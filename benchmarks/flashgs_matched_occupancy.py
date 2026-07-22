"""Fail-closed shared-node occupancy and machine-state evidence."""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELEVANT_PROCESS_PATTERN = re.compile(
    r"(?:/home/freiza/(?:agent-worktrees|benchmark)|"
    r"vectorized-gaussian-renderer|vgr-flashgs|isaac|ovrtx|"
    r"nsys|ncu|compute-sanitizer|nvcc|ninja)",
    re.IGNORECASE,
)

DEFAULT_EXECUTOR_LOCK_PATH = Path(
    "/tmp/vgr-publication-gpu-executor.lock"
)
EXECUTOR_LOCK_PATH_ENV = "VGR_GPU_EXECUTOR_LOCK_PATH"
EXECUTOR_LOCK_OWNER_ENV = "VGR_GPU_EXECUTOR_LOCK_OWNER_PID"
_ACTIVE_EXECUTOR_LOCK: Any | None = None
_ACTIVE_EXECUTOR_LOCK_EVIDENCE: dict[str, Any] | None = None


def _command(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _probe(
    command: list[str],
    *,
    accepted_nonzero_stderr: tuple[str, ...] = (),
) -> tuple[str | None, dict[str, Any]]:
    """Run a telemetry command while preserving whether its state is knowable."""

    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        return None, {
            "command": command,
            "returncode": None,
            "success": False,
            "error": str(error),
        }
    stderr = result.stderr.strip()
    normalized_stderr = stderr.casefold()
    accepted_nonzero = bool(
        result.returncode != 0
        and any(
            marker.casefold() in normalized_stderr
            for marker in accepted_nonzero_stderr
        )
    )
    success = result.returncode == 0 or accepted_nonzero
    return (result.stdout.strip() if success else None), {
        "command": command,
        "returncode": result.returncode,
        "success": success,
        "stderr": stderr,
    }


def _csv_rows(raw: str | None, fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not raw:
        return []
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values, strict=True)))
    return rows


def process_ancestry(pid: int | None = None) -> list[int]:
    """Return the current process and its Linux parent chain."""

    current = os.getpid() if pid is None else int(pid)
    result: list[int] = []
    seen: set[int] = set()
    while current > 0 and current not in seen:
        result.append(current)
        seen.add(current)
        raw = _command(["ps", "-o", "ppid=", "-p", str(current)])
        if not raw:
            break
        try:
            current = int(raw)
        except ValueError:
            break
    return result


def _executor_lock_path() -> Path:
    configured = os.environ.get(EXECUTOR_LOCK_PATH_ENV)
    return Path(configured).resolve() if configured else DEFAULT_EXECUTOR_LOCK_PATH


def _read_lock_metadata(handle: Any) -> dict[str, Any] | None:
    handle.seek(0)
    raw = handle.read().strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _release_active_executor_lock() -> None:
    global _ACTIVE_EXECUTOR_LOCK, _ACTIVE_EXECUTOR_LOCK_EVIDENCE
    if _ACTIVE_EXECUTOR_LOCK is None:
        return
    try:
        fcntl.flock(_ACTIVE_EXECUTOR_LOCK.fileno(), fcntl.LOCK_UN)
    finally:
        _ACTIVE_EXECUTOR_LOCK.close()
        _ACTIVE_EXECUTOR_LOCK = None
        _ACTIVE_EXECUTOR_LOCK_EVIDENCE = None


def ensure_cooperative_executor_lock() -> dict[str, Any]:
    """Acquire the node-wide GPU lock or prove an ancestor already holds it.

    The lock coordinates this repository's runners across all visible GPUs.
    It is paired with sampled NVIDIA process telemetry because it cannot force
    unrelated software to cooperate.
    """

    global _ACTIVE_EXECUTOR_LOCK, _ACTIVE_EXECUTOR_LOCK_EVIDENCE
    if _ACTIVE_EXECUTOR_LOCK is not None:
        assert _ACTIVE_EXECUTOR_LOCK_EVIDENCE is not None
        return dict(_ACTIVE_EXECUTOR_LOCK_EVIDENCE)

    lock_path = _executor_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_raw = os.environ.get(EXECUTOR_LOCK_OWNER_ENV)
    ancestry = process_ancestry()
    if owner_raw:
        try:
            owner_pid = int(owner_raw)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid {EXECUTOR_LOCK_OWNER_ENV}: {owner_raw!r}."
            ) from error
        if owner_pid not in ancestry[1:]:
            raise RuntimeError(
                "Declared cooperative GPU-lock owner is not a process ancestor."
            )
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                metadata = _read_lock_metadata(handle)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                raise RuntimeError(
                    "Declared ancestor does not hold the cooperative GPU lock."
                )
        if not isinstance(metadata, dict) or metadata.get("owner_pid") != owner_pid:
            raise RuntimeError(
                "Cooperative GPU-lock metadata differs from the declared owner."
            )
        return {
            "schema_version": "vgr-cooperative-gpu-lock-v1",
            "mode": "verified-ancestor-owner",
            "path": str(lock_path),
            "owner_pid": owner_pid,
            "current_pid": os.getpid(),
            "owner_in_ancestry": True,
            "lock_observed_held": True,
            "metadata": metadata,
            "pass": True,
        }

    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        metadata = _read_lock_metadata(handle)
        handle.close()
        raise RuntimeError(
            "Another publication GPU executor holds the node-wide lock: "
            f"{metadata!r}."
        ) from error
    metadata = {
        "schema_version": "vgr-cooperative-gpu-lock-metadata-v1",
        "owner_pid": os.getpid(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "command": list(os.sys.argv),
    }
    handle.seek(0)
    handle.truncate()
    json.dump(metadata, handle, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    os.environ[EXECUTOR_LOCK_PATH_ENV] = str(lock_path)
    os.environ[EXECUTOR_LOCK_OWNER_ENV] = str(os.getpid())
    evidence = {
        "schema_version": "vgr-cooperative-gpu-lock-v1",
        "mode": "direct-owner",
        "path": str(lock_path),
        "owner_pid": os.getpid(),
        "current_pid": os.getpid(),
        "owner_in_ancestry": True,
        "lock_observed_held": True,
        "metadata": metadata,
        "pass": True,
    }
    _ACTIVE_EXECUTOR_LOCK = handle
    _ACTIVE_EXECUTOR_LOCK_EVIDENCE = evidence
    atexit.register(_release_active_executor_lock)
    return dict(evidence)


def capture_compute_apps_snapshot() -> dict[str, Any]:
    """Capture the least intrusive available node-wide CUDA process sample."""

    fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory_mib")
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    raw, probe = _probe(command)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        "probe_status": probe,
        "compute_apps": _csv_rows(raw, fields),
    }


def compute_sample_failures(
    sample: dict[str, Any],
    *,
    expected_gpu_uuid: str,
    allowed_pids: set[int],
) -> list[str]:
    """Reject failed telemetry and every CUDA process outside this job."""

    failures: list[str] = []
    if (sample.get("probe_status") or {}).get("success") is not True:
        failures.append("sampled compute-process telemetry probe failed")
    for app in sample.get("compute_apps", []):
        try:
            pid = int(app.get("pid", -1))
        except (TypeError, ValueError):
            pid = -1
        if pid not in allowed_pids or app.get("gpu_uuid") != expected_gpu_uuid:
            failures.append(
                "sampled unfamiliar CUDA compute process: "
                f"{app.get('gpu_uuid')} pid={app.get('pid')} "
                f"{app.get('process_name')}"
            )
    return failures


class ComputeProcessSampler:
    """Sample NVIDIA's compute-process table during a bounded GPU job."""

    def __init__(
        self,
        *,
        expected_gpu_uuid: str,
        allowed_pids: set[int],
        interval_seconds: float = 1.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Compute-process sample interval must be positive.")
        self.expected_gpu_uuid = expected_gpu_uuid
        self.allowed_pids = set(allowed_pids)
        self.interval_seconds = float(interval_seconds)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="vgr-compute-process-sampler",
            daemon=True,
        )

    def _capture(self) -> None:
        self.samples.append(capture_compute_apps_snapshot())

    def _run(self) -> None:
        self._capture()
        while not self._stop.wait(self.interval_seconds):
            self._capture()

    def start(self) -> "ComputeProcessSampler":
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        self._capture()
        failures = [
            failure
            for sample in self.samples
            for failure in compute_sample_failures(
                sample,
                expected_gpu_uuid=self.expected_gpu_uuid,
                allowed_pids=self.allowed_pids,
            )
        ]
        return {
            "schema_version": "flashgs-matched-compute-process-sampling-v1",
            "scope": "node-wide-all-visible-nvidia-gpus",
            "coverage": "periodic-samples-not-continuous-observation",
            "interval_seconds": self.interval_seconds,
            "allowed_pids": sorted(self.allowed_pids),
            "expected_gpu_uuid": self.expected_gpu_uuid,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "failures": failures,
            "pass": not failures,
        }


def capture_node_snapshot() -> dict[str, Any]:
    """Capture GPU, process, and tmux state without initializing CUDA."""

    gpu_fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "temperature_gpu_c",
        "power_draw_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "utilization_gpu_percent",
        "memory_used_mib",
        "memory_free_mib",
    )
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,temperature.gpu,"
        "power.draw,clocks.current.sm,clocks.current.memory,"
        "utilization.gpu,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    gpu_raw, gpu_probe = _probe(gpu_command)
    gpu_rows = _csv_rows(
        gpu_raw,
        gpu_fields,
    )
    app_fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory_mib")
    apps_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    apps_raw, apps_probe = _probe(apps_command)
    compute_apps = _csv_rows(
        apps_raw,
        app_fields,
    )
    process_rows: list[dict[str, Any]] = []
    raw_processes, process_probe = _probe(
        ["ps", "-eo", "pid=,ppid=,etimes=,comm=,args="]
    )
    if raw_processes:
        for line in raw_processes.splitlines():
            parts = line.strip().split(maxsplit=4)
            if len(parts) != 5 or not RELEVANT_PROCESS_PATTERN.search(parts[4]):
                continue
            try:
                process_rows.append(
                    {
                        "pid": int(parts[0]),
                        "ppid": int(parts[1]),
                        "elapsed_seconds": int(parts[2]),
                        "command_name": parts[3],
                        "command": parts[4],
                    }
                )
            except ValueError:
                continue
    tmux_raw, tmux_probe = _probe(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        accepted_nonzero_stderr=(
            "no server running",
            "failed to connect",
            "no such file or directory",
        ),
    )
    current_tmux = _command(["tmux", "display-message", "-p", "#{session_name}"])
    ancestry = process_ancestry()
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "ancestor_pids": ancestry,
        "ancestry_complete": bool(ancestry and ancestry[-1] == 1),
        "probe_status": {
            "gpu_inventory": gpu_probe,
            "compute_apps": apps_probe,
            "process_table": process_probe,
            "tmux_sessions": tmux_probe,
        },
        "gpus": gpu_rows,
        "compute_apps": compute_apps,
        "relevant_processes": process_rows,
        "tmux_sessions": tmux_raw.splitlines() if tmux_raw else [],
        "current_tmux_session": current_tmux or None,
    }


def occupancy_failures(
    snapshot: dict[str, Any],
    *,
    expected_gpu_uuid: str,
    allow_current_gpu_process: bool,
) -> list[str]:
    """Return conflicts; an empty list is the only launch-safe verdict."""

    failures: list[str] = []
    required_probes = (
        "gpu_inventory",
        "compute_apps",
        "process_table",
        "tmux_sessions",
    )
    statuses = snapshot.get("probe_status") or {}
    for probe in required_probes:
        if (statuses.get(probe) or {}).get("success") is not True:
            failures.append(f"occupancy telemetry probe failed: {probe}")
    if snapshot.get("ancestry_complete") is not True:
        failures.append("occupancy process ancestry is incomplete")
    current_pid = int(snapshot.get("pid", -1))
    ancestry = {int(value) for value in snapshot.get("ancestor_pids", [])}
    gpu_uuids = {row.get("uuid") for row in snapshot.get("gpus", [])}
    if expected_gpu_uuid not in gpu_uuids:
        failures.append("expected GPU UUID is absent from nvidia-smi")
    for app in snapshot.get("compute_apps", []):
        try:
            pid = int(app.get("pid", -1))
        except (TypeError, ValueError):
            pid = -1
        allowed = allow_current_gpu_process and pid == current_pid and (
            app.get("gpu_uuid") == expected_gpu_uuid
        )
        if not allowed:
            failures.append(
                "unfamiliar CUDA compute process: "
                f"{app.get('gpu_uuid')} pid={app.get('pid')} "
                f"{app.get('process_name')}"
            )
    for process in snapshot.get("relevant_processes", []):
        if int(process.get("pid", -1)) not in ancestry:
            failures.append(
                "unfamiliar relevant process: "
                f"pid={process.get('pid')} {process.get('command')}"
            )
    sessions = set(snapshot.get("tmux_sessions", []))
    current_session = snapshot.get("current_tmux_session")
    allowed_sessions = {current_session} if current_session else set()
    for session in sorted(sessions - allowed_sessions):
        failures.append(f"unfamiliar tmux session: {session}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executor_lock = ensure_cooperative_executor_lock()
    snapshot = capture_node_snapshot()
    failures = occupancy_failures(
        snapshot,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=False,
    )
    result = {
        "schema_version": "flashgs-matched-occupancy-preflight-v2",
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "executor_control": {
            "cooperative_node_wide_lock": executor_lock,
            "scope": "all-visible-gpus",
        },
        "snapshot": snapshot,
        "failures": failures,
        "pass": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
