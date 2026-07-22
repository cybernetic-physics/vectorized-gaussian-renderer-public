#!/usr/bin/env python3
"""Run one publication gate with bounded execution and occupancy evidence."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


COMMAND_SCHEMA = "publication-command-record-v1"
OCCUPANCY_SCHEMA = "flashgs-matched-node-occupancy-v2"
WRAPPER_SCHEMA = "publication-bounded-gate-wrapper-v1"
RESULT_SCHEMA = "publication-gate-result-v1"
RESULT_SPEC_SCHEMA = "publication-gate-result-spec-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument(
        "--subcommands-json",
        type=Path,
        help="Optional JSON list of exact child commands represented by this gate.",
    )
    parser.add_argument(
        "--result-spec",
        type=Path,
        help="Optional declarative normalized-result spec resolved after the command succeeds.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command or any(not isinstance(token, str) or not token for token in args.command):
        parser.error("a non-empty command is required after --")
    if not 1 <= args.timeout_seconds <= 7_200:
        parser.error("--timeout-seconds must be in [1, 7200]")
    return args


def load_subcommands(path: Path | None) -> list[list[str]]:
    if path is None:
        return []
    if path.is_symlink() or not path.is_file():
        raise ValueError("Subcommand plan is missing or symlinked.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(
        not isinstance(command, list)
        or not command
        or any(not isinstance(token, str) or not token for token in command)
        for command in value
    ):
        raise ValueError("Subcommand plan must be a non-empty-command JSON list.")
    return value


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path_value: str | Path) -> dict[str, Any]:
    raw = Path(path_value)
    if raw.is_symlink():
        raise ValueError(f"Artifact may not be a symlink: {raw}")
    path = raw.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Artifact is not a regular file: {path}")
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": file_sha256(path)}


def build_normalized_result(spec_path: Path, *, gate_id: str) -> dict[str, Any]:
    if spec_path.is_symlink() or not spec_path.is_file():
        raise ValueError("Gate result spec is missing or symlinked.")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) != {
        "checks",
        "evidence",
        "gate_id",
        "identity",
        "schema_version",
    }:
        raise ValueError("Gate result spec has missing or unknown fields.")
    if spec.get("schema_version") != RESULT_SPEC_SCHEMA or spec.get("gate_id") != gate_id:
        raise ValueError("Gate result spec identity differs.")
    checks = spec.get("checks")
    evidence = spec.get("evidence")
    identity = spec.get("identity")
    if not isinstance(checks, dict) or len(checks) < 3:
        raise ValueError("Gate result spec has too few checks.")
    if (
        not isinstance(evidence, dict)
        or not evidence
        or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in evidence.items())
    ):
        raise ValueError("Gate result evidence paths are malformed.")
    if not isinstance(identity, dict) or set(identity) != {
        "gpu_uuid",
        "native_extension",
        "scene_sha256",
        "source_manifest",
    }:
        raise ValueError("Gate result identity is malformed.")
    scene_sha256 = identity.get("scene_sha256")
    if (
        not isinstance(scene_sha256, str)
        or len(scene_sha256) != 64
        or any(character not in "0123456789abcdef" for character in scene_sha256)
    ):
        raise ValueError("Gate result scene SHA-256 is malformed.")
    source = artifact_record(identity["source_manifest"])
    native = artifact_record(identity["native_extension"])
    return {
        "schema_version": RESULT_SCHEMA,
        "gate_id": gate_id,
        "created_at": utc_now(),
        "pass": True,
        "identity": {
            "gpu_uuid": identity.get("gpu_uuid"),
            "native_extension_sha256": native["sha256"],
            "scene_sha256": scene_sha256,
            "source_manifest_sha256": source["sha256"],
        },
        "checks": checks,
        "evidence": {key: artifact_record(value) for key, value in sorted(evidence.items())},
    }


def descendant_of(pid: int, ancestor_pid: int) -> bool:
    """Return whether a live PID descends from this bounded executor."""

    current = int(pid)
    seen: set[int] = set()
    while current > 0 and current not in seen:
        if current == ancestor_pid:
            return True
        seen.add(current)
        try:
            raw = subprocess.check_output(
                ["ps", "-o", "ppid=", "-p", str(current)],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            current = int(raw)
        except (OSError, subprocess.CalledProcessError, ValueError):
            return False
    return False


def sample_failures(
    sample: dict[str, Any],
    *,
    expected_gpu_uuid: str,
    executor_pid: int,
) -> list[str]:
    failures: list[str] = []
    if (sample.get("probe_status") or {}).get("success") is not True:
        failures.append("sampled compute-process telemetry probe failed")
    for app in sample.get("compute_apps", []):
        try:
            pid = int(app.get("pid", -1))
        except (TypeError, ValueError):
            pid = -1
        if app.get("gpu_uuid") != expected_gpu_uuid or not descendant_of(pid, executor_pid):
            failures.append(
                "sampled unfamiliar CUDA compute process: "
                f"{app.get('gpu_uuid')} pid={app.get('pid')} {app.get('process_name')}"
            )
    return failures


class DescendantComputeSampler:
    def __init__(
        self,
        *,
        capture: Any,
        expected_gpu_uuid: str,
        executor_pid: int,
        interval_seconds: float = 0.25,
    ) -> None:
        self.capture = capture
        self.expected_gpu_uuid = expected_gpu_uuid
        self.executor_pid = executor_pid
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _capture(self) -> None:
        self.samples.append(self.capture())

    def _run(self) -> None:
        self._capture()
        while not self._stop.wait(self.interval_seconds):
            self._capture()

    def start(self) -> "DescendantComputeSampler":
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        self._capture()
        failures = [
            failure
            for sample in self.samples
            for failure in sample_failures(
                sample,
                expected_gpu_uuid=self.expected_gpu_uuid,
                executor_pid=self.executor_pid,
            )
        ]
        return {
            "schema_version": "publication-descendant-compute-process-sampling-v1",
            "scope": "node-wide-all-visible-nvidia-gpus",
            "coverage": "periodic-samples-not-continuous-observation",
            "allowed_process_rule": "executor-descendants-on-exact-expected-gpu",
            "executor_pid": self.executor_pid,
            "expected_gpu_uuid": self.expected_gpu_uuid,
            "interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "failures": failures,
            "pass": not failures,
        }


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    source_root = args.source_root.resolve(strict=True)
    occupancy_module_root = source_root / "benchmarks"
    if not (occupancy_module_root / "flashgs_matched_occupancy.py").is_file():
        raise FileNotFoundError("Frozen source root lacks flashgs_matched_occupancy.py.")
    sys.path.insert(0, str(occupancy_module_root))
    from flashgs_matched_occupancy import (  # type: ignore[import-not-found]
        capture_compute_apps_snapshot,
        capture_node_snapshot,
        ensure_cooperative_executor_lock,
        occupancy_failures,
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to reuse gate output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    started_at = utc_now()
    subcommands = load_subcommands(args.subcommands_json)
    command_record = {
        "schema_version": COMMAND_SCHEMA,
        "gate_id": args.gate_id,
        "started_at": started_at,
        "timeout_seconds": args.timeout_seconds,
        "command": list(args.command),
        "subcommands": subcommands,
    }
    write_new(output_dir / "command.json", canonical_json_bytes(command_record))

    executor_lock = ensure_cooperative_executor_lock()
    preflight = capture_node_snapshot()
    preflight_failures = occupancy_failures(
        preflight,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=False,
    )
    if preflight_failures:
        write_new(
            output_dir / "node-occupancy.json",
            canonical_json_bytes(
                {
                    "schema_version": OCCUPANCY_SCHEMA,
                    "expected_gpu_uuid": args.expected_gpu_uuid,
                    "executor_control": {
                        "scope": "all-visible-gpus",
                        "cooperative_node_wide_lock": executor_lock,
                    },
                    "preflight": preflight,
                    "preflight_failures": preflight_failures,
                    "postflight": None,
                    "postflight_failures": ["gate not launched"],
                    "sampled_compute_process_telemetry": {
                        "sample_count": 0,
                        "pass": False,
                        "failures": ["gate not launched"],
                    },
                    "pass": False,
                }
            ),
        )
        raise RuntimeError("GPU occupancy preflight failed: " + "; ".join(preflight_failures))

    log_path = output_dir / "full.log"
    timed_out = False
    child: subprocess.Popen[str] | None = None
    sampler = DescendantComputeSampler(
        capture=capture_compute_apps_snapshot,
        expected_gpu_uuid=args.expected_gpu_uuid,
        executor_pid=os.getpid(),
    ).start()
    try:
        with log_path.open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(
                args.command,
                cwd=source_root,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = child.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    returncode = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    returncode = child.wait(timeout=10)
    finally:
        sampled = sampler.stop()
    postflight = capture_node_snapshot()
    postflight_failures = occupancy_failures(
        postflight,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=False,
    )
    occupancy = {
        "schema_version": OCCUPANCY_SCHEMA,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "executor_control": {
            "scope": "all-visible-gpus",
            "cooperative_node_wide_lock": executor_lock,
        },
        "preflight": preflight,
        "preflight_failures": preflight_failures,
        "postflight": postflight,
        "postflight_failures": postflight_failures,
        "sampled_compute_process_telemetry": sampled,
        "pass": bool(not preflight_failures and not postflight_failures and sampled["pass"]),
    }
    write_new(output_dir / "node-occupancy.json", canonical_json_bytes(occupancy))
    write_new(output_dir / "exit-status.txt", f"{returncode}\n".encode("ascii"))
    result_path: Path | None = None
    if returncode == 0 and not timed_out and occupancy["pass"] and args.result_spec is not None:
        normalized = build_normalized_result(args.result_spec, gate_id=args.gate_id)
        result_path = output_dir / "result.json"
        write_new(result_path, canonical_json_bytes(normalized))
    completed_at = utc_now()
    wrapper = {
        "schema_version": WRAPPER_SCHEMA,
        "gate_id": args.gate_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "timeout_seconds": args.timeout_seconds,
        "timed_out": timed_out,
        "returncode": returncode,
        "command_record": str(output_dir / "command.json"),
        "log": str(log_path),
        "exit_status": str(output_dir / "exit-status.txt"),
        "occupancy": str(output_dir / "node-occupancy.json"),
        "result": str(result_path) if result_path is not None else None,
        "pass": bool(returncode == 0 and not timed_out and occupancy["pass"]),
    }
    write_new(output_dir / "wrapper.json", canonical_json_bytes(wrapper))
    print("PUBLICATION_BOUNDED_GATE " + json.dumps(wrapper, sort_keys=True))
    if not wrapper["pass"]:
        raise SystemExit(returncode if returncode != 0 else 1)


if __name__ == "__main__":
    main()
