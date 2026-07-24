"""Run the finite Isaac graphical acceptance gate with UUID-safe GPU selection.

This runner deliberately uses only the Python standard library.  Project code is
loaded from a read-only source tree and every writable cache is redirected to a
per-run scratch directory outside the canonical Isaac environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .isaac_gpu_identity import normalize_gpu_uuid
else:
    from isaac_gpu_identity import normalize_gpu_uuid


SCHEMA_VERSION = "isaac-graphics-gate/v1"
OVERLAY_REQUIREMENTS = (
    "plyfile==1.1.3",
    "lpips==0.1.4",
    "tqdm==4.67.1",
)
DEFAULT_VULKAN_ICD_CANDIDATES = (
    Path("/etc/vulkan/icd.d/nvidia_icd.json"),
    Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
)
IDENTITY_MARKER = "ISAAC_GPU_IDENTITY "
GPU_BLOCK = re.compile(r"(?m)^GPU(?P<index>\d+):\s*$")
GPU_TABLE_ROW = re.compile(
    r"(?m)^\|\s*(?P<index>\d+)\s*\|\s*(?P<name>[^|]+?)\s*\|\s*"
    r"(?P<active>Yes:\s*\d+|No)\s*\|"
)
SOURCE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "outputs",
        "profiles",
    }
)
RELEVANT_PROCESS_PATTERN = re.compile(
    r"(?i)(?:^|[/\s_-])(?:python(?:\d+(?:\.\d+)*)?|isaac(?:sim)?|kit|ovrtx|"
    r"gaussian|nsys|ncu|ngfx|compute-sanitizer|cuda-gdb|vulkaninfo|"
    r"xvfb(?:-run)?|xorg)"
    r"(?:$|[/\s_.-])"
)


@dataclass(frozen=True)
class Lane:
    name: str
    command: list[str]
    marker: str
    result_json: Path | None = None
    expected_rasterize_mode: str | None = None


class GateFailure(RuntimeError):
    """An acceptance condition failed with preserved evidence."""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--canonical-runtime", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=root)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
        help="Rasterizer mode exercised by the Isaac extension lane.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--xvfb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vulkan-icd",
        type=Path,
        default=None,
        help=(
            "Explicit NVIDIA Vulkan ICD manifest. By default, the gate accepts "
            "exactly one standard NVIDIA runfile or distribution-package location."
        ),
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        parser.error("--source-commit must be a full lowercase Git commit.")
    if re.fullmatch(r"[0-9a-f]{64}", args.source_sha256) is None:
        parser.error("--source-sha256 must be a lowercase SHA-256 digest.")
    for name in ("python", "uv", "canonical_runtime", "source_root"):
        if not getattr(args, name).exists():
            parser.error(f"--{name.replace('_', '-')} does not exist.")
    if not args.canonical_runtime.is_dir() or not args.source_root.is_dir():
        parser.error("--canonical-runtime and --source-root must be directories.")
    if not os.access(args.python, os.X_OK) or not os.access(args.uv, os.X_OK):
        parser.error("--python and --uv must be executable.")
    canonical_runtime = args.canonical_runtime.resolve()
    python_path = args.python.absolute()
    if canonical_runtime not in python_path.parents:
        parser.error("--python must be invoked through --canonical-runtime.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("--output-root must be absent or empty.")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == source_root or source_root in output_root.parents:
        parser.error("--output-root must be outside --source-root.")
    if output_root == canonical_runtime or canonical_runtime in output_root.parents:
        parser.error("--output-root must be outside --canonical-runtime.")
    return args


def resolve_vulkan_icd(
    explicit: Path | None,
    *,
    candidates: tuple[Path, ...] = DEFAULT_VULKAN_ICD_CANDIDATES,
) -> Path:
    """Resolve one unambiguous NVIDIA ICD from the runfile or distro layout."""

    if explicit is not None:
        if not explicit.is_file():
            raise GateFailure(f"NVIDIA Vulkan ICD does not exist: {explicit}")
        return explicit.resolve(strict=True)

    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        locations = ", ".join(str(candidate) for candidate in candidates)
        raise GateFailure(f"NVIDIA Vulkan ICD was not found in: {locations}")

    resolved = {candidate.resolve(strict=True) for candidate in existing}
    if len(resolved) != 1:
        locations = ", ".join(str(candidate) for candidate in existing)
        raise GateFailure(
            "Multiple distinct NVIDIA Vulkan ICD manifests were found; pass "
            f"--vulkan-icd explicitly: {locations}"
        )
    return resolved.pop()


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    group_id = process.pid
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The direct child may exit while a grandchild ignores SIGTERM. Always
    # escalate the owned process group so no detached Xvfb/Kit child survives.
    try:
        os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def run_bounded(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    log_path: Path,
) -> dict[str, Any]:
    """Run one owned process group and always preserve its combined log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    timed_out = False
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            returncode = process.returncode
    return {
        "command": command,
        "command_shell": shlex.join(command),
        "cwd": str(cwd),
        "log": str(log_path),
        "started_unix": started,
        "duration_seconds": time.time() - started,
        "timeout_seconds": timeout,
        "timed_out": timed_out,
        "returncode": returncode,
    }


def capture_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        raise GateFailure(f"Command timed out: {shlex.join(command)}\n{stdout}{stderr}") from exc
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )


def parse_vulkan_summary(text: str) -> list[dict[str, Any]]:
    """Extract physical-device indexes, names, and full UUIDs."""

    matches = list(GPU_BLOCK.finditer(text))
    devices: list[dict[str, Any]] = []
    for position, match in enumerate(matches):
        body_end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end() : body_end]
        name_match = re.search(r"(?m)^\s*deviceName\s*=\s*(.+?)\s*$", body)
        uuid_match = re.search(r"(?m)^\s*deviceUUID\s*=\s*(\S+)\s*$", body)
        if name_match is None or uuid_match is None:
            continue
        devices.append(
            {
                "index": int(match.group("index")),
                "name": name_match.group(1).strip(),
                "uuid_raw": uuid_match.group(1),
                "uuid": normalize_gpu_uuid(uuid_match.group(1)),
            }
        )
    return devices


def parse_torch_identity(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.startswith(IDENTITY_MARKER)]
    if len(lines) != 1:
        raise GateFailure("Torch identity probe did not emit exactly one identity record.")
    try:
        identity = json.loads(lines[0][len(IDENTITY_MARKER) :])
        identity["cuda_device_uuid"] = normalize_gpu_uuid(identity["cuda_device_uuid"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GateFailure("Torch identity record is invalid.") from exc
    return identity


def match_vulkan_device(
    devices: list[dict[str, Any]],
    torch_uuid: str,
) -> dict[str, Any]:
    matches = [device for device in devices if device["uuid"] == torch_uuid]
    if len(matches) != 1:
        raise GateFailure(f"Torch cuda:0 UUID matched {len(matches)} Vulkan devices; expected exactly one.")
    if "NVIDIA" not in matches[0]["name"].upper():
        raise GateFailure("Torch cuda:0 matched a non-NVIDIA Vulkan physical device.")
    return matches[0]


def parse_nvidia_gpu_inventory(text: str) -> list[dict[str, Any]]:
    """Parse the fixed nvidia-smi query emitted by machine_provenance."""

    devices = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=4)]
        if len(fields) != 5:
            raise GateFailure(f"Malformed nvidia-smi GPU inventory row: {line}")
        index, name, uuid, driver, memory_mib = fields
        try:
            parsed_index = int(index)
            parsed_memory = int(memory_mib)
            parsed_uuid = normalize_gpu_uuid(uuid)
        except ValueError as exc:
            raise GateFailure(f"Invalid nvidia-smi GPU inventory row: {line}") from exc
        devices.append(
            {
                "index": parsed_index,
                "name": name,
                "uuid_raw": uuid,
                "uuid": parsed_uuid,
                "driver": driver,
                "memory_mib": parsed_memory,
            }
        )
    if not devices:
        raise GateFailure("nvidia-smi reported no GPUs.")
    return devices


def runtime_fingerprint(root: Path) -> dict[str, Any]:
    """Hash canonical-runtime metadata without reading multi-gigabyte binaries."""

    digest = hashlib.sha256()
    entries = 0
    files = 0
    bytes_total = 0
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        stat = path.lstat()
        if path.is_symlink():
            kind = "link"
            extra = os.readlink(path)
            metadata = f"{stat.st_mode}\0{stat.st_size}\0{stat.st_mtime_ns}"
        elif path.is_dir():
            kind = "dir"
            extra = ""
            # Kit may create and remove transient entries while resolving an
            # extension path, changing only the directory mtime. The entry set
            # still detects any surviving addition or deletion.
            metadata = f"{stat.st_mode}"
        else:
            kind = "file"
            extra = ""
            files += 1
            bytes_total += stat.st_size
            metadata = f"{stat.st_mode}\0{stat.st_size}\0{stat.st_mtime_ns}"
        record = f"{kind}\0{relative}\0{metadata}\0{extra}\n"
        digest.update(record.encode("utf-8", errors="surrogateescape"))
        entries += 1
    return {
        "root": str(root.resolve()),
        "metadata_sha256": digest.hexdigest(),
        "entries": entries,
        "files": files,
        "bytes": bytes_total,
    }


def source_fingerprint(root: Path) -> dict[str, Any]:
    """Content-hash the test source tree while excluding generated outputs."""

    digest = hashlib.sha256()
    files = 0
    bytes_total = 0
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative_path = path.relative_to(root)
        if any(part in SOURCE_EXCLUDED_PARTS for part in relative_path.parts):
            continue
        relative = relative_path.as_posix()
        if path.is_symlink():
            record = f"link\0{relative}\0{os.readlink(path)}\n"
        elif path.is_dir():
            record = f"dir\0{relative}\n"
        elif path.is_file():
            file_digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(block)
                    size += len(block)
            record = f"file\0{relative}\0{size}\0{file_digest.hexdigest()}\n"
            files += 1
            bytes_total += size
        else:
            record = f"other\0{relative}\n"
        digest.update(record.encode("utf-8", errors="surrogateescape"))
    return {
        "root": str(root.resolve()),
        "content_sha256": digest.hexdigest(),
        "files": files,
        "bytes": bytes_total,
        "excluded_parts": sorted(SOURCE_EXCLUDED_PARTS),
    }


def process_inventory(env: dict[str, str]) -> dict[str, Any]:
    gpu_query = capture_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=Path.cwd(),
        env=env,
        timeout=15,
    )
    if gpu_query.returncode != 0:
        raise GateFailure("nvidia-smi process inventory failed: " + gpu_query.stderr.strip())
    gpu_rows = [line.strip() for line in gpu_query.stdout.splitlines() if line.strip()]

    process_query = capture_command(
        ["ps", "-eo", "pid=,ppid=,pgid=,user=,stat=,comm=,args="],
        cwd=Path.cwd(),
        env=env,
        timeout=15,
    )
    if process_query.returncode != 0:
        raise GateFailure("Host process inventory failed: " + process_query.stderr.strip())
    host_rows: list[dict[str, Any]] = []
    for line in process_query.stdout.splitlines():
        fields = line.strip().split(maxsplit=6)
        if len(fields) < 6:
            continue
        if len(fields) == 6:
            fields.append("")
        pid, ppid, pgid, user, state, command, arguments = fields
        if command == "ps":
            continue
        host_rows.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "pgid": int(pgid),
                "user": user,
                "state": state,
                "command": command,
                "arguments": arguments,
            }
        )
    relevant_rows = [
        row for row in host_rows if RELEVANT_PROCESS_PATTERN.search(f"{row['command']} {row['arguments']}")
    ]
    return {
        "gpu_compute": {
            "command": gpu_query.args,
            "rows": gpu_rows,
            "empty": not gpu_rows,
        },
        "host": {
            "command": process_query.args,
            "rows": host_rows,
            "relevant_rows": relevant_rows,
        },
    }


def assert_no_residual_processes(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Reject GPU work or new graphical-test-related host processes."""

    if not current["gpu_compute"]["empty"]:
        raise GateFailure("A GPU process survived the lane: " + "; ".join(current["gpu_compute"]["rows"]))
    baseline_pids = {row["pid"] for row in baseline["host"]["relevant_rows"]}
    added = [row for row in current["host"]["relevant_rows"] if row["pid"] not in baseline_pids]
    if added:
        raise GateFailure(
            "Graphical-test-related host processes survived the lane: "
            + "; ".join(f"{row['pid']} {row['arguments']}" for row in added)
        )
    return {
        "pass": True,
        "gpu_rows": [],
        "new_relevant_host_rows": [],
    }


def machine_provenance(env: dict[str, str], uv: Path) -> dict[str, Any]:
    """Capture bounded host/tool versions without importing Kit."""

    commands = {
        "gpu_inventory": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_smi_version": ["nvidia-smi", "--version"],
        "kernel": ["uname", "-a"],
        "uv": [str(uv.absolute()), "--version"],
    }
    records: dict[str, Any] = {}
    for name, command in commands.items():
        captured = capture_command(
            command,
            cwd=Path.cwd(),
            env=env,
            timeout=15,
        )
        if captured.returncode != 0:
            raise GateFailure(f"Machine provenance command failed: {shlex.join(command)}: " + captured.stderr.strip())
        records[name] = {
            "command": command,
            "stdout": captured.stdout.strip(),
        }
    records["gpu_inventory"]["devices"] = parse_nvidia_gpu_inventory(records["gpu_inventory"]["stdout"])
    os_release = Path("/etc/os-release")
    records["os_release"] = os_release.read_text(encoding="utf-8", errors="replace") if os_release.is_file() else None
    return records


def assert_single_active_gpu(
    log_text: str,
    expected_index: int,
    *,
    expected_experience: Path | None = None,
    success_marker: str | None = None,
) -> dict[str, Any]:
    required_args = [
        f"--/renderer/activeGpu={expected_index}",
        "--/renderer/multiGpu/enabled=False",
        "--/renderer/multiGpu/maxGpuCount=1",
        "--/physics/cudaDevice=0",
    ]
    missing = [argument for argument in required_args if argument not in log_text]
    if missing:
        raise GateFailure("Isaac log omitted required launch args: " + ", ".join(missing))
    if expected_experience is not None and str(expected_experience) not in log_text:
        raise GateFailure("Isaac did not launch the gate's isolated Kit experience.")
    rows = [match.groupdict() for match in GPU_TABLE_ROW.finditer(log_text)]
    active_rows = [row for row in rows if row["active"].startswith("Yes:")]
    expected = [row for row in active_rows if int(row["index"]) == expected_index and "NVIDIA" in row["name"].upper()]
    if len(active_rows) != 1 or len(expected) != 1:
        raise GateFailure("GPU Foundation did not activate exactly the UUID-matched NVIDIA device.")
    if "Simulation App Startup Complete" not in log_text:
        raise GateFailure("Isaac never reported startup completion.")
    if "Simulation App Shutting Down" not in log_text:
        raise GateFailure("Isaac never entered its close path.")
    if success_marker is not None:
        marker_matches = list(
            re.finditer(
                rf"(?m)^{re.escape(success_marker)}$",
                log_text,
            )
        )
        if len(marker_matches) != 1:
            raise GateFailure("Isaac log did not contain exactly one success marker.")
        startup_position = log_text.index("Simulation App Startup Complete")
        shutdown_position = log_text.index("Simulation App Shutting Down")
        if not startup_position < marker_matches[0].start() < shutdown_position:
            raise GateFailure("Isaac success marker was outside the startup/close lifecycle.")
    registry_fallbacks = (
        "Failed to solve some dependencies locally",
        "syncing registry:",
    )
    present_fallbacks = [marker for marker in registry_fallbacks if marker in log_text]
    if present_fallbacks:
        raise GateFailure("Isaac fell back to the moving extension registry: " + ", ".join(present_fallbacks))
    return {"rows": rows, "active_rows": active_rows}


def checker_result(
    checker: Path,
    log_path: Path,
    marker: str,
    result_json: Path | None,
    *,
    env: dict[str, str],
) -> dict[str, Any]:
    command = [str(checker), str(log_path), marker]
    if result_json is not None:
        command.append(str(result_json))
    checked = capture_command(command, cwd=checker.parents[4], env=env, timeout=30)
    result = {
        "command": command,
        "returncode": checked.returncode,
        "stdout": checked.stdout,
        "stderr": checked.stderr,
    }
    if checked.returncode != 0:
        raise GateFailure(
            f"Fatal/marker scan failed for {log_path.name}: " + (checked.stderr.strip() or checked.stdout.strip())
        )
    return result


def assert_rasterize_mode_result(
    result_json: Path,
    expected_mode: str,
) -> dict[str, Any]:
    """Require evidence that the selected mode reached the CUDA backend."""

    try:
        result = json.loads(result_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateFailure(
            f"Invalid extension result JSON: {result_json}"
        ) from exc
    configuration = result.get("configuration")
    renderer = result.get("renderer_configuration")
    if not isinstance(configuration, dict) or not isinstance(renderer, dict):
        raise GateFailure(
            "Extension evidence omitted renderer-mode configuration."
        )
    acceptance = {
        "expected_rasterize_mode": expected_mode,
        "reported_rasterize_mode": configuration.get("rasterize_mode"),
        "renderer_asserted": renderer.get("asserted"),
        "renderer_requested_rasterize_mode": renderer.get(
            "requested_rasterize_mode"
        ),
        "renderer_actual_rasterize_mode": renderer.get(
            "actual_rasterize_mode"
        ),
        "renderer_matches": renderer.get("matches"),
    }
    if acceptance != {
        "expected_rasterize_mode": expected_mode,
        "reported_rasterize_mode": expected_mode,
        "renderer_asserted": True,
        "renderer_requested_rasterize_mode": expected_mode,
        "renderer_actual_rasterize_mode": expected_mode,
        "renderer_matches": True,
    }:
        raise GateFailure(
            "Extension did not prove the selected rasterize mode reached "
            f"the CUDA backend: {acceptance}."
        )
    return acceptance


def make_environment(
    source_root: Path,
    scratch: Path,
    active_gpu: int,
    vulkan_icd: Path,
    python_overlay: Path | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    directories = {
        "HOME": scratch / "home",
        "XDG_CACHE_HOME": scratch / "xdg-cache",
        "XDG_CONFIG_HOME": scratch / "xdg-config",
        "XDG_DATA_HOME": scratch / "xdg-data",
        "XDG_RUNTIME_DIR": scratch / "xdg-runtime",
        "TORCH_EXTENSIONS_DIR": scratch / "torch-extensions",
        "WARP_CACHE_PATH": scratch / "warp-cache",
        "PYTHONPYCACHEPREFIX": scratch / "pycache",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    os.chmod(directories["XDG_RUNTIME_DIR"], 0o700)
    env.update({name: str(path) for name, path in directories.items()})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VGR_ISAAC_ACTIVE_GPU"] = str(active_gpu)
    env["VGR_ISAAC_REQUIRE_EXPLICIT_GPU"] = "1"
    env["VGR_PROJECT_ROOT"] = str(source_root)
    env["OMNI_KIT_ALLOW_ROOT"] = "1"
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["VK_ICD_FILENAMES"] = str(vulkan_icd)
    python_paths = []
    if python_overlay is not None:
        python_paths.append(str(python_overlay))
    python_paths.append(str(source_root / "src"))
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def prepare_dependency_overlay(
    uv: Path,
    python: Path,
    source_root: Path,
    scratch: Path,
    logs: Path,
    base_env: dict[str, str],
) -> dict[str, Any]:
    """Install pinned project runtime dependencies outside canonical Isaac."""

    target = scratch / "python-overlay"
    target.mkdir(parents=True)
    env = dict(base_env)
    env["UV_CACHE_DIR"] = str(scratch / "uv-cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(scratch / "dependency-pycache")
    requirements = list(OVERLAY_REQUIREMENTS)
    command = [
        str(uv.absolute()),
        "pip",
        "install",
        "--python",
        str(python.absolute()),
        "--target",
        str(target),
        "--no-deps",
        "--only-binary=:all:",
        *requirements,
    ]
    receipt = run_bounded(
        command,
        cwd=source_root,
        env=env,
        timeout=180,
        log_path=logs / "dependency-overlay-install.log",
    )
    if receipt["timed_out"] or receipt["returncode"] != 0:
        raise GateFailure(f"Disposable dependency overlay installation failed; see {receipt['log']}.")
    verify_env = dict(env)
    verify_env["PYTHONPATH"] = os.pathsep.join([str(target), str(source_root / "src")])
    verification = capture_command(
        [
            str(python.absolute()),
            "-c",
            (
                "import importlib.metadata as m, json, lpips, plyfile, tqdm; "
                "print(json.dumps({'lpips': m.version('lpips'), "
                "'plyfile': m.version('plyfile'), 'tqdm': m.version('tqdm')}, "
                "sort_keys=True))"
            ),
        ],
        cwd=source_root,
        env=verify_env,
        timeout=60,
    )
    (logs / "dependency-overlay-verify.log").write_text(
        verification.stdout + verification.stderr,
        encoding="utf-8",
    )
    if verification.returncode != 0:
        raise GateFailure("Disposable dependency overlay import verification failed.")
    try:
        versions = json.loads(verification.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise GateFailure("Dependency overlay version record is invalid.") from exc
    return {
        "target": str(target),
        "requirements": requirements,
        "versions": versions,
        "install": receipt,
        "verification_log": str(logs / "dependency-overlay-verify.log"),
    }


def prepare_experience_overlay(
    canonical_runtime: Path,
    source_root: Path,
    scratch: Path,
) -> dict[str, Any]:
    """Mirror only extension-directory links around the repo-owned test app."""

    candidates = sorted((canonical_runtime / "lib").glob("python*/site-packages/isaacsim"))
    package_roots = [candidate for candidate in candidates if candidate.is_dir()]
    if len(package_roots) != 1:
        raise GateFailure(f"Expected one canonical Isaac package root, found {len(package_roots)}.")
    isaac_root = package_roots[0]
    overlay = scratch / "isaac-app-overlay"
    app_dir = overlay / "apps"
    app_dir.mkdir(parents=True)
    template = source_root / "apps" / "vgr.graphical_test.kit"
    if not template.is_file():
        raise GateFailure(f"Missing graphical test experience: {template}")
    experience = app_dir / template.name
    shutil.copy2(template, experience)
    links: dict[str, str] = {}
    for name in ("exts", "extscache", "extsDeprecated"):
        target = isaac_root / name
        if not target.is_dir():
            raise GateFailure(f"Canonical Isaac extension directory is missing: {target}")
        link = overlay / name
        link.symlink_to(target, target_is_directory=True)
        links[name] = str(target.resolve())
    project_extensions = source_root / "exts"
    if not project_extensions.is_dir():
        raise GateFailure(f"Project extension directory is missing: {project_extensions}")
    project_link = overlay / "extsUser"
    project_link.symlink_to(project_extensions, target_is_directory=True)
    links["extsUser"] = str(project_extensions.resolve())
    return {
        "experience": str(experience.absolute()),
        "template": str(template.resolve()),
        "isaac_package_root": str(isaac_root.resolve()),
        "extension_links": links,
    }


def lane_plan(args: argparse.Namespace, scratch: Path) -> list[Lane]:
    # Do not resolve the venv's python symlink: CPython uses the invoked path to
    # discover pyvenv.cfg and its site-packages.
    python = str(args.python.absolute())
    scripts = args.source_root.resolve() / "scripts"
    portable = scratch / "portable"
    result_root = args.output_root / "results"
    extension_result = result_root / "extension.json"
    return [
        Lane(
            "lifecycle-default",
            [
                python,
                str(scripts / "isaac_headless_smoke.py"),
                "--renderer",
                "default",
                "--portable-root",
                str(portable / "lifecycle-default"),
            ],
            "ISAAC_HEADLESS_SMOKE_OK",
        ),
        Lane(
            "extension",
            [
                python,
                str(scripts / "isaacsim_extension_smoke.py"),
                "--repo-root",
                str(args.source_root.resolve()),
                "--device",
                "cuda:0",
                "--rasterize-mode",
                args.rasterize_mode,
                "--output",
                str(extension_result),
                "--portable-root",
                str(portable / "extension"),
            ],
            "ISAACSIM_GAUSSIAN_RENDERER_SMOKE_OK",
            extension_result,
            args.rasterize_mode,
        ),
        Lane(
            "fabric-camera",
            [
                python,
                str(scripts / "fabric_camera_smoke.py"),
                "--repo-root",
                str(args.source_root.resolve()),
                "--batch",
                "4",
                "--output",
                str(result_root / "fabric-camera.json"),
                "--portable-root",
                str(portable / "fabric-camera"),
            ],
            "FABRIC_CAMERA_SMOKE_OK",
            result_root / "fabric-camera.json",
        ),
    ]


def main() -> None:
    args = parse_args()
    args.vulkan_icd = resolve_vulkan_icd(args.vulkan_icd)
    args.output_root.mkdir(parents=True, exist_ok=True)
    scratch = args.output_root / "scratch"
    logs = args.output_root / "logs"
    scratch.mkdir()
    logs.mkdir()
    summary_path = args.output_root / "summary.json"
    checker = (
        args.source_root / ".agents" / "skills" / "test-isaac-graphics" / "scripts" / "check-isaac-log.sh"
    ).resolve()
    base_env = dict(os.environ)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pass": False,
        "source_commit": args.source_commit,
        "expected_source_sha256": args.source_sha256,
        "source_root": str(args.source_root.resolve()),
        "rasterize_mode": args.rasterize_mode,
        "canonical_runtime": str(args.canonical_runtime.resolve()),
        "python": str(args.python.absolute()),
        "uv": str(args.uv.absolute()),
        "vulkan_icd": str(args.vulkan_icd),
        "invocation": {
            "command": [sys.executable, *sys.argv],
            "command_shell": shlex.join([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd()),
        },
        "lanes": [],
    }
    try:
        summary["source_before"] = source_fingerprint(args.source_root.resolve())
        if summary["source_before"]["content_sha256"] != args.source_sha256:
            raise GateFailure("The source tree does not match --source-sha256.")
        summary["runtime_before"] = runtime_fingerprint(args.canonical_runtime)
        summary["processes_before"] = process_inventory(base_env)
        if not summary["processes_before"]["gpu_compute"]["empty"]:
            raise GateFailure("GPU workload already present before the gate.")
        summary["machine"] = machine_provenance(base_env, args.uv)

        vulkan_env = dict(base_env)
        vulkan_env["VK_ICD_FILENAMES"] = str(args.vulkan_icd)
        vulkan_command = ["vulkaninfo", "--summary"]
        if args.xvfb:
            vulkan_command = ["xvfb-run", "-a", *vulkan_command]
        vulkan_receipt = run_bounded(
            vulkan_command,
            cwd=args.source_root,
            env=vulkan_env,
            timeout=30,
            log_path=logs / "vulkaninfo-summary.log",
        )
        summary["vulkan_probe"] = vulkan_receipt
        if vulkan_receipt["timed_out"]:
            raise GateFailure("vulkaninfo exceeded its bounded watchdog.")
        if vulkan_receipt["returncode"] != 0:
            raise GateFailure(f"vulkaninfo failed with return code {vulkan_receipt['returncode']}.")
        vulkan_text = Path(vulkan_receipt["log"]).read_text(
            encoding="utf-8",
            errors="replace",
        )
        devices = parse_vulkan_summary(vulkan_text)
        if not devices:
            raise GateFailure("vulkaninfo exposed no physical devices with full UUIDs.")
        summary["vulkan_devices"] = devices

        identity_scratch = scratch / "identity"
        identity_env = make_environment(
            args.source_root.resolve(),
            identity_scratch,
            0,
            args.vulkan_icd,
        )
        identity_receipt = run_bounded(
            [str(args.python.absolute()), str(args.source_root / "scripts" / "isaac_gpu_identity.py")],
            cwd=args.source_root,
            env=identity_env,
            timeout=60,
            log_path=logs / "torch-gpu-identity.log",
        )
        summary["torch_identity_probe"] = identity_receipt
        if identity_receipt["timed_out"]:
            raise GateFailure("Torch GPU identity probe exceeded its watchdog.")
        if identity_receipt["returncode"] != 0:
            raise GateFailure(f"Torch GPU identity probe failed with return code {identity_receipt['returncode']}.")
        identity_text = Path(identity_receipt["log"]).read_text(
            encoding="utf-8",
            errors="replace",
        )
        torch_identity = parse_torch_identity(identity_text)
        matched = match_vulkan_device(devices, torch_identity["cuda_device_uuid"])
        nvidia_matches = [
            device
            for device in summary["machine"]["gpu_inventory"]["devices"]
            if device["uuid"] == torch_identity["cuda_device_uuid"]
        ]
        if len(nvidia_matches) != 1:
            raise GateFailure("Torch cuda:0 did not match exactly one nvidia-smi GPU UUID.")
        summary["torch_identity"] = torch_identity
        summary["matched_vulkan_device"] = matched
        summary["matched_nvidia_smi_device"] = nvidia_matches[0]

        dependency_overlay = prepare_dependency_overlay(
            args.uv,
            args.python,
            args.source_root.resolve(),
            scratch,
            logs,
            base_env,
        )
        summary["dependency_overlay"] = dependency_overlay

        env = make_environment(
            args.source_root.resolve(),
            scratch / "runtime",
            matched["index"],
            args.vulkan_icd,
            python_overlay=Path(dependency_overlay["target"]),
        )
        experience_overlay = prepare_experience_overlay(
            args.canonical_runtime.resolve(),
            args.source_root.resolve(),
            scratch,
        )
        env["VGR_ISAAC_EXPERIENCE"] = experience_overlay["experience"]
        env["VGR_ISAAC_REQUIRE_EXPERIENCE"] = "1"
        summary["experience_overlay"] = experience_overlay
        experience_path = Path(experience_overlay["experience"])
        for lane in lane_plan(args, scratch):
            command = list(lane.command)
            if args.xvfb:
                command = ["xvfb-run", "-a", *command]
            receipt = run_bounded(
                command,
                cwd=args.source_root,
                env=env,
                timeout=args.timeout_seconds,
                log_path=logs / f"{lane.name}.log",
            )
            lane_record: dict[str, Any] = {
                "name": lane.name,
                "process": receipt,
            }
            summary["lanes"].append(lane_record)
            if receipt["timed_out"]:
                raise GateFailure(f"{lane.name} exceeded its watchdog.")
            if receipt["returncode"] != 0:
                raise GateFailure(f"{lane.name} exited {receipt['returncode']}; see {receipt['log']}.")
            log_path = Path(receipt["log"])
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            lane_record["log_acceptance"] = checker_result(
                checker,
                log_path,
                lane.marker,
                lane.result_json,
                env=env,
            )
            if lane.expected_rasterize_mode is not None:
                if lane.result_json is None:
                    raise GateFailure(
                        f"{lane.name} omitted its renderer result JSON."
                    )
                lane_record["rasterize_mode_acceptance"] = (
                    assert_rasterize_mode_result(
                        lane.result_json,
                        lane.expected_rasterize_mode,
                    )
                )
            lane_record["gpu_foundation"] = assert_single_active_gpu(
                log_text,
                matched["index"],
                expected_experience=experience_path,
                success_marker=lane.marker,
            )
            lane_record["processes_after"] = process_inventory(env)
            try:
                residual_acceptance = assert_no_residual_processes(
                    summary["processes_before"],
                    lane_record["processes_after"],
                )
                lane_record["residual_process_acceptance"] = residual_acceptance
            except GateFailure as exc:
                raise GateFailure(f"{lane.name}: {exc}") from exc

        summary["source_after"] = source_fingerprint(args.source_root.resolve())
        summary["source_unchanged"] = summary["source_before"] == summary["source_after"]
        if not summary["source_unchanged"]:
            raise GateFailure("The source tree changed during the gate.")
        summary["runtime_after"] = runtime_fingerprint(args.canonical_runtime)
        summary["runtime_unchanged"] = summary["runtime_before"] == summary["runtime_after"]
        if not summary["runtime_unchanged"]:
            raise GateFailure("The canonical Isaac runtime changed during the gate.")
        summary["processes_final"] = process_inventory(env)
        summary["residual_process_acceptance"] = assert_no_residual_processes(
            summary["processes_before"],
            summary["processes_final"],
        )
        summary["pass"] = True
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        try:
            summary["source_after"] = source_fingerprint(args.source_root.resolve())
            summary["source_unchanged"] = summary.get("source_before") == summary["source_after"]
            summary["runtime_after"] = runtime_fingerprint(args.canonical_runtime)
            summary["runtime_unchanged"] = summary.get("runtime_before") == summary["runtime_after"]
            summary["processes_final"] = process_inventory(base_env)
        except BaseException as evidence_exc:
            summary["evidence_error"] = f"{type(evidence_exc).__name__}: {evidence_exc}"
        raise
    finally:
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("ISAAC_GRAPHICS_GATE_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"ISAAC_GRAPHICS_GATE_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
