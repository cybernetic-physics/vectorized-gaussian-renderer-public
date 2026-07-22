#!/usr/bin/env python3
"""Bind an Nsight wrapper invocation to its exact bounded artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    artifact_record,
)


SAFE_ENVIRONMENT_KEYS = (
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "FLASHGS_SOURCE_PATH",
    "HOME_SCAN_PATH",
    "LD_LIBRARY_PATH",
    "NSYS_NVTX_PROFILER_REGISTER_ONLY",
    "PATH",
    "PROFILE_MEASURED_FRAMES",
    "PROFILE_ROOT",
    "PROJECT_ROOT",
    "PYTHONPATH",
    "TORCH_CUDA_ARCH_LIST",
    "VGR_GPU_EXECUTOR_LOCK_OWNER_PID",
    "VGR_GPU_EXECUTOR_LOCK_PATH",
    "VGR_NATIVE_BUILD_ROOT",
    "VGR_PROJECT_ROOT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occupancy-preflight", type=Path, required=True)
    parser.add_argument("--profile-exit-status", type=Path, required=True)
    parser.add_argument("--profile-control", type=Path, required=True)
    parser.add_argument("--nsys-version", type=Path, required=True)
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Exact profiled command following --.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("the exact profiled command is required after --")
    return args


def main() -> None:
    args = parse_args()
    occupancy = json.loads(args.occupancy_preflight.read_text(encoding="utf-8"))
    exit_status_raw = args.profile_exit_status.read_text(encoding="utf-8").strip()
    try:
        exit_status = int(exit_status_raw)
    except ValueError as error:
        raise ValueError("Nsight profile exit status is not an integer.") from error
    nsys_version = args.nsys_version.read_text(encoding="utf-8").strip()
    executor_control = occupancy.get("executor_control") or {}
    cooperative_lock = executor_control.get("cooperative_node_wide_lock") or {}
    failures: list[str] = []
    if (
        occupancy.get("schema_version") != "flashgs-matched-occupancy-preflight-v2"
        or occupancy.get("pass") is not True
        or executor_control.get("scope") != "all-visible-gpus"
        or cooperative_lock.get("pass") is not True
    ):
        failures.append("profile occupancy preflight did not pass")
    if exit_status != 0:
        failures.append(f"Nsight profile target exited {exit_status}")
    if not nsys_version:
        failures.append("Nsight Systems version is empty")
    if not args.profile_control.is_file():
        failures.append("profile control result is missing")
    environment = {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}
    result = {
        "schema_version": "flashgs-profile-wrapper-evidence-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pass": not failures,
        "failures": failures,
        "cwd": str(Path.cwd().resolve()),
        "command": list(args.command),
        "safe_environment_allowlist": list(SAFE_ENVIRONMENT_KEYS),
        "environment": environment,
        "nsys_version": nsys_version,
        "artifacts": {
            "occupancy_preflight": artifact_record(args.occupancy_preflight),
            "profile_exit_status": artifact_record(args.profile_exit_status),
            "profile_control": (artifact_record(args.profile_control) if args.profile_control.is_file() else None),
            "nsys_version": artifact_record(args.nsys_version),
            "source_script": artifact_record(args.source_script),
        },
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
