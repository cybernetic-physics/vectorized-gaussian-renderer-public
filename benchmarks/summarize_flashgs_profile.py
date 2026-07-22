#!/usr/bin/env python3
"""Normalize a bounded Nsight Systems control against its unprofiled run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.benchmark_manifest import file_sha256  # noqa: E402
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PROFILE_SCHEMA,
    RENDERER_RUN_SCHEMA,
    same_artifact,
    source_identity,
    verify_node_occupancy_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unprofiled-run", type=Path, required=True)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--nsys-summary", type=Path, required=True)
    parser.add_argument("--nsys-report", type=Path, required=True)
    parser.add_argument("--nsys-sqlite", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--wrapper-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path, schema: str | int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != schema:
        raise ValueError(f"Unexpected schema in {path}: {payload.get('schema_version')}.")
    return payload


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty profiler artifact: {path}.")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def command_flag_value(command: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"Profile command must contain one {flag} value.")
    return command[positions[0] + 1]


def verify_recorded_artifact(
    record: dict[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    path = Path(str((record or {}).get("path", "")))
    actual = artifact(path)
    if not same_artifact(record, actual):
        raise ValueError(f"{label} differs from its recorded hash.")
    return actual


def main() -> None:
    args = parse_args()
    control = load(args.unprofiled_run, RENDERER_RUN_SCHEMA)
    profile = load(args.profile_run, RENDERER_RUN_SCHEMA)
    nsys = load(args.nsys_summary, 1)
    wrapper = load(
        args.wrapper_evidence,
        "flashgs-profile-wrapper-evidence-v1",
    )
    failures: list[str] = []
    for label, run in (("unprofiled", control), ("profile", profile)):
        equation = run.get("equation_contract") or {}
        if equation.get("gaussian_support_sigma") != (MATCHED_GAUSSIAN_SUPPORT_SIGMA):
            failures.append(f"{label} run does not use pinned gsplat {MATCHED_GAUSSIAN_SUPPORT_SIGMA}-sigma support")
        for field, expected in MATCHED_PROJECTION_RULES.items():
            if equation.get(field) != expected:
                failures.append(f"{label} run projection rule {field} differs")
        try:
            verify_node_occupancy_evidence(
                run.get("environment", {}).get("node_occupancy"),
                expected_gpu_uuid=str(run.get("environment", {}).get("gpu_uuid", "")),
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            failures.append(f"{label} run occupancy evidence failed: {error}")
    for name, control_value, profile_value in (
        ("renderer", control["renderer"], profile["renderer"]),
        (
            "output_contract",
            control["output_contract"],
            profile["output_contract"],
        ),
        (
            "scene_sha256",
            control["scene"]["sha256"],
            profile["scene"]["sha256"],
        ),
        (
            "trajectory_id",
            control["camera_contract"]["trajectory_id"],
            profile["camera_contract"]["trajectory_id"],
        ),
        (
            "batch",
            control["camera_contract"]["batch"],
            profile["camera_contract"]["batch"],
        ),
        (
            "gpu_uuid",
            control["environment"]["gpu_uuid"],
            profile["environment"]["gpu_uuid"],
        ),
        (
            "source_git_commit",
            control["environment"]["source_git_commit"],
            profile["environment"]["source_git_commit"],
        ),
        (
            "source_identity",
            source_identity(control["environment"]["source_provenance"]),
            source_identity(profile["environment"]["source_provenance"]),
        ),
        (
            "runtime_identity",
            {
                key: control["environment"].get(key)
                for key in (
                    "gpu_name",
                    "gpu_uuid",
                    "driver",
                    "torch",
                    "cuda_runtime",
                    "compute_capability",
                    "torch_cuda_arch_list",
                )
            },
            {
                key: profile["environment"].get(key)
                for key in (
                    "gpu_name",
                    "gpu_uuid",
                    "driver",
                    "torch",
                    "cuda_runtime",
                    "compute_capability",
                    "torch_cuda_arch_list",
                )
            },
        ),
        (
            "equation_contract",
            control["equation_contract"],
            profile["equation_contract"],
        ),
        (
            "kernel_runner_config",
            {
                key: control.get("runner_config", {}).get(key)
                for key in (
                    "custom_depth_buckets",
                    "custom_depth_bucket_group",
                    "custom_max_physical_views",
                    "semantic_topology",
                )
            },
            {
                key: profile.get("runner_config", {}).get(key)
                for key in (
                    "custom_depth_buckets",
                    "custom_depth_bucket_group",
                    "custom_max_physical_views",
                    "semantic_topology",
                )
            },
        ),
        (
            "fixed_capacity",
            control["capacity"]["installed"],
            profile["capacity"]["installed"],
        ),
        (
            "physical_batch",
            control.get("backend_execution", {}).get("physical_batch"),
            profile.get("backend_execution", {}).get("physical_batch"),
        ),
        (
            "native_submissions_per_logical_batch",
            control.get("backend_execution", {}).get("native_submissions_per_logical_batch"),
            profile.get("backend_execution", {}).get("native_submissions_per_logical_batch"),
        ),
    ):
        if control_value != profile_value:
            failures.append(f"{name}: {control_value!r} != {profile_value!r}")
    if not same_artifact(
        control["environment"].get("native_extension"),
        profile["environment"].get("native_extension"),
    ):
        failures.append("loaded native extension differs from unprofiled control")
    if not same_artifact(
        control["environment"].get("flashgs_adapter_attestation"),
        profile["environment"].get("flashgs_adapter_attestation"),
    ):
        failures.append("FlashGS adapter attestation differs from unprofiled control")
    capacity_field = "calibration_artifact" if control.get("renderer") == "custom" else "demand_survey_artifact"
    config_capacity_field = "capacity_calibration" if control.get("renderer") == "custom" else "flashgs_demand_survey"
    capacity_flag = "--capacity-calibration" if control.get("renderer") == "custom" else "--flashgs-demand-survey"
    control_capacity_artifact = (control.get("capacity") or {}).get(capacity_field)
    if not same_artifact(
        control_capacity_artifact,
        (profile.get("capacity") or {}).get(capacity_field),
    ) or not same_artifact(
        control_capacity_artifact,
        (profile.get("runner_config") or {}).get(config_capacity_field),
    ):
        failures.append("profile control capacity artifact differs from unprofiled run")
    if not control.get("pass"):
        failures.append("unprofiled control is not a passing run")
    if not profile.get("pass") or not profile.get("profile_control"):
        failures.append("profile run is not a passing explicit profile control")
    if profile.get("headline_eligible"):
        failures.append("profile timing was incorrectly marked headline eligible")
    if nsys.get("limitations"):
        failures.extend(str(item) for item in nsys["limitations"])
    kernel_summary = nsys.get("kernel_summary")
    if not kernel_summary:
        failures.append("Nsight summary has no CUDA kernel rows")
        total_kernel_time_ms = 0.0
    else:
        total_kernel_time_ms = float(kernel_summary["total_kernel_time_ms"])
        if total_kernel_time_ms <= 0.0:
            failures.append("Nsight reported non-positive CUDA kernel time")
    captured_frames = int(profile["camera_contract"]["measured_frames"])
    if captured_frames != 3:
        failures.append(f"profile captured {captured_frames} measured frames, expected fixed control length 3")
    if wrapper.get("pass") is not True:
        failures.append("profile wrapper evidence did not pass")
    wrapper_artifacts = wrapper.get("artifacts") or {}
    for name in (
        "occupancy_preflight",
        "profile_exit_status",
        "profile_control",
        "nsys_version",
        "source_script",
    ):
        try:
            verify_recorded_artifact(
                wrapper_artifacts.get(name),
                label=f"profile wrapper {name}",
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            failures.append(str(error))
    if not same_artifact(
        wrapper_artifacts.get("profile_control"),
        artifact(args.profile_run),
    ):
        failures.append("profile wrapper target result differs")
    try:
        wrapper_occupancy_path = Path(str((wrapper_artifacts.get("occupancy_preflight") or {}).get("path", "")))
        wrapper_occupancy = json.loads(wrapper_occupancy_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        failures.append(f"profile wrapper occupancy is unavailable: {error}")
    else:
        wrapper_lock = (wrapper_occupancy.get("executor_control") or {}).get("cooperative_node_wide_lock") or {}
        if (
            wrapper_occupancy.get("schema_version") != "flashgs-matched-occupancy-preflight-v2"
            or wrapper_occupancy.get("expected_gpu_uuid") != control["environment"]["gpu_uuid"]
            or wrapper_occupancy.get("pass") is not True
            or wrapper_lock.get("pass") is not True
        ):
            failures.append("profile wrapper occupancy contract differs")
    try:
        profile_exit_status = int(
            Path(str((wrapper_artifacts.get("profile_exit_status") or {}).get("path", "")))
            .read_text(encoding="utf-8")
            .strip()
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        failures.append(f"profile exit status is unavailable: {error}")
    else:
        if profile_exit_status != 0:
            failures.append(f"profile wrapper target exited {profile_exit_status}")
    command = wrapper.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        failures.append("profile wrapper command is malformed")
        command = []
    try:
        if command[:2] != ["nsys", "profile"]:
            raise ValueError("Profile command is not an Nsight profile command.")
        expected_capture = (
            f"flashgs-matched-capture/{control['renderer']}/"
            f"{control['output_contract']}/"
            f"b{control['camera_contract']['batch']}"
        )
        required_literals = {
            "--trace=cuda,nvtx,osrt",
            "--capture-range=nvtx",
            "--capture-range-end=stop",
            f"--nvtx-capture={expected_capture}",
            "--profile-control",
            "--no-capture-last-output",
        }
        missing_literals = sorted(required_literals - set(command))
        if missing_literals:
            raise ValueError("Profile command lacks required arguments: " + ", ".join(missing_literals))
        runner_path = str(PROJECT_ROOT / "benchmarks/run_flashgs_matched.py")
        if command.count(runner_path) != 1:
            raise ValueError("Profile command runner path differs.")
        expected_flags = {
            "--renderer": str(control["renderer"]),
            "--output-contract": str(control["output_contract"]),
            "--trajectory": str(control["camera_contract"]["path"]),
            "--expected-gpu-uuid": str(control["environment"]["gpu_uuid"]),
            capacity_flag: str((control_capacity_artifact or {}).get("path", "")),
            "--warmup-frames": "8",
            "--measured-frames": "3",
            "--output": str(args.profile_run),
        }
        for flag, expected in expected_flags.items():
            if command_flag_value(command, flag) != expected:
                raise ValueError(f"Profile command {flag} differs.")
    except ValueError as error:
        failures.append(str(error))
    wrapper_environment = wrapper.get("environment") or {}
    if wrapper_environment.get("TORCH_CUDA_ARCH_LIST") != profile["environment"].get("torch_cuda_arch_list"):
        failures.append("profile wrapper CUDA architecture differs")
    native_path = Path(str(profile["environment"]["native_extension"]["path"])).resolve()
    if Path(str(wrapper_environment.get("VGR_NATIVE_BUILD_ROOT", ""))).resolve() != native_path.parent.parent:
        failures.append("profile wrapper native build root differs")
    for name in ("PROJECT_ROOT", "VGR_PROJECT_ROOT"):
        if Path(str(wrapper_environment.get(name, ""))).resolve() != PROJECT_ROOT:
            failures.append(f"profile wrapper {name} differs")
    if wrapper_environment.get("PROFILE_MEASURED_FRAMES") != "3":
        failures.append("profile wrapper measured-frame control differs")
    if Path(str(wrapper_environment.get("PROFILE_ROOT", ""))).resolve() != (args.profile_run.resolve().parent):
        failures.append("profile wrapper result-owned profile root differs")
    if not str(wrapper.get("nsys_version", "")).strip():
        failures.append("profile wrapper Nsight version is empty")
    csv_artifacts = [artifact(path) for path in sorted(args.stats_dir.glob("*.csv"))]
    if not csv_artifacts:
        failures.append("Nsight stats export produced no CSV files")
    cuda_api_summary = nsys.get("cuda_api_summary")
    if not isinstance(cuda_api_summary, dict):
        failures.append("Nsight CUDA API summary is unavailable")
        cuda_api_summary = {}
    api_totals = cuda_api_summary.get("totals") or {}
    total_api_calls = int(api_totals.get("calls", 0))
    if total_api_calls <= 0:
        failures.append("Nsight reported no CUDA API calls")
    kernel_launch_summary = cuda_api_summary.get("kernel_launch") or {}
    kernel_launch_calls = int(kernel_launch_summary.get("calls", 0))
    if kernel_launch_calls <= 0:
        failures.append("Nsight reported no CUDA kernel-launch API calls")
    if nsys.get("allocator_classification") != "cuda-runtime-driver-vmm-v2":
        failures.append("Nsight allocator classification is incomplete")
    allocator_summary = cuda_api_summary.get("allocator") or {
        "calls": 0.0,
        "time_ms": 0.0,
        "time_ns": 0.0,
    }
    allocator_calls = int(allocator_summary.get("calls", 0))
    if allocator_calls != 0:
        failures.append(f"Nsight measured range contains {allocator_calls} allocator calls")
    capture_range = (
        f"flashgs-matched-capture/{control['renderer']}/"
        f"{control['output_contract']}/"
        f"b{control['camera_contract']['batch']}"
    )
    frame_prefix = (
        f"flashgs-matched/{control['renderer']}/"
        f"{control['output_contract']}/"
        f"b{control['camera_contract']['batch']}/frame-"
    )
    nvtx_summary = nsys.get("nvtx_summary")
    if not isinstance(nvtx_summary, dict) or not isinstance(nvtx_summary.get("ranges"), list):
        failures.append("Nsight NVTX summary is unavailable")
        nvtx_ranges: list[dict[str, Any]] = []
    else:
        nvtx_ranges = nvtx_summary["ranges"]
    outer_calls = sum(int(row.get("calls", 0)) for row in nvtx_ranges if row.get("name") == capture_range)
    frame_calls = sum(
        int(row.get("calls", 0)) for row in nvtx_ranges if str(row.get("name", "")).startswith(frame_prefix)
    )
    if outer_calls != 1:
        failures.append(f"Nsight outer capture range count is {outer_calls}, expected 1")
    if frame_calls != captured_frames:
        failures.append(f"Nsight measured frame range count is {frame_calls}, expected {captured_frames}")
    result = {
        "schema_version": PROFILE_SCHEMA,
        "pass": not failures,
        "failures": failures,
        "renderer": control["renderer"],
        "output_contract": control["output_contract"],
        "batch": control["camera_contract"]["batch"],
        "captured_frames": captured_frames,
        "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        "projection_contract": MATCHED_PROJECTION_RULES,
        "capture_range": capture_range,
        "hardware_scope": {
            "gpu_name": control["environment"]["gpu_name"],
            "gpu_uuid": control["environment"]["gpu_uuid"],
            "unprofiled_headline_eligible": bool(control.get("headline_eligible")),
            "profile_timing_headline_eligible": False,
        },
        "unprofiled": {
            "run": str(args.unprofiled_run),
            "run_artifact": artifact(args.unprofiled_run),
            "gpu_pipeline_elapsed_ms": control["timing"]["gpu_batch_ms"]["mean"],
            "host_submission_ms": control["timing"]["host_submission_ms"]["mean"],
            "driver_process_memory_growth_mib": control["memory"].get("driver_process_memory_growth_mib"),
        },
        "profiled_control": {
            "run": str(args.profile_run),
            "run_artifact": artifact(args.profile_run),
            "gpu_pipeline_elapsed_ms": profile["timing"]["gpu_batch_ms"]["mean"],
            "total_cuda_kernel_time_ms": total_kernel_time_ms,
            "cuda_kernel_time_per_frame_ms": (total_kernel_time_ms / captured_frames if captured_frames else None),
            "kernel_stages": (kernel_summary["stages"] if kernel_summary else None),
            "top_kernels": (kernel_summary["top_kernels"] if kernel_summary else None),
            "cuda_api_summary": cuda_api_summary,
            "total_cuda_api_calls": total_api_calls,
            "kernel_launch_api_calls": kernel_launch_calls,
            "driver_process_memory_growth_mib": profile["memory"].get("driver_process_memory_growth_mib"),
            "torch_allocation_growth_bytes": profile["memory"].get("allocation_growth_bytes"),
            "torch_reservation_growth_bytes": profile["memory"].get("reservation_growth_bytes"),
            "measured_range_allocator_calls": allocator_calls,
            "measured_range_allocator_time_ms": float(allocator_summary.get("time_ms", 0.0)),
            "no_measured_range_allocator_calls": allocator_calls == 0,
            "nvtx_outer_capture_calls": outer_calls,
            "nvtx_measured_frame_calls": frame_calls,
        },
        "artifacts": {
            "nsys_report": artifact(args.nsys_report),
            "sqlite": artifact(args.nsys_sqlite),
            "stats_csv": csv_artifacts,
            "parsed_summary": artifact(args.nsys_summary),
            "wrapper_evidence": artifact(args.wrapper_evidence),
            "wrapper_occupancy_preflight": wrapper_artifacts.get("occupancy_preflight"),
            "profile_exit_status": wrapper_artifacts.get("profile_exit_status"),
            "nsys_version": wrapper_artifacts.get("nsys_version"),
            "profile_wrapper_script": wrapper_artifacts.get("source_script"),
        },
        "interpretation": (
            "Nsight kernel sums are normalized over the bounded NVTX capture. "
            "They diagnose GPU work and are not substituted for unprofiled "
            "CUDA-event batch latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "FLASHGS_MATCHED_PROFILE_OK "
        + json.dumps(
            {"output": str(args.output), "pass": result["pass"]},
            sort_keys=True,
        )
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
