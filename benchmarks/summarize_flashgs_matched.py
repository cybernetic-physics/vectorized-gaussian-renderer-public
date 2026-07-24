#!/usr/bin/env python3
"""Fail-closed summary for the decisive custom-versus-FlashGS matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from benchmarks.flashgs_matched_memory import (  # noqa: E402
    TORCH_CUMULATIVE_MEMORY_COUNTERS,
)
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    CAPACITY_CALIBRATION_SCHEMA,
    CAPACITY_CONSUMPTION_SCHEMA,
    FIDELITY_SCHEMA,
    FLASHGS_DEMAND_SURVEY_CONSUMPTION_SCHEMA,
    FLASHGS_DEMAND_SURVEY_SCHEMA,
    GSPLAT_ORACLE_SCHEMA,
    HEADLINE_GPU_NAME,
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PRIMARY_BATCHES,
    PRIMARY_CHUNKED_PHYSICAL_VIEWS,
    PRIMARY_TRAJECTORY_IDS,
    PROFILE_SCHEMA,
    RENDERER_RUN_SCHEMA,
    SUMMARY_SCHEMA,
    TIMED_CAPACITY_VERIFICATION_SCHEMA,
    artifact_record,
    derive_flashgs_prefix_capacities,
    is_nvidia_gpu_uuid,
    primary_execution_schedule_failures,
    primary_fidelity_selection,
    primary_max_physical_views,
    same_artifact,
    source_identity,
    verify_flashgs_demand_counter_source_audit,
    verify_gsplat_oracle_support_evidence,
    verify_node_occupancy_evidence,
    verify_trajectory_prefix_proof,
)
from isaacsim_gaussian_renderer.evaluation.matched_semantics import (  # noqa: E402
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
)
from isaacsim_gaussian_renderer.fidelity.metrics import (  # noqa: E402
    validate_fidelity_report,
)
from isaacsim_gaussian_renderer.flashgs_baseline_contract import (  # noqa: E402
    FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
    require_flashgs_matched_port_classification,
)

FROZEN_EXECUTION_SCHEDULE = (
    "under the frozen execution schedule (Custom: direct B1-B256, P128x4 at "
    "B512, and P128x8 at B1024; FlashGS-derived: P1xB)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Rebuild every gate and compare with --output without writing files.",
    )
    return parser.parse_args()


def load(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != schema:
        raise ValueError(f"Unexpected schema in {path}: {payload.get('schema_version')}.")
    return payload


def command_flag_value(command: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"command must contain exactly one {flag} value")
    return command[positions[0] + 1]


def expected_matrix_capacity_calibration_schedule(
    batches: tuple[int, ...],
) -> list[dict[str, int | str]]:
    ordered_batches = [batch for batch in (1024, 512) if batch in batches]
    ordered_batches.extend(batch for batch in batches if batch not in (1024, 512))
    return [{"batch": batch, "renderer": "custom"} for batch in ordered_batches]


def validated_timing_distribution(
    run: dict[str, Any],
    key: str,
    *,
    label: str,
    expected_count: int = 100,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Recompute and validate a stored timing distribution from raw samples."""

    try:
        stored = run["timing"][key]
        samples = np.asarray(stored["samples"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} {key} timing distribution is malformed") from error
    if samples.shape != (expected_count,):
        raise ValueError(f"{label} {key} requires {expected_count} ordered samples; found shape {samples.shape}.")
    if not np.isfinite(samples).all() or np.any(samples <= 0.0):
        raise ValueError(f"{label} {key} samples are not finite and positive.")
    recomputed: dict[str, float | int] = {
        "count": int(samples.size),
        "mean": float(samples.mean()),
        "stddev": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
        "min": float(samples.min()),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "max": float(samples.max()),
    }
    for field, expected in recomputed.items():
        actual = stored.get(field)
        if field == "count":
            matches = actual == expected
        else:
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual),
                float(expected),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        if not matches:
            raise ValueError(f"{label} {key} stored {field}={actual!r} does not match raw-sample value {expected!r}.")
    return samples, recomputed


@lru_cache(maxsize=None)
def load_verified_oracle_manifest(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    payload = load(path, GSPLAT_ORACLE_SCHEMA)
    verify_gsplat_oracle_support_evidence(payload)
    return payload


def steady_process_gpu_bytes(run: dict[str, Any]) -> tuple[int, str]:
    driver_samples = [
        run["memory"][key].get("current_process_used_mib")
        for key in (
            "driver_process_memory_before",
            "driver_process_memory_after",
        )
    ]
    available_driver_samples = [int(value) for value in driver_samples if value is not None]
    if available_driver_samples:
        return (
            max(available_driver_samples) * 1024 * 1024,
            "max-pre-post-nvidia-smi-process-endpoint",
        )
    return int(run["memory"]["torch_peak_allocated_bytes"]), "torch-peak-allocated-fallback"


def capacity_artifact_binding_failures(
    renderer: str,
    run: dict[str, Any],
) -> list[str]:
    """Verify the exact renderer-specific capacity protocol and direct audit."""

    failures: list[str] = []
    capacity = run.get("capacity") or {}
    config = run.get("runner_config") or {}
    expected_setup = {
        (
            "initial_workspace_from_calibration" if renderer == "custom" else "initial_workspace_from_demand_survey"
        ): True,
        "prepare_outputs_calls": 1,
        "warmup_frames": 8,
        "trajectory_preflight_frames": 0,
        "explicit_capacity_reservation_calls": 0,
        "empty_cache_calls": 0,
    }
    if capacity.get("timed_setup") != expected_setup or config.get("premeasurement_schedule") != expected_setup:
        failures.append(f"{renderer} timed capacity/setup contract differs")
    capacity_record = capacity.get("calibration_artifact" if renderer == "custom" else "demand_survey_artifact")
    config_record = config.get("capacity_calibration" if renderer == "custom" else "flashgs_demand_survey")
    unexpected_config = config.get("flashgs_demand_survey" if renderer == "custom" else "capacity_calibration")
    if not same_artifact(capacity_record, config_record):
        failures.append(f"{renderer} capacity-artifact references disagree")
    if unexpected_config is not None:
        failures.append(f"{renderer} runner recorded the wrong capacity-artifact kind")
    artifact_path = Path(str((capacity_record or {}).get("path", "")))
    try:
        actual_record = artifact_record(artifact_path)
        artifact = load(
            artifact_path,
            CAPACITY_CALIBRATION_SCHEMA if renderer == "custom" else FLASHGS_DEMAND_SURVEY_SCHEMA,
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        failures.append(f"{renderer} capacity artifact unavailable: {error}")
        return failures
    if not same_artifact(capacity_record, actual_record):
        failures.append(f"{renderer} capacity artifact hash differs")
    if artifact.get("pass") is not True or artifact.get("renderer") != renderer:
        failures.append(f"{renderer} capacity artifact identity differs")
    camera = artifact.get("camera_contract") or {}
    run_camera = run.get("camera_contract") or {}
    if (artifact.get("scene") or {}).get("sha256") != (run.get("scene") or {}).get("sha256"):
        failures.append(f"{renderer} capacity artifact scene differs")
    if (artifact.get("equation_contract") or {}).get("gaussian_support_sigma") != (
        run.get("equation_contract") or {}
    ).get("gaussian_support_sigma"):
        failures.append(f"{renderer} capacity artifact support differs")
    artifact_equation = artifact.get("equation_contract") or {}
    run_equation = run.get("equation_contract") or {}
    if artifact_equation.get("semantic_topology") != run_equation.get("semantic_topology") or any(
        artifact_equation.get(field) != run_equation.get(field) for field in MATCHED_PROJECTION_RULES
    ):
        failures.append(f"{renderer} capacity artifact equation differs")
    artifact_environment = artifact.get("environment") or {}
    run_environment = run.get("environment") or {}
    runtime_fields = (
        "gpu_name",
        "gpu_uuid",
        "driver",
        "torch",
        "cuda_runtime",
        "compute_capability",
        "torch_cuda_arch_list",
    )
    if any(artifact_environment.get(field) != run_environment.get(field) for field in runtime_fields):
        failures.append(f"{renderer} capacity artifact runtime differs")
    if source_identity(artifact_environment.get("source_provenance") or {}) != source_identity(
        run_environment.get("source_provenance") or {}
    ):
        failures.append(f"{renderer} capacity artifact source differs")
    for field in ("native_extension", "flashgs_adapter_attestation"):
        if not same_artifact(artifact_environment.get(field), run_environment.get(field)):
            failures.append(f"{renderer} capacity artifact {field} differs")
    if artifact_environment.get("native_build_contract") != run_environment.get("native_build_contract"):
        failures.append(f"{renderer} capacity artifact build contract differs")
    try:
        verify_node_occupancy_evidence(
            artifact_environment.get("node_occupancy"),
            expected_gpu_uuid=str(run_environment.get("gpu_uuid", "")),
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        failures.append(f"{renderer} capacity artifact occupancy unavailable: {error}")
    artifact_config = artifact.get("calibration_config") or artifact.get("survey_config") or {}
    expected_max_physical_views = primary_max_physical_views(renderer, int(run_camera.get("batch", 0)))
    for field, expected in (
        ("capacity_headroom", 1.05),
        ("initial_visible_per_view", 170_000),
        ("initial_intersections_per_view", 200_000),
        ("flashgs_initial_intersections", 200_000),
        ("custom_depth_buckets", 128),
        ("custom_depth_bucket_group", 8),
        ("custom_max_physical_views", expected_max_physical_views),
    ):
        if artifact_config.get(field) != expected:
            failures.append(f"{renderer} capacity artifact config {field} differs")

    if renderer == "custom":
        calibrated_capacity = artifact.get("capacity") or {}
        calibration_attempts = calibrated_capacity.get("calibration_attempts")
        if (
            artifact.get("mode") != "capacity-calibration-only"
            or artifact.get("calibration_output_contract") != "full"
            or camera.get("trajectory_id") != run_camera.get("trajectory_id")
            or camera.get("batch") != run_camera.get("batch")
            or camera.get("timesteps")
            != int(run_camera.get("warmup_frames", 0)) + int(run_camera.get("measured_frames", 0))
            or camera.get("measured_start") != run_camera.get("warmup_frames")
            or capacity.get("schema_version") != CAPACITY_CONSUMPTION_SCHEMA
            or capacity.get("capacity_source") != "hash-bound-calibration-artifact"
            or calibrated_capacity.get("schema_version") != "flashgs-matched-capacity-v1"
            or calibrated_capacity.get("headroom") != 1.05
            or not isinstance(calibrated_capacity.get("initial_preflight"), dict)
            or not isinstance(calibration_attempts, list)
            or not calibration_attempts
            or calibrated_capacity.get("preflight_passes") != 1 + len(calibration_attempts or [])
            or calibrated_capacity.get("capacity_source") != "full-trajectory-device-counter-calibration"
            or calibrated_capacity.get("preflight_frames_per_pass") != 108
            or calibrated_capacity.get("preflight_measured_frames_per_pass") != 100
            or calibrated_capacity.get("installed") != capacity.get("installed")
            or calibrated_capacity.get("verified_preflight") != capacity.get("verified_preflight")
            or calibrated_capacity.get("intersection_capacity_scope") != capacity.get("intersection_capacity_scope")
        ):
            failures.append("custom consumed capacity differs from calibration")
    else:
        proof = artifact.get("trajectory_prefix_proof") or {}
        observation = artifact.get("demand_observation") or {}
        visible = observation.get("per_camera_max_visible_gaussians")
        generated = observation.get("per_camera_max_generated_intersections")
        overflow = observation.get("per_camera_max_intersection_overflow")
        try:
            verify_flashgs_demand_counter_source_audit(
                artifact.get("counter_source_audit") or {},
                project_root=PROJECT_ROOT,
            )
            verified_proof = verify_trajectory_prefix_proof(proof)
            derived = derive_flashgs_prefix_capacities(
                visible,
                generated,
                batches=PRIMARY_BATCHES,
                headroom=1.05,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            failures.append(f"flashgs demand-survey proof unavailable: {error}")
            return failures
        batch = int(run_camera.get("batch", 0))
        selected = derived.get(str(batch)) or {}
        proof_contract = (verified_proof.get("contracts") or {}).get(str(batch)) or {}
        if (
            artifact.get("mode") != "flashgs-demand-survey-only"
            or artifact.get("timing_valid") is not False
            or artifact.get("render_outputs_valid") is not False
            or camera.get("batch") != PRIMARY_BATCHES[-1]
            or camera.get("trajectory_id") != PRIMARY_TRAJECTORY_IDS[PRIMARY_BATCHES[-1]]
            or proof_contract.get("trajectory_id") != run_camera.get("trajectory_id")
            or not isinstance(visible, list)
            or not isinstance(generated, list)
            or not isinstance(overflow, list)
            or len(visible) != PRIMARY_BATCHES[-1]
            or len(generated) != PRIMARY_BATCHES[-1]
            or len(overflow) != PRIMARY_BATCHES[-1]
            or artifact.get("derived_batch_capacities") != derived
            or capacity.get("schema_version") != FLASHGS_DEMAND_SURVEY_CONSUMPTION_SCHEMA
            or capacity.get("capacity_source") != "hash-bound-b1024-demand-survey-batch-prefix"
            or capacity.get("survey_prefix_demand") != selected
            or capacity.get("installed_from_batch_specific_prefix") != batch
            or capacity.get("survey_canonical_batch") != PRIMARY_BATCHES[-1]
            or capacity.get("survey_render_outputs_valid") is not False
            or capacity.get("installed")
            != {
                "visible_records": None,
                "intersections": selected.get("installed_intersections_per_camera"),
            }
        ):
            failures.append("flashgs consumed capacity differs from B1024 demand survey")

    verification = capacity.get("timed_verification") or {}
    observed = verification.get("observed") or {}
    if (
        verification.get("schema_version") != TIMED_CAPACITY_VERIFICATION_SCHEMA
        or verification.get("pass") is not True
        or verification.get("source") != "renderer-device-counters"
        or verification.get("frames_observed") != 108
        or verification.get("warmup_frames_observed") != 8
        or verification.get("measured_frames_observed") != 100
        or verification.get("device_max_updates") != 108
        or verification.get("cpu_readbacks_in_measured_loop") != 0
        or verification.get("final_cpu_readback_after_measurement") is not True
        or verification.get("audit_updates_in_cuda_event_timing") is not False
        or verification.get("audit_updates_in_wall_samples") is not False
        or observed.get("max_visible_overflow", 0) != 0
        or observed.get("max_intersection_overflow") != 0
    ):
        failures.append(f"{renderer} direct timed zero-overflow verification differs")
    return failures


# Retain the import name used by older analysis helpers while applying the new
# renderer-specific protocol.
capacity_calibration_binding_failures = capacity_artifact_binding_failures


def profile_metrics(
    root: Path,
    *,
    renderer: str,
    contract: str,
    batch: int,
    run_path: Path,
) -> dict[str, Any] | None:
    path = root / "profiles" / f"flashgs-matched-{renderer}-{contract}-b{batch}" / "profile-summary.json"
    if not path.is_file():
        return None
    payload = load(path, PROFILE_SCHEMA)
    if not payload.get("pass"):
        raise RuntimeError(f"Profiler control is not passing: {path}.")
    if (
        payload.get("gaussian_support_sigma") != (MATCHED_GAUSSIAN_SUPPORT_SIGMA)
        or payload.get("projection_contract") != MATCHED_PROJECTION_RULES
    ):
        raise RuntimeError(f"Profiler equation contract differs: {path}.")
    if payload["renderer"] != renderer or payload["output_contract"] != contract or int(payload["batch"]) != batch:
        raise RuntimeError(f"Profiler control identity mismatch: {path}.")
    if not same_artifact(
        payload.get("unprofiled", {}).get("run_artifact"),
        artifact_record(run_path),
    ):
        raise RuntimeError(f"Profiler control is stale for the unprofiled run: {path}.")
    raw_records = [
        payload.get("profiled_control", {}).get("run_artifact"),
        payload.get("artifacts", {}).get("nsys_report"),
        payload.get("artifacts", {}).get("sqlite"),
        payload.get("artifacts", {}).get("parsed_summary"),
        payload.get("artifacts", {}).get("wrapper_evidence"),
        payload.get("artifacts", {}).get("wrapper_occupancy_preflight"),
        payload.get("artifacts", {}).get("profile_exit_status"),
        payload.get("artifacts", {}).get("nsys_version"),
        payload.get("artifacts", {}).get("profile_wrapper_script"),
        *(payload.get("artifacts", {}).get("stats_csv") or []),
    ]
    if not raw_records:
        raise RuntimeError(f"Profiler raw artifact records are missing: {path}.")
    for record in raw_records:
        record_path = Path(str((record or {}).get("path", "")))
        if not record_path.is_file() or not same_artifact(
            record,
            artifact_record(record_path),
        ):
            raise RuntimeError(f"Profiler raw artifact changed or is missing: {record_path}.")
    return {
        "artifact": artifact_record(path),
        "captured_frames": payload["captured_frames"],
        "cuda_kernel_time_per_frame_ms": payload["profiled_control"]["cuda_kernel_time_per_frame_ms"],
        "measured_range_allocator_calls": payload["profiled_control"]["measured_range_allocator_calls"],
    }


def steady_state_fairness_failures(
    renderer: str,
    run: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    memory = run.get("memory")
    if not isinstance(memory, dict):
        return [f"{renderer} memory evidence is unavailable"]
    fairness = memory.get("steady_state_fairness")
    if not isinstance(fairness, dict):
        return [f"{renderer} strict steady-state fairness evidence is unavailable"]

    reported_failures = fairness.get("failures")
    if fairness.get("pass") is not True:
        failures.append(f"{renderer} strict steady-state fairness pass=false")
        if isinstance(reported_failures, list):
            failures.extend(f"{renderer} steady-state fairness: {reason}" for reason in reported_failures)

    baseline = fairness.get("baseline")
    if not isinstance(baseline, dict):
        failures.append(f"{renderer} stable memory baseline evidence is unavailable")
    else:
        required_count = baseline.get("required_sample_count")
        sample_count = baseline.get("sample_count")
        if (
            not isinstance(required_count, int)
            or required_count < 3
            or not isinstance(sample_count, int)
            or sample_count < required_count
        ):
            failures.append(f"{renderer} has fewer than three required baseline samples")
        for field, label in (
            ("nvml_available", "baseline NVML sample"),
            ("torch_current_state_available", "baseline Torch allocator state"),
            (
                "torch_cumulative_counters_available",
                "baseline Torch cumulative counters",
            ),
            ("driver_memory_stable", "NVML baseline stability"),
            ("torch_memory_stable", "Torch baseline stability"),
            ("stable", "combined memory baseline stability"),
        ):
            if baseline.get(field) is not True:
                failures.append(f"{renderer} {label} is not established")

    post_measurement = fairness.get("post_measurement")
    if not isinstance(post_measurement, dict):
        failures.append(f"{renderer} post-measurement memory evidence is unavailable")
    else:
        for field, label in (
            ("nvml_available", "post-measurement NVML sample"),
            (
                "torch_current_state_available",
                "post-measurement Torch allocator state",
            ),
            (
                "torch_cumulative_counters_available",
                "post-measurement Torch cumulative counters",
            ),
        ):
            if post_measurement.get(field) is not True:
                failures.append(f"{renderer} {label} is unavailable")

    for field, label, unit in (
        ("allocation_growth_bytes", "Torch allocation", "bytes"),
        ("reservation_growth_bytes", "Torch reservation", "bytes"),
        ("driver_process_memory_growth_mib", "NVML process memory", "MiB"),
    ):
        value = fairness.get(field)
        if value != 0:
            failures.append(f"{renderer} {label} delta is {value!r} {unit}, expected 0")
        if memory.get(field) != value:
            failures.append(f"{renderer} top-level {field} disagrees with fairness evidence")

    counter_deltas = fairness.get("torch_cumulative_counter_deltas")
    if not isinstance(counter_deltas, dict):
        failures.append(f"{renderer} Torch cumulative allocator deltas are unavailable")
    else:
        for name in TORCH_CUMULATIVE_MEMORY_COUNTERS:
            delta = counter_deltas.get(name)
            if delta != 0:
                failures.append(f"{renderer} Torch cumulative counter {name} delta is {delta!r}, expected 0")
        if memory.get("torch_cumulative_counter_deltas") != counter_deltas:
            failures.append(f"{renderer} top-level cumulative deltas disagree with fairness evidence")

    validation = run.get("validation")
    if not isinstance(validation, dict):
        failures.append(f"{renderer} post-measurement validation scope is unavailable")
    else:
        if validation.get("scope") != "post-measurement":
            failures.append(f"{renderer} validation is not labeled post-measurement")
        if validation.get("included_in_cuda_event_timing") is not False:
            failures.append(f"{renderer} validation timing exclusion is not established")
        if validation.get("included_in_memory_snapshots") is not False:
            failures.append(f"{renderer} validation memory exclusion is not established")
        if validation.get("post_measurement_memory_snapshots_completed_before_validation") is not True:
            failures.append(f"{renderer} pre-validation memory snapshot ordering is not established")
    return failures


def compatible(custom: dict[str, Any], flashgs: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    fields = (
        ("scene.sha256", custom["scene"]["sha256"], flashgs["scene"]["sha256"]),
        ("scene.gaussian_count", custom["scene"]["gaussian_count"], flashgs["scene"]["gaussian_count"]),
        (
            "trajectory_id",
            custom["camera_contract"]["trajectory_id"],
            flashgs["camera_contract"]["trajectory_id"],
        ),
        ("batch", custom["camera_contract"]["batch"], flashgs["camera_contract"]["batch"]),
        ("width", custom["camera_contract"]["width"], flashgs["camera_contract"]["width"]),
        ("height", custom["camera_contract"]["height"], flashgs["camera_contract"]["height"]),
        (
            "measured_frames",
            custom["camera_contract"]["measured_frames"],
            flashgs["camera_contract"]["measured_frames"],
        ),
        ("gpu_uuid", custom["environment"]["gpu_uuid"], flashgs["environment"]["gpu_uuid"]),
        ("gpu_name", custom["environment"]["gpu_name"], flashgs["environment"]["gpu_name"]),
        ("driver", custom["environment"]["driver"], flashgs["environment"]["driver"]),
        ("torch", custom["environment"]["torch"], flashgs["environment"]["torch"]),
        (
            "cuda_runtime",
            custom["environment"]["cuda_runtime"],
            flashgs["environment"]["cuda_runtime"],
        ),
        (
            "compute_capability",
            custom["environment"]["compute_capability"],
            flashgs["environment"]["compute_capability"],
        ),
        (
            "torch_cuda_arch_list",
            custom["environment"].get("torch_cuda_arch_list"),
            flashgs["environment"].get("torch_cuda_arch_list"),
        ),
        (
            "source_identity",
            source_identity(custom["environment"]["source_provenance"]),
            source_identity(flashgs["environment"]["source_provenance"]),
        ),
        (
            "flashgs_upstream_commit",
            custom["environment"]["flashgs_upstream_commit"],
            flashgs["environment"]["flashgs_upstream_commit"],
        ),
        ("equation_contract", custom["equation_contract"], flashgs["equation_contract"]),
    )
    for name, left, right in fields:
        if left != right:
            failures.append(f"{name}: {left!r} != {right!r}")
    for renderer, run in (("custom", custom), ("flashgs", flashgs)):
        failures.extend(capacity_calibration_binding_failures(renderer, run))
        if not run.get("pass"):
            failures.append(f"{renderer} run pass=false")
        if run.get("primary_workload_eligible") is not True:
            failures.append(f"{renderer} run is not the frozen primary workload")
        if run["camera_contract"]["measured_frames"] < 100:
            failures.append(f"{renderer} has fewer than 100 measured frames")
        if run["camera_contract"]["unchanged_measured_camera_frame_pairs"] != 0:
            failures.append(f"{renderer} contains unchanged camera poses")
        if run["equation_contract"]["projection_cache"]:
            failures.append(f"{renderer} projection cache enabled")
        if run["equation_contract"]["rendered_frame_cache"]:
            failures.append(f"{renderer} rendered-frame cache enabled")
        if run["equation_contract"].get("gaussian_support_sigma") != (MATCHED_GAUSSIAN_SUPPORT_SIGMA):
            failures.append(
                f"{renderer} Gaussian support cutoff is not the pinned "
                f"gsplat {MATCHED_GAUSSIAN_SUPPORT_SIGMA}-sigma contract"
            )
        for field, expected in MATCHED_PROJECTION_RULES.items():
            if run["equation_contract"].get(field) != expected:
                failures.append(f"{renderer} projection rule {field} is not frozen")
        if run["timing"]["cpu_output_copies_in_measured_loop"] != 0:
            failures.append(f"{renderer} copied outputs in measured loop")
        if run.get("profile_control") is not False:
            failures.append(f"{renderer} primary run is a profile control")
        native_record = run.get("environment", {}).get("native_extension") or {}
        try:
            native_actual = artifact_record(native_record.get("path", ""))
        except (FileNotFoundError, TypeError, ValueError) as error:
            failures.append(f"{renderer} loaded native extension unavailable: {error}")
        else:
            if not same_artifact(native_record, native_actual):
                failures.append(f"{renderer} loaded native extension hash differs")
        build_ninja_record = run.get("environment", {}).get("native_build_ninja") or {}
        try:
            build_ninja_actual = artifact_record(Path(str(build_ninja_record.get("path", ""))))
        except (FileNotFoundError, TypeError, ValueError) as error:
            failures.append(f"{renderer} native build.ninja unavailable: {error}")
        else:
            if not same_artifact(build_ninja_record, build_ninja_actual):
                failures.append(f"{renderer} native build.ninja hash differs")
        compilers = run.get("environment", {}).get("compiler_versions") or {}
        if not compilers.get("cxx") or not compilers.get("nvcc"):
            failures.append(f"{renderer} compiler versions are unavailable")
        if renderer == "flashgs":
            build_contract = run.get("environment", {}).get("native_build_contract") or {}
            if (
                build_contract.get("upstream_commit") != "cdfc4e4002318423eda356eed02df8e01fa32cb6"
                or build_contract.get("cxx_flags") != ["-O3", "-std=c++17"]
                or build_contract.get("cuda_flags") != ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"]
                or build_contract.get("torch_cuda_arch_list") != run["environment"].get("torch_cuda_arch_list")
            ):
                failures.append("FlashGS native build contract differs")
        adapter_record = run.get("environment", {}).get("flashgs_adapter_attestation") or {}
        try:
            adapter_path = Path(str(adapter_record.get("path", "")))
            adapter_actual = artifact_record(adapter_path)
            adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
            adapter_diff_record = adapter.get("adapter_diff") or {}
            adapter_diff_actual = artifact_record(Path(str(adapter_diff_record.get("path", ""))))
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
            failures.append(f"{renderer} FlashGS adapter attestation unavailable: {error}")
        else:
            if not same_artifact(adapter_record, adapter_actual):
                failures.append(f"{renderer} FlashGS adapter attestation hash differs")
            if (
                adapter.get("schema_version") != FLASHGS_ADAPTER_ATTESTATION_SCHEMA
                or adapter.get("upstream_commit") != "cdfc4e4002318423eda356eed02df8e01fa32cb6"
                or adapter.get("pass") is not True
                or source_identity(adapter.get("source_provenance") or {})
                != source_identity(run["environment"].get("source_provenance") or {})
                or not same_artifact(adapter_diff_record, adapter_diff_actual)
            ):
                failures.append(f"{renderer} FlashGS adapter attestation contract differs")
            try:
                require_flashgs_matched_port_classification(adapter.get("baseline_classification"))
            except ValueError as error:
                failures.append(f"{renderer} FlashGS adapter classification differs: {error}")
        try:
            verify_node_occupancy_evidence(
                run.get("environment", {}).get("node_occupancy"),
                expected_gpu_uuid=str(run.get("environment", {}).get("gpu_uuid", "")),
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            failures.append(f"{renderer} occupancy evidence unavailable: {error}")
        config = run.get("runner_config", {})
        for field, expected in (
            ("warmup_frames", 8),
            ("measured_frames", 100),
            ("capacity_headroom", 1.05),
            ("initial_visible_per_view", 170_000),
            ("initial_intersections_per_view", 200_000),
            ("flashgs_initial_intersections", 200_000),
            ("custom_depth_buckets", 128),
            ("custom_depth_bucket_group", 8),
            ("fixed_visible_capacity", None),
            ("fixed_intersection_capacity", None),
            ("capacity_calibration_only", False),
            ("flashgs_demand_survey_only", False),
            ("capture_last_output", True),
            ("independent_trial", 1),
            ("profile_control", False),
        ):
            if config.get(field) != expected:
                failures.append(f"{renderer} runner config {field} is not frozen")
        expected_max_physical_views = primary_max_physical_views(
            renderer,
            int(run["camera_contract"]["batch"]),
        )
        if config.get("custom_max_physical_views") != (expected_max_physical_views):
            failures.append(f"{renderer} internal camera schedule is not frozen")
        provenance = run.get("environment", {}).get("source_provenance", {})
        if provenance.get("dirty") is not False:
            failures.append(f"{renderer} source tree is not clean")
        failures.extend(steady_state_fairness_failures(renderer, run))
        for sample_name in (
            "driver_process_memory_before",
            "driver_process_memory_after",
        ):
            if run["memory"].get(sample_name, {}).get("gpu_uuid") != (run["environment"]["gpu_uuid"]):
                failures.append(f"{renderer} {sample_name} GPU UUID differs from the run")
        timed_observed = run["capacity"]["timed_verification"]["observed"]
        if timed_observed.get("max_visible_overflow", 0):
            failures.append(f"{renderer} visible capacity overflow")
        if timed_observed["max_intersection_overflow"]:
            failures.append(f"{renderer} intersection capacity overflow")
    custom_scope = custom["capacity"].get("intersection_capacity_scope")
    if custom_scope not in (
        "batch-global-workspace",
        "physical-camera-chunk-reused-workspace",
    ):
        failures.append("custom intersection capacity scope is unsupported")
    if flashgs["capacity"].get("intersection_capacity_scope") != ("per-camera-reused-workspace"):
        failures.append("FlashGS intersection capacity scope is not per-camera")
    flashgs_capacity = int(flashgs["capacity"]["installed"]["intersections"])
    flashgs_observed_per_camera = int(
        flashgs["capacity"]["timed_verification"]["observed"]["max_generated_intersections_per_camera"]
    )
    if flashgs_capacity < flashgs_observed_per_camera:
        failures.append("FlashGS per-camera intersection capacity is undersized")
    custom_execution = custom["backend_execution"]
    logical_batch = int(custom["camera_contract"]["batch"])
    physical_batch = custom_execution.get("physical_batch")
    submissions_per_logical_batch = custom_execution.get("native_submissions_per_logical_batch")
    if not isinstance(physical_batch, int) or not (0 < physical_batch <= logical_batch):
        failures.append("custom physical batch is invalid")
    expected_physical_batch = primary_max_physical_views("custom", logical_batch) or logical_batch
    if physical_batch != expected_physical_batch:
        failures.append("custom physical batch is not the frozen schedule")
    if isinstance(physical_batch, int) and physical_batch > 0:
        expected_submissions = math.ceil(logical_batch / physical_batch)
        if submissions_per_logical_batch != expected_submissions:
            failures.append("custom native submission count is inconsistent")
        if custom_execution.get("logical_batch") != logical_batch:
            failures.append("custom execution logical batch is inconsistent")
        if custom["capacity"].get("logical_batch") != logical_batch:
            failures.append("custom capacity logical batch is inconsistent")
        if custom["capacity"].get("physical_batch") != physical_batch:
            failures.append("custom capacity physical batch is inconsistent")
        if custom["capacity"].get("native_submissions_per_logical_batch") != expected_submissions:
            failures.append("custom capacity submission schedule is inconsistent")
        if custom_execution.get("logical_batch_in_single_native_submission") != (expected_submissions == 1):
            failures.append("custom logical single-submission label is inconsistent")
        if custom_execution.get("true_native_batch") != (expected_submissions == 1):
            failures.append("custom true-native-batch label is inconsistent")
        expected_measured_submissions = custom["camera_contract"]["measured_frames"] * expected_submissions
        if custom_execution.get("measured_native_batch_submissions") != (expected_measured_submissions):
            failures.append("custom measured native submissions are incomplete")
        expected_camera_coverage = custom["camera_contract"]["measured_frames"] * logical_batch
        if custom_execution.get("measured_camera_coverage") != (expected_camera_coverage):
            failures.append("custom measured camera coverage is incomplete")
        if custom_execution.get("measured_native_camera_executions") != (expected_camera_coverage):
            failures.append("custom native camera execution count is incomplete")
        if custom_execution.get("native_camera_executions") != (expected_camera_coverage):
            failures.append("custom total camera execution count is incomplete")
        expected_scope = (
            "batch-global-workspace" if expected_submissions == 1 else "physical-camera-chunk-reused-workspace"
        )
        if custom_scope != expected_scope:
            failures.append("custom capacity scope disagrees with execution schedule")
        if expected_submissions > 1:
            verified = custom["capacity"]["timed_verification"]["observed"]
            installed = custom["capacity"]["installed"]
            if int(installed["intersections"]) < int(verified["max_generated_intersections_per_physical_chunk"]):
                failures.append("custom physical intersection capacity is undersized")
            if int(installed["visible_records"]) < int(verified["max_visible_gaussians_per_physical_chunk"]):
                failures.append("custom physical visible capacity is undersized")
    if custom_execution.get("camera_order") != "contiguous-logical-order":
        failures.append("custom camera order is not established")
    if flashgs["backend_execution"]["true_native_batch"] is not False:
        failures.append("FlashGS was incorrectly labeled as a native batch")
    expected_flashgs_camera_executions = (
        flashgs["camera_contract"]["measured_frames"] * flashgs["camera_contract"]["batch"]
    )
    if flashgs["backend_execution"]["measured_native_camera_executions"] != expected_flashgs_camera_executions:
        failures.append("FlashGS per-camera execution count is not B times frames")
    flashgs_execution = flashgs["backend_execution"]
    failures.extend(
        primary_execution_schedule_failures(
            "flashgs",
            int(flashgs["camera_contract"]["batch"]),
            flashgs_execution,
            flashgs["capacity"],
        )
    )
    if flashgs_execution.get("physical_batch") != 1:
        failures.append("FlashGS physical batch is not one camera")
    if flashgs_execution.get("native_submissions_per_logical_batch") != (flashgs["camera_contract"]["batch"]):
        failures.append("FlashGS native submission schedule is inconsistent")
    if flashgs_execution.get("measured_camera_coverage") != (expected_flashgs_camera_executions):
        failures.append("FlashGS measured camera coverage is incomplete")
    return failures


def fidelity_binding_failures(
    label: str,
    *,
    root: Path,
    batch: int,
    run_path: Path,
    run: dict[str, Any],
    fidelity: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    try:
        capture_path = Path(run["fidelity_capture"]["path"])
        oracle_path = root / f"oracle/b{batch}.npz"
        oracle_manifest_path = oracle_path.with_suffix(".manifest.json")
        oracle_manifest = load_verified_oracle_manifest(str(oracle_manifest_path.resolve()))
        attestation_path = Path(oracle_manifest["gsplat_build_attestation"]["path"])
        current = {
            "run": artifact_record(run_path),
            "capture": artifact_record(capture_path),
            "oracle": artifact_record(oracle_path),
            "oracle_manifest": artifact_record(oracle_manifest_path),
            "gsplat_build_attestation": artifact_record(attestation_path),
            "camera_bundle": artifact_record(oracle_path.with_suffix(".camera-bundle.json")),
        }
        fidelity_root = root / "fidelity" / run["renderer"] / run["output_contract"] / f"b{batch}"
        current_reports = {
            "json": artifact_record(fidelity_root / "fidelity_report.json"),
            "csv": artifact_record(fidelity_root / "fidelity_report.csv"),
        }
        raw_report = json.loads((fidelity_root / "fidelity_report.json").read_text(encoding="utf-8"))
        validate_fidelity_report(raw_report)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        return [f"{label} fidelity input unavailable: {error}"]
    recorded = fidelity.get("input_artifacts")
    if not isinstance(recorded, dict):
        failures.append(f"{label} fidelity input hashes are unavailable")
    else:
        for name, actual in current.items():
            if not same_artifact(recorded.get(name), actual):
                failures.append(f"{label} fidelity {name} hash differs")
    recorded_reports = fidelity.get("report_artifacts")
    if not isinstance(recorded_reports, dict):
        failures.append(f"{label} raw fidelity report hashes are unavailable")
    else:
        for name, actual in current_reports.items():
            if not same_artifact(recorded_reports.get(name), actual):
                failures.append(f"{label} raw fidelity {name} hash differs")
    if fidelity.get("thresholds") != raw_report.get("thresholds"):
        failures.append(f"{label} fidelity thresholds differ from raw report")
    if fidelity.get("aggregate") != raw_report.get("aggregate"):
        failures.append(f"{label} fidelity aggregate differs from raw report")
    if fidelity.get("pass") is not raw_report.get("pass"):
        failures.append(f"{label} fidelity pass differs from raw report")
    expected_source = source_identity(run["environment"]["source_provenance"])
    if source_identity(oracle_manifest.get("source_provenance") or {}) != (expected_source):
        failures.append(f"{label} oracle source identity differs")
    if oracle_manifest.get("output_sha256") != current["oracle"]["sha256"]:
        failures.append(f"{label} oracle output hash differs")
    if not same_artifact(
        oracle_manifest.get("camera_bundle_artifact"),
        current["camera_bundle"],
    ):
        failures.append(f"{label} oracle camera-bundle hash differs")
    if oracle_manifest.get("trajectory_id") != (run["camera_contract"]["trajectory_id"]):
        failures.append(f"{label} oracle trajectory differs")
    if oracle_manifest.get("semantic_topology") != (run["equation_contract"]["semantic_topology"]):
        failures.append(f"{label} oracle semantic topology differs")
    if oracle_manifest.get("gpu_uuid") != run["environment"]["gpu_uuid"]:
        failures.append(f"{label} oracle GPU UUID differs")
    oracle_runtime = {
        "gpu_name": oracle_manifest.get("gpu"),
        **{
            field: oracle_manifest.get(field)
            for field in (
                "gpu_uuid",
                "driver",
                "torch",
                "cuda_runtime",
                "compute_capability",
                "torch_cuda_arch_list",
            )
        },
    }
    run_runtime = {
        field: run["environment"].get(field)
        for field in (
            "gpu_name",
            "gpu_uuid",
            "driver",
            "torch",
            "cuda_runtime",
            "compute_capability",
            "torch_cuda_arch_list",
        )
    }
    if oracle_runtime != run_runtime:
        failures.append(f"{label} oracle runtime identity differs")
    try:
        with np.load(oracle_path, allow_pickle=False) as oracle_archive:
            archive_support = float(oracle_archive["gaussian_support_sigma"].item())
            archive_trajectory = str(oracle_archive["trajectory_id"].item())
            archive_steps = np.asarray(oracle_archive["steps"], dtype=np.int64).tolist()
            archive_indices = np.asarray(oracle_archive["camera_indices"], dtype=np.int64).tolist()
        with np.load(capture_path, allow_pickle=False) as capture_archive:
            capture_steps = np.asarray(capture_archive["steps"], dtype=np.int64).tolist()
            capture_indices = np.asarray(capture_archive["camera_indices"], dtype=np.int64).tolist()
            capture_trajectory = str(capture_archive["trajectory_id"].item())
    except (KeyError, OSError, TypeError, ValueError) as error:
        failures.append(f"{label} oracle archive contract unavailable: {error}")
        archive_support = None
        archive_trajectory = None
        archive_steps = None
        archive_indices = None
        capture_steps = None
        capture_indices = None
        capture_trajectory = None
    if archive_support is None or not np.isclose(
        archive_support,
        MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        rtol=0.0,
        atol=1.0e-6,
    ):
        failures.append(f"{label} oracle archive support cutoff differs")
    if archive_trajectory != run["camera_contract"]["trajectory_id"]:
        failures.append(f"{label} oracle archive trajectory differs")
    if fidelity.get("source_identity") != expected_source:
        failures.append(f"{label} fidelity source identity differs")
    for field, expected in (
        ("renderer", run["renderer"]),
        ("output_contract", run["output_contract"]),
        ("batch", batch),
        ("trajectory_id", run["camera_contract"]["trajectory_id"]),
        (
            "semantic_topology",
            run["equation_contract"]["semantic_topology"],
        ),
        (
            "gaussian_support_sigma",
            run["equation_contract"]["gaussian_support_sigma"],
        ),
    ):
        if fidelity.get(field) != expected:
            failures.append(f"{label} fidelity {field} differs")
    expected_selection = primary_fidelity_selection(
        batch,
        trajectory_timesteps=int(run["camera_contract"]["warmup_frames"])
        + int(run["camera_contract"]["measured_frames"]),
    )
    expected_pairs = [list(pair) for pair in expected_selection]
    expected_steps = [step for step, _camera in expected_selection]
    expected_indices = [camera for _step, camera in expected_selection]
    if fidelity.get("selection_pairs") != expected_pairs:
        failures.append(f"{label} fidelity selection differs")
    if fidelity.get("steps") != expected_steps:
        failures.append(f"{label} fidelity steps differ")
    if fidelity.get("camera_indices") != expected_indices:
        failures.append(f"{label} fidelity camera sample differs")
    if oracle_manifest.get("selection_pairs") != expected_pairs:
        failures.append(f"{label} oracle selection differs")
    if oracle_manifest.get("steps") != expected_steps or archive_steps != expected_steps:
        failures.append(f"{label} oracle steps differ")
    if oracle_manifest.get("camera_indices") != expected_indices or archive_indices != expected_indices:
        failures.append(f"{label} oracle camera sample differs")
    if capture_steps != expected_steps or capture_indices != expected_indices:
        failures.append(f"{label} candidate capture selection differs")
    if capture_trajectory != run["camera_contract"]["trajectory_id"]:
        failures.append(f"{label} candidate capture trajectory differs")
    return failures


def row_for(
    root: Path,
    *,
    batch: int,
    contract: str,
) -> dict[str, Any]:
    custom_path = root / f"runs/custom/{contract}/b{batch}.json"
    flashgs_path = root / f"runs/flashgs/{contract}/b{batch}.json"
    custom = load(custom_path, RENDERER_RUN_SCHEMA)
    flashgs = load(flashgs_path, RENDERER_RUN_SCHEMA)
    custom_fidelity_path = root / f"fidelity/custom/{contract}/b{batch}/matched-fidelity-summary.json"
    flashgs_fidelity_path = root / f"fidelity/flashgs/{contract}/b{batch}/matched-fidelity-summary.json"
    custom_fidelity = load(custom_fidelity_path, FIDELITY_SCHEMA)
    flashgs_fidelity = load(flashgs_fidelity_path, FIDELITY_SCHEMA)
    custom_gpu, custom_gpu_summary = validated_timing_distribution(
        custom,
        "gpu_batch_ms",
        label=f"Custom B{batch} {contract}",
    )
    flashgs_gpu, flashgs_gpu_summary = validated_timing_distribution(
        flashgs,
        "gpu_batch_ms",
        label=f"FlashGS B{batch} {contract}",
    )
    custom_wall, custom_wall_summary = validated_timing_distribution(
        custom,
        "synchronized_wall_batch_ms",
        label=f"Custom B{batch} {contract}",
    )
    flashgs_wall, flashgs_wall_summary = validated_timing_distribution(
        flashgs,
        "synchronized_wall_batch_ms",
        label=f"FlashGS B{batch} {contract}",
    )
    custom_host, custom_host_summary = validated_timing_distribution(
        custom,
        "host_submission_ms",
        label=f"Custom B{batch} {contract}",
    )
    flashgs_host, flashgs_host_summary = validated_timing_distribution(
        flashgs,
        "host_submission_ms",
        label=f"FlashGS B{batch} {contract}",
    )
    custom_ms = float(custom_gpu_summary["mean"])
    flashgs_ms = float(flashgs_gpu_summary["mean"])
    custom_wall_ms = float(custom_wall_summary["mean"])
    flashgs_wall_ms = float(flashgs_wall_summary["mean"])
    gpu_same_step_speedups = flashgs_gpu / custom_gpu
    wall_same_step_speedups = flashgs_wall / custom_wall
    custom_memory, custom_memory_source = steady_process_gpu_bytes(custom)
    flashgs_memory, flashgs_memory_source = steady_process_gpu_bytes(flashgs)
    measured_frames = int(custom["camera_contract"]["measured_frames"])
    total_images = batch * measured_frames
    total_megapixels = (
        total_images * int(custom["camera_contract"]["width"]) * int(custom["camera_contract"]["height"]) / 1.0e6
    )
    custom_images_per_second = total_images / (float(custom_gpu.sum()) / 1000.0)
    flashgs_images_per_second = total_images / (float(flashgs_gpu.sum()) / 1000.0)
    custom_megapixels_per_second = total_megapixels / (float(custom_gpu.sum()) / 1000.0)
    flashgs_megapixels_per_second = total_megapixels / (float(flashgs_gpu.sum()) / 1000.0)
    for label, stored, recomputed in (
        (
            "Custom images/s",
            custom["timing"].get("images_per_second"),
            custom_images_per_second,
        ),
        (
            "FlashGS images/s",
            flashgs["timing"].get("images_per_second"),
            flashgs_images_per_second,
        ),
        (
            "Custom megapixels/s",
            custom["timing"].get("megapixels_per_second"),
            custom_megapixels_per_second,
        ),
        (
            "FlashGS megapixels/s",
            flashgs["timing"].get("megapixels_per_second"),
            flashgs_megapixels_per_second,
        ),
    ):
        if not isinstance(stored, (int, float)) or not math.isclose(
            float(stored), recomputed, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(f"{label} stored value {stored!r} does not match raw-sample value {recomputed!r}.")
    custom_profile = profile_metrics(
        root,
        renderer="custom",
        contract=contract,
        batch=batch,
        run_path=custom_path,
    )
    flashgs_profile = profile_metrics(
        root,
        renderer="flashgs",
        contract=contract,
        batch=batch,
        run_path=flashgs_path,
    )
    failures = compatible(custom, flashgs)
    custom_fidelity_failures = fidelity_binding_failures(
        "custom",
        root=root,
        batch=batch,
        run_path=custom_path,
        run=custom,
        fidelity=custom_fidelity,
    )
    failures.extend(custom_fidelity_failures)
    if batch in (PRIMARY_BATCHES[0], PRIMARY_BATCHES[-1]):
        if custom_profile is None:
            failures.append("custom required B1/B1024 Nsight profile is missing")
        if flashgs_profile is None:
            failures.append("FlashGS required B1/B1024 Nsight profile is missing")
    flashgs_fidelity_failures = fidelity_binding_failures(
        "flashgs",
        root=root,
        batch=batch,
        run_path=flashgs_path,
        run=flashgs,
        fidelity=flashgs_fidelity,
    )
    failures.extend(flashgs_fidelity_failures)
    fidelity_pass = bool(
        custom_fidelity["pass"]
        and flashgs_fidelity["pass"]
        and not custom_fidelity_failures
        and not flashgs_fidelity_failures
    )
    custom_timed_counters = custom["capacity"]["timed_verification"]["observed"]
    flashgs_timed_counters = flashgs["capacity"]["timed_verification"]["observed"]
    custom_visible = custom_timed_counters["measured_mean_visible_gaussians"]
    flashgs_visible = flashgs_timed_counters["measured_mean_visible_gaussians"]
    custom_intersections = custom_timed_counters["measured_mean_generated_intersections"]
    flashgs_intersections = flashgs_timed_counters["measured_mean_generated_intersections"]
    custom_sort_capacity = custom["capacity"]["installed"]["intersections"]
    flashgs_sort_capacity = flashgs["capacity"]["installed"]["intersections"]
    custom_sort_submissions = custom["backend_execution"]["native_submissions_per_logical_batch"]
    flashgs_sort_submissions = flashgs["backend_execution"]["native_submissions_per_logical_batch"]
    custom_physical_batch = custom["backend_execution"]["physical_batch"]
    flashgs_physical_batch = flashgs["backend_execution"]["physical_batch"]
    custom_installed_sort_slots = custom_sort_capacity * custom_sort_submissions
    flashgs_installed_sort_slots = flashgs_sort_capacity * flashgs_sort_submissions
    if (
        custom_installed_sort_slots <= 0
        or flashgs_installed_sort_slots <= 0
        or custom_intersections > custom_installed_sort_slots
        or flashgs_intersections > flashgs_installed_sort_slots
    ):
        failures.append("Installed fixed-sort capacity is invalid for observed work")
    return {
        "batch": batch,
        "custom_ms": custom_ms,
        "flashgs_ms": flashgs_ms,
        "custom_synchronized_wall_ms": custom_wall_ms,
        "flashgs_synchronized_wall_ms": flashgs_wall_ms,
        "custom_host_submission_ms": float(custom_host_summary["mean"]),
        "flashgs_host_submission_ms": float(flashgs_host_summary["mean"]),
        "custom_nsys_profile": custom_profile,
        "flashgs_nsys_profile": flashgs_profile,
        "speedup_custom_over_flashgs": flashgs_ms / custom_ms,
        "gpu_speedup_ratio_of_mean_latency": flashgs_ms / custom_ms,
        "gpu_speedup_observed_same_step_min": float(gpu_same_step_speedups.min()),
        "gpu_speedup_observed_same_step_p50": float(np.percentile(gpu_same_step_speedups, 50)),
        "gpu_speedup_observed_same_step_max": float(gpu_same_step_speedups.max()),
        "wall_speedup_ratio_of_mean_latency": flashgs_wall_ms / custom_wall_ms,
        "wall_speedup_observed_same_step_min": float(wall_same_step_speedups.min()),
        "wall_speedup_observed_same_step_p50": float(np.percentile(wall_same_step_speedups, 50)),
        "wall_speedup_observed_same_step_max": float(wall_same_step_speedups.max()),
        "custom_images_per_second": custom_images_per_second,
        "flashgs_images_per_second": flashgs_images_per_second,
        "custom_wall_images_per_second": total_images / (float(custom_wall.sum()) / 1000.0),
        "flashgs_wall_images_per_second": total_images / (float(flashgs_wall.sum()) / 1000.0),
        "custom_megapixels_per_second": custom_megapixels_per_second,
        "flashgs_megapixels_per_second": flashgs_megapixels_per_second,
        "custom_steady_process_gpu_bytes": custom_memory,
        "flashgs_steady_process_gpu_bytes": flashgs_memory,
        "custom_steady_process_gpu_gb": custom_memory / 1.0e9,
        "flashgs_steady_process_gpu_gb": flashgs_memory / 1.0e9,
        "custom_memory_source": custom_memory_source,
        "flashgs_memory_source": flashgs_memory_source,
        "timing_samples": {
            "custom": {
                "gpu_batch_ms": custom_gpu.tolist(),
                "synchronized_wall_batch_ms": custom_wall.tolist(),
                "host_submission_ms": custom_host.tolist(),
            },
            "flashgs": {
                "gpu_batch_ms": flashgs_gpu.tolist(),
                "synchronized_wall_batch_ms": flashgs_wall.tolist(),
                "host_submission_ms": flashgs_host.tolist(),
            },
        },
        "custom_visible_gaussians": custom_visible,
        "flashgs_visible_gaussians": flashgs_visible,
        "custom_visible_gaussians_per_image": custom_visible / batch,
        "flashgs_visible_gaussians_per_image": flashgs_visible / batch,
        "custom_generated_intersections": custom_intersections,
        "flashgs_generated_intersections": flashgs_intersections,
        "custom_generated_intersections_per_image": custom_intersections / batch,
        "flashgs_generated_intersections_per_image": flashgs_intersections / batch,
        "custom_installed_sort_capacity_per_native_submission": custom_sort_capacity,
        "flashgs_installed_sort_capacity_per_native_submission": flashgs_sort_capacity,
        "custom_physical_batch": custom_physical_batch,
        "flashgs_physical_batch": flashgs_physical_batch,
        "custom_native_submissions_per_logical_batch": custom_sort_submissions,
        "flashgs_native_submissions_per_logical_batch": flashgs_sort_submissions,
        "custom_installed_sort_slots_per_logical_request": custom_installed_sort_slots,
        "flashgs_installed_sort_slots_per_logical_request": flashgs_installed_sort_slots,
        "custom_mean_fixed_sort_utilization": (custom_intersections / custom_installed_sort_slots),
        "flashgs_mean_fixed_sort_utilization": (flashgs_intersections / flashgs_installed_sort_slots),
        "fixed_sort_includes_sentinel_entries": True,
        "visible_counts_cross_backend_comparable": False,
        "visible_counter_units": {
            "custom": "custom compact-path projected/retained visibility counter",
            "flashgs": "FlashGS-derived per-camera projected visibility counter",
        },
        "intersection_counts_cross_backend_comparable": False,
        "intersection_counter_units": {
            "custom": "custom-1x1-bin intersection records",
            "flashgs": "FlashGS-16x16-tile intersection records",
        },
        "custom_fidelity_pass": custom_fidelity["pass"],
        "flashgs_fidelity_pass": flashgs_fidelity["pass"],
        "fidelity_pass": fidelity_pass,
        "fairness_failures": failures,
        "fairness_pass": not failures,
        "pass": not failures and fidelity_pass,
        "artifacts": {
            "custom_run": artifact_record(custom_path),
            "flashgs_run": artifact_record(flashgs_path),
            "custom_fidelity": artifact_record(custom_fidelity_path),
            "flashgs_fidelity": artifact_record(flashgs_fidelity_path),
        },
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def b128_repeat_evidence(
    root: Path,
    *,
    contract: str,
) -> tuple[dict[str, Any], list[str]]:
    """Validate three fresh-process B128 trials and summarize run dispersion."""

    failures: list[str] = []
    runs: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        "custom": [],
        "flashgs": [],
    }
    for renderer in runs:
        primary_path = root / f"runs/{renderer}/{contract}/b128.json"
        try:
            primary = load(primary_path, RENDERER_RUN_SCHEMA)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            failures.append(f"B128 {contract} {renderer} primary unavailable: {error}")
            continue
        runs[renderer].append((primary_path, primary))
        for trial in (2, 3):
            path = root / "repeats" / renderer / contract / f"b128-trial{trial}.json"
            try:
                repeat = load(path, RENDERER_RUN_SCHEMA)
            except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
                failures.append(f"B128 {contract} {renderer} trial {trial} unavailable: {error}")
                continue
            runs[renderer].append((path, repeat))
            for field, actual, expected in (
                ("renderer", repeat.get("renderer"), renderer),
                ("output contract", repeat.get("output_contract"), contract),
                ("batch", repeat.get("camera_contract", {}).get("batch"), 128),
                (
                    "trajectory",
                    repeat.get("camera_contract", {}).get("trajectory_id"),
                    primary.get("camera_contract", {}).get("trajectory_id"),
                ),
                (
                    "scene",
                    repeat.get("scene", {}).get("sha256"),
                    primary.get("scene", {}).get("sha256"),
                ),
                (
                    "equation",
                    repeat.get("equation_contract"),
                    primary.get("equation_contract"),
                ),
                (
                    "runtime",
                    {
                        name: repeat.get("environment", {}).get(name)
                        for name in (
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
                        name: primary.get("environment", {}).get(name)
                        for name in (
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
                    "source",
                    source_identity(repeat.get("environment", {}).get("source_provenance") or {}),
                    source_identity(primary.get("environment", {}).get("source_provenance") or {}),
                ),
            ):
                if actual != expected:
                    failures.append(f"B128 {contract} {renderer} trial {trial} {field} differs")
            config = repeat.get("runner_config") or {}
            artifact_field = "calibration_artifact" if renderer == "custom" else "demand_survey_artifact"
            config_artifact_field = "capacity_calibration" if renderer == "custom" else "flashgs_demand_survey"
            unexpected_config_artifact_field = (
                "flashgs_demand_survey" if renderer == "custom" else "capacity_calibration"
            )
            primary_capacity_artifact = primary.get("capacity", {}).get(artifact_field)
            if (
                config.get("independent_trial") != trial
                or config.get("capture_last_output") is not False
                or config.get("fixed_intersection_capacity") is not None
                or config.get("fixed_visible_capacity") is not None
                or config.get(unexpected_config_artifact_field) is not None
                or not same_artifact(
                    config.get(config_artifact_field),
                    primary_capacity_artifact,
                )
                or not same_artifact(
                    repeat.get("capacity", {}).get(artifact_field),
                    primary_capacity_artifact,
                )
                or repeat.get("capacity", {}).get("installed") != primary.get("capacity", {}).get("installed")
                or repeat.get("fidelity_capture") is not None
            ):
                failures.append(f"B128 {contract} {renderer} trial {trial} config differs")
            failures.extend(
                f"B128 {contract} trial {trial}: {failure}"
                for failure in capacity_calibration_binding_failures(renderer, repeat)
            )
            if (
                repeat.get("pass") is not True
                or repeat.get("memory", {}).get("steady_state_fairness", {}).get("pass") is not True
            ):
                failures.append(f"B128 {contract} {renderer} trial {trial} failed")
            for record_name in (
                "native_extension",
                "native_build_ninja",
                "flashgs_adapter_attestation",
            ):
                if not same_artifact(
                    repeat.get("environment", {}).get(record_name),
                    primary.get("environment", {}).get(record_name),
                ):
                    failures.append(f"B128 {contract} {renderer} trial {trial} {record_name} differs")
            try:
                verify_node_occupancy_evidence(
                    repeat.get("environment", {}).get("node_occupancy"),
                    expected_gpu_uuid=str(repeat.get("environment", {}).get("gpu_uuid", "")),
                )
            except (
                FileNotFoundError,
                OSError,
                ValueError,
            ) as error:
                failures.append(f"B128 {contract} {renderer} trial {trial} occupancy unavailable: {error}")
    if any(len(values) != 3 for values in runs.values()):
        return {
            "contract": contract,
            "pass": False,
            "trials": {},
        }, failures

    summaries: dict[str, Any] = {}
    raw: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for renderer, records in runs.items():
        raw[renderer] = []
        renderer_rows = []
        for trial_index, (path, run) in enumerate(records, start=1):
            try:
                gpu, gpu_summary = validated_timing_distribution(
                    run,
                    "gpu_batch_ms",
                    label=f"B128 {contract} {renderer} trial {trial_index}",
                )
                wall, wall_summary = validated_timing_distribution(
                    run,
                    "synchronized_wall_batch_ms",
                    label=f"B128 {contract} {renderer} trial {trial_index}",
                )
                validated_timing_distribution(
                    run,
                    "host_submission_ms",
                    label=f"B128 {contract} {renderer} trial {trial_index}",
                )
            except ValueError as error:
                failures.append(str(error))
                continue
            raw[renderer].append((gpu, wall))
            renderer_rows.append(
                {
                    "trial": trial_index,
                    "gpu_mean_ms": gpu_summary["mean"],
                    "wall_mean_ms": wall_summary["mean"],
                    "artifact": artifact_record(path),
                }
            )
        summaries[renderer] = renderer_rows
    if any(len(values) != 3 for values in raw.values()):
        return {
            "contract": contract,
            "pass": False,
            "trials": summaries,
        }, failures

    gpu_ratios: list[float] = []
    wall_ratios: list[float] = []
    same_step_gpu_min: list[float] = []
    same_step_gpu_max: list[float] = []
    same_step_wall_min: list[float] = []
    same_step_wall_max: list[float] = []
    for trial in range(3):
        custom_gpu, custom_wall = raw["custom"][trial]
        flashgs_gpu, flashgs_wall = raw["flashgs"][trial]
        gpu_ratio_samples = flashgs_gpu / custom_gpu
        wall_ratio_samples = flashgs_wall / custom_wall
        gpu_ratios.append(float(flashgs_gpu.mean() / custom_gpu.mean()))
        wall_ratios.append(float(flashgs_wall.mean() / custom_wall.mean()))
        same_step_gpu_min.append(float(gpu_ratio_samples.min()))
        same_step_gpu_max.append(float(gpu_ratio_samples.max()))
        same_step_wall_min.append(float(wall_ratio_samples.min()))
        same_step_wall_max.append(float(wall_ratio_samples.max()))

    def descriptive_spread(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()),
            "median": float(np.median(array)),
            "max": float(array.max()),
        }

    custom_wins = bool(min(same_step_gpu_min) > 1.0 and min(same_step_wall_min) > 1.0)
    flashgs_wins = bool(max(same_step_gpu_max) < 1.0 and max(same_step_wall_max) < 1.0)
    return {
        "contract": contract,
        "pass": not failures and (custom_wins or flashgs_wins),
        "winner": ("custom" if custom_wins else "flashgs" if flashgs_wins else "mixed"),
        "trials": summaries,
        "gpu_ratio_of_mean_latency_by_trial": gpu_ratios,
        "wall_ratio_of_mean_latency_by_trial": wall_ratios,
        "gpu_geometric_mean_ratio": geometric_mean(gpu_ratios),
        "wall_geometric_mean_ratio": geometric_mean(wall_ratios),
        "gpu_run_level_ratio_descriptive_spread": descriptive_spread(gpu_ratios),
        "wall_run_level_ratio_descriptive_spread": descriptive_spread(wall_ratios),
        "gpu_same_step_min_by_trial": same_step_gpu_min,
        "gpu_same_step_max_by_trial": same_step_gpu_max,
        "wall_same_step_min_by_trial": same_step_wall_min,
        "wall_same_step_max_by_trial": same_step_wall_max,
        "independent_run_count": 3,
    }, failures


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Batch | Custom P/submits | FlashGS-derived P/submits | Custom GPU ms | FlashGS-derived GPU ms | GPU ratio-of-means | GPU same-step range | Wall same-step range | Custom steady GB | FlashGS-derived steady GB | Fidelity pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {batch} | {custom_physical}/{custom_submits} | "
            "{flashgs_physical}/{flashgs_submits} | "
            "{custom_ms:.3f} | {flashgs_ms:.3f} | {speedup:.3f}x | "
            "[{gpu_min:.3f}x, {gpu_max:.3f}x] | "
            "[{wall_min:.3f}x, {wall_max:.3f}x] | "
            "{custom_gb:.3f} | {flashgs_gb:.3f} | {fidelity} |".format(
                batch=row["batch"],
                custom_physical=row["custom_physical_batch"],
                custom_submits=row["custom_native_submissions_per_logical_batch"],
                flashgs_physical=row["flashgs_physical_batch"],
                flashgs_submits=row["flashgs_native_submissions_per_logical_batch"],
                custom_ms=row["custom_ms"],
                flashgs_ms=row["flashgs_ms"],
                speedup=row["gpu_speedup_ratio_of_mean_latency"],
                gpu_min=row["gpu_speedup_observed_same_step_min"],
                gpu_max=row["gpu_speedup_observed_same_step_max"],
                wall_min=row["wall_speedup_observed_same_step_min"],
                wall_max=row["wall_speedup_observed_same_step_max"],
                custom_gb=row["custom_steady_process_gpu_gb"],
                flashgs_gb=row["flashgs_steady_process_gpu_gb"],
                fidelity="PASS" if row["fidelity_pass"] else "FAIL",
            )
        )
    return "\n".join(lines)


def interpret_results(
    full_rows: list[dict[str, Any]],
    rgb_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    full_speedups = [row["speedup_custom_over_flashgs"] for row in full_rows]
    rgb_speedups = [row["speedup_custom_over_flashgs"] for row in rgb_rows]
    custom_wins_full = all(
        row["gpu_speedup_observed_same_step_min"] > 1.0 and row["wall_speedup_observed_same_step_min"] > 1.0
        for row in full_rows
    )
    custom_wins_rgb = all(
        row["gpu_speedup_observed_same_step_min"] > 1.0 and row["wall_speedup_observed_same_step_min"] > 1.0
        for row in rgb_rows
    )
    flashgs_wins_full = all(
        row["gpu_speedup_observed_same_step_max"] < 1.0 and row["wall_speedup_observed_same_step_max"] < 1.0
        for row in full_rows
    )
    flashgs_wins_rgb = all(
        row["gpu_speedup_observed_same_step_max"] < 1.0 and row["wall_speedup_observed_same_step_max"] < 1.0
        for row in rgb_rows
    )
    if custom_wins_full and custom_wins_rgb:
        performance_verdict = "custom-wins-rgb-and-full-sensor"
        performance_claim = (
            "The custom rasterizer outperforms the FlashGS-derived matched-"
            "contract port for dynamic, massively batched low-resolution "
            "robotics cameras under the frozen RGB and eight-region spatial-"
            f"semantic sensor topology {FROZEN_EXECUTION_SCHEDULE}."
        )
    elif custom_wins_full and flashgs_wins_rgb:
        performance_verdict = "flashgs-wins-rgb-custom-wins-full-sensor"
        performance_claim = (
            "The FlashGS-derived port is faster for pure RGB, while the custom design "
            "is faster for the frozen eight-region spatial-semantic sensor "
            f"contract {FROZEN_EXECUTION_SCHEDULE}."
        )
    elif flashgs_wins_full and flashgs_wins_rgb:
        performance_verdict = "flashgs-wins-rgb-and-full-sensor"
        performance_claim = (
            "The custom rasterizer is not justified on performance; the Isaac/Fabric "
            "sensor runtime should consider the FlashGS-derived port as its backend "
            f"{FROZEN_EXECUTION_SCHEDULE}."
        )
    else:
        performance_verdict = "mixed-by-batch"
        performance_claim = (
            "Performance crosses over by batch size "
            f"{FROZEN_EXECUTION_SCHEDULE}; no uniform renderer-performance claim "
            "is supported by this matrix."
        )

    full_fidelity_pass = all(row["fidelity_pass"] for row in full_rows)
    rgb_fidelity_pass = all(row["fidelity_pass"] for row in rgb_rows)
    full_fairness_pass = all(row.get("fairness_pass", not row.get("fairness_failures", [])) for row in full_rows)
    rgb_fairness_pass = all(row.get("fairness_pass", not row.get("fairness_failures", [])) for row in rgb_rows)
    full_claim_eligible = full_fidelity_pass and full_fairness_pass
    rgb_claim_eligible = rgb_fidelity_pass and rgb_fairness_pass
    if not rgb_claim_eligible:
        failed_contracts = []
        if not rgb_fidelity_pass:
            failed_contracts.append("pinned-gsplat fidelity")
        if not rgb_fairness_pass:
            failed_contracts.append("matched-run fairness")
        failed_verdicts = []
        if not rgb_fidelity_pass:
            failed_verdicts.append("fidelity")
        if not rgb_fairness_pass:
            failed_verdicts.append("fairness")
        verdict = "rgb-and-full-sensor-" + "-and-".join(failed_verdicts) + "-inconclusive"
        supported_claim = (
            "No renderer replacement claim is supported because at least one "
            "RGB-only row fails " + " and ".join(failed_contracts) + "."
        )
    elif not full_claim_eligible:
        if custom_wins_rgb:
            rgb_verdict = "custom-wins-rgb"
            rgb_claim = (
                "The custom rasterizer outperforms the FlashGS-derived matched-contract port "
                "for the validated RGB-only dynamic-camera contract "
                f"{FROZEN_EXECUTION_SCHEDULE}."
            )
        elif flashgs_wins_rgb:
            rgb_verdict = "flashgs-wins-rgb"
            rgb_claim = (
                "The FlashGS-derived matched-contract port outperforms the custom "
                "rasterizer for the validated RGB-only dynamic-camera contract "
                f"{FROZEN_EXECUTION_SCHEDULE}."
            )
        else:
            rgb_verdict = "rgb-mixed-by-batch"
            rgb_claim = f"Validated RGB-only performance crosses over by batch size {FROZEN_EXECUTION_SCHEDULE}."
        failed_contracts = []
        if not full_fidelity_pass:
            failed_contracts.append("fidelity")
        if not full_fairness_pass:
            failed_contracts.append("fairness")
        verdict = f"{rgb_verdict}-full-sensor-{'-and-'.join(failed_contracts)}-inconclusive"
        supported_claim = (
            f"{rgb_claim} The full-sensor timing result is not claim-eligible "
            "because at least one candidate fails the full-sensor " + " and ".join(failed_contracts) + " contract."
        )
    else:
        verdict = performance_verdict
        supported_claim = performance_claim

    return {
        "full_speedups": full_speedups,
        "rgb_speedups": rgb_speedups,
        "raw_performance_verdict": performance_verdict,
        "verdict": verdict,
        "supported_claim": supported_claim,
        "full_sensor_fidelity_pass": full_fidelity_pass,
        "rgb_fidelity_pass": rgb_fidelity_pass,
        "full_sensor_fairness_pass": full_fairness_pass,
        "rgb_fairness_pass": rgb_fairness_pass,
        "full_sensor_claim_eligible": full_claim_eligible,
        "rgb_claim_eligible": rgb_claim_eligible,
    }


def apply_matrix_claim_gates(
    *,
    verdict: str,
    supported_claim: str,
    exact_primary_batches: bool,
    matrix_failures: list[str],
    scientific_pass: bool,
    headline_eligible: bool,
    hardware_scope: dict[str, Any],
) -> tuple[str, str]:
    if not exact_primary_batches:
        verdict = "diagnostic-subset-no-primary-verdict"
        supported_claim = (
            "No all-batch renderer claim is supported because this summary "
            "does not contain the complete frozen eight-batch matrix."
        )
    if matrix_failures:
        return (
            "invalid-matrix-no-primary-verdict",
            "No renderer-performance claim is supported because the matrix "
            "failed these global contract gates: " + "; ".join(matrix_failures) + ".",
        )
    if not headline_eligible:
        scope_kind = "result" if scientific_pass else "partial result"
        supported_claim = (
            f"Hardware-scoped non-headline {scope_kind} on "
            f"{hardware_scope['gpu_name']} "
            f"({hardware_scope['gpu_uuid']}): {supported_claim} "
            f"This does not establish the frozen {HEADLINE_GPU_NAME} "
            "hardware-class headline result."
        )
    return verdict, supported_claim


def build_summary(
    root: Path,
    batches: tuple[int, ...],
    *,
    output_path: Path,
) -> tuple[dict[str, Any], str]:
    """Rebuild the scientific summary from raw artifacts without writing files.

    ``output_path`` is part of the audited command contract: it must name the
    output recorded by the matrix runner's final summary invocation.
    """

    batches_csv = ",".join(str(value) for value in batches)
    if not batches or any(batch not in PRIMARY_BATCHES for batch in batches):
        raise ValueError(f"Batches must be a subset of {PRIMARY_BATCHES}.")
    if len(set(batches)) != len(batches):
        raise ValueError("Batches may not contain duplicates.")
    full_rows = [row_for(root, batch=batch, contract="full") for batch in batches]
    rgb_rows = [row_for(root, batch=batch, contract="rgb") for batch in batches]
    interpretation = interpret_results(full_rows, rgb_rows)
    full_speedups = interpretation.pop("full_speedups")
    rgb_speedups = interpretation.pop("rgb_speedups")
    matrix_failures: list[str] = []
    matrix_invocation_artifact = None
    invocation_gpu_uuid: str | None = None
    summary_command_artifact = None
    matrix_invocation_path = root / "provenance/matrix-invocation.json"
    summary_command_path = root / "logs/summarize.command.json"
    try:
        matrix_invocation = json.loads(matrix_invocation_path.read_text(encoding="utf-8"))
        matrix_invocation_artifact = artifact_record(matrix_invocation_path)
        recorded_matrix_script = matrix_invocation.get("source_script") or {}
        matrix_argv = matrix_invocation.get("argv")
        current_matrix_script = artifact_record(PROJECT_ROOT / "benchmarks/run_flashgs_matched_matrix.py")
        if (
            matrix_invocation.get("schema_version") != "flashgs-matched-matrix-invocation-v1"
            or not isinstance(matrix_argv, list)
            or len(matrix_argv) < 2
            or not same_artifact(recorded_matrix_script, current_matrix_script)
        ):
            raise ValueError("matrix invocation contract differs")
        invocation_cwd = Path(str(matrix_invocation.get("cwd", "")))
        matrix_script_argument = Path(str(matrix_argv[1]))
        if not matrix_script_argument.is_absolute():
            matrix_script_argument = invocation_cwd / matrix_script_argument
        if matrix_script_argument.resolve() != (PROJECT_ROOT / "benchmarks/run_flashgs_matched_matrix.py").resolve():
            raise ValueError("matrix invocation argv script differs")
        parsed_matrix_args = matrix_invocation.get("parsed_arguments") or {}
        if Path(str(parsed_matrix_args.get("output_root", ""))).resolve() != root.resolve():
            raise ValueError("matrix invocation output root differs")
        invocation_gpu_uuid = parsed_matrix_args.get("expected_gpu_uuid")
        if not is_nvidia_gpu_uuid(invocation_gpu_uuid):
            raise ValueError("matrix invocation GPU UUID is not canonical")
        if parsed_matrix_args.get("batches") != batches_csv:
            raise ValueError("matrix invocation batches differ")
        if parsed_matrix_args.get("custom_chunked_physical_views") != (PRIMARY_CHUNKED_PHYSICAL_VIEWS):
            raise ValueError("matrix invocation Custom physical-view schedule differs")
        expected_calibration_schedule = expected_matrix_capacity_calibration_schedule(batches)
        if parsed_matrix_args.get("capacity_calibration_schedule") != (expected_calibration_schedule):
            raise ValueError("matrix invocation capacity-calibration schedule differs")
        expected_flashgs_capacity_protocol = {
            "schema_version": FLASHGS_DEMAND_SURVEY_SCHEMA,
            "canonical_survey_batch": PRIMARY_BATCHES[-1],
            "derived_prefix_batches": list(PRIMARY_BATCHES),
            "probe_intersections_per_camera": 200_000,
            "headroom": 1.05,
            "timing_valid": False,
            "render_outputs_valid": False,
        }
        if parsed_matrix_args.get("flashgs_capacity_protocol") != expected_flashgs_capacity_protocol:
            raise ValueError("matrix invocation FlashGS capacity protocol differs")
        if parsed_matrix_args.get("semantic_topology") != REPRESENTATIVE_SEMANTIC_TOPOLOGY:
            raise ValueError("matrix invocation semantic topology differs")
        if parsed_matrix_args.get("allow_nonheadline_gpu") is not False:
            raise ValueError("matrix invocation non-headline GPU override differs")
        expected_source_manifest_path = root / "provenance/source-manifest.json"
        if Path(str(parsed_matrix_args.get("source_manifest", ""))).resolve() != (
            expected_source_manifest_path.resolve()
        ):
            raise ValueError("matrix invocation source manifest path differs")
        representative_run = load(
            root / f"runs/custom/full/b{batches[0]}.json",
            RENDERER_RUN_SCHEMA,
        )
        expected_scene_path = Path(str((representative_run.get("scene") or {}).get("path", "")))
        if Path(str(parsed_matrix_args.get("scene_path", ""))).resolve() != expected_scene_path.resolve():
            raise ValueError("matrix invocation scene path differs")
        for flag, parsed_key in (
            ("--output-root", "output_root"),
            ("--source-manifest", "source_manifest"),
            ("--scene-path", "scene_path"),
        ):
            argv_path = Path(command_flag_value(matrix_argv, flag))
            if not argv_path.is_absolute():
                argv_path = invocation_cwd / argv_path
            parsed_path = Path(str(parsed_matrix_args.get(parsed_key, "")))
            if argv_path.resolve() != parsed_path.resolve():
                raise ValueError(f"matrix invocation argv {flag} differs")
        if command_flag_value(matrix_argv, "--expected-gpu-uuid") != str(parsed_matrix_args.get("expected_gpu_uuid")):
            raise ValueError("matrix invocation argv --expected-gpu-uuid differs")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        matrix_failures.append(f"matrix invocation evidence unavailable: {error}")
    try:
        summary_command = json.loads(summary_command_path.read_text(encoding="utf-8"))
        summary_command_artifact = artifact_record(summary_command_path)
        command = summary_command.get("command")
        if (
            not isinstance(command, list)
            or len(command) < 2
            or Path(str(command[1])).resolve() != Path(__file__).resolve()
            or "--root" not in command
            or "--output" not in command
        ):
            raise ValueError("final summary command contract differs")
        if (
            Path(command_flag_value(command, "--root")).resolve() != root.resolve()
            or Path(command_flag_value(command, "--output")).resolve() != output_path.resolve()
        ):
            raise ValueError("final summary command paths differ")
        if command_flag_value(command, "--batches") != batches_csv:
            raise ValueError("final summary command batches differ")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        matrix_failures.append(f"final summary command evidence unavailable: {error}")
    exact_primary_batches = batches == PRIMARY_BATCHES
    if not exact_primary_batches:
        matrix_failures.append(f"matrix batches {batches!r} are not the frozen {PRIMARY_BATCHES!r}")
    runs = [
        load(
            root / f"runs/{renderer}/{contract}/b{batch}.json",
            RENDERER_RUN_SCHEMA,
        )
        for renderer in ("custom", "flashgs")
        for contract in ("full", "rgb")
        for batch in batches
    ]
    environment_fields = (
        "gpu_name",
        "gpu_uuid",
        "driver",
        "torch",
        "cuda_runtime",
        "compute_capability",
        "torch_cuda_arch_list",
    )
    reference_environment = {field: runs[0]["environment"].get(field) for field in environment_fields}
    reference_source = source_identity(runs[0]["environment"]["source_provenance"])
    reference_gpu_uuid = reference_environment.get("gpu_uuid")
    if (
        reference_environment.get("gpu_name") != HEADLINE_GPU_NAME
        or not is_nvidia_gpu_uuid(reference_gpu_uuid)
        or reference_gpu_uuid != invocation_gpu_uuid
    ):
        matrix_failures.append(
            "matrix hardware is not one consistently UUID-bound headline L4"
        )
    capacity_artifacts: dict[str, dict[str, Any]] = {
        "custom": {},
        "flashgs": {},
    }
    matrix_launch_occupancy = None
    matrix_launch_path = root / "provenance/matrix-launch-occupancy.json"
    try:
        matrix_launch_payload = json.loads(matrix_launch_path.read_text(encoding="utf-8"))
        matrix_launch_occupancy = artifact_record(matrix_launch_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        matrix_failures.append(f"matrix launch occupancy evidence unavailable: {error}")
    else:
        if (
            matrix_launch_payload.get("schema_version") != "flashgs-matched-matrix-launch-occupancy-v2"
            or matrix_launch_payload.get("expected_gpu_uuid") != reference_gpu_uuid
            or ((matrix_launch_payload.get("executor_control") or {}).get("scope") != "all-visible-gpus")
            or (
                ((matrix_launch_payload.get("executor_control") or {}).get("cooperative_node_wide_lock") or {}).get(
                    "pass"
                )
                is not True
            )
            or matrix_launch_payload.get("pass") is not True
        ):
            matrix_failures.append("matrix launch occupancy gate did not pass")
    for run in runs:
        actual_environment = {field: run["environment"].get(field) for field in environment_fields}
        if actual_environment != reference_environment:
            matrix_failures.append("runtime environment differs across matrix runs")
            break
    for run in runs:
        if source_identity(run["environment"]["source_provenance"]) != (reference_source):
            matrix_failures.append("source identity differs across matrix runs")
            break
    for renderer in ("custom", "flashgs"):
        renderer_runs = [run for run in runs if run["renderer"] == renderer]
        native_reference = renderer_runs[0]["environment"].get("native_extension")
        if any(
            not same_artifact(
                native_reference,
                run["environment"].get("native_extension"),
            )
            for run in renderer_runs[1:]
        ):
            matrix_failures.append(f"{renderer} loaded native extension differs across runs")
        for batch in batches:
            full_run = load(
                root / f"runs/{renderer}/full/b{batch}.json",
                RENDERER_RUN_SCHEMA,
            )
            rgb_run = load(
                root / f"runs/{renderer}/rgb/b{batch}.json",
                RENDERER_RUN_SCHEMA,
            )
            artifact_field = "calibration_artifact" if renderer == "custom" else "demand_survey_artifact"
            full_capacity_artifact = full_run["capacity"].get(artifact_field)
            rgb_capacity_artifact = rgb_run["capacity"].get(artifact_field)
            if not same_artifact(full_capacity_artifact, rgb_capacity_artifact):
                matrix_failures.append(f"{renderer} B{batch} output contracts used different capacity artifacts")
            if full_run["capacity"].get("installed") != rgb_run["capacity"].get("installed"):
                matrix_failures.append(f"{renderer} B{batch} installed capacities differ")
            capacity_artifacts[renderer][str(batch)] = full_capacity_artifact
    if any(run["equation_contract"].get("semantic_topology") != REPRESENTATIVE_SEMANTIC_TOPOLOGY for run in runs):
        matrix_failures.append("matrix is not the representative semantic topology")
    if any(run["equation_contract"].get("gaussian_support_sigma") != MATCHED_GAUSSIAN_SUPPORT_SIGMA for run in runs):
        matrix_failures.append("matrix is not the pinned gsplat 3.33-sigma support contract")
    if any(
        any(run["equation_contract"].get(field) != expected for field, expected in MATCHED_PROJECTION_RULES.items())
        for run in runs
    ):
        matrix_failures.append("matrix projection edge rules differ from gsplat")
    if reference_source.get("dirty") is not False:
        matrix_failures.append("matrix source provenance is not a clean tree")
    repeat_evidence: dict[str, Any] = {}
    if exact_primary_batches:
        for contract, rows in (("full", full_rows), ("rgb", rgb_rows)):
            evidence, repeat_failures = b128_repeat_evidence(
                root,
                contract=contract,
            )
            repeat_evidence[contract] = evidence
            matrix_failures.extend(repeat_failures)
            primary_b128 = next(row for row in rows if row["batch"] == 128)
            if (
                primary_b128["gpu_speedup_observed_same_step_min"] > 1.0
                and primary_b128["wall_speedup_observed_same_step_min"] > 1.0
            ):
                primary_winner = "custom"
            elif (
                primary_b128["gpu_speedup_observed_same_step_max"] < 1.0
                and primary_b128["wall_speedup_observed_same_step_max"] < 1.0
            ):
                primary_winner = "flashgs"
            else:
                primary_winner = "mixed"
            if evidence.get("winner") != primary_winner:
                matrix_failures.append(f"B128 {contract} independent trials do not reproduce the primary-row winner")
            if evidence.get("pass") is not True:
                matrix_failures.append(f"B128 {contract} independent-repeat gate failed")
    all_rows_pass = bool(all(row["pass"] for row in full_rows) and all(row["pass"] for row in rgb_rows))
    scientific_pass = bool(exact_primary_batches and not matrix_failures and all_rows_pass)
    headline_eligible = bool(scientific_pass and all(run["headline_eligible"] for run in runs))
    verdict = interpretation["verdict"]
    supported_claim = interpretation["supported_claim"]
    verdict, supported_claim = apply_matrix_claim_gates(
        verdict=verdict,
        supported_claim=supported_claim,
        exact_primary_batches=exact_primary_batches,
        matrix_failures=matrix_failures,
        scientific_pass=scientific_pass,
        headline_eligible=headline_eligible,
        hardware_scope=reference_environment,
    )
    interpretation["verdict"] = verdict
    interpretation["supported_claim"] = supported_claim
    result = {
        "schema_version": SUMMARY_SCHEMA,
        "pass": scientific_pass,
        "scientific_pass": scientific_pass,
        "headline_eligible": headline_eligible,
        "primary_contract_eligible": exact_primary_batches,
        "matrix_fairness_failures": matrix_failures,
        "hardware_scope": reference_environment,
        "source_identity": reference_source,
        "command_provenance": {
            "matrix_invocation": matrix_invocation_artifact,
            "final_summary_command": summary_command_artifact,
        },
        "preflight_evidence": {
            "capacity_artifacts": capacity_artifacts,
            "matrix_launch_occupancy": matrix_launch_occupancy,
            "flashgs_adapter_attestation": runs[0]["environment"].get("flashgs_adapter_attestation"),
            "b128_independent_repeats": repeat_evidence,
        },
        "statistical_contract": {
            "matrix_comparisons": 16,
            "ordered_trajectory_samples_per_renderer": 100,
            "method": "descriptive paired-by-trajectory-step observed ratios",
            "winner_gate": (
                "the same renderer must be faster in CUDA-event and "
                "synchronized-wall latency at every one of the 100 matched "
                "trajectory steps in every scheduled row"
            ),
            "confidence_interval": None,
            "limitation": (
                "trajectory steps are correlated deterministic workloads and "
                "each renderer has one fresh process per row; no iid or "
                "run-to-run confidence claim is made"
            ),
        },
        "primary_full_sensor_dynamic_table": full_rows,
        "rgb_only_dynamic_table": rgb_rows,
        "aggregate": {
            "full_sensor_geometric_mean_speedup_custom_over_flashgs": geometric_mean(full_speedups),
            "rgb_geometric_mean_speedup_custom_over_flashgs": geometric_mean(rgb_speedups),
            **interpretation,
        },
        "interpretation_boundaries": {
            "ovrtx": "Separate complete-system comparison; image-formation equation differs.",
            "gsplat": "Correctness oracle only; pinned commit 77ab983ffe43420b2131669cb35776b883ca4c3c.",
            "flashgs": (
                "FlashGS-derived matched-contract port based on public commit "
                "cdfc4e4002318423eda356eed02df8e01fa32cb6; not upstream-faithful "
                "or integration-only."
            ),
            "semantics": (
                "Full-sensor correctness is scoped to eight deterministic "
                "spatial octants; the interleaved modulo-1024 contributor "
                "stress lane is reported separately."
            ),
            "intersections": (
                "Backend-native diagnostic units only: Custom counts 1x1-bin "
                "records and FlashGS counts 16x16-tile records."
            ),
        },
    }
    if headline_eligible:
        table_scope = "Primary headline result"
    elif scientific_pass:
        table_scope = "Complete hardware-scoped non-headline result"
    else:
        table_scope = "Diagnostic incomplete or invalid matrix"
    markdown = (
        "# Matched dynamic custom vs FlashGS-derived port\n\n"
        f"{table_scope} (independently changing cameras, full Home Scan, 128x128):\n\n"
        + markdown_table(full_rows)
        + "\n\nRGB-only control:\n\n"
        + markdown_table(rgb_rows)
        + f"\n\nVerdict: `{verdict}`\n\n{supported_claim}\n"
    )
    return result, markdown


def validate_existing_summary(
    root: Path,
    summary_path: Path,
    batches: tuple[int, ...],
) -> dict[str, Any]:
    """Recompute every summary gate and require exact stored-result equality."""

    stored = load(summary_path, SUMMARY_SCHEMA)
    rebuilt, _ = build_summary(root, batches, output_path=summary_path)
    if stored != rebuilt:
        raise ValueError("Stored matched summary differs from raw-artifact reconstruction.")
    return rebuilt


def main() -> None:
    args = parse_args()
    batches = tuple(int(value) for value in args.batches.split(",") if value)
    if args.verify_existing:
        result = validate_existing_summary(args.root, args.output, batches)
        print(
            "FLASHGS_MATCHED_SUMMARY_VERIFIED "
            + json.dumps(
                {
                    "output": str(args.output),
                    "pass": result["pass"],
                    "verdict": result["aggregate"]["verdict"],
                },
                sort_keys=True,
            )
        )
        if not result["pass"]:
            raise SystemExit(1)
        return
    result, markdown = build_summary(args.root, batches, output_path=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    verdict = str(result["aggregate"]["verdict"])
    print(
        "FLASHGS_MATCHED_SUMMARY_OK "
        + json.dumps(
            {"output": str(args.output), "pass": result["pass"], "verdict": verdict},
            sort_keys=True,
        )
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
