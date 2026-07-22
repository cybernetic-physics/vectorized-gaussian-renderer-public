#!/usr/bin/env python3
"""Run/resume the complete matched custom-versus-FlashGS matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.evaluation.matched_semantics import (  # noqa: E402
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    SEMANTIC_TOPOLOGIES,
)
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    CAPACITY_CALIBRATION_SCHEMA,
    CAPACITY_CONSUMPTION_SCHEMA,
    FIDELITY_SCHEMA,
    FLASHGS_DEMAND_SURVEY_CONSUMPTION_SCHEMA,
    FLASHGS_DEMAND_SURVEY_SCHEMA,
    GSPLAT_ORACLE_SCHEMA,
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PINNED_GSPLAT_COMMIT,
    PINNED_GSPLAT_PATCHED_UTILS_SHA256,
    PINNED_GSPLAT_PATCH_SHA256,
    PRIMARY_CHUNKED_PHYSICAL_VIEWS,
    PRIMARY_BATCHES,
    PRIMARY_TRAJECTORY_IDS,
    PROFILE_SCHEMA,
    RENDERER_RUN_SCHEMA,
    SUMMARY_SCHEMA,
    TIMED_CAPACITY_VERIFICATION_SCHEMA,
    artifact_record,
    derive_flashgs_prefix_capacities,
    load_verified_source_manifest,
    primary_fidelity_selection,
    primary_execution_schedule_failures,
    primary_max_physical_views,
    same_artifact,
    source_identity,
    trajectory_artifacts,
    verify_flashgs_demand_counter_source_audit,
    verify_gsplat_oracle_support_evidence,
    verify_node_occupancy_evidence,
    verify_trajectory_prefix_proof,
)
from benchmarks.flashgs_matched_occupancy import (  # noqa: E402
    capture_node_snapshot,
    ensure_cooperative_executor_lock,
    occupancy_failures,
)
from isaacsim_gaussian_renderer.flashgs_baseline_contract import (  # noqa: E402
    FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
    require_flashgs_matched_port_classification,
)
from isaacsim_gaussian_renderer.flashgs_native_loader import (  # noqa: E402
    FLASHGS_UPSTREAM_COMMIT,
)


_SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?i)(?:^|_)(?:api_key|access_key(?:_id)?|auth(?:orization)?|"
    r"credentials?|password|passwd|private_key|secret(?:_access_key|_key)?|"
    r"token)(?:_|$)"
)
_HOST_USER_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"file://(?:localhost)?/(?:Users|home)/[^/\s:]+/|"
    r"/(?:Users|home)/[^/\s:]+/|"
    r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s:]+[\\/]"
    r")"
)
RUNTIME_PREFLIGHT_SCHEMA = "flashgs-publication-runtime-preflight-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="Must be OUTPUT_ROOT/provenance/source-manifest.json for the sealed matrix.",
    )
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gsplat-source", type=Path, default=Path("/workspace/src/gsplat"))
    parser.add_argument(
        "--flashgs-source",
        type=Path,
        default=Path("/workspace/src/FlashGS"),
    )
    parser.add_argument("--batches", default=",".join(str(value) for value in PRIMARY_BATCHES))
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-nonheadline-gpu", action="store_true")
    parser.add_argument(
        "--custom-chunked-physical-views",
        type=int,
        default=PRIMARY_CHUNKED_PHYSICAL_VIEWS,
        help=(
            "Physical camera chunk used for the memory-bounded Custom B512 "
            "and B1024 rows. Smaller rows retain the direct path."
        ),
    )
    parser.add_argument(
        "--semantic-topology",
        choices=SEMANTIC_TOPOLOGIES,
        default=REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    )
    return parser.parse_args()


def publication_child_environment(
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Pin child imports and shell helpers to this exact checkout."""

    environment = dict(os.environ if base is None else base)
    pinned_paths = (str(SRC_ROOT), str(PROJECT_ROOT))
    inherited_paths = [
        value for value in environment.get("PYTHONPATH", "").split(os.pathsep) if value and value not in pinned_paths
    ]
    environment["PYTHONPATH"] = os.pathsep.join((*pinned_paths, *inherited_paths))
    environment["PROJECT_ROOT"] = str(PROJECT_ROOT)
    environment["VGR_PROJECT_ROOT"] = str(PROJECT_ROOT)
    return environment


def require_publication_safe_environment(environment: dict[str, str]) -> None:
    """Reject credentials and host-user paths before any child or profiler."""

    unsafe_names = sorted(
        name
        for name, value in environment.items()
        if value and _SENSITIVE_ENV_NAME_RE.search(name)
    )
    if unsafe_names:
        raise ValueError(
            "Publication environment contains credential-shaped variables: "
            + ", ".join(unsafe_names)
        )
    unsafe_paths = sorted(
        name
        for name, value in environment.items()
        if isinstance(value, str) and _HOST_USER_PATH_RE.search(value)
    )
    if unsafe_paths:
        raise ValueError(
            "Publication environment contains host-user paths in: "
            + ", ".join(unsafe_paths)
        )


def require_publication_safe_paths(paths: dict[str, str | Path]) -> None:
    """Reject nonportable launch paths before writing matrix provenance."""

    unsafe = sorted(
        label
        for label, value in paths.items()
        if _HOST_USER_PATH_RE.search(os.fspath(value))
    )
    if unsafe:
        raise ValueError(
            "Publication launch paths contain host-user identities in: "
            + ", ".join(unsafe)
        )


def capacity_calibration_schedule(
    batches: tuple[int, ...],
) -> tuple[tuple[int, str], ...]:
    """Schedule memory-risk Custom rows largest first, then caller order."""

    schedule: list[tuple[int, str]] = []
    for batch in (1024, 512):
        if batch in batches:
            schedule.append((batch, "custom"))
    for batch in batches:
        item = (batch, "custom")
        if item not in schedule:
            schedule.append(item)
    return tuple(schedule)


def custom_physical_view_args(
    renderer: str,
    batch: int,
    chunked_physical_views: int,
) -> list[str]:
    """Return the frozen Custom physical-camera schedule for one matrix row."""

    expected = primary_max_physical_views(renderer, batch)
    if expected is None:
        return []
    if chunked_physical_views != expected:
        raise ValueError(f"Custom B{batch} requires P{expected}, got P{chunked_physical_views}.")
    return ["--custom-max-physical-views", str(expected)]


def require_adapter_attestation_identity(
    attestation: dict[str, Any],
    *,
    expected_source_identity: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Reject a stale reusable adapter attestation before any benchmark row."""

    adapter_diff_path = Path(str((attestation.get("adapter_diff") or {}).get("path", "")))
    repair_audit = attestation.get("correctness_repair_audit") or {}
    repair_checks = repair_audit.get("checks") or {}
    render_path = project_root.resolve() / "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
    expected_repair_checks = {
        "repaired_slot_0_predicate_exactly_once",
        "repaired_slot_1_predicate_exactly_once",
        "buggy_slot_0_predicate_absent",
        "buggy_slot_1_predicate_absent",
        "slot_0_load_guard_present",
        "slot_1_load_guard_present",
    }
    if (
        attestation.get("schema_version") != FLASHGS_ADAPTER_ATTESTATION_SCHEMA
        or attestation.get("pass") is not True
        or attestation.get("upstream_commit") != FLASHGS_UPSTREAM_COMMIT
        or attestation.get("upstream_clean") is not True
        or source_identity(attestation.get("source_provenance") or {}) != expected_source_identity
        or not adapter_diff_path.is_file()
        or not same_artifact(
            attestation.get("adapter_diff"),
            artifact_record(adapter_diff_path),
        )
        or repair_audit.get("pass") is not True
        or set(repair_checks) != expected_repair_checks
        or not all(value is True for value in repair_checks.values())
        or not render_path.is_file()
        or not same_artifact(repair_audit.get("render_source"), artifact_record(render_path))
    ):
        raise RuntimeError("FlashGS adapter attestation is invalid or stale for this source manifest.")
    require_flashgs_matched_port_classification(attestation.get("baseline_classification"))


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    environment: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record_path = log_path.with_suffix(".command.json")
    record_path.write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(PROJECT_ROOT),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "torch_cuda_arch_list": environment.get("TORCH_CUDA_ARCH_LIST"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def gpu_compute_capability(gpu_uuid: str) -> str:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0] == gpu_uuid:
            return fields[1]
    raise RuntimeError(f"GPU UUID is not visible to nvidia-smi: {gpu_uuid}.")


def runtime_identity(
    python: str,
    *,
    environment: dict[str, str],
    gpu_uuid: str,
) -> dict[str, Any]:
    payload = json.loads(
        subprocess.check_output(
            [
                python,
                "-c",
                (
                    "import json, os, torch; torch.cuda.init(); "
                    "print(json.dumps({'gpu_name': torch.cuda.get_device_name(0), "
                    "'compute_capability': list(torch.cuda.get_device_capability(0)), "
                    "'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, "
                    "'torch_cuda_arch_list': os.environ.get('TORCH_CUDA_ARCH_LIST')}))"
                ),
            ],
            text=True,
            env=environment,
        )
    )
    payload["driver"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "-i",
            gpu_uuid,
        ],
        text=True,
    ).strip()
    payload["gpu_uuid"] = gpu_uuid
    return payload


def require_runtime_preflight_identity(
    preflight: dict[str, Any],
    *,
    expected_gpu_uuid: str,
    expected_source_identity: dict[str, Any],
    source_manifest: Path,
) -> None:
    """Validate the fail-fast CUDA/LPIPS runtime proof for this checkout."""

    operations = preflight.get("operations") or {}
    if (
        preflight.get("schema_version") != RUNTIME_PREFLIGHT_SCHEMA
        or preflight.get("pass") is not True
        or preflight.get("gpu_uuid") != expected_gpu_uuid
        or preflight.get("source_identity") != expected_source_identity
        or not same_artifact(
            preflight.get("source_manifest"),
            artifact_record(source_manifest),
        )
        or operations
        != {
            "cuda_convolution_finite": True,
            "cuda_matmul_finite": True,
            "lpips_backend": "lpips-alex",
            "lpips_finite": True,
        }
    ):
        raise RuntimeError("Publication runtime preflight identity differs.")
    modules = preflight.get("modules") or {}
    if set(modules) != {"lpips", "torch", "torchvision"}:
        raise RuntimeError("Publication runtime module origins are incomplete.")
    for name, record in modules.items():
        origin = Path(str((record or {}).get("origin", "")))
        if (
            (record or {}).get("name") != name
            or not origin.is_file()
            or _HOST_USER_PATH_RE.search(str(origin))
        ):
            raise RuntimeError(f"Publication runtime module {name} is invalid.")
    weights = preflight.get("lpips_weights") or {}
    if set(weights) != {"alexnet_imagenet", "lpips_alex_v0_1"}:
        raise RuntimeError("Publication runtime LPIPS weights are incomplete.")
    for label, record in weights.items():
        path = Path(str((record or {}).get("path", "")))
        if (
            not path.is_file()
            or _HOST_USER_PATH_RE.search(str(path))
            or not same_artifact(record, artifact_record(path))
        ):
            raise RuntimeError(f"Publication runtime weight {label} is invalid.")
    libraries = preflight.get("loaded_cuda_libraries")
    if (
        not isinstance(libraries, list)
        or not libraries
        or any(
            not isinstance(path, str) or _HOST_USER_PATH_RE.search(path)
            for path in libraries
        )
        or not any("libcublas" in Path(path).name.lower() for path in libraries)
    ):
        raise RuntimeError("Publication runtime did not prove a portable cuBLAS load.")


def require_pass(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != schema:
        raise RuntimeError(f"Unexpected schema in {path}: {payload.get('schema_version')}.")
    if not payload.get("pass"):
        raise RuntimeError(f"Existing artifact is not passing: {path}.")
    return payload


def require_schema(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != schema:
        raise RuntimeError(f"Unexpected schema in {path}: {payload.get('schema_version')}.")
    return payload


def require_direct_timed_capacity_verification(
    capacity: dict[str, Any],
    *,
    renderer: str,
    warmup_frames: int = 8,
    measured_frames: int = 100,
) -> None:
    """Require direct device-counter coverage of every timed-run frame."""

    verification = capacity.get("timed_verification") or {}
    observed = verification.get("observed") or {}
    if (
        verification.get("schema_version") != TIMED_CAPACITY_VERIFICATION_SCHEMA
        or verification.get("pass") is not True
        or verification.get("source") != "renderer-device-counters"
        or verification.get("frames_observed") != warmup_frames + measured_frames
        or verification.get("warmup_frames_observed") != warmup_frames
        or verification.get("measured_frames_observed") != measured_frames
        or verification.get("device_max_updates") != warmup_frames + measured_frames
        or verification.get("cpu_readbacks_in_measured_loop") != 0
        or verification.get("final_cpu_readback_after_measurement") is not True
        or verification.get("audit_updates_in_cuda_event_timing") is not False
        or verification.get("audit_updates_in_wall_samples") is not False
        or observed.get("max_visible_overflow", 0) != 0
        or observed.get("max_intersection_overflow") != 0
    ):
        raise RuntimeError(f"Existing {renderer} run lacks direct zero-overflow timed verification.")
    installed = capacity.get("installed") or {}
    if renderer == "flashgs":
        demand = observed.get("max_generated_intersections_per_camera")
        if not isinstance(demand, int) or installed.get("intersections", 0) < demand:
            raise RuntimeError("Existing FlashGS timed capacity is smaller than observed demand.")
    else:
        intersection_key = (
            "max_generated_intersections_per_physical_chunk"
            if capacity.get("intersection_capacity_scope") == "physical-camera-chunk-reused-workspace"
            else "max_generated_intersections"
        )
        visible_key = (
            "max_visible_gaussians_per_physical_chunk"
            if capacity.get("intersection_capacity_scope") == "physical-camera-chunk-reused-workspace"
            else "max_visible_gaussians"
        )
        if (
            not isinstance(observed.get(intersection_key), int)
            or not isinstance(observed.get(visible_key), int)
            or installed.get("intersections", 0) < observed[intersection_key]
            or installed.get("visible_records", 0) < observed[visible_key]
        ):
            raise RuntimeError("Existing Custom timed capacity is smaller than observed demand.")


def require_run_identity(
    run: dict[str, Any],
    *,
    renderer: str,
    output_contract: str,
    batch: int,
    trajectory_id: str,
    semantic_topology: str,
    custom_chunked_physical_views: int,
    expected_gpu_uuid: str,
    expected_source_identity: dict[str, Any],
    expected_scene_sha256: str,
    expected_runtime_identity: dict[str, Any],
    expected_flashgs_adapter_attestation: dict[str, Any],
    expected_capacity_calibration: dict[str, Any],
) -> None:
    if run.get("renderer") != renderer:
        raise RuntimeError(f"Existing run renderer is not {renderer}.")
    if run.get("output_contract") != output_contract:
        raise RuntimeError(f"Existing {renderer} run output contract is not {output_contract}.")
    if run.get("camera_contract", {}).get("batch") != batch:
        raise RuntimeError(f"Existing {renderer} run batch is not {batch}.")
    if run["camera_contract"].get("trajectory_id") != trajectory_id:
        raise RuntimeError(f"Existing {renderer} run does not match immutable camera contract.")
    if run.get("equation_contract", {}).get("semantic_topology") != (semantic_topology):
        raise RuntimeError(f"Existing {renderer} run does not match semantic topology {semantic_topology}.")
    if run.get("equation_contract", {}).get("gaussian_support_sigma") != (MATCHED_GAUSSIAN_SUPPORT_SIGMA):
        raise RuntimeError(
            f"Existing {renderer} run does not use the pinned gsplat "
            f"{MATCHED_GAUSSIAN_SUPPORT_SIGMA}-sigma support cutoff."
        )
    for field, expected in MATCHED_PROJECTION_RULES.items():
        if run.get("equation_contract", {}).get(field) != expected:
            raise RuntimeError(f"Existing {renderer} run projection rule {field} differs.")
    if run.get("scene", {}).get("sha256") != expected_scene_sha256:
        raise RuntimeError(f"Existing {renderer} run scene content differs.")
    if run.get("environment", {}).get("gpu_uuid") != expected_gpu_uuid:
        raise RuntimeError(f"Existing {renderer} run GPU UUID differs.")
    for field, expected in expected_runtime_identity.items():
        if run.get("environment", {}).get(field) != expected:
            raise RuntimeError(f"Existing {renderer} run runtime field {field} differs.")
    native_record = run.get("environment", {}).get("native_extension") or {}
    native_path = Path(str(native_record.get("path", "")))
    if not native_path.is_file() or not same_artifact(
        native_record,
        artifact_record(native_path),
    ):
        raise RuntimeError(f"Existing {renderer} run native extension content differs.")
    try:
        verify_node_occupancy_evidence(
            run.get("environment", {}).get("node_occupancy"),
            expected_gpu_uuid=expected_gpu_uuid,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError(f"Existing {renderer} run occupancy gate did not pass.") from error
    provenance = run.get("environment", {}).get("source_provenance")
    if not isinstance(provenance, dict) or source_identity(provenance) != (expected_source_identity):
        raise RuntimeError(f"Existing {renderer} run source identity differs.")
    upstream_commit = run.get("environment", {}).get("flashgs_upstream_commit")
    if upstream_commit != FLASHGS_UPSTREAM_COMMIT:
        raise RuntimeError(f"Existing {renderer} run FlashGS pin differs.")
    if not same_artifact(
        run.get("environment", {}).get("flashgs_adapter_attestation"),
        expected_flashgs_adapter_attestation,
    ):
        raise RuntimeError(f"Existing {renderer} run FlashGS adapter attestation differs.")
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
        ("semantic_topology", semantic_topology),
        ("capture_last_output", True),
        ("independent_trial", 1),
        ("profile_control", False),
    ):
        if config.get(field) != expected:
            raise RuntimeError(f"Existing {renderer} run config field {field} differs.")
    if config.get("fixed_visible_capacity") is not None:
        raise RuntimeError(f"Existing {renderer} run used a fixed-capacity CLI value.")
    if config.get("fixed_intersection_capacity") is not None:
        raise RuntimeError(f"Existing {renderer} run used a fixed-capacity CLI value.")
    if config.get("capacity_calibration_only") is not False:
        raise RuntimeError(f"Existing {renderer} run is not a timed run.")
    if config.get("flashgs_demand_survey_only") is not False:
        raise RuntimeError(f"Existing {renderer} run is not a timed run.")
    expected_config_artifact_field = "capacity_calibration" if renderer == "custom" else "flashgs_demand_survey"
    unexpected_config_artifact_field = "flashgs_demand_survey" if renderer == "custom" else "capacity_calibration"
    if (
        not same_artifact(
            config.get(expected_config_artifact_field),
            expected_capacity_calibration,
        )
        or config.get(unexpected_config_artifact_field) is not None
    ):
        raise RuntimeError(f"Existing {renderer} run capacity artifact differs.")
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
    if config.get("premeasurement_schedule") != expected_setup:
        raise RuntimeError(f"Existing {renderer} run premeasurement schedule differs.")
    capacity = run.get("capacity") or {}
    artifact_path = Path(str(expected_capacity_calibration.get("path", "")))
    if renderer == "custom":
        if (
            capacity.get("schema_version") != CAPACITY_CONSUMPTION_SCHEMA
            or capacity.get("capacity_source") != "hash-bound-calibration-artifact"
            or capacity.get("timed_setup") != expected_setup
            or not same_artifact(
                capacity.get("calibration_artifact"),
                expected_capacity_calibration,
            )
        ):
            raise RuntimeError("Existing Custom run did not consume the exact calibration.")
        calibration = require_pass(artifact_path, CAPACITY_CALIBRATION_SCHEMA)
        calibrated_capacity = calibration.get("capacity") or {}
        if (
            capacity.get("installed") != calibrated_capacity.get("installed")
            or capacity.get("verified_preflight") != calibrated_capacity.get("verified_preflight")
            or capacity.get("intersection_capacity_scope") != calibrated_capacity.get("intersection_capacity_scope")
        ):
            raise RuntimeError("Existing Custom run capacity differs from calibration.")
    else:
        survey = require_pass(artifact_path, FLASHGS_DEMAND_SURVEY_SCHEMA)
        selected = (survey.get("derived_batch_capacities") or {}).get(str(batch)) or {}
        expected_installed = {
            "visible_records": None,
            "intersections": selected.get("installed_intersections_per_camera"),
        }
        if (
            capacity.get("schema_version") != FLASHGS_DEMAND_SURVEY_CONSUMPTION_SCHEMA
            or capacity.get("capacity_source") != "hash-bound-b1024-demand-survey-batch-prefix"
            or capacity.get("timed_setup") != expected_setup
            or not same_artifact(
                capacity.get("demand_survey_artifact"),
                expected_capacity_calibration,
            )
            or capacity.get("installed") != expected_installed
            or capacity.get("intersection_capacity_scope") != "per-camera-reused-workspace"
            or capacity.get("survey_prefix_demand") != selected
            or capacity.get("survey_canonical_batch") != PRIMARY_BATCHES[-1]
            or capacity.get("installed_from_batch_specific_prefix") != batch
            or capacity.get("survey_render_outputs_valid") is not False
        ):
            raise RuntimeError("Existing FlashGS run did not consume its exact batch-prefix survey capacity.")
    require_direct_timed_capacity_verification(
        capacity,
        renderer=renderer,
    )
    expected_max_physical_views = primary_max_physical_views(renderer, batch)
    if config.get("custom_max_physical_views") != expected_max_physical_views:
        raise RuntimeError(f"Existing {renderer} run does not match the frozen internal camera schedule.")
    schedule_failures = primary_execution_schedule_failures(
        renderer,
        batch,
        run.get("backend_execution") or {},
        capacity,
    )
    if schedule_failures:
        raise RuntimeError("Existing run execution schedule differs: " + "; ".join(schedule_failures))
    if renderer != "custom":
        return
    if expected_max_physical_views is not None and (custom_chunked_physical_views != expected_max_physical_views):
        raise RuntimeError("Custom chunked physical-view contract differs.")
    expected_physical_batch = expected_max_physical_views or batch
    execution = run.get("backend_execution", {})
    if execution.get("physical_batch") != expected_physical_batch:
        raise RuntimeError("Existing Custom run does not match the required physical batch.")
    expected_submissions = (batch + expected_physical_batch - 1) // expected_physical_batch
    if execution.get("native_submissions_per_logical_batch") != (expected_submissions):
        raise RuntimeError("Existing Custom run does not match the required chunk schedule.")


def require_oracle_identity(
    oracle: dict[str, Any],
    *,
    trajectory_id: str,
    semantic_topology: str,
    scene_sha256: str,
    expected_gpu_uuid: str,
    expected_source_identity: dict[str, Any],
    trajectory_timesteps: int,
    batch: int,
    expected_runtime_identity: dict[str, Any],
    expected_gsplat_source: Path,
) -> None:
    verify_gsplat_oracle_support_evidence(
        oracle,
        gsplat_source=expected_gsplat_source,
    )
    if oracle.get("trajectory_id") != trajectory_id:
        raise RuntimeError("Existing oracle camera contract differs.")
    if oracle.get("semantic_topology") != semantic_topology:
        raise RuntimeError("Existing oracle semantic topology differs.")
    if oracle.get("gaussian_support_sigma") != (MATCHED_GAUSSIAN_SUPPORT_SIGMA):
        raise RuntimeError("Existing oracle Gaussian support cutoff differs.")
    support_contract = oracle.get("gsplat_support_contract") or {}
    if (
        support_contract.get("macro") != "GAUSSIAN_EXTEND"
        or support_contract.get("value") != MATCHED_GAUSSIAN_SUPPORT_SIGMA
    ):
        raise RuntimeError("Existing oracle gsplat support contract differs.")
    support_header = support_contract.get("header_artifact") or {}
    support_header_path = Path(str(support_header.get("path", "")))
    if not support_header_path.is_file() or not same_artifact(
        support_header,
        artifact_record(support_header_path),
    ):
        raise RuntimeError("Existing oracle gsplat support header differs.")
    if oracle.get("scene_sha256") != scene_sha256:
        raise RuntimeError("Existing oracle scene differs.")
    if oracle.get("gpu_uuid") != expected_gpu_uuid:
        raise RuntimeError("Existing oracle GPU UUID differs.")
    oracle_runtime = {
        "gpu_name": oracle.get("gpu"),
        **{
            field: oracle.get(field)
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
    if oracle_runtime != expected_runtime_identity:
        raise RuntimeError("Existing oracle runtime identity differs.")
    if oracle.get("gsplat_commit") != PINNED_GSPLAT_COMMIT:
        raise RuntimeError("Existing oracle gsplat pin differs.")
    expected_selection = primary_fidelity_selection(batch, trajectory_timesteps=trajectory_timesteps)
    if oracle.get("selection_pairs") != [list(pair) for pair in expected_selection]:
        raise RuntimeError("Existing oracle fidelity selection differs.")
    if oracle.get("steps") != [step for step, _camera in expected_selection]:
        raise RuntimeError("Existing oracle fidelity steps differ.")
    if oracle.get("camera_indices") != [camera for _step, camera in expected_selection]:
        raise RuntimeError("Existing oracle camera sample differs.")
    if oracle.get("selection_profile") != "primary-fidelity-suite":
        raise RuntimeError("Existing oracle is not the primary fidelity suite.")
    patch = oracle.get("gsplat_compatibility_patch") or {}
    if patch.get("sha256") != PINNED_GSPLAT_PATCH_SHA256:
        raise RuntimeError("Existing oracle compatibility patch differs.")
    if patch.get("patched_utils_sha256") != PINNED_GSPLAT_PATCHED_UTILS_SHA256:
        raise RuntimeError("Existing oracle patched Utils.cpp differs.")
    if not isinstance(oracle.get("camera_bundle_id"), str):
        raise RuntimeError("Existing oracle camera bundle ID is unavailable.")
    for field in ("gsplat_python_artifact", "gsplat_native_extension"):
        recorded = oracle.get(field)
        path = Path(str((recorded or {}).get("path", "")))
        if not path.is_file() or not same_artifact(
            recorded,
            artifact_record(path),
        ):
            raise RuntimeError(f"Existing oracle {field} content differs.")
    provenance = oracle.get("source_provenance")
    if not isinstance(provenance, dict) or source_identity(provenance) != (expected_source_identity):
        raise RuntimeError("Existing oracle source identity differs.")
    output_path = Path(str(oracle.get("output", "")))
    if not output_path.is_file():
        raise RuntimeError("Existing oracle output is missing.")
    if oracle.get("output_sha256") != artifact_record(output_path)["sha256"]:
        raise RuntimeError("Existing oracle output hash differs.")
    bundle_record = oracle.get("camera_bundle_artifact")
    bundle_path = Path(str(oracle.get("camera_bundle", "")))
    if not bundle_path.is_file() or not same_artifact(
        bundle_record,
        artifact_record(bundle_path),
    ):
        raise RuntimeError("Existing oracle camera-bundle hash differs.")


def require_fidelity_identity(
    fidelity: dict[str, Any],
    *,
    run_path: Path,
    oracle_path: Path,
    camera_bundle_path: Path,
    renderer: str,
    output_contract: str,
    batch: int,
    trajectory_id: str,
    semantic_topology: str,
    expected_source_identity: dict[str, Any],
) -> None:
    for field, expected in (
        ("renderer", renderer),
        ("output_contract", output_contract),
        ("batch", batch),
        ("trajectory_id", trajectory_id),
        ("semantic_topology", semantic_topology),
        (
            "gaussian_support_sigma",
            MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        ),
    ):
        if fidelity.get(field) != expected:
            raise RuntimeError(f"Existing fidelity {field} differs.")
    expected_selection = primary_fidelity_selection(batch)
    if fidelity.get("selection_pairs") != [list(pair) for pair in expected_selection]:
        raise RuntimeError("Existing fidelity selection differs.")
    if fidelity.get("steps") != [step for step, _camera in expected_selection]:
        raise RuntimeError("Existing fidelity steps differ.")
    if fidelity.get("camera_indices") != [camera for _step, camera in expected_selection]:
        raise RuntimeError("Existing fidelity camera sample differs.")
    if fidelity.get("source_identity") != expected_source_identity:
        raise RuntimeError("Existing fidelity source identity differs.")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    oracle_manifest_path = oracle_path.with_suffix(".manifest.json")
    oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    capture = run.get("fidelity_capture") or {}
    capture_path = Path(str(capture.get("path", "")))
    current = {
        "run": artifact_record(run_path),
        "capture": artifact_record(capture_path),
        "oracle": artifact_record(oracle_path),
        "oracle_manifest": artifact_record(oracle_manifest_path),
        "gsplat_build_attestation": artifact_record(oracle_manifest["gsplat_build_attestation"]["path"]),
        "camera_bundle": artifact_record(camera_bundle_path),
    }
    recorded = fidelity.get("input_artifacts")
    if not isinstance(recorded, dict):
        raise RuntimeError("Existing fidelity has no input artifact identities.")
    for name, actual in current.items():
        if not same_artifact(recorded.get(name), actual):
            raise RuntimeError(f"Existing fidelity input artifact changed: {name}.")


def require_capacity_calibration_identity(
    calibration: dict[str, Any],
    *,
    renderer: str,
    batch: int,
    trajectory_id: str,
    trajectory_timesteps: int,
    semantic_topology: str,
    expected_gpu_uuid: str,
    expected_source_identity: dict[str, Any],
    expected_scene_sha256: str,
    expected_runtime_identity: dict[str, Any],
    expected_flashgs_adapter_attestation: dict[str, Any],
) -> None:
    """Validate one untimed full-trajectory capacity calibration."""

    if renderer != "custom":
        raise RuntimeError("Only Custom uses the zero-overflow calibration protocol.")

    if calibration.get("schema_version") != CAPACITY_CALIBRATION_SCHEMA:
        raise RuntimeError("Capacity calibration schema differs.")
    if (
        calibration.get("mode") != "capacity-calibration-only"
        or calibration.get("renderer") != renderer
        or calibration.get("calibration_output_contract") != "full"
    ):
        raise RuntimeError("Capacity calibration identity differs.")
    camera = calibration.get("camera_contract") or {}
    if (
        camera.get("batch") != batch
        or camera.get("trajectory_id") != trajectory_id
        or camera.get("timesteps") != trajectory_timesteps
        or camera.get("width") != 128
        or camera.get("height") != 128
        or camera.get("measured_start") != 8
    ):
        raise RuntimeError("Capacity calibration camera contract differs.")
    if calibration.get("scene", {}).get("sha256") != expected_scene_sha256:
        raise RuntimeError("Capacity calibration scene differs.")
    environment = calibration.get("environment") or {}
    if environment.get("gpu_uuid") != expected_gpu_uuid:
        raise RuntimeError("Capacity calibration GPU differs.")
    for field, expected in expected_runtime_identity.items():
        if environment.get(field) != expected:
            raise RuntimeError(f"Capacity calibration runtime field {field} differs.")
    if not same_artifact(
        environment.get("flashgs_adapter_attestation"),
        expected_flashgs_adapter_attestation,
    ):
        raise RuntimeError("Capacity calibration adapter attestation differs.")
    if source_identity(environment.get("source_provenance") or {}) != expected_source_identity:
        raise RuntimeError("Capacity calibration source differs.")
    native_record = environment.get("native_extension") or {}
    native_path = Path(str(native_record.get("path", "")))
    if not native_path.is_file() or not same_artifact(
        native_record,
        artifact_record(native_path),
    ):
        raise RuntimeError("Capacity calibration native extension differs.")
    try:
        verify_node_occupancy_evidence(
            environment.get("node_occupancy"),
            expected_gpu_uuid=expected_gpu_uuid,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError("Capacity calibration occupancy did not pass.") from error
    equation = calibration.get("equation_contract") or {}
    if equation.get("gaussian_support_sigma") != MATCHED_GAUSSIAN_SUPPORT_SIGMA:
        raise RuntimeError("Capacity calibration support cutoff differs.")
    for field, expected in MATCHED_PROJECTION_RULES.items():
        if equation.get(field) != expected:
            raise RuntimeError(f"Capacity calibration projection rule {field} differs.")
    if equation.get("semantic_topology") != semantic_topology:
        raise RuntimeError("Capacity calibration semantic topology differs.")
    config = calibration.get("calibration_config") or {}
    expected_max_physical_views = primary_max_physical_views(renderer, batch)
    for field, expected in (
        ("capacity_headroom", 1.05),
        ("initial_visible_per_view", 170_000),
        ("initial_intersections_per_view", 200_000),
        ("flashgs_initial_intersections", 200_000),
        ("custom_depth_buckets", 128),
        ("custom_depth_bucket_group", 8),
        ("custom_max_physical_views", expected_max_physical_views),
    ):
        if config.get(field) != expected:
            raise RuntimeError(f"Capacity calibration config {field} differs.")
    capacity = calibration.get("capacity") or {}
    expected_physical_batch = expected_max_physical_views or batch if renderer == "custom" else 1
    expected_scope = (
        "per-camera-reused-workspace"
        if renderer == "flashgs"
        else ("physical-camera-chunk-reused-workspace" if expected_physical_batch < batch else "batch-global-workspace")
    )
    for field, expected in (
        ("schema_version", "flashgs-matched-capacity-v1"),
        ("headroom", 1.05),
        ("capacity_source", "full-trajectory-device-counter-calibration"),
        ("preflight_frames_per_pass", trajectory_timesteps),
        ("preflight_measured_frames_per_pass", trajectory_timesteps - 8),
        ("intersection_capacity_scope", expected_scope),
        ("logical_batch", batch),
        ("physical_batch", expected_physical_batch),
        (
            "native_submissions_per_logical_batch",
            (batch + expected_physical_batch - 1) // expected_physical_batch,
        ),
    ):
        if capacity.get(field) != expected:
            raise RuntimeError(f"Capacity calibration field {field} differs.")
    if not isinstance(capacity.get("initial_preflight"), dict):
        raise RuntimeError("Capacity calibration lacks initial counters.")
    if not isinstance(capacity.get("calibration_attempts"), list) or not (capacity["calibration_attempts"]):
        raise RuntimeError("Capacity calibration lacks installed-capacity attempts.")
    if capacity.get("preflight_passes") != (1 + len(capacity["calibration_attempts"])):
        raise RuntimeError("Capacity calibration preflight count differs.")
    installed = capacity.get("installed") or {}
    if not isinstance(installed.get("intersections"), int) or installed["intersections"] <= 0:
        raise RuntimeError("Capacity calibration lacks intersection capacity.")
    if renderer == "custom" and (
        not isinstance(installed.get("visible_records"), int) or installed["visible_records"] <= 0
    ):
        raise RuntimeError("Custom capacity calibration lacks visible capacity.")
    if renderer == "flashgs" and installed.get("visible_records") is not None:
        raise RuntimeError("FlashGS calibration has a visible-record capacity.")
    verified = capacity.get("verified_preflight") or {}
    if verified.get("max_visible_overflow", 0) != 0 or verified.get("max_intersection_overflow") != 0:
        raise RuntimeError("Capacity calibration reported overflow.")
    if (calibration.get("output_validation") or {}).get("pass") is not True:
        raise RuntimeError("Capacity calibration output validation failed.")
    if calibration.get("pass") is not True:
        raise RuntimeError("Capacity calibration pass=false.")


def require_flashgs_demand_survey_identity(
    survey: dict[str, Any],
    *,
    semantic_topology: str,
    expected_gpu_uuid: str,
    expected_source_identity: dict[str, Any],
    expected_scene_sha256: str,
    expected_runtime_identity: dict[str, Any],
    expected_flashgs_adapter_attestation: dict[str, Any],
    expected_contract_paths: dict[int, Path],
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Validate the sole untimed, output-invalid B1024 demand survey."""

    if (
        survey.get("schema_version") != FLASHGS_DEMAND_SURVEY_SCHEMA
        or survey.get("mode") != "flashgs-demand-survey-only"
        or survey.get("renderer") != "flashgs"
        or survey.get("pass") is not True
        or survey.get("timing_valid") is not False
        or survey.get("render_outputs_valid") is not False
    ):
        raise RuntimeError("FlashGS demand-survey identity differs.")
    camera = survey.get("camera_contract") or {}
    if (
        camera.get("batch") != PRIMARY_BATCHES[-1]
        or camera.get("trajectory_id") != PRIMARY_TRAJECTORY_IDS[PRIMARY_BATCHES[-1]]
        or camera.get("timesteps") != 108
        or camera.get("width") != 128
        or camera.get("height") != 128
        or camera.get("artifacts") != trajectory_artifacts(expected_contract_paths[PRIMARY_BATCHES[-1]])
    ):
        raise RuntimeError("FlashGS demand-survey camera contract differs.")
    if (survey.get("scene") or {}).get("sha256") != expected_scene_sha256:
        raise RuntimeError("FlashGS demand-survey scene differs.")
    environment = survey.get("environment") or {}
    if environment.get("gpu_uuid") != expected_gpu_uuid:
        raise RuntimeError("FlashGS demand-survey GPU differs.")
    for field, expected in expected_runtime_identity.items():
        if environment.get(field) != expected:
            raise RuntimeError(f"FlashGS demand-survey runtime field {field} differs.")
    if source_identity(environment.get("source_provenance") or {}) != expected_source_identity:
        raise RuntimeError("FlashGS demand-survey source differs.")
    if not same_artifact(
        environment.get("flashgs_adapter_attestation"),
        expected_flashgs_adapter_attestation,
    ):
        raise RuntimeError("FlashGS demand-survey adapter attestation differs.")
    native_record = environment.get("native_extension") or {}
    native_path = Path(str(native_record.get("path", "")))
    if not native_path.is_file() or not same_artifact(native_record, artifact_record(native_path)):
        raise RuntimeError("FlashGS demand-survey native extension differs.")
    try:
        verify_node_occupancy_evidence(
            environment.get("node_occupancy"),
            expected_gpu_uuid=expected_gpu_uuid,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError("FlashGS demand-survey occupancy did not pass.") from error
    equation = survey.get("equation_contract") or {}
    if (
        equation.get("semantic_topology") != semantic_topology
        or equation.get("gaussian_support_sigma") != MATCHED_GAUSSIAN_SUPPORT_SIGMA
        or any(equation.get(field) != expected for field, expected in MATCHED_PROJECTION_RULES.items())
    ):
        raise RuntimeError("FlashGS demand-survey equation contract differs.")
    config = survey.get("survey_config") or {}
    for field, expected in (
        ("capacity_headroom", 1.05),
        ("initial_visible_per_view", 170_000),
        ("initial_intersections_per_view", 200_000),
        ("flashgs_initial_intersections", 200_000),
        ("custom_depth_buckets", 128),
        ("custom_depth_bucket_group", 8),
        ("custom_max_physical_views", None),
        ("installed_probe_intersections_per_camera", 200_000),
        ("workspace_reuse", "per-camera-reused-workspace"),
        ("output_contract_exercised_but_invalid", "full"),
    ):
        if config.get(field) != expected:
            raise RuntimeError(f"FlashGS demand-survey config {field} differs.")
    try:
        verify_flashgs_demand_counter_source_audit(
            survey.get("counter_source_audit") or {},
            project_root=project_root,
        )
        proof = verify_trajectory_prefix_proof(survey.get("trajectory_prefix_proof") or {})
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError("FlashGS demand-survey source/prefix proof failed.") from error
    expected_artifacts = {str(batch): trajectory_artifacts(path) for batch, path in expected_contract_paths.items()}
    if set(expected_contract_paths) != set(PRIMARY_BATCHES):
        raise RuntimeError("FlashGS demand-survey expected contract set differs.")
    for batch in PRIMARY_BATCHES:
        contract = (proof.get("contracts") or {}).get(str(batch)) or {}
        if (
            contract.get("trajectory_id") != PRIMARY_TRAJECTORY_IDS[batch]
            or contract.get("artifacts") != expected_artifacts[str(batch)]
        ):
            raise RuntimeError(f"FlashGS demand-survey B{batch} prefix artifact differs.")
    observation = survey.get("demand_observation") or {}
    visible = observation.get("per_camera_max_visible_gaussians")
    generated = observation.get("per_camera_max_generated_intersections")
    overflow = observation.get("per_camera_max_intersection_overflow")
    if (
        observation.get("schema_version") != "flashgs-matched-flashgs-demand-observation-v1"
        or observation.get("counter_storage") != "cuda-int64-per-camera"
        or observation.get("frames_observed") != 108
        or observation.get("camera_count") != PRIMARY_BATCHES[-1]
        or observation.get("device_max_updates") != 108
        or observation.get("cpu_readbacks") != 1
        or not isinstance(visible, list)
        or not isinstance(generated, list)
        or not isinstance(overflow, list)
        or len(visible) != PRIMARY_BATCHES[-1]
        or len(generated) != PRIMARY_BATCHES[-1]
        or len(overflow) != PRIMARY_BATCHES[-1]
        or any(not isinstance(value, int) or value < 0 for value in (*visible, *generated, *overflow))
    ):
        raise RuntimeError("FlashGS demand-survey device observation differs.")
    recomputed = derive_flashgs_prefix_capacities(
        visible,
        generated,
        batches=PRIMARY_BATCHES,
        headroom=1.05,
    )
    if survey.get("derived_batch_capacities") != recomputed:
        raise RuntimeError("FlashGS demand-survey derived capacities differ.")


def require_b128_repeat_identity(
    run: dict[str, Any],
    *,
    renderer: str,
    output_contract: str,
    trial: int,
    trajectory_id: str,
    semantic_topology: str,
    expected_gpu_uuid: str,
    expected_source_identity: dict[str, Any],
    expected_scene_sha256: str,
    expected_runtime_identity: dict[str, Any],
    expected_flashgs_adapter_attestation: dict[str, Any],
    expected_capacity_calibration: dict[str, Any],
) -> None:
    """Validate a fresh-process B128 run-level repetition."""

    if (
        run.get("renderer") != renderer
        or run.get("output_contract") != output_contract
        or run.get("camera_contract", {}).get("batch") != 128
        or run.get("camera_contract", {}).get("trajectory_id") != trajectory_id
        or run.get("scene", {}).get("sha256") != expected_scene_sha256
        or run.get("environment", {}).get("gpu_uuid") != expected_gpu_uuid
    ):
        raise RuntimeError("B128 independent-repeat identity differs.")
    for field, expected in expected_runtime_identity.items():
        if run.get("environment", {}).get(field) != expected:
            raise RuntimeError(f"B128 repeat runtime field {field} differs.")
    if source_identity(run.get("environment", {}).get("source_provenance") or {}) != expected_source_identity:
        raise RuntimeError("B128 repeat source identity differs.")
    if not same_artifact(
        run.get("environment", {}).get("flashgs_adapter_attestation"),
        expected_flashgs_adapter_attestation,
    ):
        raise RuntimeError("B128 repeat adapter attestation differs.")
    native_record = run.get("environment", {}).get("native_extension") or {}
    native_path = Path(str(native_record.get("path", "")))
    if not native_path.is_file() or not same_artifact(native_record, artifact_record(native_path)):
        raise RuntimeError("B128 repeat native extension differs.")
    adapter_path = Path(str(expected_flashgs_adapter_attestation.get("path", "")))
    if not adapter_path.is_file() or not same_artifact(
        expected_flashgs_adapter_attestation,
        artifact_record(adapter_path),
    ):
        raise RuntimeError("B128 repeat adapter attestation is stale.")
    try:
        verify_node_occupancy_evidence(
            run.get("environment", {}).get("node_occupancy"),
            expected_gpu_uuid=expected_gpu_uuid,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError("B128 repeat occupancy did not pass.") from error
    equation = run.get("equation_contract") or {}
    if (
        equation.get("semantic_topology") != semantic_topology
        or equation.get("gaussian_support_sigma") != MATCHED_GAUSSIAN_SUPPORT_SIGMA
        or any(equation.get(field) != expected for field, expected in MATCHED_PROJECTION_RULES.items())
    ):
        raise RuntimeError("B128 repeat equation contract differs.")
    config = run.get("runner_config") or {}
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
        ("custom_max_physical_views", None),
        ("semantic_topology", semantic_topology),
        ("capture_last_output", False),
        ("independent_trial", trial),
        ("profile_control", False),
        ("capacity_calibration_only", False),
        ("flashgs_demand_survey_only", False),
    ):
        if config.get(field) != expected:
            raise RuntimeError(f"B128 repeat config field {field} differs.")
    expected_config_field = "capacity_calibration" if renderer == "custom" else "flashgs_demand_survey"
    unexpected_config_field = "flashgs_demand_survey" if renderer == "custom" else "capacity_calibration"
    if (
        not same_artifact(config.get(expected_config_field), expected_capacity_calibration)
        or config.get(unexpected_config_field) is not None
    ):
        raise RuntimeError("B128 repeat capacity artifact differs.")
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
    capacity = run.get("capacity") or {}
    if config.get("premeasurement_schedule") != expected_setup or capacity.get("timed_setup") != expected_setup:
        raise RuntimeError("B128 repeat timed setup differs.")
    artifact_path = Path(str(expected_capacity_calibration.get("path", "")))
    if renderer == "custom":
        calibration = require_pass(artifact_path, CAPACITY_CALIBRATION_SCHEMA)
        calibrated_capacity = calibration.get("capacity") or {}
        if (
            capacity.get("schema_version") != CAPACITY_CONSUMPTION_SCHEMA
            or capacity.get("capacity_source") != "hash-bound-calibration-artifact"
            or not same_artifact(capacity.get("calibration_artifact"), expected_capacity_calibration)
            or capacity.get("installed") != calibrated_capacity.get("installed")
            or capacity.get("verified_preflight") != calibrated_capacity.get("verified_preflight")
        ):
            raise RuntimeError("B128 repeat capacity differs from calibration.")
        capacity_environment = calibration.get("environment") or {}
    else:
        survey = require_pass(artifact_path, FLASHGS_DEMAND_SURVEY_SCHEMA)
        selected = (survey.get("derived_batch_capacities") or {}).get("128") or {}
        if (
            capacity.get("schema_version") != FLASHGS_DEMAND_SURVEY_CONSUMPTION_SCHEMA
            or capacity.get("capacity_source") != "hash-bound-b1024-demand-survey-batch-prefix"
            or not same_artifact(capacity.get("demand_survey_artifact"), expected_capacity_calibration)
            or capacity.get("installed_from_batch_specific_prefix") != 128
            or capacity.get("survey_prefix_demand") != selected
            or capacity.get("installed")
            != {
                "visible_records": None,
                "intersections": selected.get("installed_intersections_per_camera"),
            }
        ):
            raise RuntimeError("B128 repeat capacity differs from demand survey.")
        capacity_environment = survey.get("environment") or {}
    if not same_artifact(capacity_environment.get("native_extension"), native_record):
        raise RuntimeError("B128 repeat native extension differs from capacity artifact.")
    if run.get("fidelity_capture") is not None:
        raise RuntimeError("B128 repeat unexpectedly captured fidelity output.")
    if (run.get("memory") or {}).get("steady_state_fairness", {}).get("pass") is not True:
        raise RuntimeError("B128 repeat memory gate failed.")
    require_direct_timed_capacity_verification(capacity, renderer=renderer)
    if run.get("pass") is not True:
        raise RuntimeError("B128 independent repeat pass=false.")


def maybe_run(
    command: list[str],
    *,
    result_path: Path,
    schema: str,
    log_path: Path,
    args: argparse.Namespace,
    environment: dict[str, str],
    validate: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    if args.skip_existing and result_path.is_file():
        try:
            existing = require_pass(result_path, schema)
            validate(existing)
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ):
            pass
        else:
            return existing
    run_logged(command, log_path=log_path, environment=environment)
    created = require_pass(result_path, schema)
    validate(created)
    return created


def require_profile_identity(
    profile: dict[str, Any],
    *,
    renderer: str,
    output_contract: str,
    batch: int,
    unprofiled_run: Path,
) -> None:
    if (
        profile.get("renderer") != renderer
        or profile.get("output_contract") != output_contract
        or profile.get("batch") != batch
    ):
        raise RuntimeError("Existing Nsight profile identity differs.")
    if not same_artifact(
        (profile.get("unprofiled") or {}).get("run_artifact"),
        artifact_record(unprofiled_run),
    ):
        raise RuntimeError("Existing Nsight profile is stale for its timed run.")
    profiled_control = profile.get("profiled_control") or {}
    if (
        profile.get("captured_frames") != 3
        or profiled_control.get("nvtx_measured_frame_calls") != 3
        or profiled_control.get("no_measured_range_allocator_calls") is not True
    ):
        raise RuntimeError("Existing Nsight profile does not satisfy the fixed three-frame control.")


def result_root_has_compatible_invocation_or_is_pristine(
    output_root: Path,
    *,
    source_manifest: Path,
) -> bool:
    """Reject mixed evidence before the matrix writes anything into the root.

    A source manifest may be staged inside an otherwise new root. An existing
    invocation is handled as a resumable run, but its complete protocol is
    validated separately before any evidence file is written or overwritten.
    """

    if output_root.is_symlink():
        raise FileExistsError(f"Matrix output root may not be a symlink: {output_root}.")
    invocation_path = output_root / "provenance/matrix-invocation.json"
    if invocation_path.is_symlink():
        raise FileExistsError(f"Matrix invocation may not be a symlink: {invocation_path}.")
    if invocation_path.is_file():
        return True
    if invocation_path.exists():
        raise FileExistsError(f"Matrix invocation must be a regular file: {invocation_path}.")
    if not output_root.exists():
        return False

    allowed_files: set[Path] = set()
    try:
        source_manifest_relative = source_manifest.resolve().relative_to(output_root.resolve())
    except ValueError:
        pass
    else:
        if source_manifest.is_symlink():
            raise FileExistsError(f"Source manifest may not be a symlink: {source_manifest}.")
        allowed_files.add((output_root / source_manifest_relative).resolve())
    unexpected = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if (path.is_file() or path.is_symlink()) and path.resolve() not in allowed_files
    )
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise FileExistsError(f"Refusing to mix matrix evidence in non-pristine {output_root}: {preview}.")
    return False


def require_publication_source_manifest_location(
    output_root: Path,
    source_manifest: Path,
) -> None:
    expected = output_root / "provenance/source-manifest.json"
    lexical_source = Path(os.path.abspath(source_manifest))
    lexical_expected = Path(os.path.abspath(expected))
    if lexical_source != lexical_expected:
        raise ValueError(
            "The decisive matrix requires --source-manifest at OUTPUT_ROOT/provenance/source-manifest.json."
        )
    if not source_manifest.is_file() or source_manifest.is_symlink():
        raise ValueError("The decisive matrix source manifest must be a regular, non-symlink file.")


def write_or_validate_matrix_invocation(
    path: Path,
    document: dict[str, Any],
    *,
    resume: bool,
) -> None:
    """Create one immutable invocation record or validate an exact resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    original_bytes = path.read_bytes()
    try:
        existing = json.loads(original_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("Existing matrix invocation is malformed; refusing to overwrite it.") from error
    created_at = existing.get("created_at")
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Existing matrix invocation timestamp is malformed.") from error
    if parsed_created_at.tzinfo is None:
        raise RuntimeError("Existing matrix invocation timestamp must include a timezone.")
    expected_without_timestamp = {key: value for key, value in document.items() if key != "created_at"}
    existing_without_timestamp = {key: value for key, value in existing.items() if key != "created_at"}
    if existing_without_timestamp != expected_without_timestamp:
        raise RuntimeError("Existing matrix invocation differs; refusing to overwrite or mix result roots.")
    if path.read_bytes() != original_bytes:
        raise RuntimeError("Existing matrix invocation changed during validation.")


def main() -> None:
    args = parse_args()
    if args.custom_chunked_physical_views != PRIMARY_CHUNKED_PHYSICAL_VIEWS:
        raise ValueError(
            "The primary matrix freezes Custom B512/B1024 at 128 physical views; "
            "use run_flashgs_matched.py directly for schedule experiments."
        )
    batches = tuple(int(value) for value in args.batches.split(",") if value)
    if not batches or any(value not in PRIMARY_BATCHES for value in batches):
        raise ValueError(f"Batches must be a subset of {PRIMARY_BATCHES}.")
    if len(set(batches)) != len(batches):
        raise ValueError("Batches may not contain duplicates.")
    require_publication_safe_paths(
        {
            "cwd": Path.cwd().absolute(),
            "flashgs_source": args.flashgs_source.absolute(),
            "gsplat_source": args.gsplat_source.absolute(),
            "matrix_python": Path(args.python).absolute(),
            "output_root": args.output_root.absolute(),
            "project_root": PROJECT_ROOT.absolute(),
            "scene_path": args.scene_path.absolute(),
            "source_manifest": args.source_manifest.absolute(),
            "sys_executable": Path(sys.executable).absolute(),
        }
    )
    require_publication_source_manifest_location(args.output_root, args.source_manifest)
    calibration_schedule = capacity_calibration_schedule(batches)
    resume = result_root_has_compatible_invocation_or_is_pristine(
        args.output_root,
        source_manifest=args.source_manifest,
    )
    executor_lock = ensure_cooperative_executor_lock()
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = publication_child_environment()
    require_publication_safe_environment(environment)
    matrix_invocation_path = args.output_root / "provenance/matrix-invocation.json"
    write_or_validate_matrix_invocation(
        matrix_invocation_path,
        {
            "schema_version": "flashgs-matched-matrix-invocation-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "argv": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd().resolve()),
            "parsed_arguments": {
                "output_root": str(args.output_root.resolve()),
                "scene_path": str(args.scene_path.resolve()),
                "source_manifest": str(args.source_manifest.resolve()),
                "expected_gpu_uuid": args.expected_gpu_uuid,
                "batches": args.batches,
                "custom_chunked_physical_views": (args.custom_chunked_physical_views),
                "capacity_calibration_schedule": [
                    {"batch": batch, "renderer": renderer} for batch, renderer in calibration_schedule
                ],
                "flashgs_capacity_protocol": {
                    "schema_version": FLASHGS_DEMAND_SURVEY_SCHEMA,
                    "canonical_survey_batch": PRIMARY_BATCHES[-1],
                    "derived_prefix_batches": list(PRIMARY_BATCHES),
                    "probe_intersections_per_camera": 200_000,
                    "headroom": 1.05,
                    "timing_valid": False,
                    "render_outputs_valid": False,
                },
                "semantic_topology": args.semantic_topology,
                "allow_nonheadline_gpu": args.allow_nonheadline_gpu,
            },
            "environment": {
                name: environment.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "PROJECT_ROOT",
                    "PYTHONPATH",
                    "TORCH_CUDA_ARCH_LIST",
                    "VGR_NATIVE_BUILD_ROOT",
                    "VGR_PROJECT_ROOT",
                    "VGR_GPU_EXECUTOR_LOCK_PATH",
                )
            },
            "source_script": artifact_record(Path(__file__)),
        },
        resume=resume,
    )
    matrix_occupancy_path = args.output_root / "provenance/matrix-launch-occupancy.json"
    matrix_occupancy_snapshot = capture_node_snapshot()
    matrix_occupancy_failures = occupancy_failures(
        matrix_occupancy_snapshot,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=False,
    )
    matrix_occupancy = {
        "schema_version": "flashgs-matched-matrix-launch-occupancy-v2",
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "executor_control": {
            "cooperative_node_wide_lock": executor_lock,
            "scope": "all-visible-gpus",
        },
        "snapshot": matrix_occupancy_snapshot,
        "failures": matrix_occupancy_failures,
        "pass": not matrix_occupancy_failures,
    }
    matrix_occupancy_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_occupancy_path.write_text(
        json.dumps(matrix_occupancy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if matrix_occupancy_failures:
        raise RuntimeError("Matrix launch occupancy preflight failed: " + "; ".join(matrix_occupancy_failures))
    source_provenance = load_verified_source_manifest(
        args.source_manifest,
        project_root=PROJECT_ROOT,
    )
    if source_provenance.get("dirty") is not False:
        raise RuntimeError("The decisive matrix requires a clean source-manifest checkout.")
    expected_source_identity = source_identity(source_provenance)
    scene_artifact = artifact_record(args.scene_path)
    environment["FLASHGS_SOURCE_PATH"] = str(args.flashgs_source.resolve())
    native_build_root = (args.output_root / "provenance/native-extensions").resolve()
    configured_native_root = environment.get("VGR_NATIVE_BUILD_ROOT")
    if configured_native_root is not None and Path(configured_native_root).resolve() != native_build_root:
        raise RuntimeError("VGR_NATIVE_BUILD_ROOT conflicts with the result-owned build root.")
    environment["VGR_NATIVE_BUILD_ROOT"] = str(native_build_root)
    environment.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str((args.output_root / "provenance/torch-extensions").resolve()),
    )
    expected_arch = gpu_compute_capability(args.expected_gpu_uuid)
    configured_arch = environment.get("TORCH_CUDA_ARCH_LIST")
    if configured_arch is not None and configured_arch != expected_arch:
        raise RuntimeError(
            f"TORCH_CUDA_ARCH_LIST conflicts with the contracted GPU: {configured_arch!r} != {expected_arch!r}."
        )
    environment["TORCH_CUDA_ARCH_LIST"] = expected_arch
    runtime_preflight_path = (
        args.output_root / "provenance/publication-runtime-preflight.json"
    )
    if not runtime_preflight_path.is_file():
        run_logged(
            [
                args.python,
                str(
                    PROJECT_ROOT
                    / "benchmarks/verify_flashgs_publication_runtime.py"
                ),
                "--expected-gpu-uuid",
                args.expected_gpu_uuid,
                "--source-manifest",
                str(args.source_manifest),
                "--output",
                str(runtime_preflight_path),
            ],
            log_path=args.output_root / "logs/publication-runtime-preflight.log",
            environment=environment,
        )
    runtime_preflight = require_pass(
        runtime_preflight_path,
        RUNTIME_PREFLIGHT_SCHEMA,
    )
    require_runtime_preflight_identity(
        runtime_preflight,
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_source_identity=expected_source_identity,
        source_manifest=args.source_manifest,
    )
    adapter_attestation_path = args.output_root / "provenance/flashgs-adapter-attestation.json"
    if not adapter_attestation_path.is_file():
        run_logged(
            [
                args.python,
                str(PROJECT_ROOT / "scripts/write_flashgs_adapter_attestation.py"),
                "--flashgs-source",
                str(args.flashgs_source),
                "--source-manifest",
                str(args.source_manifest),
                "--output",
                str(adapter_attestation_path),
            ],
            log_path=args.output_root / "logs/flashgs-adapter-attestation.log",
            environment=environment,
        )
    adapter_attestation = json.loads(adapter_attestation_path.read_text(encoding="utf-8"))
    require_adapter_attestation_identity(
        adapter_attestation,
        expected_source_identity=expected_source_identity,
    )
    adapter_attestation_artifact = artifact_record(adapter_attestation_path)
    expected_runtime_identity = runtime_identity(
        args.python,
        environment=environment,
        gpu_uuid=args.expected_gpu_uuid,
    )
    contracts_root = args.output_root / "contracts"
    canonical_max_batch = PRIMARY_BATCHES[-1]
    required_contracts = tuple(contracts_root / f"b{batch}.json" for batch in PRIMARY_BATCHES)
    if not all(path.is_file() for path in required_contracts):
        run_logged(
            [
                args.python,
                str(PROJECT_ROOT / "benchmarks/generate_flashgs_dynamic_contract.py"),
                "--max-batch",
                str(canonical_max_batch),
                "--output-root",
                str(contracts_root),
            ],
            log_path=args.output_root / "logs/generate-contracts.log",
            environment=environment,
        )

    primary_contract_paths: dict[int, Path] = {}
    primary_contract_payloads: dict[int, dict[str, Any]] = {}
    for batch in PRIMARY_BATCHES:
        trajectory = contracts_root / f"b{batch}.json"
        if not trajectory.is_file():
            raise FileNotFoundError(f"Missing camera contract {trajectory}.")
        trajectory_payload = json.loads(trajectory.read_text(encoding="utf-8"))
        expected_trajectory_id = trajectory_payload.get("trajectory_id")
        if not isinstance(expected_trajectory_id, str):
            raise RuntimeError(f"Camera contract lacks trajectory_id: {trajectory}.")
        if expected_trajectory_id != PRIMARY_TRAJECTORY_IDS[batch]:
            raise RuntimeError(f"Camera contract B{batch} is not the frozen primary trajectory.")
        expected_scene_sha256 = trajectory_payload.get("scene_sha256")
        if expected_scene_sha256 != scene_artifact["sha256"]:
            raise RuntimeError(f"Camera contract scene identity differs from {args.scene_path}.")
        trajectory_timesteps = int(trajectory_payload.get("shape", {}).get("timesteps", 0))
        if trajectory_timesteps != 108:
            raise RuntimeError("Primary camera contract must contain 108 steps.")
        primary_contract_paths[batch] = trajectory
        primary_contract_payloads[batch] = trajectory_payload

    batch_contracts: dict[int, dict[str, Any]] = {}
    for batch_index, batch in enumerate(batches):
        trajectory = primary_contract_paths[batch]
        trajectory_payload = primary_contract_payloads[batch]
        expected_trajectory_id = trajectory_payload["trajectory_id"]
        expected_scene_sha256 = trajectory_payload["scene_sha256"]
        trajectory_timesteps = int(trajectory_payload["shape"]["timesteps"])
        # Alternate backend order across batch sizes to reduce monotonic thermal
        # or clock-order bias without co-resident memory contamination.
        renderer_order = ("custom", "flashgs") if batch_index % 2 == 0 else ("flashgs", "custom")
        batch_contracts[batch] = {
            "trajectory": trajectory,
            "trajectory_id": expected_trajectory_id,
            "scene_sha256": expected_scene_sha256,
            "timesteps": trajectory_timesteps,
            "renderer_order": renderer_order,
        }

    # Finish every capacity-only process before launching a timed process. The
    # largest Custom rows go first so a memory-infeasible matrix fails early,
    # before spending hours on publishable-looking partial timing results.
    run_records: dict[str, Any] = {
        "publication-runtime-preflight": str(runtime_preflight_path)
    }
    calibration_artifacts_by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    for batch, renderer in calibration_schedule:
        contract = batch_contracts[batch]
        trajectory = contract["trajectory"]
        expected_trajectory_id = contract["trajectory_id"]
        expected_scene_sha256 = contract["scene_sha256"]
        trajectory_timesteps = contract["timesteps"]
        calibration_path = args.output_root / f"capacity/{renderer}/b{batch}.json"
        calibration_command = [
            args.python,
            str(PROJECT_ROOT / "benchmarks/run_flashgs_matched.py"),
            "--renderer",
            renderer,
            "--output-contract",
            "full",
            "--trajectory",
            str(trajectory),
            "--scene-path",
            str(args.scene_path),
            "--source-manifest",
            str(args.source_manifest),
            "--flashgs-adapter-attestation",
            str(adapter_attestation_path),
            "--expected-gpu-uuid",
            args.expected_gpu_uuid,
            "--capacity-calibration-only",
            "--no-capture-last-output",
            "--semantic-topology",
            args.semantic_topology,
            "--output",
            str(calibration_path),
        ]
        if args.allow_nonheadline_gpu:
            calibration_command.append("--allow-nonheadline-gpu")
        calibration_command.extend(
            custom_physical_view_args(
                renderer,
                batch,
                args.custom_chunked_physical_views,
            )
        )
        calibration = maybe_run(
            calibration_command,
            result_path=calibration_path,
            schema=CAPACITY_CALIBRATION_SCHEMA,
            log_path=(args.output_root / f"logs/capacity-{renderer}-b{batch}.log"),
            args=args,
            environment=environment,
            validate=lambda payload, renderer=renderer, batch=batch: require_capacity_calibration_identity(
                payload,
                renderer=renderer,
                batch=batch,
                trajectory_id=expected_trajectory_id,
                trajectory_timesteps=trajectory_timesteps,
                semantic_topology=args.semantic_topology,
                expected_gpu_uuid=args.expected_gpu_uuid,
                expected_source_identity=expected_source_identity,
                expected_scene_sha256=expected_scene_sha256,
                expected_runtime_identity=expected_runtime_identity,
                expected_flashgs_adapter_attestation=(adapter_attestation_artifact),
            ),
        )
        require_capacity_calibration_identity(
            calibration,
            renderer=renderer,
            batch=batch,
            trajectory_id=expected_trajectory_id,
            trajectory_timesteps=trajectory_timesteps,
            semantic_topology=args.semantic_topology,
            expected_gpu_uuid=args.expected_gpu_uuid,
            expected_source_identity=expected_source_identity,
            expected_scene_sha256=expected_scene_sha256,
            expected_runtime_identity=expected_runtime_identity,
            expected_flashgs_adapter_attestation=(adapter_attestation_artifact),
        )
        calibration_artifacts_by_pair[(batch, renderer)] = artifact_record(calibration_path)
        run_records[f"capacity-{renderer}-b{batch}"] = str(calibration_path)

    flashgs_survey_path = args.output_root / "capacity/flashgs/b1024-demand-survey.json"
    flashgs_survey_command = [
        args.python,
        str(PROJECT_ROOT / "benchmarks/run_flashgs_matched.py"),
        "--renderer",
        "flashgs",
        "--output-contract",
        "full",
        "--trajectory",
        str(primary_contract_paths[PRIMARY_BATCHES[-1]]),
        "--scene-path",
        str(args.scene_path),
        "--source-manifest",
        str(args.source_manifest),
        "--flashgs-adapter-attestation",
        str(adapter_attestation_path),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "--flashgs-demand-survey-only",
        "--no-capture-last-output",
        "--semantic-topology",
        args.semantic_topology,
        "--output",
        str(flashgs_survey_path),
    ]
    for batch in PRIMARY_BATCHES:
        flashgs_survey_command.extend(["--prefix-trajectory", str(primary_contract_paths[batch])])
    if args.allow_nonheadline_gpu:
        flashgs_survey_command.append("--allow-nonheadline-gpu")
    flashgs_survey = maybe_run(
        flashgs_survey_command,
        result_path=flashgs_survey_path,
        schema=FLASHGS_DEMAND_SURVEY_SCHEMA,
        log_path=args.output_root / "logs/flashgs-b1024-demand-survey.log",
        args=args,
        environment=environment,
        validate=lambda payload: require_flashgs_demand_survey_identity(
            payload,
            semantic_topology=args.semantic_topology,
            expected_gpu_uuid=args.expected_gpu_uuid,
            expected_source_identity=expected_source_identity,
            expected_scene_sha256=scene_artifact["sha256"],
            expected_runtime_identity=expected_runtime_identity,
            expected_flashgs_adapter_attestation=adapter_attestation_artifact,
            expected_contract_paths=primary_contract_paths,
        ),
    )
    require_flashgs_demand_survey_identity(
        flashgs_survey,
        semantic_topology=args.semantic_topology,
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_source_identity=expected_source_identity,
        expected_scene_sha256=scene_artifact["sha256"],
        expected_runtime_identity=expected_runtime_identity,
        expected_flashgs_adapter_attestation=adapter_attestation_artifact,
        expected_contract_paths=primary_contract_paths,
    )
    flashgs_survey_artifact = artifact_record(flashgs_survey_path)
    for batch in batches:
        calibration_artifacts_by_pair[(batch, "flashgs")] = flashgs_survey_artifact
    run_records["flashgs-b1024-demand-survey"] = str(flashgs_survey_path)

    for batch in batches:
        contract = batch_contracts[batch]
        trajectory = contract["trajectory"]
        expected_trajectory_id = contract["trajectory_id"]
        expected_scene_sha256 = contract["scene_sha256"]
        trajectory_timesteps = contract["timesteps"]
        renderer_order = contract["renderer_order"]
        for renderer in renderer_order:
            calibration_path = (
                args.output_root / f"capacity/custom/b{batch}.json" if renderer == "custom" else flashgs_survey_path
            )
            calibration_artifact = calibration_artifacts_by_pair[(batch, renderer)]
            full_result_path = args.output_root / f"runs/{renderer}/full/b{batch}.json"
            full_command = [
                args.python,
                str(PROJECT_ROOT / "benchmarks/run_flashgs_matched.py"),
                "--renderer",
                renderer,
                "--output-contract",
                "full",
                "--trajectory",
                str(trajectory),
                "--scene-path",
                str(args.scene_path),
                "--source-manifest",
                str(args.source_manifest),
                "--flashgs-adapter-attestation",
                str(adapter_attestation_path),
                "--expected-gpu-uuid",
                args.expected_gpu_uuid,
                "--output",
                str(full_result_path),
                "--semantic-topology",
                args.semantic_topology,
            ]
            full_command.extend(
                [
                    ("--capacity-calibration" if renderer == "custom" else "--flashgs-demand-survey"),
                    str(calibration_path),
                ]
            )
            if args.allow_nonheadline_gpu:
                full_command.append("--allow-nonheadline-gpu")
            full_command.extend(
                custom_physical_view_args(
                    renderer,
                    batch,
                    args.custom_chunked_physical_views,
                )
            )
            full = maybe_run(
                full_command,
                result_path=full_result_path,
                schema=RENDERER_RUN_SCHEMA,
                log_path=args.output_root / f"logs/{renderer}-full-b{batch}.log",
                args=args,
                environment=environment,
                validate=lambda payload, renderer=renderer, batch=batch: require_run_identity(
                    payload,
                    renderer=renderer,
                    output_contract="full",
                    batch=batch,
                    trajectory_id=expected_trajectory_id,
                    semantic_topology=args.semantic_topology,
                    custom_chunked_physical_views=(args.custom_chunked_physical_views),
                    expected_gpu_uuid=args.expected_gpu_uuid,
                    expected_source_identity=expected_source_identity,
                    expected_scene_sha256=expected_scene_sha256,
                    expected_runtime_identity=expected_runtime_identity,
                    expected_flashgs_adapter_attestation=(adapter_attestation_artifact),
                    expected_capacity_calibration=calibration_artifact,
                ),
            )
            require_run_identity(
                full,
                renderer=renderer,
                output_contract="full",
                batch=batch,
                trajectory_id=expected_trajectory_id,
                semantic_topology=args.semantic_topology,
                custom_chunked_physical_views=(args.custom_chunked_physical_views),
                expected_gpu_uuid=args.expected_gpu_uuid,
                expected_source_identity=expected_source_identity,
                expected_scene_sha256=expected_scene_sha256,
                expected_runtime_identity=expected_runtime_identity,
                expected_flashgs_adapter_attestation=(adapter_attestation_artifact),
                expected_capacity_calibration=calibration_artifact,
            )
            rgb_result_path = args.output_root / f"runs/{renderer}/rgb/b{batch}.json"
            rgb_command = [
                args.python,
                str(PROJECT_ROOT / "benchmarks/run_flashgs_matched.py"),
                "--renderer",
                renderer,
                "--output-contract",
                "rgb",
                "--trajectory",
                str(trajectory),
                "--scene-path",
                str(args.scene_path),
                "--source-manifest",
                str(args.source_manifest),
                "--flashgs-adapter-attestation",
                str(adapter_attestation_path),
                "--expected-gpu-uuid",
                args.expected_gpu_uuid,
                "--output",
                str(rgb_result_path),
                "--semantic-topology",
                args.semantic_topology,
            ]
            rgb_command.extend(
                [
                    ("--capacity-calibration" if renderer == "custom" else "--flashgs-demand-survey"),
                    str(calibration_path),
                ]
            )
            if args.allow_nonheadline_gpu:
                rgb_command.append("--allow-nonheadline-gpu")
            rgb_command.extend(
                custom_physical_view_args(
                    renderer,
                    batch,
                    args.custom_chunked_physical_views,
                )
            )
            rgb = maybe_run(
                rgb_command,
                result_path=rgb_result_path,
                schema=RENDERER_RUN_SCHEMA,
                log_path=args.output_root / f"logs/{renderer}-rgb-b{batch}.log",
                args=args,
                environment=environment,
                validate=lambda payload, renderer=renderer, batch=batch: require_run_identity(
                    payload,
                    renderer=renderer,
                    output_contract="rgb",
                    batch=batch,
                    trajectory_id=expected_trajectory_id,
                    semantic_topology=args.semantic_topology,
                    custom_chunked_physical_views=(args.custom_chunked_physical_views),
                    expected_gpu_uuid=args.expected_gpu_uuid,
                    expected_source_identity=expected_source_identity,
                    expected_scene_sha256=expected_scene_sha256,
                    expected_runtime_identity=expected_runtime_identity,
                    expected_flashgs_adapter_attestation=(adapter_attestation_artifact),
                    expected_capacity_calibration=calibration_artifact,
                ),
            )
            require_run_identity(
                rgb,
                renderer=renderer,
                output_contract="rgb",
                batch=batch,
                trajectory_id=expected_trajectory_id,
                semantic_topology=args.semantic_topology,
                custom_chunked_physical_views=(args.custom_chunked_physical_views),
                expected_gpu_uuid=args.expected_gpu_uuid,
                expected_source_identity=expected_source_identity,
                expected_scene_sha256=expected_scene_sha256,
                expected_runtime_identity=expected_runtime_identity,
                expected_flashgs_adapter_attestation=(adapter_attestation_artifact),
                expected_capacity_calibration=calibration_artifact,
            )
            run_records[f"{renderer}-full-b{batch}"] = str(full_result_path)
            run_records[f"{renderer}-rgb-b{batch}"] = str(rgb_result_path)

        oracle_path = args.output_root / f"oracle/b{batch}.npz"
        oracle_manifest = oracle_path.with_suffix(".manifest.json")
        oracle_command = [
            args.python,
            str(PROJECT_ROOT / "benchmarks/run_flashgs_gsplat_oracle.py"),
            "--trajectory",
            str(trajectory),
            "--scene-path",
            str(args.scene_path),
            "--gsplat-source",
            str(args.gsplat_source),
            "--source-manifest",
            str(args.source_manifest),
            "--gsplat-build-attestation",
            str(args.output_root / "provenance/gsplat-build-attestation.json"),
            "--expected-gpu-uuid",
            args.expected_gpu_uuid,
            "--output",
            str(oracle_path),
            "--semantic-topology",
            args.semantic_topology,
            "--primary-fidelity-suite",
        ]
        oracle = maybe_run(
            oracle_command,
            result_path=oracle_manifest,
            schema=GSPLAT_ORACLE_SCHEMA,
            log_path=args.output_root / f"logs/gsplat-oracle-b{batch}.log",
            args=args,
            environment=environment,
            validate=lambda payload: require_oracle_identity(
                payload,
                trajectory_id=expected_trajectory_id,
                semantic_topology=args.semantic_topology,
                scene_sha256=expected_scene_sha256,
                expected_gpu_uuid=args.expected_gpu_uuid,
                expected_source_identity=expected_source_identity,
                trajectory_timesteps=trajectory_timesteps,
                batch=batch,
                expected_runtime_identity=expected_runtime_identity,
                expected_gsplat_source=args.gsplat_source,
            ),
        )
        require_oracle_identity(
            oracle,
            trajectory_id=expected_trajectory_id,
            semantic_topology=args.semantic_topology,
            scene_sha256=expected_scene_sha256,
            expected_gpu_uuid=args.expected_gpu_uuid,
            expected_source_identity=expected_source_identity,
            trajectory_timesteps=trajectory_timesteps,
            batch=batch,
            expected_runtime_identity=expected_runtime_identity,
            expected_gsplat_source=args.gsplat_source,
        )
        camera_bundle = oracle_path.with_suffix(".camera-bundle.json")
        for renderer in ("custom", "flashgs"):
            for contract in ("full", "rgb"):
                run_path = args.output_root / f"runs/{renderer}/{contract}/b{batch}.json"
                fidelity_root = args.output_root / f"fidelity/{renderer}/{contract}/b{batch}"
                fidelity_summary = fidelity_root / "matched-fidelity-summary.json"
                fidelity_command = [
                    args.python,
                    str(PROJECT_ROOT / "benchmarks/compare_flashgs_matched_fidelity.py"),
                    "--run",
                    str(run_path),
                    "--oracle",
                    str(oracle_path),
                    "--camera-bundle",
                    str(camera_bundle),
                    "--output-dir",
                    str(fidelity_root),
                ]
                reuse_fidelity = False
                if args.skip_existing and fidelity_summary.is_file():
                    try:
                        existing_fidelity = require_schema(
                            fidelity_summary,
                            FIDELITY_SCHEMA,
                        )
                        require_fidelity_identity(
                            existing_fidelity,
                            run_path=run_path,
                            oracle_path=oracle_path,
                            camera_bundle_path=camera_bundle,
                            renderer=renderer,
                            output_contract=contract,
                            batch=batch,
                            trajectory_id=expected_trajectory_id,
                            semantic_topology=args.semantic_topology,
                            expected_source_identity=expected_source_identity,
                        )
                    except (
                        FileNotFoundError,
                        KeyError,
                        TypeError,
                        ValueError,
                        RuntimeError,
                    ):
                        pass
                    else:
                        reuse_fidelity = True
                if not reuse_fidelity:
                    # Fidelity failure is a scientific result, not a reason to
                    # discard the remaining performance matrix.
                    try:
                        run_logged(
                            fidelity_command,
                            log_path=args.output_root / f"logs/fidelity-{renderer}-{contract}-b{batch}.log",
                            environment=environment,
                        )
                    except subprocess.CalledProcessError:
                        if not fidelity_summary.is_file():
                            raise
                fidelity = require_schema(fidelity_summary, FIDELITY_SCHEMA)
                require_fidelity_identity(
                    fidelity,
                    run_path=run_path,
                    oracle_path=oracle_path,
                    camera_bundle_path=camera_bundle,
                    renderer=renderer,
                    output_contract=contract,
                    batch=batch,
                    trajectory_id=expected_trajectory_id,
                    semantic_topology=args.semantic_topology,
                    expected_source_identity=expected_source_identity,
                )

    if batches == PRIMARY_BATCHES:
        repeat_batch = 128
        repeat_trajectory = contracts_root / f"b{repeat_batch}.json"
        repeat_trajectory_payload = json.loads(repeat_trajectory.read_text(encoding="utf-8"))
        repeat_trajectory_id = repeat_trajectory_payload["trajectory_id"]
        repeat_scene_sha256 = repeat_trajectory_payload["scene_sha256"]
        for trial in (2, 3):
            renderer_order = ("flashgs", "custom") if trial == 2 else ("custom", "flashgs")
            for renderer_index, renderer in enumerate(renderer_order):
                calibration_path = (
                    args.output_root / "capacity/custom/b128.json" if renderer == "custom" else flashgs_survey_path
                )
                calibration_artifact = artifact_record(calibration_path)
                contract_order = ("full", "rgb") if (trial + renderer_index) % 2 == 0 else ("rgb", "full")
                for contract in contract_order:
                    repeat_path = args.output_root / "repeats" / renderer / contract / f"b128-trial{trial}.json"
                    command = [
                        args.python,
                        str(PROJECT_ROOT / "benchmarks/run_flashgs_matched.py"),
                        "--renderer",
                        renderer,
                        "--output-contract",
                        contract,
                        "--trajectory",
                        str(repeat_trajectory),
                        "--scene-path",
                        str(args.scene_path),
                        "--source-manifest",
                        str(args.source_manifest),
                        "--flashgs-adapter-attestation",
                        str(adapter_attestation_path),
                        "--expected-gpu-uuid",
                        args.expected_gpu_uuid,
                        "--no-capture-last-output",
                        "--independent-trial",
                        str(trial),
                        "--semantic-topology",
                        args.semantic_topology,
                        "--output",
                        str(repeat_path),
                    ]
                    command.extend(
                        [
                            ("--capacity-calibration" if renderer == "custom" else "--flashgs-demand-survey"),
                            str(calibration_path),
                        ]
                    )
                    if args.allow_nonheadline_gpu:
                        command.append("--allow-nonheadline-gpu")
                    repeat = maybe_run(
                        command,
                        result_path=repeat_path,
                        schema=RENDERER_RUN_SCHEMA,
                        log_path=(args.output_root / "logs" / f"repeat-{renderer}-{contract}-b128-trial{trial}.log"),
                        args=args,
                        environment=environment,
                        validate=lambda payload, renderer=renderer, contract=contract, trial=trial, calibration_artifact=calibration_artifact: (
                            require_b128_repeat_identity(
                                payload,
                                renderer=renderer,
                                output_contract=contract,
                                trial=trial,
                                trajectory_id=repeat_trajectory_id,
                                semantic_topology=args.semantic_topology,
                                expected_gpu_uuid=args.expected_gpu_uuid,
                                expected_source_identity=expected_source_identity,
                                expected_scene_sha256=repeat_scene_sha256,
                                expected_runtime_identity=expected_runtime_identity,
                                expected_flashgs_adapter_attestation=(adapter_attestation_artifact),
                                expected_capacity_calibration=(calibration_artifact),
                            )
                        ),
                    )
                    require_b128_repeat_identity(
                        repeat,
                        renderer=renderer,
                        output_contract=contract,
                        trial=trial,
                        trajectory_id=repeat_trajectory_id,
                        semantic_topology=args.semantic_topology,
                        expected_gpu_uuid=args.expected_gpu_uuid,
                        expected_source_identity=expected_source_identity,
                        expected_scene_sha256=repeat_scene_sha256,
                        expected_runtime_identity=expected_runtime_identity,
                        expected_flashgs_adapter_attestation=(adapter_attestation_artifact),
                        expected_capacity_calibration=(calibration_artifact),
                    )
                    run_records[f"{renderer}-{contract}-b128-trial{trial}"] = str(repeat_path)

        profile_environment = dict(environment)
        profile_environment.pop("PROFILE_ROOT", None)
        profile_environment["PROFILE_MEASURED_FRAMES"] = "3"
        profile_environment["MATCHED_BENCHMARK_PYTHON"] = shutil.which(args.python) or str(Path(args.python).resolve())
        profile_environment["HOME_SCAN_PATH"] = str(args.scene_path.resolve())
        profile_environment["MATCHED_ALLOW_NONHEADLINE_GPU"] = "1" if args.allow_nonheadline_gpu else "0"
        for batch in (PRIMARY_BATCHES[0], PRIMARY_BATCHES[-1]):
            trajectory = contracts_root / f"b{batch}.json"
            for renderer in ("custom", "flashgs"):
                for contract in ("full", "rgb"):
                    config_id = f"flashgs-matched-{renderer}-{contract}-b{batch}"
                    unprofiled_path = args.output_root / f"runs/{renderer}/{contract}/b{batch}.json"
                    profile_summary_path = args.output_root / "profiles" / config_id / "profile-summary.json"
                    profile_command = [
                        str(PROJECT_ROOT / "scripts/profile_flashgs_matched_nsys.sh"),
                        renderer,
                        contract,
                        str(batch),
                        str(unprofiled_path),
                        str(trajectory),
                        config_id,
                    ]
                    maybe_run(
                        profile_command,
                        result_path=profile_summary_path,
                        schema=PROFILE_SCHEMA,
                        log_path=(args.output_root / "logs" / f"profile-{renderer}-{contract}-b{batch}.log"),
                        args=args,
                        environment=profile_environment,
                        validate=lambda payload, renderer=renderer, contract=contract, batch=batch, unprofiled_path=unprofiled_path: (
                            require_profile_identity(
                                payload,
                                renderer=renderer,
                                output_contract=contract,
                                batch=batch,
                                unprofiled_run=unprofiled_path,
                            )
                        ),
                    )
                    run_records[f"profile-{renderer}-{contract}-b{batch}"] = str(profile_summary_path)

    summary_path = args.output_root / "summary.json"
    run_logged(
        [
            args.python,
            str(PROJECT_ROOT / "benchmarks/summarize_flashgs_matched.py"),
            "--root",
            str(args.output_root),
            "--batches",
            ",".join(str(value) for value in batches),
            "--output",
            str(summary_path),
        ],
        log_path=args.output_root / "logs/summarize.log",
        environment=environment,
    )
    summary = require_schema(summary_path, SUMMARY_SCHEMA)
    print(
        "FLASHGS_MATCHED_MATRIX_COMPLETE "
        + json.dumps(
            {
                "summary": str(summary_path),
                "scientific_pass": summary["pass"],
                "runs": run_records,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
