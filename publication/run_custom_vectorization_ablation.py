#!/usr/bin/env python3
"""Run and assemble the publication Custom B128 vectorization ablation.

This is publication orchestration, not benchmark implementation.  It invokes
the frozen ``run_flashgs_matched.py`` checkout in twelve fresh processes:

* three independent trials;
* full and RGB-only output contracts; and
* P1 sequential physical chunks versus one direct P128 submission.

The timed pairs are adjacent and counterbalanced.  Fidelity work happens only
after every timed process has finished, so it cannot perturb pair adjacency.
P1 receives its own full-trajectory capacity calibration.  P128 reuses the
hash-bound direct B128 calibration from the matched matrix.

Resume is deliberately narrow.  An immutable plan binds every input and exact
command.  A completed stage is reusable only through a receipt that still
matches the plan, command, and every output byte.  Unreceipted partial output
is rejected instead of silently adopted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


PLAN_SCHEMA = "publication-custom-vectorization-ablation-plan-v1"
STAGE_RECEIPT_SCHEMA = "publication-custom-vectorization-ablation-stage-v1"
OUTPUT_SCHEMA = "publication-custom-vectorization-ablation-v1"
SOURCE_SCHEMA = "renderer-source-manifest-v2"
CAPACITY_SCHEMA = "flashgs-matched-capacity-calibration-v1"
RUN_SCHEMA = "flashgs-matched-renderer-run-v4"
FIDELITY_SCHEMA = "flashgs-matched-fidelity-v4"
OCCUPANCY_SCHEMA = "flashgs-matched-node-occupancy-v2"
PRIMARY_B128_TRAJECTORY_ID = (
    "c9a7ef7727761865263c3432954b259c54e2065dfd2326e65494583083704925"
)
DEFAULT_SCENE_SHA256 = (
    "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
)
CONTRACTS = ("full", "rgb")
TRIALS = (1, 2, 3)
PHYSICAL_BATCHES = (1, 128)


class AblationError(RuntimeError):
    """The ablation contract or its immutable evidence is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AblationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise AblationError(f"{label} is missing: {candidate}.") from error
    require(not candidate.is_symlink(), f"{label} must not be a symlink: {candidate}.")
    require(resolved.is_file(), f"{label} is not a regular file: {resolved}.")
    return resolved


def executable_path(path: str | Path, *, label: str) -> Path:
    """Validate an executable while preserving a virtualenv symlink path."""

    candidate = Path(path).absolute()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise AblationError(f"{label} is missing: {candidate}.") from error
    require(resolved.is_file(), f"{label} is not a regular file: {resolved}.")
    require(os.access(candidate, os.X_OK), f"{label} is not executable: {candidate}.")
    return candidate


def artifact_record(path: str | Path, *, label: str = "artifact") -> dict[str, Any]:
    resolved = regular_file(path, label=label)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def same_artifact(left: Any, right: Any) -> bool:
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256")
    )


def resolve_artifact(record: Any, *, label: str) -> Path:
    require(isinstance(record, dict), f"{label} artifact record is missing.")
    require(
        set(record) >= {"path", "bytes", "sha256"},
        f"{label} artifact record is malformed.",
    )
    actual = artifact_record(record["path"], label=label)
    require(same_artifact(record, actual), f"{label} content differs from its record.")
    return Path(actual["path"])


def load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = regular_file(path, label=label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AblationError(f"{label} is not valid UTF-8 JSON: {resolved}.") from error
    require(isinstance(payload, dict), f"{label} must be a JSON object.")
    return payload


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError as error:
        raise AblationError(f"Refusing to overwrite immutable file: {path}.") from error


def command_sha256(command: list[str]) -> str:
    return sha256_bytes(canonical_json_bytes(command))


def _git_output(root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    try:
        result = subprocess.check_output(
            ["git", *arguments],
            cwd=root,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AblationError(
            f"Cannot verify frozen benchmark checkout with git {' '.join(arguments)}."
        ) from error
    return result if binary else result.decode("utf-8").strip()


def verify_source_manifest(
    path: Path,
    *,
    benchmark_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the complete frozen checkout just as the timed runner will."""

    payload = load_json(path, label="source manifest")
    require(payload.get("schema_version") == SOURCE_SCHEMA, "Source manifest schema differs.")
    require(payload.get("dirty") is False, "Publication benchmark source is dirty.")
    require(payload.get("status_short") == [], "Publication source status is not empty.")

    head = _git_output(benchmark_root, "rev-parse", "HEAD")
    status = _git_output(
        benchmark_root,
        "status",
        "--short",
        "--untracked-files=all",
    )
    diff = _git_output(benchmark_root, "diff", "--binary", "HEAD", binary=True)
    require(head == payload.get("head"), "Benchmark HEAD differs from source manifest.")
    require(status.splitlines() == payload.get("status_short"), "Benchmark status differs from source manifest.")
    require(sha256_bytes(diff) == payload.get("diff_sha256"), "Benchmark diff differs from source manifest.")

    tracked_raw = _git_output(benchmark_root, "ls-files", "-z", binary=True)
    untracked_raw = _git_output(
        benchmark_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        binary=True,
    )
    tracked = {
        value
        for value in tracked_raw.decode("utf-8").split("\0")
        if value and (benchmark_root / value).is_file()
    }
    untracked = {
        value
        for value in untracked_raw.decode("utf-8").split("\0")
        if value and (benchmark_root / value).is_file()
    }
    tracked_records = payload.get("tracked_source_files")
    untracked_records = payload.get("relevant_untracked_source_files")
    require(isinstance(tracked_records, dict), "Source manifest tracked records are missing.")
    require(isinstance(untracked_records, dict), "Source manifest untracked records are missing.")
    require(tracked == set(tracked_records), "Tracked file set differs from source manifest.")
    require(untracked == set(untracked_records), "Untracked file set differs from source manifest.")
    for collection, records in (
        ("tracked", tracked_records),
        ("untracked", untracked_records),
    ):
        for relative, expected in records.items():
            candidate = (benchmark_root / relative).resolve()
            require(
                candidate.is_relative_to(benchmark_root),
                f"{collection} source path escapes the benchmark root: {relative}.",
            )
            actual = artifact_record(candidate, label=f"{collection} source {relative}")
            require(
                expected.get("bytes") == actual["bytes"]
                and expected.get("sha256") == actual["sha256"],
                f"Manifested source file changed: {relative}.",
            )
    tree_sha = sha256_bytes(
        canonical_json_bytes(
            {"tracked": tracked_records, "untracked": untracked_records}
        )
    )
    require(tree_sha == payload.get("source_tree_sha256"), "Source tree fingerprint differs.")
    return payload, artifact_record(path, label="source manifest")


def source_identity(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("manifest") or {}
    return {
        "manifest_sha256": manifest.get("sha256"),
        "source_tree_sha256": payload.get("source_tree_sha256"),
        "head": payload.get("head"),
        "dirty": payload.get("dirty"),
        "diff_sha256": payload.get("diff_sha256"),
    }


def expected_source_identity(
    manifest_payload: dict[str, Any],
    manifest_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_sha256": manifest_record["sha256"],
        "source_tree_sha256": manifest_payload.get("source_tree_sha256"),
        "head": manifest_payload.get("head"),
        "dirty": False,
        "diff_sha256": manifest_payload.get("diff_sha256"),
    }


def verify_trajectory(
    trajectory_path: Path,
    *,
    scene_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = load_json(trajectory_path, label="B128 trajectory")
    shape = payload.get("shape") or {}
    require(payload.get("trajectory_id") == PRIMARY_B128_TRAJECTORY_ID, "Trajectory is not the frozen B128 contract.")
    require(payload.get("scene_sha256") == scene_sha256, "Trajectory scene hash differs.")
    require(shape.get("batch") == 128 and shape.get("timesteps") == 108, "Trajectory must be B128 with 108 steps.")
    require(payload.get("height") == 128 and payload.get("width") == 128, "Trajectory must be 128x128.")
    require(
        payload.get("motion_classification") == "independently-changing-camera-batch",
        "Trajectory is not independently changing.",
    )
    arrays_name = payload.get("arrays_file")
    require(
        isinstance(arrays_name, str)
        and arrays_name
        and not Path(arrays_name).is_absolute(),
        "Trajectory arrays_file is invalid.",
    )
    arrays_path = (trajectory_path.parent / arrays_name).resolve()
    require(
        arrays_path.is_relative_to(trajectory_path.parent),
        "Trajectory arrays_file escapes its contract directory.",
    )
    return (
        payload,
        artifact_record(trajectory_path, label="B128 trajectory"),
        artifact_record(arrays_path, label="B128 trajectory arrays"),
    )


def verify_occupancy(record: Any, *, expected_gpu_uuid: str, label: str) -> dict[str, Any]:
    path = resolve_artifact(record, label=label)
    payload = load_json(path, label=label)
    executor = payload.get("executor_control") or {}
    lock = executor.get("cooperative_node_wide_lock") or {}
    samples = payload.get("sampled_compute_process_telemetry") or {}
    require(payload.get("schema_version") == OCCUPANCY_SCHEMA, f"{label} schema differs.")
    require(payload.get("expected_gpu_uuid") == expected_gpu_uuid, f"{label} GPU differs.")
    require(payload.get("pass") is True, f"{label} did not pass.")
    require(executor.get("scope") == "all-visible-gpus", f"{label} scope differs.")
    require(lock.get("pass") is True and lock.get("lock_observed_held") is True, f"{label} lock failed.")
    require(
        samples.get("pass") is True
        and isinstance(samples.get("sample_count"), int)
        and samples["sample_count"] >= 2,
        f"{label} periodic occupancy sampling is insufficient.",
    )
    return payload


def verify_native_record(record: Any, *, label: str) -> dict[str, Any]:
    path = resolve_artifact(record, label=label)
    return artifact_record(path, label=label)


def verify_oracle_inputs(
    *,
    oracle: Path,
    camera_bundle: Path,
    expected_gpu_uuid: str,
    expected_scene_sha256: str,
    expected_source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = oracle.with_suffix(".manifest.json")
    manifest = load_json(manifest_path, label="matrix B128 oracle manifest")
    require(
        manifest.get("schema_version") == "flashgs-matched-gsplat-oracle-v4"
        and manifest.get("pass") is True,
        "Matrix B128 oracle manifest failed.",
    )
    require(
        manifest.get("trajectory_id") == PRIMARY_B128_TRAJECTORY_ID
        and manifest.get("scene_sha256") == expected_scene_sha256
        and manifest.get("selection_profile") == "primary-fidelity-suite",
        "Matrix B128 oracle workload differs.",
    )
    require(
        manifest.get("gpu_uuid") == expected_gpu_uuid,
        "Matrix B128 oracle GPU differs.",
    )
    require(
        source_identity(manifest.get("source_provenance") or {})
        == expected_source,
        "Matrix B128 oracle source differs.",
    )
    oracle_record = artifact_record(oracle, label="matrix B128 oracle")
    require(
        manifest.get("output_sha256") == oracle_record["sha256"],
        "Matrix B128 oracle bytes differ from its manifest.",
    )
    camera_record = artifact_record(
        camera_bundle,
        label="matrix B128 camera bundle",
    )
    require(
        same_artifact(manifest.get("camera_bundle_artifact"), camera_record),
        "Matrix B128 oracle uses another camera bundle.",
    )
    verify_occupancy(
        manifest.get("node_occupancy"),
        expected_gpu_uuid=expected_gpu_uuid,
        label="matrix B128 oracle occupancy",
    )
    return manifest, artifact_record(
        manifest_path,
        label="matrix B128 oracle manifest",
    )


def verify_matrix_bindings(
    *,
    summary_path: Path,
    p128_capacity_record: dict[str, Any],
    oracle_manifest_record: dict[str, Any],
) -> None:
    """Bind reused calibration/oracle inputs to both matrix B128 rows."""

    summary = load_json(summary_path, label="matched matrix summary")
    require(
        summary.get("schema_version") == "flashgs-matched-summary-v4"
        and summary.get("pass") is True
        and summary.get("scientific_pass") is True,
        "Matched matrix summary has not passed.",
    )
    for table_name, contract in (
        ("primary_full_sensor_dynamic_table", "full"),
        ("rgb_only_dynamic_table", "rgb"),
    ):
        table = summary.get(table_name)
        require(isinstance(table, list), f"Matrix {contract} table is missing.")
        rows = [row for row in table if isinstance(row, dict) and row.get("batch") == 128]
        require(len(rows) == 1, f"Matrix {contract} table does not have exactly one B128 row.")
        artifacts = rows[0].get("artifacts") or {}
        custom_run_path = resolve_artifact(
            artifacts.get("custom_run"),
            label=f"matrix B128 {contract} Custom run",
        )
        custom_run = load_json(
            custom_run_path,
            label=f"matrix B128 {contract} Custom run",
        )
        require(
            custom_run.get("renderer") == "custom"
            and custom_run.get("output_contract") == contract
            and custom_run.get("pass") is True,
            f"Matrix B128 {contract} Custom run failed.",
        )
        require(
            same_artifact(
                (custom_run.get("capacity") or {}).get("calibration_artifact"),
                p128_capacity_record,
            ),
            f"P128 capacity is not the matrix B128 {contract} calibration.",
        )
        fidelity_path = resolve_artifact(
            artifacts.get("custom_fidelity"),
            label=f"matrix B128 {contract} Custom fidelity",
        )
        fidelity = load_json(
            fidelity_path,
            label=f"matrix B128 {contract} Custom fidelity",
        )
        require(
            fidelity.get("schema_version") == FIDELITY_SCHEMA
            and fidelity.get("pass") is True,
            f"Matrix B128 {contract} Custom fidelity failed.",
        )
        require(
            same_artifact(
                (fidelity.get("input_artifacts") or {}).get("oracle_manifest"),
                oracle_manifest_record,
            ),
            f"Selected oracle is not the matrix B128 {contract} oracle.",
        )


def verify_capacity(
    path: Path,
    *,
    physical_batch: int,
    expected_gpu_uuid: str,
    expected_scene_sha256: str,
    expected_source: dict[str, Any],
    expected_manifest_record: dict[str, Any],
    expected_native: dict[str, Any] | None,
    require_matrix_direct_config: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_json(path, label=f"P{physical_batch} capacity calibration")
    require(payload.get("schema_version") == CAPACITY_SCHEMA, f"P{physical_batch} capacity schema differs.")
    require(payload.get("pass") is True and payload.get("renderer") == "custom", f"P{physical_batch} capacity failed.")
    require(payload.get("mode") == "capacity-calibration-only", f"P{physical_batch} capacity mode differs.")
    require(payload.get("calibration_output_contract") == "full", f"P{physical_batch} calibration was not full-output.")
    camera = payload.get("camera_contract") or {}
    require(
        camera.get("trajectory_id") == PRIMARY_B128_TRAJECTORY_ID
        and camera.get("batch") == 128
        and camera.get("timesteps") == 108
        and camera.get("measured_start") == 8
        and camera.get("width") == 128
        and camera.get("height") == 128,
        f"P{physical_batch} capacity camera coverage differs.",
    )
    require(
        (payload.get("scene") or {}).get("sha256") == expected_scene_sha256,
        f"P{physical_batch} capacity scene differs.",
    )
    config = payload.get("calibration_config") or {}
    require(config.get("capacity_headroom") == 1.05, f"P{physical_batch} capacity headroom differs.")
    expected_config = None if require_matrix_direct_config else physical_batch
    require(
        config.get("custom_max_physical_views") == expected_config,
        f"P{physical_batch} calibration physical policy differs.",
    )
    capacity = payload.get("capacity") or {}
    expected_scope = (
        "physical-camera-chunk-reused-workspace"
        if physical_batch == 1
        else "batch-global-workspace"
    )
    require(
        capacity.get("logical_batch") == 128
        and capacity.get("physical_batch") == physical_batch
        and capacity.get("native_submissions_per_logical_batch") == 128 // physical_batch
        and capacity.get("intersection_capacity_scope") == expected_scope
        and capacity.get("headroom") == 1.05,
        f"P{physical_batch} capacity schedule differs.",
    )
    installed = capacity.get("installed") or {}
    require(
        isinstance(installed.get("visible_records"), int)
        and installed["visible_records"] > 0
        and isinstance(installed.get("intersections"), int)
        and installed["intersections"] > 0,
        f"P{physical_batch} installed capacity is invalid.",
    )
    verified = capacity.get("verified_preflight") or {}
    require(
        verified.get("max_visible_overflow") == 0
        and verified.get("max_intersection_overflow") == 0,
        f"P{physical_batch} capacity preflight overflowed.",
    )
    require(
        (payload.get("output_validation") or {}).get("pass") is True,
        f"P{physical_batch} capacity output validation failed.",
    )
    environment = payload.get("environment") or {}
    require(environment.get("gpu_uuid") == expected_gpu_uuid, f"P{physical_batch} capacity GPU differs.")
    provenance = environment.get("source_provenance") or {}
    require(source_identity(provenance) == expected_source, f"P{physical_batch} capacity source differs.")
    require(
        same_artifact(provenance.get("manifest"), expected_manifest_record),
        f"P{physical_batch} capacity manifest differs.",
    )
    native = verify_native_record(environment.get("native_extension"), label=f"P{physical_batch} native extension")
    if expected_native is not None:
        require(same_artifact(native, expected_native), f"P{physical_batch} native extension differs.")
    verify_occupancy(
        environment.get("node_occupancy"),
        expected_gpu_uuid=expected_gpu_uuid,
        label=f"P{physical_batch} capacity occupancy",
    )
    return payload, native


def timing_mean(payload: dict[str, Any], name: str, *, label: str) -> float:
    distribution = (payload.get("timing") or {}).get(name) or {}
    samples = distribution.get("samples")
    require(
        distribution.get("count") == 100
        and isinstance(samples, list)
        and len(samples) == 100,
        f"{label} does not contain 100 {name} samples.",
    )
    values: list[float] = []
    for value in samples:
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0,
            f"{label} has an invalid {name} sample.",
        )
        values.append(float(value))
    mean = distribution.get("mean")
    require(
        isinstance(mean, (int, float))
        and not isinstance(mean, bool)
        and math.isfinite(float(mean))
        and float(mean) > 0.0,
        f"{label} has an invalid {name} mean.",
    )
    calculated = sum(values) / len(values)
    require(
        math.isclose(float(mean), calculated, rel_tol=1.0e-12, abs_tol=1.0e-12),
        f"{label} {name} mean does not match its samples.",
    )
    return float(mean)


def verify_run(
    path: Path,
    *,
    contract: str,
    trial: int,
    physical_batch: int,
    capacity_record: dict[str, Any],
    capacity_payload: dict[str, Any],
    expected_gpu_uuid: str,
    expected_scene_sha256: str,
    expected_source: dict[str, Any],
    expected_manifest_record: dict[str, Any],
    expected_native: dict[str, Any],
    expected_adapter: dict[str, Any],
) -> dict[str, Any]:
    label = f"{contract} trial {trial} P{physical_batch} run"
    payload = load_json(path, label=label)
    require(payload.get("schema_version") == RUN_SCHEMA, f"{label} schema differs.")
    require(payload.get("pass") is True and payload.get("renderer") == "custom", f"{label} failed.")
    require(payload.get("output_contract") == contract, f"{label} contract differs.")
    camera = payload.get("camera_contract") or {}
    require(
        camera.get("trajectory_id") == PRIMARY_B128_TRAJECTORY_ID
        and camera.get("batch") == 128
        and camera.get("width") == 128
        and camera.get("height") == 128
        and camera.get("warmup_frames") == 8
        and camera.get("measured_frames") == 100
        and camera.get("motion_classification") == "independently-changing-camera-batch"
        and camera.get("unchanged_measured_camera_frame_pairs") == 0,
        f"{label} camera contract differs.",
    )
    upload = camera.get("camera_upload_seconds")
    require(
        isinstance(upload, (int, float))
        and not isinstance(upload, bool)
        and math.isfinite(float(upload))
        and upload >= 0,
        f"{label} camera upload time is invalid.",
    )
    require((payload.get("scene") or {}).get("sha256") == expected_scene_sha256, f"{label} scene differs.")
    config = payload.get("runner_config") or {}
    expected_config = {
        "warmup_frames": 8,
        "measured_frames": 100,
        "capacity_headroom": 1.05,
        "custom_depth_buckets": 128,
        "custom_depth_bucket_group": 8,
        "custom_max_physical_views": physical_batch,
        "capture_last_output": trial == 1,
        "independent_trial": trial,
        "profile_control": False,
    }
    for field, expected in expected_config.items():
        require(config.get(field) == expected, f"{label} config {field} differs.")
    require(same_artifact(config.get("capacity_calibration"), capacity_record), f"{label} runner capacity differs.")
    require(config.get("flashgs_demand_survey") is None, f"{label} unexpectedly uses a FlashGS survey.")
    execution = payload.get("backend_execution") or {}
    require(
        execution.get("logical_batch") == 128
        and execution.get("physical_batch") == physical_batch
        and execution.get("native_submissions_per_logical_batch") == 128 // physical_batch
        and execution.get("measured_render_requests") == 100
        and execution.get("measured_native_camera_executions") == 12_800,
        f"{label} execution schedule differs.",
    )
    capacity = payload.get("capacity") or {}
    require(same_artifact(capacity.get("calibration_artifact"), capacity_record), f"{label} capacity binding differs.")
    require(
        capacity.get("installed")
        == (capacity_payload.get("capacity") or {}).get("installed"),
        f"{label} installed capacity differs.",
    )
    require(
        capacity.get("logical_batch") == 128
        and capacity.get("physical_batch") == physical_batch
        and capacity.get("native_submissions_per_logical_batch") == 128 // physical_batch,
        f"{label} consumed capacity schedule differs.",
    )
    observed = ((capacity.get("timed_verification") or {}).get("observed") or {})
    require(
        observed.get("measured_max_visible_overflow") == 0
        and observed.get("measured_max_intersection_overflow") == 0,
        f"{label} overflowed.",
    )
    require((payload.get("output_validation") or {}).get("pass") is True, f"{label} output validation failed.")
    timing_mean(payload, "gpu_batch_ms", label=label)
    timing_mean(payload, "synchronized_wall_batch_ms", label=label)
    environment = payload.get("environment") or {}
    require(environment.get("gpu_uuid") == expected_gpu_uuid, f"{label} GPU differs.")
    provenance = environment.get("source_provenance") or {}
    require(source_identity(provenance) == expected_source, f"{label} source differs.")
    require(same_artifact(provenance.get("manifest"), expected_manifest_record), f"{label} source manifest differs.")
    native = verify_native_record(environment.get("native_extension"), label=f"{label} native extension")
    require(same_artifact(native, expected_native), f"{label} native extension differs.")
    require(
        same_artifact(
            environment.get("flashgs_adapter_attestation"), expected_adapter
        ),
        f"{label} adapter attestation differs.",
    )
    verify_occupancy(
        environment.get("node_occupancy"),
        expected_gpu_uuid=expected_gpu_uuid,
        label=f"{label} occupancy",
    )
    capture = payload.get("fidelity_capture")
    if trial == 1:
        resolve_artifact(capture, label=f"{label} fidelity capture")
    else:
        require(capture is None, f"{label} unexpectedly performed a fidelity capture.")
    return payload


def verify_fidelity(
    path: Path,
    *,
    contract: str,
    run_record: dict[str, Any],
    oracle_manifest_record: dict[str, Any],
) -> dict[str, Any]:
    label = f"{contract} trial 1 fidelity"
    payload = load_json(path, label=label)
    require(payload.get("schema_version") == FIDELITY_SCHEMA, f"{label} schema differs.")
    require(
        payload.get("pass") is True
        and payload.get("renderer") == "custom"
        and payload.get("output_contract") == contract
        and payload.get("batch") == 128
        and payload.get("trajectory_id") == PRIMARY_B128_TRAJECTORY_ID,
        f"{label} failed or has the wrong identity.",
    )
    inputs = payload.get("input_artifacts") or {}
    require(same_artifact(inputs.get("run"), run_record), f"{label} uses another timed run.")
    require(same_artifact(inputs.get("oracle_manifest"), oracle_manifest_record), f"{label} uses another oracle.")
    for name, record in (payload.get("report_artifacts") or {}).items():
        resolve_artifact(record, label=f"{label} report {name}")
    return payload


def run_order() -> list[tuple[str, int, int]]:
    order: list[tuple[str, int, int]] = []
    for trial in TRIALS:
        full = (1, 128) if trial % 2 else (128, 1)
        rgb = (128, 1) if trial % 2 else (1, 128)
        order.extend(("full", trial, physical) for physical in full)
        order.extend(("rgb", trial, physical) for physical in rgb)
    return order


def run_id(contract: str, trial: int, physical_batch: int) -> str:
    return f"{contract}-trial{trial}-p{physical_batch}"


def publication_environment(benchmark_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    pinned = (str(benchmark_root / "src"), str(benchmark_root))
    inherited = [
        value
        for value in environment.get("PYTHONPATH", "").split(os.pathsep)
        if value and value not in pinned
    ]
    environment["PYTHONPATH"] = os.pathsep.join((*pinned, *inherited))
    environment["PROJECT_ROOT"] = str(benchmark_root)
    environment["VGR_PROJECT_ROOT"] = str(benchmark_root)
    return environment


def build_commands(
    *,
    python: Path,
    benchmark_root: Path,
    output_root: Path,
    scene_path: Path,
    source_manifest: Path,
    trajectory: Path,
    p128_capacity: Path,
    oracle: Path,
    camera_bundle: Path,
    adapter_attestation: Path,
    expected_gpu_uuid: str,
    semantic_topology: str,
    allow_nonheadline_gpu: bool,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    runner = benchmark_root / "benchmarks/run_flashgs_matched.py"
    comparator = benchmark_root / "benchmarks/compare_flashgs_matched_fidelity.py"
    p1_capacity = output_root / "capacity/p1.json"
    common = [
        str(python),
        str(runner),
        "--renderer",
        "custom",
        "--trajectory",
        str(trajectory),
        "--scene-path",
        str(scene_path),
        "--source-manifest",
        str(source_manifest),
        "--flashgs-adapter-attestation",
        str(adapter_attestation),
        "--expected-gpu-uuid",
        expected_gpu_uuid,
        "--warmup-frames",
        "8",
        "--measured-frames",
        "100",
        "--capacity-headroom",
        "1.05",
        "--semantic-topology",
        semantic_topology,
    ]
    capacity_command = [
        *common,
        "--output-contract",
        "full",
        "--capacity-calibration-only",
        "--custom-max-physical-views",
        "1",
        "--no-capture-last-output",
        "--output",
        str(p1_capacity),
    ]
    if allow_nonheadline_gpu:
        capacity_command.append("--allow-nonheadline-gpu")

    timed: list[dict[str, Any]] = []
    fidelity: list[dict[str, Any]] = []
    for contract, trial, physical in run_order():
        identifier = run_id(contract, trial, physical)
        result = output_root / "runs" / f"{identifier}.json"
        capacity = p1_capacity if physical == 1 else p128_capacity
        command = [
            *common,
            "--output-contract",
            contract,
            "--capacity-calibration",
            str(capacity),
            "--custom-max-physical-views",
            str(physical),
            "--independent-trial",
            str(trial),
            ("--capture-last-output" if trial == 1 else "--no-capture-last-output"),
            "--output",
            str(result),
        ]
        if allow_nonheadline_gpu:
            command.append("--allow-nonheadline-gpu")
        timed.append(
            {
                "run_id": identifier,
                "contract": contract,
                "trial": trial,
                "physical_batch": physical,
                "capacity": str(capacity),
                "result": str(result),
                "log": str(output_root / "logs" / f"timed-{identifier}.log"),
                "command": command,
            }
        )
        if trial == 1:
            fidelity_root = output_root / "fidelity" / identifier
            fidelity.append(
                {
                    "run_id": identifier,
                    "contract": contract,
                    "result": str(fidelity_root / "matched-fidelity-summary.json"),
                    "log": str(output_root / "logs" / f"fidelity-{identifier}.log"),
                    "command": [
                        str(python),
                        str(comparator),
                        "--run",
                        str(result),
                        "--oracle",
                        str(oracle),
                        "--camera-bundle",
                        str(camera_bundle),
                        "--output-dir",
                        str(fidelity_root),
                    ],
                }
            )
    return capacity_command, timed, fidelity


def build_plan(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark_root = Path(args.benchmark_root).resolve(strict=True)
    require(benchmark_root.is_dir(), "Benchmark root is not a directory.")
    output_root = Path(args.output_root).resolve()
    require(
        not output_root.is_relative_to(benchmark_root),
        "Ablation output root must be outside the frozen benchmark checkout.",
    )
    python = executable_path(args.python, label="benchmark Python")
    scene = regular_file(args.scene_path, label="Home Scan scene")
    source_manifest = regular_file(args.source_manifest, label="source manifest")
    trajectory = regular_file(args.trajectory, label="B128 trajectory")
    p128_capacity = regular_file(args.p128_capacity, label="matrix P128 capacity")
    oracle = regular_file(args.oracle, label="matrix B128 oracle")
    camera_bundle = regular_file(args.camera_bundle, label="matrix B128 camera bundle")
    matrix_summary = regular_file(args.matrix_summary, label="matched matrix summary")
    runner = regular_file(
        benchmark_root / "benchmarks/run_flashgs_matched.py",
        label="matched runner",
    )
    comparator = regular_file(
        benchmark_root / "benchmarks/compare_flashgs_matched_fidelity.py",
        label="fidelity comparator",
    )
    scene_record = artifact_record(scene, label="Home Scan scene")
    require(scene_record["sha256"] == args.scene_sha256, "Home Scan scene SHA-256 differs.")
    source_payload, source_record = verify_source_manifest(
        source_manifest,
        benchmark_root=benchmark_root,
    )
    expected_source = expected_source_identity(source_payload, source_record)
    trajectory_payload, trajectory_record, arrays_record = verify_trajectory(
        trajectory,
        scene_sha256=args.scene_sha256,
    )
    p128_payload, native = verify_capacity(
        p128_capacity,
        physical_batch=128,
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_scene_sha256=args.scene_sha256,
        expected_source=expected_source,
        expected_manifest_record=source_record,
        expected_native=None,
        require_matrix_direct_config=True,
    )
    adapter_record = (p128_payload.get("environment") or {}).get(
        "flashgs_adapter_attestation"
    )
    adapter_path = resolve_artifact(
        adapter_record,
        label="matrix FlashGS adapter attestation",
    )
    _oracle_payload, oracle_manifest_record = verify_oracle_inputs(
        oracle=oracle,
        camera_bundle=camera_bundle,
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_scene_sha256=args.scene_sha256,
        expected_source=expected_source,
    )
    oracle_manifest = oracle.with_suffix(".manifest.json")
    p128_capacity_record = artifact_record(
        p128_capacity,
        label="matrix P128 capacity",
    )
    verify_matrix_bindings(
        summary_path=matrix_summary,
        p128_capacity_record=p128_capacity_record,
        oracle_manifest_record=oracle_manifest_record,
    )
    inputs = {
        "scene": scene_record,
        "source_manifest": source_record,
        "trajectory": trajectory_record,
        "trajectory_arrays": arrays_record,
        "p128_capacity": p128_capacity_record,
        "matrix_summary": artifact_record(
            matrix_summary,
            label="matched matrix summary",
        ),
        "oracle": artifact_record(oracle, label="matrix B128 oracle"),
        "oracle_manifest": oracle_manifest_record,
        "camera_bundle": artifact_record(
            camera_bundle,
            label="matrix B128 camera bundle",
        ),
        "adapter_attestation": artifact_record(
            adapter_path,
            label="matrix FlashGS adapter attestation",
        ),
        "custom_native_extension": native,
        "runner": artifact_record(runner, label="matched runner"),
        "fidelity_comparator": artifact_record(
            comparator,
            label="fidelity comparator",
        ),
    }
    capacity_command, timed, fidelity = build_commands(
        python=python,
        benchmark_root=benchmark_root,
        output_root=output_root,
        scene_path=scene,
        source_manifest=source_manifest,
        trajectory=trajectory,
        p128_capacity=p128_capacity,
        oracle=oracle,
        camera_bundle=camera_bundle,
        adapter_attestation=adapter_path,
        expected_gpu_uuid=args.expected_gpu_uuid,
        semantic_topology=args.semantic_topology,
        allow_nonheadline_gpu=args.allow_nonheadline_gpu,
    )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "batch": 128,
        "benchmark_root": str(benchmark_root),
        "python": str(python),
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "scene_sha256": args.scene_sha256,
        "semantic_topology": args.semantic_topology,
        "warmup_frames": 8,
        "measured_frames": 100,
        "capacity_headroom": 1.05,
        "output_root": str(output_root),
        "inputs": inputs,
        "capacity_command": capacity_command,
        "timed_runs": timed,
        "fidelity_runs": fidelity,
        "run_order": [item["run_id"] for item in timed],
    }
    context = {
        "benchmark_root": benchmark_root,
        "output_root": output_root,
        "python": python,
        "source_payload": source_payload,
        "source_record": source_record,
        "expected_source": expected_source,
        "trajectory_payload": trajectory_payload,
        "p128_capacity_payload": p128_payload,
        "native": native,
        "adapter": inputs["adapter_attestation"],
        "oracle_manifest": inputs["oracle_manifest"],
    }
    return plan, context


def write_or_verify_plan(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    expected_bytes = canonical_json_bytes(expected) + b"\n"
    if path.exists() or path.is_symlink():
        resolved = regular_file(path, label="ablation plan")
        actual_bytes = resolved.read_bytes()
        require(
            actual_bytes == expected_bytes,
            "Existing ablation plan differs from current paths, hashes, or commands.",
        )
    else:
        write_exclusive_json(path, expected)
    return artifact_record(path, label="ablation plan")


def write_or_verify_json(path: Path, expected: Any, *, label: str) -> dict[str, Any]:
    expected_bytes = canonical_json_bytes(expected) + b"\n"
    if path.exists() or path.is_symlink():
        resolved = regular_file(path, label=label)
        require(
            resolved.read_bytes() == expected_bytes,
            f"Existing {label} differs from the immutable plan.",
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(expected_bytes)
        except FileExistsError as error:
            raise AblationError(f"Refusing to overwrite {label}: {path}.") from error
    return artifact_record(path, label=label)


def _verify_receipt_artifacts(value: Any, *, label: str) -> None:
    if value is None:
        return
    if isinstance(value, dict) and set(value) >= {"path", "bytes", "sha256"}:
        resolve_artifact(value, label=label)
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _verify_receipt_artifacts(nested, label=f"{label}.{key}")
        return
    raise AblationError(f"{label} receipt artifact tree is malformed.")


def verify_stage_receipt(
    path: Path,
    *,
    stage_id: str,
    plan_sha256: str,
    command: list[str],
) -> dict[str, Any]:
    payload = load_json(path, label=f"{stage_id} stage receipt")
    require(
        set(payload)
        == {
            "artifacts",
            "command_sha256",
            "exit_code",
            "plan_sha256",
            "schema_version",
            "stage_id",
        },
        f"{stage_id} stage receipt fields differ.",
    )
    require(payload.get("schema_version") == STAGE_RECEIPT_SCHEMA, f"{stage_id} receipt schema differs.")
    require(payload.get("stage_id") == stage_id, f"{stage_id} receipt identity differs.")
    require(payload.get("plan_sha256") == plan_sha256, f"{stage_id} receipt uses another plan.")
    require(payload.get("command_sha256") == command_sha256(command), f"{stage_id} receipt command differs.")
    require(payload.get("exit_code") == 0, f"{stage_id} recorded a nonzero exit.")
    _verify_receipt_artifacts(payload.get("artifacts"), label=f"{stage_id} artifacts")
    return payload


def stage_receipt_path(output_root: Path, stage_id: str) -> Path:
    return output_root / "receipts" / f"{stage_id}.json"


def execute_stage(
    *,
    stage_id: str,
    command: list[str],
    result_path: Path,
    log_path: Path,
    plan_sha256: str,
    output_root: Path,
    environment: dict[str, str],
    benchmark_root: Path,
    timeout_seconds: int,
    sentinel: str,
    validate: Callable[[Path], dict[str, Any]],
    extra_artifacts: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = stage_receipt_path(output_root, stage_id)
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = verify_stage_receipt(
            receipt_path,
            stage_id=stage_id,
            plan_sha256=plan_sha256,
            command=command,
        )
        result = validate(result_path)
        print(f"CUSTOM_VECTORIZATION_ABLATION_STAGE_REUSED {stage_id}")
        return result, receipt

    partials = [path for path in (result_path, log_path) if path.exists() or path.is_symlink()]
    require(
        not partials,
        f"{stage_id} has unreceipted partial output; use a new output root: {partials}.",
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("x", encoding="utf-8") as stream:
            subprocess.run(
                command,
                cwd=benchmark_root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_seconds,
            )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise AblationError(f"{stage_id} process failed; partial evidence was preserved.") from error
    log_text = log_path.read_text(encoding="utf-8", errors="strict")
    require(log_text.count(sentinel) == 1, f"{stage_id} success sentinel is missing or duplicated.")
    result = validate(result_path)
    artifacts = {
        "result": artifact_record(result_path, label=f"{stage_id} result"),
        "log": artifact_record(log_path, label=f"{stage_id} log"),
        **extra_artifacts(result),
    }
    receipt = {
        "schema_version": STAGE_RECEIPT_SCHEMA,
        "stage_id": stage_id,
        "plan_sha256": plan_sha256,
        "command_sha256": command_sha256(command),
        "exit_code": 0,
        "artifacts": artifacts,
    }
    write_exclusive_json(receipt_path, receipt)
    print(f"CUSTOM_VECTORIZATION_ABLATION_STAGE_OK {stage_id}")
    return result, receipt


def receipt_result_record(receipt: dict[str, Any]) -> dict[str, Any]:
    record = (receipt.get("artifacts") or {}).get("result")
    resolve_artifact(record, label="stage result")
    return record


def assemble_output(
    *,
    plan: dict[str, Any],
    plan_record: dict[str, Any],
    context: dict[str, Any],
    run_results: dict[str, dict[str, Any]],
    run_receipts: dict[str, dict[str, Any]],
    fidelity_receipts: dict[str, dict[str, Any]],
    p1_capacity_payload: dict[str, Any],
    p1_capacity_record: dict[str, Any],
) -> dict[str, Any]:
    # Receipts bind the plan.  These already-validated execution objects are
    # intentionally not copied into the exact six-field public raw schema.
    del plan_record, context, p1_capacity_payload
    capacity_records = {
        1: p1_capacity_record,
        128: plan["inputs"]["p128_capacity"],
    }
    require(
        not same_artifact(capacity_records[1], capacity_records[128]),
        "P1 and P128 capacity artifacts must be distinct.",
    )
    entries: list[dict[str, Any]] = []
    seen_run_artifacts: set[tuple[str, int]] = set()
    seen_occupancy_artifacts: set[tuple[str, int]] = set()
    equations: dict[str, list[dict[str, Any]]] = {"full": [], "rgb": []}
    for item in plan["timed_runs"]:
        identifier = item["run_id"]
        payload = run_results[identifier]
        equations[item["contract"]].append(payload.get("equation_contract") or {})
        run_record = receipt_result_record(run_receipts[identifier])
        run_identity = (run_record["sha256"], run_record["bytes"])
        require(run_identity not in seen_run_artifacts, "Timed process output was reused across design cells.")
        seen_run_artifacts.add(run_identity)
        occupancy_record = (payload.get("environment") or {}).get("node_occupancy")
        occupancy_identity = (occupancy_record["sha256"], occupancy_record["bytes"])
        require(
            occupancy_identity not in seen_occupancy_artifacts,
            "Occupancy evidence was reused across fresh processes.",
        )
        seen_occupancy_artifacts.add(occupancy_identity)
        fidelity_record = None
        if item["trial"] == 1:
            fidelity_record = receipt_result_record(fidelity_receipts[identifier])
        entries.append(
            {
                "run_id": identifier,
                "contract": item["contract"],
                "trial": item["trial"],
                "physical_batch": item["physical_batch"],
                "run": run_record,
                "capacity": capacity_records[item["physical_batch"]],
                "fidelity": fidelity_record,
            }
        )
    for contract, values in equations.items():
        require(
            values and all(value == values[0] for value in values[1:]),
            f"Ablation {contract} equation contracts differ across runs.",
        )
    full_common = {
        key: value
        for key, value in equations["full"][0].items()
        if key != "output_dtypes"
    }
    rgb_common = {
        key: value
        for key, value in equations["rgb"][0].items()
        if key != "output_dtypes"
    }
    require(
        full_common == rgb_common,
        "Full and RGB ablations use different rendering equations.",
    )

    ratios: dict[str, list[dict[str, Any]]] = {"full": [], "rgb": []}
    by_key = {
        (item["contract"], item["trial"], item["physical_batch"]): run_results[item["run_id"]]
        for item in plan["timed_runs"]
    }
    for contract in CONTRACTS:
        for trial in TRIALS:
            p1 = by_key[(contract, trial, 1)]
            p128 = by_key[(contract, trial, 128)]
            p1_gpu = timing_mean(p1, "gpu_batch_ms", label=f"{contract} trial {trial} P1")
            p128_gpu = timing_mean(p128, "gpu_batch_ms", label=f"{contract} trial {trial} P128")
            p1_wall = timing_mean(p1, "synchronized_wall_batch_ms", label=f"{contract} trial {trial} P1")
            p128_wall = timing_mean(p128, "synchronized_wall_batch_ms", label=f"{contract} trial {trial} P128")
            ratios[contract].append(
                {
                    "trial": trial,
                    "cuda_speedup_p128_over_p1": p1_gpu / p128_gpu,
                    "wall_speedup_p128_over_p1": p1_wall / p128_wall,
                }
            )
    # Passing proves the design and evidence, not that P128 is faster.  Ratios
    # below one are an admissible and important publication result.
    return {
        "schema_version": OUTPUT_SCHEMA,
        "pass": True,
        "batch": 128,
        "runs": entries,
        "run_order": plan["run_order"],
        "ratios": ratios,
    }


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--scene-sha256", default=DEFAULT_SCENE_SHA256)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--p128-capacity", type=Path, required=True)
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--camera-bundle", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--raw-output",
        type=Path,
        help=(
            "Exact publication raw-result alias. Defaults to "
            "OUTPUT_ROOT/custom-vectorization-ablation.json."
        ),
    )
    parser.add_argument(
        "--gate-result-spec",
        type=Path,
        help="Write a run_bounded_gate.py normalized-result spec after success.",
    )
    parser.add_argument("--semantic-topology", default="spatial-octants-8")
    parser.add_argument("--allow-nonheadline-gpu", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--stage-timeout-seconds",
        type=int,
        default=7200,
        help="Finite timeout applied independently to each subprocess.",
    )
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    require(
        1 <= args.stage_timeout_seconds <= 7200,
        "Stage timeout must be between 1 and 7200 seconds.",
    )
    plan, context = build_plan(args)
    output_root: Path = context["output_root"]
    raw_output = (
        Path(args.raw_output).resolve()
        if args.raw_output is not None
        else output_root / "custom-vectorization-ablation.json"
    )
    require(
        not raw_output.is_relative_to(context["benchmark_root"]),
        "Raw publication output must be outside the frozen benchmark checkout.",
    )
    result_spec_path = (
        Path(args.gate_result_spec).resolve()
        if args.gate_result_spec is not None
        else None
    )
    if result_spec_path is not None:
        require(
            not result_spec_path.is_relative_to(context["benchmark_root"]),
            "Gate result spec must be outside the frozen benchmark checkout.",
        )
    # Output destinations are part of the immutable plan just like commands.
    plan["raw_output"] = str(raw_output)
    plan["gate_result_spec"] = (
        str(result_spec_path) if result_spec_path is not None else None
    )
    plan_path = output_root / "ablation-plan.json"
    plan_record = write_or_verify_plan(plan_path, plan)
    subcommands_path = output_root / "timed-subcommands.json"
    subcommands_record = write_or_verify_json(
        subcommands_path,
        [item["command"] for item in plan["timed_runs"]],
        label="timed subcommand plan",
    )
    if args.plan_only:
        print(
            "CUSTOM_VECTORIZATION_ABLATION_PLAN_OK "
            + json.dumps(
                {
                    "plan": plan_record,
                    "subcommands": subcommands_record,
                    "timed_processes": len(plan["timed_runs"]),
                },
                sort_keys=True,
            )
        )
        return

    environment = publication_environment(context["benchmark_root"])
    plan_sha = plan_record["sha256"]
    p1_path = output_root / "capacity/p1.json"
    p1_log = output_root / "logs/capacity-p1.log"

    def validate_p1(path: Path) -> dict[str, Any]:
        payload, _native = verify_capacity(
            path,
            physical_batch=1,
            expected_gpu_uuid=args.expected_gpu_uuid,
            expected_scene_sha256=args.scene_sha256,
            expected_source=context["expected_source"],
            expected_manifest_record=context["source_record"],
            expected_native=context["native"],
            require_matrix_direct_config=False,
        )
        return payload

    def capacity_extras(payload: dict[str, Any]) -> dict[str, Any]:
        environment_payload = payload.get("environment") or {}
        return {
            "occupancy": environment_payload.get("node_occupancy"),
            "native_extension": environment_payload.get("native_extension"),
        }

    p1_payload, p1_receipt = execute_stage(
        stage_id="capacity-p1",
        command=plan["capacity_command"],
        result_path=p1_path,
        log_path=p1_log,
        plan_sha256=plan_sha,
        output_root=output_root,
        environment=environment,
        benchmark_root=context["benchmark_root"],
        timeout_seconds=args.stage_timeout_seconds,
        sentinel="FLASHGS_MATCHED_CAPACITY_CALIBRATION_OK",
        validate=validate_p1,
        extra_artifacts=capacity_extras,
    )
    p1_record = receipt_result_record(p1_receipt)

    capacities = {1: (p1_payload, p1_record), 128: (context["p128_capacity_payload"], plan["inputs"]["p128_capacity"])}
    run_results: dict[str, dict[str, Any]] = {}
    run_receipts: dict[str, dict[str, Any]] = {}
    for item in plan["timed_runs"]:
        identifier = item["run_id"]
        physical = item["physical_batch"]
        capacity_payload, capacity_record = capacities[physical]

        def validate_timed(
            path: Path,
            *,
            item: dict[str, Any] = item,
            capacity_payload: dict[str, Any] = capacity_payload,
            capacity_record: dict[str, Any] = capacity_record,
        ) -> dict[str, Any]:
            return verify_run(
                path,
                contract=item["contract"],
                trial=item["trial"],
                physical_batch=item["physical_batch"],
                capacity_record=capacity_record,
                capacity_payload=capacity_payload,
                expected_gpu_uuid=args.expected_gpu_uuid,
                expected_scene_sha256=args.scene_sha256,
                expected_source=context["expected_source"],
                expected_manifest_record=context["source_record"],
                expected_native=context["native"],
                expected_adapter=context["adapter"],
            )

        def run_extras(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "occupancy": (payload.get("environment") or {}).get("node_occupancy"),
                "fidelity_capture": payload.get("fidelity_capture"),
            }

        result, receipt = execute_stage(
            stage_id=f"timed-{identifier}",
            command=item["command"],
            result_path=Path(item["result"]),
            log_path=Path(item["log"]),
            plan_sha256=plan_sha,
            output_root=output_root,
            environment=environment,
            benchmark_root=context["benchmark_root"],
            timeout_seconds=args.stage_timeout_seconds,
            sentinel="FLASHGS_MATCHED_RUN_OK",
            validate=validate_timed,
            extra_artifacts=run_extras,
        )
        run_results[identifier] = result
        run_receipts[identifier] = receipt

    # Only now, after all twelve timed processes, run the four oracle checks.
    fidelity_receipts: dict[str, dict[str, Any]] = {}
    for item in plan["fidelity_runs"]:
        identifier = item["run_id"]
        run_record = receipt_result_record(run_receipts[identifier])

        def validate_oracle(
            path: Path,
            *,
            item: dict[str, Any] = item,
            run_record: dict[str, Any] = run_record,
        ) -> dict[str, Any]:
            return verify_fidelity(
                path,
                contract=item["contract"],
                run_record=run_record,
                oracle_manifest_record=context["oracle_manifest"],
            )

        def fidelity_extras(payload: dict[str, Any]) -> dict[str, Any]:
            return {"reports": payload.get("report_artifacts") or {}}

        _result, receipt = execute_stage(
            stage_id=f"fidelity-{identifier}",
            command=item["command"],
            result_path=Path(item["result"]),
            log_path=Path(item["log"]),
            plan_sha256=plan_sha,
            output_root=output_root,
            environment=environment,
            benchmark_root=context["benchmark_root"],
            timeout_seconds=args.stage_timeout_seconds,
            sentinel="FLASHGS_MATCHED_FIDELITY_OK",
            validate=validate_oracle,
            extra_artifacts=fidelity_extras,
        )
        fidelity_receipts[identifier] = receipt

    output = assemble_output(
        plan=plan,
        plan_record=plan_record,
        context=context,
        run_results=run_results,
        run_receipts=run_receipts,
        fidelity_receipts=fidelity_receipts,
        p1_capacity_payload=p1_payload,
        p1_capacity_record=p1_record,
    )
    output_path = raw_output
    expected_bytes = canonical_json_bytes(output) + b"\n"
    if output_path.exists() or output_path.is_symlink():
        existing = regular_file(output_path, label="ablation output")
        require(existing.read_bytes() == expected_bytes, "Existing ablation output differs from verified evidence.")
    else:
        write_exclusive_json(output_path, output)
    if result_spec_path is not None:
        result_spec = {
            "schema_version": "publication-gate-result-spec-v1",
            "gate_id": "custom-vectorization-ablation",
            "checks": {
                "design_cells": 12,
                "fresh_processes": 12,
                "counterbalanced": True,
                "reported_without_win_requirement": True,
            },
            "evidence": {"raw_result": str(output_path)},
            "identity": {
                "gpu_uuid": args.expected_gpu_uuid,
                "native_extension": context["native"]["path"],
                "scene_sha256": args.scene_sha256,
                "source_manifest": context["source_record"]["path"],
            },
        }
        write_or_verify_json(
            result_spec_path,
            result_spec,
            label="vectorization gate result spec",
        )
    print(
        "CUSTOM_VECTORIZATION_ABLATION_OK "
        + json.dumps(
            {
                "output": artifact_record(output_path, label="ablation output"),
                "pass": True,
                "timed_processes": 12,
                "fidelity_comparisons": 4,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
