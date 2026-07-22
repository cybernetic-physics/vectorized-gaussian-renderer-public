"""Immutable identities for the matched Custom-versus-FlashGS evidence."""

from __future__ import annotations

import json
import hashlib
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from isaacsim_gaussian_renderer.benchmark_manifest import (
    file_sha256,
    sha256_json,
)
from isaacsim_gaussian_renderer.evaluation.evidence_bundle import (
    resolve_artifact_record,
)


RENDERER_RUN_SCHEMA = "flashgs-matched-renderer-run-v4"
CAPACITY_CALIBRATION_SCHEMA = "flashgs-matched-capacity-calibration-v1"
CAPACITY_CONSUMPTION_SCHEMA = "flashgs-matched-capacity-consumption-v1"
FLASHGS_DEMAND_SURVEY_SCHEMA = "flashgs-matched-flashgs-demand-survey-v1"
FLASHGS_DEMAND_SURVEY_CONSUMPTION_SCHEMA = "flashgs-matched-flashgs-demand-survey-consumption-v1"
FLASHGS_TRAJECTORY_PREFIX_PROOF_SCHEMA = "flashgs-matched-primary-trajectory-prefix-proof-v1"
FLASHGS_COUNTER_SOURCE_AUDIT_SCHEMA = "flashgs-matched-flashgs-counter-source-audit-v1"
TIMED_CAPACITY_VERIFICATION_SCHEMA = "flashgs-matched-timed-capacity-verification-v1"
GSPLAT_ORACLE_SCHEMA = "flashgs-matched-gsplat-oracle-v4"
FIDELITY_SCHEMA = "flashgs-matched-fidelity-v4"
SUMMARY_SCHEMA = "flashgs-matched-summary-v4"
PROFILE_SCHEMA = "flashgs-matched-profile-v3"
GSPLAT_BUILD_ATTESTATION_SCHEMA = "flashgs-matched-gsplat-build-v1"
SOURCE_MANIFEST_SCHEMA = "renderer-source-manifest-v2"
NODE_OCCUPANCY_SCHEMA = "flashgs-matched-node-occupancy-v2"

PINNED_GSPLAT_COMMIT = "77ab983ffe43420b2131669cb35776b883ca4c3c"
PINNED_GSPLAT_PATCH_SHA256 = "ea30120f728d5c728e082fec686f15056de5813a4e8fcc2c6b4e68aa324ca36d"
PINNED_GSPLAT_PATCHED_UTILS_SHA256 = "ad135b17732963793d149e32e15a33ead6efac49dfe4b50cb72c59e8c123c0a8"

PRIMARY_BATCHES = (1, 8, 32, 64, 128, 256, 512, 1024)
PRIMARY_CHUNKED_BATCHES = (512, 1024)
PRIMARY_CHUNKED_PHYSICAL_VIEWS = 128
HEADLINE_GPU_NAME = "NVIDIA L4"
HEADLINE_GPU_UUID = "GPU-ebf6dc95-db46-4e1f-6e95-492c5c787805"
HEADLINE_COMPUTE_CAPABILITY = (8, 9)
HEADLINE_TORCH_CUDA_ARCH_LIST = "8.9"
# Pinned gsplat commit 77ab983 defines GAUSSIAN_EXTEND as 3.33f in
# gsplat/cuda/include/Common.h.  The matched candidates must use that exact
# support cutoff; 3.0 is the separate OVRTX contract, not the gsplat oracle
# contract used here.
MATCHED_GAUSSIAN_SUPPORT_SIGMA = 3.33
MATCHED_PROJECTION_RULES = {
    "frustum_depth_interval": "inclusive-near-and-far",
    "invalid_covariance_2d": "determinant<=0",
    "projected_radius_covariance_diagonal_floor": None,
    "projected_radius_min_pixels": 0,
    "projected_radius_clip": "cull-only-when-both-radii<=0",
}
GSPLAT_SUPPORT_PROBE_EXPECTED_RADII = [[34, 34], [26, 26]]
GSPLAT_BOUNDARY_PROBE_EXPECTED_RADII = [[34, 34]]
PRIMARY_TRAJECTORY_IDS = {
    1: "a901ed9a3d7a62bffc2860ecefa2cdf8eaf2196085bca067215ac0469fbc8dfe",
    8: "39ab99be70fef112dd7f82636f28d61e2b67764b7f7f7eb6d1e6f401bddb3ca3",
    32: "86febe653241edbbd94d321ad22ed3214c6ff09130434a3c3f5c587107cd5d15",
    64: "e30050bf2d873825cf7cdebbe799c911ddaee5cc15d90e1c0dd8adc2c2e62cc0",
    128: "c9a7ef7727761865263c3432954b259c54e2065dfd2326e65494583083704925",
    256: "f0a68d425ca12f38b9d62c6fd445caae9d75472bdf9bacc29f563b93817233f4",
    512: "f750294341d523b1b46f626b609fbd5e2a0a43d28f0aa35a4614d6f0f6ed8c7d",
    1024: "375a6711e86333621448c94ca1ad7985bd2687c4b585f68c823624725c223910",
}


def primary_fidelity_selection(
    batch: int,
    *,
    trajectory_timesteps: int = 108,
) -> tuple[tuple[int, int], ...]:
    """Return frozen ``(step, camera)`` pairs for matched fidelity."""

    if batch <= 0:
        raise ValueError("batch must be positive.")
    if trajectory_timesteps < 108:
        raise ValueError("The primary fidelity suite requires 108 steps.")
    view_count = min(8, batch)
    if view_count == 1:
        camera_indices = {0}
    else:
        camera_indices = {int(index * (batch - 1) / (view_count - 1)) for index in range(view_count)}
    if batch in PRIMARY_CHUNKED_BATCHES:
        for boundary in range(
            PRIMARY_CHUNKED_PHYSICAL_VIEWS,
            batch,
            PRIMARY_CHUNKED_PHYSICAL_VIEWS,
        ):
            camera_indices.update((boundary - 1, boundary))
    steps = (8, 57, 107) if batch in (1, 128, 512, 1024) else (107,)
    return tuple((step, camera_index) for step in steps for camera_index in sorted(camera_indices))


def primary_max_physical_views(renderer: str, batch: int) -> int | None:
    """Return the only internal camera chunk allowed by the primary matrix."""

    if renderer == "custom" and batch in PRIMARY_CHUNKED_BATCHES:
        return PRIMARY_CHUNKED_PHYSICAL_VIEWS
    return None


def primary_execution_schedule_failures(
    renderer: str,
    batch: int,
    execution: dict[str, Any],
    capacity: dict[str, Any],
) -> list[str]:
    """Cross-bind the reported execution and capacity schedules for one row."""

    expected_physical_batch = (primary_max_physical_views(renderer, batch) or batch) if renderer == "custom" else 1
    expected_submissions = (batch + expected_physical_batch - 1) // expected_physical_batch
    failures: list[str] = []
    for label, record in (("execution", execution), ("capacity", capacity)):
        if record.get("logical_batch") != batch:
            failures.append(f"{renderer} {label} logical batch is inconsistent")
        if record.get("physical_batch") != expected_physical_batch:
            failures.append(f"{renderer} {label} physical batch is inconsistent")
        if record.get("native_submissions_per_logical_batch") != expected_submissions:
            failures.append(f"{renderer} {label} native submission count is inconsistent")
    return failures


def active_cuda_device_uuid(cuda: Any) -> str | None:
    """Resolve the current CUDA-visible device UUID without allocating."""

    properties = cuda.get_device_properties(cuda.current_device())
    value = str(getattr(properties, "uuid", "")).strip()
    if not value:
        return None
    if value.startswith(("GPU-", "MIG-")):
        return value
    return f"GPU-{value}"


def artifact_record(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {artifact_path}.")
    return {
        "path": str(artifact_path),
        "bytes": artifact_path.stat().st_size,
        "sha256": file_sha256(artifact_path),
    }


def trajectory_artifacts(path: str | Path) -> dict[str, Any]:
    """Return hash-bound records for a trajectory JSON and its declared NPZ."""

    json_path = Path(path).resolve()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    arrays_file = payload.get("arrays_file")
    if not isinstance(arrays_file, str) or not arrays_file or Path(arrays_file).is_absolute():
        raise ValueError("Trajectory arrays_file must be a relative path.")
    npz_path = (json_path.parent / arrays_file).resolve()
    if not npz_path.is_relative_to(json_path.parent):
        raise ValueError("Trajectory arrays_file escapes its contract directory.")
    return {
        "json": artifact_record(json_path),
        "npz": artifact_record(npz_path),
    }


def prove_trajectory_prefixes(
    contract_paths: Mapping[int, str | Path],
    *,
    expected_batches: Sequence[int] = PRIMARY_BATCHES,
    expected_timesteps: int = 108,
) -> dict[str, Any]:
    """Prove every smaller camera contract is an exact B-max prefix.

    Equality is byte-exact after the trajectory loader has verified each JSON
    manifest and NPZ payload.  The proof binds both files for every contract so
    it can be recomputed when a survey is resumed or consumed.
    """

    from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (
        array_sha256,
        load_trajectory,
    )

    batches = tuple(int(value) for value in expected_batches)
    if not batches or tuple(sorted(set(batches))) != batches:
        raise ValueError("expected_batches must be strictly increasing and unique.")
    if set(contract_paths) != set(batches):
        raise ValueError("Trajectory prefix proof requires exactly the expected batches.")
    canonical_batch = batches[-1]
    canonical = load_trajectory(contract_paths[canonical_batch])
    if canonical.batch != canonical_batch or canonical.timesteps != expected_timesteps:
        raise ValueError("Canonical trajectory shape differs from the prefix protocol.")

    contracts: dict[str, Any] = {}
    for batch in batches:
        path = Path(contract_paths[batch]).resolve()
        trajectory = load_trajectory(path)
        if trajectory.batch != batch or trajectory.timesteps != expected_timesteps:
            raise ValueError(f"B{batch} trajectory shape differs from the prefix protocol.")
        if (
            trajectory.width != canonical.width
            or trajectory.height != canonical.height
            or trajectory.scene_sha256 != canonical.scene_sha256
            or trajectory.route_sha256 != canonical.route_sha256
            or trajectory.near_plane != canonical.near_plane
            or trajectory.far_plane != canonical.far_plane
            or trajectory.motion_classification != canonical.motion_classification
        ):
            raise ValueError(f"B{batch} trajectory metadata differs from B{canonical_batch}.")
        canonical_viewmats = np.ascontiguousarray(canonical.viewmats[:, :batch])
        canonical_intrinsics = np.ascontiguousarray(canonical.intrinsics[:, :batch])
        viewmats_sha256 = array_sha256(trajectory.viewmats)
        intrinsics_sha256 = array_sha256(trajectory.intrinsics)
        canonical_viewmats_sha256 = array_sha256(canonical_viewmats)
        canonical_intrinsics_sha256 = array_sha256(canonical_intrinsics)
        if not np.array_equal(trajectory.viewmats, canonical_viewmats) or viewmats_sha256 != canonical_viewmats_sha256:
            raise ValueError(f"B{batch} viewmats are not an exact B{canonical_batch} prefix.")
        if (
            not np.array_equal(trajectory.intrinsics, canonical_intrinsics)
            or intrinsics_sha256 != canonical_intrinsics_sha256
        ):
            raise ValueError(f"B{batch} intrinsics are not an exact B{canonical_batch} prefix.")
        if not np.array_equal(
            trajectory.expanded_scene_ids(),
            canonical.expanded_scene_ids()[:, :batch],
        ):
            raise ValueError(f"B{batch} scene IDs are not an exact B{canonical_batch} prefix.")
        contracts[str(batch)] = {
            "batch": batch,
            "trajectory_id": trajectory.trajectory_id,
            "artifacts": trajectory_artifacts(path),
            "viewmats_sha256": viewmats_sha256,
            "intrinsics_sha256": intrinsics_sha256,
            "scene_ids_sha256": array_sha256(trajectory.expanded_scene_ids()),
            "canonical_prefix_viewmats_sha256": canonical_viewmats_sha256,
            "canonical_prefix_intrinsics_sha256": canonical_intrinsics_sha256,
        }
    return {
        "schema_version": FLASHGS_TRAJECTORY_PREFIX_PROOF_SCHEMA,
        "pass": True,
        "comparison": "numpy.array_equal-and-array-sha256-after-manifest-verified-load",
        "timestep_relation": "identical-complete-timestep-axis",
        "camera_relation": "exact-contiguous-prefix-of-b1024",
        "canonical_batch": canonical_batch,
        "timesteps": expected_timesteps,
        "batches": list(batches),
        "contracts": contracts,
    }


def verify_trajectory_prefix_proof(proof: dict[str, Any]) -> dict[str, Any]:
    """Recompute a recorded trajectory-prefix proof from its bound files."""

    if proof.get("schema_version") != FLASHGS_TRAJECTORY_PREFIX_PROOF_SCHEMA:
        raise ValueError("FlashGS trajectory-prefix proof schema differs.")
    batches = tuple(proof.get("batches") or ())
    contracts = proof.get("contracts") or {}
    try:
        paths = {int(batch): Path(str(contracts[str(batch)]["artifacts"]["json"]["path"])) for batch in batches}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("FlashGS trajectory-prefix proof is malformed.") from error
    recomputed = prove_trajectory_prefixes(
        paths,
        expected_batches=batches,
        expected_timesteps=int(proof.get("timesteps", 0)),
    )
    if recomputed != proof:
        raise ValueError("FlashGS trajectory-prefix proof or bound artifacts changed.")
    return recomputed


def audit_flashgs_demand_counter_source(
    project_root: str | Path,
) -> dict[str, Any]:
    """Audit that generated-demand is counted before capacity-bounded writes."""

    root = Path(project_root).resolve()
    source_path = root / "src/isaacsim_gaussian_renderer/native/flashgs/preprocess.cu"
    source = source_path.read_text(encoding="utf-8")
    signature = "__device__ __forceinline__ int64_t reserve_intersections("
    start = source.find(signature)
    if start < 0:
        raise ValueError("FlashGS reserve_intersections source function is missing.")
    brace = source.find("{", start)
    if brace < 0:
        raise ValueError("FlashGS reserve_intersections source body is missing.")
    depth = 0
    end = None
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError("FlashGS reserve_intersections source body is unbalanced.")
    body = source[start:end]
    generated_atomic = "reinterpret_cast<unsigned long long*>(counters + 1)"
    demand_end = "const unsigned long long end = offset + static_cast<unsigned long long>(count);"
    overflow_guard = "if (end > static_cast<unsigned long long>(capacity))"
    overflow_atomic = "reinterpret_cast<unsigned long long*>(counters + 2)"
    checks = {
        "generated_counter_atomic_precedes_capacity_test": (
            body.find(generated_atomic) >= 0
            and body.find(generated_atomic) < body.find(demand_end) < body.find(overflow_guard)
        ),
        "overflow_counter_records_unstored_suffix": (
            body.find(overflow_atomic) > body.find(overflow_guard) and "end - first_dropped" in body
        ),
        "reservation_returns_unbounded_demand_offset": ("return static_cast<int64_t>(offset);" in body),
        "single_tile_write_is_capacity_bounded": (source.count("if (offset < intersection_capacity)") == 1),
        "multi_tile_write_is_capacity_bounded": (
            source.count("if (valid && offset + local_offset < intersection_capacity)") == 1
        ),
        "visible_flag_is_set_independently_of_bounded_writes": (source.count("gaussian_emitted = true;") == 2),
        "visible_counter_uses_completed_emission_flag": (
            source.count("if (gaussian_emitted && gaussian < count)") == 1
            and source.count("reinterpret_cast<unsigned long long*>(counters),") == 1
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"FlashGS demand-counter source audit failed: {checks}.")
    return {
        "schema_version": FLASHGS_COUNTER_SOURCE_AUDIT_SCHEMA,
        "pass": True,
        "generated_intersections_semantics": ("exact-reservation-demand-counted-before-capacity-bounded-writes"),
        "visible_gaussians_semantics": (
            "exact-count-with-at-least-one-generated-intersection-independent-of-bounded-storage"
        ),
        "overflow_semantics": "exact-demand-minus-stored-suffix",
        "survey_output_validity": "invalid-by-protocol-regardless-of-overflow-count",
        "source": artifact_record(source_path),
        "reserve_intersections_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "checks": checks,
    }


def verify_flashgs_demand_counter_source_audit(
    audit: dict[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Fail closed if a recorded source audit no longer matches the checkout."""

    current = audit_flashgs_demand_counter_source(project_root)
    recorded_source = audit.get("source")
    if not same_artifact(recorded_source, current["source"]):
        raise ValueError("FlashGS demand-counter source artifact changed.")
    comparable = {**audit, "source": current["source"]}
    if comparable != current:
        raise ValueError("FlashGS demand-counter source audit differs.")
    return current


def derive_flashgs_prefix_capacities(
    per_camera_visible: Sequence[int],
    per_camera_generated: Sequence[int],
    *,
    batches: Sequence[int] = PRIMARY_BATCHES,
    headroom: float = 1.05,
) -> dict[str, Any]:
    """Derive one reused-workspace capacity for each exact camera prefix."""

    if headroom != 1.05:
        raise ValueError("The matched FlashGS demand survey requires 1.05 headroom.")
    ordered_batches = tuple(int(value) for value in batches)
    if not ordered_batches or tuple(sorted(set(ordered_batches))) != ordered_batches:
        raise ValueError("batches must be strictly increasing and unique.")
    visible = [int(value) for value in per_camera_visible]
    generated = [int(value) for value in per_camera_generated]
    if len(visible) != len(generated) or len(generated) < ordered_batches[-1]:
        raise ValueError("Per-camera demand arrays do not cover the requested prefixes.")
    if any(value < 0 for value in (*visible, *generated)):
        raise ValueError("Per-camera demands must be non-negative.")
    derived: dict[str, Any] = {}
    for batch in ordered_batches:
        visible_prefix = visible[:batch]
        generated_prefix = generated[:batch]
        maximum_generated = max(generated_prefix)
        installed = max(1, math.ceil(maximum_generated * headroom))
        if installed > 2_147_483_647:
            raise ValueError("Derived FlashGS intersection capacity exceeds INT32_MAX.")
        derived[str(batch)] = {
            "batch": batch,
            "prefix_camera_range": [0, batch],
            "max_visible_gaussians_per_camera": max(visible_prefix),
            "max_generated_intersections_per_camera": maximum_generated,
            "visible_argmax_camera": visible_prefix.index(max(visible_prefix)),
            "generated_argmax_camera": generated_prefix.index(maximum_generated),
            "headroom": headroom,
            "installed_intersections_per_camera": installed,
            "installed_capacity_is_batch_specific": True,
        }
    return derived


def _safe_source_path(project_root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"Unsafe source-manifest path: {relative!r}.")
    candidate = (project_root / relative).resolve()
    if not candidate.is_relative_to(project_root):
        raise ValueError(f"Source-manifest path escapes project root: {relative!r}.")
    return candidate


def _verify_source_files(
    project_root: Path,
    records: dict[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(records, dict):
        raise ValueError(f"Source manifest lacks {label} file records.")
    for relative, expected in records.items():
        if not isinstance(relative, str) or not isinstance(expected, dict):
            raise ValueError(f"Malformed {label} source record.")
        path = _safe_source_path(project_root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Manifested source file is missing: {path}.")
        actual_bytes = path.stat().st_size
        actual_sha256 = file_sha256(path)
        if expected.get("bytes") != actual_bytes or expected.get("sha256") != actual_sha256:
            raise ValueError(f"Manifested source file changed: {relative} ({actual_bytes} bytes, {actual_sha256}).")


def load_verified_source_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Load a source manifest and verify this checkout byte-for-byte."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError(f"Unexpected source manifest schema: {payload.get('schema_version')!r}.")
    root = Path(project_root).resolve()
    tracked = payload.get("tracked_source_files")
    untracked = payload.get("relevant_untracked_source_files")
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    current_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        text=True,
    ).strip()
    current_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    current_tracked = {
        item
        for item in subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
        if item and (root / item).is_file()
    }
    current_untracked = {
        item
        for item in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        .decode("utf-8")
        .split("\0")
        if item and (root / item).is_file()
    }
    if current_head != payload.get("head"):
        raise ValueError("Current Git HEAD differs from the source manifest.")
    if current_status.splitlines() != payload.get("status_short"):
        raise ValueError("Current Git status differs from the source manifest.")
    if payload.get("dirty") is not bool(current_status):
        raise ValueError("Source-manifest dirty flag is inconsistent.")
    if payload.get("diff_sha256") != hashlib.sha256(current_diff).hexdigest():
        raise ValueError("Source-manifest diff fingerprint is inconsistent.")
    if current_tracked != set(tracked or {}):
        raise ValueError("Current tracked file set differs from the source manifest.")
    if current_untracked != set(untracked or {}):
        raise ValueError("Current untracked file set differs from the source manifest.")
    _verify_source_files(root, tracked, label="tracked")
    _verify_source_files(root, untracked, label="untracked")
    calculated_tree_sha256 = sha256_json({"tracked": tracked, "untracked": untracked})
    if payload.get("source_tree_sha256") != calculated_tree_sha256:
        raise ValueError("Source-manifest tree fingerprint is inconsistent.")
    head = payload.get("head")
    diff_sha256 = payload.get("diff_sha256")
    if not isinstance(head, str) or not head:
        raise ValueError("Source manifest has no Git HEAD.")
    if not isinstance(diff_sha256, str) or len(diff_sha256) != 64:
        raise ValueError("Source manifest has no valid diff fingerprint.")
    return {
        "manifest": artifact_record(manifest_path),
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "source_tree_sha256": calculated_tree_sha256,
        "head": head,
        "branch": payload.get("branch"),
        "dirty": payload.get("dirty"),
        "diff_sha256": diff_sha256,
    }


def source_identity(provenance: dict[str, Any]) -> dict[str, Any]:
    """Return the relocation-independent source fields used for pairing."""

    manifest = provenance.get("manifest") or {}
    return {
        "manifest_sha256": manifest.get("sha256"),
        "source_tree_sha256": provenance.get("source_tree_sha256"),
        "head": provenance.get("head"),
        "dirty": provenance.get("dirty"),
        "diff_sha256": provenance.get("diff_sha256"),
    }


def same_artifact(
    expected: dict[str, Any] | None,
    actual: dict[str, Any] | None,
) -> bool:
    """Compare artifact content while allowing a bundle to be relocated."""

    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    return expected.get("bytes") == actual.get("bytes") and expected.get("sha256") == actual.get("sha256")


def matched_projection_rule_failures(
    contract: dict[str, Any] | None,
) -> list[str]:
    """Return exact matched-projection fields missing from an equation contract."""

    if not isinstance(contract, dict):
        return ["projection equation contract is unavailable"]
    return [
        f"{field}={contract.get(field)!r}, expected {expected!r}"
        for field, expected in MATCHED_PROJECTION_RULES.items()
        if contract.get(field) != expected
    ]


def _verified_recorded_artifact(
    record: dict[str, Any] | None,
    *,
    label: str,
    bundle_root: str | Path | None = None,
) -> dict[str, Any]:
    return resolve_artifact_record(
        record or {},
        bundle_root=bundle_root,
        label=label,
    )


def verify_node_occupancy_evidence(
    record: dict[str, Any] | None,
    *,
    expected_gpu_uuid: str,
    bundle_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify immutable, single-executor occupancy evidence for one GPU job."""

    actual = _verified_recorded_artifact(
        record,
        label="Node occupancy evidence",
        bundle_root=bundle_root,
    )
    payload = json.loads(Path(str(actual["path"])).read_text(encoding="utf-8"))
    executor_control = payload.get("executor_control") or {}
    cooperative_lock = executor_control.get("cooperative_node_wide_lock") or {}
    sampled = payload.get("sampled_compute_process_telemetry") or {}
    if (
        payload.get("schema_version") != NODE_OCCUPANCY_SCHEMA
        or payload.get("expected_gpu_uuid") != expected_gpu_uuid
        or executor_control.get("scope") != "all-visible-gpus"
        or cooperative_lock.get("schema_version") != "vgr-cooperative-gpu-lock-v1"
        or cooperative_lock.get("lock_observed_held") is not True
        or cooperative_lock.get("pass") is not True
        or sampled.get("schema_version") != "flashgs-matched-compute-process-sampling-v1"
        or sampled.get("coverage") != "periodic-samples-not-continuous-observation"
        or sampled.get("sample_count", 0) < 2
        or sampled.get("pass") is not True
        or payload.get("pass") is not True
    ):
        raise ValueError("Node occupancy evidence did not pass its contract.")
    return payload


def load_verified_gsplat_build_attestation(
    path: str | Path,
    *,
    gsplat_source: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the exact pinned source inputs and JIT binary used by gsplat."""

    attestation_path = Path(path).resolve()
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != GSPLAT_BUILD_ATTESTATION_SCHEMA:
        raise ValueError(f"Unexpected gsplat build-attestation schema: {payload.get('schema_version')!r}.")
    if payload.get("gsplat_commit") != PINNED_GSPLAT_COMMIT:
        raise ValueError("gsplat build attestation uses a different commit.")

    recorded_root = Path(str(payload.get("gsplat_source_root", ""))).resolve()
    if gsplat_source is not None and recorded_root != Path(gsplat_source).resolve():
        raise ValueError("gsplat build attestation source root differs.")
    if not recorded_root.is_dir():
        raise FileNotFoundError(f"Attested gsplat source root is missing: {recorded_root}.")
    current_commit = subprocess.check_output(
        ["git", "-C", str(recorded_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if current_commit != PINNED_GSPLAT_COMMIT:
        raise ValueError("Attested gsplat source checkout changed commits.")

    source_inputs = payload.get("source_inputs")
    if not isinstance(source_inputs, dict) or not source_inputs:
        raise ValueError("gsplat build attestation has no source-input records.")
    normalized_inputs: dict[str, dict[str, Any]] = {}
    for relative, record in source_inputs.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise ValueError("Malformed gsplat source-input record.")
        source_path = _safe_source_path(recorded_root, relative)
        recorded_path = Path(str(record.get("path", ""))).resolve()
        if source_path != recorded_path:
            raise ValueError(f"gsplat source-input path differs for {relative}.")
        actual = _verified_recorded_artifact(
            record,
            label=f"gsplat source input {relative}",
        )
        normalized_inputs[relative] = {
            "bytes": actual["bytes"],
            "sha256": actual["sha256"],
        }
    if payload.get("source_input_tree_sha256") != sha256_json(normalized_inputs):
        raise ValueError("gsplat source-input tree fingerprint differs.")

    build_directory = Path(str(payload.get("build_directory", ""))).resolve()
    for name in ("build_ninja", "build_parameters", "native_extension"):
        actual = _verified_recorded_artifact(
            payload.get(name),
            label=f"gsplat attested {name}",
        )
        if not Path(actual["path"]).resolve().is_relative_to(build_directory):
            raise ValueError(f"gsplat attested {name} is outside the JIT build directory.")

    support = payload.get("support_contract") or {}
    if support.get("macro") != "GAUSSIAN_EXTEND" or support.get("value") != MATCHED_GAUSSIAN_SUPPORT_SIGMA:
        raise ValueError("gsplat attested support contract differs.")
    _verified_recorded_artifact(
        support.get("header_artifact"),
        label="gsplat attested support header",
    )
    if payload.get("projection_contract") != MATCHED_PROJECTION_RULES:
        raise ValueError("gsplat attested projection rules differ.")

    probe = payload.get("behavior_probe") or {}
    if (
        probe.get("pass") is not True
        or probe.get("support_sigma") != MATCHED_GAUSSIAN_SUPPORT_SIGMA
        or probe.get("expected_radii") != GSPLAT_SUPPORT_PROBE_EXPECTED_RADII
        or probe.get("observed_radii") != GSPLAT_SUPPORT_PROBE_EXPECTED_RADII
        or probe.get("boundary_expected_radii") != GSPLAT_BOUNDARY_PROBE_EXPECTED_RADII
        or probe.get("boundary_observed_radii") != GSPLAT_BOUNDARY_PROBE_EXPECTED_RADII
    ):
        raise ValueError("gsplat native support/boundary behavior probe differs.")
    return payload


def verify_gsplat_oracle_support_evidence(
    oracle: dict[str, Any],
    *,
    gsplat_source: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless an oracle is tied to the verified gsplat JIT build."""

    if oracle.get("pass") is not True:
        raise ValueError("Oracle manifest pass is not true.")
    if oracle.get("gsplat_commit") != PINNED_GSPLAT_COMMIT:
        raise ValueError("Oracle gsplat commit differs.")
    patch = oracle.get("gsplat_compatibility_patch") or {}
    if (
        patch.get("sha256") != PINNED_GSPLAT_PATCH_SHA256
        or patch.get("patched_utils_sha256") != PINNED_GSPLAT_PATCHED_UTILS_SHA256
    ):
        raise ValueError("Oracle gsplat compatibility patch differs.")
    if oracle.get("gaussian_support_sigma") != MATCHED_GAUSSIAN_SUPPORT_SIGMA:
        raise ValueError("Oracle Gaussian support cutoff differs.")
    support = oracle.get("gsplat_support_contract") or {}
    if support.get("macro") != "GAUSSIAN_EXTEND" or support.get("value") != MATCHED_GAUSSIAN_SUPPORT_SIGMA:
        raise ValueError("Oracle gsplat support contract differs.")
    _verified_recorded_artifact(
        support.get("header_artifact"),
        label="Oracle gsplat support header",
    )
    if oracle.get("projection_contract") != MATCHED_PROJECTION_RULES:
        raise ValueError("Oracle projection contract differs.")

    attestation_record = oracle.get("gsplat_build_attestation")
    attestation_artifact = _verified_recorded_artifact(
        attestation_record,
        label="Oracle gsplat build attestation",
    )
    attestation = load_verified_gsplat_build_attestation(
        attestation_artifact["path"],
        gsplat_source=gsplat_source,
    )
    if not same_artifact(
        oracle.get("gsplat_native_extension"),
        attestation.get("native_extension"),
    ):
        raise ValueError("Oracle native extension differs from its build attestation.")
    _verified_recorded_artifact(
        oracle.get("gsplat_python_artifact"),
        label="Oracle gsplat Python package",
    )
    _verified_recorded_artifact(
        oracle.get("gsplat_native_extension"),
        label="Oracle gsplat native extension",
    )
    if oracle.get("gsplat_behavior_probe") != attestation.get("behavior_probe"):
        raise ValueError("Oracle behavior probe differs from its build attestation.")
    verify_node_occupancy_evidence(
        oracle.get("node_occupancy"),
        expected_gpu_uuid=str(oracle.get("gpu_uuid", "")),
    )
    return attestation
