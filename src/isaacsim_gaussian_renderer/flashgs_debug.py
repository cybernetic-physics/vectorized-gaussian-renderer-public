"""Debug-only tracing for exact IDs in the FlashGS-derived matched port."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from .evaluation.matched_artifacts import (
    artifact_record,
    same_artifact,
    source_identity,
)
from .flashgs_debug_native_loader import load_flashgs_debug_extension
from .flashgs_baseline_contract import (
    FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
    require_flashgs_short_tail_repair,
    require_flashgs_matched_port_classification,
)
from .flashgs_native_loader import (
    FLASHGS_CUDA_FLAGS,
    FLASHGS_CXX_FLAGS,
    FLASHGS_UPSTREAM_COMMIT,
)


FLASHGS_PIPELINE_DEBUG_SCHEMA = "flashgs-pipeline-debug-v4"
FLASHGS_B64_KNOWN_FAILURE_CASES_SCHEMA = "flashgs-b64-known-failure-cases-v1"
DISCOVERY_MODES = (
    "all-contributors",
    "old-predicate-false-negatives",
)
MAX_TRACED_CANDIDATES = 64
CAUSE_STAGES = frozenset(
    {
        "projected-tile-bounds",
        "old-tile-intersection-predicate",
        "tile-enumeration",
        "radix-sort",
        "tile-range-construction",
        "optimized-compositor-feature-load",
        "compositor",
    }
)
ATTESTED_ADAPTER_PATHS = frozenset(
    {
        "src/isaacsim_gaussian_renderer/native/flashgs/adapter.cpp",
        "src/isaacsim_gaussian_renderer/native/flashgs/ops.h",
        "src/isaacsim_gaussian_renderer/native/flashgs/preprocess.cu",
        "src/isaacsim_gaussian_renderer/native/flashgs/sort.cu",
        "src/isaacsim_gaussian_renderer/native/flashgs/render.cu",
    }
)


def load_verified_b64_known_failure_case(
    path: str | Path,
    *,
    trajectory_id: str,
    scene_sha256: str,
    step: int,
    camera_index: int,
    pixel_x: int,
    pixel_y: int,
    width: int,
    height: int,
    expected_semantic_id: int,
) -> dict[str, Any]:
    """Bind one diagnostic invocation to the five historical B64 failures."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != FLASHGS_B64_KNOWN_FAILURE_CASES_SCHEMA
        or payload.get("diagnostic_only") is not True
        or payload.get("batch") != 64
        or payload.get("step") != 107
        or payload.get("width") != 128
        or payload.get("height") != 128
        or payload.get("total_mismatched_pixels_across_regions") != 92
    ):
        raise ValueError("B64 known-failure manifest contract differs.")
    if (
        payload.get("trajectory_id") != trajectory_id
        or payload.get("scene_sha256") != scene_sha256
        or payload.get("step") != step
        or payload.get("width") != width
        or payload.get("height") != height
    ):
        raise ValueError("Invocation differs from the known B64 failure contract.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError("B64 known-failure manifest must contain five cases.")
    case_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("B64 known-failure case is malformed.")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("B64 known-failure case IDs must be unique.")
        case_ids.add(case_id)
        case_pixel = case.get("pixel_xy")
        case_tile = case.get("tile_xy")
        if (
            not isinstance(case_pixel, list)
            or len(case_pixel) != 2
            or not isinstance(case_tile, list)
            or len(case_tile) != 2
            or case_tile != [int(case_pixel[0]) // 16, int(case_pixel[1]) // 16]
        ):
            raise ValueError(f"B64 known-failure geometry is malformed: {case_id}.")
        if case.get("camera_index") == camera_index and case_pixel == [pixel_x, pixel_y]:
            selected.append(case)
    if len(selected) != 1:
        raise ValueError("Camera/pixel is not exactly one case in the B64 failure manifest.")
    selected_case = selected[0]
    oracle_semantic_id = selected_case.get("oracle_semantic_id")
    if expected_semantic_id not in (-1, oracle_semantic_id):
        raise ValueError("Contributor semantic filter differs from the historical oracle label.")
    return {
        "manifest": artifact_record(manifest_path),
        "schema_version": FLASHGS_B64_KNOWN_FAILURE_CASES_SCHEMA,
        "trajectory_id": trajectory_id,
        "scene_sha256": scene_sha256,
        "case": selected_case,
        "historical_artifacts": payload.get("historical_artifacts"),
        "selection": payload.get("selection"),
    }


FLOAT_FIELDS = (
    "opacity_raw",
    "camera_x",
    "camera_y",
    "camera_z",
    "point_x",
    "point_y",
    "covariance_00",
    "covariance_01",
    "covariance_11",
    "determinant",
    "opacity_extend",
    "extend",
    "half_support_squared",
    "conic_x",
    "conic_y",
    "conic_z",
    "tile_pixel_min_x",
    "tile_pixel_min_y",
    "tile_pixel_max_x",
    "tile_pixel_max_y",
    "old_first_dx",
    "old_first_a",
    "old_first_b",
    "old_first_c",
    "old_first_delta",
    "old_first_t1",
    "old_first_t2",
    "old_second_dy",
    "old_second_a",
    "old_second_b",
    "old_second_c",
    "old_second_delta",
    "old_second_t1",
    "old_second_t2",
    "tile_center_x",
    "tile_center_y",
    "tile_center_q",
    "corner_00_q",
    "corner_10_q",
    "corner_01_q",
    "corner_11_q",
    "target_power_reprojected",
    "target_alpha_reprojected",
    "workspace_point_x",
    "workspace_point_y",
    "workspace_conic_x",
    "workspace_conic_y",
    "workspace_conic_z",
    "workspace_opacity",
    "compositor_power",
    "compositor_alpha",
    "compositor_pre_transmittance",
    "compositor_next_transmittance",
    "compositor_weight",
    "compositor_loaded_point_x",
    "compositor_loaded_point_y",
    "compositor_loaded_conic_x",
    "compositor_loaded_conic_y",
    "compositor_loaded_conic_z",
    "compositor_loaded_opacity",
)

INT_FIELDS = (
    "gaussian_id",
    "gaussian_id_valid",
    "projection_rejection",
    "radius_x",
    "radius_y",
    "rect_min_x",
    "rect_min_y",
    "rect_max_x",
    "rect_max_y",
    "target_tile_x",
    "target_tile_y",
    "target_tile_id",
    "target_tile_in_rect",
    "old_contains_center",
    "old_first_segment_evaluated",
    "old_first_segment_result",
    "old_second_segment_evaluated",
    "old_second_segment_result",
    "old_predicate_accept",
    "tile_center_inside_ellipse",
    "all_tile_corners_inside_ellipse",
    "target_pixel_branch",
    "unsorted_any_count",
    "unsorted_target_count",
    "sorted_any_count",
    "sorted_target_count",
    "range_start",
    "range_end",
    "candidate_in_range_count",
    "sorted_position",
    "compositor_seen",
    "compositor_branch",
    "compositor_feature_gaussian_id",
    "compositor_feature_matches_sorted_id",
    "compositor_initial_pair_slot",
    "compositor_offset_assignment_guard",
    "compositor_load_enable_guard",
    "compositor_zero_offset_fallback",
    "compositor_feature_load_offset",
)

PROJECTION_REJECTIONS = {
    0: "accepted",
    1: "invalid-gaussian-id",
    2: "opacity-threshold",
    3: "near-far-depth",
    4: "non-positive-covariance-determinant",
    5: "zero-projected-radius",
    6: "screen-bounds",
    7: "empty-tile-rectangle",
}

PIXEL_BRANCHES = {
    0: "not-evaluated",
    1: "positive-power",
    2: "power-below-minus-20",
    3: "alpha-below-threshold",
    4: "individual-contributor",
}

COMPOSITOR_BRANCHES = {
    0: "absent",
    1: "positive-power",
    2: "power-below-minus-20",
    3: "alpha-below-threshold",
    4: "transmittance-terminated-before-candidate",
    5: "candidate-triggers-transmittance-cutoff-without-contribution",
    6: "contributed",
}


def load_verified_flashgs_adapter_attestation(
    path: str | Path,
    *,
    source_provenance: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Bind a passing adapter attestation to this exact clean source tree."""

    attestation_path = Path(path).resolve()
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    if source_provenance.get("dirty") is not False:
        raise ValueError("FlashGS diagnosis requires clean source provenance.")
    if (
        payload.get("schema_version") != FLASHGS_ADAPTER_ATTESTATION_SCHEMA
        or payload.get("pass") is not True
        or payload.get("upstream_commit") != FLASHGS_UPSTREAM_COMMIT
        or payload.get("upstream_clean") is not True
    ):
        raise ValueError("FlashGS adapter attestation did not pass its contract.")
    require_flashgs_matched_port_classification(payload.get("baseline_classification"))
    if source_identity(payload.get("source_provenance") or {}) != (source_identity(source_provenance)):
        raise ValueError("FlashGS adapter attestation is not bound to the source manifest.")
    if payload.get("build_flags") != {
        "cxx": list(FLASHGS_CXX_FLAGS),
        "cuda": list(FLASHGS_CUDA_FLAGS),
    }:
        raise ValueError("FlashGS adapter attestation build flags differ.")
    diff_record = payload.get("adapter_diff")
    diff_path = Path(str((diff_record or {}).get("path", "")))
    if not diff_path.is_file() or not same_artifact(diff_record, artifact_record(diff_path)):
        raise ValueError("FlashGS adapter attestation diff artifact differs.")

    root = Path(project_root).resolve()
    source_pairs = payload.get("source_pairs") or []
    if len(source_pairs) != len(ATTESTED_ADAPTER_PATHS):
        raise ValueError("FlashGS adapter source-pair count differs.")
    observed_paths: set[str] = set()
    for pair in source_pairs:
        if not isinstance(pair, dict):
            raise ValueError("FlashGS adapter source-pair record is malformed.")
        relative = str(pair.get("adapter_path", ""))
        candidate = (root / relative).resolve()
        if (
            not relative
            or Path(relative).is_absolute()
            or not candidate.is_relative_to(root)
            or not candidate.is_file()
        ):
            raise ValueError(f"Unsafe or missing adapter path {relative!r}.")
        actual = artifact_record(candidate)
        if pair.get("adapter_bytes") != actual["bytes"] or pair.get("adapter_sha256") != actual["sha256"]:
            raise ValueError(f"Attested adapter source changed: {relative}.")
        observed_paths.add(relative)
    if observed_paths != ATTESTED_ADAPTER_PATHS:
        raise ValueError("FlashGS adapter attestation source inventory differs.")
    render_path = root / "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
    repair_checks = require_flashgs_short_tail_repair(render_path.read_text(encoding="utf-8"))
    repair_audit = payload.get("correctness_repair_audit") or {}
    if (
        repair_audit.get("pass") is not True
        or repair_audit.get("checks") != repair_checks
        or not same_artifact(repair_audit.get("render_source"), artifact_record(render_path))
    ):
        raise ValueError("FlashGS adapter repair audit differs.")
    return {
        "attestation": artifact_record(attestation_path),
        "adapter_diff": artifact_record(diff_path),
        "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
        "baseline_classification": payload["baseline_classification"],
        "source_identity": source_identity(source_provenance),
        "adapter_paths": sorted(observed_paths),
        "correctness_repair_audit": repair_audit,
    }


def classify_candidate_trace(int_fields: dict[str, int]) -> str:
    """Name the first pipeline stage that lost an expected contributor."""

    rejection = int_fields["projection_rejection"]
    if rejection:
        return "projection"
    if int_fields["target_pixel_branch"] != 4:
        return "not-an-individual-target-pixel-contributor"
    if not int_fields["target_tile_in_rect"]:
        return "projected-tile-bounds"
    if not int_fields["unsorted_target_count"]:
        if not int_fields["old_predicate_accept"]:
            return "old-tile-intersection-predicate"
        return "tile-enumeration"
    if not int_fields["sorted_target_count"]:
        return "radix-sort"
    if not int_fields["candidate_in_range_count"]:
        return "tile-range-construction"
    if not int_fields["compositor_feature_matches_sorted_id"]:
        return "optimized-compositor-feature-load"
    if int_fields["compositor_branch"] != 6:
        return "compositor"
    return "survived-compositor"


def current_optimized_feature_source_ids(
    sorted_gaussian_ids: Iterable[int],
) -> list[int]:
    """Model the current adapter's cooperative compositor feature sources.

    This is a source-grounded diagnostic model of the repaired code currently
    in ``native/flashgs/render.cu``.  The initial offset-assignment guards now
    match the corresponding load-enable guards, including one- and two-entry
    ranges, so every logical sorted ID supplies its own features.
    """

    return [int(value) for value in sorted_gaussian_ids]


def _optional_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _record_from_rows(
    float_row: list[float],
    int_row: list[int],
    *,
    semantic_id: int,
) -> dict[str, Any]:
    floats = {name: _optional_float(value) for name, value in zip(FLOAT_FIELDS, float_row, strict=True)}
    ints = {name: int(value) for name, value in zip(INT_FIELDS, int_row, strict=True)}
    half_support_squared = floats["half_support_squared"]
    ellipse_limit_q = None if half_support_squared is None else 2.0 * half_support_squared
    replayed = [
        floats["point_x"],
        floats["point_y"],
        floats["conic_x"],
        floats["conic_y"],
        floats["conic_z"],
    ]
    retained = [
        floats["workspace_point_x"],
        floats["workspace_point_y"],
        floats["workspace_conic_x"],
        floats["workspace_conic_y"],
        floats["workspace_conic_z"],
    ]
    workspace_projection_available = all(value is not None for value in retained)
    replay_differences = (
        [
            abs(float(replayed_value) - float(retained_value))
            for replayed_value, retained_value in zip(replayed, retained, strict=True)
        ]
        if workspace_projection_available and all(value is not None for value in replayed)
        else None
    )
    replay_exact = all(value == 0.0 for value in replay_differences) if replay_differences is not None else None
    record = {
        "gaussian_id": ints["gaussian_id"],
        "semantic_id": int(semantic_id),
        "first_loss_stage": classify_candidate_trace(ints),
        "projection": {
            "result": PROJECTION_REJECTIONS.get(ints["projection_rejection"], "unknown"),
            "camera_xyz": [
                floats["camera_x"],
                floats["camera_y"],
                floats["camera_z"],
            ],
            "point_xy": [floats["point_x"], floats["point_y"]],
            "covariance_2d": [
                floats["covariance_00"],
                floats["covariance_01"],
                floats["covariance_11"],
            ],
            "determinant": floats["determinant"],
            "radii": [ints["radius_x"], ints["radius_y"]],
            "tile_rect_half_open": [
                ints["rect_min_x"],
                ints["rect_min_y"],
                ints["rect_max_x"],
                ints["rect_max_y"],
            ],
            "retained_workspace_crosscheck": {
                "available": workspace_projection_available,
                "bit_exact_values": replay_exact,
                "absolute_differences_point_xy_conic": replay_differences,
            },
        },
        "old_tile_predicate": {
            "tile_xy": [ints["target_tile_x"], ints["target_tile_y"]],
            "tile_id": ints["target_tile_id"],
            "tile_in_projected_rect": bool(ints["target_tile_in_rect"]),
            "pixel_center_bounds": [
                floats["tile_pixel_min_x"],
                floats["tile_pixel_min_y"],
                floats["tile_pixel_max_x"],
                floats["tile_pixel_max_y"],
            ],
            "contains_ellipse_center": bool(ints["old_contains_center"]),
            "first_segment": {
                "evaluated": bool(ints["old_first_segment_evaluated"]),
                "result": bool(ints["old_first_segment_result"]),
                "dx": floats["old_first_dx"],
                "a": floats["old_first_a"],
                "b": floats["old_first_b"],
                "c": floats["old_first_c"],
                "delta": floats["old_first_delta"],
                "t1": floats["old_first_t1"],
                "t2": floats["old_first_t2"],
            },
            "second_segment": {
                "evaluated": bool(ints["old_second_segment_evaluated"]),
                "result": bool(ints["old_second_segment_result"]),
                "dy": floats["old_second_dy"],
                "a": floats["old_second_a"],
                "b": floats["old_second_b"],
                "c": floats["old_second_c"],
                "delta": floats["old_second_delta"],
                "t1": floats["old_second_t1"],
                "t2": floats["old_second_t2"],
            },
            "accepted": bool(ints["old_predicate_accept"]),
            "ellipse_limit_q": ellipse_limit_q,
            "tile_center_xy": [
                floats["tile_center_x"],
                floats["tile_center_y"],
            ],
            "tile_center_q": floats["tile_center_q"],
            "tile_center_inside_ellipse": bool(ints["tile_center_inside_ellipse"]),
            "corner_q": [
                floats["corner_00_q"],
                floats["corner_10_q"],
                floats["corner_01_q"],
                floats["corner_11_q"],
            ],
            "all_tile_corners_inside_ellipse": bool(ints["all_tile_corners_inside_ellipse"]),
        },
        "target_pixel": {
            "branch": PIXEL_BRANCHES.get(ints["target_pixel_branch"], "unknown"),
            "reprojected_power": floats["target_power_reprojected"],
            "reprojected_alpha": floats["target_alpha_reprojected"],
        },
        "enumeration_sort_range": {
            key: ints[key]
            for key in (
                "unsorted_any_count",
                "unsorted_target_count",
                "sorted_any_count",
                "sorted_target_count",
                "range_start",
                "range_end",
                "candidate_in_range_count",
                "sorted_position",
            )
        }
        | {
            "range_length": max(0, ints["range_end"] - ints["range_start"]),
        },
        "compositor": {
            "feature_load_evidence": {
                "kind": "source-grounded-debug-model",
                "production_lane_registers_observed": False,
                "bound_source": "native/flashgs/render.cu",
            },
            "seen": bool(ints["compositor_seen"]),
            "logical_sorted_gaussian_id": ints["gaussian_id"],
            "feature_gaussian_id": ints["compositor_feature_gaussian_id"],
            "feature_load_offset": ints["compositor_feature_load_offset"],
            "feature_source_matches_sorted_gaussian": bool(ints["compositor_feature_matches_sorted_id"]),
            "logical_feature_workspace": {
                "point_xy": [
                    floats["workspace_point_x"],
                    floats["workspace_point_y"],
                ],
                "conic_opacity": [
                    floats["workspace_conic_x"],
                    floats["workspace_conic_y"],
                    floats["workspace_conic_z"],
                    floats["workspace_opacity"],
                ],
            },
            "loaded_feature_workspace": {
                "point_xy": [
                    floats["compositor_loaded_point_x"],
                    floats["compositor_loaded_point_y"],
                ],
                "conic_opacity": [
                    floats["compositor_loaded_conic_x"],
                    floats["compositor_loaded_conic_y"],
                    floats["compositor_loaded_conic_z"],
                    floats["compositor_loaded_opacity"],
                ],
            },
            "initial_load": {
                "pair_slot": (
                    ints["compositor_initial_pair_slot"] if ints["compositor_initial_pair_slot"] >= 0 else None
                ),
                "offset_initial_value": 0,
                "offset_assignment_guard": (
                    bool(ints["compositor_offset_assignment_guard"])
                    if ints["compositor_offset_assignment_guard"] >= 0
                    else None
                ),
                "load_enable_guard": (
                    bool(ints["compositor_load_enable_guard"]) if ints["compositor_load_enable_guard"] >= 0 else None
                ),
                "zero_offset_fallback": bool(ints["compositor_zero_offset_fallback"]),
                "slot_0_offset_condition": "point_id + 0 < range.y",
                "slot_0_load_condition": "point_id + 0 < range.y",
                "slot_1_offset_condition": "point_id + 1 < range.y",
                "slot_1_load_condition": "point_id + 1 < range.y",
            },
            "branch": COMPOSITOR_BRANCHES.get(ints["compositor_branch"], "unknown"),
            "power": floats["compositor_power"],
            "alpha": floats["compositor_alpha"],
            "loaded_feature_target_power": floats["compositor_power"],
            "loaded_feature_target_alpha": floats["compositor_alpha"],
            "pre_transmittance": floats["compositor_pre_transmittance"],
            "next_transmittance": floats["compositor_next_transmittance"],
            "weight": floats["compositor_weight"],
        },
        # Retain flat, lossless field maps so a future parser can use values
        # that are not promoted into the human-oriented sections above.
        "float_fields": floats,
        "int_fields": ints,
    }
    return record


def trace_flashgs_backend(
    backend: Any,
    *,
    viewmat: torch.Tensor,
    intrinsics: torch.Tensor,
    target_pixel_x: int,
    target_pixel_y: int,
    candidate_ids: Iterable[int] = (),
    discovery_mode: str = "all-contributors",
    expected_semantic_id: int = -1,
    max_discovered_candidates: int = 32,
    native_module: ModuleType | Any | None = None,
    native_loader: Callable[[], ModuleType] = load_flashgs_debug_extension,
) -> dict[str, Any]:
    """Trace one already-rendered camera through retained FlashGS buffers.

    FlashGS reuses one workspace per camera, so callers must render exactly the
    selected camera immediately before this function. The diagnostic performs
    synchronized readback and is never valid inside a timed benchmark range.
    """

    scene = getattr(backend, "_scene", None)
    covariances = getattr(backend, "_covariances", None)
    if scene is None or covariances is None:
        raise RuntimeError("FlashGS backend must have a resident scene.")
    if not getattr(backend, "workspace", None):
        raise RuntimeError("FlashGS backend workspace is unavailable.")
    height = int(getattr(backend, "_height", 0))
    width = int(getattr(backend, "_width", 0))
    if not (0 <= target_pixel_x < width and 0 <= target_pixel_y < height):
        raise ValueError("Target pixel lies outside the FlashGS output.")
    if discovery_mode not in DISCOVERY_MODES:
        raise ValueError(f"Unknown discovery mode {discovery_mode!r}.")
    if not 0 < max_discovered_candidates <= MAX_TRACED_CANDIDATES:
        raise ValueError(f"max_discovered_candidates must be in [1, {MAX_TRACED_CANDIDATES}].")
    viewmat = viewmat.reshape(4, 4).contiguous()
    intrinsics = intrinsics.reshape(3, 3).contiguous()
    if viewmat.device != scene.means.device or intrinsics.device != scene.means.device:
        raise ValueError("Camera and scene tensors must share one CUDA device.")
    native = native_module if native_module is not None else native_loader()
    counters = backend.workspace.get("counters")
    if counters is None or counters.ndim != 2 or counters.shape[1] < 2:
        raise RuntimeError("FlashGS per-camera counters are unavailable.")
    generated_intersections = int(counters[0, 1].item())
    stored_capacity = int(backend.workspace["keys_unsorted"].numel())
    if not 0 <= generated_intersections <= stored_capacity:
        raise RuntimeError(
            "FlashGS generated-intersection count exceeds retained capacity: "
            f"{generated_intersections} > {stored_capacity}."
        )

    explicit_ids = [int(value) for value in candidate_ids]
    invalid_ids = [value for value in explicit_ids if value < 0 or value >= scene.count]
    if invalid_ids:
        raise ValueError(f"Gaussian IDs lie outside [0, {scene.count}): {invalid_ids}.")
    require_old_predicate_rejection = discovery_mode == "old-predicate-false-negatives"
    scores = native.score_projected_contributors(
        scene.means,
        covariances,
        scene.opacities,
        scene.semantic_ids,
        viewmat,
        intrinsics,
        height,
        width,
        float(backend.near_plane),
        float(backend.far_plane),
        float(backend.gaussian_support_sigma),
        float(backend.covariance_epsilon),
        target_pixel_x,
        target_pixel_y,
        expected_semantic_id,
        require_old_predicate_rejection,
    )
    if scores.dtype != torch.float32 or scores.device != scene.means.device or scores.shape != (scene.count,):
        raise RuntimeError("FlashGS contributor score tensor has an invalid contract.")
    candidate_mask = torch.isfinite(scores)
    discovery_count = int(candidate_mask.sum().item())
    selected_count = min(discovery_count, max_discovered_candidates)
    ranked_candidates: list[dict[str, int | float]] = []
    cutoff_tie_total = 0
    cutoff_tie_selected = 0
    if selected_count:
        top_scores, _top_ids = torch.topk(
            scores,
            k=selected_count,
            largest=True,
            sorted=True,
        )
        cutoff_tensor = top_scores[-1]
        strong_ids = torch.nonzero(scores > cutoff_tensor, as_tuple=False).flatten()
        remaining = selected_count - int(strong_ids.numel())
        tie_ids = torch.nonzero(scores == cutoff_tensor, as_tuple=False).flatten()[:remaining]
        selected_ids = torch.cat((strong_ids, tie_ids))
        selected_scores = scores.index_select(0, selected_ids)
        ranked_pairs = sorted(
            zip(
                selected_ids.detach().cpu().tolist(),
                selected_scores.detach().cpu().tolist(),
                strict=True,
            ),
            key=lambda item: (-float(item[1]), int(item[0])),
        )
        ranked_candidates = [
            {
                "rank": rank,
                "gaussian_id": int(gaussian_id),
                "individual_alpha": float(alpha),
            }
            for rank, (gaussian_id, alpha) in enumerate(ranked_pairs, start=1)
        ]
        cutoff = float(ranked_candidates[-1]["individual_alpha"])
        cutoff_tie_total = int((scores == cutoff).sum().item())
        cutoff_tie_selected = remaining
    discovered_ids = [int(candidate["gaussian_id"]) for candidate in ranked_candidates]
    all_ids = [
        *discovered_ids,
        *sorted(set(explicit_ids) - set(discovered_ids)),
    ]
    if len(all_ids) > MAX_TRACED_CANDIDATES:
        raise ValueError(f"At most {MAX_TRACED_CANDIDATES} total candidate IDs may be traced.")
    records: list[dict[str, Any]] = []
    if all_ids:
        candidate_tensor = torch.tensor(
            all_ids,
            device=scene.means.device,
            dtype=torch.int64,
        )
        float_trace, int_trace = native.trace_candidates(
            scene.means,
            covariances,
            scene.opacities,
            viewmat,
            intrinsics,
            backend.workspace["keys_unsorted"][:generated_intersections],
            backend.workspace["values_unsorted"][:generated_intersections],
            backend.workspace["keys_sorted"][:generated_intersections],
            backend.workspace["values_sorted"][:generated_intersections],
            backend.workspace["ranges"],
            backend.workspace["points_xy"],
            backend.workspace["conic_opacity"],
            candidate_tensor,
            height,
            width,
            float(backend.near_plane),
            float(backend.far_plane),
            float(backend.gaussian_support_sigma),
            float(backend.covariance_epsilon),
            target_pixel_x,
            target_pixel_y,
        )
        semantic_ids = scene.semantic_ids.index_select(0, candidate_tensor).detach().cpu().tolist()
        float_rows = float_trace.detach().cpu().tolist()
        int_rows = int_trace.detach().cpu().tolist()
        records = [
            _record_from_rows(
                float_row,
                int_row,
                semantic_id=int(semantic_id),
            )
            for float_row, int_row, semantic_id in zip(float_rows, int_rows, semantic_ids, strict=True)
        ]

    discovery_by_id = {int(candidate["gaussian_id"]): candidate for candidate in ranked_candidates}
    score_crosschecks: list[bool] = []
    for record in records:
        discovery = discovery_by_id.get(record["gaussian_id"])
        record["discovery"] = discovery
        if discovery is not None:
            traced_alpha = record["target_pixel"]["reprojected_alpha"]
            score_exact = traced_alpha is not None and traced_alpha == discovery["individual_alpha"]
            record["discovery_score_bit_exact"] = score_exact
            score_crosschecks.append(score_exact)
        else:
            record["discovery_score_bit_exact"] = None

    replay_checks = [
        record["projection"]["retained_workspace_crosscheck"]["bit_exact_values"]
        for record in records
        if record["projection"]["retained_workspace_crosscheck"]["available"]
    ]
    identified_causes = [
        {
            "gaussian_id": record["gaussian_id"],
            "rank": record["discovery"]["rank"],
            "individual_alpha": record["discovery"]["individual_alpha"],
            "first_loss_stage": record["first_loss_stage"],
        }
        for record in records
        if record["discovery"] is not None and record["first_loss_stage"] in CAUSE_STAGES
    ]
    cause_identified = bool(identified_causes)
    return {
        "schema_version": FLASHGS_PIPELINE_DEBUG_SCHEMA,
        "debug_only": True,
        "measured_timing_valid": False,
        "target_pixel_xy": [target_pixel_x, target_pixel_y],
        "target_tile_xy": [target_pixel_x // 16, target_pixel_y // 16],
        "expected_semantic_id_filter": expected_semantic_id,
        "candidate_discovery": {
            "mode": discovery_mode,
            "predicate": (
                "all projected target-pixel contributors"
                if discovery_mode == "all-contributors"
                else (
                    "projected target-pixel contributors whose target tile is "
                    "in the projected rectangle and whose exact old FlashGS "
                    "tile predicate rejects"
                )
            ),
            "matched": discovery_count,
            "selected": selected_count,
            "truncated": discovery_count - selected_count,
            "maximum_selected": max_discovered_candidates,
            "ranking": "descending individual pre-compositing alpha, then Gaussian ID",
            "cutoff_tie_total": cutoff_tie_total,
            "cutoff_tie_selected": cutoff_tie_selected,
            "score_buffer_bytes": scores.numel() * scores.element_size(),
            "gaussians_scored": scene.count,
            "valid_intersections_scanned_per_candidate": (generated_intersections),
            "candidates": ranked_candidates,
        },
        "explicit_gaussian_ids": sorted(set(explicit_ids)),
        "projection_replay_crosscheck": {
            "checked_records": len(replay_checks),
            "all_bit_exact": all(replay_checks) if replay_checks else None,
        },
        "discovery_score_crosscheck": {
            "checked_records": len(score_crosschecks),
            "all_bit_exact": (all(score_crosschecks) if score_crosschecks else None),
        },
        "cause_identified": cause_identified,
        "identified_causes": identified_causes,
        "records": records,
        "native_build_contract": getattr(native, "__vgr_build_contract__", None),
    }
