from __future__ import annotations

import json
import copy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (
    HEADLINE_TORCH_CUDA_ARCH_LIST,
    artifact_record,
)
from isaacsim_gaussian_renderer.evaluation.matched_semantics import (
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
)
from isaacsim_gaussian_renderer.fidelity.camera_bundle import (
    bundle_from_tensors,
    write_camera_bundle,
)
from isaacsim_gaussian_renderer.flashgs_repair import (
    EXPECTED_MISMATCH_COUNT,
    EXPECTED_SELECTED_VIEW_MISMATCH_COUNTS,
    FLASHGS_B64_DIAGNOSIS_INDEX_SCHEMA,
    FLASHGS_B64_DIAGNOSIS_LOCK_SCHEMA,
    FLASHGS_B64_PRE_FIX_COMMIT,
    FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA,
    PRE_FIX_PIPELINE_DEBUG_SCHEMA,
    PRE_FIX_RENDERER_RUN_SCHEMA,
    _select_case_culprits,
    aggregate_b64_repair_cases,
    audit_repaired_flashgs_production,
    build_b64_diagnosis_index,
    evaluate_repaired_b64_case,
    evaluate_repaired_mismatch_corpus,
    evaluate_repaired_rgb_only_corpus,
    flashgs_adapter_module_name,
    load_bound_b64_oracle_pixels,
    load_b64_diagnosis_lock,
    load_verified_b64_diagnosis_index,
    load_verified_b64_repair_raw_outputs,
)


TRAJECTORY_ID = "e" * 64
SCENE_SHA256 = "2" * 64
CAMERAS = (0, 9, 18, 27, 36, 45, 54, 63)
CASE_SPECS = (
    ("camera18-tile0-0", 18, (7, 3), 2, 6, np.float32(0.0483288169)),
    ("camera36-tile7-0", 36, (123, 15), 87, 2, np.float32(0.25153077)),
    ("camera45-tile0-1", 45, (0, 31), 1, 6, np.float32(0.047843933)),
    ("camera54-tile5-0", 54, (95, 15), 1, 7, np.float32(0.014007568)),
    ("camera63-tile6-1", 63, (111, 31), 1, 2, np.float32(0.014948845)),
)


def _mismatch_coordinates() -> dict[int, list[tuple[int, int]]]:
    coordinates = {camera: [] for camera in CAMERAS}
    coordinates[18] = [(7, 3), (8, 3)]
    coordinates[36] = [(123, 15)] + [(index % 128, 40 + index // 128) for index in range(86)]
    coordinates[45] = [(0, 31)]
    coordinates[54] = [(95, 15)]
    coordinates[63] = [(111, 31)]
    return coordinates


def _write_input_artifacts(root: Path) -> dict[str, Any]:
    coordinates = _mismatch_coordinates()
    historical_rgb = np.zeros((64, 128, 128, 3), dtype=np.float32)
    historical_alpha = np.zeros((64, 128, 128, 1), dtype=np.float32)
    historical_depth = np.full((64, 128, 128, 1), np.inf, dtype=np.float32)
    historical_semantic = np.full((64, 128, 128, 1), -1, dtype=np.int64)

    oracle_rgb = np.zeros((8, 128, 128, 3), dtype=np.float32)
    oracle_alpha = np.zeros((8, 128, 128, 1), dtype=np.float32)
    oracle_depth = np.zeros((8, 128, 128, 1), dtype=np.float32)
    oracle_semantic = np.full((8, 128, 128, 1), -1, dtype=np.int64)
    case_by_camera = {spec[1]: spec for spec in CASE_SPECS}
    for view_index, camera in enumerate(CAMERAS):
        for coordinate_index, (x, y) in enumerate(coordinates[camera]):
            spec = case_by_camera.get(camera)
            semantic_id = int(spec[4]) if spec is not None else 1
            alpha = (
                np.float32(spec[5])
                if coordinate_index == 0 and spec is not None
                else np.float32(0.02 + coordinate_index * 1.0e-5)
            )
            oracle_rgb[view_index, y, x] = [alpha, alpha / 2, alpha / 4]
            oracle_alpha[view_index, y, x, 0] = alpha
            oracle_depth[view_index, y, x, 0] = np.float32(camera + 1)
            oracle_semantic[view_index, y, x, 0] = semantic_id

    bundle = bundle_from_tensors(
        viewmats=np.repeat(np.eye(4)[None, ...], 8, axis=0),
        intrinsics=np.repeat(np.eye(3)[None, ...], 8, axis=0),
        width=128,
        height=128,
        scene_ids=np.full(8, 404, dtype=np.int64),
        scene_checksum=SCENE_SHA256,
        view_ids=[f"t107-b{camera:04d}" for camera in CAMERAS],
    )
    bundle_path = root / "oracle.camera-bundle.json"
    write_camera_bundle(bundle, bundle_path)

    historical_path = root / "flashgs-full-b64.last-output.npz"
    np.savez_compressed(
        historical_path,
        rgb=historical_rgb,
        alpha=historical_alpha,
        depth=historical_depth,
        semantic_id=historical_semantic,
    )
    oracle_path = root / "oracle.npz"
    np.savez_compressed(
        oracle_path,
        rgb=oracle_rgb,
        alpha=oracle_alpha,
        depth=oracle_depth,
        semantic=oracle_semantic,
        camera_indices=np.asarray(CAMERAS, dtype=np.int64),
        step=np.asarray(107, dtype=np.int64),
        trajectory_id=np.asarray(TRAJECTORY_ID),
        semantic_topology=np.asarray(REPRESENTATIVE_SEMANTIC_TOPOLOGY),
        camera_bundle_id=np.asarray(bundle.bundle_id),
    )
    historical_artifact = artifact_record(historical_path)
    oracle_artifact = artifact_record(oracle_path)
    run_path = root / "flashgs-full-b64.json"
    run_path.write_text(
        json.dumps(
            {
                "schema_version": PRE_FIX_RENDERER_RUN_SCHEMA,
                "renderer": "flashgs",
                "output_contract": "full",
                "camera_contract": {
                    "trajectory_id": TRAJECTORY_ID,
                    "batch": 64,
                    "width": 128,
                    "height": 128,
                },
                "equation_contract": {"semantic_topology": REPRESENTATIVE_SEMANTIC_TOPOLOGY},
                "last_output_capture": historical_artifact,
            }
        ),
        encoding="utf-8",
    )

    cases = [
        {
            "case_id": case_id,
            "camera_index": camera,
            "pixel_xy": list(pixel),
            "tile_xy": [pixel[0] // 16, pixel[1] // 16],
            "region_mismatched_pixels": region_count,
            "oracle_alpha": float(alpha),
            "oracle_semantic_id": semantic_id,
            "historical_flashgs_alpha": 0.0,
            "historical_flashgs_semantic_id": -1,
        }
        for case_id, camera, pixel, region_count, semantic_id, alpha in CASE_SPECS
    ]
    known_path = root / "known.json"
    known_path.write_text(
        json.dumps(
            {
                "schema_version": "flashgs-b64-known-failure-cases-v1",
                "diagnostic_only": True,
                "scene_sha256": SCENE_SHA256,
                "trajectory_id": TRAJECTORY_ID,
                "batch": 64,
                "step": 107,
                "width": 128,
                "height": 128,
                "total_mismatched_pixels_across_regions": 92,
                "historical_artifacts": {
                    "flashgs_full_last_output": historical_artifact,
                    "pinned_gsplat_oracle": oracle_artifact,
                },
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    known_artifact = artifact_record(known_path)
    attestation_path = root / "adapter-attestation.json"
    attestation_path.write_text("{}", encoding="utf-8")
    production_native_path = root / "isaacsim_flashgs_adapter_fixture.so"
    production_native_path.write_bytes(b"fixture-production-extension")
    debug_native_path = root / "isaacsim_flashgs_debug_fixture.so"
    debug_native_path.write_bytes(b"fixture-debug-extension")
    attestation_artifact = artifact_record(attestation_path)
    production_native_artifact = artifact_record(production_native_path)
    diagnosis_paths: list[Path] = []
    culprit_specs_by_case: dict[str, list[dict[str, int]]] = {}
    for index, case in enumerate(cases):
        gaussian_id = 1_000 + index
        alpha = float(case["oracle_alpha"])
        record = {
            "gaussian_id": gaussian_id,
            "semantic_id": case["oracle_semantic_id"],
            "first_loss_stage": "optimized-compositor-feature-load",
            "projection": {
                "retained_workspace_crosscheck": {
                    "available": True,
                    "bit_exact_values": True,
                }
            },
            "target_pixel": {
                "reprojected_power": -0.25,
                "reprojected_alpha": alpha,
            },
            "enumeration_sort_range": {
                "range_start": 0,
                "range_end": 1,
                "range_length": 1,
                "sorted_position": 0,
                "candidate_in_range_count": 1,
            },
            "compositor": {
                "feature_load_evidence": {
                    "kind": "source-grounded-debug-model",
                    "production_lane_registers_observed": False,
                },
                "seen": True,
                "feature_gaussian_id": 0,
                "feature_load_offset": 0,
                "feature_source_matches_sorted_gaussian": False,
                "logical_feature_workspace": {
                    "point_xy": [1.0 + index, 2.0],
                    "conic_opacity": [1.0, 0.0, 1.0, alpha],
                },
                "loaded_feature_workspace": {
                    "point_xy": [-1.0, -2.0],
                    "conic_opacity": [2.0, 0.0, 2.0, 0.0],
                },
                "initial_load": {
                    "pair_slot": 0,
                    "offset_assignment_guard": False,
                    "load_enable_guard": True,
                    "zero_offset_fallback": True,
                },
                "branch": "low-alpha",
            },
            "discovery": {
                "rank": 1,
                "gaussian_id": gaussian_id,
                "individual_alpha": alpha,
            },
        }
        diagnosis = {
            "schema_version": PRE_FIX_PIPELINE_DEBUG_SCHEMA,
            "debug_only": True,
            "measured_timing_valid": False,
            "tool_integrity": {
                "pass": True,
                "checks": {
                    "historical_failure_reproduced": True,
                    "zero_intersection_overflow": True,
                },
            },
            "cause_identified": True,
            "diagnosis_complete": True,
            "source_provenance": {
                "manifest": {"sha256": "a" * 64},
                "source_tree_sha256": "b" * 64,
                "head": "c" * 40,
                "dirty": False,
                "diff_sha256": "d" * 64,
            },
            "camera": {
                "trajectory_id": TRAJECTORY_ID,
                "step": 107,
                "camera_index": case["camera_index"],
                "target_pixel_xy": case["pixel_xy"],
                "historical_failure_case": {
                    "manifest": known_artifact,
                    "case": case,
                },
            },
            "rendered_target_pixel": {
                "alpha": 0.0,
                "semantic_id": -1,
            },
            "capacity": {"counters": {"intersection_overflow": False}},
            "flashgs_adapter_attestation": {"attestation": attestation_artifact},
            "production_adapter": {"native_extension": production_native_artifact},
            "trace": {
                "schema_version": PRE_FIX_PIPELINE_DEBUG_SCHEMA,
                "cause_identified": True,
                "native_build_contract": {"module_name": debug_native_path.stem},
                "records": [record],
            },
        }
        path = root / f"diagnosis-{index}.json"
        path.write_text(json.dumps(diagnosis), encoding="utf-8")
        diagnosis_paths.append(path)
        culprit_specs_by_case[case["case_id"]] = [
            {
                "gaussian_id": gaussian_id,
                "range_start": 0,
                "range_end": 1,
                "sorted_position": 0,
                "range_relative_slot": 0,
            }
        ]

    def relative_record(path: Path) -> dict[str, Any]:
        record = artifact_record(path)
        record["path"] = path.relative_to(root).as_posix()
        return record

    lock_path = root / "diagnosis-lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": FLASHGS_B64_DIAGNOSIS_LOCK_SCHEMA,
                "known_failure_manifest": relative_record(known_path),
                "pre_fix_source_identity": {
                    "manifest_sha256": "a" * 64,
                    "source_tree_sha256": "b" * 64,
                    "head": "c" * 40,
                    "dirty": False,
                    "diff_sha256": "d" * 64,
                },
                "adapter_attestation": relative_record(attestation_path),
                "production_native_extension": relative_record(production_native_path),
                "debug_native_extension": relative_record(debug_native_path),
                "cases": [
                    {
                        "case_id": case["case_id"],
                        "diagnosis": relative_record(diagnosis_path),
                        "culprits": culprit_specs_by_case[case["case_id"]],
                    }
                    for case, diagnosis_path in zip(cases, diagnosis_paths, strict=True)
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "known": known_path,
        "lock": lock_path,
        "historical_run": run_path,
        "historical": historical_path,
        "oracle": oracle_path,
        "bundle": bundle_path,
        "diagnoses": diagnosis_paths,
    }


@pytest.fixture
def indexed_evidence(tmp_path: Path) -> dict[str, Any]:
    artifact_root = tmp_path / "artifact-root"
    artifact_root.mkdir()
    inputs = _write_input_artifacts(artifact_root)
    index = build_b64_diagnosis_index(
        inputs["known"],
        inputs["diagnoses"],
        diagnosis_lock=inputs["lock"],
        artifact_root=artifact_root,
        historical_run=inputs["historical_run"],
        historical_flashgs_output=inputs["historical"],
        pinned_oracle_output=inputs["oracle"],
        camera_bundle_path=inputs["bundle"],
        semantic_topology=REPRESENTATIVE_SEMANTIC_TOPOLOGY,
        created_at="2026-07-22T00:00:00+00:00",
    )
    index_path = artifact_root / "diagnosis-index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return {**inputs, "index": index, "index_path": index_path}


def test_diagnosis_index_freezes_five_culprits_and_92_pixels(
    indexed_evidence: dict[str, Any],
) -> None:
    index = indexed_evidence["index"]

    assert index["schema_version"] == FLASHGS_B64_DIAGNOSIS_INDEX_SCHEMA
    assert index["case_count"] == 5
    assert index["culprit_count"] == 5
    assert all(len(case["culprits"]) == 1 for case in index["cases"])
    corpus = index["historical_mismatch_corpus"]
    assert corpus["mismatch_count"] == EXPECTED_MISMATCH_COUNT
    assert corpus["per_view_mismatch_counts"] == list(EXPECTED_SELECTED_VIEW_MISMATCH_COUNTS)
    assert len(corpus["pixels"]) == 92
    assert len(corpus["coordinate_sha256"]) == 64
    assert len(corpus["oracle_values_sha256"]) == 64

    verified = load_verified_b64_diagnosis_index(
        indexed_evidence["index_path"],
        known_failure_manifest=indexed_evidence["known"],
        diagnosis_lock=indexed_evidence["lock"],
        artifact_root=indexed_evidence["root"],
    )
    assert verified["payload"] == index
    oracle_pixels = load_bound_b64_oracle_pixels(
        indexed_evidence["oracle"],
        known_failure_manifest=indexed_evidence["known"],
        semantic_topology=REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    )
    assert len(oracle_pixels["pixels"]) == 5


def test_diagnosis_index_rejects_tampered_source_artifact(
    indexed_evidence: dict[str, Any],
) -> None:
    indexed_evidence["diagnoses"][0].write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash differs"):
        load_verified_b64_diagnosis_index(
            indexed_evidence["index_path"],
            known_failure_manifest=indexed_evidence["known"],
            diagnosis_lock=indexed_evidence["lock"],
            artifact_root=indexed_evidence["root"],
        )


def test_diagnosis_index_is_relocatable(indexed_evidence: dict[str, Any], tmp_path: Path) -> None:
    relocated = tmp_path / "relocated-artifact-root"
    shutil.copytree(indexed_evidence["root"], relocated)

    verified = load_verified_b64_diagnosis_index(
        relocated / indexed_evidence["index_path"].name,
        known_failure_manifest=relocated / indexed_evidence["known"].relative_to(indexed_evidence["root"]),
        diagnosis_lock=relocated / indexed_evidence["lock"].relative_to(indexed_evidence["root"]),
        artifact_root=relocated,
    )

    assert verified["payload"] == indexed_evidence["index"]


def test_diagnosis_lock_rejects_unsafe_artifact_path(indexed_evidence: dict[str, Any], tmp_path: Path) -> None:
    payload = json.loads(indexed_evidence["lock"].read_text(encoding="utf-8"))
    payload["adapter_attestation"]["path"] = "../escape.json"
    unsafe_lock = tmp_path / "unsafe-lock.json"
    unsafe_lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="safe and relative"):
        load_b64_diagnosis_lock(
            unsafe_lock,
            artifact_root=indexed_evidence["root"],
            known_failure_manifest=indexed_evidence["known"],
        )


def test_pre_fix_culprit_requires_source_grounded_model_marker(indexed_evidence: dict[str, Any]) -> None:
    diagnosis = json.loads(indexed_evidence["diagnoses"][0].read_text(encoding="utf-8"))
    case = json.loads(indexed_evidence["known"].read_text(encoding="utf-8"))["cases"][0]
    diagnosis["trace"]["records"][0]["compositor"]["feature_load_evidence"]["production_lane_registers_observed"] = True

    with pytest.raises(ValueError, match="no short-tail culprit"):
        _select_case_culprits(diagnosis, case)


def _repaired_trace(indexed_case: dict[str, Any]) -> dict[str, Any]:
    records = []
    for culprit in indexed_case["culprits"]:
        workspace = culprit["logical_feature_workspace"]
        records.append(
            {
                "gaussian_id": culprit["gaussian_id"],
                "semantic_id": culprit["semantic_id"],
                "first_loss_stage": "survived-compositor",
                "projection": {
                    "retained_workspace_crosscheck": {
                        "available": True,
                        "bit_exact_values": True,
                    }
                },
                "target_pixel": {
                    "reprojected_power": culprit["target_reprojected_power"],
                    "reprojected_alpha": culprit["target_reprojected_alpha"],
                },
                "enumeration_sort_range": {
                    "range_start": culprit["range_start"],
                    "range_end": culprit["range_end"],
                    "range_length": culprit["range_length"],
                    "sorted_position": culprit["sorted_position"],
                    "candidate_in_range_count": 1,
                },
                "compositor": {
                    "feature_load_evidence": {
                        "kind": "source-grounded-debug-model",
                        "production_lane_registers_observed": False,
                    },
                    "seen": True,
                    "feature_gaussian_id": culprit["gaussian_id"],
                    "feature_load_offset": culprit["gaussian_id"],
                    "feature_source_matches_sorted_gaussian": True,
                    "logical_feature_workspace": workspace,
                    "loaded_feature_workspace": copy.deepcopy(workspace),
                    "initial_load": {
                        "pair_slot": culprit["range_relative_slot"],
                        "offset_assignment_guard": True,
                        "load_enable_guard": True,
                        "zero_offset_fallback": False,
                    },
                    "loaded_feature_target_power": culprit["target_reprojected_power"],
                    "loaded_feature_target_alpha": culprit["target_reprojected_alpha"],
                    "branch": "contributed",
                },
            }
        )
    return {
        "schema_version": "flashgs-pipeline-debug-v4",
        "debug_only": True,
        "measured_timing_valid": False,
        "target_pixel_xy": indexed_case["pixel_xy"],
        "expected_semantic_id_filter": indexed_case["oracle_semantic_id"],
        "explicit_gaussian_ids": sorted(culprit["gaussian_id"] for culprit in indexed_case["culprits"]),
        "candidate_discovery": {
            "mode": "explicit-only",
            "matched": len(indexed_case["culprits"]),
            "selected": len(indexed_case["culprits"]),
            "truncated": 0,
        },
        "projection_replay_crosscheck": {"all_bit_exact": True},
        "discovery_score_crosscheck": {
            "all_bit_exact": None,
            "not_applicable": True,
        },
        "records": records,
    }


def test_repaired_case_requires_correct_source_and_output(
    indexed_evidence: dict[str, Any],
) -> None:
    indexed_case = indexed_evidence["index"]["cases"][0]
    frozen_case = json.loads(indexed_evidence["known"].read_text(encoding="utf-8"))["cases"][0]
    oracle_pixel = {
        "alpha": frozen_case["oracle_alpha"],
        "semantic_id": frozen_case["oracle_semantic_id"],
    }
    trace = _repaired_trace(indexed_case)

    result = evaluate_repaired_b64_case(
        frozen_case=frozen_case,
        indexed_case=indexed_case,
        oracle_pixel=oracle_pixel,
        rendered_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        trace=trace,
        capacity_counters={"intersection_overflow": False},
    )

    assert result["pass"] is True
    trace["records"][0]["compositor"]["initial_load"]["zero_offset_fallback"] = True
    failed = evaluate_repaired_b64_case(
        frozen_case=frozen_case,
        indexed_case=indexed_case,
        oracle_pixel=oracle_pixel,
        rendered_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        trace=trace,
        capacity_counters={"intersection_overflow": False},
    )
    assert failed["pass"] is False
    assert failed["culprits"][0]["checks"]["no_zero_offset_fallback"] is False


def _offset_float32_ulps(value: float, steps: int) -> float:
    result = np.float32(value)
    direction = np.float32(np.inf)
    for _ in range(steps):
        result = np.nextafter(result, direction, dtype=np.float32)
    return float(result)


@pytest.mark.parametrize(
    ("field", "loaded_field"),
    [
        ("reprojected_alpha", "loaded_feature_target_alpha"),
        ("reprojected_power", "loaded_feature_target_power"),
    ],
)
def test_repaired_case_allows_only_bounded_cross_architecture_trace_rounding(
    indexed_evidence: dict[str, Any], field: str, loaded_field: str
) -> None:
    indexed_case = indexed_evidence["index"]["cases"][0]
    frozen_case = json.loads(indexed_evidence["known"].read_text(encoding="utf-8"))["cases"][0]

    within = _repaired_trace(indexed_case)
    within_value = _offset_float32_ulps(within["records"][0]["target_pixel"][field], 128)
    within["records"][0]["target_pixel"][field] = within_value
    within["records"][0]["compositor"][loaded_field] = within_value
    accepted = evaluate_repaired_b64_case(
        frozen_case=frozen_case,
        indexed_case=indexed_case,
        oracle_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        rendered_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        trace=within,
        capacity_counters={"intersection_overflow": False},
    )
    assert accepted["pass"] is True

    outside = _repaired_trace(indexed_case)
    outside_value = _offset_float32_ulps(outside["records"][0]["target_pixel"][field], 129)
    outside["records"][0]["target_pixel"][field] = outside_value
    outside["records"][0]["compositor"][loaded_field] = outside_value
    rejected = evaluate_repaired_b64_case(
        frozen_case=frozen_case,
        indexed_case=indexed_case,
        oracle_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        rendered_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        trace=outside,
        capacity_counters={"intersection_overflow": False},
    )
    failed_check = f"target_{field.removeprefix('reprojected_')}_matches_historical_trace_ulp_bound"
    assert rejected["pass"] is False
    assert rejected["culprits"][0]["checks"][failed_check] is False


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    [
        (
            lambda trace: trace["records"][0]["compositor"].__setitem__("seen", False),
            "compositor_seen",
        ),
        (
            lambda trace: trace["records"][0]["projection"]["retained_workspace_crosscheck"].__setitem__(
                "available", False
            ),
            "projection_workspace_replay_available",
        ),
        (
            lambda trace: trace["records"][0]["compositor"]["loaded_feature_workspace"]["conic_opacity"].__setitem__(
                1, -0.0
            ),
            "workspace_bit_equal",
        ),
        (
            lambda trace: trace["records"][0]["compositor"]["loaded_feature_workspace"]["point_xy"].__setitem__(0, 1),
            "workspace_bit_equal",
        ),
        (
            lambda trace: trace["records"][0]["compositor"]["feature_load_evidence"].__setitem__(
                "production_lane_registers_observed", True
            ),
            "production_lane_registers_not_claimed_observed",
        ),
    ],
)
def test_repaired_case_fails_closed_on_culprit_workspace_evidence(
    indexed_evidence: dict[str, Any], mutate: Any, failed_check: str
) -> None:
    indexed_case = indexed_evidence["index"]["cases"][0]
    frozen_case = json.loads(indexed_evidence["known"].read_text(encoding="utf-8"))["cases"][0]
    trace = _repaired_trace(indexed_case)
    mutate(trace)

    result = evaluate_repaired_b64_case(
        frozen_case=frozen_case,
        indexed_case=indexed_case,
        oracle_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        rendered_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        trace=trace,
        capacity_counters={"intersection_overflow": False},
    )

    assert result["pass"] is False
    assert result["culprits"][0]["checks"][failed_check] is False


def test_repaired_case_requires_current_debug_trace_contract(indexed_evidence: dict[str, Any]) -> None:
    indexed_case = indexed_evidence["index"]["cases"][0]
    frozen_case = json.loads(indexed_evidence["known"].read_text(encoding="utf-8"))["cases"][0]
    trace = _repaired_trace(indexed_case)
    trace["schema_version"] = PRE_FIX_PIPELINE_DEBUG_SCHEMA

    result = evaluate_repaired_b64_case(
        frozen_case=frozen_case,
        indexed_case=indexed_case,
        oracle_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        rendered_pixel={
            "alpha": frozen_case["oracle_alpha"],
            "semantic_id": frozen_case["oracle_semantic_id"],
        },
        trace=trace,
        capacity_counters={"intersection_overflow": False},
    )

    assert result["pass"] is False
    assert result["checks"]["trace_schema_is_current_v4"] is False


def _rendered_outputs_from_corpus(
    corpus: dict[str, Any],
) -> dict[int, dict[str, np.ndarray]]:
    rendered: dict[int, dict[str, np.ndarray]] = {}
    for pixel in corpus["pixels"]:
        camera = int(pixel["camera_index"])
        arrays = rendered.setdefault(
            camera,
            {
                "rgb": np.zeros((128, 128, 3), dtype=np.float32),
                "alpha": np.zeros((128, 128, 1), dtype=np.float32),
                "depth": np.zeros((128, 128, 1), dtype=np.float32),
                "semantic_id": np.full((128, 128, 1), -1, dtype=np.int64),
            },
        )
        x, y = pixel["pixel_xy"]
        arrays["rgb"][y, x] = pixel["oracle"]["rgb"]
        arrays["alpha"][y, x, 0] = pixel["oracle"]["alpha"]
        arrays["depth"][y, x, 0] = pixel["oracle"]["depth"]
        arrays["semantic_id"][y, x, 0] = pixel["oracle"]["semantic_id"]
    return rendered


def test_all_92_mismatch_outputs_use_explicit_tolerances(
    indexed_evidence: dict[str, Any],
) -> None:
    corpus = indexed_evidence["index"]["historical_mismatch_corpus"]
    rendered = _rendered_outputs_from_corpus(corpus)

    result = evaluate_repaired_mismatch_corpus(
        corpus=corpus,
        rendered_by_camera=rendered,
    )

    assert result["pass"] is True
    assert result["pixel_count"] == 92
    first = corpus["pixels"][0]
    x, y = first["pixel_xy"]
    rendered[int(first["camera_index"])]["alpha"][y, x, 0] += 1.0e-4
    failed = evaluate_repaired_mismatch_corpus(
        corpus=corpus,
        rendered_by_camera=rendered,
    )
    assert failed["pass"] is False
    assert failed["passed_pixel_count"] == 91

    rgb_only = evaluate_repaired_rgb_only_corpus(
        corpus=corpus,
        rendered_rgb_by_camera={camera: arrays["rgb"] for camera, arrays in rendered.items()},
    )
    assert rgb_only["pass"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        lambda outputs, camera: outputs[camera].__setitem__("rgb", outputs[camera]["rgb"].astype(np.float64)),
        lambda outputs, camera: outputs[camera].__setitem__("alpha", np.zeros((256, 256, 1), dtype=np.float32)),
        lambda outputs, camera: outputs[camera].__setitem__(
            "semantic_id", outputs[camera]["semantic_id"].astype(np.float32) + 0.9
        ),
        lambda outputs, camera: outputs[camera]["rgb"].__setitem__((0, 0, 0), np.nan),
    ),
)
def test_mismatch_evaluator_rejects_wrong_tensor_contracts(indexed_evidence: dict[str, Any], mutation: Any) -> None:
    corpus = indexed_evidence["index"]["historical_mismatch_corpus"]
    rendered = _rendered_outputs_from_corpus(corpus)
    camera = int(corpus["pixels"][0]["camera_index"])
    mutation(rendered, camera)

    with pytest.raises(ValueError):
        evaluate_repaired_mismatch_corpus(
            corpus=corpus,
            rendered_by_camera=rendered,
        )


def test_rgb_only_evaluator_rejects_float64(indexed_evidence: dict[str, Any]) -> None:
    corpus = indexed_evidence["index"]["historical_mismatch_corpus"]
    rendered = _rendered_outputs_from_corpus(corpus)

    with pytest.raises(ValueError, match="float32"):
        evaluate_repaired_rgb_only_corpus(
            corpus=corpus,
            rendered_rgb_by_camera={camera: arrays["rgb"].astype(np.float64) for camera, arrays in rendered.items()},
        )


def _write_raw_repair_report(indexed_evidence: dict[str, Any], *, directory: Path) -> Path:
    corpus = indexed_evidence["index"]["historical_mismatch_corpus"]
    sparse = _rendered_outputs_from_corpus(corpus)
    rgb = np.zeros((64, 128, 128, 3), dtype=np.float32)
    alpha = np.zeros((64, 128, 128, 1), dtype=np.float32)
    depth = np.full((64, 128, 128, 1), np.inf, dtype=np.float32)
    semantic = np.full((64, 128, 128, 1), -1, dtype=np.int64)
    for camera, arrays in sparse.items():
        rgb[camera] = arrays["rgb"]
        alpha[camera] = arrays["alpha"]
        semantic[camera] = arrays["semantic_id"]
        for pixel in corpus["pixels"]:
            if int(pixel["camera_index"]) == camera:
                x, y = pixel["pixel_xy"]
                depth[camera, y, x, 0] = arrays["depth"][y, x, 0]
    cameras = np.arange(64, dtype=np.int64)
    full_path = directory / "repair.full.npz"
    rgb_path = directory / "repair.rgb-only.npz"
    np.savez(
        full_path,
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        semantic_id=semantic,
        camera_indices=cameras,
        step=np.asarray(107, dtype=np.int64),
        trajectory_id=np.asarray(TRAJECTORY_ID),
        semantic_topology=np.asarray(REPRESENTATIVE_SEMANTIC_TOPOLOGY),
    )
    np.savez(
        rgb_path,
        rgb=rgb,
        camera_indices=cameras,
        step=np.asarray(107, dtype=np.int64),
        trajectory_id=np.asarray(TRAJECTORY_ID),
        semantic_topology=np.asarray(REPRESENTATIVE_SEMANTIC_TOPOLOGY),
    )
    full_eval = evaluate_repaired_mismatch_corpus(
        corpus=corpus,
        rendered_by_camera={
            camera: {
                "rgb": rgb[camera],
                "alpha": alpha[camera],
                "depth": depth[camera],
                "semantic_id": semantic[camera],
            }
            for camera in range(64)
        },
    )
    rgb_eval = evaluate_repaired_rgb_only_corpus(
        corpus=corpus,
        rendered_rgb_by_camera={camera: rgb[camera] for camera in range(64)},
    )

    def relative_record(path: Path) -> dict[str, Any]:
        record = artifact_record(path)
        record["path"] = path.relative_to(directory).as_posix()
        return record

    report_path = directory / "repair.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA,
                "pass": True,
                "tool_integrity": {"pass": True},
                "diagnosis_index": artifact_record(indexed_evidence["index_path"]),
                "diagnosis_lock": indexed_evidence["index"]["diagnosis_lock"],
                "camera": {"trajectory_id": TRAJECTORY_ID, "step": 107},
                "equation": {"semantic_topology": REPRESENTATIVE_SEMANTIC_TOPOLOGY},
                "post_fix_raw_outputs": {
                    "full": relative_record(full_path),
                    "rgb_only": relative_record(rgb_path),
                },
                "historical_mismatch_repair": full_eval,
                "rgb_only_specialization": {"result": rgb_eval},
            }
        ),
        encoding="utf-8",
    )
    return report_path


def test_raw_b64_loader_recomputes_and_relocates(indexed_evidence: dict[str, Any], tmp_path: Path) -> None:
    report = _write_raw_repair_report(indexed_evidence, directory=indexed_evidence["root"])
    loaded = load_verified_b64_repair_raw_outputs(
        report,
        diagnosis_index=indexed_evidence["index_path"],
        diagnosis_lock=indexed_evidence["lock"],
        artifact_root=indexed_evidence["root"],
    )
    assert loaded["recomputed_full"]["pass"] is True
    assert loaded["recomputed_rgb_only"]["pass"] is True

    relocated = tmp_path / "relocated-raw-artifact-root"
    shutil.copytree(indexed_evidence["root"], relocated)
    relocated_loaded = load_verified_b64_repair_raw_outputs(
        relocated / report.name,
        diagnosis_index=relocated / indexed_evidence["index_path"].name,
        diagnosis_lock=relocated / indexed_evidence["lock"].name,
        artifact_root=relocated,
    )
    assert relocated_loaded["full"]["camera_indices"] == list(range(64))


def test_raw_b64_loader_rejects_reported_verdict_tamper(indexed_evidence: dict[str, Any]) -> None:
    report = _write_raw_repair_report(indexed_evidence, directory=indexed_evidence["root"])
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["historical_mismatch_repair"]["passed_pixel_count"] = 91
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="reported verdicts"):
        load_verified_b64_repair_raw_outputs(
            report,
            diagnosis_index=indexed_evidence["index_path"],
            diagnosis_lock=indexed_evidence["lock"],
            artifact_root=indexed_evidence["root"],
        )


def _copy_native_sources(tmp_path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src/isaacsim_gaussian_renderer/native/flashgs"
    target_root = tmp_path / "flashgs-native"
    target_root.mkdir()
    for name in ("adapter.cpp", "preprocess.cu", "sort.cu", "render.cu", "ops.h"):
        shutil.copy2(source_root / name, target_root / name)
    return target_root


def _build_contract(native_root: Path) -> dict[str, Any]:
    from isaacsim_gaussian_renderer.flashgs_native_loader import (
        FLASHGS_CUDA_FLAGS,
        FLASHGS_CXX_FLAGS,
        FLASHGS_UPSTREAM_COMMIT,
    )

    return {
        "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
        "cxx_flags": list(FLASHGS_CXX_FLAGS),
        "cuda_flags": list(FLASHGS_CUDA_FLAGS),
        "torch_cuda_arch_list": HEADLINE_TORCH_CUDA_ARCH_LIST,
        "module_name": flashgs_adapter_module_name(native_root),
        "sources": [str(native_root / name) for name in ("adapter.cpp", "preprocess.cu", "sort.cu", "render.cu")],
        "header": str(native_root / "ops.h"),
    }


def _pre_fix_source_identity() -> dict[str, Any]:
    return {"head": FLASHGS_B64_PRE_FIX_COMMIT, "dirty": False}


def test_production_audit_rejects_wrong_build_digest(tmp_path: Path) -> None:
    native_root = _copy_native_sources(tmp_path)
    contract = _build_contract(native_root)

    passing = audit_repaired_flashgs_production(
        native_root=native_root,
        build_contract=contract,
        repository_root=Path(__file__).resolve().parents[1],
        pre_fix_source_identity=_pre_fix_source_identity(),
        loaded_module_name=contract["module_name"],
    )
    assert passing["pass"] is True

    contract["module_name"] = "isaacsim_flashgs_adapter_deadbeefdead"
    failed = audit_repaired_flashgs_production(
        native_root=native_root,
        build_contract=contract,
        repository_root=Path(__file__).resolve().parents[1],
        pre_fix_source_identity=_pre_fix_source_identity(),
        loaded_module_name=contract["module_name"],
    )
    assert failed["pass"] is False
    assert failed["checks"]["build_module_name_matches_source_digest"] is False


def test_production_audit_rejects_buggy_short_tail_guards(
    tmp_path: Path,
) -> None:
    native_root = _copy_native_sources(tmp_path)
    render_path = native_root / "render.cu"
    render_source = render_path.read_text(encoding="utf-8")
    render_path.write_text(
        render_source.replace(
            "else if ((lane & 4) == 0 && point_id + 0 < range.y)",
            "else if ((lane & 4) == 0 && point_id + 1 < range.y)",
            1,
        ).replace(
            "else if (point_id + 1 < range.y)",
            "else if (point_id + 2 < range.y)",
            1,
        ),
        encoding="utf-8",
    )
    contract = _build_contract(native_root)

    failed = audit_repaired_flashgs_production(
        native_root=native_root,
        build_contract=contract,
        repository_root=Path(__file__).resolve().parents[1],
        pre_fix_source_identity=_pre_fix_source_identity(),
        loaded_module_name=contract["module_name"],
    )

    assert failed["pass"] is False
    assert failed["checks"]["buggy_slot_0_predicate_absent"] is False
    assert failed["checks"]["buggy_slot_1_predicate_absent"] is False


def test_production_audit_rejects_any_extra_render_change(tmp_path: Path) -> None:
    native_root = _copy_native_sources(tmp_path)
    render_path = native_root / "render.cu"
    render_path.write_text(
        render_path.read_text(encoding="utf-8") + "\n// unrelated change\n",
        encoding="utf-8",
    )
    contract = _build_contract(native_root)

    failed = audit_repaired_flashgs_production(
        native_root=native_root,
        build_contract=contract,
        repository_root=Path(__file__).resolve().parents[1],
        pre_fix_source_identity=_pre_fix_source_identity(),
        loaded_module_name=contract["module_name"],
    )

    assert failed["pass"] is False
    assert failed["checks"]["production_render_diff_is_exact_two_line_repair"] is False


def test_production_audit_rejects_wrong_arch_and_flags(tmp_path: Path) -> None:
    native_root = _copy_native_sources(tmp_path)
    contract = _build_contract(native_root)
    contract["torch_cuda_arch_list"] = "8.6"
    contract["cuda_flags"] = ["-O0"]

    failed = audit_repaired_flashgs_production(
        native_root=native_root,
        build_contract=contract,
        repository_root=Path(__file__).resolve().parents[1],
        pre_fix_source_identity=_pre_fix_source_identity(),
        loaded_module_name=contract["module_name"],
    )

    assert failed["pass"] is False
    assert failed["checks"]["build_arch_matches_headline_gpu"] is False
    assert failed["checks"]["build_cuda_flags_match"] is False


def test_repair_aggregate_requires_exactly_five_passes() -> None:
    expected = {f"case-{index}" for index in range(5)}
    results = [{"case_id": case_id, "pass": True} for case_id in sorted(expected)]

    assert aggregate_b64_repair_cases(results, expected_case_ids=expected)["pass"] is True
    assert aggregate_b64_repair_cases(results[:-1], expected_case_ids=expected)["pass"] is False
