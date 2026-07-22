from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaacsim_gaussian_renderer.flashgs_baseline_contract import (
    FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
    FLASHGS_MATCHED_PORT_CLASSIFICATION,
    require_flashgs_short_tail_repair,
)
from isaacsim_gaussian_renderer.flashgs_debug import (
    ATTESTED_ADAPTER_PATHS,
    COMPOSITOR_BRANCHES,
    FLASHGS_PIPELINE_DEBUG_SCHEMA,
    FLOAT_FIELDS,
    INT_FIELDS,
    MAX_TRACED_CANDIDATES,
    PROJECTION_REJECTIONS,
    _record_from_rows,
    classify_candidate_trace,
    current_optimized_feature_source_ids,
    load_verified_b64_known_failure_case,
    load_verified_flashgs_adapter_attestation,
    trace_flashgs_backend,
)
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import artifact_record
from isaacsim_gaussian_renderer.flashgs_native_loader import (
    FLASHGS_CUDA_FLAGS,
    FLASHGS_CXX_FLAGS,
    FLASHGS_UPSTREAM_COMMIT,
)


def int_fields(**updates: int) -> dict[str, int]:
    result = {name: 0 for name in INT_FIELDS}
    result.update(
        {
            "projection_rejection": 0,
            "target_pixel_branch": 4,
            "target_tile_in_rect": 1,
            "old_predicate_accept": 1,
            "unsorted_target_count": 1,
            "sorted_target_count": 1,
            "candidate_in_range_count": 1,
            "compositor_feature_gaussian_id": 7,
            "compositor_feature_matches_sorted_id": 1,
            "compositor_branch": 6,
        }
    )
    result.update(updates)
    return result


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"projection_rejection": 3}, "projection"),
        (
            {"target_pixel_branch": 3},
            "not-an-individual-target-pixel-contributor",
        ),
        ({"target_tile_in_rect": 0}, "projected-tile-bounds"),
        (
            {"old_predicate_accept": 0, "unsorted_target_count": 0},
            "old-tile-intersection-predicate",
        ),
        ({"unsorted_target_count": 0}, "tile-enumeration"),
        ({"sorted_target_count": 0}, "radix-sort"),
        ({"candidate_in_range_count": 0}, "tile-range-construction"),
        (
            {"compositor_feature_matches_sorted_id": 0},
            "optimized-compositor-feature-load",
        ),
        ({"compositor_branch": 3}, "compositor"),
        ({}, "survived-compositor"),
    ],
)
def test_classify_candidate_trace_names_first_loss(updates: dict[str, int], expected: str) -> None:
    assert classify_candidate_trace(int_fields(**updates)) == expected


def test_record_dumps_exact_old_predicate_inputs_and_branches() -> None:
    float_row = [float("nan")] * len(FLOAT_FIELDS)
    int_row = [0] * len(INT_FIELDS)
    floats = {name: index for index, name in enumerate(FLOAT_FIELDS)}
    ints = {name: index for index, name in enumerate(INT_FIELDS)}
    float_row[floats["half_support_squared"]] = 5.5
    float_row[floats["tile_pixel_min_x"]] = 112.5
    float_row[floats["tile_pixel_min_y"]] = 0.5
    float_row[floats["tile_pixel_max_x"]] = 127.5
    float_row[floats["tile_pixel_max_y"]] = 15.5
    float_row[floats["old_first_delta"]] = 4.0
    float_row[floats["old_first_t1"]] = 3.0
    float_row[floats["old_first_t2"]] = 2.0
    float_row[floats["tile_center_q"]] = 1.25
    for name in ("corner_00_q", "corner_10_q", "corner_01_q", "corner_11_q"):
        float_row[floats[name]] = 2.0
    int_row[ints["gaussian_id"]] = 123
    int_row[ints["projection_rejection"]] = 0
    int_row[ints["target_pixel_branch"]] = 4
    int_row[ints["target_tile_in_rect"]] = 1
    int_row[ints["old_first_segment_evaluated"]] = 1
    int_row[ints["old_first_segment_result"]] = 0
    int_row[ints["old_second_segment_evaluated"]] = 1
    int_row[ints["old_second_segment_result"]] = 0
    int_row[ints["old_predicate_accept"]] = 0
    int_row[ints["tile_center_inside_ellipse"]] = 1
    int_row[ints["all_tile_corners_inside_ellipse"]] = 1

    record = _record_from_rows(float_row, int_row, semantic_id=2)

    predicate = record["old_tile_predicate"]
    assert record["first_loss_stage"] == "old-tile-intersection-predicate"
    assert predicate["pixel_center_bounds"] == [112.5, 0.5, 127.5, 15.5]
    assert predicate["first_segment"] == {
        "evaluated": True,
        "result": False,
        "dx": None,
        "a": None,
        "b": None,
        "c": None,
        "delta": 4.0,
        "t1": 3.0,
        "t2": 2.0,
    }
    assert predicate["second_segment"]["evaluated"] is True
    assert predicate["accepted"] is False
    assert predicate["ellipse_limit_q"] == 11.0
    assert predicate["tile_center_inside_ellipse"] is True
    assert predicate["all_tile_corners_inside_ellipse"] is True


def test_feature_load_evidence_uses_new_debug_schema() -> None:
    assert FLASHGS_PIPELINE_DEBUG_SCHEMA == "flashgs-pipeline-debug-v4"


@pytest.mark.parametrize("length", range(10))
def test_repaired_compositor_schedule_preserves_feature_sources(
    length: int,
) -> None:
    logical_ids = list(range(41, 41 + length))

    feature_ids = current_optimized_feature_source_ids(logical_ids)

    assert feature_ids == logical_ids


def test_one_entry_trace_records_repaired_feature_load() -> None:
    floats = {name: index for index, name in enumerate(FLOAT_FIELDS)}
    ints = {name: index for index, name in enumerate(INT_FIELDS)}
    float_row = [float("nan")] * len(FLOAT_FIELDS)
    int_row = [0] * len(INT_FIELDS)
    int_row[ints["gaussian_id"]] = 9_237_708
    int_row[ints["target_pixel_branch"]] = 4
    int_row[ints["target_tile_in_rect"]] = 1
    int_row[ints["old_predicate_accept"]] = 1
    int_row[ints["unsorted_target_count"]] = 1
    int_row[ints["sorted_target_count"]] = 1
    int_row[ints["candidate_in_range_count"]] = 1
    int_row[ints["range_start"]] = 0
    int_row[ints["range_end"]] = 1
    int_row[ints["sorted_position"]] = 0
    int_row[ints["compositor_seen"]] = 1
    int_row[ints["compositor_branch"]] = 6
    int_row[ints["compositor_feature_gaussian_id"]] = 9_237_708
    int_row[ints["compositor_feature_matches_sorted_id"]] = 1
    int_row[ints["compositor_initial_pair_slot"]] = 0
    int_row[ints["compositor_offset_assignment_guard"]] = 1
    int_row[ints["compositor_load_enable_guard"]] = 1
    int_row[ints["compositor_zero_offset_fallback"]] = 0
    int_row[ints["compositor_feature_load_offset"]] = 9_237_708
    float_row[floats["target_alpha_reprojected"]] = 0.048328828
    float_row[floats["compositor_alpha"]] = 0.048328828
    float_row[floats["workspace_point_x"]] = 111.5
    float_row[floats["workspace_point_y"]] = 31.5
    float_row[floats["workspace_conic_x"]] = 0.25
    float_row[floats["workspace_conic_y"]] = 0.0
    float_row[floats["workspace_conic_z"]] = 0.25
    float_row[floats["workspace_opacity"]] = 0.048328828
    float_row[floats["compositor_loaded_point_x"]] = 111.5
    float_row[floats["compositor_loaded_point_y"]] = 31.5
    float_row[floats["compositor_loaded_conic_x"]] = 0.25
    float_row[floats["compositor_loaded_conic_y"]] = 0.0
    float_row[floats["compositor_loaded_conic_z"]] = 0.25
    float_row[floats["compositor_loaded_opacity"]] = 0.048328828

    record = _record_from_rows(float_row, int_row, semantic_id=6)

    assert record["first_loss_stage"] == "survived-compositor"
    assert record["enumeration_sort_range"]["range_length"] == 1
    assert record["compositor"]["logical_sorted_gaussian_id"] == 9_237_708
    assert record["compositor"]["feature_load_evidence"] == {
        "kind": "source-grounded-debug-model",
        "production_lane_registers_observed": False,
        "bound_source": "native/flashgs/render.cu",
    }
    assert record["compositor"]["feature_gaussian_id"] == 9_237_708
    assert record["compositor"]["feature_load_offset"] == 9_237_708
    assert record["compositor"]["logical_feature_workspace"] == {
        "point_xy": [111.5, 31.5],
        "conic_opacity": [0.25, 0.0, 0.25, 0.048328828],
    }
    assert record["compositor"]["loaded_feature_workspace"] == record["compositor"]["logical_feature_workspace"]
    assert record["compositor"]["loaded_feature_target_alpha"] == 0.048328828
    assert record["compositor"]["initial_load"] == {
        "pair_slot": 0,
        "offset_initial_value": 0,
        "offset_assignment_guard": True,
        "load_enable_guard": True,
        "zero_offset_fallback": False,
        "slot_0_offset_condition": "point_id + 0 < range.y",
        "slot_0_load_condition": "point_id + 0 < range.y",
        "slot_1_offset_condition": "point_id + 1 < range.y",
        "slot_1_load_condition": "point_id + 1 < range.y",
    }


def test_debug_schedule_model_is_bound_to_current_render_source() -> None:
    render_source = (
        Path(__file__).resolve().parents[1] / "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
    ).read_text(encoding="utf-8")

    assert "int offset = 0;" in render_source
    assert "(lane & 4) == 0 && point_id + 0 < range.y" in render_source
    assert "else if (point_id + 1 < range.y)" in render_source
    assert "(lane & 4) == 0 && point_id + 1 < range.y" not in render_source
    assert "else if (point_id + 2 < range.y)" not in render_source
    assert "load_enable = load_enable && point_id + 0 < range.y" in (render_source)
    assert "load_enable = load_enable && point_id + 1 < range.y" in (render_source)


def test_known_b64_failure_manifest_binds_exact_camera_and_pixel() -> None:
    manifest = Path(__file__).resolve().parents[1] / "experiments/flashgs_matched/B64_KNOWN_FAILURE_CASES.json"

    selected = load_verified_b64_known_failure_case(
        manifest,
        trajectory_id=("e30050bf2d873825cf7cdebbe799c911ddaee5cc15d90e1c0dd8adc2c2e62cc0"),
        scene_sha256=("29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"),
        step=107,
        camera_index=36,
        pixel_x=123,
        pixel_y=15,
        width=128,
        height=128,
        expected_semantic_id=2,
    )

    assert selected["case"]["case_id"] == "camera36-tile7-0"
    assert selected["case"]["oracle_alpha"] == 0.2515307664871216


def test_known_b64_failure_manifest_rejects_unlisted_pixel() -> None:
    manifest = Path(__file__).resolve().parents[1] / "experiments/flashgs_matched/B64_KNOWN_FAILURE_CASES.json"

    with pytest.raises(ValueError, match="not exactly one case"):
        load_verified_b64_known_failure_case(
            manifest,
            trajectory_id=("e30050bf2d873825cf7cdebbe799c911ddaee5cc15d90e1c0dd8adc2c2e62cc0"),
            scene_sha256=("29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"),
            step=107,
            camera_index=36,
            pixel_x=0,
            pixel_y=0,
            width=128,
            height=128,
            expected_semantic_id=-1,
        )


class FakeDebugNative:
    __vgr_build_contract__ = {"purpose": "test"}

    def __init__(
        self,
        scores: list[float] | None = None,
        *,
        old_predicate_accept: int = 1,
        unsorted_target_count: int = 0,
    ) -> None:
        self.scores = torch.tensor(
            scores if scores is not None else [float("-inf")] * 10,
            dtype=torch.float32,
        )
        self.old_predicate_accept = old_predicate_accept
        self.unsorted_target_count = unsorted_target_count
        self.require_old_predicate_rejection: bool | None = None
        self.traced_intersection_lengths: tuple[int, int, int, int] | None = None

    def score_projected_contributors(self, *args: object) -> torch.Tensor:
        self.require_old_predicate_rejection = bool(args[-1])
        return self.scores.clone()

    def trace_candidates(self, *args: object) -> list[torch.Tensor]:
        self.traced_intersection_lengths = tuple(int(args[index].numel()) for index in (5, 6, 7, 8))
        candidate_ids = args[12]
        count = candidate_ids.numel()
        float_trace = torch.full((count, len(FLOAT_FIELDS)), float("nan"))
        int_trace = torch.zeros((count, len(INT_FIELDS)), dtype=torch.int64)
        id_index = INT_FIELDS.index("gaussian_id")
        projection_index = INT_FIELDS.index("projection_rejection")
        pixel_index = INT_FIELDS.index("target_pixel_branch")
        rect_index = INT_FIELDS.index("target_tile_in_rect")
        old_index = INT_FIELDS.index("old_predicate_accept")
        unsorted_index = INT_FIELDS.index("unsorted_target_count")
        sorted_index = INT_FIELDS.index("sorted_target_count")
        range_index = INT_FIELDS.index("candidate_in_range_count")
        compositor_index = INT_FIELDS.index("compositor_branch")
        compositor_feature_index = INT_FIELDS.index("compositor_feature_gaussian_id")
        compositor_match_index = INT_FIELDS.index("compositor_feature_matches_sorted_id")
        alpha_index = FLOAT_FIELDS.index("target_alpha_reprojected")
        int_trace[:, id_index] = candidate_ids.cpu()
        int_trace[:, projection_index] = 0
        int_trace[:, pixel_index] = 4
        int_trace[:, rect_index] = 1
        int_trace[:, old_index] = self.old_predicate_accept
        int_trace[:, unsorted_index] = self.unsorted_target_count
        int_trace[:, sorted_index] = 0
        int_trace[:, range_index] = 0
        int_trace[:, compositor_index] = 0
        int_trace[:, compositor_feature_index] = candidate_ids.cpu()
        int_trace[:, compositor_match_index] = 1
        float_trace[:, alpha_index] = self.scores.index_select(0, candidate_ids.cpu())
        return [float_trace, int_trace]


def fake_backend() -> SimpleNamespace:
    scene = SimpleNamespace(
        means=torch.zeros((10, 3)),
        opacities=torch.ones(10),
        semantic_ids=torch.arange(10, dtype=torch.int64),
        count=10,
    )
    workspace = {
        "keys_unsorted": torch.zeros(4, dtype=torch.uint64),
        "values_unsorted": torch.zeros(4, dtype=torch.int32),
        "keys_sorted": torch.zeros(4, dtype=torch.uint64),
        "values_sorted": torch.zeros(4, dtype=torch.int32),
        "ranges": torch.zeros((1, 2), dtype=torch.int32),
        "points_xy": torch.zeros((10, 2)),
        "conic_opacity": torch.zeros((10, 4)),
        "counters": torch.tensor([[1, 4, 0]], dtype=torch.int64),
    }
    return SimpleNamespace(
        _scene=scene,
        _covariances=torch.zeros((10, 6)),
        workspace=workspace,
        _height=4,
        _width=4,
        near_plane=0.01,
        far_plane=100.0,
        gaussian_support_sigma=3.33,
        covariance_epsilon=0.0,
    )


def test_trace_backend_ranks_all_contributors_and_finds_nonpredicate_loss() -> None:
    scores = [float("-inf")] * 10
    scores[3] = 0.8
    scores[5] = 0.5
    scores[7] = 0.2
    result = trace_flashgs_backend(
        fake_backend(),
        viewmat=torch.eye(4),
        intrinsics=torch.eye(3),
        target_pixel_x=1,
        target_pixel_y=2,
        candidate_ids=[9],
        max_discovered_candidates=2,
        native_module=FakeDebugNative(scores),
    )

    discovery = result["candidate_discovery"]
    assert discovery["mode"] == "all-contributors"
    assert discovery["matched"] == 3
    assert discovery["selected"] == 2
    assert discovery["truncated"] == 1
    assert [item["gaussian_id"] for item in discovery["candidates"]] == [3, 5]
    assert result["explicit_gaussian_ids"] == [9]
    assert [record["gaussian_id"] for record in result["records"]] == [3, 5, 9]
    assert [record["semantic_id"] for record in result["records"]] == [3, 5, 9]
    assert result["records"][0]["first_loss_stage"] == "tile-enumeration"
    assert result["records"][0]["old_tile_predicate"]["accepted"] is True
    assert result["discovery_score_crosscheck"]["all_bit_exact"] is True
    assert result["cause_identified"] is True
    assert {cause["gaussian_id"] for cause in result["identified_causes"]} == {
        3,
        5,
    }
    assert COMPOSITOR_BRANCHES[6] == "contributed"
    assert PROJECTION_REJECTIONS[0] == "accepted"


def test_trace_backend_uses_stable_id_order_at_truncated_alpha_tie() -> None:
    scores = [float("-inf")] * 10
    scores[6] = scores[2] = scores[4] = 0.8

    result = trace_flashgs_backend(
        fake_backend(),
        viewmat=torch.eye(4),
        intrinsics=torch.eye(3),
        target_pixel_x=1,
        target_pixel_y=2,
        max_discovered_candidates=2,
        native_module=FakeDebugNative(scores),
    )

    discovery = result["candidate_discovery"]
    assert [item["gaussian_id"] for item in discovery["candidates"]] == [2, 4]
    assert discovery["cutoff_tie_total"] == 3
    assert discovery["cutoff_tie_selected"] == 2


def test_trace_backend_old_predicate_mode_is_explicit_and_not_default() -> None:
    scores = [float("-inf")] * 10
    scores[7] = 0.25
    native = FakeDebugNative(scores, old_predicate_accept=0)

    result = trace_flashgs_backend(
        fake_backend(),
        viewmat=torch.eye(4),
        intrinsics=torch.eye(3),
        target_pixel_x=1,
        target_pixel_y=2,
        discovery_mode="old-predicate-false-negatives",
        native_module=native,
    )

    assert native.require_old_predicate_rejection is True
    assert result["candidate_discovery"]["mode"] == ("old-predicate-false-negatives")


def test_trace_backend_zero_candidates_is_not_a_diagnosis() -> None:
    result = trace_flashgs_backend(
        fake_backend(),
        viewmat=torch.eye(4),
        intrinsics=torch.eye(3),
        target_pixel_x=1,
        target_pixel_y=2,
        native_module=FakeDebugNative(),
    )

    assert result["candidate_discovery"]["matched"] == 0
    assert result["records"] == []
    assert result["cause_identified"] is False
    assert result["identified_causes"] == []


def test_trace_backend_scans_only_generated_intersection_prefix() -> None:
    backend = fake_backend()
    backend.workspace["counters"][0, 1] = 2
    scores = [float("-inf")] * 10
    scores[3] = 0.8
    native = FakeDebugNative(scores)

    result = trace_flashgs_backend(
        backend,
        viewmat=torch.eye(4),
        intrinsics=torch.eye(3),
        target_pixel_x=1,
        target_pixel_y=2,
        native_module=native,
    )

    assert native.traced_intersection_lengths == (2, 2, 2, 2)
    assert result["candidate_discovery"]["valid_intersections_scanned_per_candidate"] == 2


def test_trace_backend_rejects_generated_count_above_capacity() -> None:
    backend = fake_backend()
    backend.workspace["counters"][0, 1] = 5

    with pytest.raises(RuntimeError, match="exceeds retained capacity"):
        trace_flashgs_backend(
            backend,
            viewmat=torch.eye(4),
            intrinsics=torch.eye(3),
            target_pixel_x=1,
            target_pixel_y=2,
            native_module=FakeDebugNative(),
        )


def test_trace_backend_rejects_invalid_explicit_gaussian_id() -> None:
    with pytest.raises(ValueError, match="outside"):
        trace_flashgs_backend(
            fake_backend(),
            viewmat=torch.eye(4),
            intrinsics=torch.eye(3),
            target_pixel_x=1,
            target_pixel_y=2,
            candidate_ids=[10],
            native_module=FakeDebugNative(),
        )


def test_trace_backend_bounds_total_trace_candidates() -> None:
    with pytest.raises(ValueError, match="max_discovered_candidates"):
        trace_flashgs_backend(
            fake_backend(),
            viewmat=torch.eye(4),
            intrinsics=torch.eye(3),
            target_pixel_x=1,
            target_pixel_y=2,
            max_discovered_candidates=MAX_TRACED_CANDIDATES + 1,
            native_module=FakeDebugNative(),
        )


def source_provenance() -> dict[str, object]:
    return {
        "manifest": {"sha256": "a" * 64},
        "source_tree_sha256": "b" * 64,
        "head": "c" * 40,
        "dirty": False,
        "diff_sha256": "d" * 64,
    }


def write_adapter_attestation(root: Path) -> Path:
    source_pairs = []
    for index, relative in enumerate(sorted(ATTESTED_ADAPTER_PATHS)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = f"adapter-{index}\n"
        if relative.endswith("/render.cu"):
            contents = """\
else if ((lane & 4) == 0 && point_id + 0 < range.y)
else if (point_id + 1 < range.y)
load_enable = load_enable && point_id + 0 < range.y
load_enable = load_enable && point_id + 1 < range.y
"""
        path.write_text(contents, encoding="utf-8")
        artifact = artifact_record(path)
        source_pairs.append(
            {
                "upstream_path": f"upstream-{index}",
                "upstream_bytes": 1,
                "upstream_sha256": "e" * 64,
                "adapter_path": relative,
                "adapter_bytes": artifact["bytes"],
                "adapter_sha256": artifact["sha256"],
            }
        )
    diff_path = root / "evidence/adapter.patch"
    diff_path.parent.mkdir(parents=True)
    diff_path.write_text("diff\n", encoding="utf-8")
    attestation_path = root / "evidence/adapter.json"
    render_path = root / "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
    repair_checks = require_flashgs_short_tail_repair(render_path.read_text(encoding="utf-8"))
    attestation_path.write_text(
        json.dumps(
            {
                "schema_version": FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
                "pass": True,
                "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
                "upstream_clean": True,
                "source_provenance": source_provenance(),
                "source_pairs": source_pairs,
                "adapter_diff": artifact_record(diff_path),
                "baseline_classification": FLASHGS_MATCHED_PORT_CLASSIFICATION,
                "correctness_repair_audit": {
                    "render_source": artifact_record(render_path),
                    "checks": repair_checks,
                    "pass": True,
                },
                "build_flags": {
                    "cxx": list(FLASHGS_CXX_FLAGS),
                    "cuda": list(FLASHGS_CUDA_FLAGS),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return attestation_path


def test_adapter_attestation_binds_current_sources_and_manifest(
    tmp_path: Path,
) -> None:
    attestation = write_adapter_attestation(tmp_path)

    verified = load_verified_flashgs_adapter_attestation(
        attestation,
        source_provenance=source_provenance(),
        project_root=tmp_path,
    )

    assert verified["upstream_commit"] == FLASHGS_UPSTREAM_COMMIT
    assert verified["adapter_paths"] == sorted(ATTESTED_ADAPTER_PATHS)
    assert verified["correctness_repair_audit"]["pass"] is True


def test_adapter_attestation_rejects_reverted_short_tail_repair(
    tmp_path: Path,
) -> None:
    attestation = write_adapter_attestation(tmp_path)
    render_path = tmp_path / "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
    render_path.write_text(
        render_path.read_text(encoding="utf-8")
        .replace("point_id + 0 < range.y)", "point_id + 1 < range.y)", 1)
        .replace(
            "else if (point_id + 1 < range.y)",
            "else if (point_id + 2 < range.y)",
            1,
        ),
        encoding="utf-8",
    )
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    changed = artifact_record(render_path)
    for pair in payload["source_pairs"]:
        if pair["adapter_path"].endswith("/render.cu"):
            pair["adapter_bytes"] = changed["bytes"]
            pair["adapter_sha256"] = changed["sha256"]
    payload["correctness_repair_audit"]["render_source"] = changed
    attestation.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="short-tail correctness repair"):
        load_verified_flashgs_adapter_attestation(
            attestation,
            source_provenance=source_provenance(),
            project_root=tmp_path,
        )


def test_adapter_attestation_rejects_minimal_adapter_mislabel(
    tmp_path: Path,
) -> None:
    attestation = write_adapter_attestation(tmp_path)
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload["baseline_classification"] = {
        **FLASHGS_MATCHED_PORT_CLASSIFICATION,
        "integration_only": True,
        "upstream_faithful": True,
    }
    attestation.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="semantic classification differs"):
        load_verified_flashgs_adapter_attestation(
            attestation,
            source_provenance=source_provenance(),
            project_root=tmp_path,
        )


def test_adapter_attestation_rejects_different_source_manifest(
    tmp_path: Path,
) -> None:
    attestation = write_adapter_attestation(tmp_path)
    different = source_provenance()
    different["head"] = "f" * 40

    with pytest.raises(ValueError, match="not bound"):
        load_verified_flashgs_adapter_attestation(
            attestation,
            source_provenance=different,
            project_root=tmp_path,
        )


def test_adapter_attestation_rejects_dirty_source_manifest(
    tmp_path: Path,
) -> None:
    attestation = write_adapter_attestation(tmp_path)
    dirty = source_provenance()
    dirty["dirty"] = True

    with pytest.raises(ValueError, match="clean source provenance"):
        load_verified_flashgs_adapter_attestation(
            attestation,
            source_provenance=dirty,
            project_root=tmp_path,
        )


def test_adapter_attestation_rejects_changed_adapter_source(
    tmp_path: Path,
) -> None:
    attestation = write_adapter_attestation(tmp_path)
    changed = tmp_path / sorted(ATTESTED_ADAPTER_PATHS)[0]
    changed.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source changed"):
        load_verified_flashgs_adapter_attestation(
            attestation,
            source_provenance=source_provenance(),
            project_root=tmp_path,
        )
