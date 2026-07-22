from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from benchmarks.verify_flashgs_short_tail_synthetic import (
    EXPECTED_RANGE_GAUSSIAN_IDS,
    EXPECTED_RANGE_LENGTHS,
    SCHEMA_VERSION,
    compose_cpu_oracle,
    synthetic_fixture_arrays,
    validate_result_payload,
    validate_tile_layout,
)


def _manual_workspace() -> dict[str, np.ndarray]:
    fixture = synthetic_fixture_arrays()
    points_xy = np.asarray(
        [[56.5, 8.5], [24.5, 8.5], [40.5, 8.5], [40.5, 8.5]] + [[56.5, 8.5]] * 3,
        dtype=np.float32,
    )
    inverse_variance = np.float32(1.0 / 4.0)
    conic_opacity = np.zeros((7, 4), dtype=np.float32)
    conic_opacity[:, 0] = inverse_variance
    conic_opacity[:, 2] = inverse_variance
    conic_opacity[:, 3] = fixture["opacities"]
    rgb_depth = np.concatenate([fixture["features"], fixture["means"][:, 2:3]], axis=1).astype(np.float32)
    return {
        "ranges": np.asarray([[0, 0], [0, 1], [1, 3], [3, 7]], dtype=np.int32),
        "point_list": np.asarray([1, 2, 3, 0, 4, 5, 6], dtype=np.int32),
        "points_xy": points_xy,
        "rgb_depth": rgb_depth,
        "conic_opacity": conic_opacity,
        "semantic_ids": fixture["semantic_ids"],
    }


def test_tile_layout_requires_exact_zero_one_two_four_contract() -> None:
    workspace = _manual_workspace()

    lengths, ids = validate_tile_layout(workspace["ranges"], workspace["point_list"])

    assert lengths == list(EXPECTED_RANGE_LENGTHS)
    assert ids == [list(values) for values in EXPECTED_RANGE_GAUSSIAN_IDS]
    bad_ranges = workspace["ranges"].copy()
    bad_ranges[2, 1] = 2
    with pytest.raises(ValueError, match="range lengths"):
        validate_tile_layout(bad_ranges, workspace["point_list"])


def test_cpu_oracle_exercises_empty_and_short_tail_semantics() -> None:
    workspace = _manual_workspace()

    oracle = compose_cpu_oracle(width=64, height=16, **workspace)

    center_semantics = [int(oracle["semantic_id"][8, x, 0]) for x in (8, 24, 40, 56)]
    assert center_semantics == [-1, 101, 202, 302]
    assert np.isinf(oracle["depth"][8, 8, 0])
    assert oracle["alpha"][8, 8, 0] == 0.0
    assert np.all(oracle["rgb"][8, 8] == 0.0)
    assert np.all(np.isfinite(oracle["depth"][oracle["alpha"] > 0.0]))


def _passing_payload() -> dict[str, object]:
    counters = {
        "visible_gaussians": 7,
        "generated_intersections": 7,
        "intersection_overflow": 0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture": {
            "range_lengths": list(EXPECTED_RANGE_LENGTHS),
            "range_gaussian_ids": [list(values) for values in EXPECTED_RANGE_GAUSSIAN_IDS],
        },
        "specializations": {
            "full_sensor": {
                "repeat_bitwise_equal": True,
                "counters": counters,
            },
            "rgb_only": {
                "repeat_bitwise_equal": True,
                "counters": counters,
            },
        },
        "checks": {
            "full_sensor_matches_cpu_oracle": True,
            "rgb_only_matches_cpu_oracle": True,
            "rgb_specializations_bitwise_equal": True,
            "workspace_layouts_equal": True,
            "output_contracts_valid": True,
            "occupancy_pass": True,
        },
        "environment": {
            "gpu_uuid": "GPU-test",
            "native_extension": {"sha256": "a" * 64},
        },
        "pass": True,
    }


def test_result_schema_validation_is_fail_closed() -> None:
    payload = _passing_payload()
    assert validate_result_payload(payload) == []

    broken = deepcopy(payload)
    broken["specializations"]["full_sensor"]["counters"][  # type: ignore[index]
        "intersection_overflow"
    ] = 1
    broken["checks"]["occupancy_pass"] = False  # type: ignore[index]
    errors = validate_result_payload(broken)

    assert "full_sensor overflow is nonzero" in errors
    assert "check failed: occupancy_pass" in errors
