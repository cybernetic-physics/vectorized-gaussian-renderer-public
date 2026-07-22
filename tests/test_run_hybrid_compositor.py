"""CPU tests for the hybrid compositor benchmark helpers and result shape."""

from __future__ import annotations

import pytest
import torch

from benchmarks.run_hybrid_compositor import (
    DEFAULT_BATCHES,
    DEFAULT_RESOLUTIONS,
    PROTOCOL_MIN_ITERATIONS,
    assemble_case_result,
    branch_coverage,
    build_case_grid,
    case_seed,
    forbid_host_transfers,
    hybrid_config_id,
    parse_csv_ints,
    parse_resolutions,
    synthesize_hybrid_layers,
)
from isaacsim_gaussian_renderer.benchmark_schema import flatten_result
from isaacsim_gaussian_renderer.hybrid_compositor import (
    composite_gaussian_and_mesh,
)


def test_hybrid_config_id_format() -> None:
    assert hybrid_config_id(64, 512, 512) == "cfg-v1-hybrid-compositor-b64-512x512"


def test_parse_helpers() -> None:
    assert parse_csv_ints(" 1, 8 ,32") == [1, 8, 32]
    assert parse_resolutions("128x128, 256X256") == [(128, 128), (256, 256)]


def test_default_case_grid_covers_protocol_matrix() -> None:
    cases = build_case_grid(DEFAULT_BATCHES, DEFAULT_RESOLUTIONS, base_seed=7)
    assert len(cases) == len(DEFAULT_BATCHES) * len(DEFAULT_RESOLUTIONS)
    assert cases[0]["config_id"] == "cfg-v1-hybrid-compositor-b1-128x128"
    assert cases[-1]["config_id"] == "cfg-v1-hybrid-compositor-b256-512x512"
    assert len({case["config_id"] for case in cases}) == len(cases)
    assert len({case["seed"] for case in cases}) == len(cases)
    assert all(case["seed"] == case_seed(7, case["batch"], case["width"], case["height"]) for case in cases)


def test_case_grid_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_case_grid([0], [(128, 128)], base_seed=1)
    with pytest.raises(ValueError, match="at least 8x8"):
        build_case_grid([1], [(4, 4)], base_seed=1)
    with pytest.raises(ValueError, match="at least one"):
        build_case_grid([], [(128, 128)], base_seed=1)


def test_synthesized_layers_are_deterministic() -> None:
    first = synthesize_hybrid_layers(2, 16, 16, seed=1234)
    second = synthesize_hybrid_layers(2, 16, 16, seed=1234)
    other = synthesize_hybrid_layers(2, 16, 16, seed=1235)
    for name, tensor in first.items():
        torch.testing.assert_close(
            tensor,
            second[name],
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=lambda message: f"{name}: {message}",
        )
    assert not torch.equal(first["gaussian_rgb"], other["gaussian_rgb"])


def test_synthesized_layers_exercise_every_compositor_branch() -> None:
    layers = synthesize_hybrid_layers(2, 16, 16, seed=99)
    coverage = branch_coverage(layers)
    assert coverage["coverage_complete"], coverage
    for key in (
        "mesh_front_both_valid",
        "gaussian_front_blend",
        "near_tie_both_valid",
        "mesh_only",
        "gaussian_only",
        "both_invalid",
    ):
        assert coverage[key] > 0, key


def test_synthesized_layers_composite_cleanly_on_cpu() -> None:
    layers = synthesize_hybrid_layers(1, 12, 12, seed=5)
    coverage = branch_coverage(layers)
    output = composite_gaussian_and_mesh(
        layers["gaussian_rgb"],
        layers["gaussian_alpha"],
        layers["gaussian_depth"],
        layers["mesh_rgb"],
        layers["mesh_alpha"],
        layers["mesh_depth"],
    )
    assert bool(torch.isfinite(output.rgb).all())
    assert bool(torch.isfinite(output.alpha).all())
    assert bool(((output.alpha >= 0.0) & (output.alpha <= 1.0 + 1.0e-5)).all())
    depth_valid = torch.isfinite(output.depth) | (
        torch.isinf(output.depth) & (output.depth > 0)
    )
    assert bool(depth_valid.all())
    assert int(torch.isinf(output.depth).sum()) == coverage["both_invalid"]
    expected_front = coverage["mesh_front_both_valid"] + coverage["mesh_only"]
    assert int(output.mesh_in_front.sum()) == expected_front


def test_forbid_host_transfers_blocks_cpu_syncs_inside_the_guard() -> None:
    value = torch.tensor([1.0])
    with forbid_host_transfers():
        with pytest.raises(RuntimeError, match="forbidden"):
            value.item()
        with pytest.raises(RuntimeError, match="forbidden"):
            value.cpu()
        with pytest.raises(RuntimeError, match="forbidden"):
            value.tolist()
        _allowed = value + value  # device-side math stays legal
    assert value.item() == 1.0
    assert value.cpu().shape == (1,)


def _fake_result(*, iterations: int = PROTOCOL_MIN_ITERATIONS, allocation_delta: int = 0):
    case = {
        "batch": 8,
        "width": 128,
        "height": 128,
        "config_id": hybrid_config_id(8, 128, 128),
        "seed": case_seed(20260717, 8, 128, 128),
    }
    checks = {
        "outputs_gpu_resident": True,
        "rgb_finite": True,
        "no_measured_loop_allocation_growth": allocation_delta == 0,
        "meets_protocol_iteration_minimum": iterations >= PROTOCOL_MIN_ITERATIONS,
    }
    return assemble_case_result(
        case,
        environment_info={"git_commit": "deadbeef", "gpu_name": "fake", "host": "unit"},
        flags={"min_alpha": 1.0e-4, "depth_epsilon": 1.0e-4},
        warmup_iterations=10,
        measured_iterations=iterations,
        cold_start_ms=3.2,
        upload_seconds=0.01,
        gpu_samples_ms=[1.0 + 0.01 * index for index in range(iterations)],
        wall_loop_seconds=0.06,
        memory={
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
            "persistent_scene_bytes": 512,
            "reusable_workspace_bytes": 256,
            "temporary_allocation_delta_bytes": allocation_delta,
            "driver_process_memory_bytes": None,
        },
        coverage={"coverage_complete": True},
        checks=checks,
    )


def test_assemble_case_result_flattens_to_protocol_csv_row() -> None:
    result = _fake_result()
    assert result["status"]["pass"] is True
    assert result["status"]["verdict"] == "PASS"
    for key in ("mean", "std", "p50", "p95", "ci95"):
        assert result["timing"]["gpu_ms"][key] is not None
    row = flatten_result(result)
    assert row["config_id"] == "cfg-v1-hybrid-compositor-b8-128x128"
    assert row["git_commit"] == "deadbeef"
    assert row["implementation"] == "hybrid-compositor"
    assert row["scene_id"] == "synthetic-hybrid-layers"
    assert row["batch_size"] == 8
    assert row["measured_iterations"] == PROTOCOL_MIN_ITERATIONS
    assert row["mean_gpu_ms"] == result["timing"]["gpu_ms"]["mean"]
    assert row["peak_allocated_bytes"] == 1024
    assert row["temporary_allocation_delta_bytes"] == 0
    assert row["pass"] is True
    assert len(result["dataset"]["checksum_sha256"]) == 64


def test_assemble_case_result_fails_closed_on_bad_checks() -> None:
    short = _fake_result(iterations=PROTOCOL_MIN_ITERATIONS - 1)
    assert short["status"]["pass"] is False
    assert short["status"]["verdict"] == "FAIL"
    leaking = _fake_result(allocation_delta=4096)
    assert leaking["status"]["pass"] is False
