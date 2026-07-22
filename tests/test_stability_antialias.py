from argparse import Namespace

from benchmarks.run_stability import (
    build_run_command,
    resolve_covariance_epsilon,
    result_contract,
)


def stability_args(tmp_path) -> Namespace:
    return Namespace(
        scene="synthetic-small",
        batch=8,
        width=128,
        height=128,
        warmup=3,
        iterations=5,
        tile_size=1,
        depth_bucket_count=32,
        depth_bucket_group_size=8,
        gaussian_support_sigma=3.0,
        covariance_epsilon=0.3,
        rasterize_mode="antialiased",
        compact_projection_cache=True,
        projection_cache=True,
        output=tmp_path,
    )


def test_stability_antialias_default_remains_effective_for_compact_path() -> None:
    assert resolve_covariance_epsilon(
        None,
        rasterize_mode="antialiased",
        ray_gaussian_evaluation=False,
        compact_projection_cache=True,
    ) == 0.3


def test_stability_command_forwards_mode_and_resolved_epsilon(tmp_path) -> None:
    args = stability_args(tmp_path)
    command = build_run_command(args, result_path=tmp_path / "run.json")

    assert command[command.index("--rasterize-mode") + 1] == "antialiased"
    assert command[command.index("--covariance-epsilon") + 1] == "0.3"
    assert "--compact-projection-cache" in command
    assert "--projection-cache" in command


def test_stability_result_contract_fails_closed_on_ignored_mode() -> None:
    result = {
        "pass": True,
        "rasterize_mode": "classic",
        "covariance_epsilon": 0.3,
        "outputs_gpu_resident": True,
        "outputs_contract_valid": True,
        "allocation_delta_bytes": 0,
        "capacity_adaptation": {
            "parameters": {
                "rasterize_mode": "classic",
                "covariance_epsilon": 0.3,
            }
        },
        "measured_counters": {
            "visible_overflow": 0,
            "intersection_overflow": 0,
        },
    }

    contract = result_contract(
        result,
        rasterize_mode="antialiased",
        covariance_epsilon=0.3,
        expected_shape=(8, 128, 128),
    )

    assert not contract["rasterize_mode_matches"]
    assert not contract["backend_rasterize_mode_matches"]
    assert not all(contract.values())


def test_stability_result_contract_verifies_backend_parameters() -> None:
    result = {
        "schema_version": "custom-cuda-benchmark/v1",
        "pass": True,
        "rasterize_mode": "antialiased",
        "covariance_epsilon": 0.3,
        "outputs_gpu_resident": True,
        "outputs_contract_valid": True,
        "outputs": ["rgb", "depth", "alpha", "semantic_id"],
        "output_names_match": True,
        "output_shapes": {
            "rgb": [8, 128, 128, 3],
            "depth": [8, 128, 128, 1],
            "alpha": [8, 128, 128, 1],
            "semantic_id": [8, 128, 128, 1],
        },
        "output_shapes_match": True,
        "output_dtypes": {
            "rgb": "torch.float32",
            "depth": "torch.float32",
            "alpha": "torch.float32",
            "semantic_id": "torch.int64",
        },
        "output_dtypes_match": True,
        "outputs_contiguous": True,
        "alpha_in_range": True,
        "allocation_delta_bytes": 0,
        "capacity_adaptation": {
            "parameters": {
                "rasterize_mode": "antialiased",
                "covariance_epsilon": 0.3,
            }
        },
        "measured_counters": {
            "visible_overflow": 0,
            "intersection_overflow": 0,
        },
    }

    contract = result_contract(
        result,
        rasterize_mode="antialiased",
        covariance_epsilon=0.3,
        expected_shape=(8, 128, 128),
    )

    assert all(contract.values())


def test_stability_result_contract_does_not_trust_aggregate_child_flag() -> None:
    result = {
        "schema_version": "custom-cuda-benchmark/v1",
        "pass": True,
        "rasterize_mode": "antialiased",
        "covariance_epsilon": 0.3,
        "outputs_gpu_resident": True,
        "outputs_contract_valid": True,
        "outputs": ["rgb", "depth", "alpha", "semantic_id"],
        "output_names_match": True,
        "output_shapes": {
            "rgb": [8, 128, 128, 3],
            "depth": [8, 128, 128, 1],
            "alpha": [8, 128, 128, 1],
            "semantic_id": [8, 128, 128, 1],
        },
        "output_shapes_match": True,
        "output_dtypes": {
            "rgb": "torch.float32",
            "depth": "torch.float32",
            "alpha": "torch.float32",
            "semantic_id": "torch.int64",
        },
        "output_dtypes_match": True,
        "outputs_contiguous": True,
        "alpha_in_range": False,
        "allocation_delta_bytes": 0,
        "capacity_adaptation": {
            "parameters": {
                "rasterize_mode": "antialiased",
                "covariance_epsilon": 0.3,
            }
        },
        "measured_counters": {
            "visible_overflow": 0,
            "intersection_overflow": 0,
        },
    }

    contract = result_contract(
        result,
        rasterize_mode="antialiased",
        covariance_epsilon=0.3,
        expected_shape=(8, 128, 128),
    )

    assert contract["child_pass"]
    assert not contract["alpha_in_range"]
    assert not all(contract.values())
