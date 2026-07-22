"""Run independent custom-renderer processes and verify timing stability."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

if __package__:
    from .renderer_options import resolve_covariance_epsilon
else:
    from renderer_options import resolve_covariance_epsilon


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="synthetic-small")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--depth-bucket-count", type=int, default=4096)
    parser.add_argument("--depth-bucket-group-size", type=int, default=64)
    parser.add_argument("--gaussian-support-sigma", type=float, default=2.0)
    parser.add_argument("--covariance-epsilon", type=float, default=None)
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument("--compact-projection-cache", action="store_true")
    parser.add_argument("--projection-cache", action="store_true")
    parser.add_argument("--max-relative-spread", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def build_run_command(
    args: argparse.Namespace,
    *,
    result_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "benchmarks/run_custom.py",
        "--scene",
        args.scene,
        "--batch",
        str(args.batch),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--tile-size",
        str(args.tile_size),
        "--depth-bucket-count",
        str(args.depth_bucket_count),
        "--depth-bucket-group-size",
        str(args.depth_bucket_group_size),
        "--gaussian-support-sigma",
        str(args.gaussian_support_sigma),
        "--covariance-epsilon",
        str(args.covariance_epsilon),
        "--rasterize-mode",
        args.rasterize_mode,
        "--authored-display-output",
        "--output",
        str(result_path),
    ]
    if args.compact_projection_cache:
        command.append("--compact-projection-cache")
    if args.projection_cache:
        command.append("--projection-cache")
    return command


def result_contract(
    result: dict[str, object],
    *,
    rasterize_mode: str,
    covariance_epsilon: float,
    expected_shape: tuple[int, int, int],
) -> dict[str, bool]:
    counters = result.get("measured_counters")
    counters_valid = isinstance(counters, dict)
    zero_overflow = bool(
        counters_valid
        and counters.get("visible_overflow") == 0
        and counters.get("intersection_overflow") == 0
    )
    actual_epsilon = result.get("covariance_epsilon")
    epsilon_matches = isinstance(actual_epsilon, (int, float)) and math.isclose(
        float(actual_epsilon),
        covariance_epsilon,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    capacity = result.get("capacity_adaptation")
    parameters = capacity.get("parameters") if isinstance(capacity, dict) else None
    backend_epsilon = (
        parameters.get("covariance_epsilon")
        if isinstance(parameters, dict)
        else None
    )
    backend_epsilon_matches = isinstance(
        backend_epsilon,
        (int, float),
    ) and math.isclose(
        float(backend_epsilon),
        covariance_epsilon,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    expected_output_shapes = {
        "rgb": [*expected_shape, 3],
        "depth": [*expected_shape, 1],
        "alpha": [*expected_shape, 1],
        "semantic_id": [*expected_shape, 1],
    }
    expected_output_dtypes = {
        "rgb": "torch.float32",
        "depth": "torch.float32",
        "alpha": "torch.float32",
        "semantic_id": "torch.int64",
    }
    return {
        "schema_version_matches": (
            result.get("schema_version") == "custom-cuda-benchmark/v1"
        ),
        "child_pass": result.get("pass") is True,
        "rasterize_mode_matches": (
            result.get("rasterize_mode") == rasterize_mode
        ),
        "covariance_epsilon_matches": epsilon_matches,
        "backend_rasterize_mode_matches": bool(
            isinstance(parameters, dict)
            and parameters.get("rasterize_mode") == rasterize_mode
        ),
        "backend_covariance_epsilon_matches": backend_epsilon_matches,
        "outputs_gpu_resident": result.get("outputs_gpu_resident") is True,
        "outputs_contract_valid": result.get("outputs_contract_valid") is True,
        "output_names_match": bool(
            result.get("output_names_match") is True
            and result.get("outputs")
            == ["rgb", "depth", "alpha", "semantic_id"]
        ),
        "output_shapes_match": bool(
            result.get("output_shapes_match") is True
            and result.get("output_shapes") == expected_output_shapes
        ),
        "output_dtypes_match": bool(
            result.get("output_dtypes_match") is True
            and result.get("output_dtypes") == expected_output_dtypes
        ),
        "outputs_contiguous": result.get("outputs_contiguous") is True,
        "alpha_in_range": result.get("alpha_in_range") is True,
        "zero_allocation_delta": result.get("allocation_delta_bytes") == 0,
        "zero_overflow": zero_overflow,
    }


def main() -> None:
    args = parse_args()
    if args.runs < 3:
        raise ValueError("Stability evidence requires at least three independent runs.")
    args.covariance_epsilon = resolve_covariance_epsilon(
        args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        ray_gaussian_evaluation=False,
        compact_projection_cache=args.compact_projection_cache,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    results = []
    run_contracts = []
    for index in range(args.runs):
        result_path = args.output / f"run-{index + 1:02d}.json"
        command = build_run_command(args, result_path=result_path)
        subprocess.check_call(
            command,
            stdout=subprocess.DEVNULL,
        )
        result = json.loads(result_path.read_text())
        contract = result_contract(
            result,
            rasterize_mode=args.rasterize_mode,
            covariance_epsilon=args.covariance_epsilon,
            expected_shape=(args.batch, args.height, args.width),
        )
        if not all(contract.values()):
            raise RuntimeError(
                f"Stability child run {index + 1} failed its contract: "
                f"{contract}."
            )
        results.append(result)
        run_contracts.append(contract)

    means_ms = [float(result["frame_ms"]["mean"]) for result in results]
    throughput = [float(result["images_per_second"]) for result in results]
    mean_ms = statistics.fmean(means_ms)
    relative_spread = (max(means_ms) - min(means_ms)) / mean_ms
    summary = {
        "schema_version": "custom-stability/v2",
        "scene": args.scene,
        "batch": args.batch,
        "width": args.width,
        "height": args.height,
        "tile_size": args.tile_size,
        "depth_bucket_count": args.depth_bucket_count,
        "depth_bucket_group_size": args.depth_bucket_group_size,
        "gaussian_support_sigma": args.gaussian_support_sigma,
        "covariance_epsilon": args.covariance_epsilon,
        "rasterize_mode": args.rasterize_mode,
        "compact_projection_cache": args.compact_projection_cache,
        "projection_cache": args.projection_cache,
        "independent_process_runs": args.runs,
        "warmup_iterations": args.warmup,
        "measured_iterations_per_run": args.iterations,
        "frame_ms_means": means_ms,
        "images_per_second": throughput,
        "mean_frame_ms": mean_ms,
        "relative_spread": relative_spread,
        "max_relative_spread": args.max_relative_spread,
        "run_contracts": run_contracts,
        "zero_allocation_delta": all(
            int(result["allocation_delta_bytes"]) == 0 for result in results
        ),
        "zero_overflow": all(
            int(result["measured_counters"]["visible_overflow"]) == 0
            and int(result["measured_counters"]["intersection_overflow"]) == 0
            for result in results
        ),
        "driver_memory_bytes": [
            result["driver_process_memory_bytes"] for result in results
        ],
        "driver_memory_scope": [
            result["driver_memory_scope"] for result in results
        ],
    }
    summary["pass"] = bool(
        relative_spread <= args.max_relative_spread
        and summary["zero_allocation_delta"]
        and summary["zero_overflow"]
        and all(all(contract.values()) for contract in run_contracts)
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)
    print("CUSTOM_STABILITY_OK")


if __name__ == "__main__":
    main()
