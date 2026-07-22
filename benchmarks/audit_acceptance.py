"""Recompute direct acceptance verdicts from producer evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


THRESHOLDS = {
    "throughput_ratio": 5.0,
    "memory_ratio": 0.80,
    "rgb_psnr_db": 40.0,
    "rgb_ssim": 0.995,
    "lpips": 0.01,
    "alpha_mae": 0.005,
    "depth_relative_error": 0.01,
    "semantic_agreement": 0.999,
    "reproducibility_relative_range": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--isaac-integration", type=Path, required=True)
    parser.add_argument(
        "--producer-root",
        type=Path,
        required=True,
        help="Root used to resolve result paths embedded in comparison JSON.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def criterion(
    *,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "verdict": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


def aggregate_metric(
    fidelity: dict[str, Any],
    name: str,
    *,
    direction: str,
) -> float | None:
    metric = fidelity["aggregate"]["metrics"][name]
    key = "min" if direction == "minimum" else "max"
    value = metric.get(key)
    return (
        float(value)
        if isinstance(value, (int, float))
        else None
    )


def main() -> None:
    args = parse_args()
    comparison = load_json(args.comparison)
    fidelity = load_json(args.fidelity)
    stability = load_json(args.stability)
    isaac = load_json(args.isaac_integration)

    rows = comparison.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Comparison evidence contains no matched rows.")
    speedups = []
    memory_ratios = []
    custom_results = []
    for row in rows:
        ovrtx_ips = float(row["ovrtx"]["images_per_second"])
        custom_ips = float(row["custom"]["images_per_second"])
        ovrtx_memory = int(
            row["ovrtx"]["driver_process_memory_bytes"]
        )
        custom_memory = int(
            row["custom"]["driver_process_memory_bytes"]
        )
        speedups.append(custom_ips / ovrtx_ips)
        memory_ratios.append(custom_memory / ovrtx_memory)
        custom_path = resolve_path(
            args.producer_root,
            str(row["custom"]["result_json"]),
        )
        custom_results.append(load_json(custom_path))

    minimum_speedup = min(speedups)
    maximum_memory_ratio = max(memory_ratios)
    single_camera_rows = [
        result
        for result in custom_results
        if int(result["batch"]) == 1
    ]
    outputs_gpu_resident = all(
        result.get("outputs_gpu_resident") is True
        and set(result.get("outputs", []))
        == {"rgb", "depth", "alpha", "semantic_id"}
        for result in custom_results
    )
    custom_kernel_active = all(
        result.get("native_extension_loaded") is True
        and result.get("runtime_camera_loop") is False
        and result.get("batched_native_submissions_per_render") == 1
        and str(result.get("pipeline", "")).startswith(
            (
                "project-",
                "compact-project-",
                "packed-distinct-scenes-",
            )
        )
        for result in custom_results
    )

    rgb_psnr = aggregate_metric(
        fidelity,
        "rgb_psnr_db",
        direction="minimum",
    )
    rgb_ssim = aggregate_metric(
        fidelity,
        "rgb_ssim",
        direction="minimum",
    )
    lpips = aggregate_metric(
        fidelity,
        "lpips",
        direction="maximum",
    )
    alpha_mae = aggregate_metric(
        fidelity,
        "alpha_mae",
        direction="maximum",
    )
    depth_relative_error = aggregate_metric(
        fidelity,
        "depth_rel_error",
        direction="maximum",
    )
    semantic_agreement = aggregate_metric(
        fidelity,
        "semantic_agreement",
        direction="minimum",
    )

    independent_ips = stability.get("images_per_second")
    if not isinstance(independent_ips, list) or len(independent_ips) < 3:
        independent_ips = []
    independent_values = [
        float(value)
        for value in independent_ips
    ]
    reproducibility_range = (
        (
            max(independent_values) - min(independent_values)
        )
        / statistics.fmean(independent_values)
        if independent_values
        else None
    )

    renderer = isaac.get("renderer", {})
    fabric = isaac.get("fabric_scene_delegate", {})
    isaac_outputs = renderer.get("outputs", {})
    deterministic_values = renderer.get(
        "deterministic_bitwise_equal",
        {},
    )
    isaac_pass = (
        isaac.get("pass") is True
        and isaac.get("isaac_lab_required") is False
        and fabric.get("enabled") is True
        and fabric.get("extension_loaded") is True
        and renderer.get("backend") == "CustomCudaBackend"
        and renderer.get("runtime_camera_loop") is False
        and renderer.get("batched_native_submissions_per_render") == 1
        and renderer.get("render_does_not_advance_physics") is True
        and bool(isaac_outputs)
        and all(
            str(output.get("device", "")).startswith("cuda")
            for output in isaac_outputs.values()
        )
        and bool(deterministic_values)
        and all(bool(value) for value in deterministic_values.values())
    )

    criteria = {
        "visual_fidelity": criterion(
            passed=fidelity["aggregate"].get("pass") is True,
            evidence={
                "fidelity_report": str(args.fidelity),
            },
        ),
        "throughput_at_least_5x": criterion(
            passed=minimum_speedup
            >= THRESHOLDS["throughput_ratio"],
            evidence={
                "minimum_matched_speedup": minimum_speedup,
                "threshold": THRESHOLDS["throughput_ratio"],
                "matched_rows": len(rows),
            },
        ),
        "peak_memory_at_most_80_percent": criterion(
            passed=maximum_memory_ratio
            <= THRESHOLDS["memory_ratio"],
            evidence={
                "maximum_matched_driver_memory_ratio": (
                    maximum_memory_ratio
                ),
                "threshold": THRESHOLDS["memory_ratio"],
            },
        ),
        "rgb_psnr": criterion(
            passed=rgb_psnr is not None
            and rgb_psnr >= THRESHOLDS["rgb_psnr_db"],
            evidence={
                "minimum_db": rgb_psnr,
                "threshold_db": THRESHOLDS["rgb_psnr_db"],
            },
        ),
        "rgb_ssim": criterion(
            passed=rgb_ssim is not None
            and rgb_ssim >= THRESHOLDS["rgb_ssim"],
            evidence={
                "minimum": rgb_ssim,
                "threshold": THRESHOLDS["rgb_ssim"],
            },
        ),
        "lpips": criterion(
            passed=lpips is not None
            and lpips <= THRESHOLDS["lpips"],
            evidence={
                "maximum": lpips,
                "threshold": THRESHOLDS["lpips"],
            },
        ),
        "alpha_mae": criterion(
            passed=alpha_mae is not None
            and alpha_mae <= THRESHOLDS["alpha_mae"],
            evidence={
                "maximum": alpha_mae,
                "threshold": THRESHOLDS["alpha_mae"],
            },
        ),
        "depth_relative_error": criterion(
            passed=depth_relative_error is not None
            and depth_relative_error
            <= THRESHOLDS["depth_relative_error"],
            evidence={
                "maximum": depth_relative_error,
                "threshold": THRESHOLDS[
                    "depth_relative_error"
                ],
            },
        ),
        "semantic_agreement": criterion(
            passed=semantic_agreement is not None
            and semantic_agreement
            >= THRESHOLDS["semantic_agreement"],
            evidence={
                "minimum": semantic_agreement,
                "threshold": THRESHOLDS[
                    "semantic_agreement"
                ],
            },
        ),
        "three_run_reproducibility": criterion(
            passed=(
                stability.get("pass") is True
                and reproducibility_range is not None
                and reproducibility_range
                <= THRESHOLDS[
                    "reproducibility_relative_range"
                ]
            ),
            evidence={
                "independent_images_per_second": independent_values,
                "relative_range": reproducibility_range,
                "threshold": THRESHOLDS[
                    "reproducibility_relative_range"
                ],
            },
        ),
        "single_camera_reported": criterion(
            passed=bool(single_camera_rows),
            evidence={
                "single_camera_images_per_second": [
                    result["images_per_second"]
                    for result in single_camera_rows
                ],
            },
        ),
        "gpu_resident_all_outputs": criterion(
            passed=outputs_gpu_resident,
            evidence={
                "matched_custom_result_count": len(
                    custom_results
                ),
            },
        ),
        "project_owned_gpu_kernels": criterion(
            passed=custom_kernel_active,
            evidence={
                "pipelines": sorted(
                    {
                        str(result.get("pipeline"))
                        for result in custom_results
                    }
                ),
            },
        ),
        "headless_isaac_sim_kit_extension": criterion(
            passed=isaac_pass,
            evidence={
                "isaac_integration": str(
                    args.isaac_integration
                ),
                "runtime": isaac.get("runtime"),
                "fabric_scene_delegate": fabric,
            },
        ),
    }
    overall_pass = all(
        item["verdict"] == "PASS"
        for item in criteria.values()
    )
    result = {
        "schema_version": "acceptance-audit/v1",
        "overall_verdict": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "thresholds": THRESHOLDS,
        "inputs": {
            "comparison": str(args.comparison),
            "fidelity": str(args.fidelity),
            "stability": str(args.stability),
            "isaac_integration": str(
                args.isaac_integration
            ),
            "producer_root": str(args.producer_root),
        },
        "criteria": criteria,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
