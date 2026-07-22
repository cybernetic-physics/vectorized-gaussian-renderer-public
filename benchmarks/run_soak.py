"""Run a synchronized long-duration soak of the custom CUDA renderer."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .renderer_options import resolve_covariance_epsilon
else:
    from renderer_options import resolve_covariance_epsilon

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle,
    resolve_output_color_contract,
    synthetic_exact_intersections_per_visible,
    synthetic_scene_manifest,
    synthetic_scene_tensors,
)


SYNTHETIC_SCENES = {
    "synthetic-small": (10_000, 42),
    "synthetic-medium": (100_000, 43),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        choices=sorted(SYNTHETIC_SCENES),
        default="synthetic-medium",
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--renders-per-check", type=int, default=100)
    parser.add_argument("--visible-fraction", type=float)
    parser.add_argument("--intersections-per-visible", type=float)
    parser.add_argument(
        "--direct-intersections-per-camera-gaussian",
        type=float,
        default=6.0,
    )
    parser.add_argument("--gaussian-support-sigma", type=float, default=3.0)
    parser.add_argument("--semantic-min-alpha", type=float, default=0.01)
    parser.add_argument("--covariance-epsilon", type=float, default=None)
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument("--ray-gaussian-evaluation", action="store_true")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--depth-bucket-count", type=int, default=4096)
    parser.add_argument("--depth-bucket-group-size", type=int, default=64)
    parser.add_argument("--compact-projection-cache", action="store_true")
    parser.add_argument("--projection-cache", action="store_true")
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--linear-output", action="store_true")
    color_group.add_argument(
        "--authored-display-output",
        action="store_true",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def gpu_memory_snapshot() -> tuple[int | None, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        pid = str(__import__("os").getpid())
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 2 and fields[0] == pid:
                return int(fields[1]) * 1024 * 1024, "compute-process"
    except Exception:
        pass
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return (
            int(output.splitlines()[0].strip()) * 1024 * 1024,
            "isolated-gpu-total",
        )
    except Exception:
        return None, "unavailable"


def output_contract(
    outputs: dict[str, torch.Tensor],
    *,
    semantic_min_alpha: float,
    expected_shape: tuple[int, int, int],
) -> dict[str, Any]:
    expected_specs = {
        "rgb": ((*expected_shape, 3), torch.float32),
        "depth": ((*expected_shape, 1), torch.float32),
        "alpha": ((*expected_shape, 1), torch.float32),
        "semantic_id": ((*expected_shape, 1), torch.int64),
    }
    output_names_match = set(outputs) == set(expected_specs)
    if not output_names_match:
        return {
            "valid": False,
            "output_names_match": False,
            "expected_output_names": sorted(expected_specs),
            "actual_output_names": sorted(outputs),
        }
    shapes_match = all(
        tuple(outputs[name].shape) == shape
        for name, (shape, _dtype) in expected_specs.items()
    )
    dtypes_match = all(
        outputs[name].dtype == dtype
        for name, (_shape, dtype) in expected_specs.items()
    )
    contiguous = all(tensor.is_contiguous() for tensor in outputs.values())
    gpu_resident = all(tensor.is_cuda for tensor in outputs.values())
    alpha = outputs["alpha"][..., 0]
    depth = outputs["depth"][..., 0]
    semantic = outputs["semantic_id"][..., 0]
    foreground = alpha > 0
    background = ~foreground
    foreground_count = int(torch.count_nonzero(foreground).item())
    rgb_finite = bool(torch.isfinite(outputs["rgb"]).all().item())
    alpha_finite = bool(torch.isfinite(alpha).all().item())
    alpha_in_range = bool(
        ((alpha >= -1.0e-6) & (alpha <= 1.0 + 1.0e-6)).all().item()
    )
    foreground_depth_finite = (
        foreground_count > 0
        and bool(torch.isfinite(depth[foreground]).all().item())
    )
    background_depth_valid = bool(
        (
            torch.isfinite(depth[background])
            | (torch.isinf(depth[background]) & (depth[background] > 0))
        )
        .all()
        .item()
    )
    semantic_foreground = (
        foreground
        if semantic_min_alpha == 0
        else alpha >= semantic_min_alpha
    )
    semantic_background = ~semantic_foreground
    semantic_foreground_count = int(
        torch.count_nonzero(semantic_foreground).item()
    )
    foreground_semantic_valid = (
        semantic_foreground_count > 0
        and bool(
            (semantic[semantic_foreground] >= 0).all().item()
        )
    )
    background_semantic_valid = bool(
        (semantic[semantic_background] == -1).all().item()
    )
    valid = (
        output_names_match
        and shapes_match
        and dtypes_match
        and contiguous
        and gpu_resident
        and rgb_finite
        and alpha_finite
        and alpha_in_range
        and foreground_depth_finite
        and background_depth_valid
        and foreground_semantic_valid
        and background_semantic_valid
    )
    return {
        "valid": valid,
        "output_names_match": output_names_match,
        "shapes_match": shapes_match,
        "dtypes_match": dtypes_match,
        "contiguous": contiguous,
        "output_shapes": {
            name: list(tensor.shape) for name, tensor in outputs.items()
        },
        "output_dtypes": {
            name: str(tensor.dtype) for name, tensor in outputs.items()
        },
        "expected_shape": list(expected_shape),
        "outputs_gpu_resident": gpu_resident,
        "output_devices": {
            name: str(tensor.device) for name, tensor in outputs.items()
        },
        "rgb_finite": rgb_finite,
        "alpha_finite": alpha_finite,
        "alpha_in_range": alpha_in_range,
        "foreground_pixel_count": foreground_count,
        "foreground_depth_finite": foreground_depth_finite,
        "background_depth_valid": background_depth_valid,
        "foreground_semantic_valid": foreground_semantic_valid,
        "background_semantic_valid": background_semantic_valid,
        "semantic_foreground_pixel_count": semantic_foreground_count,
        "semantic_min_alpha": semantic_min_alpha,
        "depth_background_policy": (
            "finite_or_positive_infinity_when_alpha_zero"
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(
        args.batch,
        args.width,
        args.height,
        args.warmup,
        args.renders_per_check,
    ) <= 0:
        raise ValueError("Batch, resolution, warmup, and chunk size must be positive.")
    if args.compact_projection_cache and (
        args.tile_size != 1
        or args.ray_gaussian_evaluation
    ):
        raise ValueError(
            "--compact-projection-cache requires --tile-size 1 and "
            "screen-space evaluation."
        )
    visible_fraction = (
        args.visible_fraction
        if args.visible_fraction is not None
        else (0.10 if args.ray_gaussian_evaluation else 0.065)
    )
    if (
        args.duration_seconds <= 0
        or visible_fraction <= 0
        or args.gaussian_support_sigma <= 0
        or args.direct_intersections_per_camera_gaussian <= 0
        or args.depth_bucket_count <= 0
        or args.depth_bucket_group_size <= 0
        or not 0 <= args.semantic_min_alpha <= 1
    ):
        raise ValueError(
            "Duration, capacity fractions, and support sigma must be positive."
        )
    args.covariance_epsilon = resolve_covariance_epsilon(
        args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        compact_projection_cache=args.compact_projection_cache,
    )
    color_contract = resolve_output_color_contract(
        linear_output=args.linear_output,
        authored_display_output=args.authored_display_output,
    )

    device = torch.device("cuda:0")
    gaussian_count, seed = SYNTHETIC_SCENES[args.scene]
    visible_capacity = math.ceil(
        args.batch * gaussian_count * visible_fraction
    )
    resolution_scale = (
        args.width * args.height / float(128 * 128)
    )
    if args.compact_projection_cache:
        intersections_per_visible = None
        intersection_capacity = math.ceil(
            args.batch
            * gaussian_count
            * args.direct_intersections_per_camera_gaussian
            * resolution_scale
        )
    else:
        intersections_per_visible = (
            args.intersections_per_visible
            or math.ceil(
                (
                    synthetic_exact_intersections_per_visible(
                        width=args.width,
                        height=args.height,
                        tile_size=args.tile_size,
                    )
                    if args.ray_gaussian_evaluation
                    else 4.0
                    * (
                        max(args.width, args.height) / 128.0
                    )
                    ** 1.2
                )
            )
        )
        intersection_capacity = math.ceil(
            visible_capacity * intersections_per_visible
        )
    scene = synthetic_scene_tensors(
        gaussian_count,
        seed=seed,
        device=device,
    )
    cameras = camera_bundle(
        args.batch,
        args.width,
        args.height,
        device=device,
    )
    scene_ids = torch.full(
        (args.batch,),
        101,
        device=device,
        dtype=torch.int64,
    )
    backend = CustomCudaBackend(
        max_visible_records=visible_capacity,
        max_intersections=intersection_capacity,
        gaussian_support_sigma=args.gaussian_support_sigma,
        covariance_epsilon=args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        semantic_min_alpha=args.semantic_min_alpha,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        tile_size=args.tile_size,
        depth_bucket_count=args.depth_bucket_count,
        depth_bucket_group_size=args.depth_bucket_group_size,
        compact_projection_cache=args.compact_projection_cache,
        enable_projection_cache=args.projection_cache,
        output_srgb=bool(color_contract["output_srgb"]),
    )
    if backend.rasterize_mode != args.rasterize_mode or not math.isclose(
        backend.covariance_epsilon,
        args.covariance_epsilon,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise AssertionError(
            "Soak renderer configuration mismatch: requested "
            f"rasterize_mode={args.rasterize_mode!r}, "
            f"covariance_epsilon={args.covariance_epsilon}; got "
            f"rasterize_mode={backend.rasterize_mode!r}, "
            f"covariance_epsilon={backend.covariance_epsilon}."
        )
    service = RendererService(
        backend,
        height=args.height,
        width=args.width,
        max_views=args.batch,
    )
    service.initialize(stage=None, device=device)
    service.load_scene(
        101,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )
    outputs = service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
    )
    for _ in range(args.warmup - 1):
        service.render(
            cameras.viewmats,
            cameras.intrinsics,
            scene_ids,
            outputs=outputs,
        )
    service.synchronize()
    initial_counters = backend.check_capacity(synchronize=False)
    initial_zero_overflow = bool(
        initial_counters["visible_overflow"] == 0
        and initial_counters["intersection_overflow"] == 0
    )
    initial_contract = output_contract(
        outputs,
        semantic_min_alpha=args.semantic_min_alpha,
        expected_shape=(args.batch, args.height, args.width),
    )
    if not initial_contract["valid"]:
        raise RuntimeError(
            f"Initial output contract failed: {initial_contract}."
        )
    if not initial_zero_overflow:
        raise RuntimeError(
            f"Warmup overflowed renderer capacity: {initial_counters}."
        )

    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    render_calls = 0
    checks = 0
    all_outputs_valid = True
    zero_overflow = initial_zero_overflow
    last_contract = initial_contract
    while time.perf_counter() - start < args.duration_seconds:
        for _ in range(args.renders_per_check):
            service.render(
                cameras.viewmats,
                cameras.intrinsics,
                scene_ids,
                outputs=outputs,
            )
        service.synchronize()
        render_calls += args.renders_per_check
        checks += 1
        counters = backend.check_capacity(synchronize=False)
        zero_overflow = zero_overflow and (
            counters["visible_overflow"] == 0
            and counters["intersection_overflow"] == 0
        )
        last_contract = output_contract(
            outputs,
            semantic_min_alpha=args.semantic_min_alpha,
            expected_shape=(args.batch, args.height, args.width),
        )
        all_outputs_valid = (
            all_outputs_valid
            and bool(last_contract["valid"])
        )
        if not zero_overflow or not all_outputs_valid:
            break

    service.synchronize()
    duration_seconds = time.perf_counter() - start
    allocated_after = torch.cuda.memory_allocated()
    driver_memory_bytes, driver_memory_scope = gpu_memory_snapshot()
    final_counters = backend.check_capacity(synchronize=False)
    zero_overflow = bool(
        zero_overflow
        and final_counters["visible_overflow"] == 0
        and final_counters["intersection_overflow"] == 0
    )
    pass_result = (
        duration_seconds >= args.duration_seconds
        and all_outputs_valid
        and zero_overflow
        and allocated_after == allocated_before
        and driver_memory_bytes is not None
    )
    result = {
        "schema_version": "custom-cuda-soak/v2",
        "implementation": "custom-cuda",
        "pipeline": backend.pipeline_name,
        "scene": synthetic_scene_manifest(
            args.scene,
            gaussian_count,
            seed,
        ),
        "batch": args.batch,
        "width": args.width,
        "height": args.height,
        "gaussian_support_sigma": args.gaussian_support_sigma,
        "semantic_min_alpha": args.semantic_min_alpha,
        "covariance_epsilon": args.covariance_epsilon,
        "rasterize_mode": args.rasterize_mode,
        "renderer_configuration_matches": True,
        "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
        "gaussian_evaluation_model": (
            "exact_camera_ray_3d_mahalanobis"
            if args.ray_gaussian_evaluation
            else "screen_space_2d_conic"
        ),
        "tile_size": args.tile_size,
        "depth_bucket_count": args.depth_bucket_count,
        "depth_bucket_group_size": args.depth_bucket_group_size,
        "compact_projection_cache": args.compact_projection_cache,
        "direct_intersection_mode": args.compact_projection_cache,
        "projection_cache": {
            **backend.projection_cache_stats,
            "scope": backend.projection_cache_scope,
        },
        "color_space": color_contract["color_space"],
        "color_transform": color_contract["color_transform"],
        "warmup_iterations": args.warmup,
        "requested_duration_seconds": args.duration_seconds,
        "duration_seconds": duration_seconds,
        "renders_per_check": args.renders_per_check,
        "synchronized_checks": checks,
        "batched_render_calls": render_calls,
        "rendered_images": render_calls * args.batch,
        "images_per_second": (
            render_calls * args.batch / duration_seconds
        ),
        "mean_batch_wall_ms": (
            duration_seconds * 1_000.0 / render_calls
        ),
        "visible_capacity": visible_capacity,
        "final_visible_capacity": backend.current_visible_capacity,
        "visible_storage_capacity": backend.visible_storage_capacity,
        "visible_capacity_role": (
            "diagnostic-and-capacity-planning-only"
            if args.compact_projection_cache
            else "materialized-projected-record-capacity"
        ),
        "intersection_capacity": intersection_capacity,
        "final_intersection_capacity": (
            backend.current_intersection_capacity
        ),
        "capacity_adaptation": backend.capacity_stats,
        "visible_fraction": visible_fraction,
        "intersections_per_visible": intersections_per_visible,
        "direct_intersections_per_camera_gaussian": (
            args.direct_intersections_per_camera_gaussian
            if args.compact_projection_cache
            else None
        ),
        "resolution_scale_from_128x128": resolution_scale,
        "final_counters": final_counters,
        "initial_counters": initial_counters,
        "initial_output_contract": initial_contract,
        "final_output_contract": last_contract,
        "all_output_checks_valid": all_outputs_valid,
        "zero_overflow": zero_overflow,
        "synchronized_after_each_chunk": True,
        "no_cuda_errors": True,
        "allocation_delta_bytes": allocated_after - allocated_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "driver_process_memory_bytes": driver_memory_bytes,
        "driver_memory_scope": driver_memory_scope,
        "outputs_gpu_resident": all(
            tensor.is_cuda
            for tensor in outputs.values()
        ),
        "pass": pass_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("CUSTOM_CUDA_SOAK_OK", json.dumps(result, sort_keys=True))
    service.shutdown()
    if not pass_result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
