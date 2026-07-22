"""Benchmark packed distinct Gaussian scenes in one vectorized submission."""

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
    stats,
    synthetic_exact_intersections_per_visible,
    synthetic_scene_manifest,
    synthetic_scene_tensors,
)


SYNTHETIC_SCENES = {
    "synthetic-small": (10_000, 42),
    "synthetic-medium": (100_000, 43),
}
SEMANTIC_STRIDE = 1_000_000


def require_passing_result(result: dict[str, Any]) -> None:
    """Make a persisted failed benchmark observable to shell automation."""
    if result.get("pass") is not True:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        choices=sorted(SYNTHETIC_SCENES),
        default="synthetic-small",
    )
    parser.add_argument("--scene-count", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--visible-capacity", type=int)
    parser.add_argument("--visible-fraction", type=float)
    parser.add_argument("--intersection-capacity", type=int)
    parser.add_argument("--intersections-per-visible", type=float)
    parser.add_argument(
        "--direct-intersections-per-camera-gaussian",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--gaussian-support-sigma",
        type=float,
        default=3.0,
        help="Use 3.0 for the OVRTX acceptance contract; 3.33 is the gsplat control.",
    )
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
    parser.add_argument("--materialize-projected-records", action="store_true")
    parser.add_argument("--max-workspace-gib", type=float)
    parser.add_argument("--projection-cache", action="store_true")
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--linear-output", action="store_true")
    color_group.add_argument(
        "--authored-display-output",
        action="store_true",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


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
        return int(output.splitlines()[0].strip()) * 1024 * 1024, ("isolated-gpu-total")
    except Exception:
        return None, "unavailable"


def load_scene(
    service: RendererService,
    *,
    scene_id: int,
    scene: dict[str, torch.Tensor],
) -> None:
    service.load_scene(
        scene_id,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if (
        min(
            args.scene_count,
            args.batch,
            args.width,
            args.height,
            args.warmup,
            args.iterations,
        )
        <= 0
    ):
        raise ValueError(
            "All count, batch, resolution, and iteration values must be positive."
        )
    if args.scene_count > args.batch:
        raise ValueError("--scene-count cannot exceed --batch.")
    if args.compact_projection_cache and (
        args.tile_size != 1 or args.ray_gaussian_evaluation or args.deterministic
    ):
        raise ValueError(
            "--compact-projection-cache requires --tile-size 1, "
            "screen-space evaluation, and non-deterministic scheduling."
        )
    if args.materialize_projected_records and not args.compact_projection_cache:
        raise ValueError(
            "--materialize-projected-records requires --compact-projection-cache."
        )
    if args.rasterize_mode == "antialiased" and args.ray_gaussian_evaluation:
        raise ValueError(
            "--rasterize-mode antialiased is incompatible with --ray-gaussian-evaluation."
        )
    if args.max_workspace_gib is not None and args.max_workspace_gib <= 0:
        raise ValueError("--max-workspace-gib must be positive.")
    visible_fraction = (
        args.visible_fraction
        if args.visible_fraction is not None
        else (0.10 if args.ray_gaussian_evaluation else 0.065)
    )
    if visible_fraction <= 0:
        raise ValueError("--visible-fraction must be positive.")
    if args.gaussian_support_sigma <= 0:
        raise ValueError("--gaussian-support-sigma must be positive.")
    if args.depth_bucket_count <= 0 or args.depth_bucket_group_size <= 0:
        raise ValueError("Depth bucket count and group size must be positive.")
    if args.direct_intersections_per_camera_gaussian <= 0:
        raise ValueError("--direct-intersections-per-camera-gaussian must be positive.")
    if (
        not math.isfinite(args.semantic_min_alpha)
        or not 0 <= args.semantic_min_alpha <= 1
    ):
        raise ValueError("--semantic-min-alpha must be in [0, 1].")
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

    device = torch.device("cuda")
    gaussian_count, base_seed = SYNTHETIC_SCENES[args.scene]
    visible_capacity = args.visible_capacity or math.ceil(
        args.batch * gaussian_count * visible_fraction
    )
    resolution_scale = args.width * args.height / float(128 * 128)
    if args.compact_projection_cache:
        intersections_per_visible = None
        intersection_capacity = args.intersection_capacity or math.ceil(
            args.batch
            * gaussian_count
            * args.direct_intersections_per_camera_gaussian
            * resolution_scale
        )
    else:
        intersections_per_visible = args.intersections_per_visible or math.ceil(
            (
                synthetic_exact_intersections_per_visible(
                    width=args.width,
                    height=args.height,
                    tile_size=args.tile_size,
                )
                if args.ray_gaussian_evaluation
                else 4.0 * (max(args.width, args.height) / 128.0) ** 1.2
            )
        )
        intersection_capacity = args.intersection_capacity or math.ceil(
            visible_capacity * intersections_per_visible
        )

    scene_ids_host = [17 + index * 884 for index in range(args.scene_count)]
    scenes: list[dict[str, torch.Tensor]] = []
    scene_manifests = []
    for index, scene_id in enumerate(scene_ids_host):
        seed = base_seed + index * 1009
        scene = synthetic_scene_tensors(
            gaussian_count,
            seed=seed,
            device=device,
        )
        if index:
            scene["colors"] = torch.roll(
                scene["colors"],
                shifts=index % 3,
                dims=1,
            ).contiguous()
        scene["semantic_ids"] = (
            scene["semantic_ids"].to(torch.int64) + index * SEMANTIC_STRIDE
        ).contiguous()
        scenes.append(scene)
        manifest = synthetic_scene_manifest(
            f"{args.scene}-distinct-{index}",
            gaussian_count,
            seed,
        )
        manifest["scene_id"] = scene_id
        manifest["semantic_offset"] = index * SEMANTIC_STRIDE
        scene_manifests.append(manifest)

    camera_data = camera_bundle(
        args.batch,
        args.width,
        args.height,
        device=device,
    )
    assignment_indices = (
        torch.arange(args.batch, device=device, dtype=torch.int64) % args.scene_count
    )
    packed_scene_ids = torch.tensor(
        scene_ids_host,
        device=device,
        dtype=torch.int64,
    )
    scene_ids = packed_scene_ids.index_select(0, assignment_indices)
    environment_transforms = torch.eye(
        4,
        device=device,
        dtype=torch.float32,
    ).repeat(args.batch, 1, 1)
    environment_transforms[:, 0, 3] = (
        assignment_indices.to(torch.float32) - (args.scene_count - 1) * 0.5
    ) * 0.025
    environment_transforms[:, 1, 3] = (torch.arange(args.batch, device=device) % 7).to(
        torch.float32
    ) * 0.002
    environment_transforms = environment_transforms.contiguous()

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
        materialize_projected_records=args.materialize_projected_records,
        enable_projection_cache=args.projection_cache,
        output_srgb=bool(color_contract["output_srgb"]),
        deterministic=args.deterministic,
        max_workspace_bytes=(
            None
            if args.max_workspace_gib is None
            else int(args.max_workspace_gib * 1024**3)
        ),
    )
    service = RendererService(
        backend,
        height=args.height,
        width=args.width,
        max_views=args.batch,
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    service.initialize(stage=None, device=device)
    for scene_id, scene in zip(scene_ids_host, scenes, strict=True):
        load_scene(service, scene_id=scene_id, scene=scene)
    service.update_scene_transforms(scene_ids, environment_transforms)
    outputs = service.render(
        camera_data.viewmats,
        camera_data.intrinsics,
        scene_ids,
    )
    service.synchronize()
    cold_start_ms = (time.perf_counter() - cold_start) * 1_000.0

    foreground = (
        outputs["alpha"][..., 0] > 0
        if args.semantic_min_alpha == 0
        else outputs["alpha"][..., 0] >= args.semantic_min_alpha
    )
    semantic = outputs["semantic_id"][..., 0]
    expected_offsets = assignment_indices.view(args.batch, 1, 1) * SEMANTIC_STRIDE
    invalid_semantics = foreground & (
        (semantic < expected_offsets) | (semantic >= expected_offsets + SEMANTIC_STRIDE)
    )
    if bool(torch.any(invalid_semantics).item()):
        raise AssertionError(
            "A camera received semantic IDs from a different packed scene."
        )
    if not bool(torch.isfinite(outputs["rgb"]).all().item()):
        raise AssertionError("RGB output contains non-finite values.")
    valid_depth = foreground & torch.isfinite(outputs["depth"][..., 0])
    if int(torch.count_nonzero(valid_depth).item()) == 0:
        raise AssertionError("No finite foreground depth values were produced.")

    for _ in range(args.warmup - 1):
        service.render(
            camera_data.viewmats,
            camera_data.intrinsics,
            scene_ids,
            outputs=outputs,
        )
    service.synchronize()
    warmup_counters = backend.check_capacity(synchronize=False)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    wall_start = time.perf_counter()
    for start, end in zip(starts, ends, strict=True):
        start.record()
        service.render(
            camera_data.viewmats,
            camera_data.intrinsics,
            scene_ids,
            outputs=outputs,
        )
        end.record()
    service.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    frame_samples_ms = [
        float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)
    ]
    measured_counters = backend.check_capacity(synchronize=False)
    allocated_after = torch.cuda.memory_allocated()
    driver_memory_bytes, driver_memory_scope = gpu_memory_snapshot()
    alpha = outputs["alpha"][..., 0]
    depth = outputs["depth"][..., 0]
    semantic = outputs["semantic_id"][..., 0]
    foreground = alpha > 0
    background = ~foreground
    outputs_gpu_resident = all(tensor.is_cuda for tensor in outputs.values())
    rgb_finite = bool(torch.isfinite(outputs["rgb"]).all().item())
    alpha_finite = bool(torch.isfinite(alpha).all().item())
    foreground_pixel_count = int(torch.count_nonzero(foreground).item())
    foreground_depth_finite = foreground_pixel_count > 0 and bool(
        torch.isfinite(depth[foreground]).all().item()
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
        foreground if args.semantic_min_alpha == 0 else alpha >= args.semantic_min_alpha
    )
    semantic_background = ~semantic_foreground
    semantic_foreground_pixel_count = int(
        torch.count_nonzero(semantic_foreground).item()
    )
    foreground_semantic_valid = semantic_foreground_pixel_count > 0 and bool(
        (semantic[semantic_foreground] >= 0).all().item()
    )
    background_semantic_valid = bool((semantic[semantic_background] == -1).all().item())
    background_positive_infinity_pixels = int(
        torch.count_nonzero(background & torch.isinf(depth) & (depth > 0)).item()
    )
    outputs_contract_valid = (
        rgb_finite
        and alpha_finite
        and foreground_depth_finite
        and background_depth_valid
        and foreground_semantic_valid
        and background_semantic_valid
    )

    if (
        measured_counters["visible_overflow"]
        or measured_counters["intersection_overflow"]
    ):
        raise RuntimeError(f"Packed-scene capacity overflow: {measured_counters}.")

    frame_stats = stats(frame_samples_ms)
    mean_ms = frame_stats["mean"]
    result: dict[str, Any] = {
        "schema_version": "custom-multi-scene-benchmark/v1",
        "implementation": "custom-cuda",
        "pipeline": f"packed-distinct-scenes-{backend.pipeline_name}",
        "native_extension_loaded": True,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "scene_family": args.scene,
        "scene_count": args.scene_count,
        "gaussians_per_scene": gaussian_count,
        "packed_gaussians": gaussian_count * args.scene_count,
        "scenes": scene_manifests,
        "scene_assignment": "round_robin_on_cuda",
        "distinct_scene_assignment": args.scene_count > 1,
        "environment_transforms": "per_camera_gpu_tensor",
        "camera_manifest": camera_data.manifest,
        "batch": args.batch,
        "width": args.width,
        "height": args.height,
        "outputs": ["rgb", "depth", "alpha", "semantic_id"],
        "output_devices": {
            name: str(tensor.device) for name, tensor in outputs.items()
        },
        "output_shapes": {name: list(tensor.shape) for name, tensor in outputs.items()},
        "color_space": color_contract["color_space"],
        "color_transform": color_contract["color_transform"],
        "deterministic": args.deterministic,
        "gaussian_support_sigma": args.gaussian_support_sigma,
        "semantic_min_alpha": args.semantic_min_alpha,
        "covariance_epsilon": args.covariance_epsilon,
        "rasterize_mode": args.rasterize_mode,
        "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
        "gaussian_evaluation_model": (
            "exact_camera_ray_3d_mahalanobis"
            if args.ray_gaussian_evaluation
            else "screen_space_2d_conic"
        ),
        "tile_size": args.tile_size,
        "bin_tile_size": backend.bin_tile_size,
        "depth_bucket_count": args.depth_bucket_count,
        "depth_bucket_group_size": args.depth_bucket_group_size,
        "compact_projection_cache": args.compact_projection_cache,
        "direct_intersection_mode": args.compact_projection_cache,
        "projected_record_reuse": backend.capacity_stats["projected_record_reuse"],
        "projection_cache": {
            **backend.projection_cache_stats,
            "scope": backend.projection_cache_scope,
        },
        "cold_start_ms": cold_start_ms,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "frame_ms": {
            **frame_stats,
            "samples": frame_samples_ms,
        },
        "wall_frame_ms": wall_seconds * 1_000.0 / args.iterations,
        "images_per_second": args.batch * 1_000.0 / mean_ms,
        "megapixels_per_second": (args.batch * args.width * args.height / 1.0e6)
        * 1_000.0
        / mean_ms,
        "visible_capacity": visible_capacity,
        "final_visible_capacity": backend.current_visible_capacity,
        "visible_storage_capacity": backend.visible_storage_capacity,
        "visible_capacity_role": backend.capacity_stats["visible_capacity_role"],
        "intersection_capacity": intersection_capacity,
        "final_intersection_capacity": (backend.current_intersection_capacity),
        "capacity_adaptation": backend.capacity_stats,
        "capacity_planner": {
            "visible_fraction": visible_fraction,
            "intersections_per_visible": intersections_per_visible,
            "direct_intersections_per_camera_gaussian": (
                args.direct_intersections_per_camera_gaussian
                if args.compact_projection_cache
                else None
            ),
            "resolution_scale_from_128x128": resolution_scale,
            "source": (
                "explicit"
                if args.visible_capacity is not None
                or args.intersection_capacity is not None
                or args.intersections_per_visible is not None
                else "synthetic-analytical-capacity-margin"
            ),
        },
        "warmup_counters": warmup_counters,
        "measured_counters": measured_counters,
        "scene_bytes": backend.scene_bytes,
        "workspace_bytes": backend.workspace_bytes,
        "workspace_bytes_by_tensor": backend.workspace_bytes_by_tensor,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocation_delta_bytes": allocated_after - allocated_before,
        "driver_process_memory_bytes": driver_memory_bytes,
        "driver_memory_scope": driver_memory_scope,
        "semantic_scene_isolation_pass": True,
        "runtime_camera_loop": False,
        "batched_native_submissions_per_render": 1,
        "outputs_gpu_resident": outputs_gpu_resident,
        "outputs_finite": outputs_contract_valid,
        "outputs_contract_valid": outputs_contract_valid,
        "output_validity": {
            "rgb_finite": rgb_finite,
            "alpha_finite": alpha_finite,
            "foreground_pixel_count": foreground_pixel_count,
            "foreground_depth_finite": foreground_depth_finite,
            "background_depth_valid": background_depth_valid,
            "background_positive_infinity_pixels": (
                background_positive_infinity_pixels
            ),
            "foreground_semantic_valid": foreground_semantic_valid,
            "background_semantic_valid": background_semantic_valid,
            "semantic_foreground_pixel_count": (semantic_foreground_pixel_count),
            "depth_background_policy": ("finite_or_positive_infinity_when_alpha_zero"),
        },
        "pass": (
            measured_counters["visible_overflow"] == 0
            and measured_counters["intersection_overflow"] == 0
            and allocated_after == allocated_before
            and driver_memory_bytes is not None
            and outputs_gpu_resident
            and outputs_contract_valid
        ),
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    service.shutdown()
    require_passing_result(result)


if __name__ == "__main__":
    main()
