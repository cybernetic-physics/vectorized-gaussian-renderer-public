"""Profile the project-owned vectorized CUDA renderer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .renderer_options import (
        rasterize_mode_qualified,
        resolve_covariance_epsilon,
    )
else:
    from renderer_options import (
        rasterize_mode_qualified,
        resolve_covariance_epsilon,
    )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-id",
        default="custom-synth-shared-c64-n100k-128",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/profiles"),
    )
    parser.add_argument(
        "--scene",
        choices=sorted(SYNTHETIC_SCENES),
        default="synthetic-medium",
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--visible-capacity", type=int)
    parser.add_argument("--visible-fraction", type=float)
    parser.add_argument("--intersection-capacity", type=int)
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
    parser.add_argument("--materialize-projected-records", action="store_true")
    parser.add_argument("--max-workspace-gib", type=float)
    parser.add_argument("--projection-cache", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--linear-output", action="store_true")
    color_group.add_argument(
        "--authored-display-output",
        action="store_true",
    )
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--torch-profiler-iterations", type=int, default=4)
    return parser.parse_args()


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def environment() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "project_commit": os.environ.get(
            "SOURCE_GIT_COMMIT",
            command_output(["git", "rev-parse", "HEAD"], cwd=project_root),
        ),
        "project_branch": command_output(
            ["git", "branch", "--show-current"],
            cwd=project_root,
        ),
        "driver": command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(
        args.batch,
        args.width,
        args.height,
        args.warmup,
        args.iterations,
    ) <= 0:
        raise ValueError("Batch, resolution, warmup, and iterations must be positive.")
    if args.compact_projection_cache and (
        args.tile_size != 1
        or args.ray_gaussian_evaluation
        or args.deterministic
    ):
        raise ValueError(
            "--compact-projection-cache requires --tile-size 1, "
            "screen-space evaluation, and non-deterministic scheduling."
        )
    if args.rasterize_mode == "antialiased" and args.ray_gaussian_evaluation:
        raise ValueError(
            "--rasterize-mode antialiased is incompatible with "
            "--ray-gaussian-evaluation."
        )
    if (
        args.materialize_projected_records
        and not args.compact_projection_cache
    ):
        raise ValueError(
            "--materialize-projected-records requires "
            "--compact-projection-cache."
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
    if (
        args.depth_bucket_count <= 0
        or args.depth_bucket_group_size <= 0
    ):
        raise ValueError("Depth bucket count and group size must be positive.")
    if args.direct_intersections_per_camera_gaussian <= 0:
        raise ValueError(
            "--direct-intersections-per-camera-gaussian must be positive."
        )
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
    args.config_id = rasterize_mode_qualified(
        args.config_id,
        args.rasterize_mode,
    )
    if args.torch_profiler_iterations <= 0:
        raise ValueError("--torch-profiler-iterations must be positive.")
    color_contract = resolve_output_color_contract(
        linear_output=args.linear_output,
        authored_display_output=args.authored_display_output,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    gaussian_count, seed = SYNTHETIC_SCENES[args.scene]
    visible_capacity = args.visible_capacity or math.ceil(
        args.batch * gaussian_count * visible_fraction
    )
    resolution_scale = (
        args.width * args.height / float(128 * 128)
    )
    if args.compact_projection_cache:
        intersections_per_visible = None
        intersection_capacity = (
            args.intersection_capacity
            or math.ceil(
                args.batch
                * gaussian_count
                * args.direct_intersections_per_camera_gaussian
                * resolution_scale
            )
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
        intersection_capacity = (
            args.intersection_capacity
            or math.ceil(
                visible_capacity * intersections_per_visible
            )
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
    scene_id = 101
    scene_ids = torch.full(
        (args.batch,),
        scene_id,
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
        materialize_projected_records=(
            args.materialize_projected_records
        ),
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
    service.initialize(stage=None, device=device)
    service.load_scene(
        scene_id,
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
    service.synchronize()

    for _ in range(args.warmup - 1):
        service.render(
            cameras.viewmats,
            cameras.intrinsics,
            scene_ids,
            outputs=outputs,
        )
    service.synchronize()
    warmup_counters = backend.check_capacity(synchronize=False)

    starts = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(args.iterations)
    ]
    ends = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(args.iterations)
    ]
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    wall_start = time.perf_counter()
    torch.cuda.nvtx.range_push("custom_profile_window")
    for index, (start, end) in enumerate(
        zip(starts, ends, strict=True)
    ):
        torch.cuda.nvtx.range_push(f"custom_iteration_{index}")
        start.record()
        service.render(
            cameras.viewmats,
            cameras.intrinsics,
            scene_ids,
            outputs=outputs,
        )
        end.record()
        torch.cuda.nvtx.range_pop()
    service.synchronize()
    torch.cuda.nvtx.range_pop()
    wall_seconds = time.perf_counter() - wall_start
    frame_samples_ms = [
        float(start.elapsed_time(end))
        for start, end in zip(starts, ends, strict=True)
    ]
    measured_counters = backend.check_capacity(synchronize=False)
    allocated_after = torch.cuda.memory_allocated()

    profiler_trace = None
    profiler_summary: list[dict[str, Any]] = []
    if args.torch_profiler:
        profiler_trace = (
            args.output_dir
            / f"{args.config_id}_torch_trace.json"
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=False,
        ) as profiler:
            for index in range(args.torch_profiler_iterations):
                torch.cuda.nvtx.range_push(
                    f"custom_torch_profiler_iteration_{index}"
                )
                service.render(
                    cameras.viewmats,
                    cameras.intrinsics,
                    scene_ids,
                    outputs=outputs,
                )
                torch.cuda.nvtx.range_pop()
                profiler.step()
            service.synchronize()
        profiler.export_chrome_trace(str(profiler_trace))
        for item in profiler.key_averages()[:80]:
            cuda_total = getattr(
                item,
                "cuda_time_total",
                getattr(item, "device_time_total", 0.0),
            )
            self_cuda_total = getattr(
                item,
                "self_cuda_time_total",
                getattr(item, "self_device_time_total", 0.0),
            )
            cuda_memory = getattr(
                item,
                "cuda_memory_usage",
                getattr(item, "device_memory_usage", 0),
            )
            profiler_summary.append(
                {
                    "key": item.key,
                    "cpu_time_total_us": item.cpu_time_total,
                    "cuda_time_total_us": cuda_total,
                    "self_cuda_time_total_us": self_cuda_total,
                    "count": item.count,
                    "cpu_memory_usage_bytes": item.cpu_memory_usage,
                    "cuda_memory_usage_bytes": cuda_memory,
                }
            )

    frame_stats = stats(frame_samples_ms)
    mean_ms = frame_stats["mean"]
    result = {
        "schema_version": "custom-profile/v1",
        "config_id": args.config_id,
        "implementation": "custom-cuda",
        "pipeline": backend.pipeline_name,
        "scene": synthetic_scene_manifest(
            args.scene,
            gaussian_count,
            seed,
        ),
        "camera_manifest": cameras.manifest,
        "batch": args.batch,
        "width": args.width,
        "height": args.height,
        "outputs": ["rgb", "depth", "alpha", "semantic_id"],
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
        "depth_bucket_count": args.depth_bucket_count,
        "depth_bucket_group_size": args.depth_bucket_group_size,
        "compact_projection_cache": args.compact_projection_cache,
        "direct_intersection_mode": args.compact_projection_cache,
        "projected_record_reuse": backend.capacity_stats[
            "projected_record_reuse"
        ],
        "projection_cache": {
            **backend.projection_cache_stats,
            "scope": backend.projection_cache_scope,
        },
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "frame_ms": {
            **frame_stats,
            "samples": frame_samples_ms,
        },
        "wall_frame_ms": wall_seconds * 1_000.0 / args.iterations,
        "images_per_second": args.batch * 1_000.0 / mean_ms,
        "megapixels_per_second": (
            args.batch * args.width * args.height / 1.0e6
        )
        * 1_000.0
        / mean_ms,
        "visible_capacity": visible_capacity,
        "final_visible_capacity": backend.current_visible_capacity,
        "visible_storage_capacity": backend.visible_storage_capacity,
        "visible_capacity_role": backend.capacity_stats[
            "visible_capacity_role"
        ],
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
        "warmup_counters": warmup_counters,
        "measured_counters": measured_counters,
        "scene_bytes": backend.scene_bytes,
        "workspace_bytes": backend.workspace_bytes,
        "workspace_bytes_by_tensor": backend.workspace_bytes_by_tensor,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocation_delta_bytes": allocated_after - allocated_before,
        "runtime_camera_loop": False,
        "batched_native_submissions_per_render": 1,
        "output_shapes": {
            name: list(tensor.shape)
            for name, tensor in outputs.items()
        },
        "output_devices": {
            name: str(tensor.device)
            for name, tensor in outputs.items()
        },
        "environment": environment(),
        "torch_profiler_trace": (
            str(profiler_trace)
            if profiler_trace is not None
            else None
        ),
        "torch_profiler_summary": profiler_summary,
    }
    json_path = args.output_dir / f"{args.config_id}.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        args.output_dir / "profile_runs.csv",
        {
            "config_id": args.config_id,
            "implementation": "custom-cuda",
            "scene": args.scene,
            "batch": args.batch,
            "gaussians": gaussian_count,
            "width": args.width,
            "height": args.height,
            "mean_ms": frame_stats["mean"],
            "p50_ms": frame_stats["p50"],
            "p95_ms": frame_stats["p95"],
            "ci95_ms": frame_stats["ci95"],
            "images_per_second": result["images_per_second"],
            "megapixels_per_second": result["megapixels_per_second"],
            "peak_allocated_bytes": result["peak_allocated_bytes"],
            "peak_reserved_bytes": result["peak_reserved_bytes"],
            "visible_gaussians": measured_counters["visible_gaussians"],
            "tile_intersections": measured_counters["tile_intersections"],
            "allocation_delta_bytes": result["allocation_delta_bytes"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    service.shutdown()


if __name__ == "__main__":
    main()
