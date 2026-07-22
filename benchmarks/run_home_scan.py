"""Render the full Home Scan LOD0 scene with the project CUDA backend."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

if __package__:
    from .renderer_options import resolve_covariance_epsilon
else:
    from renderer_options import resolve_covariance_epsilon

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    compact_semantic_ids,
    axis_aligned_exterior_camera_bundle,
    deterministic_semantic_ids,
    deterministic_semantic_manifest,
    file_sha256,
    resolve_output_color_contract,
    spatial_semantic_ids,
    spatial_semantic_manifest,
    stats,
)
from isaacsim_gaussian_renderer.ply_loader import (
    canonicalize_3dgs_scene,
    load_ply_to_gaussians,
)
from isaacsim_gaussian_renderer.scene import GaussianScene


DATASET_SPECS = {
    "home-scan-lod0": {
        "default_path": "/workspace/datasets/home-scan-lod0.ply",
        "count": 21_497_908,
        "sha256": ("29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"),
        "manifest": "datasets/home-scan-lod0.manifest.json",
        "source_sh_degree": 0,
        "camera_bounds_quantiles": None,
    },
    "voxel51-train-30000": {
        "default_path": (
            "/workspace/datasets/public-gaussian/"
            "Voxel51_gaussian_splatting/FO_dataset/train/"
            "point_cloud/iteration_30000/point_cloud.ply"
        ),
        "count": 1_074_761,
        "sha256": ("a4c1906ce0256f5cb2255aa078d18b468dc01d8604f6eda1aaf22b235147a7a1"),
        "manifest": "datasets/voxel51-train-30000.manifest.json",
        "source_sh_degree": 3,
        "camera_bounds_quantiles": (0.1, 0.9),
    },
}


def require_passing_result(result: dict[str, Any]) -> None:
    """Make a persisted failed benchmark observable to shell automation."""
    if result.get("pass") is not True:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-id",
        choices=sorted(DATASET_SPECS),
        default="home-scan-lod0",
    )
    parser.add_argument(
        "--path",
        type=Path,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--renders-per-check", type=int, default=100)
    parser.add_argument("--visible-fraction", type=float, default=1.0)
    parser.add_argument(
        "--intersections-per-visible",
        type=float,
        default=1.25,
    )
    parser.add_argument("--visible-capacity", type=int)
    parser.add_argument("--intersection-capacity", type=int)
    parser.add_argument(
        "--direct-visible-gaussians-per-view",
        type=float,
        default=150_000.0,
    )
    parser.add_argument(
        "--direct-intersections-per-view-at-128",
        type=float,
        default=170_000.0,
    )
    parser.add_argument("--focal-scale", type=float, default=0.9)
    parser.add_argument("--fit-margin", type=float, default=1.15)
    parser.add_argument("--gaussian-support-sigma", type=float, default=3.0)
    parser.add_argument("--covariance-epsilon", type=float, default=None)
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument("--semantic-min-alpha", type=float, default=0.01)
    parser.add_argument(
        "--semantic-scheme",
        choices=("index-modulo", "spatial-grid"),
        default="index-modulo",
    )
    parser.add_argument(
        "--semantic-grid",
        type=lambda value: tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        ),
        default=(2, 2, 2),
    )
    parser.add_argument("--ray-gaussian-evaluation", action="store_true")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--depth-bucket-count", type=int, default=4096)
    parser.add_argument("--depth-bucket-group-size", type=int, default=64)
    parser.add_argument("--compact-projection-cache", action="store_true")
    parser.add_argument("--materialize-projected-records", action="store_true")
    parser.add_argument("--max-workspace-gib", type=float)
    parser.add_argument("--tight-depth-range", action="store_true")
    parser.add_argument("--projection-cache", action="store_true")
    parser.add_argument(
        "--invalidate-projection-cache-before-each-render",
        action="store_true",
        help=(
            "Keep projection caching enabled but explicitly invalidate it before every warmup and measured render."
        ),
    )
    parser.add_argument("--culling-diagnostics", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--linear-output", action="store_true")
    color_group.add_argument(
        "--authored-display-output",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/home-scan/custom-b1-128.json"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    return parser.parse_args()


def driver_memory_bytes() -> tuple[int | None, str]:
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
            fields = [item.strip() for item in line.split(",")]
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


def project_commit() -> str:
    """Resolve the project revision even when a benchmark copy has no Git metadata."""
    override = os.environ.get("SOURCE_GIT_COMMIT", "").strip()
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def output_contract(
    outputs: dict[str, torch.Tensor],
    *,
    semantic_min_alpha: float,
) -> dict[str, Any]:
    alpha = outputs["alpha"][..., 0]
    depth = outputs["depth"][..., 0]
    semantic = outputs["semantic_id"][..., 0]
    foreground = alpha > 0
    background = ~foreground
    foreground_pixel_count = int(torch.count_nonzero(foreground).item())
    rgb_finite = bool(torch.isfinite(outputs["rgb"]).all().item())
    alpha_finite = bool(torch.isfinite(alpha).all().item())
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
        alpha > 0 if semantic_min_alpha == 0 else alpha >= semantic_min_alpha
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
    valid = (
        rgb_finite
        and alpha_finite
        and foreground_depth_finite
        and background_depth_valid
        and foreground_semantic_valid
        and background_semantic_valid
    )
    return {
        "valid": valid,
        "rgb_finite": rgb_finite,
        "alpha_finite": alpha_finite,
        "foreground_pixel_count": foreground_pixel_count,
        "foreground_depth_finite": foreground_depth_finite,
        "background_depth_valid": background_depth_valid,
        "background_positive_infinity_pixels": (background_positive_infinity_pixels),
        "foreground_semantic_valid": foreground_semantic_valid,
        "background_semantic_valid": background_semantic_valid,
        "semantic_min_alpha": semantic_min_alpha,
        "semantic_foreground_pixel_count": (semantic_foreground_pixel_count),
        "depth_background_policy": ("finite_or_positive_infinity_when_alpha_zero"),
    }


def save_artifacts(
    artifact_dir: Path,
    outputs: dict[str, torch.Tensor],
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rgb = outputs["rgb"][0].detach().cpu().numpy()
    alpha = outputs["alpha"][0, ..., 0].detach().cpu().numpy()
    depth = outputs["depth"][0, ..., 0].detach().cpu().numpy()
    semantic = outputs["semantic_id"][0, ..., 0].detach().cpu().numpy()
    rgb_path = artifact_dir / "rgb-0000.png"
    alpha_path = artifact_dir / "alpha-0000.png"
    output_path = artifact_dir / "output-0000.npz"
    Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(rgb_path)
    Image.fromarray(np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(
        alpha_path
    )
    np.savez_compressed(
        output_path,
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        semantic=semantic,
        valid_depth=np.isfinite(depth) & (depth > 0),
    )
    return {
        "rgb": str(rgb_path),
        "alpha": str(alpha_path),
        "output": str(output_path),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    dataset_spec = DATASET_SPECS[args.dataset_id]
    dataset_path = (
        args.path if args.path is not None else Path(str(dataset_spec["default_path"]))
    )
    if (
        min(
            args.batch,
            args.width,
            args.height,
            args.warmup,
            args.iterations,
        )
        <= 0
    ):
        raise ValueError("Batch, resolution, warmup, and iterations must be positive.")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive.")
    if args.renders_per_check <= 0:
        raise ValueError("--renders-per-check must be positive.")
    if args.compact_projection_cache and (
        args.tile_size != 1 or args.ray_gaussian_evaluation or args.deterministic
    ):
        raise ValueError(
            "--compact-projection-cache requires --tile-size 1, "
            "screen-space evaluation, and non-deterministic scheduling."
        )
    if args.rasterize_mode == "antialiased" and args.ray_gaussian_evaluation:
        raise ValueError(
            "--rasterize-mode antialiased is incompatible with --ray-gaussian-evaluation."
        )
    if args.materialize_projected_records and not args.compact_projection_cache:
        raise ValueError(
            "--materialize-projected-records requires --compact-projection-cache."
        )
    if args.max_workspace_gib is not None and args.max_workspace_gib <= 0:
        raise ValueError("--max-workspace-gib must be positive.")
    if (
        args.invalidate_projection_cache_before_each_render
        and not args.projection_cache
    ):
        raise ValueError(
            "--invalidate-projection-cache-before-each-render requires --projection-cache."
        )
    if not 0 < args.visible_fraction <= 1:
        raise ValueError("--visible-fraction must be in (0, 1].")
    if args.intersections_per_visible <= 0:
        raise ValueError("--intersections-per-visible must be positive.")
    if (
        args.direct_visible_gaussians_per_view <= 0
        or args.direct_intersections_per_view_at_128 <= 0
    ):
        raise ValueError("Direct per-view capacity estimates must be positive.")
    if (
        args.focal_scale <= 0
        or args.fit_margin <= 0
        or args.gaussian_support_sigma <= 0
    ):
        raise ValueError("Focal scale and fit margin must be positive.")
    args.covariance_epsilon = resolve_covariance_epsilon(
        args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        compact_projection_cache=args.compact_projection_cache,
    )
    if (
        not math.isfinite(args.semantic_min_alpha)
        or args.semantic_min_alpha < 0
        or args.semantic_min_alpha > 1
    ):
        raise ValueError("--semantic-min-alpha must be finite and in [0, 1].")
    if len(args.semantic_grid) != 3 or min(args.semantic_grid) <= 0:
        raise ValueError("--semantic-grid must contain three positive dimensions.")
    if (
        args.tile_size <= 0
        or args.tile_size > 16
        or args.tile_size & (args.tile_size - 1)
    ):
        raise ValueError("--tile-size must be a power of two in [1, 16].")
    if args.depth_bucket_count <= 0:
        raise ValueError("--depth-bucket-count must be positive.")
    if args.depth_bucket_group_size <= 0:
        raise ValueError("--depth-bucket-group-size must be positive.")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    color_contract = resolve_output_color_contract(
        linear_output=args.linear_output,
        authored_display_output=args.authored_display_output,
    )

    checksum = file_sha256(dataset_path)
    expected_sha256 = str(dataset_spec["sha256"])
    if checksum != expected_sha256:
        raise ValueError(
            f"{args.dataset_id} SHA-256 mismatch: expected {expected_sha256}, got {checksum}."
        )

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    host_load_start = time.perf_counter()
    raw_scene = load_ply_to_gaussians(dataset_path)
    host_load_seconds = time.perf_counter() - host_load_start
    expected_count = int(dataset_spec["count"])
    if raw_scene.count != expected_count:
        raise ValueError(
            f"{args.dataset_id} count mismatch: expected {expected_count}, got {raw_scene.count}."
        )

    upload_start = time.perf_counter()
    canonical = canonicalize_3dgs_scene(raw_scene, device=device)
    if args.semantic_scheme == "spatial-grid":
        raw_semantic_ids = spatial_semantic_ids(
            canonical.means,
            grid=args.semantic_grid,
            dtype=torch.int64,
        )
        semantic_ids, occupied_source_ids = compact_semantic_ids(
            raw_semantic_ids,
            dtype=torch.int64,
        )
        semantic_sidecar_manifest = spatial_semantic_manifest(
            canonical.count,
            grid=args.semantic_grid,
            position_min=canonical.means.amin(dim=0),
            position_max=canonical.means.amax(dim=0),
            occupied_source_ids=occupied_source_ids,
        )
    else:
        semantic_ids = deterministic_semantic_ids(
            canonical.count,
            device=device,
            dtype=torch.int64,
        )
        semantic_sidecar_manifest = deterministic_semantic_manifest(canonical.count)
    semantic_group_count = int(semantic_ids.max().item()) + 1
    scene = GaussianScene(
        means=canonical.means,
        scales=canonical.scales,
        rotations=canonical.rotations,
        opacities=canonical.opacities,
        features=canonical.features[:, :3].contiguous(),
        semantic_ids=semantic_ids,
    )
    del canonical
    del raw_scene
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    upload_seconds = time.perf_counter() - upload_start

    position_min = scene.means.amin(dim=0)
    position_max = scene.means.amax(dim=0)
    extent = position_max - position_min
    camera_quantiles = dataset_spec["camera_bounds_quantiles"]
    if camera_quantiles is None:
        camera_min = position_min
        camera_max = position_max
        bounds_policy = "full-componentwise-min-max"
    else:
        quantiles = torch.tensor(
            camera_quantiles,
            device=device,
            dtype=scene.means.dtype,
        )
        camera_min, camera_max = torch.quantile(
            scene.means,
            quantiles,
            dim=0,
        )
        bounds_policy = (
            f"componentwise-quantile-{camera_quantiles[0]}-{camera_quantiles[1]}"
        )
    center = (camera_min + camera_max) * 0.5
    camera_extent = camera_max - camera_min
    bounding_radius = float((torch.linalg.vector_norm(camera_extent) * 0.5).item())
    cameras = axis_aligned_exterior_camera_bundle(
        center=center,
        bounding_radius=bounding_radius,
        batch_size=args.batch,
        width=args.width,
        height=args.height,
        focal_scale=args.focal_scale,
        fit_margin=args.fit_margin,
        camera_path=f"{args.dataset_id}-axis-aligned-exterior",
        bounds_policy=bounds_policy,
    )
    viewmats = cameras.viewmats
    intrinsics = cameras.intrinsics
    camera_manifest = cameras.manifest

    resolution_scale = args.width * args.height / float(128 * 128)
    if args.compact_projection_cache:
        visible_capacity = args.visible_capacity or math.ceil(
            args.batch * args.direct_visible_gaussians_per_view
        )
        intersection_capacity = args.intersection_capacity or math.ceil(
            args.batch * args.direct_intersections_per_view_at_128 * resolution_scale
        )
    else:
        visible_capacity = args.visible_capacity or math.ceil(
            args.batch * scene.count * args.visible_fraction
        )
        intersection_capacity = args.intersection_capacity or math.ceil(
            visible_capacity * args.intersections_per_visible
        )
    depth_margin = max(1.0e-3, bounding_radius * 1.0e-4)
    if args.tight_depth_range:
        near_plane = max(
            0.01,
            camera_manifest["distance"] - bounding_radius - depth_margin,
        )
        far_plane = camera_manifest["distance"] + bounding_radius + depth_margin
    else:
        near_plane = 0.01
        far_plane = max(
            100.0,
            camera_manifest["distance"] + bounding_radius * 2.0,
        )
    backend = CustomCudaBackend(
        max_visible_records=visible_capacity,
        max_intersections=intersection_capacity,
        near_plane=near_plane,
        far_plane=far_plane,
        gaussian_support_sigma=args.gaussian_support_sigma,
        covariance_epsilon=args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        semantic_min_alpha=args.semantic_min_alpha,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        tile_size=args.tile_size,
        depth_bucket_count=args.depth_bucket_count,
        depth_bucket_group_size=args.depth_bucket_group_size,
        compact_projection_cache=args.compact_projection_cache,
        materialize_projected_records=(args.materialize_projected_records),
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
        404,
        means=scene.means,
        scales=scene.scales,
        rotations=scene.rotations,
        opacities=scene.opacities,
        features=scene.features,
        semantic_ids=scene.semantic_ids,
    )
    scene_ids = torch.full(
        (args.batch,),
        404,
        device=device,
        dtype=torch.int64,
    )

    first_start = time.perf_counter()
    outputs = service.render(viewmats, intrinsics, scene_ids)
    service.synchronize()
    first_render_ms = (time.perf_counter() - first_start) * 1_000.0
    first_counters = backend.read_counters(synchronize=False)
    if first_counters["visible_overflow"] or first_counters["intersection_overflow"]:
        raise RuntimeError(
            "Home Scan capacity overflow during preflight: "
            f"{first_counters}; visible_capacity={visible_capacity}; "
            f"intersection_capacity={intersection_capacity}."
        )
    foreground = outputs["alpha"][..., 0] > 0
    if int(torch.count_nonzero(foreground).item()) == 0:
        raise AssertionError(f"{args.dataset_id} render produced no foreground pixels.")
    if not bool(torch.isfinite(outputs["rgb"]).all().item()):
        raise AssertionError(f"{args.dataset_id} RGB contains non-finite values.")
    culling_diagnostics = None
    if args.culling_diagnostics:
        if args.compact_projection_cache:
            culling_diagnostics = {
                "source": "direct-intersection-counters",
                "visible_gaussians": int(first_counters["visible_gaussians"]),
                "qualifying_pixel_intersections": int(
                    first_counters["tile_intersections"]
                ),
                "active_pixels": int(first_counters["active_tiles"]),
                "projected_record_radii_available": False,
                "note": (
                    "Compact mode consumes projected radii during intersection "
                    "emission and does not retain them in its projection cache."
                ),
            }
        else:
            visible_count = min(
                int(first_counters["visible_gaussians"]),
                backend.visible_storage_capacity,
            )
            radii = backend.workspace["visible_radii"][:visible_count]
            radius_thresholds = sorted(
                {
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    max(args.width, args.height),
                }
            )
            culling_diagnostics = {
                "source": "materialized-projected-record-radii",
                "visible_records": visible_count,
                "radius_x_max": int(radii[:, 0].max().item()),
                "radius_y_max": int(radii[:, 1].max().item()),
                "radius_x_mean": float(radii[:, 0].to(torch.float32).mean().item()),
                "radius_y_mean": float(radii[:, 1].to(torch.float32).mean().item()),
                "records_at_or_above_radius": {
                    str(threshold): int(
                        torch.count_nonzero(
                            (radii[:, 0] >= threshold) | (radii[:, 1] >= threshold)
                        ).item()
                    )
                    for threshold in radius_thresholds
                },
                "full_image_fallback_records": int(
                    torch.count_nonzero(
                        (radii[:, 0] >= max(args.width, args.height))
                        & (radii[:, 1] >= max(args.width, args.height))
                    ).item()
                ),
            }
            del radii
            torch.cuda.empty_cache()

    warmup_invalidations = 0
    for _ in range(args.warmup):
        if args.invalidate_projection_cache_before_each_render:
            backend.invalidate_projection_cache()
            warmup_invalidations += 1
        service.render(
            viewmats,
            intrinsics,
            scene_ids,
            outputs=outputs,
        )
    service.synchronize()
    warmup_counters = backend.check_capacity(synchronize=False)
    cache_stats_before_measurement = backend.projection_cache_stats

    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    synchronized_checks = 0
    all_output_checks_valid = True
    zero_overflow = True
    measured_invalidations = 0
    if args.duration_seconds is None:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        wall_start = time.perf_counter()
        for start, end in zip(starts, ends, strict=True):
            if args.invalidate_projection_cache_before_each_render:
                backend.invalidate_projection_cache()
                measured_invalidations += 1
            start.record()
            service.render(
                viewmats,
                intrinsics,
                scene_ids,
                outputs=outputs,
            )
            end.record()
        service.synchronize()
        wall_seconds = time.perf_counter() - wall_start
        frame_samples_ms = [
            float(start.elapsed_time(end))
            for start, end in zip(starts, ends, strict=True)
        ]
        render_calls = args.iterations
        synchronized_checks = 1
        measured_counters = backend.check_capacity(synchronize=False)
    else:
        frame_samples_ms = []
        render_calls = 0
        measured_counters = warmup_counters
        wall_start = time.perf_counter()
        while time.perf_counter() - wall_start < args.duration_seconds:
            chunk_start = torch.cuda.Event(enable_timing=True)
            chunk_end = torch.cuda.Event(enable_timing=True)
            chunk_start.record()
            for _ in range(args.renders_per_check):
                if args.invalidate_projection_cache_before_each_render:
                    backend.invalidate_projection_cache()
                    measured_invalidations += 1
                service.render(
                    viewmats,
                    intrinsics,
                    scene_ids,
                    outputs=outputs,
                )
            chunk_end.record()
            service.synchronize()
            frame_samples_ms.append(
                float(chunk_start.elapsed_time(chunk_end)) / args.renders_per_check
            )
            render_calls += args.renders_per_check
            synchronized_checks += 1
            measured_counters = backend.read_counters(synchronize=False)
            zero_overflow = zero_overflow and (
                measured_counters["visible_overflow"] == 0
                and measured_counters["intersection_overflow"] == 0
            )
            contract_check = output_contract(
                outputs,
                semantic_min_alpha=args.semantic_min_alpha,
            )
            all_output_checks_valid = all_output_checks_valid and bool(
                contract_check["valid"]
            )
            if not zero_overflow or not all_output_checks_valid:
                break
        wall_seconds = time.perf_counter() - wall_start
    cache_stats_after_measurement = backend.projection_cache_stats
    allocated_after = torch.cuda.memory_allocated()
    driver_bytes, driver_scope = driver_memory_bytes()
    outputs_gpu_resident = all(tensor.is_cuda for tensor in outputs.values())
    final_contract = output_contract(
        outputs,
        semantic_min_alpha=args.semantic_min_alpha,
    )
    outputs_contract_valid = bool(final_contract["valid"])
    all_output_checks_valid = all_output_checks_valid and outputs_contract_valid
    depth_diagnostics = None
    if args.tile_size == 1:
        effective_group_size = min(
            args.depth_bucket_group_size,
            args.depth_bucket_count,
        )
        group_count = math.ceil(args.depth_bucket_count / effective_group_size)
        if args.compact_projection_cache:
            nonzero_depth_buckets: list[int] = []
            visible_counts_by_group: list[int] | None = None
            bucket_population_source = "direct-pixel-optical-thickness"
        else:
            depth_bucket_counts = (
                backend.workspace["depth_bucket_counts"].detach().cpu().tolist()
            )
            nonzero_depth_buckets = [
                index for index, count in enumerate(depth_bucket_counts) if count > 0
            ]
            visible_counts_by_group = [
                int(
                    sum(
                        depth_bucket_counts[
                            group * effective_group_size : min(
                                args.depth_bucket_count,
                                (group + 1) * effective_group_size,
                            )
                        ]
                    )
                )
                for group in range(group_count)
            ]
            bucket_population_source = "projected-visible-record-counts"
        cutoff = backend.workspace["depth_cutoff"].reshape(
            args.batch,
            args.height,
            args.width,
        )
        foreground_cutoff = cutoff[outputs["alpha"][..., 0] > 0]
        cutoff_quantiles = (
            torch.quantile(
                foreground_cutoff.to(torch.float32),
                torch.tensor(
                    [0.0, 0.5, 0.95, 1.0],
                    device=foreground_cutoff.device,
                ),
            )
            .detach()
            .cpu()
            .tolist()
        )
        cutoff_group_histogram = (
            torch.bincount(
                torch.div(
                    foreground_cutoff,
                    effective_group_size,
                    rounding_mode="floor",
                ),
                minlength=group_count,
            )
            .detach()
            .cpu()
            .tolist()
        )
        depth_diagnostics = {
            "nonzero_depth_bucket_count": len(nonzero_depth_buckets),
            "first_nonzero_depth_bucket": (
                nonzero_depth_buckets[0] if nonzero_depth_buckets else None
            ),
            "last_nonzero_depth_bucket": (
                nonzero_depth_buckets[-1] if nonzero_depth_buckets else None
            ),
            "visible_counts_by_group": visible_counts_by_group,
            "bucket_population_source": bucket_population_source,
            "foreground_cutoff_bucket_quantiles": {
                "min": cutoff_quantiles[0],
                "p50": cutoff_quantiles[1],
                "p95": cutoff_quantiles[2],
                "max": cutoff_quantiles[3],
            },
            "foreground_cutoff_group_histogram": cutoff_group_histogram,
        }
    frame_stats = stats(frame_samples_ms)
    mean_ms = frame_stats["mean"]

    artifacts = (
        save_artifacts(args.artifact_dir, outputs)
        if args.artifact_dir is not None
        else None
    )
    result = {
        "schema_version": "real-scene-custom-benchmark/v1",
        "implementation": "custom-cuda",
        "pipeline": backend.pipeline_name,
        "environment": {
            "project_commit": project_commit(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "dataset": {
            "dataset_id": args.dataset_id,
            "path": str(dataset_path),
            "sha256": checksum,
            "gaussian_count": scene.count,
            "manifest": str(dataset_spec["manifest"]),
            "source_spherical_harmonics_degree": int(dataset_spec["source_sh_degree"]),
            "spherical_harmonics_degree": 0,
            "spherical_harmonics_policy": (
                "degree-zero DC reconstruction shared by OVRTX and custom"
            ),
            "semantic_sidecar": semantic_sidecar_manifest,
            "position_min": position_min.detach().cpu().tolist(),
            "position_max": position_max.detach().cpu().tolist(),
            "center": center.detach().cpu().tolist(),
            "extent": extent.detach().cpu().tolist(),
            "camera_bounds_min": camera_min.detach().cpu().tolist(),
            "camera_bounds_max": camera_max.detach().cpu().tolist(),
            "camera_bounds_policy": bounds_policy,
            "bounding_radius": bounding_radius,
        },
        "camera_manifest": camera_manifest,
        "batch": args.batch,
        "width": args.width,
        "height": args.height,
        "color_space": color_contract["color_space"],
        "color_transform": color_contract["color_transform"],
        "deterministic": args.deterministic,
        "gaussian_support_sigma": args.gaussian_support_sigma,
        "covariance_epsilon": args.covariance_epsilon,
        "rasterize_mode": args.rasterize_mode,
        "semantic_min_alpha": args.semantic_min_alpha,
        "semantic_scheme": args.semantic_scheme,
        "semantic_grid": list(args.semantic_grid),
        "semantic_group_count": semantic_group_count,
        "semantic_rule": (
            "maximum_alpha_transmittance_contribution_when_accumulated_alpha_meets_semantic_min_alpha"
        ),
        "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
        "gaussian_evaluation_model": (
            "exact_camera_ray_3d_mahalanobis"
            if args.ray_gaussian_evaluation
            else "screen_space_2d_conic"
        ),
        "near_plane": near_plane,
        "far_plane": far_plane,
        "depth_range_policy": (
            "camera-space-scene-bounding-sphere"
            if args.tight_depth_range
            else "broad-default"
        ),
        "tile_size": args.tile_size,
        "bin_tile_size": backend.bin_tile_size,
        "depth_bucket_count": args.depth_bucket_count,
        "depth_bucket_group_size": min(
            args.depth_bucket_group_size,
            args.depth_bucket_count,
        ),
        "compact_projection_cache": args.compact_projection_cache,
        "direct_intersection_mode": args.compact_projection_cache,
        "projected_record_reuse": backend.capacity_stats["projected_record_reuse"],
        "projection_cache": {
            "enabled": args.projection_cache,
            "explicit_invalidation_before_each_render": (
                args.invalidate_projection_cache_before_each_render
            ),
            "warmup_invalidations": warmup_invalidations,
            "measured_invalidations": measured_invalidations,
            "scope": backend.projection_cache_scope,
            "coherence_contract": (
                "cache key includes scene revision, projection parameters, "
                "tensor identity/version, camera transforms, intrinsics, "
                "environment transforms, scene IDs, and active camera IDs; "
                "external zero-copy writers and inference-mode tensor writes "
                "must explicitly invalidate"
            ),
            "measured_hits": (
                int(cache_stats_after_measurement["hits"])
                - int(cache_stats_before_measurement["hits"])
            ),
            "measured_misses": (
                int(cache_stats_after_measurement["misses"])
                - int(cache_stats_before_measurement["misses"])
            ),
            "cumulative_hits": int(cache_stats_after_measurement["hits"]),
            "cumulative_misses": int(cache_stats_after_measurement["misses"]),
        },
        "depth_diagnostics": depth_diagnostics,
        "culling_diagnostics": culling_diagnostics,
        "outputs": {
            name: {
                "shape": list(tensor.shape),
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in outputs.items()
        },
        "loading": {
            "host_ply_seconds": host_load_seconds,
            "canonicalize_upload_seconds": upload_seconds,
            "first_render_ms": first_render_ms,
        },
        "warmup_iterations": args.warmup,
        "measurement_mode": (
            "duration_soak" if args.duration_seconds is not None else "fixed_iterations"
        ),
        "requested_duration_seconds": args.duration_seconds,
        "duration_seconds": wall_seconds,
        "renders_per_check": args.renders_per_check,
        "synchronized_checks": synchronized_checks,
        "measured_iterations": render_calls,
        "frame_ms": {
            **frame_stats,
            "samples": frame_samples_ms,
            "sample_scope": (
                "synchronized_chunk_mean"
                if args.duration_seconds is not None
                else "individual_cuda_event"
            ),
        },
        "wall_frame_ms": wall_seconds * 1_000.0 / render_calls,
        "images_per_second": args.batch * 1_000.0 / mean_ms,
        "wall_images_per_second": (render_calls * args.batch / wall_seconds),
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
            "visible_fraction": args.visible_fraction,
            "intersections_per_visible": args.intersections_per_visible,
            "direct_visible_gaussians_per_view": (
                args.direct_visible_gaussians_per_view
                if args.compact_projection_cache
                else None
            ),
            "direct_intersections_per_view_at_128": (
                args.direct_intersections_per_view_at_128
                if args.compact_projection_cache
                else None
            ),
            "resolution_scale_from_128x128": resolution_scale,
            "source": (
                "explicit"
                if args.visible_capacity is not None
                or args.intersection_capacity is not None
                else "full-scene-exterior-preflight"
            ),
        },
        "first_counters": first_counters,
        "warmup_counters": warmup_counters,
        "measured_counters": measured_counters,
        "scene_bytes": backend.scene_bytes,
        "workspace_bytes": backend.workspace_bytes,
        "workspace_bytes_by_tensor": backend.workspace_bytes_by_tensor,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocation_delta_bytes": allocated_after - allocated_before,
        "driver_process_memory_bytes": driver_bytes,
        "driver_memory_scope": driver_scope,
        "runtime_camera_loop": False,
        "batched_native_submissions_per_render": 1,
        "outputs_gpu_resident": outputs_gpu_resident,
        "outputs_contract_valid": outputs_contract_valid,
        "output_validity": final_contract,
        "all_output_checks_valid": all_output_checks_valid,
        "zero_overflow": zero_overflow,
        "artifacts": artifacts,
        "pass": (
            measured_counters["visible_overflow"] == 0
            and measured_counters["intersection_overflow"] == 0
            and allocated_after == allocated_before
            and driver_bytes is not None
            and outputs_gpu_resident
            and outputs_contract_valid
            and all_output_checks_valid
            and zero_overflow
            and (args.duration_seconds is None or wall_seconds >= args.duration_seconds)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    service.shutdown()
    del outputs
    del scene
    del semantic_ids
    del backend
    del service
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    require_passing_result(result)
    marker = (
        "HOME_SCAN_CUSTOM_OK"
        if args.dataset_id == "home-scan-lod0"
        else "PUBLIC_SCENE_CUSTOM_OK"
    )
    print(marker, json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
