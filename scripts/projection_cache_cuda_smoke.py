"""Verify projection-cache coherence and per-frame raster execution on CUDA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle,
    synthetic_scene_tensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--gaussians", type=int, default=10_000)
    parser.add_argument("--visible-capacity", type=int)
    parser.add_argument("--intersection-capacity", type=int)
    parser.add_argument("--max-workspace-bytes", type=int)
    parser.add_argument("--ray-gaussian-evaluation", action="store_true")
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument("--covariance-epsilon", type=float)
    parser.add_argument("--compact-projection-cache", action="store_true")
    parser.add_argument("--materialize-projected-records", action="store_true")
    parser.add_argument("--depth-bucket-count", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/projection-cache/cuda-smoke.json"),
    )
    return parser.parse_args()


def _output_contract(
    outputs: dict[str, torch.Tensor],
    *,
    height: int,
    width: int,
    semantic_min_alpha: float,
) -> dict[str, bool]:
    alpha = outputs["alpha"][..., 0]
    foreground = alpha > 0.0
    semantic_foreground = (
        foreground
        if semantic_min_alpha == 0.0
        else alpha >= semantic_min_alpha
    )
    semantic_background = ~semantic_foreground
    foreground_depth = outputs["depth"][..., 0][foreground]
    semantic = outputs["semantic_id"][..., 0]
    foreground_semantic = semantic[semantic_foreground]
    background_semantic = semantic[semantic_background]
    return {
        "shapes": (
            tuple(outputs["rgb"].shape) == (1, height, width, 3)
            and tuple(outputs["depth"].shape) == (1, height, width, 1)
            and tuple(outputs["alpha"].shape) == (1, height, width, 1)
            and tuple(outputs["semantic_id"].shape)
            == (1, height, width, 1)
        ),
        "dtypes": (
            outputs["rgb"].dtype == torch.float32
            and outputs["depth"].dtype == torch.float32
            and outputs["alpha"].dtype == torch.float32
            and outputs["semantic_id"].dtype == torch.int64
        ),
        "cuda_resident": all(tensor.is_cuda for tensor in outputs.values()),
        "finite_rgb": bool(torch.isfinite(outputs["rgb"]).all().item()),
        "finite_foreground_depth": bool(
            foreground_depth.numel()
            and torch.isfinite(foreground_depth).all().item()
        ),
        "alpha_bounds": bool(
            ((alpha >= 0.0) & (alpha <= 1.0)).all().item()
        ),
        "nonempty_foreground": bool(foreground.any().item()),
        "valid_foreground_semantics": bool(
            foreground_semantic.numel()
            and (foreground_semantic >= 0).all().item()
        ),
        "background_semantics": bool(
            (background_semantic == -1).all().item()
        ),
    }


def _outputs_match(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, bool]:
    matches: dict[str, bool] = {}
    for name, expected in reference.items():
        actual = candidate[name]
        if expected.is_floating_point():
            matches[name] = bool(
                torch.allclose(
                    expected,
                    actual,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                    equal_nan=True,
                )
            )
        else:
            matches[name] = bool(torch.equal(expected, actual))
    return matches


def _resolve_covariance_epsilon(
    *,
    requested: float | None,
    rasterize_mode: str,
    ray_gaussian_evaluation: bool,
    compact_projection_cache: bool,
) -> float:
    value = requested
    if value is None:
        if rasterize_mode == "antialiased":
            value = 0.3
        elif ray_gaussian_evaluation or compact_projection_cache:
            value = 0.0
        else:
            value = 0.3
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "--covariance-epsilon must be finite and non-negative."
        )
    return value


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.gaussians <= 0:
        raise ValueError("width, height, and gaussians must be positive.")
    for name in (
        "visible_capacity",
        "intersection_capacity",
        "max_workspace_bytes",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive.")
    if args.compact_projection_cache and args.ray_gaussian_evaluation:
        raise ValueError(
            "Compact projection currently supports screen-space evaluation."
        )
    if args.rasterize_mode == "antialiased" and args.ray_gaussian_evaluation:
        raise ValueError(
            "--rasterize-mode antialiased is incompatible with "
            "--ray-gaussian-evaluation."
        )
    if args.materialize_projected_records and not args.compact_projection_cache:
        raise ValueError(
            "--materialize-projected-records requires "
            "--compact-projection-cache."
        )
    depth_bucket_count = args.depth_bucket_count or (
        128 if args.compact_projection_cache else 1024
    )
    if depth_bucket_count <= 0:
        raise ValueError("depth-bucket-count must be positive.")
    covariance_epsilon = _resolve_covariance_epsilon(
        requested=args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        compact_projection_cache=args.compact_projection_cache,
    )

    device = torch.device("cuda")
    scene = synthetic_scene_tensors(
        args.gaussians,
        seed=87,
        device=device,
    )
    cameras = camera_bundle(
        1,
        args.width,
        args.height,
        device=device,
    )
    scene_ids = torch.tensor([87], device=device, dtype=torch.int64)
    backend = CustomCudaBackend(
        max_visible_records=args.visible_capacity or args.gaussians,
        max_intersections=(
            args.intersection_capacity
            or max(500_000, args.gaussians * 16)
        ),
        max_workspace_bytes=args.max_workspace_bytes,
        gaussian_support_sigma=3.0,
        covariance_epsilon=covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        tile_size=1,
        depth_bucket_count=depth_bucket_count,
        depth_bucket_group_size=min(32, depth_bucket_count),
        compact_projection_cache=args.compact_projection_cache,
        materialize_projected_records=args.materialize_projected_records,
        enable_projection_cache=True,
        output_srgb=False,
        deterministic=not args.compact_projection_cache,
    )
    service = RendererService(
        backend,
        height=args.height,
        width=args.width,
        max_views=1,
    )
    service.initialize(stage=None, device=device)
    service.load_scene(
        87,
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
    backend.check_capacity(synchronize=False)
    initial_output_contract = _output_contract(
        outputs,
        height=args.height,
        width=args.width,
        semantic_min_alpha=backend.semantic_min_alpha,
    )
    initial_capacity_adaptation = backend.capacity_stats
    initial_projection_attempts = (
        initial_capacity_adaptation["last_render"]["retry_count"] + 1
    )
    miss_reference = {
        name: tensor.clone()
        for name, tensor in outputs.items()
    }

    service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=outputs,
    )
    service.synchronize()
    coherent_equal = {
        name: bool(torch.equal(miss_reference[name], tensor))
        for name, tensor in outputs.items()
    }
    stats_after_hit = backend.projection_cache_stats

    scene["colors"][:, :3].mul_(0.5)
    service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=outputs,
    )
    service.synchronize()
    color_update_rgb_changed = not bool(
        torch.equal(miss_reference["rgb"], outputs["rgb"])
    )
    color_update_geometry_equal = {
        name: bool(torch.equal(miss_reference[name], outputs[name]))
        for name in ("depth", "alpha", "semantic_id")
    }
    stats_after_color_update = backend.projection_cache_stats

    cameras.viewmats[0, 0, 3] = 0.05
    service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=outputs,
    )
    service.synchronize()
    stats_after_camera_update = backend.projection_cache_stats

    backend.invalidate_projection_cache()
    service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=outputs,
    )
    service.synchronize()
    stats_after_manual_invalidation = backend.projection_cache_stats

    mode_mutation = None
    if args.rasterize_mode == "antialiased":
        antialiased_reference = {
            name: tensor.clone()
            for name, tensor in outputs.items()
        }
        misses_before_mode_mutation = stats_after_manual_invalidation["misses"]
        backend.rasterize_mode = "classic"
        service.render(
            cameras.viewmats,
            cameras.intrinsics,
            scene_ids,
            outputs=outputs,
        )
        service.synchronize()
        classic_outputs = {
            name: tensor.clone()
            for name, tensor in outputs.items()
        }
        stats_after_classic_mode = backend.projection_cache_stats
        classic_projection_attempts = (
            backend.capacity_stats["last_render"]["retry_count"] + 1
        )
        backend.rasterize_mode = "antialiased"
        service.render(
            cameras.viewmats,
            cameras.intrinsics,
            scene_ids,
            outputs=outputs,
        )
        service.synchronize()
        stats_after_antialias_restore = backend.projection_cache_stats
        antialias_restore_projection_attempts = (
            backend.capacity_stats["last_render"]["retry_count"] + 1
        )
        mode_mutation = {
            "classic_changed_rgb_or_alpha": (
                not torch.equal(
                    antialiased_reference["rgb"],
                    classic_outputs["rgb"],
                )
                or not torch.equal(
                    antialiased_reference["alpha"],
                    classic_outputs["alpha"],
                )
            ),
            "restored_antialias_outputs_match": _outputs_match(
                antialiased_reference,
                outputs,
            ),
            "classic_mode_caused_cache_miss": (
                stats_after_classic_mode["misses"]
                == misses_before_mode_mutation + classic_projection_attempts
            ),
            "antialias_restore_caused_cache_miss": (
                stats_after_antialias_restore["misses"]
                == misses_before_mode_mutation
                + classic_projection_attempts
                + antialias_restore_projection_attempts
            ),
            "classic_projection_attempts": classic_projection_attempts,
            "antialias_restore_projection_attempts": (
                antialias_restore_projection_attempts
            ),
            "stats_after_classic_mode": stats_after_classic_mode,
            "stats_after_antialias_restore": stats_after_antialias_restore,
        }

    final_counters = backend.check_capacity(synchronize=False)
    final_output_contract = _output_contract(
        outputs,
        height=args.height,
        width=args.width,
        semantic_min_alpha=backend.semantic_min_alpha,
    )

    passed = (
        all(coherent_equal.values())
        and color_update_rgb_changed
        and all(color_update_geometry_equal.values())
        and stats_after_hit["hits"] == 1
        and stats_after_hit["misses"] == initial_projection_attempts
        and stats_after_color_update["hits"] == 2
        and stats_after_color_update["misses"] == initial_projection_attempts
        and stats_after_camera_update["hits"] == 2
        and stats_after_camera_update["misses"] == (
            initial_projection_attempts + 1
        )
        and stats_after_manual_invalidation["hits"] == 2
        and stats_after_manual_invalidation["misses"] == (
            initial_projection_attempts + 2
        )
        and final_counters["visible_overflow"] == 0
        and final_counters["intersection_overflow"] == 0
        and all(initial_output_contract.values())
        and all(final_output_contract.values())
        and (
            mode_mutation is None
            or (
                mode_mutation["classic_changed_rgb_or_alpha"]
                and all(
                    mode_mutation["restored_antialias_outputs_match"].values()
                )
                and mode_mutation["classic_mode_caused_cache_miss"]
                and mode_mutation["antialias_restore_caused_cache_miss"]
            )
        )
    )
    result = {
        "schema_version": "projection-cache-cuda-smoke/v1",
        "pass": passed,
        "device": torch.cuda.get_device_name(device),
        "gaussian_count": args.gaussians,
        "resolution": [args.width, args.height],
        "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
        "rasterize_mode": args.rasterize_mode,
        "covariance_epsilon": covariance_epsilon,
        "compact_projection_cache": args.compact_projection_cache,
        "direct_intersection_mode": args.compact_projection_cache,
        "projected_record_reuse": backend.capacity_stats[
            "projected_record_reuse"
        ],
        "depth_bucket_count": depth_bucket_count,
        "pipeline": backend.pipeline_name,
        "cache_scope": backend.projection_cache_scope,
        "visible_storage_capacity": backend.visible_storage_capacity,
        "initial_capacity_adaptation": initial_capacity_adaptation,
        "final_capacity_adaptation": backend.capacity_stats,
        "coherent_outputs_bitwise_equal": coherent_equal,
        "color_update_on_cache_hit": {
            "rgb_changed": color_update_rgb_changed,
            "geometry_outputs_bitwise_equal": color_update_geometry_equal,
        },
        "stats_after_coherent_hit": stats_after_hit,
        "stats_after_color_update": stats_after_color_update,
        "stats_after_camera_update": stats_after_camera_update,
        "stats_after_manual_invalidation": stats_after_manual_invalidation,
        "mode_mutation": mode_mutation,
        "final_counters": final_counters,
        "initial_output_contract": initial_output_contract,
        "final_output_contract": final_output_contract,
        "outputs_gpu_resident": final_output_contract["cuda_resident"],
        "batched_native_submissions_per_render": 1,
        "initial_native_submissions": initial_projection_attempts,
        "runtime_camera_loop": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    service.shutdown()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
