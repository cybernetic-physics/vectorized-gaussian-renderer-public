"""Benchmark the project-owned vectorized CUDA renderer."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__:
    from .renderer_options import resolve_covariance_epsilon
else:
    from renderer_options import resolve_covariance_epsilon

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    STRESS_SCENES,
    camera_bundle,
    resolve_output_color_contract,
    stats,
    synthetic_exact_intersections_per_visible,
    stress_scene_manifest,
    stress_scene_tensors,
    synthetic_scene_manifest,
    synthetic_scene_tensors,
)
from isaacsim_gaussian_renderer.fidelity import bundle_from_tensors, write_camera_bundle


SYNTHETIC_SCENES = {
    "synthetic-small": (10_000, 42),
    "synthetic-medium": (100_000, 43),
    **STRESS_SCENES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(SYNTHETIC_SCENES), default="synthetic-small")
    parser.add_argument("--batch", type=int, default=8)
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
        help=(
            "Gaussian support radius in standard deviations. The benchmark "
            "default matches OVRTX's documented hard 3-sigma support; pass "
            "3.33 for the pinned-gsplat parity control."
        ),
    )
    parser.add_argument("--semantic-min-alpha", type=float, default=0.01)
    parser.add_argument(
        "--covariance-epsilon",
        type=float,
        default=None,
        help=(
            "Screen-space covariance regularizer. Defaults to 0 for exact "
            "ray evaluation and 0.3 for screen-space gsplat parity."
        ),
    )
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
        help=(
            "Treat scene RGB values as already encoded in the OVRTX "
            "display-sRGB contract and apply no additional transfer."
        ),
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
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
        return int(output.splitlines()[0].strip()) * 1024 * 1024, "isolated-gpu-total"
    except Exception:
        return None, "unavailable"


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def output_contract(
    outputs: dict[str, torch.Tensor],
    *,
    expected_shape: tuple[int, int, int],
    semantic_min_alpha: float,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Return concrete evidence for every production output invariant."""
    expected_specs = {
        "rgb": ((*expected_shape, 3), torch.float32),
        "depth": ((*expected_shape, 1), torch.float32),
        "alpha": ((*expected_shape, 1), torch.float32),
        "semantic_id": ((*expected_shape, 1), torch.int64),
    }
    output_names = list(outputs)
    output_names_match = set(outputs) == set(expected_specs)
    required_outputs_present = set(expected_specs).issubset(outputs)
    output_shapes = {
        name: list(tensor.shape) for name, tensor in outputs.items()
    }
    output_dtypes = {
        name: str(tensor.dtype) for name, tensor in outputs.items()
    }
    output_devices = {
        name: str(tensor.device) for name, tensor in outputs.items()
    }
    if not required_outputs_present:
        return {
            "valid": False,
            "output_names": output_names,
            "output_names_match": output_names_match,
            "output_shapes": output_shapes,
            "output_shapes_match": False,
            "output_dtypes": output_dtypes,
            "output_dtypes_match": False,
            "output_devices": output_devices,
            "outputs_contiguous": False,
            "outputs_gpu_resident": False,
            "cuda_residency_required": require_cuda,
            "alpha_in_range": False,
            "missing_outputs": sorted(set(expected_specs) - set(outputs)),
        }

    output_shapes_match = all(
        tuple(outputs[name].shape) == shape
        for name, (shape, _dtype) in expected_specs.items()
    )
    output_dtypes_match = all(
        outputs[name].dtype == dtype
        for name, (_shape, dtype) in expected_specs.items()
    )
    outputs_contiguous = all(
        tensor.is_contiguous() for tensor in outputs.values()
    )
    outputs_gpu_resident = all(
        tensor.is_cuda for tensor in outputs.values()
    )
    residency_valid = outputs_gpu_resident if require_cuda else True
    alpha = outputs["alpha"][..., 0]
    depth = outputs["depth"][..., 0]
    semantic = outputs["semantic_id"][..., 0]
    foreground = alpha > 0
    background = ~foreground
    rgb_finite = bool(torch.isfinite(outputs["rgb"]).all().item())
    alpha_finite = bool(torch.isfinite(alpha).all().item())
    alpha_in_range = bool(
        ((alpha >= 0.0) & (alpha <= 1.0)).all().item()
    )
    foreground_pixel_count = int(torch.count_nonzero(foreground).item())
    foreground_depth_finite = bool(
        foreground_pixel_count > 0
        and torch.isfinite(depth[foreground]).all().item()
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
    semantic_foreground_pixel_count = int(
        torch.count_nonzero(semantic_foreground).item()
    )
    foreground_semantic_valid = bool(
        semantic_foreground_pixel_count > 0
        and (semantic[semantic_foreground] >= 0).all().item()
    )
    background_semantic_valid = bool(
        (semantic[semantic_background] == -1).all().item()
    )
    background_positive_infinity_pixels = int(
        torch.count_nonzero(
            background & torch.isinf(depth) & (depth > 0)
        ).item()
    )
    valid = bool(
        output_names_match
        and output_shapes_match
        and output_dtypes_match
        and outputs_contiguous
        and residency_valid
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
        "output_names": output_names,
        "output_names_match": output_names_match,
        "output_shapes": output_shapes,
        "output_shapes_match": output_shapes_match,
        "output_dtypes": output_dtypes,
        "output_dtypes_match": output_dtypes_match,
        "output_devices": output_devices,
        "outputs_contiguous": outputs_contiguous,
        "outputs_gpu_resident": outputs_gpu_resident,
        "cuda_residency_required": require_cuda,
        "rgb_finite": rgb_finite,
        "alpha_finite": alpha_finite,
        "alpha_in_range": alpha_in_range,
        "foreground_pixel_count": foreground_pixel_count,
        "foreground_depth_finite": foreground_depth_finite,
        "background_depth_valid": background_depth_valid,
        "background_positive_infinity_pixels": (
            background_positive_infinity_pixels
        ),
        "foreground_semantic_valid": foreground_semantic_valid,
        "background_semantic_valid": background_semantic_valid,
        "semantic_foreground_pixel_count": (
            semantic_foreground_pixel_count
        ),
        "semantic_min_alpha": semantic_min_alpha,
        "depth_background_policy": (
            "finite_or_positive_infinity_when_alpha_zero"
        ),
    }


def require_passing_result(result: dict[str, Any]) -> None:
    """Make a persisted failed benchmark observable to shell automation."""
    if result.get("pass") is not True:
        raise SystemExit(1)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("batch and resolution must be positive.")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive.")
    if args.compact_projection_cache and (args.tile_size != 1 or args.ray_gaussian_evaluation or args.deterministic):
        raise ValueError(
            "--compact-projection-cache requires --tile-size 1, "
            "screen-space evaluation, and non-deterministic scheduling."
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
    if args.max_workspace_gib is not None and args.max_workspace_gib <= 0:
        raise ValueError("--max-workspace-gib must be positive.")
    visible_fraction = (
        args.visible_fraction
        if args.visible_fraction is not None
        else (0.10 if args.ray_gaussian_evaluation else 0.065)
    )
    if visible_fraction <= 0:
        raise ValueError("visible_fraction must be positive.")
    if args.gaussian_support_sigma <= 0:
        raise ValueError("gaussian_support_sigma must be positive.")
    if args.depth_bucket_count <= 0 or args.depth_bucket_group_size <= 0:
        raise ValueError("Depth bucket count and group size must be positive.")
    if args.direct_intersections_per_camera_gaussian <= 0:
        raise ValueError("--direct-intersections-per-camera-gaussian must be positive.")
    if not math.isfinite(args.semantic_min_alpha) or not 0 <= args.semantic_min_alpha <= 1:
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
    gaussian_count, seed = SYNTHETIC_SCENES[args.scene]
    visible_capacity = args.visible_capacity or math.ceil(args.batch * gaussian_count * visible_fraction)
    resolution_scale = args.width * args.height / float(128 * 128)
    if args.compact_projection_cache:
        intersections_per_visible = None
        intersection_capacity = args.intersection_capacity or math.ceil(
            args.batch * gaussian_count * args.direct_intersections_per_camera_gaussian * resolution_scale
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
        intersection_capacity = args.intersection_capacity or math.ceil(visible_capacity * intersections_per_visible)
    if args.scene in STRESS_SCENES:
        scene = stress_scene_tensors(
            args.scene,
            gaussian_count,
            seed=seed,
            device=device,
        )
        scene_manifest = stress_scene_manifest(
            args.scene,
            gaussian_count,
            seed,
        )
    else:
        scene = synthetic_scene_tensors(
            gaussian_count,
            seed=seed,
            device=device,
        )
        scene_manifest = synthetic_scene_manifest(
            args.scene,
            gaussian_count,
            seed,
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
    cold_start_ms = (time.perf_counter() - cold_start) * 1000.0

    for _ in range(args.warmup - 1):
        service.render(cameras.viewmats, cameras.intrinsics, scene_ids, outputs=outputs)
    service.synchronize()
    warmup_counters = backend.check_capacity(synchronize=False)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    wall_start = time.perf_counter()
    for start, end in zip(starts, ends, strict=True):
        start.record()
        service.render(cameras.viewmats, cameras.intrinsics, scene_ids, outputs=outputs)
        end.record()
    service.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    frame_samples_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
    measured_counters = backend.check_capacity(synchronize=False)
    allocated_after = torch.cuda.memory_allocated()
    driver_memory_bytes, driver_memory_scope = gpu_memory_snapshot()
    output_evidence = output_contract(
        outputs,
        expected_shape=(args.batch, args.height, args.width),
        semantic_min_alpha=args.semantic_min_alpha,
    )
    outputs_gpu_resident = bool(
        output_evidence["outputs_gpu_resident"]
    )
    outputs_contract_valid = bool(output_evidence["valid"])

    frame_stats = stats(frame_samples_ms)
    mean_ms = frame_stats["mean"]
    zero_overflow = measured_counters["visible_overflow"] == 0 and measured_counters["intersection_overflow"] == 0
    result: dict[str, Any] = {
        "schema_version": "custom-cuda-benchmark/v1",
        "implementation": "custom-cuda",
        "pipeline": backend.pipeline_name,
        "native_extension_loaded": True,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "environment": {
            "host": platform.node(),
            "project_commit": os.environ.get(
                "SOURCE_GIT_COMMIT",
                command_output(["git", "rev-parse", "HEAD"]) or "unknown",
            ),
            "driver": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            ),
        },
        "scene": scene_manifest,
        "camera_manifest": cameras.manifest,
        "batch": args.batch,
        "width": args.width,
        "height": args.height,
        "outputs": output_evidence["output_names"],
        "color_space": color_contract["color_space"],
        "color_transform": color_contract["color_transform"],
        "depth_rule": "expected_depth_alpha_weighted",
        "semantic_rule": "maximum_alpha_transmittance_contribution",
        "semantic_min_alpha": args.semantic_min_alpha,
        "deterministic": args.deterministic,
        "gaussian_support_sigma": args.gaussian_support_sigma,
        "covariance_epsilon": args.covariance_epsilon,
        "rasterize_mode": args.rasterize_mode,
        "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
        "gaussian_evaluation_model": (
            "exact_camera_ray_3d_mahalanobis" if args.ray_gaussian_evaluation else "screen_space_2d_conic"
        ),
        "tile_size": args.tile_size,
        "bin_tile_size": backend.bin_tile_size,
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
        "deterministic_order": (
            "full_float_depth_bits_then_global_gaussian_id_per_tile" if args.deterministic else None
        ),
        "cold_start_ms": cold_start_ms,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "frame_ms": {
            **frame_stats,
            "samples": frame_samples_ms,
        },
        "wall_frame_ms": wall_seconds * 1000.0 / args.iterations,
        "images_per_second": args.batch * 1000.0 / mean_ms,
        "megapixels_per_second": (args.batch * args.width * args.height / 1.0e6) * 1000.0 / mean_ms,
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
        "capacity_planner": {
            "visible_fraction": visible_fraction,
            "intersections_per_visible": intersections_per_visible,
            "direct_intersections_per_camera_gaussian": (
                args.direct_intersections_per_camera_gaussian if args.compact_projection_cache else None
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
        "zero_overflow": zero_overflow,
        "scene_bytes": backend.scene_bytes,
        "workspace_bytes": backend.workspace_bytes,
        "workspace_bytes_by_tensor": backend.workspace_bytes_by_tensor,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocation_delta_bytes": allocated_after - allocated_before,
        "driver_process_memory_bytes": driver_memory_bytes,
        "driver_memory_scope": driver_memory_scope,
        "runtime_camera_loop": False,
        "batched_native_submissions_per_render": 1,
        "outputs_gpu_resident": outputs_gpu_resident,
        "outputs_finite": outputs_contract_valid,
        "outputs_contract_valid": outputs_contract_valid,
        "output_names_match": output_evidence["output_names_match"],
        "output_shapes_match": output_evidence["output_shapes_match"],
        "output_dtypes_match": output_evidence["output_dtypes_match"],
        "outputs_contiguous": output_evidence["outputs_contiguous"],
        "alpha_in_range": output_evidence["alpha_in_range"],
        "output_validity": output_evidence,
        "output_shapes": output_evidence["output_shapes"],
        "output_dtypes": output_evidence["output_dtypes"],
        "output_devices": output_evidence["output_devices"],
        "pass": (
            zero_overflow
            and allocated_after == allocated_before
            and driver_memory_bytes is not None
            and outputs_gpu_resident
            and outputs_contract_valid
        ),
    }

    if args.artifact_dir is not None:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        fidelity_bundle = bundle_from_tensors(
            viewmats=cameras.viewmats,
            intrinsics=cameras.intrinsics,
            width=args.width,
            height=args.height,
            color_space=str(color_contract["color_space"]),
            scene_ids=scene_ids,
            scene_checksum=result["scene"]["checksum_sha256"],
        )
        camera_path = args.artifact_dir / "camera-bundle.json"
        output_path = args.artifact_dir / "custom-output.npz"
        write_camera_bundle(fidelity_bundle, camera_path)
        np.savez_compressed(
            output_path,
            rgb=outputs["rgb"].detach().cpu().numpy(),
            depth=outputs["depth"].detach().cpu().numpy()[..., 0],
            alpha=outputs["alpha"].detach().cpu().numpy()[..., 0],
            semantic=outputs["semantic_id"].detach().cpu().numpy()[..., 0],
            valid_depth=(outputs["alpha"] > 0).detach().cpu().numpy()[..., 0],
            color_space=np.asarray(fidelity_bundle.color_space),
            background=np.asarray(fidelity_bundle.background, dtype=np.float32),
            camera_bundle_id=np.asarray(fidelity_bundle.bundle_id),
        )
        result["artifacts"] = {
            "camera_bundle": str(camera_path),
            "render_output": str(output_path),
            "camera_bundle_id": fidelity_bundle.bundle_id,
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
