#!/usr/bin/env python3
"""Render one matched Home Scan camera path with custom CUDA or OVRTX."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import file_sha256
from isaacsim_gaussian_renderer.camera_math import (
    opencv_viewmats_to_usd_camera_world_matrices,
)
from isaacsim_gaussian_renderer.camera_paths import (
    cinematic_orbit_camera_bundle,
    cinematic_walkthrough_camera_bundle,
    scripted_walkthrough_camera_bundle,
)
from isaacsim_gaussian_renderer.ply_loader import (
    canonicalize_3dgs_scene,
    load_ply_to_gaussians,
)


HOME_SCAN_COUNT = 21_497_908
HOME_SCAN_SHA256 = (
    "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
)
SCENE_ID = 404


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renderer",
        choices=("custom", "ovrtx"),
        required=True,
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=Path("/workspace/datasets/home-scan-lod0.ply"),
    )
    parser.add_argument("--scene-id", default="home-scan-lod0")
    parser.add_argument(
        "--expected-scene-sha256",
        default=HOME_SCAN_SHA256,
        help="Pin the exact canonical PLY; pass an empty string to opt out.",
    )
    parser.add_argument(
        "--expected-gaussian-count",
        type=int,
        default=HOME_SCAN_COUNT,
        help="Pin the canonical Gaussian count; pass 0 to opt out.",
    )
    parser.add_argument("--scene-author", default="Isaiah Sweeney")
    parser.add_argument(
        "--scene-source",
        default="https://superspl.at/scene/3f89bbd3",
    )
    parser.add_argument("--scene-license", default="CC BY 4.0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/flyby/home-scan-v1"),
    )
    parser.add_argument("--camera-contract", type=Path)
    parser.add_argument(
        "--trajectory-contract-v1",
        type=Path,
        help="Consume the exact camera-trajectory-v1 contract used by the measured evaluator.",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--focal-scale", type=float, default=0.72)
    parser.add_argument(
        "--max-gaussian-scale",
        type=float,
        help=(
            "Optional per-axis scale cap applied only to the in-memory flyby "
            "render scene. The canonical PLY remains unchanged."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument(
        "--camera-path",
        choices=("clearance", "walkthrough", "orbit"),
        default="clearance",
    )
    parser.add_argument(
        "--route-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "benchmarks/camera_paths/home-scan-walkthrough-v1.json"
        ),
    )
    parser.add_argument("--route-lookahead-frames", type=int, default=6)
    parser.add_argument("--robust-sample-stride", type=int, default=20)
    parser.add_argument("--robust-lower-quantile", type=float, default=0.005)
    parser.add_argument("--robust-upper-quantile", type=float, default=0.995)
    parser.add_argument("--orbit-margin", type=float, default=2.0)
    parser.add_argument("--elevation-scale", type=float, default=0.58)
    parser.add_argument("--path-radius-x-fraction", type=float, default=0.58)
    parser.add_argument("--path-radius-z-fraction", type=float, default=0.52)
    parser.add_argument("--eye-height-fraction", type=float, default=0.60)
    parser.add_argument("--vertical-bob-fraction", type=float, default=0.025)
    parser.add_argument("--lookahead-phase", type=float, default=0.12)
    parser.add_argument("--start-angle-degrees", type=float, default=-90.0)
    parser.add_argument(
        "--direct-visible-per-view",
        type=int,
        default=500_000,
    )
    parser.add_argument(
        "--direct-intersections-per-view-at-128",
        type=int,
        default=2_400_000,
    )
    parser.add_argument("--depth-bucket-count", type=int, default=128)
    parser.add_argument("--depth-bucket-group-size", type=int, default=8)
    parser.add_argument("--ovrtx-temporal-samples", type=int, default=64)
    parser.add_argument("--ovrtx-warmup-samples", type=int, default=4)
    parser.add_argument(
        "--ovrtx-camera-write-mode",
        choices=("xform-op", "row-major", "transposed", "static-initial"),
        default="row-major",
    )
    parser.add_argument(
        "--ovrtx-camera-layout",
        choices=("static-all", "dynamic-chunks"),
        default="dynamic-chunks",
    )
    parser.add_argument(
        "--ovrtx-static-render-batch-size",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--ovrtx-reset-between-camera-batches",
        action="store_true",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-camera-contract",
        action="store_true",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.frames,
        args.width,
        args.height,
        args.fps,
        args.chunk_size,
        args.robust_sample_stride,
        args.direct_visible_per_view,
        args.direct_intersections_per_view_at_128,
        args.depth_bucket_count,
        args.depth_bucket_group_size,
        args.ovrtx_temporal_samples,
        args.route_lookahead_frames,
        args.ovrtx_static_render_batch_size,
    )
    if min(positive) <= 0 or args.ovrtx_warmup_samples < 0:
        raise ValueError("Flyby dimensions, capacities, and samples must be valid.")
    if (
        args.max_gaussian_scale is not None
        and args.max_gaussian_scale <= 0.0
    ):
        raise ValueError("--max-gaussian-scale must be positive when supplied.")
    if not (
        0.0
        <= args.robust_lower_quantile
        < args.robust_upper_quantile
        <= 1.0
    ):
        raise ValueError("Robust quantiles must satisfy 0 <= lower < upper <= 1.")
    if args.width != args.height:
        raise ValueError(
            "The current matched OVRTX aperture contract requires square frames."
        )
    if not (
        0.0 < args.path_radius_x_fraction < 1.0
        and 0.0 < args.path_radius_z_fraction < 1.0
        and 0.0 < args.eye_height_fraction < 1.0
        and 0.0 <= args.vertical_bob_fraction < 0.25
        and args.lookahead_phase > 0.0
    ):
        raise ValueError("Walkthrough path fractions are outside valid ranges.")
    if (
        args.ovrtx_camera_layout == "dynamic-chunks"
        and args.ovrtx_camera_write_mode == "static-initial"
        and args.frames > args.chunk_size
    ):
        raise ValueError(
            "static-initial camera mode requires frames <= chunk size."
        )


def load_scene(
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not args.scene_path.is_file():
        raise FileNotFoundError(args.scene_path)
    checksum = file_sha256(args.scene_path)
    if args.expected_scene_sha256 and checksum != args.expected_scene_sha256:
        raise ValueError(
            "Scene SHA-256 mismatch: "
            f"expected {args.expected_scene_sha256}, got {checksum}."
        )

    host_start = time.perf_counter()
    raw = load_ply_to_gaussians(args.scene_path)
    host_seconds = time.perf_counter() - host_start
    if args.expected_gaussian_count and raw.count != args.expected_gaussian_count:
        raise ValueError(
            f"Expected {args.expected_gaussian_count} Gaussians, got {raw.count}."
        )
    upload_start = time.perf_counter()
    gaussian_count = raw.count
    canonical = canonicalize_3dgs_scene(raw, device="cuda")
    scales = canonical.scales
    scale_filter: dict[str, Any] | None = None
    if args.max_gaussian_scale is not None:
        oversized = scales > args.max_gaussian_scale
        scales = scales.clamp_max(args.max_gaussian_scale).contiguous()
        scale_filter = {
            "kind": "per-axis-clamp",
            "max_gaussian_scale": args.max_gaussian_scale,
            "clipped_components": int(oversized.sum().item()),
            "clipped_gaussians": int(oversized.any(dim=1).sum().item()),
        }
    scene = {
        "means": canonical.means,
        "quats": canonical.rotations,
        "scales": scales,
        "opacities": canonical.opacities,
        "colors": canonical.features[:, :3].contiguous(),
        # Semantics do not affect RGB. One class keeps the visual demo's OVRTX
        # upload compact while preserving the identical Gaussian scene.
        "semantic_ids": torch.zeros(
            canonical.count,
            device="cuda",
            dtype=torch.int64,
        ),
    }
    del canonical
    del raw
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return scene, {
        "scene_id": args.scene_id,
        "gaussian_count": gaussian_count,
        "sha256": checksum,
        "file_bytes": args.scene_path.stat().st_size,
        "host_load_seconds": host_seconds,
        "canonicalize_upload_seconds": time.perf_counter() - upload_start,
        "license": args.scene_license,
        "author": args.scene_author,
        "source": args.scene_source,
        "scale_filter": scale_filter,
    }


def camera_contract_hash(
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
    manifest: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(viewmats).tobytes())
    digest.update(np.ascontiguousarray(intrinsics).tobytes())
    digest.update(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def create_camera_contract(
    args: argparse.Namespace,
    scene: dict[str, torch.Tensor],
    scene_metadata: dict[str, Any],
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    sample = scene["means"][:: args.robust_sample_stride]
    quantiles = torch.tensor(
        [args.robust_lower_quantile, args.robust_upper_quantile],
        device=sample.device,
        dtype=sample.dtype,
    )
    robust_min, robust_max = torch.quantile(sample, quantiles, dim=0)
    route_metadata: dict[str, Any] = {}
    if args.camera_path == "clearance":
        if not args.route_path.is_file():
            raise FileNotFoundError(
                f"Clearance-aware route contract not found: {args.route_path}"
            )
        route_contract = json.loads(
            args.route_path.read_text(encoding="utf-8")
        )
        expected_route_hash = route_contract.pop("route_sha256")
        route_hash_payload = json.dumps(
            route_contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        actual_route_hash = hashlib.sha256(route_hash_payload).hexdigest()
        route_contract["route_sha256"] = expected_route_hash
        if actual_route_hash != expected_route_hash:
            raise ValueError(
                "Route contract hash mismatch: "
                f"{actual_route_hash} != {expected_route_hash}."
            )
        world_path_xz = np.asarray(
            route_contract["world_path_xz"],
            dtype=np.float32,
        )
        if (
            world_path_xz.ndim != 2
            or world_path_xz.shape[1] != 2
            or world_path_xz.shape[0] < 2
        ):
            raise ValueError(
                "Route contract world_path_xz must have shape [N, 2]."
            )
        path_points = torch.empty(
            (world_path_xz.shape[0], 3),
            device=scene["means"].device,
            dtype=scene["means"].dtype,
        )
        path_points[:, 0] = torch.from_numpy(
            world_path_xz[:, 0]
        ).to(path_points)
        path_points[:, 1] = float(route_contract["eye_height"])
        path_points[:, 2] = torch.from_numpy(
            world_path_xz[:, 1]
        ).to(path_points)
        bundle = scripted_walkthrough_camera_bundle(
            path_points=path_points,
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            focal_scale=args.focal_scale,
            lookahead_frames=args.route_lookahead_frames,
            world_up=torch.tensor(
                [0.0, -1.0, 0.0],
                device=path_points.device,
                dtype=path_points.dtype,
            ),
            route_sha256=expected_route_hash,
        )
        route_metadata = {
            "route_path": str(args.route_path.resolve()),
            "route_schema_version": route_contract["schema_version"],
            "route_sha256": expected_route_hash,
            "route_planner": route_contract["planner"],
        }
    elif args.camera_path == "walkthrough":
        bundle = cinematic_walkthrough_camera_bundle(
            bounds_min=robust_min,
            bounds_max=robust_max,
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            focal_scale=args.focal_scale,
            path_radius_x_fraction=args.path_radius_x_fraction,
            path_radius_z_fraction=args.path_radius_z_fraction,
            eye_height_fraction=args.eye_height_fraction,
            vertical_bob_fraction=args.vertical_bob_fraction,
            lookahead_phase=args.lookahead_phase,
            start_angle_degrees=args.start_angle_degrees,
        )
    else:
        bundle = cinematic_orbit_camera_bundle(
            bounds_min=robust_min,
            bounds_max=robust_max,
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            focal_scale=args.focal_scale,
            orbit_margin=args.orbit_margin,
            elevation_scale=args.elevation_scale,
            start_angle_degrees=args.start_angle_degrees,
        )
    viewmats = bundle.viewmats.detach().cpu().numpy().astype(np.float32)
    intrinsics = bundle.intrinsics.detach().cpu().numpy().astype(np.float32)
    manifest = {
        **bundle.manifest,
        "schema_version": "gaussian-flyby-camera-contract/v1",
        "scene_sha256": scene_metadata["sha256"],
        "scene_gaussian_count": scene_metadata["gaussian_count"],
        "fps": args.fps,
        "robust_sample_stride": args.robust_sample_stride,
        "robust_sample_count": int(sample.shape[0]),
        "robust_lower_quantile": args.robust_lower_quantile,
        "robust_upper_quantile": args.robust_upper_quantile,
        **route_metadata,
    }
    manifest["camera_contract_sha256"] = camera_contract_hash(
        viewmats,
        intrinsics,
        manifest,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        viewmats=viewmats,
        intrinsics=intrinsics,
        manifest_json=np.asarray(
            json.dumps(manifest, sort_keys=True),
        ),
    )
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle.viewmats, bundle.intrinsics, manifest


def load_camera_contract(
    args: argparse.Namespace,
    scene_metadata: dict[str, Any],
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as contract:
        viewmats = np.asarray(contract["viewmats"], dtype=np.float32)
        intrinsics = np.asarray(contract["intrinsics"], dtype=np.float32)
        manifest = json.loads(
            str(np.asarray(contract["manifest_json"]).item())
        )
    expected_hash = manifest.pop("camera_contract_sha256")
    actual_hash = camera_contract_hash(viewmats, intrinsics, manifest)
    manifest["camera_contract_sha256"] = expected_hash
    if actual_hash != expected_hash:
        raise ValueError(
            f"Camera contract hash mismatch: {actual_hash} != {expected_hash}."
        )
    expected_shape = (args.frames, 4, 4)
    if viewmats.shape != expected_shape:
        raise ValueError(
            f"Camera contract has shape {viewmats.shape}, expected {expected_shape}."
        )
    if intrinsics.shape != (args.frames, 3, 3):
        raise ValueError("Camera contract intrinsics have the wrong shape.")
    if (
        manifest["width"] != args.width
        or manifest["height"] != args.height
        or manifest["scene_sha256"] != scene_metadata["sha256"]
        or not math.isclose(
            float(manifest["focal_scale"]),
            args.focal_scale,
        )
    ):
        raise ValueError(
            "Camera contract resolution, focal scale, or scene does not match."
        )
    return (
        torch.from_numpy(viewmats).to(device="cuda").contiguous(),
        torch.from_numpy(intrinsics).to(device="cuda").contiguous(),
        manifest,
    )


def prepare_frames_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} already contains frames; pass --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_rgb_frame(path: Path, rgb: np.ndarray) -> dict[str, float]:
    image = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path, compress_level=4)
    signal = np.any(image > 2, axis=-1)
    luma = (
        image[..., 0].astype(np.float32) * 0.2126
        + image[..., 1].astype(np.float32) * 0.7152
        + image[..., 2].astype(np.float32) * 0.0722
    ) / 255.0
    return {
        "nonblack_fraction": float(signal.mean()),
        "mean_luma": float(luma.mean()),
        "clipped_fraction": float(np.any(image >= 254, axis=-1).mean()),
    }


def summarize_frame_stats(items: list[dict[str, float]]) -> dict[str, Any]:
    if not items:
        raise ValueError("No flyby frame statistics were recorded.")
    result: dict[str, Any] = {}
    for name in items[0]:
        values = [item[name] for item in items]
        result[name] = {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
    black_frame_count = sum(item["nonblack_fraction"] <= 0 for item in items)
    result["black_frame_count"] = black_frame_count
    result["pass"] = black_frame_count == 0
    result["failures"] = (
        []
        if black_frame_count == 0
        else [f"{black_frame_count} flyby frames are completely black"]
    )
    return result


def flyby_status_marker(passed: bool) -> str:
    """Return a log marker that cannot label a failed artifact as successful."""
    return "GAUSSIAN_FLYBY_OK" if passed else "GAUSSIAN_FLYBY_FAIL"


@torch.inference_mode()
def render_custom(
    args: argparse.Namespace,
    scene: dict[str, torch.Tensor],
    scene_metadata: dict[str, Any],
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_manifest: dict[str, Any],
) -> dict[str, Any]:
    output_dir = args.output_root / "custom"
    frames_dir = output_dir / "frames"
    prepare_frames_dir(frames_dir, overwrite=args.overwrite)
    resolution_scale = (
        args.width * args.height / float(128 * 128)
    )
    backend = CustomCudaBackend(
        max_visible_records=(
            args.chunk_size * args.direct_visible_per_view
        ),
        max_intersections=math.ceil(
            args.chunk_size
            * args.direct_intersections_per_view_at_128
            * resolution_scale
        ),
        near_plane=float(camera_manifest["near_plane"]),
        far_plane=float(camera_manifest["far_plane"]),
        gaussian_support_sigma=3.0,
        covariance_epsilon=0.0,
        semantic_min_alpha=0.01,
        tile_size=1,
        depth_bucket_count=args.depth_bucket_count,
        depth_bucket_group_size=args.depth_bucket_group_size,
        compact_projection_cache=True,
        enable_projection_cache=False,
        output_srgb=False,
        deterministic=False,
    )
    service = RendererService(
        backend,
        height=args.height,
        width=args.width,
        max_views=args.chunk_size,
    )
    service.initialize(stage=None, device="cuda")
    service.load_scene(
        SCENE_ID,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"],
    )

    start_time = time.perf_counter()
    stats: list[dict[str, float]] = []
    counter_high_water: dict[str, int] = {}
    for start in range(0, args.frames, args.chunk_size):
        stop = min(start + args.chunk_size, args.frames)
        batch = stop - start
        scene_ids = torch.full(
            (batch,),
            SCENE_ID,
            device="cuda",
            dtype=torch.int64,
        )
        outputs = service.render(
            viewmats[start:stop],
            intrinsics[start:stop],
            scene_ids,
        )
        service.synchronize()
        counters = backend.check_capacity(synchronize=False)
        for name, value in counters.items():
            counter_high_water[name] = max(
                counter_high_water.get(name, 0),
                int(value),
            )
        if counters["visible_overflow"] or counters["intersection_overflow"]:
            raise RuntimeError(
                f"Custom flyby capacity overflow at frames {start}:{stop}: "
                f"{counters}."
            )
        rgb = outputs["rgb"].detach().cpu().numpy()
        for local_index in range(batch):
            frame_index = start + local_index
            stats.append(
                save_rgb_frame(
                    frames_dir / f"frame-{frame_index:06d}.png",
                    rgb[local_index],
                )
            )
        print(
            f"FLYBY_CUSTOM_PROGRESS frames={stop}/{args.frames}",
            flush=True,
        )
    elapsed = time.perf_counter() - start_time
    capacity_adaptation = backend.capacity_stats
    service.shutdown()
    return {
        "schema_version": "gaussian-flyby-render/v1",
        "renderer": "custom-cuda",
        "gaussian_model": "deterministic-screen-space-2d-ewa",
        "scene": scene_metadata,
        "camera_contract_sha256": camera_manifest[
            "camera_contract_sha256"
        ],
        "frames": args.frames,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "chunk_size": args.chunk_size,
        "elapsed_seconds": elapsed,
        "frames_per_second": args.frames / elapsed,
        "counter_high_water": counter_high_water,
        "capacity_adaptation": capacity_adaptation,
        "frame_signal": summarize_frame_stats(stats),
        "visual_demo_only": True,
    }


def padded_chunk(
    tensors: torch.Tensor,
    start: int,
    stop: int,
    size: int,
) -> torch.Tensor:
    chunk = tensors[start:stop]
    if chunk.shape[0] == size:
        return chunk
    return torch.cat(
        (
            chunk,
            chunk[-1:].expand(size - chunk.shape[0], *chunk.shape[1:]),
        ),
        dim=0,
    ).contiguous()


@torch.inference_mode()
def render_ovrtx(
    args: argparse.Namespace,
    scene: dict[str, torch.Tensor],
    scene_metadata: dict[str, Any],
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_manifest: dict[str, Any],
) -> dict[str, Any]:
    import ovrtx

    from benchmarks.run_ovrtx import (
        build_stage,
        copy_cpu_render_var,
        product_paths,
        splat_prim_semantic_ids,
        upload_scene,
    )

    output_dir = args.output_root / "ovrtx"
    frames_dir = output_dir / "frames"
    prepare_frames_dir(frames_dir, overwrite=args.overwrite)
    camera_count = (
        args.frames
        if args.ovrtx_camera_layout == "static-all"
        else args.chunk_size
    )
    initial_viewmats = (
        viewmats
        if args.ovrtx_camera_layout == "static-all"
        else padded_chunk(
            viewmats,
            0,
            min(args.chunk_size, args.frames),
            args.chunk_size,
        )
    )
    prim_semantic_ids = splat_prim_semantic_ids(
        [int(scene["means"].shape[0])]
    )
    stage = build_stage(
        camera_count,
        args.width,
        args.height,
        args.focal_scale,
        initial_viewmats,
        "none",
        float(camera_manifest["near_plane"]),
        float(camera_manifest["far_plane"]),
        1,
        color_only=True,
        splat_prim_semantic_ids_override=prim_semantic_ids,
    )
    stage = stage.replace(
        'token omni:rtx:rendermode = "RealTimePathTracing"',
        'token omni:rtx:rendermode = "RealTimePathTracing"\n'
        "        bool omni:rtx:pt:fractionalCutoutOpacity = 1\n"
        "        bool omni:rtx:rt:fractionalOpacity = 1",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scene.usda").write_text(stage, encoding="utf-8")
    paths = product_paths(camera_count)
    camera_paths = [
        f"/World/Camera_{index}"
        for index in range(camera_count)
    ]

    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            keep_system_alive=True,
            log_level="warning",
            active_cuda_gpus="0",
            use_vulkan=True,
        )
    )
    renderer.open_usd_from_string(stage)
    upload_contract, upload_keepalive = upload_scene(
        renderer,
        scene,
        semantic_group_count=1,
    )
    del scene
    gc.collect()
    torch.cuda.empty_cache()

    start_time = time.perf_counter()
    stats: list[dict[str, float]] = []
    if args.ovrtx_camera_layout == "static-all":
        static_batch_size = min(
            args.ovrtx_static_render_batch_size,
            args.frames,
        )
        for start in range(0, args.frames, static_batch_size):
            stop = min(start + static_batch_size, args.frames)
            batch_paths = paths[start:stop]
            renderer.reset()
            for _ in range(args.ovrtx_warmup_samples):
                warmup = renderer.step(
                    render_products=set(batch_paths),
                    delta_time=1.0 / args.fps,
                )
                del warmup

            rgba_sum = np.zeros(
                (
                    stop - start,
                    args.height,
                    args.width,
                    4,
                ),
                dtype=np.float32,
            )
            for _ in range(args.ovrtx_temporal_samples):
                products = renderer.step(
                    render_products=set(batch_paths),
                    delta_time=1.0 / args.fps,
                )
                for local_index, path in enumerate(batch_paths):
                    rgba_sum[local_index] += copy_cpu_render_var(
                        products[path].frames[0],
                        "LdrColor",
                    ).astype(np.float32)
                del products
            rgba = rgba_sum / (255.0 * args.ovrtx_temporal_samples)
            for local_index in range(stop - start):
                frame_index = start + local_index
                stats.append(
                    save_rgb_frame(
                        frames_dir / f"frame-{frame_index:06d}.png",
                        rgba[local_index, ..., :3],
                    )
                )
            print(
                f"FLYBY_OVRTX_PROGRESS frames={stop}/{args.frames}",
                flush=True,
            )
    else:
        renderer.reset()
        for start in range(0, args.frames, args.chunk_size):
            stop = min(start + args.chunk_size, args.frames)
            batch = stop - start
            chunk_viewmats = padded_chunk(
                viewmats,
                start,
                stop,
                args.chunk_size,
            )
            world_matrices = (
                opencv_viewmats_to_usd_camera_world_matrices(
                    chunk_viewmats
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            if args.ovrtx_reset_between_camera_batches and start > 0:
                renderer.reset()
            if args.ovrtx_camera_write_mode == "static-initial":
                if start > 0:
                    raise ValueError(
                        "static-initial camera mode requires "
                        "frames <= chunk size."
                    )
            else:
                if args.ovrtx_camera_write_mode == "transposed":
                    world_matrices = np.ascontiguousarray(
                        world_matrices.transpose(0, 2, 1)
                    )
                if args.ovrtx_camera_write_mode == "xform-op":
                    renderer.write_attribute(
                        prim_paths=camera_paths,
                        attribute_name="xformOp:transform",
                        tensor=world_matrices,
                    )
                else:
                    renderer.write_attribute(
                        prim_paths=camera_paths,
                        attribute_name="omni:xform",
                        tensor=world_matrices,
                        semantic=ovrtx.Semantic.XFORM_MAT4x4,
                    )
            for _ in range(args.ovrtx_warmup_samples):
                warmup = renderer.step(
                    render_products=set(paths),
                    delta_time=1.0 / args.fps,
                )
                del warmup

            rgba_sum = np.zeros(
                (
                    args.chunk_size,
                    args.height,
                    args.width,
                    4,
                ),
                dtype=np.float32,
            )
            for _ in range(args.ovrtx_temporal_samples):
                products = renderer.step(
                    render_products=set(paths),
                    delta_time=1.0 / args.fps,
                )
                for index, path in enumerate(paths):
                    rgba_sum[index] += copy_cpu_render_var(
                        products[path].frames[0],
                        "LdrColor",
                    ).astype(np.float32)
                del products
            rgba = rgba_sum / (255.0 * args.ovrtx_temporal_samples)
            for local_index in range(batch):
                frame_index = start + local_index
                stats.append(
                    save_rgb_frame(
                        frames_dir / f"frame-{frame_index:06d}.png",
                        rgba[local_index, ..., :3],
                    )
                )
            print(
                f"FLYBY_OVRTX_PROGRESS frames={stop}/{args.frames}",
                flush=True,
            )
    elapsed = time.perf_counter() - start_time
    return {
        "schema_version": "gaussian-flyby-render/v1",
        "renderer": "ovrtx",
        "gaussian_model": "stochastic-3d-ray-probabilistic-first-hit",
        "scene": scene_metadata,
        "camera_contract_sha256": camera_manifest[
            "camera_contract_sha256"
        ],
        "frames": args.frames,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "chunk_size": args.chunk_size,
        "temporal_samples_per_camera": args.ovrtx_temporal_samples,
        "warmup_samples_per_camera": args.ovrtx_warmup_samples,
        "camera_layout": args.ovrtx_camera_layout,
        "authored_camera_count": camera_count,
        "static_render_batch_size": (
            min(args.ovrtx_static_render_batch_size, args.frames)
            if args.ovrtx_camera_layout == "static-all"
            else None
        ),
        "camera_write_mode": args.ovrtx_camera_write_mode,
        "reset_between_camera_batches": (
            args.ovrtx_reset_between_camera_batches
        ),
        "elapsed_seconds": elapsed,
        "frames_per_second": args.frames / elapsed,
        "upload_contract": upload_contract,
        "upload_keepalive_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in upload_keepalive["buffers"].values()
        ),
        "frame_signal": summarize_frame_stats(stats),
        "visual_demo_only": True,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_root = args.output_root.resolve()
    camera_contract = (
        args.camera_contract.resolve()
        if args.camera_contract is not None
        else args.output_root / "camera-path.npz"
    )
    scene, scene_metadata = load_scene(args)
    if args.trajectory_contract_v1 is not None:
        from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (
            load_trajectory,
        )

        trajectory = load_trajectory(args.trajectory_contract_v1)
        if trajectory.batch != 1:
            raise ValueError("The flyby visualizer requires a B1 camera-trajectory-v1 contract.")
        if (
            trajectory.timesteps != args.frames
            or trajectory.width != args.width
            or trajectory.height != args.height
            or trajectory.scene_sha256 != scene_metadata["sha256"]
        ):
            raise ValueError("Flyby arguments or scene do not match camera-trajectory-v1.")
        viewmats = torch.from_numpy(trajectory.viewmats[:, 0]).to("cuda")
        intrinsics = torch.from_numpy(trajectory.intrinsics[:, 0]).to("cuda")
        camera_manifest = {
            "schema_version": "camera-trajectory-v1-consumer",
            "camera_contract_sha256": trajectory.trajectory_id,
            "route_sha256": trajectory.route_sha256,
            "near_plane": trajectory.near_plane,
            "far_plane": trajectory.far_plane,
            "frames": trajectory.timesteps,
            "width": trajectory.width,
            "height": trajectory.height,
        }
        camera_contract = args.trajectory_contract_v1.resolve()
    elif camera_contract.is_file() and not args.overwrite_camera_contract:
        viewmats, intrinsics, camera_manifest = load_camera_contract(
            args,
            scene_metadata,
            camera_contract,
        )
    else:
        viewmats, intrinsics, camera_manifest = create_camera_contract(
            args,
            scene,
            scene_metadata,
            camera_contract,
        )

    if args.renderer == "custom":
        result = render_custom(
            args,
            scene,
            scene_metadata,
            viewmats,
            intrinsics,
            camera_manifest,
        )
    else:
        result = render_ovrtx(
            args,
            scene,
            scene_metadata,
            viewmats,
            intrinsics,
            camera_manifest,
        )
    result["camera_contract"] = str(camera_contract)
    result["camera_manifest"] = camera_manifest
    result["pass"] = result["frame_signal"]["pass"]
    result_path = args.output_root / args.renderer / "render-manifest.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        flyby_status_marker(result["pass"])
        + " "
        + json.dumps(
            {
                "renderer": args.renderer,
                "result": str(result_path),
                "camera_contract_sha256": camera_manifest[
                    "camera_contract_sha256"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
