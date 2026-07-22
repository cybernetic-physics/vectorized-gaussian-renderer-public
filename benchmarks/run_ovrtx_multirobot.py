#!/usr/bin/env python3
"""Pure-OVRTX multi-robot scaling lane: Home Scan splats + N G1 meshes rendered
in ONE unified ray-traced OVRTX scene (no Isaac RTX, no compositor).

Counterpart to run_g1_hybrid_multirobot.py. N cameras (one per env), one G1
mesh placed in front of each camera. Two passes isolate the robot cost:
  1. splats-only        (control)
  2. splats + N robots  (unified hybrid)

Scope note (honest): OVRTX 0.3 exposes no per-frame USD transform update in
this harness, so robots are STATIC — acceleration structures may cache, making
this an OPTIMISTIC lower bound versus walking robots. Robot pose variation
per-env is approximated by distinct placements; joint-level animation is out of
scope without physics.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
HOME_SCAN_COUNT = 21_497_908


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--home-scan-path", type=Path, default=Path("/workspace/datasets/home-scan-lod0.ply"))
    p.add_argument("--g1-usda", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--min-rgb-delta-pixels", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.g1_usda.is_file():
        raise FileNotFoundError(args.g1_usda)

    import numpy as np
    import ovrtx
    import torch

    from benchmarks.run_ovrtx import (
        build_stage,
        copy_cpu_render_var,
        file_sha256,
        product_paths,
        splat_prim_semantic_ids,
        upload_scene,
    )
    from isaacsim_gaussian_renderer.benchmark_manifest import (
        axis_aligned_exterior_camera_bundle,
        compact_semantic_ids,
        spatial_semantic_ids,
    )
    from isaacsim_gaussian_renderer.ply_loader import (
        canonicalize_3dgs_scene,
        load_ply_to_gaussians,
    )

    checksum = file_sha256(args.home_scan_path)
    if checksum != HOME_SCAN_SHA256:
        raise ValueError(f"Home Scan SHA-256 mismatch: {checksum}")
    raw = load_ply_to_gaussians(args.home_scan_path)
    if raw.count != HOME_SCAN_COUNT:
        raise ValueError(f"Home Scan count mismatch: {raw.count}")
    canonical = canonicalize_3dgs_scene(raw, device="cuda")
    scene = {
        "means": canonical.means,
        "quats": canonical.rotations,
        "scales": canonical.scales,
        "opacities": canonical.opacities,
        "colors": canonical.features[:, :3].contiguous(),
        "semantic_ids": compact_semantic_ids(
            spatial_semantic_ids(canonical.means, grid=(2, 2, 2), dtype=torch.int64),
            dtype=torch.int64,
        )[0],
    }
    position_min = canonical.means.amin(dim=0)
    position_max = canonical.means.amax(dim=0)
    del canonical, raw
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    center = (position_min + position_max) * 0.5
    radius = float((torch.linalg.vector_norm(position_max - position_min) * 0.5).item())
    cameras = axis_aligned_exterior_camera_bundle(
        center=center, bounding_radius=radius, batch_size=args.envs,
        width=args.width, height=args.height, focal_scale=0.9, fit_margin=1.15,
        camera_path="home-scan-ovrtx-multirobot", bounds_policy="full-componentwise-min-max",
    )
    semantic_group_count = int(scene["semantic_ids"].max().item()) + 1
    semantic_counts = torch.bincount(
        scene["semantic_ids"], minlength=semantic_group_count
    ).detach().cpu().tolist()
    prim_semantic_ids = splat_prim_semantic_ids(semantic_counts)

    # One robot in front of each env camera (same placement as the hybrid lane).
    robot_ref = args.g1_usda.resolve().as_posix()
    robot_prims = []
    for i in range(args.envs):
        vm = cameras.viewmats[i]
        cam_center = -vm[:3, :3].transpose(0, 1) @ vm[:3, 3]
        base = (cam_center + vm[2, :3] * 2.5 + vm[1, :3] * 0.35).detach().cpu().tolist()
        robot_prims.append(
            f'''    def Xform "G1_{i:03d}" (
        prepend references = @{robot_ref}@
    )
    {{
        double3 xformOp:translate = ({base[0]:.9g}, {base[1]:.9g}, {base[2]:.9g})
        float3 xformOp:rotateXYZ = (90, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}'''
        )
    robots_world_prims = "\n".join(robot_prims)

    def run_pass(label: str, world_prims: str) -> dict:
        stage = build_stage(
            args.envs, args.width, args.height, 0.9, cameras.viewmats, "none",
            float(cameras.manifest["near_plane"]), float(cameras.manifest["far_plane"]),
            semantic_group_count=semantic_group_count,
            splat_prim_semantic_ids_override=prim_semantic_ids,
            projection_mode="perspective", sorting_mode="zDepth",
            world_prims=world_prims,
        )
        renderer = ovrtx.Renderer(
            ovrtx.RendererConfig(
                keep_system_alive=True, log_level="info", active_cuda_gpus="0", use_vulkan=True
            )
        )
        upload_keepalive = None
        try:
            renderer.open_usd_from_string(stage)
            _contract, upload_keepalive = upload_scene(
                renderer, scene, semantic_group_count=semantic_group_count
            )
            paths = product_paths(args.envs)
            for _ in range(args.warmup):
                renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
            torch.cuda.synchronize()
            samples_ms = []
            for _ in range(args.frames):
                t0 = time.perf_counter()
                outputs = renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
                torch.cuda.synchronize()
                samples_ms.append((time.perf_counter() - t0) * 1000.0)
                del outputs
            outputs = renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
            rgba0 = copy_cpu_render_var(outputs[paths[0]].frames[0], "LdrColor").copy()
            del outputs
        finally:
            del upload_keepalive
            del renderer
            gc.collect()
            torch.cuda.empty_cache()
        mean_ms = sum(samples_ms) / len(samples_ms)
        print(f"OVRTX_MULTIROBOT_PASS {label} mean_ms={mean_ms:.1f}", flush=True)
        return {
            "label": label,
            "mean_frame_ms": mean_ms,
            "samples_ms": samples_ms,
            "fps": 1000.0 / mean_ms,
            "images_per_second": args.envs * 1000.0 / mean_ms,
            "_rgba0": rgba0,
        }

    splats_only = run_pass("splats-only", "")
    with_robots = run_pass("splats-plus-robots", robots_world_prims)

    delta = int(
        np.count_nonzero(
            np.any(splats_only["_rgba0"][..., :3] != with_robots["_rgba0"][..., :3], axis=-1)
        )
    )
    robots_visible = delta >= args.min_rgb_delta_pixels
    for entry in (splats_only, with_robots):
        entry.pop("_rgba0")

    report = {
        "schema_version": "ovrtx-multirobot/v1",
        "pass": bool(robots_visible),
        "envs": args.envs,
        "robots": args.envs,
        "robots_static": True,
        "scope": (
            "Unified OVRTX scene (splats + G1 meshes, no compositor). Robots are "
            "static: accel structures may cache, so this is an optimistic lower "
            "bound vs walking robots."
        ),
        "resolution": [args.width, args.height],
        "rgb_delta_pixels_cam0": delta,
        "splats_only": splats_only,
        "splats_plus_robots": with_robots,
        "robot_increment_ms": with_robots["mean_frame_ms"] - splats_only["mean_frame_ms"],
        "robot_increment_ms_per_env": (
            (with_robots["mean_frame_ms"] - splats_only["mean_frame_ms"]) / max(args.envs, 1)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not robots_visible:
        raise AssertionError(f"Robots not visible in OVRTX render (delta={delta}).")
    print("OVRTX_MULTIROBOT_OK " + json.dumps(
        {"envs": args.envs,
         "splats_only_ms": round(splats_only["mean_frame_ms"], 1),
         "with_robots_ms": round(with_robots["mean_frame_ms"], 1),
         "images_per_second": round(with_robots["images_per_second"], 1)}))


if __name__ == "__main__":
    main()
