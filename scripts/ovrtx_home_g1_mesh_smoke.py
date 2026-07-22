#!/usr/bin/env python3
"""Prove that OVRTX composes an imported G1 USD mesh with Home Scan splats.

This is deliberately a renderer-boundary smoke, not a claim that the custom
Gaussian CUDA kernel is an OVRTX plug-in.  The production design keeps the
static Gaussian layer in the project kernel and uses a separate RTX/OVRTX mesh
layer, followed by a GPU depth compositor.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path


HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
HOME_SCAN_COUNT = 21_497_908


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--home-scan-path", type=Path, default=Path("/workspace/datasets/home-scan-lod0.ply")
    )
    parser.add_argument("--g1-usda", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--min-rgb-delta-pixels",
        type=int,
        default=64,
        help="Minimum visible RGB footprint required to call mesh composition usable.",
    )
    return parser.parse_args()


def png_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.warmup < 0
        or args.min_rgb_delta_pixels <= 0
    ):
        raise ValueError("Resolution and RGB-delta threshold must be positive.")
    if not args.g1_usda.is_file():
        raise FileNotFoundError(args.g1_usda)

    # Import project helpers only after the headless runtime has supplied its
    # CUDA/OVRTX paths through scripts/remote_env.sh.
    import numpy as np
    import ovrtx
    import torch
    from PIL import Image

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        # Match the established Home Scan control's compact spatial groups.
        # They avoid the known single-field headless LDR edge case while still
        # keeping the stage compact enough for a mesh-composition smoke.
        "semantic_ids": compact_semantic_ids(
            spatial_semantic_ids(
                canonical.means,
                grid=(2, 2, 2),
                dtype=torch.int64,
            ),
            dtype=torch.int64,
        )[0],
    }
    del canonical, raw
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    position_min = scene["means"].amin(dim=0)
    position_max = scene["means"].amax(dim=0)
    center = (position_min + position_max) * 0.5
    radius = float((torch.linalg.vector_norm(position_max - position_min) * 0.5).item())
    cameras = axis_aligned_exterior_camera_bundle(
        center=center,
        bounding_radius=radius,
        batch_size=1,
        width=args.width,
        height=args.height,
        focal_scale=0.9,
        fit_margin=1.15,
        camera_path="home-scan-exterior-g1-mesh-smoke",
        bounds_policy="full-componentwise-min-max",
    )
    center_values = center.detach().cpu().tolist()
    # Imported G1 is metres-scale.  Place it beside, rather than inside, the
    # scan bounds so its mesh contribution is independently observable.
    robot_translate = (
        center_values[0] + radius * 1.12,
        center_values[1],
        center_values[2],
    )
    robot_ref = args.g1_usda.resolve().as_posix()
    semantic_group_count = int(scene["semantic_ids"].max().item()) + 1
    semantic_counts = torch.bincount(
        scene["semantic_ids"], minlength=semantic_group_count
    ).detach().cpu().tolist()
    prim_semantic_ids = splat_prim_semantic_ids(semantic_counts)
    robot_world_prim = f'''    def Xform "G1" (
        prepend references = @{robot_ref}@
    )
    {{
        double3 xformOp:translate = ({robot_translate[0]:.9g}, {robot_translate[1]:.9g}, {robot_translate[2]:.9g})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}'''

    def render(label: str, world_prims: str) -> tuple[np.ndarray, Path]:
        stage = build_stage(
            1,
            args.width,
            args.height,
            0.9,
            cameras.viewmats,
            "none",
            float(cameras.manifest["near_plane"]),
            float(cameras.manifest["far_plane"]),
            semantic_group_count=semantic_group_count,
            splat_prim_semantic_ids_override=prim_semantic_ids,
            projection_mode="perspective",
            sorting_mode="zDepth",
            world_prims=world_prims,
        )
        stage_path = args.output_dir / f"{label}.usda"
        stage_path.write_text(stage, encoding="utf-8")
        renderer = ovrtx.Renderer(
            ovrtx.RendererConfig(
                keep_system_alive=True, log_level="info", active_cuda_gpus="0", use_vulkan=True
            )
        )
        upload_keepalive = None
        try:
            renderer.open_usd_from_string(stage)
            # OVRTX copies asynchronously; retain every grouped CUDA view until
            # the render has completed or the Gaussian field can disappear.
            _upload_contract, upload_keepalive = upload_scene(
                renderer, scene, semantic_group_count=semantic_group_count
            )
            paths = product_paths(1)
            for _ in range(args.warmup):
                renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
            outputs = renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
            rgba = copy_cpu_render_var(outputs[paths[0]].frames[0], "LdrColor").copy()
        finally:
            del upload_keepalive
            # OVRTX 0.3 owns the Kit system and exposes no close() method.
            # Dropping the renderer before the next isolated stage releases
            # its Python-side handles without pretending there is a lifecycle
            # API that does not exist in this runtime.
            del renderer
            gc.collect()
            torch.cuda.empty_cache()
        image_path = args.output_dir / f"{label}.png"
        Image.fromarray(rgba).save(image_path)
        return rgba, image_path

    start = time.perf_counter()
    baseline, baseline_path = render("home-scan-only", "")
    with_g1, g1_path = render("home-scan-plus-g1", robot_world_prim)
    rgb_delta_pixels = int(np.count_nonzero(np.any(baseline[..., :3] != with_g1[..., :3], axis=-1)))
    alpha_delta_pixels = int(np.count_nonzero(baseline[..., 3] != with_g1[..., 3]))
    # A few alpha-only pixels can arise from headless accumulation noise.  A
    # usable hybrid render requires a visible, nontrivial robot RGB footprint.
    passed = rgb_delta_pixels >= args.min_rgb_delta_pixels
    report = {
        "schema_version": "ovrtx-home-g1-mesh-smoke/v1",
        "pass": passed,
        "scope": "OVRTX mesh plus Home Scan Gaussian visual composition; not a custom-kernel plug-in or physics-contact test.",
        "scene": {"gaussian_count": HOME_SCAN_COUNT, "sha256": checksum},
        "g1_usda": str(args.g1_usda.resolve()),
        "g1_world_translate": robot_translate,
        "projection_mode": "perspective",
        "resolution": [args.width, args.height],
        "warmup_frames": args.warmup,
        "min_rgb_delta_pixels": args.min_rgb_delta_pixels,
        "rgb_delta_pixels": rgb_delta_pixels,
        "alpha_delta_pixels": alpha_delta_pixels,
        "artifacts": {
            "baseline_png": str(baseline_path),
            "baseline_png_sha256": png_sha256(baseline_path),
            "with_g1_png": str(g1_path),
            "with_g1_png_sha256": png_sha256(g1_path),
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise AssertionError("OVRTX G1 mesh did not change the Home Scan render.")
    print("OVRTX_HOME_G1_MESH_SMOKE_OK " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
