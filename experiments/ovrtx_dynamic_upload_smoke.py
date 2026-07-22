#!/usr/bin/env python3
"""Prove direct CUDA tensor upload into an OVRTX Gaussian particle field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ovrtx
import torch
from PIL import Image

from ovrtx_gaussian_smoke import (
    build_scene,
    copy_cpu_render_var,
    render_product_paths,
)


SPLAT_PATH = "/World/Splats"
SH_C0 = 0.28209479177387814


def synthetic_gaussians(count: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    means = torch.randn((count, 3), device="cuda", generator=generator)
    means[:, :2] *= 1.25
    means[:, 2] *= 0.25
    quats = torch.zeros((count, 4), device="cuda", dtype=torch.float32)
    quats[:, 0] = 1.0
    scales = torch.full((count, 3), 0.025, device="cuda", dtype=torch.float32)
    opacities = torch.full((count,), 0.65, device="cuda", dtype=torch.float32)
    rgb = torch.rand((count, 3), device="cuda", generator=generator)
    sh_dc = (rgb - 0.5) / SH_C0
    return {
        "positions": means.contiguous(),
        "orientations": quats.contiguous(),
        "scales": scales.contiguous(),
        "opacities": opacities.contiguous(),
        "radiance:sphericalHarmonicsCoefficients": sh_dc.contiguous(),
    }


def upload_scene(
    renderer: ovrtx.Renderer,
    tensors: dict[str, torch.Tensor],
) -> None:
    torch.cuda.synchronize()
    for attribute_name, tensor in tensors.items():
        renderer.write_array_attribute(
            prim_paths=[SPLAT_PATH],
            attribute_name=attribute_name,
            tensors=[tensor],
            data_access=ovrtx.DataAccess.ASYNC,
        )
    renderer.reset()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gaussians", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene(512, 512, color_aov="ldr", camera_count=1, layout="products")
    (args.output_dir / "scene.usda").write_text(scene, encoding="utf-8")
    products = render_product_paths(1, "products")
    tensors = synthetic_gaussians(args.gaussians, args.seed)

    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            keep_system_alive=True,
            log_level="info",
            active_cuda_gpus="0",
            use_vulkan=True,
        )
    )
    renderer.open_usd_from_string(scene)
    upload_scene(renderer, tensors)

    for _ in range(args.warmup):
        renderer.step(render_products=set(products), delta_time=1.0 / 60.0)
    outputs = renderer.step(render_products=set(products), delta_time=1.0 / 60.0)
    frame = outputs[products[0]].frames[0]
    ldr = copy_cpu_render_var(frame, "LdrColor")
    distance = copy_cpu_render_var(frame, "DistanceToImagePlaneSD")
    Image.fromarray(ldr).save(args.output_dir / "ldr.png")
    np.save(args.output_dir / "distance_to_image_plane.npy", distance)

    rgb = ldr[..., :3]
    valid_distance = distance[np.isfinite(distance) & (distance > 0)]
    summary = {
        "gaussians": args.gaussians,
        "rgb_nonzero_pixels": int(np.count_nonzero(np.any(rgb != 0, axis=-1))),
        "rgb_unique_colors": int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0]),
        "finite_positive_distance_pixels": int(valid_distance.size),
        "cuda_upload": True,
        "tensor_devices": {
            name: str(tensor.device)
            for name, tensor in tensors.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["rgb_nonzero_pixels"] and valid_distance.size else 1


if __name__ == "__main__":
    raise SystemExit(main())
