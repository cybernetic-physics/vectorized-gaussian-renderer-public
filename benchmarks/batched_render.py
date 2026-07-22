"""Measure batched gsplat throughput for a shared static scene."""

from __future__ import annotations

import argparse
import time

import torch
from gsplat.rendering import rasterization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cameras", type=int, default=64)
    parser.add_argument("--num-gaussians", type=int, default=100_000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--packed", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(7)

    means = torch.randn((args.num_gaussians, 3), generator=generator, device=device)
    means[:, :2] *= 3.0
    means[:, 2] = means[:, 2].abs() * 1.5 + 4.0
    quats = torch.zeros((args.num_gaussians, 4), device=device)
    quats[:, 0] = 1.0
    scales = torch.full((args.num_gaussians, 3), 0.02, device=device)
    opacities = torch.full((args.num_gaussians,), 0.6, device=device)
    colors = torch.rand((args.num_gaussians, 3), generator=generator, device=device)

    viewmats = torch.eye(4, device=device).repeat(args.num_cameras, 1, 1)
    camera_offsets = torch.linspace(-1.5, 1.5, args.num_cameras, device=device)
    viewmats[:, 0, 3] = -camera_offsets

    focal = 0.9 * args.width
    intrinsics = torch.zeros((args.num_cameras, 3, 3), device=device)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = args.width / 2.0
    intrinsics[:, 1, 2] = args.height / 2.0
    intrinsics[:, 2, 2] = 1.0

    def render_once() -> tuple[torch.Tensor, torch.Tensor]:
        rendered, alpha, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            intrinsics,
            args.width,
            args.height,
            packed=args.packed,
            render_mode="RGB+D",
        )
        return rendered, alpha

    for _ in range(args.warmup):
        render_once()
    torch.cuda.synchronize()

    start = time.perf_counter()
    rendered = alpha = None
    for _ in range(args.iterations):
        rendered, alpha = render_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    milliseconds = elapsed * 1000.0 / args.iterations
    camera_fps = args.num_cameras * args.iterations / elapsed
    megapixels_per_second = (
        args.num_cameras * args.width * args.height * args.iterations / elapsed / 1.0e6
    )
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"cameras={args.num_cameras} gaussians={args.num_gaussians} resolution={args.width}x{args.height}")
    print(f"packed={args.packed} frame_ms={milliseconds:.3f}")
    print(f"camera_fps={camera_fps:.1f} megapixels_per_second={megapixels_per_second:.1f}")
    print(f"peak_allocated_gib={peak_gib:.3f}")
    print(f"render_shape={tuple(rendered.shape)} alpha_shape={tuple(alpha.shape)}")


if __name__ == "__main__":
    main()
