"""Verify deterministic depth-tie ordering and bitwise-repeat outputs on CUDA."""

from __future__ import annotations

import argparse
import json

import torch

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import camera_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--gaussians", type=int, default=1_024)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--ray-gaussian-evaluation", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.gaussians <= 0 or args.iterations < 2:
        raise ValueError("batch and gaussians must be positive; iterations must be at least 2.")

    device = torch.device("cuda")
    count = args.gaussians
    means = torch.zeros((count, 3), device=device, dtype=torch.float32)
    means[:, 2] = 5.0
    scales = torch.full((count, 3), 0.03, device=device, dtype=torch.float32)
    rotations = torch.zeros((count, 4), device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.full((count,), 0.2, device=device, dtype=torch.float32)
    features = torch.zeros((count, 3), device=device, dtype=torch.float32)
    features[:, 0] = torch.linspace(1.0, 0.0, count, device=device)
    features[:, 2] = torch.linspace(0.0, 1.0, count, device=device)
    semantic_ids = torch.arange(1_000, 1_000 + count, device=device, dtype=torch.int64)

    one_camera = camera_bundle(1, args.width, args.height, device=device)
    centered_viewmat = torch.eye(4, device=device, dtype=torch.float32)
    centered_viewmat[2, 3] = -4.0
    viewmats = centered_viewmat.repeat(args.batch, 1, 1).contiguous()
    intrinsics = one_camera.intrinsics.repeat(args.batch, 1, 1).contiguous()
    scene_ids = torch.full((args.batch,), 73, device=device, dtype=torch.int64)

    backend = CustomCudaBackend(
        max_visible_records=args.batch * count,
        max_intersections=(
            args.batch
            * count
            * (256 if args.tile_size == 1 else 64)
        ),
        gaussian_support_sigma=3.0,
        covariance_epsilon=(
            0.0 if args.ray_gaussian_evaluation else 0.3
        ),
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        tile_size=args.tile_size,
        output_srgb=False,
        deterministic=True,
    )
    service = RendererService(
        backend,
        height=args.height,
        width=args.width,
        max_views=args.batch,
    )
    service.initialize(stage=None, device=device)
    service.load_scene(
        73,
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        features=features,
        semantic_ids=semantic_ids,
    )
    outputs = service.render(viewmats, intrinsics, scene_ids)
    service.synchronize()
    reference = {name: tensor.clone() for name, tensor in outputs.items()}

    for _ in range(args.iterations - 1):
        service.render(viewmats, intrinsics, scene_ids, outputs=outputs)
    service.synchronize()
    bitwise_equal = {
        name: bool(torch.equal(reference[name], outputs[name]))
        for name in reference
    }
    if not all(bitwise_equal.values()):
        raise AssertionError(f"Repeated outputs were not bitwise equal: {bitwise_equal}")

    tiles_x = (
        args.width + args.tile_size - 1
    ) // args.tile_size
    center_tile_x = (args.width // 2) // args.tile_size
    center_tile_y = (args.height // 2) // args.tile_size
    center_tile = center_tile_y * tiles_x + center_tile_x
    start = int(backend.workspace["tile_starts"][center_tile].item())
    end = int(backend.workspace["tile_ends"][center_tile].item())
    visible_indices = backend.workspace["values_out"][start:end].to(torch.int64)
    sorted_gaussian_ids = backend.workspace["visible_gaussian_ids"].index_select(
        0,
        visible_indices,
    )
    expected_gaussian_ids = torch.arange(count, device=device, dtype=torch.int32)
    if not torch.equal(sorted_gaussian_ids, expected_gaussian_ids):
        mismatch = torch.nonzero(
            sorted_gaussian_ids != expected_gaussian_ids,
            as_tuple=False,
        ).flatten()
        first_mismatch = int(mismatch[0].item()) if mismatch.numel() else -1
        raise AssertionError(
            "Equal-depth tile records are not ordered by global Gaussian ID: "
            f"range=[{start}, {end}), count={sorted_gaussian_ids.numel()}, "
            f"first_mismatch={first_mismatch}, "
            f"actual_head={sorted_gaussian_ids[:32].tolist()}, "
            f"actual_tail={sorted_gaussian_ids[-32:].tolist()}."
        )

    center_semantic = int(
        outputs["semantic_id"][0, args.height // 2, args.width // 2, 0].item()
    )
    if center_semantic != 1_000:
        raise AssertionError(
            f"Expected the lowest-ID equal-depth Gaussian to dominate; got {center_semantic}."
        )

    counters = backend.check_capacity(synchronize=False)
    tight_backend = CustomCudaBackend(
        max_visible_records=counters["visible_gaussians"],
        max_intersections=counters["tile_intersections"],
        gaussian_support_sigma=3.0,
        covariance_epsilon=(
            0.0 if args.ray_gaussian_evaluation else 0.3
        ),
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        tile_size=args.tile_size,
        output_srgb=False,
        deterministic=True,
    )
    tight_service = RendererService(
        tight_backend,
        height=args.height,
        width=args.width,
        max_views=args.batch,
    )
    tight_service.initialize(stage=None, device=device)
    tight_service.load_scene(
        73,
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        features=features,
        semantic_ids=semantic_ids,
    )
    tight_outputs = tight_service.render(viewmats, intrinsics, scene_ids)
    tight_service.synchronize()
    tight_counters = tight_backend.check_capacity(synchronize=False)
    capacity_invariant = {
        name: bool(torch.equal(reference[name], tight_outputs[name]))
        for name in reference
    }
    if not all(capacity_invariant.values()):
        raise AssertionError(
            "Wide and exact-capacity deterministic outputs differ: "
            f"{capacity_invariant}"
        )
    if tight_counters != counters:
        raise AssertionError(
            "Wide and exact-capacity counters differ: "
            f"wide={counters}, tight={tight_counters}"
        )
    result = {
        "schema_version": "deterministic-cuda-smoke/v1",
        "batch": args.batch,
        "gaussians": count,
        "iterations": args.iterations,
        "tile_size": args.tile_size,
        "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
        "bitwise_equal": bitwise_equal,
        "capacity_invariant_bitwise_equal": capacity_invariant,
        "wide_capacity": {
            "visible_gaussians": args.batch * count,
            "tile_intersections": (
                args.batch
                * count
                * (256 if args.tile_size == 1 else 64)
            ),
        },
        "exact_capacity": {
            "visible_gaussians": counters["visible_gaussians"],
            "tile_intersections": counters["tile_intersections"],
        },
        "equal_depth_records_checked": int(sorted_gaussian_ids.numel()),
        "equal_depth_gaussian_id_order": "ascending",
        "center_semantic": center_semantic,
        "counters": counters,
        "pass": True,
    }
    print("DETERMINISTIC_CUDA_SMOKE_OK", json.dumps(result, sort_keys=True))
    tight_service.shutdown()
    service.shutdown()


if __name__ == "__main__":
    main()
