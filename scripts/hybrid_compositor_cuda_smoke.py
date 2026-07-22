#!/usr/bin/env python3
"""Verify the hybrid tensor compositor executes entirely on CUDA."""

from __future__ import annotations

import json

import torch

from isaacsim_gaussian_renderer.hybrid_compositor import (
    composite_gaussian_and_mesh,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke.")
    device = torch.device("cuda")
    gaussian_rgb = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], device=device
    )
    gaussian_alpha = torch.tensor([[[[1.0], [0.0]]]], device=device)
    gaussian_depth = torch.tensor([[[[3.0], [float("inf")]]]], device=device)
    mesh_rgb = torch.tensor(
        [[[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]], device=device
    )
    mesh_alpha = torch.tensor([[[[0.5], [1.0]]]], device=device)
    mesh_depth = torch.tensor([[[[2.0], [4.0]]]], device=device)
    output = composite_gaussian_and_mesh(
        gaussian_rgb,
        gaussian_alpha,
        gaussian_depth,
        mesh_rgb,
        mesh_alpha,
        mesh_depth,
    )
    torch.cuda.synchronize()
    if any(tensor.device.type != "cuda" for tensor in (
        output.rgb,
        output.alpha,
        output.depth,
        output.mesh_in_front,
    )):
        raise AssertionError("Hybrid compositor copied an output off CUDA.")
    expected = torch.tensor(
        [[[[0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]]], device=device
    )
    torch.testing.assert_close(output.rgb, expected)
    print(
        "HYBRID_COMPOSITOR_CUDA_OK "
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "shape": list(output.rgb.shape),
                "mesh_in_front_pixels": int(output.mesh_in_front.sum().item()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
