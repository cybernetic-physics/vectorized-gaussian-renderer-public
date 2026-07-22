"""GPU tensor compositor for static Gaussians and dynamic mesh layers."""

from __future__ import annotations


from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HybridComposite:
    """Straight-alpha RGB, alpha, depth, and front-layer selection tensors."""

    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    mesh_in_front: torch.Tensor


def composite_gaussian_and_mesh(
    gaussian_rgb: torch.Tensor,
    gaussian_alpha: torch.Tensor,
    gaussian_depth: torch.Tensor,
    mesh_rgb: torch.Tensor,
    mesh_alpha: torch.Tensor,
    mesh_depth: torch.Tensor,
    *,
    min_alpha: float = 1.0e-4,
    depth_epsilon: float = 1.0e-4,
) -> HybridComposite:
    """Depth-order and straight-alpha composite two GPU-resident render layers.

    Every tensor uses the project output layout: RGB is ``[B,H,W,3]`` and
    alpha/depth are ``[B,H,W,1]``.  Depth is positive camera distance for
    foreground pixels; invalid/empty pixels are selected by alpha rather than
    relying on a renderer-specific sentinel depth.
    """
    _validate_layer(gaussian_rgb, gaussian_alpha, gaussian_depth, "gaussian")
    _validate_layer(mesh_rgb, mesh_alpha, mesh_depth, "mesh")
    if gaussian_rgb.shape != mesh_rgb.shape:
        raise ValueError("Gaussian and mesh RGB tensors must have equal shapes.")
    if gaussian_rgb.device != mesh_rgb.device:
        raise ValueError("Gaussian and mesh tensors must share one device.")
    if not 0.0 <= min_alpha <= 1.0:
        raise ValueError("min_alpha must be in [0, 1].")
    if depth_epsilon < 0.0:
        raise ValueError("depth_epsilon must be non-negative.")

    gaussian_valid = (
        (gaussian_alpha > min_alpha)
        & torch.isfinite(gaussian_alpha)
        & torch.isfinite(gaussian_depth)
    )
    mesh_valid = (
        (mesh_alpha > min_alpha)
        & torch.isfinite(mesh_alpha)
        & torch.isfinite(mesh_depth)
    )
    mesh_in_front = mesh_valid & (
        ~gaussian_valid | (mesh_depth + depth_epsilon < gaussian_depth)
    )

    # A layer rejected by the validity contract must not contribute as the
    # nominal back layer. Merely excluding it from depth ordering is
    # insufficient: its nonzero alpha (or non-finite RGB) would otherwise be
    # blended behind the valid layer.
    gaussian_rgb_valid = torch.where(
        gaussian_valid.expand_as(gaussian_rgb),
        gaussian_rgb,
        torch.zeros_like(gaussian_rgb),
    )
    mesh_rgb_valid = torch.where(
        mesh_valid.expand_as(mesh_rgb),
        mesh_rgb,
        torch.zeros_like(mesh_rgb),
    )
    gaussian_alpha_valid = torch.where(
        gaussian_valid,
        gaussian_alpha,
        torch.zeros_like(gaussian_alpha),
    )
    mesh_alpha_valid = torch.where(
        mesh_valid,
        mesh_alpha,
        torch.zeros_like(mesh_alpha),
    )

    front_rgb = torch.where(
        mesh_in_front.expand_as(mesh_rgb), mesh_rgb_valid, gaussian_rgb_valid
    )
    front_alpha = torch.where(
        mesh_in_front, mesh_alpha_valid, gaussian_alpha_valid
    )
    back_rgb = torch.where(
        mesh_in_front.expand_as(mesh_rgb), gaussian_rgb_valid, mesh_rgb_valid
    )
    back_alpha = torch.where(
        mesh_in_front, gaussian_alpha_valid, mesh_alpha_valid
    )
    alpha = front_alpha + back_alpha * (1.0 - front_alpha)
    premultiplied_rgb = (
        front_rgb * front_alpha
        + back_rgb * back_alpha * (1.0 - front_alpha)
    )
    rgb = torch.where(
        alpha > min_alpha,
        premultiplied_rgb / alpha.clamp_min(torch.finfo(alpha.dtype).eps),
        torch.zeros_like(premultiplied_rgb),
    )
    depth = torch.where(
        mesh_in_front,
        mesh_depth,
        gaussian_depth,
    )
    depth = torch.where(
        gaussian_valid | mesh_valid,
        depth,
        torch.full_like(depth, float("inf")),
    )
    return HybridComposite(
        rgb=rgb.contiguous(),
        alpha=alpha.contiguous(),
        depth=depth.contiguous(),
        mesh_in_front=mesh_in_front.contiguous(),
    )


def _validate_layer(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    name: str,
) -> None:
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or not torch.is_floating_point(rgb):
        raise ValueError(f"{name}_rgb must be floating [B,H,W,3].")
    expected = (*rgb.shape[:3], 1)
    for tensor, suffix in ((alpha, "alpha"), (depth, "depth")):
        if tensor.shape != expected or not torch.is_floating_point(tensor):
            raise ValueError(
                f"{name}_{suffix} must be floating [B,H,W,1], got {tuple(tensor.shape)}."
            )
        if tensor.device != rgb.device:
            raise ValueError(f"{name}_{suffix} must share the RGB device.")
