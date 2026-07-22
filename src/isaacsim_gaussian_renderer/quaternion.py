"""Quaternion component-order conversions at renderer integration boundaries."""

from __future__ import annotations

import torch


def quaternion_wxyz_to_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert canonical 3DGS WXYZ quaternions to Fabric/OVRTX XYZW lanes."""
    if quaternions.ndim < 1 or quaternions.shape[-1] != 4:
        raise ValueError("quaternions must have a final dimension of length 4.")
    return quaternions[..., [1, 2, 3, 0]].contiguous()


def quaternion_xyzw_to_wxyz(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert Fabric/OVRTX XYZW quaternion lanes to canonical 3DGS WXYZ."""
    if quaternions.ndim < 1 or quaternions.shape[-1] != 4:
        raise ValueError("quaternions must have a final dimension of length 4.")
    return quaternions[..., [3, 0, 1, 2]].contiguous()
