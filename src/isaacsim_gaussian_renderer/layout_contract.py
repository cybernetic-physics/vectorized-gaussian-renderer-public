"""Tensor layout checks for the custom Gaussian renderer boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CanonicalSceneLayout:
    """Canonical structure-of-arrays scene tensors accepted by the CUDA backend."""

    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    features: torch.Tensor
    semantic_ids: torch.Tensor
    scene_offsets: torch.Tensor

    def validate(self) -> None:
        count = _expect_2d_last(self.means, "means", 3)
        _expect_2d_last(self.quats, "quats", 4, count)
        _expect_2d_last(self.scales, "scales", 3, count)
        _expect_1d(self.opacities, "opacities", count)
        _expect_rank_at_least(self.features, "features", 2)
        if self.features.shape[0] != count:
            raise ValueError(f"features first dimension must match gaussian count {count}, got {self.features.shape}.")
        _expect_1d(self.semantic_ids, "semantic_ids", count)
        _expect_1d(self.scene_offsets, "scene_offsets")
        if self.scene_offsets.numel() < 2:
            raise ValueError("scene_offsets must contain at least [0, G].")
        if self.scene_offsets.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"scene_offsets must be int32 or int64, got {self.scene_offsets.dtype}.")
        offsets = self.scene_offsets.detach().cpu()
        if int(offsets[0]) != 0 or int(offsets[-1]) != count:
            raise ValueError(f"scene_offsets must start at 0 and end at gaussian count {count}.")
        if not torch.all(offsets[1:] >= offsets[:-1]):
            raise ValueError("scene_offsets must be monotonically nondecreasing.")
        _expect_same_device(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            self.features,
            self.semantic_ids,
            self.scene_offsets,
        )
        _expect_contiguous(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            self.features,
            self.semantic_ids,
            self.scene_offsets,
        )


@dataclass(frozen=True)
class RenderBatchLayout:
    """Per-render camera, transform, selection, and output tensors."""

    viewmats: torch.Tensor
    intrinsics: torch.Tensor
    env_xforms: torch.Tensor
    scene_ids: torch.Tensor
    active_camera_ids: torch.Tensor | None
    rgb: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    semantic_id: torch.Tensor

    def validate(self) -> None:
        batch = _expect_3d_last2(self.viewmats, "viewmats", 4, 4)
        _expect_3d_last2(self.intrinsics, "intrinsics", 3, 3, batch)
        _expect_3d_last2(self.env_xforms, "env_xforms", 4, 4, batch)
        _expect_1d(self.scene_ids, "scene_ids", batch)
        if self.active_camera_ids is not None:
            _expect_1d(self.active_camera_ids, "active_camera_ids")
        _expect_output(self.rgb, "rgb", batch, channels=3, floating=True)
        height, width = self.rgb.shape[1:3]
        _expect_output(self.depth, "depth", batch, height, width, channels=1, floating=True)
        _expect_output(self.alpha, "alpha", batch, height, width, channels=1, floating=True)
        _expect_output(self.semantic_id, "semantic_id", batch, height, width, channels=1, floating=False)
        tensors = [
            self.viewmats,
            self.intrinsics,
            self.env_xforms,
            self.scene_ids,
            self.rgb,
            self.depth,
            self.alpha,
            self.semantic_id,
        ]
        if self.active_camera_ids is not None:
            tensors.append(self.active_camera_ids)
        _expect_same_device(*tensors)
        _expect_contiguous(*tensors)


def _expect_rank_at_least(tensor: torch.Tensor, name: str, rank: int) -> None:
    if tensor.dim() < rank:
        raise ValueError(f"{name} must have rank >= {rank}, got {tuple(tensor.shape)}.")


def _expect_1d(tensor: torch.Tensor, name: str, length: int | None = None) -> int:
    if tensor.dim() != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(tensor.shape)}.")
    if length is not None and tensor.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {tuple(tensor.shape)}.")
    return int(tensor.shape[0])


def _expect_2d_last(tensor: torch.Tensor, name: str, last: int, rows: int | None = None) -> int:
    if tensor.dim() != 2 or tensor.shape[-1] != last:
        raise ValueError(f"{name} must have shape [N, {last}], got {tuple(tensor.shape)}.")
    if rows is not None and tensor.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got {tuple(tensor.shape)}.")
    return int(tensor.shape[0])


def _expect_3d_last2(tensor: torch.Tensor, name: str, d0: int, d1: int, batch: int | None = None) -> int:
    if tensor.dim() != 3 or tensor.shape[-2:] != (d0, d1):
        raise ValueError(f"{name} must have shape [B, {d0}, {d1}], got {tuple(tensor.shape)}.")
    if batch is not None and tensor.shape[0] != batch:
        raise ValueError(f"{name} must have batch {batch}, got {tuple(tensor.shape)}.")
    return int(tensor.shape[0])


def _expect_output(
    tensor: torch.Tensor,
    name: str,
    batch: int,
    height: int | None = None,
    width: int | None = None,
    *,
    channels: int,
    floating: bool,
) -> None:
    if tensor.dim() != 4 or tensor.shape[0] != batch or tensor.shape[-1] != channels:
        raise ValueError(f"{name} must have shape [B, H, W, {channels}], got {tuple(tensor.shape)}.")
    if height is not None and tensor.shape[1] != height:
        raise ValueError(f"{name} height must be {height}, got {tuple(tensor.shape)}.")
    if width is not None and tensor.shape[2] != width:
        raise ValueError(f"{name} width must be {width}, got {tuple(tensor.shape)}.")
    if floating and not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}.")
    if not floating and tensor.dtype not in (torch.int32, torch.int64, torch.uint32):
        raise TypeError(f"{name} must be int32, int64, or uint32, got {tensor.dtype}.")


def _expect_same_device(*tensors: torch.Tensor) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"all tensors must be on the same device, got {sorted(map(str, devices))}.")


def _expect_contiguous(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError(f"tensor with shape {tuple(tensor.shape)} must be contiguous.")
