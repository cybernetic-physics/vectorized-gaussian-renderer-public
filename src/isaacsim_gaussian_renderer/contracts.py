"""Strict tensor contracts for Isaac Sim Gaussian rendering."""

from __future__ import annotations

import torch

OUTPUT_SPECS: dict[str, tuple[int, torch.dtype]] = {
    "rgb": (3, torch.float32),
    "depth": (1, torch.float32),
    "alpha": (1, torch.float32),
    "semantic_id": (1, torch.int64),
}


def ensure_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int | None, ...],
    dtype: torch.dtype | None,
    device: torch.device,
    require_cuda: bool,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if require_cuda and tensor.device.type != "cuda":
        raise ValueError(f"{name} must be CUDA-resident; got device {tensor.device}.")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}; got {tensor.device}.")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}; got {tensor.dtype}.")
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}; got shape {tuple(tensor.shape)}.")
    for index, (actual, expected) in enumerate(zip(tensor.shape, shape, strict=True)):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} dimension {index} must be {expected}; got {actual}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")
    return tensor


def validate_camera_batch(
    camera_transforms: torch.Tensor,
    intrinsics: torch.Tensor,
    scene_ids: torch.Tensor,
    *,
    device: torch.device,
    require_cuda: bool,
) -> int:
    batch = camera_transforms.shape[0]
    ensure_tensor(
        camera_transforms,
        name="camera_transforms",
        shape=(None, 4, 4),
        dtype=torch.float32,
        device=device,
        require_cuda=require_cuda,
    )
    ensure_tensor(
        intrinsics,
        name="intrinsics",
        shape=(batch, 3, 3),
        dtype=torch.float32,
        device=device,
        require_cuda=require_cuda,
    )
    ensure_tensor(
        scene_ids,
        name="scene_ids",
        shape=(batch,),
        dtype=torch.int64,
        device=device,
        require_cuda=require_cuda,
    )
    return batch


def validate_active_camera_ids(
    active_camera_ids: torch.Tensor | None,
    *,
    batch: int,
    device: torch.device,
    require_cuda: bool,
) -> torch.Tensor | None:
    if active_camera_ids is None:
        return None
    ensure_tensor(
        active_camera_ids,
        name="active_camera_ids",
        shape=(None,),
        dtype=torch.int64,
        device=device,
        require_cuda=require_cuda,
    )
    # Reading CUDA min/max in Python would synchronize the render path back to
    # the host. CPU tests retain eager value validation; production backends
    # validate or mask device-resident indices without a host round trip.
    if (
        not require_cuda
        and active_camera_ids.numel()
        and (active_camera_ids.min() < 0 or active_camera_ids.max() >= batch)
    ):
        raise ValueError("active_camera_ids contains an index outside the camera batch.")
    return active_camera_ids


def validate_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    require_cuda: bool,
) -> dict[str, torch.Tensor]:
    if not outputs:
        raise ValueError("At least one output tensor is required.")
    unknown = sorted(set(outputs) - set(OUTPUT_SPECS))
    if unknown:
        raise ValueError(f"Unsupported output tensor(s): {', '.join(unknown)}.")

    for name, tensor in outputs.items():
        channels, dtype = OUTPUT_SPECS[name]
        ensure_tensor(
            tensor,
            name=f"outputs[{name!r}]",
            shape=(batch, height, width, channels),
            dtype=dtype,
            device=device,
            require_cuda=require_cuda,
        )
    return outputs
