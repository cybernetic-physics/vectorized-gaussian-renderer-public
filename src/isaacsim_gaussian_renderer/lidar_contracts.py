"""Strict tensor contracts for the opt-in Gaussian LiDAR service."""

from __future__ import annotations

import torch

from .contracts import ensure_tensor


LIDAR_OUTPUT_SPECS: dict[str, torch.dtype] = {
    "range_m": torch.float32,
    "position_world_m": torch.float32,
    "intensity": torch.float32,
    "semantic_id": torch.int64,
    "valid": torch.bool,
    "time_offset_ns": torch.int64,
    "return_count": torch.int32,
}


def validate_lidar_request(
    ray_directions: torch.Tensor,
    time_offsets_ns: torch.Tensor,
    sensor_to_world: torch.Tensor,
    scene_ids: torch.Tensor,
    *,
    device: torch.device,
    require_cuda: bool,
) -> tuple[int, int]:
    """Validate shapes without synchronizing CUDA tensors back to the host.

    Direction values are checked asynchronously by the native trace kernel so
    the steady-state API keeps current-stream semantics. ``check_errors()``
    reports a non-unit direction after synchronization. CPU contract tests do
    the same value check eagerly.
    """

    ensure_tensor(
        ray_directions,
        name="ray_directions",
        shape=(None, 3),
        dtype=torch.float32,
        device=device,
        require_cuda=require_cuda,
    )
    rays = int(ray_directions.shape[0])
    if rays <= 0:
        raise ValueError("ray_directions must contain at least one ray.")
    if time_offsets_ns.dtype not in (torch.int32, torch.int64):
        raise ValueError("time_offsets_ns must have dtype torch.int32 or torch.int64.")
    ensure_tensor(
        time_offsets_ns,
        name="time_offsets_ns",
        shape=(rays,),
        dtype=time_offsets_ns.dtype,
        device=device,
        require_cuda=require_cuda,
    )
    ensure_tensor(
        sensor_to_world,
        name="sensor_to_world",
        shape=(None, 4, 4),
        dtype=torch.float32,
        device=device,
        require_cuda=require_cuda,
    )
    batch = int(sensor_to_world.shape[0])
    if batch <= 0:
        raise ValueError("sensor_to_world must contain at least one sensor.")
    ensure_tensor(
        scene_ids,
        name="scene_ids",
        shape=(batch,),
        dtype=torch.int64,
        device=device,
        require_cuda=require_cuda,
    )
    if not require_cuda:
        norms = torch.linalg.vector_norm(ray_directions, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=1.0e-4):
            raise ValueError("ray_directions must be normalized within absolute tolerance 1e-4.")
    return batch, rays


def validate_active_sensor_ids(
    active_sensor_ids: torch.Tensor | None,
    *,
    batch: int,
    device: torch.device,
    require_cuda: bool,
) -> torch.Tensor | None:
    if active_sensor_ids is None:
        return None
    ensure_tensor(
        active_sensor_ids,
        name="active_sensor_ids",
        shape=(None,),
        dtype=torch.int64,
        device=device,
        require_cuda=require_cuda,
    )
    if not require_cuda and active_sensor_ids.numel():
        if active_sensor_ids.min() < 0 or active_sensor_ids.max() >= batch:
            raise ValueError("active_sensor_ids contains an index outside the sensor batch.")
        if active_sensor_ids.numel() > 1 and not torch.all(
            active_sensor_ids[1:] > active_sensor_ids[:-1]
        ):
            raise ValueError("active_sensor_ids must be strictly increasing and unique.")
    return active_sensor_ids


def validate_scene_transforms(
    scene_to_world: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
    require_cuda: bool,
) -> torch.Tensor:
    return ensure_tensor(
        scene_to_world,
        name="scene_to_world",
        shape=(batch, 4, 4),
        dtype=torch.float32,
        device=device,
        require_cuda=require_cuda,
    )


def validate_lidar_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    batch: int,
    rays: int,
    returns: int,
    device: torch.device,
    require_cuda: bool,
) -> dict[str, torch.Tensor]:
    required = set(LIDAR_OUTPUT_SPECS)
    missing = sorted(required - set(outputs))
    unknown = sorted(set(outputs) - required)
    if missing:
        raise ValueError(f"Missing LiDAR output tensor(s): {', '.join(missing)}.")
    if unknown:
        raise ValueError(f"Unsupported LiDAR output tensor(s): {', '.join(unknown)}.")
    for name, dtype in LIDAR_OUTPUT_SPECS.items():
        shape = (
            (batch, rays, returns, 3)
            if name == "position_world_m"
            else (batch, rays)
            if name == "return_count"
            else (batch, rays, returns)
        )
        ensure_tensor(
            outputs[name],
            name=f"outputs[{name!r}]",
            shape=shape,
            dtype=dtype,
            device=device,
            require_cuda=require_cuda,
        )
    return outputs
