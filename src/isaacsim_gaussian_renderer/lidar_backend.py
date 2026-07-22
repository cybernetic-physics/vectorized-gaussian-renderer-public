"""Backend protocol for the opt-in Gaussian LiDAR service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class LidarRenderRequest:
    ray_directions: torch.Tensor
    time_offsets_ns: torch.Tensor
    sensor_to_world: torch.Tensor
    scene_to_world: torch.Tensor
    scene_ids: torch.Tensor
    outputs: dict[str, torch.Tensor]
    returns: int
    active_sensor_ids: torch.Tensor | None = None


class GaussianLidarBackend(Protocol):
    """Allocation-free steady-state backend for direct Gaussian ray tracing."""

    def initialize(self, stage: Any, device: torch.device) -> None: ...

    def register_scene(self, scene_id: int, scene: Any) -> None: ...

    def revise_scene(self, scene_id: int, scene: Any) -> None: ...

    def allocate_workspace(
        self,
        *,
        max_sensors: int,
        max_rays: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]: ...

    def render_lidar(self, request: LidarRenderRequest) -> None: ...

    def synchronize(self) -> None: ...

    def shutdown(self) -> None: ...
