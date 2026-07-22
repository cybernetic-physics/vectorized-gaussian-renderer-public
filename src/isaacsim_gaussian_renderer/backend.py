"""Backend protocol for project-owned Gaussian rendering kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class RenderStats:
    """Optional counters emitted by a backend implementation."""

    rendered_views: int
    visible_gaussians: int | None = None
    tile_intersections: int | None = None


@dataclass(frozen=True)
class RenderRequest:
    """Validated tensors passed to the custom-kernel backend."""

    camera_transforms: torch.Tensor
    intrinsics: torch.Tensor
    scene_ids: torch.Tensor
    scene_offsets: torch.Tensor
    outputs: dict[str, torch.Tensor]
    active_camera_ids: torch.Tensor | None = None
    workspace: dict[str, torch.Tensor] = field(default_factory=dict)


class GaussianRendererBackend(Protocol):
    """Injection seam for the project-owned CUDA/C++ kernels.

    Implementations must not advance Isaac simulation state, allocate steady-state
    output tensors, or depend on stock ``gsplat`` rasterization.
    """

    def initialize(self, stage: Any, device: torch.device) -> None:
        """Bind backend resources to an Isaac USD stage and target device."""

    def register_scene(self, scene_id: int, scene: Any, offset: int) -> None:
        """Receive immutable scene registration and packed-scene offset metadata."""

    def allocate_workspace(
        self,
        *,
        max_views: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Allocate reusable non-output work tensors owned by the backend."""

    def update_scene_transforms(self, scene_ids: torch.Tensor, transforms: torch.Tensor) -> None:
        """Update per-scene/environment transforms without duplicating static data."""

    def render(self, request: RenderRequest) -> RenderStats:
        """Render into the prevalidated output tensors in ``request.outputs``."""

    def synchronize(self) -> None:
        """Synchronize backend work for correctness checks and benchmark timing."""

    def shutdown(self) -> None:
        """Release backend-owned resources."""
