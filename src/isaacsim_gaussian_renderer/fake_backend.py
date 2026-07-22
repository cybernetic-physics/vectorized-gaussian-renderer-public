"""Deterministic backend used only by smoke tests and local contract tests."""

from __future__ import annotations

from typing import Any

import torch

from .backend import RenderRequest, RenderStats


class DeterministicFakeBackend:
    """Backend double that validates lifecycle without rasterizing Gaussians."""

    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_called = False
        self.render_calls = 0
        self.registered_offsets: dict[int, int] = {}
        self.last_request: RenderRequest | None = None

    def initialize(self, stage: Any, device: torch.device) -> None:
        self.initialized = True
        self.device = device
        self.stage = stage

    def register_scene(self, scene_id: int, scene: Any, offset: int) -> None:
        self.registered_offsets[scene_id] = offset

    def allocate_workspace(
        self,
        *,
        max_views: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "visible_counts": torch.empty((max_views,), device=device, dtype=torch.int32),
            "tile_counts": torch.empty((max_views,), device=device, dtype=torch.int32),
        }

    def update_scene_transforms(self, scene_ids: torch.Tensor, transforms: torch.Tensor) -> None:
        self.last_scene_ids = scene_ids
        self.last_scene_transforms = transforms

    def render(self, request: RenderRequest) -> RenderStats:
        self.render_calls += 1
        self.last_request = request
        active = (
            request.active_camera_ids
            if request.active_camera_ids is not None
            else torch.arange(request.scene_ids.shape[0], device=request.scene_ids.device, dtype=torch.int64)
        )
        if request.active_camera_ids is not None:
            # Match CudaRendererBackend: active-subset renders still define
            # every inactive row, preventing stale observations when callers
            # reuse or alternate output buffers.
            if "rgb" in request.outputs:
                request.outputs["rgb"].zero_()
            if "depth" in request.outputs:
                request.outputs["depth"].fill_(float("inf"))
            if "alpha" in request.outputs:
                request.outputs["alpha"].zero_()
            if "semantic_id" in request.outputs:
                request.outputs["semantic_id"].fill_(-1)
        scene_values = request.scene_ids.index_select(0, active)
        selected_views = int(active.numel())
        if "rgb" in request.outputs:
            output = request.outputs["rgb"]
            values = (scene_values.to(output.dtype) / 255.0).view(-1, 1, 1, 1)
            output.index_copy_(0, active, values.expand(-1, output.shape[1], output.shape[2], output.shape[3]))
        if "depth" in request.outputs:
            output = request.outputs["depth"]
            values = (scene_values.to(output.dtype) + 1.0).view(-1, 1, 1, 1)
            output.index_copy_(0, active, values.expand(-1, output.shape[1], output.shape[2], output.shape[3]))
        if "alpha" in request.outputs:
            output = request.outputs["alpha"]
            values = torch.ones(
                (selected_views, output.shape[1], output.shape[2], output.shape[3]),
                device=output.device,
                dtype=output.dtype,
            )
            output.index_copy_(0, active, values)
        if "semantic_id" in request.outputs:
            output = request.outputs["semantic_id"]
            values = scene_values.to(output.dtype).view(-1, 1, 1, 1)
            output.index_copy_(0, active, values.expand(-1, output.shape[1], output.shape[2], output.shape[3]))
        return RenderStats(rendered_views=int(active.numel()))

    def synchronize(self) -> None:
        self.synchronized = True

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.initialized = False
