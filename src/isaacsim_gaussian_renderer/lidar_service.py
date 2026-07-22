"""Public, explicitly constructed Gaussian LiDAR service."""

from __future__ import annotations

from typing import Any

import torch

from .lidar_backend import GaussianLidarBackend, LidarRenderRequest
from .lidar_contracts import (
    LIDAR_OUTPUT_SPECS,
    validate_active_sensor_ids,
    validate_lidar_outputs,
    validate_lidar_request,
    validate_scene_transforms,
)
from .lidar_scene import GaussianLidarScene, LidarSceneRegistry


class GaussianLidarService:
    """Trace shared sensor-local rays against registered static Gaussian scenes.

    Every call submits fresh tracing and rewrites every output. The service has
    no cadence, cross-call output cache, background worker, or private stream.
    Static per-scene LBVHs are legal only under explicit scene revisions.
    """

    mode = "gaussian_surface"

    def __init__(
        self,
        backend: GaussianLidarBackend,
        *,
        max_sensors: int,
        max_rays: int,
        returns: int = 1,
        allow_cpu_for_tests: bool = False,
    ) -> None:
        if max_sensors <= 0 or max_rays <= 0:
            raise ValueError("max_sensors and max_rays must be positive.")
        if returns not in (1, 2):
            raise ValueError("returns must be exactly 1 or 2.")
        self.backend = backend
        self.max_sensors = max_sensors
        self.max_rays = max_rays
        self.returns = returns
        self.allow_cpu_for_tests = allow_cpu_for_tests
        self.device: torch.device | None = None
        self.stage: Any | None = None
        self.registry = LidarSceneRegistry()
        self.workspace: dict[str, torch.Tensor] = {}
        self._owned_outputs: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._initialized = False

    def initialize(self, stage: Any, device: torch.device | str) -> None:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda" and not self.allow_cpu_for_tests:
            raise ValueError(f"GaussianLidarService requires a CUDA device; got {device}.")
        self.stage = stage
        self.device = device
        self.backend.initialize(stage, device)
        self.workspace = self.backend.allocate_workspace(
            max_sensors=self.max_sensors,
            max_rays=self.max_rays,
            device=device,
        )
        self._initialized = True

    def load_scene(
        self,
        scene_id: int,
        *,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        semantic_ids: torch.Tensor,
        reflectivity: torch.Tensor | None = None,
        surface_confidence: torch.Tensor | None = None,
        revision: int = 0,
    ) -> None:
        self._require_initialized()
        assert self.device is not None
        scene = GaussianLidarScene(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            semantic_ids=semantic_ids,
            reflectivity=reflectivity,
            surface_confidence=surface_confidence,
            revision=revision,
        ).validate(device=self.device, require_cuda=not self.allow_cpu_for_tests)
        self.registry.register(scene_id, scene)
        self.backend.register_scene(scene_id, scene)

    def revise_scene(self, scene_id: int, *, revision: int) -> None:
        """Explicitly rebuild the scene LBVH after same-storage geometry edits."""

        self._require_initialized()
        assert self.device is not None
        scene = self.registry.revise(
            scene_id,
            revision,
            validate=lambda candidate: candidate.validate(
                device=self.device,
                require_cuda=not self.allow_cpu_for_tests,
            ),
        )
        self.backend.revise_scene(scene_id, scene)

    def render_lidar(
        self,
        ray_directions: torch.Tensor,
        time_offsets_ns: torch.Tensor,
        sensor_to_world: torch.Tensor,
        scene_ids: torch.Tensor,
        *,
        scene_to_world: torch.Tensor | None = None,
        active_sensor_ids: torch.Tensor | None = None,
        outputs: dict[str, torch.Tensor] | None = None,
        returns: int | None = None,
    ) -> dict[str, torch.Tensor]:
        self._require_initialized()
        assert self.device is not None
        selected_returns = self.returns if returns is None else returns
        if selected_returns not in (1, 2):
            raise ValueError("returns must be exactly 1 or 2.")
        require_cuda = not self.allow_cpu_for_tests
        batch, rays = validate_lidar_request(
            ray_directions,
            time_offsets_ns,
            sensor_to_world,
            scene_ids,
            device=self.device,
            require_cuda=require_cuda,
        )
        if batch > self.max_sensors or rays > self.max_rays:
            raise ValueError(
                f"Request B={batch}, R={rays} exceeds configured "
                f"B={self.max_sensors}, R={self.max_rays}."
            )
        active_sensor_ids = validate_active_sensor_ids(
            active_sensor_ids,
            batch=batch,
            device=self.device,
            require_cuda=require_cuda,
        )
        if not require_cuda:
            self.registry.require_scene_ids(scene_ids)
        if scene_to_world is None:
            scene_to_world = self.workspace["identity_scene_to_world"][:batch]
        else:
            validate_scene_transforms(
                scene_to_world,
                batch=batch,
                device=self.device,
                require_cuda=require_cuda,
            )
        target_outputs = (
            outputs
            if outputs is not None
            else self._owned_output_set(batch, rays, selected_returns)
        )
        validate_lidar_outputs(
            target_outputs,
            batch=batch,
            rays=rays,
            returns=selected_returns,
            device=self.device,
            require_cuda=require_cuda,
        )
        request = LidarRenderRequest(
            ray_directions=ray_directions,
            time_offsets_ns=time_offsets_ns,
            sensor_to_world=sensor_to_world,
            scene_to_world=scene_to_world,
            scene_ids=scene_ids,
            outputs=target_outputs,
            returns=selected_returns,
            active_sensor_ids=active_sensor_ids,
        )
        self.backend.render_lidar(request)
        return target_outputs

    def synchronize(self) -> None:
        self._require_initialized()
        self.backend.synchronize()

    def shutdown(self) -> None:
        if self._initialized:
            self.backend.shutdown()
        self.workspace = {}
        self._owned_outputs = {}
        self._initialized = False

    def _owned_output_set(self, batch: int, rays: int, returns: int) -> dict[str, torch.Tensor]:
        assert self.device is not None
        key = (batch, rays, returns)
        cached = self._owned_outputs.get(key)
        if cached is not None:
            return cached
        outputs: dict[str, torch.Tensor] = {}
        for name, dtype in LIDAR_OUTPUT_SPECS.items():
            shape = (
                (batch, rays, returns, 3)
                if name == "position_world_m"
                else (batch, rays)
                if name == "return_count"
                else (batch, rays, returns)
            )
            outputs[name] = torch.empty(shape, device=self.device, dtype=dtype)
        self._owned_outputs[key] = outputs
        return outputs

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("GaussianLidarService.initialize(stage, device) must be called first.")
