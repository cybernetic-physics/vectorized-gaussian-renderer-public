"""Public Isaac Sim Gaussian renderer service API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .backend import GaussianRendererBackend, RenderRequest
from .contracts import OUTPUT_SPECS, ensure_tensor, validate_active_camera_ids, validate_camera_batch, validate_outputs
from .scene import GaussianScene, SceneRegistry


class RendererService:
    """Stable process-local API between Isaac Sim and the custom kernels."""

    def __init__(
        self,
        backend: GaussianRendererBackend,
        *,
        height: int,
        width: int,
        outputs: Iterable[str] = ("rgb", "depth", "alpha", "semantic_id"),
        max_views: int | None = None,
        render_every_n: int = 1,
        allow_cpu_for_tests: bool = False,
    ) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive.")
        if render_every_n <= 0:
            raise ValueError("render_every_n must be positive.")
        unknown = sorted(set(outputs) - set(OUTPUT_SPECS))
        if unknown:
            raise ValueError(f"Unsupported output name(s): {', '.join(unknown)}.")

        self.backend = backend
        self.height = height
        self.width = width
        self.output_names = tuple(outputs)
        self.max_views = max_views
        self.render_every_n = render_every_n
        self.allow_cpu_for_tests = allow_cpu_for_tests

        self.device: torch.device | None = None
        self.stage: Any | None = None
        self.registry = SceneRegistry()
        self.workspace: dict[str, torch.Tensor] = {}
        self._owned_outputs: dict[int, dict[str, torch.Tensor]] = {}
        self._last_outputs: dict[str, torch.Tensor] | None = None
        self._render_request_count = 0
        self._initialized = False

    def initialize(self, stage: Any, device: torch.device | str) -> None:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda" and not self.allow_cpu_for_tests:
            raise ValueError(f"RendererService requires a CUDA device; got {device}.")
        self.stage = stage
        self.device = device
        self.backend.initialize(stage, device)
        if self.max_views is not None:
            self.workspace = self.backend.allocate_workspace(
                max_views=self.max_views,
                height=self.height,
                width=self.width,
                device=device,
                dtype=torch.float32,
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
        features: torch.Tensor,
        semantic_ids: torch.Tensor,
    ) -> None:
        self._require_initialized()
        assert self.device is not None
        scene = GaussianScene(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            features=features,
            semantic_ids=semantic_ids,
        ).validate(device=self.device, require_cuda=not self.allow_cpu_for_tests)
        offset = self.registry.register(scene_id, scene)
        self.backend.register_scene(scene_id, scene, offset)
        # Materialize packed offsets at registration time so render() only
        # retrieves an already resident tensor.
        self.registry.offsets_tensor(
            device=self.device,
            require_cuda=not self.allow_cpu_for_tests,
        )

    def update_scene_transforms(self, scene_ids: torch.Tensor, transforms: torch.Tensor) -> None:
        self._require_initialized()
        assert self.device is not None
        require_cuda = not self.allow_cpu_for_tests
        ensure_tensor(
            transforms,
            name="transforms",
            shape=(None, 4, 4),
            dtype=torch.float32,
            device=self.device,
            require_cuda=require_cuda,
        )
        if scene_ids.shape != (transforms.shape[0],):
            raise ValueError("scene_ids must have one entry per transform.")
        ensure_tensor(
            scene_ids,
            name="scene_ids",
            shape=(transforms.shape[0],),
            dtype=torch.int64,
            device=self.device,
            require_cuda=require_cuda,
        )
        if not require_cuda:
            self.registry.require_scene_ids(scene_ids)
        self.backend.update_scene_transforms(scene_ids, transforms)

    def render(
        self,
        camera_transforms: torch.Tensor,
        intrinsics: torch.Tensor,
        scene_ids: torch.Tensor,
        outputs: dict[str, torch.Tensor] | None = None,
        active_camera_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._require_initialized()
        assert self.device is not None
        require_cuda = not self.allow_cpu_for_tests
        batch = validate_camera_batch(
            camera_transforms,
            intrinsics,
            scene_ids,
            device=self.device,
            require_cuda=require_cuda,
        )
        if self.max_views is not None and batch > self.max_views:
            raise ValueError(f"Batch has {batch} cameras but service max_views is {self.max_views}.")
        active_camera_ids = validate_active_camera_ids(
            active_camera_ids,
            batch=batch,
            device=self.device,
            require_cuda=require_cuda,
        )
        if not require_cuda:
            self.registry.require_scene_ids(scene_ids)

        target_outputs = outputs if outputs is not None else self._owned_output_set(batch)
        validate_outputs(
            target_outputs,
            batch=batch,
            height=self.height,
            width=self.width,
            device=self.device,
            require_cuda=require_cuda,
        )

        self._render_request_count += 1
        if self.render_every_n > 1 and (self._render_request_count - 1) % self.render_every_n != 0:
            if self._last_outputs is None:
                raise RuntimeError("Render cadence skipped before any output exists.")
            return self._last_outputs

        request = RenderRequest(
            camera_transforms=camera_transforms,
            intrinsics=intrinsics,
            scene_ids=scene_ids,
            scene_offsets=self.registry.offsets_tensor(device=self.device, require_cuda=require_cuda),
            outputs=target_outputs,
            active_camera_ids=active_camera_ids,
            workspace=self.workspace,
        )
        self.backend.render(request)
        self._last_outputs = target_outputs
        return target_outputs

    def prepare_outputs(self, batch: int) -> dict[str, torch.Tensor]:
        """Allocate and retain one logical output set before steady state.

        Callers with a measured render loop can prepare its full logical batch
        during setup, then pass the returned tensors back through ``render``.
        This keeps output storage allocation outside the measured boundary,
        including when the backend uses bounded physical chunks internally.
        """
        self._require_initialized()
        if batch <= 0:
            raise ValueError("batch must be positive.")
        if self.max_views is not None and batch > self.max_views:
            raise ValueError(
                f"Batch has {batch} cameras but service max_views is "
                f"{self.max_views}."
            )
        return self._owned_output_set(batch)

    def synchronize(self) -> None:
        self._require_initialized()
        self.backend.synchronize()

    def shutdown(self) -> None:
        if self._initialized:
            self.backend.shutdown()
        self.workspace = {}
        self._owned_outputs = {}
        self._last_outputs = None
        self._initialized = False

    def _owned_output_set(self, batch: int) -> dict[str, torch.Tensor]:
        assert self.device is not None
        cached = self._owned_outputs.get(batch)
        if cached is not None and tuple(cached) == self.output_names:
            return cached
        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            channels, dtype = OUTPUT_SPECS[name]
            outputs[name] = torch.zeros(
                (batch, self.height, self.width, channels),
                device=self.device,
                dtype=dtype,
            )
            if name == "semantic_id":
                outputs[name].fill_(-1)
        self._owned_outputs[batch] = outputs
        return outputs

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("RendererService.initialize(stage, device) must be called first.")
