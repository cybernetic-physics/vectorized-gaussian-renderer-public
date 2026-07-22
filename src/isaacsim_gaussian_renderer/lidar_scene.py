"""Gaussian LiDAR scene tensors and explicit revision tracking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .contracts import ensure_tensor


@dataclass(frozen=True)
class GaussianLidarScene:
    """Caller-owned static Gaussian tensors used by one LiDAR LBVH.

    Rotations are WXYZ. Scales are activated positive axis scales and opacity,
    reflectivity, and surface confidence are activated values in ``[0, 1]``.
    Missing reflectivity uses the backend's documented constant fallback and
    is nonphysical. Missing surface confidence is exactly one.
    """

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    semantic_ids: torch.Tensor
    reflectivity: torch.Tensor | None = None
    surface_confidence: torch.Tensor | None = None
    revision: int = 0

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def validate(self, *, device: torch.device, require_cuda: bool) -> "GaussianLidarScene":
        if self.revision < 0:
            raise ValueError("scene revision must be non-negative.")
        count = self.count
        if count <= 0:
            raise ValueError("A Gaussian LiDAR scene must contain at least one Gaussian.")
        for name, tensor, shape, dtype in (
            ("means", self.means, (None, 3), torch.float32),
            ("scales", self.scales, (count, 3), torch.float32),
            ("rotations", self.rotations, (count, 4), torch.float32),
            ("opacities", self.opacities, (count,), torch.float32),
            ("semantic_ids", self.semantic_ids, (count,), torch.int64),
        ):
            ensure_tensor(
                tensor,
                name=name,
                shape=shape,
                dtype=dtype,
                device=device,
                require_cuda=require_cuda,
            )
        for name, tensor in (
            ("reflectivity", self.reflectivity),
            ("surface_confidence", self.surface_confidence),
        ):
            if tensor is not None:
                ensure_tensor(
                    tensor,
                    name=name,
                    shape=(count,),
                    dtype=torch.float32,
                    device=device,
                    require_cuda=require_cuda,
                )
        # Registration and explicit revision are already cold-path operations;
        # validating CUDA values here may synchronize, but prevents a malformed
        # static scene from entering the persistent acceleration structure.
        if not torch.isfinite(self.means).all():
            raise ValueError("means must be finite.")
        if not torch.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("scales must be finite and positive.")
        if (self.semantic_ids < 0).any():
            raise ValueError("semantic_ids must be non-negative; -1 is reserved for no-hit outputs.")
        quaternion_norms = torch.linalg.vector_norm(self.rotations, dim=1)
        if not torch.allclose(
            quaternion_norms,
            torch.ones_like(quaternion_norms),
            rtol=0.0,
            atol=1.0e-4,
        ):
            raise ValueError("rotations must be normalized WXYZ quaternions.")
        for name, tensor in (
            ("opacities", self.opacities),
            ("reflectivity", self.reflectivity),
            ("surface_confidence", self.surface_confidence),
        ):
            if tensor is not None and (
                not torch.isfinite(tensor).all() or (tensor < 0).any() or (tensor > 1).any()
            ):
                raise ValueError(f"{name} must be finite and in [0, 1].")
        return self


class LidarSceneRegistry:
    """Retain one caller-owned scene and one explicit revision per scene ID."""

    def __init__(self) -> None:
        self._scenes: dict[int, GaussianLidarScene] = {}

    @property
    def scene_ids(self) -> tuple[int, ...]:
        return tuple(self._scenes)

    def register(self, scene_id: int, scene: GaussianLidarScene) -> None:
        if scene_id in self._scenes:
            raise ValueError(f"LiDAR scene {scene_id} is already registered.")
        self._scenes[scene_id] = scene

    def scene(self, scene_id: int) -> GaussianLidarScene:
        try:
            return self._scenes[scene_id]
        except KeyError as error:
            raise ValueError(f"Unregistered LiDAR scene ID: {scene_id}.") from error

    def revise(
        self,
        scene_id: int,
        revision: int,
        *,
        validate: Callable[[GaussianLidarScene], GaussianLidarScene] | None = None,
    ) -> GaussianLidarScene:
        scene = self.scene(scene_id)
        if revision <= scene.revision:
            raise ValueError(
                f"LiDAR scene {scene_id} revision must increase from {scene.revision}; got {revision}."
            )
        revised = GaussianLidarScene(
            means=scene.means,
            scales=scene.scales,
            rotations=scene.rotations,
            opacities=scene.opacities,
            semantic_ids=scene.semantic_ids,
            reflectivity=scene.reflectivity,
            surface_confidence=scene.surface_confidence,
            revision=revision,
        )
        if validate is not None:
            revised = validate(revised)
        self._scenes[scene_id] = revised
        return revised

    def require_scene_ids(self, scene_ids: torch.Tensor) -> None:
        if scene_ids.device.type != "cpu":
            raise ValueError("Host LiDAR scene-ID validation accepts CPU tensors only.")
        missing = sorted(set(int(value) for value in scene_ids.tolist()) - set(self._scenes))
        if missing:
            raise ValueError(f"Unregistered LiDAR scene ID(s): {missing}.")
