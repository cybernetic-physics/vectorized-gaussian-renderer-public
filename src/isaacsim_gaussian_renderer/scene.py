"""Static Gaussian scene registration and packed-offset metadata."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import ensure_tensor


@dataclass(frozen=True)
class GaussianScene:
    """Canonical inference tensors owned by the caller.

    ``scales`` are positive activated scales, ``opacities`` are activated
    probabilities, ``rotations`` are WXYZ quaternions, and the first three
    ``features`` channels are linear RGB. ``load_ply_to_gaussians`` intentionally
    returns raw 3DGS values, so callers must canonicalize those values before
    registering them with a renderer backend.
    """

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    features: torch.Tensor
    semantic_ids: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def validate(self, *, device: torch.device, require_cuda: bool) -> "GaussianScene":
        count = self.count
        ensure_tensor(
            self.means,
            name="means",
            shape=(None, 3),
            dtype=torch.float32,
            device=device,
            require_cuda=require_cuda,
        )
        ensure_tensor(
            self.scales,
            name="scales",
            shape=(count, 3),
            dtype=torch.float32,
            device=device,
            require_cuda=require_cuda,
        )
        ensure_tensor(
            self.rotations,
            name="rotations",
            shape=(count, 4),
            dtype=torch.float32,
            device=device,
            require_cuda=require_cuda,
        )
        ensure_tensor(
            self.opacities,
            name="opacities",
            shape=(count,),
            dtype=torch.float32,
            device=device,
            require_cuda=require_cuda,
        )
        ensure_tensor(
            self.features,
            name="features",
            shape=(count, None),
            dtype=torch.float32,
            device=device,
            require_cuda=require_cuda,
        )
        ensure_tensor(
            self.semantic_ids,
            name="semantic_ids",
            shape=(count,),
            dtype=torch.int64,
            device=device,
            require_cuda=require_cuda,
        )
        return self


class SceneRegistry:
    """Track scene IDs, offsets, and total packed Gaussian count."""

    def __init__(self) -> None:
        self._scenes: dict[int, GaussianScene] = {}
        self._offsets: list[int] = [0]
        self._offset_tensors: dict[torch.device, torch.Tensor] = {}

    @property
    def scene_ids(self) -> tuple[int, ...]:
        return tuple(self._scenes)

    @property
    def total_gaussians(self) -> int:
        return self._offsets[-1]

    def register(self, scene_id: int, scene: GaussianScene) -> int:
        if scene_id in self._scenes:
            raise ValueError(f"Scene {scene_id} is already registered.")
        offset = self._offsets[-1]
        self._scenes[scene_id] = scene
        self._offsets.append(offset + scene.count)
        self._offset_tensors.clear()
        return offset

    def require_scene_ids(self, scene_ids: torch.Tensor) -> None:
        if scene_ids.device.type != "cpu":
            raise ValueError("Host scene-ID validation accepts CPU tensors only.")
        values = set(int(value) for value in scene_ids.tolist())
        missing = sorted(values - set(self._scenes))
        if missing:
            raise ValueError(f"Unregistered scene ID(s): {missing}.")

    def offsets_tensor(self, *, device: torch.device, require_cuda: bool) -> torch.Tensor:
        tensor = self._offset_tensors.get(device)
        if tensor is None:
            tensor = torch.tensor(self._offsets, dtype=torch.int64, device=device)
            self._offset_tensors[device] = tensor
        if require_cuda and tensor.device.type != "cuda":
            raise ValueError("scene_offsets must be CUDA-resident.")
        return tensor
