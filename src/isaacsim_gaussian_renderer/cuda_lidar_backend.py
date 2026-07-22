"""Project-owned CUDA LBVH backend for Gaussian surface LiDAR."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch

from .lidar_backend import LidarRenderRequest
from .lidar_native_loader import load_lidar_native_extension
from .lidar_scene import GaussianLidarScene


LIDAR_COUNTER_NAMES = (
    "calls_started",
    "calls_completed",
    "rays_traced",
    "node_visits",
    "leaf_tests",
    "candidates",
    "returns",
    "stack_overflow",
    "semantic_overflow",
    "invalid_directions",
    "invalid_scene_ids",
    "invalid_active_sensor_ids",
)


class LidarTraversalError(RuntimeError):
    """Raised by an explicit post-submit device-counter check."""


@dataclass
class _SceneAcceleration:
    scene: GaussianLidarScene
    sorted_indices: torch.Tensor
    node_bounds: torch.Tensor
    leaf_count: int
    leaf_capacity: int

    @property
    def bvh_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.sorted_indices, self.node_bounds)
        )


class CudaLidarBackend:
    """Morton-sorted packet LBVH and batched current-stream surface tracing.

    Thresholds are public equation parameters, not benchmark-tuned toggles.
    The backend owns one acceleration structure per registered static scene and
    never expands scene tensors or BVHs over the sensor batch.
    """

    def __init__(
        self,
        *,
        max_scenes: int = 64,
        packet_size: int = 8,
        near_plane_m: float = 0.05,
        far_plane_m: float = 200.0,
        support_sigma: float = 3.0,
        detection_threshold: float = 0.01,
        planarity_ratio_max: float = 0.35,
        min_incidence_cos: float = 0.05,
        cluster_abs_m: float = 0.02,
        cluster_relative: float = 0.001,
        fallback_reflectivity: float = 0.5,
        direction_norm_tolerance: float = 1.0e-4,
        semantic_slots: int = 16,
        native_module: ModuleType | Any | None = None,
        native_loader: Callable[[], ModuleType] = load_lidar_native_extension,
        allow_cpu_for_tests: bool = False,
    ) -> None:
        if max_scenes <= 0:
            raise ValueError("max_scenes must be positive.")
        if packet_size not in (8, 16):
            raise ValueError("packet_size must be 8 or 16.")
        if near_plane_m <= 0 or far_plane_m <= near_plane_m:
            raise ValueError("Expected 0 < near_plane_m < far_plane_m.")
        if support_sigma <= 0:
            raise ValueError("support_sigma must be positive.")
        if not 0 < detection_threshold <= 1:
            raise ValueError("detection_threshold must be in (0, 1].")
        if not 0 < planarity_ratio_max <= 1:
            raise ValueError("planarity_ratio_max must be in (0, 1].")
        if not 0 < min_incidence_cos <= 1:
            raise ValueError("min_incidence_cos must be in (0, 1].")
        if cluster_abs_m < 0 or cluster_relative < 0:
            raise ValueError("cluster thresholds must be non-negative.")
        if not 0 <= fallback_reflectivity <= 1:
            raise ValueError("fallback_reflectivity must be in [0, 1].")
        if not 0 < direction_norm_tolerance < 1:
            raise ValueError("direction_norm_tolerance must be in (0, 1).")
        if semantic_slots <= 0 or semantic_slots > 32:
            raise ValueError("semantic_slots must be in [1, 32].")
        self.max_scenes = max_scenes
        self.packet_size = packet_size
        self.near_plane_m = near_plane_m
        self.far_plane_m = far_plane_m
        self.support_sigma = support_sigma
        self.detection_threshold = detection_threshold
        self.planarity_ratio_max = planarity_ratio_max
        self.min_incidence_cos = min_incidence_cos
        self.cluster_abs_m = cluster_abs_m
        self.cluster_relative = cluster_relative
        self.fallback_reflectivity = fallback_reflectivity
        self.direction_norm_tolerance = direction_norm_tolerance
        self.semantic_slots = semantic_slots
        self._native = native_module
        self._native_loader = native_loader
        self.allow_cpu_for_tests = allow_cpu_for_tests
        self.device: torch.device | None = None
        self.stage: Any | None = None
        self.workspace: dict[str, torch.Tensor] = {}
        self._scenes: OrderedDict[int, _SceneAcceleration] = OrderedDict()
        self._pending_build_temporaries: list[tuple[torch.Tensor, ...]] = []

    def initialize(self, stage: Any, device: torch.device) -> None:
        if device.type != "cuda" and not self.allow_cpu_for_tests:
            raise ValueError(f"CudaLidarBackend requires CUDA; got {device}.")
        self.stage = stage
        self.device = device
        if self._native is None:
            self._native = self._native_loader()

    def allocate_workspace(
        self,
        *,
        max_sensors: int,
        max_rays: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        self._require_initialized()
        identity = torch.eye(4, device=device, dtype=torch.float32)
        self.workspace = {
            "scene_descriptors": torch.zeros(
                (self.max_scenes, 14), device=device, dtype=torch.int64
            ),
            "counters": torch.zeros(
                (len(LIDAR_COUNTER_NAMES),), device=device, dtype=torch.int64
            ),
            "identity_scene_to_world": identity.repeat(max_sensors, 1, 1).contiguous(),
            "all_active_sensor_ids": torch.arange(
                max_sensors, device=device, dtype=torch.int64
            ),
            "empty_active_sensor_ids": torch.empty((0,), device=device, dtype=torch.int64),
            "empty_optional_attribute": torch.empty((0,), device=device, dtype=torch.float32),
        }
        return self.workspace

    def register_scene(self, scene_id: int, scene: GaussianLidarScene) -> None:
        self._require_workspace()
        if scene_id in self._scenes:
            raise ValueError(f"LiDAR scene {scene_id} is already registered.")
        if len(self._scenes) >= self.max_scenes:
            raise ValueError(f"LiDAR scene count exceeds max_scenes={self.max_scenes}.")
        acceleration = self._allocate_acceleration(scene)
        self._scenes[scene_id] = acceleration
        self._build_and_pack_scene(len(self._scenes) - 1, scene_id, acceleration)

    def revise_scene(self, scene_id: int, scene: GaussianLidarScene) -> None:
        self._require_workspace()
        if scene_id not in self._scenes:
            raise ValueError(f"Unregistered LiDAR scene ID: {scene_id}.")
        acceleration = self._scenes[scene_id]
        if scene.count != acceleration.scene.count:
            raise ValueError("A scene revision must retain the registered Gaussian count.")
        acceleration.scene = scene
        slot = tuple(self._scenes).index(scene_id)
        self._build_and_pack_scene(slot, scene_id, acceleration)

    def render_lidar(self, request: LidarRenderRequest) -> None:
        self._require_workspace()
        if not self._scenes:
            raise RuntimeError("At least one Gaussian LiDAR scene must be registered.")
        active = (
            request.active_sensor_ids
            if request.active_sensor_ids is not None
            else self.workspace["all_active_sensor_ids"][: request.scene_ids.shape[0]]
        )
        outputs = [
            request.outputs["range_m"],
            request.outputs["position_world_m"],
            request.outputs["intensity"],
            request.outputs["semantic_id"],
            request.outputs["valid"],
            request.outputs["time_offset_ns"],
            request.outputs["return_count"],
        ]
        inputs = [
            request.ray_directions,
            request.time_offsets_ns,
            request.sensor_to_world,
            request.scene_to_world,
            request.scene_ids,
            active,
        ]
        assert self._native is not None
        nvtx_range = (
            torch.cuda.nvtx.range("GaussianLidarService.render_lidar")
            if request.ray_directions.is_cuda
            else _nullcontext()
        )
        with nvtx_range:
            self._native.render_lidar(
                self.workspace["scene_descriptors"],
                len(self._scenes),
                inputs,
                outputs,
                self.workspace["counters"],
                request.returns,
                self.packet_size,
                self.near_plane_m,
                self.far_plane_m,
                self.support_sigma,
                self.detection_threshold,
                self.planarity_ratio_max,
                self.min_incidence_cos,
                self.cluster_abs_m,
                self.cluster_relative,
                self.fallback_reflectivity,
                self.direction_norm_tolerance,
                self.semantic_slots,
            )

    def synchronize(self) -> None:
        self._require_initialized()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._pending_build_temporaries.clear()

    def read_counters(self, *, synchronize: bool = True) -> dict[str, int]:
        if synchronize:
            self.synchronize()
        values = self.workspace["counters"].detach().cpu().tolist()
        return dict(zip(LIDAR_COUNTER_NAMES, (int(value) for value in values), strict=True))

    def check_errors(self, *, synchronize: bool = True) -> dict[str, int]:
        counters = self.read_counters(synchronize=synchronize)
        errors = {
            name: counters[name]
            for name in (
                "stack_overflow",
                "semantic_overflow",
                "invalid_directions",
                "invalid_scene_ids",
                "invalid_active_sensor_ids",
            )
            if counters[name]
        }
        if errors:
            raise LidarTraversalError(f"Gaussian LiDAR device errors: {errors}.")
        return counters

    @property
    def bvh_bytes(self) -> int:
        return sum(acceleration.bvh_bytes for acceleration in self._scenes.values())

    @property
    def scene_attribute_bytes(self) -> int:
        total = 0
        for acceleration in self._scenes.values():
            scene = acceleration.scene
            tensors = (
                scene.means,
                scene.scales,
                scene.rotations,
                scene.opacities,
                scene.semantic_ids,
                scene.reflectivity,
                scene.surface_confidence,
            )
            total += sum(
                tensor.numel() * tensor.element_size()
                for tensor in tensors
                if tensor is not None
            )
        return total

    @property
    def workspace_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.workspace.values())

    def shutdown(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            self.synchronize()
        self.workspace = {}
        self._scenes.clear()
        self._pending_build_temporaries.clear()
        self.stage = None
        self.device = None

    def _allocate_acceleration(self, scene: GaussianLidarScene) -> _SceneAcceleration:
        assert self.device is not None
        leaf_count = math.ceil(scene.count / self.packet_size)
        leaf_capacity = 1 << (leaf_count - 1).bit_length()
        node_count = 2 * leaf_capacity - 1
        return _SceneAcceleration(
            scene=scene,
            sorted_indices=torch.empty((scene.count,), device=self.device, dtype=torch.int32),
            node_bounds=torch.empty((node_count, 6), device=self.device, dtype=torch.float32),
            leaf_count=leaf_count,
            leaf_capacity=leaf_capacity,
        )

    def _build_and_pack_scene(
        self,
        slot: int,
        scene_id: int,
        acceleration: _SceneAcceleration,
    ) -> None:
        assert self.device is not None
        assert self._native is not None
        scene = acceleration.scene
        keys_in = torch.empty((scene.count,), device=self.device, dtype=torch.uint64)
        keys_out = torch.empty_like(keys_in)
        indices_in = torch.arange(scene.count, device=self.device, dtype=torch.int32)
        sort_bytes = int(self._native.sort_temp_bytes(scene.count))
        sort_temp = torch.empty((sort_bytes,), device=self.device, dtype=torch.uint8)
        scene_bounds = torch.stack((scene.means.amin(dim=0), scene.means.amax(dim=0))).contiguous()
        self._native.build_scene_lbvh(
            [scene.means, scene.scales, scene.rotations],
            [
                keys_in,
                keys_out,
                indices_in,
                acceleration.sorted_indices,
                sort_temp,
                scene_bounds,
                acceleration.node_bounds,
            ],
            self.packet_size,
            acceleration.leaf_count,
            acceleration.leaf_capacity,
            self.support_sigma,
            self.planarity_ratio_max,
        )
        empty = self.workspace["empty_optional_attribute"]
        self._native.pack_scene_descriptor(
            self.workspace["scene_descriptors"],
            slot,
            scene_id,
            [
                scene.means,
                scene.scales,
                scene.rotations,
                scene.opacities,
                scene.semantic_ids,
                scene.reflectivity if scene.reflectivity is not None else empty,
                scene.surface_confidence if scene.surface_confidence is not None else empty,
                acceleration.sorted_indices,
                acceleration.node_bounds,
            ],
            acceleration.leaf_count,
            acceleration.leaf_capacity,
        )
        self._pending_build_temporaries.append(
            (keys_in, keys_out, indices_in, sort_temp, scene_bounds)
        )

    def _require_initialized(self) -> None:
        if self.device is None:
            raise RuntimeError("CudaLidarBackend.initialize(stage, device) must be called first.")

    def _require_workspace(self) -> None:
        self._require_initialized()
        if not self.workspace:
            raise RuntimeError("allocate_workspace() must be called first.")


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None
