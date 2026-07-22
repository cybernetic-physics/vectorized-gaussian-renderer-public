"""RendererService backend for a pinned FlashGS-derived matched-contract port.

The port preserves FlashGS's per-camera key-emission, global-radix-sort,
tile-range, and optimized tile-compositor topology. It changes projection and
image formation to match the pinned-gsplat contract, in addition to adding the
service integration and optional robotics sensor outputs. It is therefore not
an upstream-faithful or integration-only FlashGS baseline. A batched request
deliberately executes its native pipeline once per active camera on the same
CUDA stream.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from .backend import RenderRequest, RenderStats
from .flashgs_native_loader import load_flashgs_native_extension
from .scene import GaussianScene


FLASHGS_UPSTREAM_COMMIT = "cdfc4e4002318423eda356eed02df8e01fa32cb6"
FULL_SENSOR_OUTPUTS = ("rgb", "depth", "alpha", "semantic_id")
RGB_ONLY_OUTPUTS = ("rgb",)
COUNTER_NAMES = (
    "visible_gaussians",
    "generated_intersections",
    "intersection_overflow",
)
_INT32_MAX = 2_147_483_647


class FlashGSBackend:
    """Pinned FlashGS backend with fixed-capacity, synchronization-free sort.

    One scene is registered and retained on the target GPU. Geometry scratch
    and sort storage are preallocated and reused across cameras and frames.
    The public FlashGS implementation has no true camera batch scheduler, so
    the native binding launches the unbatched pipeline once per camera; this
    cost is intentionally part of every measurement.
    """

    def __init__(
        self,
        *,
        max_intersections: int = 1_000_000,
        near_plane: float = 0.01,
        far_plane: float = 100.0,
        gaussian_support_sigma: float = 3.0,
        covariance_epsilon: float = 0.0,
        semantic_min_alpha: float = 0.01,
        tile_size: int = 16,
        native_module: ModuleType | Any | None = None,
        native_loader: Callable[[], ModuleType] = load_flashgs_native_extension,
        allow_cpu_for_tests: bool = False,
    ) -> None:
        if max_intersections <= 0 or max_intersections > _INT32_MAX:
            raise ValueError(
                f"max_intersections must be in [1, {_INT32_MAX}]."
            )
        if not (0.0 < near_plane < far_plane):
            raise ValueError("Expected 0 < near_plane < far_plane.")
        if not math.isfinite(gaussian_support_sigma) or gaussian_support_sigma <= 0:
            raise ValueError("gaussian_support_sigma must be finite and positive.")
        if not math.isfinite(covariance_epsilon) or covariance_epsilon < 0:
            raise ValueError("covariance_epsilon must be finite and non-negative.")
        if not 0.0 <= semantic_min_alpha <= 1.0:
            raise ValueError("semantic_min_alpha must be in [0, 1].")
        if tile_size != 16:
            raise ValueError(
                "The pinned FlashGS comparison uses its public 16x16 optimized kernel."
            )

        self.max_intersections = int(max_intersections)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.gaussian_support_sigma = float(gaussian_support_sigma)
        self.covariance_epsilon = float(covariance_epsilon)
        self.semantic_min_alpha = float(semantic_min_alpha)
        self.tile_size = tile_size
        self._native = native_module
        self._native_loader = native_loader
        self.allow_cpu_for_tests = allow_cpu_for_tests

        self.device: torch.device | None = None
        self.stage: Any | None = None
        self.workspace: dict[str, torch.Tensor] = {}
        self._scene: GaussianScene | None = None
        self._scene_id: int | None = None
        self._covariances: torch.Tensor | None = None
        self._max_views = 0
        self._height = 0
        self._width = 0
        self._last_full_sensor_output: bool | None = None
        self._render_requests = 0
        self._camera_executions = 0

    def initialize(self, stage: Any, device: torch.device) -> None:
        if device.type != "cuda" and not self.allow_cpu_for_tests:
            raise ValueError(f"FlashGSBackend requires CUDA; got {device}.")
        self.stage = stage
        self.device = device
        if self._native is None:
            self._native = self._native_loader()

    def register_scene(self, scene_id: int, scene: GaussianScene, offset: int) -> None:
        self._require_initialized()
        if self._scene is not None:
            raise ValueError(
                "The matched FlashGS backend supports one shared resident scene."
            )
        if offset != 0:
            raise ValueError("The single FlashGS scene must have packed offset zero.")
        if scene.features.shape[1] != 3:
            raise ValueError(
                "The matched FlashGS contract requires exactly three canonical RGB channels."
            )
        self._scene = scene
        self._scene_id = int(scene_id)
        self._covariances = torch.empty(
            (scene.count, 6),
            device=scene.means.device,
            dtype=torch.float32,
        )
        assert self._native is not None
        self._native.precompute_covariances(
            scene.scales,
            scene.rotations,
            self._covariances,
        )
        # Public FlashGS materializes these three arrays once per scene and
        # reuses them across camera renders. Their size is O(G), not O(BG).
        self.workspace["points_xy"] = torch.empty(
            (scene.count, 2), device=scene.means.device, dtype=torch.float32
        )
        self.workspace["rgb_depth"] = torch.empty(
            (scene.count, 4), device=scene.means.device, dtype=torch.float32
        )
        self.workspace["conic_opacity"] = torch.empty(
            (scene.count, 4), device=scene.means.device, dtype=torch.float32
        )

    def allocate_workspace(
        self,
        *,
        max_views: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        self._require_initialized()
        if max_views <= 0:
            raise ValueError("max_views must be positive.")
        if dtype != torch.float32:
            raise ValueError("FlashGS matched outputs use float32.")
        assert self._native is not None
        sort_temp_bytes = int(
            self._native.get_sort_buffer_size(self.max_intersections)
        )
        if sort_temp_bytes <= 0:
            raise RuntimeError("FlashGS returned an invalid CUB sort size.")
        tiles = math.ceil(width / self.tile_size) * math.ceil(
            height / self.tile_size
        )
        self.workspace.update(
            {
                "keys_unsorted": torch.empty(
                    (self.max_intersections,), device=device, dtype=torch.uint64
                ),
                "values_unsorted": torch.empty(
                    (self.max_intersections,), device=device, dtype=torch.int32
                ),
                "keys_sorted": torch.empty(
                    (self.max_intersections,), device=device, dtype=torch.uint64
                ),
                "values_sorted": torch.empty(
                    (self.max_intersections,), device=device, dtype=torch.int32
                ),
                "sort_temp": torch.empty(
                    (sort_temp_bytes,), device=device, dtype=torch.uint8
                ),
                "ranges": torch.empty((tiles, 2), device=device, dtype=torch.int32),
                "counters": torch.empty(
                    (max_views, len(COUNTER_NAMES)),
                    device=device,
                    dtype=torch.int64,
                ),
                "empty_active_ids": torch.empty(
                    (0,), device=device, dtype=torch.int64
                ),
                "dummy_depth": torch.empty((1,), device=device, dtype=torch.float32),
                "dummy_alpha": torch.empty((1,), device=device, dtype=torch.float32),
                "dummy_semantic": torch.empty((1,), device=device, dtype=torch.int64),
            }
        )
        self._max_views = int(max_views)
        self._height = int(height)
        self._width = int(width)
        return self.workspace

    def update_scene_transforms(
        self,
        scene_ids: torch.Tensor,
        transforms: torch.Tensor,
    ) -> None:
        raise NotImplementedError(
            "The matched FlashGS baseline covers one immutable shared scene; "
            "environment-transform motion is outside this benchmark contract."
        )

    def reserve_intersection_capacity(self, capacity: int) -> None:
        """Replace fixed sort storage after an out-of-band trajectory preflight."""
        self._require_initialized()
        if capacity <= 0 or capacity > _INT32_MAX:
            raise ValueError(f"capacity must be in [1, {_INT32_MAX}].")
        if not self.workspace or self.device is None:
            raise RuntimeError("allocate_workspace() must be called before reservation.")
        self.synchronize()
        for name in (
            "keys_unsorted",
            "values_unsorted",
            "keys_sorted",
            "values_sorted",
            "sort_temp",
        ):
            self.workspace.pop(name, None)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        assert self._native is not None
        sort_temp_bytes = int(self._native.get_sort_buffer_size(capacity))
        self.workspace.update(
            {
                "keys_unsorted": torch.empty(
                    (capacity,), device=self.device, dtype=torch.uint64
                ),
                "values_unsorted": torch.empty(
                    (capacity,), device=self.device, dtype=torch.int32
                ),
                "keys_sorted": torch.empty(
                    (capacity,), device=self.device, dtype=torch.uint64
                ),
                "values_sorted": torch.empty(
                    (capacity,), device=self.device, dtype=torch.int32
                ),
                "sort_temp": torch.empty(
                    (sort_temp_bytes,), device=self.device, dtype=torch.uint8
                ),
            }
        )
        self.max_intersections = int(capacity)

    def render(self, request: RenderRequest) -> RenderStats:
        self._require_initialized()
        if self._scene is None or self._covariances is None:
            raise RuntimeError("A scene must be registered before rendering.")
        if "points_xy" not in self.workspace:
            raise RuntimeError("FlashGS scene scratch is not allocated.")
        output_names = tuple(request.outputs)
        if output_names not in (RGB_ONLY_OUTPUTS, FULL_SENSOR_OUTPUTS):
            raise ValueError(
                "FlashGSBackend supports exactly the RGB-only or fused "
                "RGB/depth/alpha/semantic_id output contract; got "
                f"{output_names}."
            )
        full_sensor_output = output_names == FULL_SENSOR_OUTPUTS
        batch = int(request.camera_transforms.shape[0])
        if batch > self._max_views:
            raise ValueError(
                f"Render batch {batch} exceeds workspace max_views {self._max_views}."
            )
        if request.active_camera_ids is not None:
            raise NotImplementedError(
                "The matched FlashGS baseline does not add active-subset scheduling."
            )
        active_ids = self.workspace["empty_active_ids"]
        rendered_views = batch
        output_tensors = [
            request.outputs["rgb"],
            request.outputs.get("depth", self.workspace["dummy_depth"]),
            request.outputs.get("alpha", self.workspace["dummy_alpha"]),
            request.outputs.get("semantic_id", self.workspace["dummy_semantic"]),
        ]
        scene_tensors = [
            self._scene.means,
            self._covariances,
            self._scene.opacities,
            self._scene.features,
            self._scene.semantic_ids,
        ]
        camera_tensors = [
            request.camera_transforms,
            request.intrinsics,
            request.scene_ids,
            active_ids,
        ]
        workspace_tensors = [
            self.workspace["points_xy"],
            self.workspace["rgb_depth"],
            self.workspace["conic_opacity"],
            self.workspace["keys_unsorted"],
            self.workspace["values_unsorted"],
            self.workspace["keys_sorted"],
            self.workspace["values_sorted"],
            self.workspace["sort_temp"],
            self.workspace["ranges"],
            self.workspace["counters"],
        ]
        assert self._native is not None
        self._native.render_batch(
            scene_tensors,
            camera_tensors,
            output_tensors,
            workspace_tensors,
            self._height,
            self._width,
            self.near_plane,
            self.far_plane,
            self.gaussian_support_sigma,
            self.covariance_epsilon,
            self.semantic_min_alpha,
            full_sensor_output,
            int(self._scene_id),
        )
        self._render_requests += 1
        self._camera_executions += rendered_views
        self._last_full_sensor_output = full_sensor_output
        return RenderStats(rendered_views=rendered_views)

    def synchronize(self) -> None:
        self._require_initialized()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def read_counters(self, *, synchronize: bool = True) -> dict[str, int]:
        if synchronize:
            self.synchronize()
        values = self.workspace["counters"].sum(dim=0).detach().cpu().tolist()
        return dict(zip(COUNTER_NAMES, (int(value) for value in values), strict=True))

    def check_capacity(self, *, synchronize: bool = True) -> dict[str, int]:
        counters = self.read_counters(synchronize=synchronize)
        if counters["intersection_overflow"]:
            raise RuntimeError(
                "FlashGS intersection capacity overflow: "
                f"{counters['intersection_overflow']} dropped intersections."
            )
        return counters

    @property
    def scene_bytes(self) -> int:
        if self._scene is None or self._covariances is None:
            return 0
        tensors = (
            self._scene.means,
            self._scene.scales,
            self._scene.rotations,
            self._covariances,
            self._scene.opacities,
            self._scene.features,
            self._scene.semantic_ids,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def workspace_bytes(self) -> int:
        return sum(
            tensor.untyped_storage().nbytes()
            for tensor in self.workspace.values()
        )

    @property
    def execution_stats(self) -> dict[str, Any]:
        return {
            "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
            "render_requests": self._render_requests,
            "native_camera_executions": self._camera_executions,
            "true_native_batch": False,
            "fixed_capacity_sort": True,
            "max_intersections_per_camera": self.max_intersections,
            "full_sensor_output": self._last_full_sensor_output,
        }

    def shutdown(self) -> None:
        self.workspace.clear()
        self._scene = None
        self._covariances = None
        self.stage = None
        self.device = None

    def _require_initialized(self) -> None:
        if self.device is None:
            raise RuntimeError("FlashGSBackend.initialize(stage, device) must be called first.")
