"""Degree-zero RendererService lane preserving public FlashGS's RGB equation.

This control is intentionally distinct from :class:`FlashGSBackend`, which is
a gsplat-matched FlashGS-derived port for the equal-equation benchmark. The
upstream-faithful lane keeps FlashGS's native support, alpha, termination,
quantization, and 16x16 tile shader.  It specializes color input to degree zero
and therefore is not a byte-identical performance build of public FlashGS.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from .backend import RenderRequest, RenderStats
from .scene import GaussianScene
from .upstream_faithful_flashgs_native_loader import (
    load_upstream_faithful_flashgs_native_extension,
)
from .upstream_faithful_flashgs_sources import FLASHGS_UPSTREAM_COMMIT


UPSTREAM_FAITHFUL_RGB_OUTPUTS = ("rgb",)
UPSTREAM_FAITHFUL_COUNTER_NAMES = (
    "generated_intersections",
    "intersection_overflow",
    "camera_contract_errors",
)
_INT32_MAX = 2_147_483_647


class UpstreamFaithfulFlashGSBackend:
    """Pinned FlashGS-equation RGB control with reviewed integration glue.

    The public renderer is unbatched. A logical camera batch therefore invokes
    its projection, global radix sort, and exact 16x16 compositor once per
    camera on the active PyTorch stream. The native uint8 RGB target is reused
    and converted into RendererService's float32 RGB output on that same stream;
    conversion time is part of every render.
    """

    def __init__(
        self,
        *,
        max_intersections: int = 12_000_000,
        near_plane: float = 0.01,
        far_plane: float = 100.0,
        tile_size: int = 16,
        native_module: ModuleType | Any | None = None,
        native_loader: Callable[[], ModuleType] = (
            load_upstream_faithful_flashgs_native_extension
        ),
        allow_cpu_for_tests: bool = False,
    ) -> None:
        if max_intersections <= 0 or max_intersections > _INT32_MAX:
            raise ValueError(
                f"max_intersections must be in [1, {_INT32_MAX}]."
            )
        if not (0.0 < near_plane < far_plane):
            raise ValueError("Expected 0 < near_plane < far_plane.")
        if tile_size != 16:
            raise ValueError("Public FlashGS's optimized control uses 16x16 tiles.")
        self.max_intersections = int(max_intersections)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.tile_size = int(tile_size)
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
        self._last_batch = 0
        self._render_requests = 0
        self._camera_executions = 0

    def initialize(self, stage: Any, device: torch.device) -> None:
        if device.type != "cuda" and not self.allow_cpu_for_tests:
            raise ValueError(
                f"UpstreamFaithfulFlashGSBackend requires CUDA; got {device}."
            )
        self.stage = stage
        self.device = device
        if self._native is None:
            self._native = self._native_loader()

    def register_scene(
        self,
        scene_id: int,
        scene: GaussianScene,
        offset: int,
    ) -> None:
        self._require_initialized()
        if self._scene is not None:
            raise ValueError(
                "The upstream-faithful FlashGS control supports one resident scene."
            )
        if offset != 0:
            raise ValueError("The single upstream FlashGS scene must start at offset zero.")
        if scene.features.shape[1] != 3:
            raise ValueError(
                "The upstream-faithful control accepts exactly three canonical "
                "degree-zero linear RGB channels."
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
        self.workspace.update(
            {
                "points_xy": torch.empty(
                    (scene.count, 2),
                    device=scene.means.device,
                    dtype=torch.float32,
                ),
                "rgb_depth": torch.empty(
                    (scene.count, 4),
                    device=scene.means.device,
                    dtype=torch.float32,
                ),
                "conic_opacity": torch.empty(
                    (scene.count, 4),
                    device=scene.means.device,
                    dtype=torch.float32,
                ),
            }
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
            raise ValueError("Upstream FlashGS control outputs use float32.")
        if height % self.tile_size or width % self.tile_size:
            raise ValueError(
                "Exact public FlashGS RGB requires height and width divisible by 16."
            )
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
                # One extra range is the sentinel tile sorted immediately after
                # every real tile. Exact upstream range discovery writes it,
                # while the exact upstream compositor never launches it.
                "ranges": torch.empty(
                    (tiles + 1, 2), device=device, dtype=torch.int32
                ),
                "camera_state": torch.empty(
                    (39,), device=device, dtype=torch.float32
                ),
                "rgb_u8": torch.empty(
                    (height, width, 3), device=device, dtype=torch.uint8
                ),
                "counters": torch.empty(
                    (max_views, len(UPSTREAM_FAITHFUL_COUNTER_NAMES)),
                    device=device,
                    dtype=torch.int32,
                ),
                "empty_active_ids": torch.empty(
                    (0,), device=device, dtype=torch.int64
                ),
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
            "The upstream-faithful FlashGS control covers one immutable scene."
        )

    def reserve_intersection_capacity(self, capacity: int) -> None:
        """Install fixed storage after an out-of-band, untimed calibration."""
        self._require_initialized()
        if capacity <= 0 or capacity > _INT32_MAX:
            raise ValueError(f"capacity must be in [1, {_INT32_MAX}].")
        if not self.workspace or self.device is None:
            raise RuntimeError("allocate_workspace() must run before reservation.")
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
        if tuple(request.outputs) != UPSTREAM_FAITHFUL_RGB_OUTPUTS:
            raise ValueError(
                "UpstreamFaithfulFlashGSBackend supports RGB only; depth, alpha, "
                "and semantics would change the public tile shader."
            )
        batch = int(request.camera_transforms.shape[0])
        if batch > self._max_views:
            raise ValueError(
                f"Render batch {batch} exceeds workspace max_views {self._max_views}."
            )
        if request.active_camera_ids is not None:
            raise NotImplementedError(
                "The upstream-faithful control does not add active-subset scheduling."
            )
        workspace = [
            self.workspace["points_xy"],
            self.workspace["rgb_depth"],
            self.workspace["conic_opacity"],
            self.workspace["keys_unsorted"],
            self.workspace["values_unsorted"],
            self.workspace["keys_sorted"],
            self.workspace["values_sorted"],
            self.workspace["sort_temp"],
            self.workspace["ranges"],
            self.workspace["camera_state"],
            self.workspace["rgb_u8"],
            self.workspace["counters"],
        ]
        assert self._native is not None
        self._native.render_batch(
            [
                self._scene.means,
                self._covariances,
                self._scene.opacities,
                self._scene.features,
            ],
            [
                request.camera_transforms,
                request.intrinsics,
                request.scene_ids,
                self.workspace["empty_active_ids"],
            ],
            [request.outputs["rgb"]],
            workspace,
            self._height,
            self._width,
            self.near_plane,
            self.far_plane,
            int(self._scene_id),
        )
        self._render_requests += 1
        self._camera_executions += batch
        self._last_batch = batch
        return RenderStats(rendered_views=batch)

    def synchronize(self) -> None:
        self._require_initialized()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def read_counters(self, *, synchronize: bool = True) -> dict[str, int]:
        if synchronize:
            self.synchronize()
        values = (
            self.workspace["counters"][: self._last_batch]
            .to(dtype=torch.int64)
            .sum(dim=0)
            .detach()
            .cpu()
            .tolist()
        )
        return dict(
            zip(
                UPSTREAM_FAITHFUL_COUNTER_NAMES,
                (int(value) for value in values),
                strict=True,
            )
        )

    def check_capacity(self, *, synchronize: bool = True) -> dict[str, int]:
        counters = self.read_counters(synchronize=synchronize)
        if counters["camera_contract_errors"]:
            raise RuntimeError(
                "Upstream FlashGS camera contract failed: expected centered, "
                "zero-skew intrinsics, affine OpenCV view matrices, and the "
                "registered scene ID."
            )
        if counters["intersection_overflow"]:
            raise RuntimeError(
                "Upstream FlashGS fixed-capacity overflow: "
                f"{counters['intersection_overflow']} intersections were dropped."
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
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def workspace_bytes(self) -> int:
        return sum(
            tensor.untyped_storage().nbytes() for tensor in self.workspace.values()
        )

    @property
    def execution_stats(self) -> dict[str, Any]:
        return {
            "lane": "upstream-faithful-flashgs-rgb",
            "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
            "render_requests": self._render_requests,
            "native_camera_executions": self._camera_executions,
            "true_native_batch": False,
            "fixed_capacity_sort": True,
            "max_intersections_per_camera": self.max_intersections,
            "output_contract": "rgb-only",
            "native_output": "uint8-rgb",
            "service_output": "float32-rgb",
            "conversion_in_measured_render": True,
            "full_sensor_output": False,
            "equation_contract": "public-flashgs-native-rgb-degree-zero-specialized",
            "byte_exact_public_preprocess": False,
            "byte_exact_public_sort_and_render": True,
            "performance_identical_to_unmodified_public_binary": False,
        }

    def shutdown(self) -> None:
        self.workspace.clear()
        self._scene = None
        self._covariances = None
        self._scene_id = None
        self.stage = None
        self.device = None

    def _require_initialized(self) -> None:
        if self.device is None:
            raise RuntimeError(
                "UpstreamFaithfulFlashGSBackend.initialize(stage, device) must "
                "be called first."
            )
