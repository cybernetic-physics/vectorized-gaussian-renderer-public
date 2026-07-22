"""Project-owned vectorized C++/CUDA Gaussian renderer backend."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Callable
from types import ModuleType
from typing import Any, Literal, NoReturn

import torch

from .backend import RenderRequest, RenderStats
from .native_loader import load_native_extension
from .scene import GaussianScene


FULL_SENSOR_OUTPUTS = ("rgb", "depth", "alpha", "semantic_id")
RGB_ONLY_OUTPUTS = ("rgb",)
DEFAULT_GAUSSIAN_SUPPORT_SIGMA = 3.33
COUNTER_NAMES = (
    "visible_gaussians",
    "tile_intersections",
    "visible_overflow",
    "intersection_overflow",
    "active_tiles",
)
_INT32_MAX = 2_147_483_647
_PROJECTED_RECORD_BYTES = 44
_DTYPE_BYTES = {
    torch.float32: 4,
    torch.int32: 4,
    torch.int64: 8,
    torch.uint8: 1,
    torch.uint64: 8,
}


def _emitted_sort_key_end_bit(global_tile_capacity: int) -> int:
    if global_tile_capacity <= 0:
        raise ValueError("Global tile capacity must be positive.")
    if global_tile_capacity > _INT32_MAX:
        raise ValueError(
            "Global tile capacity exceeds the int32 implementation limit."
        )
    return 32 + (global_tile_capacity - 1).bit_length()


class RendererCapacityError(RuntimeError):
    """Raised when safe capacity handling cannot produce a valid render."""

    def __init__(
        self,
        message: str,
        *,
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.telemetry = copy.deepcopy(telemetry)


class CustomCudaBackend:
    """Bounded adaptive Gaussian renderer with bounded native submissions.

    Scene tensors use the canonical inference layout: positive activated scales,
    activated opacity in ``[0, 1]``, normalized WXYZ quaternions, and RGB values
    in the first three feature channels. Raw 3DGS PLY parameters must be
    canonicalized before registration.

    ``compact_projection_cache`` selects the dense per-pixel intersection
    path. By default it uses ``max_visible_records`` only as capacity-planning
    metadata and stores one-element placeholders for projected-record arrays.
    ``materialize_projected_records`` instead stores one 44-byte screen-space
    record per first-pass camera/Gaussian candidate that contributes to at
    least one pixel. The record includes its camera ID and radii so cutoff-
    retained intersections can be emitted without projecting the scene a
    second time. If that complete workspace exceeds ``max_workspace_bytes``,
    allocation records the reason and falls back to the direct-reprojection
    path.

    A new projection input is checked once against device counters. Overflow
    triggers bounded grow/retry before output is returned. Growth releases the
    synchronized old workspace before allocating its replacement so peak memory
    does not include both complete workspaces; a failed replacement attempts to
    restore the old capacity. Identical validated inputs remain asynchronous and
    allocation-free. Set ``adaptive_capacity`` false only when the caller
    explicitly checks capacity outside measured work. Inputs without a PyTorch
    version counter, including tensors created inside ``torch.inference_mode()``,
    conservatively bypass projection and capacity-token reuse because in-place
    mutation cannot be detected from their storage identity.
    """

    def __init__(
        self,
        *,
        max_visible_records: int | None = None,
        max_intersections: int | None = None,
        visible_capacity_per_view: int = 100_000,
        intersections_per_visible: float = 4.0,
        near_plane: float = 0.01,
        far_plane: float = 100.0,
        gaussian_support_sigma: float = DEFAULT_GAUSSIAN_SUPPORT_SIGMA,
        covariance_epsilon: float = 0.3,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        semantic_min_alpha: float = 0.01,
        ray_gaussian_evaluation: bool = False,
        tile_size: int = 16,
        depth_bucket_count: int = 4096,
        depth_bucket_group_size: int = 64,
        compact_projection_cache: bool = False,
        materialize_projected_records: bool = False,
        enable_projection_cache: bool = False,
        output_srgb: bool = True,
        deterministic: bool = False,
        fixed_capacity_sort: bool = False,
        max_physical_views: int | None = None,
        adaptive_capacity: bool = True,
        capacity_growth_factor: float = 1.25,
        capacity_headroom: float = 1.25,
        max_capacity_retries: int = 4,
        max_workspace_bytes: int | None = None,
        native_module: ModuleType | Any | None = None,
        native_loader: Callable[[], ModuleType] = load_native_extension,
        allow_cpu_for_tests: bool = False,
    ) -> None:
        if max_visible_records is not None and max_visible_records <= 0:
            raise ValueError("max_visible_records must be positive.")
        if max_intersections is not None and max_intersections <= 0:
            raise ValueError("max_intersections must be positive.")
        if visible_capacity_per_view <= 0:
            raise ValueError("visible_capacity_per_view must be positive.")
        if intersections_per_visible <= 0:
            raise ValueError("intersections_per_visible must be positive.")
        if near_plane <= 0 or far_plane <= near_plane:
            raise ValueError("Expected 0 < near_plane < far_plane.")
        if gaussian_support_sigma <= 0:
            raise ValueError("gaussian_support_sigma must be positive.")
        if not math.isfinite(covariance_epsilon) or covariance_epsilon < 0:
            raise ValueError(
                "covariance_epsilon must be finite and non-negative."
            )
        if rasterize_mode not in ("classic", "antialiased"):
            raise ValueError(
                "rasterize_mode must be 'classic' or 'antialiased'."
            )
        if (
            not math.isfinite(semantic_min_alpha)
            or semantic_min_alpha < 0
            or semantic_min_alpha > 1
        ):
            raise ValueError(
                "semantic_min_alpha must be finite and in [0, 1]."
            )
        if (
            tile_size <= 0
            or tile_size > 16
            or tile_size & (tile_size - 1)
        ):
            raise ValueError(
                "tile_size must be a power of two in [1, 16]."
            )
        if depth_bucket_count <= 0:
            raise ValueError("depth_bucket_count must be positive.")
        if depth_bucket_group_size <= 0:
            raise ValueError("depth_bucket_group_size must be positive.")
        if compact_projection_cache and tile_size != 1:
            raise ValueError(
                "compact_projection_cache requires tile_size=1."
            )
        if compact_projection_cache and ray_gaussian_evaluation:
            raise ValueError(
                "compact_projection_cache currently supports only the "
                "screen-space Gaussian model."
            )
        if rasterize_mode == "antialiased" and ray_gaussian_evaluation:
            raise ValueError(
                "rasterize_mode='antialiased' is incompatible with "
                "ray_gaussian_evaluation=True because the exact-ray path "
                "does not use the screen-space covariance dilation being "
                "compensated."
            )
        if compact_projection_cache and deterministic:
            raise ValueError(
                "compact_projection_cache currently requires "
                "deterministic=False."
            )
        if materialize_projected_records and not compact_projection_cache:
            raise ValueError(
                "materialize_projected_records requires "
                "compact_projection_cache=True."
            )
        if fixed_capacity_sort and deterministic:
            raise ValueError(
                "fixed_capacity_sort is only meaningful for the "
                "nondeterministic global-radix path."
            )
        if max_physical_views is not None and max_physical_views <= 0:
            raise ValueError("max_physical_views must be positive when set.")
        if max_physical_views is not None and not fixed_capacity_sort:
            raise ValueError(
                "max_physical_views requires fixed_capacity_sort=True so "
                "logical chunks never synchronize on an emitted count."
            )
        if max_physical_views is not None and adaptive_capacity:
            raise ValueError(
                "max_physical_views requires adaptive_capacity=False; "
                "physical capacities must be calibrated and installed before "
                "steady-state logical-batch rendering."
            )
        if max_physical_views is not None and enable_projection_cache:
            raise ValueError(
                "max_physical_views currently requires "
                "enable_projection_cache=False."
            )
        if (
            not math.isfinite(capacity_growth_factor)
            or capacity_growth_factor <= 1.0
        ):
            raise ValueError(
                "capacity_growth_factor must be finite and greater than 1."
            )
        if not math.isfinite(capacity_headroom) or capacity_headroom < 1.0:
            raise ValueError(
                "capacity_headroom must be finite and at least 1."
            )
        if (
            isinstance(max_capacity_retries, bool)
            or not isinstance(max_capacity_retries, int)
            or max_capacity_retries < 0
        ):
            raise ValueError("max_capacity_retries must be non-negative.")
        if max_workspace_bytes is not None and max_workspace_bytes <= 0:
            raise ValueError("max_workspace_bytes must be positive when set.")

        self.max_visible_records = max_visible_records
        self.max_intersections = max_intersections
        self.visible_capacity_per_view = visible_capacity_per_view
        self.intersections_per_visible = intersections_per_visible
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.gaussian_support_sigma = gaussian_support_sigma
        self.covariance_epsilon = covariance_epsilon
        self._rasterize_mode: Literal["classic", "antialiased"] = "classic"
        self.rasterize_mode = rasterize_mode
        self.semantic_min_alpha = semantic_min_alpha
        self.ray_gaussian_evaluation = ray_gaussian_evaluation
        self.tile_size = tile_size
        self.depth_bucket_count = depth_bucket_count
        self.depth_bucket_group_size = min(
            depth_bucket_group_size,
            depth_bucket_count,
        )
        self.compact_projection_cache = compact_projection_cache
        self.materialize_projected_records = materialize_projected_records
        self.enable_projection_cache = enable_projection_cache
        self.output_srgb = output_srgb
        self.deterministic = deterministic
        self.fixed_capacity_sort = fixed_capacity_sort
        self.max_physical_views = max_physical_views
        self.adaptive_capacity = adaptive_capacity
        self.capacity_growth_factor = capacity_growth_factor
        self.capacity_headroom = capacity_headroom
        self.max_capacity_retries = max_capacity_retries
        self.max_workspace_bytes = max_workspace_bytes
        self._native = native_module
        self._native_loader = native_loader
        self.allow_cpu_for_tests = allow_cpu_for_tests

        self.device: torch.device | None = None
        self.stage: Any | None = None
        self.workspace: dict[str, torch.Tensor] = {}
        self._scenes: OrderedDict[int, GaussianScene] = OrderedDict()
        self._packed_scene: dict[str, torch.Tensor] = {}
        self._max_scene_gaussians = 0
        self._env_scene_ids: torch.Tensor | None = None
        self._env_transforms: torch.Tensor | None = None
        self._max_views = 0
        self._physical_max_views = 0
        self._max_logical_chunks = 1
        self._height = 0
        self._width = 0
        self._scene_revision = 0
        self._projection_cache_token: tuple[Any, ...] | None = None
        self._projection_cache_tensor_refs: tuple[torch.Tensor, ...] = ()
        self._projection_cache_hits = 0
        self._projection_cache_misses = 0
        self._allocated_visible_capacity = 0
        self._allocated_intersection_capacity = 0
        self._materialize_projected_records_active = False
        self._projected_record_fallback_reason: str | None = None
        self._initial_visible_capacity = 0
        self._initial_intersection_capacity = 0
        self._workspace_config: tuple[
            int,
            int,
            int,
            torch.device,
            torch.dtype,
        ] | None = None
        self._capacity_validation_token: tuple[Any, ...] | None = None
        self._capacity_validation_tensor_refs: tuple[torch.Tensor, ...] = ()
        self._capacity_validation_counters: dict[str, int] | None = None
        self._capacity_validations = 0
        self._capacity_validation_reuses = 0
        self._capacity_growth_events = 0
        self._capacity_total_retries = 0
        self._capacity_last_render: dict[str, Any] | None = None
        self._capacity_last_growth: dict[str, Any] | None = None
        self._capacity_last_allocation: dict[str, Any] | None = None

    def initialize(self, stage: Any, device: torch.device) -> None:
        if device.type != "cuda" and not self.allow_cpu_for_tests:
            raise ValueError(f"CustomCudaBackend requires CUDA; got {device}.")
        self.stage = stage
        self.device = device
        if self._native is None:
            self._native = self._native_loader()
        self.invalidate_projection_cache()
        self._projection_cache_hits = 0
        self._projection_cache_misses = 0
        self._capacity_validations = 0
        self._capacity_validation_reuses = 0
        self._capacity_growth_events = 0
        self._capacity_total_retries = 0
        self._capacity_last_render = None
        self._capacity_last_growth = None
        self._capacity_last_allocation = None

    def register_scene(self, scene_id: int, scene: GaussianScene, offset: int) -> None:
        self._require_initialized()
        expected_offset = sum(registered.count for registered in self._scenes.values())
        if offset != expected_offset:
            raise ValueError(
                f"Scene {scene_id} offset {offset} does not match packed offset {expected_offset}."
            )
        if scene.features.shape[1] < 3:
            raise ValueError("CustomCudaBackend requires at least three RGB feature channels.")
        if self._scenes:
            feature_width = next(iter(self._scenes.values())).features.shape[1]
            if scene.features.shape[1] != feature_width:
                raise ValueError(
                    "All packed scenes must use the same feature width; "
                    f"expected {feature_width}, got {scene.features.shape[1]}."
                )
        self._scenes[scene_id] = scene
        self._rebuild_packed_scene()

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
        if dtype != torch.float32:
            raise ValueError("The first custom CUDA backend workspace uses float32.")
        if max_views <= 0:
            raise ValueError("max_views must be positive.")

        physical_max_views = min(
            max_views,
            self.max_physical_views or max_views,
        )
        visible_capacity = self.max_visible_records or (
            physical_max_views * self.visible_capacity_per_view
        )
        intersection_capacity = self.max_intersections or math.ceil(
            visible_capacity * self.intersections_per_visible
        )
        self._max_views = max_views
        self._physical_max_views = physical_max_views
        self._max_logical_chunks = math.ceil(max_views / physical_max_views)
        self._workspace_config = (
            physical_max_views,
            height,
            width,
            device,
            dtype,
        )
        self._install_workspace(
            visible_capacity=visible_capacity,
            intersection_capacity=intersection_capacity,
            reason="initial-allocation",
        )
        self._initial_visible_capacity = visible_capacity
        self._initial_intersection_capacity = intersection_capacity
        return self.workspace

    def update_scene_transforms(
        self,
        scene_ids: torch.Tensor,
        transforms: torch.Tensor,
    ) -> None:
        self._env_scene_ids = scene_ids
        self._env_transforms = transforms
        self.invalidate_projection_cache()

    def reserve_workspace_capacity(
        self,
        *,
        visible_records: int,
        tile_intersections: int,
    ) -> None:
        """Install an explicit synchronized capacity outside measured work."""
        self._require_initialized()
        if not self.workspace:
            raise RuntimeError("allocate_workspace() must be called before reservation.")
        self.synchronize()
        self._install_workspace(
            visible_capacity=visible_records,
            intersection_capacity=tile_intersections,
            reason="explicit-benchmark-reservation",
        )

    def render(self, request: RenderRequest) -> RenderStats:
        self._require_initialized()
        if not self.workspace:
            raise RuntimeError("allocate_workspace() must be called before render().")
        if not self._packed_scene:
            raise RuntimeError("At least one Gaussian scene must be registered.")
        output_names = tuple(request.outputs)
        if output_names not in (RGB_ONLY_OUTPUTS, FULL_SENSOR_OUTPUTS):
            raise ValueError(
                "CustomCudaBackend supports exactly the RGB-only or fused "
                "RGB/depth/alpha/semantic_id output contract; got "
                f"{output_names}."
            )
        full_sensor_output = output_names == FULL_SENSOR_OUTPUTS
        batch = int(request.camera_transforms.shape[0])
        if batch > self._max_views:
            raise ValueError(f"Render batch {batch} exceeds workspace max_views {self._max_views}.")
        if request.outputs["rgb"].shape[1:3] != (self._height, self._width):
            raise ValueError("Output resolution does not match the allocated workspace.")

        attempts: list[dict[str, Any]] = []
        for retry_count in range(self.max_capacity_retries + 1):
            if batch > self._physical_max_views:
                stats, input_token, tensor_refs = (
                    self._submit_chunked_render(
                        request,
                        full_sensor_output=full_sensor_output,
                    )
                )
            else:
                stats, input_token, tensor_refs = self._submit_render(
                    request,
                    full_sensor_output=full_sensor_output,
                )
                if self.max_physical_views is not None:
                    self.workspace["physical_capacity_counters"].copy_(
                        self.workspace["counters"]
                    )
            if not self.adaptive_capacity:
                self._capacity_last_render = {
                    "status": "submitted-unvalidated",
                    "validation": "explicit-manual-capacity-mode",
                    "synchronized": False,
                    "retry_count": 0,
                    "attempts": [],
                    "final_capacity": self._capacity_snapshot(),
                    "final_counters": None,
                    "headroom": None,
                }
                return stats

            if (
                input_token == self._capacity_validation_token
                and self._capacity_validation_counters is not None
            ):
                counters = dict(self._capacity_validation_counters)
                self._capacity_validation_reuses += 1
                attempt = self._capacity_attempt_record(
                    retry_count=retry_count,
                    counters=counters,
                    counter_source="validated-identical-input",
                    synchronized=False,
                )
                self._capacity_last_render = {
                    "status": "success",
                    "validation": "validated-identical-input",
                    "synchronized": False,
                    "retry_count": retry_count,
                    "attempts": [attempt],
                    "final_capacity": self._capacity_snapshot(),
                    "final_counters": counters,
                    "headroom": self._capacity_headroom(counters),
                }
                return RenderStats(
                    rendered_views=stats.rendered_views,
                    visible_gaussians=counters["visible_gaussians"],
                    tile_intersections=counters["tile_intersections"],
                )

            counters = self.read_counters(synchronize=True)
            attempt = self._capacity_attempt_record(
                retry_count=retry_count,
                counters=counters,
                counter_source="device-counters",
                synchronized=True,
            )
            attempts.append(attempt)
            overflow = bool(
                counters["visible_overflow"]
                or counters["intersection_overflow"]
            )
            if not overflow:
                self._capacity_validation_token = input_token
                self._capacity_validation_tensor_refs = tensor_refs
                self._capacity_validation_counters = dict(counters)
                self._capacity_validations += 1
                self._capacity_total_retries += retry_count
                render_record = {
                    "status": "success",
                    "validation": "device-counters",
                    "synchronized": True,
                    "retry_count": retry_count,
                    "attempts": attempts,
                    "final_capacity": self._capacity_snapshot(),
                    "final_counters": counters,
                    "headroom": self._capacity_headroom(counters),
                }
                self._capacity_last_render = render_record
                if retry_count:
                    self._capacity_last_growth = copy.deepcopy(render_record)
                return RenderStats(
                    rendered_views=stats.rendered_views,
                    visible_gaussians=counters["visible_gaussians"],
                    tile_intersections=counters["tile_intersections"],
                )

            if retry_count >= self.max_capacity_retries:
                self._capacity_total_retries += retry_count
                self._raise_render_capacity_error(
                    request,
                    message=(
                        "Custom renderer capacity remained insufficient after "
                        f"{retry_count} bounded retries."
                    ),
                    attempts=attempts,
                    reason="retry-limit",
                )

            (
                proposed_visible,
                proposed_intersections,
                workspace_ceiling_adjustment,
            ) = self._propose_capacity_growth(counters)
            attempt["growth_request"] = {
                "visible_records": proposed_visible,
                "tile_intersections": proposed_intersections,
            }
            if workspace_ceiling_adjustment is not None:
                attempt["workspace_ceiling_adjustment"] = (
                    workspace_ceiling_adjustment
                )
            try:
                self._install_workspace(
                    visible_capacity=proposed_visible,
                    intersection_capacity=proposed_intersections,
                    reason="overflow-retry",
                )
            except RendererCapacityError as error:
                attempt["growth_failure"] = copy.deepcopy(error.telemetry)
                self._capacity_total_retries += retry_count
                self._raise_render_capacity_error(
                    request,
                    message=str(error),
                    attempts=attempts,
                    reason="growth-allocation-failed",
                    cause=error,
                )
            self._capacity_growth_events += 1

        raise AssertionError("Unreachable capacity retry state.")

    def _submit_render(
        self,
        request: RenderRequest,
        *,
        full_sensor_output: bool,
    ) -> tuple[
        RenderStats,
        tuple[Any, ...] | None,
        tuple[torch.Tensor, ...],
    ]:
        batch = int(request.camera_transforms.shape[0])

        if self._env_transforms is None:
            env_xforms = self.workspace["identity_env_xforms"][:batch]
        else:
            if self._env_transforms.shape[0] != batch:
                raise ValueError(
                    "update_scene_transforms() must provide one transform per render camera; "
                    f"got {self._env_transforms.shape[0]} transforms for batch {batch}."
                )
            env_xforms = self._env_transforms

        active_ids = (
            request.active_camera_ids
            if request.active_camera_ids is not None
            else self.workspace["empty_active_ids"]
        )
        projection_cache_token: tuple[Any, ...] | None = None
        reuse_projection = False
        if self.enable_projection_cache or self.adaptive_capacity:
            projection_cache_token = self._projection_input_token(
                batch=batch,
                request=request,
                env_xforms=env_xforms,
                active_ids=active_ids,
            )
        if self.enable_projection_cache:
            reuse_projection = (
                projection_cache_token == self._projection_cache_token
            )
        scene_tensors = [
            self._packed_scene["means"],
            self._packed_scene["covariances"],
            self._packed_scene["opacities"],
            self._packed_scene["features"],
            self._packed_scene["semantic_ids"],
            self._packed_scene["registered_scene_ids"],
            self._packed_scene["scene_offsets"],
        ]
        camera_tensors = [
            request.camera_transforms,
            request.intrinsics,
            env_xforms,
            request.scene_ids,
            active_ids,
        ]
        output_tensors = [
            request.outputs["rgb"],
            (
                request.outputs["depth"]
                if full_sensor_output
                else self.workspace["dummy_depth"]
            ),
            (
                request.outputs["alpha"]
                if full_sensor_output
                else self.workspace["dummy_alpha"]
            ),
            (
                request.outputs["semantic_id"]
                if full_sensor_output
                else self.workspace["dummy_semantic"]
            ),
        ]
        if request.active_camera_ids is not None:
            # The native kernels only visit active cameras. Define every
            # inactive output on every logical render so caller-supplied or
            # alternating buffers can never expose stale observations.
            request.outputs["rgb"].zero_()
            if full_sensor_output:
                request.outputs["depth"].fill_(float("inf"))
                request.outputs["alpha"].zero_()
                request.outputs["semantic_id"].fill_(-1)
        workspace_tensors = [
            self.workspace["visible_means2d"],
            self.workspace["visible_conics"],
            self.workspace["visible_depths"],
            self.workspace["visible_opacities"],
            self.workspace["visible_camera_ids"],
            self.workspace["visible_gaussian_ids"],
            self.workspace["visible_radii"],
            self.workspace["keys_in"],
            self.workspace["keys_out"],
            self.workspace["values_in"],
            self.workspace["values_out"],
            self.workspace["tile_starts"],
            self.workspace["tile_ends"],
            self.workspace["counters"],
            self.workspace["sort_temp"],
            self.workspace["depth_bucket_tau"],
            self.workspace["depth_cutoff"],
            self.workspace["depth_bucket_counts"],
            self.workspace["depth_bucket_offsets"],
            self.workspace["depth_bucket_write_offsets"],
            self.workspace["depth_ordered_visible_indices"],
            self.workspace["depth_accumulated_tau"],
            self.workspace["visible_ray_precisions"],
            self.workspace["visible_ray_precision_means"],
        ]
        assert self._native is not None
        sorted_buffer_selector = int(
            self._native.render(
                scene_tensors,
                camera_tensors,
                output_tensors,
                workspace_tensors,
                self._height,
                self._width,
                self._max_scene_gaussians,
                self.near_plane,
                self.far_plane,
                self.gaussian_support_sigma,
                self.covariance_epsilon,
                self.rasterize_mode == "antialiased",
                self.semantic_min_alpha,
                self.ray_gaussian_evaluation,
                self.tile_size,
                self.depth_bucket_count,
                self.depth_bucket_group_size,
                self.compact_projection_cache,
                self._materialize_projected_records_active,
                reuse_projection,
                self.output_srgb,
                self.deterministic,
                full_sensor_output,
                self.fixed_capacity_sort,
            )
        )
        if sorted_buffer_selector not in (0, 1):
            raise RuntimeError(
                "Native renderer returned an invalid sorted-buffer selector: "
                f"{sorted_buffer_selector}."
            )
        if sorted_buffer_selector == 0:
            # CUB's DoubleBuffer API may leave the sorted prefix in either
            # caller-owned buffer. Normalize the public workspace contract so
            # values_out remains the sorted buffer for projection-cache hits,
            # diagnostics, and deterministic smoke scripts, without copying a
            # potentially multi-gigabyte emitted prefix.
            self.workspace["keys_in"], self.workspace["keys_out"] = (
                self.workspace["keys_out"],
                self.workspace["keys_in"],
            )
            self.workspace["values_in"], self.workspace["values_out"] = (
                self.workspace["values_out"],
                self.workspace["values_in"],
            )
        if self.enable_projection_cache:
            self._projection_cache_token = projection_cache_token
            # Retain the tensors whose storage identities form the cache key.
            # Without strong references, an allocator can recycle the same
            # address for a new moving-camera tensor with version zero and
            # create a false hit despite different values.
            self._projection_cache_tensor_refs = tuple(camera_tensors)
            if reuse_projection:
                self._projection_cache_hits += 1
            else:
                self._projection_cache_misses += 1
        rendered_views = (
            int(request.active_camera_ids.numel())
            if request.active_camera_ids is not None
            else batch
        )
        return (
            RenderStats(rendered_views=rendered_views),
            projection_cache_token,
            tuple(camera_tensors),
        )

    def _submit_chunked_render(
        self,
        request: RenderRequest,
        *,
        full_sensor_output: bool,
    ) -> tuple[
        RenderStats,
        tuple[Any, ...] | None,
        tuple[torch.Tensor, ...],
    ]:
        """Submit one logical request through bounded physical workspaces.

        The logical camera/output tensors remain in caller order.  Only
        contiguous first-dimension views are passed to each native submission,
        so the measured path creates no CUDA storage and performs no host
        readback.  Chunk counters are reduced on the active CUDA stream after
        the final submission.
        """
        batch = int(request.camera_transforms.shape[0])
        if self.max_physical_views is None:
            raise AssertionError("Chunked submission requires max_physical_views.")
        if not self.fixed_capacity_sort or self.adaptive_capacity:
            raise AssertionError(
                "Chunked submission requires fixed, preflighted capacities."
            )
        if self.enable_projection_cache:
            raise AssertionError("Chunked projection-cache reuse is unsupported.")
        chunk_count = math.ceil(batch / self._physical_max_views)
        if chunk_count > self._max_logical_chunks:
            raise ValueError(
                f"Logical batch requires {chunk_count} chunks but only "
                f"{self._max_logical_chunks} were allocated."
            )

        if self._env_transforms is None:
            env_xforms = None
        else:
            if self._env_transforms.shape[0] != batch:
                raise ValueError(
                    "update_scene_transforms() must provide one transform per "
                    "logical render camera; got "
                    f"{self._env_transforms.shape[0]} transforms for batch "
                    f"{batch}."
                )
            env_xforms = self._env_transforms

        logical_active_ids = request.active_camera_ids
        if logical_active_ids is not None:
            # Define inactive logical outputs once.  Every native chunk then
            # writes only the valid local IDs prepared on-device below.
            request.outputs["rgb"].zero_()
            if full_sensor_output:
                request.outputs["depth"].fill_(float("inf"))
                request.outputs["alpha"].zero_()
                request.outputs["semantic_id"].fill_(-1)
            assert self._native is not None
            self._native.prepare_chunked_active_ids(
                logical_active_ids,
                self.workspace["logical_active_mask"],
                self.workspace["logical_chunk_active_ids"],
                batch,
                self._physical_max_views,
            )

        scene_tensors = [
            self._packed_scene["means"],
            self._packed_scene["covariances"],
            self._packed_scene["opacities"],
            self._packed_scene["features"],
            self._packed_scene["semantic_ids"],
            self._packed_scene["registered_scene_ids"],
            self._packed_scene["scene_offsets"],
        ]
        chunk_counters = self.workspace["logical_chunk_counters"]
        assert self._native is not None
        for chunk_index in range(chunk_count):
            start = chunk_index * self._physical_max_views
            count = min(self._physical_max_views, batch - start)
            camera_tensors = [
                request.camera_transforms.narrow(0, start, count),
                request.intrinsics.narrow(0, start, count),
                (
                    self.workspace["identity_env_xforms"].narrow(
                        0,
                        0,
                        count,
                    )
                    if env_xforms is None
                    else env_xforms.narrow(0, start, count)
                ),
                request.scene_ids.narrow(0, start, count),
                (
                    self.workspace["empty_active_ids"]
                    if logical_active_ids is None
                    else self.workspace["logical_chunk_active_ids"][
                        chunk_index
                    ].narrow(0, 0, count)
                ),
            ]
            output_tensors = [
                request.outputs["rgb"].narrow(0, start, count),
                (
                    request.outputs["depth"].narrow(0, start, count)
                    if full_sensor_output
                    else self.workspace["dummy_depth"]
                ),
                (
                    request.outputs["alpha"].narrow(0, start, count)
                    if full_sensor_output
                    else self.workspace["dummy_alpha"]
                ),
                (
                    request.outputs["semantic_id"].narrow(0, start, count)
                    if full_sensor_output
                    else self.workspace["dummy_semantic"]
                ),
            ]
            workspace_tensors = [
                self.workspace["visible_means2d"],
                self.workspace["visible_conics"],
                self.workspace["visible_depths"],
                self.workspace["visible_opacities"],
                self.workspace["visible_camera_ids"],
                self.workspace["visible_gaussian_ids"],
                self.workspace["visible_radii"],
                self.workspace["keys_in"],
                self.workspace["keys_out"],
                self.workspace["values_in"],
                self.workspace["values_out"],
                self.workspace["tile_starts"],
                self.workspace["tile_ends"],
                chunk_counters[chunk_index],
                self.workspace["sort_temp"],
                self.workspace["depth_bucket_tau"],
                self.workspace["depth_cutoff"],
                self.workspace["depth_bucket_counts"],
                self.workspace["depth_bucket_offsets"],
                self.workspace["depth_bucket_write_offsets"],
                self.workspace["depth_ordered_visible_indices"],
                self.workspace["depth_accumulated_tau"],
                self.workspace["visible_ray_precisions"],
                self.workspace["visible_ray_precision_means"],
            ]
            sorted_buffer_selector = int(
                self._native.render(
                    scene_tensors,
                    camera_tensors,
                    output_tensors,
                    workspace_tensors,
                    self._height,
                    self._width,
                    self._max_scene_gaussians,
                    self.near_plane,
                    self.far_plane,
                    self.gaussian_support_sigma,
                    self.covariance_epsilon,
                    self.rasterize_mode == "antialiased",
                    self.semantic_min_alpha,
                    self.ray_gaussian_evaluation,
                    self.tile_size,
                    self.depth_bucket_count,
                    self.depth_bucket_group_size,
                    self.compact_projection_cache,
                    self._materialize_projected_records_active,
                    False,
                    self.output_srgb,
                    self.deterministic,
                    full_sensor_output,
                    self.fixed_capacity_sort,
                )
            )
            if sorted_buffer_selector not in (0, 1):
                raise RuntimeError(
                    "Native renderer returned an invalid sorted-buffer "
                    f"selector: {sorted_buffer_selector}."
                )
            if sorted_buffer_selector == 0:
                self.workspace["keys_in"], self.workspace["keys_out"] = (
                    self.workspace["keys_out"],
                    self.workspace["keys_in"],
                )
                self.workspace["values_in"], self.workspace["values_out"] = (
                    self.workspace["values_out"],
                    self.workspace["values_in"],
                )

        self._native.aggregate_chunk_counters(
            chunk_counters,
            self.workspace["counters"],
            self.workspace["physical_capacity_counters"],
            chunk_count,
        )
        rendered_views = (
            int(logical_active_ids.numel())
            if logical_active_ids is not None
            else batch
        )
        tensor_refs = (
            request.camera_transforms,
            request.intrinsics,
            request.scene_ids,
            *(() if env_xforms is None else (env_xforms,)),
            *(
                ()
                if logical_active_ids is None
                else (logical_active_ids,)
            ),
        )
        return RenderStats(rendered_views=rendered_views), None, tensor_refs

    def synchronize(self) -> None:
        self._require_initialized()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def read_counters(self, *, synchronize: bool = True) -> dict[str, int]:
        """Read device counters outside the steady-state render submission."""
        if synchronize:
            self.synchronize()
        values = self.workspace["counters"].detach().cpu().tolist()
        return dict(zip(COUNTER_NAMES, (int(value) for value in values), strict=True))

    def read_physical_capacity_counters(
        self,
        *,
        synchronize: bool = True,
    ) -> dict[str, int]:
        """Read maximum per-chunk demand outside steady-state submission."""
        if "physical_capacity_counters" not in self.workspace:
            return self.read_counters(synchronize=synchronize)
        if synchronize:
            self.synchronize()
        values = (
            self.workspace["physical_capacity_counters"]
            .detach()
            .cpu()
            .tolist()
        )
        return dict(
            zip(
                COUNTER_NAMES,
                (int(value) for value in values),
                strict=True,
            )
        )

    def check_capacity(self, *, synchronize: bool = True) -> dict[str, int]:
        counters = self.read_counters(synchronize=synchronize)
        if counters["visible_overflow"] or counters["intersection_overflow"]:
            raise RendererCapacityError(
                "Custom renderer workspace overflow: "
                f"visible={counters['visible_overflow']} "
                f"intersections={counters['intersection_overflow']}.",
                telemetry={
                    "schema_version": "renderer-capacity-error/v1",
                    "reason": "explicit-check-overflow",
                    "counters": counters,
                    "capacity": self.capacity_stats,
                },
            )
        return counters

    def _workspace_plan(
        self,
        *,
        visible_capacity: int,
        intersection_capacity: int,
    ) -> dict[str, Any]:
        if self._workspace_config is None:
            raise RuntimeError("Workspace configuration is unavailable.")
        max_views, height, width, device, _dtype = self._workspace_config
        if visible_capacity <= 0 or intersection_capacity <= 0:
            raise ValueError("Workspace capacities must be positive.")
        materialize_projected_records_active = bool(
            self.compact_projection_cache
            and self.materialize_projected_records
        )
        projected_record_fallback_reason: str | None = None
        visible_storage_capacity = (
            visible_capacity
            if (
                not self.compact_projection_cache
                or materialize_projected_records_active
            )
            else 1
        )
        projected_candidate_metadata_capacity = visible_storage_capacity
        if visible_storage_capacity > _INT32_MAX:
            raise ValueError(
                "Visible storage capacity exceeds the int32 implementation "
                "limit."
            )
        if intersection_capacity > _INT32_MAX:
            raise ValueError(
                "Intersection capacity exceeds the int32 implementation "
                "limit."
            )

        tiles_x = math.ceil(width / self.bin_tile_size)
        tiles_y = math.ceil(height / self.bin_tile_size)
        global_tiles = max_views * tiles_x * tiles_y
        assert self._native is not None
        sort_temp_strategy = (
            "pointer-api-global-segmented-scan"
            if self.deterministic
            else "double-buffer-global-radix"
        )
        sort_key_end_bit = (
            64
            if self.deterministic
            else _emitted_sort_key_end_bit(global_tiles)
        )
        sort_radix_digit_bits_assumption = 8
        sort_radix_passes_expected = math.ceil(
            sort_key_end_bit / sort_radix_digit_bits_assumption
        )
        if self.deterministic:
            sort_temp_bytes = int(
                self._native.sort_temp_bytes(
                    intersection_capacity,
                    global_tiles,
                )
            )
        else:
            sort_temp_bytes = int(
                self._native.radix_sort_double_buffer_temp_bytes(
                    intersection_capacity,
                    global_tiles,
                )
            )
        dense_depth_shape = (
            (
                global_tiles,
                (
                    self.depth_bucket_count
                    if self.compact_projection_cache
                    else self.depth_bucket_group_size
                ),
            )
            if self.tile_size == 1
            else (1, 1)
        )
        depth_bucket_tau_bytes = (
            math.prod(dense_depth_shape) * _DTYPE_BYTES[torch.float32]
        )
        shared_phase_scratch_bytes = max(
            sort_temp_bytes,
            depth_bucket_tau_bytes,
        )
        dense_pixel_shape = (
            (global_tiles,) if self.tile_size == 1 else (1,)
        )
        dense_bucket_shape = (
            (self.depth_bucket_count,)
            if self.tile_size == 1 and not self.compact_projection_cache
            else (1,)
        )
        dense_bucket_offset_shape = (
            (self.depth_bucket_count + 1,)
            if self.tile_size == 1 and not self.compact_projection_cache
            else (1,)
        )
        dense_visible_shape = (
            (visible_capacity,)
            if self.tile_size == 1 and not self.compact_projection_cache
            else (1,)
        )
        ray_precision_shape = (
            (visible_storage_capacity, 6)
            if self.ray_gaussian_evaluation
            else (1, 6)
        )
        ray_precision_mean_shape = (
            (visible_storage_capacity, 4)
            if self.ray_gaussian_evaluation
            else (1, 4)
        )
        specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "visible_means2d": (
                (visible_storage_capacity, 2),
                torch.float32,
            ),
            "visible_conics": (
                (visible_storage_capacity, 3),
                torch.float32,
            ),
            "visible_depths": (
                (visible_storage_capacity,),
                torch.float32,
            ),
            "visible_opacities": (
                (visible_storage_capacity,),
                torch.float32,
            ),
            "visible_camera_ids": (
                (projected_candidate_metadata_capacity,),
                torch.int32,
            ),
            "visible_gaussian_ids": (
                (visible_storage_capacity,),
                torch.int32,
            ),
            "visible_radii": (
                (projected_candidate_metadata_capacity, 2),
                torch.int32,
            ),
            "keys_in": ((intersection_capacity,), torch.uint64),
            "keys_out": ((intersection_capacity,), torch.uint64),
            "values_in": ((intersection_capacity,), torch.int32),
            "values_out": ((intersection_capacity,), torch.int32),
            # Native intersection capacity is capped at INT_MAX, so every
            # inclusive/exclusive tile-range endpoint is exactly representable
            # in int32. Keeping these dense per-tile arrays int64 only doubled
            # their footprint at large camera batches.
            "tile_starts": ((global_tiles,), torch.int32),
            "tile_ends": ((global_tiles,), torch.int32),
            "counters": ((len(COUNTER_NAMES),), torch.int64),
            # Depth-bucket accumulation finishes before radix sorting begins
            # on the same CUDA stream. Back both phase-disjoint views with one
            # allocation sized for the larger requirement.
            "sort_temp": ((shared_phase_scratch_bytes,), torch.uint8),
            "depth_bucket_tau": (dense_depth_shape, torch.float32),
            "depth_cutoff": (dense_pixel_shape, torch.int32),
            "depth_bucket_counts": (dense_bucket_shape, torch.int64),
            "depth_bucket_offsets": (
                dense_bucket_offset_shape,
                torch.int64,
            ),
            "depth_bucket_write_offsets": (
                dense_bucket_shape,
                torch.int64,
            ),
            "depth_ordered_visible_indices": (
                dense_visible_shape,
                torch.int32,
            ),
            "depth_accumulated_tau": (
                dense_pixel_shape,
                torch.float32,
            ),
            "visible_ray_precisions": (
                ray_precision_shape,
                torch.float32,
            ),
            "visible_ray_precision_means": (
                ray_precision_mean_shape,
                torch.float32,
            ),
            "identity_env_xforms": ((max_views, 4, 4), torch.float32),
            "empty_active_ids": ((0,), torch.int64),
            # Native signatures stay fixed across output contracts. RGB-only
            # uses these one-element sinks, which are never indexed by the
            # compile-time RGB compositor specialization.
            "dummy_depth": ((1,), torch.float32),
            "dummy_alpha": ((1,), torch.float32),
            "dummy_semantic": ((1,), torch.int64),
        }
        if self.max_physical_views is not None:
            specs.update(
                {
                    # Each physical submission writes its own counters.  A
                    # tiny device kernel reduces these rows into ``counters``
                    # after the final chunk, retaining logical-batch totals
                    # without a host readback or stream synchronization.
                    "logical_chunk_counters": (
                        (self._max_logical_chunks, len(COUNTER_NAMES)),
                        torch.int64,
                    ),
                    "physical_capacity_counters": (
                        (len(COUNTER_NAMES),),
                        torch.int64,
                    ),
                    # Active subsets stay device-resident.  The native
                    # preparation kernel converts logical IDs into padded
                    # chunk-local IDs; -1 entries are ignored by the existing
                    # projection and compositor kernels.
                    "logical_active_mask": (
                        (self._max_views,),
                        torch.int32,
                    ),
                    "logical_chunk_active_ids": (
                        (self._max_logical_chunks, max_views),
                        torch.int64,
                    ),
                }
            )

        def workspace_sizes() -> tuple[int, int, int]:
            logical_bytes = sum(
                math.prod(shape) * _DTYPE_BYTES[tensor_dtype]
                for shape, tensor_dtype in specs.values()
            )
            physical_bytes = logical_bytes - depth_bucket_tau_bytes
            unaliased_bytes = physical_bytes + min(
                sort_temp_bytes,
                depth_bucket_tau_bytes,
            )
            return logical_bytes, physical_bytes, unaliased_bytes

        (
            workspace_logical_bytes,
            workspace_bytes,
            workspace_unaliased_bytes,
        ) = workspace_sizes()
        materialized_workspace_bytes = (
            workspace_bytes
            if materialize_projected_records_active
            else None
        )
        if (
            materialize_projected_records_active
            and self.max_workspace_bytes is not None
            and workspace_bytes > self.max_workspace_bytes
        ):
            materialize_projected_records_active = False
            projected_record_fallback_reason = "workspace-memory-ceiling"
            visible_storage_capacity = 1
            for name, shape in (
                ("visible_means2d", (1, 2)),
                ("visible_conics", (1, 3)),
                ("visible_depths", (1,)),
                ("visible_opacities", (1,)),
                ("visible_camera_ids", (1,)),
                ("visible_gaussian_ids", (1,)),
                ("visible_radii", (1, 2)),
            ):
                _old_shape, tensor_dtype = specs[name]
                specs[name] = (shape, tensor_dtype)
            (
                workspace_logical_bytes,
                workspace_bytes,
                workspace_unaliased_bytes,
            ) = workspace_sizes()
        return {
            "visible_capacity": visible_capacity,
            "visible_storage_capacity": visible_storage_capacity,
            "intersection_capacity": intersection_capacity,
            "global_tiles": global_tiles,
            "workspace_bytes": workspace_bytes,
            "workspace_logical_bytes": workspace_logical_bytes,
            "workspace_unaliased_bytes": workspace_unaliased_bytes,
            "workspace_alias_savings_bytes": min(
                sort_temp_bytes,
                depth_bucket_tau_bytes,
            ),
            "projected_record_reuse": {
                "requested": self.materialize_projected_records,
                "active": materialize_projected_records_active,
                "fallback_reason": projected_record_fallback_reason,
                "bytes_per_record": _PROJECTED_RECORD_BYTES,
                "materialized_workspace_bytes": (
                    materialized_workspace_bytes
                ),
            },
            "sort_temp_bytes": sort_temp_bytes,
            "sort_temp_strategy": sort_temp_strategy,
            "sort_key_begin_bit": 0,
            "sort_key_end_bit": sort_key_end_bit,
            "sort_key_bits": sort_key_end_bit,
            "sort_radix_digit_bits_assumption": (
                sort_radix_digit_bits_assumption
            ),
            "sort_radix_passes_expected": sort_radix_passes_expected,
            "workspace_storage_aliases": {
                "depth_bucket_tau": "sort_temp",
            },
            "device": device,
            "specs": specs,
        }

    def _install_workspace(
        self,
        *,
        visible_capacity: int,
        intersection_capacity: int,
        reason: str,
    ) -> None:
        try:
            plan = self._workspace_plan(
                visible_capacity=visible_capacity,
                intersection_capacity=intersection_capacity,
            )
        except ValueError as error:
            if reason == "initial-allocation":
                raise
            raise RendererCapacityError(
                str(error),
                telemetry={
                    "schema_version": "renderer-capacity-allocation/v1",
                    "reason": reason,
                    "failure": "native-int32-limit",
                    "requested_capacity": {
                        "visible_records": visible_capacity,
                        "tile_intersections": intersection_capacity,
                    },
                    "native_limit": _INT32_MAX,
                },
            ) from error
        requested_bytes = int(plan["workspace_bytes"])
        current_bytes = self.workspace_bytes
        release_before_allocate = (
            reason in {"overflow-retry", "explicit-benchmark-reservation"}
            and bool(self.workspace)
        )
        telemetry = {
            "schema_version": "renderer-capacity-allocation/v1",
            "reason": reason,
            "requested_capacity": {
                "visible_records": visible_capacity,
                "visible_storage_records": int(
                    plan["visible_storage_capacity"]
                ),
                "tile_intersections": intersection_capacity,
            },
            "requested_workspace_bytes": requested_bytes,
            "requested_workspace_unaliased_bytes": int(
                plan["workspace_unaliased_bytes"]
            ),
            "workspace_alias_savings_bytes": int(
                plan["workspace_alias_savings_bytes"]
            ),
            "workspace_storage_aliases": dict(
                plan["workspace_storage_aliases"]
            ),
            "projected_record_reuse": copy.deepcopy(
                plan["projected_record_reuse"]
            ),
            "sort_temp_bytes": int(plan["sort_temp_bytes"]),
            "sort_temp_strategy": str(plan["sort_temp_strategy"]),
            "sort_key_begin_bit": int(plan["sort_key_begin_bit"]),
            "sort_key_end_bit": int(plan["sort_key_end_bit"]),
            "sort_key_bits": int(plan["sort_key_bits"]),
            "sort_radix_digit_bits_assumption": int(
                plan["sort_radix_digit_bits_assumption"]
            ),
            "sort_radix_passes_expected": int(
                plan["sort_radix_passes_expected"]
            ),
            "current_workspace_bytes": current_bytes,
            "allocation_strategy": (
                "release-old-workspace-first"
                if release_before_allocate
                else "transactional-replacement"
            ),
            "transient_logical_bytes": (
                max(current_bytes, requested_bytes)
                if release_before_allocate
                else current_bytes + requested_bytes
            ),
            "transactional_transient_logical_bytes": (
                current_bytes + requested_bytes
            ),
            "max_workspace_bytes": self.max_workspace_bytes,
        }
        if (
            self.max_workspace_bytes is not None
            and requested_bytes > self.max_workspace_bytes
        ):
            telemetry["failure"] = "workspace-memory-limit"
            telemetry["status"] = "failure"
            self._capacity_last_allocation = copy.deepcopy(telemetry)
            raise RendererCapacityError(
                "Adaptive capacity growth requires "
                f"{requested_bytes} workspace bytes, exceeding the configured "
                f"limit of {self.max_workspace_bytes} bytes.",
                telemetry=telemetry,
            )

        device = plan["device"]
        old_plan: dict[str, Any] | None = None
        if release_before_allocate:
            old_plan = self._workspace_plan(
                visible_capacity=self._allocated_visible_capacity,
                intersection_capacity=self._allocated_intersection_capacity,
            )
            self.invalidate_projection_cache()
            self.workspace.clear()
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

        try:
            replacement = self._allocate_workspace_plan(plan)
        except torch.OutOfMemoryError as error:
            telemetry["failure"] = "allocator-out-of-memory"
            telemetry["status"] = "failure"
            if release_before_allocate:
                assert old_plan is not None
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                try:
                    restored = self._allocate_workspace_plan(old_plan)
                except torch.OutOfMemoryError:
                    self.workspace.clear()
                    self._allocated_visible_capacity = 0
                    self._allocated_intersection_capacity = 0
                    telemetry["restoration"] = {
                        "status": "failed",
                        "workspace_usable": False,
                    }
                else:
                    self.workspace.update(restored)
                    telemetry["restoration"] = {
                        "status": "restored-old-capacity",
                        "workspace_usable": True,
                        "workspace_bytes": self.workspace_bytes,
                    }
            self._capacity_last_allocation = copy.deepcopy(telemetry)
            raise RendererCapacityError(
                "CUDA allocator could not satisfy bounded adaptive workspace "
                f"growth to {requested_bytes} bytes.",
                telemetry=telemetry,
            ) from error

        physical_max_views, height, width, _device, _dtype = (
            self._workspace_config
        )
        # Preserve the public workspace dictionary identity held by
        # RendererService while replacing only backend-owned tensor storage.
        self.workspace.clear()
        self.workspace.update(replacement)
        self._allocated_visible_capacity = visible_capacity
        self._allocated_intersection_capacity = intersection_capacity
        self._materialize_projected_records_active = bool(
            plan["projected_record_reuse"]["active"]
        )
        self._projected_record_fallback_reason = plan[
            "projected_record_reuse"
        ]["fallback_reason"]
        self._physical_max_views = physical_max_views
        self._height = height
        self._width = width
        self.invalidate_projection_cache()
        telemetry["status"] = "success"
        telemetry["installed_workspace_bytes"] = self.workspace_bytes
        telemetry["installed_workspace_storage_aliases"] = (
            self.workspace_storage_aliases
        )
        self._capacity_last_allocation = copy.deepcopy(telemetry)

    def _allocate_workspace_plan(
        self,
        plan: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        device = plan["device"]
        replacement: dict[str, torch.Tensor] = {}
        try:
            for name, (shape, tensor_dtype) in plan["specs"].items():
                if name in plan["workspace_storage_aliases"]:
                    continue
                replacement[name] = torch.empty(
                    shape,
                    device=device,
                    dtype=tensor_dtype,
                )
            for alias, owner in plan["workspace_storage_aliases"].items():
                shape, tensor_dtype = plan["specs"][alias]
                logical_bytes = (
                    math.prod(shape) * _DTYPE_BYTES[tensor_dtype]
                )
                replacement[alias] = (
                    replacement[owner]
                    .flatten()[:logical_bytes]
                    .view(tensor_dtype)
                    .view(shape)
                )
            replacement["identity_env_xforms"].copy_(
                torch.eye(4, device=device, dtype=torch.float32)
                .repeat(self._workspace_config[0], 1, 1)
                .contiguous()
            )
        except torch.OutOfMemoryError:
            replacement.clear()
            raise
        return replacement

    def _propose_capacity_growth(
        self,
        counters: dict[str, int],
    ) -> tuple[int, int, dict[str, Any] | None]:
        visible_capacity = self._allocated_visible_capacity
        intersection_capacity = self._allocated_intersection_capacity
        visible_required = max(
            int(counters["visible_gaussians"]),
            visible_capacity + int(counters["visible_overflow"]),
        )
        proposed_visible = visible_capacity
        if counters["visible_overflow"]:
            proposed_visible = max(
                visible_capacity + 1,
                math.ceil(visible_capacity * self.capacity_growth_factor),
                math.ceil(visible_required * self.capacity_headroom),
            )

        intersection_required = max(
            int(counters["tile_intersections"]),
            intersection_capacity + int(counters["intersection_overflow"]),
        )
        # Paths that emit from stored projected records cannot count
        # intersections for candidates that did not fit, so extrapolate their
        # observed intersection demand. The compact direct-reprojection path
        # visits every camera/Gaussian task and therefore counts its complete
        # demand without projected-record storage.
        if (
            proposed_visible > visible_capacity
            and (
                not self.compact_projection_cache
                or self._materialize_projected_records_active
            )
        ):
            materialized_visible = max(
                1,
                min(int(counters["visible_gaussians"]), visible_capacity),
            )
            intersection_required = max(
                intersection_required,
                math.ceil(
                    int(counters["tile_intersections"])
                    * visible_required
                    / materialized_visible
                ),
            )
        proposed_intersections = intersection_capacity
        if (
            counters["intersection_overflow"]
            or proposed_visible > visible_capacity
        ):
            proposed_intersections = max(
                intersection_capacity + 1,
                math.ceil(
                    intersection_capacity * self.capacity_growth_factor
                ),
                math.ceil(
                    intersection_required * self.capacity_headroom
                ),
            )

        ceiling_adjustment: dict[str, Any] | None = None
        if self.max_workspace_bytes is not None:
            try:
                target_plan = self._workspace_plan(
                    visible_capacity=proposed_visible,
                    intersection_capacity=proposed_intersections,
                )
            except ValueError:
                # Preserve native-limit error handling in _install_workspace().
                pass
            else:
                target_bytes = int(target_plan["workspace_bytes"])
                if target_bytes > self.max_workspace_bytes:
                    minimum_visible = visible_capacity
                    if counters["visible_overflow"]:
                        minimum_visible = max(
                            visible_capacity + 1,
                            visible_required,
                        )
                    minimum_intersections = intersection_capacity
                    if (
                        counters["intersection_overflow"]
                        or minimum_visible > visible_capacity
                    ):
                        minimum_intersections = max(
                            intersection_capacity + 1,
                            intersection_required,
                        )
                    minimum_plan = self._workspace_plan(
                        visible_capacity=minimum_visible,
                        intersection_capacity=minimum_intersections,
                    )
                    minimum_bytes = int(minimum_plan["workspace_bytes"])
                    ceiling_adjustment = {
                        "policy": "relax-headroom-to-observed-demand",
                        "target_capacity": {
                            "visible_records": proposed_visible,
                            "tile_intersections": proposed_intersections,
                        },
                        "target_workspace_bytes": target_bytes,
                        "minimum_observed_demand_capacity": {
                            "visible_records": minimum_visible,
                            "tile_intersections": minimum_intersections,
                        },
                        "minimum_observed_demand_workspace_bytes": (
                            minimum_bytes
                        ),
                        "max_workspace_bytes": self.max_workspace_bytes,
                        "status": (
                            "headroom-relaxed"
                            if minimum_bytes <= self.max_workspace_bytes
                            else "minimum-observed-demand-exceeds-limit"
                        ),
                    }
                    proposed_visible = minimum_visible
                    proposed_intersections = minimum_intersections
        return (
            proposed_visible,
            proposed_intersections,
            ceiling_adjustment,
        )

    def _capacity_snapshot(self) -> dict[str, int]:
        return {
            "visible_records": self._allocated_visible_capacity,
            "visible_storage_records": self.visible_storage_capacity,
            "tile_intersections": self._allocated_intersection_capacity,
            "workspace_bytes": self.workspace_bytes,
        }

    def _capacity_attempt_record(
        self,
        *,
        retry_count: int,
        counters: dict[str, int],
        counter_source: str,
        synchronized: bool,
    ) -> dict[str, Any]:
        return {
            "attempt": retry_count + 1,
            "requested_capacity": self._capacity_snapshot(),
            "actual_demand": dict(counters),
            "counter_source": counter_source,
            "synchronized": synchronized,
            "overflow": bool(
                counters["visible_overflow"]
                or counters["intersection_overflow"]
            ),
        }

    def _capacity_headroom(
        self,
        counters: dict[str, int],
    ) -> dict[str, float | None]:
        visible_demand = int(counters["visible_gaussians"])
        intersection_demand = int(counters["tile_intersections"])
        return {
            "visible_records_ratio": (
                self._allocated_visible_capacity / visible_demand
                if visible_demand
                and (
                    not self.compact_projection_cache
                    or self._materialize_projected_records_active
                )
                else None
            ),
            "tile_intersections_ratio": (
                self._allocated_intersection_capacity / intersection_demand
                if intersection_demand
                else None
            ),
        }

    def _raise_render_capacity_error(
        self,
        request: RenderRequest,
        *,
        message: str,
        attempts: list[dict[str, Any]],
        reason: str,
        cause: BaseException | None = None,
    ) -> NoReturn:
        self.invalidate_projection_cache()
        for name, tensor in request.outputs.items():
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
            elif name == "semantic_id":
                tensor.fill_(-1)
        final_counters = (
            copy.deepcopy(attempts[-1]["actual_demand"])
            if attempts
            else None
        )
        self._capacity_last_render = {
            "status": "failure",
            "validation": "device-counters",
            "synchronized": any(
                bool(attempt["synchronized"]) for attempt in attempts
            ),
            "retry_count": max(0, len(attempts) - 1),
            "attempts": copy.deepcopy(attempts),
            "final_capacity": self._capacity_snapshot(),
            "final_counters": final_counters,
            "headroom": (
                self._capacity_headroom(final_counters)
                if final_counters is not None
                else None
            ),
            "failure_reason": reason,
        }
        if any("growth_request" in attempt for attempt in attempts):
            self._capacity_last_growth = copy.deepcopy(
                self._capacity_last_render
            )
        telemetry = {
            "schema_version": "renderer-capacity-error/v1",
            "reason": reason,
            "capacity": self.capacity_stats,
        }
        if attempts:
            last_attempt = attempts[-1]
            demand = last_attempt["actual_demand"]
            requested = last_attempt["requested_capacity"]
            message = (
                f"{message} Last attempt: "
                f"visible_gaussians={demand['visible_gaussians']} "
                f"tile_intersections={demand['tile_intersections']} "
                f"visible_capacity={requested['visible_records']} "
                f"intersection_capacity={requested['tile_intersections']} "
                f"visible_overflow={demand['visible_overflow']} "
                f"intersection_overflow={demand['intersection_overflow']}."
            )
        if cause is None:
            raise RendererCapacityError(message, telemetry=telemetry)
        raise RendererCapacityError(message, telemetry=telemetry) from cause

    @property
    def workspace_bytes(self) -> int:
        return sum(self.workspace_bytes_by_tensor.values())

    @property
    def workspace_bytes_by_tensor(self) -> dict[str, int]:
        """Attribute each unique persistent storage to its first named view."""
        seen: set[tuple[str, int | None, int]] = set()
        result: dict[str, int] = {}
        for name, tensor in self.workspace.items():
            storage = tensor.untyped_storage()
            storage_bytes = int(storage.nbytes())
            if storage_bytes == 0:
                result[name] = 0
                continue
            key = (
                tensor.device.type,
                tensor.device.index,
                int(storage.data_ptr()),
            )
            if key in seen:
                result[name] = 0
                continue
            seen.add(key)
            result[name] = storage_bytes
        return result

    @property
    def workspace_logical_bytes_by_tensor(self) -> dict[str, int]:
        """Return the logical byte span exposed by every named tensor view."""
        return {
            name: tensor.numel() * tensor.element_size()
            for name, tensor in self.workspace.items()
        }

    @property
    def workspace_storage_aliases(self) -> dict[str, str]:
        """Return named tensor views that share an earlier workspace storage."""
        owners: dict[tuple[str, int | None, int], str] = {}
        aliases: dict[str, str] = {}
        for name, tensor in self.workspace.items():
            storage = tensor.untyped_storage()
            if storage.nbytes() == 0:
                continue
            key = (
                tensor.device.type,
                tensor.device.index,
                int(storage.data_ptr()),
            )
            owner = owners.setdefault(key, name)
            if owner != name:
                aliases[name] = owner
        return aliases

    @property
    def current_visible_capacity(self) -> int:
        return self._allocated_visible_capacity

    @property
    def current_intersection_capacity(self) -> int:
        return self._allocated_intersection_capacity

    @property
    def capacity_stats(self) -> dict[str, Any]:
        """Return JSON-serializable adaptive-capacity provenance."""
        return {
            "schema_version": "renderer-capacity/v1",
            "policy": (
                "adaptive-bounded-grow-retry"
                if self.adaptive_capacity
                else "explicit-manual-check"
            ),
            "adaptive": self.adaptive_capacity,
            "initial_capacity": {
                "visible_records": self._initial_visible_capacity,
                "tile_intersections": self._initial_intersection_capacity,
            },
            "current_capacity": self._capacity_snapshot(),
            "execution_schedule": {
                "logical_max_views": self._max_views,
                "physical_max_views": self._physical_max_views,
                "maximum_logical_chunks": self._max_logical_chunks,
                "chunking_enabled": self.max_physical_views is not None,
                "counter_reduction": (
                    "device-sum-and-physical-maximum"
                    if self.max_physical_views is not None
                    else "single-native-submission"
                ),
            },
            "visible_capacity_role": (
                "materialized-projected-candidate-capacity"
                if (
                    not self.compact_projection_cache
                    or self._materialize_projected_records_active
                )
                else "diagnostic-and-capacity-planning-only"
            ),
            "projected_record_reuse": {
                "requested": self.materialize_projected_records,
                "active": self._materialize_projected_records_active,
                "fallback_reason": self._projected_record_fallback_reason,
                "bytes_per_record": _PROJECTED_RECORD_BYTES,
            },
            "parameters": {
                "rasterize_mode": self.rasterize_mode,
                "covariance_epsilon": self.covariance_epsilon,
                "growth_factor": self.capacity_growth_factor,
                "headroom_factor": self.capacity_headroom,
                "max_retries": self.max_capacity_retries,
                "max_workspace_bytes": self.max_workspace_bytes,
                "workspace_ceiling_policy": (
                    "relax-headroom-to-observed-demand"
                ),
            },
            "totals": {
                "device_validations": self._capacity_validations,
                "validated_input_reuses": self._capacity_validation_reuses,
                "growth_events": self._capacity_growth_events,
                "retries": self._capacity_total_retries,
            },
            "synchronization_policy": (
                "new or explicitly invalidated projection inputs read device "
                "counters; identical validated inputs remain asynchronous"
                if self.adaptive_capacity
                else "caller must explicitly synchronize and check counters"
            ),
            "intersection_ordering_synchronization": (
                "fixed-capacity sentinel sort is device-only and performs no "
                "device-to-host count copy or stream synchronization"
                if self.fixed_capacity_sort
                else "projection-cache misses copy the emitted intersection "
                "count to the host and synchronize the current CUDA stream "
                "before global radix sorting; projection-cache hits reuse "
                "sorted intersections without that synchronization"
                if not self.deterministic
                else "deterministic projection-cache misses use a device-only "
                "count, scan, scatter, and segmented radix-sort path"
            ),
            "sort_temp_strategy": (
                "pointer-api-global-segmented-scan"
                if self.deterministic
                else "double-buffer-global-radix"
            ),
            "last_render": copy.deepcopy(self._capacity_last_render),
            "last_growth": copy.deepcopy(self._capacity_last_growth),
            "last_allocation": copy.deepcopy(
                self._capacity_last_allocation
            ),
        }

    @property
    def scene_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._packed_scene.values()
        )

    @property
    def projection_cache_stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enable_projection_cache,
            "hits": self._projection_cache_hits,
            "misses": self._projection_cache_misses,
        }

    @property
    def pipeline_name(self) -> str:
        nondeterministic_ordering = (
            "fixed-capacity-sentinel-double-buffer-global-radix"
            if self.fixed_capacity_sort
            else "emitted-count-sync-double-buffer-global-radix"
        )
        if self.compact_projection_cache:
            if self._materialize_projected_records_active:
                return (
                    "compact-project-once-candidate-tau-cutoff-"
                    "cached-record-intersections-emitted-count-sync-"
                    "double-buffer-global-radix-linear-ranges-"
                    "fused-stored-projection-rgba-depth-semantic"
                )
            return (
                "compact-project-tau-cutoff-direct-intersections-"
                f"{nondeterministic_ordering}-linear-ranges-"
                "fused-reproject-rgba-depth-semantic"
            )
        binning = (
            "macrobin2-"
            if self.bin_tile_size != self.tile_size
            else ""
        )
        ordering = (
            "device-tile-segmented-radix"
            if self.deterministic
            else f"{nondeterministic_ordering}-linear-ranges"
        )
        return (
            f"project-fused-project-{binning}bin-{ordering}-"
            "fused-rgba-depth-semantic"
        )

    @property
    def bin_tile_size(self) -> int:
        """Return the conservative scheduling tile used for intersection ordering.

        Exact-ray evaluation at the 16x16 raster tile size shares one sorted
        32x32 macro-tile candidate range across four raster blocks.  The exact
        ray support test rejects candidates outside each pixel footprint, so
        this changes scheduling and workspace size without changing the
        compositing equation or depth order.
        """
        if self.ray_gaussian_evaluation and self.tile_size == 16:
            return self.tile_size * 2
        return self.tile_size

    @property
    def projection_cache_scope(self) -> str:
        if self.compact_projection_cache:
            if self._materialize_projected_records_active:
                return (
                    "first-pass screen-space candidates, including camera "
                    "IDs and radii, reused for cutoff-filtered intersection "
                    "emission and compositing, with per-pixel depth cutoffs, "
                    "sorted projected-record intersections, and dense pixel "
                    "ranges; "
                    "RGB, depth, alpha, and semantic outputs are rewritten on "
                    "every render"
                )
            return (
                "per-pixel depth cutoffs, sorted Gaussian-ID intersections, "
                "and dense pixel ranges; Gaussian projection is recomputed "
                "cooperatively during compositing and RGB, depth, alpha, and "
                "semantic outputs are rewritten on every render"
            )
        return (
            "projected visible records, depth cutoffs, pixel bins, sorted "
            "intersections, and tile ranges; fused rasterization and RGB, "
            "depth, alpha, and semantic outputs are rewritten on every render"
        )

    @property
    def visible_storage_capacity(self) -> int:
        visible_depths = self.workspace.get("visible_depths")
        return int(visible_depths.numel()) if visible_depths is not None else 0

    @property
    def materialize_projected_records_active(self) -> bool:
        return self._materialize_projected_records_active

    @property
    def rasterize_mode(self) -> Literal["classic", "antialiased"]:
        return self._rasterize_mode

    @rasterize_mode.setter
    def rasterize_mode(
        self,
        value: Literal["classic", "antialiased"],
    ) -> None:
        if value not in ("classic", "antialiased"):
            raise ValueError(
                "rasterize_mode must be 'classic' or 'antialiased'."
            )
        if value == "antialiased" and getattr(
            self,
            "ray_gaussian_evaluation",
            False,
        ):
            raise ValueError(
                "rasterize_mode='antialiased' is incompatible with "
                "ray_gaussian_evaluation=True because the exact-ray path "
                "does not use the screen-space covariance dilation being "
                "compensated."
            )
        if value == getattr(self, "_rasterize_mode", None):
            return
        self._rasterize_mode = value
        if hasattr(self, "_projection_cache_token"):
            self.invalidate_projection_cache()

    def invalidate_projection_cache(self) -> None:
        """Invalidate cached geometry and its safe-capacity validation.

        Call this after writes that bypass PyTorch version counters, including
        external Fabric/Warp writes and writes to inference-mode tensors.
        """
        self._projection_cache_token = None
        self._projection_cache_tensor_refs = ()
        self._capacity_validation_token = None
        self._capacity_validation_tensor_refs = ()
        self._capacity_validation_counters = None

    def shutdown(self) -> None:
        self.invalidate_projection_cache()
        self.workspace = {}
        self._scenes.clear()
        self._packed_scene = {}
        self._env_scene_ids = None
        self._env_transforms = None
        self._workspace_config = None
        self._allocated_visible_capacity = 0
        self._allocated_intersection_capacity = 0
        self._materialize_projected_records_active = False
        self._projected_record_fallback_reason = None
        self._initial_visible_capacity = 0
        self._initial_intersection_capacity = 0
        self.stage = None
        self.device = None

    def _rebuild_packed_scene(self) -> None:
        assert self.device is not None
        scenes = list(self._scenes.values())
        names = ("means", "scales", "rotations", "opacities", "features", "semantic_ids")
        packed: dict[str, torch.Tensor] = {}
        for name in names:
            tensors = [getattr(scene, name) for scene in scenes]
            packed[name] = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        packed["covariances"] = self._precompute_covariances(
            packed["scales"],
            packed["rotations"],
        )
        offsets = [0]
        for scene in scenes:
            offsets.append(offsets[-1] + scene.count)
        packed["registered_scene_ids"] = torch.tensor(
            list(self._scenes),
            device=self.device,
            dtype=torch.int64,
        )
        packed["scene_offsets"] = torch.tensor(
            offsets,
            device=self.device,
            dtype=torch.int64,
        )
        self._packed_scene = packed
        self._max_scene_gaussians = max(scene.count for scene in scenes)
        self._scene_revision += 1
        self.invalidate_projection_cache()

    def _projection_input_token(
        self,
        *,
        batch: int,
        request: RenderRequest,
        env_xforms: torch.Tensor,
        active_ids: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            self._scene_revision,
            batch,
            self._height,
            self._width,
            self._max_scene_gaussians,
            self.near_plane,
            self.far_plane,
            self.gaussian_support_sigma,
            self.covariance_epsilon,
            self.rasterize_mode,
            self.ray_gaussian_evaluation,
            self.tile_size,
            self.depth_bucket_count,
            self.depth_bucket_group_size,
            self.compact_projection_cache,
            self._materialize_projected_records_active,
            self.deterministic,
            self._tensor_cache_token(self._packed_scene["means"]),
            self._tensor_cache_token(self._packed_scene["covariances"]),
            self._tensor_cache_token(self._packed_scene["opacities"]),
            self._tensor_cache_token(
                self._packed_scene["registered_scene_ids"]
            ),
            self._tensor_cache_token(self._packed_scene["scene_offsets"]),
            self._tensor_cache_token(request.camera_transforms),
            self._tensor_cache_token(request.intrinsics),
            self._tensor_cache_token(env_xforms),
            self._tensor_cache_token(request.scene_ids),
            self._tensor_cache_token(active_ids),
        )

    @staticmethod
    def _tensor_cache_token(tensor: torch.Tensor) -> tuple[Any, ...]:
        try:
            version: int | object = int(tensor._version)
        except RuntimeError:
            # Inference tensors do not track versions. A stable data pointer,
            # shape, and stride therefore cannot prove that their contents are
            # unchanged after an in-place write. A fresh identity makes every
            # projection/capacity token containing such a tensor non-reusable.
            version = object()
        return (
            tensor.data_ptr(),
            version,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    def _precompute_covariances(
        self,
        scales: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        assert self.device is not None
        covariances = torch.empty(
            (scales.shape[0], 6),
            device=self.device,
            dtype=torch.float32,
        )
        assert self._native is not None
        native_precompute = getattr(
            self._native,
            "precompute_covariances",
            None,
        )
        if self.device.type == "cuda" and native_precompute is not None:
            native_precompute(scales, rotations, covariances)
            return covariances

        with torch.no_grad():
            quaternion = rotations * torch.rsqrt(
                rotations.square().sum(dim=1, keepdim=True).clamp_min(1.0e-20)
            )
            w, x, y, z = quaternion.unbind(dim=1)
            rotation = torch.stack(
                (
                    1.0 - 2.0 * (y.square() + z.square()),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x.square() + z.square()),
                    2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x.square() + y.square()),
                ),
                dim=1,
            ).reshape(-1, 3, 3)
            scale_squared = scales.square()
            covariance = (
                rotation
                * scale_squared.unsqueeze(1)
            ) @ rotation.transpose(1, 2)
            covariances.copy_(
                torch.stack(
                    (
                        covariance[:, 0, 0],
                        covariance[:, 0, 1],
                        covariance[:, 0, 2],
                        covariance[:, 1, 1],
                        covariance[:, 1, 2],
                        covariance[:, 2, 2],
                    ),
                    dim=1,
                )
            )
        return covariances

    def _require_initialized(self) -> None:
        if self.device is None:
            raise RuntimeError("CustomCudaBackend.initialize() must be called first.")
