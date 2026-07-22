"""Kit lifecycle hooks for the Gaussian renderer service extension."""

from __future__ import annotations

from typing import Any, Literal

import omni.ext
import torch

from isaacsim_gaussian_renderer import (
    DEFAULT_GAUSSIAN_SUPPORT_SIGMA,
    CustomCudaBackend,
    DeterministicFakeBackend,
    IsaacCameraSource,
    RendererService,
    UsdrtFabricCameraTransformSource,
)

STARTUP_EXT_ID: str | None = None
STARTUP_INSTANCE: IsaacSimGaussianRendererExtension | None = None


class IsaacSimGaussianRendererExtension(omni.ext.IExt):
    """Kit lifecycle owner for the Isaac Sim-native renderer service."""

    def on_startup(self, ext_id: str) -> None:
        global STARTUP_EXT_ID, STARTUP_INSTANCE
        STARTUP_EXT_ID = ext_id
        STARTUP_INSTANCE = self
        self.ext_id = ext_id
        self.service: RendererService | None = None
        self.lidar_service: Any | None = None
        self.camera_source: IsaacCameraSource | None = None
        self.fabric_transform_source: UsdrtFabricCameraTransformSource | None = None

    def create_service(
        self,
        *,
        stage: Any,
        device: str,
        height: int,
        width: int,
        max_views: int,
        backend: Any | None = None,
        max_visible_records: int | None = None,
        max_intersections: int | None = None,
        visible_capacity_per_view: int = 100_000,
        intersections_per_visible: float = 4.0,
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
        capacity_growth_factor: float | None = None,
        capacity_headroom: float = 1.25,
        max_capacity_retries: int = 4,
        max_workspace_bytes: int | None = None,
        render_every_n: int = 1,
        allow_cpu_for_tests: bool = False,
    ) -> RendererService:
        resolved_device = torch.device(device)
        if backend is None:
            if resolved_device.type == "cuda":
                capacity_kwargs = {}
                if capacity_growth_factor is not None:
                    capacity_kwargs["capacity_growth_factor"] = (
                        capacity_growth_factor
                    )
                backend = CustomCudaBackend(
                    max_visible_records=max_visible_records,
                    max_intersections=max_intersections,
                    visible_capacity_per_view=visible_capacity_per_view,
                    intersections_per_visible=intersections_per_visible,
                    gaussian_support_sigma=gaussian_support_sigma,
                    covariance_epsilon=covariance_epsilon,
                    rasterize_mode=rasterize_mode,
                    semantic_min_alpha=semantic_min_alpha,
                    ray_gaussian_evaluation=ray_gaussian_evaluation,
                    tile_size=tile_size,
                    depth_bucket_count=depth_bucket_count,
                    depth_bucket_group_size=depth_bucket_group_size,
                    compact_projection_cache=compact_projection_cache,
                    materialize_projected_records=(
                        materialize_projected_records
                    ),
                    enable_projection_cache=enable_projection_cache,
                    output_srgb=output_srgb,
                    deterministic=deterministic,
                    fixed_capacity_sort=fixed_capacity_sort,
                    max_physical_views=max_physical_views,
                    adaptive_capacity=adaptive_capacity,
                    capacity_headroom=capacity_headroom,
                    max_capacity_retries=max_capacity_retries,
                    max_workspace_bytes=max_workspace_bytes,
                    **capacity_kwargs,
                )
            elif allow_cpu_for_tests:
                backend = DeterministicFakeBackend()
            else:
                raise ValueError(
                    "The default Isaac Sim Gaussian renderer requires CUDA. "
                    "Pass allow_cpu_for_tests=True only for contract tests."
                )
        self.service = RendererService(
            backend,
            height=height,
            width=width,
            max_views=max_views,
            render_every_n=render_every_n,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
        self.service.initialize(stage, resolved_device)
        return self.service

    def create_lidar_service(
        self,
        *,
        stage: Any,
        device: str,
        max_sensors: int,
        max_rays: int,
        returns: int = 1,
        backend: Any | None = None,
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
        allow_cpu_for_tests: bool = False,
    ) -> Any:
        """Explicitly create the opt-in LiDAR path.

        Imports are intentionally local: the existing ``create_service`` camera
        path neither imports the LiDAR CUDA backend nor invokes its lazy native
        loader, acceleration builder, workspace allocator, or kernels.
        """

        from isaacsim_gaussian_renderer.cuda_lidar_backend import CudaLidarBackend
        from isaacsim_gaussian_renderer.lidar_service import GaussianLidarService

        resolved_device = torch.device(device)
        if backend is None:
            if resolved_device.type != "cuda":
                raise ValueError("The default Gaussian LiDAR backend requires CUDA.")
            backend = CudaLidarBackend(
                max_scenes=max_scenes,
                packet_size=packet_size,
                near_plane_m=near_plane_m,
                far_plane_m=far_plane_m,
                support_sigma=support_sigma,
                detection_threshold=detection_threshold,
                planarity_ratio_max=planarity_ratio_max,
                min_incidence_cos=min_incidence_cos,
                cluster_abs_m=cluster_abs_m,
                cluster_relative=cluster_relative,
                fallback_reflectivity=fallback_reflectivity,
                direction_norm_tolerance=direction_norm_tolerance,
                semantic_slots=semantic_slots,
            )
        self.lidar_service = GaussianLidarService(
            backend,
            max_sensors=max_sensors,
            max_rays=max_rays,
            returns=returns,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
        self.lidar_service.initialize(stage, resolved_device)
        return self.lidar_service

    def create_camera_source(
        self,
        *,
        stage: Any,
        stage_id: int,
        camera_paths: tuple[str, ...],
        device: str = "cuda:0",
        update_world_xforms: bool = True,
    ) -> IsaacCameraSource:
        self.fabric_transform_source = UsdrtFabricCameraTransformSource(
            stage_id=stage_id,
            camera_paths=camera_paths,
            device=device,
            update_world_xforms=update_world_xforms,
        )
        self.camera_source = IsaacCameraSource(
            stage,
            transform_source=self.fabric_transform_source,
        )
        return self.camera_source

    def on_shutdown(self) -> None:
        global STARTUP_EXT_ID, STARTUP_INSTANCE
        if self.lidar_service is not None:
            self.lidar_service.shutdown()
        if self.service is not None:
            self.service.shutdown()
        if self.fabric_transform_source is not None:
            self.fabric_transform_source.close()
        self.service = None
        self.lidar_service = None
        self.camera_source = None
        self.fabric_transform_source = None
        STARTUP_EXT_ID = None
        STARTUP_INSTANCE = None
