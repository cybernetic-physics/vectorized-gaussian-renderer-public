"""Isaac Sim-native Gaussian renderer service boundary."""

from .backend import GaussianRendererBackend, RenderRequest, RenderStats
from .camera_math import (
    isaac_world_cameras_to_viewmats,
    opencv_viewmats_to_usd_camera_world_matrices,
    quaternion_xyzw_to_matrix,
    usd_camera_world_matrices_to_viewmats,
)
from .camera_paths import (
    cinematic_orbit_camera_bundle,
    cinematic_walkthrough_camera_bundle,
    look_at_opencv_viewmats,
    scripted_walkthrough_camera_bundle,
)
from .camera_source import (
    IsaacCameraSource,
    UsdCameraIntrinsics,
    UsdrtFabricCameraTransformSource,
)
from .cuda_backend import (
    DEFAULT_GAUSSIAN_SUPPORT_SIGMA,
    CustomCudaBackend,
    RendererCapacityError,
)
from .fake_backend import DeterministicFakeBackend
from .flashgs_backend import FlashGSBackend
from .lidar_backend import GaussianLidarBackend, LidarRenderRequest
from .lidar_scene import GaussianLidarScene, LidarSceneRegistry
from .lidar_service import GaussianLidarService
from .ply_loader import (
    canonicalize_3dgs_scene,
    load_ply_to_canonical_gaussians,
    load_ply_to_gaussians,
)
from .quaternion import (
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_wxyz,
)
from .renderer_service import RendererService
from .scene import GaussianScene, SceneRegistry
from .upstream_faithful_flashgs_backend import UpstreamFaithfulFlashGSBackend

__all__ = [
    "DeterministicFakeBackend",
    "DEFAULT_GAUSSIAN_SUPPORT_SIGMA",
    "CustomCudaBackend",
    "GaussianRendererBackend",
    "FlashGSBackend",
    "UpstreamFaithfulFlashGSBackend",
    "GaussianLidarBackend",
    "GaussianLidarScene",
    "GaussianLidarService",
    "GaussianScene",
    "IsaacCameraSource",
    "RenderRequest",
    "RenderStats",
    "LidarRenderRequest",
    "LidarSceneRegistry",
    "RendererService",
    "RendererCapacityError",
    "SceneRegistry",
    "UsdCameraIntrinsics",
    "UsdrtFabricCameraTransformSource",
    "canonicalize_3dgs_scene",
    "cinematic_orbit_camera_bundle",
    "cinematic_walkthrough_camera_bundle",
    "isaac_world_cameras_to_viewmats",
    "look_at_opencv_viewmats",
    "scripted_walkthrough_camera_bundle",
    "opencv_viewmats_to_usd_camera_world_matrices",
    "load_ply_to_canonical_gaussians",
    "load_ply_to_gaussians",
    "quaternion_wxyz_to_xyzw",
    "quaternion_xyzw_to_matrix",
    "quaternion_xyzw_to_wxyz",
    "usd_camera_world_matrices_to_viewmats",
]
