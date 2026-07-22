"""Isaac camera source boundary for USD intrinsics and high-rate transforms."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any, Protocol

import torch


def _load_fabric_modules() -> tuple[Any, Any, Any]:
    """Load Fabric/Warp modules and initialize Warp before tensor interop."""

    try:
        usdrt = importlib.import_module("usdrt")
        warp = importlib.import_module("warp")
        kernels = importlib.import_module(
            "isaacsim_gaussian_renderer.fabric_camera_kernels"
        )
    except ImportError as error:
        raise RuntimeError(
            "USDRT Fabric camera ingestion must run inside Isaac Sim with "
            "usdrt.scenegraph and omni.warp.core enabled."
        ) from error
    # ``warp.from_torch`` reads the global CUDA device registry directly;
    # unlike launches, it does not lazily initialize that registry.
    warp.init()
    return usdrt, warp, kernels


@dataclass(frozen=True)
class UsdCameraIntrinsics:
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    width: int
    height: int

    def matrix(self, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        fx = self.focal_length / self.horizontal_aperture * self.width
        fy = self.focal_length / self.vertical_aperture * self.height
        cx = self.width * 0.5
        cy = self.height * 0.5
        return torch.tensor(
            ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
            device=device,
            dtype=dtype,
        )


class HighFrequencyTransformSource(Protocol):
    """Fabric/USDRT transform ingestion seam.

    Implementations may use Fabric or USDRT to avoid slow USD traversal, but
    this interface deliberately makes no zero-copy promise.
    """

    def read_camera_transforms(self, camera_paths: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
        """Return contiguous float32 world-to-camera transforms with shape [B, 4, 4]."""


class UsdrtFabricCameraTransformSource:
    """Read selected USD camera world matrices from Fabric entirely on CUDA.

    Initialization tags the selected camera prims with a private integer index
    and builds one reusable USDRT GPU selection. Each steady-state read launches
    one Warp kernel on PyTorch's current CUDA stream. The kernel converts
    Fabric/Gf row-vector matrices and USD camera axes directly into the
    renderer's OpenCV world-to-camera tensor.
    """

    def __init__(
        self,
        *,
        stage_id: int,
        camera_paths: tuple[str, ...],
        device: torch.device | str = "cuda:0",
        update_world_xforms: bool = True,
    ) -> None:
        if not camera_paths:
            raise ValueError("camera_paths must contain at least one camera.")
        if len(set(camera_paths)) != len(camera_paths):
            raise ValueError("camera_paths must be unique.")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("USDRT Fabric camera ingestion requires a CUDA device.")
        if self.device.index not in (None, 0):
            raise ValueError("USDRT GPU prim selection currently supports only cuda:0.")
        self.device = torch.device("cuda:0")
        self.camera_paths = tuple(camera_paths)
        self.update_world_xforms = update_world_xforms
        self.read_calls = 0
        self.topology_rebuilds = 0

        self._usdrt, self._warp, kernels = _load_fabric_modules()
        self._kernel = kernels.fabric_world_to_opencv_viewmats

        Usd = self._usdrt.Usd
        Rt = self._usdrt.Rt
        Sdf = self._usdrt.Sdf
        self._stage = Usd.Stage.Attach(int(stage_id))
        if not self._stage:
            raise RuntimeError(f"Could not attach USDRT to stage ID {stage_id}.")

        path_digest = hashlib.sha256(
            "\0".join(self.camera_paths).encode("utf-8")
        ).hexdigest()[:16]
        self._index_attribute = (
            f"omni:gaussianRenderer:cameraIndex_{path_digest}"
        )
        for index, path in enumerate(self.camera_paths):
            prim = self._stage.GetPrimAtPath(Sdf.Path(path))
            if not prim:
                raise ValueError(f"USDRT camera prim does not exist: {path}")
            xformable = Rt.Xformable(prim)
            world_matrix = xformable.GetFabricHierarchyWorldMatrixAttr()
            if not world_matrix:
                raise RuntimeError(
                    f"Fabric world matrix is unavailable for camera: {path}"
                )
            world_matrix.Get()
            index_attr = (
                prim.GetAttribute(self._index_attribute)
                if prim.HasAttribute(self._index_attribute)
                else prim.CreateAttribute(
                    self._index_attribute,
                    Sdf.ValueTypeNames.Int,
                    True,
                )
            )
            index_attr.Set(index)

        self._selection = self._stage.SelectPrims(
            require_attrs=[
                (
                    Sdf.ValueTypeNames.Matrix4d,
                    Rt.Tokens.fabricHierarchyWorldMatrix,
                    Usd.Access.Read,
                ),
                (
                    Sdf.ValueTypeNames.Int,
                    self._index_attribute,
                    Usd.Access.Read,
                ),
            ],
            require_prim_type="Camera",
            device="cuda:0",
            want_paths=False,
        )
        self._require_selection_size()
        self._hierarchy = self._usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            self._stage.GetFabricId(),
            self._stage.GetStageIdAsStageId(),
        )
        self._viewmats = torch.empty(
            (len(self.camera_paths), 4, 4),
            device=self.device,
            dtype=torch.float32,
        )
        self._warp_viewmats = self._warp.from_torch(
            self._viewmats,
            dtype=self._warp.mat44f,
        )
        self._bind_fabric_arrays()

    def read_camera_transforms(
        self,
        camera_paths: tuple[str, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())
        if tuple(camera_paths) != self.camera_paths:
            raise ValueError(
                "Camera path order differs from the reusable Fabric selection."
            )
        if requested_device != self.device:
            raise ValueError(
                f"Fabric source is bound to {self.device}; got {requested_device}."
            )

        if self.update_world_xforms:
            self._hierarchy.update_world_xforms()
        if self._selection.PrepareForReuse():
            self.topology_rebuilds += 1
            self._require_selection_size()
            self._bind_fabric_arrays()

        stream = self._warp.stream_from_torch(
            torch.cuda.current_stream(self.device)
        )
        self._warp.launch(
            self._kernel,
            dim=len(self.camera_paths),
            inputs=[
                self._world_matrices,
                self._camera_indices,
                self._warp_viewmats,
            ],
            stream=stream,
        )
        self.read_calls += 1
        return self._viewmats

    @property
    def output_tensor(self) -> torch.Tensor:
        return self._viewmats

    def close(self) -> None:
        self._world_matrices = None
        self._camera_indices = None
        self._selection = None
        self._hierarchy = None
        self._stage = None

    def _bind_fabric_arrays(self) -> None:
        Rt = self._usdrt.Rt
        self._world_matrices = self._warp.fabricarray(
            self._selection,
            Rt.Tokens.fabricHierarchyWorldMatrix,
        )
        self._camera_indices = self._warp.fabricarray(
            self._selection,
            self._index_attribute,
        )

    def _require_selection_size(self) -> None:
        count = int(self._selection.GetCount())
        if count != len(self.camera_paths):
            raise RuntimeError(
                "Fabric camera selection cardinality changed: "
                f"expected {len(self.camera_paths)}, got {count}."
            )


class IsaacCameraSource:
    """Cache USD camera intrinsics and delegate high-frequency transforms."""

    def __init__(self, stage: Any, transform_source: HighFrequencyTransformSource | None = None) -> None:
        self.stage = stage
        self.transform_source = transform_source
        self._intrinsics_cache: dict[tuple[str, int, int], UsdCameraIntrinsics] = {}

    def get_intrinsics(self, camera_path: str, *, width: int, height: int, device: torch.device) -> torch.Tensor:
        key = (camera_path, width, height)
        if key not in self._intrinsics_cache:
            self._intrinsics_cache[key] = self._read_usd_intrinsics(camera_path, width=width, height=height)
        return self._intrinsics_cache[key].matrix(device=device).contiguous()

    def get_batched_intrinsics(
        self,
        camera_paths: tuple[str, ...],
        *,
        width: int,
        height: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.stack(
            [self.get_intrinsics(path, width=width, height=height, device=device) for path in camera_paths],
            dim=0,
        ).contiguous()

    def read_transforms(self, camera_paths: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
        if self.transform_source is None:
            raise RuntimeError("No Fabric/USDRT high-frequency transform source is configured.")
        transforms = self.transform_source.read_camera_transforms(camera_paths, device=device)
        if transforms.shape != (len(camera_paths), 4, 4):
            raise ValueError(f"Transform source returned {tuple(transforms.shape)} for {len(camera_paths)} cameras.")
        return transforms.to(device=device, dtype=torch.float32).contiguous()

    def _read_usd_intrinsics(self, camera_path: str, *, width: int, height: int) -> UsdCameraIntrinsics:
        prim = self.stage.GetPrimAtPath(camera_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"USD camera prim does not exist: {camera_path}")

        def attr_float(name: str, default: float) -> float:
            attr = prim.GetAttribute(name)
            value = attr.Get() if attr and attr.IsValid() else None
            return float(default if value is None else value)

        focal_length = attr_float("focalLength", 50.0)
        horizontal_aperture = attr_float("horizontalAperture", 20.955)
        vertical_aperture = attr_float("verticalAperture", horizontal_aperture * height / width)
        return UsdCameraIntrinsics(
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            width=width,
            height=height,
        )
