"""Shared camera-bundle format for Isaac Sim renderer fidelity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CAMERA_BUNDLE_SCHEMA_VERSION = "camera-bundle-v1"


@dataclass(frozen=True)
class CameraRecord:
    """One deterministic world-to-camera OpenCV camera."""

    view_id: str
    viewmat: list[list[float]]
    intrinsics: list[list[float]]
    scene_id: int = 0

    def validate(self) -> None:
        viewmat = np.asarray(self.viewmat, dtype=np.float64)
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        if viewmat.shape != (4, 4):
            raise ValueError(f"Camera {self.view_id!r} viewmat must have shape 4x4, got {viewmat.shape}.")
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Camera {self.view_id!r} intrinsics must have shape 3x3, got {intrinsics.shape}.")
        if not np.isfinite(viewmat).all():
            raise ValueError(f"Camera {self.view_id!r} viewmat contains non-finite values.")
        if not np.isfinite(intrinsics).all():
            raise ValueError(f"Camera {self.view_id!r} intrinsics contains non-finite values.")
        if int(self.scene_id) != self.scene_id or self.scene_id < 0:
            raise ValueError(f"Camera {self.view_id!r} scene_id must be a non-negative integer.")

    def to_json(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "scene_id": int(self.scene_id),
            "viewmat": _round_nested(self.viewmat),
            "intrinsics": _round_nested(self.intrinsics),
        }


@dataclass(frozen=True)
class CameraBundle:
    """Immutable camera/background/color-space metadata shared by fair runs."""

    width: int
    height: int
    cameras: tuple[CameraRecord, ...]
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color_space: str = "linear_rgb"
    coordinate_system: str = "opencv_world_to_camera"
    scene_checksum: str | None = None
    schema_version: str = CAMERA_BUNDLE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != CAMERA_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported camera bundle schema: {self.schema_version!r}.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera bundle width and height must be positive.")
        if len(self.cameras) == 0:
            raise ValueError("Camera bundle must contain at least one camera.")
        if self.coordinate_system != "opencv_world_to_camera":
            raise ValueError("Camera bundle coordinate_system must be 'opencv_world_to_camera'.")
        if len(self.background) != 3:
            raise ValueError("Camera bundle background must have exactly three RGB values.")
        if not np.isfinite(np.asarray(self.background, dtype=np.float64)).all():
            raise ValueError("Camera bundle background contains non-finite values.")
        seen = set()
        for camera in self.cameras:
            camera.validate()
            if camera.view_id in seen:
                raise ValueError(f"Duplicate camera view_id: {camera.view_id!r}.")
            seen.add(camera.view_id)

    @property
    def bundle_id(self) -> str:
        """Stable SHA-256 over canonical JSON content."""
        payload = json.dumps(self.to_json(include_bundle_id=False), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_json(self, *, include_bundle_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "coordinate_system": self.coordinate_system,
            "width": int(self.width),
            "height": int(self.height),
            "background": [float(v) for v in self.background],
            "color_space": self.color_space,
            "scene_checksum": self.scene_checksum,
            "cameras": [camera.to_json() for camera in self.cameras],
        }
        if include_bundle_id:
            payload["bundle_id"] = self.bundle_id
        return payload


def bundle_from_tensors(
    *,
    viewmats: Any,
    intrinsics: Any,
    width: int,
    height: int,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    color_space: str = "linear_rgb",
    scene_ids: Any | None = None,
    scene_checksum: str | None = None,
    view_ids: list[str] | None = None,
) -> CameraBundle:
    """Create a bundle from array-like tensors without depending on torch."""
    viewmats_np = np.asarray(_to_numpy(viewmats), dtype=np.float64)
    intrinsics_np = np.asarray(_to_numpy(intrinsics), dtype=np.float64)
    if viewmats_np.ndim != 3 or viewmats_np.shape[1:] != (4, 4):
        raise ValueError(f"viewmats must have shape [V,4,4], got {viewmats_np.shape}.")
    if intrinsics_np.ndim == 2 and intrinsics_np.shape == (3, 3):
        intrinsics_np = np.broadcast_to(intrinsics_np, (viewmats_np.shape[0], 3, 3))
    if intrinsics_np.ndim != 3 or intrinsics_np.shape[1:] != (3, 3):
        raise ValueError(f"intrinsics must have shape [V,3,3] or [3,3], got {intrinsics_np.shape}.")
    if intrinsics_np.shape[0] != viewmats_np.shape[0]:
        raise ValueError("viewmats and intrinsics must contain the same number of views.")

    num_views = viewmats_np.shape[0]
    if scene_ids is None:
        scene_ids_np = np.zeros((num_views,), dtype=np.int64)
    else:
        scene_ids_np = np.asarray(_to_numpy(scene_ids), dtype=np.int64)
        if scene_ids_np.shape != (num_views,):
            raise ValueError(f"scene_ids must have shape [{num_views}], got {scene_ids_np.shape}.")

    if view_ids is None:
        view_ids = [f"view_{index:06d}" for index in range(num_views)]
    if len(view_ids) != num_views:
        raise ValueError("view_ids length must match the number of views.")

    cameras = tuple(
        CameraRecord(
            view_id=str(view_ids[index]),
            scene_id=int(scene_ids_np[index]),
            viewmat=viewmats_np[index].tolist(),
            intrinsics=intrinsics_np[index].tolist(),
        )
        for index in range(num_views)
    )
    bundle = CameraBundle(
        width=width,
        height=height,
        background=tuple(float(v) for v in background),
        color_space=color_space,
        scene_checksum=scene_checksum,
        cameras=cameras,
    )
    bundle.validate()
    return bundle


def load_camera_bundle(path: str | Path) -> CameraBundle:
    payload = json.loads(Path(path).read_text())
    cameras = tuple(
        CameraRecord(
            view_id=str(camera["view_id"]),
            scene_id=int(camera.get("scene_id", 0)),
            viewmat=camera["viewmat"],
            intrinsics=camera["intrinsics"],
        )
        for camera in payload["cameras"]
    )
    bundle = CameraBundle(
        schema_version=payload["schema_version"],
        coordinate_system=payload.get("coordinate_system", "opencv_world_to_camera"),
        width=int(payload["width"]),
        height=int(payload["height"]),
        background=tuple(float(v) for v in payload["background"]),
        color_space=str(payload["color_space"]),
        scene_checksum=payload.get("scene_checksum"),
        cameras=cameras,
    )
    bundle.validate()
    expected_id = payload.get("bundle_id")
    if expected_id is not None and expected_id != bundle.bundle_id:
        raise ValueError(f"Camera bundle_id mismatch: expected {expected_id}, recomputed {bundle.bundle_id}.")
    return bundle


def write_camera_bundle(bundle: CameraBundle, path: str | Path) -> None:
    bundle.validate()
    Path(path).write_text(json.dumps(bundle.to_json(), indent=2, sort_keys=True) + "\n")


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _round_nested(value: Any) -> list[list[float]]:
    array = np.asarray(value, dtype=np.float64)
    return [[float(f"{item:.12g}") for item in row] for row in array]
