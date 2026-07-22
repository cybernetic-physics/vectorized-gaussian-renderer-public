"""Versioned, portable camera trajectory contract.

The JSON document contains canonical metadata and SHA-256 digests. Dense arrays
live in a sibling NPZ so a contract stays practical for long vectorized paths.
Neither the contract hash nor provenance may contain machine-local paths.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from ..fidelity.camera_bundle import CameraBundle, bundle_from_tensors

CAMERA_TRAJECTORY_SCHEMA_VERSION = "camera-trajectory-v1"
_ARRAY_NAMES = (
    "viewmats",
    "intrinsics",
    "scene_ids",
    "environment_transforms",
    "active_camera_ids",
)
_CACHE_EVENTS = {"hit", "miss", "skip", "disabled"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(canonical_json_bytes(list(value.shape)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _portable_path(value: str, *, name: str) -> str:
    if "\\" in value:
        raise ValueError(f"{name} must use portable POSIX separators.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a portable repository-relative path.")
    return str(path)


@dataclass(frozen=True)
class CameraTrajectory:
    """Immutable trajectory arrays with time-major ``[T,B]`` semantics."""

    viewmats: np.ndarray
    intrinsics: np.ndarray
    scene_ids: np.ndarray
    width: int
    height: int
    fps: float
    seed: int
    scene_sha256: str
    route_sha256: str
    motion_classification: str
    expected_cache_events: tuple[str, ...]
    near_plane: float = 0.01
    far_plane: float = 100.0
    near_far_policy: str = "fixed-metric"
    environment_phase_offsets: tuple[int, ...] = ()
    environment_transforms: np.ndarray | None = None
    active_camera_ids: np.ndarray | None = None
    scene_id_broadcast: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    route_validation: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CAMERA_TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "viewmats", np.asarray(self.viewmats, dtype=np.float32))
        object.__setattr__(self, "intrinsics", np.asarray(self.intrinsics, dtype=np.float32))
        object.__setattr__(self, "scene_ids", np.asarray(self.scene_ids, dtype=np.int64))
        if self.environment_transforms is not None:
            object.__setattr__(
                self,
                "environment_transforms",
                np.asarray(self.environment_transforms, dtype=np.float32),
            )
        if self.active_camera_ids is not None:
            object.__setattr__(
                self,
                "active_camera_ids",
                np.asarray(self.active_camera_ids, dtype=np.int64),
            )
        self.validate()

    @property
    def timesteps(self) -> int:
        return int(self.viewmats.shape[0])

    @property
    def batch(self) -> int:
        return int(self.viewmats.shape[1])

    def validate(self) -> None:
        if self.schema_version != CAMERA_TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported trajectory schema {self.schema_version!r}.")
        if self.viewmats.ndim != 4 or self.viewmats.shape[2:] != (4, 4):
            raise ValueError(f"viewmats must have shape [T,B,4,4], got {self.viewmats.shape}.")
        if self.intrinsics.shape != self.viewmats.shape[:2] + (3, 3):
            raise ValueError("intrinsics must have shape [T,B,3,3].")
        if self.scene_ids.shape == (self.batch,):
            if self.scene_id_broadcast != "time":
                raise ValueError("[B] scene_ids require scene_id_broadcast='time'.")
        elif self.scene_ids.shape != (self.timesteps, self.batch):
            raise ValueError("scene_ids must have shape [T,B], or [B] with explicit time broadcast.")
        if self.environment_transforms is not None and self.environment_transforms.shape != (
            self.timesteps,
            self.batch,
            4,
            4,
        ):
            raise ValueError("environment_transforms must have shape [T,B,4,4].")
        if self.active_camera_ids is not None:
            if self.active_camera_ids.ndim != 2 or self.active_camera_ids.shape[0] != self.timesteps:
                raise ValueError("active_camera_ids must have shape [T,A], with optional trailing -1 padding.")
            if np.any(self.active_camera_ids < -1) or np.any(self.active_camera_ids >= self.batch):
                raise ValueError("active_camera_ids contains an out-of-range camera.")
            for row in self.active_camera_ids:
                padding = np.flatnonzero(row == -1)
                if padding.size and np.any(row[padding[0] :] != -1):
                    raise ValueError("active_camera_ids -1 padding must be trailing within each timestep.")
        if not np.isfinite(self.viewmats).all() or not np.isfinite(self.intrinsics).all():
            raise ValueError("Trajectory camera arrays contain non-finite values.")
        if self.environment_transforms is not None and not np.isfinite(self.environment_transforms).all():
            raise ValueError("Environment transforms contain non-finite values.")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("Width, height, and FPS must be positive.")
        if not (math.isfinite(self.near_plane) and math.isfinite(self.far_plane)):
            raise ValueError("Near and far planes must be finite.")
        if not 0 < self.near_plane < self.far_plane:
            raise ValueError("Near/far policy requires 0 < near < far.")
        if len(self.expected_cache_events) != self.timesteps:
            raise ValueError("expected_cache_events must contain one event per timestep.")
        unknown = set(self.expected_cache_events) - _CACHE_EVENTS
        if unknown:
            raise ValueError(f"Unknown cache events: {sorted(unknown)}.")
        if self.environment_phase_offsets and len(self.environment_phase_offsets) != self.batch:
            raise ValueError("environment_phase_offsets must be empty or length B.")
        if len(self.scene_sha256) != 64 or len(self.route_sha256) != 64:
            raise ValueError("scene_sha256 and route_sha256 must be 64-character digests.")
        for key in ("route_path", "scene_path", "source_manifest"):
            value = self.provenance.get(key)
            if value is not None:
                _portable_path(str(value), name=f"provenance.{key}")

    def expanded_scene_ids(self) -> np.ndarray:
        if self.scene_ids.shape == (self.batch,):
            return np.broadcast_to(self.scene_ids[None, :], (self.timesteps, self.batch))
        return self.scene_ids

    def arrays(self) -> dict[str, np.ndarray]:
        result = {
            "viewmats": np.ascontiguousarray(self.viewmats),
            "intrinsics": np.ascontiguousarray(self.intrinsics),
            "scene_ids": np.ascontiguousarray(self.scene_ids),
        }
        if self.environment_transforms is not None:
            result["environment_transforms"] = np.ascontiguousarray(self.environment_transforms)
        if self.active_camera_ids is not None:
            result["active_camera_ids"] = np.ascontiguousarray(self.active_camera_ids)
        return result

    def metadata(self, *, include_id: bool = True, npz_name: str | None = None) -> dict[str, Any]:
        arrays = self.arrays()
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "coordinate_system": "opencv_world_to_camera",
            "shape": {"timesteps": self.timesteps, "batch": self.batch},
            "width": int(self.width),
            "height": int(self.height),
            "fps": float(self.fps),
            "seed": int(self.seed),
            "scene_sha256": self.scene_sha256,
            "route_sha256": self.route_sha256,
            "near_far": {
                "policy": self.near_far_policy,
                "near": float(self.near_plane),
                "far": float(self.far_plane),
            },
            "motion_classification": self.motion_classification,
            "expected_cache_events": list(self.expected_cache_events),
            "environment_phase_offsets": [int(value) for value in self.environment_phase_offsets],
            "scene_id_broadcast": self.scene_id_broadcast,
            "provenance": dict(self.provenance),
            "route_validation": dict(self.route_validation),
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "sha256": array_sha256(value),
                }
                for name, value in sorted(arrays.items())
            },
        }
        if npz_name is not None:
            payload["arrays_file"] = _portable_path(npz_name, name="arrays_file")
        if include_id:
            payload["trajectory_id"] = self.trajectory_id
        return payload

    @property
    def trajectory_id(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.metadata(include_id=False))).hexdigest()

    def flatten_fidelity_bundle(
        self,
        *,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color_space: str = "linear_rgb",
    ) -> CameraBundle:
        view_ids = [
            f"t{time_index:06d}_b{batch_index:04d}"
            for time_index in range(self.timesteps)
            for batch_index in range(self.batch)
        ]
        return bundle_from_tensors(
            viewmats=self.viewmats.reshape(-1, 4, 4),
            intrinsics=self.intrinsics.reshape(-1, 3, 3),
            scene_ids=self.expanded_scene_ids().reshape(-1),
            width=self.width,
            height=self.height,
            background=background,
            color_space=color_space,
            scene_checksum=self.scene_sha256,
            view_ids=view_ids,
        )


def save_trajectory(trajectory: CameraTrajectory, json_path: str | Path) -> tuple[Path, Path]:
    trajectory.validate()
    json_path = Path(json_path)
    npz_path = json_path.with_suffix(".npz")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **trajectory.arrays())
    payload = trajectory.metadata(npz_name=npz_path.name)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path, npz_path


def load_trajectory(json_path: str | Path) -> CameraTrajectory:
    json_path = Path(json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CAMERA_TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported trajectory schema {payload.get('schema_version')!r}.")
    arrays_file = _portable_path(str(payload["arrays_file"]), name="arrays_file")
    with np.load(json_path.parent / arrays_file, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    declared_arrays = payload["arrays"]
    if set(arrays) != set(declared_arrays):
        raise ValueError("Trajectory array inventory does not match its manifest.")
    for name, value in arrays.items():
        declared = declared_arrays[name]
        if list(value.shape) != declared["shape"] or value.dtype.str != declared["dtype"]:
            raise ValueError(f"Trajectory array contract mismatch for {name}.")
        if array_sha256(value) != declared["sha256"]:
            raise ValueError(f"Trajectory array SHA-256 mismatch for {name}.")
    near_far = payload["near_far"]
    trajectory = CameraTrajectory(
        viewmats=arrays["viewmats"],
        intrinsics=arrays["intrinsics"],
        scene_ids=arrays["scene_ids"],
        environment_transforms=arrays.get("environment_transforms"),
        active_camera_ids=arrays.get("active_camera_ids"),
        width=int(payload["width"]),
        height=int(payload["height"]),
        fps=float(payload["fps"]),
        seed=int(payload["seed"]),
        scene_sha256=str(payload["scene_sha256"]),
        route_sha256=str(payload["route_sha256"]),
        motion_classification=str(payload["motion_classification"]),
        expected_cache_events=tuple(str(value) for value in payload["expected_cache_events"]),
        near_plane=float(near_far["near"]),
        far_plane=float(near_far["far"]),
        near_far_policy=str(near_far["policy"]),
        environment_phase_offsets=tuple(int(value) for value in payload.get("environment_phase_offsets", [])),
        scene_id_broadcast=payload.get("scene_id_broadcast"),
        provenance=payload.get("provenance", {}),
        route_validation=payload.get("route_validation", {}),
    )
    if trajectory.trajectory_id != payload.get("trajectory_id"):
        raise ValueError("Trajectory canonical JSON hash mismatch.")
    return trajectory
