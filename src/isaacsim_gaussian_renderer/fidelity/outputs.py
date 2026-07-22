"""Canonical render-output schema loading for fidelity comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RENDER_OUTPUT_SCHEMA_VERSION = "render-output-v1"


@dataclass(frozen=True)
class RenderOutput:
    """Machine-readable renderer output tensors in canonical CPU ndarray form."""

    rgb: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray
    semantic: np.ndarray
    valid_depth: np.ndarray | None = None
    color_space: str | None = None
    background: tuple[float, float, float] | None = None
    camera_bundle_id: str | None = None
    source: str | None = None

    def validate(
        self,
        *,
        expected_views: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        if self.rgb.ndim != 4 or self.rgb.shape[-1] != 3:
            raise ValueError(f"rgb must have shape [V,H,W,3], got {self.rgb.shape}.")
        expected_shape = self.rgb.shape[:3]
        for name, array in (("alpha", self.alpha), ("depth", self.depth), ("semantic", self.semantic)):
            if array.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}.")
        if self.valid_depth is not None and self.valid_depth.shape != expected_shape:
            raise ValueError(f"valid_depth must have shape {expected_shape}, got {self.valid_depth.shape}.")
        if expected_views is not None and self.rgb.shape[0] != expected_views:
            raise ValueError(f"Expected {expected_views} views, got {self.rgb.shape[0]}.")
        if height is not None and self.rgb.shape[1] != height:
            raise ValueError(f"Expected height {height}, got {self.rgb.shape[1]}.")
        if width is not None and self.rgb.shape[2] != width:
            raise ValueError(f"Expected width {width}, got {self.rgb.shape[2]}.")
        if not np.isfinite(self.rgb).all():
            raise ValueError("rgb contains non-finite values.")
        if not np.isfinite(self.alpha).all():
            raise ValueError("alpha contains non-finite values.")


def load_render_output(path: str | Path) -> RenderOutput:
    """Load either a `.npz` tensor bundle or a JSON manifest pointing at `.npy` arrays."""
    path = Path(path)
    if path.suffix == ".json":
        return _load_manifest(path)
    if path.suffix == ".npz":
        return _load_npz(path)
    raise ValueError(f"Unsupported render-output path {path}; expected .json or .npz.")


def _load_npz(path: Path) -> RenderOutput:
    with np.load(path, allow_pickle=False) as data:
        output = RenderOutput(
            rgb=np.asarray(data["rgb"], dtype=np.float32),
            alpha=_squeeze_last(np.asarray(data["alpha"], dtype=np.float32)),
            depth=_squeeze_last(np.asarray(data["depth"], dtype=np.float32)),
            semantic=_squeeze_last(np.asarray(data["semantic"])),
            valid_depth=_optional_array(data, "valid_depth", dtype=bool),
            color_space=_optional_scalar(data, "color_space"),
            background=_optional_background(data),
            camera_bundle_id=_optional_scalar(data, "camera_bundle_id"),
            source=str(path),
        )
    output.validate()
    return output


def _load_manifest(path: Path) -> RenderOutput:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != RENDER_OUTPUT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported render-output schema: {payload.get('schema_version')!r}.")
    base = path.parent
    arrays: dict[str, Any] = payload["arrays"]
    valid_depth = None
    if arrays.get("valid_depth") is not None:
        valid_depth = _squeeze_last(np.load(base / arrays["valid_depth"], allow_pickle=False).astype(bool))
    output = RenderOutput(
        rgb=np.load(base / arrays["rgb"], allow_pickle=False).astype(np.float32),
        alpha=_squeeze_last(np.load(base / arrays["alpha"], allow_pickle=False).astype(np.float32)),
        depth=_squeeze_last(np.load(base / arrays["depth"], allow_pickle=False).astype(np.float32)),
        semantic=_squeeze_last(np.load(base / arrays["semantic"], allow_pickle=False)),
        valid_depth=valid_depth,
        color_space=payload.get("color_space"),
        background=tuple(float(v) for v in payload["background"]) if payload.get("background") is not None else None,
        camera_bundle_id=payload.get("camera_bundle_id"),
        source=str(path),
    )
    output.validate()
    return output


def _squeeze_last(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    return array


def _optional_array(data: Any, key: str, *, dtype: Any) -> np.ndarray | None:
    if key not in data:
        return None
    return _squeeze_last(np.asarray(data[key], dtype=dtype))


def _optional_scalar(data: Any, key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if np.asarray(value).shape == ():
        return str(np.asarray(value).item())
    return str(value)


def _optional_background(data: Any) -> tuple[float, float, float] | None:
    if "background" not in data:
        return None
    values = np.asarray(data["background"], dtype=np.float64).reshape(-1)
    if values.shape != (3,):
        raise ValueError(f"background must contain three values, got {values.shape}.")
    return tuple(float(v) for v in values)
