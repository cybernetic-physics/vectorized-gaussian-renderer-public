"""Shared validation and identity helpers for renderer benchmark options."""

from __future__ import annotations

import math
from typing import Literal, cast


RasterizeMode = Literal["classic", "antialiased"]
RASTERIZE_MODES = ("classic", "antialiased")
GSPLAT_RENDERED_OUTPUTS = ("rgb", "depth", "alpha")


def validate_rasterize_mode(rasterize_mode: str) -> RasterizeMode:
    """Return a validated rasterization mode."""
    if rasterize_mode not in RASTERIZE_MODES:
        raise ValueError("rasterize_mode must be 'classic' or 'antialiased'.")
    return cast(RasterizeMode, rasterize_mode)


def resolve_covariance_epsilon(
    covariance_epsilon: float | None,
    *,
    rasterize_mode: str,
    ray_gaussian_evaluation: bool,
    compact_projection_cache: bool,
) -> float:
    """Resolve the covariance regularizer shared by benchmark runners.

    Explicit values, including zero, always win. Otherwise exact-ray
    evaluation uses no screen-space regularizer, antialiasing uses gsplat's
    0.3 default, the classic compact path keeps its zero-epsilon contract,
    and other screen-space paths use 0.3.
    """
    mode = validate_rasterize_mode(rasterize_mode)
    if mode == "antialiased" and ray_gaussian_evaluation:
        raise ValueError(
            "antialiased rasterization is incompatible with exact-ray "
            "Gaussian evaluation."
        )
    if covariance_epsilon is not None:
        value = float(covariance_epsilon)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "covariance_epsilon must be finite and non-negative."
            )
        return value
    if ray_gaussian_evaluation:
        return 0.0
    if mode == "antialiased":
        return 0.3
    if compact_projection_cache:
        return 0.0
    return 0.3


def rasterize_mode_qualified(identifier: str, rasterize_mode: str) -> str:
    """Add a stable rasterization-mode component to an artifact identity."""
    mode = validate_rasterize_mode(rasterize_mode)
    suffix = f"-rasterize-{mode}"
    return identifier if identifier.endswith(suffix) else f"{identifier}{suffix}"


def gsplat_config_id(
    *,
    scene_id: str,
    scene_mode: str,
    batch_size: int,
    width: int,
    height: int,
    rasterize_mode: str,
) -> str:
    """Build the core-output gsplat configuration identity."""
    base = (
        f"cfg-v1-gsplat-{scene_id}-{scene_mode}-b{batch_size}-"
        f"{width}x{height}-rgb-depth-alpha"
    )
    return rasterize_mode_qualified(base, rasterize_mode)
