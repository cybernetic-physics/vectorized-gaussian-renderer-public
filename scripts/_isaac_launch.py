"""Shared, fail-closed launch policy for finite Isaac graphical tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ACTIVE_GPU_ENV = "VGR_ISAAC_ACTIVE_GPU"
REQUIRE_ACTIVE_GPU_ENV = "VGR_ISAAC_REQUIRE_EXPLICIT_GPU"
EXPERIENCE_ENV = "VGR_ISAAC_EXPERIENCE"
REQUIRE_EXPERIENCE_ENV = "VGR_ISAAC_REQUIRE_EXPERIENCE"


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def selected_vulkan_gpu() -> int | None:
    """Return the preflight-matched Vulkan index supplied by the gate."""

    raw = os.environ.get(ACTIVE_GPU_ENV)
    if raw is None or not raw.strip():
        if _enabled(os.environ.get(REQUIRE_ACTIVE_GPU_ENV)):
            raise RuntimeError(
                f"{REQUIRE_ACTIVE_GPU_ENV}=1 requires {ACTIVE_GPU_ENV}. "
                "Run the UUID-matching graphical gate instead of guessing an index."
            )
        return None
    try:
        index = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{ACTIVE_GPU_ENV} must be a non-negative integer.") from exc
    if index < 0:
        raise RuntimeError(f"{ACTIVE_GPU_ENV} must be a non-negative integer.")
    return index


def selected_experience() -> str | None:
    """Return the gate-created Kit experience without guessing runtime paths."""

    raw = os.environ.get(EXPERIENCE_ENV)
    if raw is None or not raw.strip():
        if _enabled(os.environ.get(REQUIRE_EXPERIENCE_ENV)):
            raise RuntimeError(f"{REQUIRE_EXPERIENCE_ENV}=1 requires {EXPERIENCE_ENV}.")
        return None
    path = Path(raw).expanduser().absolute()
    if not path.is_file():
        raise RuntimeError(f"{EXPERIENCE_ENV} does not name a file: {path}")
    return str(path)


def isaac_app_config(
    *,
    renderer: str | None = None,
    disable_viewport_updates: bool = False,
) -> dict[str, Any]:
    """Build the common one-GPU SimulationApp configuration."""

    config: dict[str, Any] = {
        "headless": True,
        "physics_gpu": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
    }
    active_gpu = selected_vulkan_gpu()
    if active_gpu is not None:
        config["active_gpu"] = active_gpu
    experience = selected_experience()
    if experience is not None:
        config["experience"] = experience
    if renderer is not None:
        config["renderer"] = renderer
    if disable_viewport_updates:
        config["disable_viewport_updates"] = True
    return config


def close_simulation_app(simulation_app: object, *, failed: bool) -> None:
    """Close Kit without letting fast shutdown replace a failure with exit 0."""

    simulation_app.close(  # type: ignore[attr-defined]
        wait_for_replicator=False,
        exit_code=1 if failed else 0,
    )
