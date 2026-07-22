"""Explicit trajectory execution scenarios and cache expectations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .trajectory_contract import CameraTrajectory

ScenarioName = Literal[
    "contract-exact",
    "static-repeat",
    "appearance-updates",
    "sequential-flyby",
    "phase-offset-vectorized-flyby",
    "mixed-motion",
    "intrinsics-sweep",
    "environment-transform-motion",
    "teleport",
    "active-subset-cadence",
    "multi-scene-trajectory",
]
ExecutionSemantics = Literal["spatial-batch", "sequential-temporal", "vectorized-temporal"]


@dataclass(frozen=True)
class ScenarioSpec:
    name: ScenarioName
    execution_semantics: ExecutionSemantics
    viewmats: np.ndarray
    intrinsics: np.ndarray
    scene_ids: np.ndarray
    expected_cache_events: tuple[str, ...]
    environment_transforms: np.ndarray | None = None
    active_camera_ids: tuple[np.ndarray | None, ...] = ()
    appearance_update_steps: tuple[int, ...] = ()
    moved_fraction: float | None = None
    render_every_n: int = 1
    environment_phase_offsets: tuple[int, ...] = ()

    @property
    def timesteps(self) -> int:
        return int(self.viewmats.shape[0])

    @property
    def batch(self) -> int:
        return int(self.viewmats.shape[1])

    @property
    def rendered_requests(self) -> int:
        return sum(event != "skip" for event in self.expected_cache_events)


def _repeat_first(trajectory: CameraTrajectory) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.repeat(trajectory.viewmats[:1], trajectory.timesteps, axis=0),
        np.repeat(trajectory.intrinsics[:1], trajectory.timesteps, axis=0),
    )


def _phase_offset(trajectory: CameraTrajectory, batch: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    offsets = np.floor(np.arange(batch, dtype=np.float64) * trajectory.timesteps / batch).astype(np.int64)
    time = np.arange(trajectory.timesteps, dtype=np.int64)[:, None]
    indices = (time + offsets[None, :]) % trajectory.timesteps
    base_view = trajectory.viewmats[:, 0]
    base_intrinsics = trajectory.intrinsics[:, 0]
    return base_view[indices], base_intrinsics[indices], tuple(int(value) for value in offsets)


def build_scenario(
    trajectory: CameraTrajectory,
    name: ScenarioName,
    *,
    batch: int | None = None,
    moved_fraction: float = 1.0,
    render_every_n: int = 1,
) -> ScenarioSpec:
    """Construct a scenario without mutating the source contract."""
    if name == "contract-exact":
        if batch is not None and batch != trajectory.batch:
            raise ValueError(
                "contract-exact requires batch to match the trajectory contract "
                f"({batch} != {trajectory.batch})."
            )
        active: tuple[np.ndarray | None, ...]
        if trajectory.active_camera_ids is None:
            active = tuple(None for _ in range(trajectory.timesteps))
        else:
            active = tuple(
                None
                if event == "skip"
                else np.ascontiguousarray(row[row >= 0])
                for row, event in zip(
                    trajectory.active_camera_ids,
                    trajectory.expected_cache_events,
                    strict=True,
                )
            )
        return ScenarioSpec(
            name=name,
            execution_semantics=(
                "sequential-temporal" if trajectory.batch == 1 else "vectorized-temporal"
            ),
            viewmats=np.ascontiguousarray(trajectory.viewmats.copy()),
            intrinsics=np.ascontiguousarray(trajectory.intrinsics.copy()),
            scene_ids=np.ascontiguousarray(trajectory.expanded_scene_ids().copy()),
            expected_cache_events=trajectory.expected_cache_events,
            environment_transforms=(
                None
                if trajectory.environment_transforms is None
                else np.ascontiguousarray(trajectory.environment_transforms.copy())
            ),
            active_camera_ids=active,
            environment_phase_offsets=trajectory.environment_phase_offsets,
        )
    if batch is None:
        batch = trajectory.batch
    if batch <= 0 or render_every_n <= 0:
        raise ValueError("batch and render_every_n must be positive.")
    if not 0.0 <= moved_fraction <= 1.0:
        raise ValueError("moved_fraction must be in [0,1].")

    scene_base = trajectory.expanded_scene_ids()[:, :1]
    phase_offsets: tuple[int, ...] = ()
    if name in {"static-repeat", "appearance-updates", "intrinsics-sweep"}:
        base_view, base_intrinsics = _repeat_first(trajectory)
        viewmats = np.repeat(base_view[:, :1], batch, axis=1)
        intrinsics = np.repeat(base_intrinsics[:, :1], batch, axis=1)
    elif name == "sequential-flyby":
        batch = 1
        viewmats = trajectory.viewmats[:, :1].copy()
        intrinsics = trajectory.intrinsics[:, :1].copy()
    else:
        viewmats, intrinsics, phase_offsets = _phase_offset(trajectory, batch)
    scene_ids = np.repeat(scene_base, batch, axis=1)
    environment_transforms = None
    appearance_steps: tuple[int, ...] = ()
    active: tuple[np.ndarray | None, ...] = tuple(None for _ in range(trajectory.timesteps))
    events = ["miss"] * trajectory.timesteps
    semantics: ExecutionSemantics = "vectorized-temporal"

    if name == "static-repeat":
        semantics = "spatial-batch"
        events = ["miss", *("hit" for _ in range(trajectory.timesteps - 1))]
    elif name == "appearance-updates":
        semantics = "spatial-batch"
        appearance_steps = tuple(range(1, trajectory.timesteps))
        events = ["miss", *("hit" for _ in range(trajectory.timesteps - 1))]
    elif name == "sequential-flyby":
        semantics = "sequential-temporal"
    elif name == "phase-offset-vectorized-flyby":
        pass
    elif name == "mixed-motion":
        moving = int(np.ceil(batch * moved_fraction))
        static_view = viewmats[:1].copy()
        if moving < batch:
            viewmats[:, moving:] = np.repeat(static_view[:, moving:], trajectory.timesteps, axis=0)
        if moving == 0:
            events = ["miss", *("hit" for _ in range(trajectory.timesteps - 1))]
    elif name == "intrinsics-sweep":
        for step in range(trajectory.timesteps):
            scale = 0.9 + 0.2 * step / max(trajectory.timesteps - 1, 1)
            intrinsics[step, :, 0, 0] *= scale
            intrinsics[step, :, 1, 1] *= scale
            intrinsics[step, :, 0, 2] += 0.125 * step
    elif name == "environment-transform-motion":
        environment_transforms = np.repeat(
            np.eye(4, dtype=np.float32)[None, None],
            trajectory.timesteps * batch,
            axis=0,
        ).reshape(trajectory.timesteps, batch, 4, 4)
        environment_transforms[:, :, 0, 3] = np.linspace(0.0, 0.25, trajectory.timesteps)[:, None]
    elif name == "teleport":
        order = np.empty(trajectory.timesteps, dtype=np.int64)
        low, high = 0, trajectory.timesteps - 1
        for index in range(trajectory.timesteps):
            if index % 2 == 0:
                order[index], low = low, low + 1
            else:
                order[index], high = high, high - 1
        viewmats = viewmats[order]
        intrinsics = intrinsics[order]
    elif name == "active-subset-cadence":
        active_items: list[np.ndarray | None] = []
        for step in range(trajectory.timesteps):
            if step % render_every_n:
                events[step] = "skip"
                active_items.append(None)
            else:
                count = max(1, 1 + (step // render_every_n) % batch)
                active_items.append(np.arange(count, dtype=np.int64))
        active = tuple(active_items)
    elif name == "multi-scene-trajectory":
        scene_ids = np.empty((trajectory.timesteps, batch), dtype=np.int64)
        arbitrary = np.asarray([7, 404, 99991], dtype=np.int64)
        for step in range(trajectory.timesteps):
            scene_ids[step] = arbitrary[(np.arange(batch) + step) % arbitrary.size]
    else:  # pragma: no cover - Literal keeps typed callers honest.
        raise ValueError(f"Unknown scenario {name!r}.")

    return ScenarioSpec(
        name=name,
        execution_semantics=semantics,
        viewmats=np.ascontiguousarray(viewmats),
        intrinsics=np.ascontiguousarray(intrinsics),
        scene_ids=np.ascontiguousarray(scene_ids),
        expected_cache_events=tuple(events),
        environment_transforms=environment_transforms,
        active_camera_ids=active,
        appearance_update_steps=appearance_steps,
        moved_fraction=moved_fraction if name == "mixed-motion" else None,
        render_every_n=render_every_n,
        environment_phase_offsets=phase_offsets,
    )
