"""Immutable logical-batch camera contracts and microbatch scheduling audits."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .trajectory_contract import CameraTrajectory, array_sha256


def build_logical_phase_contract(
    source: CameraTrajectory,
    *,
    logical_batch: int,
    measured_steps: int,
    step_stride: int,
) -> CameraTrajectory:
    """Turn a dense B1 route into a few all-distinct, all-moving logical batches.

    Every logical environment receives a different phase of the same immutable
    route. Advancing a measured step rotates every environment by
    ``step_stride`` route samples, so no camera can inherit a static cache hit.
    """
    if source.batch != 1:
        raise ValueError("The source route must have batch 1.")
    if logical_batch <= 0 or measured_steps <= 1 or step_stride <= 0:
        raise ValueError(
            "logical_batch and step_stride must be positive; measured_steps must exceed one."
        )
    if source.timesteps < logical_batch:
        raise ValueError(
            "The source route must contain at least one unique sample per logical camera."
        )

    phase_offsets = np.floor(
        np.arange(logical_batch, dtype=np.float64)
        * source.timesteps
        / logical_batch
    ).astype(np.int64)
    if np.unique(phase_offsets).size != logical_batch:
        raise ValueError("Phase offsets are not unique.")
    step_offsets = (
        np.arange(measured_steps, dtype=np.int64) * step_stride
    ) % source.timesteps
    route_indices = (
        step_offsets[:, None] + phase_offsets[None, :]
    ) % source.timesteps

    source_scene_ids = source.expanded_scene_ids()[:, 0]
    contract = CameraTrajectory(
        viewmats=np.ascontiguousarray(source.viewmats[:, 0][route_indices]),
        intrinsics=np.ascontiguousarray(source.intrinsics[:, 0][route_indices]),
        scene_ids=np.ascontiguousarray(source_scene_ids[route_indices]),
        width=source.width,
        height=source.height,
        fps=source.fps,
        seed=source.seed,
        scene_sha256=source.scene_sha256,
        route_sha256=source.route_sha256,
        motion_classification="logical-batch-independent-moving-camera",
        expected_cache_events=("miss",) * measured_steps,
        near_plane=source.near_plane,
        far_plane=source.far_plane,
        near_far_policy=source.near_far_policy,
        environment_phase_offsets=tuple(int(value) for value in phase_offsets),
        provenance={
            **source.provenance,
            "logical_batch_policy": "phase-offset-dense-route-v1",
            "logical_batch": logical_batch,
            "measured_steps": measured_steps,
            "step_stride": step_stride,
            "source_route_samples": source.timesteps,
        },
        route_validation=source.route_validation,
    )
    audit = audit_logical_camera_contract(contract)
    if not audit["pass"]:
        raise ValueError(f"Logical camera contract failed its audit: {audit}")
    return contract


def audit_logical_camera_contract(trajectory: CameraTrajectory) -> dict[str, Any]:
    """Prove exact per-step uniqueness and between-step camera motion."""
    viewmats = np.ascontiguousarray(trajectory.viewmats)
    flattened = viewmats.reshape(trajectory.timesteps, trajectory.batch, -1)
    distinct_per_step = [
        int(np.unique(step, axis=0).shape[0])
        for step in flattened
    ]
    deltas = np.linalg.norm(
        viewmats[1:] - viewmats[:-1],
        axis=(2, 3),
    )
    moved_between_steps = [
        int(np.count_nonzero(step_delta > 0.0))
        for step_delta in deltas
    ]
    min_moving_delta = (
        float(deltas[deltas > 0.0].min())
        if np.any(deltas > 0.0)
        else 0.0
    )
    return {
        "schema_version": "logical-camera-audit/v1",
        "logical_batch": trajectory.batch,
        "measured_steps": trajectory.timesteps,
        "distinct_cameras_per_step": distinct_per_step,
        "moved_cameras_between_steps": moved_between_steps,
        "minimum_moving_viewmat_frobenius_delta": min_moving_delta,
        "maximum_viewmat_frobenius_delta": (
            float(deltas.max()) if deltas.size else 0.0
        ),
        "viewmats_sha256": array_sha256(viewmats),
        "intrinsics_sha256": array_sha256(trajectory.intrinsics),
        "pass": (
            trajectory.timesteps > 1
            and all(value == trajectory.batch for value in distinct_per_step)
            and all(value == trajectory.batch for value in moved_between_steps)
            and set(trajectory.expected_cache_events) == {"miss"}
        ),
    }


def deterministic_camera_order(
    logical_batch: int,
    *,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Return one checksum-frozen permutation shared by all renderers."""
    if logical_batch <= 0:
        raise ValueError("logical_batch must be positive.")
    order = np.random.default_rng(seed).permutation(logical_batch).astype(
        np.int64,
        copy=False,
    )
    digest = hashlib.sha256(np.ascontiguousarray(order).tobytes()).hexdigest()
    return order, digest


def stratified_source_indices(source_count: int, target_count: int) -> np.ndarray:
    """Select one deterministic source index from each equal-count stratum."""
    if not 0 < target_count < source_count:
        raise ValueError("target_count must be positive and smaller than source_count.")
    if source_count * target_count > np.iinfo(np.uint64).max // 2:
        raise ValueError("Source/target product exceeds the exact uint64 sampling range.")
    strata = np.arange(target_count, dtype=np.uint64)
    selected = (
        (2 * strata + 1) * np.uint64(source_count)
        // np.uint64(2 * target_count)
    ).astype(np.int64)
    if selected.shape != (target_count,) or np.any(np.diff(selected) <= 0):
        raise AssertionError("Stratified source selection was not exact and unique.")
    return selected


def select_logical_cameras(
    trajectory: CameraTrajectory,
    camera_ids: np.ndarray,
    *,
    purpose: str,
) -> CameraTrajectory:
    """Create an immutable all-step subset for held-out fidelity evaluation."""
    camera_ids = np.asarray(camera_ids, dtype=np.int64)
    if camera_ids.ndim != 1 or camera_ids.size == 0:
        raise ValueError("camera_ids must be a nonempty one-dimensional array.")
    if (
        np.unique(camera_ids).size != camera_ids.size
        or camera_ids.min() < 0
        or camera_ids.max() >= trajectory.batch
    ):
        raise ValueError("camera_ids must be unique and within the source batch.")
    selected_hash = hashlib.sha256(
        np.ascontiguousarray(camera_ids).tobytes()
    ).hexdigest()
    selected = CameraTrajectory(
        viewmats=np.ascontiguousarray(trajectory.viewmats[:, camera_ids]),
        intrinsics=np.ascontiguousarray(trajectory.intrinsics[:, camera_ids]),
        scene_ids=np.ascontiguousarray(
            trajectory.expanded_scene_ids()[:, camera_ids]
        ),
        width=trajectory.width,
        height=trajectory.height,
        fps=trajectory.fps,
        seed=trajectory.seed,
        scene_sha256=trajectory.scene_sha256,
        route_sha256=trajectory.route_sha256,
        motion_classification=f"{trajectory.motion_classification}-{purpose}",
        expected_cache_events=trajectory.expected_cache_events,
        near_plane=trajectory.near_plane,
        far_plane=trajectory.far_plane,
        near_far_policy=trajectory.near_far_policy,
        environment_phase_offsets=tuple(
            trajectory.environment_phase_offsets[int(value)]
            for value in camera_ids
        ),
        provenance={
            **trajectory.provenance,
            "camera_subset_purpose": purpose,
            "source_trajectory_id": trajectory.trajectory_id,
            "selected_camera_ids_sha256": selected_hash,
            "selected_camera_ids": [int(value) for value in camera_ids],
        },
        route_validation=trajectory.route_validation,
    )
    audit = audit_logical_camera_contract(selected)
    if not audit["pass"]:
        raise ValueError(f"Selected camera contract failed its audit: {audit}")
    return selected


def audit_microbatch_schedule(
    order: np.ndarray,
    *,
    physical_batch: int,
) -> dict[str, Any]:
    """Fail closed unless fixed-size microbatches cover each camera once."""
    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 1 or order.size == 0:
        raise ValueError("order must be a nonempty one-dimensional array.")
    if physical_batch <= 0 or order.size % physical_batch:
        raise ValueError("physical_batch must be positive and divide logical_batch.")
    expected = np.arange(order.size, dtype=np.int64)
    exact_coverage = np.array_equal(np.sort(order), expected)
    return {
        "schema_version": "microbatch-schedule-audit/v1",
        "logical_batch": int(order.size),
        "physical_batch": int(physical_batch),
        "microbatches_per_logical_batch": int(order.size // physical_batch),
        "camera_order_sha256": hashlib.sha256(
            np.ascontiguousarray(order).tobytes()
        ).hexdigest(),
        "exact_camera_coverage": bool(exact_coverage),
        "pass": bool(exact_coverage),
    }
