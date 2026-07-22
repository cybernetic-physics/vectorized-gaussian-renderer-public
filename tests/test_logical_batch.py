from __future__ import annotations

import numpy as np
import pytest

from isaacsim_gaussian_renderer.evaluation.logical_batch import (
    audit_logical_camera_contract,
    audit_microbatch_schedule,
    build_logical_phase_contract,
    deterministic_camera_order,
    select_logical_cameras,
    stratified_source_indices,
)
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (
    CameraTrajectory,
    load_trajectory,
    save_trajectory,
)


def dense_route(samples: int = 16) -> CameraTrajectory:
    viewmats = np.repeat(
        np.eye(4, dtype=np.float32)[None, None],
        samples,
        axis=0,
    )
    viewmats[:, 0, 0, 3] = np.arange(samples, dtype=np.float32)
    viewmats[:, 0, 1, 3] = np.arange(samples, dtype=np.float32) ** 2
    intrinsics = np.repeat(
        np.eye(3, dtype=np.float32)[None, None],
        samples,
        axis=0,
    )
    return CameraTrajectory(
        viewmats=viewmats,
        intrinsics=intrinsics,
        scene_ids=np.asarray([404], dtype=np.int64),
        scene_id_broadcast="time",
        width=128,
        height=128,
        fps=60.0,
        seed=56,
        scene_sha256="a" * 64,
        route_sha256="b" * 64,
        motion_classification="dense-b1-route",
        expected_cache_events=("miss",) * samples,
        provenance={"route_path": "benchmarks/camera_paths/test.json"},
    )


def test_build_logical_phase_contract_is_distinct_and_fully_moving(tmp_path) -> None:
    contract = build_logical_phase_contract(
        dense_route(),
        logical_batch=16,
        measured_steps=4,
        step_stride=3,
    )
    audit = audit_logical_camera_contract(contract)

    assert audit["pass"] is True
    assert audit["distinct_cameras_per_step"] == [16, 16, 16, 16]
    assert audit["moved_cameras_between_steps"] == [16, 16, 16]
    assert contract.expected_cache_events == ("miss",) * 4
    assert contract.environment_phase_offsets == tuple(range(16))

    json_path, _ = save_trajectory(contract, tmp_path / "logical.json")
    assert load_trajectory(json_path).trajectory_id == contract.trajectory_id


def test_logical_contract_requires_enough_dense_route_samples() -> None:
    with pytest.raises(ValueError, match="at least one unique sample"):
        build_logical_phase_contract(
            dense_route(8),
            logical_batch=16,
            measured_steps=2,
            step_stride=1,
        )


def test_camera_order_and_microbatch_schedule_are_deterministic() -> None:
    first, first_hash = deterministic_camera_order(1024, seed=5601024)
    second, second_hash = deterministic_camera_order(1024, seed=5601024)

    assert np.array_equal(first, second)
    assert first_hash == second_hash
    audit = audit_microbatch_schedule(first, physical_batch=16)
    assert audit["pass"] is True
    assert audit["microbatches_per_logical_batch"] == 64
    assert audit["camera_order_sha256"] == first_hash


def test_select_logical_cameras_preserves_all_step_motion() -> None:
    contract = build_logical_phase_contract(
        dense_route(),
        logical_batch=16,
        measured_steps=4,
        step_stride=3,
    )
    selected = select_logical_cameras(
        contract,
        np.asarray([2, 7, 11, 15], dtype=np.int64),
        purpose="held-out-fidelity",
    )

    assert selected.batch == 4
    assert selected.timesteps == 4
    assert audit_logical_camera_contract(selected)["pass"] is True
    assert selected.provenance["source_trajectory_id"] == contract.trajectory_id


def test_microbatch_schedule_rejects_partial_or_duplicate_coverage() -> None:
    order = np.arange(16, dtype=np.int64)
    with pytest.raises(ValueError, match="divide"):
        audit_microbatch_schedule(order, physical_batch=6)

    order[-1] = 0
    audit = audit_microbatch_schedule(order, physical_batch=4)
    assert audit["pass"] is False
    assert audit["exact_camera_coverage"] is False


def test_stratified_source_indices_are_exact_unique_and_spread() -> None:
    selected = stratified_source_indices(21, 5)

    assert selected.tolist() == [2, 6, 10, 14, 18]
    assert np.unique(selected).size == 5
    assert selected[0] >= 0
    assert selected[-1] < 21


@pytest.mark.parametrize("source_count,target_count", [(5, 0), (5, 5), (5, 6)])
def test_stratified_source_indices_reject_invalid_counts(
    source_count: int,
    target_count: int,
) -> None:
    with pytest.raises(ValueError, match="positive and smaller"):
        stratified_source_indices(source_count, target_count)
