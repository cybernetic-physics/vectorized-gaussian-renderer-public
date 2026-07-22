from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from isaacsim_gaussian_renderer.evaluation.scenarios import build_scenario
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (
    CameraTrajectory,
    load_trajectory,
    save_trajectory,
)


def trajectory(*, timesteps: int = 4, batch: int = 2) -> CameraTrajectory:
    viewmats = np.repeat(np.eye(4, dtype=np.float32)[None, None], timesteps * batch, axis=0).reshape(
        timesteps, batch, 4, 4
    )
    viewmats[:, :, 0, 3] = np.arange(timesteps, dtype=np.float32)[:, None]
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, None], timesteps * batch, axis=0).reshape(
        timesteps, batch, 3, 3
    )
    return CameraTrajectory(
        viewmats=viewmats,
        intrinsics=intrinsics,
        scene_ids=np.asarray([7, 9], dtype=np.int64),
        scene_id_broadcast="time",
        width=128,
        height=96,
        fps=30,
        seed=123,
        scene_sha256="a" * 64,
        route_sha256="b" * 64,
        motion_classification="moving-camera",
        expected_cache_events=("miss",) * timesteps,
        environment_phase_offsets=(0, 2),
        provenance={"route_path": "benchmarks/camera_paths/test.json"},
    )


def test_round_trip_and_fidelity_flatten(tmp_path) -> None:
    source = trajectory()
    json_path, npz_path = save_trajectory(source, tmp_path / "trajectory.json")
    assert npz_path.is_file()
    loaded = load_trajectory(json_path)
    assert loaded.trajectory_id == source.trajectory_id
    assert np.array_equal(loaded.viewmats, source.viewmats)
    bundle = loaded.flatten_fidelity_bundle()
    assert len(bundle.cameras) == loaded.timesteps * loaded.batch
    assert bundle.cameras[-1].view_id == "t000003_b0001"


def test_hash_rejects_array_tamper(tmp_path) -> None:
    json_path, npz_path = save_trajectory(trajectory(), tmp_path / "trajectory.json")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["viewmats"] = arrays["viewmats"].copy()
    arrays["viewmats"][1, 0, 0, 3] += 1
    np.savez_compressed(npz_path, **arrays)
    with pytest.raises(ValueError, match="SHA-256"):
        load_trajectory(json_path)


def test_portable_provenance_rejects_absolute_path() -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        CameraTrajectory(
            **{
                **trajectory().__dict__,
                "provenance": {"route_path": "/workspace/private/route.json"},
            }
        )


def test_scenario_cache_contracts() -> None:
    source = trajectory()
    static = build_scenario(source, "static-repeat", batch=8)
    assert static.expected_cache_events == ("miss", "hit", "hit", "hit")
    moving = build_scenario(source, "phase-offset-vectorized-flyby", batch=8)
    assert set(moving.expected_cache_events) == {"miss"}
    cadence = build_scenario(source, "active-subset-cadence", batch=8, render_every_n=2)
    assert cadence.rendered_requests == 2
    assert cadence.expected_cache_events == ("miss", "skip", "miss", "skip")


def test_contract_exact_preserves_measured_vectorized_tensors() -> None:
    source = trajectory()
    exact = build_scenario(source, "contract-exact", batch=source.batch)

    assert exact.execution_semantics == "vectorized-temporal"
    assert np.array_equal(exact.viewmats, source.viewmats)
    assert np.array_equal(exact.intrinsics, source.intrinsics)
    assert np.array_equal(exact.scene_ids, source.expanded_scene_ids())
    assert exact.expected_cache_events == source.expected_cache_events
    assert exact.environment_phase_offsets == source.environment_phase_offsets
    assert not np.shares_memory(exact.viewmats, source.viewmats)


def test_contract_exact_rejects_batch_reinterpretation() -> None:
    source = trajectory()
    with pytest.raises(ValueError, match="requires batch to match"):
        build_scenario(source, "contract-exact", batch=8)


def test_json_manifest_contains_no_absolute_workspace_path(tmp_path) -> None:
    json_path, _ = save_trajectory(trajectory(), tmp_path / "trajectory.json")
    payload = json.loads(json_path.read_text())
    assert "/workspace/" not in json.dumps(payload)


def test_checked_in_route_hashes_are_canonical() -> None:
    route_root = Path(__file__).resolve().parents[1] / "benchmarks" / "camera_paths"
    for name in ("home-scan-representative-v1.json", "home-scan-stress-v1.json"):
        payload = json.loads((route_root / name).read_text(encoding="utf-8"))
        expected = payload.pop("route_sha256")
        payload.pop("validation", None)
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == expected


def test_active_camera_padding_round_trip(tmp_path) -> None:
    source = trajectory()
    padded = CameraTrajectory(
        **{
            **source.__dict__,
            "active_camera_ids": np.asarray([[0, 1], [0, -1], [-1, -1], [1, -1]]),
        }
    )
    loaded = load_trajectory(save_trajectory(padded, tmp_path / "padded.json")[0])
    assert np.array_equal(loaded.active_camera_ids, padded.active_camera_ids)
