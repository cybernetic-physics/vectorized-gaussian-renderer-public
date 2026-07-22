import json
from types import SimpleNamespace

import numpy as np
import pytest
import isaacsim_gaussian_renderer.evaluation.isaac_fabric_trajectory as trajectory_module

from isaacsim_gaussian_renderer.evaluation.isaac_fabric_trajectory import (
    PublishedCameraContract,
    camera_contract_hash,
    load_published_camera_contract,
    phase_offset_contract,
    resolve_camera_route_sha256,
    resolve_source_commit,
    renderer_configuration_contract,
    validate_scene_release,
    view_population_pack_contract,
)


def camera_arrays(frames: int = 8) -> tuple[np.ndarray, np.ndarray]:
    viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    viewmats[:, 0, 3] = np.arange(frames, dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0)
    intrinsics[:, 0, 0] = 64.0
    intrinsics[:, 1, 1] = 64.0
    intrinsics[:, 0, 2] = 32.0
    intrinsics[:, 1, 2] = 32.0
    return viewmats, intrinsics


def write_camera_contract(tmp_path, *, scene_sha256: str):
    viewmats, intrinsics = camera_arrays()
    manifest = {
        "schema_version": "home-scan-flyby-camera-contract/v1",
        "frame_count": len(viewmats),
        "width": 64,
        "height": 64,
        "fps": 24,
        "near_plane": 0.01,
        "far_plane": 100.0,
        "scene_sha256": scene_sha256,
        "scene_gaussian_count": 3,
    }
    manifest["camera_contract_sha256"] = camera_contract_hash(
        viewmats,
        intrinsics,
        manifest,
    )
    npz_path = tmp_path / "camera-path.npz"
    json_path = tmp_path / "camera-path.json"
    np.savez_compressed(
        npz_path,
        viewmats=viewmats,
        intrinsics=intrinsics,
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )
    json_path.write_text(json.dumps(manifest), encoding="utf-8")
    return npz_path, json_path


def test_published_contract_validates_both_files_and_hash(tmp_path) -> None:
    npz_path, json_path = write_camera_contract(tmp_path, scene_sha256="a" * 64)

    contract = load_published_camera_contract(npz_path, json_path)

    assert contract.viewmats.shape == (8, 4, 4)
    assert contract.manifest["camera_contract_sha256"]
    modified = json.loads(json_path.read_text(encoding="utf-8"))
    modified["fps"] = 30
    json_path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="different manifests"):
        load_published_camera_contract(npz_path, json_path)


def test_camera_route_identity_supports_procedural_contracts() -> None:
    route = "a" * 64
    checksum = "b" * 64
    contract = "c" * 64

    assert resolve_camera_route_sha256(
        {
            "route_sha256": route,
            "checksum_sha256": checksum,
            "camera_contract_sha256": contract,
        }
    ) == (route, "route_sha256")
    assert resolve_camera_route_sha256(
        {
            "checksum_sha256": checksum,
            "camera_contract_sha256": contract,
        }
    ) == (checksum, "checksum_sha256")
    assert resolve_camera_route_sha256(
        {"camera_contract_sha256": contract}
    ) == (contract, "camera_contract_sha256")


def test_camera_route_identity_rejects_missing_digest() -> None:
    with pytest.raises(ValueError, match="lacks a 64-character"):
        resolve_camera_route_sha256({"checksum_sha256": "not-a-digest"})


def test_phase_offset_contract_uses_reproducible_source_indices() -> None:
    viewmats, intrinsics = camera_arrays()
    contract = PublishedCameraContract(
        viewmats=viewmats,
        intrinsics=intrinsics,
        manifest={},
        npz_sha256="n",
        manifest_sha256="m",
    )

    batched_views, _, indices, offsets = phase_offset_contract(
        contract,
        batch=4,
        frames=3,
        stride=2,
        start_frame=1,
    )

    assert offsets == (0, 2, 4, 6)
    assert indices.tolist() == [[1, 3, 5, 7], [3, 5, 7, 1], [5, 7, 1, 3]]
    assert batched_views[..., 0, 3].tolist() == indices.tolist()


def test_view_population_pack_uses_each_source_view_once() -> None:
    viewmats, intrinsics = camera_arrays()
    contract = PublishedCameraContract(
        viewmats=viewmats,
        intrinsics=intrinsics,
        manifest={},
        npz_sha256="n",
        manifest_sha256="m",
    )

    batched_views, _, indices, offsets = view_population_pack_contract(
        contract,
        batch=4,
        frames=None,
        stride=1,
        start_frame=0,
    )

    assert indices.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert offsets == (0, 1, 2, 3)
    assert sorted(indices.reshape(-1).tolist()) == list(range(8))
    assert batched_views[..., 0, 3].tolist() == indices.tolist()


def test_view_population_pack_rejects_duplicate_source_views() -> None:
    viewmats, intrinsics = camera_arrays()
    contract = PublishedCameraContract(
        viewmats=viewmats,
        intrinsics=intrinsics,
        manifest={},
        npz_sha256="n",
        manifest_sha256="m",
    )

    with pytest.raises(ValueError, match="must not repeat"):
        view_population_pack_contract(
            contract,
            batch=4,
            frames=3,
            stride=1,
            start_frame=0,
        )


def test_scene_release_binds_camera_to_exact_ply(tmp_path) -> None:
    scene_path = tmp_path / "scene.ply"
    scene_path.write_bytes(b"exact-scene-bytes")
    import hashlib

    scene_sha256 = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    release = {
        "dataset_id": "indoor",
        "selection": {"record_count": 3},
        "canonical_ply": {
            "sha256": scene_sha256,
            "byte_count": scene_path.stat().st_size,
            "vertex_properties": ["x", "y", "z"],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(release), encoding="utf-8")

    result = validate_scene_release(
        scene_path,
        manifest_path,
        scene_id_label="indoor",
        camera_manifest={
            "scene_sha256": scene_sha256,
            "scene_gaussian_count": 3,
        },
    )

    assert result["sha256"] == scene_sha256
    assert result["gaussian_count"] == 3
    assert result["gaussian_count_source"] == "selection.record_count"


def test_external_scene_release_accepts_canonical_gaussian_count(tmp_path) -> None:
    scene_path = tmp_path / "external.ply"
    scene_path.write_bytes(b"external-scene-bytes")
    import hashlib

    scene_sha256 = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    release = {
        "dataset_id": "external-indoor",
        "canonical_ply": {
            "sha256": scene_sha256,
            "byte_count": scene_path.stat().st_size,
            "gaussian_count": 7,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(release), encoding="utf-8")

    result = validate_scene_release(
        scene_path,
        manifest_path,
        scene_id_label="external-indoor",
        camera_manifest={
            "scene_sha256": scene_sha256,
            "scene_gaussian_count": 7,
        },
    )

    assert result["gaussian_count"] == 7
    assert result["gaussian_count_source"] == "canonical_ply.gaussian_count"


def test_source_commit_falls_back_to_running_checkout(monkeypatch) -> None:
    monkeypatch.delenv("SOURCE_GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        trajectory_module,
        "_command_output",
        lambda command: "1" * 40 if command == ["git", "rev-parse", "HEAD"] else None,
    )

    assert resolve_source_commit() == "1" * 40


def test_source_commit_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "2" * 40)
    monkeypatch.setattr(trajectory_module, "_command_output", lambda command: "1" * 40)

    assert resolve_source_commit() == "2" * 40


def test_trajectory_renderer_configuration_contract_records_match() -> None:
    contract = renderer_configuration_contract(
        SimpleNamespace(
            rasterize_mode="antialiased",
            covariance_epsilon=0.3,
        ),
        rasterize_mode="antialiased",
        covariance_epsilon=0.3,
    )

    assert contract["matches"]
    assert contract["actual_rasterize_mode"] == "antialiased"
    assert contract["actual_covariance_epsilon"] == 0.3


def test_trajectory_renderer_configuration_contract_rejects_ignored_mode() -> None:
    with pytest.raises(AssertionError, match="configuration mismatch"):
        renderer_configuration_contract(
            SimpleNamespace(
                rasterize_mode="classic",
                covariance_epsilon=0.0,
            ),
            rasterize_mode="antialiased",
            covariance_epsilon=0.3,
        )
