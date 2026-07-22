from pathlib import Path

import numpy as np
import pytest

from isaacsim_gaussian_renderer.fidelity.camera_bundle import (
    bundle_from_tensors,
    load_camera_bundle,
    write_camera_bundle,
)


def test_camera_bundle_round_trip_is_deterministic(tmp_path: Path) -> None:
    viewmats = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    intrinsics = np.array([[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]])

    bundle = bundle_from_tensors(
        viewmats=viewmats,
        intrinsics=intrinsics,
        width=8,
        height=6,
        background=(0.1, 0.2, 0.3),
        scene_ids=np.array([0, 3]),
        view_ids=["left", "right"],
        scene_checksum="sha256:test",
    )
    path = tmp_path / "cameras.json"
    write_camera_bundle(bundle, path)

    loaded = load_camera_bundle(path)
    assert loaded.bundle_id == bundle.bundle_id
    assert loaded.cameras[1].scene_id == 3
    assert loaded.color_space == "linear_rgb"


def test_camera_bundle_rejects_duplicate_view_ids() -> None:
    viewmats = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    intrinsics = np.repeat(np.eye(3)[None, ...], 2, axis=0)

    with pytest.raises(ValueError, match="Duplicate camera"):
        bundle_from_tensors(
            viewmats=viewmats,
            intrinsics=intrinsics,
            width=4,
            height=4,
            view_ids=["same", "same"],
        )


def test_camera_bundle_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="viewmats"):
        bundle_from_tensors(
            viewmats=np.eye(3),
            intrinsics=np.eye(3),
            width=4,
            height=4,
        )
