from pathlib import Path

import numpy as np

from isaacsim_gaussian_renderer.fidelity.camera_bundle import (
    bundle_from_tensors,
    write_camera_bundle,
)
from isaacsim_gaussian_renderer.fidelity.cli import main


def test_compare_fidelity_cli_consumes_npz_outputs(tmp_path: Path) -> None:
    bundle = bundle_from_tensors(
        viewmats=np.eye(4)[None, ...],
        intrinsics=np.eye(3)[None, ...],
        width=2,
        height=2,
    )
    camera_path = tmp_path / "cameras.json"
    write_camera_bundle(bundle, camera_path)

    arrays = {
        "rgb": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "alpha": np.ones((1, 2, 2), dtype=np.float32),
        "depth": np.ones((1, 2, 2), dtype=np.float32),
        "semantic": np.zeros((1, 2, 2), dtype=np.int32),
        "color_space": np.array("linear_rgb"),
        "background": np.array([0.0, 0.0, 0.0]),
        "camera_bundle_id": np.array(bundle.bundle_id),
    }
    reference_path = tmp_path / "reference.npz"
    candidate_path = tmp_path / "candidate.npz"
    np.savez(reference_path, **arrays)
    np.savez(candidate_path, **arrays)

    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "--reference",
            str(reference_path),
            "--candidate",
            str(candidate_path),
            "--camera-bundle",
            str(camera_path),
            "--output-dir",
            str(output_dir),
            "--config-id",
            "unit-cli",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "fidelity_report.json").is_file()
    assert (output_dir / "fidelity_report.csv").is_file()
    assert (output_dir / "0000_view_000000_rgb_absdiff.png").is_file()


def test_compare_fidelity_cli_returns_nonzero_for_perturbation(tmp_path: Path) -> None:
    bundle = bundle_from_tensors(
        viewmats=np.eye(4)[None, ...],
        intrinsics=np.eye(3)[None, ...],
        width=2,
        height=2,
    )
    camera_path = tmp_path / "cameras.json"
    write_camera_bundle(bundle, camera_path)

    reference = {
        "rgb": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "alpha": np.ones((1, 2, 2), dtype=np.float32),
        "depth": np.ones((1, 2, 2), dtype=np.float32),
        "semantic": np.zeros((1, 2, 2), dtype=np.int32),
        "color_space": np.array("linear_rgb"),
        "background": np.array([0.0, 0.0, 0.0]),
        "camera_bundle_id": np.array(bundle.bundle_id),
    }
    candidate = dict(reference)
    candidate["rgb"] = np.ones((1, 2, 2, 3), dtype=np.float32)

    reference_path = tmp_path / "reference.npz"
    candidate_path = tmp_path / "candidate.npz"
    np.savez(reference_path, **reference)
    np.savez(candidate_path, **candidate)

    exit_code = main(
        [
            "--reference",
            str(reference_path),
            "--candidate",
            str(candidate_path),
            "--camera-bundle",
            str(camera_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-lpips-if-unavailable",
        ]
    )

    assert exit_code == 1
