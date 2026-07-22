import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from isaacsim_gaussian_renderer.fidelity.camera_bundle import bundle_from_tensors
from isaacsim_gaussian_renderer.fidelity.metrics import (
    compare_render_outputs,
    validate_fidelity_report,
)
from isaacsim_gaussian_renderer.fidelity.outputs import RenderOutput


def _bundle():
    return bundle_from_tensors(
        viewmats=np.repeat(np.eye(4)[None, ...], 2, axis=0),
        intrinsics=np.repeat(np.eye(3)[None, ...], 2, axis=0),
        width=4,
        height=3,
        background=(0.0, 0.0, 0.0),
    )


def _output(*, camera_bundle_id: str | None = None, color_space: str = "linear_rgb") -> RenderOutput:
    rgb = np.zeros((2, 3, 4, 3), dtype=np.float32)
    rgb[0, :, :, 0] = 0.25
    rgb[1, :, :, 1] = 0.5
    depth = np.ones((2, 3, 4), dtype=np.float32)
    alpha = np.ones((2, 3, 4), dtype=np.float32) * 0.75
    semantic = np.zeros((2, 3, 4), dtype=np.int32)
    return RenderOutput(
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        semantic=semantic,
        color_space=color_space,
        background=(0.0, 0.0, 0.0),
        camera_bundle_id=camera_bundle_id,
    )


def test_identical_outputs_pass_and_write_artifacts(tmp_path: Path) -> None:
    bundle = _bundle()
    reference = _output(camera_bundle_id=bundle.bundle_id)
    candidate = _output(camera_bundle_id=bundle.bundle_id)

    report = compare_render_outputs(
        reference=reference,
        candidate=candidate,
        camera_bundle=bundle,
        output_dir=tmp_path,
        config_id="synthetic-self",
    )

    assert report["pass"] is True
    assert report["lpips_backend"] == "exact_identity"
    assert report["aggregate"]["metrics"]["rgb_psnr_db"]["min"] == float("inf")
    assert (tmp_path / "fidelity_report.json").is_file()
    assert (tmp_path / "fidelity_report.csv").is_file()
    assert (tmp_path / "0000_view_000000_rgb_side_by_side.png").is_file()
    payload = json.loads((tmp_path / "fidelity_report.json").read_text())
    assert payload["aggregate"]["metrics"]["rgb_psnr_db"]["min"] == "Infinity"
    validate_fidelity_report(payload)


def test_perturbed_outputs_fail_without_relaxed_thresholds() -> None:
    bundle = _bundle()
    reference = _output(camera_bundle_id=bundle.bundle_id)
    candidate = _output(camera_bundle_id=bundle.bundle_id)
    candidate.rgb[1] += 0.2
    candidate.alpha[1] += 0.02
    candidate.depth[1] += 0.05
    candidate.semantic[1, 0, 0] = 9

    report = compare_render_outputs(
        reference=reference,
        candidate=candidate,
        camera_bundle=bundle,
        require_lpips=False,
    )

    assert report["pass"] is False
    assert report["worst_view"]["view_id"] == "view_000001"
    assert report["per_view"][1]["thresholds"]["alpha_mae"]["pass"] is False
    assert report["per_view"][1]["thresholds"]["depth_rel_error"]["pass"] is False
    assert (
        report["per_view"][1]["thresholds"][
            "semantic_foreground_agreement"
        ]["pass"]
        is False
    )


def test_worst_view_visuals_and_report_mutation_gate(tmp_path: Path) -> None:
    bundle = _bundle()
    reference = _output(camera_bundle_id=bundle.bundle_id)
    candidate = _output(camera_bundle_id=bundle.bundle_id)
    candidate.rgb[1] += 0.2

    report = compare_render_outputs(
        reference=reference,
        candidate=candidate,
        camera_bundle=bundle,
        output_dir=tmp_path,
        require_lpips=False,
        max_artifact_views=1,
    )

    assert report["artifact_view_indices"] == [1]
    assert (tmp_path / "0001_view_000001_rgb_absdiff.png").is_file()
    payload = json.loads((tmp_path / "fidelity_report.json").read_text())
    validate_fidelity_report(payload)
    payload["per_view"][1]["pass"] = True
    with pytest.raises(ValueError, match="pass differs"):
        validate_fidelity_report(payload)


def test_valid_depth_mask_ignores_invalid_reference_pixels() -> None:
    bundle = _bundle()
    reference = _output(camera_bundle_id=bundle.bundle_id)
    candidate = _output(camera_bundle_id=bundle.bundle_id)
    valid_depth = np.zeros((2, 3, 4), dtype=bool)
    valid_depth[:, 0, 0] = True
    candidate_depth = candidate.depth.copy()
    candidate_depth += 100.0
    candidate_depth[:, 0, 0] = reference.depth[:, 0, 0]
    reference = replace(reference, valid_depth=valid_depth)
    candidate = replace(candidate, depth=candidate_depth, valid_depth=valid_depth)

    report = compare_render_outputs(
        reference=reference,
        candidate=candidate,
        camera_bundle=bundle,
        require_lpips=False,
    )

    assert report["per_view"][0]["metrics"]["valid_depth_pixels"] == 1
    assert report["per_view"][0]["metrics"]["depth_rel_error"] == 0.0


def test_no_valid_depth_pixels_fails_depth_threshold() -> None:
    bundle = _bundle()
    reference = _output(camera_bundle_id=bundle.bundle_id)
    candidate = _output(camera_bundle_id=bundle.bundle_id)
    reference = replace(reference, valid_depth=np.zeros((2, 3, 4), dtype=bool))

    report = compare_render_outputs(
        reference=reference,
        candidate=candidate,
        camera_bundle=bundle,
        require_lpips=False,
    )

    assert report["pass"] is False
    assert report["per_view"][0]["metrics"]["depth_rel_error"] is None
    assert report["per_view"][0]["thresholds"]["depth_rel_error"]["pass"] is False


def test_contract_mismatch_is_rejected() -> None:
    bundle = _bundle()
    reference = _output(camera_bundle_id=bundle.bundle_id)
    candidate = _output(camera_bundle_id=bundle.bundle_id, color_space="srgb")

    with pytest.raises(ValueError, match="color_space"):
        compare_render_outputs(reference=reference, candidate=candidate, camera_bundle=bundle)
