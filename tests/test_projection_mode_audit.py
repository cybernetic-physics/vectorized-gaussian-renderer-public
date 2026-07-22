from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import benchmarks.audit_ovrtx_projection_modes as audit
from benchmarks.audit_ovrtx_projection_modes import (
    authored_focal_lengths,
    authored_modes,
    compare_array,
    effect_exceeds_repeat_noise,
    runtime_mode_readback,
)


def test_compare_array_treats_matching_nan_as_equal() -> None:
    perspective = np.asarray([1.0, np.nan], dtype=np.float32)
    tangential = np.asarray([1.0, np.nan], dtype=np.float32)

    result = compare_array(perspective, tangential)

    assert result["bitwise_equal"] is True
    assert result["different_elements"] == 0
    assert result["total_different_elements"] == 0
    assert result["max_abs_difference"] == 0.0


def test_compare_array_reports_distribution_statistics() -> None:
    perspective = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    tangential = np.asarray([1.0, 2.25, 3.5, 5.0], dtype=np.float32)

    result = compare_array(perspective, tangential)

    assert result["bitwise_equal"] is False
    assert result["different_elements"] == 3
    assert result["different_fraction"] == 0.75
    assert result["max_abs_difference"] == 1.0
    assert result["mean_abs_difference"] == 0.4375
    assert result["p99_abs_difference"] > 0.9


def test_depth_numeric_comparison_uses_jointly_valid_pixels() -> None:
    perspective = np.asarray([1.0, np.inf, 3.0, np.inf], dtype=np.float32)
    tangential = np.asarray([1.0, 2.0, 3.25, np.inf], dtype=np.float32)
    jointly_valid = np.asarray([True, False, True, False])

    depth = compare_array(
        perspective,
        tangential,
        valid_mask=jointly_valid,
    )
    validity = compare_array(
        np.asarray([True, False, True, False]),
        np.asarray([True, True, True, False]),
    )

    assert depth["compared_elements"] == 2
    assert depth["excluded_elements"] == 2
    assert depth["different_elements"] == 1
    assert depth["max_abs_difference"] == 0.25
    assert validity["significant_difference_elements"] == 1


def test_authored_stage_helpers_read_mode_and_focal_length(tmp_path: Path) -> None:
    stage = tmp_path / "scene.usda"
    stage.write_text(
        'float focalLength = 32.4\nuniform token projectionModeHint = "tangential"\n',
        encoding="utf-8",
    )

    assert authored_modes(stage) == ["tangential"]
    assert authored_focal_lengths(stage) == [32.4]


def test_cross_mode_effect_uses_each_statistic_repeat_envelope() -> None:
    cross = compare_array(
        np.zeros(8, dtype=np.float32),
        np.asarray([0.25] * 4 + [0.0] * 4, dtype=np.float32),
    )
    quiet_repeat = compare_array(
        np.zeros(8, dtype=np.float32),
        np.asarray([0.001] * 4 + [0.0] * 4, dtype=np.float32),
    )

    assert (
        effect_exceeds_repeat_noise(
            cross,
            quiet_repeat,
            quiet_repeat,
            key="rgb",
        )
        is True
    )
    assert (
        effect_exceeds_repeat_noise(
            quiet_repeat,
            cross,
            cross,
            key="rgb",
        )
        is False
    )


def canonical_summary(mode: str, *, legacy: str | None = None) -> dict:
    summary = {
        "runtime_token_readback": {
            "projection_mode": {
                "attribute": "projectionModeHint",
                "requested": mode,
                "observed": [mode],
                "prim_count": 1,
                "all_match": True,
            }
        }
    }
    if legacy is not None:
        summary["ovrtx_projection_mode_observed"] = legacy
    return summary


def write_summary(
    path: Path,
    mode: str,
    *,
    legacy: str | None = None,
) -> Path:
    path.write_text(
        json.dumps(canonical_summary(mode, legacy=legacy)),
        encoding="utf-8",
    )
    return path


def test_runtime_mode_readback_accepts_canonical_run_result_shape(
    tmp_path: Path,
) -> None:
    summary = write_summary(tmp_path / "summary.json", "tangential")

    result = runtime_mode_readback(summary, expected="tangential")

    assert result["pass"] is True
    assert result["legacy_field_present"] is False
    assert result["canonical_detail_valid"] is True


def test_runtime_mode_readback_accepts_consistent_legacy_scalar(
    tmp_path: Path,
) -> None:
    summary = write_summary(
        tmp_path / "summary.json",
        "tangential",
        legacy="tangential",
    )

    result = runtime_mode_readback(summary, expected="tangential")

    assert result["pass"] is True
    assert result["legacy_field_present"] is True


def test_runtime_mode_readback_rejects_legacy_disagreement(
    tmp_path: Path,
) -> None:
    summary = write_summary(
        tmp_path / "summary.json",
        "tangential",
        legacy="perspective",
    )

    result = runtime_mode_readback(summary, expected="tangential")

    assert result["pass"] is False
    assert result["legacy_field_consistent"] is False


def test_runtime_mode_readback_rejects_missing_or_partial_detail(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"ovrtx_projection_mode_observed": "tangential"}),
        encoding="utf-8",
    )
    assert not runtime_mode_readback(
        summary,
        expected="tangential",
    )["pass"]


def test_runtime_mode_readback_rejects_configuration_duplicate_disagreement(
    tmp_path: Path,
) -> None:
    payload = canonical_summary("tangential")
    payload["configuration"] = {
        "runtime_token_readback": {
            "projection_mode": {
                "attribute": "projectionModeHint",
                "requested": "perspective",
                "observed": ["perspective"],
                "prim_count": 1,
                "all_match": True,
            }
        }
    }
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(payload), encoding="utf-8")

    result = runtime_mode_readback(summary, expected="tangential")

    assert result["pass"] is False
    assert result["duplicate_sources_consistent"] is False

    partial = canonical_summary("tangential")
    partial["runtime_token_readback"]["projection_mode"]["prim_count"] = 0
    summary.write_text(json.dumps(partial), encoding="utf-8")
    assert not runtime_mode_readback(
        summary,
        expected="tangential",
    )["pass"]


def write_candidate(
    path: Path,
    *,
    offset: float = 0.0,
    control_geometry: bool = False,
    camera_bundle_id: str = "a" * 64,
    blank: bool = False,
) -> Path:
    height = width = 16
    rgb = np.zeros((1, height, width, 3), dtype=np.float32)
    alpha = np.zeros((1, height, width), dtype=np.float32)
    depth = np.full((1, height, width), np.inf, dtype=np.float32)
    semantic = np.full((1, height, width), -1, dtype=np.int64)
    valid_depth = np.zeros((1, height, width), dtype=np.bool_)
    if not blank:
        region = np.s_[:, 2:8, 2:8] if control_geometry else np.s_[:, 4:8, 4:8]
        rgb_region = np.s_[:, 2:8, 2:8, :] if control_geometry else np.s_[:, 4:8, 4:8, :]
        rgb[rgb_region] = 0.25 + offset
        alpha[region] = 0.40 + offset
        depth[region] = 2.0 + offset
        semantic[region] = 0
        valid_depth[region] = True
    np.savez(
        path,
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        semantic=semantic,
        valid_depth=valid_depth,
        color_space=np.asarray("display_srgb"),
        background=np.zeros(3, dtype=np.float32),
        camera_bundle_id=np.asarray(camera_bundle_id),
    )
    return path


def write_stage(path: Path, *, mode: str, focal_length: float) -> Path:
    path.write_text(
        "#usda 1.0\n"
        'def ParticleField3DGaussianSplat "Splats_0000" {\n'
        f' uniform token projectionModeHint = "{mode}"\n'
        ' uniform token sortingModeHint = "zDepth"\n'
        "}\n"
        'def Camera "Camera_0" {\n'
        f" float focalLength = {focal_length}\n"
        "}\n",
        encoding="utf-8",
    )
    return path


def write_full_summary(
    path: Path,
    *,
    mode: str,
    camera_bundle_id: str,
    stage: Path,
) -> Path:
    projection_readback = {
        "attribute": "projectionModeHint",
        "requested": mode,
        "observed": [mode],
        "prim_count": 1,
        "all_match": True,
    }
    sorting_readback = {
        "attribute": "sortingModeHint",
        "requested": "zDepth",
        "observed": ["zDepth"],
        "prim_count": 1,
        "all_match": True,
    }
    summary = {
        "schema_version": "ovrtx-temporal-fidelity/v1",
        "scene_id": "projection-activation",
        "gaussian_count": 4,
        "semantic_scheme": "index-modulo",
        "semantic_group_count": 1,
        "ovrtx_projection_mode_hint": mode,
        "ovrtx_projection_mode_observed": mode,
        "ovrtx_sorting_mode": "zDepth",
        "ovrtx_sorting_mode_observed": "zDepth",
        "ovrtx_fractional_opacity": True,
        "ovrtx_color_space": "display_srgb",
        "resumed_from_frames": 0,
        "resume_sample_sequence_advanced_frames": 0,
        "resume_accumulators": None,
        "final_report_only": True,
        "camera_bundle_id": camera_bundle_id,
        "scene": {
            "checksum_sha256": "c" * 64,
            "tensor_checksum_sha256": "d" * 64,
        },
        "scene_tensor_sha256": "d" * 64,
        "runtime_token_readback": {
            "projection_mode": projection_readback,
            "sorting_mode": sorting_readback,
        },
        "provenance": {
            "project_commit": "1" * 40,
            "ovrtx_commit": "2" * 40,
            "ovrtx_version": "0.3.0",
            "renderer_version": [0, 3, 0],
            "script_sha256": "e" * 64,
            "stage_sha256": audit.file_sha256(stage),
        },
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def run_audit(
    tmp_path: Path,
    monkeypatch,
    *,
    cross: float = 0.0,
    perspective_repeat_noise: float = 0.0,
    tangential_repeat_noise: float = 0.0,
    control: float = 0.1,
    require_activation_proof: bool = True,
    tangential_focal_length: float = 32.4,
    blank_mode_candidates: bool = False,
    control_repeat_noise: float = 0.0,
) -> tuple[int, dict]:
    base_camera_id = "a" * 64
    control_camera_id = "b" * 64
    candidates = {
        "perspective": write_candidate(
            tmp_path / "perspective.npz",
            camera_bundle_id=base_camera_id,
            blank=blank_mode_candidates,
        ),
        "tangential": write_candidate(
            tmp_path / "tangential.npz",
            offset=cross,
            camera_bundle_id=base_camera_id,
            blank=blank_mode_candidates,
        ),
        "perspective_repeat": write_candidate(
            tmp_path / "perspective-repeat.npz",
            offset=perspective_repeat_noise,
            camera_bundle_id=base_camera_id,
            blank=blank_mode_candidates,
        ),
        "tangential_repeat": write_candidate(
            tmp_path / "tangential-repeat.npz",
            offset=cross + tangential_repeat_noise,
            camera_bundle_id=base_camera_id,
            blank=blank_mode_candidates,
        ),
        "control": write_candidate(
            tmp_path / "control.npz",
            control_geometry=control != 0,
            camera_bundle_id=control_camera_id,
        ),
        "control_repeat": write_candidate(
            tmp_path / "control-repeat.npz",
            offset=control_repeat_noise,
            control_geometry=control != 0,
            camera_bundle_id=control_camera_id,
        ),
    }
    stages = {
        "perspective": write_stage(
            tmp_path / "perspective.usda",
            mode="perspective",
            focal_length=32.4,
        ),
        "tangential": write_stage(
            tmp_path / "tangential.usda",
            mode="tangential",
            focal_length=tangential_focal_length,
        ),
        "perspective_repeat": write_stage(
            tmp_path / "perspective-repeat.usda",
            mode="perspective",
            focal_length=32.4,
        ),
        "tangential_repeat": write_stage(
            tmp_path / "tangential-repeat.usda",
            mode="tangential",
            focal_length=tangential_focal_length,
        ),
        "control": write_stage(
            tmp_path / "control.usda",
            mode="perspective",
            focal_length=34.02,
        ),
        "control_repeat": write_stage(
            tmp_path / "control-repeat.usda",
            mode="perspective",
            focal_length=34.02,
        ),
    }
    summaries = {
        "perspective": write_full_summary(
            tmp_path / "perspective.json",
            mode="perspective",
            camera_bundle_id=base_camera_id,
            stage=stages["perspective"],
        ),
        "tangential": write_full_summary(
            tmp_path / "tangential.json",
            mode="tangential",
            camera_bundle_id=base_camera_id,
            stage=stages["tangential"],
        ),
        "perspective_repeat": write_full_summary(
            tmp_path / "perspective-repeat.json",
            mode="perspective",
            camera_bundle_id=base_camera_id,
            stage=stages["perspective_repeat"],
        ),
        "tangential_repeat": write_full_summary(
            tmp_path / "tangential-repeat.json",
            mode="tangential",
            camera_bundle_id=base_camera_id,
            stage=stages["tangential_repeat"],
        ),
        "control": write_full_summary(
            tmp_path / "control.json",
            mode="perspective",
            camera_bundle_id=control_camera_id,
            stage=stages["control"],
        ),
        "control_repeat": write_full_summary(
            tmp_path / "control-repeat.json",
            mode="perspective",
            camera_bundle_id=control_camera_id,
            stage=stages["control_repeat"],
        ),
    }
    output = tmp_path / "report.json"
    argv = [
        "audit",
        "--perspective-candidate",
        str(candidates["perspective"]),
        "--tangential-candidate",
        str(candidates["tangential"]),
        "--perspective-repeat",
        str(candidates["perspective_repeat"]),
        "--tangential-repeat",
        str(candidates["tangential_repeat"]),
        "--perspective-repeat-stage",
        str(stages["perspective_repeat"]),
        "--tangential-repeat-stage",
        str(stages["tangential_repeat"]),
        "--perspective-stage",
        str(stages["perspective"]),
        "--tangential-stage",
        str(stages["tangential"]),
        "--perspective-summary",
        str(summaries["perspective"]),
        "--tangential-summary",
        str(summaries["tangential"]),
        "--perspective-repeat-summary",
        str(summaries["perspective_repeat"]),
        "--tangential-repeat-summary",
        str(summaries["tangential_repeat"]),
        "--positive-control-candidate",
        str(candidates["control"]),
        "--positive-control-stage",
        str(stages["control"]),
        "--positive-control-summary",
        str(summaries["control"]),
        "--positive-control-repeat-candidate",
        str(candidates["control_repeat"]),
        "--positive-control-repeat-stage",
        str(stages["control_repeat"]),
        "--positive-control-repeat-summary",
        str(summaries["control_repeat"]),
        "--require-positive-control",
        "--output",
        str(output),
    ]
    if require_activation_proof:
        argv.insert(-2, "--require-activation-proof")
    monkeypatch.setattr(sys, "argv", argv)
    try:
        audit.main()
        exit_code = 0
    except SystemExit as exc:
        exit_code = int(exc.code)
    return exit_code, json.loads(output.read_text(encoding="utf-8"))


def test_zero_mode_effect_with_detected_control_is_honest_negative(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exit_code, report = run_audit(tmp_path, monkeypatch)

    assert exit_code == 1
    assert report["classification"] == "NO_OBSERVABLE_MODE_EFFECT"
    assert report["token_proof_valid"] is True
    assert report["positive_control_valid"] is True
    assert report["behavioral_activation_valid"] is False


def test_same_mode_repeat_noise_prevents_false_activation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exit_code, report = run_audit(
        tmp_path,
        monkeypatch,
        cross=0.05,
        perspective_repeat_noise=0.05,
    )

    assert exit_code == 1
    assert report["classification"] == "MODE_EFFECT_WITHIN_REPEAT_NOISE"
    assert report["effect_above_repeat_noise"] is False


def test_genuine_mode_effect_above_noise_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exit_code, report = run_audit(
        tmp_path,
        monkeypatch,
        cross=0.2,
        perspective_repeat_noise=0.001,
        tangential_repeat_noise=0.001,
    )

    assert exit_code == 0
    assert report["classification"] == "OBSERVABLE_MODE_EFFECT"
    assert report["activation_proof_valid"] is True


def test_insensitive_positive_control_fails_before_mode_verdict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exit_code, report = run_audit(
        tmp_path,
        monkeypatch,
        control=0.0,
    )

    assert exit_code == 1
    assert report["classification"] == "POSITIVE_CONTROL_NOT_DETECTED"
    assert report["positive_control_valid"] is False


def test_positive_control_requirement_cannot_fail_open_without_activation_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exit_code, report = run_audit(
        tmp_path,
        monkeypatch,
        cross=0.2,
        control=0.0,
        require_activation_proof=False,
    )

    assert exit_code == 1
    assert report["classification"] == "POSITIVE_CONTROL_NOT_DETECTED"
    assert report["pass"] is False


def test_mismatched_tangential_camera_cannot_be_credited_as_mode_effect(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exit_code, report = run_audit(
        tmp_path,
        monkeypatch,
        cross=0.2,
        tangential_focal_length=40.0,
    )

    assert exit_code == 1
    assert report["classification"] == "INVALID_MATCHED_LANE_CONTRACT"
    assert report["matched_lane_contract_valid"] is False
    assert report["pass"] is False


def test_blank_mode_candidates_are_invalid_not_a_no_effect_verdict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exit_code, report = run_audit(
        tmp_path,
        monkeypatch,
        blank_mode_candidates=True,
    )

    assert exit_code == 1
    assert report["classification"] == "INVALID_CANDIDATE_OUTPUT"
    assert report["candidate_outputs_valid"] is False
    assert report["behavioral_activation_valid"] is False


def test_unstable_control_repeat_blocks_a_mode_verdict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exit_code, report = run_audit(
        tmp_path,
        monkeypatch,
        cross=0.2,
        control_repeat_noise=0.15,
    )

    assert exit_code == 1
    assert report["classification"] == "POSITIVE_CONTROL_UNSTABLE"
    assert report["positive_control_stable"] is False
    assert report["positive_control_valid"] is False
