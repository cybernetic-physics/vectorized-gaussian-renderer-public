from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import compare_flashgs_b64_repair_oracle as target
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (
    GSPLAT_BUILD_ATTESTATION_SCHEMA,
    GSPLAT_ORACLE_SCHEMA,
    HEADLINE_COMPUTE_CAPABILITY,
    HEADLINE_GPU_NAME,
    HEADLINE_GPU_UUID,
    HEADLINE_TORCH_CUDA_ARCH_LIST,
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PINNED_GSPLAT_COMMIT,
    PINNED_GSPLAT_PATCHED_UTILS_SHA256,
    PINNED_GSPLAT_PATCH_SHA256,
    artifact_record,
)
from isaacsim_gaussian_renderer.evaluation.matched_semantics import (
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
)
from isaacsim_gaussian_renderer.fidelity.camera_bundle import (
    bundle_from_tensors,
    write_camera_bundle,
)
from isaacsim_gaussian_renderer.flashgs_repair import (
    FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA,
)


TRAJECTORY_ID = "9" * 64
GPU_UUID = HEADLINE_GPU_UUID
GPU_NAME = HEADLINE_GPU_NAME


def _source_provenance() -> dict[str, Any]:
    return {
        "manifest": {"sha256": "1" * 64},
        "source_tree_sha256": "2" * 64,
        "head": "3" * 40,
        "dirty": False,
        "diff_sha256": "4" * 64,
    }


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    record = artifact_record(path)
    record["path"] = path.relative_to(root).as_posix()
    return record


def _arrays() -> dict[str, np.ndarray]:
    rgb = np.zeros(target.EXPECTED_RGB_SHAPE, dtype=np.float32)
    alpha = np.zeros(target.EXPECTED_SCALAR_SHAPE, dtype=np.float32)
    depth = np.full(target.EXPECTED_SCALAR_SHAPE, np.inf, dtype=np.float32)
    semantic = np.full(target.EXPECTED_SCALAR_SHAPE, -1, dtype=np.int64)
    for view in range(64):
        y = view % 128
        x = (view * 7) % 128
        rgb[view, y, x] = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
        alpha[view, y, x, 0] = np.float32(0.5)
        depth[view, y, x, 0] = np.float32(view + 1)
        semantic[view, y, x, 0] = view % 8
    return {"rgb": rgb, "alpha": alpha, "depth": depth, "semantic": semantic}


def _write_candidate(
    root: Path,
    arrays: dict[str, np.ndarray] | None = None,
    *,
    prefix: str = "repair",
) -> None:
    values = arrays or _arrays()
    np.savez_compressed(
        root / f"{prefix}.full.npz",
        rgb=values["rgb"],
        alpha=values["alpha"],
        depth=values["depth"],
        semantic_id=values["semantic"],
    )
    np.savez_compressed(root / f"{prefix}.rgb-only.npz", rgb=values["rgb"])


def _write_oracle(root: Path, arrays: dict[str, np.ndarray] | None = None, **metadata: Any) -> None:
    values = arrays or _arrays()
    np.savez_compressed(
        root / "oracle.npz",
        rgb=values["rgb"],
        alpha=values["alpha"],
        depth=values["depth"],
        semantic=values["semantic"],
        valid_depth=values["alpha"] > 0.01,
        color_space=np.asarray("linear_rgb"),
        background=np.zeros(3, dtype=np.float32),
        camera_bundle_id=np.asarray(metadata["camera_bundle_id"]),
        camera_indices=metadata.get("camera_indices", np.arange(64, dtype=np.int64)),
        steps=metadata.get("steps", np.full(64, 107, dtype=np.int64)),
        trajectory_id=np.asarray(metadata.get("trajectory_id", TRAJECTORY_ID)),
        semantic_topology=np.asarray(metadata.get("semantic_topology", REPRESENTATIVE_SEMANTIC_TOPOLOGY)),
        gaussian_support_sigma=np.asarray(MATCHED_GAUSSIAN_SUPPORT_SIGMA, dtype=np.float32),
    )


def _fake_raw_loader(
    report_path: str | Path,
    *,
    diagnosis_index: str | Path,
    diagnosis_lock: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    del diagnosis_lock
    report = Path(report_path).resolve()
    root = Path(artifact_root).resolve()
    assert report.is_relative_to(root)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    raw_records = report_payload["post_fix_raw_outputs"]
    full_path = report.parent / raw_records["full"]["path"]
    rgb_path = report.parent / raw_records["rgb_only"]["path"]
    with np.load(full_path, allow_pickle=False) as archive:
        full = {name: np.asarray(archive[name]).copy() for name in archive.files}
    with np.load(rgb_path, allow_pickle=False) as archive:
        rgb_only = {name: np.asarray(archive[name]).copy() for name in archive.files}
    metadata = {
        "camera_indices": list(range(64)),
        "step": 107,
        "trajectory_id": TRAJECTORY_ID,
        "semantic_topology": REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    }
    return {
        "report": artifact_record(report),
        "diagnosis_index": artifact_record(diagnosis_index),
        "full": {"arrays": full, **metadata},
        "rgb_only": {"arrays": rgb_only, **metadata},
    }


def _write_fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    _write_candidate(root)
    _write_candidate(root, prefix="repeat-repair")
    (root / "diagnosis-index.json").write_text("{}\n", encoding="utf-8")
    (root / "diagnosis-lock.json").write_text("{}\n", encoding="utf-8")
    for name in (
        "gsplat-build.json",
        "gsplat-native.so",
        "gsplat-python.py",
        "support.h",
        "occupancy.json",
    ):
        (root / name).write_bytes(f"fixture-{name}".encode())

    bundle = bundle_from_tensors(
        viewmats=np.repeat(np.eye(4, dtype=np.float64)[None], 64, axis=0),
        intrinsics=np.repeat(np.eye(3, dtype=np.float64)[None], 64, axis=0),
        width=128,
        height=128,
        background=(0.0, 0.0, 0.0),
        scene_ids=np.full(64, 404, dtype=np.int64),
        scene_checksum=target.HOME_SCAN_SHA256,
        view_ids=[f"t107-b{camera:04d}" for camera in range(64)],
    )
    bundle_path = root / "oracle.camera-bundle.json"
    write_camera_bundle(bundle, bundle_path)
    _write_oracle(root, camera_bundle_id=bundle.bundle_id)

    report = {
        "schema_version": FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA,
        "created_at": "2026-07-22T10:00:00+00:00",
        "debug_only": True,
        "measured_timing_valid": False,
        "pass": True,
        "tool_integrity": {"pass": True, "checks": {"fixture_integrity": True}},
        "camera": {"trajectory_id": TRAJECTORY_ID, "step": 107},
        "equation": {
            "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            "covariance_epsilon": 0.0,
            "alpha_threshold": 1.0 / 255.0,
            "alpha_cap": 0.99,
            "transmittance_threshold": 1.0e-4,
            "semantic_min_alpha": 0.01,
            "semantic_topology": REPRESENTATIVE_SEMANTIC_TOPOLOGY,
        },
        "scene": {
            "sha256": target.HOME_SCAN_SHA256,
            "gaussian_count": target.HOME_SCAN_COUNT,
            "semantic_topology": REPRESENTATIVE_SEMANTIC_TOPOLOGY,
            "canonical_precision": "float32",
        },
        "environment": {"gpu_uuid": GPU_UUID, "gpu_name": GPU_NAME},
        "source_provenance": _source_provenance(),
        "production_adapter": {"source_digest_and_repair_audit": {"pass": True}},
        "post_fix_raw_outputs": {
            "full": _relative_record(root / "repair.full.npz", root),
            "rgb_only": _relative_record(root / "repair.rgb-only.npz", root),
        },
    }
    report_path = root / "repair.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    repeat_report = json.loads(json.dumps(report))
    repeat_report["created_at"] = "2026-07-22T10:01:00+00:00"
    repeat_report["post_fix_raw_outputs"] = {
        "full": _relative_record(root / "repeat-repair.full.npz", root),
        "rgb_only": _relative_record(root / "repeat-repair.rgb-only.npz", root),
    }
    repeat_report_path = root / "repeat-repair.json"
    repeat_report_path.write_text(json.dumps(repeat_report), encoding="utf-8")

    manifest = {
        "schema_version": GSPLAT_ORACLE_SCHEMA,
        "pass": True,
        "output": str(root / "oracle.npz"),
        "output_sha256": artifact_record(root / "oracle.npz")["sha256"],
        "camera_bundle": str(bundle_path),
        "camera_bundle_artifact": artifact_record(bundle_path),
        "camera_bundle_id": bundle.bundle_id,
        "camera_indices": list(range(64)),
        "steps": [107] * 64,
        "selection_pairs": [[107, camera] for camera in range(64)],
        "selection_profile": "diagnostic-single-step",
        "trajectory_id": TRAJECTORY_ID,
        "scene_sha256": target.HOME_SCAN_SHA256,
        "gaussian_count": target.HOME_SCAN_COUNT,
        "semantic_topology": REPRESENTATIVE_SEMANTIC_TOPOLOGY,
        "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        "projection_contract": MATCHED_PROJECTION_RULES,
        "gsplat_commit": PINNED_GSPLAT_COMMIT,
        "gsplat_compatibility_patch": {
            "sha256": PINNED_GSPLAT_PATCH_SHA256,
            "patched_utils_sha256": PINNED_GSPLAT_PATCHED_UTILS_SHA256,
        },
        "gsplat_support_contract": {
            "macro": "GAUSSIAN_EXTEND",
            "value": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            "header_artifact": artifact_record(root / "support.h"),
        },
        "gsplat_build_attestation": artifact_record(root / "gsplat-build.json"),
        "gsplat_behavior_probe": {"pass": True},
        "gsplat_python_artifact": artifact_record(root / "gsplat-python.py"),
        "gsplat_native_extension": artifact_record(root / "gsplat-native.so"),
        "node_occupancy": artifact_record(root / "occupancy.json"),
        "source_provenance": _source_provenance(),
        "gpu": GPU_NAME,
        "gpu_uuid": GPU_UUID,
        "compute_capability": list(HEADLINE_COMPUTE_CAPABILITY),
        "torch_cuda_arch_list": HEADLINE_TORCH_CUDA_ARCH_LIST,
    }
    manifest_path = root / "oracle.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "root": root,
        "report": report_path,
        "repeat_report": repeat_report_path,
        "index": root / "diagnosis-index.json",
        "lock": root / "diagnosis-lock.json",
        "oracle": root / "oracle.npz",
        "manifest": manifest_path,
        "bundle": bundle_path,
    }


def _fake_support_evidence(_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": GSPLAT_BUILD_ATTESTATION_SCHEMA,
        "build_environment": {
            "TORCH_CUDA_ARCH_LIST": HEADLINE_TORCH_CUDA_ARCH_LIST
        },
    }


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    paths = _write_fixture(tmp_path / "evidence")
    monkeypatch.setattr(target, "load_verified_b64_repair_raw_outputs", _fake_raw_loader)
    monkeypatch.setattr(target, "verify_gsplat_oracle_support_evidence", _fake_support_evidence)
    return paths


def _run(paths: dict[str, Path], *, output_name: str = "comparison.json") -> dict[str, Any]:
    return target.compare_b64_repair_oracle(
        repair_report=paths["report"],
        repeat_repair_report=paths["repeat_report"],
        diagnosis_index=paths["index"],
        diagnosis_lock=paths["lock"],
        artifact_root=paths["root"],
        oracle_path=paths["oracle"],
        oracle_manifest_path=paths["manifest"],
        oracle_camera_bundle_path=paths["bundle"],
        output_path=paths["root"] / output_name,
    )


def test_all_pixel_gate_passes_and_persists_contract(evidence: dict[str, Path]) -> None:
    result = _run(evidence)

    assert result["pass"] is True
    for run_label in ("primary", "repeat"):
        assert result["oracle_comparisons"][run_label]["full_sensor"]["aggregate"]["passed_views"] == 64
        assert result["oracle_comparisons"][run_label]["rgb_only"]["aggregate"]["passed_views"] == 64
    assert result["acceptance_policy"]["all_required_oracle_comparisons_pass"] is True
    assert result["acceptance_policy"]["repeatability_drift_is_acceptance_gate"] is False
    assert result["repeatability"]["bitwise_determinism_expected"] is False
    assert result["debug_only"] is True
    assert result["measured_timing_valid"] is False
    assert result["performance_claim_valid"] is False
    persisted = json.loads((evidence["root"] / "comparison.json").read_text())
    assert persisted["oracle_comparisons"]["primary"]["full_sensor"]["aggregate"]["rgb_psnr_db_worst"] == ("Infinity")
    for record in persisted["input_artifacts"].values():
        if "path" in record:
            assert not Path(record["path"]).is_absolute()
            assert ".." not in Path(record["path"]).parts


def test_rejects_oracle_artifact_tamper(evidence: dict[str, Path]) -> None:
    with evidence["oracle"].open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="Oracle manifest"):
        _run(evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("shape", "RGB tensor contract"),
        ("dtype", "RGB tensor contract"),
    ],
)
def test_rejects_candidate_shape_and_dtype(evidence: dict[str, Path], mutation: str, message: str) -> None:
    values = _arrays()
    if mutation == "shape":
        values["rgb"] = values["rgb"][:63]
    else:
        values["rgb"] = values["rgb"].astype(np.float64)
    _write_candidate(evidence["root"], values)

    with pytest.raises(ValueError, match=message):
        _run(evidence)


def test_rejects_nonfinite_repeat_alpha(evidence: dict[str, Path]) -> None:
    values = _arrays()
    values["alpha"][0, 0, 0, 0] = np.float32(np.nan)
    _write_candidate(evidence["root"], values, prefix="repeat-repair")

    with pytest.raises(ValueError, match="full-sensor tensor contract"):
        _run(evidence)


def test_rejects_oracle_selection_metadata(evidence: dict[str, Path]) -> None:
    bundle_id = json.loads(evidence["bundle"].read_text())["bundle_id"]
    steps = np.full(64, 107, dtype=np.int64)
    steps[-1] = 106
    _write_oracle(evidence["root"], camera_bundle_id=bundle_id, steps=steps)

    with pytest.raises(ValueError, match="Oracle tensor or metadata contract"):
        _run(evidence)


def test_rejects_source_and_gpu_metadata_tamper(evidence: dict[str, Path]) -> None:
    manifest = json.loads(evidence["manifest"].read_text())
    manifest["source_provenance"]["source_tree_sha256"] = "a" * 64
    manifest["gpu_uuid"] = "GPU-wrong"
    evidence["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Oracle manifest|source identities"):
        _run(evidence)


def test_rejects_missing_untimed_debug_markers(evidence: dict[str, Path]) -> None:
    report = json.loads(evidence["report"].read_text())
    report["measured_timing_valid"] = True
    evidence["report"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="untimed production-repair contract"):
        _run(evidence)


def test_rgb_specialization_difference_is_reported_but_does_not_fail(evidence: dict[str, Path]) -> None:
    with np.load(evidence["root"] / "repair.rgb-only.npz", allow_pickle=False) as archive:
        rgb = np.asarray(archive["rgb"]).copy()
    rgb[0, 0, 0, 0] = np.nextafter(rgb[0, 0, 0, 0], np.float32(1.0))
    np.savez_compressed(evidence["root"] / "repair.rgb-only.npz", rgb=rgb)

    result = _run(evidence)

    drift = result["repeatability"]["cross_specialization"]["primary_full_rgb_vs_rgb_only"]["rgb"]
    assert result["oracle_comparisons"]["primary"]["rgb_only"]["pass"] is True
    assert drift["bitwise_equal"] is False
    assert drift["bit_mismatch_values"] == 1
    assert result["pass"] is True


def test_rgb_specialization_uses_float32_bits(evidence: dict[str, Path]) -> None:
    with np.load(evidence["root"] / "repair.rgb-only.npz", allow_pickle=False) as archive:
        rgb = np.asarray(archive["rgb"]).copy()
    rgb[0, 0, 1, 0] = np.float32(-0.0)
    np.savez_compressed(evidence["root"] / "repair.rgb-only.npz", rgb=rgb)

    result = _run(evidence)

    drift = result["repeatability"]["cross_specialization"]["primary_full_rgb_vs_rgb_only"]["rgb"]
    assert result["oracle_comparisons"]["primary"]["rgb_only"]["pass"] is True
    assert drift["max_absolute_error"] == 0.0
    assert drift["bit_mismatch_values"] == 1
    assert drift["bitwise_equal"] is False
    assert result["pass"] is True


def test_same_specialization_drift_and_semantic_change_are_diagnostic(evidence: dict[str, Path]) -> None:
    values = _arrays()
    values["rgb"][0, 0, 1, 0] = np.float32(1.0e-3)
    values["alpha"][0, 0, 0, 0] += np.float32(1.0e-3)
    values["depth"][0, 0, 0, 0] += np.float32(1.0e-3)
    values["semantic"][0, 0, 1, 0] = 7
    _write_candidate(evidence["root"], values, prefix="repeat-repair")

    result = _run(evidence)

    repeat_drift = result["repeatability"]["same_specialization_repeats"]["full_sensor_primary_vs_repeat"]
    assert repeat_drift["rgb"]["bit_mismatch_values"] == 1
    assert repeat_drift["alpha"]["max_absolute_error"] == pytest.approx(1.0e-3, rel=2.0e-5)
    assert repeat_drift["depth"]["max_absolute_error"] == pytest.approx(1.0e-3, rel=2.0e-4)
    assert repeat_drift["semantic_id"]["mismatch_values"] == 1
    assert result["pass"] is True


def test_repeat_oracle_failure_fails_verdict(evidence: dict[str, Path]) -> None:
    values = _arrays()
    values["rgb"][0] = np.float32(1.0)
    _write_candidate(evidence["root"], values, prefix="repeat-repair")

    result = _run(evidence)

    assert result["oracle_comparisons"]["repeat"]["full_sensor"]["pass"] is False
    assert result["oracle_comparisons"]["repeat"]["rgb_only"]["pass"] is False
    assert result["pass"] is False


def test_rejects_repeat_contract_mismatch(evidence: dict[str, Path]) -> None:
    report = json.loads(evidence["repeat_report"].read_text())
    report["source_provenance"]["source_tree_sha256"] = "a" * 64
    evidence["repeat_report"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="different source/GPU/render contracts"):
        _run(evidence)


def test_artifact_tree_relocates(evidence: dict[str, Path], tmp_path: Path) -> None:
    relocated_root = tmp_path / "relocated"
    shutil.copytree(evidence["root"], relocated_root)
    relocated = {
        "root": relocated_root,
        "report": relocated_root / evidence["report"].name,
        "repeat_report": relocated_root / evidence["repeat_report"].name,
        "index": relocated_root / evidence["index"].name,
        "lock": relocated_root / evidence["lock"].name,
        "oracle": relocated_root / evidence["oracle"].name,
        "manifest": relocated_root / evidence["manifest"].name,
        "bundle": relocated_root / evidence["bundle"].name,
    }

    result = _run(relocated, output_name="relocated-comparison.json")

    assert result["pass"] is True
    assert result["input_artifacts"]["oracle"]["path"] == "oracle.npz"
