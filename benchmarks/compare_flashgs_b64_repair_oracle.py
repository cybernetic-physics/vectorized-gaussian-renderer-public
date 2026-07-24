#!/usr/bin/env python3
"""Gate repaired FlashGS B64 tensors against a fresh pinned gsplat oracle.

This is an untimed correctness comparison.  It intentionally validates the
producer evidence before comparing every pixel of both output contracts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    GSPLAT_BUILD_ATTESTATION_SCHEMA,
    GSPLAT_ORACLE_SCHEMA,
    HEADLINE_COMPUTE_CAPABILITY,
    HEADLINE_GPU_NAME,
    HEADLINE_TORCH_CUDA_ARCH_LIST,
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PINNED_GSPLAT_COMMIT,
    PINNED_GSPLAT_PATCH_SHA256,
    PINNED_GSPLAT_PATCHED_UTILS_SHA256,
    artifact_record,
    is_nvidia_gpu_uuid,
    same_artifact,
    source_identity,
    verify_gsplat_oracle_support_evidence,
)
from isaacsim_gaussian_renderer.evaluation.matched_semantics import (  # noqa: E402
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
)
from isaacsim_gaussian_renderer.fidelity.camera_bundle import (  # noqa: E402
    load_camera_bundle,
)
from isaacsim_gaussian_renderer.fidelity.metrics import (  # noqa: E402
    FIDELITY_THRESHOLDS,
    _rgb_ssim,
)
from isaacsim_gaussian_renderer.flashgs_repair import (  # noqa: E402
    FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA,
    load_verified_b64_repair_raw_outputs,
)

SCHEMA_VERSION = "flashgs-b64-repair-gsplat-all-pixel-v2"
HOME_SCAN_COUNT = 21_497_908
HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
EXPECTED_CAMERAS = np.arange(64, dtype=np.int64)
EXPECTED_STEPS = np.full(64, 107, dtype=np.int64)
EXPECTED_SELECTION = [[107, camera] for camera in range(64)]
EXPECTED_RGB_SHAPE = (64, 128, 128, 3)
EXPECTED_SCALAR_SHAPE = (64, 128, 128, 1)
THRESHOLDS = {
    "rgb_psnr_db_min": FIDELITY_THRESHOLDS["rgb_psnr_db"],
    "rgb_ssim_min": FIDELITY_THRESHOLDS["rgb_ssim"],
    "alpha_mae_max": FIDELITY_THRESHOLDS["alpha_mae"],
    "depth_relative_error_mean_max": FIDELITY_THRESHOLDS["depth_rel_error"],
    "semantic_foreground_agreement_min": FIDELITY_THRESHOLDS["semantic_foreground_agreement"],
    # This additional all-pixel semantic gate is at least as strict as the
    # repository's foreground-only matched gate.
    "semantic_all_pixel_agreement_min": FIDELITY_THRESHOLDS["semantic_foreground_agreement"],
    "depth_finiteness_mismatch_max": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Untimed, all-pixel B64 comparison of repaired FlashGS outputs against a fresh pinned-gsplat oracle."
        )
    )
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument(
        "--repeat-repair-report",
        type=Path,
        required=True,
        help="A distinct second B64 repair-verification report produced by a separate render run.",
    )
    parser.add_argument("--diagnosis-index", type=Path, required=True)
    parser.add_argument("--diagnosis-lock", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Portable root containing the repair, diagnosis, oracle, and output evidence.",
    )
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path)
    parser.add_argument("--oracle-camera-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object.")
    return value


def _resolved_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Artifact root is not a directory: {root}.")
    return root


def _inside_root(path: str | Path, *, root: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} lies outside --artifact-root: {resolved}.")
    return resolved


def _portable_artifact(path: str | Path, *, root: Path, label: str) -> dict[str, Any]:
    resolved = _inside_root(path, root=root, label=label)
    record = artifact_record(resolved)
    relative = resolved.relative_to(root)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} has an unsafe relative path.")
    record["path"] = relative.as_posix()
    return record


def _identity_only(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    byte_count = record.get("bytes")
    sha256 = record.get("sha256")
    if not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError(f"{label} has no valid byte count.")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"{label} has no valid SHA-256.")
    try:
        int(sha256, 16)
    except ValueError as error:
        raise ValueError(f"{label} has no valid SHA-256.") from error
    return {"bytes": byte_count, "sha256": sha256}


def _valid_source_identity(provenance: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    identity = source_identity(dict(provenance))
    for field in ("manifest_sha256", "source_tree_sha256", "diff_sha256"):
        value = identity.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{label} has no valid {field}.")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{label} has no valid {field}.") from error
    head = identity.get("head")
    if not isinstance(head, str) or len(head) != 40:
        raise ValueError(f"{label} has no full Git commit.")
    try:
        int(head, 16)
    except ValueError as error:
        raise ValueError(f"{label} has no full Git commit.") from error
    if identity.get("dirty") is not False:
        raise ValueError(f"{label} is not a clean source identity.")
    return identity


def _validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    tool_integrity = _mapping(report.get("tool_integrity"), "repair tool_integrity")
    integrity_checks = _mapping(tool_integrity.get("checks"), "repair integrity checks")
    source_audit = _mapping(
        _mapping(report.get("production_adapter"), "repair production_adapter").get("source_digest_and_repair_audit"),
        "repair source audit",
    )
    camera = _mapping(report.get("camera"), "repair camera")
    equation = _mapping(report.get("equation"), "repair equation")
    scene = _mapping(report.get("scene"), "repair scene")
    environment = _mapping(report.get("environment"), "repair environment")
    created_at = report.get("created_at")
    if (
        report.get("schema_version") != FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA
        or report.get("pass") is not True
        or report.get("debug_only") is not True
        or report.get("measured_timing_valid") is not False
        or tool_integrity.get("pass") is not True
        or not integrity_checks
        or not all(value is True for value in integrity_checks.values())
        or source_audit.get("pass") is not True
        or not isinstance(created_at, str)
        or not created_at
    ):
        raise ValueError("Repair report did not pass the untimed production-repair contract.")
    trajectory_id = camera.get("trajectory_id")
    if (
        not isinstance(trajectory_id, str)
        or len(trajectory_id) != 64
        or camera.get("step") != 107
        or equation.get("gaussian_support_sigma") != MATCHED_GAUSSIAN_SUPPORT_SIGMA
        or equation.get("covariance_epsilon") != 0.0
        or equation.get("alpha_threshold") != 1.0 / 255.0
        or equation.get("alpha_cap") != 0.99
        or equation.get("transmittance_threshold") != 1.0e-4
        or equation.get("semantic_min_alpha") != 0.01
        or equation.get("semantic_topology") != REPRESENTATIVE_SEMANTIC_TOPOLOGY
    ):
        raise ValueError("Repair camera/equation contract is not the frozen B64 contract.")
    if (
        scene.get("sha256") != HOME_SCAN_SHA256
        or scene.get("gaussian_count") != HOME_SCAN_COUNT
        or scene.get("semantic_topology") != REPRESENTATIVE_SEMANTIC_TOPOLOGY
        or scene.get("canonical_precision") != "float32"
    ):
        raise ValueError("Repair scene is not the canonical full Home Scan.")
    gpu_uuid = environment.get("gpu_uuid")
    gpu_name = environment.get("gpu_name")
    if not is_nvidia_gpu_uuid(gpu_uuid) or gpu_name != HEADLINE_GPU_NAME:
        raise ValueError("Repair report is not from one identified headline L4.")
    return {
        "trajectory_id": trajectory_id,
        "semantic_topology": equation["semantic_topology"],
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "created_at": created_at,
        "source_identity": _valid_source_identity(
            _mapping(report.get("source_provenance"), "repair source_provenance"),
            label="repair source provenance",
        ),
    }


def _validate_loaded_candidate(loaded: Mapping[str, Any]) -> None:
    for label, required in (
        ("full", ("rgb", "alpha", "depth", "semantic_id")),
        ("rgb_only", ("rgb",)),
    ):
        item = _mapping(loaded.get(label), f"loaded {label}")
        arrays = _mapping(item.get("arrays"), f"loaded {label} arrays")
        if (
            item.get("camera_indices") != list(range(64))
            or item.get("step") != 107
            or not isinstance(item.get("trajectory_id"), str)
            or item.get("semantic_topology") != REPRESENTATIVE_SEMANTIC_TOPOLOGY
        ):
            raise ValueError(f"Loaded {label} B64 metadata differs.")
        for name in required:
            if name not in arrays or not isinstance(arrays[name], np.ndarray):
                raise ValueError(f"Loaded {label} lacks ndarray {name}.")
        rgb = arrays["rgb"]
        if rgb.shape != EXPECTED_RGB_SHAPE or rgb.dtype != np.dtype(np.float32) or not np.isfinite(rgb).all():
            raise ValueError(f"Loaded {label} RGB tensor contract differs.")
        if label == "full":
            alpha = arrays["alpha"]
            depth = arrays["depth"]
            semantic = arrays["semantic_id"]
            if (
                alpha.shape != EXPECTED_SCALAR_SHAPE
                or alpha.dtype != np.dtype(np.float32)
                or not np.isfinite(alpha).all()
                or np.any((alpha < 0.0) | (alpha > 1.0))
                or depth.shape != EXPECTED_SCALAR_SHAPE
                or depth.dtype != np.dtype(np.float32)
                or not np.logical_or(np.isfinite(depth), np.isposinf(depth)).all()
                or semantic.shape != EXPECTED_SCALAR_SHAPE
                or semantic.dtype != np.dtype(np.int64)
            ):
                raise ValueError("Loaded full-sensor tensor contract differs.")


def _load_oracle(path: Path) -> dict[str, Any]:
    required = {
        "rgb",
        "alpha",
        "depth",
        "semantic",
        "valid_depth",
        "color_space",
        "background",
        "camera_bundle_id",
        "camera_indices",
        "steps",
        "trajectory_id",
        "semantic_topology",
        "gaussian_support_sigma",
    }
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError("Oracle NPZ lacks required arrays or metadata.")
        values = {name: np.asarray(archive[name]).copy() for name in required}
    rgb = values["rgb"]
    alpha = values["alpha"]
    depth = values["depth"]
    semantic = values["semantic"]
    valid_depth = values["valid_depth"]
    cameras = values["camera_indices"]
    steps = values["steps"]
    background = values["background"]
    for name in (
        "color_space",
        "camera_bundle_id",
        "trajectory_id",
        "semantic_topology",
        "gaussian_support_sigma",
    ):
        if values[name].shape != ():
            raise ValueError(f"Oracle {name} metadata is not scalar.")
    if (
        rgb.shape != EXPECTED_RGB_SHAPE
        or rgb.dtype != np.dtype(np.float32)
        or not np.isfinite(rgb).all()
        or alpha.shape != EXPECTED_SCALAR_SHAPE
        or alpha.dtype != np.dtype(np.float32)
        or not np.isfinite(alpha).all()
        or np.any((alpha < 0.0) | (alpha > 1.0))
        or depth.shape != EXPECTED_SCALAR_SHAPE
        or depth.dtype != np.dtype(np.float32)
        or not np.logical_or(np.isfinite(depth), np.isposinf(depth)).all()
        or semantic.shape != EXPECTED_SCALAR_SHAPE
        or semantic.dtype != np.dtype(np.int64)
        or valid_depth.shape != EXPECTED_SCALAR_SHAPE
        or valid_depth.dtype != np.dtype(np.bool_)
        or not np.array_equal(valid_depth, alpha > 0.01)
        or cameras.shape != (64,)
        or cameras.dtype != np.dtype(np.int64)
        or not np.array_equal(cameras, EXPECTED_CAMERAS)
        or steps.shape != (64,)
        or steps.dtype != np.dtype(np.int64)
        or not np.array_equal(steps, EXPECTED_STEPS)
        or background.shape != (3,)
        or background.dtype != np.dtype(np.float32)
        or not np.array_equal(background, np.zeros(3, dtype=np.float32))
        or str(values["color_space"].item()) != "linear_rgb"
        or str(values["semantic_topology"].item()) != REPRESENTATIVE_SEMANTIC_TOPOLOGY
        or not np.isclose(
            float(values["gaussian_support_sigma"].item()),
            MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise ValueError("Oracle tensor or metadata contract differs.")
    return {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "semantic": semantic,
        "valid_depth": valid_depth,
        "camera_indices": cameras,
        "steps": steps,
        "trajectory_id": str(values["trajectory_id"].item()),
        "semantic_topology": str(values["semantic_topology"].item()),
        "camera_bundle_id": str(values["camera_bundle_id"].item()),
    }


def _validate_oracle_manifest(
    manifest: Mapping[str, Any],
    *,
    oracle_record: Mapping[str, Any],
    camera_bundle_record: Mapping[str, Any],
    oracle: Mapping[str, Any],
    repair_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        manifest.get("schema_version") != GSPLAT_ORACLE_SCHEMA
        or manifest.get("pass") is not True
        or manifest.get("output_sha256") != oracle_record.get("sha256")
        or not same_artifact(
            _mapping(manifest.get("camera_bundle_artifact"), "oracle camera_bundle_artifact"),
            dict(camera_bundle_record),
        )
        or manifest.get("camera_bundle_id") != oracle.get("camera_bundle_id")
        or manifest.get("camera_indices") != list(range(64))
        or manifest.get("steps") != [107] * 64
        or manifest.get("selection_pairs") != EXPECTED_SELECTION
        or manifest.get("selection_profile") != "diagnostic-single-step"
        or manifest.get("trajectory_id") != repair_contract.get("trajectory_id")
        or manifest.get("trajectory_id") != oracle.get("trajectory_id")
        or manifest.get("scene_sha256") != HOME_SCAN_SHA256
        or manifest.get("gaussian_count") != HOME_SCAN_COUNT
        or manifest.get("semantic_topology") != REPRESENTATIVE_SEMANTIC_TOPOLOGY
        or manifest.get("semantic_topology") != oracle.get("semantic_topology")
        or manifest.get("gaussian_support_sigma") != MATCHED_GAUSSIAN_SUPPORT_SIGMA
        or manifest.get("projection_contract") != MATCHED_PROJECTION_RULES
        or manifest.get("gsplat_commit") != PINNED_GSPLAT_COMMIT
        or _mapping(
            manifest.get("gsplat_compatibility_patch"),
            "oracle gsplat_compatibility_patch",
        ).get("sha256")
        != PINNED_GSPLAT_PATCH_SHA256
        or _mapping(
            manifest.get("gsplat_compatibility_patch"),
            "oracle gsplat_compatibility_patch",
        ).get("patched_utils_sha256")
        != PINNED_GSPLAT_PATCHED_UTILS_SHA256
        or manifest.get("gpu_uuid") != repair_contract.get("gpu_uuid")
        or manifest.get("gpu") != repair_contract.get("gpu_name")
        or manifest.get("compute_capability") != list(HEADLINE_COMPUTE_CAPABILITY)
        or manifest.get("torch_cuda_arch_list")
        != HEADLINE_TORCH_CUDA_ARCH_LIST
    ):
        raise ValueError("Oracle manifest is not the exact fresh B64 matched contract.")
    oracle_source_identity = _valid_source_identity(
        _mapping(manifest.get("source_provenance"), "oracle source_provenance"),
        label="oracle source provenance",
    )
    if oracle_source_identity != repair_contract.get("source_identity"):
        raise ValueError("Repair and oracle source identities differ.")
    attestation = verify_gsplat_oracle_support_evidence(dict(manifest))
    if (
        attestation.get("schema_version") != GSPLAT_BUILD_ATTESTATION_SCHEMA
        or _mapping(attestation.get("build_environment"), "gsplat build environment").get("TORCH_CUDA_ARCH_LIST")
        != HEADLINE_TORCH_CUDA_ARCH_LIST
    ):
        raise ValueError("Oracle gsplat build attestation does not match the headline GPU architecture.")
    return {
        "source_identity": oracle_source_identity,
        "gsplat_build_attestation": _identity_only(
            _mapping(manifest.get("gsplat_build_attestation"), "gsplat build attestation"),
            label="gsplat build attestation",
        ),
        "gsplat_native_extension": _identity_only(
            _mapping(manifest.get("gsplat_native_extension"), "gsplat native extension"),
            label="gsplat native extension",
        ),
    }


def _validate_camera_bundle(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> None:
    bundle = load_camera_bundle(path)
    expected_ids = [f"t107-b{camera:04d}" for camera in range(64)]
    if (
        bundle.bundle_id != manifest.get("camera_bundle_id")
        or bundle.bundle_id != oracle.get("camera_bundle_id")
        or bundle.width != 128
        or bundle.height != 128
        or len(bundle.cameras) != 64
        or [camera.view_id for camera in bundle.cameras] != expected_ids
        or any(camera.scene_id != 404 for camera in bundle.cameras)
        or bundle.background != (0.0, 0.0, 0.0)
        or bundle.color_space != "linear_rgb"
        or bundle.scene_checksum != HOME_SCAN_SHA256
    ):
        raise ValueError("Oracle camera bundle differs from the exact B64/step-107 contract.")


def _rgb_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    clipped_reference = np.clip(reference, 0.0, 1.0)
    clipped_candidate = np.clip(candidate, 0.0, 1.0)
    clipped_difference = clipped_candidate.astype(np.float64) - clipped_reference.astype(np.float64)
    mse = float(np.mean(clipped_difference**2))
    psnr = math.inf if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    ssim = (
        1.0
        if np.array_equal(reference, candidate)
        else _rgb_ssim(
            clipped_reference,
            clipped_candidate,
        )
    )
    return {
        "psnr_db": psnr,
        "psnr_and_ssim_input": "linear RGB clipped to [0,1], matching the repository fidelity gate",
        "ssim": float(ssim),
        "mae": float(np.mean(np.abs(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mismatched_values": int(np.count_nonzero(reference != candidate)),
    }


def _compare_full(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> dict[str, Any]:
    per_view: list[dict[str, Any]] = []
    for view in range(64):
        rgb = _rgb_metrics(reference["rgb"][view], candidate["rgb"][view])
        alpha_difference = np.abs(
            candidate["alpha"][view].astype(np.float64) - reference["alpha"][view].astype(np.float64)
        )
        ref_depth = reference["depth"][view, ..., 0]
        cand_depth = candidate["depth"][view, ..., 0]
        ref_finite = np.isfinite(ref_depth)
        cand_finite = np.isfinite(cand_depth)
        finiteness_mismatches = int(np.count_nonzero(ref_finite != cand_finite))
        valid_depth = reference["valid_depth"][view, ..., 0]
        missing_valid_depth = int(np.count_nonzero(valid_depth & ~cand_finite))
        comparable_depth = valid_depth & cand_finite
        if np.any(comparable_depth):
            depth_relative = np.abs(
                cand_depth[comparable_depth].astype(np.float64) - ref_depth[comparable_depth].astype(np.float64)
            ) / np.maximum(np.abs(ref_depth[comparable_depth].astype(np.float64)), 1.0e-8)
            depth_relative_mean: float | None = float(np.mean(depth_relative))
            depth_relative_max: float | None = float(np.max(depth_relative))
        else:
            depth_relative_mean = None
            depth_relative_max = None
        ref_semantic = reference["semantic"][view, ..., 0]
        cand_semantic = candidate["semantic_id"][view, ..., 0]
        semantic_equal = ref_semantic == cand_semantic
        semantic_all = float(np.mean(semantic_equal))
        foreground = (reference["alpha"][view, ..., 0] >= 0.01) | (candidate["alpha"][view, ..., 0] >= 0.01)
        semantic_foreground = float(np.mean(semantic_equal[foreground])) if np.any(foreground) else 1.0
        checks = {
            "rgb_psnr": rgb["psnr_db"] >= THRESHOLDS["rgb_psnr_db_min"],
            "rgb_ssim": rgb["ssim"] >= THRESHOLDS["rgb_ssim_min"],
            "alpha_mae": float(np.mean(alpha_difference)) <= THRESHOLDS["alpha_mae_max"],
            "depth_relative_error": depth_relative_mean is not None
            and depth_relative_mean <= THRESHOLDS["depth_relative_error_mean_max"],
            "depth_finiteness": finiteness_mismatches == 0 and missing_valid_depth == 0,
            "semantic_foreground": semantic_foreground >= THRESHOLDS["semantic_foreground_agreement_min"],
            "semantic_all_pixels": semantic_all >= THRESHOLDS["semantic_all_pixel_agreement_min"],
        }
        per_view.append(
            {
                "view_index": view,
                "camera_index": view,
                "step": 107,
                "pixel_count": 128 * 128,
                "rgb": rgb,
                "alpha": {
                    "mae": float(np.mean(alpha_difference)),
                    "max_absolute_error": float(np.max(alpha_difference)),
                    "mismatched_values": int(np.count_nonzero(reference["alpha"][view] != candidate["alpha"][view])),
                },
                "depth": {
                    "relative_error_mean": depth_relative_mean,
                    "relative_error_max": depth_relative_max,
                    "reference_valid_pixels": int(np.count_nonzero(valid_depth)),
                    "compared_valid_pixels": int(np.count_nonzero(comparable_depth)),
                    "missing_valid_pixels": missing_valid_depth,
                    "finiteness_mismatch_pixels": finiteness_mismatches,
                },
                "semantic": {
                    "all_pixel_agreement": semantic_all,
                    "foreground_agreement": semantic_foreground,
                    "foreground_pixels": int(np.count_nonzero(foreground)),
                    "mismatch_pixels": int(np.count_nonzero(~semantic_equal)),
                },
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    depth_relative_means = [
        float(row["depth"]["relative_error_mean"])
        for row in per_view
        if row["depth"]["relative_error_mean"] is not None
    ]
    aggregate = {
        "view_count": len(per_view),
        "pixel_count": 64 * 128 * 128,
        "rgb_psnr_db_worst": min(float(row["rgb"]["psnr_db"]) for row in per_view),
        "rgb_ssim_worst": min(float(row["rgb"]["ssim"]) for row in per_view),
        "rgb_max_absolute_error": max(float(row["rgb"]["max_absolute_error"]) for row in per_view),
        "alpha_mae_worst": max(float(row["alpha"]["mae"]) for row in per_view),
        "alpha_max_absolute_error": max(float(row["alpha"]["max_absolute_error"]) for row in per_view),
        "depth_relative_error_mean_worst": max(depth_relative_means) if depth_relative_means else None,
        "depth_finiteness_mismatch_pixels": sum(int(row["depth"]["finiteness_mismatch_pixels"]) for row in per_view),
        "semantic_all_pixel_agreement_worst": min(float(row["semantic"]["all_pixel_agreement"]) for row in per_view),
        "semantic_foreground_agreement_worst": min(float(row["semantic"]["foreground_agreement"]) for row in per_view),
        "semantic_mismatch_pixels": sum(int(row["semantic"]["mismatch_pixels"]) for row in per_view),
        "passed_views": sum(row["pass"] is True for row in per_view),
        "pass": all(row["pass"] is True for row in per_view),
    }
    return {"per_view": per_view, "aggregate": aggregate, "pass": aggregate["pass"]}


def _compare_rgb(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    per_view = []
    for view in range(64):
        metrics = _rgb_metrics(reference[view], candidate[view])
        checks = {
            "rgb_psnr": metrics["psnr_db"] >= THRESHOLDS["rgb_psnr_db_min"],
            "rgb_ssim": metrics["ssim"] >= THRESHOLDS["rgb_ssim_min"],
        }
        per_view.append(
            {
                "view_index": view,
                "camera_index": view,
                "step": 107,
                "pixel_count": 128 * 128,
                "rgb": metrics,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    aggregate = {
        "view_count": 64,
        "pixel_count": 64 * 128 * 128,
        "rgb_psnr_db_worst": min(float(row["rgb"]["psnr_db"]) for row in per_view),
        "rgb_ssim_worst": min(float(row["rgb"]["ssim"]) for row in per_view),
        "rgb_max_absolute_error": max(float(row["rgb"]["max_absolute_error"]) for row in per_view),
        "passed_views": sum(row["pass"] is True for row in per_view),
        "pass": all(row["pass"] is True for row in per_view),
    }
    return {"per_view": per_view, "aggregate": aggregate, "pass": aggregate["pass"]}


def _tensor_contract(array: np.ndarray) -> dict[str, Any]:
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _float32_bit_mismatch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.dtype != np.dtype(np.float32) or right.dtype != np.dtype(np.float32):
        raise ValueError("Float drift comparison requires two float32 arrays.")
    if left.shape != right.shape:
        raise ValueError("Float drift comparison requires equal shapes.")
    left_bits = np.ascontiguousarray(left).view(np.uint32)
    right_bits = np.ascontiguousarray(right).view(np.uint32)
    return left_bits != right_bits


def _float32_drift(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    bit_mismatch = _float32_bit_mismatch(left, right)
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    finiteness_mismatch = left_finite != right_finite
    comparable = left_finite & right_finite
    if np.any(comparable):
        absolute = np.abs(left[comparable].astype(np.float64) - right[comparable].astype(np.float64))
        maximum: float | None = float(np.max(absolute))
        mean: float | None = float(np.mean(absolute))
    else:
        maximum = None
        mean = None
    return {
        "comparison_representation": "IEEE-754 float32 bit patterns viewed as uint32",
        "value_count": int(left.size),
        "bitwise_equal": bool(not np.any(bit_mismatch)),
        "bit_mismatch_values": int(np.count_nonzero(bit_mismatch)),
        "numeric_mismatch_values": int(np.count_nonzero(left != right)),
        "finiteness_mismatch_values": int(np.count_nonzero(finiteness_mismatch)),
        "finite_compared_values": int(np.count_nonzero(comparable)),
        "max_absolute_error": maximum,
        "mean_absolute_error": mean,
    }


def _semantic_drift(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.dtype != np.dtype(np.int64) or right.dtype != np.dtype(np.int64):
        raise ValueError("Semantic drift comparison requires two int64 arrays.")
    if left.shape != right.shape:
        raise ValueError("Semantic drift comparison requires equal shapes.")
    mismatch = left != right
    return {
        "value_count": int(left.size),
        "exactly_equal": bool(not np.any(mismatch)),
        "mismatch_values": int(np.count_nonzero(mismatch)),
        "agreement": float(np.mean(~mismatch)),
    }


def _full_output_drift(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "acceptance_role": "diagnostic-only-not-an-acceptance-gate",
        "rgb": _float32_drift(left["rgb"], right["rgb"]),
        "alpha": _float32_drift(left["alpha"], right["alpha"]),
        "depth": _float32_drift(left["depth"], right["depth"]),
        "semantic_id": _semantic_drift(left["semantic_id"], right["semantic_id"]),
    }


def _rgb_output_drift(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    return {
        "acceptance_role": "diagnostic-only-not-an-acceptance-gate",
        "rgb": _float32_drift(left, right),
        "non_rgb_outputs": {
            "available": False,
            "reason": "The RGB-only specialization does not produce alpha, depth, or semantic ID.",
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _repair_raw_paths(report: Mapping[str, Any], *, report_path: Path) -> dict[str, Path]:
    raw_records = _mapping(report.get("post_fix_raw_outputs"), "post_fix_raw_outputs")
    raw_paths: dict[str, Path] = {}
    for label in ("full", "rgb_only"):
        record = _mapping(raw_records.get(label), f"raw {label}")
        raw_value = record.get("path")
        if not isinstance(raw_value, str) or not raw_value:
            raise ValueError(f"Raw {label} report path is missing.")
        raw_relative = Path(raw_value)
        if raw_relative.is_absolute() or ".." in raw_relative.parts:
            raise ValueError(f"Raw {label} report path is unsafe.")
        raw_paths[label] = (report_path.parent / raw_relative).resolve()
    return raw_paths


def _load_repair_evidence(
    report_path: Path,
    *,
    diagnosis_index: Path,
    diagnosis_lock: Path,
    artifact_root: Path,
    label: str,
) -> dict[str, Any]:
    loaded = load_verified_b64_repair_raw_outputs(
        report_path,
        diagnosis_index=diagnosis_index,
        diagnosis_lock=diagnosis_lock,
        artifact_root=artifact_root,
    )
    if not same_artifact(
        _mapping(loaded.get("report"), f"loaded {label} repair report"),
        artifact_record(report_path),
    ) or not same_artifact(
        _mapping(loaded.get("diagnosis_index"), f"loaded {label} diagnosis index"),
        artifact_record(diagnosis_index),
    ):
        raise ValueError(f"{label} raw-output loader returned different repair/diagnosis inputs.")
    _validate_loaded_candidate(loaded)
    report = _mapping(
        json.loads(report_path.read_text(encoding="utf-8")),
        f"{label} repair report",
    )
    contract = _validate_report(report)
    if any(
        loaded[output_label].get("trajectory_id") != contract["trajectory_id"] for output_label in ("full", "rgb_only")
    ):
        raise ValueError(f"{label} B64 tensors and repair report have different trajectories.")
    return {
        "loaded": loaded,
        "report": report,
        "contract": contract,
        "raw_paths": _repair_raw_paths(report, report_path=report_path),
    }


def compare_b64_repair_oracle(
    *,
    repair_report: str | Path,
    repeat_repair_report: str | Path,
    diagnosis_index: str | Path,
    diagnosis_lock: str | Path,
    artifact_root: str | Path,
    oracle_path: str | Path,
    output_path: str | Path,
    oracle_manifest_path: str | Path | None = None,
    oracle_camera_bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate producer evidence and persist an all-pixel correctness verdict."""

    root = _resolved_root(artifact_root)
    repair_path = _inside_root(repair_report, root=root, label="repair report")
    repeat_repair_path = _inside_root(
        repeat_repair_report,
        root=root,
        label="repeat repair report",
    )
    if repair_path == repeat_repair_path:
        raise ValueError("Primary and repeat repair reports must be distinct files.")
    index_path = _inside_root(diagnosis_index, root=root, label="diagnosis index")
    lock_path = _inside_root(diagnosis_lock, root=root, label="diagnosis lock")
    resolved_oracle = _inside_root(oracle_path, root=root, label="oracle")
    manifest_path = _inside_root(
        oracle_manifest_path or resolved_oracle.with_suffix(".manifest.json"),
        root=root,
        label="oracle manifest",
    )
    camera_bundle_path = _inside_root(
        oracle_camera_bundle_path or resolved_oracle.with_suffix(".camera-bundle.json"),
        root=root,
        label="oracle camera bundle",
    )
    destination = _inside_root(output_path, root=root, label="output report")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite correctness evidence: {destination}.")

    primary = _load_repair_evidence(
        repair_path,
        diagnosis_index=index_path,
        diagnosis_lock=lock_path,
        artifact_root=root,
        label="primary",
    )
    repeat = _load_repair_evidence(
        repeat_repair_path,
        diagnosis_index=index_path,
        diagnosis_lock=lock_path,
        artifact_root=root,
        label="repeat",
    )
    primary_report_record = artifact_record(repair_path)
    repeat_report_record = artifact_record(repeat_repair_path)
    binding_fields = (
        "trajectory_id",
        "semantic_topology",
        "gpu_uuid",
        "gpu_name",
        "source_identity",
    )
    if any(primary["contract"][field] != repeat["contract"][field] for field in binding_fields):
        raise ValueError("Primary and repeat repair reports have different source/GPU/render contracts.")
    if (
        primary_report_record["sha256"] == repeat_report_record["sha256"]
        or primary["contract"]["created_at"] == repeat["contract"]["created_at"]
    ):
        raise ValueError("Primary and repeat reports do not prove distinct render runs.")
    if set(primary["raw_paths"].values()) & set(repeat["raw_paths"].values()):
        raise ValueError("Primary and repeat reports reuse a raw output path.")
    repair_contract = primary["contract"]

    oracle_record = artifact_record(resolved_oracle)
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "oracle manifest",
    )
    camera_bundle_record = artifact_record(camera_bundle_path)
    oracle = _load_oracle(resolved_oracle)
    oracle_bindings = _validate_oracle_manifest(
        manifest,
        oracle_record=oracle_record,
        camera_bundle_record=camera_bundle_record,
        oracle=oracle,
        repair_contract=repair_contract,
    )
    _validate_camera_bundle(
        camera_bundle_path,
        manifest=manifest,
        oracle=oracle,
    )

    primary_full = primary["loaded"]["full"]["arrays"]
    primary_rgb = primary["loaded"]["rgb_only"]["arrays"]["rgb"]
    repeat_full = repeat["loaded"]["full"]["arrays"]
    repeat_rgb = repeat["loaded"]["rgb_only"]["arrays"]["rgb"]
    oracle_comparisons = {
        "primary": {
            "full_sensor": _compare_full(oracle, primary_full),
            "rgb_only": _compare_rgb(oracle["rgb"], primary_rgb),
        },
        "repeat": {
            "full_sensor": _compare_full(oracle, repeat_full),
            "rgb_only": _compare_rgb(oracle["rgb"], repeat_rgb),
        },
    }
    required_oracle_passes = {
        f"{run_label}_{output_label}": oracle_comparisons[run_label][output_label]["pass"]
        for run_label in ("primary", "repeat")
        for output_label in ("full_sensor", "rgb_only")
    }
    passed = all(required_oracle_passes.values())

    repeatability = {
        "acceptance_role": "diagnostic-only-not-an-acceptance-gate",
        "bitwise_determinism_expected": False,
        "explanation_status": (
            "empirical repeat observation plus source-grounded mechanism inference; not a direct scheduler trace"
        ),
        "empirical_and_source_grounded_reason": (
            "Independent B64 runs showed comparable same-specialization and cross-specialization drift. "
            "The production preprocess kernel reserves emitted intersection spans with a global atomicAdd; its "
            "64-bit sort key contains tile ID and float depth but no Gaussian-ID tie-breaker. CUB sorts those "
            "keys, so equal tile/depth entries can retain reservation-order differences and alter compositing."
        ),
        "same_specialization_repeats": {
            "full_sensor_primary_vs_repeat": _full_output_drift(primary_full, repeat_full),
            "rgb_only_primary_vs_repeat": _rgb_output_drift(primary_rgb, repeat_rgb),
        },
        "cross_specialization": {
            "primary_full_rgb_vs_rgb_only": _rgb_output_drift(primary_full["rgb"], primary_rgb),
            "repeat_full_rgb_vs_rgb_only": _rgb_output_drift(repeat_full["rgb"], repeat_rgb),
        },
    }

    tool_record = artifact_record(Path(__file__))
    mechanism_source_artifacts = {}
    for relative in (
        "src/isaacsim_gaussian_renderer/native/flashgs/preprocess.cu",
        "src/isaacsim_gaussian_renderer/native/flashgs/sort.cu",
    ):
        source_record = artifact_record(PROJECT_ROOT / relative)
        mechanism_source_artifacts[Path(relative).name] = {
            "repository_path": relative,
            "bytes": source_record["bytes"],
            "sha256": source_record["sha256"],
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "debug_only": True,
        "measured_timing_valid": False,
        "evidence_kind": "untimed-all-pixel-correctness-gate",
        "performance_claim_valid": False,
        "pass": passed,
        "acceptance_policy": {
            "pass_rule": (
                "Pass if and only if primary/full, primary/RGB-only, repeat/full, and repeat/RGB-only each "
                "pass every unchanged per-view oracle threshold; full outputs must also have exact oracle "
                "depth-finiteness masks. Repeat or cross-specialization bitwise equality is not required."
            ),
            "required_oracle_comparisons": required_oracle_passes,
            "all_required_oracle_comparisons_pass": passed,
            "repeatability_drift_is_acceptance_gate": False,
            "thresholds_weakened_for_repeatability": False,
        },
        "checks": {
            "primary_repair_raw_loader_passed": True,
            "repeat_repair_raw_loader_passed": True,
            "distinct_repair_runs": True,
            "oracle_manifest_support_build_and_occupancy_verified": True,
            "same_source_identity_across_both_repairs_and_oracle": True,
            "same_gpu_uuid_across_both_repairs_and_oracle": True,
            "exact_b64_step107_selection": True,
            "same_trajectory_and_semantic_topology": True,
            **required_oracle_passes,
        },
        "contract": {
            "batch": 64,
            "width": 128,
            "height": 128,
            "step": 107,
            "camera_indices": list(range(64)),
            "selection_pairs": EXPECTED_SELECTION,
            "trajectory_id": repair_contract["trajectory_id"],
            "semantic_topology": REPRESENTATIVE_SEMANTIC_TOPOLOGY,
            "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            "gpu_uuid": repair_contract["gpu_uuid"],
            "source_identity": repair_contract["source_identity"],
        },
        "thresholds": THRESHOLDS,
        "tensor_contracts": {
            "primary_full": {
                name: _tensor_contract(primary_full[name]) for name in ("rgb", "alpha", "depth", "semantic_id")
            },
            "primary_rgb_only": {"rgb": _tensor_contract(primary_rgb)},
            "repeat_full": {
                name: _tensor_contract(repeat_full[name]) for name in ("rgb", "alpha", "depth", "semantic_id")
            },
            "repeat_rgb_only": {"rgb": _tensor_contract(repeat_rgb)},
            "oracle": {
                name: _tensor_contract(oracle[name]) for name in ("rgb", "alpha", "depth", "semantic", "valid_depth")
            },
        },
        "oracle_comparisons": oracle_comparisons,
        "repeatability": repeatability,
        "repeatability_mechanism_source_artifacts": mechanism_source_artifacts,
        "input_artifacts": {
            "primary_repair_report": _portable_artifact(repair_path, root=root, label="primary repair report"),
            "primary_repair_full_raw": _portable_artifact(
                primary["raw_paths"]["full"], root=root, label="primary repair full raw"
            ),
            "primary_repair_rgb_only_raw": _portable_artifact(
                primary["raw_paths"]["rgb_only"], root=root, label="primary repair RGB-only raw"
            ),
            "repeat_repair_report": _portable_artifact(repeat_repair_path, root=root, label="repeat repair report"),
            "repeat_repair_full_raw": _portable_artifact(
                repeat["raw_paths"]["full"], root=root, label="repeat repair full raw"
            ),
            "repeat_repair_rgb_only_raw": _portable_artifact(
                repeat["raw_paths"]["rgb_only"], root=root, label="repeat repair RGB-only raw"
            ),
            "diagnosis_index": _portable_artifact(index_path, root=root, label="diagnosis index"),
            "diagnosis_lock": _portable_artifact(lock_path, root=root, label="diagnosis lock"),
            "oracle": _portable_artifact(resolved_oracle, root=root, label="oracle"),
            "oracle_manifest": _portable_artifact(manifest_path, root=root, label="oracle manifest"),
            "oracle_camera_bundle": _portable_artifact(camera_bundle_path, root=root, label="oracle camera bundle"),
            "gsplat_build_attestation": oracle_bindings["gsplat_build_attestation"],
            "gsplat_native_extension": oracle_bindings["gsplat_native_extension"],
            "comparison_tool": {
                "repository_path": "benchmarks/compare_flashgs_b64_repair_oracle.py",
                "bytes": tool_record["bytes"],
                "sha256": tool_record["sha256"],
            },
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    result = compare_b64_repair_oracle(
        repair_report=args.repair_report,
        repeat_repair_report=args.repeat_repair_report,
        diagnosis_index=args.diagnosis_index,
        diagnosis_lock=args.diagnosis_lock,
        artifact_root=args.artifact_root,
        oracle_path=args.oracle,
        oracle_manifest_path=args.oracle_manifest,
        oracle_camera_bundle_path=args.oracle_camera_bundle,
        output_path=args.output,
    )
    print(
        "FLASHGS_B64_REPAIR_GSPLAT_ALL_PIXEL "
        + json.dumps(
            {
                "output": str(args.output),
                "pass": result["pass"],
                "primary_full_views_passed": result["oracle_comparisons"]["primary"]["full_sensor"]["aggregate"][
                    "passed_views"
                ],
                "primary_rgb_only_views_passed": result["oracle_comparisons"]["primary"]["rgb_only"]["aggregate"][
                    "passed_views"
                ],
                "repeat_full_views_passed": result["oracle_comparisons"]["repeat"]["full_sensor"]["aggregate"][
                    "passed_views"
                ],
                "repeat_rgb_only_views_passed": result["oracle_comparisons"]["repeat"]["rgb_only"]["aggregate"][
                    "passed_views"
                ],
            },
            sort_keys=True,
        )
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
