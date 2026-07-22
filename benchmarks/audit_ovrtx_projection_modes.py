"""Fail-closed audit for OVRTX Gaussian projection-mode experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_KEYS = ("rgb", "alpha", "depth", "semantic", "valid_depth")
GEOMETRY_OUTPUT_KEYS = ("alpha", "depth", "semantic", "valid_depth")
NUMERIC_EFFECT_STATS = (
    "max_abs_difference",
    "mean_abs_difference",
    "significant_difference_fraction",
)
CATEGORICAL_EFFECT_STATS = ("significant_difference_fraction",)
NUMERIC_SIGNIFICANCE_FLOOR = 1.0e-6
MAX_ABS_EFFECT_FLOOR = 1.0e-6
MEAN_ABS_EFFECT_FLOOR = 1.0e-10
MIN_EFFECT_ELEMENTS = 4
MIN_FOREGROUND_PIXELS = 4
MODE_PATTERN = re.compile(r'projectionModeHint\s*=\s*"(perspective|tangential)"')
FOCAL_LENGTH_PATTERN = re.compile(r"focalLength\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
MODE_SUB_PATTERN = re.compile(r'(projectionModeHint\s*=\s*)"(?:perspective|tangential)"')
FOCAL_SUB_PATTERN = re.compile(r"(focalLength\s*=\s*)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
SORTING_PATTERN = re.compile(r'sortingModeHint\s*=\s*"(zDepth|distanceToCamera|none)"')
CANDIDATE_METADATA_KEYS = ("color_space", "background", "camera_bundle_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perspective-candidate", type=Path, required=True)
    parser.add_argument("--tangential-candidate", type=Path, required=True)
    parser.add_argument("--perspective-stage", type=Path, required=True)
    parser.add_argument("--tangential-stage", type=Path, required=True)
    parser.add_argument("--perspective-repeat", type=Path)
    parser.add_argument("--tangential-repeat", type=Path)
    parser.add_argument("--perspective-repeat-stage", type=Path)
    parser.add_argument("--tangential-repeat-stage", type=Path)
    parser.add_argument("--perspective-summary", type=Path)
    parser.add_argument("--tangential-summary", type=Path)
    parser.add_argument("--perspective-repeat-summary", type=Path)
    parser.add_argument("--tangential-repeat-summary", type=Path)
    parser.add_argument("--positive-control-candidate", type=Path)
    parser.add_argument("--positive-control-stage", type=Path)
    parser.add_argument("--positive-control-summary", type=Path)
    parser.add_argument("--positive-control-repeat-candidate", type=Path)
    parser.add_argument("--positive-control-repeat-stage", type=Path)
    parser.add_argument("--positive-control-repeat-summary", type=Path)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    parser.add_argument("--expected-control-focal-ratio", type=float, default=1.05)
    parser.add_argument(
        "--require-positive-control",
        action="store_true",
        help=(
            "Require a perspective control with a different authored camera "
            "focal length, valid composed-token readback, and a geometry-AOV "
            "effect above the same-mode repeat-noise envelope."
        ),
    )
    parser.add_argument(
        "--require-activation-proof",
        action="store_true",
        help=(
            "Require runtime readback, same-mode repeat controls, and a "
            "cross-mode effect above the repeat-noise envelope."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authored_modes(path: Path) -> list[str]:
    return sorted(set(MODE_PATTERN.findall(path.read_text(encoding="utf-8"))))


def authored_focal_lengths(path: Path) -> list[float]:
    return sorted(set(float(value) for value in FOCAL_LENGTH_PATTERN.findall(path.read_text(encoding="utf-8"))))


def _comparison_mask(
    array: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    if valid_mask is None:
        return np.ones(array.shape, dtype=np.bool_)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if mask.shape == array.shape:
        return mask
    if array.ndim == mask.ndim + 1 and array.shape[-1] == 1:
        mask = mask[..., None]
    try:
        return np.broadcast_to(mask, array.shape)
    except ValueError as exc:
        raise ValueError(f"Comparison mask shape {mask.shape} cannot cover array shape {array.shape}.") from exc


def compare_array(
    perspective: np.ndarray,
    tangential: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    same_shape = perspective.shape == tangential.shape
    same_dtype = perspective.dtype == tangential.dtype
    bitwise_equal = same_shape and same_dtype and np.array_equal(perspective, tangential, equal_nan=True)
    if not same_shape:
        return {
            "shape": list(perspective.shape),
            "other_shape": list(tangential.shape),
            "dtype": str(perspective.dtype),
            "other_dtype": str(tangential.dtype),
            "same_shape": False,
            "same_dtype": same_dtype,
            "bitwise_equal": False,
            "compared_elements": None,
            "excluded_elements": None,
            "different_elements": None,
            "total_different_elements": None,
            "different_fraction": None,
            "significant_difference_elements": None,
            "significant_difference_fraction": None,
            "nonfinite_mismatch_elements": None,
            "max_abs_difference": None,
            "mean_abs_difference": None,
            "p99_abs_difference": None,
            "rmse": None,
        }

    mask = _comparison_mask(perspective, valid_mask)
    compared_elements = int(np.count_nonzero(mask))
    equality = np.equal(perspective, tangential)
    if np.issubdtype(perspective.dtype, np.floating) and np.issubdtype(tangential.dtype, np.floating):
        equality |= np.isnan(perspective) & np.isnan(tangential)
    difference_mask = ~equality
    total_different_elements = int(np.count_nonzero(difference_mask))
    different_elements = int(np.count_nonzero(difference_mask & mask))
    different_fraction = float(different_elements / compared_elements) if compared_elements else 0.0

    max_abs_difference = None
    mean_abs_difference = None
    p99_abs_difference = None
    rmse = None
    nonfinite_mismatch_elements = 0
    significant_mask = difference_mask & mask
    if np.issubdtype(perspective.dtype, np.number) and np.issubdtype(tangential.dtype, np.number):
        perspective_float = perspective.astype(np.float64)
        tangential_float = tangential.astype(np.float64)
        jointly_finite = mask & np.isfinite(perspective_float) & np.isfinite(tangential_float)
        finite_difference = np.abs(perspective_float[jointly_finite] - tangential_float[jointly_finite])
        max_abs_difference = float(finite_difference.max()) if finite_difference.size else 0.0
        mean_abs_difference = float(finite_difference.mean()) if finite_difference.size else 0.0
        p99_abs_difference = float(np.percentile(finite_difference, 99.0)) if finite_difference.size else 0.0
        rmse = float(np.sqrt(np.mean(np.square(finite_difference)))) if finite_difference.size else 0.0
        nonfinite_mismatch = mask & difference_mask & ~(np.isfinite(perspective_float) & np.isfinite(tangential_float))
        nonfinite_mismatch_elements = int(np.count_nonzero(nonfinite_mismatch))
        significant_mask = nonfinite_mismatch.copy()
        finite_significant = np.zeros(perspective.shape, dtype=np.bool_)
        if finite_difference.size:
            finite_significant[jointly_finite] = finite_difference > NUMERIC_SIGNIFICANCE_FLOOR
        significant_mask |= finite_significant

    significant_difference_elements = int(np.count_nonzero(significant_mask))
    significant_difference_fraction = (
        float(significant_difference_elements / compared_elements) if compared_elements else 0.0
    )
    return {
        "shape": list(perspective.shape),
        "dtype": str(perspective.dtype),
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "bitwise_equal": bitwise_equal,
        "compared_elements": compared_elements,
        "excluded_elements": int(perspective.size - compared_elements),
        "different_elements": different_elements,
        "total_different_elements": total_different_elements,
        "different_fraction": different_fraction,
        "significant_difference_elements": (significant_difference_elements),
        "significant_difference_fraction": (significant_difference_fraction),
        "numeric_significance_floor": (
            NUMERIC_SIGNIFICANCE_FLOOR if np.issubdtype(perspective.dtype, np.number) else None
        ),
        "nonfinite_mismatch_elements": nonfinite_mismatch_elements,
        "max_abs_difference": max_abs_difference,
        "mean_abs_difference": mean_abs_difference,
        "p99_abs_difference": p99_abs_difference,
        "rmse": rmse,
    }


def effect_against_repeat_noise(
    cross_mode: dict[str, Any],
    perspective_repeat: dict[str, Any],
    tangential_repeat: dict[str, Any],
    *,
    key: str,
    additional_repeats: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    repeat_comparisons = (
        perspective_repeat,
        tangential_repeat,
        *additional_repeats,
    )
    if not cross_mode["same_shape"] or any(not repeat["same_shape"] for repeat in repeat_comparisons):
        return {"detected": False, "reason": "shape_mismatch", "statistics": {}}

    effect_stats = CATEGORICAL_EFFECT_STATS if key in {"semantic", "valid_depth"} else NUMERIC_EFFECT_STATS
    statistics: dict[str, Any] = {}
    enough_elements = int(cross_mode.get("significant_difference_elements") or 0) >= MIN_EFFECT_ELEMENTS
    for statistic in effect_stats:
        cross_value = cross_mode.get(statistic)
        repeat_values = tuple(repeat.get(statistic) for repeat in repeat_comparisons)
        if cross_value is None or any(value is None for value in repeat_values):
            statistics[statistic] = {
                "pass": False,
                "reason": "statistic_unavailable",
            }
            continue
        noise = max(float(value) for value in repeat_values)
        if statistic == "max_abs_difference":
            absolute_floor = MAX_ABS_EFFECT_FLOOR
        elif statistic == "mean_abs_difference":
            absolute_floor = MEAN_ABS_EFFECT_FLOOR
        else:
            compared = max(int(cross_mode.get("compared_elements") or 0), 1)
            absolute_floor = 0.5 / compared
        tolerance = max(absolute_floor, noise * 0.10)
        passed = enough_elements and float(cross_value) > noise + tolerance
        statistics[statistic] = {
            "pass": passed,
            "cross_mode": float(cross_value),
            "repeat_noise": noise,
            "absolute_floor": absolute_floor,
            "tolerance": tolerance,
        }
    return {
        "detected": any(row["pass"] for row in statistics.values()),
        "minimum_effect_elements": MIN_EFFECT_ELEMENTS,
        "significant_difference_elements": cross_mode.get("significant_difference_elements"),
        "statistics": statistics,
    }


def effect_exceeds_repeat_noise(
    cross_mode: dict[str, Any],
    perspective_repeat: dict[str, Any],
    tangential_repeat: dict[str, Any],
    *,
    key: str = "rgb",
) -> bool:
    return bool(
        effect_against_repeat_noise(
            cross_mode,
            perspective_repeat,
            tangential_repeat,
            key=key,
        )["detected"]
    )


def runtime_token_readback(
    path: Path | None,
    *,
    key: str,
    attribute: str,
    expected: str,
    legacy_field: str | None = None,
) -> dict[str, Any]:
    if path is None:
        return {"present": False, "pass": False, "expected": expected}
    summary = json.loads(path.read_text(encoding="utf-8"))
    sources: dict[str, Any] = {}
    top_level = summary.get("runtime_token_readback")
    if isinstance(top_level, dict) and key in top_level:
        sources["runtime_token_readback"] = top_level[key]
    configuration = summary.get("configuration")
    if isinstance(configuration, dict):
        duplicate = configuration.get("runtime_token_readback")
        if isinstance(duplicate, dict) and key in duplicate:
            sources["configuration.runtime_token_readback"] = duplicate[key]
    detail_valid_by_source = {
        source: (
            isinstance(detail, dict)
            and detail.get("attribute") == attribute
            and detail.get("requested") == expected
            and detail.get("observed") == [expected]
            and detail.get("all_match") is True
            and type(detail.get("prim_count")) is int
            and detail["prim_count"] == 1
        )
        for source, detail in sources.items()
    }
    duplicate_consistent = len({json.dumps(detail, sort_keys=True) for detail in sources.values()}) <= 1
    canonical_source = (
        "runtime_token_readback"
        if "runtime_token_readback" in sources
        else ("configuration.runtime_token_readback" if "configuration.runtime_token_readback" in sources else None)
    )
    detail = sources.get(canonical_source, {}) if canonical_source is not None else {}
    detail_valid = bool(sources) and all(detail_valid_by_source.values()) and duplicate_consistent
    legacy_present = legacy_field is not None and legacy_field in summary
    legacy_observed = summary.get(legacy_field) if legacy_field is not None else None
    legacy_consistent = not legacy_present or legacy_observed == expected
    canonical_observed = (
        detail.get("observed", [None])[0]
        if isinstance(detail.get("observed"), list) and len(detail["observed"]) == 1
        else None
    )
    return {
        "present": True,
        "pass": detail_valid and legacy_consistent,
        "expected": expected,
        "observed": canonical_observed,
        "canonical_source": canonical_source,
        "canonical_detail_valid": detail_valid,
        "source_details": sources,
        "source_validity": detail_valid_by_source,
        "duplicate_sources_consistent": duplicate_consistent,
        "legacy_field_present": legacy_present,
        "legacy_field_observed": legacy_observed,
        "legacy_field_consistent": legacy_consistent,
        "detail": detail,
        "summary": str(path),
    }


def runtime_mode_readback(
    path: Path | None,
    *,
    expected: str,
) -> dict[str, Any]:
    return runtime_token_readback(
        path,
        key="projection_mode",
        attribute="projectionModeHint",
        expected=expected,
        legacy_field="ovrtx_projection_mode_observed",
    )


def runtime_sorting_readback(
    path: Path | None,
    *,
    expected: str,
) -> dict[str, Any]:
    return runtime_token_readback(
        path,
        key="sorting_mode",
        attribute="sortingModeHint",
        expected=expected,
        legacy_field="ovrtx_sorting_mode_observed",
    )


def load_candidate(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as candidate:
        required = set(OUTPUT_KEYS) | set(CANDIDATE_METADATA_KEYS)
        missing = sorted(required - set(candidate.files))
        if missing:
            raise ValueError(f"Projection-mode candidate is missing required arrays: {missing}")
        extras = sorted(set(candidate.files) - required)
        if extras:
            raise ValueError(f"Projection-mode candidate has unexpected arrays: {extras}")
        return {key: np.asarray(candidate[key]).copy() for key in required}


def _candidate_scalar(candidate: dict[str, np.ndarray], key: str) -> Any:
    value = np.asarray(candidate[key])
    if value.shape != ():
        raise ValueError(f"Candidate metadata {key!r} must be scalar, got {value.shape}.")
    return value.item()


def validate_candidate_contract(
    candidate: dict[str, np.ndarray],
    *,
    expected_width: int | None,
    expected_height: int | None,
) -> dict[str, Any]:
    errors: list[str] = []
    rgb = candidate["rgb"]
    alpha = candidate["alpha"]
    depth = candidate["depth"]
    semantic = candidate["semantic"]
    valid_depth = candidate["valid_depth"]
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3:
        errors.append(f"rgb_shape:{rgb.shape}")
        height = width = None
    else:
        height, width = int(rgb.shape[1]), int(rgb.shape[2])
    expected_scalar_shape = (1, height, width) if height is not None and width is not None else None
    for key in ("alpha", "depth", "semantic", "valid_depth"):
        if expected_scalar_shape is None or candidate[key].shape != expected_scalar_shape:
            errors.append(f"{key}_shape:{candidate[key].shape}")
    if expected_width is not None and width != expected_width:
        errors.append(f"width:{width}")
    if expected_height is not None and height != expected_height:
        errors.append(f"height:{height}")
    expected_dtypes = {
        "rgb": np.dtype(np.float32),
        "alpha": np.dtype(np.float32),
        "depth": np.dtype(np.float32),
        "semantic": np.dtype(np.int64),
        "valid_depth": np.dtype(np.bool_),
    }
    for key, expected_dtype in expected_dtypes.items():
        if candidate[key].dtype != expected_dtype:
            errors.append(f"{key}_dtype:{candidate[key].dtype}")

    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        errors.append("rgb_values")
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        errors.append("alpha_values")
    expected_valid = np.isfinite(depth) & (depth > 0)
    if valid_depth.dtype == np.dtype(np.bool_) and not np.array_equal(valid_depth, expected_valid):
        errors.append("valid_depth_mask")
    if valid_depth.shape == depth.shape:
        if not np.isposinf(depth[~valid_depth.astype(bool)]).all():
            errors.append("invalid_depth_sentinel")
        if alpha.shape == depth.shape and np.any(valid_depth.astype(bool) & (alpha <= 0)):
            errors.append("valid_depth_without_alpha")
        finite_depth = depth[valid_depth.astype(bool)]
        if finite_depth.size and np.any((finite_depth < 0.01) | (finite_depth > 100.0)):
            errors.append("valid_depth_range")
    if np.any((semantic < -1) | (semantic > 0)):
        errors.append("semantic_values")
    if semantic.shape == alpha.shape:
        if np.any((semantic >= 0) & (alpha <= 0)):
            errors.append("semantic_without_alpha")
        if np.any((alpha == 0) & (semantic != -1)):
            errors.append("background_semantic")

    foreground_counts = {
        "alpha": int(np.count_nonzero(alpha > 0)),
        "partial_alpha": int(np.count_nonzero((alpha > 0) & (alpha < 1))),
        "background_alpha": int(np.count_nonzero(alpha == 0)),
        "depth": int(np.count_nonzero(valid_depth)),
        "background_depth": int(np.count_nonzero(~valid_depth.astype(bool))),
        "semantic": int(np.count_nonzero(semantic == 0)),
        "background_semantic": int(np.count_nonzero(semantic == -1)),
        "nonzero_rgb": int(np.count_nonzero(rgb > 0)),
    }
    for key, count in foreground_counts.items():
        if count < MIN_FOREGROUND_PIXELS:
            errors.append(f"insufficient_{key}:{count}")
    if alpha.ndim == 3 and rgb.ndim == 4 and rgb.shape[:3] == alpha.shape:
        if np.any(rgb[alpha == 0] != 0):
            errors.append("nonblack_background_rgb")

    try:
        color_space = str(_candidate_scalar(candidate, "color_space"))
        camera_bundle_id = str(_candidate_scalar(candidate, "camera_bundle_id"))
    except ValueError as exc:
        errors.append(str(exc))
        color_space = None
        camera_bundle_id = None
    if color_space != "display_srgb":
        errors.append(f"color_space:{color_space!r}")
    if camera_bundle_id is None or not re.fullmatch(r"[0-9a-f]{64}", camera_bundle_id):
        errors.append(f"camera_bundle_id:{camera_bundle_id!r}")
    background = candidate["background"]
    if (
        background.shape != (3,)
        or background.dtype != np.dtype(np.float32)
        or not np.array_equal(background, np.zeros(3, dtype=np.float32))
    ):
        errors.append("background")
    return {
        "pass": not errors,
        "errors": errors,
        "shape": [1, height, width] if height is not None and width is not None else None,
        "camera_bundle_id": camera_bundle_id,
        "color_space": color_space,
        "foreground_counts": foreground_counts,
    }


def compare_candidates(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key in OUTPUT_KEYS:
        valid_mask = None
        if key == "depth":
            valid_mask = np.asarray(first["valid_depth"], dtype=np.bool_) & np.asarray(
                second["valid_depth"], dtype=np.bool_
            )
        rows[key] = compare_array(
            first[key],
            second[key],
            valid_mask=valid_mask,
        )
    return rows


def normalized_stage_text(
    path: Path,
    *,
    normalize_mode: bool,
    normalize_focal: bool,
) -> str:
    text = path.read_text(encoding="utf-8")
    if normalize_mode:
        text = MODE_SUB_PATTERN.sub(r'\1"__PROJECTION_MODE__"', text)
    if normalize_focal:
        text = FOCAL_SUB_PATTERN.sub(r"\1__FOCAL_LENGTH__", text)
    return text


def validate_stage_contract(
    path: Path | None,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    if path is None:
        return {"present": False, "pass": False, "errors": ["missing_stage"]}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return {
            "present": True,
            "pass": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    modes = MODE_PATTERN.findall(text)
    sorting_modes = SORTING_PATTERN.findall(text)
    focal_lengths = authored_focal_lengths(path)
    particle_fields = len(re.findall(r'\bdef\s+ParticleField3DGaussianSplat\s+"', text))
    errors = []
    if modes != [expected_mode]:
        errors.append(f"projection_modes:{modes}")
    if sorting_modes != ["zDepth"]:
        errors.append(f"sorting_modes:{sorting_modes}")
    if len(focal_lengths) != 1:
        errors.append(f"focal_lengths:{focal_lengths}")
    if particle_fields != 1:
        errors.append(f"particle_fields:{particle_fields}")
    return {
        "present": True,
        "pass": not errors,
        "errors": errors,
        "mode": expected_mode,
        "sorting_modes": sorting_modes,
        "focal_lengths": focal_lengths,
        "particle_field_count": particle_fields,
        "sha256": file_sha256(path),
    }


def validate_summary_contract(
    path: Path | None,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    if path is None:
        return {"present": False, "pass": False, "errors": ["missing_summary"]}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "present": True,
            "pass": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    projection = runtime_mode_readback(path, expected=expected_mode)
    sorting = runtime_sorting_readback(path, expected="zDepth")
    scene = summary.get("scene")
    provenance = summary.get("provenance")
    camera_bundle_id = summary.get("camera_bundle_id")
    errors: list[str] = []
    exact_fields = {
        "schema_version": "ovrtx-temporal-fidelity/v1",
        "scene_id": "projection-activation",
        "gaussian_count": 4,
        "semantic_scheme": "index-modulo",
        "semantic_group_count": 1,
        "ovrtx_projection_mode_hint": expected_mode,
        "ovrtx_projection_mode_observed": expected_mode,
        "ovrtx_sorting_mode": "zDepth",
        "ovrtx_sorting_mode_observed": "zDepth",
        "ovrtx_fractional_opacity": True,
        "ovrtx_color_space": "display_srgb",
        "resumed_from_frames": 0,
        "resume_sample_sequence_advanced_frames": 0,
        "resume_accumulators": None,
        "final_report_only": True,
    }
    for key, expected in exact_fields.items():
        if summary.get(key) != expected:
            errors.append(f"{key}:{summary.get(key)!r}")
    if not projection["pass"]:
        errors.append("projection_readback")
    if not sorting["pass"]:
        errors.append("sorting_readback")
    if not isinstance(scene, dict) or not isinstance(scene.get("checksum_sha256"), str):
        errors.append("scene_manifest")
        scene_checksum = None
        scene_tensor_checksum = None
    else:
        scene_checksum = scene.get("checksum_sha256")
        scene_tensor_checksum = scene.get("tensor_checksum_sha256")
        if summary.get("scene_tensor_sha256") != scene_tensor_checksum:
            errors.append("scene_tensor_checksum")
    if not isinstance(camera_bundle_id, str) or not re.fullmatch(r"[0-9a-f]{64}", camera_bundle_id):
        errors.append("camera_bundle_id")
    required_provenance = (
        "project_commit",
        "ovrtx_commit",
        "ovrtx_version",
        "renderer_version",
        "script_sha256",
        "stage_sha256",
    )
    if not isinstance(provenance, dict):
        errors.append("provenance")
        provenance_contract = None
    else:
        missing_provenance = [key for key in required_provenance if provenance.get(key) in (None, "", [])]
        if missing_provenance:
            errors.append(f"provenance_missing:{missing_provenance}")
        provenance_contract = {
            key: provenance.get(key)
            for key in (
                "project_commit",
                "ovrtx_commit",
                "ovrtx_version",
                "renderer_version",
                "script_sha256",
            )
        }
    return {
        "present": True,
        "pass": not errors,
        "errors": errors,
        "projection_readback": projection,
        "sorting_readback": sorting,
        "camera_bundle_id": camera_bundle_id,
        "scene_checksum": scene_checksum,
        "scene_tensor_checksum": scene_tensor_checksum,
        "provenance_contract": provenance_contract,
        "payload": summary,
    }


def validate_matched_lane_contract(
    *,
    stage_paths: dict[str, Path | None],
    stage_contracts: dict[str, dict[str, Any]],
    summary_contracts: dict[str, dict[str, Any]],
    candidate_contracts: dict[str, dict[str, Any]],
    expected_control_focal_ratio: float,
) -> dict[str, Any]:
    errors: list[str] = []
    required_names = (
        "perspective",
        "tangential",
        "perspective_repeat",
        "tangential_repeat",
        "positive_control",
        "positive_control_repeat",
    )
    if any(not stage_contracts[name]["pass"] for name in required_names):
        errors.append("stage_contract")
    if any(not summary_contracts[name]["pass"] for name in required_names):
        errors.append("summary_contract")
    if any(not candidate_contracts[name]["pass"] for name in required_names):
        errors.append("candidate_contract")
    if errors:
        return {"pass": False, "errors": errors}
    for name in required_names:
        provenance = summary_contracts[name]["payload"]["provenance"]
        if provenance.get("stage_sha256") != stage_contracts[name]["sha256"]:
            errors.append(f"stage_provenance:{name}")
    if errors:
        return {"pass": False, "errors": errors}
    paths = {name: stage_paths[name] for name in required_names}
    assert all(path is not None for path in paths.values())
    concrete_paths = {name: path for name, path in paths.items() if path is not None}
    stage_texts = {name: path.read_text(encoding="utf-8") for name, path in concrete_paths.items()}
    if stage_texts["perspective"] != stage_texts["perspective_repeat"]:
        errors.append("perspective_repeat_stage")
    if stage_texts["tangential"] != stage_texts["tangential_repeat"]:
        errors.append("tangential_repeat_stage")
    if stage_texts["positive_control"] != stage_texts["positive_control_repeat"]:
        errors.append("positive_control_repeat_stage")
    mode_normalized = {
        normalized_stage_text(
            concrete_paths[name],
            normalize_mode=True,
            normalize_focal=False,
        )
        for name in (
            "perspective",
            "tangential",
            "perspective_repeat",
            "tangential_repeat",
        )
    }
    if len(mode_normalized) != 1:
        errors.append("mode_stage_mismatch")
    fully_normalized = {
        normalized_stage_text(
            concrete_paths[name],
            normalize_mode=True,
            normalize_focal=True,
        )
        for name in required_names
    }
    if len(fully_normalized) != 1:
        errors.append("control_stage_mismatch")

    base_focal = stage_contracts["perspective"]["focal_lengths"][0]
    control_focal = stage_contracts["positive_control"]["focal_lengths"][0]
    if not np.isclose(
        control_focal / base_focal,
        expected_control_focal_ratio,
        rtol=0,
        atol=1.0e-9,
    ):
        errors.append("control_focal_ratio")
    if stage_contracts["positive_control_repeat"]["focal_lengths"] != [control_focal]:
        errors.append("control_repeat_focal")

    base_names = (
        "perspective",
        "tangential",
        "perspective_repeat",
        "tangential_repeat",
    )
    control_names = ("positive_control", "positive_control_repeat")
    base_camera_ids = {candidate_contracts[name]["camera_bundle_id"] for name in base_names} | {
        summary_contracts[name]["camera_bundle_id"] for name in base_names
    }
    control_camera_ids = {candidate_contracts[name]["camera_bundle_id"] for name in control_names} | {
        summary_contracts[name]["camera_bundle_id"] for name in control_names
    }
    if len(base_camera_ids) != 1:
        errors.append("base_camera_mismatch")
    if len(control_camera_ids) != 1:
        errors.append("control_camera_mismatch")
    if base_camera_ids == control_camera_ids:
        errors.append("control_camera_unchanged")
    for contract_key in (
        "scene_checksum",
        "scene_tensor_checksum",
        "provenance_contract",
    ):
        values = {json.dumps(summary_contracts[name][contract_key], sort_keys=True) for name in required_names}
        if len(values) != 1:
            errors.append(f"cross_lane_{contract_key}")
    return {
        "pass": not errors,
        "errors": errors,
        "base_camera_bundle_ids": sorted(base_camera_ids),
        "control_camera_bundle_ids": sorted(control_camera_ids),
        "base_focal_length": base_focal,
        "control_focal_length": control_focal,
        "expected_control_focal_ratio": expected_control_focal_ratio,
    }


def alpha_radial_second_moment(alpha: np.ndarray) -> float:
    if alpha.ndim != 3 or alpha.shape[0] != 1:
        raise ValueError(f"alpha must have shape [1,H,W], got {alpha.shape}.")
    height, width = alpha.shape[1:]
    y, x = np.mgrid[:height, :width]
    radius_squared = (x - width * 0.5) ** 2 + (y - height * 0.5) ** 2
    weights = alpha[0].astype(np.float64)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("alpha radial moment requires nonzero coverage.")
    return float(np.sum(weights * radius_squared) / total)


def main() -> None:
    args = parse_args()
    candidate_paths = {
        "perspective": args.perspective_candidate,
        "tangential": args.tangential_candidate,
        "perspective_repeat": args.perspective_repeat,
        "tangential_repeat": args.tangential_repeat,
        "positive_control": args.positive_control_candidate,
        "positive_control_repeat": args.positive_control_repeat_candidate,
    }
    stage_paths = {
        "perspective": args.perspective_stage,
        "tangential": args.tangential_stage,
        "perspective_repeat": args.perspective_repeat_stage,
        "tangential_repeat": args.tangential_repeat_stage,
        "positive_control": args.positive_control_stage,
        "positive_control_repeat": args.positive_control_repeat_stage,
    }
    summary_paths = {
        "perspective": args.perspective_summary,
        "tangential": args.tangential_summary,
        "perspective_repeat": args.perspective_repeat_summary,
        "tangential_repeat": args.tangential_repeat_summary,
        "positive_control": args.positive_control_summary,
        "positive_control_repeat": args.positive_control_repeat_summary,
    }
    expected_modes = {
        "perspective": "perspective",
        "tangential": "tangential",
        "perspective_repeat": "perspective",
        "tangential_repeat": "tangential",
        "positive_control": "perspective",
        "positive_control_repeat": "perspective",
    }
    repeat_controls_present = all(
        candidate_paths[name] is not None and stage_paths[name] is not None and summary_paths[name] is not None
        for name in ("perspective_repeat", "tangential_repeat")
    )
    positive_control_present = all(
        candidate_paths[name] is not None and stage_paths[name] is not None and summary_paths[name] is not None
        for name in ("positive_control", "positive_control_repeat")
    )

    candidates: dict[str, dict[str, np.ndarray] | None] = {}
    candidate_contracts: dict[str, dict[str, Any]] = {}
    for name, path in candidate_paths.items():
        if path is None:
            candidates[name] = None
            candidate_contracts[name] = {
                "pass": False,
                "errors": ["missing_candidate"],
                "camera_bundle_id": None,
            }
            continue
        try:
            candidate = load_candidate(path)
            contract = validate_candidate_contract(
                candidate,
                expected_width=args.expected_width,
                expected_height=args.expected_height,
            )
        except Exception as exc:
            candidate = None
            contract = {
                "pass": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
                "camera_bundle_id": None,
            }
        candidates[name] = candidate
        candidate_contracts[name] = contract

    stage_contracts = {
        name: validate_stage_contract(path, expected_mode=expected_modes[name]) for name, path in stage_paths.items()
    }
    summary_contracts = {
        name: validate_summary_contract(path, expected_mode=expected_modes[name])
        for name, path in summary_paths.items()
    }
    stage_authoring_valid = all(contract["pass"] for contract in stage_contracts.values())
    runtime_readback_valid = all(
        contract.get("projection_readback", {}).get("pass") is True
        and contract.get("sorting_readback", {}).get("pass") is True
        for contract in summary_contracts.values()
    )
    candidate_outputs_valid = all(contract["pass"] for contract in candidate_contracts.values())
    matched_lane_contract = validate_matched_lane_contract(
        stage_paths=stage_paths,
        stage_contracts=stage_contracts,
        summary_contracts=summary_contracts,
        candidate_contracts=candidate_contracts,
        expected_control_focal_ratio=args.expected_control_focal_ratio,
    )
    matched_lane_contract_valid = matched_lane_contract["pass"]

    arrays: dict[str, Any] = {}
    repeat_noise: dict[str, Any] | None = None
    effect_above_repeat_noise = False
    observably_different = False
    mode_common_effects: list[dict[str, str]] = []
    if candidate_outputs_valid and repeat_controls_present and positive_control_present:
        perspective = candidates["perspective"]
        tangential = candidates["tangential"]
        perspective_repeat = candidates["perspective_repeat"]
        tangential_repeat = candidates["tangential_repeat"]
        control = candidates["positive_control"]
        control_repeat = candidates["positive_control_repeat"]
        assert all(
            candidate is not None
            for candidate in (
                perspective,
                tangential,
                perspective_repeat,
                tangential_repeat,
                control,
                control_repeat,
            )
        )
        primary_cross = compare_candidates(perspective, tangential)
        repeat_cross = compare_candidates(perspective_repeat, tangential_repeat)
        perspective_repeat_comparison = compare_candidates(perspective, perspective_repeat)
        tangential_repeat_comparison = compare_candidates(tangential, tangential_repeat)
        control_repeat_comparison = compare_candidates(control, control_repeat)
        repeat_noise = {
            key: {
                "perspective": perspective_repeat_comparison[key],
                "tangential": tangential_repeat_comparison[key],
                "positive_control": control_repeat_comparison[key],
            }
            for key in OUTPUT_KEYS
        }
        for key in OUTPUT_KEYS:
            for label, cross in (
                ("candidate_pair", primary_cross),
                ("repeat_pair", repeat_cross),
            ):
                analysis = effect_against_repeat_noise(
                    cross[key],
                    perspective_repeat_comparison[key],
                    tangential_repeat_comparison[key],
                    key=key,
                    additional_repeats=(control_repeat_comparison[key],),
                )
                cross[key]["effect_analysis"] = analysis
                cross[key]["effect_exceeds_repeat_noise"] = analysis["detected"]
                cross[key]["pair"] = label
            first_stats = primary_cross[key]["effect_analysis"]["statistics"]
            second_stats = repeat_cross[key]["effect_analysis"]["statistics"]
            for statistic in sorted(set(first_stats) & set(second_stats)):
                if first_stats[statistic]["pass"] and second_stats[statistic]["pass"]:
                    mode_common_effects.append({"key": key, "statistic": statistic})
        arrays = {
            "candidate_pair": primary_cross,
            "repeat_pair": repeat_cross,
        }
        observably_different = any(not row["bitwise_equal"] for pair in arrays.values() for row in pair.values())
        effect_above_repeat_noise = bool(mode_common_effects)

    positive_control_detected = False
    positive_control_stable = False
    positive_control_valid = False
    positive_control: dict[str, Any] | None = None
    if candidate_outputs_valid and repeat_controls_present and positive_control_present:
        assert repeat_noise is not None
        perspective = candidates["perspective"]
        perspective_repeat = candidates["perspective_repeat"]
        control = candidates["positive_control"]
        control_repeat = candidates["positive_control_repeat"]
        assert all(candidate is not None for candidate in (perspective, perspective_repeat, control, control_repeat))
        first_control_cross = compare_candidates(perspective, control)
        second_control_cross = compare_candidates(perspective_repeat, control_repeat)
        control_repeat_cross = compare_candidates(control, control_repeat)
        control_common_effects: list[dict[str, str]] = []
        control_instability: list[dict[str, str]] = []
        for key in OUTPUT_KEYS:
            for cross in (first_control_cross, second_control_cross):
                analysis = effect_against_repeat_noise(
                    cross[key],
                    repeat_noise[key]["perspective"],
                    repeat_noise[key]["tangential"],
                    key=key,
                    additional_repeats=(repeat_noise[key]["positive_control"],),
                )
                cross[key]["effect_analysis"] = analysis
            instability = effect_against_repeat_noise(
                control_repeat_cross[key],
                repeat_noise[key]["perspective"],
                repeat_noise[key]["tangential"],
                key=key,
            )
            control_repeat_cross[key]["instability_analysis"] = instability
            if instability["detected"]:
                control_instability.append({"key": key})
            first_stats = first_control_cross[key]["effect_analysis"]["statistics"]
            second_stats = second_control_cross[key]["effect_analysis"]["statistics"]
            for statistic in sorted(set(first_stats) & set(second_stats)):
                if first_stats[statistic]["pass"] and second_stats[statistic]["pass"]:
                    control_common_effects.append({"key": key, "statistic": statistic})
        geometry_common_effects = [effect for effect in control_common_effects if effect["key"] in GEOMETRY_OUTPUT_KEYS]
        moment_values = {
            "perspective_candidate": alpha_radial_second_moment(perspective["alpha"]),
            "perspective_repeat": alpha_radial_second_moment(perspective_repeat["alpha"]),
            "positive_control_candidate": alpha_radial_second_moment(control["alpha"]),
            "positive_control_repeat": alpha_radial_second_moment(control_repeat["alpha"]),
        }
        moment_noise = max(
            abs(moment_values["perspective_candidate"] - moment_values["perspective_repeat"]),
            abs(moment_values["positive_control_candidate"] - moment_values["positive_control_repeat"]),
        )
        moment_tolerance = max(1.0e-6, moment_noise * 0.10)
        moment_deltas = {
            "candidate_pair": (moment_values["positive_control_candidate"] - moment_values["perspective_candidate"]),
            "repeat_pair": (moment_values["positive_control_repeat"] - moment_values["perspective_repeat"]),
        }
        signed_moment_detected = all(delta > moment_noise + moment_tolerance for delta in moment_deltas.values())
        positive_control_stable = not control_instability
        positive_control_detected = bool(geometry_common_effects) and signed_moment_detected
        positive_control_valid = matched_lane_contract_valid and positive_control_stable and positive_control_detected
        positive_control = {
            "present": True,
            "pass": positive_control_valid,
            "scope": (
                "camera-focal-length authoring and output-capture sensitivity; "
                "this does not prove ParticleField hint consumption"
            ),
            "stable": positive_control_stable,
            "effect_detected": positive_control_detected,
            "common_geometry_effects": geometry_common_effects,
            "instability": control_instability,
            "signed_alpha_radial_second_moment": {
                "values": moment_values,
                "repeat_noise": moment_noise,
                "tolerance": moment_tolerance,
                "control_minus_base": moment_deltas,
                "detected": signed_moment_detected,
            },
            "candidate_pair_arrays": first_control_cross,
            "repeat_pair_arrays": second_control_cross,
            "control_repeat_arrays": control_repeat_cross,
        }
    elif positive_control_present:
        positive_control = {
            "present": True,
            "pass": False,
            "reason": "candidate_or_repeat_contract_invalid",
        }

    token_proof_valid = stage_authoring_valid and runtime_readback_valid and matched_lane_contract_valid
    behavioral_activation_valid = (
        candidate_outputs_valid
        and matched_lane_contract_valid
        and observably_different
        and repeat_controls_present
        and effect_above_repeat_noise
    )
    activation_proof_valid = (
        token_proof_valid
        and behavioral_activation_valid
        and (not args.require_positive_control or positive_control_valid)
    )
    if not repeat_controls_present:
        classification = "MISSING_REPEAT_CONTROLS"
    elif args.require_positive_control and not positive_control_present:
        classification = "MISSING_POSITIVE_CONTROL"
    elif not candidate_outputs_valid:
        classification = "INVALID_CANDIDATE_OUTPUT"
    elif not stage_authoring_valid:
        classification = "INVALID_MODE_AUTHORING"
    elif not runtime_readback_valid:
        classification = "INVALID_RUNTIME_MODE_READBACK"
    elif not matched_lane_contract_valid:
        classification = "INVALID_MATCHED_LANE_CONTRACT"
    elif args.require_positive_control and not positive_control_stable:
        classification = "POSITIVE_CONTROL_UNSTABLE"
    elif args.require_positive_control and not positive_control_detected:
        classification = "POSITIVE_CONTROL_NOT_DETECTED"
    elif not observably_different:
        classification = "NO_OBSERVABLE_MODE_EFFECT"
    elif not effect_above_repeat_noise:
        classification = "MODE_EFFECT_WITHIN_REPEAT_NOISE"
    else:
        classification = "OBSERVABLE_MODE_EFFECT"

    report_pass = (
        activation_proof_valid
        if args.require_activation_proof
        else (
            token_proof_valid
            and candidate_outputs_valid
            and observably_different
            and (not args.require_positive_control or positive_control_valid)
        )
    )
    report = {
        "schema_version": "ovrtx-projection-mode-audit/v3",
        "pass": bool(report_pass),
        "classification": classification,
        "activation_proof_required": args.require_activation_proof,
        "activation_proof_valid": activation_proof_valid,
        "token_proof_valid": token_proof_valid,
        "behavioral_activation_valid": behavioral_activation_valid,
        "candidate_outputs_valid": candidate_outputs_valid,
        "matched_lane_contract_valid": matched_lane_contract_valid,
        "stage_authoring_valid": stage_authoring_valid,
        "observably_different": observably_different,
        "repeat_controls_present": repeat_controls_present,
        "effect_above_repeat_noise": effect_above_repeat_noise,
        "mode_common_effects": mode_common_effects,
        "runtime_readback_valid": runtime_readback_valid,
        "runtime_readback": {
            name: {
                "projection": contract.get("projection_readback"),
                "sorting": contract.get("sorting_readback"),
            }
            for name, contract in summary_contracts.items()
        },
        "positive_control_required": args.require_positive_control,
        "positive_control_present": positive_control_present,
        "positive_control_stable": positive_control_stable,
        "positive_control_detected": positive_control_detected,
        "positive_control_valid": positive_control_valid,
        "positive_control": positive_control,
        "candidate_contracts": candidate_contracts,
        "stage_contracts": stage_contracts,
        "summary_contracts": summary_contracts,
        "matched_lane_contract": matched_lane_contract,
        "interpretation": (
            "The USD schema defines projectionModeHint as advisory. Runtime "
            "readback proves composition, not ParticleField render-kernel "
            "consumption. The bracketed focal-length control proves only "
            "that matched camera authoring and output capture are sensitive "
            "in this run. A null mode effect is scoped to this tested OVRTX "
            "configuration and is not a schema-violation or global support claim."
        ),
        "arrays": arrays,
        "repeat_noise": repeat_noise,
        "artifacts": {
            name: {
                "candidate": str(candidate_paths[name]) if candidate_paths[name] else None,
                "candidate_sha256": (
                    file_sha256(candidate_paths[name])
                    if candidate_paths[name] is not None and candidate_paths[name].is_file()
                    else None
                ),
                "stage": str(stage_paths[name]) if stage_paths[name] else None,
                "summary": str(summary_paths[name]) if summary_paths[name] else None,
            }
            for name in candidate_paths
        },
        "effect_detection_contract": {
            "pairing": "C1-P1, P1-T1, T2-P2, P2-C2 with C/P/T repeat envelopes",
            "both_mode_pairs_required_same_aov_statistic": True,
            "both_control_pairs_required_same_geometry_aov_statistic": True,
            "positive_control_signed_statistic": "alpha radial second moment",
            "numeric_statistics": list(NUMERIC_EFFECT_STATS),
            "categorical_statistics": list(CATEGORICAL_EFFECT_STATS),
            "numeric_significance_floor": NUMERIC_SIGNIFICANCE_FLOOR,
            "max_abs_effect_floor": MAX_ABS_EFFECT_FLOOR,
            "mean_abs_effect_floor": MEAN_ABS_EFFECT_FLOOR,
            "minimum_effect_elements": MIN_EFFECT_ELEMENTS,
            "minimum_foreground_pixels": MIN_FOREGROUND_PIXELS,
            "depth_numeric_mask": "jointly-valid pixels",
            "depth_validity_changes": "valid_depth categorical AOV",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
