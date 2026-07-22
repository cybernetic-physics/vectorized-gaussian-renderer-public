#!/usr/bin/env python3
"""Diagnose native exact-ray output and semantic aggregation behavior.

The CUDA capture path renders the deterministic ``synthetic-small`` scene with
``CustomCudaBackend(ray_gaussian_evaluation=True)`` and stores a portable NPZ
containing the native outputs plus the sorted per-pixel contributions.  The
analysis path needs only NumPy, so the same artifact can be inspected on a
machine without CUDA.  When no artifact is available, analysis-only mode runs
a small in-memory contract check and reports that no renderer evidence was
evaluated.

The full Python reconstruction follows the equations used by
``experiments/ovrtx_exact_ray_model.py``: exact 3D ray/Gaussian alpha,
front-to-back Bernoulli first-hit weights, center-Z depth, and either the
strongest individual contribution or the aggregate contribution per semantic
label.  An additional reconstruction applies the native compositor's
transmittance cutoff and semantic alpha threshold for implementation parity.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_SCHEMA_VERSION = "native-exact-semantics-artifact/v1"
REPORT_SCHEMA_VERSION = "native-exact-semantics-diagnostic/v1"
SCENE_ID = 101
SCENE_NAME = "synthetic-small"
GAUSSIAN_COUNT = 10_000
SCENE_SEED = 42
NATIVE_TRANSMITTANCE_THRESHOLD = 1.0e-4
SUCCESS_MARKER = "NATIVE_EXACT_SEMANTICS_DIAGNOSTIC_OK"
DEFAULT_ACCUMULATOR_CAPACITIES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
REQUIRED_ARTIFACT_ARRAYS = (
    "pixel_offsets",
    "contribution_alpha",
    "contribution_depth",
    "contribution_color",
    "contribution_semantic",
    "native_rgb",
    "native_alpha",
    "native_depth",
    "native_semantic",
)


def positive_int_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, unique, increasing list of positive integers."""

    try:
        values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.") from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return values


def _validate_contributions(
    *,
    pixel_offsets: np.ndarray,
    alphas: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    semantic_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(pixel_offsets, dtype=np.int64)
    alpha = np.asarray(alphas, dtype=np.float32)
    depth = np.asarray(depths, dtype=np.float32)
    color = np.asarray(colors, dtype=np.float32)
    semantic = np.asarray(semantic_ids, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 1:
        raise ValueError("pixel_offsets must have shape [pixel_count + 1].")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("pixel_offsets must start at zero and be nondecreasing.")
    contribution_count = int(offsets[-1])
    if alpha.shape != (contribution_count,):
        raise ValueError(
            f"alphas must have one entry per contribution; expected {(contribution_count,)}, got {alpha.shape}."
        )
    if depth.shape != (contribution_count,):
        raise ValueError(
            f"depths must have one entry per contribution; expected {(contribution_count,)}, got {depth.shape}."
        )
    if color.shape != (contribution_count, 3):
        raise ValueError(f"colors must have shape [contribution_count, 3]; got {color.shape}.")
    if semantic.shape != (contribution_count,):
        raise ValueError(
            "semantic_ids must have one entry per contribution; "
            f"expected {(contribution_count,)}, got {semantic.shape}."
        )
    if not np.isfinite(alpha).all() or np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("alphas must be finite and in [0, 1].")
    if not np.isfinite(depth).all():
        raise ValueError("contribution depths must be finite.")
    if not np.isfinite(color).all():
        raise ValueError("contribution colors must be finite.")
    if np.any(semantic < 0):
        raise ValueError("contribution semantic IDs must be non-negative.")
    return offsets, alpha, depth, color, semantic


def composite_exact_contributions(
    *,
    pixel_offsets: np.ndarray,
    alphas: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    semantic_ids: np.ndarray,
    semantic_min_alpha: float,
    transmittance_threshold: float | None,
) -> dict[str, np.ndarray]:
    """Composite sorted exact-ray contributions with explicit semantic rules.

    ``transmittance_threshold=None`` evaluates the full Python exact-ray model.
    A finite threshold reproduces the native early-exit rule.  The returned
    strongest-individual label uses the largest single ``T * alpha`` weight;
    the aggregate label sums those weights for every distinct semantic ID.
    Aggregate ties choose the smallest label, matching dense label ``argmax``.
    """

    if not math.isfinite(semantic_min_alpha) or not 0.0 <= semantic_min_alpha <= 1.0:
        raise ValueError("semantic_min_alpha must be finite and in [0, 1].")
    if transmittance_threshold is not None and (
        not math.isfinite(transmittance_threshold) or not 0.0 < transmittance_threshold < 1.0
    ):
        raise ValueError("transmittance_threshold must be None or finite and in (0, 1).")
    offsets, alpha, depth, color, semantic = _validate_contributions(
        pixel_offsets=pixel_offsets,
        alphas=alphas,
        depths=depths,
        colors=colors,
        semantic_ids=semantic_ids,
    )
    pixel_count = offsets.size - 1
    rgb = np.zeros((pixel_count, 3), dtype=np.float32)
    accumulated_alpha = np.zeros(pixel_count, dtype=np.float32)
    composited_depth = np.full(pixel_count, np.inf, dtype=np.float32)
    strongest_semantic = np.full(pixel_count, -1, dtype=np.int64)
    aggregate_semantic = np.full(pixel_count, -1, dtype=np.int64)
    strongest_weight = np.zeros(pixel_count, dtype=np.float32)
    aggregate_top_weight = np.zeros(pixel_count, dtype=np.float32)
    aggregate_second_weight = np.zeros(pixel_count, dtype=np.float32)
    processed_contribution_count = np.zeros(pixel_count, dtype=np.int64)
    processed_distinct_label_count = np.zeros(pixel_count, dtype=np.int64)
    raster_positive_contribution_count = np.zeros(pixel_count, dtype=np.int64)
    raster_distinct_label_count = np.zeros(pixel_count, dtype=np.int64)
    omitted_positive_contribution_count = np.zeros(pixel_count, dtype=np.int64)
    omitted_distinct_label_count = np.zeros(pixel_count, dtype=np.int64)

    for pixel_id in range(pixel_count):
        start = int(offsets[pixel_id])
        end = int(offsets[pixel_id + 1])
        positive_indices = [index for index in range(start, end) if alpha[index] > 0.0]
        raster_positive_contribution_count[pixel_id] = len(positive_indices)
        raster_labels = {int(semantic[index]) for index in positive_indices}
        raster_distinct_label_count[pixel_id] = len(raster_labels)

        transmittance = np.float32(1.0)
        accumulated_rgb = np.zeros(3, dtype=np.float32)
        accumulated_depth = np.float32(0.0)
        best_weight = np.float32(-1.0)
        best_label = -1
        label_weights: dict[int, np.float32] = {}
        processed_indices: list[int] = []
        for index in range(start, end):
            splat_alpha = np.float32(alpha[index])
            if not splat_alpha > 0.0:
                continue
            weight = np.float32(transmittance * splat_alpha)
            accumulated_rgb = np.asarray(
                accumulated_rgb + np.asarray(weight * color[index], dtype=np.float32),
                dtype=np.float32,
            )
            accumulated_depth = np.float32(accumulated_depth + np.float32(weight * depth[index]))
            label = int(semantic[index])
            label_weights[label] = np.float32(label_weights.get(label, np.float32(0.0)) + weight)
            if weight > best_weight:
                best_weight = weight
                best_label = label
            processed_indices.append(index)
            transmittance = np.float32(transmittance * np.float32(1.0 - splat_alpha))
            if transmittance_threshold is not None and transmittance <= np.float32(transmittance_threshold):
                break

        alpha_value = np.float32(1.0 - transmittance)
        rgb[pixel_id] = accumulated_rgb
        accumulated_alpha[pixel_id] = alpha_value
        if alpha_value > np.float32(1.0e-8):
            composited_depth[pixel_id] = np.float32(accumulated_depth / alpha_value)

        processed_contribution_count[pixel_id] = len(processed_indices)
        processed_labels = {int(semantic[index]) for index in processed_indices}
        processed_distinct_label_count[pixel_id] = len(processed_labels)
        omitted_indices = positive_indices[len(processed_indices) :]
        omitted_positive_contribution_count[pixel_id] = len(omitted_indices)
        omitted_distinct_label_count[pixel_id] = len({int(semantic[index]) for index in omitted_indices})

        if best_label >= 0:
            strongest_weight[pixel_id] = best_weight
        if best_label < 0 or alpha_value < np.float32(semantic_min_alpha):
            continue
        strongest_semantic[pixel_id] = best_label
        ordered_labels = sorted(
            label_weights.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
        aggregate_semantic[pixel_id] = ordered_labels[0][0]
        aggregate_top_weight[pixel_id] = ordered_labels[0][1]
        if len(ordered_labels) > 1:
            aggregate_second_weight[pixel_id] = ordered_labels[1][1]

    return {
        "rgb": rgb,
        "alpha": accumulated_alpha,
        "depth": composited_depth,
        "strongest_individual_semantic": strongest_semantic,
        "aggregate_label_semantic": aggregate_semantic,
        "strongest_individual_weight": strongest_weight,
        "aggregate_top_weight": aggregate_top_weight,
        "aggregate_second_weight": aggregate_second_weight,
        "aggregate_top_minus_second_weight": (aggregate_top_weight - aggregate_second_weight).astype(
            np.float32, copy=False
        ),
        "processed_contribution_count": processed_contribution_count,
        "processed_distinct_label_count": processed_distinct_label_count,
        "raster_positive_contribution_count": raster_positive_contribution_count,
        "raster_distinct_label_count": raster_distinct_label_count,
        "omitted_positive_contribution_count": omitted_positive_contribution_count,
        "omitted_distinct_label_count": omitted_distinct_label_count,
    }


def integer_histogram(values: np.ndarray) -> dict[str, int]:
    """Return a stable JSON-ready histogram for non-negative integer values."""

    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(array < 0):
        raise ValueError("Histogram values must be non-negative.")
    if array.size == 0:
        return {}
    unique, counts = np.unique(array, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def accumulator_capacity_summary(
    distinct_label_counts: np.ndarray,
    capacities: Sequence[int],
) -> dict[str, Any]:
    """Summarize accumulator occupancy and overflow at candidate capacities."""

    counts = np.asarray(distinct_label_counts, dtype=np.int64).reshape(-1)
    if np.any(counts < 0):
        raise ValueError("Distinct-label counts must be non-negative.")
    normalized_capacities = tuple(sorted({int(capacity) for capacity in capacities}))
    if not normalized_capacities or normalized_capacities[0] <= 0:
        raise ValueError("capacities must contain positive integers.")
    if counts.size:
        percentiles = {
            name: int(np.quantile(counts, quantile, method="higher"))
            for name, quantile in (
                ("p50", 0.50),
                ("p90", 0.90),
                ("p95", 0.95),
                ("p99", 0.99),
                ("p99_9", 0.999),
            )
        }
        maximum = int(counts.max())
        mean = float(counts.mean())
    else:
        percentiles = {name: 0 for name in ("p50", "p90", "p95", "p99", "p99_9")}
        maximum = 0
        mean = 0.0
    return {
        "pixel_count": int(counts.size),
        "histogram": integer_histogram(counts),
        "mean": mean,
        "maximum": maximum,
        "minimum_capacity_for_zero_overflow": maximum,
        "percentiles_higher": percentiles,
        "candidate_capacities": [
            {
                "capacity": capacity,
                "pixels_exceeding_capacity": int(np.count_nonzero(counts > capacity)),
                "fraction_exceeding_capacity": (float(np.mean(counts > capacity)) if counts.size else 0.0),
            }
            for capacity in normalized_capacities
        ],
    }


def _finite_quantiles(values: np.ndarray) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    quantile_values = np.quantile(array, (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p25", "median", "p75", "p90", "p99", "max"),
            quantile_values,
            strict=True,
        )
    }


def summarize_semantic_rules(
    composite: Mapping[str, np.ndarray],
    *,
    capacities: Sequence[int] = DEFAULT_ACCUMULATOR_CAPACITIES,
) -> dict[str, Any]:
    """Compare strongest-individual and aggregate-label semantic decisions."""

    individual = np.asarray(composite["strongest_individual_semantic"], dtype=np.int64).reshape(-1)
    aggregate = np.asarray(composite["aggregate_label_semantic"], dtype=np.int64).reshape(-1)
    processed_distinct = np.asarray(composite["processed_distinct_label_count"], dtype=np.int64).reshape(-1)
    raster_distinct = np.asarray(composite["raster_distinct_label_count"], dtype=np.int64).reshape(-1)
    margin = np.asarray(composite["aggregate_top_minus_second_weight"], dtype=np.float32).reshape(-1)
    if not (individual.shape == aggregate.shape == processed_distinct.shape == raster_distinct.shape == margin.shape):
        raise ValueError("Semantic analysis arrays must have identical flattened shapes.")
    mismatch = individual != aggregate
    both_foreground = (individual >= 0) & (aggregate >= 0)
    foreground_mismatch = mismatch & both_foreground
    foreground_match = (~mismatch) & both_foreground
    foreground = (individual >= 0) | (aggregate >= 0)
    total = int(individual.size)
    return {
        "total_pixels": total,
        "strongest_individual_vs_aggregate_label": {
            "agreement": float(np.mean(~mismatch)) if total else 1.0,
            "mismatch_pixels": int(np.count_nonzero(mismatch)),
            "both_foreground_mismatch_pixels": int(np.count_nonzero(foreground_mismatch)),
            "strongest_background_aggregate_foreground_pixels": int(
                np.count_nonzero((individual < 0) & (aggregate >= 0))
            ),
            "strongest_foreground_aggregate_background_pixels": int(
                np.count_nonzero((individual >= 0) & (aggregate < 0))
            ),
            "aggregate_top_minus_second_weight_on_foreground_mismatch": _finite_quantiles(margin[foreground_mismatch]),
            "aggregate_top_minus_second_weight_on_foreground_match": _finite_quantiles(margin[foreground_match]),
        },
        "distinct_contributing_label_histograms": {
            "processed_all_pixels": integer_histogram(processed_distinct),
            "processed_foreground_pixels": integer_histogram(processed_distinct[foreground]),
            "processed_rule_mismatch_pixels": integer_histogram(processed_distinct[mismatch]),
            "raster_positive_all_pixels": integer_histogram(raster_distinct),
            "raster_positive_rule_mismatch_pixels": integer_histogram(raster_distinct[mismatch]),
        },
        "cuda_accumulator_sizing": {
            "processed_until_native_transmittance_cutoff": accumulator_capacity_summary(processed_distinct, capacities),
            "conservative_all_positive_raster_contributors": accumulator_capacity_summary(raster_distinct, capacities),
        },
    }


def _validate_render_arrays(name: str, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("rgb", "alpha", "depth", "semantic")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{name} is missing render arrays: {', '.join(missing)}.")
    rgb = np.asarray(arrays["rgb"], dtype=np.float32)
    alpha = np.asarray(arrays["alpha"], dtype=np.float32)
    depth = np.asarray(arrays["depth"], dtype=np.float32)
    semantic = np.asarray(arrays["semantic"], dtype=np.int64)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"{name}.rgb must have shape [V,H,W,3], got {rgb.shape}.")
    expected = rgb.shape[:3]
    for key, array in (("alpha", alpha), ("depth", depth), ("semantic", semantic)):
        if array.shape != expected:
            raise ValueError(f"{name}.{key} must have shape {expected}, got {array.shape}.")
    if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
        raise ValueError(f"{name} contains non-finite RGB or alpha values.")
    valid_depth = arrays.get("valid_depth")
    if valid_depth is None:
        valid = np.isfinite(depth) & (depth > 0.0) & (alpha > 0.0)
    else:
        valid = np.asarray(valid_depth, dtype=bool)
        if valid.shape != expected:
            raise ValueError(f"{name}.valid_depth must have shape {expected}, got {valid.shape}.")
    return {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "semantic": semantic,
        "valid_depth": valid,
    }


def compare_render_arrays(
    *,
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compare RGB, alpha, depth, and semantic tensors without CUDA dependencies."""

    reference_arrays = _validate_render_arrays("reference", reference)
    candidate_arrays = _validate_render_arrays("candidate", candidate)
    if reference_arrays["rgb"].shape != candidate_arrays["rgb"].shape:
        raise ValueError(
            "Reference and candidate render shapes differ: "
            f"{reference_arrays['rgb'].shape} != {candidate_arrays['rgb'].shape}."
        )
    reference_rgb = np.clip(reference_arrays["rgb"], 0.0, 1.0).astype(np.float64)
    candidate_rgb = np.clip(candidate_arrays["rgb"], 0.0, 1.0).astype(np.float64)
    rgb_delta = candidate_rgb - reference_rgb
    rgb_mse = float(np.mean(np.square(rgb_delta)))
    valid_depth = reference_arrays["valid_depth"] & candidate_arrays["valid_depth"]
    valid_depth_count = int(np.count_nonzero(valid_depth))
    if valid_depth_count:
        depth_delta = np.abs(candidate_arrays["depth"][valid_depth] - reference_arrays["depth"][valid_depth]).astype(
            np.float64
        )
        depth_relative = depth_delta / np.maximum(
            np.abs(reference_arrays["depth"][valid_depth]).astype(np.float64),
            1.0e-8,
        )
        depth_mae: float | None = float(depth_delta.mean())
        depth_relative_error: float | None = float(depth_relative.mean())
        depth_max_relative_error: float | None = float(depth_relative.max())
    else:
        depth_mae = None
        depth_relative_error = None
        depth_max_relative_error = None
    semantic_equal = candidate_arrays["semantic"] == reference_arrays["semantic"]
    semantic_both_foreground = (candidate_arrays["semantic"] >= 0) & (reference_arrays["semantic"] >= 0)
    return {
        "rgb": {
            "identical": bool(rgb_mse == 0.0),
            "psnr_db": None if rgb_mse == 0.0 else float(10.0 * math.log10(1.0 / rgb_mse)),
            "mae": float(np.mean(np.abs(rgb_delta))),
            "max_abs_error": float(np.max(np.abs(rgb_delta))),
        },
        "alpha": {
            "mae": float(np.mean(np.abs(candidate_arrays["alpha"] - reference_arrays["alpha"]))),
            "max_abs_error": float(np.max(np.abs(candidate_arrays["alpha"] - reference_arrays["alpha"]))),
        },
        "depth": {
            "valid_pixels": valid_depth_count,
            "mae": depth_mae,
            "relative_error": depth_relative_error,
            "max_relative_error": depth_max_relative_error,
        },
        "semantic": {
            "agreement": float(np.mean(semantic_equal)),
            "mismatch_pixels": int(np.count_nonzero(~semantic_equal)),
            "both_foreground_mismatch_pixels": int(np.count_nonzero((~semantic_equal) & semantic_both_foreground)),
            "reference_background_candidate_foreground_pixels": int(
                np.count_nonzero((reference_arrays["semantic"] < 0) & (candidate_arrays["semantic"] >= 0))
            ),
            "reference_foreground_candidate_background_pixels": int(
                np.count_nonzero((reference_arrays["semantic"] >= 0) & (candidate_arrays["semantic"] < 0))
            ),
        },
    }


def _composite_render_arrays(
    composite: Mapping[str, np.ndarray],
    *,
    width: int,
    height: int,
    semantic_key: str,
) -> dict[str, np.ndarray]:
    pixel_count = width * height
    if np.asarray(composite["alpha"]).size != pixel_count:
        raise ValueError(
            "Composite pixel count does not match artifact dimensions: "
            f"{np.asarray(composite['alpha']).size} != {pixel_count}."
        )
    alpha = np.asarray(composite["alpha"], dtype=np.float32).reshape(1, height, width)
    depth = np.asarray(composite["depth"], dtype=np.float32).reshape(1, height, width)
    return {
        "rgb": np.asarray(composite["rgb"], dtype=np.float32).reshape(1, height, width, 3),
        "alpha": alpha,
        "depth": depth,
        "semantic": np.asarray(composite[semantic_key], dtype=np.int64).reshape(1, height, width),
        "valid_depth": np.isfinite(depth) & (depth > 0.0) & (alpha > 0.0),
    }


def _native_render_arrays(artifact: Mapping[str, Any]) -> dict[str, np.ndarray]:
    alpha = np.asarray(artifact["native_alpha"], dtype=np.float32)
    depth = np.asarray(artifact["native_depth"], dtype=np.float32)
    return {
        "rgb": np.asarray(artifact["native_rgb"], dtype=np.float32),
        "alpha": alpha,
        "depth": depth,
        "semantic": np.asarray(artifact["native_semantic"], dtype=np.int64),
        "valid_depth": np.isfinite(depth) & (depth > 0.0) & (alpha > 0.0),
    }


def _load_reference(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Render reference does not exist: {path}")
    from isaacsim_gaussian_renderer.fidelity import load_render_output

    output = load_render_output(path)
    arrays = {
        "rgb": output.rgb,
        "alpha": output.alpha,
        "depth": output.depth,
        "semantic": output.semantic,
    }
    if output.valid_depth is not None:
        arrays["valid_depth"] = output.valid_depth
    metadata = {
        "path": str(path.resolve()),
        "shape": list(output.rgb.shape),
        "color_space": output.color_space,
        "background": list(output.background) if output.background is not None else None,
        "camera_bundle_id": output.camera_bundle_id,
    }
    return arrays, metadata


def _named_comparison(
    *,
    reference_name: str,
    reference: Mapping[str, np.ndarray],
    candidate_name: str,
    candidate: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "metrics": compare_render_arrays(reference=reference, candidate=candidate),
    }


def validate_reference_contract(
    *,
    reference_name: str,
    reference_metadata: Mapping[str, Any],
    expected_shape: Sequence[int] | None,
    expected_camera_bundle_id: str | None,
    expected_color_space: str | None,
    expected_background: Sequence[float] | None,
) -> dict[str, Any]:
    """Reject known fidelity-contract mismatches and name unverifiable fields."""

    checks: dict[str, str] = {}
    observed_shape = tuple(int(value) for value in reference_metadata["shape"])
    if expected_shape is None:
        checks["shape"] = "not_checked_without_cuda_artifact"
    elif observed_shape != tuple(expected_shape):
        raise ValueError(f"{reference_name} shape {observed_shape} does not match expected {tuple(expected_shape)}.")
    else:
        checks["shape"] = "matched"

    for field, observed, expected in (
        (
            "camera_bundle_id",
            reference_metadata.get("camera_bundle_id"),
            expected_camera_bundle_id,
        ),
        ("color_space", reference_metadata.get("color_space"), expected_color_space),
        ("background", reference_metadata.get("background"), expected_background),
    ):
        if expected is None:
            checks[field] = "not_checked_without_expected_contract"
        elif observed is None:
            checks[field] = "unverified_missing_from_reference"
        elif field == "background":
            if not np.array_equal(
                np.asarray(observed, dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
            ):
                raise ValueError(f"{reference_name} background {observed!r} does not match {expected!r}.")
            checks[field] = "matched"
        elif observed != expected:
            raise ValueError(f"{reference_name} {field} {observed!r} does not match {expected!r}.")
        else:
            checks[field] = "matched"
    return checks


def analyze_artifact(
    artifact: Mapping[str, Any],
    *,
    capacities: Sequence[int] = DEFAULT_ACCUMULATOR_CAPACITIES,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Reconstruct native-cutoff and full exact-ray outputs from an artifact."""

    width = int(artifact["width"])
    height = int(artifact["height"])
    semantic_min_alpha = float(artifact["semantic_min_alpha"])
    native_threshold = float(artifact["transmittance_threshold"])
    contributions = {
        "pixel_offsets": artifact["pixel_offsets"],
        "alphas": artifact["contribution_alpha"],
        "depths": artifact["contribution_depth"],
        "colors": artifact["contribution_color"],
        "semantic_ids": artifact["contribution_semantic"],
    }
    native_python = composite_exact_contributions(
        **contributions,
        semantic_min_alpha=semantic_min_alpha,
        transmittance_threshold=native_threshold,
    )
    full_python = composite_exact_contributions(
        **contributions,
        semantic_min_alpha=0.0,
        transmittance_threshold=None,
    )
    renders = {
        "custom_cuda_native": _native_render_arrays(artifact),
        "python_native_cutoff_strongest_individual": _composite_render_arrays(
            native_python,
            width=width,
            height=height,
            semantic_key="strongest_individual_semantic",
        ),
        "python_native_cutoff_aggregate_label": _composite_render_arrays(
            native_python,
            width=width,
            height=height,
            semantic_key="aggregate_label_semantic",
        ),
        "python_full_exact_strongest_individual": _composite_render_arrays(
            full_python,
            width=width,
            height=height,
            semantic_key="strongest_individual_semantic",
        ),
        "python_full_exact_aggregate_label": _composite_render_arrays(
            full_python,
            width=width,
            height=height,
            semantic_key="aggregate_label_semantic",
        ),
    }
    comparisons = {
        "native_vs_python_native_cutoff_strongest_individual": _named_comparison(
            reference_name="python_native_cutoff_strongest_individual",
            reference=renders["python_native_cutoff_strongest_individual"],
            candidate_name="custom_cuda_native",
            candidate=renders["custom_cuda_native"],
        ),
        "native_vs_python_native_cutoff_aggregate_label": _named_comparison(
            reference_name="python_native_cutoff_aggregate_label",
            reference=renders["python_native_cutoff_aggregate_label"],
            candidate_name="custom_cuda_native",
            candidate=renders["custom_cuda_native"],
        ),
        "native_vs_python_full_exact_strongest_individual": _named_comparison(
            reference_name="python_full_exact_strongest_individual",
            reference=renders["python_full_exact_strongest_individual"],
            candidate_name="custom_cuda_native",
            candidate=renders["custom_cuda_native"],
        ),
        "native_vs_python_full_exact_aggregate_label": _named_comparison(
            reference_name="python_full_exact_aggregate_label",
            reference=renders["python_full_exact_aggregate_label"],
            candidate_name="custom_cuda_native",
            candidate=renders["custom_cuda_native"],
        ),
        "native_cutoff_vs_full_exact_strongest_individual": _named_comparison(
            reference_name="python_full_exact_strongest_individual",
            reference=renders["python_full_exact_strongest_individual"],
            candidate_name="python_native_cutoff_strongest_individual",
            candidate=renders["python_native_cutoff_strongest_individual"],
        ),
    }
    analysis = {
        "scene": SCENE_NAME,
        "width": width,
        "height": height,
        "contribution_count": int(np.asarray(artifact["contribution_alpha"]).size),
        "native_semantic_min_alpha": semantic_min_alpha,
        "native_transmittance_threshold": native_threshold,
        "python_model_provenance": {
            "source": "experiments/ovrtx_exact_ray_model.py",
            "equations": (
                "alpha_i=opacity_i*exp(-0.5*minimum_ray_mahalanobis_squared); "
                "front-to-back Bernoulli first-hit weights; center-Z depth"
            ),
            "full_model_semantic_min_alpha": 0.0,
        },
        "strongest_individual_vs_aggregate_label": {
            "native_cutoff": summarize_semantic_rules(native_python, capacities=capacities),
            "full_exact_python_model": summarize_semantic_rules(full_python, capacities=capacities),
        },
        "comparisons": comparisons,
    }
    return analysis, renders


def _artifact_scalar(data: Any, key: str) -> Any:
    if key not in data:
        raise ValueError(f"Artifact is missing scalar {key!r}.")
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"Artifact scalar {key!r} has shape {value.shape}.")
    return value.item()


def load_artifact(path: Path) -> dict[str, Any]:
    """Load and validate a portable native exact-semantics capture."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        schema_version = str(_artifact_scalar(data, "schema_version"))
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema {schema_version!r}.")
        missing = [key for key in REQUIRED_ARTIFACT_ARRAYS if key not in data]
        if missing:
            raise ValueError(f"Artifact is missing arrays: {', '.join(missing)}.")
        artifact: dict[str, Any] = {key: np.asarray(data[key]) for key in REQUIRED_ARTIFACT_ARRAYS}
        artifact.update(
            {
                "schema_version": schema_version,
                "width": int(_artifact_scalar(data, "width")),
                "height": int(_artifact_scalar(data, "height")),
                "semantic_min_alpha": float(_artifact_scalar(data, "semantic_min_alpha")),
                "transmittance_threshold": float(_artifact_scalar(data, "transmittance_threshold")),
                "metadata": json.loads(str(_artifact_scalar(data, "metadata_json"))),
            }
        )
    width = artifact["width"]
    height = artifact["height"]
    if width <= 0 or height <= 0:
        raise ValueError("Artifact width and height must be positive.")
    _validate_contributions(
        pixel_offsets=artifact["pixel_offsets"],
        alphas=artifact["contribution_alpha"],
        depths=artifact["contribution_depth"],
        colors=artifact["contribution_color"],
        semantic_ids=artifact["contribution_semantic"],
    )
    if artifact["pixel_offsets"].shape != (width * height + 1,):
        raise ValueError("Artifact pixel offsets do not match width * height.")
    _validate_render_arrays("artifact.native", _native_render_arrays(artifact))
    return artifact


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cuda_status() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on workstation environment.
        return False, f"torch import failed: {type(exc).__name__}: {exc}"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false"
    return True, f"CUDA available on {torch.cuda.get_device_name(0)}"


def _capture_sorted_contributions(
    *,
    backend: Any,
    intrinsics: Any,
    counters: Mapping[str, int],
    width: int,
    height: int,
    support_sigma: float,
) -> dict[str, np.ndarray]:
    """Copy exact-ray contribution inputs from a completed tile-size-1 render."""

    import torch

    pixel_count = width * height
    starts = backend.workspace["tile_starts"][:pixel_count].detach().cpu().numpy()
    ends = backend.workspace["tile_ends"][:pixel_count].detach().cpu().numpy()
    lengths = np.where((starts >= 0) & (ends > starts), ends - starts, 0).astype(np.int64, copy=False)
    intersection_count = int(counters["tile_intersections"])
    expected_starts = np.cumsum(lengths, dtype=np.int64) - lengths
    nonempty = lengths > 0
    if int(lengths.sum()) != intersection_count or (
        np.any(nonempty) and not np.array_equal(starts[nonempty], expected_starts[nonempty])
    ):
        raise RuntimeError("Sorted tile ranges do not form a contiguous cover of measured intersections.")
    offsets = np.empty(pixel_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    pixel_ids_cpu = np.repeat(np.arange(pixel_count, dtype=np.int64), lengths)
    pixel_ids = torch.from_numpy(pixel_ids_cpu).to(device=intrinsics.device)
    visible_indices = backend.workspace["values_out"][:intersection_count].to(torch.int64)
    opacities = backend.workspace["visible_opacities"].index_select(0, visible_indices)
    precisions = backend.workspace["visible_ray_precisions"].index_select(0, visible_indices)
    precision_means = backend.workspace["visible_ray_precision_means"].index_select(0, visible_indices)
    pixel_x = pixel_ids.remainder(width).to(torch.float32) + 0.5
    pixel_y = torch.div(pixel_ids, width, rounding_mode="floor").to(torch.float32) + 0.5
    ray_x = (pixel_x - intrinsics[0, 2]) / intrinsics[0, 0]
    ray_y = (pixel_y - intrinsics[1, 2]) / intrinsics[1, 1]
    ray_precision_ray = (
        precisions[:, 0] * ray_x.square()
        + 2.0 * precisions[:, 1] * ray_x * ray_y
        + 2.0 * precisions[:, 2] * ray_x
        + precisions[:, 3] * ray_y.square()
        + 2.0 * precisions[:, 4] * ray_y
        + precisions[:, 5]
    )
    ray_precision_mean = precision_means[:, 0] * ray_x + precision_means[:, 1] * ray_y + precision_means[:, 2]
    valid_precision = ray_precision_ray > 0.0
    safe_ray_precision_ray = torch.where(valid_precision, ray_precision_ray, torch.ones_like(ray_precision_ray))
    mahalanobis_squared = (precision_means[:, 3] - ray_precision_mean.square() / safe_ray_precision_ray).clamp_min(0.0)
    alpha = torch.minimum(
        torch.ones_like(opacities),
        opacities * torch.exp(-0.5 * mahalanobis_squared),
    )
    alpha = torch.where(
        valid_precision & (mahalanobis_squared <= support_sigma * support_sigma),
        alpha,
        torch.zeros_like(alpha),
    )
    gaussian_ids = backend.workspace["visible_gaussian_ids"].index_select(0, visible_indices).to(torch.int64)
    return {
        "pixel_offsets": offsets,
        "contribution_alpha": alpha.detach().cpu().numpy().astype(np.float32, copy=False),
        "contribution_depth": backend.workspace["visible_depths"]
        .index_select(0, visible_indices)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "contribution_color": backend._packed_scene["features"]
        .index_select(0, gaussian_ids)[:, :3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "contribution_semantic": backend._packed_scene["semantic_ids"]
        .index_select(0, gaussian_ids)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False),
    }


def render_synthetic_small(args: argparse.Namespace, artifact_path: Path) -> dict[str, Any]:
    """Render synthetic-small through the production exact-ray CUDA backend."""

    import torch

    from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
    from isaacsim_gaussian_renderer.benchmark_manifest import (
        camera_bundle,
        synthetic_scene_manifest,
        synthetic_scene_tensors,
    )
    from isaacsim_gaussian_renderer.fidelity import bundle_from_tensors

    device = torch.device("cuda:0")
    scene = synthetic_scene_tensors(GAUSSIAN_COUNT, seed=SCENE_SEED, device=device)
    scene_manifest = synthetic_scene_manifest(SCENE_NAME, GAUSSIAN_COUNT, SCENE_SEED)
    cameras = camera_bundle(1, args.width, args.height, device=device)
    backend = CustomCudaBackend(
        max_visible_records=args.visible_capacity,
        max_intersections=args.intersection_capacity,
        near_plane=float(cameras.manifest["near_plane"]),
        far_plane=float(cameras.manifest["far_plane"]),
        gaussian_support_sigma=args.support_sigma,
        covariance_epsilon=0.0,
        semantic_min_alpha=args.semantic_min_alpha,
        ray_gaussian_evaluation=True,
        tile_size=1,
        depth_bucket_count=args.depth_bucket_count,
        depth_bucket_group_size=args.depth_bucket_group_size,
        output_srgb=False,
        deterministic=True,
    )
    service = RendererService(backend, height=args.height, width=args.width, max_views=1)
    initialized = False
    try:
        service.initialize(stage=None, device=device)
        initialized = True
        service.load_scene(
            SCENE_ID,
            means=scene["means"],
            scales=scene["scales"],
            rotations=scene["quats"],
            opacities=scene["opacities"],
            features=scene["colors"],
            semantic_ids=scene["semantic_ids"].to(torch.int64),
        )
        camera_scene_ids = torch.tensor([SCENE_ID], device=device, dtype=torch.int64)
        outputs = service.render(
            cameras.viewmats,
            cameras.intrinsics,
            camera_scene_ids,
        )
        service.synchronize()
        counters = backend.check_capacity(synchronize=False)
        alpha_output = outputs["alpha"][..., 0]
        depth_output = outputs["depth"][..., 0]
        semantic_output = outputs["semantic_id"][..., 0]
        foreground = alpha_output > 0.0
        semantic_foreground = foreground if args.semantic_min_alpha == 0.0 else alpha_output >= args.semantic_min_alpha
        semantic_background = ~semantic_foreground
        expected_shapes = {
            "rgb": (1, args.height, args.width, 3),
            "depth": (1, args.height, args.width, 1),
            "alpha": (1, args.height, args.width, 1),
            "semantic_id": (1, args.height, args.width, 1),
        }
        output_assertions = {
            "all_outputs_cuda_resident": all(tensor.is_cuda for tensor in outputs.values()),
            "all_outputs_contiguous": all(tensor.is_contiguous() for tensor in outputs.values()),
            "shapes_match": all(tuple(outputs[name].shape) == shape for name, shape in expected_shapes.items()),
            "dtypes_match": (
                outputs["rgb"].dtype == torch.float32
                and outputs["depth"].dtype == torch.float32
                and outputs["alpha"].dtype == torch.float32
                and outputs["semantic_id"].dtype == torch.int64
            ),
            "rgb_finite": bool(torch.isfinite(outputs["rgb"]).all().item()),
            "alpha_finite_and_bounded": bool(
                (torch.isfinite(outputs["alpha"]) & (outputs["alpha"] >= 0.0) & (outputs["alpha"] <= 1.0)).all().item()
            ),
            "foreground_nonempty": bool(torch.count_nonzero(foreground).item()),
            "foreground_depth_finite": bool(torch.isfinite(depth_output[foreground]).all().item()),
            "background_depth_positive_infinity": bool(
                (torch.isinf(depth_output[~foreground]) & (depth_output[~foreground] > 0.0)).all().item()
            ),
            "foreground_semantics_in_scene_range": bool(
                ((semantic_output[semantic_foreground] >= 0) & (semantic_output[semantic_foreground] < 1024))
                .all()
                .item()
            ),
            "background_semantics_minus_one": bool((semantic_output[semantic_background] == -1).all().item()),
            "zero_visible_overflow": counters["visible_overflow"] == 0,
            "zero_intersection_overflow": counters["intersection_overflow"] == 0,
        }
        if not all(output_assertions.values()):
            raise RuntimeError(f"Native output contract failed: {output_assertions}")
        contributions = _capture_sorted_contributions(
            backend=backend,
            intrinsics=cameras.intrinsics[0],
            counters=counters,
            width=args.width,
            height=args.height,
            support_sigma=args.support_sigma,
        )
        fidelity_bundle = bundle_from_tensors(
            viewmats=cameras.viewmats,
            intrinsics=cameras.intrinsics,
            width=args.width,
            height=args.height,
            color_space="display_srgb",
            scene_ids=camera_scene_ids,
            scene_checksum=scene_manifest["checksum_sha256"],
        )
        metadata = {
            "scene": scene_manifest,
            "camera": cameras.manifest,
            "fidelity_camera_bundle": fidelity_bundle.to_json(),
            "configuration": {
                "backend": "CustomCudaBackend",
                "ray_gaussian_evaluation": True,
                "tile_size": 1,
                "deterministic": True,
                "output_srgb": False,
                "color_space": "display_srgb",
                "color_transform": "identity_authored_display_rgb",
                "support_sigma": args.support_sigma,
                "semantic_min_alpha": args.semantic_min_alpha,
                "visible_capacity": args.visible_capacity,
                "intersection_capacity": args.intersection_capacity,
                "depth_bucket_count": args.depth_bucket_count,
                "depth_bucket_group_size": args.depth_bucket_group_size,
            },
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "git_commit": _git_commit(),
            },
            "projection_counters": dict(counters),
            "output_assertions": output_assertions,
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            schema_version=np.asarray(ARTIFACT_SCHEMA_VERSION),
            width=np.asarray(args.width, dtype=np.int64),
            height=np.asarray(args.height, dtype=np.int64),
            semantic_min_alpha=np.asarray(args.semantic_min_alpha, dtype=np.float64),
            transmittance_threshold=np.asarray(NATIVE_TRANSMITTANCE_THRESHOLD, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
            **contributions,
            native_rgb=outputs["rgb"].detach().cpu().numpy().astype(np.float32, copy=False),
            native_alpha=alpha_output.detach().cpu().numpy().astype(np.float32, copy=False),
            native_depth=depth_output.detach().cpu().numpy().astype(np.float32, copy=False),
            native_semantic=semantic_output.detach().cpu().numpy().astype(np.int64, copy=False),
        )
    finally:
        if initialized:
            service.shutdown()
    return load_artifact(artifact_path)


def analysis_only_self_check(*, capacities: Sequence[int] = DEFAULT_ACCUMULATOR_CAPACITIES) -> dict[str, Any]:
    """Exercise pure analysis helpers when no remote CUDA artifact exists."""

    composite = composite_exact_contributions(
        pixel_offsets=np.asarray([0, 3, 3, 5], dtype=np.int64),
        alphas=np.asarray([0.35, 0.30, 0.40, 0.70, 0.50], dtype=np.float32),
        depths=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        colors=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        semantic_ids=np.asarray([10, 20, 20, 4, 3], dtype=np.int64),
        semantic_min_alpha=0.0,
        transmittance_threshold=None,
    )
    summary = summarize_semantic_rules(composite, capacities=capacities)
    observed = summary["strongest_individual_vs_aggregate_label"]["mismatch_pixels"]
    return {
        "fixture_only_not_renderer_evidence": True,
        "expected_rule_mismatch_pixels": 1,
        "observed_rule_mismatch_pixels": observed,
        "pass": observed == 1,
        "semantic_summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "render", "analysis-only"),
        default="auto",
        help=(
            "render requires CUDA; analysis-only never initializes CUDA; auto analyzes an "
            "existing artifact, otherwise renders when CUDA is available and falls back to "
            "the local analysis self-check."
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help=("Portable capture to read or write. Render mode defaults to <output-stem>.artifact.npz beside --output."),
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON diagnostic report.")
    parser.add_argument(
        "--exact-ray-reference",
        type=Path,
        default=None,
        help=("Optional canonical .npz/.json render output produced by the existing exact-ray Python model."),
    )
    parser.add_argument(
        "--ovrtx-reference",
        type=Path,
        default=None,
        help="Optional canonical temporally accumulated OVRTX .npz/.json render output.",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument("--semantic-min-alpha", type=float, default=0.01)
    parser.add_argument("--visible-capacity", type=int, default=5_000)
    parser.add_argument("--intersection-capacity", type=int, default=500_000)
    parser.add_argument("--depth-bucket-count", type=int, default=1024)
    parser.add_argument("--depth-bucket-group-size", type=int, default=32)
    parser.add_argument(
        "--accumulator-capacities",
        type=positive_int_list,
        default=DEFAULT_ACCUMULATOR_CAPACITIES,
        help="Candidate fixed accumulator slots per pixel.",
    )
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive.")
    if not math.isfinite(args.support_sigma) or args.support_sigma <= 0.0:
        parser.error("--support-sigma must be finite and positive.")
    if not math.isfinite(args.semantic_min_alpha) or not 0.0 <= args.semantic_min_alpha <= 1.0:
        parser.error("--semantic-min-alpha must be finite and in [0, 1].")
    for name in (
        "visible_capacity",
        "intersection_capacity",
        "depth_bucket_count",
        "depth_bucket_group_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    return args


def _resolve_mode(args: argparse.Namespace) -> tuple[str, Path | None, str]:
    artifact_exists = args.artifact is not None and args.artifact.is_file()
    if args.mode == "analysis-only":
        return "analysis-only", args.artifact if artifact_exists else None, "explicit analysis-only"
    if args.mode == "render":
        artifact_path = args.artifact or args.output.with_name(f"{args.output.stem}.artifact.npz")
        return "render", artifact_path, "explicit render"
    if artifact_exists:
        return "analysis-only", args.artifact, "auto selected existing artifact"
    cuda_available, cuda_reason = _cuda_status()
    if cuda_available:
        artifact_path = args.artifact or args.output.with_name(f"{args.output.stem}.artifact.npz")
        return "render", artifact_path, f"auto selected render: {cuda_reason}"
    return "analysis-only", None, f"auto selected local analysis: {cuda_reason}"


def main() -> None:
    args = parse_args()
    resolved_mode, artifact_path, mode_reason = _resolve_mode(args)
    if resolved_mode == "render":
        cuda_available, cuda_reason = _cuda_status()
        if not cuda_available:
            raise RuntimeError(f"Render mode requires CUDA: {cuda_reason}")
        assert artifact_path is not None
        artifact = render_synthetic_small(args, artifact_path)
        artifact_status = "rendered"
    elif artifact_path is not None:
        artifact = load_artifact(artifact_path)
        artifact_status = "loaded"
    else:
        artifact = None
        artifact_status = "absent"

    reference_arrays: dict[str, dict[str, np.ndarray]] = {}
    references: dict[str, dict[str, Any]] = {}
    for name, path in (
        ("exact_ray_python_reference", args.exact_ray_reference),
        ("ovrtx_temporal_reference", args.ovrtx_reference),
    ):
        if path is None:
            references[name] = {"status": "not_provided"}
            continue
        arrays, metadata = _load_reference(path)
        reference_arrays[name] = arrays
        references[name] = {"status": "loaded", **metadata}

    if artifact is not None:
        fidelity_contract = artifact.get("metadata", {}).get("fidelity_camera_bundle", {})
        expected_shape: tuple[int, int, int, int] | None = (
            1,
            int(artifact["height"]),
            int(artifact["width"]),
            3,
        )
        expected_camera_bundle_id = fidelity_contract.get("bundle_id")
        expected_color_space = fidelity_contract.get("color_space", "display_srgb")
        expected_background = fidelity_contract.get("background", [0.0, 0.0, 0.0])
    else:
        expected_shape = None
        expected_camera_bundle_id = None
        expected_color_space = None
        expected_background = None
    for name, metadata in references.items():
        if metadata["status"] != "loaded":
            continue
        metadata["contract_validation"] = validate_reference_contract(
            reference_name=name,
            reference_metadata=metadata,
            expected_shape=expected_shape,
            expected_camera_bundle_id=expected_camera_bundle_id,
            expected_color_space=expected_color_space,
            expected_background=expected_background,
        )

    loaded_reference_metadata = [metadata for metadata in references.values() if metadata["status"] == "loaded"]
    if len(loaded_reference_metadata) > 1:
        first_bundle_id = loaded_reference_metadata[0].get("camera_bundle_id")
        for metadata in loaded_reference_metadata[1:]:
            next_bundle_id = metadata.get("camera_bundle_id")
            if first_bundle_id is not None and next_bundle_id is not None and first_bundle_id != next_bundle_id:
                raise ValueError("Exact-ray and OVRTX references use different camera bundles.")

    comparisons: dict[str, Any] = {}
    if artifact is not None:
        artifact_analysis, renders = analyze_artifact(artifact, capacities=args.accumulator_capacities)
        comparisons.update(artifact_analysis.pop("comparisons"))
        for reference_name, reference in reference_arrays.items():
            for candidate_name in (
                "custom_cuda_native",
                "python_full_exact_strongest_individual",
                "python_full_exact_aggregate_label",
            ):
                comparisons[f"{candidate_name}_vs_{reference_name}"] = _named_comparison(
                    reference_name=reference_name,
                    reference=reference,
                    candidate_name=candidate_name,
                    candidate=renders[candidate_name],
                )
        analysis_payload: dict[str, Any] = artifact_analysis
        self_check = None
    else:
        analysis_payload = {
            "status": "no_cuda_artifact",
            "message": (
                "No native CUDA artifact was available; renderer comparisons and "
                "synthetic-small accumulator histograms were not evaluated."
            ),
        }
        self_check = analysis_only_self_check(capacities=args.accumulator_capacities)

    if {
        "exact_ray_python_reference",
        "ovrtx_temporal_reference",
    }.issubset(reference_arrays):
        comparisons["exact_ray_python_reference_vs_ovrtx_temporal_reference"] = _named_comparison(
            reference_name="ovrtx_temporal_reference",
            reference=reference_arrays["ovrtx_temporal_reference"],
            candidate_name="exact_ray_python_reference",
            candidate=reference_arrays["exact_ray_python_reference"],
        )

    pass_status = self_check is None or bool(self_check["pass"])
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "diagnostic_only": True,
        "acceptance_gate": False,
        "pass": pass_status,
        "failure_reason": None if pass_status else "analysis-only helper self-check failed",
        "mode": {
            "requested": args.mode,
            "resolved": resolved_mode,
            "reason": mode_reason,
        },
        "artifact": {
            "status": artifact_status,
            "path": str(artifact_path.resolve()) if artifact_path is not None else None,
            "schema_version": artifact.get("schema_version") if artifact is not None else None,
            "metadata": artifact.get("metadata") if artifact is not None else None,
        },
        "references": references,
        "analysis": analysis_payload,
        "comparisons": comparisons,
        "analysis_only_self_check": self_check,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    print(SUCCESS_MARKER)


if __name__ == "__main__":
    main()
