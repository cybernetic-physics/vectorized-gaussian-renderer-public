from __future__ import annotations

import numpy as np
import pytest

from experiments.native_exact_semantics_diagnostic import (
    accumulator_capacity_summary,
    analyze_artifact,
    compare_render_arrays,
    composite_exact_contributions,
    summarize_semantic_rules,
)


def test_native_exact_semantics_distinguish_strongest_from_aggregate_label() -> None:
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

    assert composite["strongest_individual_semantic"].tolist() == [10, -1, 4]
    assert composite["aggregate_label_semantic"].tolist() == [20, -1, 4]
    assert composite["processed_distinct_label_count"].tolist() == [2, 0, 2]
    assert composite["depth"][1] == np.inf

    summary = summarize_semantic_rules(composite, capacities=(1, 2))
    difference = summary["strongest_individual_vs_aggregate_label"]
    assert difference["mismatch_pixels"] == 1
    assert difference["both_foreground_mismatch_pixels"] == 1
    assert summary["distinct_contributing_label_histograms"]["processed_all_pixels"] == {
        "0": 1,
        "2": 2,
    }
    sizing = summary["cuda_accumulator_sizing"]["processed_until_native_transmittance_cutoff"]
    assert sizing["minimum_capacity_for_zero_overflow"] == 2
    assert sizing["candidate_capacities"] == [
        {
            "capacity": 1,
            "pixels_exceeding_capacity": 2,
            "fraction_exceeding_capacity": pytest.approx(2.0 / 3.0),
        },
        {
            "capacity": 2,
            "pixels_exceeding_capacity": 0,
            "fraction_exceeding_capacity": 0.0,
        },
    ]


def test_native_exact_semantics_cutoff_reports_conservative_omitted_labels() -> None:
    contributions = {
        "pixel_offsets": np.asarray([0, 3], dtype=np.int64),
        "alphas": np.asarray([0.99995, 0.50, 0.50], dtype=np.float32),
        "depths": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "colors": np.ones((3, 3), dtype=np.float32),
        "semantic_ids": np.asarray([1, 2, 3], dtype=np.int64),
        "semantic_min_alpha": 0.0,
    }
    native_cutoff = composite_exact_contributions(
        **contributions,
        transmittance_threshold=1.0e-4,
    )
    full = composite_exact_contributions(
        **contributions,
        transmittance_threshold=None,
    )

    assert native_cutoff["processed_contribution_count"].tolist() == [1]
    assert native_cutoff["processed_distinct_label_count"].tolist() == [1]
    assert native_cutoff["raster_distinct_label_count"].tolist() == [3]
    assert native_cutoff["omitted_positive_contribution_count"].tolist() == [2]
    assert native_cutoff["omitted_distinct_label_count"].tolist() == [2]
    assert full["processed_contribution_count"].tolist() == [3]


def test_native_exact_semantics_capacity_summary_uses_integer_higher_quantiles() -> None:
    summary = accumulator_capacity_summary(
        np.asarray([0, 1, 2, 2, 5], dtype=np.int64),
        capacities=(4, 1, 2, 2),
    )

    assert summary["histogram"] == {"0": 1, "1": 1, "2": 2, "5": 1}
    assert summary["maximum"] == 5
    assert summary["minimum_capacity_for_zero_overflow"] == 5
    assert summary["percentiles_higher"]["p50"] == 2
    assert [row["capacity"] for row in summary["candidate_capacities"]] == [1, 2, 4]
    assert [row["pixels_exceeding_capacity"] for row in summary["candidate_capacities"]] == [
        3,
        1,
        1,
    ]


def test_native_exact_semantics_render_comparison_covers_all_outputs() -> None:
    reference = {
        "rgb": np.zeros((1, 1, 2, 3), dtype=np.float32),
        "alpha": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "depth": np.asarray([[[2.0, np.inf]]], dtype=np.float32),
        "semantic": np.asarray([[[7, -1]]], dtype=np.int64),
    }
    candidate = {
        "rgb": np.asarray([[[[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32),
        "alpha": np.asarray([[[1.0, 1.0]]], dtype=np.float32),
        "depth": np.asarray([[[3.0, 99.0]]], dtype=np.float32),
        "semantic": np.asarray([[[8, 9]]], dtype=np.int64),
    }

    metrics = compare_render_arrays(reference=reference, candidate=candidate)

    assert metrics["rgb"]["identical"] is False
    assert metrics["rgb"]["mae"] == pytest.approx(0.05)
    assert metrics["alpha"]["mae"] == pytest.approx(0.5)
    assert metrics["depth"]["valid_pixels"] == 1
    assert metrics["depth"]["relative_error"] == pytest.approx(0.5)
    assert metrics["semantic"]["agreement"] == pytest.approx(0.0)
    assert metrics["semantic"]["mismatch_pixels"] == 2
    assert metrics["semantic"]["both_foreground_mismatch_pixels"] == 1
    assert metrics["semantic"]["reference_background_candidate_foreground_pixels"] == 1


def test_native_exact_semantics_artifact_analysis_compares_native_and_full_model() -> None:
    contribution_data = {
        "pixel_offsets": np.asarray([0, 3, 3], dtype=np.int64),
        "alphas": np.asarray([0.35, 0.30, 0.40], dtype=np.float32),
        "depths": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "colors": np.eye(3, dtype=np.float32),
        "semantic_ids": np.asarray([10, 20, 20], dtype=np.int64),
    }
    native_model = composite_exact_contributions(
        **contribution_data,
        semantic_min_alpha=0.01,
        transmittance_threshold=1.0e-4,
    )
    artifact = {
        "width": 2,
        "height": 1,
        "semantic_min_alpha": 0.01,
        "transmittance_threshold": 1.0e-4,
        "pixel_offsets": contribution_data["pixel_offsets"],
        "contribution_alpha": contribution_data["alphas"],
        "contribution_depth": contribution_data["depths"],
        "contribution_color": contribution_data["colors"],
        "contribution_semantic": contribution_data["semantic_ids"],
        "native_rgb": native_model["rgb"].reshape(1, 1, 2, 3),
        "native_alpha": native_model["alpha"].reshape(1, 1, 2),
        "native_depth": native_model["depth"].reshape(1, 1, 2),
        "native_semantic": native_model["strongest_individual_semantic"].reshape(1, 1, 2),
    }

    analysis, renders = analyze_artifact(artifact, capacities=(1, 2))

    native_parity = analysis["comparisons"]["native_vs_python_native_cutoff_strongest_individual"]["metrics"]
    aggregate_comparison = analysis["comparisons"]["native_vs_python_full_exact_aggregate_label"]["metrics"]
    assert native_parity["rgb"]["identical"] is True
    assert native_parity["semantic"]["agreement"] == 1.0
    assert aggregate_comparison["semantic"]["mismatch_pixels"] == 1
    assert renders["python_full_exact_aggregate_label"]["semantic"].tolist() == [[[20, -1]]]
