from __future__ import annotations

import sys

import numpy as np
import pytest

from benchmarks import summarize_flashgs_matched
from benchmarks.summarize_flashgs_matched import (
    apply_matrix_claim_gates,
    expected_matrix_capacity_calibration_schedule,
    interpret_results,
    markdown_table,
    validated_timing_distribution,
)


def test_expected_matrix_capacity_schedule_runs_memory_risk_rows_first() -> None:
    assert expected_matrix_capacity_calibration_schedule((1, 8, 32, 64, 128, 256, 512, 1024)) == [
        {"batch": 1024, "renderer": "custom"},
        {"batch": 512, "renderer": "custom"},
        {"batch": 1, "renderer": "custom"},
        {"batch": 8, "renderer": "custom"},
        {"batch": 32, "renderer": "custom"},
        {"batch": 64, "renderer": "custom"},
        {"batch": 128, "renderer": "custom"},
        {"batch": 256, "renderer": "custom"},
    ]


def row(
    speedup: float,
    *,
    fidelity: bool = True,
    fairness: bool = True,
) -> dict[str, object]:
    return {
        "speedup_custom_over_flashgs": speedup,
        "gpu_speedup_observed_same_step_min": speedup * 0.95,
        "gpu_speedup_observed_same_step_max": speedup * 1.05,
        "wall_speedup_observed_same_step_min": speedup * 0.95,
        "wall_speedup_observed_same_step_max": speedup * 1.05,
        "fidelity_pass": fidelity,
        "fairness_pass": fairness,
    }


def test_interpretation_allows_full_claim_only_when_full_fidelity_passes() -> None:
    interpretation = interpret_results([row(3.0)], [row(2.0)])

    assert interpretation["verdict"] == "custom-wins-rgb-and-full-sensor"
    assert interpretation["full_sensor_fidelity_pass"] is True
    assert interpretation["rgb_fidelity_pass"] is True
    assert "direct B1-B256" in interpretation["supported_claim"]
    assert "P128x4 at B512" in interpretation["supported_claim"]


def test_markdown_table_discloses_physical_batch_and_native_submissions() -> None:
    table = markdown_table(
        [
            {
                "batch": 512,
                "custom_physical_batch": 128,
                "custom_native_submissions_per_logical_batch": 4,
                "flashgs_physical_batch": 1,
                "flashgs_native_submissions_per_logical_batch": 512,
                "custom_ms": 1.0,
                "flashgs_ms": 2.0,
                "gpu_speedup_ratio_of_mean_latency": 2.0,
                "gpu_speedup_observed_same_step_min": 1.9,
                "gpu_speedup_observed_same_step_max": 2.1,
                "wall_speedup_observed_same_step_min": 1.8,
                "wall_speedup_observed_same_step_max": 2.2,
                "custom_steady_process_gpu_gb": 3.0,
                "flashgs_steady_process_gpu_gb": 4.0,
                "fidelity_pass": True,
            }
        ]
    )

    assert "Custom P/submits" in table
    assert "128/4" in table
    assert "1/512" in table


def test_interpretation_gates_full_claim_on_full_sensor_fidelity() -> None:
    interpretation = interpret_results(
        [row(3.0, fidelity=False)],
        [row(2.0)],
    )

    assert interpretation["raw_performance_verdict"] == ("custom-wins-rgb-and-full-sensor")
    assert interpretation["verdict"] == ("custom-wins-rgb-full-sensor-fidelity-inconclusive")
    assert "validated RGB-only" in interpretation["supported_claim"]
    assert "not claim-eligible" in interpretation["supported_claim"]


def test_interpretation_gates_every_claim_on_rgb_fidelity() -> None:
    interpretation = interpret_results(
        [row(3.0)],
        [row(2.0, fidelity=False)],
    )

    assert interpretation["verdict"] == ("rgb-and-full-sensor-fidelity-inconclusive")
    assert interpretation["rgb_fidelity_pass"] is False


def test_interpretation_gates_every_claim_on_rgb_fairness() -> None:
    interpretation = interpret_results(
        [row(3.0)],
        [row(2.0, fairness=False)],
    )

    assert interpretation["verdict"] == ("rgb-and-full-sensor-fairness-inconclusive")
    assert interpretation["rgb_fairness_pass"] is False
    assert "matched-run fairness" in interpretation["supported_claim"]


def test_interpretation_gates_full_claim_on_full_fairness() -> None:
    interpretation = interpret_results(
        [row(3.0, fairness=False)],
        [row(2.0)],
    )

    assert interpretation["verdict"] == ("custom-wins-rgb-full-sensor-fairness-inconclusive")
    assert interpretation["rgb_claim_eligible"] is True
    assert interpretation["full_sensor_claim_eligible"] is False


def test_interpretation_does_not_claim_a_winner_when_observed_range_crosses_one() -> None:
    ambiguous = row(1.0001)
    ambiguous["gpu_speedup_observed_same_step_min"] = 0.99
    ambiguous["gpu_speedup_observed_same_step_max"] = 1.01

    interpretation = interpret_results([ambiguous], [ambiguous])

    assert interpretation["verdict"] == "mixed-by-batch"
    assert "no uniform" in interpretation["supported_claim"]


def test_interpretation_requires_wall_and_gpu_wins() -> None:
    gpu_only = row(2.0)
    gpu_only["wall_speedup_observed_same_step_min"] = 0.9
    gpu_only["wall_speedup_observed_same_step_max"] = 1.1

    interpretation = interpret_results([gpu_only], [gpu_only])

    assert interpretation["verdict"] == "mixed-by-batch"


def test_timing_distribution_rejects_stored_aggregate_inconsistent_with_samples() -> None:
    samples = np.arange(1.0, 101.0, dtype=np.float64)
    run = {
        "timing": {
            "gpu_batch_ms": {
                "count": 100,
                "mean": float(samples.mean()) + 1.0,
                "stddev": float(samples.std(ddof=1)),
                "min": float(samples.min()),
                "p50": float(np.percentile(samples, 50)),
                "p95": float(np.percentile(samples, 95)),
                "p99": float(np.percentile(samples, 99)),
                "max": float(samples.max()),
                "samples": samples.tolist(),
            }
        }
    }

    with pytest.raises(ValueError, match="does not match raw-sample"):
        validated_timing_distribution(run, "gpu_batch_ms", label="fixture")


def test_global_matrix_failure_suppresses_renderer_winner_claim() -> None:
    verdict, claim = apply_matrix_claim_gates(
        verdict="custom-wins-rgb-and-full-sensor",
        supported_claim="The custom renderer wins.",
        exact_primary_batches=True,
        matrix_failures=["source identity differs"],
        scientific_pass=False,
        headline_eligible=False,
        hardware_scope={"gpu_name": "RTX 3090", "gpu_uuid": "GPU-test"},
    )

    assert verdict == "invalid-matrix-no-primary-verdict"
    assert "No renderer-performance claim" in claim
    assert "custom renderer wins" not in claim


def test_partial_winner_on_other_gpu_is_hardware_scoped_nonheadline() -> None:
    verdict, claim = apply_matrix_claim_gates(
        verdict="custom-wins-rgb-full-sensor-fidelity-inconclusive",
        supported_claim="The custom rasterizer outperforms FlashGS for RGB.",
        exact_primary_batches=True,
        matrix_failures=[],
        scientific_pass=False,
        headline_eligible=False,
        hardware_scope={
            "gpu_name": "NVIDIA L4",
            "gpu_uuid": "GPU-test",
        },
    )

    assert verdict == "custom-wins-rgb-full-sensor-fidelity-inconclusive"
    assert claim.startswith("Hardware-scoped non-headline partial result on NVIDIA L4 (GPU-test):")
    assert (
        "This does not establish the frozen NVIDIA L4 (GPU-b3c9268d-2b06-d924-90cc-d2171c86ef34) headline result."
    ) in claim


def test_verify_existing_cli_does_not_rewrite_summary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / "summary.json"
    sentinel = b"existing-summary-bytes\n"
    summary_path.write_bytes(sentinel)
    result = {"pass": True, "aggregate": {"verdict": "verified"}}
    monkeypatch.setattr(
        summarize_flashgs_matched,
        "validate_existing_summary",
        lambda root, output, batches: result,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_flashgs_matched.py",
            "--root",
            str(tmp_path),
            "--batches",
            "1,8",
            "--output",
            str(summary_path),
            "--verify-existing",
        ],
    )

    summarize_flashgs_matched.main()

    assert summary_path.read_bytes() == sentinel
