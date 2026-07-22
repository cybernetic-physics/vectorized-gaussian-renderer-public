from __future__ import annotations

from benchmarks.summarize_headline_acceptance import fidelity_snapshot


def test_fidelity_snapshot_requires_top_and_aggregate_pass() -> None:
    report = {
        "pass": True,
        "aggregate": {
            "pass": False,
            "metrics": {"semantic_agreement": {"mean": 0.995}},
            "thresholds": {
                "semantic_agreement": {
                    "pass": False,
                    "threshold": 0.999,
                }
            },
        },
    }

    snapshot = fidelity_snapshot(report)

    assert snapshot["pass"] is False
    assert snapshot["metrics"]["semantic_agreement"]["mean"] == 0.995
