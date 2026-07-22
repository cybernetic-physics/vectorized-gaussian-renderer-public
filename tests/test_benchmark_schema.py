from __future__ import annotations

import csv

from isaacsim_gaussian_renderer.benchmark_schema import (
    CSV_FIELDS,
    fabric_scene_delegate_record,
    flatten_result,
    write_results_csv,
)


def test_fabric_scene_delegate_record_is_not_applicable() -> None:
    record = fabric_scene_delegate_record()

    assert record["state"] == "not_applicable"
    assert record["citable_as_fsd_evidence"] is False
    assert "no Kit application" in record["reason"]
    assert "Fabric Scene Delegate" in record["reason"]


def test_flatten_result_carries_fabric_scene_delegate_state() -> None:
    row = flatten_result(
        {
            "configuration": {
                "fabric_scene_delegate": fabric_scene_delegate_record(),
            },
        }
    )

    assert "fabric_scene_delegate" in CSV_FIELDS
    assert row["fabric_scene_delegate"] == "not_applicable"


def test_flatten_result_leaves_unrecorded_fsd_state_empty() -> None:
    row = flatten_result({"configuration": {}})

    assert row["fabric_scene_delegate"] is None


def test_results_csv_contains_fabric_scene_delegate_column(tmp_path) -> None:
    path = tmp_path / "baseline-results.csv"
    write_results_csv(
        [
            {
                "run_id": "run-1",
                "configuration": {
                    "fabric_scene_delegate": fabric_scene_delegate_record(),
                },
            }
        ],
        path,
    )

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["fabric_scene_delegate"] == "not_applicable"
