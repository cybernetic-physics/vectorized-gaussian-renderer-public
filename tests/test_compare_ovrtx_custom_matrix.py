from pathlib import Path

import pytest

from benchmarks.compare_ovrtx_custom_matrix import (
    comparison_pass,
    custom_result_path,
    rasterize_mode_contract_mismatches,
    require_passing_report,
)


def test_custom_result_path_is_rasterize_mode_qualified(tmp_path: Path) -> None:
    classic = custom_result_path(
        tmp_path,
        scene="synthetic-small",
        batch=8,
        width=128,
        height=128,
        rasterize_mode="classic",
    )
    antialiased = custom_result_path(
        tmp_path,
        scene="synthetic-small",
        batch=8,
        width=128,
        height=128,
        rasterize_mode="antialiased",
    )

    assert classic.name == (
        "synthetic-small-b8-128x128-rasterize-classic.json"
    )
    assert antialiased.name == (
        "synthetic-small-b8-128x128-rasterize-antialiased.json"
    )
    assert classic != antialiased


def test_rasterize_mode_contract_rejects_mismatched_or_missing_evidence() -> None:
    assert not rasterize_mode_contract_mismatches(
        {"rasterize_mode": "antialiased"},
        expected="antialiased",
    )
    assert rasterize_mode_contract_mismatches(
        {"rasterize_mode": "classic"},
        expected="antialiased",
    ) == ["custom_rasterize_mode"]
    assert rasterize_mode_contract_mismatches(
        {},
        expected="classic",
    ) == ["custom_rasterize_mode"]


def test_comparison_report_is_fail_closed() -> None:
    passing_row = {
        "throughput_pass": True,
        "memory_pass": True,
        "zero_steady_state_allocation": True,
        "zero_overflow": True,
    }
    assert comparison_pass(
        ovrtx_manifest_complete=True,
        rows=[passing_row],
        failures=[],
    )
    assert not comparison_pass(
        ovrtx_manifest_complete=True,
        rows=[passing_row],
        failures=[{"status": "MISSING"}],
    )
    assert not comparison_pass(
        ovrtx_manifest_complete=True,
        rows=[{**passing_row, "throughput_pass": False}],
        failures=[],
    )
    assert not comparison_pass(
        ovrtx_manifest_complete=True,
        rows=[],
        failures=[],
    )

    require_passing_report({"pass": True})
    with pytest.raises(SystemExit) as error:
        require_passing_report({"pass": False})
    assert error.value.code == 1
