from __future__ import annotations

from benchmarks.run_baselines import (
    GSPAT_OUTPUT_CONTRACT,
    gsplat_output_bytes,
    gsplat_output_contract,
)


def test_gsplat_baseline_excludes_fake_semantic_output() -> None:
    contract = gsplat_output_contract()

    assert contract == {
        "outputs": ["rgb", "depth", "alpha"],
        "output_contract": GSPAT_OUTPUT_CONTRACT,
        "output_match_scope": "rgb_depth_alpha_only",
        "semantic_contract": "unavailable_not_measured",
    }
    assert "semantic_id" not in contract["outputs"]


def test_gsplat_output_bytes_count_only_native_aovs() -> None:
    assert gsplat_output_bytes(batch_size=2, width=3, height=5) == (
        2 * 3 * 5 * 20
    )
