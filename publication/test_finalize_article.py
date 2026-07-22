#!/usr/bin/env python3
"""Tests for the deterministic publication article finalizer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from finalize_article import (
    CONCLUSION_PLACEHOLDER,
    FINALIZATION_SCHEMA,
    RESULTS_PLACEHOLDER,
    STATUS_PLACEHOLDER,
    FinalizationError,
    finalize,
)
from verify_claim_ledger import (
    ABLATION_GATE_ID,
    ABLATION_LOGICAL_PATH,
    ABLATION_SCHEMA,
    ACCEPTED_BASELINE_LABEL,
    ACCEPTED_HARDWARE_NAME,
    ACCEPTED_HARDWARE_UUID,
    ARTICLE_LOGICAL_PATH,
    LEDGER_LOGICAL_PATH,
    PRIMARY_BATCHES,
    SUMMARY_SCHEMA,
    VERIFICATION_LOGICAL_PATH,
    VERIFICATION_SCHEMA,
    canonical_json_bytes,
    file_sha256,
    parse_claim_markers,
    verify_claim_ledger,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def identity(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def template_text(*, extra: str = "") -> str:
    return (
        "# Publication fixture\n\n"
        f"{STATUS_PLACEHOLDER}\n\n"
        "The renderer scope is deliberately narrow.\n\n"
        f"{RESULTS_PLACEHOLDER}\n\n"
        f"{CONCLUSION_PLACEHOLDER}\n"
        f"{extra}"
    )


def create_fixture(root: Path, *, states: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    states = states or {"full": "custom", "rgb": "custom"}
    rows_by_contract: dict[str, list[dict[str, Any]]] = {"full": [], "rgb": []}
    for contract in ("full", "rgb"):
        for batch in PRIMARY_BATCHES:
            state = states[contract]
            if state == "custom":
                custom_gpu, baseline_gpu = float(batch), float(batch) * 2.0
                custom_wall, baseline_wall = float(batch) + 1.0, (float(batch) + 1.0) * 2.0
                gpu_min, gpu_max, wall_min, wall_max = 1.25, 2.50, 1.20, 2.40
            elif state == "baseline":
                custom_gpu, baseline_gpu = float(batch) * 2.0, float(batch)
                custom_wall, baseline_wall = (float(batch) + 1.0) * 2.0, float(batch) + 1.0
                gpu_min, gpu_max, wall_min, wall_max = 0.40, 0.80, 0.42, 0.82
            else:
                custom_gpu, baseline_gpu = float(batch), float(batch) * 1.1
                custom_wall, baseline_wall = float(batch) + 1.0, (float(batch) + 1.0) * 0.9
                gpu_min, gpu_max, wall_min, wall_max = 0.90, 1.30, 0.80, 1.20
            gpu_ratio = baseline_gpu / custom_gpu
            wall_ratio = baseline_wall / custom_wall
            rows_by_contract[contract].append(
                {
                    "batch": batch,
                    "pass": True,
                    "fidelity_pass": True,
                    "fairness_pass": True,
                    "fairness_failures": [],
                    "custom_ms": custom_gpu,
                    "flashgs_ms": baseline_gpu,
                    "custom_synchronized_wall_ms": custom_wall,
                    "flashgs_synchronized_wall_ms": baseline_wall,
                    "speedup_custom_over_flashgs": gpu_ratio,
                    "gpu_speedup_ratio_of_mean_latency": gpu_ratio,
                    "wall_speedup_ratio_of_mean_latency": wall_ratio,
                    "gpu_speedup_observed_same_step_min": gpu_min,
                    "gpu_speedup_observed_same_step_max": gpu_max,
                    "wall_speedup_observed_same_step_min": wall_min,
                    "wall_speedup_observed_same_step_max": wall_max,
                    "custom_physical_batch": batch if batch <= 256 else 128,
                    "custom_native_submissions_per_logical_batch": (
                        1 if batch <= 256 else batch // 128
                    ),
                    "flashgs_physical_batch": 1,
                    "flashgs_native_submissions_per_logical_batch": batch,
                }
            )

    if states == {"full": "custom", "rgb": "custom"}:
        verdict = "custom-wins-rgb-and-full-sensor"
    elif states == {"full": "custom", "rgb": "baseline"}:
        verdict = "flashgs-wins-rgb-custom-wins-full-sensor"
    elif states == {"full": "baseline", "rgb": "baseline"}:
        verdict = "flashgs-wins-rgb-and-full-sensor"
    else:
        verdict = "mixed-by-batch"

    def geometric_mean(contract: str) -> float:
        product = 1.0
        for row in rows_by_contract[contract]:
            product *= row["flashgs_ms"] / row["custom_ms"]
        return product ** (1.0 / 8.0)

    repeat_evidence: dict[str, dict[str, Any]] = {}
    for contract in ("full", "rgb"):
        if states[contract] == "baseline":
            winner = "flashgs"
            gpu_ratios = [0.50, 0.55, 0.45]
            wall_ratios = [0.52, 0.57, 0.47]
        else:
            winner = "custom"
            gpu_ratios = [2.00, 1.80, 2.20]
            wall_ratios = [1.90, 1.70, 2.10]

        def spread(values: list[float]) -> dict[str, float]:
            ordered = sorted(values)
            return {"min": ordered[0], "median": ordered[1], "max": ordered[2]}

        repeat_evidence[contract] = {
            "contract": contract,
            "pass": True,
            "winner": winner,
            "trials": {},
            "gpu_ratio_of_mean_latency_by_trial": gpu_ratios,
            "wall_ratio_of_mean_latency_by_trial": wall_ratios,
            "gpu_geometric_mean_ratio": 1.0,
            "wall_geometric_mean_ratio": 1.0,
            "gpu_run_level_ratio_descriptive_spread": spread(gpu_ratios),
            "wall_run_level_ratio_descriptive_spread": spread(wall_ratios),
            "gpu_same_step_min_by_trial": gpu_ratios,
            "gpu_same_step_max_by_trial": gpu_ratios,
            "wall_same_step_min_by_trial": wall_ratios,
            "wall_same_step_max_by_trial": wall_ratios,
            "independent_run_count": 3,
        }

    write_json(
        root / "summary.json",
        {
            "schema_version": SUMMARY_SCHEMA,
            "pass": True,
            "scientific_pass": True,
            "headline_eligible": True,
            "primary_contract_eligible": True,
            "matrix_fairness_failures": [],
            "hardware_scope": {
                "gpu_name": ACCEPTED_HARDWARE_NAME,
                "gpu_uuid": ACCEPTED_HARDWARE_UUID,
            },
            "preflight_evidence": {
                "b128_independent_repeats": repeat_evidence,
            },
            "primary_full_sensor_dynamic_table": rows_by_contract["full"],
            "rgb_only_dynamic_table": rows_by_contract["rgb"],
            "aggregate": {
                "full_sensor_geometric_mean_speedup_custom_over_flashgs": geometric_mean("full"),
                "rgb_geometric_mean_speedup_custom_over_flashgs": geometric_mean("rgb"),
                "raw_performance_verdict": verdict,
                "verdict": verdict,
            },
        },
    )

    ablation_path = root / ABLATION_LOGICAL_PATH
    ratios = {
        "full": [
            {"trial": 1, "cuda_speedup_p128_over_p1": 0.800, "wall_speedup_p128_over_p1": 0.750},
            {"trial": 2, "cuda_speedup_p128_over_p1": 1.100, "wall_speedup_p128_over_p1": 1.050},
            {"trial": 3, "cuda_speedup_p128_over_p1": 0.900, "wall_speedup_p128_over_p1": 0.850},
        ],
        "rgb": [
            {"trial": 1, "cuda_speedup_p128_over_p1": 1.200, "wall_speedup_p128_over_p1": 1.100},
            {"trial": 2, "cuda_speedup_p128_over_p1": 0.950, "wall_speedup_p128_over_p1": 0.900},
            {"trial": 3, "cuda_speedup_p128_over_p1": 1.050, "wall_speedup_p128_over_p1": 1.000},
        ],
    }
    run_ids = [
        f"{contract}-t{trial}-p{physical}"
        for contract in ("full", "rgb")
        for trial in (1, 2, 3)
        for physical in (1, 128)
    ]
    write_json(
        ablation_path,
        {
            "schema_version": ABLATION_SCHEMA,
            "pass": True,
            "batch": 128,
            "runs": [{"run_id": run_id} for run_id in run_ids],
            "run_order": run_ids,
            "ratios": ratios,
        },
    )
    gate_result_path = root / "publication/evidence/custom-vectorization-ablation-gate-result.json"
    write_json(
        gate_result_path,
        {
            "schema_version": "publication-gate-result-v1",
            "pass": True,
            "gate_id": ABLATION_GATE_ID,
            "created_at": "2026-01-01T00:00:00+00:00",
            "checks": {"design_cells": 12, "reported_without_win_requirement": True},
            "identity": {},
            "evidence": {"raw_result": identity(ablation_path)},
        },
    )
    write_json(
        root / VERIFICATION_LOGICAL_PATH,
        {
            "schema_version": VERIFICATION_SCHEMA,
            "pass": True,
            "created_at": "2026-01-01T00:00:01+00:00",
            "benchmark": {},
            "matrix": {},
            "required_gate_ids": [ABLATION_GATE_ID],
            "gates": [{"gate_id": ABLATION_GATE_ID, "result": identity(gate_result_path)}],
        },
    )
    template = root / "post.md"
    template.write_text(template_text(), encoding="utf-8")
    return template


class FinalizeArticleTests(unittest.TestCase):
    def test_finalizes_and_strictly_verifies_all_required_values(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-") as raw:
            root = Path(raw)
            template = create_fixture(root)
            receipt = finalize(evidence_root=root, template_path=template)
            self.assertEqual(receipt["schema_version"], FINALIZATION_SCHEMA)
            self.assertTrue(receipt["pass"])
            self.assertEqual(receipt["conclusion"], "custom-uniform")
            self.assertEqual(receipt["claim_count"], 110)
            self.assertTrue(receipt["publication_eligible"])
            self.assertEqual(receipt["mode"], "complete")
            verified = verify_claim_ledger(root)
            self.assertTrue(verified["pass"])
            self.assertEqual(verified["claim_count"], 110)

            article = (root / ARTICLE_LOGICAL_PATH).read_text(encoding="utf-8")
            markers = parse_claim_markers(article)
            self.assertEqual(len(markers), 110)
            self.assertEqual(markers["full-b1-custom-gpu-ms"], "1.000 ms")
            self.assertEqual(markers["full-b1-baseline-gpu-ms"], "2.000 ms")
            self.assertEqual(markers["full-b1-gpu-speedup"], "2.000×")
            self.assertEqual(markers["full-b1-wall-speedup"], "2.000×")
            self.assertEqual(markers["full-b1-custom-wall-ms"], "2.000 ms")
            self.assertEqual(markers["full-b1-baseline-wall-ms"], "4.000 ms")
            self.assertEqual(
                markers["custom-ablation-full-t1-cuda-p128-over-p1"],
                "0.800×",
            )
            self.assertIn(ACCEPTED_BASELINE_LABEL, article)
            self.assertIn("Custom is the scoped empirical winner", article)
            self.assertIn("P1/1", article)
            self.assertIn("three fresh processes", article)
            self.assertIn("P1 latency / P128 latency", article)

            ledger = json.loads((root / LEDGER_LOGICAL_PATH).read_text(encoding="utf-8"))
            self.assertEqual(ledger["summary"]["logical_path"], "summary.json")
            self.assertEqual(ledger["summary"]["path"], "summary.json")
            self.assertEqual(ledger["article"]["logical_path"], ARTICLE_LOGICAL_PATH)
            self.assertNotIn("created_at", ledger)

    def test_second_run_is_byte_identical_and_allowed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-repeat-") as raw:
            root = Path(raw)
            template = create_fixture(root)
            first = finalize(evidence_root=root, template_path=template)
            article = (root / ARTICLE_LOGICAL_PATH).read_bytes()
            ledger = (root / LEDGER_LOGICAL_PATH).read_bytes()
            second = finalize(evidence_root=root, template_path=template)
            self.assertEqual(first, second)
            self.assertEqual(article, (root / ARTICLE_LOGICAL_PATH).read_bytes())
            self.assertEqual(ledger, (root / LEDGER_LOGICAL_PATH).read_bytes())

    def test_refuses_to_overwrite_different_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-overwrite-") as raw:
            root = Path(raw)
            template = create_fixture(root)
            target = root / ARTICLE_LOGICAL_PATH
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("user-owned article\n", encoding="utf-8")
            with self.assertRaisesRegex(FinalizationError, "Refusing to overwrite"):
                finalize(evidence_root=root, template_path=template)
            self.assertEqual(target.read_text(encoding="utf-8"), "user-owned article\n")
            self.assertFalse((root / LEDGER_LOGICAL_PATH).exists())

    def test_rejects_missing_or_unmarked_template_claims(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-template-") as raw:
            root = Path(raw)
            template = create_fixture(root)
            original = template.read_text(encoding="utf-8")
            template.write_text(original.replace(RESULTS_PLACEHOLDER, ""), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationError, "placeholders differ"):
                finalize(evidence_root=root, template_path=template)

            template.write_text(template_text(extra="The renderer is 999x faster.\n"), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationError, "unmarked quantitative performance"):
                finalize(evidence_root=root, template_path=template)

    def test_cuda_only_advantage_cannot_produce_uniform_conclusion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-wall-gate-") as raw:
            root = Path(raw)
            template = create_fixture(root)
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            # CUDA still favors Custom in every sample and row. One wall-clock
            # row crosses one, which must be enough to forbid a uniform claim.
            row = summary["primary_full_sensor_dynamic_table"][0]
            row["wall_speedup_observed_same_step_min"] = 0.90
            row["wall_speedup_observed_same_step_max"] = 1.20
            summary["aggregate"]["raw_performance_verdict"] = "mixed-by-batch"
            summary["aggregate"]["verdict"] = "mixed-by-batch"
            write_json(summary_path, summary)
            receipt = finalize(evidence_root=root, template_path=template)
            self.assertEqual(receipt["conclusion"], "mixed")
            article = (root / ARTICLE_LOGICAL_PATH).read_text(encoding="utf-8")
            self.assertNotIn("Custom is the scoped empirical winner:", article)
            self.assertIn("does not support a uniform renderer winner", article)
            self.assertTrue(verify_claim_ledger(root)["pass"])

    def test_complete_baseline_win_is_reported_without_preselecting_custom(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-baseline-") as raw:
            root = Path(raw)
            template = create_fixture(root, states={"full": "baseline", "rgb": "baseline"})
            receipt = finalize(evidence_root=root, template_path=template)
            self.assertTrue(receipt["publication_eligible"])
            self.assertEqual(receipt["conclusion"], "baseline-uniform")
            article = (root / ARTICLE_LOGICAL_PATH).read_text(encoding="utf-8")
            self.assertIn(
                "FlashGS-derived matched-contract port is the scoped empirical winner",
                article,
            )
            self.assertNotIn("Custom is the scoped empirical winner:", article)
            self.assertTrue(verify_claim_ledger(root)["pass"])

    def test_rejects_stored_geometric_mean_that_differs_from_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-geomean-") as raw:
            root = Path(raw)
            template = create_fixture(root)
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["aggregate"]["full_sensor_geometric_mean_speedup_custom_over_flashgs"] = 999.0
            write_json(summary_path, summary)
            with self.assertRaisesRegex(FinalizationError, "geometric mean differs from recomputation"):
                finalize(evidence_root=root, template_path=template)
            self.assertFalse((root / ARTICLE_LOGICAL_PATH).exists())
            self.assertFalse((root / LEDGER_LOGICAL_PATH).exists())

    def test_fidelity_failure_generates_non_equivalent_diagnostic_article(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-fidelity-") as raw:
            root = Path(raw)
            template = create_fixture(root)
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row = summary["primary_full_sensor_dynamic_table"][0]
            row["pass"] = False
            row["fidelity_pass"] = False
            summary["pass"] = False
            summary["scientific_pass"] = False
            summary["headline_eligible"] = False
            write_json(summary_path, summary)

            receipt = finalize(evidence_root=root, template_path=template)
            self.assertFalse(receipt["pass"])
            self.assertTrue(receipt["generated"])
            self.assertFalse(receipt["publication_eligible"])
            self.assertEqual(receipt["mode"], "diagnostic-fidelity-failure")
            self.assertEqual(receipt["claim_count"], 0)
            self.assertIsNone(receipt["claim_ledger"])
            self.assertFalse((root / LEDGER_LOGICAL_PATH).exists())
            article = (root / ARTICLE_LOGICAL_PATH).read_text(encoding="utf-8")
            self.assertEqual(parse_claim_markers(article), {})
            self.assertIn("non-equivalent fidelity failure", article)
            self.assertIn("No strict claim ledger was emitted", article)
            self.assertNotIn("scoped empirical winner", article)

    def test_missing_row_generates_incomplete_diagnostic_without_interpolation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-incomplete-") as raw:
            root = Path(raw)
            template = create_fixture(root)
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["rgb_only_dynamic_table"].pop()
            summary["pass"] = False
            summary["scientific_pass"] = False
            summary["headline_eligible"] = False
            summary["primary_contract_eligible"] = False
            write_json(summary_path, summary)

            receipt = finalize(evidence_root=root, template_path=template)
            self.assertFalse(receipt["pass"])
            self.assertTrue(receipt["generated"])
            self.assertFalse(receipt["publication_eligible"])
            self.assertEqual(receipt["mode"], "diagnostic-incomplete")
            self.assertFalse((root / LEDGER_LOGICAL_PATH).exists())
            article = (root / ARTICLE_LOGICAL_PATH).read_text(encoding="utf-8")
            self.assertEqual(parse_claim_markers(article), {})
            self.assertIn("Missing rows remain missing", article)
            self.assertIn("nothing is interpolated", article)
            self.assertNotIn("scoped empirical winner", article)

    def test_missing_post_matrix_evidence_generates_incomplete_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-missing-evidence-") as raw:
            root = Path(raw)
            template = create_fixture(root)
            (root / ABLATION_LOGICAL_PATH).unlink()

            receipt = finalize(evidence_root=root, template_path=template)
            self.assertFalse(receipt["pass"])
            self.assertEqual(receipt["mode"], "diagnostic-incomplete")
            self.assertFalse(receipt["publication_eligible"])
            self.assertFalse((root / LEDGER_LOGICAL_PATH).exists())
            article = (root / ARTICLE_LOGICAL_PATH).read_text(encoding="utf-8")
            self.assertIn("schedule-control evidence is missing", article)
            self.assertNotIn("scoped empirical winner", article)

    def test_outputs_use_only_portable_ledger_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-finalizer-roots-") as raw:
            base = Path(raw)
            first_root = base / "first"
            second_root = base / "second"
            first_template = create_fixture(first_root)
            second_template = create_fixture(second_root)
            # Nested aggregate records contain absolute fixture paths, so
            # their input hashes legitimately differ by fixture. The final
            # article remains root-independent, and the ledger must not copy
            # either staging directory into any of its path fields.
            finalize(evidence_root=first_root, template_path=first_template)
            finalize(evidence_root=second_root, template_path=second_template)
            self.assertEqual(
                (first_root / ARTICLE_LOGICAL_PATH).read_bytes(),
                (second_root / ARTICLE_LOGICAL_PATH).read_bytes(),
            )
            for root in (first_root, second_root):
                ledger_text = (root / LEDGER_LOGICAL_PATH).read_text(encoding="utf-8")
                self.assertNotIn(str(root), ledger_text)
                ledger = json.loads(ledger_text)
                for record_name in ("summary", "verification", "ablation", "article"):
                    self.assertEqual(ledger[record_name]["path"], ledger[record_name]["logical_path"])


if __name__ == "__main__":
    unittest.main()
