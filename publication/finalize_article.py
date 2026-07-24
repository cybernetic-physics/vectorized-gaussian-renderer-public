#!/usr/bin/env python3
"""Deterministically finalize the benchmark article and strict claim ledger.

This is a publication-only adapter.  It does not run or reinterpret a GPU
benchmark.  It reads the already validated matrix and Custom P128/P1 ablation,
recomputes every displayed number and replaces three explicit template tokens.
A complete claim-eligible matrix emits the two fixed publication artifacts
accepted by ``verify_claim_ledger.py``. A recognized fidelity failure or
incomplete/invalid matrix instead emits a marker-free diagnostic article with
no claim ledger, no performance values, and no winner language.

The template must contain each of these standalone tokens exactly once:

* ``<!-- publication:status -->``
* ``<!-- publication:results -->``
* ``<!-- publication:conclusion -->``

Existing output files are accepted only when their bytes are identical to the
new deterministic output.  A different existing file is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from verify_claim_ledger import (
    ABLATION_LOGICAL_PATH,
    ACCEPTED_BASELINE_LABEL,
    ACCEPTED_EQUATION_CONTRACT,
    ACCEPTED_HARDWARE,
    ACCEPTED_HARDWARE_NAME,
    ACCEPTED_SCENE,
    ALLOWED_RELEASE_CLAIMS,
    ARTICLE_LOGICAL_PATH,
    LEDGER_LOGICAL_PATH,
    LEDGER_SCHEMA,
    PRIMARY_BATCHES,
    REQUIRED_RELEASE_CLAIM_IDS,
    STRICT_LEDGER_SCHEMA,
    SUMMARY_LOGICAL_PATH,
    SUMMARY_SCHEMA,
    VERIFICATION_LOGICAL_PATH,
    ClaimLedgerError,
    EvidenceRoot,
    canonical_json_bytes,
    file_sha256,
    format_result,
    is_nvidia_gpu_uuid,
    load_decimal_json,
    normalized_format,
    numeric,
    parse_claim_markers,
    recompute,
    reject_unmarked_performance_claims,
    resolve_json_pointer,
    resolve_validated_ablation,
    verify_claim_ledger,
)

FINALIZATION_SCHEMA = "publication-article-finalization-v2"
STATUS_PLACEHOLDER = "<!-- publication:status -->"
RESULTS_PLACEHOLDER = "<!-- publication:results -->"
CONCLUSION_PLACEHOLDER = "<!-- publication:conclusion -->"
REQUIRED_PLACEHOLDERS = (
    STATUS_PLACEHOLDER,
    RESULTS_PLACEHOLDER,
    CONCLUSION_PLACEHOLDER,
)
PLACEHOLDER_RE = re.compile(r"<!-- publication:[a-z0-9-]+ -->")

TABLES = {
    "full": ("primary_full_sensor_dynamic_table", "Full sensor"),
    "rgb": ("rgb_only_dynamic_table", "RGB only"),
}
PUBLISHED_ROW_SUFFIXES = (
    "custom-gpu-ms",
    "baseline-gpu-ms",
    "custom-wall-ms",
    "baseline-wall-ms",
    "gpu-speedup",
    "wall-speedup",
)

MILLISECONDS_FORMAT = {
    "notation": "fixed",
    "precision": 3,
    "prefix": "",
    "suffix": " ms",
    "grouping": False,
    "trim_trailing_zeros": False,
    "exponent_case": "lower",
}
RATIO_FORMAT = {
    "notation": "fixed",
    "precision": 3,
    "prefix": "",
    "suffix": "×",
    "grouping": False,
    "trim_trailing_zeros": False,
    "exponent_case": "lower",
}


class FinalizationError(ValueError):
    """Raised before publication output is accepted or written."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_record(path: Path, *, logical_path: str) -> dict[str, Any]:
    """Return a root-independent, strict artifact record.

    ``path`` is deliberately the canonical logical path.  The strict verifier
    resolves it through ``logical_path`` in staging mode and through the object
    inventory after relocation.  This keeps ledger bytes independent of the
    temporary staging directory while retaining the required path field.
    """

    return {
        "bytes": path.stat().st_size,
        "logical_path": logical_path,
        "path": logical_path,
        "sha256": file_sha256(path),
    }


def _artifact_record_from_bytes(payload: bytes, *, logical_path: str) -> dict[str, Any]:
    return {
        "bytes": len(payload),
        "logical_path": logical_path,
        "path": logical_path,
        "sha256": _sha256_bytes(payload),
    }


def _read_template(path: Path) -> str:
    raw = path
    if raw.is_symlink():
        raise FinalizationError(f"Article template may not be a symlink: {raw}.")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise FinalizationError(f"Article template is missing: {raw}.") from error
    if not resolved.is_file():
        raise FinalizationError(f"Article template is not a regular file: {raw}.")
    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise FinalizationError("Article template is not UTF-8 text.") from error


def _validate_template(template: str) -> None:
    try:
        existing_markers = parse_claim_markers(template)
        reject_unmarked_performance_claims(template)
    except ClaimLedgerError as error:
        raise FinalizationError(str(error)) from error
    if existing_markers:
        raise FinalizationError("Article template already contains generated claim markers.")
    observed = PLACEHOLDER_RE.findall(template)
    expected = list(REQUIRED_PLACEHOLDERS)
    if sorted(observed) != sorted(expected):
        raise FinalizationError(
            "Article template placeholders differ; "
            f"expected={sorted(expected)}, observed={sorted(observed)}."
        )
    for placeholder in REQUIRED_PLACEHOLDERS:
        if template.count(placeholder) != 1 or re.search(
            rf"(?m)^[ \t]*{re.escape(placeholder)}[ \t]*$",
            template,
        ) is None:
            raise FinalizationError(
                f"Article template must contain standalone placeholder {placeholder!r} exactly once."
            )


def _close(left: Decimal, right: Decimal, *, label: str) -> None:
    tolerance = max(Decimal("1e-12"), abs(right) * Decimal("1e-12"))
    if abs(left - right) > tolerance:
        raise FinalizationError(f"{label} differs from recomputation: {left} != {right}.")


def _geometric_mean_eight(values: list[Decimal]) -> Decimal:
    if len(values) != 8 or any(value <= 0 for value in values):
        raise FinalizationError("Geometric-mean input must contain eight positive ratios.")
    with localcontext() as context:
        context.prec = 50
        product = Decimal(1)
        for value in values:
            product *= value
        # The frozen matrix has exactly eight rows, so three square roots avoid
        # platform-dependent binary logarithms.
        return product.sqrt().sqrt().sqrt()


def _winner_state(rows: list[dict[str, Any]], *, label: str) -> str:
    custom_wins = True
    baseline_wins = True
    for row in rows:
        gpu_min = numeric(
            row.get("gpu_speedup_observed_same_step_min"),
            label=f"{label} CUDA same-step minimum",
        )
        gpu_max = numeric(
            row.get("gpu_speedup_observed_same_step_max"),
            label=f"{label} CUDA same-step maximum",
        )
        wall_min = numeric(
            row.get("wall_speedup_observed_same_step_min"),
            label=f"{label} wall same-step minimum",
        )
        wall_max = numeric(
            row.get("wall_speedup_observed_same_step_max"),
            label=f"{label} wall same-step maximum",
        )
        if min(gpu_min, gpu_max, wall_min, wall_max) <= 0:
            raise FinalizationError(f"{label} contains a nonpositive same-step ratio.")
        if gpu_min > gpu_max or wall_min > wall_max:
            raise FinalizationError(f"{label} contains an inverted same-step range.")
        custom_wins = custom_wins and gpu_min > 1 and wall_min > 1
        baseline_wins = baseline_wins and gpu_max < 1 and wall_max < 1
    if custom_wins:
        return "custom"
    if baseline_wins:
        return "baseline"
    return "mixed"


def _classify_summary(summary: Any) -> tuple[str, tuple[str, ...]]:
    """Classify a frozen summary without turning a failed run into a headline.

    Only the complete outcome may enter the strict quantitative claim path.
    Other recognized outcomes deliberately produce a marker-free diagnostic
    article: the current strict ledger requires every headline row and cannot
    safely bind a partial or non-equivalent matrix.
    """

    if not isinstance(summary, dict):
        raise FinalizationError("Canonical matrix summary is not an object.")
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise FinalizationError("Canonical matrix summary schema differs.")

    incomplete = False
    fidelity_failed = False
    row_invalid = False
    for _contract, (table_name, _label) in TABLES.items():
        rows = summary.get(table_name)
        if not isinstance(rows, list) or len(rows) != len(PRIMARY_BATCHES):
            incomplete = True
            continue
        for expected_batch, row in zip(PRIMARY_BATCHES, rows):
            if not isinstance(row, dict) or row.get("batch") != Decimal(expected_batch):
                incomplete = True
                continue
            fidelity_failed = fidelity_failed or row.get("fidelity_pass") is not True
            row_invalid = row_invalid or (
                row.get("pass") is not True
                or row.get("fairness_pass") is not True
                or row.get("fairness_failures") != []
            )

    preflight = summary.get("preflight_evidence")
    repeats = preflight.get("b128_independent_repeats") if isinstance(preflight, dict) else None
    if not isinstance(repeats, dict) or any(
        not isinstance(repeats.get(contract), dict) for contract in TABLES
    ):
        incomplete = True
    elif any(repeats[contract].get("pass") is not True for contract in TABLES):
        row_invalid = True

    if summary.get("primary_contract_eligible") is not True:
        incomplete = True
    if incomplete:
        return "diagnostic-incomplete", (
            "one or more predeclared rows are missing, duplicated, or out of order",
        )
    if fidelity_failed:
        return "diagnostic-fidelity-failure", (
            "one or more candidate rows failed the same-equation fidelity gate",
        )

    hardware = summary.get("hardware_scope")
    hardware_invalid = not isinstance(hardware, dict) or (
        hardware.get("gpu_name") != ACCEPTED_HARDWARE_NAME
        or not is_nvidia_gpu_uuid(hardware.get("gpu_uuid"))
    )
    complete = bool(
        summary.get("pass") is True
        and summary.get("scientific_pass") is True
        and summary.get("headline_eligible") is True
        and summary.get("matrix_fairness_failures") == []
        and not row_invalid
        and not hardware_invalid
    )
    if complete:
        return "complete", ()
    return "diagnostic-fairness-failure", (
        "the complete matrix did not satisfy every provenance, fairness, hardware, and headline gate",
    )


def _integer_field(row: dict[str, Any], field: str, *, label: str) -> int:
    value = numeric(row.get(field), label=label)
    integral = value.to_integral_value()
    if value != integral or integral <= 0:
        raise FinalizationError(f"{label} must be a positive integer.")
    return int(integral)


def _parity_range(row: dict[str, Any], *, clock: str, label: str) -> str:
    minimum = numeric(
        row.get(f"{clock}_speedup_observed_same_step_min"),
        label=f"{label} {clock} same-step minimum",
    )
    maximum = numeric(
        row.get(f"{clock}_speedup_observed_same_step_max"),
        label=f"{label} {clock} same-step maximum",
    )
    if minimum <= 0 or minimum > maximum:
        raise FinalizationError(f"{label} has an invalid {clock} same-step range.")
    if minimum > 1:
        return "entire range above parity"
    if maximum < 1:
        return "entire range below parity"
    return "range touches or crosses parity"


def _repeat_evidence(summary: Any) -> dict[str, dict[str, str]]:
    preflight = summary.get("preflight_evidence")
    repeats = preflight.get("b128_independent_repeats") if isinstance(preflight, dict) else None
    if not isinstance(repeats, dict):
        raise FinalizationError("Canonical B128 independent-repeat evidence is missing.")
    rendered: dict[str, dict[str, str]] = {}
    for contract in TABLES:
        evidence = repeats.get(contract)
        if not isinstance(evidence, dict) or evidence.get("pass") is not True:
            raise FinalizationError(f"Canonical B128 {contract} independent-repeat gate did not pass.")
        winner = evidence.get("winner")
        if winner not in {"custom", "flashgs"}:
            raise FinalizationError(f"Canonical B128 {contract} repeat winner is invalid.")
        if numeric(
            evidence.get("independent_run_count"),
            label=f"B128 {contract} independent run count",
        ) != 3:
            raise FinalizationError(f"Canonical B128 {contract} does not contain three runs.")
        spreads: dict[str, str] = {}
        for clock in ("gpu", "wall"):
            values = evidence.get(f"{clock}_ratio_of_mean_latency_by_trial")
            if not isinstance(values, list) or len(values) != 3:
                raise FinalizationError(f"Canonical B128 {contract} {clock} repeat ratios are incomplete.")
            ratios = [
                numeric(value, label=f"B128 {contract} {clock} repeat ratio")
                for value in values
            ]
            if any(value <= 0 for value in ratios):
                raise FinalizationError(f"Canonical B128 {contract} {clock} repeat ratio is nonpositive.")
            same_step_minimums = evidence.get(f"{clock}_same_step_min_by_trial")
            same_step_maximums = evidence.get(f"{clock}_same_step_max_by_trial")
            if (
                not isinstance(same_step_minimums, list)
                or not isinstance(same_step_maximums, list)
                or len(same_step_minimums) != 3
                or len(same_step_maximums) != 3
            ):
                raise FinalizationError(
                    f"Canonical B128 {contract} {clock} same-step repeat ranges are incomplete."
                )
            minimums = [
                numeric(value, label=f"B128 {contract} {clock} repeat minimum")
                for value in same_step_minimums
            ]
            maximums = [
                numeric(value, label=f"B128 {contract} {clock} repeat maximum")
                for value in same_step_maximums
            ]
            if any(minimum <= 0 or minimum > maximum for minimum, maximum in zip(minimums, maximums)):
                raise FinalizationError(
                    f"Canonical B128 {contract} {clock} same-step repeat range is invalid."
                )
            if winner == "custom" and min(minimums) <= 1:
                raise FinalizationError(
                    f"Canonical B128 {contract} {clock} repeats do not support the Custom winner."
                )
            if winner == "flashgs" and max(maximums) >= 1:
                raise FinalizationError(
                    f"Canonical B128 {contract} {clock} repeats do not support the derived-port winner."
                )
            stored = evidence.get(f"{clock}_run_level_ratio_descriptive_spread")
            if not isinstance(stored, dict):
                raise FinalizationError(f"Canonical B128 {contract} {clock} spread is missing.")
            ordered = sorted(ratios)
            for statistic, expected in (
                ("min", ordered[0]),
                ("median", ordered[1]),
                ("max", ordered[2]),
            ):
                _close(
                    numeric(
                        stored.get(statistic),
                        label=f"B128 {contract} {clock} repeat {statistic}",
                    ),
                    expected,
                    label=f"Canonical B128 {contract} {clock} repeat {statistic}",
                )
            spreads[clock] = (
                "entire spread above parity"
                if min(ratios) > 1
                else "entire spread below parity"
                if max(ratios) < 1
                else "spread touches or crosses parity"
            )
        rendered[contract] = {
            "winner": "Custom in every repeat" if winner == "custom" else "Derived port in every repeat",
            "gpu": spreads["gpu"],
            "wall": spreads["wall"],
        }
    return rendered


def _validate_summary(
    summary: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, dict[str, str]]]:
    if not isinstance(summary, dict):
        raise FinalizationError("Canonical matrix summary is not an object.")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("pass") is not True
        or summary.get("scientific_pass") is not True
        or summary.get("headline_eligible") is not True
        or summary.get("primary_contract_eligible") is not True
        or summary.get("matrix_fairness_failures") != []
    ):
        raise FinalizationError("Canonical matrix is not a complete passing headline result.")
    hardware = summary.get("hardware_scope")
    if not isinstance(hardware, dict) or (
        hardware.get("gpu_name") != ACCEPTED_HARDWARE_NAME
        or not is_nvidia_gpu_uuid(hardware.get("gpu_uuid"))
    ):
        raise FinalizationError(
            "Canonical matrix is not bound to one identified publication L4."
        )

    rows_by_contract: dict[str, list[dict[str, Any]]] = {}
    speedups_by_contract: dict[str, list[Decimal]] = {}
    for contract, (table_name, _label) in TABLES.items():
        rows = summary.get(table_name)
        if not isinstance(rows, list) or len(rows) != len(PRIMARY_BATCHES):
            raise FinalizationError(f"Canonical {contract} table is incomplete.")
        normalized_rows: list[dict[str, Any]] = []
        speedups: list[Decimal] = []
        for expected_batch, row in zip(PRIMARY_BATCHES, rows):
            if not isinstance(row, dict) or row.get("batch") != Decimal(expected_batch):
                raise FinalizationError(f"Canonical {contract} table batch order differs.")
            if (
                row.get("pass") is not True
                or row.get("fidelity_pass") is not True
                or row.get("fairness_pass") is not True
                or row.get("fairness_failures") != []
            ):
                raise FinalizationError(f"Canonical {contract} B{expected_batch} row is not claim eligible.")
            custom_gpu = numeric(row.get("custom_ms"), label=f"{contract} B{expected_batch} Custom GPU ms")
            baseline_gpu = numeric(row.get("flashgs_ms"), label=f"{contract} B{expected_batch} baseline GPU ms")
            custom_wall = numeric(
                row.get("custom_synchronized_wall_ms"),
                label=f"{contract} B{expected_batch} Custom wall ms",
            )
            baseline_wall = numeric(
                row.get("flashgs_synchronized_wall_ms"),
                label=f"{contract} B{expected_batch} baseline wall ms",
            )
            if min(custom_gpu, baseline_gpu, custom_wall, baseline_wall) <= 0:
                raise FinalizationError(f"Canonical {contract} B{expected_batch} has nonpositive latency.")
            gpu_ratio = baseline_gpu / custom_gpu
            wall_ratio = baseline_wall / custom_wall
            for field in ("speedup_custom_over_flashgs", "gpu_speedup_ratio_of_mean_latency"):
                _close(
                    numeric(row.get(field), label=f"{contract} B{expected_batch} {field}"),
                    gpu_ratio,
                    label=f"Canonical {contract} B{expected_batch} {field}",
                )
            _close(
                numeric(
                    row.get("wall_speedup_ratio_of_mean_latency"),
                    label=f"{contract} B{expected_batch} wall ratio",
                ),
                wall_ratio,
                label=f"Canonical {contract} B{expected_batch} wall ratio",
            )
            # Validate the stronger all-sample fields used for qualitative
            # winner language before retaining this row.
            _winner_state([row], label=f"{contract} B{expected_batch}")
            expected_custom_physical = expected_batch if expected_batch <= 256 else 128
            expected_custom_submissions = 1 if expected_batch <= 256 else expected_batch // 128
            schedule = {
                "custom_physical_batch": expected_custom_physical,
                "custom_native_submissions_per_logical_batch": expected_custom_submissions,
                "flashgs_physical_batch": 1,
                "flashgs_native_submissions_per_logical_batch": expected_batch,
            }
            for field, expected in schedule.items():
                observed = _integer_field(
                    row,
                    field,
                    label=f"{contract} B{expected_batch} {field}",
                )
                if observed != expected:
                    raise FinalizationError(
                        f"Canonical {contract} B{expected_batch} {field} differs: "
                        f"{observed} != {expected}."
                    )
            normalized_rows.append(row)
            speedups.append(gpu_ratio)
        rows_by_contract[contract] = normalized_rows
        speedups_by_contract[contract] = speedups

    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict):
        raise FinalizationError("Canonical matrix aggregate is missing.")
    for contract, field in (
        ("full", "full_sensor_geometric_mean_speedup_custom_over_flashgs"),
        ("rgb", "rgb_geometric_mean_speedup_custom_over_flashgs"),
    ):
        _close(
            numeric(aggregate.get(field), label=f"Aggregate {contract} geometric mean"),
            _geometric_mean_eight(speedups_by_contract[contract]),
            label=f"Aggregate {contract} geometric mean",
        )

    states = {
        contract: _winner_state(rows, label=f"{contract} matrix")
        for contract, rows in rows_by_contract.items()
    }
    if states == {"full": "custom", "rgb": "custom"}:
        expected_verdict = "custom-wins-rgb-and-full-sensor"
    elif states == {"full": "custom", "rgb": "baseline"}:
        expected_verdict = "flashgs-wins-rgb-custom-wins-full-sensor"
    elif states == {"full": "baseline", "rgb": "baseline"}:
        expected_verdict = "flashgs-wins-rgb-and-full-sensor"
    else:
        expected_verdict = "mixed-by-batch"
    if aggregate.get("raw_performance_verdict") != expected_verdict or aggregate.get("verdict") != expected_verdict:
        raise FinalizationError(
            "Canonical matrix verdict differs from the row-level CUDA/wall winner gate: "
            f"expected {expected_verdict!r}."
        )
    return rows_by_contract, states, _repeat_evidence(summary)


def _published_claim_ids() -> list[str]:
    claim_ids: set[str] = set()
    for contract in TABLES:
        for batch in PRIMARY_BATCHES:
            claim_ids.update(f"{contract}-b{batch}-{suffix}" for suffix in PUBLISHED_ROW_SUFFIXES)
    claim_ids.update(("full-geometric-mean-speedup", "rgb-geometric-mean-speedup"))
    for contract in TABLES:
        for trial in (1, 2, 3):
            for clock in ("cuda", "wall"):
                claim_ids.add(f"custom-ablation-{contract}-t{trial}-{clock}-p128-over-p1")
    if not REQUIRED_RELEASE_CLAIM_IDS.issubset(claim_ids):
        raise FinalizationError("Finalizer claim set omits a required strict-ledger claim.")
    unknown = claim_ids - set(ALLOWED_RELEASE_CLAIMS)
    if unknown:
        raise FinalizationError(f"Finalizer claim set contains unknown claims: {sorted(unknown)}.")
    return sorted(claim_ids)


def _claim_format(claim_id: str) -> dict[str, Any]:
    if claim_id.endswith(
        ("custom-gpu-ms", "baseline-gpu-ms", "custom-wall-ms", "baseline-wall-ms")
    ):
        return dict(MILLISECONDS_FORMAT)
    return dict(RATIO_FORMAT)


def _build_claims(
    *,
    summary: Any,
    summary_record: dict[str, Any],
    ablation: Any,
    ablation_record: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    targets = {
        "summary": (summary, summary_record),
        "ablation": (ablation, ablation_record),
    }
    claims: list[dict[str, Any]] = []
    display_by_id: dict[str, str] = {}
    for claim_id in _published_claim_ids():
        binding = ALLOWED_RELEASE_CLAIMS[claim_id]
        target, artifact = targets[binding["artifact_role"]]
        values = [
            numeric(
                resolve_json_pointer(target, pointer),
                label=f"Claim {claim_id} field {pointer}",
            )
            for pointer in binding["json_fields"]
        ]
        format_specification = _claim_format(claim_id)
        display = format_result(
            recompute(binding["operation"], values),
            normalized_format(format_specification),
        )
        display_by_id[claim_id] = display
        claims.append(
            {
                "artifact": artifact,
                "article_location": f"claim:{claim_id}",
                "baseline_label": binding["baseline_label"],
                "claim_id": claim_id,
                "display_value": display,
                "equation_contract": ACCEPTED_EQUATION_CONTRACT,
                "format": format_specification,
                "hardware": ACCEPTED_HARDWARE,
                "json_fields": list(binding["json_fields"]),
                "operation": binding["operation"],
                "output_contract": binding["output_contract"],
                "scene": ACCEPTED_SCENE,
            }
        )
    return claims, display_by_id


def _marker(claim_id: str, display_by_id: dict[str, str]) -> str:
    display = display_by_id[claim_id]
    return f"<!-- claim:{claim_id} -->{display}<!-- /claim:{claim_id} -->"


def _render_status() -> str:
    return (
        "> **Benchmark evidence status — finalized and hash-bound.** The canonical matrix, "
        "aggregate publication verification, and Custom physical-batch ablation "
        "are passing and hash-bound. Every displayed performance value below is "
        "recomputed from those JSON artifacts by the strict claim ledger. This "
        "status does not assert that the external release or anonymous retrieval "
        "checks are complete."
    )


def _render_results(
    display_by_id: dict[str, str],
    *,
    rows_by_contract: dict[str, list[dict[str, Any]]],
    repeats: dict[str, dict[str, str]],
) -> str:
    blocks = [
        "## Measured results",
        "",
        "The claim scope is fixed to these exact labels:",
        "",
        f"- Baseline: {ACCEPTED_BASELINE_LABEL}",
        f"- Hardware: {ACCEPTED_HARDWARE}",
        f"- Scene: {ACCEPTED_SCENE}",
        f"- Image equation: {ACCEPTED_EQUATION_CONTRACT}",
        "",
        "Each speedup is baseline latency divided by Custom latency. The table "
        "reports ratios of mean latency for both CUDA-event and synchronized-wall "
        "clocks; values on either side of parity are retained. Exact numerical "
        "same-step ranges remain in the hash-bound summary; the table states "
        "whether each complete range is above, below, or touches parity.",
    ]
    for contract, (_table_name, label) in TABLES.items():
        blocks.extend(
            [
                "",
                f"### {label}",
                "",
                "Execution schedule and all-sample winner gates:",
                "",
                "| Logical batch | Custom P/submits | FlashGS-derived port P/submits | CUDA same-step range | Wall same-step range |",
                "|---:|---:|---:|:---|:---|",
            ]
        )
        for batch, row in zip(PRIMARY_BATCHES, rows_by_contract[contract]):
            blocks.append(
                "| {batch} | P{custom_physical}/{custom_submits} | "
                "P{baseline_physical}/{baseline_submits} | {gpu_range} | "
                "{wall_range} |".format(
                    batch=batch,
                    custom_physical=_integer_field(
                        row,
                        "custom_physical_batch",
                        label=f"{contract} B{batch} Custom physical batch",
                    ),
                    custom_submits=_integer_field(
                        row,
                        "custom_native_submissions_per_logical_batch",
                        label=f"{contract} B{batch} Custom native submissions",
                    ),
                    baseline_physical=_integer_field(
                        row,
                        "flashgs_physical_batch",
                        label=f"{contract} B{batch} derived-port physical batch",
                    ),
                    baseline_submits=_integer_field(
                        row,
                        "flashgs_native_submissions_per_logical_batch",
                        label=f"{contract} B{batch} derived-port native submissions",
                    ),
                    gpu_range=_parity_range(
                        row,
                        clock="gpu",
                        label=f"{contract} B{batch}",
                    ),
                    wall_range=_parity_range(
                        row,
                        clock="wall",
                        label=f"{contract} B{batch}",
                    ),
                )
            )
        blocks.extend(
            [
                "",
                "Absolute latency and ratio-of-means results:",
                "",
                "| Logical batch | Custom GPU | FlashGS-derived port GPU | Custom wall | FlashGS-derived port wall | GPU speedup | Wall speedup |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for batch in PRIMARY_BATCHES:
            blocks.append(
                "| {batch} | {custom_gpu} | {baseline_gpu} | {custom_wall} | "
                "{baseline_wall} | {gpu} | {wall} |".format(
                    batch=batch,
                    custom_gpu=_marker(f"{contract}-b{batch}-custom-gpu-ms", display_by_id),
                    baseline_gpu=_marker(f"{contract}-b{batch}-baseline-gpu-ms", display_by_id),
                    custom_wall=_marker(f"{contract}-b{batch}-custom-wall-ms", display_by_id),
                    baseline_wall=_marker(f"{contract}-b{batch}-baseline-wall-ms", display_by_id),
                    gpu=_marker(f"{contract}-b{batch}-gpu-speedup", display_by_id),
                    wall=_marker(f"{contract}-b{batch}-wall-speedup", display_by_id),
                )
            )
        blocks.extend(
            [
                "",
                "Geometric-mean GPU speedup: "
                + _marker(f"{contract}-geometric-mean-speedup", display_by_id)
                + ". It equally weights the eight predeclared, nested batch "
                "points and is descriptive only; it is neither a scaling "
                "exponent nor a deployment-weighted average.",
            ]
        )

    blocks.extend(
        [
            "",
            "### Independent middle-row repeat dispersion",
            "",
            "The primary middle row and two fresh-process repetitions must agree "
            "on the same renderer in both clocks. Exact run-level ratios and "
            "minimum/median/maximum values are retained in the hash-bound summary. "
            "The strict article ledger does not currently bind those repeat fields, "
            "so this table reports their validated direction rather than copying "
            "unledgered numbers.",
            "",
            "| Contract | Processes | Repeat winner gate | CUDA run-level spread | Wall run-level spread |",
            "|:---|:---|:---|:---|:---|",
        ]
    )
    for contract, (_table_name, label) in TABLES.items():
        repeat = repeats[contract]
        blocks.append(
            f"| {label} | three fresh processes | {repeat['winner']} | "
            f"{repeat['gpu']} | {repeat['wall']} |"
        )

    blocks.extend(
        [
            "",
            "### Custom physical-batch ablation",
            "",
            "This control compares the Custom renderer with itself at the same "
            "logical batch and camera contract: one physical view per native "
            "submission versus all physical views in one native submission. A "
            "displayed speedup is P1 latency divided by P128 latency, so a ratio "
            "above parity favors P128 and a ratio below parity favors P1. This is "
            "an aggregate physical-submission-schedule control at logical batch "
            "128. It does not "
            "isolate a kernel mechanism, explain the cross-renderer result, or "
            "establish a batch-scaling law. The three fresh trials are reported "
            "without a win gate.",
            "",
            "| Contract | Trial | CUDA speedup of P128 over P1 (P1 latency / P128 latency) | Wall speedup of P128 over P1 (P1 latency / P128 latency) |",
            "|:---|---:|---:|---:|",
        ]
    )
    for contract, (_table_name, label) in TABLES.items():
        for trial in (1, 2, 3):
            blocks.append(
                "| {label} | {trial} | {cuda} | {wall} |".format(
                    label=label,
                    trial=trial,
                    cuda=_marker(
                        f"custom-ablation-{contract}-t{trial}-cuda-p128-over-p1",
                        display_by_id,
                    ),
                    wall=_marker(
                        f"custom-ablation-{contract}-t{trial}-wall-p128-over-p1",
                        display_by_id,
                    ),
                )
            )
    return "\n".join(blocks)


def _render_conclusion(states: dict[str, str]) -> tuple[str, str]:
    full = states["full"]
    rgb = states["rgb"]
    if full == "custom" and rgb == "custom":
        kind = "custom-uniform"
        result = (
            "Custom is the scoped empirical winner: every scheduled row favors "
            "it in both CUDA-event and synchronized-wall latency at every matched "
            "trajectory step for both output contracts."
        )
    elif full == "baseline" and rgb == "baseline":
        kind = "baseline-uniform"
        result = (
            "The FlashGS-derived matched-contract port is the scoped empirical "
            "winner: every scheduled row favors it in both CUDA-event and "
            "synchronized-wall latency at every matched trajectory step for both "
            "output contracts."
        )
    elif full == "custom" and rgb == "baseline":
        kind = "split-by-contract"
        result = (
            "The result splits by output contract: Custom is the scoped winner "
            "for full-sensor output, while the FlashGS-derived matched-contract "
            "port is the scoped winner for RGB-only output. There is no uniform "
            "renderer winner."
        )
    elif full == "baseline" and rgb == "custom":
        kind = "split-by-contract"
        result = (
            "The result splits by output contract: the FlashGS-derived matched-"
            "contract port is the scoped winner for full-sensor output, while "
            "Custom is the scoped winner for RGB-only output. There is no "
            "uniform renderer winner."
        )
    else:
        kind = "mixed"
        descriptions = {
            "custom": "uniformly favors Custom",
            "baseline": "uniformly favors the FlashGS-derived matched-contract port",
            "mixed": "does not support a uniform winner",
        }
        result = (
            f"The full-sensor matrix {descriptions[full]}, while the RGB-only "
            f"matrix {descriptions[rgb]}. The combined evidence does not support "
            "a uniform renderer winner."
        )
    block = (
        "## Scoped conclusion\n\n"
        + result
        + " This conclusion is limited to one identified L4, one scene, one "
        "authored route, and one low-resolution robotics-shaped renderer "
        "microbenchmark. Isaac/Fabric ingestion, physics, policy or model work, "
        "and deployed-robot behavior are outside the timed result. It is not an "
        "upstream FlashGS comparison, an RTX replacement claim, a linear-scaling "
        "claim, or a high-resolution result."
    )
    return kind, block


def _render_article(
    template: str,
    *,
    display_by_id: dict[str, str],
    states: dict[str, str],
    rows_by_contract: dict[str, list[dict[str, Any]]],
    repeats: dict[str, dict[str, str]],
) -> tuple[bytes, str]:
    conclusion_kind, conclusion = _render_conclusion(states)
    replacements = {
        STATUS_PLACEHOLDER: _render_status(),
        RESULTS_PLACEHOLDER: _render_results(
            display_by_id,
            rows_by_contract=rows_by_contract,
            repeats=repeats,
        ),
        CONCLUSION_PLACEHOLDER: conclusion,
    }
    article = template
    for placeholder in REQUIRED_PLACEHOLDERS:
        article = article.replace(placeholder, replacements[placeholder])
    if PLACEHOLDER_RE.search(article):
        raise FinalizationError("Generated article retains a publication placeholder.")
    try:
        markers = parse_claim_markers(article)
        reject_unmarked_performance_claims(article)
    except ClaimLedgerError as error:
        raise FinalizationError(str(error)) from error
    if set(markers) != set(display_by_id):
        raise FinalizationError("Generated article marker set differs from the generated claim set.")
    payload = article.encode("utf-8")
    return payload, conclusion_kind


def _render_diagnostic_article(
    template: str,
    *,
    mode: str,
    reasons: tuple[str, ...],
) -> bytes:
    labels = {
        "diagnostic-incomplete": (
            "incomplete matrix",
            "The predeclared matrix is incomplete. Missing rows remain missing; "
            "nothing is interpolated and no renderer winner is inferred.",
        ),
        "diagnostic-fidelity-failure": (
            "non-equivalent fidelity failure",
            "At least one candidate row failed the same-equation fidelity gate. "
            "Its timing is non-equivalent diagnostic evidence and cannot support "
            "a speedup, geometric mean, or renderer-winner claim.",
        ),
        "diagnostic-fairness-failure": (
            "failed publication gates",
            "The matrix did not satisfy every provenance, fairness, hardware, and "
            "headline gate. Its measurements remain diagnostic and cannot support "
            "a renderer-winner claim.",
        ),
    }
    if mode not in labels:
        raise FinalizationError(f"Unknown diagnostic article mode {mode!r}.")
    label, conclusion = labels[mode]
    status = (
        f"> **Benchmark evidence status — diagnostic ({label}), not publication "
        "eligible.** The finalizer intentionally omitted all latency, throughput, "
        "speedup, and aggregate values. No strict claim ledger was emitted."
    )
    result_lines = [
        "## Diagnostic matrix outcome",
        "",
        f"- Baseline label: {ACCEPTED_BASELINE_LABEL}",
        f"- Outcome: {label}",
    ]
    result_lines.extend(f"- Gate finding: {reason}" for reason in reasons)
    result_lines.extend(
        [
            "",
            "The frozen summary remains the diagnostic record. This article does "
            "not copy failed or partial performance values into publishable prose.",
        ]
    )
    scoped_conclusion = (
        "## Scoped conclusion\n\n"
        + conclusion
        + " This says nothing about upstream FlashGS, RTX or OVRTX replacement, "
        "other hardware, scenes, routes, or resolutions, end-to-end Isaac/Fabric "
        "execution, policy or model work, deployed robots, or proportional batch "
        "scaling."
    )
    replacements = {
        STATUS_PLACEHOLDER: status,
        RESULTS_PLACEHOLDER: "\n".join(result_lines),
        CONCLUSION_PLACEHOLDER: scoped_conclusion,
    }
    article = template
    for placeholder in REQUIRED_PLACEHOLDERS:
        article = article.replace(placeholder, replacements[placeholder])
    if PLACEHOLDER_RE.search(article):
        raise FinalizationError("Generated diagnostic article retains a publication placeholder.")
    try:
        markers = parse_claim_markers(article)
        reject_unmarked_performance_claims(article)
    except ClaimLedgerError as error:
        raise FinalizationError(str(error)) from error
    if markers:
        raise FinalizationError("Generated diagnostic article unexpectedly contains claim markers.")
    return article.encode("utf-8")


def _preflight_output(path: Path, payload: bytes) -> bool:
    """Return True when a byte-identical output already exists."""

    if path.is_symlink():
        raise FinalizationError(f"Publication output may not be a symlink: {path}.")
    if not path.exists():
        return False
    if not path.is_file():
        raise FinalizationError(f"Publication output is not a regular file: {path}.")
    if path.read_bytes() != payload:
        raise FinalizationError(f"Refusing to overwrite different publication output: {path}.")
    return True


def _write_exclusive(path: Path, payload: bytes) -> bool:
    """Create ``path`` without replacement; return whether it was created."""

    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return True
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FinalizationError(f"Concurrent different output appeared at {path}.")
        return False


def _finalize_diagnostic(
    *,
    root: Path,
    template: str,
    summary_record: dict[str, Any],
    mode: str,
    reasons: tuple[str, ...],
) -> dict[str, Any]:
    article_bytes = _render_diagnostic_article(
        template,
        mode=mode,
        reasons=reasons,
    )
    article_record = _artifact_record_from_bytes(
        article_bytes,
        logical_path=ARTICLE_LOGICAL_PATH,
    )
    article_path = root / ARTICLE_LOGICAL_PATH
    ledger_path = root / LEDGER_LOGICAL_PATH
    if ledger_path.exists() or ledger_path.is_symlink():
        raise FinalizationError(
            "Refusing to generate a diagnostic article beside an existing claim ledger."
        )
    article_exists = _preflight_output(article_path, article_bytes)
    article_path.parent.mkdir(parents=True, exist_ok=True)
    if not article_exists:
        _write_exclusive(article_path, article_bytes)
    return {
        "schema_version": FINALIZATION_SCHEMA,
        "pass": False,
        "generated": True,
        "publication_eligible": False,
        "mode": mode,
        "article": article_record,
        "summary": summary_record,
        "claim_count": 0,
        "claim_ledger": None,
        "conclusion": mode,
        "strict_verification_claim_count": None,
    }


def finalize(*, evidence_root: Path, template_path: Path) -> dict[str, Any]:
    root_view = EvidenceRoot.open(evidence_root)
    if root_view.mode != "staging":
        raise FinalizationError("Article finalization requires a staging root, not a relocated bundle.")
    root = root_view.root
    template = _read_template(template_path)
    _validate_template(template)

    summary_path = root_view.resolve_logical(SUMMARY_LOGICAL_PATH, label="Canonical matrix summary")
    summary_record = _artifact_record(summary_path, logical_path=SUMMARY_LOGICAL_PATH)
    summary = load_decimal_json(summary_path, label="Canonical matrix summary")
    outcome, diagnostic_reasons = _classify_summary(summary)
    if outcome != "complete":
        return _finalize_diagnostic(
            root=root,
            template=template,
            summary_record=summary_record,
            mode=outcome,
            reasons=diagnostic_reasons,
        )

    try:
        verification_path = root_view.resolve_logical(
            VERIFICATION_LOGICAL_PATH,
            label="Aggregate publication verification",
        )
        ablation_path = root_view.resolve_logical(
            ABLATION_LOGICAL_PATH,
            label="Canonical Custom vectorization ablation",
        )
    except (ClaimLedgerError, FileNotFoundError, OSError):
        return _finalize_diagnostic(
            root=root,
            template=template,
            summary_record=summary_record,
            mode="diagnostic-incomplete",
            reasons=("required post-matrix verification or schedule-control evidence is missing",),
        )
    verification_record = _artifact_record(
        verification_path,
        logical_path=VERIFICATION_LOGICAL_PATH,
    )
    ablation_record = _artifact_record(ablation_path, logical_path=ABLATION_LOGICAL_PATH)
    rows, states, repeats = _validate_summary(summary)
    try:
        validated_verification, validated_ablation, ablation = resolve_validated_ablation(
            root_view,
            verification_value=verification_record,
            ablation_value=ablation_record,
        )
    except ClaimLedgerError:
        return _finalize_diagnostic(
            root=root,
            template=template,
            summary_record=summary_record,
            mode="diagnostic-fairness-failure",
            reasons=("post-matrix verification or schedule-control evidence did not validate",),
        )
    if validated_verification != verification_record or validated_ablation != ablation_record:
        raise FinalizationError("Validated publication input identities changed during finalization.")

    claims, display_by_id = _build_claims(
        summary=summary,
        summary_record=summary_record,
        ablation=ablation,
        ablation_record=ablation_record,
    )
    article_bytes, conclusion_kind = _render_article(
        template,
        display_by_id=display_by_id,
        states=states,
        rows_by_contract=rows,
        repeats=repeats,
    )
    article_record = _artifact_record_from_bytes(article_bytes, logical_path=ARTICLE_LOGICAL_PATH)
    ledger = {
        "ablation": ablation_record,
        "article": article_record,
        "baseline_label": ACCEPTED_BASELINE_LABEL,
        "claims": claims,
        "pass": True,
        "schema_version": LEDGER_SCHEMA,
        "strict_schema_version": STRICT_LEDGER_SCHEMA,
        "summary": summary_record,
        "verification": verification_record,
    }
    ledger_bytes = canonical_json_bytes(ledger)

    article_path = root / ARTICLE_LOGICAL_PATH
    ledger_path = root / LEDGER_LOGICAL_PATH
    article_exists = _preflight_output(article_path, article_bytes)
    ledger_exists = _preflight_output(ledger_path, ledger_bytes)
    article_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[tuple[Path, bytes]] = []
    try:
        if not article_exists and _write_exclusive(article_path, article_bytes):
            created.append((article_path, article_bytes))
        if not ledger_exists and _write_exclusive(ledger_path, ledger_bytes):
            created.append((ledger_path, ledger_bytes))
        receipt = verify_claim_ledger(root)
    except Exception:
        for path, expected in reversed(created):
            if path.is_file() and not path.is_symlink() and path.read_bytes() == expected:
                path.unlink()
        raise

    return {
        "schema_version": FINALIZATION_SCHEMA,
        "pass": True,
        "generated": True,
        "publication_eligible": True,
        "mode": "complete",
        "article": article_record,
        "claim_count": len(claims),
        "claim_ledger": {
            "bytes": len(ledger_bytes),
            "logical_path": LEDGER_LOGICAL_PATH,
            "path": LEDGER_LOGICAL_PATH,
            "sha256": _sha256_bytes(ledger_bytes),
        },
        "conclusion": conclusion_kind,
        "strict_verification_claim_count": receipt["claim_count"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Staging root containing summary.json and publication evidence.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("post.md"),
        help="UTF-8 article template containing the three required placeholders.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        receipt = finalize(
            evidence_root=arguments.evidence_root,
            template_path=arguments.template,
        )
    except (FinalizationError, ClaimLedgerError, OSError, ValueError) as error:
        print(f"PUBLICATION_ARTICLE_FINALIZATION_FAILED {error}", file=sys.stderr)
        return 1
    print(canonical_json_bytes(receipt).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
