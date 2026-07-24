#!/usr/bin/env python3
"""Publication-local positive and mutation tests for verify_claim_ledger.py."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from verify_claim_ledger import (
    ABLATION_GATE_ID,
    ABLATION_LOGICAL_PATH,
    ABLATION_SCHEMA,
    ACCEPTED_BASELINE_LABEL,
    ACCEPTED_EQUATION_CONTRACT,
    ACCEPTED_HARDWARE,
    ACCEPTED_HARDWARE_NAME,
    ACCEPTED_SCENE,
    ALLOWED_RELEASE_CLAIMS,
    ARTICLE_LOGICAL_PATH,
    BUNDLE_SCHEMA,
    LEDGER_LOGICAL_PATH,
    LEDGER_SCHEMA,
    PRIMARY_BATCHES,
    REQUIRED_RELEASE_CLAIM_IDS,
    STRICT_LEDGER_SCHEMA,
    SUMMARY_SCHEMA,
    VERIFICATION_LOGICAL_PATH,
    VERIFICATION_SCHEMA,
    ClaimLedgerError,
    canonical_json_bytes,
    file_sha256,
    format_result,
    load_decimal_json,
    normalized_format,
    numeric,
    recompute,
    resolve_json_pointer,
    verify_claim_ledger,
)

TEST_GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "logical_path": path.relative_to(root).as_posix(),
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def fixed(precision: int, suffix: str = "", *, grouping: bool = False) -> dict[str, Any]:
    return {
        "notation": "fixed",
        "precision": precision,
        "prefix": "",
        "suffix": suffix,
        "grouping": grouping,
        "trim_trailing_zeros": False,
    }


def claim(
    claim_id: str,
    display_value: str,
    artifact: dict[str, Any],
    fields: list[str],
    operation: str,
    format_specification: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "article_location": f"claim:{claim_id}",
        "baseline_label": ALLOWED_RELEASE_CLAIMS[claim_id]["baseline_label"],
        "claim_id": claim_id,
        "display_value": display_value,
        "equation_contract": ACCEPTED_EQUATION_CONTRACT,
        "format": format_specification,
        "hardware": ACCEPTED_HARDWARE,
        "json_fields": fields,
        "operation": operation,
        "output_contract": ALLOWED_RELEASE_CLAIMS[claim_id]["output_contract"],
        "scene": ACCEPTED_SCENE,
    }


def marker(claim_id: str, display_value: str) -> str:
    return f"<!-- claim:{claim_id} -->{display_value}<!-- /claim:{claim_id} -->"


def create_staging(root: Path) -> None:
    summary_path = root / "summary.json"
    full_rows = []
    rgb_rows = []
    for batch in PRIMARY_BATCHES:
        row = {
            "batch": batch,
            "pass": True,
            "custom_ms": float(batch),
            "flashgs_ms": float(batch) * 2.0,
        }
        full_rows.append(dict(row))
        rgb_rows.append(dict(row))
    write_json(
        summary_path,
        {
            "schema_version": SUMMARY_SCHEMA,
            "pass": True,
            "scientific_pass": True,
            "headline_eligible": True,
            "hardware_scope": {
                "gpu_name": ACCEPTED_HARDWARE_NAME,
                "gpu_uuid": TEST_GPU_UUID,
            },
            "primary_full_sensor_dynamic_table": full_rows,
            "rgb_only_dynamic_table": rgb_rows,
            "aggregate": {
                "full_sensor_geometric_mean_speedup_custom_over_flashgs": 2.0,
                "rgb_geometric_mean_speedup_custom_over_flashgs": 2.0,
            },
        },
    )
    summary_record = record(summary_path, root=root)

    ablation_path = root / ABLATION_LOGICAL_PATH
    ablation_rows = {
        "full": [
            {"trial": 1, "cuda_speedup_p128_over_p1": 0.80, "wall_speedup_p128_over_p1": 0.75},
            {"trial": 2, "cuda_speedup_p128_over_p1": 1.10, "wall_speedup_p128_over_p1": 1.05},
            {"trial": 3, "cuda_speedup_p128_over_p1": 0.90, "wall_speedup_p128_over_p1": 0.85},
        ],
        "rgb": [
            {"trial": 1, "cuda_speedup_p128_over_p1": 1.20, "wall_speedup_p128_over_p1": 1.10},
            {"trial": 2, "cuda_speedup_p128_over_p1": 0.95, "wall_speedup_p128_over_p1": 0.90},
            {"trial": 3, "cuda_speedup_p128_over_p1": 1.05, "wall_speedup_p128_over_p1": 1.00},
        ],
    }
    run_ids = [f"{contract}-t{trial}-p{physical}" for contract in ("full", "rgb") for trial in (1, 2, 3) for physical in (1, 128)]
    write_json(
        ablation_path,
        {
            "schema_version": ABLATION_SCHEMA,
            "pass": True,
            "batch": 128,
            "runs": [{"run_id": run_id} for run_id in run_ids],
            "run_order": run_ids,
            "ratios": ablation_rows,
        },
    )
    ablation_record = record(ablation_path, root=root)

    gate_result_path = root / "publication/evidence/custom-vectorization-ablation-gate-result.json"
    raw_identity = {key: ablation_record[key] for key in ("bytes", "path", "sha256")}
    write_json(
        gate_result_path,
        {
            "schema_version": "publication-gate-result-v1",
            "pass": True,
            "gate_id": ABLATION_GATE_ID,
            "created_at": "2026-01-01T00:00:01+00:00",
            "checks": {"design_cells": 12, "reported_without_win_requirement": True},
            "identity": {},
            "evidence": {"raw_result": raw_identity},
        },
    )
    gate_result_record = record(gate_result_path, root=root)
    gate_result_identity = {key: gate_result_record[key] for key in ("bytes", "path", "sha256")}
    verification_path = root / VERIFICATION_LOGICAL_PATH
    write_json(
        verification_path,
        {
            "schema_version": VERIFICATION_SCHEMA,
            "pass": True,
            "created_at": "2026-01-01T00:00:02+00:00",
            "benchmark": {},
            "matrix": {},
            "required_gate_ids": [ABLATION_GATE_ID],
            "gates": [{"gate_id": ABLATION_GATE_ID, "result": gate_result_identity}],
        },
    )
    verification_record = record(verification_path, root=root)

    optional_ablation_statistics = {
        claim_id
        for claim_id, binding in ALLOWED_RELEASE_CLAIMS.items()
        if binding["artifact_role"] == "ablation" and binding["operation"] in {"min", "median", "max"}
    }
    published_claim_ids = sorted(REQUIRED_RELEASE_CLAIM_IDS | optional_ablation_statistics)
    targets = {
        "summary": (summary_record, load_decimal_json(summary_path, label="fixture summary")),
        "ablation": (ablation_record, load_decimal_json(ablation_path, label="fixture ablation")),
    }
    specifications = []
    for claim_id in published_claim_ids:
        binding = ALLOWED_RELEASE_CLAIMS[claim_id]
        target_record, target = targets[binding["artifact_role"]]
        format_specification = fixed(2, "×")
        values = [
            numeric(resolve_json_pointer(target, pointer), label=f"fixture {claim_id} {pointer}")
            for pointer in binding["json_fields"]
        ]
        display = format_result(
            recompute(binding["operation"], values),
            normalized_format(format_specification),
        )
        specifications.append(
            (
                claim_id,
                display,
                target_record,
                binding["json_fields"],
                binding["operation"],
                format_specification,
            )
        )
    article = "# Fixture publication\n\n" + "\n\n".join(
        f"{claim_id}: {marker(claim_id, display)}"
        for claim_id, display, _artifact, _fields, _operation, _format in specifications
    ) + "\n"
    article_path = root / ARTICLE_LOGICAL_PATH
    article_path.parent.mkdir(parents=True, exist_ok=True)
    article_path.write_text(article, encoding="utf-8")
    ledger = {
        "ablation": ablation_record,
        "article": record(article_path, root=root),
        "baseline_label": ACCEPTED_BASELINE_LABEL,
        "claims": [
            claim(claim_id, display, artifact, fields, operation, format_specification)
            for claim_id, display, artifact, fields, operation, format_specification in specifications
        ],
        "pass": True,
        "schema_version": LEDGER_SCHEMA,
        "summary": summary_record,
        "strict_schema_version": STRICT_LEDGER_SCHEMA,
        "verification": verification_record,
    }
    write_json(root / LEDGER_LOGICAL_PATH, ledger)


def refresh_article_record(root: Path) -> None:
    ledger_path = root / LEDGER_LOGICAL_PATH
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["article"] = record(root / ARTICLE_LOGICAL_PATH, root=root)
    write_json(ledger_path, ledger)


def mutate_ledger(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    ledger_path = root / LEDGER_LOGICAL_PATH
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    mutation(ledger)
    write_json(ledger_path, ledger)


def create_bundle(staging: Path, bundle: Path) -> None:
    publication_artifacts: list[dict[str, Any]] = []
    for logical_path in (
        "summary.json",
        ABLATION_LOGICAL_PATH,
        "publication/evidence/custom-vectorization-ablation-gate-result.json",
        VERIFICATION_LOGICAL_PATH,
        ARTICLE_LOGICAL_PATH,
        LEDGER_LOGICAL_PATH,
    ):
        source = staging / logical_path
        sha256 = file_sha256(source)
        target = bundle / "objects" / "sha256" / sha256[:2] / sha256
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        publication_artifacts.append(
            {
                "bytes": source.stat().st_size,
                "logical_path": logical_path,
                "media_type": "application/json" if logical_path.endswith(".json") else "text/markdown",
                "object_path": f"objects/sha256/{sha256[:2]}/{sha256}",
                "roles": ["supporting-artifact"],
                "sha256": sha256,
            }
        )
    write_json(
        bundle / "manifest.json",
        {
            "schema_version": BUNDLE_SCHEMA,
            "publication_artifacts": sorted(publication_artifacts, key=lambda item: item["logical_path"]),
        },
    )


def expect_failure(name: str, pristine: Path, mutation: Callable[[Path], None]) -> str:
    case = pristine.parent / f"mutation-{name}"
    shutil.copytree(pristine, case)
    mutation(case)
    try:
        verify_claim_ledger(case)
    except ClaimLedgerError as error:
        return str(error)
    raise AssertionError(f"Mutation {name!r} unexpectedly passed strict claim verification.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="publication-claim-ledger-self-test-") as raw:
        temporary = Path(raw)
        pristine = temporary / "staging"
        create_staging(pristine)
        staging_receipt = verify_claim_ledger(pristine)
        assert staging_receipt["pass"] is True
        assert staging_receipt["root_mode"] == "staging"
        assert staging_receipt["claim_count"] >= len(REQUIRED_RELEASE_CLAIM_IDS)
        verified_by_id = {item["claim_id"]: item for item in staging_receipt["claims"]}
        assert verified_by_id["custom-ablation-full-t1-cuda-p128-over-p1"]["display_value"] == "0.80×"
        assert "custom-ablation-full-cuda-median-p128-over-p1" in verified_by_id

        bundle = temporary / "bundle"
        create_bundle(pristine, bundle)
        bundle_receipt = verify_claim_ledger(bundle)
        assert bundle_receipt["pass"] is True
        assert bundle_receipt["root_mode"] == "bundle"
        assert bundle_receipt["claim_count"] == staging_receipt["claim_count"]

        failures: dict[str, str] = {}
        failures["changed-display"] = expect_failure(
            "changed-display",
            pristine,
            lambda root: mutate_ledger(
                root,
                lambda ledger: ledger["claims"][1].__setitem__("display_value", "999.00×"),
            ),
        )

        def remove_marker(root: Path) -> None:
            article_path = root / ARTICLE_LOGICAL_PATH
            article = article_path.read_text(encoding="utf-8")
            ledger = json.loads((root / LEDGER_LOGICAL_PATH).read_text(encoding="utf-8"))
            claim_id = ledger["claims"][0]["claim_id"]
            article = article.replace(f"<!-- claim:{claim_id} -->", "").replace(
                f"<!-- /claim:{claim_id} -->",
                "",
            )
            article_path.write_text(article, encoding="utf-8")
            refresh_article_record(root)

        failures["missing-marker"] = expect_failure("missing-marker", pristine, remove_marker)
        failures["missing-claim"] = expect_failure(
            "missing-claim",
            pristine,
            lambda root: mutate_ledger(root, lambda ledger: ledger["claims"].pop()),
        )
        failures["bad-pointer"] = expect_failure(
            "bad-pointer",
            pristine,
            lambda root: mutate_ledger(
                root,
                lambda ledger: ledger["claims"][0].__setitem__("json_fields", ["/absent"]),
            ),
        )
        failures["forbidden-label"] = expect_failure(
            "forbidden-label",
            pristine,
            lambda root: mutate_ledger(
                root,
                lambda ledger: ledger.__setitem__("baseline_label", "upstream FlashGS"),
            ),
        )

        def corrupt_article(root: Path) -> None:
            with (root / ARTICLE_LOGICAL_PATH).open("ab") as stream:
                stream.write(b"corruption")

        failures["article-hash"] = expect_failure("article-hash", pristine, corrupt_article)
        def add_unmarked_overclaim(root: Path) -> None:
            article_path = root / ARTICLE_LOGICAL_PATH
            article_path.write_text(
                article_path.read_text(encoding="utf-8")
                + "\nOur renderer is 999999x faster than upstream FlashGS on every robot.\n",
                encoding="utf-8",
            )
            refresh_article_record(root)

        failures["unmarked-performance"] = expect_failure(
            "unmarked-performance",
            pristine,
            add_unmarked_overclaim,
        )

        def add_unmarked_capacity_disclosure(root: Path) -> None:
            article_path = root / ARTICLE_LOGICAL_PATH
            article_path.write_text(
                article_path.read_text(encoding="utf-8")
                + "\nThe calibrated workspace peaked at 22.03 GiB.\n",
                encoding="utf-8",
            )
            refresh_article_record(root)

        failures["unmarked-capacity"] = expect_failure(
            "unmarked-capacity",
            pristine,
            add_unmarked_capacity_disclosure,
        )
        failures["false-metadata"] = expect_failure(
            "false-metadata",
            pristine,
            lambda root: mutate_ledger(
                root,
                lambda ledger: ledger["claims"][0].__setitem__("hardware", "imaginary GPU"),
            ),
        )

        def arbitrary_artifact(root: Path) -> None:
            fabricated = root / "fabricated.json"
            write_json(fabricated, {"value": 999999})
            mutate_ledger(
                root,
                lambda ledger: ledger["claims"][0].__setitem__(
                    "artifact",
                    record(fabricated, root=root),
                ),
            )

        failures["arbitrary-artifact"] = expect_failure(
            "arbitrary-artifact",
            pristine,
            arbitrary_artifact,
        )

        def replace_aggregate_bound_ablation(root: Path) -> None:
            ablation_path = root / ABLATION_LOGICAL_PATH
            ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
            ablation["ratios"]["full"][0]["cuda_speedup_p128_over_p1"] = 999999.0
            write_json(ablation_path, ablation)
            replacement = record(ablation_path, root=root)

            def update(ledger: dict[str, Any]) -> None:
                ledger["ablation"] = replacement
                for item in ledger["claims"]:
                    if ALLOWED_RELEASE_CLAIMS[item["claim_id"]]["artifact_role"] == "ablation":
                        item["artifact"] = replacement

            mutate_ledger(root, update)

        failures["aggregate-ablation-binding"] = expect_failure(
            "aggregate-ablation-binding",
            pristine,
            replace_aggregate_bound_ablation,
        )

        def remove_required_claim_and_marker(root: Path) -> None:
            ledger_path = root / LEDGER_LOGICAL_PATH
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            removed = ledger["claims"].pop()
            claim_id = removed["claim_id"]
            display = removed["display_value"]
            article_path = root / ARTICLE_LOGICAL_PATH
            article = article_path.read_text(encoding="utf-8")
            article = article.replace(
                f"{claim_id}: {marker(claim_id, display)}\n",
                "",
            )
            article_path.write_text(article, encoding="utf-8")
            ledger["article"] = record(article_path, root=root)
            write_json(ledger_path, ledger)

        failures["required-claim"] = expect_failure(
            "required-claim",
            pristine,
            remove_required_claim_and_marker,
        )
        assert set(failures) == {
            "article-hash",
            "aggregate-ablation-binding",
            "arbitrary-artifact",
            "bad-pointer",
            "changed-display",
            "false-metadata",
            "forbidden-label",
            "missing-claim",
            "missing-marker",
            "required-claim",
            "unmarked-capacity",
            "unmarked-performance",
        }
        print(
            "PUBLICATION_CLAIM_LEDGER_SELF_TEST_OK "
            + json.dumps(
                {
                    "required_release_claims": len(REQUIRED_RELEASE_CLAIM_IDS),
                    "bundle_claim_count": bundle_receipt["claim_count"],
                    "mutation_failures": sorted(failures),
                    "staging_claim_count": staging_receipt["claim_count"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
