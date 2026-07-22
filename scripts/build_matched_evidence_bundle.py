#!/usr/bin/env python3
"""Build a closed, content-addressed matched-benchmark evidence bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "src/isaacsim_gaussian_renderer/evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from evidence_bundle import (  # noqa: E402
    MATCHED_BATCHES,
    SEMANTIC_VALIDATION_SCHEMA,
    build_evidence_bundle,
    file_sha256,
    sha256_json,
    write_deterministic_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--semantic-root",
        type=Path,
        required=True,
        help="Original matched result root whose host-local artifact paths are still live.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Exact clean benchmark checkout used to rerun the canonical read-only validator.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional deterministic uncompressed tar path outside the bundle root.",
    )
    return parser.parse_args()


def artifact_identity(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def semantic_validator(
    *,
    staged_root: Path,
    semantic_root: Path,
    project_root: Path,
) -> dict[str, object]:
    validator = (project_root / "benchmarks/summarize_flashgs_matched.py").resolve()
    staged_validator = (
        staged_root / "publication/validator/summarize_flashgs_matched.py"
    )
    staged_source_manifest = staged_root / "provenance/source-manifest.json"
    staged_summary = staged_root / "summary.json"
    original_summary = semantic_root / "summary.json"
    for path in (
        validator,
        staged_validator,
        staged_source_manifest,
        staged_summary,
        original_summary,
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Semantic-validation input is missing or symlinked: {path}.")
    if artifact_identity(validator) != artifact_identity(staged_validator):
        raise ValueError("Staged semantic validator differs from the executing validator.")
    source_manifest = json.loads(staged_source_manifest.read_text(encoding="utf-8"))
    tracked_validator = (source_manifest.get("tracked_source_files") or {}).get(
        "benchmarks/summarize_flashgs_matched.py"
    )
    if tracked_validator != artifact_identity(validator):
        raise ValueError("Semantic validator differs from the clean source manifest.")
    run = json.loads(
        (semantic_root / "runs/custom/full/b1.json").read_text(encoding="utf-8")
    )
    recorded_source_manifest = (
        (run.get("environment") or {}).get("source_provenance") or {}
    ).get("manifest")
    if not isinstance(recorded_source_manifest, dict) or {
        "bytes": recorded_source_manifest.get("bytes"),
        "sha256": recorded_source_manifest.get("sha256"),
    } != artifact_identity(staged_source_manifest):
        raise ValueError("Staged source manifest differs from the timed-run provenance.")
    if artifact_identity(original_summary) != artifact_identity(staged_summary):
        raise ValueError("Staged canonical summary differs from the original result root.")
    command = [
        sys.executable,
        str(validator),
        "--root",
        str(semantic_root.resolve()),
        "--batches",
        ",".join(str(value) for value in MATCHED_BATCHES),
        "--output",
        str(original_summary.resolve()),
        "--verify-existing",
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or "FLASHGS_MATCHED_SUMMARY_VERIFIED " not in completed.stdout:
        raise RuntimeError(
            "Canonical matched semantic validation failed with exit status "
            f"{completed.returncode}."
        )
    summary_payload = json.loads(staged_summary.read_text(encoding="utf-8"))
    if (
        summary_payload.get("pass") is not True
        or summary_payload.get("scientific_pass") is not True
        or summary_payload.get("headline_eligible") is not True
    ):
        raise ValueError("Canonical matched summary is not publication-eligible.")
    return {
        "batches": list(MATCHED_BATCHES),
        "pass": True,
        "recomputed_summary_sha256": sha256_json(summary_payload),
        "schema_version": SEMANTIC_VALIDATION_SCHEMA,
        "source_manifest": artifact_identity(staged_source_manifest),
        "summary": artifact_identity(staged_summary),
        "validator": artifact_identity(staged_validator),
    }


def main() -> None:
    args = parse_args()
    manifest = build_evidence_bundle(
        input_root=args.input_root,
        inventory_path=args.inventory,
        output_root=args.output_root,
        semantic_validator=lambda staged_root, _summary_path: semantic_validator(
            staged_root=staged_root,
            semantic_root=args.semantic_root.resolve(),
            project_root=args.project_root.resolve(),
        ),
    )
    result: dict[str, object] = {
        "bundle_root": str(args.output_root.resolve()),
        "manifest_id": manifest["manifest_id"],
        "object_count": len(manifest["objects"]),
    }
    if args.archive is not None:
        result["archive"] = write_deterministic_archive(
            args.output_root,
            args.archive,
        )
    print("MATCHED_EVIDENCE_BUNDLE_OK " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
