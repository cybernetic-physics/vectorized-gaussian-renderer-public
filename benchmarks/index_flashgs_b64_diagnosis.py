#!/usr/bin/env python3
"""Freeze five pre-fix FlashGS traces and all 92 historical mismatches."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.evaluation.matched_semantics import (  # noqa: E402
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    SEMANTIC_TOPOLOGIES,
)
from isaacsim_gaussian_renderer.flashgs_repair import (  # noqa: E402
    build_b64_diagnosis_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and index exactly five historical B64 pipeline traces, "
            "the hash-pinned full FlashGS output, and its 92 oracle mismatches."
        )
    )
    parser.add_argument("--known-failure-manifest", type=Path, required=True)
    parser.add_argument("--diagnosis-lock", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Common root used for every portable relative artifact path.",
    )
    parser.add_argument(
        "--diagnosis",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly five times, once for each pre-fix trace JSON.",
    )
    parser.add_argument("--historical-run", type=Path, required=True)
    parser.add_argument("--historical-flashgs-output", type=Path, required=True)
    parser.add_argument("--pinned-oracle-output", type=Path, required=True)
    parser.add_argument("--camera-bundle", type=Path, required=True)
    parser.add_argument(
        "--semantic-topology",
        choices=SEMANTIC_TOPOLOGIES,
        default=REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnosis index: {args.output}.")
    payload = build_b64_diagnosis_index(
        args.known_failure_manifest,
        args.diagnosis,
        diagnosis_lock=args.diagnosis_lock,
        artifact_root=args.artifact_root,
        historical_run=args.historical_run,
        historical_flashgs_output=args.historical_flashgs_output,
        pinned_oracle_output=args.pinned_oracle_output,
        camera_bundle_path=args.camera_bundle,
        semantic_topology=args.semantic_topology,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "FLASHGS_B64_DIAGNOSIS_INDEX_OK "
        + json.dumps(
            {
                "output": str(args.output),
                "cases": payload["case_count"],
                "culprits": payload["culprit_count"],
                "mismatch_pixels": payload["historical_mismatch_corpus"]["mismatch_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
