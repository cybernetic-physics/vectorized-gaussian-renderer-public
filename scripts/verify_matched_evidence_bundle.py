#!/usr/bin/env python3
"""Verify a relocated matched-benchmark evidence bundle and optional tar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "src/isaacsim_gaussian_renderer/evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from evidence_bundle import (  # noqa: E402
    verify_deterministic_archive,
    verify_evidence_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = verify_evidence_bundle(args.bundle_root)
    result: dict[str, object] = {
        "bundle_root": str(args.bundle_root.resolve()),
        "manifest_id": manifest["manifest_id"],
        "object_count": len(manifest["objects"]),
    }
    if args.archive is not None:
        result["archive"] = verify_deterministic_archive(
            args.bundle_root,
            args.archive,
        )
    print("MATCHED_EVIDENCE_BUNDLE_VERIFIED " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
