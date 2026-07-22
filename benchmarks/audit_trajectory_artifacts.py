#!/usr/bin/env python3
"""Reopen and fail-closed audit one trajectory evaluation bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaacsim_gaussian_renderer.evaluation.artifacts import EvaluationArtifactBundle, audit_artifact_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_artifact_bundle(args.bundle)
    output = args.output or args.bundle / "artifact-audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output.parent.resolve() == args.bundle.resolve():
        EvaluationArtifactBundle(args.bundle).write_hash_manifest()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
