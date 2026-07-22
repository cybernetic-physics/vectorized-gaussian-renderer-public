#!/usr/bin/env python3
"""Freeze the exact files and roles intended for matched-result publication."""

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
    write_publication_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument(
        "--roles",
        type=Path,
        required=True,
        help="JSON object mapping every input-relative path to an explicit role list.",
    )
    parser.add_argument(
        "--external-dependencies",
        type=Path,
        required=True,
        help="JSON list of content-addressed dependencies intentionally omitted from the bundle.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roles = json.loads(args.roles.read_text(encoding="utf-8"))
    external = json.loads(args.external_dependencies.read_text(encoding="utf-8"))
    if not isinstance(roles, dict):
        raise ValueError("--roles must contain a JSON object.")
    if not isinstance(external, list):
        raise ValueError("--external-dependencies must contain a JSON list.")
    payload = write_publication_inventory(
        input_root=args.input_root,
        bundle_name=args.bundle_name,
        roles_by_path=roles,
        external_dependencies=external,
        output_path=args.output,
    )
    print(
        "MATCHED_EVIDENCE_INVENTORY_OK "
        + json.dumps(
            {
                "artifact_count": len(payload["artifacts"]),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
