"""Merge isolated OVRTX matrix reruns without rewriting producer evidence."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--override",
        type=Path,
        action="append",
        required=True,
        help=(
            "A later matrix manifest whose cases replace matching base cases. "
            "May be supplied more than once; later arguments win."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("cases"), list):
        raise ValueError(f"{path} does not contain a matrix case list.")
    return data


def case_key(case: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(case["scene"]),
        int(case["batch"]),
        int(case["width"]),
        int(case["height"]),
    )


def main() -> None:
    args = parse_args()
    manifests = [
        (args.base, load_manifest(args.base)),
        *[
            (path, load_manifest(path))
            for path in args.override
        ],
    ]
    base_cases = manifests[0][1]["cases"]
    expected_keys = {
        case_key(case)
        for case in base_cases
    }
    merged: dict[
        tuple[str, int, int, int],
        dict[str, Any],
    ] = {}
    replacement_counts: dict[str, int] = {}
    for path, manifest in manifests:
        replaced = 0
        for original in manifest["cases"]:
            key = case_key(original)
            if path != args.base and key not in expected_keys:
                raise ValueError(
                    f"Override {path} contains an unknown case: {key}."
                )
            case = dict(original)
            case["source_manifest"] = str(path)
            if key in merged:
                replaced += 1
            merged[key] = case
        replacement_counts[str(path)] = replaced

    missing = expected_keys - merged.keys()
    if missing:
        raise ValueError(
            "Merged matrix is missing base cases: "
            + ", ".join(str(key) for key in sorted(missing))
        )
    cases = [
        merged[key]
        for key in sorted(
            expected_keys,
            key=lambda item: (
                item[0],
                item[2],
                item[3],
                item[1],
            ),
        )
    ]
    status_counts: dict[str, int] = {}
    for case in cases:
        status = str(case["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    output = {
        "schema_version": "ovrtx-matrix-merged/v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "complete": all(
            bool(manifest.get("complete"))
            for _path, manifest in manifests
        ),
        "case_count": len(cases),
        "status_counts": status_counts,
        "source_manifests": [
            str(path)
            for path, _manifest in manifests
        ],
        "replacement_counts": replacement_counts,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
