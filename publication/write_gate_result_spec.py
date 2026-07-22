#!/usr/bin/env python3
"""Write one immutable normalized-result spec for run_bounded_gate.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


def parse_evidence(value: str) -> tuple[str, str]:
    key, separator, raw_path = value.partition("=")
    if not separator or not key or not raw_path or not key.replace("_", "a").isalnum():
        raise argparse.ArgumentTypeError("--evidence must be SAFE_KEY=PATH")
    return key, raw_path


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--native-extension", type=Path, required=True)
    parser.add_argument("--scene-sha256", required=True)
    parser.add_argument("--gpu-uuid", help="Omit only for the Python unit gate.")
    parser.add_argument("--evidence", type=parse_evidence, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def regular(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink.")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file.")
    return resolved


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    checks_path = regular(args.checks, label="Checks JSON")
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    if not isinstance(checks, dict) or len(checks) < 3:
        raise ValueError("Checks JSON must be an object with at least three fields.")
    if (
        len(args.scene_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.scene_sha256)
    ):
        raise ValueError("Scene SHA-256 is malformed.")
    evidence: dict[str, str] = {}
    for key, path_value in args.evidence:
        if key in evidence:
            raise ValueError(f"Duplicate evidence key: {key}.")
        evidence[key] = str(regular(Path(path_value), label=f"Evidence {key}"))
    return {
        "schema_version": "publication-gate-result-spec-v1",
        "gate_id": args.gate_id,
        "identity": {
            "gpu_uuid": args.gpu_uuid,
            "native_extension": str(regular(args.native_extension, label="Native extension")),
            "scene_sha256": args.scene_sha256,
            "source_manifest": str(regular(args.source_manifest, label="Source manifest")),
        },
        "checks": checks,
        "evidence": dict(sorted(evidence.items())),
    }


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    specification = build_spec(args)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(specification, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print("PUBLICATION_GATE_RESULT_SPEC_OK " + json.dumps({"gate_id": args.gate_id, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
