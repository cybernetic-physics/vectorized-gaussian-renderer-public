#!/usr/bin/env python3
"""Write exact role and external-dependency inputs for the evidence builder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlsplit


HOME_SCAN = {
    "bytes": 1_203_883_212,
    "content_addressed_key": (
        "datasets/home-scan-lod0/29cee1594654/home-scan-lod0.ply"
    ),
    "dependency_id": "home-scan-lod0",
    "license": "CC-BY-4.0",
    "media_type": "application/octet-stream",
    "roles": ["scene-dataset"],
    "sha256": "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267",
}


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--retrieval-manifest-url", required=True)
    parser.add_argument("--roles-output", type=Path, required=True)
    parser.add_argument("--external-dependencies-output", type=Path, required=True)
    return parser.parse_args(arguments)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def required_paths(source_root: Path) -> dict[str, frozenset[str]]:
    module_path = (
        source_root
        / "src/isaacsim_gaussian_renderer/evaluation/evidence_bundle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "publication_evidence_bundle_contract",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen evidence-bundle contract.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module._matched_required_paths()
    if not isinstance(value, dict) or not value:
        raise RuntimeError("Frozen evidence-bundle required path set is empty.")
    return value


def safe_relative(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in ("", ".", "..") for part in parsed.parts):
        raise ValueError(f"Unsafe staged path: {relative!r}")
    return relative


def build_roles(stage: Path, required: dict[str, frozenset[str]]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for path in sorted(stage.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Staged evidence may not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = safe_relative(stage, path)
        if relative.lower().endswith(".ply"):
            raise ValueError(f"The external Home Scan PLY may not be staged: {relative}")
        roles[relative] = sorted(required.get(relative, {"supporting-artifact"}))
    missing = sorted(set(required) - set(roles))
    if missing:
        raise FileNotFoundError(
            "Evidence stage is missing required publication paths: "
            + ", ".join(missing[:20])
        )
    return roles


def validate_manifest_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Retrieval manifest URL must be plain credential-free HTTPS.")
    expected_digest = "b231602598b1eb039175dcb7edbd475167c2fc92011c5e975a4727c62b9f74b9"
    if expected_digest not in parsed.path:
        raise ValueError("Retrieval manifest URL must contain the full manifest SHA-256.")
    return value


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    stage = args.stage_root.resolve(strict=True)
    source = args.source_root.resolve(strict=True)
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("Evidence stage is missing or symlinked.")
    required = required_paths(source)
    roles = build_roles(stage, required)
    dependency: dict[str, Any] = dict(HOME_SCAN)
    dependency["retrieval_manifest_url"] = validate_manifest_url(
        args.retrieval_manifest_url
    )
    write_new(args.roles_output, canonical_json_bytes(roles))
    write_new(
        args.external_dependencies_output,
        canonical_json_bytes([dependency]),
    )
    print(
        "PUBLICATION_BUNDLE_INPUTS_OK "
        + json.dumps(
            {
                "artifact_count": len(roles),
                "external_dependency_count": 1,
                "roles": str(args.roles_output.resolve()),
                "external_dependencies": str(
                    args.external_dependencies_output.resolve()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
