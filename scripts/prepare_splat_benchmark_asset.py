#!/usr/bin/env python3
"""Convert a supplied SOG/LOD Gaussian scene into a canonical benchmark PLY.

The script deliberately keeps the source asset outside the repository.  It
writes an auditable manifest beside a canonical PLY so the rendering and R2
publication workflows can use identical, hash-pinned Gaussian attributes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lod", type=int, default=0)
    parser.add_argument("--splat-transform", default="splat-transform")
    parser.add_argument("--author")
    parser.add_argument("--source-url")
    parser.add_argument("--license")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def natural_chunk_key(path: Path) -> tuple[int, int, str]:
    """Sort ``0_2`` before ``0_10`` without relying on platform collation."""
    match = re.fullmatch(r"(\d+)_(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"Unexpected LOD chunk directory: {path.parent}")
    return int(match.group(1)), int(match.group(2)), str(path)


def select_sog_inputs(source: Path, lod: int) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() != ".sog":
            raise ValueError(f"Expected a .sog archive, got: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not (source / "lod-meta.json").is_file():
        raise ValueError(
            f"Extracted SOG directory is missing lod-meta.json: {source}"
        )
    inputs = sorted(
        source.glob(f"{lod}_*/meta.json"),
        key=natural_chunk_key,
    )
    if not inputs:
        raise ValueError(f"No LOD {lod} meta.json chunks found in {source}")
    return inputs


def parse_license_file(source: Path) -> dict[str, str]:
    raw_text: str | None = None
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            try:
                raw_text = archive.read("license.txt").decode("utf-8")
            except KeyError:
                raw_text = None
    candidates = (
        [source.with_name("license.txt")]
        if source.is_file()
        else [source / "license.txt"]
    )
    for path in candidates:
        if path.is_file():
            raw_text = path.read_text(encoding="utf-8")
    if raw_text is not None:
        values: dict[str, str] = {}
        for line in raw_text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip().lower()] = value.strip()
        return {
            "title": values.get("title", ""),
            "author": values.get("author", ""),
            "source": values.get("source", ""),
            "license": values.get("license", ""),
            "license_url": values.get("license url", ""),
        }
    return {}


def read_ply_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(64 * 1024).split(b"end_header\n", 1)[0]
    for line in header.decode("ascii").splitlines():
        if line.startswith("element vertex "):
            return int(line.rsplit(" ", 1)[1])
    raise ValueError(f"PLY vertex count missing from {path}")


def command_version(command: str) -> str:
    completed = subprocess.run(
        [command, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_records(inputs: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in inputs
    ]


def source_inventory(source: Path) -> dict[str, Any]:
    """Hash every source byte needed to reproduce a directory conversion."""
    paths = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.is_file()
    )
    root = source.parent if source.is_file() else source
    records = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "file_count": len(records),
        "files": records,
        "relative_file_manifest_sha256": digest,
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    inputs = select_sog_inputs(source, args.lod)
    license_info = parse_license_file(source)
    metadata = {
        "title": license_info.get("title", args.scene_id),
        "author": args.author or license_info.get("author"),
        "source": args.source_url or license_info.get("source"),
        "license": args.license or license_info.get("license"),
        "license_url": license_info.get("license_url"),
    }
    missing = [key for key in ("author", "source", "license") if not metadata[key]]
    if missing:
        raise ValueError(
            "Missing attribution metadata "
            + ", ".join(missing)
            + "; add a license.txt or pass the corresponding option."
        )
    plan = {
        "scene_id": args.scene_id,
        "source": str(source),
        "lod": args.lod if source.is_dir() else None,
        "input_count": len(inputs),
        "inputs": [str(path) for path in inputs],
        "output_ply": str(args.output_ply.resolve()),
        "manifest": str(args.manifest.resolve()),
        "attribution": metadata,
    }
    if args.dry_run:
        print("SPLAT_PREPARE_PLAN " + json.dumps(plan, sort_keys=True))
        return

    if shutil.which(args.splat_transform) is None:
        raise FileNotFoundError(
            f"splat-transform executable is not available: {args.splat_transform}"
        )

    output_ply = args.output_ply.resolve()
    manifest_path = args.manifest.resolve()
    if (output_ply.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("Output exists; pass --overwrite to replace it.")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    command = [args.splat_transform, "--mem", *map(str, inputs), str(output_ply)]
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not output_ply.is_file():
        raise FileNotFoundError(output_ply)
    manifest = {
        "schema_version": "vectorized-gaussian-renderer-external-scene/v1",
        "dataset_id": args.scene_id,
        "source": {
            "format": "PlayCanvas SOG" if source.is_file() else "PlayCanvas SOG LOD",
            "local_path": str(source),
            "lod": args.lod if source.is_dir() else None,
            "inputs": source_records(inputs),
            "inventory": source_inventory(source),
        },
        "canonical_ply": {
            "path": str(output_ply),
            "byte_count": output_ply.stat().st_size,
            "sha256": sha256_file(output_ply),
            "gaussian_count": read_ply_count(output_ply),
            "format": "binary_little_endian",
        },
        "attribution": metadata,
        "conversion": {
            "tool": "splat-transform",
            "version": command_version(args.splat_transform),
            "command": command,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "SPLAT_PREPARE_OK "
        + json.dumps(
            {
                "count": manifest["canonical_ply"]["gaussian_count"],
                "manifest": str(manifest_path),
                "ply": str(output_ply),
                "sha256": manifest["canonical_ply"]["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
