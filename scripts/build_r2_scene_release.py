#!/usr/bin/env python3
"""Build a hash-pinned R2 publication manifest for one prepared scene release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARTIFACT_TYPES = {
    "custom.mp4": ("video/mp4", "inline"),
    "ovrtx-perspective.mp4": ("video/mp4", "inline"),
    "side-by-side.mp4": ("video/mp4", "inline"),
    "side-by-side-preview.gif": ("image/gif", "inline"),
    "poster.png": ("image/png", "inline"),
    "storyboard.png": ("image/png", "inline"),
    "artifact-manifest.json": ("application/json", "inline"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--canonical-ply", type=Path, required=True)
    parser.add_argument("--flyby-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("cloudflare/r2-assets.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ValueError(f"Artifact must be inside repository root: {path}") from error


def asset(source: Path, key: str, content_type: str, disposition: str) -> dict[str, Any]:
    return {
        "source": source,
        "key": key,
        "content_type": content_type,
        "content_disposition": disposition,
        "upload_method": "wrangler" if source.stat().st_size <= 315 * 1024 * 1024 else "multipart-worker",
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def main() -> None:
    args = parse_args()
    base_path = args.base_manifest.resolve()
    root = base_path.parent.parent
    base = json.loads(base_path.read_text(encoding="utf-8"))
    scene_manifest = json.loads(args.scene_manifest.read_text(encoding="utf-8"))
    canonical = scene_manifest["canonical_ply"]
    scene_id = scene_manifest["dataset_id"]
    scene_hash_prefix = canonical["sha256"][:12]
    flyby_root = args.flyby_root.resolve()
    artifacts = flyby_root / "artifacts"
    required = [
        args.canonical_ply.resolve(),
        args.scene_manifest.resolve(),
        flyby_root / "camera-path.json",
        flyby_root / "camera-path.npz",
        *(artifacts / name for name in ARTIFACT_TYPES),
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(args.canonical_ply) != canonical["sha256"]:
        raise ValueError("Canonical PLY hash does not match scene manifest.")

    dataset_prefix = f"datasets/{scene_id}/{scene_hash_prefix}"
    flyby_prefix = f"flybys/{scene_id}/{args.release_id}"
    items = [
        asset(
            args.scene_manifest.resolve(),
            f"{dataset_prefix}/manifest.json",
            "application/json",
            "inline",
        ),
        asset(
            args.canonical_ply.resolve(),
            f"{dataset_prefix}/{scene_id}.ply",
            "application/octet-stream",
            f'attachment; filename="{scene_id}.ply"',
        ),
    ]
    for name, (content_type, disposition) in ARTIFACT_TYPES.items():
        items.append(asset(artifacts / name, f"{flyby_prefix}/{name}", content_type, disposition))
    for name, content_type, disposition in (
        ("camera-path.json", "application/json", "inline"),
        ("camera-path.npz", "application/octet-stream", "attachment"),
    ):
        items.append(asset(flyby_root / name, f"{flyby_prefix}/{name}", content_type, disposition))

    output = {
        "schema_version": "vectorized-gaussian-r2-assets/v1",
        "bucket": base["bucket"],
        "public_base_url": base["public_base_url"],
        "wrangler_version": base["wrangler_version"],
        "default_cache_control": base["default_cache_control"],
        "release": {
            "scene_id": scene_id,
            "scene_sha256": canonical["sha256"],
            "source": scene_manifest["attribution"]["source"],
            "author": scene_manifest["attribution"]["author"],
            "license": scene_manifest["attribution"]["license"],
            "license_url": scene_manifest["attribution"].get("license_url"),
        },
        "assets": [
            {**item, "source": repo_relative(root, item["source"])}
            for item in items
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "R2_SCENE_RELEASE_MANIFEST_OK "
        + json.dumps(
            {
                "assets": len(items),
                "manifest": str(args.output),
                "scene_id": scene_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
