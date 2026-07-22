#!/usr/bin/env python3
"""Validate, upload, and publicly verify versioned renderer artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


WRANGLER_MAX_BYTES = 315 * 1024 * 1024
RANGE_SAMPLE_BYTES = 64 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cloudflare/r2-assets.json"),
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=("wrangler", "multipart-worker"),
        help="Limit work to one upload method; may be repeated.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not upload; validate local inputs and public objects.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Validate local inputs without uploading or public requests.",
    )
    parser.add_argument(
        "--full-download-max-bytes",
        type=int,
        default=32 * 1024 * 1024,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "vectorized-gaussian-r2-assets/v1":
        raise ValueError(f"Unsupported publication manifest: {path}")
    return value


def selected_assets(
    manifest: dict[str, Any],
    methods: list[str] | None,
) -> list[dict[str, Any]]:
    allowed = set(methods or ("wrangler", "multipart-worker"))
    return [
        asset
        for asset in manifest["assets"]
        if asset["upload_method"] in allowed
    ]


def validate_local(root: Path, asset: dict[str, Any]) -> Path:
    source = (root / asset["source"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_bytes = source.stat().st_size
    if actual_bytes != int(asset["bytes"]):
        raise ValueError(
            f"{source}: expected {asset['bytes']} bytes, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(source)
    if actual_sha256 != asset["sha256"]:
        raise ValueError(
            f"{source}: expected SHA-256 {asset['sha256']}, "
            f"got {actual_sha256}"
        )
    print(
        "LOCAL_ASSET_OK "
        + json.dumps(
            {
                "bytes": actual_bytes,
                "key": asset["key"],
                "sha256": actual_sha256,
                "source": str(source),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return source


def upload_with_wrangler(
    root: Path,
    manifest: dict[str, Any],
    asset: dict[str, Any],
    source: Path,
) -> None:
    if source.stat().st_size > WRANGLER_MAX_BYTES:
        raise ValueError(
            f"{source} exceeds Wrangler's {WRANGLER_MAX_BYTES}-byte limit"
        )
    version = manifest["wrangler_version"]
    command = [
        "npx",
        "--yes",
        f"wrangler@{version}",
        "r2",
        "object",
        "put",
        f"{manifest['bucket']}/{asset['key']}",
        "--remote",
        "--file",
        str(source),
        "--content-type",
        asset["content_type"],
        "--content-disposition",
        asset["content_disposition"],
        "--cache-control",
        asset.get(
            "cache_control",
            manifest["default_cache_control"],
        ),
        "--force",
    ]
    print("UPLOAD " + asset["key"], flush=True)
    subprocess.run(command, cwd=root, check=True)


def request_bytes(
    request: urllib.request.Request,
    *,
    attempts: int = 6,
) -> tuple[int, Any, bytes]:
    if not request.has_header("User-agent"):
        request.add_header(
            "User-Agent",
            "vectorized-gaussian-renderer-r2-verifier/1.0",
        )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, response.headers, response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(min(2**attempt, 15))
    assert last_error is not None
    raise last_error


def public_url(manifest: dict[str, Any], asset: dict[str, Any]) -> str:
    return manifest["public_base_url"].rstrip("/") + "/" + asset["key"]


def verify_public(
    source: Path,
    manifest: dict[str, Any],
    asset: dict[str, Any],
    full_download_max_bytes: int,
) -> None:
    url = public_url(manifest, asset)
    status, headers, _ = request_bytes(
        urllib.request.Request(url, method="HEAD"),
    )
    if status != 200:
        raise ValueError(f"{url}: HEAD returned {status}")
    content_length = int(headers.get("Content-Length", "-1"))
    if content_length != int(asset["bytes"]):
        raise ValueError(
            f"{url}: expected Content-Length {asset['bytes']}, "
            f"got {content_length}"
        )
    content_type = headers.get_content_type()
    if content_type != asset["content_type"]:
        raise ValueError(
            f"{url}: expected Content-Type {asset['content_type']}, "
            f"got {content_type}"
        )

    size = int(asset["bytes"])
    if size <= full_download_max_bytes:
        get_status, _, body = request_bytes(
            urllib.request.Request(url, method="GET"),
        )
        digest = hashlib.sha256(body).hexdigest()
        if get_status != 200 or len(body) != size or digest != asset["sha256"]:
            raise ValueError(
                f"{url}: public download did not match the local artifact"
            )
        verification = "full-sha256"
    else:
        sample_size = min(RANGE_SAMPLE_BYTES, size)
        ranges = (
            (0, sample_size - 1),
            (size - sample_size, size - 1),
        )
        with source.open("rb") as handle:
            for start, end in ranges:
                handle.seek(start)
                expected = handle.read(end - start + 1)
                request = urllib.request.Request(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    method="GET",
                )
                range_status, range_headers, body = request_bytes(request)
                if (
                    range_status != 206
                    or body != expected
                    or range_headers.get("Content-Range")
                    != f"bytes {start}-{end}/{size}"
                ):
                    raise ValueError(
                        f"{url}: public byte range {start}-{end} "
                        "did not match the local artifact"
                    )
        verification = "head-and-tail-range"

    print(
        "PUBLIC_ASSET_OK "
        + json.dumps(
            {
                "bytes": size,
                "key": asset["key"],
                "url": url,
                "verification": verification,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent.parent
    manifest = load_manifest(manifest_path)
    assets = selected_assets(manifest, args.method)
    if not assets:
        raise ValueError("No assets selected.")

    sources: dict[str, Path] = {}
    for asset in assets:
        sources[asset["key"]] = validate_local(root, asset)

    if args.local_only:
        return

    if not args.verify_only:
        for asset in assets:
            if asset["upload_method"] == "wrangler":
                upload_with_wrangler(
                    root,
                    manifest,
                    asset,
                    sources[asset["key"]],
                )
            else:
                print(
                    "SKIP_MULTIPART_UPLOAD "
                    + json.dumps(
                        {
                            "key": asset["key"],
                            "reason": (
                                "Use scripts/upload_r2_large_object.py "
                                "through the temporary multipart Worker."
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    for asset in assets:
        verify_public(
            sources[asset["key"]],
            manifest,
            asset,
            args.full_download_max_bytes,
        )


if __name__ == "__main__":
    main()
