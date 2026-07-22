#!/usr/bin/env python3
"""Upload one artifact to Cloudflare R2 and verify its round-trip SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


WRANGLER_MAX_BYTES = 300 * 1024 * 1024
PART_BYTES = 256 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_remote_sha256(object_path: str) -> tuple[str, int]:
    with tempfile.TemporaryFile() as stderr_stream:
        process = subprocess.Popen(
            [
                "npx",
                "--yes",
                "wrangler@4.111.0",
                "r2",
                "object",
                "get",
                object_path,
                "--remote",
                "--pipe",
            ],
            stdout=subprocess.PIPE,
            stderr=stderr_stream,
        )
        if process.stdout is None:
            raise RuntimeError("Wrangler did not expose an object stream.")
        digest = hashlib.sha256()
        byte_count = 0
        for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
        return_code = process.wait()
        stderr_stream.seek(0)
        stderr = stderr_stream.read().decode("utf-8", errors="replace")
    if return_code:
        raise RuntimeError(f"R2 verification download failed with exit {return_code}: {stderr.strip()}")
    return digest.hexdigest(), byte_count


def upload_and_verify(
    *, bucket: str, key: str, path: Path, content_type: str
) -> dict[str, object]:
    object_path = f"{bucket}/{key}"
    local_sha256 = sha256_file(path)
    local_bytes = path.stat().st_size
    subprocess.run(
        [
            "npx",
            "--yes",
            "wrangler@4.111.0",
            "r2",
            "object",
            "put",
            object_path,
            "--remote",
            "--force",
            "--content-type",
            content_type,
            "--file",
            str(path),
        ],
        check=True,
    )
    remote_sha256, remote_bytes = verified_remote_sha256(object_path)
    passed = remote_sha256 == local_sha256 and remote_bytes == local_bytes
    return {
        "key": key,
        "bytes": local_bytes,
        "sha256": local_sha256,
        "round_trip_bytes": remote_bytes,
        "round_trip_sha256": remote_sha256,
        "verification": "pass" if passed else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--content-type", default="application/octet-stream")
    args = parser.parse_args()

    artifact = args.file.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if args.key.startswith("/") or ".." in Path(args.key).parts:
        raise ValueError("R2 key must be a safe relative object key.")
    local_sha256 = sha256_file(artifact)
    local_bytes = artifact.stat().st_size
    parts: list[dict[str, object]] = []
    index_object: dict[str, object] | None = None
    if local_bytes <= WRANGLER_MAX_BYTES:
        parts.append(
            upload_and_verify(
                bucket=args.bucket,
                key=args.key,
                path=artifact,
                content_type=args.content_type,
            )
        )
    else:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            with artifact.open("rb") as source:
                part_index = 0
                while True:
                    payload = source.read(PART_BYTES)
                    if not payload:
                        break
                    part_path = temporary_root / f"part-{part_index:05d}"
                    part_path.write_bytes(payload)
                    part_key = f"{args.key}.part-{part_index:05d}"
                    parts.append(
                        upload_and_verify(
                            bucket=args.bucket,
                            key=part_key,
                            path=part_path,
                            content_type="application/octet-stream",
                        )
                    )
                    part_path.unlink()
                    part_index += 1
            index_payload = {
                "schema_version": "r2-multipart-artifact-index-v1",
                "artifact_key": args.key,
                "artifact_bytes": local_bytes,
                "artifact_sha256": local_sha256,
                "part_bytes": PART_BYTES,
                "parts": [
                    {"key": part["key"], "bytes": part["bytes"], "sha256": part["sha256"]}
                    for part in parts
                ],
            }
            index_path = temporary_root / "index.json"
            index_path.write_text(
                json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            index_object = upload_and_verify(
                bucket=args.bucket,
                key=f"{args.key}.index.json",
                path=index_path,
                content_type="application/json",
            )
    passed = all(part["verification"] == "pass" for part in parts) and (
        index_object is None or index_object["verification"] == "pass"
    )
    receipt = {
        "schema_version": "r2-artifact-publication-v1",
        "bucket": args.bucket,
        "key": args.key,
        "bytes": local_bytes,
        "sha256": local_sha256,
        "parts": parts,
        "index_object": index_object,
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification": "pass" if passed else "fail",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
