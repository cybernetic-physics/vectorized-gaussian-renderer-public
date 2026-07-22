#!/usr/bin/env python3
"""Upload one large R2 object through the authenticated multipart Worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument(
        "--content-type",
        default="application/octet-stream",
    )
    parser.add_argument(
        "--content-disposition",
        default="attachment",
    )
    parser.add_argument(
        "--cache-control",
        default="public, max-age=31536000, immutable",
    )
    parser.add_argument(
        "--part-size-mib",
        type=int,
        default=64,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def api_request(
    worker_url: str,
    token: str,
    path: str,
    *,
    method: str,
    body: bytes,
    content_type: str,
    attempts: int = 5,
) -> dict[str, Any]:
    url = worker_url.rstrip("/") + path
    display_path = path.split("?", 1)[0]
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "User-Agent": (
                    "vectorized-gaussian-renderer-r2-uploader/1.0"
                ),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = response.read()
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(
                        f"{method} {path} returned {response.status}"
                    )
                return json.loads(payload)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
        ) as error:
            last_error = error
            if attempt == attempts:
                break
            delay = min(2**attempt, 20)
            print(
                f"RETRY {method} {display_path} "
                f"attempt={attempt} delay={delay}s",
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    token = os.environ.get("R2_UPLOAD_TOKEN")
    if not token:
        raise RuntimeError("R2_UPLOAD_TOKEN is required.")
    source = args.file.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_bytes = source.stat().st_size
    if actual_bytes != args.bytes:
        raise ValueError(
            f"Expected {args.bytes} bytes, got {actual_bytes}: {source}"
        )
    actual_sha256 = sha256_file(source)
    if actual_sha256 != args.sha256:
        raise ValueError(
            f"Expected SHA-256 {args.sha256}, got {actual_sha256}: {source}"
        )
    part_size = args.part_size_mib * 1024 * 1024
    if part_size < 5 * 1024 * 1024 or part_size > 96 * 1024 * 1024:
        raise ValueError("--part-size-mib must be between 5 and 96.")

    create_body = json.dumps(
        {
            "key": args.key,
            "contentType": args.content_type,
            "contentDisposition": args.content_disposition,
            "cacheControl": args.cache_control,
            "sha256": args.sha256,
        }
    ).encode()
    created = api_request(
        args.worker_url,
        token,
        "/multipart/create",
        method="POST",
        body=create_body,
        content_type="application/json",
    )
    upload_id = created["uploadId"]
    parts: list[dict[str, Any]] = []
    try:
        with source.open("rb") as handle:
            part_number = 1
            uploaded = 0
            while True:
                block = handle.read(part_size)
                if not block:
                    break
                query = urllib.parse.urlencode(
                    {
                        "key": args.key,
                        "uploadId": upload_id,
                        "partNumber": part_number,
                    }
                )
                part = api_request(
                    args.worker_url,
                    token,
                    f"/multipart/part?{query}",
                    method="PUT",
                    body=block,
                    content_type="application/octet-stream",
                )
                parts.append(
                    {
                        "partNumber": int(part["partNumber"]),
                        "etag": part["etag"],
                    }
                )
                uploaded += len(block)
                print(
                    "MULTIPART_PROGRESS "
                    + json.dumps(
                        {
                            "bytes": uploaded,
                            "part": part_number,
                            "percent": round(
                                uploaded * 100.0 / actual_bytes,
                                3,
                            ),
                            "total_bytes": actual_bytes,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                part_number += 1

        complete_body = json.dumps(
            {
                "key": args.key,
                "uploadId": upload_id,
                "parts": parts,
            }
        ).encode()
        completed = api_request(
            args.worker_url,
            token,
            "/multipart/complete",
            method="POST",
            body=complete_body,
            content_type="application/json",
        )
        if int(completed["size"]) != actual_bytes:
            raise ValueError(
                f"Completed object has {completed['size']} bytes, "
                f"expected {actual_bytes}"
            )
        print(
            "MULTIPART_UPLOAD_OK "
            + json.dumps(
                {
                    "bytes": actual_bytes,
                    "key": args.key,
                    "parts": len(parts),
                    "sha256": actual_sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception:
        abort_body = json.dumps(
            {
                "key": args.key,
                "uploadId": upload_id,
            }
        ).encode()
        try:
            api_request(
                args.worker_url,
                token,
                "/multipart/abort",
                method="POST",
                body=abort_body,
                content_type="application/json",
                attempts=2,
            )
        except Exception as abort_error:
            print(f"MULTIPART_ABORT_FAILED {abort_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
