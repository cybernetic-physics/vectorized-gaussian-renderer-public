#!/usr/bin/env python3
"""Download the selected public real-world Gaussian dataset and checksum it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


DATASET = {
    "dataset_id": "voxel51-gaussian-splatting-train-30000",
    "source": "Hugging Face dataset Voxel51/gaussian_splatting",
    "license_source": "https://huggingface.co/datasets/Voxel51/gaussian_splatting",
    "license": "apache-2.0",
    "revision": "6aaba1b1dd0439c0db464842f8a6600a37e0eae9",
    "url": "https://huggingface.co/datasets/Voxel51/gaussian_splatting/resolve/main/FO_dataset/train/point_cloud/iteration_30000/point_cloud.ply?download=true",
    "relative_path": "Voxel51_gaussian_splatting/FO_dataset/train/point_cloud/iteration_30000/point_cloud.ply",
    "expected_size_bytes": 266_542_260,
    "expected_sha256": (
        "a4c1906ce0256f5cb2255aa078d18b468dc01d8604f6eda1aaf22b235147a7a1"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/datasets/public-gaussian"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target = args.output_root / DATASET["relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        if shutil.which("curl"):
            subprocess.check_call(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "8",
                    "--retry-all-errors",
                    "--retry-delay",
                    "5",
                    "--connect-timeout",
                    "30",
                    "--continue-at",
                    "-",
                    "-o",
                    str(partial),
                    DATASET["url"],
                ]
            )
        else:
            with urllib.request.urlopen(DATASET["url"], timeout=60) as response:
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
        if partial.stat().st_size != DATASET["expected_size_bytes"]:
            raise RuntimeError(
                "Downloaded public Gaussian size mismatch: "
                f"expected {DATASET['expected_size_bytes']}, "
                f"got {partial.stat().st_size}."
            )
        downloaded_sha256 = sha256(partial)
        if downloaded_sha256 != DATASET["expected_sha256"]:
            raise RuntimeError(
                "Downloaded public Gaussian SHA-256 mismatch: "
                f"expected {DATASET['expected_sha256']}, "
                f"got {downloaded_sha256}."
            )
        partial.replace(target)

    target_sha256 = sha256(target)
    if target.stat().st_size != DATASET["expected_size_bytes"]:
        raise RuntimeError("Existing public Gaussian has the wrong size.")
    if target_sha256 != DATASET["expected_sha256"]:
        raise RuntimeError("Existing public Gaussian has the wrong SHA-256.")
    manifest = {
        **DATASET,
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "sha256": target_sha256,
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
