#!/usr/bin/env python3
"""Write a relocation-independent diff/manifest for the FlashGS adapter."""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.benchmark_manifest import file_sha256  # noqa: E402
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    artifact_record,
    load_verified_source_manifest,
)
from isaacsim_gaussian_renderer.flashgs_native_loader import (  # noqa: E402
    FLASHGS_CUDA_FLAGS,
    FLASHGS_CXX_FLAGS,
    FLASHGS_UPSTREAM_COMMIT,
)
from isaacsim_gaussian_renderer.flashgs_baseline_contract import (  # noqa: E402
    FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
    FLASHGS_MATCHED_PORT_CLASSIFICATION,
    require_flashgs_short_tail_repair,
)


SOURCE_PAIRS = (
    ("csrc/pybind.cpp", "adapter.cpp"),
    ("csrc/ops.h", "ops.h"),
    ("csrc/cuda_rasterizer/preprocess.cu", "preprocess.cu"),
    ("csrc/cuda_rasterizer/sort.cu", "sort.cu"),
    ("csrc/cuda_rasterizer/render.cu", "render.cu"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flashgs-source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_provenance = load_verified_source_manifest(
        args.source_manifest,
        project_root=PROJECT_ROOT,
    )
    upstream_root = args.flashgs_source.resolve()
    commit = subprocess.check_output(["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(upstream_root),
            "status",
            "--short",
            "--untracked-files=all",
        ],
        text=True,
    ).strip()
    if commit != FLASHGS_UPSTREAM_COMMIT or status:
        raise RuntimeError("FlashGS attestation requires the clean pinned upstream checkout.")
    adapter_root = PROJECT_ROOT / "src" / "isaacsim_gaussian_renderer" / "native" / "flashgs"
    diff_lines: list[str] = []
    source_pairs: list[dict[str, object]] = []
    for upstream_relative, adapter_relative in SOURCE_PAIRS:
        upstream_path = upstream_root / upstream_relative
        adapter_path = adapter_root / adapter_relative
        upstream_text = upstream_path.read_text(encoding="utf-8").splitlines(keepends=True)
        adapter_text = adapter_path.read_text(encoding="utf-8").splitlines(keepends=True)
        diff_lines.extend(
            difflib.unified_diff(
                upstream_text,
                adapter_text,
                fromfile=f"upstream/{upstream_relative}",
                tofile=f"adapter/{adapter_relative}",
            )
        )
        source_pairs.append(
            {
                "upstream_path": upstream_relative,
                "upstream_bytes": upstream_path.stat().st_size,
                "upstream_sha256": file_sha256(upstream_path),
                "adapter_path": str(adapter_path.relative_to(PROJECT_ROOT)),
                "adapter_bytes": adapter_path.stat().st_size,
                "adapter_sha256": file_sha256(adapter_path),
            }
        )
    render_path = adapter_root / "render.cu"
    repair_checks = require_flashgs_short_tail_repair(render_path.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    patch_path = args.output.with_suffix(".patch")
    patch_path.write_text("".join(diff_lines), encoding="utf-8")
    manifest = {
        "schema_version": FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": commit,
        "upstream_clean": True,
        "source_provenance": source_provenance,
        "source_pairs": source_pairs,
        "adapter_diff": artifact_record(patch_path),
        "baseline_classification": FLASHGS_MATCHED_PORT_CLASSIFICATION,
        "correctness_repair_audit": {
            "render_source": artifact_record(render_path),
            "checks": repair_checks,
            "pass": True,
        },
        "build_flags": {
            "cxx": list(FLASHGS_CXX_FLAGS),
            "cuda": list(FLASHGS_CUDA_FLAGS),
        },
        "pass": True,
    }
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "FLASHGS_ADAPTER_ATTESTATION_OK "
        + json.dumps(
            {"manifest": str(args.output), "patch": str(patch_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
