#!/usr/bin/env python3
"""Parse Nsight Systems CSV summaries into stage-level JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGES = {
    "lidar_trace": ("trace_lidar_kernel",),
    "lidar_output_init": ("clear_outputs_kernel",),
    "lidar_call_marker": ("begin_call_kernel", "end_call_kernel"),
    "lidar_lbvh_build": (
        "build_leaf_bounds_kernel",
        "build_internal_bounds_kernel",
        "compute_morton_codes_kernel",
        "pack_descriptor_kernel",
    ),
    "projected_intersection_emission": (
        "emit_cached_projected_intersections",
    ),
    "projection": ("projection", "project", "fully_fused"),
    "tile_grouping": (
        "count_intersections_by_tile",
        "initialize_tile_scatter",
        "scatter_intersections_by_tile",
        "mark_sorted_tile_ranges",
    ),
    "tile_intersection": (
        "intersect_tile",
        "isect",
        "intersection",
        "bin_tiles",
    ),
    "offsets": (
        "intersect_offset",
        "offset",
        "find_dense_tile_ranges",
        "tile_ranges",
        "devicescan",
    ),
    "sorting": (
        "radixsort",
        "radix_sort",
        "sortpairs",
        "segmentedradixsort",
        "build_deterministic_keys",
    ),
    "raster": (
        "rasterize_to_pixels",
        "rasterize",
        "render",
        "composite_tiles",
        "composite",
    ),
    "concat_or_fill": ("cat", "concatenate", "fill", "copy", "memset"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def numeric(value: str | None, *, field: str = "numeric value") -> float:
    """Parse one required Nsight counter without inventing a zero."""

    if value is None:
        raise ValueError(f"Missing required Nsight {field}.")
    cleaned = value.replace(",", "").replace("%", "").strip()
    if not cleaned:
        raise ValueError(f"Empty required Nsight {field}.")
    try:
        result = float(cleaned)
    except ValueError as error:
        raise ValueError(
            f"Invalid Nsight {field}: {value!r}."
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"Invalid Nsight {field}: {value!r}.")
    return result


def required_name(value: str | None, *, row_kind: str, row_index: int) -> str:
    """Return a non-empty row identity for diagnostics and classification."""

    name = (value or "").strip()
    if not name:
        raise ValueError(
            f"Nsight {row_kind} row {row_index} has no required name."
        )
    return name


def pick(row: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    lower = {key.lower().strip(): key for key in row}
    for candidate in candidates:
        if candidate in lower:
            return row[lower[candidate]]
    for key_lower, key in lower.items():
        if any(candidate in key_lower for candidate in candidates):
            return row[key]
    return None


def classify_kernel(name: str) -> str:
    lowered = re.sub(r"\s+", "", name.lower())
    for stage, needles in STAGES.items():
        if any(needle in lowered for needle in needles):
            return stage
    return "other_kernel"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_kernel_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_stage: dict[str, dict[str, float]] = defaultdict(lambda: {"time_ns": 0.0, "calls": 0.0})
    top: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        name = required_name(
            pick(row, ("name", "kernel name", "demangled name")),
            row_kind="kernel",
            row_index=row_index,
        )
        time_ns = numeric(
            pick(row, ("total time (ns)", "time (ns)", "total time", "sum")),
            field=f"kernel row {row_index} total time",
        )
        calls = numeric(
            pick(row, ("instances", "calls", "num calls", "count")),
            field=f"kernel row {row_index} call count",
        )
        stage = classify_kernel(name)
        by_stage[stage]["time_ns"] += time_ns
        by_stage[stage]["calls"] += calls
        top.append({"stage": stage, "name": name, "time_ns": time_ns, "calls": calls})
    total = sum(value["time_ns"] for value in by_stage.values())
    stages = {
        stage: {
            **value,
            "time_ms": value["time_ns"] / 1e6,
            "percent": (100.0 * value["time_ns"] / total) if total else 0.0,
        }
        for stage, value in sorted(by_stage.items(), key=lambda item: item[1]["time_ns"], reverse=True)
    }
    return {
        "total_kernel_time_ms": total / 1e6,
        "stages": stages,
        "top_kernels": sorted(top, key=lambda item: item["time_ns"], reverse=True)[:30],
    }


def parse_api_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, dict[str, float]] = defaultdict(lambda: {"time_ns": 0.0, "calls": 0.0})
    for row_index, row in enumerate(rows, start=1):
        name = required_name(
            pick(row, ("name", "api name")),
            row_kind="CUDA API",
            row_index=row_index,
        )
        time_ns = numeric(
            pick(row, ("total time (ns)", "time (ns)", "total time", "sum")),
            field=f"CUDA API row {row_index} total time",
        )
        calls = numeric(
            pick(row, ("instances", "calls", "num calls", "count")),
            field=f"CUDA API row {row_index} call count",
        )
        lowered = name.lower()
        if "launch" in lowered:
            bucket = "kernel_launch"
        elif "synchroniz" in lowered:
            bucket = "synchronization"
        elif any(
            needle in lowered
            for needle in (
                "malloc",
                "free",
                "alloc",
                "memcreate",
                "memmap",
                "memunmap",
                "memrelease",
                "addressreserve",
                "addressfree",
                "mempool",
                "arraycreate",
                "arraydestroy",
                "mipmappedarray",
            )
        ):
            bucket = "allocator"
        else:
            bucket = "other_cuda_api"
        summary[bucket]["time_ns"] += time_ns
        summary[bucket]["calls"] += calls
    buckets = {
        key: {**value, "time_ms": value["time_ns"] / 1e6}
        for key, value in sorted(summary.items(), key=lambda item: item[1]["time_ns"], reverse=True)
    }
    total_time_ns = sum(value["time_ns"] for value in summary.values())
    total_calls = sum(value["calls"] for value in summary.values())
    return {
        "totals": {
            "time_ns": total_time_ns,
            "time_ms": total_time_ns / 1e6,
            "calls": total_calls,
        },
        **buckets,
    }


def parse_nvtx_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    ranges: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        name = required_name(
            pick(row, ("range", "name", "message")),
            row_kind="NVTX",
            row_index=row_index,
        )
        # Nsight prefixes default-domain PushPop ranges with one colon in
        # nvtx_sum CSVs (for example, ``:flashgs-matched-capture/...``).
        if name.startswith(":"):
            name = name[1:]
        calls = numeric(
            pick(row, ("instances", "calls", "count")),
            field=f"NVTX row {row_index} call count",
        )
        time_ns = numeric(
            pick(row, ("total time (ns)", "time (ns)", "total time", "sum")),
            field=f"NVTX row {row_index} total time",
        )
        ranges.append(
            {"name": name, "calls": calls, "time_ns": time_ns}
        )
    return {"ranges": ranges}


def main() -> None:
    args = parse_args()
    kernel_rows: list[dict[str, str]] = []
    api_rows: list[dict[str, str]] = []
    nvtx_rows: list[dict[str, str]] = []
    files = sorted(args.stats_dir.glob("*.csv"))
    for path in files:
        if "cuda_gpu_kern_sum" in path.name:
            kernel_rows.extend(read_csv(path))
        elif "cuda_api_sum" in path.name:
            api_rows.extend(read_csv(path))
        elif "nvtx_sum" in path.name:
            nvtx_rows.extend(read_csv(path))
    report = {
        "schema_version": 1,
        "stats_dir": str(args.stats_dir),
        "input_csv_files": [str(path) for path in files],
        "kernel_summary": parse_kernel_rows(kernel_rows) if kernel_rows else None,
        "cuda_api_summary": parse_api_rows(api_rows) if api_rows else None,
        "nvtx_summary": parse_nvtx_rows(nvtx_rows) if nvtx_rows else None,
        "allocator_classification": "cuda-runtime-driver-vmm-v2",
        "limitations": [],
    }
    if not kernel_rows:
        report["limitations"].append("No Nsight Systems cuda_gpu_kern_sum CSV was found.")
    if not api_rows:
        report["limitations"].append("No Nsight Systems cuda_api_sum CSV was found.")
    if not nvtx_rows:
        report["limitations"].append("No Nsight Systems nvtx_sum CSV was found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
