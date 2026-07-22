"""Machine-readable benchmark result schema and writers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "baseline-result/v1"


def fabric_scene_delegate_record() -> dict[str, Any]:
    """Fabric Scene Delegate state for standalone OVRTX API rows.

    BENCHMARK_PROTOCOL.md requires Fabric Scene Delegate to be enabled and
    verified for Kit/Isaac RTX runs, and: "The standalone OVRTX API does
    not instantiate a Kit application or Fabric Scene Delegate; those rows
    must record that the setting is not applicable and cannot be cited as
    FSD-specific evidence."
    """
    return {
        "state": "not_applicable",
        "reason": (
            "standalone OVRTX API instantiates no Kit application or "
            "Fabric Scene Delegate"
        ),
        "citable_as_fsd_evidence": False,
    }


CSV_FIELDS = [
    "schema_version",
    "run_id",
    "config_id",
    "timestamp_utc",
    "host",
    "gpu_name",
    "driver_version",
    "git_commit",
    "implementation",
    "implementation_commit",
    "scene_id",
    "scene_checksum",
    "scene_gaussians",
    "scene_mode",
    "batch_size",
    "width",
    "height",
    "outputs",
    "semantic_contract",
    "warmup_iterations",
    "measured_iterations",
    "fabric_scene_delegate",
    "mean_gpu_ms",
    "std_gpu_ms",
    "p50_gpu_ms",
    "p95_gpu_ms",
    "ci95_gpu_ms",
    "mean_wall_ms",
    "images_per_second",
    "megapixels_per_second",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "persistent_scene_bytes",
    "temporary_allocation_delta_bytes",
    "driver_process_memory_bytes",
    "visible_gaussians",
    "tile_intersections",
    "status",
    "pass",
    "notes",
]


JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://example.invalid/vectorized-gaussian/baseline-result-v1.schema.json",
    "title": "Vectorized Gaussian baseline result",
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "config_id",
        "timestamp_utc",
        "environment",
        "implementation",
        "configuration",
        "dataset",
        "timing",
        "memory",
        "work",
        "fidelity",
        "status",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "run_id": {"type": "string"},
        "config_id": {"type": "string"},
        "timestamp_utc": {"type": "string"},
        "environment": {"type": "object"},
        "implementation": {"type": "object"},
        "configuration": {"type": "object"},
        "dataset": {"type": "object"},
        "timing": {"type": "object"},
        "memory": {"type": "object"},
        "work": {"type": "object"},
        "fidelity": {"type": "object"},
        "status": {"type": "object"},
    },
    "additionalProperties": True,
}


def flatten_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the protocol CSV row for a nested JSON result."""
    env = result.get("environment", {})
    impl = result.get("implementation", {})
    cfg = result.get("configuration", {})
    dataset = result.get("dataset", {})
    timing = result.get("timing", {})
    memory = result.get("memory", {})
    work = result.get("work", {})
    status = result.get("status", {})
    outputs = cfg.get("outputs", [])
    fabric_scene_delegate = cfg.get("fabric_scene_delegate")

    row = {
        "schema_version": result.get("schema_version"),
        "run_id": result.get("run_id"),
        "config_id": result.get("config_id"),
        "timestamp_utc": result.get("timestamp_utc"),
        "host": env.get("host"),
        "gpu_name": env.get("gpu_name"),
        "driver_version": env.get("driver_version"),
        "git_commit": env.get("git_commit"),
        "implementation": impl.get("name"),
        "implementation_commit": impl.get("commit"),
        "scene_id": dataset.get("scene_id"),
        "scene_checksum": dataset.get("checksum_sha256"),
        "scene_gaussians": dataset.get("gaussian_count"),
        "scene_mode": cfg.get("scene_mode"),
        "batch_size": cfg.get("batch_size"),
        "width": cfg.get("width"),
        "height": cfg.get("height"),
        "outputs": "+".join(outputs) if isinstance(outputs, list) else outputs,
        "semantic_contract": cfg.get("semantic_contract"),
        "warmup_iterations": cfg.get("warmup_iterations"),
        "measured_iterations": cfg.get("measured_iterations"),
        "fabric_scene_delegate": (
            fabric_scene_delegate.get("state")
            if isinstance(fabric_scene_delegate, Mapping)
            else fabric_scene_delegate
        ),
        "mean_gpu_ms": timing.get("gpu_ms", {}).get("mean"),
        "std_gpu_ms": timing.get("gpu_ms", {}).get("std"),
        "p50_gpu_ms": timing.get("gpu_ms", {}).get("p50"),
        "p95_gpu_ms": timing.get("gpu_ms", {}).get("p95"),
        "ci95_gpu_ms": timing.get("gpu_ms", {}).get("ci95"),
        "mean_wall_ms": timing.get("wall_ms", {}).get("mean"),
        "images_per_second": timing.get("images_per_second"),
        "megapixels_per_second": timing.get("megapixels_per_second"),
        "peak_allocated_bytes": memory.get("peak_allocated_bytes"),
        "peak_reserved_bytes": memory.get("peak_reserved_bytes"),
        "persistent_scene_bytes": memory.get("persistent_scene_bytes"),
        "temporary_allocation_delta_bytes": memory.get("temporary_allocation_delta_bytes"),
        "driver_process_memory_bytes": memory.get("driver_process_memory_bytes"),
        "visible_gaussians": work.get("visible_gaussians"),
        "tile_intersections": work.get("tile_intersections"),
        "status": status.get("verdict"),
        "pass": status.get("pass"),
        "notes": status.get("notes"),
    }
    return {field: row.get(field) for field in CSV_FIELDS}


def write_json_schema(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(JSON_SCHEMA, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_result_json(result: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_results_csv(results: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(flatten_result(result))
