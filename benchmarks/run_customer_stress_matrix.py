"""Run fail-closed customer stress and cache-regime benchmark lanes."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from isaacsim_gaussian_renderer.benchmark_manifest import STRESS_SCENES


SCHEMA_VERSION = "customer-stress-matrix/v1"
COMPARISON_SCHEMA_VERSION = "custom-gsplat-fidelity-control/v1"
CUSTOM_SCHEMA_VERSION = "custom-cuda-benchmark/v1"
PUBLIC_SCHEMA_VERSION = "real-scene-custom-benchmark/v1"
OUTPUTS = {"rgb", "depth", "alpha", "semantic_id"}
SYNTHETIC_SCENES = {
    "synthetic-small": (10_000, 42),
    "synthetic-medium": (100_000, 43),
    **STRESS_SCENES,
}
PUBLIC_DATASET_ID = "voxel51-train-30000"
PUBLIC_DATASET_DEFAULT = Path(
    "/workspace/datasets/public-gaussian/Voxel51_gaussian_splatting/"
    "FO_dataset/train/point_cloud/iteration_30000/point_cloud.ply"
)


@dataclass(frozen=True)
class MatrixCase:
    """One immutable synthetic scene/camera workload."""

    scene: str
    batch: int
    width: int
    height: int

    @property
    def case_id(self) -> str:
        return f"{self.scene}-b{self.batch}-{self.width}x{self.height}"

    @property
    def gaussian_count(self) -> int:
        return int(SYNTHETIC_SCENES[self.scene][0])


@dataclass(frozen=True)
class CapacityPlan:
    """Explicit capacity reservation and its acceptance role."""

    regime: str
    visible_records: int
    tile_intersections: int
    acceptance_gate: bool
    source: str


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_positive_ints(value: str) -> list[int]:
    parsed = [int(item) for item in parse_csv(value)]
    if not parsed or min(parsed) <= 0:
        raise ValueError("Expected one or more positive integers.")
    return parsed


def parse_resolutions(value: str) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for item in parse_csv(value):
        width, height = item.lower().split("x", 1)
        pair = (int(width), int(height))
        if min(pair) <= 0:
            raise ValueError("Resolutions must be positive.")
        parsed.append(pair)
    if not parsed:
        raise ValueError("Expected one or more resolutions.")
    return parsed


def production_capacity(case: MatrixCase) -> CapacityPlan:
    """Mirror the ordinary screen-space planner in run_custom.py."""
    visible_records = math.ceil(case.batch * case.gaussian_count * 0.065)
    intersections_per_visible = math.ceil(4.0 * (max(case.width, case.height) / 128.0) ** 1.2)
    return CapacityPlan(
        regime="production-default",
        visible_records=visible_records,
        tile_intersections=(visible_records * intersections_per_visible),
        acceptance_gate=True,
        source=(
            "run_custom screen-space defaults: visible_fraction=0.065; "
            "ceil(4*(max_resolution/128)^1.2) intersections per visible"
        ),
    )


def conservative_capacity(case: MatrixCase) -> CapacityPlan:
    """Reserve every camera/Gaussian pair and resolution-scaled slack."""
    visible_records = case.batch * case.gaussian_count
    resolution_scale = case.width * case.height / float(128 * 128)
    intersections_per_visible = max(
        8,
        math.ceil(8.0 * resolution_scale),
    )
    return CapacityPlan(
        regime="conservative-non-oracle",
        visible_records=visible_records,
        tile_intersections=(visible_records * intersections_per_visible),
        acceptance_gate=False,
        source=("all camera/Gaussian pairs; eight intersections per pair at 128x128 scaled by pixel area"),
    )


def oracle_capacity(
    *,
    counters: dict[str, Any],
    headroom: float,
    source_case_id: str,
) -> CapacityPlan:
    """Return a diagnostic-only capacity derived from observed demand."""
    if not math.isfinite(headroom) or headroom <= 1.0:
        raise ValueError("Oracle headroom must be finite and greater than 1.")
    visible = int(counters["visible_gaussians"])
    intersections = int(counters["tile_intersections"])
    if visible <= 0 or intersections <= 0:
        raise ValueError("Oracle source counters must be positive.")
    return CapacityPlan(
        regime="hindsight-oracle-diagnostic",
        visible_records=math.ceil(visible * headroom),
        tile_intersections=math.ceil(intersections * headroom),
        acceptance_gate=False,
        source=(f"{headroom:.6g}x observed demand from {source_case_id}; never valid for production acceptance"),
    )


def child_environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    entries = [str(project_root), str(project_root / "src")]
    existing = environment.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def project_commit(project_root: Path) -> str:
    """Resolve the exact source revision propagated to every child result."""
    override = os.environ.get("SOURCE_GIT_COMMIT", "").strip()
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def classify_failure(log: str, returncode: int) -> str:
    lowered = log.lower()
    if "workspace overflow" in lowered or "capacity overflow" in lowered:
        return "CAPACITY_OVERFLOW"
    if "outofmemoryerror" in lowered or "out of memory" in lowered:
        return "OUT_OF_DEVICE_MEMORY"
    if "exceeds int32" in lowered or "at most int_max" in lowered:
        return "IMPLEMENTATION_LIMIT"
    if returncode < 0:
        return f"SIGNAL_{-returncode}"
    return "FAILED"


def capacity_failure_evidence(
    log: str,
    *,
    visible_capacity: int,
    intersection_capacity: int,
) -> dict[str, Any] | None:
    """Extract overflow evidence without claiming censored counters are exact."""
    named_counters = {
        name: int(match.group(1))
        for name in (
            "visible_gaussians",
            "tile_intersections",
            "visible_overflow",
            "intersection_overflow",
            "active_tiles",
        )
        if (
            match := re.search(
                rf"['\"]?{name}['\"]?\s*[:=]\s*(\d+)",
                log,
            )
        )
    }
    if "visible_overflow" in named_counters or "intersection_overflow" in named_counters:
        final_visible = re.search(r"visible_capacity\s*=\s*(\d+)", log)
        final_intersections = re.search(
            r"intersection_capacity\s*=\s*(\d+)",
            log,
        )
        return {
            "initial_requested_visible_capacity": visible_capacity,
            "initial_requested_intersection_capacity": intersection_capacity,
            "requested_visible_capacity": (
                int(final_visible.group(1))
                if final_visible is not None
                else visible_capacity
            ),
            "requested_intersection_capacity": (
                int(final_intersections.group(1))
                if final_intersections is not None
                else intersection_capacity
            ),
            "counters": named_counters,
            "demand_scope": "exact device counters printed by the child preflight",
            "censored": False,
        }

    custom = re.search(
        r"Custom renderer workspace overflow:\s*visible=(\d+)\s+intersections=(\d+)",
        log,
    )
    if custom is None:
        return None
    visible_overflow = int(custom.group(1))
    intersection_overflow = int(custom.group(2))
    return {
        "requested_visible_capacity": visible_capacity,
        "requested_intersection_capacity": intersection_capacity,
        "counters": {
            "visible_overflow": visible_overflow,
            "intersection_overflow": intersection_overflow,
            "visible_gaussians_lower_bound": visible_capacity + visible_overflow,
            "tile_intersections_lower_bound": (
                intersection_capacity + intersection_overflow
            ),
        },
        "demand_scope": (
            "lower bounds reconstructed from capacity plus dropped-record counters; "
            "upstream truncation can censor later intersection demand"
        ),
        "censored": True,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_capacity_adaptation(
    adaptation: Any,
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(adaptation, dict):
        return [f"{label} is missing adaptive-capacity telemetry"]
    if adaptation.get("schema_version") != "renderer-capacity/v1":
        errors.append(f"{label} has an unexpected capacity schema")
    if adaptation.get("adaptive") is not True:
        errors.append(f"{label} did not use bounded adaptive capacity")
    current = adaptation.get("current_capacity")
    if not isinstance(current, dict):
        errors.append(f"{label} is missing final capacity")
    else:
        for name in ("visible_records", "tile_intersections", "workspace_bytes"):
            if not isinstance(current.get(name), int) or current[name] <= 0:
                errors.append(f"{label} has invalid final {name}")
    last_render = adaptation.get("last_render")
    if not isinstance(last_render, dict):
        errors.append(f"{label} is missing last-render capacity telemetry")
    elif last_render.get("status") != "success":
        errors.append(f"{label} capacity handling did not finish successfully")
    else:
        counters = last_render.get("final_counters")
        if not isinstance(counters, dict):
            errors.append(f"{label} is missing final capacity counters")
        elif (
            counters.get("visible_overflow") != 0
            or counters.get("intersection_overflow") != 0
        ):
            errors.append(f"{label} adaptive capacity ended with overflow")
    return errors


def validate_custom_result(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("schema_version") != CUSTOM_SCHEMA_VERSION:
        errors.append("unexpected custom result schema")
    if result.get("pass") is not True:
        errors.append("custom result did not pass")
    if set(result.get("outputs", [])) != OUTPUTS:
        errors.append("custom result does not contain the four-output contract")
    if result.get("outputs_gpu_resident") is not True:
        errors.append("custom outputs are not all GPU resident")
    if result.get("outputs_contract_valid") is not True:
        errors.append("custom output assertions failed")
    if result.get("zero_overflow") is not True:
        errors.append("custom result reported capacity overflow")
    environment = result.get("environment")
    if not isinstance(environment, dict) or environment.get("project_commit") in {None, "", "unknown"}:
        errors.append("custom result is missing project source provenance")
    counters = result.get("measured_counters")
    if not isinstance(counters, dict):
        errors.append("custom result is missing measured counters")
    elif counters.get("visible_overflow") != 0 or counters.get("intersection_overflow") != 0:
        errors.append("custom measured counters are not overflow-free")
    timing = result.get("frame_ms")
    if not isinstance(timing, dict):
        errors.append("custom result is missing GPU timing")
    else:
        for key in ("mean", "p50", "p95", "ci95"):
            if not _is_number(timing.get(key)):
                errors.append(f"custom timing is missing finite {key}")
    if not _is_number(result.get("wall_frame_ms")):
        errors.append("custom result is missing finite wall timing")
    if not isinstance(result.get("driver_process_memory_bytes"), int):
        errors.append("custom result is missing process GPU memory")
    scene = result.get("scene")
    if not isinstance(scene, dict) or not scene.get("checksum_sha256"):
        errors.append("custom result is missing a scene checksum")
    camera = result.get("camera_manifest")
    if not isinstance(camera, dict) or not camera.get("checksum_sha256"):
        errors.append("custom result is missing a camera checksum")
    errors.extend(
        validate_capacity_adaptation(
            result.get("capacity_adaptation"),
            label="custom result",
        )
    )
    return errors


def validate_comparison_summary(
    summary: dict[str, Any],
    *,
    expected_regime: str,
) -> list[str]:
    errors: list[str] = []
    if summary.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        errors.append("unexpected custom-gsplat summary schema")
    if summary.get("pass") is not True:
        errors.append("custom-gsplat fidelity failed")
    environment = summary.get("environment")
    if not isinstance(environment, dict) or environment.get("project_commit") in {None, "", "unknown"}:
        errors.append("custom-gsplat summary is missing project source provenance")
    candidate = summary.get("candidate")
    if not isinstance(candidate, dict):
        errors.append("custom-gsplat summary is missing candidate metadata")
        return errors
    capacity = candidate.get("capacity")
    if not isinstance(capacity, dict):
        errors.append("custom-gsplat summary is missing capacity metadata")
    elif capacity.get("regime") != expected_regime:
        errors.append("custom-gsplat capacity regime is mislabeled")
    counters = candidate.get("counters")
    if not isinstance(counters, dict):
        errors.append("custom-gsplat summary is missing counters")
    elif counters.get("visible_overflow") != 0 or counters.get("intersection_overflow") != 0:
        errors.append("custom-gsplat counters are not overflow-free")
    errors.extend(
        validate_capacity_adaptation(
            candidate.get("capacity_adaptation"),
            label="custom-gsplat candidate",
        )
    )
    timing = summary.get("timing")
    if not isinstance(timing, dict):
        errors.append("custom-gsplat comparison is missing timing")
    else:
        for name in (
            "custom_all_output_gpu_ms",
            "gsplat_all_output_gpu_ms",
        ):
            stats = timing.get(name)
            if not isinstance(stats, dict):
                errors.append(f"custom-gsplat timing is missing {name}")
                continue
            for key in ("mean", "p50", "p95", "ci95"):
                if not _is_number(stats.get(key)):
                    errors.append(f"{name} is missing finite {key}")
    scene = summary.get("scene")
    if not isinstance(scene, dict) or not scene.get("checksum_sha256"):
        errors.append("custom-gsplat summary is missing a scene checksum")
    if not summary.get("camera_bundle_id"):
        errors.append("custom-gsplat summary is missing a camera bundle ID")
    return errors


def validate_public_result(
    result: dict[str, Any],
    *,
    expected_cache_enabled: bool,
    expected_invalidation: bool,
) -> list[str]:
    errors: list[str] = []
    if result.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        errors.append("unexpected public-scene result schema")
    if result.get("pass") is not True:
        errors.append("public-scene result did not pass")
    if result.get("zero_overflow") is not True:
        errors.append("public-scene result reported capacity overflow")
    if result.get("all_output_checks_valid") is not True:
        errors.append("public-scene output assertions failed")
    if result.get("outputs_gpu_resident") is not True:
        errors.append("public-scene outputs are not all GPU resident")
    environment = result.get("environment")
    if not isinstance(environment, dict) or environment.get("project_commit") in {None, "", "unknown"}:
        errors.append("public-scene result is missing project source provenance")
    if set(result.get("outputs", [])) != OUTPUTS:
        errors.append("public-scene result does not contain all four outputs")
    cache = result.get("projection_cache")
    if not isinstance(cache, dict):
        errors.append("public-scene result is missing cache metadata")
    elif cache.get("enabled") is not expected_cache_enabled:
        errors.append("public-scene cache regime is mislabeled")
    else:
        if cache.get("explicit_invalidation_before_each_render") is not expected_invalidation:
            errors.append("public-scene invalidation regime is mislabeled")
        render_calls = result.get("measured_iterations")
        measured_invalidations = cache.get("measured_invalidations")
        if expected_invalidation and measured_invalidations != render_calls:
            errors.append("public-scene invalidation count does not match measured renders")
        if expected_invalidation and cache.get("measured_misses") != render_calls:
            errors.append("public-scene invalidation lane did not miss on every measured render")
        if expected_invalidation and cache.get("measured_hits") != 0:
            errors.append("public-scene invalidation lane unexpectedly reported cache hits")
        if expected_cache_enabled and not expected_invalidation:
            if cache.get("measured_hits") != render_calls:
                errors.append("public-scene cache-hit lane did not hit on every measured render")
            if cache.get("measured_misses") != 0:
                errors.append("public-scene cache-hit lane unexpectedly reported cache misses")
        if not expected_cache_enabled:
            if cache.get("measured_hits") != 0 or cache.get("measured_misses") != 0:
                errors.append("public-scene cache-disabled lane reported cache activity")
    timing = result.get("frame_ms")
    if not isinstance(timing, dict):
        errors.append("public-scene result is missing GPU timing")
    else:
        for key in ("mean", "p50", "p95", "ci95"):
            if not _is_number(timing.get(key)):
                errors.append(f"public-scene timing is missing finite {key}")
    if not _is_number(result.get("wall_frame_ms")):
        errors.append("public-scene result is missing finite wall timing")
    if not isinstance(result.get("driver_process_memory_bytes"), int):
        errors.append("public-scene result is missing process GPU memory")
    errors.extend(
        validate_capacity_adaptation(
            result.get("capacity_adaptation"),
            label="public-scene result",
        )
    )
    return errors


def acceptance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    production = [record for record in records if record["acceptance_gate"]]
    required_evidence = [record for record in records if record["required_evidence"]]
    production_pass = bool(production) and all(record["pass"] for record in production)
    evidence_complete = bool(required_evidence) and all(record["pass"] for record in required_evidence)
    return {
        "production_default": {
            "pass": production_pass,
            "case_count": len(production),
            "failed_case_ids": [record["case_id"] for record in production if not record["pass"]],
        },
        "required_evidence": {
            "pass": evidence_complete,
            "case_count": len(required_evidence),
            "failed_case_ids": [record["case_id"] for record in required_evidence if not record["pass"]],
        },
        "oracle_can_satisfy_production_gate": False,
        "pass": production_pass and evidence_complete,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _remove_stale_files(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _run_command(
    *,
    command: list[str],
    output_dir: Path,
    project_root: Path,
    timeout_seconds: float,
) -> tuple[int, str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_path = output_dir / "command.json"
    log_path = output_dir / "run.log"
    _write_json(command_path, command)
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                env=child_environment(project_root),
                timeout=timeout_seconds,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
    log = log_path.read_text(encoding="utf-8", errors="replace")
    return returncode, log, "TIMEOUT" if timed_out else None


def _base_record(
    *,
    case_id: str,
    lane: str,
    capacity_regime: str,
    acceptance_gate: bool,
    required_evidence: bool,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "lane": lane,
        "capacity_regime": capacity_regime,
        "acceptance_gate": acceptance_gate,
        "required_evidence": required_evidence,
        "output_dir": str(output_dir),
    }


def _finish_record(
    record: dict[str, Any],
    *,
    returncode: int,
    log: str,
    timeout_status: str | None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    errors = list(errors or [])
    if timeout_status is not None:
        status = timeout_status
        errors.append("case exceeded its timeout")
    elif returncode != 0:
        status = classify_failure(log, returncode)
        errors.append(f"child exited with status {returncode}")
    elif errors:
        status = "INVALID_EVIDENCE"
    else:
        status = "PASS"
    return {
        **record,
        "status": status,
        "returncode": returncode,
        "pass": status == "PASS",
        "errors": errors,
        "log_tail": "\n".join(log.splitlines()[-40:]),
    }


def run_production_case(
    *,
    case: MatrixCase,
    output_root: Path,
    project_root: Path,
    warmup: int,
    iterations: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    output_dir = output_root / "production-default" / case.case_id
    result_path = output_dir / "result.json"
    artifact_dir = output_dir / "artifacts"
    capacity = production_capacity(case)
    _remove_stale_files(result_path)
    command = [
        sys.executable,
        "benchmarks/run_custom.py",
        "--scene",
        case.scene,
        "--batch",
        str(case.batch),
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--gaussian-support-sigma",
        "3.33",
        "--semantic-min-alpha",
        "0.01",
        "--linear-output",
        "--output",
        str(result_path),
        "--artifact-dir",
        str(artifact_dir),
    ]
    returncode, log, timeout_status = _run_command(
        command=command,
        output_dir=output_dir,
        project_root=project_root,
        timeout_seconds=timeout_seconds,
    )
    errors: list[str] = []
    result: dict[str, Any] | None = None
    if returncode == 0:
        if not result_path.is_file():
            errors.append("successful child did not write result.json")
        else:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                errors.extend(validate_custom_result(result))
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                errors.append(f"could not validate custom result: {exc}")
    record = _base_record(
        case_id=f"production-default/{case.case_id}",
        lane="custom-production-default",
        capacity_regime="production-default",
        acceptance_gate=True,
        required_evidence=True,
        output_dir=output_dir,
    )
    record["workload"] = {
        "scene": case.scene,
        "batch": case.batch,
        "width": case.width,
        "height": case.height,
    }
    record["capacity_plan"] = capacity.__dict__
    record["result_json"] = str(result_path)
    if returncode != 0:
        record["failure_capacity_evidence"] = capacity_failure_evidence(
            log,
            visible_capacity=capacity.visible_records,
            intersection_capacity=capacity.tile_intersections,
        )
    if result is not None:
        record["observed_counters"] = result.get("measured_counters")
        record["capacity_adaptation"] = result.get(
            "capacity_adaptation"
        )
        record["timing"] = {
            "gpu_frame_ms": result.get("frame_ms"),
            "wall_frame_ms": result.get("wall_frame_ms"),
        }
        record["memory"] = {
            "driver_process_memory_bytes": result.get("driver_process_memory_bytes"),
            "workspace_bytes": result.get("workspace_bytes"),
        }
        record["checksums"] = {
            "scene": result.get("scene", {}).get("checksum_sha256"),
            "camera": result.get("camera_manifest", {}).get("checksum_sha256"),
        }
        record["environment"] = result.get("environment")
    return _finish_record(
        record,
        returncode=returncode,
        log=log,
        timeout_status=timeout_status,
        errors=errors,
    )


def run_comparison_case(
    *,
    case: MatrixCase,
    capacity: CapacityPlan,
    output_root: Path,
    project_root: Path,
    warmup: int,
    iterations: int,
    timeout_seconds: float,
    required_evidence: bool,
) -> dict[str, Any]:
    output_dir = output_root / capacity.regime / case.case_id
    _remove_stale_files(
        output_dir / "summary.json",
        output_dir / "report" / "fidelity_report.json",
    )
    command = [
        sys.executable,
        "benchmarks/compare_custom_gsplat.py",
        "--scene",
        case.scene,
        "--batch",
        str(case.batch),
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--visible-capacity",
        str(capacity.visible_records),
        "--intersection-capacity",
        str(capacity.tile_intersections),
        "--capacity-regime",
        capacity.regime,
        "--semantic-min-alpha",
        "0.01",
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--require-lpips",
        "--gsplat-compatibility-patch",
        str(project_root / "patches" / "gsplat-cuda-event-flags.patch"),
        "--output",
        str(output_dir),
    ]
    returncode, log, timeout_status = _run_command(
        command=command,
        output_dir=output_dir,
        project_root=project_root,
        timeout_seconds=timeout_seconds,
    )
    errors: list[str] = []
    summary: dict[str, Any] | None = None
    fidelity: dict[str, Any] | None = None
    summary_path = output_dir / "summary.json"
    if returncode == 0:
        if not summary_path.is_file():
            errors.append("successful child did not write summary.json")
        else:
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                errors.extend(
                    validate_comparison_summary(
                        summary,
                        expected_regime=capacity.regime,
                    )
                )
                fidelity_path = output_dir / str(summary.get("fidelity_report", ""))
                if not fidelity_path.is_file():
                    errors.append("custom-gsplat fidelity report is missing")
                else:
                    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
                    per_view = fidelity.get("per_view")
                    if not isinstance(per_view, list) or not per_view:
                        errors.append("custom-gsplat fidelity report is missing per-view metrics")
                    worst_view = fidelity.get("worst_view")
                    if not isinstance(worst_view, dict):
                        errors.append("custom-gsplat fidelity report is missing the worst-view result")
                    elif not isinstance(per_view, list) or not isinstance(
                        worst_view.get("index"), int
                    ) or not (
                        0 <= worst_view["index"] < len(per_view)
                    ):
                        errors.append("custom-gsplat worst-view index is invalid")
            except (json.JSONDecodeError, KeyError, OSError, TypeError) as exc:
                errors.append(f"could not validate comparison summary: {exc}")
    record = _base_record(
        case_id=f"{capacity.regime}/{case.case_id}",
        lane="custom-vs-gsplat-all-output",
        capacity_regime=capacity.regime,
        acceptance_gate=capacity.acceptance_gate,
        required_evidence=required_evidence,
        output_dir=output_dir,
    )
    record["workload"] = {
        "scene": case.scene,
        "batch": case.batch,
        "width": case.width,
        "height": case.height,
    }
    record["capacity_plan"] = capacity.__dict__
    record["summary_json"] = str(summary_path)
    if summary is not None:
        record["observed_counters"] = summary["candidate"]["counters"]
        record["capacity_adaptation"] = summary["candidate"].get(
            "capacity_adaptation"
        )
        record["timing"] = summary.get("timing")
        record["checksums"] = {
            "scene": summary.get("scene", {}).get("checksum_sha256"),
            "camera_bundle": summary.get("camera_bundle_id"),
        }
        record["environment"] = summary.get("environment")
    if fidelity is not None:
        worst_view = fidelity.get("worst_view")
        per_view = fidelity.get("per_view")
        if (
            isinstance(worst_view, dict)
            and isinstance(worst_view.get("index"), int)
            and isinstance(per_view, list)
            and 0 <= worst_view["index"] < len(per_view)
        ):
            record["worst_view"] = {
                **per_view[worst_view["index"]],
                "worst_metric": worst_view.get("worst_metric"),
            }
    return _finish_record(
        record,
        returncode=returncode,
        log=log,
        timeout_status=timeout_status,
        errors=errors,
    )


def _public_command(
    *,
    dataset_path: Path,
    batch: int,
    warmup: int,
    iterations: int,
    output_path: Path,
    projection_cache: bool,
    invalidate_projection_cache: bool,
    intersections_per_view: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "benchmarks/run_home_scan.py",
        "--dataset-id",
        PUBLIC_DATASET_ID,
        "--path",
        str(dataset_path),
        "--batch",
        str(batch),
        "--width",
        "128",
        "--height",
        "128",
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--compact-projection-cache",
        "--tile-size",
        "1",
        "--tight-depth-range",
        "--semantic-scheme",
        "spatial-grid",
        "--semantic-grid",
        "2,1,1",
        "--authored-display-output",
        "--output",
        str(output_path),
    ]
    if projection_cache:
        command.append("--projection-cache")
    if invalidate_projection_cache:
        command.append("--invalidate-projection-cache-before-each-render")
    if intersections_per_view is not None:
        command.extend(
            [
                "--direct-intersections-per-view-at-128",
                str(intersections_per_view),
            ]
        )
    return command


def run_public_case(
    *,
    dataset_path: Path,
    batch: int,
    role: str,
    projection_cache: bool,
    invalidate_projection_cache: bool,
    intersections_per_view: int | None,
    acceptance_gate: bool,
    output_root: Path,
    project_root: Path,
    warmup: int,
    iterations: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    output_dir = output_root / role / f"b{batch}-128x128"
    result_path = output_dir / "result.json"
    visible_capacity = batch * 150_000
    intersection_capacity = batch * (
        170_000 if intersections_per_view is None else intersections_per_view
    )
    _remove_stale_files(result_path)
    command = _public_command(
        dataset_path=dataset_path,
        batch=batch,
        warmup=warmup,
        iterations=iterations,
        output_path=result_path,
        projection_cache=projection_cache,
        invalidate_projection_cache=invalidate_projection_cache,
        intersections_per_view=intersections_per_view,
    )
    returncode, log, timeout_status = _run_command(
        command=command,
        output_dir=output_dir,
        project_root=project_root,
        timeout_seconds=timeout_seconds,
    )
    errors: list[str] = []
    result: dict[str, Any] | None = None
    if returncode == 0:
        if not result_path.is_file():
            errors.append("successful child did not write result.json")
        else:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                errors.extend(
                    validate_public_result(
                        result,
                        expected_cache_enabled=projection_cache,
                        expected_invalidation=invalidate_projection_cache,
                    )
                )
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                errors.append(f"could not validate public result: {exc}")
    record = _base_record(
        case_id=f"{role}/b{batch}-128x128",
        lane="public-direct-compact",
        capacity_regime=("production-default" if intersections_per_view is None else "conservative-non-oracle"),
        acceptance_gate=acceptance_gate,
        required_evidence=True,
        output_dir=output_dir,
    )
    record["workload"] = {
        "scene": PUBLIC_DATASET_ID,
        "batch": batch,
        "width": 128,
        "height": 128,
    }
    if invalidate_projection_cache:
        record["cache_role"] = "explicit-invalidation-before-each-identical-camera-render"
    elif projection_cache:
        record["cache_role"] = "repeated-identical-camera-cache-hit"
    else:
        record["cache_role"] = "forced-full-recompute-not-moving-camera"
    record["intersections_per_view"] = intersections_per_view
    record["capacity_plan"] = {
        "visible_records": visible_capacity,
        "tile_intersections": intersection_capacity,
        "source": (
            "run_home_scan direct compact defaults"
            if intersections_per_view is None
            else "explicit conservative per-view reservation"
        ),
    }
    record["result_json"] = str(result_path)
    if returncode != 0:
        record["failure_capacity_evidence"] = capacity_failure_evidence(
            log,
            visible_capacity=visible_capacity,
            intersection_capacity=intersection_capacity,
        )
    if result is not None:
        record["wall_frame_ms"] = result["wall_frame_ms"]
        record["projection_cache"] = result["projection_cache"]
        record["observed_counters"] = result.get("measured_counters")
        record["capacity_adaptation"] = result.get(
            "capacity_adaptation"
        )
        record["timing"] = {
            "gpu_frame_ms": result.get("frame_ms"),
            "wall_frame_ms": result.get("wall_frame_ms"),
        }
        record["memory"] = {
            "driver_process_memory_bytes": result.get("driver_process_memory_bytes"),
            "workspace_bytes": result.get("workspace_bytes"),
        }
        record["checksums"] = {
            "scene": result.get("dataset", {}).get("sha256"),
            "semantic_sidecar": result.get("dataset", {})
            .get("semantic_sidecar", {})
            .get("checksum_sha256"),
            "camera": result.get("camera_manifest", {}).get("checksum_sha256"),
        }
        record["environment"] = result.get("environment")
    return _finish_record(
        record,
        returncode=returncode,
        log=log,
        timeout_status=timeout_status,
        errors=errors,
    )


def cache_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    batches = sorted(
        {int(record["workload"]["batch"]) for record in records if record.get("lane") == "public-direct-compact"}
    )
    for batch in batches:
        hit = next(
            (
                record
                for record in records
                if record["case_id"].startswith("cache-hit/") and int(record["workload"]["batch"]) == batch
            ),
            None,
        )
        miss = next(
            (
                record
                for record in records
                if record["case_id"].startswith("cache-miss/") and int(record["workload"]["batch"]) == batch
            ),
            None,
        )
        invalidated = next(
            (
                record
                for record in records
                if record["case_id"].startswith("cache-invalidated/")
                and int(record["workload"]["batch"]) == batch
            ),
            None,
        )
        if (
            not hit
            or not miss
            or not invalidated
            or not hit["pass"]
            or not miss["pass"]
            or not invalidated["pass"]
        ):
            comparisons.append(
                {
                    "batch": batch,
                    "pass": False,
                    "reason": "cache hit/disabled/invalidated evidence is incomplete",
                }
            )
            continue
        hit_ms = float(hit["wall_frame_ms"])
        miss_ms = float(miss["wall_frame_ms"])
        invalidated_ms = float(invalidated["wall_frame_ms"])
        comparisons.append(
            {
                "batch": batch,
                "pass": hit_ms > 0 and miss_ms > 0 and invalidated_ms > 0,
                "cache_hit_wall_ms": hit_ms,
                "cache_disabled_wall_ms": miss_ms,
                "cache_invalidated_wall_ms": invalidated_ms,
                "cache_benefit_vs_disabled": miss_ms / hit_ms,
                "cache_benefit_vs_invalidated": invalidated_ms / hit_ms,
                "invalidation_overhead_vs_disabled": invalidated_ms / miss_ms,
                "interpretation": (
                    "Projection-reuse benefit for repeated identical camera geometry; not a moving-camera result."
                ),
            }
        )
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/customer-stress-matrix"),
    )
    parser.add_argument(
        "--scenes",
        default=("synthetic-small,synthetic-medium,stress-anisotropic,stress-overdraw"),
    )
    parser.add_argument("--batches", default="1,8,64")
    parser.add_argument("--resolutions", default="128x128,256x256")
    parser.add_argument("--public-cache-batches", default="1,8")
    parser.add_argument("--public-path", type=Path)
    parser.add_argument(
        "--public-conservative-intersections-per-view",
        type=int,
        default=310_000,
    )
    parser.add_argument("--skip-public-cache", action="store_true")
    parser.add_argument("--skip-oracle-diagnostics", action="store_true")
    parser.add_argument("--oracle-headroom", type=float, default=1.25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--case-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Always exit zero while preserving a FAIL verdict in the manifest.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=("Run a bounded synthetic smoke only; this is not the full customer acceptance matrix."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("Warmup and iterations must be positive.")
    if args.case_timeout_seconds <= 0:
        raise ValueError("Case timeout must be positive.")
    if args.public_conservative_intersections_per_view <= 0:
        raise ValueError("Public conservative capacity must be positive.")
    if not math.isfinite(args.oracle_headroom) or args.oracle_headroom <= 1:
        raise ValueError("Oracle headroom must be finite and greater than 1.")

    project_root = Path(__file__).resolve().parents[1]
    source_commit = project_commit(project_root)
    if source_commit != "unknown":
        os.environ.setdefault("SOURCE_GIT_COMMIT", source_commit)
    scenes = parse_csv(args.scenes)
    batches = parse_positive_ints(args.batches)
    resolutions = parse_resolutions(args.resolutions)
    public_batches = parse_positive_ints(args.public_cache_batches)
    skip_public_cache = args.skip_public_cache
    if args.quick:
        scenes = ["synthetic-small", "stress-anisotropic"]
        batches = [1]
        resolutions = [(128, 128)]
        skip_public_cache = True
    unsupported = set(scenes) - set(SYNTHETIC_SCENES)
    if unsupported:
        raise ValueError("Unsupported scene(s): " + ", ".join(sorted(unsupported)))

    args.output.mkdir(parents=True, exist_ok=True)
    _remove_stale_files(args.output / "run-manifest.json")
    records: list[dict[str, Any]] = []
    if source_commit == "unknown":
        records.append(
            {
                **_base_record(
                    case_id="source-provenance/commit",
                    lane="source-provenance-preflight",
                    capacity_regime="not-applicable",
                    acceptance_gate=True,
                    required_evidence=True,
                    output_dir=args.output,
                ),
                "status": "MISSING_PROVENANCE",
                "returncode": 2,
                "pass": False,
                "errors": [
                    "Project source commit is unknown; run from a Git checkout or set SOURCE_GIT_COMMIT."
                ],
                "log_tail": "",
            }
        )
    cases = [
        MatrixCase(scene, batch, width, height)
        for scene in scenes
        for width, height in resolutions
        for batch in batches
    ]
    for case in cases:
        production = run_production_case(
            case=case,
            output_root=args.output / "synthetic",
            project_root=project_root,
            warmup=args.warmup,
            iterations=args.iterations,
            timeout_seconds=args.case_timeout_seconds,
        )
        records.append(production)
        conservative = run_comparison_case(
            case=case,
            capacity=conservative_capacity(case),
            output_root=args.output / "synthetic",
            project_root=project_root,
            warmup=args.warmup,
            iterations=args.iterations,
            timeout_seconds=args.case_timeout_seconds,
            required_evidence=True,
        )
        records.append(conservative)
        if (
            not production["pass"]
            and conservative["pass"]
            and isinstance(conservative.get("observed_counters"), dict)
        ):
            production["uncensored_demand_control"] = {
                "source_case_id": conservative["case_id"],
                "capacity_regime": conservative["capacity_regime"],
                "counters": conservative["observed_counters"],
                "interpretation": (
                    "Same scene and camera with a non-oracle reservation; diagnostic demand only, "
                    "not a production pass."
                ),
            }
        if (
            not args.skip_oracle_diagnostics
            and conservative["pass"]
            and isinstance(conservative.get("observed_counters"), dict)
        ):
            capacity = oracle_capacity(
                counters=conservative["observed_counters"],
                headroom=args.oracle_headroom,
                source_case_id=conservative["case_id"],
            )
            records.append(
                run_comparison_case(
                    case=case,
                    capacity=capacity,
                    output_root=args.output / "synthetic",
                    project_root=project_root,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    timeout_seconds=args.case_timeout_seconds,
                    required_evidence=True,
                )
            )

    limitations: list[str] = []
    if skip_public_cache:
        limitations.append(
            "Public-scene cache lanes were explicitly skipped; this is not the full customer acceptance matrix."
        )
    else:
        dataset_path = args.public_path or PUBLIC_DATASET_DEFAULT
        if not dataset_path.is_file():
            records.append(
                {
                    **_base_record(
                        case_id="public-dataset/presence",
                        lane="public-dataset-preflight",
                        capacity_regime="not-applicable",
                        acceptance_gate=True,
                        required_evidence=True,
                        output_dir=args.output / "public",
                    ),
                    "status": "MISSING_DATASET",
                    "returncode": 2,
                    "pass": False,
                    "errors": [f"Public dataset not found: {dataset_path}"],
                    "log_tail": "",
                }
            )
        else:
            for batch in public_batches:
                production = run_public_case(
                    dataset_path=dataset_path,
                    batch=batch,
                    role="production-default",
                    projection_cache=False,
                    invalidate_projection_cache=False,
                    intersections_per_view=None,
                    acceptance_gate=True,
                    output_root=args.output / "public",
                    project_root=project_root,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    timeout_seconds=args.case_timeout_seconds,
                )
                records.append(production)
                conservative_recompute: dict[str, Any] | None = None
                for role, projection_cache, invalidate_projection_cache in (
                    ("cache-miss", False, False),
                    ("cache-invalidated", True, True),
                    ("cache-hit", True, False),
                ):
                    cache_record = run_public_case(
                        dataset_path=dataset_path,
                        batch=batch,
                        role=role,
                        projection_cache=projection_cache,
                        invalidate_projection_cache=(
                            invalidate_projection_cache
                        ),
                        intersections_per_view=(args.public_conservative_intersections_per_view),
                        acceptance_gate=False,
                        output_root=args.output / "public",
                        project_root=project_root,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        timeout_seconds=args.case_timeout_seconds,
                    )
                    records.append(cache_record)
                    if role == "cache-miss":
                        conservative_recompute = cache_record
                if (
                    not production["pass"]
                    and conservative_recompute is not None
                    and conservative_recompute["pass"]
                    and isinstance(conservative_recompute.get("observed_counters"), dict)
                ):
                    production["uncensored_demand_control"] = {
                        "source_case_id": conservative_recompute["case_id"],
                        "capacity_regime": conservative_recompute["capacity_regime"],
                        "counters": conservative_recompute["observed_counters"],
                        "interpretation": (
                            "Same scene and camera with a non-oracle reservation; diagnostic demand only, "
                            "not a production pass."
                        ),
                    }

    acceptance = acceptance_summary(records)
    cache = cache_comparisons(records) if not skip_public_cache else []
    if cache and not all(item["pass"] for item in cache):
        acceptance["required_evidence"]["pass"] = False
        acceptance["pass"] = False
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "configuration": {
            "scenes": scenes,
            "batches": batches,
            "resolutions": [list(pair) for pair in resolutions],
            "public_cache_batches": public_batches,
            "public_cache_skipped": skip_public_cache,
            "public_dataset_path": (
                None
                if skip_public_cache
                else str(args.public_path or PUBLIC_DATASET_DEFAULT)
            ),
            "public_conservative_intersections_per_view": (
                args.public_conservative_intersections_per_view
            ),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "case_timeout_seconds": args.case_timeout_seconds,
            "semantic_min_alpha": 0.01,
            "oracle_headroom": args.oracle_headroom,
            "oracle_diagnostics_skipped": args.skip_oracle_diagnostics,
        },
        "acceptance": acceptance,
        "cache_comparisons": cache,
        "limitations": limitations,
        "record_count": len(records),
        "records": records,
        "pass": acceptance["pass"],
    }
    manifest_path = args.output / "run-manifest.json"
    _write_json(manifest_path, manifest)
    marker = "CUSTOMER_STRESS_MATRIX_PASS" if manifest["pass"] else "CUSTOMER_STRESS_MATRIX_FAIL"
    print(f"{marker} {manifest_path}")
    if not manifest["pass"] and not args.evidence_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
