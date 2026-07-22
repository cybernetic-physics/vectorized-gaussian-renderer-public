"""Independent acceptance-evidence verifier.

The verifier intentionally consumes evidence artifacts instead of renderer
internals. It recomputes every acceptance verdict it can from machine-readable
results and fails closed when required proof is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REMOTE_HOST = "vast-gsplat-isaac"
REMOTE_ROOT = "/workspace/agent-worktrees/verification"
INSTANCE_ID = "45069639"
HOME_SCAN_MANIFEST_SHA256 = "ac0c8226d8f0e5359deb09026d12954bb7096143ebff8061c50303909e8f285d"

REQUIRED_BATCHES = {1, 8, 32, 64, 128, 256, "max_fitting"}
REQUIRED_RESOLUTIONS = {(128, 128), (256, 256), (512, 512)}
REQUIRED_SCENES = {"synthetic-small", "synthetic-medium", "public-real-world", "home-scan-lod0"}
REQUIRED_SCENE_MODES = {"shared-scene", "scene-ids"}
REQUIRED_OUTPUTS = {"rgb", "depth", "alpha", "semantic_id"}
REQUIRED_IMPLEMENTATIONS = {"ovrtx", "gsplat", "custom"}

THRESHOLDS = {
    "throughput_ratio": 5.0,
    "memory_ratio": 0.80,
    "rgb_psnr_db": 40.0,
    "rgb_ssim": 0.995,
    "lpips": 0.01,
    "alpha_mae": 0.005,
    "depth_relative_error": 0.01,
    "semantic_agreement": 0.999,
    "reproducibility_relative_range": 0.05,
    "stability_seconds": 600.0,
}

CRITERIA = {
    "visual_fidelity",
    "throughput",
    "peak_memory",
    "rgb_psnr",
    "rgb_ssim",
    "lpips",
    "alpha_mae",
    "depth_relative_error",
    "semantic_agreement",
    "reproducibility",
    "single_camera_reported",
    "gpu_resident_outputs",
    "setup_validation",
    "dataset_manifests",
    "benchmark_schema",
    "statistical_computations",
    "deterministic_replay",
    "shared_scene_memory",
    "active_subset_cadence",
    "cuda_error_checks",
    "ten_minute_stability",
}


class VerificationError(ValueError):
    """Raised when evidence is missing, malformed, or insufficient."""


def load_acceptance(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise VerificationError(f"missing acceptance evidence: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise VerificationError("acceptance evidence must be a JSON object")
    return data


def require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise VerificationError(f"{key} must be an object")
    return value


def require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise VerificationError(f"{key} must be a non-empty list")
    return value


def require_bool(data: dict[str, Any], key: str, expected: bool = True) -> None:
    if data.get(key) is not expected:
        raise VerificationError(f"{key} must be {expected}")


def require_positive_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise VerificationError(f"{key} must be a positive number")
    return float(value)


def require_nonnegative_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise VerificationError(f"{key} must be a non-negative number")
    return float(value)


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise VerificationError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def close_enough(actual: float, expected: float, *, abs_tol: float = 1e-9, rel_tol: float = 1e-6) -> bool:
    return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def recompute_stats(samples: list[float]) -> dict[str, float]:
    if len(samples) < 50:
        raise VerificationError("timing samples must contain at least 50 measured iterations")
    mean = statistics.fmean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "mean_ms": mean,
        "std_ms": std,
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "ci95_ms": 1.96 * std / math.sqrt(len(samples)),
    }


def verify_setup(setup: dict[str, Any], evidence_dir: Path) -> None:
    require_bool(setup, "fresh_setup_rerun")
    require_bool(setup, "isaac_headless_smoke_ok")
    if setup.get("remote_host") != REMOTE_HOST:
        raise VerificationError(f"setup remote_host must be {REMOTE_HOST}")
    if setup.get("remote_root") != REMOTE_ROOT:
        raise VerificationError(f"setup remote_root must be {REMOTE_ROOT}")
    if str(setup.get("vast_instance_id")) != INSTANCE_ID:
        raise VerificationError(f"setup vast_instance_id must be {INSTANCE_ID}")
    logs = setup.get("logs")
    if not isinstance(logs, list) or not logs:
        raise VerificationError("setup logs must be a non-empty list")
    for log in logs:
        if not isinstance(log, str) or not log:
            raise VerificationError("setup log paths must be non-empty strings")
        if not (evidence_dir / log).is_file():
            raise VerificationError(f"setup log does not exist: {log}")


def verify_dataset_manifests(manifests: list[Any]) -> None:
    by_id: dict[str, dict[str, Any]] = {}
    for item in manifests:
        if not isinstance(item, dict):
            raise VerificationError("dataset manifest entries must be objects")
        dataset_id = item.get("dataset_id")
        if dataset_id in by_id:
            raise VerificationError(f"duplicate dataset manifest: {dataset_id}")
        if not isinstance(dataset_id, str):
            raise VerificationError("dataset_id must be a string")
        digest = require_sha256(item.get("manifest_sha256"), f"{dataset_id}.manifest_sha256")
        require_positive_number(item, "file_count")
        require_positive_number(item, "record_count")
        by_id[dataset_id] = item
        if dataset_id == "home-scan-lod0" and digest != HOME_SCAN_MANIFEST_SHA256:
            raise VerificationError("Home Scan manifest SHA-256 does not match BENCHMARK_PROTOCOL.md")

    missing = REQUIRED_SCENES - by_id.keys()
    if missing:
        raise VerificationError("missing dataset manifests: " + ", ".join(sorted(missing)))


def normalize_batch(value: Any) -> int | str:
    if value == "max_fitting":
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise VerificationError(f"invalid batch_size: {value!r}")


def normalize_resolution(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise VerificationError(f"resolution must be [width, height], got {value!r}")
    width, height = value
    if not isinstance(width, int) or not isinstance(height, int):
        raise VerificationError(f"resolution must contain integers, got {value!r}")
    return width, height


def verify_benchmark_runs(runs: list[Any]) -> None:
    coverage: set[tuple[str, int | str, tuple[int, int], str, str]] = set()
    config_ids: set[str] = set()

    for run in runs:
        if not isinstance(run, dict):
            raise VerificationError("benchmark runs must be objects")
        config_id = run.get("config_id")
        if not isinstance(config_id, str) or not config_id:
            raise VerificationError("benchmark run config_id must be a non-empty string")
        config_ids.add(config_id)

        batch = normalize_batch(run.get("batch_size"))
        resolution = normalize_resolution(run.get("resolution"))
        scene = run.get("scene")
        scene_mode = run.get("scene_mode")
        outputs = set(run.get("outputs", []))
        if batch not in REQUIRED_BATCHES:
            raise VerificationError(f"{config_id}: unsupported batch_size {batch!r}")
        if resolution not in REQUIRED_RESOLUTIONS:
            raise VerificationError(f"{config_id}: unsupported resolution {resolution!r}")
        if scene not in REQUIRED_SCENES:
            raise VerificationError(f"{config_id}: unsupported scene {scene!r}")
        if scene_mode not in REQUIRED_SCENE_MODES:
            raise VerificationError(f"{config_id}: unsupported scene_mode {scene_mode!r}")
        if outputs != REQUIRED_OUTPUTS:
            raise VerificationError(f"{config_id}: outputs must be {sorted(REQUIRED_OUTPUTS)}")

        implementation = run.get("implementation")
        if implementation not in REQUIRED_IMPLEMENTATIONS:
            raise VerificationError(f"{config_id}: invalid implementation {implementation!r}")
        require_sha256(run.get("dataset_manifest_sha256"), f"{config_id}.dataset_manifest_sha256")
        if not isinstance(run.get("source_commit"), str) or not run["source_commit"]:
            raise VerificationError(f"{config_id}: source_commit is required")

        samples = run.get("timing_samples_ms")
        if not isinstance(samples, list) or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            for value in samples
        ):
            raise VerificationError(f"{config_id}: timing_samples_ms must be positive numbers")
        expected = recompute_stats([float(v) for v in samples])
        reported = require_mapping(run, "timing_stats")
        for key, expected_value in expected.items():
            actual = (
                require_nonnegative_number(reported, key)
                if key in {"std_ms", "ci95_ms"}
                else require_positive_number(reported, key)
            )
            if not close_enough(actual, expected_value):
                raise VerificationError(
                    f"{config_id}: {key} mismatch, reported {actual}, recomputed {expected_value}"
                )

        memory = require_mapping(run, "memory")
        for key in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "persistent_scene_bytes",
            "workspace_bytes",
            "driver_process_bytes",
        ):
            require_positive_number(memory, key)
        require_nonnegative_number(memory, "temporary_allocation_delta_bytes")
        work_counts = require_mapping(run, "work_counts")
        if implementation in {"custom", "gsplat"}:
            require_positive_number(work_counts, "visible_gaussians")
            require_positive_number(work_counts, "tile_intersections")
        else:
            reason = work_counts.get("unavailable_reason")
            if not isinstance(reason, str) or not reason:
                require_positive_number(work_counts, "visible_gaussians")
                require_positive_number(work_counts, "tile_intersections")

        coverage.add((str(implementation), batch, resolution, str(scene), str(scene_mode)))

    required_coverage = {
        (implementation, batch, resolution, scene, scene_mode)
        for implementation in REQUIRED_IMPLEMENTATIONS
        for batch in REQUIRED_BATCHES
        for resolution in REQUIRED_RESOLUTIONS
        for scene in REQUIRED_SCENES
        for scene_mode in REQUIRED_SCENE_MODES
    }
    missing = required_coverage - coverage
    if missing:
        sample = sorted(
            missing,
            key=lambda item: (
                str(item[0]),
                str(item[1]),
                item[2],
                item[3],
                item[4],
            ),
        )[:5]
        raise VerificationError(f"benchmark matrix is incomplete; first missing entries: {sample}")
    if len(config_ids) != len(runs):
        raise VerificationError("benchmark run config_id values must be unique")


def verify_fidelity(fidelity: list[Any], evidence_dir: Path) -> None:
    seen: set[tuple[str, str]] = set()
    for item in fidelity:
        if not isinstance(item, dict):
            raise VerificationError("fidelity entries must be objects")
        scene = item.get("scene")
        view_class = item.get("view_class")
        if scene not in REQUIRED_SCENES:
            raise VerificationError(f"invalid fidelity scene {scene!r}")
        if view_class not in {"representative", "worst_case"}:
            raise VerificationError(f"invalid fidelity view_class {view_class!r}")
        metrics = require_mapping(item, "metrics")
        checks = {
            "rgb_psnr_db": require_positive_number(metrics, "rgb_psnr_db") >= THRESHOLDS["rgb_psnr_db"],
            "rgb_ssim": require_positive_number(metrics, "rgb_ssim") >= THRESHOLDS["rgb_ssim"],
            "lpips": require_nonnegative_number(metrics, "lpips") <= THRESHOLDS["lpips"],
            "alpha_mae": require_nonnegative_number(metrics, "alpha_mae") <= THRESHOLDS["alpha_mae"],
            "depth_relative_error": require_nonnegative_number(metrics, "depth_relative_error")
            <= THRESHOLDS["depth_relative_error"],
            "semantic_agreement": require_positive_number(metrics, "semantic_agreement")
            >= THRESHOLDS["semantic_agreement"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise VerificationError(f"{scene}/{view_class}: fidelity threshold failures: {', '.join(failed)}")
        artifacts = item.get("artifacts")
        if not isinstance(artifacts, dict):
            raise VerificationError(f"{scene}/{view_class}: artifacts must be recorded")
        for key in (
            "reference_rgb",
            "candidate_rgb",
            "rgb_abs_diff",
            "depth_relative_error_image",
            "alpha_diff",
            "semantic_mismatch_mask",
        ):
            if not isinstance(artifacts.get(key), str) or not artifacts[key]:
                raise VerificationError(f"{scene}/{view_class}: missing artifact {key}")
            if not (evidence_dir / artifacts[key]).is_file():
                raise VerificationError(
                    f"{scene}/{view_class}: artifact does not exist: {artifacts[key]}"
                )
        seen.add((str(scene), str(view_class)))

    required = {(scene, view_class) for scene in REQUIRED_SCENES for view_class in {"representative", "worst_case"}}
    missing = required - seen
    if missing:
        raise VerificationError("missing fidelity views: " + ", ".join(f"{s}/{v}" for s, v in sorted(missing)))


def verify_replay(replay: dict[str, Any]) -> None:
    require_bool(replay, "passed")
    hashes = replay.get("output_hashes_by_run")
    if not isinstance(hashes, list) or len(hashes) < 2:
        raise VerificationError("deterministic replay requires at least two output hash records")
    first = hashes[0]
    if not isinstance(first, dict):
        raise VerificationError("deterministic replay hash records must be objects")
    for record in hashes[1:]:
        if record != first:
            raise VerificationError("deterministic replay output hashes differ")
    require_bool(replay, "selected_environment_rendering_checked")


def verify_shared_scene_memory(memory: dict[str, Any]) -> None:
    require_bool(memory, "passed")
    if memory.get("unique_static_scene_copies") != 1:
        raise VerificationError("shared-scene rendering must keep one static scene copy")
    require_bool(memory, "per_environment_transforms_without_static_duplication")


def verify_active_subset_cadence(checks: dict[str, Any]) -> None:
    require_bool(checks, "passed")
    require_bool(checks, "active_subset_passed")
    require_bool(checks, "render_every_n_steps_passed")


def verify_cuda_checks(checks: dict[str, Any]) -> None:
    require_bool(checks, "passed")
    require_bool(checks, "synchronize_after_measured_range")
    errors = checks.get("errors")
    if errors != []:
        raise VerificationError("CUDA error list must be empty")


def verify_stability(stability: dict[str, Any]) -> None:
    require_bool(stability, "passed")
    duration = require_positive_number(stability, "duration_seconds")
    if duration < THRESHOLDS["stability_seconds"]:
        raise VerificationError("stability run must last at least 600 seconds")
    require_bool(stability, "no_cuda_errors")
    require_bool(stability, "no_nan_or_inf")


def verify_headline(headline: dict[str, Any]) -> None:
    require_bool(headline, "custom_kernel_loaded")
    require_bool(headline, "gpu_resident_outputs")
    require_bool(headline, "no_cpu_output_copies_in_steady_state")
    require_positive_number(headline, "single_camera_images_per_second")

    custom_ips = require_positive_number(headline, "custom_images_per_second")
    rtx_ips = require_positive_number(headline, "rtx_images_per_second")
    throughput_ratio = custom_ips / rtx_ips
    if throughput_ratio < THRESHOLDS["throughput_ratio"]:
        raise VerificationError(f"throughput ratio {throughput_ratio:.3f} is below 5.0x")

    custom_memory = require_positive_number(headline, "custom_peak_memory_bytes")
    rtx_memory = require_positive_number(headline, "rtx_peak_memory_bytes")
    memory_ratio = custom_memory / rtx_memory
    if memory_ratio > THRESHOLDS["memory_ratio"]:
        raise VerificationError(f"memory ratio {memory_ratio:.3f} exceeds 80 percent of RTX baseline")

    runs = headline.get("independent_run_images_per_second")
    if not isinstance(runs, list) or len(runs) < 3:
        raise VerificationError("headline requires at least three independent runs")
    values = [float(v) for v in runs if isinstance(v, (int, float)) and v > 0]
    if len(values) != len(runs):
        raise VerificationError("independent run throughputs must be positive numbers")
    relative_range = (max(values) - min(values)) / statistics.fmean(values)
    if relative_range > THRESHOLDS["reproducibility_relative_range"]:
        raise VerificationError(f"three-run reproducibility range {relative_range:.3f} exceeds 5 percent")


def verify_declared_criteria(criteria: dict[str, Any]) -> None:
    missing = CRITERIA - criteria.keys()
    if missing:
        raise VerificationError("missing criteria verdicts: " + ", ".join(sorted(missing)))
    for name in CRITERIA:
        value = criteria[name]
        if not isinstance(value, dict):
            raise VerificationError(f"criterion {name} must be an object")
        if value.get("verdict") != "PASS":
            raise VerificationError(f"criterion {name} is not PASS")


def verify_csv_rows(path: Path, expected_count: int) -> None:
    if not path.exists():
        raise VerificationError(f"missing benchmark CSV evidence: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_count:
        raise VerificationError(f"benchmark CSV row count {len(rows)} does not match JSON run count {expected_count}")
    required_columns = {
        "config_id",
        "dataset_manifest_sha256",
        "implementation",
        "mean_ms",
        "p95_ms",
        "peak_allocated_bytes",
        "rgb_psnr_db",
        "semantic_agreement",
        "pass",
    }
    missing = required_columns - set(rows[0].keys())
    if missing:
        raise VerificationError("benchmark CSV missing columns: " + ", ".join(sorted(missing)))
    failing = [row.get("config_id", "<missing>") for row in rows if row.get("pass") != "PASS"]
    if failing:
        raise VerificationError("benchmark CSV has failing rows: " + ", ".join(failing[:5]))


def verify_acceptance(evidence_dir: Path) -> dict[str, str]:
    acceptance = load_acceptance(evidence_dir / "acceptance.json")
    if acceptance.get("schema_version") != SCHEMA_VERSION:
        raise VerificationError(f"schema_version must be {SCHEMA_VERSION}")

    verify_setup(require_mapping(acceptance, "setup_validation"), evidence_dir)
    verify_dataset_manifests(require_list(acceptance, "dataset_manifests"))
    benchmark_runs = require_list(acceptance, "benchmark_runs")
    verify_benchmark_runs(benchmark_runs)
    verify_csv_rows(evidence_dir / "benchmark_rows.csv", len(benchmark_runs))
    verify_fidelity(require_list(acceptance, "fidelity"), evidence_dir)
    verify_replay(require_mapping(acceptance, "deterministic_replay"))
    verify_shared_scene_memory(require_mapping(acceptance, "shared_scene_memory"))
    verify_active_subset_cadence(require_mapping(acceptance, "active_subset_cadence"))
    verify_cuda_checks(require_mapping(acceptance, "cuda_error_checks"))
    verify_stability(require_mapping(acceptance, "stability_10_min"))
    verify_headline(require_mapping(acceptance, "headline"))
    verify_declared_criteria(require_mapping(acceptance, "criteria"))
    return {name: "PASS" for name in sorted(CRITERIA)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("outputs/verification"),
        help="Directory containing acceptance.json and benchmark_rows.csv.",
    )
    args = parser.parse_args(argv)

    try:
        verdicts = verify_acceptance(args.evidence_dir)
    except VerificationError as exc:
        print(f"VERIFICATION_FAIL {exc}")
        return 1
    print(json.dumps({"verdict": "PASS", "criteria": verdicts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
