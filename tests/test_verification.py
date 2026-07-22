from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from isaacsim_gaussian_renderer.verification import (
    CRITERIA,
    HOME_SCAN_MANIFEST_SHA256,
    REQUIRED_BATCHES,
    REQUIRED_IMPLEMENTATIONS,
    REQUIRED_RESOLUTIONS,
    REQUIRED_SCENE_MODES,
    REQUIRED_SCENES,
    VerificationError,
    recompute_stats,
    verify_acceptance,
)


def _digest(index: int) -> str:
    return f"{index:064x}"[-64:]


def _samples(seed: int) -> list[float]:
    base = 1.0 + seed * 0.001
    return [base + (idx % 7) * 0.01 for idx in range(50)]


def _benchmark_run(
    index: int,
    implementation: str,
    batch: int | str,
    resolution: tuple[int, int],
    scene: str,
    scene_mode: str,
) -> dict:
    samples = _samples(index)
    return {
        "config_id": f"cfg-{index:03d}",
        "implementation": implementation,
        "dataset_manifest_sha256": HOME_SCAN_MANIFEST_SHA256 if scene == "home-scan-lod0" else _digest(index + 1),
        "source_commit": "abc1234",
        "batch_size": batch,
        "resolution": list(resolution),
        "scene": scene,
        "scene_mode": scene_mode,
        "outputs": ["rgb", "depth", "alpha", "semantic_id"],
        "timing_samples_ms": samples,
        "timing_stats": recompute_stats(samples),
        "memory": {
            "peak_allocated_bytes": 8_000_000_000,
            "peak_reserved_bytes": 9_000_000_000,
            "persistent_scene_bytes": 5_000_000_000,
            "workspace_bytes": 1_000_000_000,
            "temporary_allocation_delta_bytes": 0,
            "driver_process_bytes": 10_000_000_000,
        },
        "work_counts": {
            "visible_gaussians": 1_000_000,
            "tile_intersections": 2_000_000,
        },
    }


def _write_fixture(root: Path) -> Path:
    evidence_dir = root / "outputs" / "verification"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "setup.log").write_text("SETUP_OK\n")

    benchmark_runs = []
    index = 0
    for implementation in sorted(REQUIRED_IMPLEMENTATIONS):
        for batch in sorted(REQUIRED_BATCHES, key=str):
            for resolution in sorted(REQUIRED_RESOLUTIONS):
                for scene in sorted(REQUIRED_SCENES):
                    for scene_mode in sorted(REQUIRED_SCENE_MODES):
                        benchmark_runs.append(
                            _benchmark_run(index, implementation, batch, resolution, scene, scene_mode)
                        )
                        index += 1

    acceptance = {
        "schema_version": 1,
        "setup_validation": {
            "fresh_setup_rerun": True,
            "isaac_headless_smoke_ok": True,
            "remote_host": "vast-gsplat-isaac",
            "remote_root": "/workspace/agent-worktrees/verification",
            "vast_instance_id": "45069639",
            "logs": ["setup.log"],
        },
        "dataset_manifests": [
            {
                "dataset_id": "synthetic-small",
                "manifest_sha256": _digest(10),
                "file_count": 1,
                "record_count": 1_000,
            },
            {
                "dataset_id": "synthetic-medium",
                "manifest_sha256": _digest(11),
                "file_count": 1,
                "record_count": 100_000,
            },
            {
                "dataset_id": "public-real-world",
                "manifest_sha256": _digest(12),
                "file_count": 4,
                "record_count": 500_000,
            },
            {
                "dataset_id": "home-scan-lod0",
                "manifest_sha256": HOME_SCAN_MANIFEST_SHA256,
                "file_count": 434,
                "record_count": 21_497_908,
            },
        ],
        "benchmark_runs": benchmark_runs,
        "fidelity": [
            {
                "scene": scene,
                "view_class": view_class,
                "metrics": {
                    "rgb_psnr_db": 45.0,
                    "rgb_ssim": 0.998,
                    "lpips": 0.005,
                    "alpha_mae": 0.001,
                    "depth_relative_error": 0.004,
                    "semantic_agreement": 0.9995,
                },
                "artifacts": {
                    "reference_rgb": f"artifacts/{scene}-{view_class}-reference.png",
                    "candidate_rgb": f"artifacts/{scene}-{view_class}-candidate.png",
                    "rgb_abs_diff": f"artifacts/{scene}-{view_class}-rgb-diff.png",
                    "depth_relative_error_image": f"artifacts/{scene}-{view_class}-depth.png",
                    "alpha_diff": f"artifacts/{scene}-{view_class}-alpha.png",
                    "semantic_mismatch_mask": f"artifacts/{scene}-{view_class}-semantic.png",
                },
            }
            for scene in sorted(REQUIRED_SCENES)
            for view_class in ("representative", "worst_case")
        ],
        "deterministic_replay": {
            "passed": True,
            "selected_environment_rendering_checked": True,
            "output_hashes_by_run": [{"rgb": _digest(20), "depth": _digest(21), "alpha": _digest(22)}] * 2,
        },
        "shared_scene_memory": {
            "passed": True,
            "unique_static_scene_copies": 1,
            "per_environment_transforms_without_static_duplication": True,
        },
        "active_subset_cadence": {
            "passed": True,
            "active_subset_passed": True,
            "render_every_n_steps_passed": True,
        },
        "cuda_error_checks": {
            "passed": True,
            "synchronize_after_measured_range": True,
            "errors": [],
        },
        "stability_10_min": {
            "passed": True,
            "duration_seconds": 600.0,
            "no_cuda_errors": True,
            "no_nan_or_inf": True,
        },
        "headline": {
            "custom_kernel_loaded": True,
            "gpu_resident_outputs": True,
            "no_cpu_output_copies_in_steady_state": True,
            "single_camera_images_per_second": 1_000.0,
            "custom_images_per_second": 6_000.0,
            "rtx_images_per_second": 1_000.0,
            "custom_peak_memory_bytes": 8_000_000_000,
            "rtx_peak_memory_bytes": 10_000_000_000,
            "independent_run_images_per_second": [5_950.0, 6_000.0, 6_050.0],
        },
        "criteria": {name: {"verdict": "PASS"} for name in sorted(CRITERIA)},
    }
    for fidelity_item in acceptance["fidelity"]:
        for artifact in fidelity_item["artifacts"].values():
            artifact_path = evidence_dir / artifact
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"fixture")
    (evidence_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2))

    with (evidence_dir / "benchmark_rows.csv").open("w", newline="") as stream:
        fieldnames = [
            "config_id",
            "dataset_manifest_sha256",
            "implementation",
            "mean_ms",
            "p95_ms",
            "peak_allocated_bytes",
            "rgb_psnr_db",
            "semantic_agreement",
            "pass",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in benchmark_runs:
            writer.writerow(
                {
                    "config_id": run["config_id"],
                    "dataset_manifest_sha256": run["dataset_manifest_sha256"],
                    "implementation": run["implementation"],
                    "mean_ms": run["timing_stats"]["mean_ms"],
                    "p95_ms": run["timing_stats"]["p95_ms"],
                    "peak_allocated_bytes": run["memory"]["peak_allocated_bytes"],
                    "rgb_psnr_db": 45.0,
                    "semantic_agreement": 0.9995,
                    "pass": "PASS",
                }
            )

    return evidence_dir


def test_synthetic_fixture_acceptance_passes(tmp_path: Path) -> None:
    evidence_dir = _write_fixture(tmp_path)

    verdicts = verify_acceptance(evidence_dir)

    assert verdicts["throughput"] == "PASS"
    assert verdicts["benchmark_schema"] == "PASS"


def test_missing_acceptance_fails_closed(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "outputs" / "verification"
    evidence_dir.mkdir(parents=True)

    with pytest.raises(VerificationError, match="missing acceptance evidence"):
        verify_acceptance(evidence_dir)


def test_invalid_statistics_fail_closed(tmp_path: Path) -> None:
    evidence_dir = _write_fixture(tmp_path)
    path = evidence_dir / "acceptance.json"
    data = json.loads(path.read_text())
    data["benchmark_runs"][0]["timing_stats"]["mean_ms"] += 1.0
    path.write_text(json.dumps(data))

    with pytest.raises(VerificationError, match="mean_ms mismatch"):
        verify_acceptance(evidence_dir)


def test_failing_csv_row_fails_closed(tmp_path: Path) -> None:
    evidence_dir = _write_fixture(tmp_path)
    csv_path = evidence_dir / "benchmark_rows.csv"
    rows = list(csv.DictReader(csv_path.open()))
    rows[0]["pass"] = "FAIL"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(VerificationError, match="benchmark CSV has failing rows"):
        verify_acceptance(evidence_dir)
