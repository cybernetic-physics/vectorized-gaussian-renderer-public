"""Build a fail-closed headline acceptance report from raw benchmark runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


THROUGHPUT_RATIO_MIN = 5.0
MEMORY_RATIO_MAX = 0.8
REPRODUCIBILITY_SPREAD_MAX = 0.05
REQUIRED_OUTPUTS = {"rgb", "depth", "alpha", "semantic_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom-run",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--ovrtx-run",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--ewa-gsplat-fidelity",
        type=Path,
        required=True,
        help=(
            "Strict fidelity report for the production screen-space EWA "
            "renderer against the pinned conventional gsplat reference."
        ),
    )
    parser.add_argument(
        "--ovrtx-perspective-fidelity",
        type=Path,
        required=True,
        help=(
            "Strict fidelity report for the production renderer against a "
            "temporally accumulated OVRTX perspective reference."
        ),
    )
    parser.add_argument(
        "--ovrtx-exact-ray-diagnostic",
        type=Path,
        required=True,
        help=(
            "Diagnostic-only exact 3D-ray model comparison against the "
            "converged OVRTX perspective reference."
        ),
    )
    parser.add_argument(
        "--home-finite-frame-control",
        type=Path,
        required=True,
        help=(
            "Matched Home Scan finite-frame control. This is retained as a "
            "control and must not replace the converged perspective contract."
        ),
    )
    parser.add_argument(
        "--projection-mode-audit",
        type=Path,
        required=True,
        help=(
            "Audit of whether authored OVRTX perspective and tangential "
            "projection hints produced observably different tensors."
        ),
    )
    parser.add_argument("--custom-single", type=Path, required=True)
    parser.add_argument("--ovrtx-single", type=Path, required=True)
    parser.add_argument(
        "--ovrtx-next-batch-failure-log",
        type=Path,
        required=True,
    )
    parser.add_argument("--headline-batch", type=int, default=1024)
    parser.add_argument("--next-batch", type=int, default=2048)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--dynamic-trajectory-run", type=Path, action="append", default=[])
    parser.add_argument("--isaac-fabric-run", type=Path, action="append", default=[])
    parser.add_argument("--cache-audit", type=Path)
    parser.add_argument("--artifact-audit", type=Path, action="append", default=[])
    parser.add_argument("--stability-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_result(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = sorted(
        candidate
        for candidate in path.glob("*.json")
        if candidate.name
        not in {
            "baseline-result.schema.json",
            "semantic-id-map.json",
            "run-manifest.json",
        }
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one result JSON under {path}, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def load_result(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_result(path)
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def custom_metrics(result: dict[str, Any]) -> dict[str, Any]:
    outputs = result.get("outputs")
    output_names = set(outputs) if isinstance(outputs, dict) else set(outputs)
    return {
        "batch": int(result["batch"]),
        "width": int(result["width"]),
        "height": int(result["height"]),
        "frame_ms": float(result["frame_ms"]["mean"]),
        "images_per_second": float(result["images_per_second"]),
        "driver_memory_bytes": int(
            result["driver_process_memory_bytes"]
        ),
        "camera_checksum": str(
            result["camera_manifest"]["checksum_sha256"]
        ),
        "scene_checksum": str(
            result["dataset"]["sha256"]
            if "dataset" in result
            else result["scene"]["checksum_sha256"]
        ),
        "outputs": sorted(output_names),
        "pass": result.get("pass") is True,
        "outputs_contract_valid": (
            result.get("outputs_contract_valid") is True
        ),
        "outputs_gpu_resident": (
            result.get("outputs_gpu_resident") is True
        ),
        "allocation_delta_bytes": int(
            result["allocation_delta_bytes"]
        ),
        "zero_overflow": (
            int(result["measured_counters"]["visible_overflow"]) == 0
            and int(
                result["measured_counters"]["intersection_overflow"]
            )
            == 0
        ),
        "pipeline": str(result.get("pipeline", "")),
        "projection_cache": result.get("projection_cache", {}),
        "gaussian_evaluation_model": str(
            result.get("gaussian_evaluation_model", "")
        ),
        "runtime_camera_loop": result.get("runtime_camera_loop"),
        "batched_native_submissions_per_render": result.get(
            "batched_native_submissions_per_render"
        ),
    }


def ovrtx_metrics(result: dict[str, Any]) -> dict[str, Any]:
    configuration = result["configuration"]
    dataset = result["dataset"]
    return {
        "batch": int(configuration["batch_size"]),
        "width": int(configuration["width"]),
        "height": int(configuration["height"]),
        "frame_ms": float(result["timing"]["wall_ms"]["mean"]),
        "images_per_second": float(
            result["timing"]["images_per_second"]
        ),
        "driver_memory_bytes": int(
            result["memory"]["driver_process_memory_bytes"]
        ),
        "camera_checksum": str(
            configuration["camera_manifest"]["checksum_sha256"]
        ),
        "scene_checksum": str(
            dataset.get(
                "file_sha256",
                dataset.get("checksum_sha256"),
            )
        ),
        "outputs": sorted(configuration["outputs"]),
        "pass": result["status"]["pass"] is True,
        "frame_generation": bool(
            result["implementation"]["flags"][
                "dlss_frame_generation"
            ]
        ),
        "fabric_scene_delegate": configuration.get(
            "fabric_scene_delegate"
        ),
    }


def relative_spread(values: list[float]) -> float:
    return (max(values) - min(values)) / statistics.fmean(values)


def criterion(passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


def fidelity_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate", {})
    metrics = aggregate.get("metrics", {})
    return {
        "pass": (
            report.get("pass") is True
            and aggregate.get("pass") is True
        ),
        "config_id": report.get("config_id"),
        "contract": report.get("contract"),
        "metrics": metrics,
        "thresholds": aggregate.get(
            "thresholds",
            report.get("thresholds"),
        ),
        "worst_view": report.get("worst_view"),
    }


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_rows(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def main() -> None:
    args = parse_args()
    if len(args.custom_run) < 3 or len(args.ovrtx_run) < 3:
        raise ValueError(
            "Headline reproducibility requires at least three independent "
            "custom and OVRTX runs."
        )
    if args.headline_batch <= 0 or args.next_batch <= args.headline_batch:
        raise ValueError("Expected 0 < headline batch < next batch.")

    custom_rows = []
    custom_paths = []
    for path in args.custom_run:
        resolved, result = load_result(path)
        custom_paths.append(str(resolved))
        custom_rows.append(custom_metrics(result))

    ovrtx_rows = []
    ovrtx_paths = []
    for path in args.ovrtx_run:
        resolved, result = load_result(path)
        ovrtx_paths.append(str(resolved))
        ovrtx_rows.append(ovrtx_metrics(result))

    custom_single_path, custom_single_raw = load_result(
        args.custom_single
    )
    ovrtx_single_path, ovrtx_single_raw = load_result(
        args.ovrtx_single
    )
    custom_single = custom_metrics(custom_single_raw)
    ovrtx_single = ovrtx_metrics(ovrtx_single_raw)
    ewa_gsplat_fidelity = json.loads(
        args.ewa_gsplat_fidelity.read_text(encoding="utf-8")
    )
    ovrtx_perspective_fidelity = json.loads(
        args.ovrtx_perspective_fidelity.read_text(encoding="utf-8")
    )
    ovrtx_exact_ray_diagnostic = json.loads(
        args.ovrtx_exact_ray_diagnostic.read_text(encoding="utf-8")
    )
    home_finite_frame_control = json.loads(
        args.home_finite_frame_control.read_text(encoding="utf-8")
    )
    projection_mode_audit = json.loads(
        args.projection_mode_audit.read_text(encoding="utf-8")
    )
    next_failure_log = args.ovrtx_next_batch_failure_log.read_text(
        encoding="utf-8",
        errors="replace",
    )
    source_manifest = load_optional_json(args.source_manifest)
    dynamic_runs = load_json_rows(args.dynamic_trajectory_run)
    isaac_runs = load_json_rows(args.isaac_fabric_run)
    cache_audit = load_optional_json(args.cache_audit)
    artifact_audits = load_json_rows(args.artifact_audit)
    stability_run = load_optional_json(args.stability_run)

    custom_ips = [
        row["images_per_second"]
        for row in custom_rows
    ]
    ovrtx_ips = [
        row["images_per_second"]
        for row in ovrtx_rows
    ]
    custom_mean_ips = statistics.fmean(custom_ips)
    ovrtx_mean_ips = statistics.fmean(ovrtx_ips)
    speedup = custom_mean_ips / ovrtx_mean_ips
    custom_memory = max(
        row["driver_memory_bytes"]
        for row in custom_rows
    )
    ovrtx_memory = max(
        row["driver_memory_bytes"]
        for row in ovrtx_rows
    )
    memory_ratio = custom_memory / ovrtx_memory
    custom_spread = relative_spread(custom_ips)
    ovrtx_spread = relative_spread(ovrtx_ips)

    contract_rows = custom_rows + ovrtx_rows
    matched_contract = (
        all(
            row["batch"] == args.headline_batch
            for row in contract_rows
        )
        and len(
            {
                (
                    row["width"],
                    row["height"],
                    row["camera_checksum"],
                    row["scene_checksum"],
                    tuple(row["outputs"]),
                )
                for row in contract_rows
            }
        )
        == 1
    )
    all_outputs = all(
        set(row["outputs"]) == REQUIRED_OUTPUTS
        for row in contract_rows
    )
    custom_execution_valid = all(
        row["pass"]
        and row["outputs_contract_valid"]
        and row["outputs_gpu_resident"]
        and row["allocation_delta_bytes"] == 0
        and row["zero_overflow"]
        and row["runtime_camera_loop"] is False
        and row["batched_native_submissions_per_render"] == 1
        and row["pipeline"].startswith("compact-project-")
        and row["gaussian_evaluation_model"]
        == "screen_space_2d_conic"
        and row["projection_cache"].get("enabled") is True
        and "outputs are rewritten on every render"
        in str(row["projection_cache"].get("scope", ""))
        for row in custom_rows
    )
    ovrtx_execution_valid = all(
        row["pass"] and not row["frame_generation"]
        for row in ovrtx_rows
    )
    next_batch_oom = (
        "ERROR_OUT_OF_DEVICE_MEMORY" in next_failure_log
        or "Out of GPU memory" in next_failure_log
    )

    criteria = {
        "matched_comparison_contract": criterion(
            matched_contract and all_outputs,
            {
                "headline_batch": args.headline_batch,
                "required_outputs": sorted(REQUIRED_OUTPUTS),
            },
        ),
        "custom_execution_contract": criterion(
            custom_execution_valid,
            {
                "pipeline": sorted(
                    {row["pipeline"] for row in custom_rows}
                ),
                "run_count": len(custom_rows),
            },
        ),
        "ovrtx_execution_contract": criterion(
            ovrtx_execution_valid,
            {
                "run_count": len(ovrtx_rows),
                "frame_generation_disabled": True,
                # Carried verbatim from each row's configuration; absence
                # (null) means the row predates FSD-state recording.
                "fabric_scene_delegate": [
                    row["fabric_scene_delegate"]
                    for row in ovrtx_rows
                ],
            },
        ),
        "throughput": criterion(
            speedup >= THROUGHPUT_RATIO_MIN,
            {
                "custom_mean_images_per_second": custom_mean_ips,
                "ovrtx_mean_images_per_second": ovrtx_mean_ips,
                "speedup": speedup,
                "threshold": THROUGHPUT_RATIO_MIN,
            },
        ),
        "peak_gpu_memory": criterion(
            memory_ratio <= MEMORY_RATIO_MAX,
            {
                "custom_peak_driver_memory_bytes": custom_memory,
                "ovrtx_peak_driver_memory_bytes": ovrtx_memory,
                "ratio": memory_ratio,
                "reduction": 1.0 - memory_ratio,
                "threshold": MEMORY_RATIO_MAX,
            },
        ),
        "three_run_reproducibility": criterion(
            custom_spread <= REPRODUCIBILITY_SPREAD_MAX
            and ovrtx_spread <= REPRODUCIBILITY_SPREAD_MAX,
            {
                "custom_images_per_second": custom_ips,
                "custom_relative_spread": custom_spread,
                "ovrtx_images_per_second": ovrtx_ips,
                "ovrtx_relative_spread": ovrtx_spread,
                "threshold": REPRODUCIBILITY_SPREAD_MAX,
            },
        ),
        "ewa_gsplat_fidelity": criterion(
            fidelity_snapshot(ewa_gsplat_fidelity)["pass"],
            {
                "contract": (
                    "deterministic screen-space 2D EWA projection and "
                    "front-to-back alpha compositing"
                ),
                **fidelity_snapshot(ewa_gsplat_fidelity),
            },
        ),
        "ovrtx_perspective_fidelity": criterion(
            fidelity_snapshot(ovrtx_perspective_fidelity)["pass"],
            {
                "contract": (
                    "temporally accumulated OVRTX perspective Gaussian "
                    "reconstruction/compositing"
                ),
                **fidelity_snapshot(ovrtx_perspective_fidelity),
            },
        ),
        "largest_common_batch": criterion(
            next_batch_oom,
            {
                "largest_common_batch": args.headline_batch,
                "next_tested_ovrtx_batch": args.next_batch,
                "next_batch_ovrtx_out_of_memory": next_batch_oom,
                "failure_log": str(
                    args.ovrtx_next_batch_failure_log
                ),
            },
        ),
        "single_camera_reported": criterion(
            custom_single["batch"] == 1
            and ovrtx_single["batch"] == 1
            and custom_single["pass"]
            and ovrtx_single["pass"],
            {
                "custom_result": str(custom_single_path),
                "custom_images_per_second": custom_single[
                    "images_per_second"
                ],
                "ovrtx_result": str(ovrtx_single_path),
                "ovrtx_images_per_second": ovrtx_single[
                    "images_per_second"
                ],
                "ovrtx_fabric_scene_delegate": ovrtx_single[
                    "fabric_scene_delegate"
                ],
                "speedup": (
                    custom_single["images_per_second"]
                    / ovrtx_single["images_per_second"]
                ),
            },
        ),
    }
    dynamic_nonstatic = [
        row
        for row in dynamic_runs
        if row.get("scenario") not in {None, "static-repeat"}
    ]
    dynamic_pass = bool(dynamic_nonstatic) and all(row.get("pass") is True for row in dynamic_nonstatic)
    dynamic_ips = []
    for row in dynamic_nonstatic:
        timing_path = Path(row.get("timing", ""))
        if not timing_path.is_absolute() and args.dynamic_trajectory_run:
            source_index = dynamic_runs.index(row)
            timing_path = args.dynamic_trajectory_run[source_index].parent / timing_path
        if timing_path.is_file():
            dynamic_ips.append(float(json.loads(timing_path.read_text())["images_per_second"]))
    dynamic_spread = relative_spread(dynamic_ips) if len(dynamic_ips) >= 2 else None
    dynamic_reproducible = (
        len(dynamic_ips) >= 3
        and dynamic_spread is not None
        and dynamic_spread <= REPRODUCIBILITY_SPREAD_MAX
    )
    isaac_pass = bool(isaac_runs) and all(
        row.get("pass") is True
        and row.get("renderer", {}).get("render_does_not_advance_physics") is True
        and row.get("physics_coupled_performance", {}).get("steps", 0) > 0
        for row in isaac_runs
    )
    provenance_pass = (
        source_manifest is not None
        and source_manifest.get("schema_version")
        in {"renderer-source-manifest-v1", "renderer-source-manifest-v2"}
        and source_manifest.get("diff_sha256")
        and source_manifest.get("head")
        and (
            source_manifest.get("schema_version") == "renderer-source-manifest-v1"
            or (
                source_manifest.get("source_tree_sha256")
                and source_manifest.get("tracked_source_files")
            )
        )
    )
    cache_pass = cache_audit is not None and cache_audit.get("pass") is True
    artifact_pass = bool(artifact_audits) and all(row.get("pass") is True for row in artifact_audits)
    stability_pass = stability_run is not None and stability_run.get("pass") is True
    independent_verdicts = {
        "contract_provenance": criterion(
            bool(matched_contract and all_outputs and provenance_pass),
            {"matched_static_contract": matched_contract, "source_manifest": str(args.source_manifest) if args.source_manifest else None},
        ),
        "ewa_vs_pinned_gsplat_fidelity": criteria["ewa_gsplat_fidelity"],
        "ovrtx_perspective_fidelity": criteria["ovrtx_perspective_fidelity"],
        "static_coherent_performance": criterion(
            criteria["throughput"]["verdict"] == "PASS" and custom_execution_valid,
            {
                "scope": "unchanged-camera projection-cache hits only",
                "speedup": speedup,
                "cannot_satisfy_dynamic_gate": True,
            },
        ),
        "dynamic_trajectory_performance": criterion(
            dynamic_pass,
            {
                "run_count": len(dynamic_nonstatic),
                "scenarios": [row.get("scenario") for row in dynamic_nonstatic],
                "static_results_accepted_here": False,
            },
        ),
        "isaac_fabric_integration": criterion(
            isaac_pass,
            {"run_count": len(isaac_runs), "inputs": [str(path) for path in args.isaac_fabric_run]},
        ),
        "cache_correctness": criterion(cache_pass, {"input": str(args.cache_audit) if args.cache_audit else None}),
        "artifact_integrity": criterion(
            artifact_pass,
            {"run_count": len(artifact_audits), "inputs": [str(path) for path in args.artifact_audit]},
        ),
        "stability": criterion(stability_pass, {"input": str(args.stability_run) if args.stability_run else None}),
        "reproducibility": criterion(
            dynamic_reproducible,
            {
                "dynamic_run_count": len(dynamic_ips),
                "dynamic_images_per_second": dynamic_ips,
                "relative_spread": dynamic_spread,
                "threshold": REPRODUCIBILITY_SPREAD_MAX,
            },
        ),
        "memory": criteria["peak_gpu_memory"],
    }
    overall_pass = all(item["verdict"] == "PASS" for item in independent_verdicts.values())
    report = {
        "schema_version": "headline-acceptance/v3",
        "overall_verdict": "PASS" if overall_pass else "FAIL",
        "headline": {
            "batch": args.headline_batch,
            "custom_mean_images_per_second": custom_mean_ips,
            "ovrtx_mean_images_per_second": ovrtx_mean_ips,
            "speedup": speedup,
            "custom_peak_driver_memory_bytes": custom_memory,
            "ovrtx_peak_driver_memory_bytes": ovrtx_memory,
            "driver_memory_ratio": memory_ratio,
            "driver_memory_reduction": 1.0 - memory_ratio,
            "custom_relative_spread": custom_spread,
            "ovrtx_relative_spread": ovrtx_spread,
        },
        "criteria": criteria,
        "independent_verdicts": independent_verdicts,
        "scope_warning": (
            "Static coherent cache-hit speed is publishable only as static performance; "
            "it cannot satisfy the dynamic trajectory or Isaac/Fabric gates."
        ),
        "diagnostics": {
            "ovrtx_exact_ray": {
                "acceptance_path": False,
                "status": "experimental",
                "model": (
                    "full 3D Gaussian ray evaluation with probabilistic "
                    "front-to-back first-hit compositing"
                ),
                **fidelity_snapshot(ovrtx_exact_ray_diagnostic),
            },
            "home_finite_frame_control": {
                "acceptance_path": False,
                "status": "finite-frame-tuned-control",
                "limitation": (
                    "A passing finite-frame Home Scan comparison does not "
                    "establish equation-level OVRTX perspective parity."
                ),
                **fidelity_snapshot(home_finite_frame_control),
            },
            "ovrtx_projection_mode": projection_mode_audit,
        },
        "inputs": {
            "custom_runs": custom_paths,
            "ovrtx_runs": ovrtx_paths,
            "ewa_gsplat_fidelity": str(
                args.ewa_gsplat_fidelity
            ),
            "ovrtx_perspective_fidelity": str(
                args.ovrtx_perspective_fidelity
            ),
            "ovrtx_exact_ray_diagnostic": str(
                args.ovrtx_exact_ray_diagnostic
            ),
            "home_finite_frame_control": str(
                args.home_finite_frame_control
            ),
            "projection_mode_audit": str(
                args.projection_mode_audit
            ),
            "custom_single": str(custom_single_path),
            "ovrtx_single": str(ovrtx_single_path),
            "ovrtx_next_batch_failure_log": str(
                args.ovrtx_next_batch_failure_log
            ),
            "source_manifest": str(args.source_manifest) if args.source_manifest else None,
            "dynamic_trajectory_runs": [str(path) for path in args.dynamic_trajectory_run],
            "isaac_fabric_runs": [str(path) for path in args.isaac_fabric_run],
            "cache_audit": str(args.cache_audit) if args.cache_audit else None,
            "artifact_audits": [str(path) for path in args.artifact_audit],
            "stability_run": str(args.stability_run) if args.stability_run else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
