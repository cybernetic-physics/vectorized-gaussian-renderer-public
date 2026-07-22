#!/usr/bin/env python3
"""Verify the FlashGS short-tail repair on five traces and all 92 pixels."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from benchmarks.flashgs_matched_occupancy import (  # noqa: E402
    ComputeProcessSampler,
    capture_node_snapshot,
    ensure_cooperative_executor_lock,
    occupancy_failures,
)
from benchmarks.run_flashgs_matched import (  # noqa: E402
    HOME_SCAN_SHA256,
    SCENE_ID,
    load_home_scan,
)
from isaacsim_gaussian_renderer import FlashGSBackend, RendererService  # noqa: E402
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    active_cuda_device_uuid,
    artifact_record,
    load_verified_source_manifest,
)
from isaacsim_gaussian_renderer.evaluation.matched_semantics import (  # noqa: E402
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    SEMANTIC_TOPOLOGIES,
)
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (  # noqa: E402
    load_trajectory,
)
from isaacsim_gaussian_renderer.flashgs_debug import (  # noqa: E402
    load_verified_flashgs_adapter_attestation,
)
from isaacsim_gaussian_renderer.flashgs_repair import (  # noqa: E402
    FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA,
    aggregate_b64_repair_cases,
    audit_repaired_flashgs_production,
    evaluate_repaired_b64_case,
    evaluate_repaired_mismatch_corpus,
    evaluate_repaired_rgb_only_corpus,
    load_b64_failure_manifest,
    load_verified_b64_diagnosis_index,
    trace_explicit_flashgs_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Untimed, synchronized validation of all frozen B64 short-tail "
            "culprits and all 92 historical mismatch pixels."
        )
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--scene-sha256", default=HOME_SCAN_SHA256)
    parser.add_argument("--known-failure-manifest", type=Path, required=True)
    parser.add_argument("--diagnosis-lock", type=Path, required=True)
    parser.add_argument("--diagnosis-index", type=Path, required=True)
    parser.add_argument(
        "--diagnosis-artifact-root",
        type=Path,
        required=True,
        help="Root for the diagnosis index's portable relative paths.",
    )
    parser.add_argument("--intersection-capacity", type=int, required=True)
    parser.add_argument(
        "--semantic-topology",
        choices=SEMANTIC_TOPOLOGIES,
        default=REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    )
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--flashgs-adapter-attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _report_relative_artifact(path: Path, *, report_path: Path) -> dict[str, object]:
    record = artifact_record(path)
    record["path"] = path.resolve().relative_to(report_path.resolve().parent).as_posix()
    return record


def _rendered_pixel(
    arrays: dict[str, np.ndarray],
    *,
    x: int,
    y: int,
) -> dict[str, object]:
    return {
        "rgb": [_json_number(value) for value in arrays["rgb"][y, x]],
        "depth": _json_number(float(arrays["depth"][y, x, 0])),
        "alpha": _json_number(float(arrays["alpha"][y, x, 0])),
        "semantic_id": int(arrays["semantic_id"][y, x, 0]),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    full_raw_path = args.output.with_suffix(".full.npz")
    rgb_raw_path = args.output.with_suffix(".rgb-only.npz")
    existing_outputs = [path for path in (args.output, full_raw_path, rgb_raw_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite repair evidence: {existing_outputs}.")
    if args.intersection_capacity <= 0:
        raise ValueError("intersection_capacity must be positive.")

    known = load_b64_failure_manifest(args.known_failure_manifest)
    diagnosis = load_verified_b64_diagnosis_index(
        args.diagnosis_index,
        known_failure_manifest=args.known_failure_manifest,
        diagnosis_lock=args.diagnosis_lock,
        artifact_root=args.diagnosis_artifact_root,
    )
    corpus = diagnosis["payload"]["historical_mismatch_corpus"]
    if corpus.get("semantic_topology") != args.semantic_topology:
        raise ValueError("Diagnosis index semantic topology differs.")
    mismatch_cameras = {int(pixel["camera_index"]) for pixel in corpus.get("pixels", [])}
    case_cameras = {int(case["camera_index"]) for case in known["cases"]}
    if mismatch_cameras != case_cameras:
        raise ValueError("The 92-pixel corpus and five traced cameras differ.")
    source_provenance = load_verified_source_manifest(
        args.source_manifest,
        project_root=PROJECT_ROOT,
    )
    if source_provenance.get("dirty") is not False:
        raise RuntimeError("Repair verification requires a clean source manifest.")
    adapter_attestation = load_verified_flashgs_adapter_attestation(
        args.flashgs_adapter_attestation,
        source_provenance=source_provenance,
        project_root=PROJECT_ROOT,
    )

    trajectory = load_trajectory(args.trajectory)
    if (
        trajectory.trajectory_id != known["payload"]["trajectory_id"]
        or trajectory.scene_sha256 != args.scene_sha256
        or args.scene_sha256 != known["payload"]["scene_sha256"]
        or trajectory.batch != known["payload"]["batch"]
        or trajectory.width != known["payload"]["width"]
        or trajectory.height != known["payload"]["height"]
        or not 0 <= known["payload"]["step"] < trajectory.timesteps
    ):
        raise ValueError("Repair trajectory differs from the frozen B64 contract.")

    executor_lock = ensure_cooperative_executor_lock()
    preflight = capture_node_snapshot()
    preflight_failures = occupancy_failures(
        preflight,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=False,
    )
    if preflight_failures:
        raise RuntimeError("Shared RTX node occupancy preflight failed: " + "; ".join(preflight_failures))
    occupancy_sampler: ComputeProcessSampler | None = ComputeProcessSampler(
        expected_gpu_uuid=args.expected_gpu_uuid,
        allowed_pids={os.getpid()},
    ).start()
    service: RendererService | None = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("FlashGS repair verification requires CUDA.")
        torch.cuda.init()
        gpu_uuid = active_cuda_device_uuid(torch.cuda)
        if gpu_uuid != args.expected_gpu_uuid:
            raise RuntimeError(f"Active GPU UUID {gpu_uuid!r} != {args.expected_gpu_uuid!r}.")

        scene, scene_metadata = load_home_scan(
            args.scene_path,
            args.scene_sha256,
            args.semantic_topology,
        )
        backend = FlashGSBackend(
            max_intersections=args.intersection_capacity,
            near_plane=trajectory.near_plane,
            far_plane=trajectory.far_plane,
            gaussian_support_sigma=MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            covariance_epsilon=0.0,
            semantic_min_alpha=0.01,
            tile_size=16,
        )
        service = RendererService(
            backend,
            height=trajectory.height,
            width=trajectory.width,
            outputs=("rgb", "depth", "alpha", "semantic_id"),
            max_views=64,
        )
        service.initialize(stage=None, device="cuda")
        service.load_scene(SCENE_ID, **scene)
        step = int(known["payload"]["step"])
        b64_viewmats = torch.from_numpy(trajectory.viewmats[step]).to("cuda")
        b64_intrinsics = torch.from_numpy(trajectory.intrinsics[step]).to("cuda")
        b64_scene_ids = torch.full((64,), SCENE_ID, device="cuda", dtype=torch.int64)
        if b64_viewmats.shape != (64, 4, 4) or b64_intrinsics.shape != (
            64,
            3,
            3,
        ):
            raise ValueError("Frozen step does not contain an exact B64 camera batch.")

        full_outputs = service.render(
            b64_viewmats,
            b64_intrinsics,
            b64_scene_ids,
        )
        service.synchronize()
        b64_full_capacity = backend.check_capacity(synchronize=False)
        full_b64_arrays = {name: value.detach().cpu().numpy().copy() for name, value in full_outputs.items()}
        rgb_only_output = {
            "rgb": torch.empty(
                (64, trajectory.height, trajectory.width, 3),
                device="cuda",
                dtype=torch.float32,
            )
        }
        rgb_outputs = service.render(
            b64_viewmats,
            b64_intrinsics,
            b64_scene_ids,
            outputs=rgb_only_output,
        )
        service.synchronize()
        b64_rgb_only_capacity = backend.check_capacity(synchronize=False)
        rgb_only_execution_stats = dict(backend.execution_stats)
        rgb_b64_array = rgb_outputs["rgb"].detach().cpu().numpy().copy()

        indexed_by_id = diagnosis["cases_by_id"]
        corpus_by_coordinate = {
            (
                int(pixel["camera_index"]),
                int(pixel["pixel_xy"][0]),
                int(pixel["pixel_xy"][1]),
            ): pixel
            for pixel in corpus["pixels"]
        }
        rendered_by_camera = {
            camera_index: {name: array[camera_index] for name, array in full_b64_arrays.items()}
            for camera_index in range(64)
        }
        rgb_only_by_camera = {camera_index: rgb_b64_array[camera_index] for camera_index in range(64)}
        case_results: list[dict[str, object]] = []
        all_zero_overflow = not bool(
            b64_full_capacity["intersection_overflow"] or b64_rgb_only_capacity["intersection_overflow"]
        )
        debug_build_contract: dict[str, object] | None = None
        for frozen_case in known["cases"]:
            case_id = frozen_case["case_id"]
            indexed_case = indexed_by_id[case_id]
            camera_index = int(frozen_case["camera_index"])
            x, y = (int(value) for value in frozen_case["pixel_xy"])
            coordinate = (camera_index, x, y)
            frozen_mismatch = corpus_by_coordinate.get(coordinate)
            if frozen_mismatch is None:
                raise ValueError(f"Mismatch corpus lacks representative {case_id}.")
            viewmat = b64_viewmats[camera_index : camera_index + 1]
            intrinsics = b64_intrinsics[camera_index : camera_index + 1]
            outputs = service.render(
                viewmat,
                intrinsics,
                b64_scene_ids[:1],
            )
            service.synchronize()
            trace_seed_capacity = backend.check_capacity(synchronize=False)
            all_zero_overflow = all_zero_overflow and not bool(trace_seed_capacity["intersection_overflow"])
            trace = trace_explicit_flashgs_candidates(
                backend,
                viewmat=viewmat[0],
                intrinsics=intrinsics[0],
                target_pixel_x=x,
                target_pixel_y=y,
                candidate_ids=[int(culprit["gaussian_id"]) for culprit in indexed_case["culprits"]],
                expected_semantic_id=int(frozen_case["oracle_semantic_id"]),
            )
            debug_build_contract = trace.get("native_build_contract")
            del outputs
            arrays = rendered_by_camera[camera_index]
            oracle_pixel = {
                "view_index": frozen_mismatch["view_index"],
                "step": frozen_mismatch["step"],
                "camera_index": camera_index,
                "pixel_xy": [x, y],
                "alpha": frozen_mismatch["oracle"]["alpha"],
                "semantic_id": frozen_mismatch["oracle"]["semantic_id"],
            }
            evaluation = evaluate_repaired_b64_case(
                frozen_case=frozen_case,
                indexed_case=indexed_case,
                oracle_pixel=oracle_pixel,
                rendered_pixel=_rendered_pixel(arrays, x=x, y=y),
                trace=trace,
                capacity_counters=b64_full_capacity,
            )
            case_results.append(
                {
                    **evaluation,
                    "b64_render_capacity": b64_full_capacity,
                    "trace_seed_b1_capacity": trace_seed_capacity,
                    "trace": trace,
                }
            )

        mismatch_result = evaluate_repaired_mismatch_corpus(
            corpus=corpus,
            rendered_by_camera=rendered_by_camera,
        )
        rgb_only_result = evaluate_repaired_rgb_only_corpus(
            corpus=corpus,
            rendered_rgb_by_camera=rgb_only_by_camera,
        )
        aggregate = aggregate_b64_repair_cases(
            case_results,
            expected_case_ids=known["cases_by_id"],
        )
        sampled_telemetry = occupancy_sampler.stop()
        occupancy_sampler = None
        postflight = capture_node_snapshot()
        postflight_failures = occupancy_failures(
            postflight,
            expected_gpu_uuid=args.expected_gpu_uuid,
            allow_current_gpu_process=True,
        )
        production_native_path = Path(str(backend._native.__file__)).resolve()
        production_source_audit = audit_repaired_flashgs_production(
            native_root=(PROJECT_ROOT / "src/isaacsim_gaussian_renderer/native/flashgs"),
            build_contract=getattr(backend._native, "__vgr_build_contract__", None),
            repository_root=PROJECT_ROOT,
            pre_fix_source_identity=diagnosis["payload"]["diagnosis_source_identity"],
            loaded_module_name=str(getattr(backend._native, "__name__", "")),
            loaded_module_path=production_native_path,
        )
        integrity_checks = {
            "clean_source_manifest": source_provenance.get("dirty") is False,
            "adapter_attestation_bound": bool(adapter_attestation),
            "diagnosis_index_verified": diagnosis["payload"].get("pass") is True,
            "historical_mismatch_count_92": corpus.get("mismatch_count") == 92,
            "preflight_occupancy": not preflight_failures,
            "postflight_occupancy": not postflight_failures,
            "cooperative_node_wide_lock": executor_lock["pass"],
            "sampled_compute_process_telemetry": sampled_telemetry["pass"],
            "production_native_build_contract": getattr(backend._native, "__vgr_build_contract__", None) is not None,
            "debug_native_build_contract": debug_build_contract is not None,
            "production_source_digest_and_repair_predicates": (production_source_audit["pass"]),
            "exact_b64_full_render_captured": all(array.shape[0] == 64 for array in full_b64_arrays.values())
            and len(rendered_by_camera) == 64,
            "exact_b64_rgb_only_render_captured": rgb_b64_array.shape == (64, trajectory.height, trajectory.width, 3),
            "rgb_only_specialization_exercised": (
                rgb_only_execution_stats.get("full_sensor_output") is False and len(rgb_only_by_camera) == 64
            ),
            "zero_intersection_overflow_b64_and_trace_seeds": all_zero_overflow,
        }
        tool_integrity_pass = all(integrity_checks.values())
        passed = bool(tool_integrity_pass and aggregate["pass"] and mismatch_result["pass"] and rgb_only_result["pass"])
        camera_order = list(range(64))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            full_raw_path,
            rgb=full_b64_arrays["rgb"],
            depth=full_b64_arrays["depth"],
            alpha=full_b64_arrays["alpha"],
            semantic_id=full_b64_arrays["semantic_id"],
            camera_indices=np.asarray(camera_order, dtype=np.int64),
            step=np.asarray(step, dtype=np.int64),
            trajectory_id=np.asarray(trajectory.trajectory_id),
            semantic_topology=np.asarray(args.semantic_topology),
        )
        np.savez(
            rgb_raw_path,
            rgb=rgb_b64_array,
            camera_indices=np.asarray(camera_order, dtype=np.int64),
            step=np.asarray(step, dtype=np.int64),
            trajectory_id=np.asarray(trajectory.trajectory_id),
            semantic_topology=np.asarray(args.semantic_topology),
        )
        raw_output_artifacts = {
            "full": _report_relative_artifact(full_raw_path, report_path=args.output),
            "rgb_only": _report_relative_artifact(rgb_raw_path, report_path=args.output),
        }
        result = {
            "schema_version": FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "debug_only": True,
            "measured_timing_valid": False,
            "pass": passed,
            "tool_integrity": {
                "pass": tool_integrity_pass,
                "checks": integrity_checks,
            },
            "aggregate": aggregate,
            "historical_mismatch_repair": mismatch_result,
            "rgb_only_specialization": {
                "result": rgb_only_result,
                "b64_capacity": b64_rgb_only_capacity,
                "execution_stats": rgb_only_execution_stats,
            },
            "b64_full_sensor_capacity": b64_full_capacity,
            "post_fix_raw_outputs": raw_output_artifacts,
            "cases": case_results,
            "known_failure_manifest": known["artifact"],
            "diagnosis_index": diagnosis["artifact"],
            "diagnosis_lock": diagnosis["payload"]["diagnosis_lock"],
            "historical_mismatch_inputs": {
                key: corpus[key]
                for key in (
                    "historical_run",
                    "historical_flashgs_output",
                    "pinned_oracle_output",
                    "camera_bundle",
                    "coordinate_sha256",
                    "oracle_values_sha256",
                )
            },
            "scene": scene_metadata,
            "camera": {
                "trajectory": artifact_record(args.trajectory),
                "trajectory_id": trajectory.trajectory_id,
                "step": step,
            },
            "source_provenance": source_provenance,
            "flashgs_adapter_attestation": adapter_attestation,
            "production_adapter": {
                "native_extension": artifact_record(production_native_path),
                "native_build_contract": getattr(backend._native, "__vgr_build_contract__", None),
                "render_source": artifact_record(
                    PROJECT_ROOT / "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
                ),
                "source_digest_and_repair_audit": production_source_audit,
            },
            "debug_native_build_contract": debug_build_contract,
            "equation": {
                "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
                "covariance_epsilon": 0.0,
                "alpha_threshold": 1.0 / 255.0,
                "alpha_cap": 0.99,
                "transmittance_threshold": 1.0e-4,
                "semantic_min_alpha": 0.01,
                "semantic_topology": args.semantic_topology,
            },
            "environment": {
                "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "gpu_uuid": gpu_uuid,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "occupancy": {
                "executor_control": {
                    "cooperative_node_wide_lock": executor_lock,
                    "scope": "all-visible-gpus",
                },
                "preflight": preflight,
                "preflight_failures": preflight_failures,
                "sampled_compute_process_telemetry": sampled_telemetry,
                "postflight": postflight,
                "postflight_failures": postflight_failures,
            },
        }
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            "FLASHGS_B64_REPAIR_VERIFICATION "
            + json.dumps(
                {
                    "output": str(args.output),
                    "pass": passed,
                    "cases_passed": aggregate["passed_case_count"],
                    "pixels_passed": mismatch_result["passed_pixel_count"],
                    "rgb_only_pixels_passed": rgb_only_result["passed_pixel_count"],
                },
                sort_keys=True,
            )
        )
        if not passed:
            raise SystemExit(1)
    finally:
        if service is not None:
            service.shutdown()
        if occupancy_sampler is not None:
            occupancy_sampler.stop()


if __name__ == "__main__":
    main()
