#!/usr/bin/env python3
"""Offline positive and mutation tests for aggregate_verification.py."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from aggregate_verification import (
    ABLATION_LOGICAL_PATH,
    GATE_RESULT_SCHEMA,
    GPU_REQUIRED,
    ISAAC_LANES,
    MACHINE_PROVENANCE_LOGICAL_PATH,
    NATIVE_ROLE,
    NATIVE_SENTINELS,
    OUTPUTS,
    REPLAY_ARGUMENTS,
    REPLAY_CLAIM_SCOPE,
    REPLAY_COMPARISON_METHOD,
    REPLAY_RENDERER_CONFIGURATION,
    REPLAY_SCHEMA,
    REPLAY_WORKER_SCHEMA,
    REQUIRED_GATE_IDS,
    SPEC_SCHEMA,
    VerificationError,
    artifact_record,
    canonical_json_bytes,
    verify_aggregate_verification,
    write_verification_from_spec,
)
from write_machine_provenance import build_machine_provenance


COMMIT = "1" * 40
GPU_UUID = "GPU-b3c9268d-2b06-d924-90cc-d2171c86ef34"
SCENE_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
STARTED = "2026-01-01T00:00:00+00:00"
RESULT_TIME = "2026-01-01T00:00:01+00:00"
COMPLETED = "2026-01-01T00:00:02+00:00"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def timing_distribution(value: float) -> dict[str, Any]:
    samples = [value for _ in range(100)]
    return {
        "samples": samples,
        "count": 100,
        "mean": value,
    }


def output_validation(contract: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pass": True,
        "rgb_cuda": True,
        "rgb_contiguous": True,
        "rgb_finite": True,
        "rgb_dtype": "torch.float32",
    }
    if contract == "full":
        result.update(
            {
                "alpha_finite": True,
                "foreground_depth_finite": True,
                "background_depth_inf": True,
                "foreground_semantic_nonnegative": True,
                "background_semantic_minus_one": True,
                "alpha_dtype": "torch.float32",
                "depth_dtype": "torch.float32",
                "semantic_dtype": "torch.int64",
                "foreground_fraction": 0.5,
            }
        )
    return result


def native_payload(gate_id: str) -> dict[str, Any]:
    counters = {
        "visible_gaussians": 10,
        "intersection_overflow": 0,
        "visible_overflow": 0,
    }
    if gate_id == "native-cuda-smoke":
        return {
            "nonempty": counters,
            "empty": {
                "visible_gaussians": 0,
                "intersection_overflow": 0,
                "visible_overflow": 0,
            },
            "adaptive": {"pass": True},
            "workspace_storage_aliases": {"pass": True},
            "chunked": {"logical_batch": 8, "physical_batch": 4},
        }
    if gate_id == "native-determinism":
        return {
            "schema_version": "deterministic-cuda-smoke/v1",
            "batch": 4,
            "iterations": 32,
            "bitwise_equal": {name: True for name in OUTPUTS},
            "capacity_invariant_bitwise_equal": {name: True for name in OUTPUTS},
            "counters": counters,
            "pass": True,
        }
    return {
        "scene_ids": [1, 2, 1, 2],
        "inactive_output_policy": {"rgb": 0, "depth": "+inf", "alpha": 0, "semantic_id": -1},
        "packed_gaussians": 12000,
        **counters,
    }


def replay_worker_payload(
    *,
    root: Path,
    native: Path,
    source_smoke: Path,
) -> dict[str, Any]:
    total = REPLAY_ARGUMENTS["batch"] * REPLAY_ARGUMENTS["height"] * REPLAY_ARGUMENTS["width"]
    foreground = 1_024
    semantic_foreground = 512
    output_contract = {
        "alpha_in_range": True,
        "background_pixel_count": total - foreground,
        "contiguous": True,
        "cuda_resident": True,
        "dtypes_match": True,
        "finite_alpha": True,
        "finite_foreground_depth": True,
        "finite_rgb": True,
        "foreground_pixel_count": foreground,
        "output_devices": {name: "cuda:0" for name in OUTPUTS},
        "output_names_match": True,
        "semantic_background_pixel_count": total - semantic_foreground,
        "semantic_foreground_pixel_count": semantic_foreground,
        "shapes_match": True,
        "single_cuda_device": True,
        "valid": True,
        "valid_background_depth": True,
        "valid_background_semantics": True,
        "valid_foreground_semantics": True,
    }
    specs = {
        "rgb": (196_608, "torch.float32", [4, 64, 64, 3]),
        "alpha": (65_536, "torch.float32", [4, 64, 64, 1]),
        "depth": (65_536, "torch.float32", [4, 64, 64, 1]),
        "semantic_id": (131_072, "torch.int64", [4, 64, 64, 1]),
    }
    sources_root = root / "replay-native-sources"
    sources_root.mkdir(exist_ok=True)
    renderer = sources_root / "renderer.cpp"
    renderer_cuda = sources_root / "renderer_cuda.cu"
    renderer.write_text("// fixture renderer\n", encoding="utf-8")
    renderer_cuda.write_text("// fixture renderer cuda\n", encoding="utf-8")
    return {
        "arguments": dict(REPLAY_ARGUMENTS),
        "bitwise_equal": {name: True for name in OUTPUTS},
        "capacity_invariant_bitwise_equal": {name: True for name in OUTPUTS},
        "center_semantic": 1_000,
        "claim_scope": dict(REPLAY_CLAIM_SCOPE),
        "comparison_method": REPLAY_COMPARISON_METHOD,
        "counters": {
            "active_tiles": 4,
            "intersection_overflow": 0,
            "tile_intersections": 4_096,
            "visible_gaussians": 4_096,
            "visible_overflow": 0,
        },
        "equal_depth_gaussian_id_order": "ascending",
        "equal_depth_records_checked": 1_024,
        "fixture_sha256": "f" * 64,
        "native_build_contract": {
            "build_directory": str(native.parent.resolve()),
            "cuda_flags": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
            "cxx_flags": ["-O3", "-std=c++17"],
            "module_name": "isaacsim_gaussian_renderer_cuda_0123456789ab",
            "sources": [str(renderer.resolve()), str(renderer_cuda.resolve())],
            "torch_cuda_arch_list": "8.9",
        },
        "native_extension": artifact_record(native),
        "output_contract": output_contract,
        "output_digests": {
            name: {
                "bytes": byte_count,
                "dtype": dtype,
                "sha256": str(index + 1) * 64,
                "shape": shape,
            }
            for index, (name, (byte_count, dtype, shape)) in enumerate(specs.items())
        },
        "pass": True,
        "renderer_configuration": dict(REPLAY_RENDERER_CONFIGURATION),
        "schema_version": REPLAY_WORKER_SCHEMA,
        "source_smoke": artifact_record(source_smoke),
    }


def command_for_gate(gate_id: str, *, root: Path, summary: Path) -> list[str]:
    replay_root = root / "gates/deterministic-replay"
    commands = {
        "flashgs-b64-repair": ["python", "benchmarks/compare_flashgs_b64_repair_oracle.py"],
        "repair-compute-sanitizer": [
            "compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99",
            "python", "benchmarks/verify_flashgs_short_tail_synthetic.py",
        ],
        "python-unit-contract": ["python", "-m", "pytest", "tests"],
        "native-cuda-smoke": ["python", "scripts/custom_backend_smoke.py"],
        "native-determinism": ["python", "scripts/deterministic_cuda_smoke.py"],
        "native-multi-scene": ["python", "scripts/multi_scene_cuda_smoke.py"],
        "isaac-graphics": ["python", "scripts/run_isaac_graphics_gate.py"],
        "all-output-fidelity": [
            "python",
            "benchmarks/summarize_flashgs_matched.py",
            "--root",
            str(root),
            "--batches",
            "1,8,32,64,128,256,512,1024",
            "--output",
            str(summary),
            "--verify-existing",
        ],
        "custom-vectorization-ablation": ["python", "benchmarks/run_flashgs_matched.py"],
        "deterministic-replay": [
            "python",
            "run_gate_and_write_result.py",
            "--",
            "python",
            str((replay_root / "run_deterministic_replay.py").resolve()),
            "--python",
            "python",
            "--source-root",
            str(root.resolve()),
            "--worker-script",
            str((replay_root / "deterministic_digest_smoke.py").resolve()),
            "--source-smoke",
            str((replay_root / "deterministic_cuda_smoke.py").resolve()),
            "--output",
            str((replay_root / "raw-result.json").resolve()),
            "--batch",
            "4",
            "--gaussians",
            "1024",
            "--iterations",
            "32",
            "--width",
            "64",
            "--height",
            "64",
            "--tile-size",
            "1",
        ],
        "compute-sanitizer-memcheck": [
            "compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99",
            "python", "scripts/custom_backend_smoke.py",
        ],
        "bounded-stability-soak": [
            "python", "benchmarks/run_home_scan.py", "--duration-seconds", "600",
        ],
    }
    return commands[gate_id]


def normalized_result(
    gate_id: str,
    *,
    source: Path,
    native: Path,
    checks: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": GATE_RESULT_SCHEMA,
        "created_at": RESULT_TIME,
        "gate_id": gate_id,
        "pass": True,
        "identity": {
            "source_manifest_sha256": artifact_record(source)["sha256"],
            "native_extension_sha256": artifact_record(native)["sha256"],
            "gpu_uuid": GPU_UUID if GPU_REQUIRED[gate_id] else None,
            "scene_sha256": SCENE_SHA256,
        },
        "checks": checks,
        "evidence": evidence,
    }


def repair_evidence(
    root: Path,
    *,
    source: Path,
    native: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = root / "raw" / "b64-artifacts"
    source_record = artifact_record(source)
    native_record = artifact_record(native)

    def occupancy_payload(lane: str) -> dict[str, Any]:
        return {
            "schema_version": "flashgs-matched-node-occupancy-v2",
            "fixture_lane": lane,
            "expected_gpu_uuid": GPU_UUID,
            "executor_control": {
                "scope": "all-visible-gpus",
                "cooperative_node_wide_lock": {
                    "pass": True,
                    "lock_observed_held": True,
                },
            },
            "sampled_compute_process_telemetry": {
                "pass": True,
                "sample_count": 2,
            },
            "pass": True,
        }

    known_path = artifact_root / "known-failure-manifest.json"
    write_json(
        known_path,
        {
            "schema_version": "flashgs-b64-known-failure-cases-v1",
            "diagnostic_only": True,
            "batch": 64,
            "step": 107,
            "trajectory_id": "b64-diagnostic-fixture",
            "scene_sha256": SCENE_SHA256,
            "total_mismatched_pixels_across_regions": 92,
            "cases": [{"case_id": f"case-{index}"} for index in range(5)],
        },
    )
    known_record = artifact_record(known_path)
    diagnosis_lock_path = artifact_root / "diagnosis-lock.json"
    write_json(
        diagnosis_lock_path,
        {
            "schema_version": "flashgs-b64-diagnosis-lock-v1",
            "known_failure_manifest": known_record,
        },
    )
    diagnosis_lock_record = artifact_record(diagnosis_lock_path)
    diagnosis_index_path = artifact_root / "diagnosis-index.json"
    write_json(
        diagnosis_index_path,
        {
            "schema_version": "flashgs-b64-diagnosis-index-v1",
            "diagnostic_only": True,
            "pass": True,
            "case_count": 5,
            "known_failure_manifest": known_record,
            "diagnosis_lock": diagnosis_lock_record,
            "historical_mismatch_corpus": {"mismatch_count": 92},
        },
    )
    diagnosis_index_record = artifact_record(diagnosis_index_path)

    repair_records: dict[str, dict[str, Any]] = {}
    raw_records: dict[str, dict[str, Any]] = {}
    report_times = {
        "primary": "2025-12-31T23:59:53+00:00",
        "repeat": "2025-12-31T23:59:57+00:00",
    }
    for lane in ("primary", "repeat"):
        lane_root = artifact_root / lane
        full_path = lane_root / "full.npz"
        rgb_path = lane_root / "rgb-only.npz"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(f"{lane}-full-fixture".encode("utf-8"))
        rgb_path.write_bytes(f"{lane}-rgb-fixture".encode("utf-8"))
        full_record = artifact_record(full_path)
        rgb_record = artifact_record(rgb_path)
        embedded_full = {**full_record, "path": full_path.name}
        embedded_rgb = {**rgb_record, "path": rgb_path.name}
        report_path = lane_root / "repair-report.json"
        write_json(
            report_path,
            {
                "schema_version": "flashgs-b64-repair-verification-v1",
                "created_at": report_times[lane],
                "pass": True,
                "aggregate": {"case_count": 5, "passed_case_count": 5},
                "historical_mismatch_repair": {
                    "pixel_count": 92,
                    "passed_pixel_count": 92,
                },
                "rgb_only_specialization": {
                    "result": {"pixel_count": 92, "passed_pixel_count": 92}
                },
                "tool_integrity": {
                    "pass": True,
                    "checks": {
                        "zero_intersection_overflow_b64_and_trace_seeds": True,
                        "production_adapter_loaded": True,
                    },
                },
                "post_fix_raw_outputs": {
                    "full": embedded_full,
                    "rgb_only": embedded_rgb,
                },
                "known_failure_manifest": known_record,
                "diagnosis_index": diagnosis_index_record,
                "diagnosis_lock": diagnosis_lock_record,
                "scene": {"sha256": SCENE_SHA256},
                "environment": {"gpu_uuid": GPU_UUID},
                "camera": {"step": 107, "trajectory_id": "b64-diagnostic-fixture"},
                "equation": {"semantic_topology": "spatial-octants-8"},
                "source_provenance": {"manifest": source_record},
                "production_adapter": {"native_extension": native_record},
                "cases": [
                    {
                        "culprits": [
                            {
                                "trace_float32_contract": {
                                    "target_alpha_ulp_distance": 0,
                                    "target_power_ulp_distance": 0,
                                }
                            }
                        ]
                    }
                    for _ in range(5)
                ],
            },
        )
        repair_records[lane] = artifact_record(report_path)
        raw_records[f"{lane}_full"] = full_record
        raw_records[f"{lane}_rgb"] = rgb_record

    oracle_root = artifact_root / "oracle"
    oracle_path = oracle_root / "oracle.npz"
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_path.write_bytes(b"fixture-gsplat-oracle")
    oracle_record = artifact_record(oracle_path)
    camera_path = oracle_root / "camera-bundle.json"
    camera_payload = {
        "schema_version": "camera-bundle-v1",
        "coordinate_system": "opencv_world_to_camera",
        "width": 128,
        "height": 128,
        "background": [0.0, 0.0, 0.0],
        "color_space": "linear_rgb",
        "scene_checksum": SCENE_SHA256,
        "cameras": [{"view_id": f"t107-b{camera:04d}"} for camera in range(64)],
    }
    camera_payload["bundle_id"] = hashlib.sha256(
        json.dumps(camera_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(camera_path, camera_payload)
    camera_record = artifact_record(camera_path)
    oracle_occupancy_path = oracle_root / "node-occupancy.json"
    write_json(oracle_occupancy_path, occupancy_payload("oracle-inner"))
    oracle_occupancy_record = artifact_record(oracle_occupancy_path)
    oracle_manifest_path = oracle_root / "oracle-manifest.json"
    write_json(
        oracle_manifest_path,
        {
            "schema_version": "flashgs-matched-gsplat-oracle-v4",
            "pass": True,
            "output": str(oracle_path.resolve()),
            "output_sha256": oracle_record["sha256"],
            "camera_bundle": str(camera_path.resolve()),
            "camera_bundle_artifact": camera_record,
            "camera_bundle_id": camera_payload["bundle_id"],
            "node_occupancy": oracle_occupancy_record,
            "gpu_uuid": GPU_UUID,
            "scene_sha256": SCENE_SHA256,
            "camera_indices": list(range(64)),
            "steps": [107] * 64,
            "selection_pairs": [[107, camera] for camera in range(64)],
            "selection_profile": "diagnostic-single-step",
            "semantic_topology": "spatial-octants-8",
            "trajectory_id": "b64-diagnostic-fixture",
            "source_provenance": {"manifest": source_record},
        },
    )
    oracle_manifest_record = artifact_record(oracle_manifest_path)
    oracle_report_path = oracle_root / "comparison-report.json"
    write_json(
        oracle_report_path,
        {
            "schema_version": "flashgs-b64-repair-gsplat-all-pixel-v2",
            "pass": True,
            "checks": {
                "primary_full_sensor_all_pixels": True,
                "primary_rgb_only_all_pixels": True,
                "repeat_full_sensor_all_pixels": True,
                "repeat_rgb_only_all_pixels": True,
            },
            "acceptance_policy": {
                "all_required_oracle_comparisons_pass": True,
                "required_oracle_comparisons": {
                    "primary_full_sensor": True,
                    "primary_rgb_only": True,
                    "repeat_full_sensor": True,
                    "repeat_rgb_only": True,
                },
            },
            "contract": {
                "batch": 64,
                "width": 128,
                "height": 128,
                "step": 107,
                "camera_indices": list(range(64)),
                "selection_pairs": [[107, camera] for camera in range(64)],
                "semantic_topology": "spatial-octants-8",
                "gpu_uuid": GPU_UUID,
                "trajectory_id": "b64-diagnostic-fixture",
            },
            "input_artifacts": {
                "primary_repair_report": repair_records["primary"],
                "primary_repair_full_raw": raw_records["primary_full"],
                "primary_repair_rgb_only_raw": raw_records["primary_rgb"],
                "repeat_repair_report": repair_records["repeat"],
                "repeat_repair_full_raw": raw_records["repeat_full"],
                "repeat_repair_rgb_only_raw": raw_records["repeat_rgb"],
                "diagnosis_index": diagnosis_index_record,
                "diagnosis_lock": diagnosis_lock_record,
                "oracle": oracle_record,
                "oracle_manifest": oracle_manifest_record,
                "oracle_camera_bundle": camera_record,
            },
        },
    )
    oracle_report_record = artifact_record(oracle_report_path)

    evidence: dict[str, Any] = {
        "repair_report": repair_records["primary"],
        "repeat_repair_report": repair_records["repeat"],
        "primary_repair_full_raw": raw_records["primary_full"],
        "primary_repair_rgb_only_raw": raw_records["primary_rgb"],
        "repeat_repair_full_raw": raw_records["repeat_full"],
        "repeat_repair_rgb_only_raw": raw_records["repeat_rgb"],
        "oracle_report": oracle_report_record,
        "oracle": oracle_record,
        "oracle_manifest": oracle_manifest_record,
        "oracle_camera_bundle": camera_record,
        "oracle_node_occupancy": oracle_occupancy_record,
        "diagnosis_index": diagnosis_index_record,
        "diagnosis_lock": diagnosis_lock_record,
        "known_failure_manifest": known_record,
    }

    support_times = {
        "primary": ("2025-12-31T23:59:50+00:00", "2025-12-31T23:59:53+00:00"),
        "repeat": ("2025-12-31T23:59:54+00:00", "2025-12-31T23:59:57+00:00"),
        "oracle": ("2025-12-31T23:59:58+00:00", "2026-01-01T00:00:00+00:00"),
    }
    trajectory_path = root / "contracts" / "b64.json"
    for lane, gate_id, script_name, output_record in (
        ("primary", "support-b64-repair-primary", "verify_flashgs_b64_repair.py", repair_records["primary"]),
        ("repeat", "support-b64-repair-repeat", "verify_flashgs_b64_repair.py", repair_records["repeat"]),
        ("oracle", "support-b64-gsplat-oracle", "run_flashgs_gsplat_oracle.py", oracle_record),
    ):
        support_root = root / "raw" / f"support-{lane}"
        command_path = support_root / "command.json"
        argv = [
            "python",
            f"benchmarks/{script_name}",
            "--output",
            output_record["path"],
            "--expected-gpu-uuid",
            GPU_UUID,
            "--source-manifest",
            str(source.resolve()),
            "--trajectory",
            str(trajectory_path.resolve()),
            "--scene-path",
            str(root / "home-scene.ply"),
        ]
        if lane in {"primary", "repeat"}:
            argv.extend(
                [
                    "--scene-sha256",
                    SCENE_SHA256,
                    "--flashgs-adapter-attestation",
                    str(root / "flashgs-adapter-attestation.json"),
                    "--semantic-topology",
                    "spatial-octants-8",
                    "--intersection-capacity",
                    "1000000",
                    "--known-failure-manifest",
                    known_record["path"],
                    "--diagnosis-index",
                    diagnosis_index_record["path"],
                    "--diagnosis-lock",
                    diagnosis_lock_record["path"],
                    "--diagnosis-artifact-root",
                    str(artifact_root.resolve()),
                ]
            )
        else:
            argv.extend(
                [
                    "--step",
                    "107",
                    "--max-views",
                    "64",
                    "--semantic-topology",
                    "spatial-octants-8",
                    "--gsplat-source",
                    str(root / "gsplat-source"),
                    "--gsplat-build-attestation",
                    str(root / "gsplat-build-attestation.json"),
                ]
            )
        started_at, completed_at = support_times[lane]
        write_json(
            command_path,
            {
                "schema_version": "publication-command-record-v1",
                "gate_id": gate_id,
                "command": argv,
                "subcommands": [],
                "started_at": started_at,
                "timeout_seconds": 7200,
            },
        )
        log_path = support_root / "full.log"
        sentinel = (
            "FLASHGS_GSPLAT_ORACLE_OK "
            if lane == "oracle"
            else "FLASHGS_B64_REPAIR_VERIFICATION "
        )
        log_path.write_text(f"{sentinel}fixture\n", encoding="utf-8")
        exit_path = support_root / "exit.txt"
        exit_path.write_text("0\n", encoding="utf-8")
        occupancy_path = support_root / "occupancy.json"
        write_json(occupancy_path, occupancy_payload(f"{lane}-support"))
        wrapper_path = support_root / "wrapper.json"
        write_json(
            wrapper_path,
            {
                "schema_version": "publication-bounded-gate-wrapper-v1",
                "gate_id": gate_id,
                "started_at": started_at,
                "completed_at": completed_at,
                "timeout_seconds": 7200,
                "timed_out": False,
                "returncode": 0,
                "command_record": str(command_path.resolve()),
                "log": str(log_path.resolve()),
                "exit_status": str(exit_path.resolve()),
                "occupancy": str(occupancy_path.resolve()),
                "result": None,
                "pass": True,
            },
        )
        for suffix, path in (
            ("wrapper", wrapper_path),
            ("command", command_path),
            ("log", log_path),
            ("exit_status", exit_path),
            ("occupancy", occupancy_path),
        ):
            evidence[f"{lane}_support_{suffix}"] = artifact_record(path)

    checks = {
        "cases_expected": 5,
        "cases_passed": 5,
        "historical_pixels_expected": 92,
        "historical_pixels_passed": 92,
        "rgb_only_pixels_passed": 92,
        "output_contracts": ["full", "rgb"],
        "primary_full_views_passed": 64,
        "primary_rgb_views_passed": 64,
        "repeat_full_views_passed": 64,
        "repeat_rgb_views_passed": 64,
        "zero_overflow": True,
        "oracle_pinned": True,
        "max_intermediate_ulp": 0,
        "oracle_comparisons": {
            "primary_full": True,
            "primary_rgb": True,
            "repeat_full": True,
            "repeat_rgb": True,
        },
    }
    return checks, evidence


def create_ablation_evidence(
    root: Path,
    *,
    source: Path,
    native: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[list[str]]]:
    gate_root = root / "gates" / "custom-vectorization-ablation"
    source_record = artifact_record(source)
    native_record = artifact_record(native)
    capacity_records: dict[int, dict[str, Any]] = {}
    for physical in (1, 128):
        capacity_path = gate_root / f"capacity-p{physical}.json"
        write_json(
            capacity_path,
            {
                "schema_version": "flashgs-matched-capacity-calibration-v1",
                "pass": True,
                "renderer": "custom",
                "camera_contract": {
                    "batch": 128,
                    "timesteps": 108,
                    "measured_start": 8,
                },
                "capacity": {
                    "logical_batch": 128,
                    "physical_batch": physical,
                    "native_submissions_per_logical_batch": 128 // physical,
                    "intersection_capacity_scope": (
                        "physical-camera-chunk-reused-workspace"
                        if physical == 1
                        else "batch-global-workspace"
                    ),
                    "headroom": 1.05,
                    "verified_preflight": {
                        "max_visible_overflow": 0,
                        "max_intersection_overflow": 0,
                    },
                },
                "output_validation": {"pass": True},
                "environment": {
                    "gpu_uuid": GPU_UUID,
                    "native_extension": native_record,
                    "source_provenance": {"manifest": source_record},
                },
            },
        )
        capacity_records[physical] = artifact_record(capacity_path)

    entries: list[dict[str, Any]] = []
    subcommands: list[list[str]] = []
    by_id: dict[tuple[str, int, int], str] = {}
    values: dict[tuple[str, int, int], tuple[float, float]] = {}
    for contract in ("full", "rgb"):
        reference = json.loads((root / "runs" / "custom" / contract / "b128.json").read_text())
        reference_fidelity = json.loads((root / "fidelity" / "custom" / contract / "b128.json").read_text())
        for trial in (1, 2, 3):
            for physical in (1, 128):
                run_id = f"{contract}-trial{trial}-p{physical}"
                by_id[(contract, trial, physical)] = run_id
                occupancy_path = gate_root / "occupancy" / f"{run_id}.json"
                write_json(
                    occupancy_path,
                    {
                        "schema_version": "flashgs-matched-node-occupancy-v2",
                        "expected_gpu_uuid": GPU_UUID,
                        "pass": True,
                        "nonce": run_id,
                    },
                )
                gpu = (12.0 + trial if physical == 1 else 5.0 + trial)
                wall = gpu + 0.5
                values[(contract, trial, physical)] = gpu, wall
                run_path = gate_root / "runs" / f"{run_id}.json"
                run = {
                    **reference,
                    "headline_eligible": False,
                    "primary_workload_eligible": False,
                    "runner_config": {
                        **reference["runner_config"],
                        "independent_trial": trial,
                        "custom_max_physical_views": physical,
                    },
                    "camera_contract": {
                        **reference["camera_contract"],
                        "minimum_consecutive_viewmat_max_abs_delta": 0.001,
                        "camera_upload_seconds": 0.01 + trial / 1000.0,
                    },
                    "backend_execution": {
                        "logical_batch": 128,
                        "physical_batch": physical,
                        "native_submissions_per_logical_batch": 128 // physical,
                    },
                    "capacity": {
                        "calibration_artifact": capacity_records[physical],
                        "timed_verification": {
                            "observed": {
                                "measured_max_visible_overflow": 0,
                                "measured_max_intersection_overflow": 0,
                            }
                        },
                    },
                    "timing": {
                        "gpu_batch_ms": timing_distribution(gpu),
                        "synchronized_wall_batch_ms": timing_distribution(wall),
                        "host_submission_ms": timing_distribution(gpu / 10.0),
                    },
                    "environment": {
                        **reference["environment"],
                        "node_occupancy": artifact_record(occupancy_path),
                    },
                }
                write_json(run_path, run)
                run_record = artifact_record(run_path)
                fidelity_record = None
                if trial == 1:
                    fidelity_path = gate_root / "fidelity" / f"{run_id}.json"
                    write_json(
                        fidelity_path,
                        {
                            **reference_fidelity,
                            "input_artifacts": {
                                "run": run_record,
                                "oracle_manifest": reference_fidelity["input_artifacts"]["oracle_manifest"],
                            },
                        },
                    )
                    fidelity_record = artifact_record(fidelity_path)
                entries.append(
                    {
                        "run_id": run_id,
                        "contract": contract,
                        "trial": trial,
                        "physical_batch": physical,
                        "run": run_record,
                        "capacity": capacity_records[physical],
                        "fidelity": fidelity_record,
                    }
                )
                subcommands.append(
                    [
                        "python",
                        "benchmarks/run_flashgs_matched.py",
                        "--renderer",
                        "custom",
                        "--output-contract",
                        contract,
                        "--independent-trial",
                        str(trial),
                        "--custom-max-physical-views",
                        str(physical),
                    ]
                )

    order: list[str] = []
    for trial in (1, 2, 3):
        full_order = (1, 128) if trial % 2 else (128, 1)
        rgb_order = (128, 1) if trial % 2 else (1, 128)
        order.extend(by_id[("full", trial, physical)] for physical in full_order)
        order.extend(by_id[("rgb", trial, physical)] for physical in rgb_order)
    ratios: dict[str, list[dict[str, Any]]] = {"full": [], "rgb": []}
    for contract in ("full", "rgb"):
        for trial in (1, 2, 3):
            p1_gpu, p1_wall = values[(contract, trial, 1)]
            p128_gpu, p128_wall = values[(contract, trial, 128)]
            ratios[contract].append(
                {
                    "trial": trial,
                    "cuda_speedup_p128_over_p1": p1_gpu / p128_gpu,
                    "wall_speedup_p128_over_p1": p1_wall / p128_wall,
                }
            )
    raw_path = gate_root / "raw-result.json"
    write_json(
        raw_path,
        {
            "schema_version": "publication-custom-vectorization-ablation-v1",
            "pass": True,
            "batch": 128,
            "runs": entries,
            "run_order": order,
            "ratios": ratios,
        },
    )
    return (
        {
            "design_cells": 12,
            "fresh_processes": 12,
            "counterbalanced": True,
            "reported_without_win_requirement": True,
        },
        {"raw_result": artifact_record(raw_path)},
        subcommands,
    )


def create_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    source = root / "provenance/source-manifest.json"
    write_json(
        source,
        {
            "schema_version": "renderer-source-manifest-v2",
            "head": COMMIT,
            "dirty": False,
            "status_short": [],
        },
    )
    source_record = artifact_record(source)
    custom_native = (
        root
        / "custom-native-build"
        / "isaacsim_gaussian_renderer_cuda_0123456789ab.so"
    )
    custom_native.parent.mkdir()
    custom_native.write_bytes(b"publication-fixture-custom-native\0")
    flashgs_native = root / "flashgs-native-extension.so"
    flashgs_native.write_bytes(b"publication-fixture-flashgs-native\0")
    native_paths = {"custom": custom_native, "flashgs": flashgs_native}
    summary = root / "summary.json"
    rows_by_contract: dict[str, list[dict[str, Any]]] = {"full": [], "rgb": []}
    oracle_records: dict[str, dict[str, Any]] = {}
    for contract in ("full", "rgb"):
        oracle_path = root / "oracle" / f"b128-{contract}-manifest.json"
        write_json(
            oracle_path,
            {
                "schema_version": "flashgs-matched-gsplat-oracle-v4",
                "pass": True,
                "contract": contract,
                "batch": 128,
            },
        )
        oracle_records[contract] = artifact_record(oracle_path)
    trajectory_ids = {
        1: "a901ed9a3d7a62bffc2860ecefa2cdf8eaf2196085bca067215ac0469fbc8dfe",
        8: "39ab99be70fef112dd7f82636f28d61e2b67764b7f7f7eb6d1e6f401bddb3ca3",
        32: "86febe653241edbbd94d321ad22ed3214c6ff09130434a3c3f5c587107cd5d15",
        64: "e30050bf2d873825cf7cdebbe799c911ddaee5cc15d90e1c0dd8adc2c2e62cc0",
        128: "c9a7ef7727761865263c3432954b259c54e2065dfd2326e65494583083704925",
        256: "f0a68d425ca12f38b9d62c6fd445caae9d75472bdf9bacc29f563b93817233f4",
        512: "f750294341d523b1b46f626b609fbd5e2a0a43d28f0aa35a4614d6f0f6ed8c7d",
        1024: "375a6711e86333621448c94ca1ad7985bd2687c4b585f68c823624725c223910",
    }
    equation = {
        "gaussian_support_sigma": 3.33,
        "semantic_topology": "spatial-octants-8",
        "frustum_depth_interval": "inclusive-near-and-far",
        "invalid_covariance_2d": "determinant<=0",
        "projected_radius_covariance_diagonal_floor": None,
        "projected_radius_min_pixels": 0,
        "projected_radius_clip": "cull-only-when-both-radii<=0",
    }
    for contract in ("full", "rgb"):
        for batch in (1, 8, 32, 64, 128, 256, 512, 1024):
            artifacts: dict[str, Any] = {}
            timings: dict[str, dict[str, list[float]]] = {}
            means: dict[str, dict[str, float]] = {}
            for renderer, native in native_paths.items():
                base = float(batch) / (100.0 if renderer == "custom" else 50.0)
                gpu = base + (0.01 if contract == "rgb" else 0.02)
                wall = gpu + 0.5
                host = gpu / 10.0
                physical = (128 if batch in (512, 1024) else batch) if renderer == "custom" else 1
                submissions = batch // physical
                occupancy_path = root / "matrix-occupancy" / renderer / contract / f"b{batch}.json"
                write_json(occupancy_path, {"schema_version": "flashgs-matched-node-occupancy-v2", "pass": True, "expected_gpu_uuid": GPU_UUID})
                matrix_run = root / "runs" / renderer / contract / f"b{batch}.json"
                run = {
                    "schema_version": "flashgs-matched-renderer-run-v4",
                    "pass": True,
                    "headline_eligible": True,
                    "primary_workload_eligible": True,
                    "profile_control": False,
                    "renderer": renderer,
                    "output_contract": contract,
                    "scene": {"sha256": SCENE_SHA256},
                    "camera_contract": {
                        "batch": batch,
                        "trajectory_id": trajectory_ids[batch],
                        "width": 128,
                        "height": 128,
                        "warmup_frames": 8,
                        "measured_frames": 100,
                        "motion_classification": "independently-changing-camera-batch",
                        "unchanged_measured_camera_frame_pairs": 0,
                        "minimum_consecutive_viewmat_max_abs_delta": 0.001,
                        "camera_upload_seconds": 0.01,
                    },
                    "equation_contract": equation,
                    "runner_config": {"independent_trial": 1, "custom_max_physical_views": physical},
                    "backend_execution": {
                        "logical_batch": batch,
                        "physical_batch": physical,
                        "native_submissions_per_logical_batch": submissions,
                    },
                    "capacity": {
                        "logical_batch": batch,
                        "timed_verification": {
                            "observed": {
                                "measured_max_visible_overflow": 0,
                                "measured_max_intersection_overflow": 0,
                            }
                        }
                    },
                    "output_validation": output_validation(contract),
                    "timing": {
                        "gpu_batch_ms": timing_distribution(gpu),
                        "synchronized_wall_batch_ms": timing_distribution(wall),
                        "host_submission_ms": timing_distribution(host),
                    },
                    "environment": {
                        "compiler_versions": {
                            "cxx": "c++ (fixture) 11.4.0",
                            "nvcc": "Cuda compilation tools, release 12.9, V12.9.41",
                        },
                        "compute_capability": [8, 9],
                        "cuda_runtime": "12.8",
                        "driver": "580.159.03",
                        "gpu_name": "NVIDIA L4",
                        "gpu_uuid": GPU_UUID,
                        "native_extension": artifact_record(native),
                        "node_occupancy": artifact_record(occupancy_path),
                        "source_git_commit": COMMIT,
                        "source_provenance": {
                            "head": COMMIT,
                            "dirty": False,
                            "manifest": source_record,
                        },
                        "torch": "2.11.0+cu128",
                        "torch_cuda_arch_list": "8.9",
                    },
                }
                write_json(matrix_run, run)
                fidelity_path = root / "fidelity" / renderer / contract / f"b{batch}.json"
                write_json(
                    fidelity_path,
                    {
                        "schema_version": "flashgs-matched-fidelity-v4",
                        "pass": True,
                        "renderer": renderer,
                        "output_contract": contract,
                        "batch": batch,
                        "trajectory_id": trajectory_ids[batch],
                        "input_artifacts": {"oracle_manifest": oracle_records[contract]},
                    },
                )
                artifacts[f"{renderer}_run"] = artifact_record(matrix_run)
                artifacts[f"{renderer}_fidelity"] = artifact_record(fidelity_path)
                timings[renderer] = {
                    "gpu_batch_ms": [gpu] * 100,
                    "synchronized_wall_batch_ms": [wall] * 100,
                    "host_submission_ms": [host] * 100,
                }
                means[renderer] = {"gpu": gpu, "wall": wall, "host": host}
            rows_by_contract[contract].append(
                {
                    "batch": batch,
                    "pass": True,
                    "fidelity_pass": True,
                    "fairness_pass": True,
                    "fairness_failures": [],
                    "custom_ms": means["custom"]["gpu"],
                    "flashgs_ms": means["flashgs"]["gpu"],
                    "custom_synchronized_wall_ms": means["custom"]["wall"],
                    "flashgs_synchronized_wall_ms": means["flashgs"]["wall"],
                    "custom_host_submission_ms": means["custom"]["host"],
                    "flashgs_host_submission_ms": means["flashgs"]["host"],
                    "speedup_custom_over_flashgs": means["flashgs"]["gpu"] / means["custom"]["gpu"],
                    "gpu_speedup_ratio_of_mean_latency": means["flashgs"]["gpu"] / means["custom"]["gpu"],
                    "wall_speedup_ratio_of_mean_latency": means["flashgs"]["wall"] / means["custom"]["wall"],
                    "timing_samples": timings,
                    "artifacts": artifacts,
                }
            )
    write_json(
        summary,
        {
            "schema_version": "flashgs-matched-summary-v4",
            "pass": True,
            "scientific_pass": True,
            "headline_eligible": True,
            "primary_contract_eligible": True,
            "matrix_fairness_failures": [],
            "hardware_scope": {"gpu_name": "NVIDIA L4", "gpu_uuid": GPU_UUID},
            "source_identity": {
                "head": COMMIT,
                "manifest": source_record,
            },
            "scene_sha256": SCENE_SHA256,
            "primary_full_sensor_dynamic_table": rows_by_contract["full"],
            "rgb_only_dynamic_table": rows_by_contract["rgb"],
        },
    )
    invocation = root / "matrix-invocation.json"
    write_json(
        invocation,
        {
            "schema_version": "flashgs-matched-matrix-invocation-v1",
            "argv": ["python", "benchmarks/run_flashgs_matched_matrix.py"],
            "parsed_arguments": {
                "expected_gpu_uuid": GPU_UUID,
                "batches": "1,8,32,64,128,256,512,1024",
                "custom_chunked_physical_views": 128,
                "allow_nonheadline_gpu": False,
            },
        },
    )
    occupancy = root / "occupancy.json"
    write_json(
        occupancy,
        {
            "schema_version": "flashgs-matched-node-occupancy-v2",
            "expected_gpu_uuid": GPU_UUID,
            "executor_control": {
                "scope": "all-visible-gpus",
                "cooperative_node_wide_lock": {
                    "pass": True,
                    "lock_observed_held": True,
                },
            },
            "sampled_compute_process_telemetry": {
                "pass": True,
                "sample_count": 2,
            },
            "pass": True,
        },
    )

    gate_specs: list[dict[str, Any]] = []
    for gate_id in REQUIRED_GATE_IDS:
        native = native_paths[NATIVE_ROLE[gate_id]]
        gate_root = root / "gates" / gate_id
        gate_root.mkdir(parents=True)
        raw_result = gate_root / "raw-result.json"
        checks: dict[str, Any]
        evidence: dict[str, Any]
        log_text = f"{gate_id} fixture log\n"
        if gate_id == "flashgs-b64-repair":
            checks, evidence = repair_evidence(
                gate_root,
                source=source,
                native=native,
            )
        elif gate_id in {"repair-compute-sanitizer", "compute-sanitizer-memcheck"}:
            if gate_id == "repair-compute-sanitizer":
                write_json(
                    raw_result,
                    {
                        "schema_version": "flashgs-short-tail-synthetic-verification-v1",
                        "pass": True,
                        "checks": {
                            "full_sensor_matches_cpu_oracle": True,
                            "rgb_only_matches_cpu_oracle": True,
                            "rgb_specializations_bitwise_equal": True,
                            "workspace_layouts_equal": True,
                            "output_contracts_valid": True,
                            "occupancy_pass": True,
                        },
                        "specializations": {
                            "full_sensor": {
                                "counters": {"visible_overflow": 0, "intersection_overflow": 0},
                                "repeat_bitwise_equal": {name: True for name in OUTPUTS},
                            },
                            "rgb_only": {
                                "counters": {"visible_overflow": 0, "intersection_overflow": 0},
                                "repeat_bitwise_equal": {"rgb": True},
                            },
                        },
                    },
                )
            else:
                payload = native_payload("native-cuda-smoke")
                write_json(raw_result, payload)
                log_text += "CUSTOM_CUDA_BACKEND_SMOKE_OK " + repr(payload) + "\n"
            checks = {
                "tool": "compute-sanitizer",
                "error_summary": 0,
                "application_exit_code": 0,
                "bounded": True,
            }
            evidence = {"target_result": artifact_record(raw_result)}
            log_text += "========= ERROR SUMMARY: 0 errors\nCOMPUTE_SANITIZER_OK\n"
        elif gate_id == "python-unit-contract":
            junit = gate_root / "junit.xml"
            junit.write_text(
                '<testsuite name="publication" tests="2" failures="0" errors="0" skipped="0">'
                '<testcase classname="x" name="a"/><testcase classname="x" name="b"/></testsuite>\n',
                encoding="utf-8",
            )
            checks = {
                "collected": 2,
                "passed": 2,
                "failures": 0,
                "errors": 0,
                "skipped_required": 0,
                "required_tests": ["x::a", "x::b"],
            }
            evidence = {"junit_xml": artifact_record(junit)}
            log_text += "2 passed in 0.01s\nPYTEST_PUBLICATION_OK\n"
        elif gate_id in NATIVE_SENTINELS:
            payload = native_payload(gate_id)
            write_json(raw_result, payload)
            checks = {
                "sentinel": NATIVE_SENTINELS[gate_id],
                "json_pass": True,
                "all_outputs_valid": True,
                "zero_overflow": True,
                "cuda_resident": True,
                "finite_values": True,
            }
            if gate_id == "native-determinism":
                checks["bitwise_equal"] = True
            if gate_id == "native-multi-scene":
                checks["scene_count"] = 2
            evidence = {"raw_result": artifact_record(raw_result)}
            encoded = json.dumps(payload, sort_keys=True) if gate_id == "native-determinism" else repr(payload)
            log_text += NATIVE_SENTINELS[gate_id] + " " + encoded + "\n"
        elif gate_id == "isaac-graphics":
            lanes = [
                {
                    "name": name,
                    "process": {"returncode": 0, "timed_out": False},
                    "log_acceptance": {"returncode": 0},
                    "residual_process_acceptance": {"pass": True},
                    "gpu_foundation": {"active_rows": [{"index": "0", "name": "NVIDIA L4"}]},
                }
                for name in ISAAC_LANES
            ]
            write_json(
                raw_result,
                {
                    "schema_version": "isaac-graphics-gate/v1",
                    "pass": True,
                    "source_unchanged": True,
                    "runtime_unchanged": True,
                    "residual_process_acceptance": {"pass": True},
                    "processes_before": {"gpu_compute": {"empty": True}},
                    "lanes": lanes,
                },
            )
            checks = {
                "source_unchanged": True,
                "runtime_unchanged": True,
                "simulation_app_closed": True,
                "all_outputs_valid": True,
                "zero_overflow": True,
                "residual_process_count": 0,
                "active_gpu_count": 1,
                "lanes": ISAAC_LANES,
            }
            evidence = {"isaac_summary": artifact_record(raw_result)}
            log_text += "\n".join((*ISAAC_LANES.values(), "ISAAC_GRAPHICS_GATE_OK")) + "\n"
        elif gate_id == "all-output-fidelity":
            checks = {
                "output_contracts": list(OUTPUTS),
                "all_views_pass": True,
                "finite_values": True,
                "zero_overflow": True,
            }
            evidence = {"raw_result": artifact_record(summary)}
        elif gate_id == "custom-vectorization-ablation":
            checks, evidence, ablation_subcommands = create_ablation_evidence(
                root,
                source=source,
                native=custom_native,
            )
        elif gate_id == "deterministic-replay":
            runner = gate_root / "run_deterministic_replay.py"
            worker = gate_root / "deterministic_digest_smoke.py"
            source_smoke = gate_root / "deterministic_cuda_smoke.py"
            runner.write_text("# fixture replay runner\n", encoding="utf-8")
            worker.write_text("# fixture replay worker\n", encoding="utf-8")
            source_smoke.write_text("# frozen deterministic smoke\n", encoding="utf-8")
            payload = replay_worker_payload(
                root=root,
                native=custom_native,
                source_smoke=source_smoke,
            )
            process_paths = [
                gate_root / "deterministic-replay.process-1.json",
                gate_root / "deterministic-replay.process-2.json",
            ]
            for path in process_paths:
                write_json(path, payload)
            child_command = [
                "python",
                str(worker.resolve()),
                "--source-smoke",
                str(source_smoke.resolve()),
                "--batch",
                "4",
                "--gaussians",
                "1024",
                "--iterations",
                "32",
                "--width",
                "64",
                "--height",
                "64",
                "--tile-size",
                "1",
            ]
            raw_replay = {
                "claim_scope": dict(REPLAY_CLAIM_SCOPE),
                "comparison_method": REPLAY_COMPARISON_METHOD,
                "cross_process_output_hashes_equal": True,
                "fixture_sha256": payload["fixture_sha256"],
                "fresh_processes": 2,
                "native_build_contract": payload["native_build_contract"],
                "native_extension": payload["native_extension"],
                "output_contract": payload["output_contract"],
                "output_digests": payload["output_digests"],
                "pass": True,
                "process_payloads_equal": True,
                "processes": [
                    {
                        "command": child_command,
                        "index": index,
                        "payload": artifact_record(path),
                        "pid": 1_000 + index,
                        "returncode": 0,
                    }
                    for index, path in enumerate(process_paths, start=1)
                ],
                "replays_per_process": 32,
                "renderer_configuration": dict(REPLAY_RENDERER_CONFIGURATION),
                "runner_script": artifact_record(runner),
                "schema_version": REPLAY_SCHEMA,
                "source_smoke": artifact_record(source_smoke),
                "within_process_bitwise_equal": True,
                "worker_script": artifact_record(worker),
                "zero_overflow": True,
            }
            write_json(raw_result, raw_replay)
            checks = {
                "output_contracts": list(OUTPUTS),
                "fresh_processes": 2,
                "replays_per_process": 32,
                "bitwise_equal": True,
                "cross_process_output_hashes_equal": True,
                "zero_overflow": True,
            }
            evidence = {
                "process_1": artifact_record(process_paths[0]),
                "process_2": artifact_record(process_paths[1]),
                "raw_result": artifact_record(raw_result),
                "runner_script": artifact_record(runner),
                "source_smoke": artifact_record(source_smoke),
                "worker_script": artifact_record(worker),
            }
            encoded_worker = json.dumps(payload, sort_keys=True)
            encoded_parent = json.dumps(raw_replay, sort_keys=True)
            log_text += (
                "DETERMINISTIC_DIGEST_SMOKE_OK " + encoded_worker + "\n"
                "DETERMINISTIC_DIGEST_SMOKE_OK " + encoded_worker + "\n"
                "DETERMINISTIC_CROSS_PROCESS_REPLAY_OK " + encoded_parent + "\n"
            )
        elif gate_id == "bounded-stability-soak":
            write_json(
                raw_result,
                {
                    "schema_version": "real-scene-custom-benchmark/v1",
                    "pass": True,
                    "measurement_mode": "duration_soak",
                    "batch": 128,
                    "width": 128,
                    "height": 128,
                    "dataset": {"sha256": SCENE_SHA256},
                    "requested_duration_seconds": 600.0,
                    "duration_seconds": 601.0,
                    "zero_overflow": True,
                    "all_output_checks_valid": True,
                    "outputs_gpu_resident": True,
                    "outputs_contract_valid": True,
                    "allocation_delta_bytes": 0,
                    "measured_counters": {"visible_overflow": 0, "intersection_overflow": 0},
                },
            )
            checks = {
                "requested_duration_seconds": 600.0,
                "duration_seconds": 601.0,
                "output_contracts": list(OUTPUTS),
                "zero_overflow": True,
                "all_outputs_valid": True,
                "outputs_gpu_resident": True,
                "no_cuda_errors": True,
            }
            evidence = {"soak_result": artifact_record(raw_result)}
            log_text += "CUSTOM_CUDA_SOAK_OK\n"
        else:  # pragma: no cover
            raise AssertionError(gate_id)

        result_path = gate_root / "result.json"
        write_json(
            result_path,
            normalized_result(
                gate_id,
                source=source,
                native=native,
                checks=checks,
                evidence=evidence,
            ),
        )
        command = gate_root / "command.json"
        write_json(
            command,
            {
                "schema_version": "publication-command-record-v1",
                "gate_id": gate_id,
                "command": command_for_gate(gate_id, root=root, summary=summary),
                "subcommands": ablation_subcommands if gate_id == "custom-vectorization-ablation" else [],
                "started_at": STARTED,
                "timeout_seconds": 1800 if gate_id == "bounded-stability-soak" else 300,
            },
        )
        log = gate_root / "full.log"
        log.write_text(log_text, encoding="utf-8")
        exit_status = gate_root / "exit.txt"
        exit_status.write_text("0\n", encoding="utf-8")
        gate_specs.append(
            {
                "gate_id": gate_id,
                "result": str(result_path),
                "command": str(command),
                "log": str(log),
                "exit_status": str(exit_status),
                "source_manifest": str(source),
                "native_extension": str(native),
                "gpu_uuid": GPU_UUID if GPU_REQUIRED[gate_id] else None,
                "occupancy": str(occupancy) if GPU_REQUIRED[gate_id] else None,
                "started_at": STARTED,
                "completed_at": COMPLETED,
            }
        )

    spec = root / "verification-spec.json"
    write_json(
        spec,
        {
            "schema_version": SPEC_SCHEMA,
            "benchmark": {
                "commit": COMMIT,
                "tag": "fixture-benchmark-v1",
                "repository_root": str(root),
                "source_manifest": str(source),
            },
            "matrix": {
                "summary": str(summary),
                "invocation": str(invocation),
                "custom_native_extension": str(custom_native),
                "flashgs_native_extension": str(flashgs_native),
                "gpu_uuid": GPU_UUID,
                "scene_sha256": SCENE_SHA256,
            },
            "gates": gate_specs,
        },
    )
    canonical_ablation = root / ABLATION_LOGICAL_PATH
    canonical_ablation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        root / "gates" / "custom-vectorization-ablation" / "raw-result.json",
        canonical_ablation,
    )
    machine_path = root / "publication/machine-provenance.json"
    write_json(
        machine_path,
        build_machine_provenance(root, expected_gpu_uuid=GPU_UUID),
    )
    output = root / "publication" / "verification.json"
    write_verification_from_spec(spec, output, verify_tag=False)
    return output


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def refresh(record: dict[str, Any], path: Path) -> None:
    record.update(artifact_record(path))


def iter_records(value: Any):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        for child in value.values():
            yield from iter_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_records(child)


def relocate(verification: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    root_record = artifact_record(verification)
    pending: list[tuple[dict[str, Any], Path | None]] = [(root_record, None)]
    copied: set[str] = set()
    while pending:
        record, relative_to = pending.pop()
        sha256 = record["sha256"]
        if sha256 in copied:
            continue
        source = Path(record["path"])
        if not source.is_absolute() and relative_to is not None:
            source = relative_to / source
        target = destination / "objects" / "sha256" / sha256[:2] / sha256
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.add(sha256)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        pending.extend((nested, source.parent) for nested in iter_records(payload))
    ablation_source = verification.parent.parent / ABLATION_LOGICAL_PATH
    ablation_record = artifact_record(ablation_source)
    machine_source = verification.parent.parent / MACHINE_PROVENANCE_LOGICAL_PATH
    machine_record = artifact_record(machine_source)
    machine = load(machine_source)
    machine_target = (
        destination
        / "objects"
        / "sha256"
        / machine_record["sha256"][:2]
        / machine_record["sha256"]
    )
    machine_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(machine_source, machine_target)
    machine_logical_artifacts = [machine["source_manifest"], *machine["primary_runs"]]
    write_json(
        destination / "manifest.json",
        {
            "schema_version": "flashgs-matched-evidence-bundle-v1",
            "publication_artifacts": [
                {
                    **root_record,
                    "logical_path": "publication/verification.json",
                },
                {
                    **ablation_record,
                    "logical_path": ABLATION_LOGICAL_PATH,
                },
                {
                    **machine_record,
                    "logical_path": MACHINE_PROVENANCE_LOGICAL_PATH,
                },
                *machine_logical_artifacts,
            ],
        },
    )


class AggregateVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "stage"
        self.verification = create_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_rejected(self) -> None:
        with self.assertRaises(VerificationError):
            verify_aggregate_verification(self.root)

    def refresh_summary(self, record: dict[str, Any], path: Path) -> None:
        refresh(record["matrix"]["summary"], path)
        fidelity_gate = next(item for item in record["gates"] if item["gate_id"] == "all-output-fidelity")
        result_path = Path(fidelity_gate["result"]["path"])
        result = load(result_path)
        refresh(result["evidence"]["raw_result"], path)
        write_json(result_path, result)
        refresh(fidelity_gate["result"], result_path)

    def b64_graph(self) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "flashgs-b64-repair")
        result_path = Path(gate["result"]["path"])
        return record, gate, result_path, load(result_path)

    def save_b64_graph(
        self,
        record: dict[str, Any],
        gate: dict[str, Any],
        result_path: Path,
        result: dict[str, Any],
    ) -> None:
        write_json(result_path, result)
        refresh(gate["result"], result_path)
        write_json(self.verification, record)

    def replay_graph(
        self,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        Path,
        dict[str, Any],
        Path,
        dict[str, Any],
    ]:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "deterministic-replay")
        result_path = Path(gate["result"]["path"])
        result = load(result_path)
        raw_path = Path(result["evidence"]["raw_result"]["path"])
        return record, gate, result_path, result, raw_path, load(raw_path)

    def save_replay_graph(
        self,
        record: dict[str, Any],
        gate: dict[str, Any],
        result_path: Path,
        result: dict[str, Any],
        raw_path: Path,
        raw: dict[str, Any],
        *,
        rewrite_log: bool = True,
    ) -> None:
        write_json(raw_path, raw)
        refresh(result["evidence"]["raw_result"], raw_path)
        if rewrite_log:
            process_payloads = [
                load(Path(result["evidence"][f"process_{index}"]["path"]))
                for index in (1, 2)
            ]
            log_path = Path(gate["log"]["path"])
            log_path.write_text(
                "deterministic-replay fixture log\n"
                + "".join(
                    "DETERMINISTIC_DIGEST_SMOKE_OK "
                    + json.dumps(payload, sort_keys=True)
                    + "\n"
                    for payload in process_payloads
                )
                + "DETERMINISTIC_CROSS_PROCESS_REPLAY_OK "
                + json.dumps(raw, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            refresh(gate["log"], log_path)
        write_json(result_path, result)
        refresh(gate["result"], result_path)
        write_json(self.verification, record)

    def test_positive(self) -> None:
        receipt = verify_aggregate_verification(self.root)
        self.assertTrue(receipt["pass"])
        self.assertEqual(receipt["gate_ids"], list(REQUIRED_GATE_IDS))
        self.assertEqual(
            receipt["machine_provenance"]["logical_path"],
            MACHINE_PROVENANCE_LOGICAL_PATH,
        )

    def test_machine_provenance_mutation_fails(self) -> None:
        path = self.root / MACHINE_PROVENANCE_LOGICAL_PATH
        machine = load(path)
        machine["driver"] = "substituted-driver"
        write_json(path, machine)
        self.assert_rejected()

    def test_relocated_object_store_positive(self) -> None:
        bundle = Path(self.temporary.name) / "bundle"
        relocate(self.verification, bundle)
        receipt = verify_aggregate_verification(bundle)
        self.assertTrue(receipt["pass"])
        self.assertEqual(receipt["root_mode"], "bundle")

    def test_replay_missing_process_evidence_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result, raw_path, raw = self.replay_graph()
        result["evidence"].pop("process_2")
        self.save_replay_graph(
            record,
            gate,
            result_path,
            result,
            raw_path,
            raw,
            rewrite_log=False,
        )
        self.assert_rejected()

    def test_replay_process_digest_divergence_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result, raw_path, raw = self.replay_graph()
        second_path = Path(result["evidence"]["process_2"]["path"])
        second = load(second_path)
        second["output_digests"]["rgb"]["sha256"] = "e" * 64
        write_json(second_path, second)
        refresh(result["evidence"]["process_2"], second_path)
        raw["processes"][1]["payload"] = result["evidence"]["process_2"]
        self.save_replay_graph(record, gate, result_path, result, raw_path, raw)
        self.assert_rejected()

    def test_replay_native_substitution_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result, raw_path, raw = self.replay_graph()
        module_name = "isaacsim_gaussian_renderer_cuda_0123456789ab"
        alternate = self.root / "alternate-native-build" / f"{module_name}.so"
        alternate.parent.mkdir()
        alternate.write_bytes(b"substituted native extension\0")
        alternate_record = artifact_record(alternate)
        for index in (1, 2):
            process_path = Path(result["evidence"][f"process_{index}"]["path"])
            payload = load(process_path)
            payload["native_extension"] = alternate_record
            payload["native_build_contract"]["build_directory"] = str(
                alternate.parent.resolve()
            )
            write_json(process_path, payload)
            refresh(result["evidence"][f"process_{index}"], process_path)
            raw["processes"][index - 1]["payload"] = result["evidence"][
                f"process_{index}"
            ]
        raw["native_extension"] = alternate_record
        raw["native_build_contract"]["build_directory"] = str(
            alternate.parent.resolve()
        )
        self.save_replay_graph(record, gate, result_path, result, raw_path, raw)
        self.assert_rejected()

    def test_replay_inert_bound_runner_token_fails_with_refreshed_command(self) -> None:
        record, gate, _result_path, result, _raw_path, _raw = self.replay_graph()
        command_path = Path(gate["command"]["path"])
        command = load(command_path)
        separator = command["command"].index("--")
        runner_path = result["evidence"]["runner_script"]["path"]
        command["command"][separator + 2] = "other_replay_program.py"
        command["command"].append(runner_path)
        write_json(command_path, command)
        refresh(gate["command"], command_path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_replay_output_contract_tamper_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result, raw_path, raw = self.replay_graph()
        for index in (1, 2):
            process_path = Path(result["evidence"][f"process_{index}"]["path"])
            payload = load(process_path)
            payload["output_contract"]["finite_rgb"] = False
            write_json(process_path, payload)
            refresh(result["evidence"][f"process_{index}"], process_path)
            raw["processes"][index - 1]["payload"] = result["evidence"][
                f"process_{index}"
            ]
        raw["output_contract"]["finite_rgb"] = False
        self.save_replay_graph(record, gate, result_path, result, raw_path, raw)
        self.assert_rejected()

    def test_b64_missing_transitive_evidence_key_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result = self.b64_graph()
        result["evidence"].pop("repeat_support_log")
        self.save_b64_graph(record, gate, result_path, result)
        self.assert_rejected()

    def test_b64_missing_raw_output_fails(self) -> None:
        _record, _gate, _result_path, result = self.b64_graph()
        Path(result["evidence"]["primary_repair_full_raw"]["path"]).unlink()
        self.assert_rejected()

    def test_b64_report_relative_escape_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result = self.b64_graph()
        evidence = result["evidence"]
        report_path = Path(evidence["repair_report"]["path"])
        report = load(report_path)
        report["post_fix_raw_outputs"]["full"]["path"] = "../full.npz"
        write_json(report_path, report)
        refresh(evidence["repair_report"], report_path)
        oracle_report_path = Path(evidence["oracle_report"]["path"])
        oracle_report = load(oracle_report_path)
        oracle_report["input_artifacts"]["primary_repair_report"] = evidence["repair_report"]
        write_json(oracle_report_path, oracle_report)
        refresh(evidence["oracle_report"], oracle_report_path)
        self.save_b64_graph(record, gate, result_path, result)
        self.assert_rejected()

    def test_b64_oracle_input_substitution_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result = self.b64_graph()
        evidence = result["evidence"]
        oracle_report_path = Path(evidence["oracle_report"]["path"])
        oracle_report = load(oracle_report_path)
        oracle_report["input_artifacts"]["primary_repair_full_raw"] = evidence[
            "repeat_repair_full_raw"
        ]
        write_json(oracle_report_path, oracle_report)
        refresh(evidence["oracle_report"], oracle_report_path)
        self.save_b64_graph(record, gate, result_path, result)
        self.assert_rejected()

    def test_b64_oracle_inner_occupancy_substitution_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result = self.b64_graph()
        evidence = result["evidence"]
        manifest_path = Path(evidence["oracle_manifest"]["path"])
        manifest = load(manifest_path)
        manifest["node_occupancy"] = evidence["oracle_support_occupancy"]
        write_json(manifest_path, manifest)
        refresh(evidence["oracle_manifest"], manifest_path)
        oracle_report_path = Path(evidence["oracle_report"]["path"])
        oracle_report = load(oracle_report_path)
        oracle_report["input_artifacts"]["oracle_manifest"] = evidence["oracle_manifest"]
        write_json(oracle_report_path, oracle_report)
        refresh(evidence["oracle_report"], oracle_report_path)
        self.save_b64_graph(record, gate, result_path, result)
        self.assert_rejected()

    def test_b64_support_command_output_substitution_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result = self.b64_graph()
        evidence = result["evidence"]
        command_path = Path(evidence["primary_support_command"]["path"])
        command = load(command_path)
        output_index = command["command"].index("--output") + 1
        command["command"][output_index] = evidence["repeat_repair_report"]["path"]
        write_json(command_path, command)
        refresh(evidence["primary_support_command"], command_path)
        self.save_b64_graph(record, gate, result_path, result)
        self.assert_rejected()

    def test_b64_support_wrapper_binding_substitution_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result = self.b64_graph()
        evidence = result["evidence"]
        wrapper_path = Path(evidence["primary_support_wrapper"]["path"])
        wrapper = load(wrapper_path)
        wrapper["command_record"] = evidence["repeat_support_command"]["path"]
        write_json(wrapper_path, wrapper)
        refresh(evidence["primary_support_wrapper"], wrapper_path)
        self.save_b64_graph(record, gate, result_path, result)
        self.assert_rejected()

    def test_b64_support_nonzero_exit_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result = self.b64_graph()
        evidence = result["evidence"]
        exit_path = Path(evidence["repeat_support_exit_status"]["path"])
        exit_path.write_text("1\n", encoding="utf-8")
        refresh(evidence["repeat_support_exit_status"], exit_path)
        self.save_b64_graph(record, gate, result_path, result)
        self.assert_rejected()

    def test_b64_support_success_log_tamper_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result = self.b64_graph()
        evidence = result["evidence"]
        log_path = Path(evidence["oracle_support_log"]["path"])
        log_path.write_text("oracle fixture without success record\n", encoding="utf-8")
        refresh(evidence["oracle_support_log"], log_path)
        self.save_b64_graph(record, gate, result_path, result)
        self.assert_rejected()

    def test_b64_support_occupancy_tamper_fails_with_refreshed_graph(self) -> None:
        record, gate, result_path, result = self.b64_graph()
        evidence = result["evidence"]
        occupancy_path = Path(evidence["primary_support_occupancy"]["path"])
        occupancy = load(occupancy_path)
        occupancy["pass"] = False
        write_json(occupancy_path, occupancy)
        refresh(evidence["primary_support_occupancy"], occupancy_path)
        self.save_b64_graph(record, gate, result_path, result)
        self.assert_rejected()

    def test_missing_gate_fails(self) -> None:
        record = load(self.verification)
        record["gates"].pop()
        write_json(self.verification, record)
        self.assert_rejected()

    def test_pass_only_fake_fails(self) -> None:
        record = load(self.verification)
        gate = record["gates"][0]
        path = Path(gate["result"]["path"])
        write_json(path, {"pass": True})
        refresh(gate["result"], path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_wrong_gpu_fails(self) -> None:
        record = load(self.verification)
        record["gates"][0]["gpu_uuid"] = "GPU-wrong"
        write_json(self.verification, record)
        self.assert_rejected()

    def test_logical_verification_symlink_escape_fails(self) -> None:
        outside = self.root.parent / "outside-verification.json"
        self.verification.replace(outside)
        self.verification.symlink_to(outside)
        self.assert_rejected()

    def test_placeholder_command_fails_even_with_refreshed_hash(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "native-cuda-smoke")
        path = Path(gate["command"]["path"])
        command = load(path)
        command["command"] = ["fixture", "native-cuda-smoke"]
        write_json(path, command)
        refresh(gate["command"], path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_fidelity_reconstruction_without_verify_existing_fails(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "all-output-fidelity")
        path = Path(gate["command"]["path"])
        command = load(path)
        command["command"].remove("--verify-existing")
        write_json(path, command)
        refresh(gate["command"], path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_fidelity_reconstruction_wrong_root_fails(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "all-output-fidelity")
        path = Path(gate["command"]["path"])
        command = load(path)
        root_index = command["command"].index("--root") + 1
        command["command"][root_index] = str(self.root / "other-result-root")
        write_json(path, command)
        refresh(gate["command"], path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_matrix_run_reuse_fails_even_with_refreshed_graph(self) -> None:
        record = load(self.verification)
        summary_path = Path(record["matrix"]["summary"]["path"])
        summary = load(summary_path)
        rows = summary["primary_full_sensor_dynamic_table"]
        rows[1]["artifacts"]["custom_run"] = rows[0]["artifacts"]["custom_run"]
        write_json(summary_path, summary)
        self.refresh_summary(record, summary_path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_matrix_timing_fabrication_fails_even_with_refreshed_graph(self) -> None:
        record = load(self.verification)
        summary_path = Path(record["matrix"]["summary"]["path"])
        summary = load(summary_path)
        summary["rgb_only_dynamic_table"][4]["custom_ms"] = 0.000001
        write_json(summary_path, summary)
        self.refresh_summary(record, summary_path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_native_log_payload_mismatch_fails(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "native-determinism")
        result_path = Path(gate["result"]["path"])
        result = load(result_path)
        raw_path = Path(result["evidence"]["raw_result"]["path"])
        raw = load(raw_path)
        raw["bitwise_equal"]["rgb"] = False
        write_json(raw_path, raw)
        refresh(result["evidence"]["raw_result"], raw_path)
        write_json(result_path, result)
        refresh(gate["result"], result_path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_ablation_ratio_fabrication_fails(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "custom-vectorization-ablation")
        result_path = Path(gate["result"]["path"])
        result = load(result_path)
        raw_path = Path(result["evidence"]["raw_result"]["path"])
        raw = load(raw_path)
        raw["ratios"]["full"][0]["cuda_speedup_p128_over_p1"] = 999.0
        write_json(raw_path, raw)
        refresh(result["evidence"]["raw_result"], raw_path)
        write_json(result_path, result)
        refresh(gate["result"], result_path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_ablation_missing_design_cell_fails_with_refreshed_graph(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "custom-vectorization-ablation")
        result_path = Path(gate["result"]["path"])
        result = load(result_path)
        raw_path = Path(result["evidence"]["raw_result"]["path"])
        raw = load(raw_path)
        removed = raw["runs"].pop()["run_id"]
        raw["run_order"].remove(removed)
        write_json(raw_path, raw)
        shutil.copyfile(raw_path, self.root / ABLATION_LOGICAL_PATH)
        refresh(result["evidence"]["raw_result"], raw_path)
        write_json(result_path, result)
        refresh(gate["result"], result_path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_ablation_wrong_physical_capacity_fails_with_refreshed_graph(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "custom-vectorization-ablation")
        result_path = Path(gate["result"]["path"])
        result = load(result_path)
        raw_path = Path(result["evidence"]["raw_result"]["path"])
        raw = load(raw_path)
        entry = next(item for item in raw["runs"] if item["physical_batch"] == 1)
        capacity_path = Path(entry["capacity"]["path"])
        capacity = load(capacity_path)
        capacity["capacity"]["physical_batch"] = 128
        write_json(capacity_path, capacity)
        refresh(entry["capacity"], capacity_path)
        run_path = Path(entry["run"]["path"])
        run = load(run_path)
        refresh(run["capacity"]["calibration_artifact"], capacity_path)
        write_json(run_path, run)
        refresh(entry["run"], run_path)
        write_json(raw_path, raw)
        shutil.copyfile(raw_path, self.root / ABLATION_LOGICAL_PATH)
        refresh(result["evidence"]["raw_result"], raw_path)
        write_json(result_path, result)
        refresh(gate["result"], result_path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_canonical_ablation_substitution_fails(self) -> None:
        canonical = self.root / ABLATION_LOGICAL_PATH
        value = load(canonical)
        value["ratios"]["rgb"][0]["wall_speedup_p128_over_p1"] = 999.0
        write_json(canonical, value)
        self.assert_rejected()

    def test_stale_hash_fails(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "native-cuda-smoke")
        result = load(Path(gate["result"]["path"]))
        raw_path = Path(result["evidence"]["raw_result"]["path"])
        raw_path.write_text(raw_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assert_rejected()

    def test_sanitizer_error_fails_even_with_refreshed_hash(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "compute-sanitizer-memcheck")
        log_path = Path(gate["log"]["path"])
        log_path.write_text("========= ERROR SUMMARY: 1 error\nCOMPUTE_SANITIZER_OK\n", encoding="utf-8")
        refresh(gate["log"], log_path)
        write_json(self.verification, record)
        self.assert_rejected()

    def test_short_soak_fails_even_with_refreshed_graph(self) -> None:
        record = load(self.verification)
        gate = next(item for item in record["gates"] if item["gate_id"] == "bounded-stability-soak")
        result_path = Path(gate["result"]["path"])
        result = load(result_path)
        raw_path = Path(result["evidence"]["soak_result"]["path"])
        raw = load(raw_path)
        raw["duration_seconds"] = 10.0
        write_json(raw_path, raw)
        refresh(result["evidence"]["soak_result"], raw_path)
        write_json(result_path, result)
        refresh(gate["result"], result_path)
        write_json(self.verification, record)
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main(verbosity=2)
