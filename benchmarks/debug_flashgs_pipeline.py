#!/usr/bin/env python3
"""Trace exact Gaussian IDs through one untimed adapted-FlashGS camera render."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from isaacsim_gaussian_renderer import (  # noqa: E402
    FlashGSBackend,
    RendererService,
)
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
    DISCOVERY_MODES,
    FLASHGS_PIPELINE_DEBUG_SCHEMA,
    load_verified_b64_known_failure_case,
    load_verified_flashgs_adapter_attestation,
    trace_flashgs_backend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Untimed, synchronized diagnostic for one camera/pixel. Run only "
            "after the shared-node occupancy gate and never report its timing."
        )
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--scene-sha256", default=HOME_SCAN_SHA256)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--camera-index", type=int, required=True)
    parser.add_argument("--pixel-x", type=int, required=True)
    parser.add_argument("--pixel-y", type=int, required=True)
    parser.add_argument("--expected-semantic-id", type=int, default=-1)
    parser.add_argument("--gaussian-id", action="append", type=int, default=[])
    parser.add_argument(
        "--discovery-mode",
        choices=DISCOVERY_MODES,
        default="all-contributors",
    )
    parser.add_argument("--max-discovered-candidates", type=int, default=32)
    parser.add_argument(
        "--intersection-capacity",
        type=int,
        required=True,
        help="Fixed per-camera capacity from the matching zero-overflow run.",
    )
    parser.add_argument(
        "--semantic-topology",
        choices=SEMANTIC_TOPOLOGIES,
        default=REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    )
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--known-failure-manifest",
        type=Path,
        required=True,
        help="Frozen five-case B64 diagnostic manifest.",
    )
    parser.add_argument(
        "--flashgs-adapter-attestation",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def json_number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite diagnostic evidence: {args.output}."
        )
    executor_lock = ensure_cooperative_executor_lock()
    source_provenance = load_verified_source_manifest(
        args.source_manifest,
        project_root=PROJECT_ROOT,
    )
    if source_provenance.get("dirty") is not False:
        raise RuntimeError(
            "FlashGS pipeline diagnosis requires a clean source manifest."
        )
    adapter_attestation = load_verified_flashgs_adapter_attestation(
        args.flashgs_adapter_attestation,
        source_provenance=source_provenance,
        project_root=PROJECT_ROOT,
    )
    preflight = capture_node_snapshot()
    preflight_failures = occupancy_failures(
        preflight,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=False,
    )
    if preflight_failures:
        raise RuntimeError(
            "Shared RTX node occupancy preflight failed: "
            + "; ".join(preflight_failures)
        )
    occupancy_sampler = ComputeProcessSampler(
        expected_gpu_uuid=args.expected_gpu_uuid,
        allowed_pids={os.getpid()},
    ).start()
    if args.intersection_capacity <= 0:
        raise ValueError("intersection_capacity must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("FlashGS pipeline debugging requires CUDA.")
    torch.cuda.init()
    gpu_uuid = active_cuda_device_uuid(torch.cuda)
    if gpu_uuid != args.expected_gpu_uuid:
        raise RuntimeError(
            f"Active GPU UUID {gpu_uuid!r} != {args.expected_gpu_uuid!r}."
        )

    trajectory = load_trajectory(args.trajectory)
    if trajectory.scene_sha256 != args.scene_sha256:
        raise ValueError("Trajectory and scene checksums differ.")
    if not 0 <= args.step < trajectory.timesteps:
        raise ValueError("step lies outside the trajectory.")
    if not 0 <= args.camera_index < trajectory.batch:
        raise ValueError("camera_index lies outside the trajectory batch.")
    if not 0 <= args.pixel_x < trajectory.width:
        raise ValueError("pixel_x lies outside the image.")
    if not 0 <= args.pixel_y < trajectory.height:
        raise ValueError("pixel_y lies outside the image.")
    historical_failure_case = load_verified_b64_known_failure_case(
        args.known_failure_manifest,
        trajectory_id=trajectory.trajectory_id,
        scene_sha256=args.scene_sha256,
        step=args.step,
        camera_index=args.camera_index,
        pixel_x=args.pixel_x,
        pixel_y=args.pixel_y,
        width=trajectory.width,
        height=trajectory.height,
        expected_semantic_id=args.expected_semantic_id,
    )

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
        max_views=1,
    )
    try:
        service.initialize(stage=None, device="cuda")
        service.load_scene(SCENE_ID, **scene)
        viewmat = torch.from_numpy(
            trajectory.viewmats[
                args.step, args.camera_index : args.camera_index + 1
            ]
        ).to("cuda")
        intrinsics = torch.from_numpy(
            trajectory.intrinsics[
                args.step, args.camera_index : args.camera_index + 1
            ]
        ).to("cuda")
        scene_ids = torch.tensor([SCENE_ID], device="cuda", dtype=torch.int64)
        outputs = service.render(viewmat, intrinsics, scene_ids)
        service.synchronize()
        counters = backend.check_capacity(synchronize=False)
        trace = trace_flashgs_backend(
            backend,
            viewmat=viewmat[0],
            intrinsics=intrinsics[0],
            target_pixel_x=args.pixel_x,
            target_pixel_y=args.pixel_y,
            candidate_ids=args.gaussian_id,
            discovery_mode=args.discovery_mode,
            expected_semantic_id=args.expected_semantic_id,
            max_discovered_candidates=args.max_discovered_candidates,
        )
        production_native_path = Path(str(backend._native.__file__)).resolve()
        pixel = {
            "rgb": [
                json_number(value)
                for value in outputs["rgb"][0, args.pixel_y, args.pixel_x]
                .detach()
                .cpu()
                .tolist()
            ],
            "depth": json_number(
                outputs["depth"][0, args.pixel_y, args.pixel_x, 0].item()
            ),
            "alpha": json_number(
                outputs["alpha"][0, args.pixel_y, args.pixel_x, 0].item()
            ),
            "semantic_id": int(
                outputs["semantic_id"][
                    0, args.pixel_y, args.pixel_x, 0
                ].item()
            ),
        }
        sampled_compute_process_telemetry = occupancy_sampler.stop()
        postflight = capture_node_snapshot()
        postflight_failures = occupancy_failures(
            postflight,
            expected_gpu_uuid=args.expected_gpu_uuid,
            allow_current_gpu_process=True,
        )
        integrity_checks = {
            "clean_source_manifest": source_provenance.get("dirty") is False,
            "adapter_attestation_bound": bool(adapter_attestation),
            "preflight_occupancy": not preflight_failures,
            "postflight_occupancy": not postflight_failures,
            "cooperative_node_wide_lock": executor_lock["pass"],
            "sampled_compute_process_telemetry": (
                sampled_compute_process_telemetry["pass"]
            ),
            "historical_failure_case_bound": bool(historical_failure_case),
            "historical_failure_reproduced": (
                pixel["alpha"]
                == historical_failure_case["case"][
                    "historical_flashgs_alpha"
                ]
                and pixel["semantic_id"]
                == historical_failure_case["case"][
                    "historical_flashgs_semantic_id"
                ]
            ),
            "zero_intersection_overflow": not counters[
                "intersection_overflow"
            ],
            "production_native_build_contract": getattr(
                backend._native, "__vgr_build_contract__", None
            )
            is not None,
            "debug_native_build_contract": trace["native_build_contract"]
            is not None,
            "projection_replay_crosscheck": trace[
                "projection_replay_crosscheck"
            ]["all_bit_exact"]
            is not False,
            "discovery_score_crosscheck": trace[
                "discovery_score_crosscheck"
            ]["all_bit_exact"]
            is not False,
        }
        tool_integrity_pass = all(integrity_checks.values())
        cause_identified = bool(trace["cause_identified"])
        diagnosis_complete = tool_integrity_pass and cause_identified
        result = {
            "schema_version": FLASHGS_PIPELINE_DEBUG_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "debug_only": True,
            "measured_timing_valid": False,
            "tool_integrity": {
                "pass": tool_integrity_pass,
                "checks": integrity_checks,
            },
            "cause_identified": cause_identified,
            "diagnosis_complete": diagnosis_complete,
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
                "sampled_compute_process_telemetry": (
                    sampled_compute_process_telemetry
                ),
                "postflight": postflight,
                "postflight_failures": postflight_failures,
            },
            "scene": scene_metadata,
            "source_provenance": source_provenance,
            "flashgs_adapter_attestation": adapter_attestation,
            "camera": {
                "trajectory": artifact_record(args.trajectory),
                "trajectory_arrays": artifact_record(
                    args.trajectory.with_suffix(".npz")
                ),
                "trajectory_id": trajectory.trajectory_id,
                "step": args.step,
                "camera_index": args.camera_index,
                "target_pixel_xy": [args.pixel_x, args.pixel_y],
                "historical_failure_case": historical_failure_case,
            },
            "equation": {
                "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
                "covariance_epsilon": 0.0,
                "alpha_threshold": 1.0 / 255.0,
                "alpha_cap": 0.99,
                "transmittance_threshold": 1.0e-4,
                "pixel_center": "x+0.5,y+0.5",
            },
            "capacity": {
                "installed_intersections": args.intersection_capacity,
                "counters": counters,
            },
            "production_adapter": {
                "native_extension": artifact_record(production_native_path),
                "native_build_contract": getattr(
                    backend._native, "__vgr_build_contract__", None
                ),
                "preprocess_source": artifact_record(
                    PROJECT_ROOT
                    / "src/isaacsim_gaussian_renderer/native/flashgs/preprocess.cu"
                ),
                "render_source": artifact_record(
                    PROJECT_ROOT
                    / "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
                ),
            },
            "rendered_target_pixel": pixel,
            "trace": trace,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "FLASHGS_PIPELINE_DEBUG_OK "
            + json.dumps(
                {
                    "output": str(args.output),
                    "records": len(trace["records"]),
                    "tool_integrity_pass": tool_integrity_pass,
                    "cause_identified": cause_identified,
                    "diagnosis_complete": diagnosis_complete,
                },
                sort_keys=True,
            )
        )
        if not tool_integrity_pass:
            raise SystemExit(1)
        if not cause_identified:
            raise SystemExit(2)
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
