#!/usr/bin/env python3
"""Run one matched dynamic RendererService benchmark process.

Each invocation owns exactly one renderer so peak memory is not contaminated by
the comparison backend. A matrix driver launches matching custom and FlashGS
processes against the same immutable camera contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from benchmarks.flashgs_matched_memory import (  # noqa: E402
    MEMORY_BASELINE_SAMPLE_COUNT,
    TORCH_CUMULATIVE_MEMORY_COUNTERS,
    assess_steady_state_memory,
)
from benchmarks.flashgs_matched_occupancy import (  # noqa: E402
    ComputeProcessSampler,
    capture_node_snapshot,
    ensure_cooperative_executor_lock,
    occupancy_failures,
)
from isaacsim_gaussian_renderer import (  # noqa: E402
    CustomCudaBackend,
    FlashGSBackend,
    RendererService,
)
from isaacsim_gaussian_renderer.benchmark_manifest import file_sha256  # noqa: E402
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    CAPACITY_CALIBRATION_SCHEMA,
    CAPACITY_CONSUMPTION_SCHEMA,
    FLASHGS_DEMAND_SURVEY_CONSUMPTION_SCHEMA,
    FLASHGS_DEMAND_SURVEY_SCHEMA,
    HEADLINE_GPU_NAME,
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PRIMARY_BATCHES,
    PRIMARY_TRAJECTORY_IDS,
    RENDERER_RUN_SCHEMA,
    TIMED_CAPACITY_VERIFICATION_SCHEMA,
    active_cuda_device_uuid,
    artifact_record,
    audit_flashgs_demand_counter_source,
    derive_flashgs_prefix_capacities,
    is_nvidia_gpu_uuid,
    load_verified_source_manifest,
    primary_fidelity_selection,
    primary_max_physical_views,
    prove_trajectory_prefixes,
    same_artifact,
    source_identity,
    trajectory_artifacts,
    verify_flashgs_demand_counter_source_audit,
    verify_node_occupancy_evidence,
    verify_trajectory_prefix_proof,
)
from isaacsim_gaussian_renderer.evaluation.matched_semantics import (  # noqa: E402
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    SEMANTIC_TOPOLOGIES,
    matched_semantic_ids,
)
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import load_trajectory  # noqa: E402
from isaacsim_gaussian_renderer.flashgs_debug import (  # noqa: E402
    load_verified_flashgs_adapter_attestation,
)
from isaacsim_gaussian_renderer.flashgs_native_loader import (  # noqa: E402
    FLASHGS_UPSTREAM_COMMIT,
)
from isaacsim_gaussian_renderer.ply_loader import (  # noqa: E402
    canonicalize_3dgs_scene,
    load_ply_to_gaussians,
)

HOME_SCAN_COUNT = 21_497_908
HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
SCENE_ID = 404
FULL_OUTPUTS = ("rgb", "depth", "alpha", "semantic_id")
RGB_OUTPUTS = ("rgb",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", choices=("custom", "flashgs"), required=True)
    parser.add_argument("--output-contract", choices=("rgb", "full"), required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--scene-sha256", default=HOME_SCAN_SHA256)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--measured-frames", type=int, default=100)
    parser.add_argument("--capacity-headroom", type=float, default=1.05)
    parser.add_argument("--initial-visible-per-view", type=int, default=170_000)
    parser.add_argument("--initial-intersections-per-view", type=int, default=200_000)
    parser.add_argument("--flashgs-initial-intersections", type=int, default=200_000)
    parser.set_defaults(
        fixed_visible_capacity=None,
        fixed_intersection_capacity=None,
    )
    parser.add_argument(
        "--capacity-calibration",
        type=Path,
        help=("Required hash-bound zero-overflow calibration for timed Custom runs."),
    )
    parser.add_argument(
        "--capacity-calibration-only",
        action="store_true",
        help=(
            "Run the full 108-step device-counter calibration and write a capacity artifact instead of timing renders."
        ),
    )
    parser.add_argument(
        "--flashgs-demand-survey",
        type=Path,
        help=(
            "Required hash-bound B1024 demand survey for timed FlashGS runs. "
            "The batch-specific derived capacity is installed during backend construction."
        ),
    )
    parser.add_argument(
        "--flashgs-demand-survey-only",
        action="store_true",
        help=("Run one untimed, output-invalid B1024 low-capacity demand survey instead of timing renders."),
    )
    parser.add_argument(
        "--prefix-trajectory",
        type=Path,
        action="append",
        default=[],
        help=(
            "Primary trajectory contract included in the B1024 prefix proof; "
            "the survey requires exactly B1,8,32,64,128,256,512,1024."
        ),
    )
    parser.add_argument("--custom-depth-buckets", type=int, default=128)
    parser.add_argument("--custom-depth-bucket-group", type=int, default=8)
    parser.add_argument(
        "--custom-max-physical-views",
        type=int,
        help=(
            "Optional fixed internal camera chunk for Custom. The matched "
            "B512 and B1024 lanes use 128; B1 through B256 leave this unset "
            "and retain the direct single-submission path."
        ),
    )
    parser.add_argument(
        "--semantic-topology",
        choices=SEMANTIC_TOPOLOGIES,
        default=REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    )
    parser.add_argument("--capture-last-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--independent-trial", type=int, default=1)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="Byte-level source manifest generated by scripts/write_source_manifest.py.",
    )
    parser.add_argument("--flashgs-adapter-attestation", type=Path)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-nonheadline-gpu", action="store_true")
    parser.add_argument(
        "--profile-control",
        action="store_true",
        help=(
            "Permit a short, explicitly non-headline Nsight control while "
            "retaining the same runner and fixed-capacity path."
        ),
    )
    return parser.parse_args()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty timing distribution.")
    stddev = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "stddev": stddev,
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        # Preserve the exact ordered trajectory samples. The 100 camera states
        # are correlated deterministic work, not iid replicates, so this runner
        # deliberately does not manufacture a confidence interval from their
        # within-trajectory variance.
        "samples": [float(value) for value in array],
    }


def process_driver_memory(expected_gpu_uuid: str) -> dict[str, Any]:
    raw = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    used_mib = None
    if raw:
        for line in raw.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 3 and fields[0] == expected_gpu_uuid and fields[1] == str(os.getpid()):
                try:
                    used_mib = int(fields[2])
                except ValueError:
                    pass
    return {
        "source": "nvidia-smi-query-compute-apps",
        "gpu_uuid": expected_gpu_uuid,
        "raw": raw,
        "current_process_used_mib": used_mib,
        "available": used_mib is not None,
    }


def current_process_gpu_uuid() -> str | None:
    # Resolve the CUDA-visible device directly first.  Some Torch/driver pairs
    # do not register a compute process until the first allocation, so a
    # process-table-only probe can return no UUID before scene upload.  Device
    # properties establish the same active-device identity without perturbing
    # allocator or NVML memory baselines.
    property_uuid = active_cuda_device_uuid(torch.cuda)
    if property_uuid:
        return property_uuid

    # Retain the process-table fallback for Torch builds that do not expose a
    # UUID in device properties and have already established a CUDA context.
    raw = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        return None
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[1] == str(os.getpid()):
            return fields[0]
    return None


def torch_memory_snapshot() -> dict[str, Any]:
    stats = torch.cuda.memory_stats()
    cumulative_counters = {
        name: int(stats[name]) if name in stats else None for name in TORCH_CUMULATIVE_MEMORY_COUNTERS
    }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "cumulative_counters": cumulative_counters,
        "missing_cumulative_counters": [name for name, value in cumulative_counters.items() if value is None],
    }


def steady_state_memory_sample(expected_gpu_uuid: str) -> dict[str, Any]:
    return {
        "torch": torch_memory_snapshot(),
        "driver": process_driver_memory(expected_gpu_uuid),
    }


def load_home_scan(
    path: Path,
    expected_sha256: str,
    semantic_topology: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    started = time.perf_counter()
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Scene SHA-256 mismatch: {actual_sha256} != {expected_sha256}.")
    raw = load_ply_to_gaussians(path)
    if raw.count != HOME_SCAN_COUNT:
        raise ValueError(f"Home Scan count mismatch: {raw.count} != {HOME_SCAN_COUNT}.")
    canonical = canonicalize_3dgs_scene(raw, device="cuda")
    scene = {
        "means": canonical.means,
        "scales": canonical.scales,
        "rotations": canonical.rotations,
        "opacities": canonical.opacities,
        "features": canonical.features[:, :3].contiguous(),
        "semantic_ids": matched_semantic_ids(
            canonical.means,
            semantic_topology,
        ),
    }
    torch.cuda.synchronize()
    return scene, {
        "path": str(path),
        "sha256": actual_sha256,
        "gaussian_count": canonical.count,
        "load_and_upload_seconds": time.perf_counter() - started,
        "canonical_precision": "float32",
        "color_space": "linear-rgb",
        "semantic_topology": semantic_topology,
        "semantic_rule_input": (
            "spatially-coherent-scene-bounds-octants"
            if semantic_topology == REPRESENTATIVE_SEMANTIC_TOPOLOGY
            else "gaussian-index-modulo-1024-interleaved-stress"
        ),
    }


def build_backend(
    args: argparse.Namespace,
    *,
    batch: int,
    near_plane: float,
    far_plane: float,
    installed_capacity: dict[str, int | None] | None = None,
) -> CustomCudaBackend | FlashGSBackend:
    if args.renderer == "custom":
        physical_batch = min(
            batch,
            args.custom_max_physical_views or batch,
        )
        visible_capacity = (
            installed_capacity.get("visible_records")
            if installed_capacity is not None
            else physical_batch * args.initial_visible_per_view
        )
        intersection_capacity = (
            installed_capacity.get("intersections")
            if installed_capacity is not None
            else physical_batch * args.initial_intersections_per_view
        )
        if not isinstance(visible_capacity, int) or visible_capacity <= 0:
            raise ValueError("Custom installed visible capacity must be positive.")
        if not isinstance(intersection_capacity, int) or intersection_capacity <= 0:
            raise ValueError("Custom installed intersection capacity must be positive.")
        return CustomCudaBackend(
            max_visible_records=visible_capacity,
            max_intersections=intersection_capacity,
            near_plane=near_plane,
            far_plane=far_plane,
            gaussian_support_sigma=MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            covariance_epsilon=0.0,
            semantic_min_alpha=0.01,
            tile_size=1,
            depth_bucket_count=args.custom_depth_buckets,
            depth_bucket_group_size=args.custom_depth_bucket_group,
            compact_projection_cache=True,
            enable_projection_cache=False,
            output_srgb=False,
            deterministic=False,
            fixed_capacity_sort=True,
            max_physical_views=args.custom_max_physical_views,
            adaptive_capacity=False,
        )
    flashgs_capacity = (
        installed_capacity.get("intersections")
        if installed_capacity is not None
        else args.flashgs_initial_intersections
    )
    if not isinstance(flashgs_capacity, int) or flashgs_capacity <= 0:
        raise ValueError("FlashGS installed intersection capacity must be positive.")
    return FlashGSBackend(
        max_intersections=flashgs_capacity,
        near_plane=near_plane,
        far_plane=far_plane,
        gaussian_support_sigma=MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        covariance_epsilon=0.0,
        semantic_min_alpha=0.01,
        tile_size=16,
    )


def current_device_counter_snapshots(
    backend: CustomCudaBackend | FlashGSBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batch-total work and the workspace-capacity demand.

    Scientific counters are logical-batch totals. Capacity demand is the
    maximum over the storage-reuse unit: the full batch for direct Custom, a
    physical camera chunk for chunked Custom, or one camera for FlashGS.
    """
    counters = backend.workspace["counters"]
    if isinstance(backend, CustomCudaBackend):
        physical_capacity_counters = backend.workspace.get("physical_capacity_counters")
        return (
            counters,
            counters if physical_capacity_counters is None else physical_capacity_counters,
        )
    per_camera = counters[: backend._max_views]
    return per_camera.sum(dim=0), per_camera.amax(dim=0)


def custom_uses_physical_chunks(backend: CustomCudaBackend) -> bool:
    return backend.max_physical_views is not None and (
        backend._max_views == 0 or backend._physical_max_views < backend._max_views
    )


def preflight_trajectory(
    service: RendererService,
    backend: CustomCudaBackend | FlashGSBackend,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    scene_ids: torch.Tensor,
    *,
    measured_start: int,
) -> tuple[dict[str, int], dict[str, torch.Tensor]]:
    if measured_start < 0 or measured_start >= int(viewmats.shape[0]):
        raise ValueError("measured_start must leave at least one measured frame.")
    max_counters: torch.Tensor | None = None
    sum_counters: torch.Tensor | None = None
    measured_max_counters: torch.Tensor | None = None
    measured_sum_counters: torch.Tensor | None = None
    capacity_max_counters: torch.Tensor | None = None
    measured_capacity_max_counters: torch.Tensor | None = None
    outputs: dict[str, torch.Tensor] | None = None
    for step in range(viewmats.shape[0]):
        outputs = service.render(viewmats[step], intrinsics[step], scene_ids, outputs=outputs)
        snapshot, capacity_snapshot = current_device_counter_snapshots(backend)
        if max_counters is None:
            max_counters = snapshot.clone()
            sum_counters = snapshot.clone()
            capacity_max_counters = capacity_snapshot.clone()
        else:
            torch.maximum(max_counters, snapshot, out=max_counters)
            sum_counters.add_(snapshot)
            assert capacity_max_counters is not None
            torch.maximum(
                capacity_max_counters,
                capacity_snapshot,
                out=capacity_max_counters,
            )
        if step >= measured_start:
            if measured_max_counters is None:
                measured_max_counters = snapshot.clone()
                measured_sum_counters = snapshot.clone()
                measured_capacity_max_counters = capacity_snapshot.clone()
            else:
                torch.maximum(
                    measured_max_counters,
                    snapshot,
                    out=measured_max_counters,
                )
                measured_sum_counters.add_(snapshot)
                assert measured_capacity_max_counters is not None
                torch.maximum(
                    measured_capacity_max_counters,
                    capacity_snapshot,
                    out=measured_capacity_max_counters,
                )
    service.synchronize()
    assert (
        max_counters is not None
        and sum_counters is not None
        and measured_max_counters is not None
        and measured_sum_counters is not None
        and capacity_max_counters is not None
        and measured_capacity_max_counters is not None
        and outputs is not None
    )
    maximum = [int(value) for value in max_counters.detach().cpu().tolist()]
    total = [int(value) for value in sum_counters.detach().cpu().tolist()]
    measured_maximum = [int(value) for value in measured_max_counters.detach().cpu().tolist()]
    measured_total = [int(value) for value in measured_sum_counters.detach().cpu().tolist()]
    capacity_maximum = [int(value) for value in capacity_max_counters.detach().cpu().tolist()]
    measured_capacity_maximum = [int(value) for value in measured_capacity_max_counters.detach().cpu().tolist()]
    del (
        max_counters,
        sum_counters,
        measured_max_counters,
        measured_sum_counters,
        capacity_max_counters,
        measured_capacity_max_counters,
    )
    if isinstance(backend, CustomCudaBackend):
        names = (
            "visible_gaussians",
            "generated_intersections",
            "visible_overflow",
            "intersection_overflow",
            "active_tiles",
        )
    else:
        names = (
            "visible_gaussians",
            "generated_intersections",
            "intersection_overflow",
        )
    result = {f"max_{name}": maximum[index] for index, name in enumerate(names)}
    result.update({f"mean_{name}": int(round(total[index] / viewmats.shape[0])) for index, name in enumerate(names)})
    measured_frames = int(viewmats.shape[0]) - measured_start
    result.update({f"measured_max_{name}": measured_maximum[index] for index, name in enumerate(names)})
    result.update(
        {
            f"measured_mean_{name}": int(round(measured_total[index] / measured_frames))
            for index, name in enumerate(names)
        }
    )
    if isinstance(backend, FlashGSBackend):
        result.update({f"max_{name}_per_camera": capacity_maximum[index] for index, name in enumerate(names)})
        result.update(
            {f"measured_max_{name}_per_camera": (measured_capacity_maximum[index]) for index, name in enumerate(names)}
        )
    elif custom_uses_physical_chunks(backend):
        result.update({f"max_{name}_per_physical_chunk": capacity_maximum[index] for index, name in enumerate(names)})
        result.update(
            {
                f"measured_max_{name}_per_physical_chunk": (measured_capacity_maximum[index])
                for index, name in enumerate(names)
            }
        )
    return result, outputs


def intersection_capacity_demand(
    backend: CustomCudaBackend | FlashGSBackend,
    counters: dict[str, int],
) -> int:
    if isinstance(backend, FlashGSBackend):
        return counters["max_generated_intersections_per_camera"]
    if custom_uses_physical_chunks(backend):
        return counters["max_generated_intersections_per_physical_chunk"]
    return counters["max_generated_intersections"]


def visible_capacity_demand(
    backend: CustomCudaBackend | FlashGSBackend,
    counters: dict[str, int],
) -> int:
    if isinstance(backend, CustomCudaBackend) and custom_uses_physical_chunks(backend):
        return counters["max_visible_gaussians_per_physical_chunk"]
    return counters["max_visible_gaussians"]


def intersection_capacity_scope(
    backend: CustomCudaBackend | FlashGSBackend,
) -> str:
    if isinstance(backend, FlashGSBackend):
        return "per-camera-reused-workspace"
    if custom_uses_physical_chunks(backend):
        return "physical-camera-chunk-reused-workspace"
    return "batch-global-workspace"


def calibrate_capacity(
    args: argparse.Namespace,
    service: RendererService,
    backend: CustomCudaBackend | FlashGSBackend,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    scene_ids: torch.Tensor,
    *,
    measured_start: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    initial, outputs = preflight_trajectory(
        service,
        backend,
        viewmats,
        intrinsics,
        scene_ids,
        measured_start=measured_start,
    )
    visible_target = max(
        1,
        math.ceil(visible_capacity_demand(backend, initial) * args.capacity_headroom),
    )
    intersection_target = max(
        1,
        math.ceil(intersection_capacity_demand(backend, initial) * args.capacity_headroom),
    )
    attempts: list[dict[str, Any]] = []
    # A too-small custom visible-record preflight can count total visibility
    # exactly while undercounting downstream intersections. Re-run with the
    # observed demand until both device overflow counters are zero instead of
    # treating a single hindsight-sized retry as trustworthy.
    for _attempt in range(3):
        if isinstance(backend, CustomCudaBackend):
            backend.reserve_workspace_capacity(
                visible_records=visible_target,
                tile_intersections=intersection_target,
            )
        else:
            backend.reserve_intersection_capacity(intersection_target)
        torch.cuda.empty_cache()
        verified, outputs = preflight_trajectory(
            service,
            backend,
            viewmats,
            intrinsics,
            scene_ids,
            measured_start=measured_start,
        )
        attempts.append(
            {
                "installed_visible_records": (visible_target if isinstance(backend, CustomCudaBackend) else None),
                "installed_intersections": intersection_target,
                "observed": verified,
            }
        )
        visible_overflow = verified.get("max_visible_overflow", 0)
        intersection_overflow = verified["max_intersection_overflow"]
        if not visible_overflow and not intersection_overflow:
            return (
                {
                    "schema_version": "flashgs-matched-capacity-v1",
                    "headroom": args.capacity_headroom,
                    "initial_preflight": initial,
                    "calibration_attempts": attempts,
                    "installed": {
                        "visible_records": (visible_target if isinstance(backend, CustomCudaBackend) else None),
                        "intersections": intersection_target,
                    },
                    "intersection_capacity_scope": intersection_capacity_scope(backend),
                    "verified_preflight": verified,
                    "preflight_frames_per_pass": int(viewmats.shape[0]),
                    "preflight_measured_frames_per_pass": (int(viewmats.shape[0]) - measured_start),
                    "preflight_passes": 1 + len(attempts),
                    "capacity_source": "full-trajectory-device-counter-calibration",
                },
                outputs,
            )
        visible_target = max(
            visible_target,
            math.ceil(visible_capacity_demand(backend, verified) * args.capacity_headroom),
        )
        intersection_target = max(
            intersection_target,
            math.ceil(intersection_capacity_demand(backend, verified) * args.capacity_headroom),
        )
    raise RuntimeError(
        "Capacity calibration failed closed after three installed-capacity "
        f"passes; final observed counters were {attempts[-1]['observed']}."
    )


def survey_flashgs_capacity_demand(
    service: RendererService,
    backend: FlashGSBackend,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    scene_ids: torch.Tensor,
) -> dict[str, Any]:
    """Collect exact per-camera demand maxima from device counters.

    The intentionally small workspace may overflow.  Native preprocess counts
    generated demand before guarding writes, so counters remain valid while
    every rendered output is invalid by protocol.
    """

    if int(viewmats.shape[1]) != PRIMARY_BATCHES[-1]:
        raise ValueError("The FlashGS demand survey requires the complete B1024 trajectory.")
    counter_maxima = torch.zeros_like(backend.workspace["counters"][: backend._max_views])
    outputs: dict[str, torch.Tensor] | None = None
    for step in range(int(viewmats.shape[0])):
        outputs = service.render(viewmats[step], intrinsics[step], scene_ids, outputs=outputs)
        torch.maximum(
            counter_maxima,
            backend.workspace["counters"][: backend._max_views],
            out=counter_maxima,
        )
    service.synchronize()
    maxima = counter_maxima.detach().cpu().numpy()
    if maxima.shape != (PRIMARY_BATCHES[-1], 3) or np.any(maxima < 0):
        raise RuntimeError("FlashGS demand survey produced malformed device counters.")
    visible = [int(value) for value in maxima[:, 0]]
    generated = [int(value) for value in maxima[:, 1]]
    overflow = [int(value) for value in maxima[:, 2]]
    return {
        "schema_version": "flashgs-matched-flashgs-demand-observation-v1",
        "counter_storage": "cuda-int64-per-camera",
        "frames_observed": int(viewmats.shape[0]),
        "camera_count": int(viewmats.shape[1]),
        "device_max_updates": int(viewmats.shape[0]),
        "cpu_readbacks": 1,
        "per_camera_max_visible_gaussians": visible,
        "per_camera_max_generated_intersections": generated,
        "per_camera_max_intersection_overflow": overflow,
        "max_visible_gaussians_per_camera": max(visible),
        "max_generated_intersections_per_camera": max(generated),
        "max_intersection_overflow_per_camera": max(overflow),
    }


def create_timed_capacity_audit(
    backend: CustomCudaBackend | FlashGSBackend,
    *,
    expected_frames: int,
    measured_start: int,
) -> dict[str, Any]:
    """Preallocate device-only state for a direct timed-run overflow audit."""

    counters = backend.workspace["counters"]
    logical_width = int(counters.shape[-1])
    if isinstance(backend, CustomCudaBackend):
        capacity_counters = backend.workspace.get("physical_capacity_counters", counters)
    else:
        capacity_counters = counters
    return {
        "expected_frames": int(expected_frames),
        "measured_start": int(measured_start),
        "updates": 0,
        "measured_updates": 0,
        "all_logical_max": torch.zeros((logical_width,), device=counters.device, dtype=torch.int64),
        "all_logical_sum": torch.zeros((logical_width,), device=counters.device, dtype=torch.int64),
        "measured_logical_max": torch.zeros((logical_width,), device=counters.device, dtype=torch.int64),
        "measured_logical_sum": torch.zeros((logical_width,), device=counters.device, dtype=torch.int64),
        "all_capacity_max": torch.zeros((logical_width,), device=counters.device, dtype=torch.int64),
        "measured_capacity_max": torch.zeros((logical_width,), device=counters.device, dtype=torch.int64),
        "logical_scratch": torch.zeros((logical_width,), device=counters.device, dtype=torch.int64),
        "capacity_scratch": torch.zeros((logical_width,), device=capacity_counters.device, dtype=torch.int64),
    }


def update_timed_capacity_audit(
    state: dict[str, Any],
    backend: CustomCudaBackend | FlashGSBackend,
    *,
    measured: bool,
) -> None:
    """Update preallocated maxima/sums without a CPU readback or allocation."""

    counters = backend.workspace["counters"]
    logical_scratch = state["logical_scratch"]
    capacity_scratch = state["capacity_scratch"]
    if isinstance(backend, FlashGSBackend):
        active = counters[: backend._max_views]
        torch.sum(active, dim=0, out=logical_scratch)
        torch.amax(active, dim=0, out=capacity_scratch)
    else:
        logical_scratch.copy_(counters)
        capacity_scratch.copy_(backend.workspace.get("physical_capacity_counters", counters))
    torch.maximum(state["all_logical_max"], logical_scratch, out=state["all_logical_max"])
    state["all_logical_sum"].add_(logical_scratch)
    torch.maximum(state["all_capacity_max"], capacity_scratch, out=state["all_capacity_max"])
    state["updates"] += 1
    if measured:
        torch.maximum(
            state["measured_logical_max"],
            logical_scratch,
            out=state["measured_logical_max"],
        )
        state["measured_logical_sum"].add_(logical_scratch)
        torch.maximum(
            state["measured_capacity_max"],
            capacity_scratch,
            out=state["measured_capacity_max"],
        )
        state["measured_updates"] += 1


def finalize_timed_capacity_audit(
    state: dict[str, Any],
    backend: CustomCudaBackend | FlashGSBackend,
) -> dict[str, Any]:
    """Read one complete direct overflow audit after all timed frames."""

    backend.synchronize()
    if state["updates"] != state["expected_frames"]:
        raise RuntimeError("Timed capacity audit did not observe every rendered frame.")
    expected_measured = state["expected_frames"] - state["measured_start"]
    if state["measured_updates"] != expected_measured:
        raise RuntimeError("Timed capacity audit did not observe every measured frame.")
    maximum = [int(value) for value in state["all_logical_max"].detach().cpu().tolist()]
    total = [int(value) for value in state["all_logical_sum"].detach().cpu().tolist()]
    measured_maximum = [int(value) for value in state["measured_logical_max"].detach().cpu().tolist()]
    measured_total = [int(value) for value in state["measured_logical_sum"].detach().cpu().tolist()]
    capacity_maximum = [int(value) for value in state["all_capacity_max"].detach().cpu().tolist()]
    measured_capacity_maximum = [int(value) for value in state["measured_capacity_max"].detach().cpu().tolist()]
    if isinstance(backend, CustomCudaBackend):
        names = (
            "visible_gaussians",
            "generated_intersections",
            "visible_overflow",
            "intersection_overflow",
            "active_tiles",
        )
        capacity_suffix = "_per_physical_chunk" if custom_uses_physical_chunks(backend) else ""
    else:
        names = (
            "visible_gaussians",
            "generated_intersections",
            "intersection_overflow",
        )
        capacity_suffix = "_per_camera"
    observed = {f"max_{name}": maximum[index] for index, name in enumerate(names)}
    observed.update(
        {f"mean_{name}": int(round(total[index] / state["expected_frames"])) for index, name in enumerate(names)}
    )
    observed.update({f"measured_max_{name}": measured_maximum[index] for index, name in enumerate(names)})
    observed.update(
        {
            f"measured_mean_{name}": int(round(measured_total[index] / expected_measured))
            for index, name in enumerate(names)
        }
    )
    if capacity_suffix:
        observed.update({f"max_{name}{capacity_suffix}": capacity_maximum[index] for index, name in enumerate(names)})
        observed.update(
            {
                f"measured_max_{name}{capacity_suffix}": measured_capacity_maximum[index]
                for index, name in enumerate(names)
            }
        )
    passed = bool(observed.get("max_visible_overflow", 0) == 0 and observed["max_intersection_overflow"] == 0)
    return {
        "schema_version": TIMED_CAPACITY_VERIFICATION_SCHEMA,
        "pass": passed,
        "source": "renderer-device-counters",
        "frames_observed": state["updates"],
        "warmup_frames_observed": state["measured_start"],
        "measured_frames_observed": state["measured_updates"],
        "device_max_updates": state["updates"],
        "cpu_readbacks_in_measured_loop": 0,
        "final_cpu_readback_after_measurement": True,
        "audit_updates_in_cuda_event_timing": False,
        "audit_updates_in_wall_samples": False,
        "observed": observed,
    }


def load_timed_capacity_artifact(
    args: argparse.Namespace,
    *,
    trajectory: Any,
    source_provenance: dict[str, Any],
    adapter_attestation: dict[str, Any] | None,
    gpu_name: str,
    gpu_uuid: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int | None]]:
    """Load the renderer-specific capacity artifact for one timed run."""

    artifact_path = args.capacity_calibration if args.renderer == "custom" else args.flashgs_demand_survey
    if artifact_path is None:
        option = "--capacity-calibration" if args.renderer == "custom" else "--flashgs-demand-survey"
        raise ValueError(f"Timed {args.renderer} runs require {option}.")
    artifact_path = artifact_path.resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("pass") is not True:
        raise ValueError("Capacity artifact pass=false.")
    scene = payload.get("scene") or {}
    if scene.get("sha256") != args.scene_sha256 or scene.get("gaussian_count") != HOME_SCAN_COUNT:
        raise ValueError("Capacity artifact scene differs.")
    equation = payload.get("equation_contract") or {}
    if (
        equation.get("semantic_topology") != args.semantic_topology
        or equation.get("gaussian_support_sigma") != MATCHED_GAUSSIAN_SUPPORT_SIGMA
        or any(equation.get(field) != expected for field, expected in MATCHED_PROJECTION_RULES.items())
    ):
        raise ValueError("Capacity artifact equation contract differs.")
    config = payload.get("calibration_config") or payload.get("survey_config") or {}
    for field, expected in (
        ("capacity_headroom", args.capacity_headroom),
        ("initial_visible_per_view", args.initial_visible_per_view),
        ("initial_intersections_per_view", args.initial_intersections_per_view),
        ("flashgs_initial_intersections", args.flashgs_initial_intersections),
        ("custom_depth_buckets", args.custom_depth_buckets),
        ("custom_depth_bucket_group", args.custom_depth_bucket_group),
        ("custom_max_physical_views", args.custom_max_physical_views),
    ):
        if config.get(field) != expected:
            raise ValueError(f"Capacity artifact config {field} differs.")
    environment = payload.get("environment") or {}
    current_runtime = {
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "compute_capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                gpu_uuid,
            ]
        ),
    }
    if any(environment.get(field) != expected for field, expected in current_runtime.items()):
        raise ValueError("Capacity artifact runtime identity differs.")
    if source_identity(environment.get("source_provenance") or {}) != source_identity(source_provenance):
        raise ValueError("Capacity artifact source identity differs.")
    if not same_artifact(environment.get("flashgs_adapter_attestation"), adapter_attestation):
        raise ValueError("Capacity artifact adapter attestation differs.")
    verify_node_occupancy_evidence(environment.get("node_occupancy"), expected_gpu_uuid=gpu_uuid)

    if args.renderer == "flashgs":
        if (
            payload.get("schema_version") != FLASHGS_DEMAND_SURVEY_SCHEMA
            or payload.get("mode") != "flashgs-demand-survey-only"
            or payload.get("renderer") != "flashgs"
            or payload.get("timing_valid") is not False
            or payload.get("render_outputs_valid") is not False
        ):
            raise ValueError("FlashGS demand-survey identity differs.")
        camera = payload.get("camera_contract") or {}
        if (
            camera.get("batch") != PRIMARY_BATCHES[-1]
            or camera.get("trajectory_id") != PRIMARY_TRAJECTORY_IDS[PRIMARY_BATCHES[-1]]
            or camera.get("timesteps") != 108
            or camera.get("width") != trajectory.width
            or camera.get("height") != trajectory.height
        ):
            raise ValueError("FlashGS demand-survey B1024 camera contract differs.")
        verify_flashgs_demand_counter_source_audit(
            payload.get("counter_source_audit") or {},
            project_root=PROJECT_ROOT,
        )
        prefix_proof = verify_trajectory_prefix_proof(payload.get("trajectory_prefix_proof") or {})
        recorded_contract = (prefix_proof.get("contracts") or {}).get(str(trajectory.batch)) or {}
        if recorded_contract.get("trajectory_id") != trajectory.trajectory_id or recorded_contract.get(
            "artifacts"
        ) != trajectory_artifacts(args.trajectory):
            raise ValueError("Timed FlashGS trajectory is not the survey-bound prefix contract.")
        observation = payload.get("demand_observation") or {}
        visible = observation.get("per_camera_max_visible_gaussians")
        generated = observation.get("per_camera_max_generated_intersections")
        overflow = observation.get("per_camera_max_intersection_overflow")
        if (
            observation.get("schema_version") != "flashgs-matched-flashgs-demand-observation-v1"
            or observation.get("counter_storage") != "cuda-int64-per-camera"
            or observation.get("frames_observed") != 108
            or observation.get("camera_count") != PRIMARY_BATCHES[-1]
            or not isinstance(visible, list)
            or not isinstance(generated, list)
            or not isinstance(overflow, list)
            or len(visible) != PRIMARY_BATCHES[-1]
            or len(generated) != PRIMARY_BATCHES[-1]
            or len(overflow) != PRIMARY_BATCHES[-1]
            or any(not isinstance(value, int) or value < 0 for value in (*visible, *generated, *overflow))
        ):
            raise ValueError("FlashGS demand-survey device observation is malformed.")
        recomputed = derive_flashgs_prefix_capacities(
            visible,
            generated,
            batches=PRIMARY_BATCHES,
            headroom=args.capacity_headroom,
        )
        if payload.get("derived_batch_capacities") != recomputed:
            raise ValueError("FlashGS demand-survey derived capacities changed.")
        selected = recomputed.get(str(trajectory.batch)) or {}
        intersections = selected.get("installed_intersections_per_camera")
        if not isinstance(intersections, int) or intersections <= 0:
            raise ValueError("FlashGS demand survey has no batch-specific capacity.")
        return (
            payload,
            artifact_record(artifact_path),
            {"visible_records": None, "intersections": intersections},
        )

    if (
        payload.get("schema_version") != CAPACITY_CALIBRATION_SCHEMA
        or payload.get("mode") != "capacity-calibration-only"
        or payload.get("renderer") != "custom"
        or payload.get("calibration_output_contract") != "full"
    ):
        raise ValueError("Custom capacity calibration identity differs.")
    camera = payload.get("camera_contract") or {}
    if (
        camera.get("trajectory_id") != trajectory.trajectory_id
        or camera.get("batch") != trajectory.batch
        or camera.get("timesteps") != trajectory.timesteps
        or camera.get("width") != trajectory.width
        or camera.get("height") != trajectory.height
        or camera.get("measured_start") != args.warmup_frames
    ):
        raise ValueError("Custom capacity calibration camera contract differs.")
    capacity = payload.get("capacity") or {}
    expected_physical_batch = (
        min(
            trajectory.batch,
            args.custom_max_physical_views or trajectory.batch,
        )
        if args.renderer == "custom"
        else 1
    )
    expected_scope = (
        "physical-camera-chunk-reused-workspace"
        if expected_physical_batch < trajectory.batch
        else "batch-global-workspace"
    )
    if (
        capacity.get("schema_version") != "flashgs-matched-capacity-v1"
        or capacity.get("capacity_source") != "full-trajectory-device-counter-calibration"
        or capacity.get("preflight_frames_per_pass") != trajectory.timesteps
        or capacity.get("preflight_measured_frames_per_pass") != trajectory.timesteps - args.warmup_frames
        or capacity.get("logical_batch") != trajectory.batch
        or capacity.get("physical_batch") != expected_physical_batch
        or capacity.get("native_submissions_per_logical_batch") != math.ceil(trajectory.batch / expected_physical_batch)
        or capacity.get("intersection_capacity_scope") != expected_scope
    ):
        raise ValueError("Custom capacity calibration protocol differs.")
    verified = capacity.get("verified_preflight") or {}
    if verified.get("max_visible_overflow", 0) != 0 or verified.get("max_intersection_overflow") != 0:
        raise ValueError("Custom capacity calibration overflowed.")
    installed = capacity.get("installed") or {}
    visible = installed.get("visible_records")
    intersections = installed.get("intersections")
    if not isinstance(intersections, int) or intersections <= 0:
        raise ValueError("Capacity calibration has no positive intersection capacity.")
    if not isinstance(visible, int) or visible <= 0:
        raise ValueError("Custom capacity calibration has no visible capacity.")
    if (payload.get("output_validation") or {}).get("pass") is not True:
        raise ValueError("Capacity calibration output validation failed.")
    return (
        payload,
        artifact_record(artifact_path),
        {
            "visible_records": visible,
            "intersections": intersections,
        },
    )


def validate_calibration_native_extension(
    calibration: dict[str, Any],
    *,
    native_extension: dict[str, Any],
    native_build_contract: dict[str, Any] | None,
) -> None:
    environment = calibration.get("environment") or {}
    if not same_artifact(environment.get("native_extension"), native_extension):
        raise ValueError("Capacity calibration native extension differs.")
    if environment.get("native_build_contract") != native_build_contract:
        raise ValueError("Capacity calibration native build contract differs.")


def validate_last_outputs(outputs: dict[str, torch.Tensor], output_contract: str) -> dict[str, Any]:
    rgb = outputs["rgb"]
    checks: dict[str, Any] = {
        "rgb_dtype": str(rgb.dtype),
        "rgb_cuda": rgb.is_cuda,
        "rgb_contiguous": rgb.is_contiguous(),
        "rgb_finite": bool(torch.isfinite(rgb).all().item()),
        "rgb_min": float(rgb.min().item()),
        "rgb_max": float(rgb.max().item()),
    }
    passed = (
        checks["rgb_cuda"]
        and checks["rgb_contiguous"]
        and checks["rgb_finite"]
        and checks["rgb_dtype"] == "torch.float32"
        and checks["rgb_min"] >= -1.0e-5
        and checks["rgb_max"] <= 1.00001
        and checks["rgb_max"] > 1.0e-6
    )
    if output_contract == "full":
        alpha = outputs["alpha"]
        depth = outputs["depth"]
        semantic = outputs["semantic_id"]
        foreground = alpha[..., 0] > 1.0e-8
        semantic_foreground = alpha[..., 0] >= 0.01
        checks.update(
            {
                "alpha_dtype": str(alpha.dtype),
                "depth_dtype": str(depth.dtype),
                "semantic_dtype": str(semantic.dtype),
                "alpha_finite": bool(torch.isfinite(alpha).all().item()),
                "alpha_min": float(alpha.min().item()),
                "alpha_max": float(alpha.max().item()),
                "foreground_fraction": float(foreground.float().mean().item()),
                "foreground_depth_finite": bool(torch.isfinite(depth[..., 0][foreground]).all().item()),
                "background_depth_inf": bool(torch.isinf(depth[..., 0][~foreground]).all().item()),
                "foreground_semantic_nonnegative": bool((semantic[..., 0][semantic_foreground] >= 0).all().item()),
                "background_semantic_minus_one": bool((semantic[..., 0][~semantic_foreground] == -1).all().item()),
            }
        )
        passed = passed and all(
            (
                checks["alpha_dtype"] == "torch.float32",
                checks["depth_dtype"] == "torch.float32",
                checks["semantic_dtype"] == "torch.int64",
                checks["alpha_finite"],
                checks["alpha_min"] >= -1.0e-6,
                checks["alpha_max"] <= 1.000001,
                checks["foreground_fraction"] > 0.0,
                checks["foreground_depth_finite"],
                checks["background_depth_inf"],
                checks["foreground_semantic_nonnegative"],
                checks["background_semantic_minus_one"],
            )
        )
    checks["pass"] = bool(passed)
    return checks


def capture_fidelity_outputs(
    path: Path,
    *,
    service: RendererService,
    outputs: dict[str, torch.Tensor],
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    scene_ids: torch.Tensor,
    trajectory_id: str,
) -> dict[str, Any]:
    """Capture the frozen sparse fidelity suite outside measurement."""

    started = time.perf_counter()
    selection = primary_fidelity_selection(int(viewmats.shape[1]), trajectory_timesteps=int(viewmats.shape[0]))
    final_step = int(viewmats.shape[0]) - 1
    selected_steps = [final_step, *sorted({step for step, _camera in selection if step != final_step})]
    captured_by_pair: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for step in selected_steps:
        if step != final_step:
            outputs = service.render(viewmats[step], intrinsics[step], scene_ids, outputs=outputs)
            service.synchronize()
        for selected_step, camera_index in selection:
            if selected_step != step:
                continue
            captured_by_pair[(step, camera_index)] = {
                name: tensor[camera_index : camera_index + 1].detach().cpu().numpy() for name, tensor in outputs.items()
            }
    arrays = {name: np.concatenate([captured_by_pair[pair][name] for pair in selection], axis=0) for name in outputs}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **arrays,
        steps=np.asarray([step for step, _camera in selection], dtype=np.int64),
        camera_indices=np.asarray([camera for _step, camera in selection], dtype=np.int64),
        trajectory_id=np.asarray(trajectory_id),
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "selection_pairs": [list(pair) for pair in selection],
        "readback_and_write_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    executor_lock = ensure_cooperative_executor_lock()
    occupancy_path = args.output.with_suffix(".node-occupancy.json")
    preflight_snapshot = capture_node_snapshot()
    preflight_occupancy_failures = occupancy_failures(
        preflight_snapshot,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=False,
    )
    occupancy_payload: dict[str, Any] = {
        "schema_version": "flashgs-matched-node-occupancy-v2",
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "executor_control": {
            "cooperative_node_wide_lock": executor_lock,
            "scope": "all-visible-gpus",
            "limitation": (
                "The lock coordinates participating repository runners; "
                "periodic NVIDIA process samples detect but cannot make a "
                "continuous absence claim about uncooperative processes."
            ),
        },
        "preflight": preflight_snapshot,
        "preflight_failures": preflight_occupancy_failures,
        "sampled_compute_process_telemetry": None,
        "postflight": None,
        "postflight_failures": None,
        "pass": False,
    }
    occupancy_path.parent.mkdir(parents=True, exist_ok=True)
    occupancy_path.write_text(
        json.dumps(occupancy_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if preflight_occupancy_failures:
        raise RuntimeError("Shared RTX node occupancy preflight failed: " + "; ".join(preflight_occupancy_failures))
    occupancy_sampler = ComputeProcessSampler(
        expected_gpu_uuid=args.expected_gpu_uuid,
        allowed_pids={os.getpid()},
    ).start()
    source_provenance = load_verified_source_manifest(
        args.source_manifest,
        project_root=PROJECT_ROOT,
    )
    adapter_attestation = None
    if args.flashgs_adapter_attestation is not None:
        load_verified_flashgs_adapter_attestation(
            args.flashgs_adapter_attestation,
            source_provenance=source_provenance,
            project_root=PROJECT_ROOT,
        )
        adapter_attestation = artifact_record(args.flashgs_adapter_attestation)
    if args.custom_max_physical_views is not None and args.custom_max_physical_views <= 0:
        raise ValueError("custom_max_physical_views must be positive.")
    if args.independent_trial <= 0:
        raise ValueError("independent_trial must be positive.")
    if args.renderer != "custom" and args.custom_max_physical_views is not None:
        raise ValueError("custom_max_physical_views applies only to the custom renderer.")
    if args.fixed_visible_capacity is not None or args.fixed_intersection_capacity is not None:
        raise ValueError(
            "Direct fixed-capacity arguments are no longer accepted; use a hash-bound --capacity-calibration artifact."
        )
    if args.capacity_calibration_only and args.flashgs_demand_survey_only:
        raise ValueError("Custom calibration and FlashGS demand-survey modes are mutually exclusive.")
    if args.capacity_calibration_only:
        if args.renderer != "custom":
            raise ValueError("The zero-overflow capacity calibration applies only to Custom.")
        if args.capacity_calibration is not None or args.flashgs_demand_survey is not None:
            raise ValueError("Capacity calibration mode cannot consume another capacity artifact.")
    elif args.flashgs_demand_survey_only:
        if args.renderer != "flashgs":
            raise ValueError("The demand-survey mode applies only to FlashGS.")
        if args.capacity_calibration is not None or args.flashgs_demand_survey is not None:
            raise ValueError("Demand-survey mode cannot consume another capacity artifact.")
        if args.capture_last_output:
            raise ValueError("Demand-survey outputs are invalid and cannot be captured.")
    elif args.renderer == "custom":
        if args.capacity_calibration is None or args.flashgs_demand_survey is not None:
            raise ValueError("Timed Custom runs require only --capacity-calibration.")
    elif args.flashgs_demand_survey is None or args.capacity_calibration is not None:
        raise ValueError("Timed FlashGS runs require only --flashgs-demand-survey.")
    if args.capacity_calibration_only and args.output_contract != "full":
        raise ValueError("Capacity calibration must use the full output contract.")
    if args.flashgs_demand_survey_only and args.output_contract != "full":
        raise ValueError("FlashGS demand survey uses the full compositor path.")
    if (args.capacity_calibration_only or args.flashgs_demand_survey_only) and args.profile_control:
        raise ValueError("Capacity-only modes are not profiler controls.")
    if bool(args.prefix_trajectory) != bool(args.flashgs_demand_survey_only):
        raise ValueError("--prefix-trajectory is required only for FlashGS demand-survey mode.")
    if args.measured_frames < 100 and not args.profile_control:
        raise ValueError("At least 100 synchronized measured frames are required.")
    if args.profile_control and args.measured_frames <= 0:
        raise ValueError("Profile controls require at least one measured frame.")
    if args.warmup_frames != 8:
        raise ValueError("The matched timing contract requires exactly 8 warmups.")
    if args.capacity_headroom != 1.05:
        raise ValueError("The matched capacity contract requires 1.05 headroom.")
    if not torch.cuda.is_available():
        raise RuntimeError("The matched benchmark requires CUDA.")
    torch.cuda.init()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    gpu_uuid = current_process_gpu_uuid()
    if not is_nvidia_gpu_uuid(args.expected_gpu_uuid):
        raise ValueError("--expected-gpu-uuid is not a canonical NVIDIA GPU UUID.")
    if gpu_uuid != args.expected_gpu_uuid:
        raise RuntimeError(
            "Active CUDA process GPU UUID differs from the explicit contract: "
            f"{gpu_uuid!r} != {args.expected_gpu_uuid!r}."
        )
    headline_gpu_pass = gpu_name == HEADLINE_GPU_NAME
    if not headline_gpu_pass and not args.allow_nonheadline_gpu:
        raise RuntimeError(
            f"Headline benchmark requires {HEADLINE_GPU_NAME}; current device "
            f"is {gpu_name} ({gpu_uuid}). The exact UUID remains bound by "
            "--expected-gpu-uuid."
        )
    trajectory = load_trajectory(args.trajectory)
    if trajectory.scene_sha256 != args.scene_sha256:
        raise ValueError("Trajectory and scene SHA-256 contracts differ.")
    measured_end = args.warmup_frames + args.measured_frames
    valid_timestep_count = (
        trajectory.timesteps >= measured_end if args.profile_control else trajectory.timesteps == measured_end
    )
    if not valid_timestep_count:
        raise ValueError(
            f"Trajectory has {trajectory.timesteps} steps; expected "
            f"{'at least ' if args.profile_control else ''}{measured_end}."
        )
    if trajectory.width != 128 or trajectory.height != 128:
        raise ValueError("Primary FlashGS comparison is fixed at 128x128.")
    if trajectory.motion_classification != "independently-changing-camera-batch":
        raise ValueError("Trajectory is not the independently-changing primary contract.")
    measured_changes = np.max(
        np.abs(
            np.diff(
                trajectory.viewmats[args.warmup_frames - 1 : measured_end],
                axis=0,
            )
        ),
        axis=(2, 3),
    )
    unchanged_pairs = int(np.count_nonzero(measured_changes == 0.0))
    if unchanged_pairs:
        raise ValueError(f"Primary contract contains {unchanged_pairs} unchanged camera/frame pairs.")

    trajectory_prefix_proof: dict[str, Any] | None = None
    counter_source_audit: dict[str, Any] | None = None
    if args.flashgs_demand_survey_only:
        if (
            trajectory.batch != PRIMARY_BATCHES[-1]
            or trajectory.trajectory_id != PRIMARY_TRAJECTORY_IDS[PRIMARY_BATCHES[-1]]
            or trajectory.timesteps != 108
        ):
            raise ValueError("FlashGS demand survey requires the frozen primary B1024 contract.")
        prefix_paths: dict[int, Path] = {}
        for path in args.prefix_trajectory:
            candidate = load_trajectory(path)
            if candidate.batch in prefix_paths:
                raise ValueError(f"Duplicate B{candidate.batch} prefix trajectory.")
            prefix_paths[candidate.batch] = path
        trajectory_prefix_proof = prove_trajectory_prefixes(
            prefix_paths,
            expected_batches=PRIMARY_BATCHES,
            expected_timesteps=108,
        )
        if trajectory_prefix_proof["contracts"][str(PRIMARY_BATCHES[-1])]["artifacts"] != trajectory_artifacts(
            args.trajectory
        ):
            raise ValueError("Survey trajectory differs from the canonical prefix-proof B1024 artifact.")
        counter_source_audit = audit_flashgs_demand_counter_source(PROJECT_ROOT)

    calibration_payload: dict[str, Any] | None = None
    calibration_artifact: dict[str, Any] | None = None
    installed_capacity: dict[str, int | None] | None = None
    if not (args.capacity_calibration_only or args.flashgs_demand_survey_only):
        (
            calibration_payload,
            calibration_artifact,
            installed_capacity,
        ) = load_timed_capacity_artifact(
            args,
            trajectory=trajectory,
            source_provenance=source_provenance,
            adapter_attestation=adapter_attestation,
            gpu_name=gpu_name,
            gpu_uuid=gpu_uuid,
        )

    scene, scene_metadata = load_home_scan(
        args.scene_path,
        args.scene_sha256,
        args.semantic_topology,
    )
    camera_upload_started = time.perf_counter()
    viewmats = torch.from_numpy(trajectory.viewmats).to("cuda")
    intrinsics = torch.from_numpy(trajectory.intrinsics).to("cuda")
    scene_ids = torch.full((trajectory.batch,), SCENE_ID, device="cuda", dtype=torch.int64)
    torch.cuda.synchronize()
    camera_upload_seconds = time.perf_counter() - camera_upload_started
    backend = build_backend(
        args,
        batch=trajectory.batch,
        near_plane=trajectory.near_plane,
        far_plane=trajectory.far_plane,
        installed_capacity=installed_capacity,
    )
    output_names = RGB_OUTPUTS if args.output_contract == "rgb" else FULL_OUTPUTS
    service = RendererService(
        backend,
        height=trajectory.height,
        width=trajectory.width,
        outputs=output_names,
        max_views=trajectory.batch,
    )
    service.initialize(stage=None, device="cuda")
    service.load_scene(SCENE_ID, **scene)
    native_module_path = getattr(backend._native, "__file__", None)
    if not native_module_path:
        raise RuntimeError("Loaded renderer native extension has no file identity.")
    native_extension = artifact_record(native_module_path)
    native_build_ninja_path = Path(native_module_path).resolve().parent / "build.ninja"
    native_build_ninja = artifact_record(native_build_ninja_path) if native_build_ninja_path.is_file() else None
    native_build_contract = getattr(backend._native, "__vgr_build_contract__", None)

    if args.flashgs_demand_survey_only:
        assert isinstance(backend, FlashGSBackend)
        assert trajectory_prefix_proof is not None
        assert counter_source_audit is not None
        demand_observation = survey_flashgs_capacity_demand(
            service,
            backend,
            viewmats,
            intrinsics,
            scene_ids,
        )
        derived_capacities = derive_flashgs_prefix_capacities(
            demand_observation["per_camera_max_visible_gaussians"],
            demand_observation["per_camera_max_generated_intersections"],
            batches=PRIMARY_BATCHES,
            headroom=args.capacity_headroom,
        )
        sampled_compute_process_telemetry = occupancy_sampler.stop()
        postflight_snapshot = capture_node_snapshot()
        postflight_occupancy_failures = occupancy_failures(
            postflight_snapshot,
            expected_gpu_uuid=args.expected_gpu_uuid,
            allow_current_gpu_process=True,
        )
        occupancy_payload.update(
            {
                "postflight": postflight_snapshot,
                "postflight_failures": postflight_occupancy_failures,
                "sampled_compute_process_telemetry": sampled_compute_process_telemetry,
                "pass": bool(
                    not postflight_occupancy_failures
                    and sampled_compute_process_telemetry["pass"]
                    and executor_lock["pass"]
                ),
            }
        )
        occupancy_path.write_text(
            json.dumps(occupancy_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        occupancy_artifact = artifact_record(occupancy_path)
        survey_result = {
            "schema_version": FLASHGS_DEMAND_SURVEY_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "flashgs-demand-survey-only",
            "renderer": "flashgs",
            "pass": bool(
                occupancy_payload["pass"]
                and source_provenance["dirty"] is False
                and trajectory_prefix_proof["pass"]
                and counter_source_audit["pass"]
                and (headline_gpu_pass or args.allow_nonheadline_gpu)
            ),
            "timing_valid": False,
            "timing_invalid_reason": "capacity-demand-survey-is-untimed",
            "render_outputs_valid": False,
            "render_outputs_invalid_reason": (
                "low-capacity bounded writes may drop intersections; only pre-write device demand counters are valid"
            ),
            "scene": scene_metadata,
            "camera_contract": {
                "path": str(args.trajectory),
                "trajectory_id": trajectory.trajectory_id,
                "batch": trajectory.batch,
                "timesteps": trajectory.timesteps,
                "width": trajectory.width,
                "height": trajectory.height,
                "motion_classification": trajectory.motion_classification,
                "artifacts": trajectory_artifacts(args.trajectory),
            },
            "trajectory_prefix_proof": trajectory_prefix_proof,
            "equation_contract": {
                "precision": "float32",
                "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
                **MATCHED_PROJECTION_RULES,
                "semantic_topology": args.semantic_topology,
                "rendered_frame_cache": False,
                "projection_cache": False,
            },
            "survey_config": {
                "capacity_headroom": args.capacity_headroom,
                "initial_visible_per_view": args.initial_visible_per_view,
                "initial_intersections_per_view": args.initial_intersections_per_view,
                "flashgs_initial_intersections": args.flashgs_initial_intersections,
                "custom_depth_buckets": args.custom_depth_buckets,
                "custom_depth_bucket_group": args.custom_depth_bucket_group,
                "custom_max_physical_views": None,
                "installed_probe_intersections_per_camera": backend.max_intersections,
                "workspace_reuse": "per-camera-reused-workspace",
                "output_contract_exercised_but_invalid": "full",
            },
            "counter_source_audit": counter_source_audit,
            "demand_observation": demand_observation,
            "derived_batch_capacities": derived_capacities,
            "environment": {
                "gpu_name": gpu_name,
                "gpu_uuid": gpu_uuid,
                "compute_capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
                "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "driver": command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                        "-i",
                        gpu_uuid,
                    ]
                ),
                "source_provenance": source_provenance,
                "native_extension": native_extension,
                "native_build_ninja": native_build_ninja,
                "native_build_contract": native_build_contract,
                "node_occupancy": occupancy_artifact,
                "flashgs_adapter_attestation": adapter_attestation,
                "flashgs_upstream_commit": backend.execution_stats["upstream_commit"],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(survey_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        service.shutdown()
        print(
            "FLASHGS_MATCHED_DEMAND_SURVEY_OK "
            + json.dumps(
                {
                    "output": str(args.output),
                    "pass": survey_result["pass"],
                    "batch": trajectory.batch,
                    "probe_capacity": backend.max_intersections,
                },
                sort_keys=True,
            )
        )
        return

    if args.capacity_calibration_only:
        capacity, calibration_outputs = calibrate_capacity(
            args,
            service,
            backend,
            viewmats,
            intrinsics,
            scene_ids,
            measured_start=args.warmup_frames,
        )
        if isinstance(backend, CustomCudaBackend):
            calibration_physical_batch = min(
                trajectory.batch,
                backend.max_physical_views or trajectory.batch,
            )
        else:
            calibration_physical_batch = 1
        capacity["logical_batch"] = trajectory.batch
        capacity["physical_batch"] = calibration_physical_batch
        capacity["native_submissions_per_logical_batch"] = math.ceil(trajectory.batch / calibration_physical_batch)
        capacity["camera_coverage"] = "complete-contiguous-logical-batch"
        output_validation = validate_last_outputs(calibration_outputs, "full")
        sampled_compute_process_telemetry = occupancy_sampler.stop()
        postflight_snapshot = capture_node_snapshot()
        postflight_occupancy_failures = occupancy_failures(
            postflight_snapshot,
            expected_gpu_uuid=args.expected_gpu_uuid,
            allow_current_gpu_process=True,
        )
        occupancy_payload.update(
            {
                "postflight": postflight_snapshot,
                "postflight_failures": postflight_occupancy_failures,
                "sampled_compute_process_telemetry": (sampled_compute_process_telemetry),
                "pass": bool(
                    not postflight_occupancy_failures
                    and sampled_compute_process_telemetry["pass"]
                    and executor_lock["pass"]
                ),
            }
        )
        occupancy_path.write_text(
            json.dumps(occupancy_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        occupancy_artifact = artifact_record(occupancy_path)
        calibration_result = {
            "schema_version": CAPACITY_CALIBRATION_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "capacity-calibration-only",
            "renderer": args.renderer,
            "calibration_output_contract": "full",
            "pass": bool(
                output_validation["pass"]
                and capacity["verified_preflight"].get("max_visible_overflow", 0) == 0
                and capacity["verified_preflight"]["max_intersection_overflow"] == 0
                and occupancy_payload["pass"]
                and source_provenance["dirty"] is False
                and (headline_gpu_pass or args.allow_nonheadline_gpu)
            ),
            "scene": scene_metadata,
            "camera_contract": {
                "path": str(args.trajectory),
                "trajectory_id": trajectory.trajectory_id,
                "batch": trajectory.batch,
                "timesteps": trajectory.timesteps,
                "width": trajectory.width,
                "height": trajectory.height,
                "measured_start": args.warmup_frames,
                "motion_classification": trajectory.motion_classification,
            },
            "equation_contract": {
                "precision": "float32",
                "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
                **MATCHED_PROJECTION_RULES,
                "semantic_topology": args.semantic_topology,
                "rendered_frame_cache": False,
                "projection_cache": False,
            },
            "calibration_config": {
                "capacity_headroom": args.capacity_headroom,
                "initial_visible_per_view": args.initial_visible_per_view,
                "initial_intersections_per_view": (args.initial_intersections_per_view),
                "flashgs_initial_intersections": (args.flashgs_initial_intersections),
                "custom_depth_buckets": args.custom_depth_buckets,
                "custom_depth_bucket_group": args.custom_depth_bucket_group,
                "custom_max_physical_views": (args.custom_max_physical_views),
            },
            "capacity": capacity,
            "output_validation": output_validation,
            "environment": {
                "gpu_name": gpu_name,
                "gpu_uuid": gpu_uuid,
                "compute_capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
                "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "driver": command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                        "-i",
                        gpu_uuid,
                    ]
                ),
                "source_provenance": source_provenance,
                "native_extension": native_extension,
                "native_build_ninja": native_build_ninja,
                "native_build_contract": native_build_contract,
                "node_occupancy": occupancy_artifact,
                "flashgs_adapter_attestation": adapter_attestation,
                "flashgs_upstream_commit": (
                    backend.execution_stats["upstream_commit"]
                    if isinstance(backend, FlashGSBackend)
                    else "cdfc4e4002318423eda356eed02df8e01fa32cb6"
                ),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(calibration_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        service.shutdown()
        print(
            "FLASHGS_MATCHED_CAPACITY_CALIBRATION_OK "
            + json.dumps(
                {
                    "output": str(args.output),
                    "pass": calibration_result["pass"],
                    "renderer": args.renderer,
                    "batch": trajectory.batch,
                    "installed": capacity["installed"],
                },
                sort_keys=True,
            )
        )
        return

    assert calibration_payload is not None
    assert calibration_artifact is not None
    assert installed_capacity is not None
    validate_calibration_native_extension(
        calibration_payload,
        native_extension=native_extension,
        native_build_contract=native_build_contract,
    )
    timed_setup = {
        (
            "initial_workspace_from_calibration"
            if isinstance(backend, CustomCudaBackend)
            else "initial_workspace_from_demand_survey"
        ): True,
        "prepare_outputs_calls": 1,
        "warmup_frames": args.warmup_frames,
        "trajectory_preflight_frames": 0,
        "explicit_capacity_reservation_calls": 0,
        "empty_cache_calls": 0,
    }
    if isinstance(backend, CustomCudaBackend):
        calibrated_capacity = calibration_payload["capacity"]
        capacity = {
            "schema_version": CAPACITY_CONSUMPTION_SCHEMA,
            "capacity_source": "hash-bound-calibration-artifact",
            "calibration_artifact": calibration_artifact,
            "installed": dict(installed_capacity),
            "intersection_capacity_scope": calibrated_capacity["intersection_capacity_scope"],
            "verified_preflight": calibrated_capacity["verified_preflight"],
            "calibration_preflight_frames_per_pass": calibrated_capacity["preflight_frames_per_pass"],
            "calibration_preflight_passes": calibrated_capacity["preflight_passes"],
            "timed_setup": timed_setup,
        }
    else:
        selected_demand = calibration_payload["derived_batch_capacities"][str(trajectory.batch)]
        capacity = {
            "schema_version": FLASHGS_DEMAND_SURVEY_CONSUMPTION_SCHEMA,
            "capacity_source": "hash-bound-b1024-demand-survey-batch-prefix",
            "demand_survey_artifact": calibration_artifact,
            "installed": dict(installed_capacity),
            "intersection_capacity_scope": "per-camera-reused-workspace",
            "survey_prefix_demand": selected_demand,
            "survey_canonical_batch": PRIMARY_BATCHES[-1],
            "installed_from_batch_specific_prefix": trajectory.batch,
            "survey_render_outputs_valid": False,
            "timed_setup": timed_setup,
        }
    outputs = service.prepare_outputs(trajectory.batch)
    timed_capacity_audit = create_timed_capacity_audit(
        backend,
        expected_frames=measured_end,
        measured_start=args.warmup_frames,
    )
    if isinstance(backend, CustomCudaBackend):
        physical_batch = min(
            trajectory.batch,
            backend.max_physical_views or trajectory.batch,
        )
    else:
        physical_batch = 1
    capacity["logical_batch"] = trajectory.batch
    capacity["physical_batch"] = physical_batch
    capacity["native_submissions_per_logical_batch"] = math.ceil(trajectory.batch / physical_batch)
    capacity["camera_coverage"] = "complete-contiguous-logical-batch"
    for step in range(args.warmup_frames):
        outputs = service.render(viewmats[step], intrinsics[step], scene_ids, outputs=outputs)
        update_timed_capacity_audit(timed_capacity_audit, backend, measured=False)
        service.synchronize()
    # Reuse the same timing events for every sample. Creating CUDA events in
    # the measured loop would violate the steady-state allocation contract.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    end_event.record()
    end_event.synchronize()
    # Preserve the warmed allocator state. Emptying the cache here would turn
    # the first measured render into allocator re-population rather than the
    # steady-state workload this benchmark claims to measure.
    torch.cuda.reset_peak_memory_stats()
    baseline_memory_samples = [steady_state_memory_sample(gpu_uuid) for _ in range(MEMORY_BASELINE_SAMPLE_COUNT)]
    memory_before = baseline_memory_samples[-1]
    flashgs_execution_before = dict(backend.execution_stats) if isinstance(backend, FlashGSBackend) else None
    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    submission_samples: list[float] = []
    measured_started = time.perf_counter()
    torch.cuda.nvtx.range_push(f"flashgs-matched-capture/{args.renderer}/{args.output_contract}/b{trajectory.batch}")
    for measured_index, step in enumerate(range(args.warmup_frames, measured_end)):
        wall_started = time.perf_counter()
        torch.cuda.nvtx.range_push(
            f"flashgs-matched/{args.renderer}/{args.output_contract}/b{trajectory.batch}/frame-{measured_index:03d}"
        )
        start_event.record()
        submission_started = time.perf_counter()
        outputs = service.render(viewmats[step], intrinsics[step], scene_ids, outputs=outputs)
        submission_samples.append((time.perf_counter() - submission_started) * 1000.0)
        end_event.record()
        end_event.synchronize()
        torch.cuda.nvtx.range_pop()
        gpu_samples.append(float(start_event.elapsed_time(end_event)))
        wall_samples.append((time.perf_counter() - wall_started) * 1000.0)
        # This preallocated device-counter reduction is deliberately after the
        # CUDA event and wall sample, then synchronized before the next sample.
        update_timed_capacity_audit(timed_capacity_audit, backend, measured=True)
        service.synchronize()
    torch.cuda.nvtx.range_pop()
    service.synchronize()
    measured_loop_with_audit_seconds = time.perf_counter() - measured_started
    measured_seconds = sum(wall_samples) / 1000.0
    timed_capacity_verification = finalize_timed_capacity_audit(
        timed_capacity_audit,
        backend,
    )
    capacity["timed_verification"] = timed_capacity_verification
    # Snapshot every memory source before output validation. Validation uses
    # temporary CUDA tensors and reductions, so including it here would make a
    # valid steady-state render look like measured-loop allocation growth.
    post_measurement_memory = steady_state_memory_sample(gpu_uuid)
    flashgs_execution_after_measurement = dict(backend.execution_stats) if isinstance(backend, FlashGSBackend) else None
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    steady_state_fairness = assess_steady_state_memory(
        baseline_memory_samples,
        post_measurement_memory,
    )

    validation_nvtx_range = f"flashgs-matched-validation/{args.renderer}/{args.output_contract}/b{trajectory.batch}"
    validation_started = time.perf_counter()
    torch.cuda.nvtx.range_push(validation_nvtx_range)
    try:
        output_validation = validate_last_outputs(outputs, args.output_contract)
    finally:
        torch.cuda.nvtx.range_pop()
    validation_seconds = time.perf_counter() - validation_started
    output_bytes = sum(tensor.numel() * tensor.element_size() for tensor in outputs.values())
    output_capture = None
    if args.capture_last_output:
        output_capture = capture_fidelity_outputs(
            args.output.with_suffix(".fidelity-capture.npz"),
            service=service,
            outputs=outputs,
            viewmats=viewmats,
            intrinsics=intrinsics,
            scene_ids=scene_ids,
            trajectory_id=trajectory.trajectory_id,
        )
    gpu_timing = distribution(gpu_samples)
    wall_timing = distribution(wall_samples)
    submission_timing = distribution(submission_samples)
    total_images = args.measured_frames * trajectory.batch
    total_megapixels = total_images * trajectory.width * trajectory.height / 1.0e6
    total_gpu_seconds = sum(gpu_samples) / 1000.0
    if isinstance(backend, FlashGSBackend):
        flashgs_execution_after = flashgs_execution_after_measurement
        assert flashgs_execution_before is not None
        assert flashgs_execution_after is not None
        backend_execution = {
            **flashgs_execution_after,
            "measured_render_requests": (
                int(flashgs_execution_after["render_requests"]) - int(flashgs_execution_before["render_requests"])
            ),
            "measured_native_camera_executions": (
                int(flashgs_execution_after["native_camera_executions"])
                - int(flashgs_execution_before["native_camera_executions"])
            ),
            "logical_batch": trajectory.batch,
            "physical_batch": 1,
            "native_submissions_per_logical_batch": trajectory.batch,
            "measured_native_batch_submissions": (args.measured_frames * trajectory.batch),
            "measured_camera_coverage": (args.measured_frames * trajectory.batch),
            "camera_order": "sequential-logical-order",
            "logical_batch_in_single_native_submission": (trajectory.batch == 1),
        }
    else:
        physical_batch = min(
            trajectory.batch,
            backend.max_physical_views or trajectory.batch,
        )
        native_submissions_per_request = math.ceil(trajectory.batch / physical_batch)
        measured_native_submissions = args.measured_frames * native_submissions_per_request
        measured_camera_executions = args.measured_frames * trajectory.batch
        backend_execution = {
            "render_requests": args.measured_frames,
            "native_camera_executions": measured_camera_executions,
            "native_batch_submissions": measured_native_submissions,
            "measured_render_requests": args.measured_frames,
            "measured_native_camera_executions": (measured_camera_executions),
            "true_native_batch": native_submissions_per_request == 1,
            "logical_batch_in_single_native_submission": (native_submissions_per_request == 1),
            "logical_batch": trajectory.batch,
            "physical_batch": physical_batch,
            "native_submissions_per_logical_batch": (native_submissions_per_request),
            "measured_native_batch_submissions": (measured_native_submissions),
            "measured_camera_coverage": (args.measured_frames * trajectory.batch),
            "camera_order": "contiguous-logical-order",
            "workspace_reuse": ("physical-camera-chunk" if native_submissions_per_request > 1 else "logical-batch"),
            "fixed_capacity_sort": backend.fixed_capacity_sort,
            "projection_cache_enabled": backend.enable_projection_cache,
            "compact_projection_cache": backend.compact_projection_cache,
        }
    memory = {
        "scene_bytes": backend.scene_bytes,
        "workspace_bytes": backend.workspace_bytes,
        "output_bytes": output_bytes,
        "torch_allocated_before_measurement": memory_before["torch"]["allocated_bytes"],
        "torch_allocated_after_measurement": post_measurement_memory["torch"]["allocated_bytes"],
        "torch_reserved_before_measurement": memory_before["torch"]["reserved_bytes"],
        "torch_reserved_after_measurement": post_measurement_memory["torch"]["reserved_bytes"],
        "torch_peak_allocated_bytes": peak_allocated_bytes,
        "torch_peak_reserved_bytes": peak_reserved_bytes,
        "torch_cumulative_counters_before_measurement": memory_before["torch"]["cumulative_counters"],
        "torch_cumulative_counters_after_measurement": (post_measurement_memory["torch"]["cumulative_counters"]),
        "torch_cumulative_counter_deltas": steady_state_fairness["torch_cumulative_counter_deltas"],
        "allocation_growth_bytes": steady_state_fairness["allocation_growth_bytes"],
        "reservation_growth_bytes": steady_state_fairness["reservation_growth_bytes"],
        "driver_process_memory_before": memory_before["driver"],
        "driver_process_memory_after": post_measurement_memory["driver"],
        "driver_process_memory_growth_mib": steady_state_fairness["driver_process_memory_growth_mib"],
        "baseline_samples": baseline_memory_samples,
        "post_measurement_sample": post_measurement_memory,
        "steady_state_fairness": steady_state_fairness,
        "measurement_boundary": (
            "baseline immediately before measured render loop; post-measurement "
            "Torch and NVML snapshots immediately after final synchronization "
            "and before validation"
        ),
    }
    expected_max_physical_views = primary_max_physical_views(
        args.renderer,
        trajectory.batch,
    )
    if args.renderer == "custom":
        primary_schedule_pass = (
            args.custom_max_physical_views == expected_max_physical_views
            and backend_execution["physical_batch"] == (expected_max_physical_views or trajectory.batch)
            and backend_execution["logical_batch_in_single_native_submission"] == (expected_max_physical_views is None)
        )
    else:
        primary_schedule_pass = args.custom_max_physical_views == expected_max_physical_views
    primary_workload_pass = bool(
        trajectory.batch in PRIMARY_BATCHES
        and trajectory.trajectory_id == PRIMARY_TRAJECTORY_IDS.get(trajectory.batch)
        and args.semantic_topology == REPRESENTATIVE_SEMANTIC_TOPOLOGY
        and primary_schedule_pass
    )
    sampled_compute_process_telemetry = occupancy_sampler.stop()
    postflight_snapshot = capture_node_snapshot()
    postflight_occupancy_failures = occupancy_failures(
        postflight_snapshot,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=True,
    )
    occupancy_payload.update(
        {
            "postflight": postflight_snapshot,
            "postflight_failures": postflight_occupancy_failures,
            "sampled_compute_process_telemetry": (sampled_compute_process_telemetry),
            "pass": bool(
                not postflight_occupancy_failures
                and sampled_compute_process_telemetry["pass"]
                and executor_lock["pass"]
            ),
        }
    )
    occupancy_path.write_text(
        json.dumps(occupancy_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    occupancy_artifact = artifact_record(occupancy_path)
    result = {
        "schema_version": RENDERER_RUN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "renderer": args.renderer,
        "output_contract": args.output_contract,
        "pass": bool(
            output_validation["pass"]
            and timed_capacity_verification["pass"]
            and steady_state_fairness["pass"]
            and occupancy_payload["pass"]
            and (headline_gpu_pass or args.allow_nonheadline_gpu)
        ),
        "headline_eligible": bool(
            headline_gpu_pass
            and primary_workload_pass
            and source_provenance["dirty"] is False
            and occupancy_payload["pass"]
            and adapter_attestation is not None
            and timed_capacity_verification["pass"]
            and args.capture_last_output
            and args.independent_trial == 1
            and not args.profile_control
        ),
        "primary_workload_eligible": primary_workload_pass,
        "profile_control": args.profile_control,
        "runner_config": {
            "warmup_frames": args.warmup_frames,
            "measured_frames": args.measured_frames,
            "capacity_headroom": args.capacity_headroom,
            "initial_visible_per_view": args.initial_visible_per_view,
            "initial_intersections_per_view": (args.initial_intersections_per_view),
            "flashgs_initial_intersections": (args.flashgs_initial_intersections),
            "fixed_visible_capacity": args.fixed_visible_capacity,
            "fixed_intersection_capacity": args.fixed_intersection_capacity,
            "capacity_calibration": (calibration_artifact if args.renderer == "custom" else None),
            "flashgs_demand_survey": (calibration_artifact if args.renderer == "flashgs" else None),
            "capacity_calibration_only": False,
            "flashgs_demand_survey_only": False,
            "custom_depth_buckets": args.custom_depth_buckets,
            "custom_depth_bucket_group": args.custom_depth_bucket_group,
            "custom_max_physical_views": args.custom_max_physical_views,
            "semantic_topology": args.semantic_topology,
            "capture_last_output": args.capture_last_output,
            "independent_trial": args.independent_trial,
            "profile_control": args.profile_control,
            "premeasurement_schedule": timed_setup,
        },
        "scene": scene_metadata,
        "camera_contract": {
            "path": str(args.trajectory),
            "trajectory_id": trajectory.trajectory_id,
            "batch": trajectory.batch,
            "width": trajectory.width,
            "height": trajectory.height,
            "warmup_frames": args.warmup_frames,
            "measured_frames": args.measured_frames,
            "motion_classification": trajectory.motion_classification,
            "unchanged_measured_camera_frame_pairs": unchanged_pairs,
            "minimum_consecutive_viewmat_max_abs_delta": float(measured_changes.min()),
            "camera_upload_seconds": camera_upload_seconds,
        },
        "equation_contract": {
            "precision": "float32",
            "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            **MATCHED_PROJECTION_RULES,
            "covariance_epsilon": 0.0,
            "alpha_cap": 0.99,
            "alpha_threshold": 1.0 / 255.0,
            "transmittance_threshold": 1.0e-4,
            "pixel_center": "x+0.5,y+0.5",
            "background": [0.0, 0.0, 0.0],
            "depth": "sum(T*alpha*z)/sum(T*alpha), +inf at zero alpha",
            "semantic": "strongest individual T*alpha; -1 below accumulated alpha 0.01",
            "semantic_topology": args.semantic_topology,
            "output_dtypes": {name: str(tensor.dtype) for name, tensor in outputs.items()},
            "rendered_frame_cache": False,
            "projection_cache": False,
            "same_active_pytorch_cuda_stream": True,
        },
        "capacity": capacity,
        "timing": {
            "gpu_batch_ms": gpu_timing,
            "synchronized_wall_batch_ms": wall_timing,
            "host_submission_ms": submission_timing,
            "images_per_second": total_images / total_gpu_seconds,
            "megapixels_per_second": total_megapixels / total_gpu_seconds,
            "measured_wall_seconds": measured_seconds,
            "measured_loop_with_out_of_band_capacity_audit_seconds": (measured_loop_with_audit_seconds),
            "synchronization": "CUDA event synchronized after every measured frame",
            "measured_boundary": "event immediately before RendererService.render through event immediately after submission",
            "cpu_output_copies_in_measured_loop": 0,
        },
        "backend_execution": backend_execution,
        "memory": memory,
        "validation": {
            "scope": "post-measurement",
            "nvtx_range": validation_nvtx_range,
            "wall_seconds": validation_seconds,
            "included_in_cuda_event_timing": False,
            "included_in_memory_snapshots": False,
            "post_measurement_memory_snapshots_completed_before_validation": True,
            "result": output_validation,
        },
        "output_validation": output_validation,
        "fidelity_capture": output_capture,
        "environment": {
            "gpu_name": gpu_name,
            "gpu_uuid": gpu_uuid,
            "compute_capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "driver": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                    "-i",
                    gpu_uuid,
                ]
            ),
            "source_provenance": source_provenance,
            "source_git_commit": source_provenance["head"],
            "source_git_diff_sha256": source_provenance["diff_sha256"],
            "native_extension": native_extension,
            "native_build_ninja": native_build_ninja,
            "native_build_contract": native_build_contract,
            "compiler_versions": {
                "cxx": command_output(["c++", "--version"]),
                "nvcc": command_output(["nvcc", "--version"]),
            },
            "node_occupancy": occupancy_artifact,
            "flashgs_adapter_attestation": adapter_attestation,
            "flashgs_upstream_commit": (
                backend.execution_stats["upstream_commit"]
                if isinstance(backend, FlashGSBackend)
                else FLASHGS_UPSTREAM_COMMIT
            ),
            "cuda_stream": int(torch.cuda.current_stream().cuda_stream),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    service.shutdown()
    print(
        "FLASHGS_MATCHED_RUN_OK "
        + json.dumps(
            {
                "output": str(args.output),
                "pass": result["pass"],
                "renderer": args.renderer,
                "contract": args.output_contract,
                "batch": trajectory.batch,
                "gpu_mean_ms": gpu_timing["mean"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
