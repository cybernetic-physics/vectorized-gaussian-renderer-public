#!/usr/bin/env python3
"""Bounded 128x128 Home Scan smoke/performance control for native FlashGS RGB.

This finite hook is deliberately capped at 64 cameras and 16 total frames. It
validates the new upstream-equation lane before that lane is admitted to a
publication matrix; it is not itself headline evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
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

from isaacsim_gaussian_renderer import (  # noqa: E402
    RendererService,
    UpstreamFaithfulFlashGSBackend,
    load_ply_to_gaussians,
)
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (  # noqa: E402
    load_trajectory,
)
from isaacsim_gaussian_renderer.ply_loader import SH_C0  # noqa: E402


HOME_SCAN_COUNT = 21_497_908
HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
SCENE_ID = 404
MAX_CONTROL_BATCH = 64
MAX_CONTROL_FRAMES = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-sha256", default=HOME_SCAN_SHA256)
    parser.add_argument("--max-intersections", type=int, default=12_000_000)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--measured-frames", type=int, default=5)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-name")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sample_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "max_ms": ordered[-1],
    }


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def summarize_counter_samples(
    samples: list[dict[str, int]],
) -> dict[str, Any]:
    """Retain the fail-closed verdict for every reused-counter frame."""
    if not samples:
        raise ValueError("At least one counter sample is required.")
    expected_names = set(samples[0])
    if any(set(sample) != expected_names for sample in samples):
        raise ValueError("Counter samples have inconsistent fields.")
    maxima = {
        name: max(sample[name] for sample in samples)
        for name in samples[0]
    }
    totals = {
        name: sum(sample[name] for sample in samples)
        for name in samples[0]
    }
    return {
        "pass": bool(
            maxima["intersection_overflow"] == 0
            and maxima["camera_contract_errors"] == 0
        ),
        "sample_count": len(samples),
        "samples": samples,
        "maxima": maxima,
        "totals": totals,
    }


def main() -> None:
    args = parse_args()
    total_frames = args.warmup_frames + args.measured_frames
    if args.warmup_frames < 1 or args.measured_frames < 1:
        raise ValueError("warmup-frames and measured-frames must be positive.")
    if total_frames > MAX_CONTROL_FRAMES:
        raise ValueError(
            f"Bounded control permits at most {MAX_CONTROL_FRAMES} total frames."
        )
    trajectory = load_trajectory(args.trajectory)
    if trajectory.width != 128 or trajectory.height != 128:
        raise ValueError("Home control is fixed at 128x128.")
    if trajectory.batch > MAX_CONTROL_BATCH:
        raise ValueError(
            f"Bounded control permits at most {MAX_CONTROL_BATCH} cameras."
        )
    if trajectory.timesteps < total_frames:
        raise ValueError("Trajectory is shorter than the bounded control request.")
    if trajectory.motion_classification != "independently-changing-camera-batch":
        raise ValueError("Control requires independently changing cameras.")
    changes = np.max(
        np.abs(np.diff(trajectory.viewmats[:total_frames], axis=0)),
        axis=(2, 3),
    )
    if np.any(changes == 0.0):
        raise ValueError("Every camera pose must change on every control frame.")
    if trajectory.scene_sha256 != args.scene_sha256:
        raise ValueError("Trajectory and requested scene digests differ.")
    actual_scene_sha256 = file_sha256(args.scene_path)
    if actual_scene_sha256 != args.scene_sha256:
        raise ValueError(
            f"Home Scan digest mismatch: {actual_scene_sha256} != {args.scene_sha256}."
        )

    torch.cuda.init()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices != args.expected_gpu_uuid:
        raise RuntimeError(
            "Bounded control requires CUDA_VISIBLE_DEVICES to be the one exact "
            f"GPU UUID {args.expected_gpu_uuid!r}; got {visible_devices!r}."
        )
    device_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_index)
    if args.expected_gpu_name is not None and gpu_name != args.expected_gpu_name:
        raise RuntimeError(
            f"GPU name mismatch: {gpu_name!r} != {args.expected_gpu_name!r}."
        )
    scene_load_started = time.perf_counter()
    raw = load_ply_to_gaussians(args.scene_path)
    if raw.count != HOME_SCAN_COUNT:
        raise ValueError(f"Home Scan count mismatch: {raw.count} != {HOME_SCAN_COUNT}.")
    rotations = raw.rotations.to(device="cuda", dtype=torch.float32)
    rotation_norms = torch.linalg.vector_norm(rotations, dim=-1, keepdim=True)
    rotations = (
        rotations
        / rotation_norms.clamp_min(torch.finfo(torch.float32).eps)
    ).contiguous()
    # Public FlashGS evaluates degree-zero as max(0.5 + SH_C0 * f_dc, 0)
    # and only clamps during final uint8 encoding. Do not use the matched lane's
    # [0,1] canonical color clamp here: it would change translucent blending.
    degree_zero_colors = (
        0.5
        + SH_C0
        * raw.features[:, :3].to(device="cuda", dtype=torch.float32)
    ).clamp_min_(0.0).contiguous()
    scene = {
        "means": raw.means.to(device="cuda", dtype=torch.float32).contiguous(),
        "scales": raw.scales.to(device="cuda", dtype=torch.float32).exp_().contiguous(),
        "rotations": rotations,
        "opacities": raw.opacities.to(
            device="cuda", dtype=torch.float32
        ).sigmoid_().contiguous(),
        "features": degree_zero_colors,
        "semantic_ids": torch.zeros(
            (raw.count,), dtype=torch.int64, device="cuda"
        ),
    }
    viewmats = torch.from_numpy(
        np.ascontiguousarray(trajectory.viewmats[:total_frames])
    ).to(device="cuda")
    intrinsics = torch.from_numpy(
        np.ascontiguousarray(trajectory.intrinsics[:total_frames])
    ).to(device="cuda")
    scene_ids = torch.full(
        (trajectory.batch,), SCENE_ID, dtype=torch.int64, device="cuda"
    )
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - scene_load_started

    backend = UpstreamFaithfulFlashGSBackend(
        max_intersections=args.max_intersections,
        near_plane=trajectory.near_plane,
        far_plane=trajectory.far_plane,
    )
    service = RendererService(
        backend,
        height=trajectory.height,
        width=trajectory.width,
        outputs=("rgb",),
        max_views=trajectory.batch,
    )
    service.initialize(stage=None, device="cuda")
    service.load_scene(SCENE_ID, **scene)
    outputs = service.prepare_outputs(trajectory.batch)
    counter_samples: list[dict[str, int]] = []
    for step in range(args.warmup_frames):
        outputs = service.render(
            viewmats[step], intrinsics[step], scene_ids, outputs=outputs
        )
        # The backend reuses and clears its per-camera counter storage on every
        # render.  Preserve each frame's verdict after synchronization so an
        # early overflow or camera-contract error cannot be hidden by a later
        # clean frame.  Warmup is outside every timing boundary.
        service.synchronize()
        counter_samples.append(backend.read_counters(synchronize=False))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    for step in range(args.warmup_frames, total_frames):
        wall_started = time.perf_counter()
        start.record()
        outputs = service.render(
            viewmats[step], intrinsics[step], scene_ids, outputs=outputs
        )
        end.record()
        end.synchronize()
        gpu_samples.append(float(start.elapsed_time(end)))
        wall_samples.append((time.perf_counter() - wall_started) * 1000.0)
        # The end event has already synchronized the measured render.  Read the
        # just-completed counters only after recording both timing samples.
        counter_samples.append(backend.read_counters(synchronize=False))
    counters = counter_samples[-1]
    counter_audit = summarize_counter_samples(counter_samples)
    rgb = outputs["rgb"]
    output_validation = {
        "shape": list(rgb.shape),
        "dtype": str(rgb.dtype),
        "cuda": rgb.is_cuda,
        "finite": bool(torch.isfinite(rgb).all().item()),
        "minimum": float(rgb.min().item()),
        "maximum": float(rgb.max().item()),
        "foreground_pixels": int(torch.count_nonzero(rgb).item()),
    }
    output_validation["pass"] = bool(
        output_validation["shape"]
        == [trajectory.batch, trajectory.height, trajectory.width, 3]
        and output_validation["dtype"] == "torch.float32"
        and output_validation["cuda"]
        and output_validation["finite"]
        and 0.0 <= output_validation["minimum"]
        and output_validation["maximum"] <= 1.0
        and output_validation["foreground_pixels"] > 0
    )
    build_contract = backend._native.__vgr_build_contract__
    source_audit_path = Path(build_contract["source_audit"])
    result = {
        "schema_version": "upstream-faithful-flashgs-home-control-v1",
        "pass": bool(output_validation["pass"] and counter_audit["pass"]),
        "classification": "bounded-diagnostic-control-not-headline-evidence",
        "scene": {
            "sha256": actual_scene_sha256,
            "gaussians": raw.count,
            "color_ingestion": (
                "public-flashgs-degree-zero-max(0.5+SH_C0*f_dc,0)-no-upper-clamp"
            ),
        },
        "camera_contract": {
            "trajectory_id": trajectory.trajectory_id,
            "batch": trajectory.batch,
            "width": trajectory.width,
            "height": trajectory.height,
            "warmup_frames": args.warmup_frames,
            "measured_frames": args.measured_frames,
            "every_pose_changes": True,
        },
        "renderer": {
            "execution_stats": backend.execution_stats,
            "counters": counters,
            "counter_audit": counter_audit,
            "scene_bytes": backend.scene_bytes,
            "workspace_bytes": backend.workspace_bytes,
            "source_audit": artifact(source_audit_path),
            "native_extension": artifact(Path(backend._native.__file__)),
        },
        "timing": {
            "gpu_ms": sample_summary(gpu_samples),
            "wall_ms": sample_summary(wall_samples),
            "raw_gpu_ms": gpu_samples,
            "raw_wall_ms": wall_samples,
            "images_per_second": trajectory.batch
            / (statistics.fmean(gpu_samples) / 1000.0),
            "megapixels_per_second": (
                trajectory.batch
                * trajectory.width
                * trajectory.height
                / 1_000_000.0
                / (statistics.fmean(gpu_samples) / 1000.0)
            ),
        },
        "output_validation": output_validation,
        "environment": {
            "gpu_name": gpu_name,
            "gpu_uuid": args.expected_gpu_uuid,
            "device_index": device_index,
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "setup_seconds": setup_seconds,
        },
        "limitations": [
            "RGB only",
            "public FlashGS uint8 quantization retained; float32 conversion measured",
            "centered zero-skew intrinsics only",
            "fixed-capacity sort processes the installed capacity",
            "no occupancy, fidelity-oracle, memory-growth, profiler, or repeat gate",
        ],
    }
    if not all(math.isfinite(value) and value > 0 for value in gpu_samples):
        result["pass"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    service.shutdown()
    print("UPSTREAM_FAITHFUL_FLASHGS_HOME_CONTROL " + json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
