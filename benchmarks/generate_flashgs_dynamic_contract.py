#!/usr/bin/env python3
"""Generate immutable independently-moving Home Scan camera contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (  # noqa: E402
    CameraTrajectory,
    canonical_json_bytes,
    save_trajectory,
)

HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
SCENE_ID = 404
PRIMARY_BATCHES = (1, 8, 32, 64, 128, 256, 512, 1024)
CANONICAL_MAX_BATCH = PRIMARY_BATCHES[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        type=Path,
        default=PROJECT_ROOT / "benchmarks/camera_paths/home-scan-walkthrough-v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/flashgs-matched/contracts",
    )
    parser.add_argument("--timesteps", type=int, default=108)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=1024)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--focal-scale", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def route_hash(payload: dict[str, Any]) -> str:
    content = dict(payload)
    expected = str(content.pop("route_sha256"))
    content.pop("validation", None)
    actual = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    if actual != expected:
        raise ValueError(f"Route SHA-256 mismatch: {actual} != {expected}.")
    return actual


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise ValueError("Camera contract contains a zero-length basis vector.")
    return vector / norm


def interpolate_path(
    points_xz: np.ndarray,
    cumulative: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [np.interp(distance, cumulative, points_xz[:, axis]) for axis in range(2)],
        axis=-1,
    )


def reflected_distance(phase: np.ndarray, length: float) -> np.ndarray:
    wrapped = np.mod(phase, 2.0 * length)
    return np.where(wrapped <= length, wrapped, 2.0 * length - wrapped)


def build_master(args: argparse.Namespace) -> tuple[CameraTrajectory, dict[str, Any]]:
    if args.timesteps <= args.warmup_frames or args.warmup_frames < 1:
        raise ValueError("timesteps must exceed a positive warmup frame count.")
    if args.timesteps - args.warmup_frames < 100:
        raise ValueError("The primary contract requires at least 100 measured frames.")
    route = json.loads(args.route.read_text(encoding="utf-8"))
    route_sha256 = route_hash(route)
    points_xz = np.asarray(route["world_path_xz"], dtype=np.float64)
    segment = np.linalg.norm(np.diff(points_xz, axis=0), axis=1)
    keep = np.concatenate(([True], segment > 1.0e-8))
    points_xz = points_xz[keep]
    segment = np.linalg.norm(np.diff(points_xz, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    length = float(cumulative[-1])
    if not length > 0.0:
        raise ValueError("Route has zero arc length.")

    rng = np.random.default_rng(args.seed)
    # Always construct the same 1024-camera master, even when a bounded run
    # asks to emit only its smaller prefixes. Otherwise --max-batch changes
    # the permutation and phase spacing, so resuming a larger matrix silently
    # replaces the camera contract behind already-recorded results.
    phase_order = rng.permutation(CANONICAL_MAX_BATCH)
    phase0 = (phase_order.astype(np.float64) + 0.371) * (
        2.0 * length / CANONICAL_MAX_BATCH
    )
    # Per-camera speeds make the tracks genuinely independent while preserving
    # the clearance-audited route. Every camera traverses roughly one quarter
    # of the path over the complete contract.
    golden = 0.5 * (1.0 + np.sqrt(5.0))
    speed_scale = 0.85 + 0.30 * np.mod(
        (np.arange(CANONICAL_MAX_BATCH, dtype=np.float64) + 1.0) / golden,
        1.0,
    )
    base_step = length / (4.0 * max(1, args.timesteps - 1))
    time_index = np.arange(args.timesteps, dtype=np.float64)[:, None]
    phase = phase0[None, :] + time_index * base_step * speed_scale[None, :]
    distance = reflected_distance(phase, length)
    positions_xz = interpolate_path(points_xz, cumulative, distance)
    centers = np.empty(
        (args.timesteps, CANONICAL_MAX_BATCH, 3), dtype=np.float64
    )
    centers[..., 0] = positions_xz[..., 0]
    centers[..., 1] = float(route["eye_height"])
    centers[..., 2] = positions_xz[..., 1]

    look_phase = phase + np.maximum(base_step * speed_scale[None, :] * 0.25, 1.0e-4)
    look_distance = reflected_distance(look_phase, length)
    look_xz = interpolate_path(points_xz, cumulative, look_distance)
    look = centers.copy()
    look[..., 0] = look_xz[..., 0]
    look[..., 2] = look_xz[..., 1]
    direction = normalize(look - centers)
    world_up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    right = normalize(np.cross(direction, world_up[None, None, :]))
    down = normalize(np.cross(direction, right))
    camera_to_world = np.stack((right, down, direction), axis=-1)
    world_to_camera = np.swapaxes(camera_to_world, -1, -2)
    viewmats = np.broadcast_to(
        np.eye(4, dtype=np.float32),
        (args.timesteps, CANONICAL_MAX_BATCH, 4, 4),
    ).copy()
    viewmats[..., :3, :3] = world_to_camera.astype(np.float32)
    viewmats[..., :3, 3] = -np.einsum(
        "tbij,tbj->tbi", world_to_camera, centers
    ).astype(np.float32)
    intrinsics = np.zeros(
        (args.timesteps, CANONICAL_MAX_BATCH, 3, 3), dtype=np.float32
    )
    intrinsics[..., 0, 0] = args.focal_scale * args.width
    intrinsics[..., 1, 1] = args.focal_scale * args.height
    intrinsics[..., 0, 2] = args.width * 0.5
    intrinsics[..., 1, 2] = args.height * 0.5
    intrinsics[..., 2, 2] = 1.0

    per_camera_delta = np.max(
        np.abs(np.diff(viewmats[args.warmup_frames - 1 :], axis=0)),
        axis=(2, 3),
    )
    minimum_pose_delta = float(per_camera_delta.min())
    unchanged_pose_pairs = int(np.count_nonzero(per_camera_delta == 0.0))
    duplicate_cameras = 0
    for step_centers in centers[args.warmup_frames:]:
        duplicate_cameras += CANONICAL_MAX_BATCH - np.unique(
            np.round(step_centers, decimals=7), axis=0
        ).shape[0]
    if unchanged_pose_pairs:
        raise RuntimeError(
            f"Generated contract has {unchanged_pose_pairs} unchanged camera/frame pairs."
        )
    if duplicate_cameras:
        raise RuntimeError(
            f"Generated contract has {duplicate_cameras} duplicate cameras."
        )
    motion_audit = {
        "warmup_frames": args.warmup_frames,
        "measured_frames": args.timesteps - args.warmup_frames,
        "minimum_consecutive_viewmat_max_abs_delta": minimum_pose_delta,
        "unchanged_measured_camera_frame_pairs": unchanged_pose_pairs,
        "duplicate_measured_camera_poses_within_frames": duplicate_cameras,
        "independent_phase_count": CANONICAL_MAX_BATCH,
        "per_camera_speed_scale_min": float(speed_scale.min()),
        "per_camera_speed_scale_max": float(speed_scale.max()),
    }
    trajectory = CameraTrajectory(
        viewmats=viewmats,
        intrinsics=intrinsics,
        scene_ids=np.full((CANONICAL_MAX_BATCH,), SCENE_ID, dtype=np.int64),
        scene_id_broadcast="time",
        width=args.width,
        height=args.height,
        fps=args.fps,
        seed=args.seed,
        scene_sha256=HOME_SCAN_SHA256,
        route_sha256=route_sha256,
        motion_classification="independently-changing-camera-batch",
        expected_cache_events=("disabled",) * args.timesteps,
        near_plane=0.01,
        far_plane=100.0,
        provenance={
            "route_path": f"benchmarks/camera_paths/{args.route.name}",
            "generator": "benchmarks/generate_flashgs_dynamic_contract.py",
        },
        route_validation={**route.get("validation", {}), "motion_audit": motion_audit},
    )
    return trajectory, motion_audit


def main() -> None:
    args = parse_args()
    master, motion_audit = build_master(args)
    batches = tuple(batch for batch in PRIMARY_BATCHES if batch <= args.max_batch)
    if not batches or batches[-1] != args.max_batch:
        raise ValueError(
            "max_batch must be one of the required primary batches through 1024."
        )
    generated: dict[str, Any] = {}
    for batch in batches:
        trajectory = CameraTrajectory(
            viewmats=np.ascontiguousarray(master.viewmats[:, :batch]),
            intrinsics=np.ascontiguousarray(master.intrinsics[:, :batch]),
            scene_ids=np.full((batch,), SCENE_ID, dtype=np.int64),
            scene_id_broadcast="time",
            width=master.width,
            height=master.height,
            fps=master.fps,
            seed=master.seed,
            scene_sha256=master.scene_sha256,
            route_sha256=master.route_sha256,
            motion_classification=master.motion_classification,
            expected_cache_events=master.expected_cache_events,
            near_plane=master.near_plane,
            far_plane=master.far_plane,
            provenance=master.provenance,
            route_validation=master.route_validation,
        )
        json_path, npz_path = save_trajectory(
            trajectory, args.output_root / f"b{batch}.json"
        )
        generated[str(batch)] = {
            "trajectory_id": trajectory.trajectory_id,
            "json": str(json_path),
            "npz": str(npz_path),
        }
    print(
        "FLASHGS_DYNAMIC_CONTRACT_OK "
        + json.dumps(
            {"contracts": generated, "motion_audit": motion_audit},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
