#!/usr/bin/env python3
"""Measure geometric, motion, coverage, and review fields for a route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaacsim_gaussian_renderer.evaluation.trajectory_contract import load_trajectory
from isaacsim_gaussian_renderer.fidelity.outputs import load_render_output
from isaacsim_gaussian_renderer.ply_loader import load_ply_to_gaussians


def camera_centers(viewmats: np.ndarray) -> np.ndarray:
    rotation = viewmats[..., :3, :3]
    translation = viewmats[..., :3, 3]
    return -np.einsum("...ji,...j->...i", rotation, translation)


def rotation_angles(viewmats: np.ndarray) -> np.ndarray:
    relative = np.einsum(
        "...ij,...jk->...ik",
        viewmats[1:, :, :3, :3],
        np.swapaxes(viewmats[:-1, :, :3, :3], -1, -2),
    )
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--render-output", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path)
    parser.add_argument("--clearance-sample-stride", type=int, default=32)
    parser.add_argument(
        "--manual-storyboard-review",
        choices=("pending", "pass", "fail"),
        default="pending",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trajectory = load_trajectory(args.trajectory)
    rendered = load_render_output(args.render_output)
    centers = camera_centers(trajectory.viewmats)
    translations = np.linalg.norm(np.diff(centers, axis=0), axis=-1)
    rotations = rotation_angles(trajectory.viewmats)
    angular_velocity = rotations * trajectory.fps
    angular_acceleration = np.diff(angular_velocity, axis=0) * trajectory.fps
    duplicate = np.all(np.isclose(trajectory.viewmats[1:], trajectory.viewmats[:-1], atol=1.0e-7), axis=(2, 3))
    alpha = rendered.alpha.reshape(trajectory.timesteps, trajectory.batch, trajectory.height, trajectory.width)
    depth = rendered.depth.reshape(trajectory.timesteps, trajectory.batch, trajectory.height, trajectory.width)
    foreground = alpha > 0.01
    valid_depth = foreground & np.isfinite(depth) & (depth > trajectory.near_plane)
    near_pressure = valid_depth & (depth < trajectory.near_plane * 2.0)
    minimum_clearance = None
    clearance_policy = "not-measured"
    if args.scene_path is not None:
        scene = load_ply_to_gaussians(args.scene_path)
        means = scene.means.numpy()[:: args.clearance_sample_stride]
        flat_centers = centers.reshape(-1, 3)
        minimum = float("inf")
        for start in range(0, means.shape[0], 250_000):
            points = means[start : start + 250_000]
            # Eye-level horizontal clearance excludes floor/ceiling points.
            for center in flat_centers:
                vertical = np.abs(points[:, 1] - center[1]) < 0.75
                if np.any(vertical):
                    horizontal = np.linalg.norm(points[vertical][:, [0, 2]] - center[[0, 2]], axis=-1)
                    minimum = min(minimum, float(horizontal.min()))
        minimum_clearance = minimum if np.isfinite(minimum) else None
        clearance_policy = f"sampled-every-{args.clearance_sample_stride}-gaussians-eye-level-band"
    validation = {
        "schema_version": "trajectory-route-validation-v1",
        "trajectory_id": trajectory.trajectory_id,
        "minimum_clearance_m": minimum_clearance,
        "minimum_clearance_policy": clearance_policy,
        "path_length_m": float(translations.sum(axis=0).mean()),
        "translation_per_frame_m": {
            "mean": float(translations.mean()),
            "p95": float(np.percentile(translations, 95)),
            "max": float(translations.max()),
        },
        "rotation_per_frame_rad": {
            "mean": float(rotations.mean()),
            "p95": float(np.percentile(rotations, 95)),
            "max": float(rotations.max()),
        },
        "angular_acceleration_rad_s2": {
            "mean_absolute": float(np.mean(np.abs(angular_acceleration))) if angular_acceleration.size else 0.0,
            "p95_absolute": float(np.percentile(np.abs(angular_acceleration), 95)) if angular_acceleration.size else 0.0,
            "max_absolute": float(np.max(np.abs(angular_acceleration))) if angular_acceleration.size else 0.0,
        },
        "duplicate_poses": int(np.count_nonzero(duplicate)),
        "foreground_coverage": {
            "mean": float(foreground.mean()),
            "min": float(foreground.mean(axis=(-2, -1)).min()),
        },
        "valid_depth_coverage": {
            "mean": float(valid_depth.mean()),
            "min": float(valid_depth.mean(axis=(-2, -1)).min()),
        },
        "near_plane_pressure_fraction": float(near_pressure.sum() / max(valid_depth.sum(), 1)),
        "manual_storyboard_review": args.manual_storyboard_review,
        "pass": bool(
            not np.any(duplicate)
            and foreground.any()
            and valid_depth.any()
            and args.manual_storyboard_review == "pass"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not validation["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
