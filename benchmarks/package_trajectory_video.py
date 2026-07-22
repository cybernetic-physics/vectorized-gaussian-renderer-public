#!/usr/bin/env python3
"""Package a dedicated renderer's frames into the standard trajectory bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaacsim_gaussian_renderer.evaluation.artifacts import EvaluationArtifactBundle
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import load_trajectory, save_trajectory


def frame_stats(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return {
        "nonblack_fraction": float(np.any(rgb > 2.0 / 255.0, axis=-1).mean()),
        "clipped_fraction": float(np.any(rgb >= 254.0 / 255.0, axis=-1).mean()),
        "mean": float(rgb.mean()),
        "std": float(rgb.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    trajectory = load_trajectory(args.trajectory)
    if trajectory.batch != 1:
        raise ValueError("Video packaging requires a B1 trajectory.")
    sources = sorted(args.frames_dir.glob("frame-*.png"))
    if len(sources) != trajectory.timesteps:
        raise ValueError(f"Expected {trajectory.timesteps} frames, found {len(sources)}.")
    source_manifest = (
        json.loads(args.source_manifest.read_text(encoding="utf-8"))
        if args.source_manifest is not None
        else {
            "schema_version": "missing-source-render-manifest/v1",
            "pass": False,
            "failure": "source renderer did not produce a manifest",
        }
    )
    bundle = EvaluationArtifactBundle.create(args.output_root / args.run_id)
    stats = []
    for index, source in enumerate(sources):
        destination = bundle.root / "frames" / f"frame-{index:06d}.png"
        shutil.copy2(source, destination)
        stats.append(frame_stats(destination))
    black_frame_count = sum(item["nonblack_fraction"] <= 0 for item in stats)
    save_trajectory(trajectory, bundle.root / "trajectory.json")
    bundle.write_json("metrics/source-render-manifest.json", source_manifest)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-framerate", str(trajectory.fps),
            "-i", str(bundle.root / "frames" / "frame-%06d.png"),
            "-frames:v", str(trajectory.timesteps), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(bundle.root / "videos" / "trajectory.mp4"),
        ],
        check=True,
    )
    bundle.write_json(
        "timing.json",
        {
            "schema_version": "trajectory-timing-v1",
            "status": "visualization-timing-only",
            "elapsed_seconds_including_readback_and_encoding": source_manifest.get("elapsed_seconds"),
            "temporal_samples_per_camera": source_manifest.get("temporal_samples_per_camera"),
        },
    )
    bundle.write_json(
        "cache-events.json",
        {"schema_version": "trajectory-cache-events-v1", "events": [], "status": "not-applicable"},
    )
    bundle.write_json(
        "metrics/summary.json",
        {"schema_version": "trajectory-metrics-v1", "status": "not-compared", "rendering_equation": source_manifest.get("gaussian_model")},
    )
    bundle.write_json(
        "manifest.json",
        {
            "schema_version": "trajectory-evaluation-run-v1",
            "run_id": args.run_id,
            "renderer": args.renderer,
            "scenario": "sequential-flyby-visualization",
            "trajectory_id": trajectory.trajectory_id,
            "frame_count": trajectory.timesteps,
            "expected_rendered_frame_count": trajectory.timesteps,
            "width": trajectory.width,
            "height": trajectory.height,
            "fps": trajectory.fps,
            "frame_statistics": stats,
            "video_status": "encoded",
            "max_explained_duplicate_run": 1,
            "cited_frames": ["frame-000000.png", f"frame-{trajectory.timesteps - 1:06d}.png"],
            "source_render_manifest": "metrics/source-render-manifest.json",
            "visual_demo_only": True,
            "black_frame_count": black_frame_count,
            "failures": [
                *([f"{black_frame_count} completely black frames"] if black_frame_count else []),
                *(["source render manifest missing or failed"] if source_manifest.get("pass") is not True else []),
            ],
            "pass": black_frame_count == 0 and source_manifest.get("pass") is True,
        },
    )
    bundle.write_hash_manifest()
    print(json.dumps({"run_id": args.run_id, "bundle": str(bundle.root)}, sort_keys=True))
    if black_frame_count or source_manifest.get("pass") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
