#!/usr/bin/env python3
"""Encode, inspect, and checksum the matched Home Scan flyby artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/flyby/home-scan-v1"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preview-width", type=int, default=512)
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        help="Prepared external-scene manifest; omit for the pinned Home Scan release.",
    )
    parser.add_argument("--custom-label", default="Custom 2D EWA")
    parser.add_argument(
        "--ovrtx-label",
        help="Override the generated OVRTX perspective temporal-sample label.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_from_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "id": "home-scan-lod0",
            "gaussian_count": 21_497_908,
            "sha256": (
                "29cee159465406d94f2b24954eefb9da"
                "76ba80cab827b558a6e75676b8809267"
            ),
            "author": "Isaiah Sweeney",
            "source": "https://superspl.at/scene/3f89bbd3",
            "license": "CC BY 4.0",
        }
    value = load_json(path)
    canonical = value["canonical_ply"]
    attribution = value["attribution"]
    return {
        "id": value["dataset_id"],
        "gaussian_count": canonical["gaussian_count"],
        "sha256": canonical["sha256"],
        "author": attribution["author"],
        "source": attribution["source"],
        "license": attribution["license"],
        "license_url": attribution.get("license_url"),
        "prepared_manifest": str(path.resolve()),
    }


def validate_frames(
    frames_dir: Path,
    expected_count: int,
) -> None:
    frames = sorted(frames_dir.glob("frame-*.png"))
    expected_names = [
        f"frame-{index:06d}.png"
        for index in range(expected_count)
    ]
    actual_names = [path.name for path in frames]
    if actual_names != expected_names:
        raise ValueError(
            f"{frames_dir} does not contain the expected contiguous "
            f"{expected_count}-frame sequence."
        )


def probe_video(ffprobe: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "stream=codec_name,width,height,pix_fmt,r_frame_rate,"
                "nb_frames:format=duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    custom_manifest_path = root / "custom/render-manifest.json"
    ovrtx_manifest_path = root / "ovrtx/render-manifest.json"
    camera_manifest_path = root / "camera-path.json"
    camera_contract_path = root / "camera-path.npz"
    required = (
        custom_manifest_path,
        ovrtx_manifest_path,
        camera_manifest_path,
        camera_contract_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    custom_manifest = load_json(custom_manifest_path)
    ovrtx_manifest = load_json(ovrtx_manifest_path)
    ovrtx_label = args.ovrtx_label or (
        "OVRTX perspective - "
        f"{int(ovrtx_manifest['temporal_samples_per_camera'])}x"
    )
    camera_manifest = load_json(camera_manifest_path)
    camera_hash = custom_manifest["camera_contract_sha256"]
    if (
        ovrtx_manifest["camera_contract_sha256"] != camera_hash
        or camera_manifest["camera_contract_sha256"] != camera_hash
    ):
        raise ValueError("Custom, OVRTX, and camera contract hashes differ.")
    common_fields = ("frames", "width", "height", "fps")
    for field in common_fields:
        if custom_manifest[field] != ovrtx_manifest[field]:
            raise ValueError(f"Renderer manifest mismatch for {field}.")
    frame_count = int(custom_manifest["frames"])
    fps = int(custom_manifest["fps"])
    width = int(custom_manifest["width"])
    height = int(custom_manifest["height"])
    validate_frames(root / "custom/frames", frame_count)
    validate_frames(root / "ovrtx/frames", frame_count)

    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    custom_video = artifacts / "custom.mp4"
    ovrtx_video = artifacts / "ovrtx-perspective.mp4"
    comparison_video = artifacts / "side-by-side.mp4"
    preview = artifacts / "side-by-side-preview.gif"
    poster = artifacts / "poster.png"
    storyboard = artifacts / "storyboard.png"

    common_encode = [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    for frames_dir, output in (
        (root / "custom/frames", custom_video),
        (root / "ovrtx/frames", ovrtx_video),
    ):
        run(
            [
                args.ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "frame-%06d.png"),
                "-frames:v",
                str(frame_count),
                *common_encode,
                str(output),
            ]
        )

    label_filter = (
        f"[0:v]drawtext=text='{args.custom_label}':x=16:y=14:"
        "fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[left];"
        f"[1:v]drawtext=text='{ovrtx_label}':x=16:y=14:"
        "fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[right];"
        "[left][right]hstack=inputs=2[comparison]"
    )
    run(
        [
            args.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-thread_queue_size",
            "512",
            "-framerate",
            str(fps),
            "-i",
            str(root / "custom/frames/frame-%06d.png"),
            "-thread_queue_size",
            "512",
            "-framerate",
            str(fps),
            "-i",
            str(root / "ovrtx/frames/frame-%06d.png"),
            "-filter_complex",
            label_filter,
            "-map",
            "[comparison]",
            "-frames:v",
            str(frame_count),
            *common_encode,
            str(comparison_video),
        ]
    )

    run(
        [
            args.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-ss",
            f"{frame_count / fps * 0.25:.6f}",
            "-i",
            str(comparison_video),
            "-frames:v",
            "1",
            "-update",
            "1",
            str(poster),
        ]
    )
    storyboard_fps = 12.0 / (frame_count / fps)
    run(
        [
            args.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(comparison_video),
            "-vf",
            (
                f"fps={storyboard_fps:.8f},"
                "scale=384:-2:flags=lanczos,"
                "tile=4x3:padding=4:margin=4:color=black"
            ),
            "-frames:v",
            "1",
            "-update",
            "1",
            str(storyboard),
        ]
    )
    run(
        [
            args.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(comparison_video),
            "-filter_complex",
            (
                f"[0:v]fps=10,scale={args.preview_width}:-2:"
                "flags=lanczos,split[a][b];"
                "[a]palettegen=max_colors=128:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=3:"
                "diff_mode=rectangle[preview]"
            ),
            "-map",
            "[preview]",
            "-loop",
            "0",
            str(preview),
        ]
    )

    artifact_files = (
        custom_video,
        ovrtx_video,
        comparison_video,
        preview,
        poster,
        storyboard,
    )
    probes = {
        path.name: probe_video(args.ffprobe, path)
        for path in (custom_video, ovrtx_video, comparison_video)
    }
    manifest = {
        "schema_version": "gaussian-flyby-artifacts/v1",
        "scene": scene_from_manifest(args.scene_manifest),
        "camera_contract_sha256": camera_hash,
        "route_sha256": camera_manifest.get("route_sha256"),
        "frames": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps,
        "per_renderer_resolution": [width, height],
        "custom_model": custom_manifest["gaussian_model"],
        "ovrtx_model": ovrtx_manifest["gaussian_model"],
        "ovrtx_temporal_samples_per_camera": ovrtx_manifest[
            "temporal_samples_per_camera"
        ],
        "ovrtx_splat_prim_counts": ovrtx_manifest["upload_contract"][
            "splat_prim_counts"
        ],
        "render_manifests": {
            "custom": custom_manifest,
            "ovrtx": ovrtx_manifest,
        },
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_files
        },
        "video_probes": probes,
        "camera_contract": {
            "npz_bytes": camera_contract_path.stat().st_size,
            "npz_sha256": sha256_file(camera_contract_path),
            "json_bytes": camera_manifest_path.stat().st_size,
            "json_sha256": sha256_file(camera_manifest_path),
        },
        "scope": (
            "Matched visual demonstration only; equation-level fidelity "
            "acceptance is reported separately."
        ),
    }
    manifest_path = artifacts / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "HOME_SCAN_FLYBY_PACKAGE_OK "
        + json.dumps(
            {
                "artifacts": str(artifacts),
                "camera_contract_sha256": camera_hash,
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
