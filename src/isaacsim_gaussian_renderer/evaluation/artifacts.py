"""Durable trajectory-evaluation artifact layout and fail-closed auditing."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationArtifactBundle:
    root: Path

    @classmethod
    def create(cls, root: str | Path) -> "EvaluationArtifactBundle":
        bundle = cls(Path(root))
        for relative in (
            "metrics",
            "tensors",
            "frames",
            "storyboards",
            "videos",
            "profiles",
        ):
            (bundle.root / relative).mkdir(parents=True, exist_ok=True)
        return bundle

    def write_json(self, relative: str, payload: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_hash_manifest(self) -> Path:
        entries = {
            path.relative_to(self.root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(self.root.rglob("*"))
            if path.is_file() and path.name != "artifact-sha256.json"
        }
        return self.write_json(
            "artifact-sha256.json",
            {"schema_version": "trajectory-artifact-sha256-v1", "files": entries},
        )


def _frame_stats(image: np.ndarray) -> dict[str, float]:
    rgb = image[..., :3].astype(np.float64) / 255.0
    return {
        "nonblack_fraction": float(np.any(rgb > (2.0 / 255.0), axis=-1).mean()),
        "clipped_fraction": float(np.any(rgb >= (254.0 / 255.0), axis=-1).mean()),
        "mean": float(rgb.mean()),
        "std": float(rgb.std()),
    }


def _ffprobe(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def _ratio(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    return float(numerator) / float(denominator)


def _audit_decoded_video_samples(
    video: Path,
    frame_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not frame_paths:
        return [], []
    indices = sorted({0, len(frame_paths) // 2, len(frame_paths) - 1})
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    failures: list[str] = []
    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="trajectory-video-audit-") as temporary:
        pattern = str(Path(temporary) / "decoded-%06d.png")
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select={expression}",
            "-vsync",
            "vfr",
            pattern,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        decoded = sorted(Path(temporary).glob("decoded-*.png"))
        if len(decoded) != len(indices):
            return [], [f"video-decode-sample-count:{video.name}:{len(decoded)}"]
        for source_index, decoded_path in zip(indices, decoded, strict=True):
            with Image.open(frame_paths[source_index]) as image:
                source = np.asarray(image.convert("RGB"), dtype=np.float64)
            with Image.open(decoded_path) as image:
                candidate = np.asarray(image.convert("RGB"), dtype=np.float64)
            if source.shape != candidate.shape:
                failures.append(f"video-decode-dimensions:{video.name}:{source_index}")
                continue
            mae = float(np.mean(np.abs(source - candidate)) / 255.0)
            mse = float(np.mean((source - candidate) ** 2) / (255.0**2))
            psnr = math.inf if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
            reports.append({"source_frame_index": source_index, "mae": mae, "psnr_db": psnr})
            if mae > 0.08 or psnr < 20.0:
                failures.append(f"video-source-mismatch:{video.name}:{source_index}")
    return reports, failures


def audit_artifact_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    hash_path = root / "artifact-sha256.json"
    if not manifest_path.is_file() or not hash_path.is_file():
        raise ValueError("Artifact bundle is missing manifest.json or artifact-sha256.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = json.loads(hash_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in hashes.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            failures.append(f"hash-or-size:{relative}")

    expected_frames = int(manifest["frame_count"])
    width, height = int(manifest["width"]), int(manifest["height"])
    expected_fps = float(manifest.get("fps", 0.0))
    frame_paths = sorted((root / "frames").glob("*.png"))
    if len(frame_paths) != expected_frames:
        failures.append(f"frame-count:{len(frame_paths)}!={expected_frames}")
    frame_stats: list[dict[str, float]] = []
    frame_hashes: list[str] = []
    for path in frame_paths:
        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                failures.append(f"frame-dimensions:{path.name}:{image.size}")
            array = np.asarray(image.convert("RGB"))
        stats = _frame_stats(array)
        frame_stats.append(stats)
        frame_hashes.append(hashlib.sha256(array.tobytes()).hexdigest())
        if stats["nonblack_fraction"] <= 0.0:
            failures.append(f"black:{path.name}")
        if stats["clipped_fraction"] >= 1.0:
            failures.append(f"fully-clipped:{path.name}")

    max_frozen = int(manifest.get("max_explained_duplicate_run", 1))
    run = 1
    for index in range(1, len(frame_hashes)):
        run = run + 1 if frame_hashes[index] == frame_hashes[index - 1] else 1
        if run > max_frozen:
            failures.append(f"unexplained-frozen-run:{index - run + 1}:{index}")
            break

    recorded_stats = manifest.get("frame_statistics")
    if recorded_stats is not None and len(recorded_stats) == len(frame_stats):
        for index, (recorded, actual) in enumerate(zip(recorded_stats, frame_stats, strict=True)):
            for key in actual:
                if abs(float(recorded[key]) - actual[key]) > 1.0e-9:
                    failures.append(f"frame-stat-mismatch:{index}:{key}")
    elif recorded_stats is not None:
        failures.append("frame-stat-count")

    video_reports: dict[str, Any] = {}
    for video in sorted((root / "videos").glob("*")):
        if not video.is_file():
            continue
        try:
            report = _ffprobe(video)
            decoded_count = int(report.get("nb_read_frames") or 0)
            if decoded_count != expected_frames:
                failures.append(f"video-frame-count:{video.name}:{decoded_count}")
            if int(report["width"]) != width or int(report["height"]) != height:
                failures.append(f"video-dimensions:{video.name}")
            decoded_fps = _ratio(str(report.get("avg_frame_rate") or report.get("r_frame_rate")))
            if expected_fps > 0 and abs(decoded_fps - expected_fps) > max(1.0e-3, expected_fps * 1.0e-3):
                failures.append(f"video-fps:{video.name}:{decoded_fps}")
            duration = float(report.get("duration") or 0.0)
            expected_duration = expected_frames / expected_fps if expected_fps > 0 else duration
            if duration > 0 and abs(duration - expected_duration) > max(0.05, 1.0 / max(expected_fps, 1.0)):
                failures.append(f"video-duration:{video.name}:{duration}")
            decoded_samples, decoded_failures = _audit_decoded_video_samples(video, frame_paths)
            report["decoded_source_samples"] = decoded_samples
            failures.extend(decoded_failures)
            video_reports[video.name] = report
        except (FileNotFoundError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
            failures.append(f"video-probe:{video.name}:{exc}")

    cited = set(manifest.get("cited_frames", []))
    available = {path.name for path in frame_paths}
    for name in sorted(cited - available):
        failures.append(f"missing-cited-frame:{name}")
    result = {
        "schema_version": "trajectory-artifact-audit-v1",
        "pass": not failures,
        "failures": failures,
        "frame_count": len(frame_paths),
        "video_reports": video_reports,
    }
    return result
