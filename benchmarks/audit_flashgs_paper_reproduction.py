#!/usr/bin/env python3
"""Recompute the archived RTX 3090 FlashGS paper-regime control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


FLASHGS_COMMIT = "cdfc4e4002318423eda356eed02df8e01fa32cb6"
VANILLA_COMMIT = "472689c0dc70417448fb451bf529ae532d32c095"
VANILLA_RASTER_COMMIT = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
FREIZA_CONTROL_GPU_UUID = "GPU-a206daaf-efc2-8eb0-2abe-a9700f440616"
OFFICIAL_MODELS_URL = (
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
    "datasets/pretrained/models.zip"
)
EXPECTED_INPUT_SHA256 = {
    "models.zip": "b922a42bb4e938fe8e2ac5fbc2577724884335ada7deb3c4b86c5a4c92c5813d",
    "prepared/matrixcity.ply": "d242165805a04663a0d5e00a8fa3e2778c72f87a5c95d3c50137559944e79447",
    "matrixcity-cameras/cameras.json": "81989799b78968ee6e566b444913588a5be7b1cb57e1962accd171e45501c7c3",
    "prepared/rubble.ply": "fdc1e250b848787c982242224a403b49b316ce4e0c2cc6ed22053e5dbcedcb99",
    "rubble-colmap-cameras/cameras.json": "4efc8ae13bca035bbc224bf381b0c7451352b72a845f4f7807e6a3380c136683",
}

CASES = {
    "truck_4k": {
        "resolution": [3840, 2160],
        "paper_vanilla_fps": 41.3,
        "paper_flash_fps": 289.0,
        "exact_public_checkpoint": True,
        "scene_directory": "truck",
    },
    "train_4k": {
        "resolution": [3840, 2160],
        "paper_vanilla_fps": 40.2,
        "paper_flash_fps": 301.2,
        "exact_public_checkpoint": True,
        "scene_directory": "train",
    },
    "playroom_4k": {
        "resolution": [3840, 2160],
        "paper_vanilla_fps": 47.2,
        "paper_flash_fps": 367.7,
        "exact_public_checkpoint": True,
        "scene_directory": "playroom",
    },
    "drjohnson_4k": {
        "resolution": [3840, 2160],
        "paper_vanilla_fps": 34.8,
        "paper_flash_fps": 334.4,
        "exact_public_checkpoint": True,
        "scene_directory": "drjohnson",
    },
    "matrixcity_1080p": {
        "resolution": [1920, 1080],
        "paper_vanilla_fps": 47.9,
        "paper_flash_fps": 310.6,
        "exact_public_checkpoint": False,
        "model_suffix": "prepared/matrixcity.ply",
        "cameras_suffix": "matrixcity-cameras/cameras.json",
    },
    "matrixcity_4k": {
        "resolution": [3840, 2160],
        "paper_vanilla_fps": 15.0,
        "paper_flash_fps": 207.4,
        "exact_public_checkpoint": False,
        "model_suffix": "prepared/matrixcity.ply",
        "cameras_suffix": "matrixcity-cameras/cameras.json",
    },
    "rubble_native": {
        "resolution": [4608, 3456],
        "paper_vanilla_fps": 20.7,
        "paper_flash_fps": 183.9,
        "exact_public_checkpoint": False,
        "model_suffix": "prepared/rubble.ply",
        "cameras_suffix": "rubble-colmap-cameras/cameras.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-12
    )


def read_ppm_token(stream: BinaryIO) -> bytes:
    """Read one whitespace-delimited PPM token, skipping comments."""
    while True:
        byte = stream.read(1)
        if not byte:
            raise ValueError("Unexpected EOF in PPM header.")
        if byte == b"#":
            stream.readline()
            continue
        if not byte.isspace():
            break
    token = bytearray(byte)
    while True:
        byte = stream.read(1)
        if not byte:
            raise ValueError("Unexpected EOF in PPM header.")
        if byte.isspace():
            return bytes(token)
        token.extend(byte)


def read_ppm_header(stream: BinaryIO) -> tuple[int, int]:
    if read_ppm_token(stream) != b"P6":
        raise ValueError("Only binary P6 PPM samples are supported.")
    try:
        width = int(read_ppm_token(stream))
        height = int(read_ppm_token(stream))
        maximum = int(read_ppm_token(stream))
    except ValueError as error:
        raise ValueError("Malformed PPM dimensions or maximum value.") from error
    if width <= 0 or height <= 0 or maximum != 255:
        raise ValueError("PPM dimensions or maximum value differ from contract.")
    return width, height


def compare_ppm_samples(
    reference: Path,
    candidate: Path,
    *,
    expected_resolution: list[int],
) -> dict[str, Any]:
    """Recompute byte-domain fidelity and hashes from two P6 PPM samples."""
    reference_hash = hashlib.sha256()
    candidate_hash = hashlib.sha256()
    total_absolute_error = 0
    total_squared_error = 0
    maximum_absolute_error = 0
    exact_samples = 0
    sample_count = 0
    with reference.open("rb") as reference_stream, candidate.open(
        "rb"
    ) as candidate_stream:
        reference_size = read_ppm_header(reference_stream)
        candidate_size = read_ppm_header(candidate_stream)
        expected_size = tuple(expected_resolution)
        if reference_size != expected_size or candidate_size != expected_size:
            raise ValueError(
                "PPM dimensions differ from the benchmark resolution."
            )
        payload_bytes = expected_size[0] * expected_size[1] * 3
        remaining = payload_bytes
        while remaining:
            block_size = min(1024 * 1024, remaining)
            reference_block = reference_stream.read(block_size)
            candidate_block = candidate_stream.read(block_size)
            if (
                len(reference_block) != block_size
                or len(candidate_block) != block_size
            ):
                raise ValueError("PPM sample payload is truncated.")
            reference_hash.update(reference_block)
            candidate_hash.update(candidate_block)
            reference_values = np.frombuffer(
                reference_block, dtype=np.uint8
            ).astype(np.int16)
            candidate_values = np.frombuffer(
                candidate_block, dtype=np.uint8
            ).astype(np.int16)
            difference = reference_values - candidate_values
            absolute_difference = np.abs(difference)
            total_absolute_error += int(
                absolute_difference.sum(dtype=np.int64)
            )
            total_squared_error += int(
                np.square(difference, dtype=np.int32).sum(dtype=np.int64)
            )
            maximum_absolute_error = max(
                maximum_absolute_error,
                int(absolute_difference.max(initial=0)),
            )
            exact_samples += int(np.count_nonzero(difference == 0))
            sample_count += len(difference)
            remaining -= block_size
        if reference_stream.read(1) or candidate_stream.read(1):
            raise ValueError("PPM sample payload has trailing bytes.")
    root_mean_square_error = math.sqrt(total_squared_error / sample_count)
    if root_mean_square_error == 0.0:
        raise ValueError(
            "Exact PPM equality needs an explicit finite-metric representation."
        )
    return {
        "shape": [expected_size[1], expected_size[0], 3],
        "mean_absolute_error_u8": total_absolute_error / sample_count,
        "root_mean_square_error_u8": root_mean_square_error,
        "maximum_absolute_error_u8": maximum_absolute_error,
        "psnr_db": 20.0 * math.log10(255.0 / root_mean_square_error),
        "exact_sample_fraction": exact_samples / sample_count,
        "reference_payload_sha256": reference_hash.hexdigest(),
        "candidate_payload_sha256": candidate_hash.hexdigest(),
        "reference_file_sha256": sha256(reference),
        "candidate_file_sha256": sha256(candidate),
    }


def parse_checksums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, recorded_path = line.partition("  ")
        name = Path(recorded_path).name
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name
            or name in records
        ):
            raise ValueError(f"Malformed checksum record: {line!r}.")
        records[name] = digest
    return records


def parse_source_commits(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, commit = line.partition("\t")
        if not separator or not name or len(commit) != 40:
            raise ValueError(f"Malformed source-commit record: {line!r}.")
        records[name] = commit
    return records


def parse_input_checksums(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, recorded_path = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not recorded_path.startswith("/")
        ):
            raise ValueError(f"Malformed input-checksum record: {line!r}.")
        records.append((digest, recorded_path))
    return records


def checksum_for_suffix(
    records: list[tuple[str, str]], suffix: str
) -> str | None:
    matches = [digest for digest, path in records if path.endswith(f"/{suffix}")]
    return matches[0] if len(matches) == 1 else None


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return payload


def audit(root: Path, *, verify_samples: bool = True) -> dict[str, Any]:
    root = root.resolve()
    results_root = root / "results"
    provenance_root = root / "provenance"
    failures: list[str] = []
    try:
        checksums = parse_checksums(provenance_root / "result-sha256.txt")
        input_checksums = parse_input_checksums(
            provenance_root / "input-sha256.txt"
        )
        commits = parse_source_commits(
            provenance_root / "source-commits.tsv"
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        return {
            "schema_version": "flashgs-paper-reproduction-audit-v2",
            "pass": False,
            "failures": [str(error)],
            "cases": [],
        }
    for suffix, expected in EXPECTED_INPUT_SHA256.items():
        if checksum_for_suffix(input_checksums, suffix) != expected:
            failures.append(f"input identity differs: {suffix}")
    expected_commits = {
        "FlashGS": FLASHGS_COMMIT,
        "gaussian-splatting": VANILLA_COMMIT,
        "diff-gaussian-rasterization": VANILLA_RASTER_COMMIT,
    }
    for name, expected in expected_commits.items():
        if commits.get(name) != expected:
            failures.append(f"source commit differs for {name}")

    machine_path = provenance_root / "machine-and-toolchain.txt"
    try:
        machine_text = machine_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError) as error:
        failures.append(f"machine provenance is unreadable: {error}")
        machine_text = ""
    if FREIZA_CONTROL_GPU_UUID not in machine_text:
        failures.append("machine provenance does not identify the control GPU UUID")

    case_rows: list[dict[str, Any]] = []
    for case, contract in CASES.items():
        runs: dict[str, dict[str, Any]] = {}
        for backend in ("vanilla", "flash"):
            path = results_root / f"{case}-{backend}.json"
            expected_digest = checksums.get(path.name)
            if not path.is_file() or expected_digest != sha256(path):
                failures.append(f"result checksum differs: {path.name}")
                continue
            run = load_json(path)
            runs[backend] = run
            frames = run.get("frames")
            if contract["exact_public_checkpoint"]:
                scene_directory = contract["scene_directory"]
                model_suffix = (
                    f"standard/{scene_directory}/point_cloud/iteration_30000/"
                    "point_cloud.ply"
                )
                cameras_suffix = f"standard/{scene_directory}/cameras.json"
            else:
                model_suffix = contract["model_suffix"]
                cameras_suffix = contract["cameras_suffix"]
            if (
                run.get("schema_version") != 1
                or run.get("scene") != case
                or run.get("backend") != backend
                or run.get("resolution") != contract["resolution"]
                or run.get("gpu") != "NVIDIA GeForce RTX 3090"
                or run.get("gpu_capability") != [8, 6]
                or run.get("hostname") != "freiza-1"
                or run.get("cuda_visible_devices") != "1"
                or not str(run.get("model", "")).endswith(f"/{model_suffix}")
                or not str(run.get("cameras", "")).endswith(
                    f"/{cameras_suffix}"
                )
                or run.get("warmup_per_camera") != 1
                or run.get("repeats_per_camera") != 10
                or run.get("camera_stride") != 1
                or not isinstance(run.get("vertex_count"), int)
                or run["vertex_count"] <= 0
                or not isinstance(frames, list)
                or len(frames) != run.get("camera_count")
            ):
                failures.append(f"run contract differs: {path.name}")
                continue
            frame_fps: list[float] = []
            frame_ms: list[float] = []
            frame_contract_valid = True
            for index, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    frame_contract_valid = False
                    break
                milliseconds = frame.get("milliseconds")
                fps = frame.get("fps")
                if (
                    frame.get("index") != index
                    or frame.get("camera_id") != index
                    or frame.get("width") != contract["resolution"][0]
                    or frame.get("height") != contract["resolution"][1]
                    or not isinstance(milliseconds, (int, float))
                    or not isinstance(fps, (int, float))
                    or not math.isfinite(float(milliseconds))
                    or not math.isfinite(float(fps))
                    or float(milliseconds) <= 0.0
                    or float(fps) <= 0.0
                    or not close(float(fps), 1000.0 / float(milliseconds))
                ):
                    frame_contract_valid = False
                    break
                frame_ms.append(float(milliseconds))
                frame_fps.append(float(fps))
            if not frame_fps or not frame_contract_valid:
                failures.append(f"frame timing is invalid: {path.name}")
                continue
            recomputed_mean = sum(frame_fps) / len(frame_fps)
            recomputed_aggregate = 1000.0 / (sum(frame_ms) / len(frame_ms))
            if (
                not close(run.get("mean_frame_fps"), recomputed_mean)
                or not close(
                    run.get("aggregate_throughput_fps"),
                    recomputed_aggregate,
                )
                or not close(run.get("minimum_frame_fps"), min(frame_fps))
                or not close(
                    run.get("median_frame_fps"),
                    statistics.median(frame_fps),
                )
            ):
                failures.append(f"timing aggregates differ from raw frames: {path.name}")

        comparison_path = results_root / f"{case}-pixel-comparison.json"
        expected_digest = checksums.get(comparison_path.name)
        comparison_stats: dict[str, Any] | None = None
        if (
            not comparison_path.is_file()
            or expected_digest != sha256(comparison_path)
        ):
            failures.append(
                f"pixel-comparison checksum differs: {comparison_path.name}"
            )
            comparison: dict[str, Any] = {}
        else:
            comparison = load_json(comparison_path)
            expected_shape = [
                contract["resolution"][1],
                contract["resolution"][0],
                3,
            ]
            metric_names = (
                "mean_absolute_error_u8",
                "root_mean_square_error_u8",
                "psnr_db",
                "exact_sample_fraction",
            )
            metric_values = [comparison.get(name) for name in metric_names]
            maximum_error = comparison.get("maximum_absolute_error_u8")
            reference_suffix = f"/samples/{case}-vanilla.ppm"
            candidate_suffix = f"/samples/{case}-flash.ppm"
            if (
                comparison.get("shape") != expected_shape
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in metric_values
                )
                or not isinstance(maximum_error, int)
                or not 0 <= maximum_error <= 255
                or not 0.0
                <= float(comparison.get("exact_sample_fraction", -1.0))
                <= 1.0
                or float(comparison["psnr_db"]) < 40.0
                or not str(comparison.get("reference", "")).endswith(
                    reference_suffix
                )
                or not str(comparison.get("candidate", "")).endswith(
                    candidate_suffix
                )
            ):
                failures.append(f"pixel parity failed: {comparison_path.name}")
            elif verify_samples:
                try:
                    comparison_stats = compare_ppm_samples(
                        root / "samples" / f"{case}-vanilla.ppm",
                        root / "samples" / f"{case}-flash.ppm",
                        expected_resolution=contract["resolution"],
                    )
                except (OSError, ValueError) as error:
                    failures.append(f"sample fidelity failed for {case}: {error}")
                else:
                    stored_metric_names = (
                        "mean_absolute_error_u8",
                        "root_mean_square_error_u8",
                        "psnr_db",
                        "exact_sample_fraction",
                    )
                    if (
                        any(
                            not close(
                                comparison.get(name),
                                float(comparison_stats[name]),
                            )
                            for name in stored_metric_names
                        )
                        or comparison.get("maximum_absolute_error_u8")
                        != comparison_stats["maximum_absolute_error_u8"]
                        or comparison.get("shape") != comparison_stats["shape"]
                    ):
                        failures.append(
                            f"sample fidelity differs: {comparison_path.name}"
                        )
            else:
                comparison_stats = comparison
        if set(runs) != {"vanilla", "flash"}:
            continue
        vanilla_frames = runs["vanilla"]["frames"]
        flash_frames = runs["flash"]["frames"]
        paired_fields = (
            "model",
            "cameras",
            "resolution",
            "camera_count",
            "camera_stride",
            "warmup_per_camera",
            "repeats_per_camera",
            "vertex_count",
            "gpu",
            "gpu_capability",
            "hostname",
            "cuda_visible_devices",
        )
        if any(
            runs["vanilla"].get(field) != runs["flash"].get(field)
            for field in paired_fields
        ) or any(
            any(
                vanilla_frame.get(field) != flash_frame.get(field)
                for field in (
                    "index",
                    "camera_id",
                    "image_name",
                    "width",
                    "height",
                )
            )
            for vanilla_frame, flash_frame in zip(
                vanilla_frames, flash_frames, strict=True
            )
        ):
            failures.append(f"backend inputs or frame sequence differ: {case}")
            continue
        vanilla_fps = float(runs["vanilla"]["mean_frame_fps"])
        flash_fps = float(runs["flash"]["mean_frame_fps"])
        case_rows.append(
            {
                "case": case,
                "resolution": contract["resolution"],
                "camera_count": runs["flash"]["camera_count"],
                "checkpoint_scope": (
                    "exact-public-paper-checkpoint"
                    if contract["exact_public_checkpoint"]
                    else "public-substitute-checkpoint"
                ),
                "vanilla_fps": vanilla_fps,
                "flash_fps": flash_fps,
                "speedup": flash_fps / vanilla_fps,
                "paper_speedup": (
                    contract["paper_flash_fps"]
                    / contract["paper_vanilla_fps"]
                ),
                "pixel_psnr_db": (
                    comparison_stats.get("psnr_db")
                    if comparison_stats is not None
                    else comparison.get("psnr_db")
                ),
                "sample_fidelity": comparison_stats,
            }
        )

    exact_rows = [
        row
        for row in case_rows
        if row["checkpoint_scope"] == "exact-public-paper-checkpoint"
    ]
    substitute_rows = case_rows
    exact_mean = (
        sum(row["speedup"] for row in exact_rows) / len(exact_rows)
        if exact_rows
        else None
    )
    seven_mean = (
        sum(row["speedup"] for row in substitute_rows) / len(substitute_rows)
        if substitute_rows
        else None
    )
    exact_paper_mean = (
        sum(row["paper_speedup"] for row in exact_rows) / len(exact_rows)
        if exact_rows
        else None
    )
    seven_paper_mean = (
        sum(row["paper_speedup"] for row in case_rows) / len(case_rows)
        if case_rows
        else None
    )
    if len(case_rows) != len(CASES) or len(exact_rows) != 4:
        failures.append("complete seven-case/four-exact-case inventory is missing")
    for name, rows, recomputed_mean in (
        ("exact-four-summary.json", exact_rows, exact_mean),
        ("public-seven-summary.json", case_rows, seven_mean),
    ):
        path = results_root / name
        if not path.is_file() or checksums.get(name) != sha256(path):
            failures.append(f"summary checksum differs: {name}")
            continue
        summary = load_json(path)
        expected_case_names = [row["case"] for row in rows]
        summary_rows = summary.get("cases")
        if (
            summary.get("case_count") != len(rows)
            or recomputed_mean is None
            or not close(
                summary.get("arithmetic_mean_speedup"), recomputed_mean
            )
            or not isinstance(summary_rows, list)
            or not all(isinstance(row, dict) for row in summary_rows)
            or [row.get("case") for row in summary_rows] != expected_case_names
            or any(
                not close(stored.get("speedup"), recomputed["speedup"])
                or not close(stored.get("vanilla_fps"), recomputed["vanilla_fps"])
                or not close(stored.get("flash_fps"), recomputed["flash_fps"])
                or stored.get("resolution") != recomputed["resolution"]
                or stored.get("camera_count") != recomputed["camera_count"]
                for stored, recomputed in zip(summary_rows, rows, strict=True)
            )
        ):
            failures.append(f"summary aggregate differs: {name}")

    return {
        "schema_version": "flashgs-paper-reproduction-audit-v2",
        "pass": not failures,
        "failures": failures,
        "source_commits": expected_commits,
        "input_provenance": {
            "official_pretrained_models_url": OFFICIAL_MODELS_URL,
            "official_pretrained_models_sha256": EXPECTED_INPUT_SHA256[
                "models.zip"
            ],
            "recorded_input_sha256": EXPECTED_INPUT_SHA256,
        },
        "hardware_scope": {
            "host": "freiza-1",
            "gpu_index": 1,
            "gpu_name": "NVIDIA GeForce RTX 3090",
            "gpu_uuid": FREIZA_CONTROL_GPU_UUID,
        },
        "cases": case_rows,
        "exact_public_four": {
            "case_count": len(exact_rows),
            "frame_records_per_backend": sum(
                row["camera_count"] for row in exact_rows
            ),
            "arithmetic_mean_speedup": exact_mean,
            "paper_arithmetic_mean_speedup_for_these_cases": (
                exact_paper_mean
            ),
            "claim": (
                "Archived rerun of four public-checkpoint paper rows; not "
                "the paper's seven-case 8.7x aggregate or a current-protocol "
                "benchmark."
            ),
        },
        "public_substitute_seven": {
            "case_count": len(case_rows),
            "frame_records_per_backend": sum(
                row["camera_count"] for row in case_rows
            ),
            "arithmetic_mean_speedup": seven_mean,
            "paper_arithmetic_mean_speedup_for_these_cases": (
                seven_paper_mean
            ),
            "claim": (
                "Installation control only: MatrixCity and Rubble use "
                "locally prepared substitute inputs and are not an exact "
                "reproduction."
            ),
        },
        "limitations": [
            "The archived control predates per-run cooperative-lock and sampled occupancy evidence.",
            "Mean frame FPS follows the paper-reproduction harness and is the arithmetic mean of per-camera FPS.",
            "Only per-camera aggregates survive; the ten individual timing repeats per camera do not.",
            (
                "The copied archive omits the invocation, runner source, build "
                "and clean-tree attestations, and extracted exact-four inputs."
            ),
            (
                "Substitute-source provenance sidecars named by the input "
                "checksum record are not present in the copied archive."
            ),
            (
                "This is an upstream-installation sanity check; it is not a "
                "matched robotics-renderer result."
            ),
        ],
        "fidelity_recomputed_from_samples": verify_samples,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
