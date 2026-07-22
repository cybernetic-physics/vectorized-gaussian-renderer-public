"""Convenience CLI for the benchmark matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TextIO

if __package__:
    from .renderer_options import (
        rasterize_mode_qualified,
        resolve_covariance_epsilon,
    )
else:
    from renderer_options import (
        rasterize_mode_qualified,
        resolve_covariance_epsilon,
    )


SYNTHETIC_SCENES = {"synthetic-small", "synthetic-medium"}
REAL_SCENES = {"home-scan-lod0", "voxel51-train-30000"}
REAL_SCENE_DIRECT_CAPACITY = {
    "home-scan-lod0": {
        "visible_gaussians_per_view": 150_000.0,
        "intersections_per_view_at_128": 170_000.0,
    },
    "voxel51-train-30000": {
        "visible_gaussians_per_view": 250_000.0,
        "intersections_per_view_at_128": 260_000.0,
    },
}


def gsplat_rasterization_arguments(
    *,
    rasterize_mode: str,
    covariance_epsilon: float,
) -> list[str]:
    """Return the matched gsplat rasterization options for a matrix child."""
    return [
        "--rasterize-mode",
        rasterize_mode,
        "--eps2d",
        str(covariance_epsilon),
    ]


def mode_result_filename(stem: str, rasterize_mode: str) -> str:
    """Return a result filename that cannot collide across modes."""
    return f"{rasterize_mode_qualified(stem, rasterize_mode)}.json"


def child_environment() -> dict[str, str]:
    """Ensure child runners import this checkout, not a stale editable install."""
    project_src = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{project_src}:{existing}" if existing else str(project_src)
    )
    return environment


def result_scene_id(result: dict[str, object]) -> str:
    scene = result.get("scene")
    if isinstance(scene, dict):
        return str(scene["scene_id"])
    dataset = result.get("dataset")
    if isinstance(dataset, dict):
        return str(dataset["dataset_id"])
    raise KeyError("Result does not contain a scene or dataset identifier.")


def load_passing_result(path: Path) -> dict[str, object]:
    """Load a child result and reject missing or non-passing verdicts."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("pass") is not True:
        raise ValueError(f"Child result did not pass: {path}")
    return result


def matrix_pass(
    *,
    expected_result_count: int,
    results: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> bool:
    """Return a fail-closed aggregate verdict for one matrix lane."""
    return (
        expected_result_count > 0
        and not failures
        and len(results) == expected_result_count
        and all(result.get("pass") is True for result in results)
    )


def remove_stale_files(*paths: Path) -> None:
    """Remove artifacts that must be created by the current child run."""
    for path in paths:
        path.unlink(missing_ok=True)


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate and reap one owned child process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_bounded_child(
    command: list[str],
    *,
    log_file: TextIO,
    timeout_seconds: float,
) -> tuple[int, bool]:
    """Run one child in an owned process group with a hard timeout."""
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=child_environment(),
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_seconds), False
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        return int(process.returncode or 124), True
    except BaseException:
        terminate_process_group(process)
        raise


def failure_status(log: str, returncode: int) -> str:
    if (
        "CUDA out of memory" in log
        or "OutOfMemoryError" in log
        or "out of memory" in log.lower()
    ):
        return "OUT_OF_DEVICE_MEMORY"
    if "workspace overflow" in log or "capacity overflow" in log:
        return "CAPACITY_OVERFLOW"
    if (
        "exceeds int32" in log
        or "int32 implementation limit" in log
        or "at most INT_MAX" in log
    ):
        return "IMPLEMENTATION_LIMIT"
    if returncode < 0:
        return f"SIGNAL_{-returncode}"
    return "FAILED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("BENCHMARK_PROTOCOL.md"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument(
        "--quick", action="store_true", help="Run a small smoke subset."
    )
    parser.add_argument("--implementations", default="gsplat")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--depth-bucket-count", type=int, default=32)
    parser.add_argument("--depth-bucket-group-size", type=int, default=8)
    parser.add_argument(
        "--batches",
        default="1,8,32,64,128,256,1024",
    )
    parser.add_argument(
        "--resolutions",
        default="128x128,256x256,512x512",
    )
    parser.add_argument(
        "--scenes",
        default="synthetic-small,synthetic-medium",
    )
    parser.add_argument("--visible-fraction", type=float)
    parser.add_argument("--intersections-per-visible", type=float)
    parser.add_argument(
        "--direct-intersections-per-camera-gaussian",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--real-direct-visible-gaussians-per-view",
        type=float,
        help="Optional override applied to every real scene.",
    )
    parser.add_argument(
        "--real-direct-intersections-per-view-at-128",
        type=float,
        help="Optional override applied to every real scene.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--case-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--compact-projection-cache", action="store_true")
    parser.add_argument("--materialize-projected-records", action="store_true")
    parser.add_argument("--max-workspace-gib", type=float)
    parser.add_argument("--projection-cache", action="store_true")
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--linear-output", action="store_true")
    color_group.add_argument(
        "--authored-display-output",
        action="store_true",
    )
    parser.add_argument(
        "--gaussian-support-sigma",
        type=float,
        default=3.0,
        help="Use 3.0 for the OVRTX acceptance contract; 3.33 is the gsplat control.",
    )
    parser.add_argument("--semantic-min-alpha", type=float, default=0.01)
    parser.add_argument("--covariance-epsilon", type=float, default=None)
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument("--ray-gaussian-evaluation", action="store_true")
    parser.add_argument(
        "--multi-scene-counts",
        default="4",
        help="Comma-separated packed scene counts for custom-cuda-multiscene.",
    )
    args = parser.parse_args()

    if not args.protocol.is_file():
        raise FileNotFoundError(f"Benchmark protocol not found: {args.protocol}")
    if args.gaussian_support_sigma <= 0:
        raise ValueError("--gaussian-support-sigma must be positive.")
    if not 0 <= args.semantic_min_alpha <= 1:
        raise ValueError("--semantic-min-alpha must be in [0, 1].")
    if (
        args.tile_size <= 0
        or args.tile_size > 16
        or args.tile_size & (args.tile_size - 1)
    ):
        raise ValueError("--tile-size must be a power of two in [1, 16].")
    if (
        args.depth_bucket_count <= 0
        or args.depth_bucket_group_size <= 0
        or args.depth_bucket_count % args.depth_bucket_group_size
    ):
        raise ValueError(
            "Depth bucket count must be positive and divisible by the positive group size."
        )
    if args.visible_fraction is not None and args.visible_fraction <= 0:
        raise ValueError("--visible-fraction must be positive.")
    if (
        args.intersections_per_visible is not None
        and args.intersections_per_visible <= 0
    ):
        raise ValueError("--intersections-per-visible must be positive.")
    if args.direct_intersections_per_camera_gaussian <= 0:
        raise ValueError("--direct-intersections-per-camera-gaussian must be positive.")
    for value in (
        args.real_direct_visible_gaussians_per_view,
        args.real_direct_intersections_per_view_at_128,
    ):
        if value is not None and value <= 0:
            raise ValueError(
                "Real-scene direct per-view capacity overrides must be positive."
            )
    args.covariance_epsilon = resolve_covariance_epsilon(
        args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        compact_projection_cache=args.compact_projection_cache,
    )
    if args.compact_projection_cache and (
        args.tile_size != 1 or args.ray_gaussian_evaluation
    ):
        raise ValueError(
            "--compact-projection-cache requires --tile-size 1 and screen-space evaluation."
        )
    if args.rasterize_mode == "antialiased" and args.ray_gaussian_evaluation:
        raise ValueError(
            "--rasterize-mode antialiased is incompatible with --ray-gaussian-evaluation."
        )
    if args.materialize_projected_records and not args.compact_projection_cache:
        raise ValueError(
            "--materialize-projected-records requires --compact-projection-cache."
        )
    if args.max_workspace_gib is not None and args.max_workspace_gib <= 0:
        raise ValueError("--max-workspace-gib must be positive.")
    if (
        not math.isfinite(args.case_timeout_seconds)
        or args.case_timeout_seconds <= 0
    ):
        raise ValueError("--case-timeout-seconds must be finite and positive.")

    if args.quick:
        batches = [1, 8]
        resolutions = [(128, 128)]
        scenes = ["synthetic-small"]
    else:
        batches = [
            int(value.strip()) for value in args.batches.split(",") if value.strip()
        ]
        resolutions = []
        for value in args.resolutions.split(","):
            value = value.strip()
            if not value:
                continue
            width, height = value.lower().split("x", 1)
            resolutions.append((int(width), int(height)))
        scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    if (
        not batches
        or min(batches) <= 0
        or not resolutions
        or min(min(pair) for pair in resolutions) <= 0
        or not scenes
    ):
        raise ValueError(
            "Scenes, batches, and resolutions must contain positive values."
        )
    unsupported_scenes = set(scenes) - (SYNTHETIC_SCENES | REAL_SCENES)
    if unsupported_scenes:
        raise ValueError(
            "Unsupported scene(s): " + ", ".join(sorted(unsupported_scenes))
        )
    implementations = {
        item.strip() for item in args.implementations.split(",") if item.strip()
    }

    if "gsplat" in implementations:
        command = [
            sys.executable,
            "benchmarks/run_baselines.py",
            "--protocol",
            str(args.protocol),
            "--output",
            str(args.output / "gsplat"),
            "--implementations",
            "gsplat",
            "--scenes",
            ",".join(scenes),
            "--batches",
            ",".join(str(value) for value in batches),
            "--resolutions",
            ",".join(f"{width}x{height}" for width, height in resolutions),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
        ]
        command.extend(
            gsplat_rasterization_arguments(
                rasterize_mode=args.rasterize_mode,
                covariance_epsilon=args.covariance_epsilon,
            )
        )
        subprocess.check_call(command, env=child_environment())

    custom_results: list[dict[str, object]] = []
    custom_failures: list[dict[str, object]] = []
    matrix_verdicts: list[bool] = []
    if "custom-cuda" in implementations:
        custom_root = args.output / "custom-cuda"
        custom_root.mkdir(parents=True, exist_ok=True)
        custom_manifest = custom_root / mode_result_filename(
            "run-manifest",
            args.rasterize_mode,
        )
        remove_stale_files(custom_manifest)
        for scene in scenes:
            for width, height in resolutions:
                for batch in batches:
                    output_path = custom_root / mode_result_filename(
                        f"{scene}-b{batch}-{width}x{height}",
                        args.rasterize_mode,
                    )
                    if scene in SYNTHETIC_SCENES:
                        command = [
                            sys.executable,
                            "benchmarks/run_custom.py",
                            "--scene",
                            scene,
                            "--batch",
                            str(batch),
                            "--width",
                            str(width),
                            "--height",
                            str(height),
                            "--warmup",
                            str(args.warmup),
                            "--iterations",
                            str(args.iterations),
                            "--gaussian-support-sigma",
                            str(args.gaussian_support_sigma),
                            "--semantic-min-alpha",
                            str(args.semantic_min_alpha),
                            "--covariance-epsilon",
                            str(args.covariance_epsilon),
                            "--rasterize-mode",
                            args.rasterize_mode,
                            "--tile-size",
                            str(args.tile_size),
                            "--depth-bucket-count",
                            str(args.depth_bucket_count),
                            "--depth-bucket-group-size",
                            str(args.depth_bucket_group_size),
                            "--direct-intersections-per-camera-gaussian",
                            str(args.direct_intersections_per_camera_gaussian),
                            "--output",
                            str(output_path),
                        ]
                        if args.visible_fraction is not None:
                            command.extend(
                                [
                                    "--visible-fraction",
                                    str(args.visible_fraction),
                                ]
                            )
                        if args.intersections_per_visible is not None:
                            command.extend(
                                [
                                    "--intersections-per-visible",
                                    str(args.intersections_per_visible),
                                ]
                            )
                    else:
                        real_capacity = REAL_SCENE_DIRECT_CAPACITY[scene]
                        visible_per_view = (
                            args.real_direct_visible_gaussians_per_view
                            or real_capacity["visible_gaussians_per_view"]
                        )
                        intersections_per_view_at_128 = (
                            args.real_direct_intersections_per_view_at_128
                            or real_capacity["intersections_per_view_at_128"]
                        )
                        command = [
                            sys.executable,
                            "benchmarks/run_home_scan.py",
                            "--dataset-id",
                            scene,
                            "--batch",
                            str(batch),
                            "--width",
                            str(width),
                            "--height",
                            str(height),
                            "--warmup",
                            str(args.warmup),
                            "--iterations",
                            str(args.iterations),
                            "--gaussian-support-sigma",
                            str(args.gaussian_support_sigma),
                            "--semantic-min-alpha",
                            str(args.semantic_min_alpha),
                            "--semantic-scheme",
                            "spatial-grid",
                            "--semantic-grid",
                            "2,1,1",
                            "--covariance-epsilon",
                            str(args.covariance_epsilon),
                            "--rasterize-mode",
                            args.rasterize_mode,
                            "--tile-size",
                            str(args.tile_size),
                            "--depth-bucket-count",
                            str(args.depth_bucket_count),
                            "--depth-bucket-group-size",
                            str(args.depth_bucket_group_size),
                            "--direct-visible-gaussians-per-view",
                            str(visible_per_view),
                            "--direct-intersections-per-view-at-128",
                            str(intersections_per_view_at_128),
                            "--tight-depth-range",
                            "--output",
                            str(output_path),
                        ]
                    if args.ray_gaussian_evaluation:
                        command.append("--ray-gaussian-evaluation")
                    if args.compact_projection_cache:
                        command.append("--compact-projection-cache")
                    if args.materialize_projected_records:
                        command.append("--materialize-projected-records")
                    if args.max_workspace_gib is not None:
                        command.extend(
                            [
                                "--max-workspace-gib",
                                str(args.max_workspace_gib),
                            ]
                        )
                    if args.projection_cache:
                        command.append("--projection-cache")
                    if args.linear_output:
                        command.append("--linear-output")
                    elif args.authored_display_output:
                        command.append("--authored-display-output")
                    log_path = output_path.with_suffix(".log")
                    command_path = output_path.with_suffix(".command.json")
                    remove_stale_files(output_path)
                    command_path.write_text(
                        json.dumps(command, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    with log_path.open(
                        "w",
                        encoding="utf-8",
                    ) as log_file:
                        returncode, timed_out = run_bounded_child(
                            command,
                            log_file=log_file,
                            timeout_seconds=args.case_timeout_seconds,
                        )
                    if timed_out or returncode != 0:
                        if not args.continue_on_error:
                            if timed_out:
                                raise subprocess.TimeoutExpired(
                                    command,
                                    args.case_timeout_seconds,
                                )
                            raise subprocess.CalledProcessError(
                                returncode,
                                command,
                            )
                        log = log_path.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )
                        custom_failures.append(
                            {
                                "scene": scene,
                                "batch": batch,
                                "width": width,
                                "height": height,
                                "status": (
                                    "TIMEOUT"
                                    if timed_out
                                    else failure_status(log, returncode)
                                ),
                                "returncode": returncode,
                                "timed_out": timed_out,
                                "timeout_seconds": args.case_timeout_seconds,
                                "output": str(output_path),
                                "log": str(log_path),
                                "command": str(command_path),
                                "log_tail": "\n".join(log.splitlines()[-80:]),
                            }
                        )
                        continue
                    try:
                        result = load_passing_result(output_path)
                    except (OSError, json.JSONDecodeError, ValueError) as error:
                        if not args.continue_on_error:
                            raise RuntimeError(
                                f"Custom child result failed: {output_path}"
                            ) from error
                        log = log_path.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )
                        custom_failures.append(
                            {
                                "scene": scene,
                                "batch": batch,
                                "width": width,
                                "height": height,
                                "status": "RESULT_CONTRACT_FAILED",
                                "returncode": returncode,
                                "timed_out": False,
                                "timeout_seconds": args.case_timeout_seconds,
                                "output": str(output_path),
                                "log": str(log_path),
                                "command": str(command_path),
                                "detail": str(error),
                                "log_tail": "\n".join(log.splitlines()[-80:]),
                            }
                        )
                        continue
                    custom_results.append(result)
        expected_custom_result_count = len(scenes) * len(resolutions) * len(batches)
        custom_pass = matrix_pass(
            expected_result_count=expected_custom_result_count,
            results=custom_results,
            failures=custom_failures,
        )
        matrix_verdicts.append(custom_pass)
        manifest = {
            "schema_version": "custom-matrix/v1",
            "pass": custom_pass,
            "expected_result_count": expected_custom_result_count,
            "case_timeout_seconds": args.case_timeout_seconds,
            "covariance_epsilon": args.covariance_epsilon,
            "rasterize_mode": args.rasterize_mode,
            "semantic_min_alpha": args.semantic_min_alpha,
            "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
            "compact_projection_cache": args.compact_projection_cache,
            "materialize_projected_records": (args.materialize_projected_records),
            "max_workspace_gib": args.max_workspace_gib,
            "projection_cache": args.projection_cache,
            "depth_bucket_count": args.depth_bucket_count,
            "depth_bucket_group_size": args.depth_bucket_group_size,
            "real_scene_direct_capacity_defaults": (REAL_SCENE_DIRECT_CAPACITY),
            "real_direct_visible_gaussians_per_view_override": (
                args.real_direct_visible_gaussians_per_view
            ),
            "real_direct_intersections_per_view_at_128_override": (
                args.real_direct_intersections_per_view_at_128
            ),
            "direct_intersections_per_camera_gaussian": (
                args.direct_intersections_per_camera_gaussian
            ),
            "result_count": len(custom_results),
            "failure_count": len(custom_failures),
            "failures": custom_failures,
            "results": [
                {
                    "scene": result_scene_id(result),
                    "batch": result["batch"],
                    "width": result["width"],
                    "height": result["height"],
                    "images_per_second": result["images_per_second"],
                    "frame_ms_mean": result["frame_ms"]["mean"],
                    "peak_allocated_bytes": result["peak_allocated_bytes"],
                    "path": mode_result_filename(
                        (
                            f"{result_scene_id(result)}-b{result['batch']}-{result['width']}x{result['height']}"
                        ),
                        args.rasterize_mode,
                    ),
                }
                for result in custom_results
            ],
        }
        custom_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    multi_scene_results: list[dict[str, object]] = []
    multi_scene_failures: list[dict[str, object]] = []
    if "custom-cuda-multiscene" in implementations:
        scene_counts = [
            int(value) for value in args.multi_scene_counts.split(",") if value.strip()
        ]
        if not scene_counts or min(scene_counts) <= 0:
            raise ValueError("--multi-scene-counts must contain positive integers.")
        multi_root = args.output / "custom-cuda-multiscene"
        multi_root.mkdir(parents=True, exist_ok=True)
        multi_manifest_path = multi_root / mode_result_filename(
            "run-manifest",
            args.rasterize_mode,
        )
        remove_stale_files(multi_manifest_path)
        for scene in scenes:
            if scene not in SYNTHETIC_SCENES:
                continue
            for scene_count in scene_counts:
                for width, height in resolutions:
                    for batch in (value for value in batches if value >= scene_count):
                        output_path = multi_root / mode_result_filename(
                            (f"{scene}-scenes{scene_count}-b{batch}-{width}x{height}"),
                            args.rasterize_mode,
                        )
                        command = [
                            sys.executable,
                            "benchmarks/run_multi_scene.py",
                            "--scene",
                            scene,
                            "--scene-count",
                            str(scene_count),
                            "--batch",
                            str(batch),
                            "--width",
                            str(width),
                            "--height",
                            str(height),
                            "--warmup",
                            str(args.warmup),
                            "--iterations",
                            str(args.iterations),
                            "--gaussian-support-sigma",
                            str(args.gaussian_support_sigma),
                            "--semantic-min-alpha",
                            str(args.semantic_min_alpha),
                            "--covariance-epsilon",
                            str(args.covariance_epsilon),
                            "--rasterize-mode",
                            args.rasterize_mode,
                            "--tile-size",
                            str(args.tile_size),
                            "--depth-bucket-count",
                            str(args.depth_bucket_count),
                            "--depth-bucket-group-size",
                            str(args.depth_bucket_group_size),
                            "--direct-intersections-per-camera-gaussian",
                            str(args.direct_intersections_per_camera_gaussian),
                            "--output",
                            str(output_path),
                        ]
                        if args.visible_fraction is not None:
                            command.extend(
                                [
                                    "--visible-fraction",
                                    str(args.visible_fraction),
                                ]
                            )
                        if args.intersections_per_visible is not None:
                            command.extend(
                                [
                                    "--intersections-per-visible",
                                    str(args.intersections_per_visible),
                                ]
                            )
                        if args.ray_gaussian_evaluation:
                            command.append("--ray-gaussian-evaluation")
                        if args.compact_projection_cache:
                            command.append("--compact-projection-cache")
                        if args.materialize_projected_records:
                            command.append("--materialize-projected-records")
                        if args.max_workspace_gib is not None:
                            command.extend(
                                [
                                    "--max-workspace-gib",
                                    str(args.max_workspace_gib),
                                ]
                            )
                        if args.projection_cache:
                            command.append("--projection-cache")
                        if args.linear_output:
                            command.append("--linear-output")
                        elif args.authored_display_output:
                            command.append("--authored-display-output")
                        log_path = output_path.with_suffix(".log")
                        command_path = output_path.with_suffix(".command.json")
                        remove_stale_files(output_path)
                        command_path.write_text(
                            json.dumps(command, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        with log_path.open(
                            "w",
                            encoding="utf-8",
                        ) as log_file:
                            returncode, timed_out = run_bounded_child(
                                command,
                                log_file=log_file,
                                timeout_seconds=args.case_timeout_seconds,
                            )
                        if timed_out or returncode != 0:
                            if not args.continue_on_error:
                                if timed_out:
                                    raise subprocess.TimeoutExpired(
                                        command,
                                        args.case_timeout_seconds,
                                    )
                                raise subprocess.CalledProcessError(
                                    returncode,
                                    command,
                                )
                            log = log_path.read_text(
                                encoding="utf-8",
                                errors="replace",
                            )
                            multi_scene_failures.append(
                                {
                                    "scene": scene,
                                    "scene_count": scene_count,
                                    "batch": batch,
                                    "width": width,
                                    "height": height,
                                    "status": (
                                        "TIMEOUT"
                                        if timed_out
                                        else failure_status(log, returncode)
                                    ),
                                    "returncode": returncode,
                                    "timed_out": timed_out,
                                    "timeout_seconds": args.case_timeout_seconds,
                                    "output": str(output_path),
                                    "log": str(log_path),
                                    "command": str(command_path),
                                    "log_tail": "\n".join(log.splitlines()[-80:]),
                                }
                            )
                            continue
                        try:
                            result = load_passing_result(output_path)
                        except (
                            OSError,
                            json.JSONDecodeError,
                            ValueError,
                        ) as error:
                            if not args.continue_on_error:
                                raise RuntimeError(
                                    f"Multi-scene child result failed: {output_path}"
                                ) from error
                            log = log_path.read_text(
                                encoding="utf-8",
                                errors="replace",
                            )
                            multi_scene_failures.append(
                                {
                                    "scene": scene,
                                    "scene_count": scene_count,
                                    "batch": batch,
                                    "width": width,
                                    "height": height,
                                    "status": "RESULT_CONTRACT_FAILED",
                                    "returncode": returncode,
                                    "timed_out": False,
                                    "timeout_seconds": args.case_timeout_seconds,
                                    "output": str(output_path),
                                    "log": str(log_path),
                                    "command": str(command_path),
                                    "detail": str(error),
                                    "log_tail": "\n".join(log.splitlines()[-80:]),
                                }
                            )
                            continue
                        multi_scene_results.append(result)
        expected_multi_scene_result_count = sum(
            1
            for scene in scenes
            if scene in SYNTHETIC_SCENES
            for scene_count in scene_counts
            for _width, _height in resolutions
            for batch in batches
            if batch >= scene_count
        )
        multi_scene_pass = matrix_pass(
            expected_result_count=expected_multi_scene_result_count,
            results=multi_scene_results,
            failures=multi_scene_failures,
        )
        matrix_verdicts.append(multi_scene_pass)
        multi_manifest = {
            "schema_version": "custom-multi-scene-matrix/v1",
            "pass": multi_scene_pass,
            "expected_result_count": expected_multi_scene_result_count,
            "case_timeout_seconds": args.case_timeout_seconds,
            "covariance_epsilon": args.covariance_epsilon,
            "rasterize_mode": args.rasterize_mode,
            "semantic_min_alpha": args.semantic_min_alpha,
            "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
            "compact_projection_cache": args.compact_projection_cache,
            "materialize_projected_records": (args.materialize_projected_records),
            "max_workspace_gib": args.max_workspace_gib,
            "projection_cache": args.projection_cache,
            "depth_bucket_count": args.depth_bucket_count,
            "depth_bucket_group_size": args.depth_bucket_group_size,
            "direct_intersections_per_camera_gaussian": (
                args.direct_intersections_per_camera_gaussian
            ),
            "result_count": len(multi_scene_results),
            "failure_count": len(multi_scene_failures),
            "failures": multi_scene_failures,
            "results": [
                {
                    "scene_family": result["scene_family"],
                    "scene_count": result["scene_count"],
                    "gaussians_per_scene": result["gaussians_per_scene"],
                    "batch": result["batch"],
                    "width": result["width"],
                    "height": result["height"],
                    "images_per_second": result["images_per_second"],
                    "frame_ms_mean": result["frame_ms"]["mean"],
                    "peak_allocated_bytes": result["peak_allocated_bytes"],
                    "path": mode_result_filename(
                        (
                            f"{result['scene_family']}-"
                            f"scenes{result['scene_count']}-"
                            f"b{result['batch']}-"
                            f"{result['width']}x{result['height']}"
                        ),
                        args.rasterize_mode,
                    ),
                }
                for result in multi_scene_results
            ],
        }
        multi_manifest_path.write_text(
            json.dumps(multi_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    unsupported = implementations - {
        "gsplat",
        "custom-cuda",
        "custom-cuda-multiscene",
    }
    if unsupported:
        raise ValueError(
            "Unsupported matrix implementation(s): " + ", ".join(sorted(unsupported))
        )
    if matrix_verdicts and not all(matrix_verdicts):
        raise SystemExit(1)
    if "custom-cuda" in implementations:
        print("CUSTOM_CUDA_MATRIX_OK")
    if "custom-cuda-multiscene" in implementations:
        print("CUSTOM_CUDA_MULTISCENE_MATRIX_OK")


if __name__ == "__main__":
    main()
