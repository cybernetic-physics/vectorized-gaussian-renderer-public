"""Run the full OVRTX matrix in isolated processes and retain capacity failures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCHMARK_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "isaacsim_gaussian_renderer"
    / "benchmark_schema.py"
)


def _load_benchmark_schema() -> Any:
    """Load the stdlib-only schema module directly by file path.

    This orchestrator stays importable without torch, so it must not
    import the ``isaacsim_gaussian_renderer`` package whose ``__init__``
    pulls in the renderer stack.
    """
    spec = importlib.util.spec_from_file_location(
        "_vgr_benchmark_schema",
        BENCHMARK_SCHEMA_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load benchmark schema module: {BENCHMARK_SCHEMA_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fabric_scene_delegate_record = (
    _load_benchmark_schema().fabric_scene_delegate_record
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_resolutions(value: str) -> list[tuple[int, int]]:
    resolutions = []
    for item in parse_csv(value):
        width, height = item.lower().split("x", 1)
        resolutions.append((int(width), int(height)))
    return resolutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("BENCHMARK_PROTOCOL.md"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=os.environ.get(
            "ISAACSIM_PYTHON",
            "/isaac-sim/python.sh",
        ),
    )
    parser.add_argument(
        "--scenes",
        default="synthetic-small,synthetic-medium",
    )
    parser.add_argument("--batches", default="1,8,32,64,128,256")
    parser.add_argument(
        "--resolutions",
        default="128x128,256x256,512x512",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--projection-mode",
        choices=("perspective", "tangential"),
        default="perspective",
    )
    parser.add_argument(
        "--sorting-mode",
        choices=("zDepth", "cameraDistance", "rayHitDistance"),
        default="zDepth",
    )
    parser.add_argument(
        "--home-scan-path",
        type=Path,
        default=Path("/workspace/datasets/home-scan-lod0.ply"),
    )
    parser.add_argument(
        "--public-scene-path",
        type=Path,
        default=Path(
            "/workspace/datasets/public-gaussian/"
            "Voxel51_gaussian_splatting/FO_dataset/train/"
            "point_cloud/iteration_30000/point_cloud.ply"
        ),
    )
    parser.add_argument("--home-scan-fit-margin", type=float, default=1.15)
    parser.add_argument("--focal-scale", type=float, default=0.9)
    parser.add_argument(
        "--semantic-scheme",
        choices=("index-modulo", "spatial-grid"),
        default="index-modulo",
    )
    parser.add_argument("--semantic-grid", default="2,2,2")
    parser.add_argument(
        "--aa-op",
        choices=("none", "taa", "fxaa", "dlss", "rtxaa"),
        default="none",
    )
    parser.add_argument("--fractional-opacity", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse valid MEASURED rows from an incomplete manifest with the "
            "same immutable configuration."
        ),
    )
    parser.add_argument(
        "--save-reference-artifacts",
        action="store_true",
        help="Save large CPU arrays and PNGs for every matrix case.",
    )
    return parser.parse_args()


def result_path(case_dir: Path) -> Path | None:
    manifest_path = case_dir / "run-manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("result_count") != 1:
        return None
    path = case_dir / f"{manifest['results'][0]}.json"
    return path if path.is_file() else None


def reusable_cases(
    manifest_path: Path,
    *,
    configuration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return verified successful rows from a compatible interrupted matrix."""
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("configuration") != configuration:
        raise ValueError(
            "--resume requires the existing matrix configuration to match."
        )
    reusable: dict[str, dict[str, Any]] = {}
    for case in manifest.get("cases", []):
        case_name = str(case.get("case", ""))
        if not case_name or case_name in reusable:
            raise ValueError(
                "Existing matrix manifest contains missing or duplicate case names."
            )
        if case.get("status") != "MEASURED":
            continue
        output_path = result_path(manifest_path.parent / case_name)
        if output_path is None:
            continue
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result.get("status", {}).get("pass") is True:
            reusable[case_name] = case
    return reusable


def log_tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()[-lines:]
    )


def failure_status(log: str) -> str:
    if (
        "ERROR_OUT_OF_DEVICE_MEMORY" in log
        or "Out of GPU memory" in log
    ):
        return "OUT_OF_DEVICE_MEMORY"
    if (
        "Too many open files" in log
        or "file descriptors: 0" in log
    ):
        return "FILE_DESCRIPTOR_LIMIT"
    return "FAILED"


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
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


def write_manifest(
    path: Path,
    *,
    created_utc: str,
    cases: list[dict[str, Any]],
    complete: bool,
    configuration: dict[str, Any],
) -> None:
    counts: dict[str, int] = {}
    for case in cases:
        status = str(case["status"])
        counts[status] = counts.get(status, 0) + 1
    manifest = {
        "schema_version": "ovrtx-matrix/v1",
        "created_utc": created_utc,
        "updated_utc": datetime.now(UTC).isoformat(),
        "complete": complete,
        "case_count": len(cases),
        "status_counts": counts,
        "configuration": configuration,
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "cases": cases,
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError(f"Benchmark protocol not found: {args.protocol}")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Warmup cannot be negative and iterations must be positive.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if args.home_scan_fit_margin <= 0 or args.focal_scale <= 0:
        raise ValueError("Fit margin and focal scale must be positive.")

    scenes = parse_csv(args.scenes)
    batches = parse_int_csv(args.batches)
    resolutions = parse_resolutions(args.resolutions)
    if not scenes or not batches or not resolutions:
        raise ValueError("Scenes, batches, and resolutions must be non-empty.")
    if min(batches) <= 0 or min(min(pair) for pair in resolutions) <= 0:
        raise ValueError("Batch sizes and resolutions must be positive.")

    args.output.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(UTC).isoformat()
    cases: list[dict[str, Any]] = []
    manifest_path = args.output / "run-manifest.json"
    total_cases = len(scenes) * len(resolutions) * len(batches)
    matrix_configuration = {
        "fabric_scene_delegate": fabric_scene_delegate_record(),
        "projection_mode": args.projection_mode,
        "sorting_mode": args.sorting_mode,
        "fractional_opacity": args.fractional_opacity,
        "semantic_scheme": args.semantic_scheme,
        "semantic_grid": args.semantic_grid,
        "anti_aliasing": args.aa_op,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "focal_scale": args.focal_scale,
        "home_scan_fit_margin": args.home_scan_fit_margin,
    }
    resumed = (
        reusable_cases(
            manifest_path,
            configuration=matrix_configuration,
        )
        if args.resume
        else {}
    )
    if not args.resume:
        manifest_path.unlink(missing_ok=True)
        write_manifest(
            manifest_path,
            created_utc=created_utc,
            cases=cases,
            complete=False,
            configuration=matrix_configuration,
        )

    for scene in scenes:
        for width, height in resolutions:
            for batch in batches:
                case_name = (
                    f"{scene}-{args.projection_mode}-"
                    f"b{batch}-{width}x{height}"
                )
                case_dir = args.output / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                if case_name in resumed:
                    case = dict(resumed[case_name])
                    case["resumed"] = True
                    cases.append(case)
                    write_manifest(
                        manifest_path,
                        created_utc=created_utc,
                        cases=cases,
                        complete=False,
                        configuration=matrix_configuration,
                    )
                    print(
                        "OVRTX_MATRIX_CASE "
                        + json.dumps(
                            {
                                "case": case_name,
                                "index": len(cases),
                                "total": total_cases,
                                "status": "RESUMED",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                (case_dir / "run-manifest.json").unlink(missing_ok=True)
                log_path = case_dir / "run.log"
                command = [
                    args.python,
                    "benchmarks/run_ovrtx.py",
                    "--protocol",
                    str(args.protocol),
                    "--output",
                    str(case_dir),
                    "--scene",
                    scene,
                    "--home-scan-path",
                    str(args.home_scan_path),
                    "--public-scene-path",
                    str(args.public_scene_path),
                    "--home-scan-fit-margin",
                    str(args.home_scan_fit_margin),
                    "--batch-size",
                    str(batch),
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--focal-scale",
                    str(args.focal_scale),
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                    "--projection-mode",
                    args.projection_mode,
                    "--sorting-mode",
                    args.sorting_mode,
                    "--semantic-scheme",
                    args.semantic_scheme,
                    "--semantic-grid",
                    args.semantic_grid,
                    "--aa-op",
                    args.aa_op,
                ]
                if args.fractional_opacity:
                    command.append("--fractional-opacity")
                if not args.save_reference_artifacts:
                    command.append("--skip-reference-artifacts")
                (case_dir / "command.json").write_text(
                    json.dumps(command, indent=2) + "\n",
                    encoding="utf-8",
                )

                started_utc = datetime.now(UTC).isoformat()
                start = time.perf_counter()
                with log_path.open("wb") as log_file:
                    process = subprocess.Popen(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    timed_out = False
                    try:
                        returncode = process.wait(
                            timeout=args.timeout_seconds
                        )
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        terminate_process_group(process)
                        returncode = process.returncode
                elapsed_seconds = time.perf_counter() - start

                output_path = result_path(case_dir)
                parsed_result = (
                    json.loads(output_path.read_text(encoding="utf-8"))
                    if output_path is not None
                    else None
                )
                if timed_out:
                    status = "TIMEOUT"
                elif (
                    returncode == 0
                    and parsed_result is not None
                    and parsed_result["status"]["pass"]
                ):
                    status = "MEASURED"
                elif returncode is not None and returncode < 0:
                    status = f"SIGNAL_{-returncode}"
                else:
                    status = failure_status(log_tail(log_path, lines=2_000))

                case: dict[str, Any] = {
                    "case": case_name,
                    "scene": scene,
                    "batch": batch,
                    "width": width,
                    "height": height,
                    "projection_mode": args.projection_mode,
                    "sorting_mode": args.sorting_mode,
                    "fractional_opacity": args.fractional_opacity,
                    "fabric_scene_delegate": fabric_scene_delegate_record(),
                    "status": status,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "timeout_seconds": args.timeout_seconds,
                    "started_utc": started_utc,
                    "elapsed_seconds": elapsed_seconds,
                    "output_dir": str(case_dir),
                    "result_json": (
                        str(output_path)
                        if output_path is not None
                        else None
                    ),
                    "log_tail": (
                        None
                        if status == "MEASURED"
                        else log_tail(log_path)
                    ),
                }
                if parsed_result is not None:
                    case["images_per_second"] = parsed_result["timing"][
                        "images_per_second"
                    ]
                    case["wall_ms_mean"] = parsed_result["timing"]["wall_ms"][
                        "mean"
                    ]
                    case["driver_process_memory_bytes"] = parsed_result[
                        "memory"
                    ]["driver_process_memory_bytes"]
                    case["driver_total_memory_bytes"] = parsed_result["memory"][
                        "driver_total_memory_bytes"
                    ]
                    case["runtime_token_readback"] = parsed_result.get(
                        "runtime_token_readback"
                    )
                cases.append(case)
                write_manifest(
                    manifest_path,
                    created_utc=created_utc,
                    cases=cases,
                    complete=False,
                    configuration=matrix_configuration,
                )
                print(
                    "OVRTX_MATRIX_CASE",
                    json.dumps(
                        {
                            "index": len(cases),
                            "total": total_cases,
                            "case": case_name,
                            "status": status,
                            "elapsed_seconds": elapsed_seconds,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    write_manifest(
        manifest_path,
        created_utc=created_utc,
        cases=cases,
        complete=True,
        configuration=matrix_configuration,
    )
    print(
        "OVRTX_MATRIX_COMPLETE",
        json.dumps(
            {
                "case_count": len(cases),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
