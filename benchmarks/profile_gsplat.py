"""Profile reproducible synthetic shared-scene gsplat matrix points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from gsplat.rendering import rasterization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", default="gsplat-synth-shared")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/profiles"))
    parser.add_argument("--num-cameras", type=int, default=64)
    parser.add_argument("--num-gaussians", type=int, default=100_000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--packed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--semantic-classes", type=int, default=64)
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--torch-profiler-iterations", type=int, default=4)
    return parser.parse_args()


def run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def stats(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * stdev / math.sqrt(len(values)) if values else math.nan
    return {
        "mean_ms": mean,
        "stdev_ms": stdev,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "ci95_ms": ci95,
    }


def make_scene(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    means = torch.randn((args.num_gaussians, 3), generator=generator, device=device)
    means[:, :2] *= 2.6
    means[:, 2] = means[:, 2].abs() * 1.5 + 4.0

    quats = torch.zeros((args.num_gaussians, 4), device=device)
    quats[:, 0] = 1.0
    scales = torch.full((args.num_gaussians, 3), 0.02, device=device)
    opacities = torch.full((args.num_gaussians,), 0.6, device=device)
    colors = torch.rand((args.num_gaussians, 3), generator=generator, device=device)
    semantic_ids = (torch.arange(args.num_gaussians, device=device) % args.semantic_classes).float()
    extra_signals = semantic_ids[:, None] / max(float(args.semantic_classes - 1), 1.0)

    viewmats = torch.eye(4, device=device).repeat(args.num_cameras, 1, 1)
    camera_offsets = torch.linspace(-1.5, 1.5, args.num_cameras, device=device)
    viewmats[:, 0, 3] = -camera_offsets

    focal = 0.9 * args.width
    intrinsics = torch.zeros((args.num_cameras, 3, 3), device=device)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = args.width / 2.0
    intrinsics[:, 1, 2] = args.height / 2.0
    intrinsics[:, 2, 2] = 1.0
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "extra_signals": extra_signals,
        "viewmats": viewmats,
        "intrinsics": intrinsics,
        "semantic_ids": semantic_ids,
    }


def tensor_checksum(scene: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in ("means", "quats", "scales", "opacities", "colors", "extra_signals", "viewmats", "intrinsics"):
        tensor = scene[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    gsplat_root = Path(os.environ.get("GSPLAT_SOURCE_PATH", "/workspace/src/gsplat"))
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "project_commit": run(["git", "rev-parse", "HEAD"], cwd=str(project_root)),
        "project_branch": run(["git", "branch", "--show-current"], cwd=str(project_root)),
        "gsplat_commit": run(["git", "rev-parse", "HEAD"], cwd=str(gsplat_root)),
        "driver": run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    }


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    scene = make_scene(args, device)
    checksum = tensor_checksum(scene)

    def render_once() -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return rasterization(
            scene["means"],
            scene["quats"],
            scene["scales"],
            scene["opacities"],
            scene["colors"],
            scene["viewmats"],
            scene["intrinsics"],
            args.width,
            args.height,
            packed=args.packed,
            render_mode="RGB+D",
            extra_signals=scene["extra_signals"],
            radius_clip=args.radius_clip,
        )

    for _ in range(args.warmup):
        render_once()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_before = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }

    event_times_ms: list[float] = []
    wall_start = time.perf_counter()
    rendered = alpha = None
    meta: dict[str, Any] = {}
    for index in range(args.iterations):
        torch.cuda.nvtx.range_push(f"iteration_{index}")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        rendered, alpha, meta = render_once()
        end_event.record()
        torch.cuda.nvtx.range_pop()
        end_event.synchronize()
        event_times_ms.append(start_event.elapsed_time(end_event))
    torch.cuda.synchronize()
    wall_elapsed_s = time.perf_counter() - wall_start

    profiler_trace = None
    profiler_summary: list[dict[str, Any]] = []
    if args.torch_profiler:
        profiler_trace = args.output_dir / f"{args.config_id}_torch_trace.json"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=False,
        ) as prof:
            for index in range(args.torch_profiler_iterations):
                torch.cuda.nvtx.range_push(f"torch_profiler_iteration_{index}")
                render_once()
                torch.cuda.nvtx.range_pop()
                prof.step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(profiler_trace))
        for item in prof.key_averages()[:80]:
            cuda_total = getattr(item, "cuda_time_total", getattr(item, "device_time_total", 0.0))
            self_cuda_total = getattr(item, "self_cuda_time_total", getattr(item, "self_device_time_total", 0.0))
            cuda_memory = getattr(item, "cuda_memory_usage", getattr(item, "device_memory_usage", 0))
            profiler_summary.append(
                {
                    "key": item.key,
                    "cpu_time_total_us": item.cpu_time_total,
                    "cuda_time_total_us": cuda_total,
                    "self_cuda_time_total_us": self_cuda_total,
                    "count": item.count,
                    "cpu_memory_usage_bytes": item.cpu_memory_usage,
                    "cuda_memory_usage_bytes": cuda_memory,
                }
            )

    visible = int((meta["radii"] > 0).all(dim=-1).sum().item())
    tile_intersections = int(meta["flatten_ids"].numel())
    tiles_per_gauss_sum = int(meta["tiles_per_gauss"].sum().item())
    time_stats = stats(event_times_ms)
    memory_after = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    result = {
        "schema_version": 1,
        "config_id": args.config_id,
        "implementation": "gsplat",
        "scene": "synthetic-shared",
        "dataset_checksum": checksum,
        "contract": {
            "outputs": ["rgb", "depth", "alpha", "semantic_id_scalar_proxy"],
            "note": "Semantic ID is represented as one extra rasterized scalar channel for profiling cost only.",
            "packed": args.packed,
            "render_mode": "RGB+D",
            "radius_clip": args.radius_clip,
        },
        "matrix_point": {
            "num_cameras": args.num_cameras,
            "num_gaussians": args.num_gaussians,
            "width": args.width,
            "height": args.height,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "timing": {
            **time_stats,
            "event_times_ms": event_times_ms,
            "wall_mean_ms": wall_elapsed_s * 1000.0 / args.iterations,
            "images_per_second": args.num_cameras * args.iterations / wall_elapsed_s,
            "megapixels_per_second": args.num_cameras * args.width * args.height * args.iterations / wall_elapsed_s / 1e6,
        },
        "memory": {
            "before": memory_before,
            "after": memory_after,
            "peak_allocated_gib": memory_after["max_allocated_bytes"] / (1024**3),
            "peak_reserved_gib": memory_after["max_reserved_bytes"] / (1024**3),
        },
        "work_counts": {
            "visible_gaussian_camera_pairs": visible,
            "tile_intersections": tile_intersections,
            "tiles_per_gauss_sum": tiles_per_gauss_sum,
            "tile_width": int(meta["tile_width"]),
            "tile_height": int(meta["tile_height"]),
            "tile_size": int(meta["tile_size"]),
        },
        "output_shapes": {
            "rendered": list(rendered.shape) if rendered is not None else None,
            "alpha": list(alpha.shape) if alpha is not None else None,
            "semantic": list(meta["render_extra_signals"].shape) if "render_extra_signals" in meta else None,
        },
        "environment": environment(),
        "torch_profiler_trace": str(profiler_trace) if profiler_trace is not None else None,
        "torch_profiler_summary": profiler_summary,
    }

    json_path = args.output_dir / f"{args.config_id}.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    row = {
        "config_id": args.config_id,
        "implementation": "gsplat",
        "scene": "synthetic-shared",
        "num_cameras": args.num_cameras,
        "num_gaussians": args.num_gaussians,
        "width": args.width,
        "height": args.height,
        "packed": args.packed,
        "mean_ms": time_stats["mean_ms"],
        "p50_ms": time_stats["p50_ms"],
        "p95_ms": time_stats["p95_ms"],
        "ci95_ms": time_stats["ci95_ms"],
        "wall_mean_ms": result["timing"]["wall_mean_ms"],
        "images_per_second": result["timing"]["images_per_second"],
        "megapixels_per_second": result["timing"]["megapixels_per_second"],
        "peak_allocated_gib": result["memory"]["peak_allocated_gib"],
        "peak_reserved_gib": result["memory"]["peak_reserved_gib"],
        "visible_gaussian_camera_pairs": visible,
        "tile_intersections": tile_intersections,
        "dataset_checksum": checksum,
    }
    csv_path = args.output_dir / "profile_runs.csv"
    write_csv(csv_path, row)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
