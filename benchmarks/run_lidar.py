"""Full-output latency and throughput benchmark for GaussianLidarService."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import time

import torch

from isaacsim_gaussian_renderer import GaussianLidarService
from isaacsim_gaussian_renderer.cuda_lidar_backend import CudaLidarBackend
from isaacsim_gaussian_renderer.lidar_contracts import LIDAR_OUTPUT_SPECS
from isaacsim_gaussian_renderer.ply_loader import load_ply_to_canonical_gaussians


HOME_SCAN_COUNT = 21_497_908
HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("analytic", "medium", "home-scan"), required=True)
    parser.add_argument("--home-scan-ply", type=Path)
    parser.add_argument("--batches", type=parse_int_list, default=parse_int_list("1,8,32,64,128"))
    parser.add_argument("--rays", type=parse_int_list, default=parse_int_list("16384,65536,131072"))
    parser.add_argument("--returns", type=parse_int_list, default=parse_int_list("1,2"))
    parser.add_argument("--modes", choices=("latency", "throughput", "both"), default="both")
    parser.add_argument("--transform-mode", choices=("static", "changing", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--ring-size", type=int, default=4)
    parser.add_argument("--rebuild-iterations", type=int, default=3)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lidar/benchmarks"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_scene(count: int, device: torch.device) -> dict[str, torch.Tensor]:
    width = math.ceil(math.sqrt(count))
    index = torch.arange(count, device=device)
    y = ((index % width).to(torch.float32) - 0.5 * width) * 0.04
    z = ((index // width).to(torch.float32) - 0.5 * width) * 0.04
    layer = (index % 7).to(torch.float32)
    means = torch.stack((8.0 + 0.75 * layer, y, z), dim=1).contiguous()
    scales = torch.empty((count, 3), device=device, dtype=torch.float32)
    scales[:, 0] = 0.006
    scales[:, 1] = 0.035
    scales[:, 2] = 0.035
    return {
        "means": means,
        "scales": scales,
        "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(count, 1).contiguous(),
        "opacities": torch.full((count,), 0.8, device=device),
        "semantic_ids": (index % 32).to(torch.int64).contiguous(),
        "reflectivity": (0.2 + 0.6 * ((index % 17).to(torch.float32) / 16.0)).contiguous(),
    }


def load_scene(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, torch.Tensor], str]:
    if args.scene == "analytic":
        return synthetic_scene(4_096, device), "synthetic-analytic-v1"
    if args.scene == "medium":
        return synthetic_scene(100_000, device), "synthetic-medium-100k-v1"
    if args.home_scan_ply is None or not args.home_scan_ply.is_file():
        raise FileNotFoundError("Full Home Scan requires --home-scan-ply; no substitute or LOD is permitted.")
    checksum = sha256(args.home_scan_ply)
    if checksum != HOME_SCAN_SHA256:
        raise ValueError(f"Home Scan SHA-256 mismatch: {checksum}")
    canonical = load_ply_to_canonical_gaussians(args.home_scan_ply, device=device)
    if canonical.means.shape[0] != HOME_SCAN_COUNT:
        raise ValueError(f"Home Scan Gaussian count mismatch: {canonical.means.shape[0]}")
    return {
        "means": canonical.means,
        "scales": canonical.scales,
        "rotations": canonical.rotations,
        "opacities": canonical.opacities,
        "semantic_ids": torch.zeros((HOME_SCAN_COUNT,), device=device, dtype=torch.int64),
        "reflectivity": None,
    }, checksum


def ray_pattern(count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(count, device=device, dtype=torch.float32)
    azimuth = -math.pi + (2.0 * math.pi * index / count)
    elevation = -0.26 + 0.52 * ((index % 128.0) / 127.0)
    horizontal = torch.cos(elevation)
    directions = torch.stack(
        (horizontal * torch.cos(azimuth), horizontal * torch.sin(azimuth), torch.sin(elevation)),
        dim=1,
    ).contiguous()
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    times = (torch.arange(count, device=device, dtype=torch.int64) * 100).contiguous()
    return directions, times


def allocate_outputs(batch: int, rays: int, returns: int, device: torch.device):
    result = {}
    for name, dtype in LIDAR_OUTPUT_SPECS.items():
        shape = (
            (batch, rays, returns, 3)
            if name == "position_world_m"
            else (batch, rays)
            if name == "return_count"
            else (batch, rays, returns)
        )
        result[name] = torch.empty(shape, device=device, dtype=dtype)
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[position]


def output_bytes(outputs: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in outputs.values())


def benchmark_row(
    service: GaussianLidarService,
    *,
    batch: int,
    rays: int,
    returns: int,
    mode: str,
    changing: bool,
    warmup: int,
    iterations: int,
    ring_size: int,
    device: torch.device,
) -> dict[str, object]:
    directions, times = ray_pattern(rays, device)
    scene_ids = torch.zeros((batch,), device=device, dtype=torch.int64)
    transforms = []
    for index in range(max(iterations, warmup, ring_size)):
        value = torch.eye(4, device=device).repeat(batch, 1, 1).contiguous()
        if changing:
            value[:, 1, 3] = 0.002 * index
        transforms.append(value)
    ring = [allocate_outputs(batch, rays, returns, device) for _ in range(ring_size)]
    for index in range(warmup):
        service.render_lidar(
            directions,
            times,
            transforms[index],
            scene_ids,
            outputs=ring[index % ring_size],
            returns=returns,
        )
    service.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    counters_before = service.backend.read_counters(synchronize=False)
    gpu_samples_ms: list[float] = []
    wall_samples_ms: list[float] = []
    if mode == "latency":
        for index in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            service.render_lidar(
                directions,
                times,
                transforms[index],
                scene_ids,
                outputs=ring[index % ring_size],
                returns=returns,
            )
            end.record()
            end.synchronize()
            wall_samples_ms.append((time.perf_counter() - wall_start) * 1000.0)
            gpu_samples_ms.append(start.elapsed_time(end))
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for index in range(iterations):
            service.render_lidar(
                directions,
                times,
                transforms[index],
                scene_ids,
                outputs=ring[index % ring_size],
                returns=returns,
            )
        end.record()
        end.synchronize()
        elapsed_gpu = start.elapsed_time(end)
        elapsed_wall = (time.perf_counter() - wall_start) * 1000.0
        gpu_samples_ms = [elapsed_gpu / iterations] * iterations
        wall_samples_ms = [elapsed_wall / iterations] * iterations
    allocated_after = torch.cuda.memory_allocated(device)
    counters_after = service.backend.check_errors(synchronize=False)
    counter_deltas = {name: counters_after[name] - counters_before[name] for name in counters_after}
    p50 = median(gpu_samples_ms)
    p95 = percentile(gpu_samples_ms, 0.95)
    seconds = mean(gpu_samples_ms) / 1000.0
    bytes_per_call = output_bytes(ring[0])
    return {
        "batch": batch,
        "rays": rays,
        "returns": returns,
        "mode": mode,
        "transform_mode": "changing" if changing else "static",
        "iterations": iterations,
        "gpu_mean_ms": mean(gpu_samples_ms),
        "gpu_p50_ms": p50,
        "gpu_p95_ms": p95,
        "wall_mean_ms": mean(wall_samples_ms),
        "rays_per_second": batch * rays / seconds,
        "scans_per_second": batch / seconds,
        "scans_per_environment_per_second": 1.0 / seconds,
        "output_bytes_per_call": bytes_per_call,
        "output_write_gb_per_second": bytes_per_call / seconds / 1.0e9,
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
        "steady_state_allocation_delta": allocated_after - allocated_before,
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "counter_deltas": counter_deltas,
        "node_visits_per_ray": counter_deltas["node_visits"] / counter_deltas["rays_traced"],
        "primitive_tests_per_ray": counter_deltas["leaf_tests"] / counter_deltas["rays_traced"],
    }


def main() -> None:
    args = parse_args()
    if (
        args.warmup <= 0
        or args.iterations <= 0
        or args.ring_size <= 0
        or args.rebuild_iterations <= 0
    ):
        raise ValueError("warmup, iterations, ring size and rebuild iterations must be positive")
    if any(value not in (1, 2) for value in args.returns):
        raise ValueError("returns list may contain only 1 and 2")
    device = torch.device("cuda:0")
    scene, checksum = load_scene(args, device)
    backend = CudaLidarBackend(max_scenes=1)
    service = GaussianLidarService(
        backend,
        max_sensors=max(args.batches),
        max_rays=max(args.rays),
        returns=max(args.returns),
    )
    service.initialize(stage=None, device=device)
    build_start = torch.cuda.Event(enable_timing=True)
    build_end = torch.cuda.Event(enable_timing=True)
    build_wall_start = time.perf_counter()
    build_start.record()
    service.load_scene(0, **scene)
    build_end.record()
    build_end.synchronize()
    build_gpu_ms = build_start.elapsed_time(build_end)
    build_wall_ms = (time.perf_counter() - build_wall_start) * 1000.0
    service.synchronize()
    rebuild_gpu_ms = []
    rebuild_wall_ms = []
    for revision in range(1, args.rebuild_iterations + 1):
        rebuild_start = torch.cuda.Event(enable_timing=True)
        rebuild_end = torch.cuda.Event(enable_timing=True)
        rebuild_wall_start = time.perf_counter()
        rebuild_start.record()
        service.revise_scene(0, revision=revision)
        rebuild_end.record()
        rebuild_end.synchronize()
        rebuild_gpu_ms.append(rebuild_start.elapsed_time(rebuild_end))
        rebuild_wall_ms.append((time.perf_counter() - rebuild_wall_start) * 1000.0)
        service.synchronize()
    modes = ("latency", "throughput") if args.modes == "both" else (args.modes,)
    transform_modes = (False, True) if args.transform_mode == "both" else (args.transform_mode == "changing",)
    rows = []
    for batch in args.batches:
        for rays in args.rays:
            for returns in args.returns:
                for changing in transform_modes:
                    for mode in modes:
                        rows.append(
                            benchmark_row(
                                service,
                                batch=batch,
                                rays=rays,
                                returns=returns,
                                mode=mode,
                                changing=changing,
                                warmup=args.warmup,
                                iterations=args.iterations,
                                ring_size=args.ring_size,
                                device=device,
                            )
                        )
    result = {
        "schema_version": "gaussian-lidar-benchmark/v1",
        "pass": True,
        "run_id": args.run_id,
        "scene": args.scene,
        "scene_checksum": checksum,
        "gaussian_count": int(scene["means"].shape[0]),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "build_gpu_ms": build_gpu_ms,
        "build_wall_ms": build_wall_ms,
        "explicit_rebuild_gpu_ms": rebuild_gpu_ms,
        "explicit_rebuild_wall_ms": rebuild_wall_ms,
        "persistent_bvh_bytes": backend.bvh_bytes,
        "scene_attribute_bytes": backend.scene_attribute_bytes,
        "service_workspace_bytes": backend.workspace_bytes,
        "rows": rows,
    }
    output_root = args.output_dir / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "counter_deltas"}
        flat.update({f"counter_{key}": value for key, value in row["counter_deltas"].items()})
        flat_rows.append(flat)
    with (output_root / "rows.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("GAUSSIAN_LIDAR_BENCHMARK_OK")
    service.shutdown()


if __name__ == "__main__":
    main()
