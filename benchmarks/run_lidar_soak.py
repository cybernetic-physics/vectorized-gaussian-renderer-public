"""Finite stability soak for the opt-in Gaussian LiDAR backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from benchmarks.run_lidar import allocate_outputs, ray_pattern, synthetic_scene
from isaacsim_gaussian_renderer import GaussianLidarService
from isaacsim_gaussian_renderer.cuda_lidar_backend import CudaLidarBackend


def validate_output(outputs: dict[str, torch.Tensor]) -> None:
    """Check valid returns; callers synchronize before invoking this audit."""

    valid = outputs["valid"]
    if not torch.isfinite(outputs["range_m"][valid]).all().item():
        raise AssertionError("Non-finite valid range during soak.")
    if not torch.isfinite(outputs["position_world_m"][valid]).all().item():
        raise AssertionError("Non-finite valid position during soak.")
    if not torch.isfinite(outputs["intensity"][valid]).all().item():
        raise AssertionError("Non-finite valid intensity during soak.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rays", type=int, default=16_384)
    parser.add_argument("--returns", type=int, choices=(1, 2), default=2)
    parser.add_argument("--gaussians", type=int, default=100_000)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/lidar/soak-10min.json")
    )
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.check_every <= 0:
        raise ValueError("duration-seconds and check-every must be positive")
    device = torch.device("cuda:0")
    service = GaussianLidarService(
        CudaLidarBackend(max_scenes=1),
        max_sensors=args.batch,
        max_rays=args.rays,
        returns=args.returns,
    )
    service.initialize(None, device)
    service.load_scene(0, **synthetic_scene(args.gaussians, device))
    service.synchronize()
    directions, times = ray_pattern(args.rays, device)
    scene_ids = torch.zeros((args.batch,), device=device, dtype=torch.int64)
    transforms = [torch.eye(4, device=device).repeat(args.batch, 1, 1).contiguous() for _ in range(4)]
    outputs = [allocate_outputs(args.batch, args.rays, args.returns, device) for _ in range(4)]
    for index, transform in enumerate(transforms):
        transform[:, 1, 3] = 0.001 * index
        service.render_lidar(directions, times, transform, scene_ids, outputs=outputs[index])
    service.synchronize()
    # Boolean indexing in the periodic audit itself uses temporary allocator
    # storage. Prewarm those checks before taking the renderer memory baseline
    # so allocator growth from the audit cannot be mistaken for a LiDAR leak.
    for output in outputs:
        validate_output(output)
    service.backend.check_errors(synchronize=False)
    allocated_start = torch.cuda.memory_allocated(device)
    reserved_start = torch.cuda.memory_reserved(device)
    checks = 0
    iterations = 0
    started = time.monotonic()
    while time.monotonic() - started < args.duration_seconds:
        slot = iterations % len(outputs)
        transforms[slot][:, 2, 3].fill_(0.0001 * (iterations % 101))
        service.render_lidar(
            directions,
            times,
            transforms[slot],
            scene_ids,
            outputs=outputs[slot],
        )
        iterations += 1
        if iterations % args.check_every == 0:
            service.synchronize()
            validate_output(outputs[slot])
            service.backend.check_errors(synchronize=False)
            checks += 1
    service.synchronize()
    counters = service.backend.check_errors(synchronize=False)
    elapsed = time.monotonic() - started
    allocated_end = torch.cuda.memory_allocated(device)
    reserved_end = torch.cuda.memory_reserved(device)
    if allocated_end != allocated_start or reserved_end != reserved_start:
        raise AssertionError(
            "Steady-state CUDA memory changed during soak: "
            f"allocated {allocated_start}->{allocated_end}, reserved {reserved_start}->{reserved_end}."
        )
    result = {
        "schema_version": "gaussian-lidar-soak/v1",
        "pass": True,
        "duration_seconds": elapsed,
        "iterations": iterations,
        "batch": args.batch,
        "rays": args.rays,
        "returns": args.returns,
        "gaussians": args.gaussians,
        "periodic_checks": checks,
        "allocated_start": allocated_start,
        "allocated_end": allocated_end,
        "reserved_start": reserved_start,
        "reserved_end": reserved_end,
        "counters": counters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("GAUSSIAN_LIDAR_SOAK_OK")
    service.shutdown()


if __name__ == "__main__":
    main()
