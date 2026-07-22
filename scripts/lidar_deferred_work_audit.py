"""Detect cross-call batching, cached outputs, deferred writes, and guard damage."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from isaacsim_gaussian_renderer import GaussianLidarService
from isaacsim_gaussian_renderer.cuda_lidar_backend import CudaLidarBackend
from isaacsim_gaussian_renderer.lidar_contracts import LIDAR_OUTPUT_SPECS


@dataclass
class GuardedOutputs:
    outputs: dict[str, torch.Tensor]
    storage: dict[str, torch.Tensor]
    guard: int
    sentinels: dict[str, float | int | bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-call", action="store_true")
    parser.add_argument("--seed", type=int, default=92317)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lidar/deferred-work-audit.json"),
    )
    return parser.parse_args()


def output_shapes(batch: int, rays: int, returns: int) -> dict[str, tuple[int, ...]]:
    return {
        name: (
            (batch, rays, returns, 3)
            if name == "position_world_m"
            else (batch, rays)
            if name == "return_count"
            else (batch, rays, returns)
        )
        for name in LIDAR_OUTPUT_SPECS
    }


def allocate_guarded_outputs(
    batch: int,
    rays: int,
    returns: int,
    *,
    device: torch.device,
    poison_id: int,
    guard: int = 37,
) -> GuardedOutputs:
    outputs: dict[str, torch.Tensor] = {}
    storage: dict[str, torch.Tensor] = {}
    sentinels: dict[str, float | int | bool] = {}
    for offset, (name, dtype) in enumerate(LIDAR_OUTPUT_SPECS.items()):
        shape = output_shapes(batch, rays, returns)[name]
        numel = 1
        for dimension in shape:
            numel *= dimension
        if dtype == torch.bool:
            sentinel: float | int | bool = True
            poison: float | int | bool = bool(poison_id % 2)
        elif dtype.is_floating_point:
            sentinel = -9000.5 - offset
            poison = 1000.25 + poison_id + offset
        else:
            sentinel = -9_000_000 - offset
            poison = 1_000_000 + poison_id + offset
        base = torch.full((numel + 2 * guard,), sentinel, device=device, dtype=dtype)
        view = base[guard : guard + numel].view(shape)
        view.fill_(poison)
        outputs[name] = view
        storage[name] = base
        sentinels[name] = sentinel
    return GuardedOutputs(outputs, storage, guard, sentinels)


def fingerprint(outputs: dict[str, torch.Tensor], weights: dict[str, torch.Tensor]) -> torch.Tensor:
    values = []
    for name, tensor in outputs.items():
        flattened = tensor.reshape(-1).to(torch.float64)
        values.append(torch.sum(flattened * weights[name]))
    return torch.stack(values)


def guards_intact(guarded: GuardedOutputs) -> bool:
    for name, base in guarded.storage.items():
        sentinel = guarded.sentinels[name]
        if not torch.all(base[: guarded.guard] == sentinel).item():
            return False
        if not torch.all(base[-guarded.guard :] == sentinel).item():
            return False
    return True


def make_service(device: torch.device, *, max_rays: int) -> GaussianLidarService:
    service = GaussianLidarService(
        CudaLidarBackend(max_scenes=1),
        max_sensors=2,
        max_rays=max_rays,
        returns=2,
    )
    service.initialize(stage=None, device=device)
    count = 64
    y = torch.linspace(-2.0, 2.0, count, device=device)
    service.load_scene(
        7,
        means=torch.stack((torch.full_like(y, 6.0), y, torch.zeros_like(y)), dim=1).contiguous(),
        scales=torch.tensor([[0.01, 0.25, 1.0]], device=device).repeat(count, 1).contiguous(),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(count, 1).contiguous(),
        opacities=torch.full((count,), 0.8, device=device),
        semantic_ids=(torch.arange(count, device=device, dtype=torch.int64) % 4).contiguous(),
        reflectivity=torch.linspace(0.2, 0.8, count, device=device),
    )
    service.synchronize()
    return service


def planned_inputs(count: int, rays: int, device: torch.device, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    plans = []
    for index in range(count):
        angles = torch.linspace(-0.22, 0.22, rays) + 0.003 * index
        directions = torch.stack(
            (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)),
            dim=1,
        ).to(device=device, dtype=torch.float32).contiguous()
        transforms = torch.eye(4).repeat(2, 1, 1).contiguous()
        transforms[0, 1, 3] = float(torch.rand((), generator=generator) * 0.2 - 0.1)
        transforms[1, 1, 3] = float(torch.rand((), generator=generator) * 0.2 - 0.1)
        plans.append((directions, transforms.to(device), torch.tensor([7, 7], device=device, dtype=torch.int64)))
    return plans


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    rays = 257
    returns = 2
    invocation_count = 1 if args.single_call else random.SystemRandom().choice([5, 7, 11, 13])
    service = make_service(device, max_rays=rays)
    times = (torch.arange(rays, device=device, dtype=torch.int64) * 100).contiguous()
    plans = planned_inputs(invocation_count, rays, device, args.seed)
    guarded = [
        allocate_guarded_outputs(2, rays, returns, device=device, poison_id=index + 1)
        for index in range(invocation_count)
    ]
    weights = {
        name: (torch.arange(tensor.numel(), device=device, dtype=torch.float64) + 1.0) * (index + 0.125)
        for index, (name, tensor) in enumerate(guarded[0].outputs.items())
    }
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(invocation_count)]
    completions = [torch.cuda.Event(enable_timing=True) for _ in range(invocation_count)]
    immediate_fingerprints: list[torch.Tensor] = []
    counters_before = service.backend.read_counters()
    for index, ((directions, transforms, scene_ids), target) in enumerate(zip(plans, guarded, strict=True)):
        starts[index].record()
        service.render_lidar(
            directions,
            times,
            transforms,
            scene_ids,
            outputs=target.outputs,
        )
        immediate_fingerprints.append(fingerprint(target.outputs, weights))
        completions[index].record()
        directions[:, 1].neg_()
        directions[:, 0].copy_(torch.sqrt(torch.clamp(1.0 - directions[:, 1].square(), min=0.0)))
        transforms[:, 2, 3].add_(0.001 * (index + 1))
    completions[-1].synchronize()
    immediate_cpu = [value.cpu().clone() for value in immediate_fingerprints]
    later_cpu = [fingerprint(target.outputs, weights).cpu() for target in guarded]
    fingerprints_stable = [
        torch.equal(before, after)
        for before, after in zip(immediate_cpu, later_cpu, strict=True)
    ]
    guards_ok = [guards_intact(target) for target in guarded]
    counters_after = service.backend.check_errors(synchronize=False)
    deltas = {name: counters_after[name] - counters_before[name] for name in counters_after}
    if not all(fingerprints_stable):
        raise AssertionError(f"Completed output changed after a later call: {fingerprints_stable}")
    if not all(guards_ok):
        raise AssertionError(f"Output guard changed: {guards_ok}")
    if deltas["calls_started"] != invocation_count or deltas["calls_completed"] != invocation_count:
        raise AssertionError(f"Call counter delta mismatch: {deltas}")
    if deltas["rays_traced"] != invocation_count * 2 * rays:
        raise AssertionError(f"Ray counter delta mismatch: {deltas}")

    synchronized_outputs = []
    for index, (directions, transforms, scene_ids) in enumerate(
        planned_inputs(invocation_count, rays, device, args.seed)
    ):
        target = allocate_guarded_outputs(2, rays, returns, device=device, poison_id=100 + index)
        service.render_lidar(directions, times, transforms, scene_ids, outputs=target.outputs)
        service.synchronize()
        synchronized_outputs.append({name: tensor.clone() for name, tensor in target.outputs.items()})
    ring_matches_synchronized = []
    for ring, synchronized in zip(guarded, synchronized_outputs, strict=True):
        ring_matches_synchronized.append(
            all(torch.equal(ring.outputs[name], synchronized[name]) for name in synchronized)
        )
    if not all(ring_matches_synchronized):
        raise AssertionError("Ring-buffer and synchronize-after-each modes differ.")

    cloned_directions = plans[0][0].clone()
    same_values_a = allocate_guarded_outputs(2, rays, returns, device=device, poison_id=501)
    same_values_b = allocate_guarded_outputs(2, rays, returns, device=device, poison_id=502)
    service.render_lidar(plans[0][0], times, plans[0][1], plans[0][2], outputs=same_values_a.outputs)
    service.render_lidar(
        cloned_directions,
        times.clone(),
        plans[0][1].clone(),
        plans[0][2].clone(),
        outputs=same_values_b.outputs,
    )
    service.synchronize()
    cloned_storage_equal = all(
        torch.equal(same_values_a.outputs[name], same_values_b.outputs[name])
        for name in same_values_a.outputs
    )
    if not cloned_storage_equal:
        raise AssertionError("Cloned storage with identical values changed the result.")

    reused_directions, reused_transforms, reused_scene_ids = planned_inputs(1, rays, device, args.seed + 2)[0]
    reused_before = allocate_guarded_outputs(2, rays, returns, device=device, poison_id=601)
    reused_after = allocate_guarded_outputs(2, rays, returns, device=device, poison_id=602)
    service.render_lidar(
        reused_directions,
        times,
        reused_transforms,
        reused_scene_ids,
        outputs=reused_before.outputs,
    )
    reused_directions[:, 1].add_(0.04)
    reused_directions.copy_(
        reused_directions / torch.linalg.vector_norm(reused_directions, dim=1, keepdim=True)
    )
    service.render_lidar(
        reused_directions,
        times,
        reused_transforms,
        reused_scene_ids,
        outputs=reused_after.outputs,
    )
    service.synchronize()
    same_storage_mutation_observed = not torch.equal(
        reused_before.outputs["range_m"], reused_after.outputs["range_m"]
    )
    if not same_storage_mutation_observed:
        raise AssertionError("Same-storage in-place input mutation did not affect later output.")

    independent_streams_equal = True
    if not args.single_call:
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        services = []
        stream_outputs = []
        for stream in streams:
            with torch.cuda.stream(stream):
                independent = make_service(device, max_rays=rays)
                target = allocate_guarded_outputs(2, rays, returns, device=device, poison_id=700)
                directions, transforms, scene_ids = planned_inputs(1, rays, device, args.seed + 1)[0]
                independent.render_lidar(directions, times, transforms, scene_ids, outputs=target.outputs)
                services.append(independent)
                stream_outputs.append(target)
        for stream in streams:
            stream.synchronize()
        independent_streams_equal = all(
            torch.equal(stream_outputs[0].outputs[name], stream_outputs[1].outputs[name])
            for name in stream_outputs[0].outputs
        )
        for independent in services:
            independent.shutdown()
        if not independent_streams_equal:
            raise AssertionError("Independent service/workspace streams disagree.")

    # Time a second, warmed render-only sequence. There are no input-mutation or
    # fingerprint kernels between these calls, so the enclosing range and the
    # sum of public-invocation event pairs describe the same GPU work.
    timing_plans = planned_inputs(invocation_count, rays, device, args.seed + 101)
    timing_outputs = [
        allocate_guarded_outputs(2, rays, returns, device=device, poison_id=800 + index)
        for index in range(invocation_count)
    ]
    warm_directions, warm_transforms, warm_scene_ids = timing_plans[0]
    service.render_lidar(
        warm_directions,
        times,
        warm_transforms,
        warm_scene_ids,
        outputs=timing_outputs[0].outputs,
    )
    service.synchronize()
    timing_starts = [torch.cuda.Event(enable_timing=True) for _ in range(invocation_count)]
    timing_ends = [torch.cuda.Event(enable_timing=True) for _ in range(invocation_count)]
    range_start = torch.cuda.Event(enable_timing=True)
    range_end = torch.cuda.Event(enable_timing=True)
    range_start.record()
    for index, ((directions, transforms, scene_ids), target) in enumerate(
        zip(timing_plans, timing_outputs, strict=True)
    ):
        timing_starts[index].record()
        service.render_lidar(directions, times, transforms, scene_ids, outputs=target.outputs)
        timing_ends[index].record()
    range_end.record()
    range_end.synchronize()
    per_call_ms = [
        start.elapsed_time(end)
        for start, end in zip(timing_starts, timing_ends, strict=True)
    ]
    range_ms = range_start.elapsed_time(range_end)
    timing_ratio = sum(per_call_ms) / range_ms
    timing_gap_ms = abs(sum(per_call_ms) - range_ms)
    timing_tolerance_ms = max(0.10, 0.10 * range_ms)
    if timing_gap_ms > timing_tolerance_ms:
        raise AssertionError(
            f"N-call range time {range_ms} ms is inconsistent with per-call sum {sum(per_call_ms)} ms"
        )
    result = {
        "schema_version": "gaussian-lidar-deferred-work-audit/v1",
        "pass": True,
        "fresh_process_single_call": args.single_call,
        "invocation_count": invocation_count,
        "invocation_count_is_prime": invocation_count in (5, 7, 11, 13),
        "distinct_padded_outputs": invocation_count,
        "fingerprints_stable_after_later_calls": fingerprints_stable,
        "guards_intact": guards_ok,
        "ring_matches_synchronize_each": ring_matches_synchronized,
        "cloned_storage_identical_values_equal": cloned_storage_equal,
        "same_storage_mutation_observed": same_storage_mutation_observed,
        "independent_streams_equal": independent_streams_equal,
        "counter_deltas": deltas,
        "per_call_gpu_ms": per_call_ms,
        "range_gpu_ms": range_ms,
        "sum_to_range_ratio": timing_ratio,
        "sum_to_range_gap_ms": timing_gap_ms,
        "sum_to_range_tolerance_ms": timing_tolerance_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("GAUSSIAN_LIDAR_DEFERRED_WORK_AUDIT_OK")
    service.shutdown()


if __name__ == "__main__":
    main()
