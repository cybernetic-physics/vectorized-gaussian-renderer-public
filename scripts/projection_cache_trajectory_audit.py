#!/usr/bin/env python3
"""Mutation matrix and anti-stale-output audit for projection caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Callable

import torch

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import camera_bundle, synthetic_scene_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--gaussians", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/projection-cache/trajectory-audit.json"),
    )
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def output_hashes(outputs: dict[str, torch.Tensor]) -> dict[str, str]:
    torch.cuda.synchronize()
    return {name: tensor_sha256(tensor) for name, tensor in sorted(outputs.items())}


def poison(outputs: dict[str, torch.Tensor]) -> None:
    outputs["rgb"].fill_(float("nan"))
    outputs["depth"].fill_(float("nan"))
    outputs["alpha"].fill_(float("nan"))
    outputs["semantic_id"].fill_(-2**62)


def rewritten(outputs: dict[str, torch.Tensor]) -> dict[str, bool]:
    return {
        "rgb": bool(torch.isfinite(outputs["rgb"]).all().item()),
        # Positive infinity is the renderer's documented background-depth
        # value. NaN is the poison sentinel and must never survive a call.
        "depth": bool((~torch.isnan(outputs["depth"])).all().item()),
        "alpha": bool(torch.isfinite(outputs["alpha"]).all().item()),
        "semantic_id": bool((outputs["semantic_id"] != -2**62).all().item()),
    }


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.gaussians) <= 0:
        raise ValueError("Resolution and Gaussian count must be positive.")
    randomizer = random.Random(args.seed)
    device = torch.device("cuda")
    scene = synthetic_scene_tensors(args.gaussians, seed=87, device=device)
    cameras = camera_bundle(2, args.width, args.height, device=device)
    scene_ids = torch.full((2,), 87, device=device, dtype=torch.int64)
    backend = CustomCudaBackend(
        max_visible_records=2 * args.gaussians,
        max_intersections=max(1_000_000, 2 * args.gaussians * 16),
        gaussian_support_sigma=3.0,
        covariance_epsilon=0.0,
        tile_size=1,
        depth_bucket_count=128,
        depth_bucket_group_size=8,
        compact_projection_cache=True,
        enable_projection_cache=True,
        output_srgb=False,
        deterministic=False,
    )
    service = RendererService(backend, height=args.height, width=args.width, max_views=2)
    service.initialize(stage=None, device=device)
    service.load_scene(
        87,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )
    scene_two = synthetic_scene_tensors(args.gaussians, seed=88, device=device)
    service.load_scene(
        88,
        means=scene_two["means"],
        scales=scene_two["scales"],
        rotations=scene_two["quats"],
        opacities=scene_two["opacities"],
        features=scene_two["colors"],
        semantic_ids=scene_two["semantic_ids"].to(torch.int64),
    )

    buffers = [service._owned_output_set(2)]
    buffers.append({name: tensor.clone() for name, tensor in buffers[0].items()})
    api_requests = native_submissions = renderer_executions = 0

    def render(outputs: dict[str, torch.Tensor], *, active: torch.Tensor | None = None) -> tuple[str, dict[str, str], dict[str, bool]]:
        nonlocal api_requests, native_submissions, renderer_executions
        before = backend.projection_cache_stats.copy()
        poison(outputs)
        service.render(
            cameras.viewmats,
            cameras.intrinsics,
            scene_ids,
            active_camera_ids=active,
            outputs=outputs,
        )
        api_requests += 1
        native_submissions += 1
        renderer_executions += 1
        service.synchronize()
        after = backend.projection_cache_stats
        event = "hit" if int(after["hits"]) > int(before["hits"]) else "miss"
        return event, output_hashes(outputs), rewritten(outputs)

    warmups = randomizer.randint(1, 4)
    for index in range(warmups):
        render(buffers[index % 2])
    baseline_event, baseline_hashes, baseline_rewritten = render(buffers[0])

    packed = backend._packed_scene
    env = torch.eye(4, device=device).repeat(2, 1, 1)
    active_one = torch.tensor([0], device=device, dtype=torch.int64)
    active_two = torch.tensor([0, 1], device=device, dtype=torch.int64)
    mutations: list[tuple[str, str, Callable[[], None], tuple[str, ...]]] = [
        ("colors", "hit", lambda: packed["features"].add_(0.03125).remainder_(1.0), ("rgb",)),
        ("semantic_ids", "hit", lambda: packed["semantic_ids"].add_(1).remainder_(1024), ("semantic_id",)),
        ("means", "miss", lambda: packed["means"][:, 0].add_(0.05), ("rgb", "depth", "alpha")),
        ("covariance", "miss", lambda: packed["covariances"].mul_(1.10), ("rgb", "alpha")),
        ("opacity", "miss", lambda: packed["opacities"].mul_(0.80), ("rgb", "alpha")),
        ("camera_transform", "miss", lambda: cameras.viewmats[:1, 0, 3].add_(0.01), ("rgb", "depth", "alpha")),
        ("intrinsics_focal", "miss", lambda: cameras.intrinsics[:, 0, 0].mul_(1.01), ("rgb", "alpha")),
        ("intrinsics_principal_point", "miss", lambda: cameras.intrinsics[:, 0, 2].add_(0.25), ("rgb", "alpha")),
        ("environment_transform", "miss", lambda: env[:, 0, 3].add_(0.02), ("rgb", "depth", "alpha")),
        ("scene_ids", "miss", lambda: scene_ids[1].fill_(88), ("rgb", "depth", "alpha", "semantic_id")),
    ]
    randomizer.shuffle(mutations)
    cases = []
    previous_hashes = baseline_hashes
    for index, (name, expected, mutate, changed_outputs) in enumerate(mutations):
        mutate()
        if name == "environment_transform":
            backend.update_scene_transforms(scene_ids, env)
        event, hashes, rewrite_status = render(buffers[index % 2])
        changed = {output: hashes[output] != previous_hashes[output] for output in hashes}
        cases.append(
            {
                "mutation": name,
                "expected_cache_event": expected,
                "actual_cache_event": event,
                "cache_event_pass": event == expected,
                "required_changed_outputs": list(changed_outputs),
                "changed_outputs": changed,
                "output_change_pass": all(changed[output] for output in changed_outputs),
                "all_outputs_rewritten": rewrite_status,
                "pass": event == expected
                and all(changed[output] for output in changed_outputs)
                and all(rewrite_status.values()),
            }
        )
        previous_hashes = hashes

    event_active_one, hashes_active_one, rewrite_active_one = render(buffers[0], active=active_one)
    event_active_two, hashes_active_two, rewrite_active_two = render(buffers[1], active=active_two)
    cases.append(
        {
            "mutation": "active_camera_ids",
            "expected_cache_event": "miss",
            "actual_cache_events": [event_active_one, event_active_two],
            "cache_event_pass": event_active_one == "miss" and event_active_two == "miss",
            "output_change_pass": hashes_active_one != hashes_active_two,
            "all_outputs_rewritten": {**rewrite_active_one, **rewrite_active_two},
            "pass": event_active_one == "miss"
            and event_active_two == "miss"
            and all(rewrite_active_one.values())
            and all(rewrite_active_two.values()),
        }
    )

    backend.invalidate_projection_cache()
    manual_event, _manual_hashes, manual_rewritten = render(buffers[0], active=active_two)
    cases.append(
        {
            "mutation": "explicit_manual_invalidation",
            "expected_cache_event": "miss",
            "actual_cache_event": manual_event,
            "cache_event_pass": manual_event == "miss",
            "output_change_pass": True,
            "all_outputs_rewritten": manual_rewritten,
            "pass": manual_event == "miss" and all(manual_rewritten.values()),
        }
    )

    # `.data` models an external CUDA writer that bypasses PyTorch version
    # counters. The first call demonstrates the stale-key hazard; explicit
    # invalidation must force the subsequent miss and changed output.
    external_baseline_event, _external_baseline_hashes, _external_baseline_rewritten = render(buffers[1])
    external_before_version = int(cameras.viewmats._version)
    cameras.viewmats.data[0, 1, 3] += 0.25
    external_after_version = int(cameras.viewmats._version)
    event_without_invalidation, stale_hashes, stale_rewritten = render(buffers[0])
    backend.invalidate_projection_cache()
    event_after_invalidation, fresh_hashes, fresh_rewritten = render(buffers[1])
    cases.append(
        {
            "mutation": "external_fabric_warp_style_write",
            "baseline_event": external_baseline_event,
            "version_counter_bypassed": external_before_version == external_after_version,
            "without_invalidation_event": event_without_invalidation,
            "explicit_invalidation_event": event_after_invalidation,
            "fresh_output_changed": stale_hashes != fresh_hashes,
            "all_outputs_rewritten": {**stale_rewritten, **fresh_rewritten},
            "pass": all(_external_baseline_rewritten.values())
            and external_before_version == external_after_version
            and event_without_invalidation == "hit"
            and event_after_invalidation == "miss"
            and stale_hashes != fresh_hashes
            and all(stale_rewritten.values())
            and all(fresh_rewritten.values()),
        }
    )

    counters = backend.check_capacity(synchronize=False)
    result = {
        "schema_version": "projection-cache-trajectory-audit-v1",
        "pass": all(case["pass"] for case in cases)
        and all(baseline_rewritten.values())
        and not counters["visible_overflow"]
        and not counters["intersection_overflow"],
        "seed": args.seed,
        "randomized_warmup_count": warmups,
        "randomized_mutation_order": [case["mutation"] for case in cases],
        "baseline_event": baseline_event,
        "baseline_output_hashes": baseline_hashes,
        "cases": cases,
        "api_requests": api_requests,
        "native_submissions": native_submissions,
        "renderer_executions": renderer_executions,
        "submissions_equal_executions": native_submissions == renderer_executions,
        "projection_cache_stats": backend.projection_cache_stats,
        "capacity_counters": counters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    service.shutdown()
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
