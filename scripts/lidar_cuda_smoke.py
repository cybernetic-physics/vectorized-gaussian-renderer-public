"""Finite CUDA correctness smoke for the opt-in Gaussian LiDAR backend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.lidar_cpu_oracle import render_lidar_oracle
from isaacsim_gaussian_renderer import GaussianLidarService
from isaacsim_gaussian_renderer.cuda_lidar_backend import CudaLidarBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lidar/lidar-cuda-smoke.json"),
    )
    return parser.parse_args()


def scene_tensors(device: torch.device, *, second_depth: float = 8.0) -> dict[str, torch.Tensor]:
    half = math.sqrt(0.5)
    return {
        "means": torch.tensor(
            [[5.0, 0.0, 0.0], [5.01, 0.0, 0.0], [second_depth, 0.0, 0.0], [0.0, 5.0, 0.0]],
            device=device,
            dtype=torch.float32,
        ),
        "scales": torch.tensor([[0.01, 1.0, 1.0]] * 4, device=device, dtype=torch.float32),
        "rotations": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [half, 0.0, 0.0, half],
            ],
            device=device,
            dtype=torch.float32,
        ),
        "opacities": torch.ones((4,), device=device, dtype=torch.float32),
        "semantic_ids": torch.tensor([9, 3, 11, 21], device=device, dtype=torch.int64),
        "reflectivity": torch.tensor([0.2, 0.6, 0.8, 0.7], device=device, dtype=torch.float32),
        "surface_confidence": torch.ones((4,), device=device, dtype=torch.float32),
    }


def boundary_scene_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "means": torch.tensor([[0.05, 0.0, 0.0], [200.0, 0.0, 0.0]], device=device),
        "scales": torch.tensor([[0.001, 1.0, 1.0], [0.001, 1.0, 1.0]], device=device),
        "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device),
        "opacities": torch.ones((2,), device=device),
        "semantic_ids": torch.tensor([50, 200], device=device, dtype=torch.int64),
        "reflectivity": torch.ones((2,), device=device),
    }


def numpy_scene(scene: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy().astype(np.float64, copy=True) for name, tensor in scene.items()}


def assert_matches_oracle(
    outputs: dict[str, torch.Tensor],
    oracle: dict[str, np.ndarray],
) -> dict[str, float]:
    gpu = {name: tensor.detach().cpu().numpy() for name, tensor in outputs.items()}
    np.testing.assert_array_equal(gpu["valid"], oracle["valid"])
    np.testing.assert_array_equal(gpu["semantic_id"], oracle["semantic_id"])
    np.testing.assert_array_equal(gpu["return_count"], oracle["return_count"])
    np.testing.assert_array_equal(gpu["time_offset_ns"], oracle["time_offset_ns"])
    mask = oracle["valid"]
    range_error = float(np.max(np.abs(gpu["range_m"][mask] - oracle["range_m"][mask])))
    intensity_error = float(np.max(np.abs(gpu["intensity"][mask] - oracle["intensity"][mask])))
    position_mask = np.repeat(mask[..., None], 3, axis=-1)
    position_error = float(
        np.max(np.abs(gpu["position_world_m"][position_mask] - oracle["position_world_m"][position_mask]))
    )
    if range_error > 2.0e-5 or intensity_error > 2.0e-5 or position_error > 3.0e-5:
        raise AssertionError(
            f"GPU/oracle error range={range_error} intensity={intensity_error} position={position_error}"
        )
    return {
        "max_range_abs_m": range_error,
        "max_intensity_abs": intensity_error,
        "max_position_abs_m": position_error,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    backend = CudaLidarBackend(max_scenes=4, packet_size=8)
    service = GaussianLidarService(backend, max_sensors=4, max_rays=8, returns=2)
    service.initialize(stage=None, device=device)
    scene_a = scene_tensors(device)
    scene_b = scene_tensors(device, second_depth=9.0)
    boundary_scene = boundary_scene_tensors(device)
    service.load_scene(17, **scene_a)
    service.load_scene(901, **scene_b)
    service.load_scene(404, **boundary_scene)
    service.synchronize()
    build_bvh_bytes = backend.bvh_bytes

    rays = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    times = torch.tensor([0, 125, 250], device=device, dtype=torch.int32)
    sensor_to_world = torch.eye(4, device=device).repeat(3, 1, 1).contiguous()
    scene_to_world = torch.eye(4, device=device).repeat(3, 1, 1).contiguous()
    scene_to_world[2, 0, 3] = 2.0
    scene_ids = torch.tensor([17, 901, 17], device=device, dtype=torch.int64)

    counters_before = backend.read_counters()
    outputs = service.render_lidar(
        rays,
        times,
        sensor_to_world,
        scene_ids,
        scene_to_world=scene_to_world,
    )
    service.synchronize()
    backend.check_errors(synchronize=False)
    oracle = render_lidar_oracle(
        {17: numpy_scene(scene_a), 901: numpy_scene(scene_b)},
        rays.cpu().numpy(),
        times.cpu().numpy(),
        sensor_to_world.cpu().numpy(),
        scene_ids.cpu().numpy(),
        scene_to_world=scene_to_world.cpu().numpy(),
        returns=2,
    )
    oracle_errors = assert_matches_oracle(outputs, oracle)
    baseline = {name: tensor.clone() for name, tensor in outputs.items()}
    first_hashes = {
        name: torch.sum(tensor.view(torch.uint8).to(torch.int64)).item()
        for name, tensor in outputs.items()
    }
    replay = service.render_lidar(
        rays,
        times,
        sensor_to_world,
        scene_ids,
        scene_to_world=scene_to_world,
    )
    service.synchronize()
    replay_bitwise = {name: torch.equal(baseline[name], replay[name]) for name in baseline}
    if not all(replay_bitwise.values()):
        raise AssertionError(f"Fixed-state replay mismatch: {replay_bitwise}")

    single_return = service.render_lidar(
        rays,
        times,
        sensor_to_world,
        scene_ids,
        scene_to_world=scene_to_world,
        returns=1,
    )
    service.synchronize()
    for name in ("range_m", "intensity", "semantic_id", "valid", "time_offset_ns"):
        if not torch.equal(single_return[name][..., 0], baseline[name][..., 0]):
            raise AssertionError(f"K=2 changed the K=1 equation for {name}.")
    if not torch.equal(
        single_return["position_world_m"][..., 0, :],
        baseline["position_world_m"][..., 0, :],
    ):
        raise AssertionError("K=2 changed the K=1 equation for position_world_m.")

    boundary_outputs = service.render_lidar(
        torch.tensor([[1.0, 0.0, 0.0]], device=device),
        torch.zeros((1,), device=device, dtype=torch.int64),
        torch.eye(4, device=device).unsqueeze(0),
        torch.tensor([404], device=device, dtype=torch.int64),
        returns=2,
    )
    service.synchronize()
    torch.testing.assert_close(
        boundary_outputs["range_m"][0, 0],
        torch.tensor([0.05, 200.0], device=device),
        rtol=0.0,
        atol=1.0e-6,
    )
    if not torch.equal(
        boundary_outputs["semantic_id"][0, 0],
        torch.tensor([50, 200], device=device, dtype=torch.int64),
    ):
        raise AssertionError("Inclusive near/far boundary semantics are incorrect.")

    rigid = torch.eye(4, device=device).unsqueeze(0)
    rigid[0, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    rigid[0, :3, 3] = torch.tensor([1.0, -2.0, 0.5], device=device)
    rotated_outputs = service.render_lidar(
        rays[:1],
        times[:1],
        rigid,
        torch.tensor([17], device=device, dtype=torch.int64),
        scene_to_world=rigid,
        returns=2,
    )
    service.synchronize()
    torch.testing.assert_close(
        rotated_outputs["range_m"][0, 0],
        baseline["range_m"][0, 0],
        rtol=0.0,
        atol=2.0e-6,
    )
    torch.testing.assert_close(
        rotated_outputs["position_world_m"][0, 0],
        baseline["position_world_m"][0, 0] @ rigid[0, :3, :3].T + rigid[0, :3, 3],
        rtol=0.0,
        atol=2.0e-6,
    )

    active = torch.tensor([0, 2], device=device, dtype=torch.int64)
    active_outputs = service.render_lidar(
        rays,
        times,
        sensor_to_world,
        scene_ids,
        scene_to_world=scene_to_world,
        active_sensor_ids=active,
    )
    service.synchronize()
    if active_outputs["valid"][1].any().item():
        raise AssertionError("Inactive sensor outputs were not reset to no-hit values.")

    before_empty = backend.read_counters()
    empty_active_outputs = service.render_lidar(
        rays,
        times,
        sensor_to_world,
        scene_ids,
        scene_to_world=scene_to_world,
        active_sensor_ids=torch.empty((0,), device=device, dtype=torch.int64),
    )
    service.synchronize()
    after_empty = backend.check_errors(synchronize=False)
    if empty_active_outputs["valid"].any().item():
        raise AssertionError("An explicit empty active subset produced LiDAR returns.")
    if after_empty["rays_traced"] != before_empty["rays_traced"]:
        raise AssertionError("An explicit empty active subset traced rays.")

    previous_intensity = baseline["intensity"]
    scene_a["reflectivity"].mul_(0.5)
    mutated = service.render_lidar(
        rays,
        times,
        sensor_to_world,
        scene_ids,
        scene_to_world=scene_to_world,
    )
    service.synchronize()
    if torch.equal(previous_intensity[0], mutated["intensity"][0]):
        raise AssertionError("Same-storage reflectivity mutation did not affect LiDAR intensity.")

    scene_a["means"][:, 0].add_(1.0)
    service.revise_scene(17, revision=1)
    service.synchronize()
    revised = service.render_lidar(
        rays,
        times,
        sensor_to_world,
        scene_ids,
        scene_to_world=scene_to_world,
    )
    service.synchronize()
    if not torch.allclose(revised["range_m"][0, 0, 0], baseline["range_m"][0, 0, 0] + 1.0, atol=2e-5):
        raise AssertionError("Explicit scene revision did not rebuild geometry.")

    final_counters = backend.check_errors(synchronize=False)
    deltas = {name: final_counters[name] - counters_before[name] for name in final_counters}
    if deltas["calls_started"] != deltas["calls_completed"] or deltas["calls_started"] != 9:
        raise AssertionError(f"Unexpected per-call counters: {deltas}")
    if backend.bvh_bytes != build_bvh_bytes:
        raise AssertionError("Scene BVH memory changed with sensor batch or same-count revision.")
    result = {
        "schema_version": "gaussian-lidar-cuda-smoke/v1",
        "pass": True,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "oracle_errors": oracle_errors,
        "replay_bitwise_equal": replay_bitwise,
        "inclusive_boundary_ranges_m": boundary_outputs["range_m"][0, 0].cpu().tolist(),
        "rigid_transform_equivariance": True,
        "initial_output_byte_sums": first_hashes,
        "counter_deltas": deltas,
        "bvh_bytes": backend.bvh_bytes,
        "scene_attribute_bytes": backend.scene_attribute_bytes,
        "workspace_bytes": backend.workspace_bytes,
        "registered_static_scenes": 3,
        "scene_copies_per_sensor": 0,
        "batch": 3,
        "rays": 3,
        "returns": 2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("GAUSSIAN_LIDAR_CUDA_SMOKE_OK")
    service.shutdown()


if __name__ == "__main__":
    main()
