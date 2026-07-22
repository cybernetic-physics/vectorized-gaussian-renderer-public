"""Randomized GPU-vs-independent-float64-oracle and metamorphic audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.lidar_cpu_oracle import render_lidar_oracle
from isaacsim_gaussian_renderer import GaussianLidarService
from isaacsim_gaussian_renderer.cuda_lidar_backend import CudaLidarBackend


SEEDS = (731, 2027, 99173, 104729)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lidar/randomized-oracle.json"),
    )
    return parser.parse_args()


def make_scene(seed: int, count: int = 96) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    depths = generator.uniform(2.0, 18.0, count)
    lateral = generator.normal(0.0, 1.2, (count, 2))
    angles = generator.uniform(-0.35, 0.35, count)
    half = 0.5 * angles
    return {
        "means": np.column_stack((depths, lateral)).astype(np.float32),
        "scales": np.column_stack(
            (
                generator.uniform(0.005, 0.025, count),
                generator.uniform(0.20, 0.75, count),
                generator.uniform(0.20, 0.75, count),
            )
        ).astype(np.float32),
        "rotations": np.column_stack(
            (np.cos(half), np.zeros(count), np.zeros(count), np.sin(half))
        ).astype(np.float32),
        "opacities": generator.uniform(0.35, 1.0, count).astype(np.float32),
        "semantic_ids": generator.integers(0, 8, count, dtype=np.int64),
        "reflectivity": generator.uniform(0.05, 0.95, count).astype(np.float32),
        "surface_confidence": generator.uniform(0.7, 1.0, count).astype(np.float32),
    }


def make_rays(seed: int, count: int = 41) -> np.ndarray:
    generator = np.random.default_rng(seed ^ 0xA5A5)
    directions = generator.normal(0.0, (0.25, 0.20, 0.15), (count, 3))
    directions[:, 0] = np.abs(directions[:, 0]) + 1.0
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions.astype(np.float32)


def cuda_scene(scene: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(values, device=device).contiguous()
        for name, values in scene.items()
    }


def oracle_scene(scene: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: values.astype(np.float64, copy=True) for name, values in scene.items()}


def compare(
    gpu: dict[str, torch.Tensor],
    cpu: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    actual = {name: tensor.cpu().numpy() for name, tensor in gpu.items()}
    np.testing.assert_array_equal(actual["valid"], cpu["valid"])
    np.testing.assert_array_equal(actual["semantic_id"], cpu["semantic_id"])
    np.testing.assert_array_equal(actual["return_count"], cpu["return_count"])
    np.testing.assert_array_equal(actual["time_offset_ns"], cpu["time_offset_ns"])
    valid = cpu["valid"]
    if not valid.any():
        return 0.0, 0.0, 0.0
    range_error = float(np.max(np.abs(actual["range_m"][valid] - cpu["range_m"][valid])))
    intensity_error = float(np.max(np.abs(actual["intensity"][valid] - cpu["intensity"][valid])))
    position_valid = np.repeat(valid[..., None], 3, axis=-1)
    position_error = float(
        np.max(
            np.abs(
                actual["position_world_m"][position_valid]
                - cpu["position_world_m"][position_valid]
            )
        )
    )
    if range_error > 4.0e-5 or intensity_error > 4.0e-5 or position_error > 6.0e-5:
        raise AssertionError(
            f"oracle mismatch range={range_error} intensity={intensity_error} position={position_error}"
        )
    return range_error, intensity_error, position_error


def run_service(
    scenes: dict[int, dict[str, np.ndarray]],
    rays: np.ndarray,
    transforms: np.ndarray,
    scene_ids: np.ndarray,
    scene_to_world: np.ndarray,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    service = GaussianLidarService(
        CudaLidarBackend(max_scenes=len(scenes), packet_size=8),
        max_sensors=len(scene_ids),
        max_rays=len(rays),
        returns=2,
    )
    service.initialize(None, device)
    for scene_id, scene in scenes.items():
        service.load_scene(scene_id, **cuda_scene(scene, device))
    service.synchronize()
    output = service.render_lidar(
        torch.as_tensor(rays, device=device),
        torch.arange(len(rays), device=device, dtype=torch.int64) * 137,
        torch.as_tensor(transforms, device=device),
        torch.as_tensor(scene_ids, device=device),
        scene_to_world=torch.as_tensor(scene_to_world, device=device),
    )
    service.synchronize()
    copied = {name: tensor.cpu() for name, tensor in output.items()}
    counters = service.backend.check_errors(synchronize=False)
    service.shutdown()
    return copied, counters


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    records = []
    maxima = {"range_m": 0.0, "intensity": 0.0, "position_m": 0.0}
    for seed in SEEDS:
        scenes = {17: make_scene(seed), 901: make_scene(seed + 1)}
        rays = make_rays(seed)
        batch = 4
        sensor_to_world = np.repeat(np.eye(4, dtype=np.float32)[None], batch, axis=0)
        sensor_to_world[:, 1, 3] = np.array([-0.3, 0.1, 0.4, -0.2], dtype=np.float32)
        scene_to_world = np.repeat(np.eye(4, dtype=np.float32)[None], batch, axis=0)
        scene_to_world[:, :3, 3] = np.array(
            [[0.0, 0.0, 0.0], [1.0, -0.5, 0.2], [-0.5, 0.3, 0.0], [0.2, 0.1, -0.1]],
            dtype=np.float32,
        )
        scene_ids = np.array([17, 901, 17, 901], dtype=np.int64)
        gpu, counters = run_service(
            scenes, rays, sensor_to_world, scene_ids, scene_to_world, device=device
        )
        cpu = render_lidar_oracle(
            {key: oracle_scene(value) for key, value in scenes.items()},
            rays.astype(np.float64),
            np.arange(len(rays), dtype=np.int64) * 137,
            sensor_to_world.astype(np.float64),
            scene_ids,
            scene_to_world=scene_to_world.astype(np.float64),
            returns=2,
        )
        errors = compare(gpu, cpu)
        maxima["range_m"] = max(maxima["range_m"], errors[0])
        maxima["intensity"] = max(maxima["intensity"], errors[1])
        maxima["position_m"] = max(maxima["position_m"], errors[2])

        permutation = np.random.default_rng(seed ^ 0x55AA).permutation(len(scenes[17]["means"]))
        permuted = {
            scene_id: {name: values[permutation].copy() for name, values in scene.items()}
            for scene_id, scene in scenes.items()
        }
        permuted_gpu, permuted_counters = run_service(
            permuted, rays, sensor_to_world, scene_ids, scene_to_world, device=device
        )
        for name in ("valid", "semantic_id", "return_count"):
            if not torch.equal(gpu[name], permuted_gpu[name]):
                raise AssertionError(f"seed {seed}: permutation changed {name}")
        for name in ("range_m", "position_world_m", "intensity"):
            torch.testing.assert_close(gpu[name], permuted_gpu[name], rtol=0.0, atol=2.0e-6)

        for sensor in range(batch):
            individual, _ = run_service(
                scenes,
                rays,
                sensor_to_world[sensor : sensor + 1],
                scene_ids[sensor : sensor + 1],
                scene_to_world[sensor : sensor + 1],
                device=device,
            )
            for name in gpu:
                if not torch.equal(gpu[name][sensor], individual[name][0]):
                    raise AssertionError(f"seed {seed}: batch/individual mismatch in {name}")

        for checked in (counters, permuted_counters):
            if checked["stack_overflow"] or checked["semantic_overflow"]:
                raise AssertionError(f"seed {seed}: overflow counters {checked}")
        records.append(
            {
                "seed": seed,
                "gaussians_per_scene": len(scenes[17]["means"]),
                "batch": batch,
                "rays": len(rays),
                "valid_returns": int(gpu["valid"].sum().item()),
                "max_range_abs_m": errors[0],
                "max_intensity_abs": errors[1],
                "max_position_abs_m": errors[2],
                "permutation_invariant_atol": 2.0e-6,
                "batch_bitwise_equals_individual": True,
            }
        )
    result = {
        "schema_version": "gaussian-lidar-randomized-oracle/v1",
        "pass": True,
        "held_out_seeds": list(SEEDS),
        "maxima": maxima,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("GAUSSIAN_LIDAR_RANDOMIZED_ORACLE_OK")


if __name__ == "__main__":
    main()
