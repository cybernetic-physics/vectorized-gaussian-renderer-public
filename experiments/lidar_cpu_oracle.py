"""Independent float64 CPU oracle for the Gaussian LiDAR surface contract.

This deliberately uses NumPy and a collect-sort-reduce formulation rather
than mirroring the production LBVH traversal or CUDA helper structure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OracleConfig:
    near_plane_m: float = 0.05
    far_plane_m: float = 200.0
    support_sigma: float = 3.0
    detection_threshold: float = 0.01
    planarity_ratio_max: float = 0.35
    min_incidence_cos: float = 0.05
    cluster_abs_m: float = 0.02
    cluster_relative: float = 0.001
    fallback_reflectivity: float = 0.5


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(float(w * w + x * x + y * y + z * z))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
        raise ValueError("Oracle rotations must be normalized WXYZ quaternions.")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _candidate(
    scene: dict[str, np.ndarray],
    gaussian: int,
    origin: np.ndarray,
    direction: np.ndarray,
    config: OracleConfig,
) -> tuple[float, float, float, int] | None:
    scales = np.asarray(scene["scales"][gaussian], dtype=np.float64)
    normal_axis = int(np.argmin(scales))
    tangent_axes = [axis for axis in range(3) if axis != normal_axis]
    if scales[normal_axis] <= 0 or scales[normal_axis] / min(scales[tangent_axes]) > config.planarity_ratio_max:
        return None
    rotation = _rotation_matrix_wxyz(scene["rotations"][gaussian])
    normal = rotation[:, normal_axis]
    denominator = float(np.dot(normal, direction))
    incidence = abs(denominator)
    if incidence < config.min_incidence_cos:
        return None
    mean = np.asarray(scene["means"][gaussian], dtype=np.float64)
    distance = float(np.dot(normal, mean - origin) / denominator)
    if distance < config.near_plane_m or distance > config.far_plane_m:
        return None
    offset = origin + distance * direction - mean
    tangent_coordinates = [
        float(np.dot(offset, rotation[:, axis]) / scales[axis])
        for axis in tangent_axes
    ]
    q = math.fsum(coordinate * coordinate for coordinate in tangent_coordinates)
    if q > config.support_sigma * config.support_sigma:
        return None
    support = math.exp(-0.5 * q)
    confidence_values = scene.get("surface_confidence")
    confidence = 1.0 if confidence_values is None else float(confidence_values[gaussian])
    weight = float(scene["opacities"][gaussian]) * support * confidence
    if not math.isfinite(weight) or weight < config.detection_threshold:
        return None
    reflectivity_values = scene.get("reflectivity")
    reflectivity = (
        config.fallback_reflectivity
        if reflectivity_values is None
        else float(reflectivity_values[gaussian])
    )
    intensity = min(max(reflectivity * support * incidence, 0.0), 1.0)
    return distance, weight, intensity, int(scene["semantic_ids"][gaussian])


def render_lidar_oracle(
    scenes: dict[int, dict[str, np.ndarray]],
    ray_directions: np.ndarray,
    time_offsets_ns: np.ndarray,
    sensor_to_world: np.ndarray,
    scene_ids: np.ndarray,
    *,
    scene_to_world: np.ndarray | None = None,
    active_sensor_ids: np.ndarray | None = None,
    returns: int = 1,
    config: OracleConfig = OracleConfig(),
) -> dict[str, np.ndarray]:
    """Collect all valid candidates, depth-cluster them, and reduce in float64."""

    if returns not in (1, 2):
        raise ValueError("returns must be 1 or 2.")
    directions = np.asarray(ray_directions, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-4):
        raise ValueError("ray_directions must be normalized.")
    sensor_to_world = np.asarray(sensor_to_world, dtype=np.float64)
    batch = sensor_to_world.shape[0]
    rays = directions.shape[0]
    if scene_to_world is None:
        scene_to_world = np.repeat(np.eye(4, dtype=np.float64)[None], batch, axis=0)
    else:
        scene_to_world = np.asarray(scene_to_world, dtype=np.float64)
    active = range(batch) if active_sensor_ids is None else [int(value) for value in active_sensor_ids]
    outputs = {
        "range_m": np.full((batch, rays, returns), np.inf, dtype=np.float64),
        "position_world_m": np.zeros((batch, rays, returns, 3), dtype=np.float64),
        "intensity": np.zeros((batch, rays, returns), dtype=np.float64),
        "semantic_id": np.full((batch, rays, returns), -1, dtype=np.int64),
        "valid": np.zeros((batch, rays, returns), dtype=np.bool_),
        "time_offset_ns": np.broadcast_to(
            np.asarray(time_offsets_ns, dtype=np.int64)[None, :, None],
            (batch, rays, returns),
        ).copy(),
        "return_count": np.zeros((batch, rays), dtype=np.int32),
    }
    for sensor in active:
        scene_id = int(scene_ids[sensor])
        if scene_id not in scenes:
            raise ValueError(f"Unregistered oracle scene ID: {scene_id}.")
        scene = scenes[scene_id]
        sensor_transform = sensor_to_world[sensor]
        scene_transform = scene_to_world[sensor]
        world_origin = sensor_transform[:3, 3]
        local_origin = scene_transform[:3, :3].T @ (world_origin - scene_transform[:3, 3])
        for ray in range(rays):
            world_direction = sensor_transform[:3, :3] @ directions[ray]
            local_direction = scene_transform[:3, :3].T @ world_direction
            candidates = [
                candidate
                for gaussian in range(len(scene["means"]))
                if (
                    candidate := _candidate(
                        scene,
                        gaussian,
                        local_origin,
                        local_direction,
                        config,
                    )
                )
                is not None
            ]
            candidates.sort(key=lambda value: (value[0], value[3]))
            previous_end = config.near_plane_m - 1.0e-12
            for return_index in range(returns):
                remaining = [candidate for candidate in candidates if candidate[0] > previous_end]
                if not remaining:
                    break
                first_t = remaining[0][0]
                cluster_end = min(
                    config.far_plane_m,
                    first_t + max(config.cluster_abs_m, config.cluster_relative * first_t),
                )
                cluster = [candidate for candidate in remaining if candidate[0] <= cluster_end]
                total_weight = math.fsum(candidate[1] for candidate in cluster)
                range_m = math.fsum(candidate[0] * candidate[1] for candidate in cluster) / total_weight
                intensity = math.fsum(candidate[2] * candidate[1] for candidate in cluster) / total_weight
                semantic_weights: dict[int, list[float]] = {}
                for candidate in cluster:
                    semantic_weights.setdefault(candidate[3], []).append(candidate[1])
                aggregates = {
                    semantic: math.fsum(weights)
                    for semantic, weights in semantic_weights.items()
                }
                winning_semantic = min(
                    aggregates,
                    key=lambda semantic: (-aggregates[semantic], semantic),
                )
                outputs["range_m"][sensor, ray, return_index] = range_m
                outputs["position_world_m"][sensor, ray, return_index] = (
                    world_origin + range_m * world_direction
                )
                outputs["intensity"][sensor, ray, return_index] = intensity
                outputs["semantic_id"][sensor, ray, return_index] = winning_semantic
                outputs["valid"][sensor, ray, return_index] = True
                outputs["return_count"][sensor, ray] += 1
                previous_end = cluster_end
    return outputs
