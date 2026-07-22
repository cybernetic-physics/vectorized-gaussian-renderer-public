from __future__ import annotations

import math

import numpy as np

from experiments.lidar_cpu_oracle import render_lidar_oracle


def make_scene(
    depths: list[float],
    *,
    semantics: list[int] | None = None,
    reflectivity: list[float] | None = None,
    rotation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    count = len(depths)
    return {
        "means": np.array([[depth, 0.0, 0.0] for depth in depths], dtype=np.float64),
        "scales": np.array([[0.01, 1.0, 1.0]] * count, dtype=np.float64),
        "rotations": np.array(
            [rotation if rotation is not None else [1.0, 0.0, 0.0, 0.0]] * count,
            dtype=np.float64,
        ),
        "opacities": np.ones(count, dtype=np.float64),
        "semantic_ids": np.array(semantics or list(range(count)), dtype=np.int64),
        "reflectivity": None if reflectivity is None else np.array(reflectivity, dtype=np.float64),
    }


def identity_batch(batch: int) -> np.ndarray:
    return np.repeat(np.eye(4, dtype=np.float64)[None], batch, axis=0)


def test_oracle_one_two_returns_clustering_semantics_intensity_and_no_hit() -> None:
    scene = make_scene(
        [5.0, 5.01, 8.0],
        semantics=[9, 3, 11],
        reflectivity=[0.2, 0.6, 0.8],
    )
    rays = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = render_lidar_oracle(
        {4: scene},
        rays,
        np.array([0, 125], dtype=np.int64),
        identity_batch(1),
        np.array([4]),
        returns=2,
    )

    assert result["valid"][0, 0].tolist() == [True, True]
    np.testing.assert_allclose(result["range_m"][0, 0], [5.005, 8.0], atol=1e-12)
    assert result["semantic_id"][0, 0].tolist() == [3, 11]
    np.testing.assert_allclose(result["intensity"][0, 0], [0.4, 0.8], atol=1e-12)
    assert result["return_count"][0, 0] == 2
    assert not result["valid"][0, 1].any()
    assert np.isinf(result["range_m"][0, 1]).all()
    assert (result["semantic_id"][0, 1] == -1).all()
    assert (result["position_world_m"][0, 1] == 0).all()


def test_oracle_nonidentity_wxyz_rotation_and_transform_equivariance() -> None:
    half = math.sqrt(0.5)
    rotation_z_90 = np.array([half, 0.0, 0.0, half])
    scene = make_scene([0.0], semantics=[5], rotation=rotation_z_90)
    scene["means"][0] = [0.0, 5.0, 0.0]
    base = render_lidar_oracle(
        {1: scene},
        np.array([[0.0, 1.0, 0.0]]),
        np.array([9]),
        identity_batch(1),
        np.array([1]),
    )
    assert base["valid"][0, 0, 0]
    np.testing.assert_allclose(base["position_world_m"][0, 0, 0], [0.0, 5.0, 0.0], atol=1e-12)

    translated_sensors = identity_batch(1)
    translated_sensors[0, :3, 3] = [3.0, -2.0, 1.0]
    translated_scene = identity_batch(1)
    translated_scene[0, :3, 3] = [3.0, -2.0, 1.0]
    moved = render_lidar_oracle(
        {1: scene},
        np.array([[0.0, 1.0, 0.0]]),
        np.array([9]),
        translated_sensors,
        np.array([1]),
        scene_to_world=translated_scene,
    )
    np.testing.assert_allclose(moved["range_m"], base["range_m"], atol=1e-12)
    np.testing.assert_allclose(
        moved["position_world_m"][0, 0, 0],
        base["position_world_m"][0, 0, 0] + [3.0, -2.0, 1.0],
        atol=1e-12,
    )

    rigid = identity_batch(1)
    rigid[0, :3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = render_lidar_oracle(
        {1: scene},
        np.array([[0.0, 1.0, 0.0]]),
        np.array([9]),
        rigid,
        np.array([1]),
        scene_to_world=rigid,
    )
    np.testing.assert_allclose(rotated["range_m"], base["range_m"], atol=1e-12)
    np.testing.assert_allclose(
        rotated["position_world_m"][0, 0, 0],
        rigid[0, :3, :3] @ base["position_world_m"][0, 0, 0],
        atol=1e-12,
    )


def test_oracle_frozen_boundaries_parallel_grazing_angled_and_thin_surfaces() -> None:
    angle = math.radians(30.0)
    angled_direction = np.array([math.cos(angle), math.sin(angle), 0.0])
    scenes = {
        1: make_scene([0.05], semantics=[1]),
        2: make_scene([200.0], semantics=[2]),
        3: make_scene([0.049], semantics=[3]),
        4: make_scene([200.001], semantics=[4]),
        5: make_scene([5.0], semantics=[5]),
        6: make_scene(
            [0.0],
            semantics=[6],
            rotation=np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]),
        ),
        7: make_scene([4.0], semantics=[7]),
        8: make_scene([4.0], semantics=[8]),
    }
    scenes[6]["means"][0] = 5.0 * angled_direction
    scenes[7]["means"][0] = [4.0, 0.05, 0.0]
    scenes[7]["scales"][0] = [0.01, 0.06, 0.5]
    scenes[8]["scales"][0] = [0.5, 1.0, 1.0]
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            angled_direction,
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    # Each batch entry uses one selected ray; the other rays are irrelevant to
    # this assertion but keep the shared-pattern public contract intact.
    result = render_lidar_oracle(
        scenes,
        directions,
        np.arange(len(directions), dtype=np.int64),
        identity_batch(8),
        np.arange(1, 9, dtype=np.int64),
    )
    selected = result["valid"][np.arange(8), np.arange(8), 0]
    assert selected.tolist() == [True, True, False, False, False, True, True, False]
    np.testing.assert_allclose(
        result["range_m"][[0, 1, 5, 6], [0, 1, 5, 6], 0],
        [0.05, 200.0, 5.0, 4.0],
        atol=1e-10,
    )


def test_oracle_batch_individual_active_subset_and_scene_transforms() -> None:
    scenes = {2: make_scene([4.0], semantics=[2]), 8: make_scene([7.0], semantics=[8])}
    sensors = identity_batch(3)
    scene_transforms = identity_batch(3)
    scene_transforms[2, 0, 3] = 2.0
    batch = render_lidar_oracle(
        scenes,
        np.array([[1.0, 0.0, 0.0]]),
        np.array([0]),
        sensors,
        np.array([2, 8, 2]),
        scene_to_world=scene_transforms,
        active_sensor_ids=np.array([0, 2]),
    )
    np.testing.assert_allclose(batch["range_m"][[0, 2], 0, 0], [4.0, 6.0])
    assert not batch["valid"][1].any()
    for sensor in (0, 2):
        individual = render_lidar_oracle(
            scenes,
            np.array([[1.0, 0.0, 0.0]]),
            np.array([0]),
            sensors[sensor : sensor + 1],
            np.array([2]),
            scene_to_world=scene_transforms[sensor : sensor + 1],
        )
        np.testing.assert_array_equal(batch["valid"][sensor], individual["valid"][0])
        np.testing.assert_allclose(batch["range_m"][sensor], individual["range_m"][0])


def test_oracle_randomized_permutation_invariance_held_out_seeds() -> None:
    for seed in (731, 2027, 99173):
        generator = np.random.default_rng(seed)
        count = 24
        scene = make_scene(
            generator.uniform(2.0, 12.0, size=count).tolist(),
            semantics=generator.integers(0, 5, size=count).tolist(),
            reflectivity=generator.uniform(0.1, 0.9, size=count).tolist(),
        )
        scene["means"][:, 1:] = generator.normal(0.0, 0.4, size=(count, 2))
        rays = generator.normal(size=(9, 3))
        rays[:, 0] = np.abs(rays[:, 0]) + 0.5
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        reference = render_lidar_oracle(
            {6: scene}, rays, np.arange(9), identity_batch(1), np.array([6]), returns=2
        )
        permutation = generator.permutation(count)
        permuted_scene = {
            key: None if value is None else value[permutation]
            for key, value in scene.items()
        }
        candidate = render_lidar_oracle(
            {6: permuted_scene}, rays, np.arange(9), identity_batch(1), np.array([6]), returns=2
        )
        np.testing.assert_array_equal(candidate["valid"], reference["valid"])
        np.testing.assert_array_equal(candidate["semantic_id"], reference["semantic_id"])
        np.testing.assert_allclose(candidate["range_m"], reference["range_m"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(candidate["intensity"], reference["intensity"], rtol=0, atol=1e-12)
