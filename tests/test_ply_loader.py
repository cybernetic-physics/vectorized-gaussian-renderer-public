from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from isaacsim_gaussian_renderer.ply_loader import (
    SH_C0,
    canonicalize_3dgs_scene,
    load_ply_to_gaussians,
    ply_vertex_property_names,
)
from isaacsim_gaussian_renderer.scene import GaussianScene


def test_load_standard_3dgs_ply(tmp_path: Path) -> None:
    fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("semantic_id", "i4"),
    ]
    vertices = np.zeros(2, dtype=fields)
    vertices["x"] = [1.0, 2.0]
    vertices["rot_0"] = 1.0
    vertices["semantic_id"] = [4, 5]

    path = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)

    scene = load_ply_to_gaussians(path)
    assert "semantic_id" in ply_vertex_property_names(path)
    torch.testing.assert_close(scene.means[:, 0], torch.tensor([1.0, 2.0]))
    assert scene.features.shape == (2, 3)
    torch.testing.assert_close(scene.rotations[:, 0], torch.ones(2))
    torch.testing.assert_close(scene.semantic_ids, torch.tensor([4, 5]))


def test_canonicalize_raw_3dgs_parameters() -> None:
    raw = GaussianScene(
        means=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        scales=torch.tensor([[0.0, 0.1, -0.2], [0.3, -0.4, 0.5]]),
        rotations=torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        opacities=torch.tensor([0.0, 1.0]),
        features=torch.tensor(
            [[0.0, 1.0, -1.0, 9.0], [2.0, -2.0, 0.5, 8.0]],
        ),
        semantic_ids=torch.tensor([4, 5], dtype=torch.int64),
    )
    canonical = canonicalize_3dgs_scene(raw)

    torch.testing.assert_close(canonical.scales, raw.scales.exp())
    torch.testing.assert_close(canonical.opacities, raw.opacities.sigmoid())
    torch.testing.assert_close(
        canonical.rotations,
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        canonical.features[:, :3],
        (0.5 + SH_C0 * raw.features[:, :3]).clamp(0.0, 1.0),
    )
    assert canonical.features.is_contiguous()
