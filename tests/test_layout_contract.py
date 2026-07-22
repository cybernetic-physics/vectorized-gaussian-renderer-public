import pytest
import torch

from isaacsim_gaussian_renderer.layout_contract import (
    CanonicalSceneLayout,
    RenderBatchLayout,
)


def _scene(count: int = 5) -> CanonicalSceneLayout:
    return CanonicalSceneLayout(
        means=torch.zeros((count, 3)),
        quats=torch.zeros((count, 4)),
        scales=torch.ones((count, 3)),
        opacities=torch.ones(count),
        features=torch.zeros((count, 3)),
        semantic_ids=torch.arange(count, dtype=torch.int32),
        scene_offsets=torch.tensor([0, 2, count], dtype=torch.int64),
    )


def _batch(batch: int = 3, height: int = 8, width: int = 12) -> RenderBatchLayout:
    return RenderBatchLayout(
        viewmats=torch.eye(4).repeat(batch, 1, 1),
        intrinsics=torch.eye(3).repeat(batch, 1, 1),
        env_xforms=torch.eye(4).repeat(batch, 1, 1),
        scene_ids=torch.zeros(batch, dtype=torch.int32),
        active_camera_ids=torch.tensor([0, 2], dtype=torch.int32),
        rgb=torch.zeros((batch, height, width, 3)),
        depth=torch.zeros((batch, height, width, 1)),
        alpha=torch.zeros((batch, height, width, 1)),
        semantic_id=torch.zeros((batch, height, width, 1), dtype=torch.int32),
    )


def test_canonical_scene_layout_accepts_valid_soa() -> None:
    _scene().validate()


def test_canonical_scene_layout_requires_offsets_cover_count() -> None:
    scene = _scene()
    bad = CanonicalSceneLayout(
        scene.means,
        scene.quats,
        scene.scales,
        scene.opacities,
        scene.features,
        scene.semantic_ids,
        torch.tensor([0, 99], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="end at gaussian count"):
        bad.validate()


def test_render_batch_layout_accepts_valid_outputs() -> None:
    _batch().validate()


def test_render_batch_layout_rejects_noncontiguous_output() -> None:
    batch = _batch()
    bad_rgb = torch.zeros((*batch.rgb.shape[:-1], 6))[..., ::2]
    bad = RenderBatchLayout(
        batch.viewmats,
        batch.intrinsics,
        batch.env_xforms,
        batch.scene_ids,
        batch.active_camera_ids,
        bad_rgb,
        batch.depth,
        batch.alpha,
        batch.semantic_id,
    )
    with pytest.raises(ValueError, match="contiguous"):
        bad.validate()


def test_render_batch_layout_requires_semantic_integer_output() -> None:
    batch = _batch()
    bad = RenderBatchLayout(
        batch.viewmats,
        batch.intrinsics,
        batch.env_xforms,
        batch.scene_ids,
        batch.active_camera_ids,
        batch.rgb,
        batch.depth,
        batch.alpha,
        torch.zeros_like(batch.alpha),
    )
    with pytest.raises(TypeError, match="semantic_id"):
        bad.validate()
