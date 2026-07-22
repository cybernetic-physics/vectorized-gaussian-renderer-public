import pytest
import torch

pytest.importorskip("ovrtx")

from benchmarks.run_ovrtx import (
    OVRTX_MAX_GAUSSIANS_PER_PRIM,
    build_stage,
    splat_prim_semantic_ids,
)


def test_splat_prim_semantic_ids_shards_at_24_bit_limit() -> None:
    assert splat_prim_semantic_ids(
        [OVRTX_MAX_GAUSSIANS_PER_PRIM + 1, 3]
    ) == [0, 0, 1]


def test_build_stage_authors_repeated_semantic_labels_for_shards() -> None:
    stage = build_stage(
        1,
        64,
        64,
        0.9,
        torch.eye(4).unsqueeze(0),
        "none",
        semantic_group_count=2,
        color_only=True,
        splat_prim_semantic_ids_override=[0, 0, 1],
    )

    assert stage.count('ParticleField3DGaussianSplat "Splats_') == 3
    assert stage.count('semanticData = "0"') == 2
    assert stage.count('semanticData = "1"') == 1


def test_build_stage_authors_projection_sorting_and_fractional_opacity() -> None:
    stage = build_stage(
        1,
        64,
        64,
        0.9,
        torch.eye(4).unsqueeze(0),
        "none",
        semantic_group_count=2,
        color_only=True,
        projection_mode="tangential",
        sorting_mode="rayHitDistance",
        fractional_opacity=True,
    )

    assert stage.count('projectionModeHint = "tangential"') == 2
    assert stage.count('sortingModeHint = "rayHitDistance"') == 2
    assert "omni:rtx:pt:fractionalCutoutOpacity = 1" in stage
    assert "omni:rtx:rt:fractionalOpacity = 1" in stage
