import torch

from benchmarks.run_trajectory import (
    semantic_output_from_top_contributors,
    stats,
)


def test_trajectory_stats_preserve_raw_samples() -> None:
    samples = [1.25, 2.5, 3.75]

    result = stats(samples)

    assert result["count"] == 3
    assert result["samples"] == samples


def test_gsplat_semantics_obey_production_alpha_background_policy() -> None:
    contributor_ids = torch.tensor([[[[2], [1], [-1]]]], dtype=torch.int32)
    alpha = torch.tensor([[[[0.5], [0.009], [0.8]]]], dtype=torch.float32)
    scene_semantic_ids = torch.tensor([10, 11, 12], dtype=torch.int64)

    semantic = semantic_output_from_top_contributors(
        contributor_ids,
        alpha,
        scene_semantic_ids,
    )

    assert semantic[..., 0].tolist() == [[[12, -1, -1]]]
