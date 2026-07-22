from __future__ import annotations

import pytest
import torch

from isaacsim_gaussian_renderer.evaluation.matched_semantics import (
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    STRESS_SEMANTIC_TOPOLOGY,
    matched_semantic_ids,
)


def test_spatial_octants_are_deterministic_and_coherent() -> None:
    means = torch.tensor(
        [
            [-2.0, -2.0, -2.0],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [2.0, -2.0, 2.0],
        ],
        dtype=torch.float32,
    )

    labels = matched_semantic_ids(
        means, REPRESENTATIVE_SEMANTIC_TOPOLOGY
    )

    assert labels.dtype == torch.int64
    assert labels.is_contiguous()
    assert labels.tolist() == [0, 0, 7, 7, 5]


def test_interleaved_topology_remains_an_explicit_stress_case() -> None:
    means = torch.zeros((1026, 3), dtype=torch.float32)

    labels = matched_semantic_ids(means, STRESS_SEMANTIC_TOPOLOGY)

    assert labels[:3].tolist() == [0, 1, 2]
    assert labels[-3:].tolist() == [1023, 0, 1]


@pytest.mark.parametrize(
    "means",
    (
        torch.zeros((0, 3), dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float32),
        torch.zeros((2, 3), dtype=torch.float64),
    ),
)
def test_matched_semantics_rejects_invalid_means(means: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        matched_semantic_ids(means, REPRESENTATIVE_SEMANTIC_TOPOLOGY)
