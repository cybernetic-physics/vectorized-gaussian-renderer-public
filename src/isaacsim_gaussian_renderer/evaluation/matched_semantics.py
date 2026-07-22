"""Deterministic semantic-label topologies for matched renderer tests."""

from __future__ import annotations

from typing import Literal

import torch


SemanticTopology = Literal[
    "spatial-octants-8",
    "interleaved-index-modulo-1024",
]
SEMANTIC_TOPOLOGIES: tuple[SemanticTopology, ...] = (
    "spatial-octants-8",
    "interleaved-index-modulo-1024",
)
REPRESENTATIVE_SEMANTIC_TOPOLOGY: SemanticTopology = "spatial-octants-8"
STRESS_SEMANTIC_TOPOLOGY: SemanticTopology = (
    "interleaved-index-modulo-1024"
)


def matched_semantic_ids(
    means: torch.Tensor,
    topology: SemanticTopology | str,
) -> torch.Tensor:
    """Return one int64 label per Gaussian without changing Gaussian order.

    Home Scan has no canonical semantic annotation.  The representative
    contract therefore uses eight spatially coherent octants.  Index-modulo
    labels are retained as an explicitly adversarial high-cardinality stress
    topology; they must not be presented as the representative robotics
    semantic workload.
    """

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape [G, 3].")
    if means.dtype != torch.float32:
        raise ValueError("means must use float32.")
    if means.shape[0] <= 0:
        raise ValueError("means must contain at least one Gaussian.")

    if topology == STRESS_SEMANTIC_TOPOLOGY:
        return torch.arange(
            means.shape[0],
            device=means.device,
            dtype=torch.int64,
        ).remainder_(1024)
    if topology != REPRESENTATIVE_SEMANTIC_TOPOLOGY:
        raise ValueError(
            f"Unsupported semantic topology {topology!r}; expected one of "
            f"{SEMANTIC_TOPOLOGIES}."
        )

    bounds_min = means.amin(dim=0)
    bounds_max = means.amax(dim=0)
    center = (bounds_min + bounds_max) * 0.5
    positive_half = means >= center
    return (
        positive_half[:, 0].to(torch.int64)
        | (positive_half[:, 1].to(torch.int64) << 1)
        | (positive_half[:, 2].to(torch.int64) << 2)
    ).contiguous()
