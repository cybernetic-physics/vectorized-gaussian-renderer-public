#!/usr/bin/env python3
"""Report raw and renderer-canonical Gaussian parameter distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from isaacsim_gaussian_renderer.ply_loader import (
    canonicalize_3dgs_scene,
    load_ply_to_gaussians,
)


def summary(value: torch.Tensor) -> dict[str, float]:
    flattened = value.detach().float().flatten().cpu()
    sample_limit = 1_000_000
    stride = max(1, (flattened.numel() + sample_limit - 1) // sample_limit)
    sample = flattened[::stride]
    quantiles = torch.quantile(
        sample,
        torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0]),
    )
    return {
        "min": float(flattened.min()),
        "p01": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(flattened.max()),
        "mean": float(flattened.mean()),
        "quantile_sample_stride": float(stride),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()

    raw = load_ply_to_gaussians(args.scene)
    canonical = canonicalize_3dgs_scene(raw)
    report = {
        "scene": str(args.scene),
        "count": raw.count,
        "raw": {
            "opacity": summary(raw.opacities),
            "dc": summary(raw.features[:, :3]),
            "log_scale": summary(raw.scales),
        },
        "canonical": {
            "opacity": summary(canonical.opacities),
            "rgb": summary(canonical.features[:, :3]),
            "scale": summary(canonical.scales),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
