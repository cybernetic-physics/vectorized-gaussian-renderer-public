"""Compare one deterministic render against saved OVRTX temporal checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from isaacsim_gaussian_renderer.fidelity import load_render_output


CHECKPOINT_PATTERN = re.compile(
    r"candidate-ovrtx-temporal-(?P<frames>\d+)\.npz$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(
        np.square(
            candidate.astype(np.float64)
            - reference.astype(np.float64)
        ).mean()
    )
    return math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)


def compare(
    reference: Any,
    candidate: Any,
) -> dict[str, Any]:
    if reference.rgb.shape != candidate.rgb.shape:
        raise ValueError(
            "Reference and checkpoint shapes differ: "
            f"{reference.rgb.shape} != {candidate.rgb.shape}."
        )
    reference_valid = (
        reference.valid_depth
        if reference.valid_depth is not None
        else (
            np.isfinite(reference.depth)
            & (reference.depth > 0)
            & (reference.alpha > 0)
        )
    )
    candidate_valid = (
        candidate.valid_depth
        if candidate.valid_depth is not None
        else (
            np.isfinite(candidate.depth)
            & (candidate.depth > 0)
            & (candidate.alpha > 0)
        )
    )
    valid_depth = reference_valid & candidate_valid
    valid_depth_count = int(np.count_nonzero(valid_depth))
    depth_relative_error = (
        float(
            np.mean(
                np.abs(
                    candidate.depth[valid_depth]
                    - reference.depth[valid_depth]
                )
                / np.maximum(
                    np.abs(reference.depth[valid_depth]),
                    1.0e-8,
                )
            )
        )
        if valid_depth_count
        else None
    )
    semantic_mismatches = int(
        np.count_nonzero(reference.semantic != candidate.semantic)
    )
    return {
        "rgb_psnr_db": psnr(
            np.clip(reference.rgb, 0.0, 1.0),
            np.clip(candidate.rgb, 0.0, 1.0),
        ),
        "alpha_mae": float(
            np.abs(candidate.alpha - reference.alpha).mean()
        ),
        "depth_relative_error": depth_relative_error,
        "valid_depth_pixels": valid_depth_count,
        "semantic_agreement": float(
            np.mean(reference.semantic == candidate.semantic)
        ),
        "semantic_mismatch_pixels": semantic_mismatches,
    }


def main() -> None:
    args = parse_args()
    reference = load_render_output(args.reference)
    rows = []
    for checkpoint_path in sorted(
        args.checkpoint_root.glob(
            "candidate-ovrtx-temporal-*.npz"
        )
    ):
        match = CHECKPOINT_PATTERN.match(checkpoint_path.name)
        if match is None:
            continue
        rows.append(
            {
                "frames": int(match.group("frames")),
                "checkpoint": str(checkpoint_path),
                **compare(
                    reference,
                    load_render_output(checkpoint_path),
                ),
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No temporal checkpoints found under {args.checkpoint_root}."
        )
    payload = {
        "schema_version": "ovrtx-temporal-checkpoint-comparison/v1",
        "diagnostic_only": True,
        "reference": str(args.reference),
        "checkpoint_root": str(args.checkpoint_root),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
