from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.generate_flashgs_dynamic_contract import build_master


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generator_args(max_batch: int) -> argparse.Namespace:
    return argparse.Namespace(
        route=PROJECT_ROOT
        / "benchmarks/camera_paths/home-scan-walkthrough-v1.json",
        output_root=PROJECT_ROOT / "outputs/unused-test-contracts",
        timesteps=108,
        warmup_frames=8,
        max_batch=max_batch,
        width=128,
        height=128,
        fps=60.0,
        focal_scale=0.72,
        seed=20260718,
    )


def test_camera_master_is_independent_of_bounded_emission_batch() -> None:
    bounded, bounded_audit = build_master(generator_args(8))
    complete, complete_audit = build_master(generator_args(1024))

    assert bounded.trajectory_id == complete.trajectory_id
    assert bounded_audit == complete_audit
    assert np.array_equal(bounded.viewmats, complete.viewmats)
    assert np.array_equal(bounded.intrinsics, complete.intrinsics)
