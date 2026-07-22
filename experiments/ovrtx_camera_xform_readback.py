#!/usr/bin/env python3
"""Compare authored camera matrices with OVRTX composed transform readback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import ovrtx
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_ovrtx import build_stage
from isaacsim_gaussian_renderer.camera_math import (
    opencv_viewmats_to_usd_camera_world_matrices,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_contract", type=Path)
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()

    with np.load(args.camera_contract, allow_pickle=False) as contract:
        all_viewmats = np.asarray(contract["viewmats"], dtype=np.float32)
    viewmats = torch.from_numpy(all_viewmats[: args.count])
    target_viewmats = torch.from_numpy(
        all_viewmats[args.count : args.count * 2]
    )
    expected = (
        opencv_viewmats_to_usd_camera_world_matrices(viewmats)
        .to(dtype=torch.float64)
        .numpy()
    )
    stage = build_stage(
        args.count,
        64,
        64,
        0.72,
        viewmats,
        "none",
        semantic_group_count=1,
        color_only=True,
    )
    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            keep_system_alive=True,
            log_level="warning",
            active_cuda_gpus="0",
            use_vulkan=True,
        )
    )
    renderer.open_usd_from_string(stage)
    paths = [f"/World/Camera_{index}" for index in range(args.count)]
    composed = np.from_dlpack(
        renderer.read_attribute(
            attribute_name="omni:xform",
            prim_paths=paths,
        )
    ).reshape(args.count, 4, 4).copy()
    target_expected = (
        opencv_viewmats_to_usd_camera_world_matrices(target_viewmats)
        .to(dtype=torch.float64)
        .numpy()
    )
    renderer.write_attribute(
        prim_paths=paths,
        attribute_name="omni:xform",
        tensor=target_expected,
        semantic=ovrtx.Semantic.XFORM_MAT4x4,
    )
    updated = np.from_dlpack(
        renderer.read_attribute(
            attribute_name="omni:xform",
            prim_paths=paths,
        )
    ).reshape(args.count, 4, 4).copy()
    print(
        "OVRTX_CAMERA_XFORM_READBACK "
        + json.dumps(
            {
                "expected": expected.tolist(),
                "composed": composed.tolist(),
                "target_expected": target_expected.tolist(),
                "updated": updated.tolist(),
                "composed_max_abs_error": float(
                    np.max(np.abs(composed - expected))
                ),
                "composed_transpose_max_abs_error": float(
                    np.max(np.abs(composed.transpose(0, 2, 1) - expected))
                ),
                "updated_max_abs_error": float(
                    np.max(np.abs(updated - target_expected))
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
