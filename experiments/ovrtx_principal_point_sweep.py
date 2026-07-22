"""Sweep camera pixel-center conventions against a fixed OVRTX render."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.ovrtx_parameter_sweep import load_candidate, metrics
from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle,
    synthetic_scene_manifest,
    synthetic_scene_tensors,
)


def finite_float_list(value: str) -> list[float]:
    values = [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of finite numbers."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--offsets",
        type=finite_float_list,
        default=finite_float_list(
            "-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5"
        ),
        help="Principal-point offsets applied independently to X and Y.",
    )
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument("--covariance-epsilon", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-render-output", type=Path, default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if (
        args.support_sigma <= 0
        or not math.isfinite(args.covariance_epsilon)
        or args.covariance_epsilon < 0
    ):
        raise ValueError(
            "Support sigma must be positive and covariance epsilon must be "
            "finite and non-negative."
        )
    candidate = load_candidate(args.candidate)
    height = int(candidate["rgb"].shape[1])
    width = int(candidate["rgb"].shape[2])
    device = torch.device("cuda:0")
    gaussian_count = 10_000
    seed = 42
    scene = synthetic_scene_tensors(
        gaussian_count,
        seed=seed,
        device=device,
    )
    base_camera = camera_bundle(
        1,
        width,
        height,
        device=device,
    )
    offset_pairs = [
        (offset_x, offset_y)
        for offset_y in args.offsets
        for offset_x in args.offsets
    ]
    batch = len(offset_pairs)
    viewmats = base_camera.viewmats.repeat(batch, 1, 1).contiguous()
    intrinsics = base_camera.intrinsics.repeat(batch, 1, 1)
    intrinsics[:, 0, 2] += torch.tensor(
        [pair[0] for pair in offset_pairs],
        device=device,
        dtype=torch.float32,
    )
    intrinsics[:, 1, 2] += torch.tensor(
        [pair[1] for pair in offset_pairs],
        device=device,
        dtype=torch.float32,
    )
    intrinsics = intrinsics.contiguous()
    scene_ids = torch.full(
        (batch,),
        101,
        device=device,
        dtype=torch.int64,
    )

    backend = CustomCudaBackend(
        max_visible_records=batch * 650,
        max_intersections=batch * 2_600,
        gaussian_support_sigma=args.support_sigma,
        covariance_epsilon=args.covariance_epsilon,
        output_srgb=False,
        deterministic=True,
    )
    service = RendererService(
        backend,
        height=height,
        width=width,
        max_views=batch,
    )
    service.initialize(stage=None, device=device)
    service.load_scene(
        101,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )
    outputs = service.render(
        viewmats,
        intrinsics,
        scene_ids,
    )
    service.synchronize()
    counters = backend.check_capacity(synchronize=False)
    custom_batch = {
        "rgb": outputs["rgb"].detach().cpu().numpy(),
        "alpha": outputs["alpha"][..., 0].detach().cpu().numpy(),
        "depth": outputs["depth"][..., 0].detach().cpu().numpy(),
        "semantic": (
            outputs["semantic_id"][..., 0].detach().cpu().numpy()
        ),
    }
    rows = []
    for index, (offset_x, offset_y) in enumerate(offset_pairs):
        custom = {
            name: values[index : index + 1]
            for name, values in custom_batch.items()
        }
        row_metrics = metrics(custom=custom, candidate=candidate)
        rows.append(
            {
                "principal_point_offset_x": offset_x,
                "principal_point_offset_y": offset_y,
                "camera_attributes_identical": (
                    offset_x == 0.0 and offset_y == 0.0
                ),
                "acceptance_eligible": (
                    offset_x == 0.0 and offset_y == 0.0
                ),
                "metrics": row_metrics,
                "_index": index,
            }
        )
    rows.sort(key=lambda row: row["metrics"]["threshold_distance"])
    best = rows[0]
    best_index = int(best.pop("_index"))
    for row in rows[1:]:
        row.pop("_index")
    if args.best_render_output is not None:
        args.best_render_output.parent.mkdir(parents=True, exist_ok=True)
        best_custom = {
            name: values[best_index : best_index + 1]
            for name, values in custom_batch.items()
        }
        np.savez_compressed(
            args.best_render_output,
            **best_custom,
            valid_depth=(
                np.isfinite(best_custom["depth"])
                & (best_custom["depth"] > 0)
                & (best_custom["alpha"] > 0)
            ),
            color_space=np.asarray("display_srgb"),
            background=np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
        )
    result = {
        "schema_version": "ovrtx-principal-point-sweep/v1",
        "diagnostic_only": True,
        "candidate": str(args.candidate),
        "scene": synthetic_scene_manifest(
            "synthetic-small",
            gaussian_count,
            seed,
        ),
        "camera_manifest": base_camera.manifest,
        "support_sigma": args.support_sigma,
        "covariance_epsilon": args.covariance_epsilon,
        "one_batched_render_call": True,
        "counters": counters,
        "row_count": len(rows),
        "best_row": best,
        "best_identical_camera_row": next(
            row for row in rows if row["camera_attributes_identical"]
        ),
        "best_render_output": (
            str(args.best_render_output)
            if args.best_render_output is not None
            else None
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    service.shutdown()


if __name__ == "__main__":
    main()
