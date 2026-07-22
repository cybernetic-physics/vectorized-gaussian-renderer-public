"""Diagnose OVRTX/custom convention gaps without producing acceptance claims."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle,
    synthetic_scene_manifest,
    synthetic_scene_tensors,
)


def float_list(value: str) -> list[float]:
    values = [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of positive numbers."
        )
    return values


def nonnegative_float_list(value: str) -> list[float]:
    values = [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    if (
        not values
        or any(not math.isfinite(item) for item in values)
        or min(values) < 0
    ):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of finite non-negative numbers."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Fixed temporally averaged OVRTX render-output NPZ.",
    )
    parser.add_argument(
        "--scale-multipliers",
        type=float_list,
        default=float_list("0.85,0.9,0.95,1.0,1.05,1.1"),
    )
    parser.add_argument(
        "--opacity-multipliers",
        type=float_list,
        default=float_list("0.85,0.9,0.95,1.0,1.05,1.1,1.15"),
    )
    parser.add_argument(
        "--support-sigmas",
        type=float_list,
        default=float_list("3.0,3.33"),
    )
    parser.add_argument(
        "--covariance-epsilons",
        type=nonnegative_float_list,
        default=nonnegative_float_list("0.3"),
    )
    parser.add_argument(
        "--ray-gaussian-evaluation",
        action="store_true",
        help="Use exact ray/3D-Gaussian evaluation in the native rasterizer.",
    )
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument(
        "--best-identical-render-output",
        type=Path,
        default=None,
        help=(
            "Optionally save the custom tensors for the best row whose scale "
            "and opacity match the authored scene."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_candidate(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"rgb", "alpha", "depth", "semantic"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"{path} is missing arrays: {sorted(missing)}."
            )
        candidate = {
            name: np.asarray(data[name])
            for name in required
        }
    if candidate["rgb"].shape[0] != 1:
        raise ValueError("The diagnostic currently expects one camera.")
    return candidate


def metrics(
    *,
    custom: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    rgb_mse = float(
        np.square(custom["rgb"] - candidate["rgb"]).mean()
    )
    rgb_psnr_db = (
        math.inf
        if rgb_mse == 0
        else -10.0 * math.log10(rgb_mse)
    )
    alpha_mae = float(
        np.abs(custom["alpha"] - candidate["alpha"]).mean()
    )
    semantic_agreement = float(
        np.mean(custom["semantic"] == candidate["semantic"])
    )
    valid_depth = (
        np.isfinite(custom["depth"])
        & np.isfinite(candidate["depth"])
        & (custom["depth"] > 0)
        & (candidate["depth"] > 0)
        & (custom["alpha"] > 0)
        & (candidate["alpha"] > 0)
    )
    valid_depth_count = int(np.count_nonzero(valid_depth))
    depth_relative_error = (
        float(
            np.mean(
                np.abs(
                    custom["depth"][valid_depth]
                    - candidate["depth"][valid_depth]
                )
                / np.maximum(
                    np.abs(candidate["depth"][valid_depth]),
                    1.0e-8,
                )
            )
        )
        if valid_depth_count
        else None
    )
    threshold_distance = (
        max(0.0, 40.0 - rgb_psnr_db) / 40.0
        + alpha_mae / 0.005
        + max(0.0, 0.999 - semantic_agreement) / 0.001
        + (
            depth_relative_error / 0.01
            if depth_relative_error is not None
            else 100.0
        )
    )
    return {
        "rgb_psnr_db": rgb_psnr_db,
        "alpha_mae": alpha_mae,
        "semantic_agreement": semantic_agreement,
        "valid_depth_pixels": valid_depth_count,
        "depth_relative_error": depth_relative_error,
        "threshold_distance": threshold_distance,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if (
        args.tile_size <= 0
        or args.tile_size > 16
        or args.tile_size & (args.tile_size - 1)
    ):
        raise ValueError("--tile-size must be a power of two in [1, 16].")
    candidate = load_candidate(args.candidate)
    height = int(candidate["rgb"].shape[1])
    width = int(candidate["rgb"].shape[2])
    device = torch.device("cuda:0")
    gaussian_count = 10_000
    seed = 42
    base_scene = synthetic_scene_tensors(
        gaussian_count,
        seed=seed,
        device=device,
    )
    cameras = camera_bundle(
        1,
        width,
        height,
        device=device,
    )
    scene_ids = torch.tensor(
        [101],
        device=device,
        dtype=torch.int64,
    )
    rows = []
    best_identical_custom: dict[str, np.ndarray] | None = None
    best_identical_score = math.inf
    for support_sigma in args.support_sigmas:
        for covariance_epsilon in args.covariance_epsilons:
            for scale_multiplier in args.scale_multipliers:
                for opacity_multiplier in args.opacity_multipliers:
                    backend = CustomCudaBackend(
                        max_visible_records=gaussian_count,
                        max_intersections=gaussian_count * 40,
                        gaussian_support_sigma=support_sigma,
                        covariance_epsilon=covariance_epsilon,
                        ray_gaussian_evaluation=(
                            args.ray_gaussian_evaluation
                        ),
                        tile_size=args.tile_size,
                        output_srgb=False,
                    )
                    service = RendererService(
                        backend,
                        height=height,
                        width=width,
                        max_views=1,
                    )
                    service.initialize(stage=None, device=device)
                    service.load_scene(
                        101,
                        means=base_scene["means"],
                        scales=(
                            base_scene["scales"] * scale_multiplier
                        ).contiguous(),
                        rotations=base_scene["quats"],
                        opacities=(
                            base_scene["opacities"]
                            * opacity_multiplier
                        )
                        .clamp_max(1.0)
                        .contiguous(),
                        features=base_scene["colors"],
                        semantic_ids=base_scene["semantic_ids"].to(
                            torch.int64
                        ),
                    )
                    outputs = service.render(
                        cameras.viewmats,
                        cameras.intrinsics,
                        scene_ids,
                    )
                    service.synchronize()
                    counters = backend.check_capacity(
                        synchronize=False
                    )
                    custom = {
                        "rgb": outputs["rgb"].detach().cpu().numpy(),
                        "alpha": (
                            outputs["alpha"][..., 0]
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                        "depth": (
                            outputs["depth"][..., 0]
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                        "semantic": (
                            outputs["semantic_id"][..., 0]
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                    }
                    row = {
                        "support_sigma": support_sigma,
                        "covariance_epsilon": covariance_epsilon,
                        "tile_size": args.tile_size,
                        "scale_multiplier": scale_multiplier,
                        "opacity_multiplier": opacity_multiplier,
                        "scene_attributes_identical": (
                            scale_multiplier == 1.0
                            and opacity_multiplier == 1.0
                        ),
                        "acceptance_eligible": (
                            scale_multiplier == 1.0
                            and opacity_multiplier == 1.0
                        ),
                        "counters": counters,
                        "metrics": metrics(
                            custom=custom,
                            candidate=candidate,
                        ),
                    }
                    rows.append(row)
                    if (
                        row["scene_attributes_identical"]
                        and row["metrics"]["threshold_distance"]
                        < best_identical_score
                    ):
                        best_identical_score = row["metrics"][
                            "threshold_distance"
                        ]
                        best_identical_custom = custom
                    service.shutdown()
                    torch.cuda.empty_cache()

    rows.sort(
        key=lambda row: row["metrics"]["threshold_distance"]
    )
    result = {
        "schema_version": "ovrtx-custom-parameter-sweep/v1",
        "diagnostic_only": True,
        "candidate": str(args.candidate),
        "scene": synthetic_scene_manifest(
            "synthetic-small",
            gaussian_count,
            seed,
        ),
        "camera_manifest": cameras.manifest,
        "color_contract": "identity_authored_display_rgb",
        "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
        "tile_size": args.tile_size,
        "warning": (
            "Rows that modify scale or opacity do not use identical Gaussian "
            "attributes and cannot support acceptance claims."
        ),
        "row_count": len(rows),
        "best_row": rows[0],
        "best_identical_scene_row": min(
            (
                row
                for row in rows
                if row["scene_attributes_identical"]
            ),
            key=lambda row: row["metrics"][
                "threshold_distance"
            ],
        ),
        "rows": rows,
    }
    if args.best_identical_render_output is not None:
        if best_identical_custom is None:
            raise RuntimeError("No identical-scene render was produced.")
        args.best_identical_render_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        np.savez_compressed(
            args.best_identical_render_output,
            **best_identical_custom,
            valid_depth=(
                np.isfinite(best_identical_custom["depth"])
                & (best_identical_custom["depth"] > 0)
                & (best_identical_custom["alpha"] > 0)
            ),
            color_space=np.asarray("display_srgb"),
            background=np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
        )
        result["best_identical_render_output"] = str(
            args.best_identical_render_output
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
