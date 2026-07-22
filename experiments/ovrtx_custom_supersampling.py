"""Test whether OVRTX temporal convergence is subpixel sample integration."""

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
    synthetic_scene_manifest,
    synthetic_scene_tensors,
)
from isaacsim_gaussian_renderer.fidelity import (
    FIDELITY_THRESHOLDS,
    RenderOutput,
    compare_render_outputs,
    load_camera_bundle,
    load_render_output,
)


def positive_int_list(value: str) -> list[int]:
    values = [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of positive integers."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Converged OVRTX render-output NPZ.",
    )
    parser.add_argument(
        "--camera-bundle",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--grid-sizes",
        type=positive_int_list,
        default=positive_int_list("1,2,4,8"),
        help="Uniform NxN subpixel grids, each rendered in one batched call.",
    )
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument("--covariance-epsilon", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-render-output", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    return parser.parse_args()


def aggregate_samples(
    outputs: dict[str, torch.Tensor],
    *,
    semantic_count: int,
) -> dict[str, torch.Tensor]:
    rgb = outputs["rgb"].mean(dim=0, keepdim=True)
    alpha_samples = outputs["alpha"][..., 0]
    alpha = alpha_samples.mean(dim=0, keepdim=True)
    depth_samples = outputs["depth"][..., 0]
    valid_depth = (
        torch.isfinite(depth_samples)
        & (depth_samples > 0)
        & (alpha_samples > 0)
    )
    depth_weight = torch.where(
        valid_depth,
        alpha_samples,
        torch.zeros_like(alpha_samples),
    )
    depth_numerator = torch.where(
        valid_depth,
        depth_samples * alpha_samples,
        torch.zeros_like(depth_samples),
    ).sum(dim=0, keepdim=True)
    depth_weight_sum = depth_weight.sum(dim=0, keepdim=True)
    depth = depth_numerator / depth_weight_sum.clamp_min(
        torch.finfo(depth_numerator.dtype).eps
    )
    depth = torch.where(
        depth_weight_sum > 0,
        depth,
        torch.full_like(depth, float("inf")),
    )

    labels = outputs["semantic_id"][..., 0].reshape(
        outputs["semantic_id"].shape[0],
        -1,
    )
    pixel_count = labels.shape[1]
    pixel_ids = torch.arange(
        pixel_count,
        device=labels.device,
        dtype=torch.int64,
    ).expand(labels.shape[0], -1)
    valid_labels = (labels >= 0) & (labels < semantic_count)
    linear_indices = (
        pixel_ids[valid_labels] * semantic_count
        + labels[valid_labels]
    )
    semantic_counts = torch.zeros(
        pixel_count * semantic_count,
        device=labels.device,
        dtype=torch.int32,
    )
    semantic_counts.scatter_add_(
        0,
        linear_indices,
        torch.ones_like(linear_indices, dtype=torch.int32),
    )
    semantic = semantic_counts.view(
        pixel_count,
        semantic_count,
    ).argmax(dim=1)
    semantic = semantic.view(1, *alpha.shape[1:]).to(torch.int64)
    semantic[alpha == 0] = -1
    return {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "semantic": semantic,
        "valid_depth": depth_weight_sum > 0,
    }


def as_render_output(
    tensors: dict[str, torch.Tensor],
    *,
    camera_bundle_id: str,
    source: str,
) -> RenderOutput:
    output = RenderOutput(
        rgb=tensors["rgb"].detach().cpu().numpy(),
        alpha=tensors["alpha"].detach().cpu().numpy(),
        depth=tensors["depth"].detach().cpu().numpy(),
        semantic=tensors["semantic"].detach().cpu().numpy(),
        valid_depth=tensors["valid_depth"].detach().cpu().numpy(),
        color_space="display_srgb",
        background=(0.0, 0.0, 0.0),
        camera_bundle_id=camera_bundle_id,
        source=source,
    )
    output.validate()
    return output


def save_render_output(path: Path, output: RenderOutput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rgb=output.rgb,
        alpha=output.alpha,
        depth=output.depth,
        semantic=output.semantic,
        valid_depth=output.valid_depth,
        color_space=np.asarray(output.color_space),
        background=np.asarray(output.background, dtype=np.float32),
        camera_bundle_id=np.asarray(output.camera_bundle_id),
    )


def threshold_distance(metrics: dict[str, float | None]) -> float:
    depth = metrics["depth_rel_error"]
    return (
        max(
            0.0,
            FIDELITY_THRESHOLDS["rgb_psnr_db"]
            - float(metrics["rgb_psnr_db"]),
        )
        / FIDELITY_THRESHOLDS["rgb_psnr_db"]
        + max(
            0.0,
            FIDELITY_THRESHOLDS["rgb_ssim"]
            - float(metrics["rgb_ssim"]),
        )
        / (1.0 - FIDELITY_THRESHOLDS["rgb_ssim"])
        + float(metrics["alpha_mae"])
        / FIDELITY_THRESHOLDS["alpha_mae"]
        + (
            float(depth) / FIDELITY_THRESHOLDS["depth_rel_error"]
            if depth is not None
            else 100.0
        )
        + max(
            0.0,
            FIDELITY_THRESHOLDS["semantic_agreement"]
            - float(metrics["semantic_agreement"]),
        )
        / (1.0 - FIDELITY_THRESHOLDS["semantic_agreement"])
    )


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
    fidelity_bundle = load_camera_bundle(args.camera_bundle)
    if len(fidelity_bundle.cameras) != 1:
        raise ValueError("This diagnostic currently expects one camera.")
    candidate = load_render_output(args.candidate)
    device = torch.device("cuda:0")
    gaussian_count = 10_000
    seed = 42
    scene = synthetic_scene_tensors(
        gaussian_count,
        seed=seed,
        device=device,
    )
    scene_manifest = synthetic_scene_manifest(
        "synthetic-small",
        gaussian_count,
        seed,
    )
    if fidelity_bundle.scene_checksum != scene_manifest["checksum_sha256"]:
        raise ValueError("Camera bundle scene checksum does not match.")

    camera = fidelity_bundle.cameras[0]
    base_viewmat = torch.tensor(
        camera.viewmat,
        device=device,
        dtype=torch.float32,
    )
    base_intrinsics = torch.tensor(
        camera.intrinsics,
        device=device,
        dtype=torch.float32,
    )
    rows: list[dict[str, Any]] = []
    rendered: dict[int, RenderOutput] = {}
    for grid_size in args.grid_sizes:
        samples = grid_size * grid_size
        offsets_1d = (
            (torch.arange(
                grid_size,
                device=device,
                dtype=torch.float32,
            ) + 0.5)
            / grid_size
            - 0.5
        )
        offset_y, offset_x = torch.meshgrid(
            offsets_1d,
            offsets_1d,
            indexing="ij",
        )
        viewmats = base_viewmat.unsqueeze(0).repeat(
            samples,
            1,
            1,
        ).contiguous()
        intrinsics = base_intrinsics.unsqueeze(0).repeat(
            samples,
            1,
            1,
        )
        intrinsics[:, 0, 2] -= offset_x.reshape(-1)
        intrinsics[:, 1, 2] -= offset_y.reshape(-1)
        intrinsics = intrinsics.contiguous()
        scene_ids = torch.full(
            (samples,),
            int(camera.scene_id),
            device=device,
            dtype=torch.int64,
        )

        backend = CustomCudaBackend(
            max_visible_records=samples * 650,
            max_intersections=samples * 2_600,
            gaussian_support_sigma=args.support_sigma,
            covariance_epsilon=args.covariance_epsilon,
            output_srgb=False,
            deterministic=True,
        )
        service = RendererService(
            backend,
            height=fidelity_bundle.height,
            width=fidelity_bundle.width,
            max_views=samples,
        )
        service.initialize(stage=None, device=device)
        service.load_scene(
            int(camera.scene_id),
            means=scene["means"],
            scales=scene["scales"],
            rotations=scene["quats"],
            opacities=scene["opacities"],
            features=scene["colors"],
            semantic_ids=scene["semantic_ids"].to(torch.int64),
        )
        sample_outputs = service.render(
            viewmats,
            intrinsics,
            scene_ids,
        )
        service.synchronize()
        counters = backend.check_capacity(synchronize=False)
        aggregated = aggregate_samples(
            sample_outputs,
            semantic_count=int(scene["semantic_ids"].max().item()) + 1,
        )
        custom = as_render_output(
            aggregated,
            camera_bundle_id=fidelity_bundle.bundle_id,
            source=f"custom-cuda-{grid_size}x{grid_size}-subpixel-grid",
        )
        report = compare_render_outputs(
            reference=custom,
            candidate=candidate,
            camera_bundle=fidelity_bundle,
            config_id=(
                f"ovrtx-temporal-vs-custom-{grid_size}x{grid_size}-"
                f"subpixel-epsilon-{args.covariance_epsilon:g}"
            ),
            require_lpips=False,
        )
        metrics = report["per_view"][0]["metrics"]
        rows.append(
            {
                "grid_size": grid_size,
                "samples_per_pixel": samples,
                "one_batched_render_call": True,
                "counters": counters,
                "metrics": metrics,
                "threshold_distance_without_lpips": threshold_distance(
                    metrics
                ),
            }
        )
        rendered[grid_size] = custom
        service.shutdown()
        torch.cuda.empty_cache()

    rows.sort(
        key=lambda row: row["threshold_distance_without_lpips"]
    )
    best_grid = int(rows[0]["grid_size"])
    best_render = rendered[best_grid]
    if args.best_render_output is not None:
        save_render_output(args.best_render_output, best_render)
    full_report = compare_render_outputs(
        reference=best_render,
        candidate=candidate,
        camera_bundle=fidelity_bundle,
        output_dir=args.report_dir,
        config_id=(
            f"ovrtx-temporal-vs-custom-{best_grid}x{best_grid}-"
            f"subpixel-epsilon-{args.covariance_epsilon:g}"
        ),
        require_lpips=True,
        max_artifact_views=1,
    )
    result = {
        "schema_version": "ovrtx-custom-supersampling/v1",
        "diagnostic_only": True,
        "candidate": str(args.candidate),
        "camera_bundle": str(args.camera_bundle),
        "scene": scene_manifest,
        "support_sigma": args.support_sigma,
        "covariance_epsilon": args.covariance_epsilon,
        "subpixel_pattern": "uniform-cell-centers",
        "row_count": len(rows),
        "rows": rows,
        "best_grid_size": best_grid,
        "best_full_report": full_report,
        "best_render_output": (
            str(args.best_render_output)
            if args.best_render_output is not None
            else None
        ),
        "report_dir": (
            str(args.report_dir)
            if args.report_dir is not None
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
