"""Compare exact ray/3D-Gaussian hit probabilities with converged OVRTX."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.ovrtx_parameter_sweep import metrics
from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    synthetic_scene_manifest,
    synthetic_scene_tensors,
)
from isaacsim_gaussian_renderer.fidelity import (
    RenderOutput,
    compare_render_outputs,
    load_camera_bundle,
    load_render_output,
)


def positive_float_list(value: str) -> list[float]:
    values = [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    if (
        not values
        or min(values) <= 0
        or any(not math.isfinite(item) for item in values)
    ):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of finite positive numbers."
        )
    return values


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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--camera-bundle", type=Path, required=True)
    parser.add_argument(
        "--support-sigmas",
        type=positive_float_list,
        default=positive_float_list("2.5,2.7,3.0,3.33"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-render-output", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--semantic-prediction-frames",
        type=positive_int_list,
        default=positive_int_list(
            "65536,262144,1048576,4194304"
        ),
    )
    return parser.parse_args()


def covariance_matrices(packed: torch.Tensor) -> torch.Tensor:
    matrices = torch.empty(
        (packed.shape[0], 3, 3),
        device=packed.device,
        dtype=packed.dtype,
    )
    matrices[:, 0, 0] = packed[:, 0]
    matrices[:, 0, 1] = packed[:, 1]
    matrices[:, 0, 2] = packed[:, 2]
    matrices[:, 1, 0] = packed[:, 1]
    matrices[:, 1, 1] = packed[:, 3]
    matrices[:, 1, 2] = packed[:, 4]
    matrices[:, 2, 0] = packed[:, 2]
    matrices[:, 2, 1] = packed[:, 4]
    matrices[:, 2, 2] = packed[:, 5]
    return matrices


def render_output_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    camera_bundle_id: str,
    source: str,
) -> RenderOutput:
    output = RenderOutput(
        rgb=arrays["rgb"].astype(np.float32, copy=False),
        alpha=arrays["alpha"].astype(np.float32, copy=False),
        depth=arrays["depth"].astype(np.float32, copy=False),
        semantic=arrays["semantic"].astype(np.int64, copy=False),
        valid_depth=(
            np.isfinite(arrays["depth"])
            & (arrays["depth"] > 0)
            & (arrays["alpha"] > 0)
        ),
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


def simple_arrays(output: RenderOutput) -> dict[str, np.ndarray]:
    return {
        "rgb": output.rgb,
        "alpha": output.alpha,
        "depth": output.depth,
        "semantic": output.semantic,
    }


def quantiles(tensor: torch.Tensor) -> dict[str, float] | None:
    if tensor.numel() == 0:
        return None
    values = torch.quantile(
        tensor.to(torch.float32),
        torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
            device=tensor.device,
        ),
    ).detach().cpu().tolist()
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p25", "median", "p75", "p90", "p99", "max"),
            values,
            strict=True,
        )
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    fidelity_bundle = load_camera_bundle(args.camera_bundle)
    if len(fidelity_bundle.cameras) != 1:
        raise ValueError("This diagnostic currently expects one camera.")
    candidate_output = load_render_output(args.candidate)
    candidate = simple_arrays(candidate_output)
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
    viewmat = torch.tensor(
        camera.viewmat,
        device=device,
        dtype=torch.float32,
    )
    intrinsics = torch.tensor(
        camera.intrinsics,
        device=device,
        dtype=torch.float32,
    )
    backend = CustomCudaBackend(
        max_visible_records=5_000,
        max_intersections=500_000,
        gaussian_support_sigma=max(args.support_sigmas),
        covariance_epsilon=0.0,
        ray_gaussian_evaluation=True,
        output_srgb=False,
        deterministic=True,
    )
    service = RendererService(
        backend,
        height=fidelity_bundle.height,
        width=fidelity_bundle.width,
        max_views=1,
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
    service.render(
        viewmat.unsqueeze(0).contiguous(),
        intrinsics.unsqueeze(0).contiguous(),
        torch.tensor(
            [camera.scene_id],
            device=device,
            dtype=torch.int64,
        ),
    )
    service.synchronize()
    counters = backend.check_capacity(synchronize=False)
    visible_count = counters["visible_gaussians"]
    visible_ids = backend.workspace["visible_gaussian_ids"][
        :visible_count
    ].to(torch.int64)
    visible_ids = torch.unique(visible_ids, sorted=True)

    means_world = backend._packed_scene["means"].index_select(
        0,
        visible_ids,
    )
    packed_covariances = backend._packed_scene["covariances"].index_select(
        0,
        visible_ids,
    )
    opacities = backend._packed_scene["opacities"].index_select(
        0,
        visible_ids,
    )
    colors = backend._packed_scene["features"].index_select(
        0,
        visible_ids,
    )[:, :3]
    semantic_ids = backend._packed_scene["semantic_ids"].index_select(
        0,
        visible_ids,
    )
    rotation = viewmat[:3, :3]
    translation = viewmat[:3, 3]
    means_camera = torch.einsum(
        "ij,gj->gi",
        rotation,
        means_world,
    ) + translation
    covariances_camera = torch.einsum(
        "ij,gjk,lk->gil",
        rotation,
        covariance_matrices(packed_covariances),
        rotation,
    )
    precisions = torch.linalg.inv(covariances_camera)

    gaussian_order = torch.argsort(visible_ids, stable=True)
    gaussian_order = gaussian_order[
        torch.argsort(
            means_camera[gaussian_order, 2],
            stable=True,
        )
    ]
    means_camera = means_camera.index_select(0, gaussian_order)
    precisions = precisions.index_select(0, gaussian_order)
    opacities = opacities.index_select(0, gaussian_order)
    colors = colors.index_select(0, gaussian_order)
    semantic_ids = semantic_ids.index_select(0, gaussian_order)

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(
            fidelity_bundle.height,
            device=device,
            dtype=torch.float32,
        ) + 0.5,
        torch.arange(
            fidelity_bundle.width,
            device=device,
            dtype=torch.float32,
        ) + 0.5,
        indexing="ij",
    )
    rays = torch.stack(
        (
            (pixel_x.reshape(-1) - intrinsics[0, 2])
            / intrinsics[0, 0],
            (pixel_y.reshape(-1) - intrinsics[1, 2])
            / intrinsics[1, 1],
            torch.ones(
                fidelity_bundle.height * fidelity_bundle.width,
                device=device,
                dtype=torch.float32,
            ),
        ),
        dim=1,
    )
    precision_means = torch.einsum(
        "gij,gj->gi",
        precisions,
        means_camera,
    )
    ray_precision_ray = torch.einsum(
        "pi,gij,pj->gp",
        rays,
        precisions,
        rays,
    )
    ray_precision_mean = torch.einsum(
        "pi,gi->gp",
        rays,
        precision_means,
    )
    mean_precision_mean = torch.einsum(
        "gi,gi->g",
        means_camera,
        precision_means,
    )
    ray_t_z = ray_precision_mean / ray_precision_ray
    mahalanobis_squared = (
        mean_precision_mean[:, None]
        - ray_precision_mean.square() / ray_precision_ray
    ).clamp_min(0.0)
    ray_lengths = torch.linalg.vector_norm(rays, dim=1)
    ray_t_distance = ray_t_z * ray_lengths[None, :]

    rows: list[dict[str, Any]] = []
    outputs_by_config: dict[str, RenderOutput] = {}
    semantic_diagnostics: dict[str, Any] = {}
    candidate_semantic = torch.from_numpy(
        candidate_output.semantic[0].astype(np.int64, copy=False)
    ).to(device=device).reshape(-1)
    for support_sigma in args.support_sigmas:
        splat_alpha = (
            opacities[:, None]
            * torch.exp(-0.5 * mahalanobis_squared)
        )
        splat_alpha = torch.where(
            mahalanobis_squared <= support_sigma * support_sigma,
            splat_alpha,
            torch.zeros_like(splat_alpha),
        )
        transmittance = torch.cumprod(
            1.0 - splat_alpha,
            dim=0,
        )
        exclusive_transmittance = torch.cat(
            (
                torch.ones_like(transmittance[:1]),
                transmittance[:-1],
            ),
            dim=0,
        )
        weights = exclusive_transmittance * splat_alpha
        accumulated_alpha = weights.sum(dim=0)
        best_gaussian_indices = weights.argmax(dim=0)
        individual_semantic = semantic_ids.index_select(
            0,
            best_gaussian_indices,
        )
        individual_semantic = torch.where(
            accumulated_alpha > 0,
            individual_semantic,
            torch.full_like(individual_semantic, -1),
        )
        semantic_count = int(
            backend._packed_scene["semantic_ids"].max().item()
        ) + 1
        semantic_probabilities = torch.zeros(
            (
                semantic_count,
                fidelity_bundle.height * fidelity_bundle.width,
            ),
            device=device,
            dtype=weights.dtype,
        )
        semantic_probabilities.index_add_(
            0,
            semantic_ids,
            weights,
        )
        aggregate_semantic = semantic_probabilities.argmax(dim=0)
        aggregate_semantic = torch.where(
            accumulated_alpha > 0,
            aggregate_semantic,
            torch.full_like(aggregate_semantic, -1),
        )
        semantic_mismatch = aggregate_semantic != candidate_semantic
        both_foreground = (
            (aggregate_semantic >= 0)
            & (candidate_semantic >= 0)
        )
        foreground_mismatch = semantic_mismatch & both_foreground
        foreground_match = (~semantic_mismatch) & both_foreground
        top_probabilities, top_labels = torch.topk(
            semantic_probabilities,
            k=2,
            dim=0,
        )
        valid_candidate_label = (
            (candidate_semantic >= 0)
            & (candidate_semantic < semantic_count)
        )
        candidate_probability = torch.zeros_like(accumulated_alpha)
        pixel_ids = torch.arange(
            candidate_semantic.numel(),
            device=device,
            dtype=torch.int64,
        )
        candidate_probability[valid_candidate_label] = (
            semantic_probabilities[
                candidate_semantic[valid_candidate_label],
                pixel_ids[valid_candidate_label],
            ]
        )
        candidate_rank = (
            semantic_probabilities > candidate_probability[None, :]
        ).sum(dim=0) + 1
        top_probability_delta = (
            top_probabilities[0] - top_probabilities[1]
        )
        top_difference_variance = (
            top_probabilities[0]
            + top_probabilities[1]
            - top_probability_delta.square()
        ).clamp_min(1.0e-20)
        exact_foreground = aggregate_semantic >= 0
        semantic_convergence_prediction = {}
        for prediction_frames in args.semantic_prediction_frames:
            normal_z = (
                top_probability_delta
                * math.sqrt(prediction_frames)
                / torch.sqrt(top_difference_variance)
            )
            label_confusion_probability = (
                0.5 * torch.erfc(normal_z / math.sqrt(2.0))
            )
            no_hit_probability = torch.exp(
                prediction_frames
                * torch.log1p(
                    -accumulated_alpha.clamp(
                        min=0.0,
                        max=1.0 - 1.0e-7,
                    )
                )
            )
            mismatch_probability = torch.where(
                exact_foreground,
                no_hit_probability
                + (1.0 - no_hit_probability)
                * label_confusion_probability,
                torch.zeros_like(accumulated_alpha),
            ).clamp(0.0, 1.0)
            semantic_convergence_prediction[
                str(prediction_frames)
            ] = {
                "expected_mismatch_pixels_normal_approx": float(
                    mismatch_probability.sum().item()
                ),
                "expected_semantic_agreement_normal_approx": float(
                    1.0
                    - mismatch_probability.sum().item()
                    / mismatch_probability.numel()
                ),
                "expected_no_hit_boundary_mismatches": float(
                    torch.where(
                        exact_foreground,
                        no_hit_probability,
                        torch.zeros_like(no_hit_probability),
                    ).sum().item()
                ),
                "pixels_with_mismatch_probability_at_least_0_5": int(
                    torch.count_nonzero(
                        mismatch_probability >= 0.5
                    ).item()
                ),
                "pixels_with_mismatch_probability_at_least_0_1": int(
                    torch.count_nonzero(
                        mismatch_probability >= 0.1
                    ).item()
                ),
                "pixels_with_mismatch_probability_at_least_0_01": int(
                    torch.count_nonzero(
                        mismatch_probability >= 0.01
                    ).item()
                ),
            }
        semantic_diagnostics[f"{support_sigma:g}"] = {
            "mismatch_pixels": int(semantic_mismatch.sum().item()),
            "candidate_background_exact_foreground": int(
                (
                    (candidate_semantic < 0)
                    & (aggregate_semantic >= 0)
                ).sum().item()
            ),
            "candidate_foreground_exact_background": int(
                (
                    (candidate_semantic >= 0)
                    & (aggregate_semantic < 0)
                ).sum().item()
            ),
            "both_foreground_mismatch": int(
                foreground_mismatch.sum().item()
            ),
            "candidate_is_second_best_on_foreground_mismatch": int(
                (
                    foreground_mismatch
                    & (top_labels[1] == candidate_semantic)
                ).sum().item()
            ),
            "candidate_rank_1_on_foreground_mismatch": int(
                (
                    foreground_mismatch
                    & (candidate_rank == 1)
                ).sum().item()
            ),
            "candidate_rank_2_on_foreground_mismatch": int(
                (
                    foreground_mismatch
                    & (candidate_rank == 2)
                ).sum().item()
            ),
            "candidate_rank_3plus_on_foreground_mismatch": int(
                (
                    foreground_mismatch
                    & (candidate_rank >= 3)
                ).sum().item()
            ),
            "top1_minus_top2_probability_on_foreground_mismatch": quantiles(
                top_probability_delta[foreground_mismatch]
            ),
            "top1_minus_top2_probability_on_foreground_match": quantiles(
                top_probability_delta[foreground_match]
            ),
            "expected_top1_minus_candidate_votes_at_16384_frames": quantiles(
                (
                    (
                        top_probabilities[0]
                        - candidate_probability
                    )
                    * 16_384.0
                )[foreground_mismatch]
            ),
            "semantic_convergence_prediction": (
                semantic_convergence_prediction
            ),
        }
        for quantized_color in (False, True):
            evaluated_colors = (
                torch.round(colors.clamp(0.0, 1.0) * 255.0) / 255.0
                if quantized_color
                else colors
            )
            rgb = torch.einsum(
                "gp,gc->pc",
                weights,
                evaluated_colors,
            )
            for semantic_mode, semantic in (
                ("individual_gaussian", individual_semantic),
                ("aggregate_label_probability", aggregate_semantic),
            ):
                for depth_mode, depths in (
                    (
                        "center_z",
                        means_camera[:, 2, None].expand_as(weights),
                    ),
                    ("ray_t_z", ray_t_z),
                    ("ray_distance", ray_t_distance),
                ):
                    depth = (
                        (weights * depths).sum(dim=0)
                        / accumulated_alpha.clamp_min(
                            torch.finfo(weights.dtype).eps
                        )
                    )
                    depth = torch.where(
                        accumulated_alpha > 0,
                        depth,
                        torch.full_like(depth, float("inf")),
                    )
                    arrays = {
                        "rgb": rgb.view(
                            1,
                            fidelity_bundle.height,
                            fidelity_bundle.width,
                            3,
                        ).detach().cpu().numpy(),
                        "alpha": accumulated_alpha.view(
                            1,
                            fidelity_bundle.height,
                            fidelity_bundle.width,
                        ).detach().cpu().numpy(),
                        "depth": depth.view(
                            1,
                            fidelity_bundle.height,
                            fidelity_bundle.width,
                        ).detach().cpu().numpy(),
                        "semantic": semantic.view(
                            1,
                            fidelity_bundle.height,
                            fidelity_bundle.width,
                        ).detach().cpu().numpy(),
                    }
                    config_id = (
                        f"exact-ray-sigma-{support_sigma:g}-"
                        f"{'quantized' if quantized_color else 'float'}-"
                        f"{semantic_mode}-{depth_mode}"
                    )
                    output = render_output_from_arrays(
                        arrays,
                        camera_bundle_id=fidelity_bundle.bundle_id,
                        source=config_id,
                    )
                    row_metrics = metrics(
                        custom=simple_arrays(output),
                        candidate=candidate,
                    )
                    rows.append(
                        {
                            "config_id": config_id,
                            "support_sigma": support_sigma,
                            "quantized_gaussian_color": quantized_color,
                            "semantic_mode": semantic_mode,
                            "depth_mode": depth_mode,
                            "metrics": row_metrics,
                        }
                    )
                    outputs_by_config[config_id] = output

    rows.sort(key=lambda row: row["metrics"]["threshold_distance"])
    best = rows[0]
    best_output = outputs_by_config[str(best["config_id"])]
    if args.best_render_output is not None:
        save_render_output(args.best_render_output, best_output)
    full_report = compare_render_outputs(
        reference=best_output,
        candidate=candidate_output,
        camera_bundle=fidelity_bundle,
        output_dir=args.report_dir,
        config_id=str(best["config_id"]),
        require_lpips=True,
        max_artifact_views=1,
    )
    result = {
        "schema_version": "ovrtx-exact-ray-model/v1",
        "diagnostic_only": True,
        "candidate": str(args.candidate),
        "camera_bundle": str(args.camera_bundle),
        "scene": scene_manifest,
        "visible_gaussians": int(visible_ids.numel()),
        "projection_counters": counters,
        "model": (
            "alpha_i=opacity_i*exp(-0.5*minimum_ray_mahalanobis_squared); "
            "front-to-back Bernoulli first-hit probabilities"
        ),
        "semantic_diagnostics": semantic_diagnostics,
        "row_count": len(rows),
        "best_row": best,
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
