"""Predict OVRTX semantic convergence from sparse custom intersections."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmarks.run_ovrtx import (
    PLY_SCENES,
    SYNTHETIC_SCENES,
    load_scene_and_cameras,
)
from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.fidelity import load_render_output


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
        "--scene",
        choices=sorted((*SYNTHETIC_SCENES, *PLY_SCENES)),
        default="synthetic-small",
    )
    parser.add_argument(
        "--home-scan-path",
        type=Path,
        default=Path("/workspace/datasets/home-scan-lod0.ply"),
    )
    parser.add_argument(
        "--public-scene-path",
        type=Path,
        default=Path(
            "/workspace/datasets/public-gaussian/"
            "Voxel51_gaussian_splatting/FO_dataset/train/"
            "point_cloud/iteration_30000/point_cloud.ply"
        ),
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--focal-scale", type=float, default=0.9)
    parser.add_argument("--home-scan-fit-margin", type=float, default=1.15)
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument(
        "--model",
        choices=("screen", "ray"),
        default="screen",
    )
    parser.add_argument("--semantic-classes", type=int, default=1024)
    parser.add_argument("--visible-capacity", type=int, default=None)
    parser.add_argument("--intersection-capacity", type=int, default=500_000)
    parser.add_argument(
        "--prediction-frames",
        type=positive_int_list,
        default=positive_int_list(
            "65536,262144,1048576,4194304"
        ),
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Optional temporal OVRTX render output to compare with the exact model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normal_loss_probability(
    *,
    target_probability: np.ndarray,
    competitor_probability: np.ndarray,
    accumulated_alpha: np.ndarray,
    frames: int,
) -> np.ndarray:
    delta = target_probability - competitor_probability
    variance = (
        target_probability
        + competitor_probability
        - delta * delta
    )
    variance = np.maximum(variance, 1.0e-20)
    z = delta * math.sqrt(frames) / np.sqrt(variance)
    confusion = 0.5 * np.fromiter(
        (math.erfc(float(value) / math.sqrt(2.0)) for value in z),
        dtype=np.float64,
        count=z.size,
    )
    no_hit = np.exp(
        frames
        * np.log1p(
            -np.clip(accumulated_alpha, 0.0, 1.0 - 1.0e-7)
        )
    )
    return np.clip(
        no_hit + (1.0 - no_hit) * confusion,
        0.0,
        1.0,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.focal_scale <= 0
        or args.home_scan_fit_margin <= 0
        or args.support_sigma <= 0
        or args.semantic_classes <= 1
        or args.intersection_capacity <= 0
        or (
            args.visible_capacity is not None
            and args.visible_capacity <= 0
        )
    ):
        raise ValueError("Geometry, capacities, and class count must be positive.")

    device = torch.device("cuda:0")
    args.batch_size = 1
    scene, scene_manifest, cameras, scene_loading = (
        load_scene_and_cameras(args)
    )
    gaussian_count = int(scene["means"].shape[0])
    visible_capacity = (
        args.visible_capacity
        if args.visible_capacity is not None
        else (
            gaussian_count
            if args.scene in PLY_SCENES
            else min(gaussian_count, 5_000)
        )
    )
    backend = CustomCudaBackend(
        max_visible_records=visible_capacity,
        max_intersections=args.intersection_capacity,
        near_plane=float(cameras.manifest["near_plane"]),
        far_plane=float(cameras.manifest["far_plane"]),
        gaussian_support_sigma=args.support_sigma,
        covariance_epsilon=0.0,
        ray_gaussian_evaluation=args.model == "ray",
        tile_size=1,
        depth_bucket_count=1024,
        depth_bucket_group_size=32,
        output_srgb=False,
        deterministic=False,
    )
    service = RendererService(
        backend,
        height=args.height,
        width=args.width,
        max_views=1,
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
    rendered = service.render(
        cameras.viewmats,
        cameras.intrinsics,
        torch.tensor([101], device=device, dtype=torch.int64),
    )
    service.synchronize()
    counters = backend.check_capacity(synchronize=False)
    intersection_count = int(counters["tile_intersections"])
    if intersection_count <= 0:
        raise RuntimeError("The exact renderer produced no intersections.")

    starts = (
        backend.workspace["tile_starts"][: args.height * args.width]
        .detach()
        .cpu()
        .numpy()
    )
    ends = (
        backend.workspace["tile_ends"][: args.height * args.width]
        .detach()
        .cpu()
        .numpy()
    )
    depth_cutoff = (
        backend.workspace["depth_cutoff"][: args.height * args.width]
        .detach()
        .cpu()
        .numpy()
    )
    lengths = (ends - starts).astype(np.int64, copy=False)
    nonempty = np.flatnonzero(lengths > 0)
    expected_starts = np.cumsum(lengths, dtype=np.int64) - lengths
    if (
        int(lengths.sum()) != intersection_count
        or nonempty.size == 0
        or int(starts[nonempty[0]]) != 0
        or not np.array_equal(
            starts[nonempty],
            expected_starts[nonempty],
        )
    ):
        raise RuntimeError(
            "Sorted tile ranges do not cover the measured intersections."
        )
    pixel_ids_cpu = np.repeat(
        np.arange(args.height * args.width, dtype=np.int64),
        lengths,
    )
    pixel_ids = torch.from_numpy(pixel_ids_cpu).to(device=device)
    visible_indices = backend.workspace["values_out"][
        :intersection_count
    ].to(torch.int64)
    opacities = backend.workspace["visible_opacities"].index_select(
        0,
        visible_indices,
    )
    gaussian_ids = backend.workspace["visible_gaussian_ids"].index_select(
        0,
        visible_indices,
    ).to(torch.int64)
    semantic_ids = backend._packed_scene["semantic_ids"].index_select(
        0,
        gaussian_ids,
    )

    pixel_x = (
        pixel_ids.remainder(args.width).to(torch.float32) + 0.5
    )
    pixel_y = (
        torch.div(
            pixel_ids,
            args.width,
            rounding_mode="floor",
        ).to(torch.float32)
        + 0.5
    )
    if args.model == "ray":
        precisions = backend.workspace[
            "visible_ray_precisions"
        ].index_select(0, visible_indices)
        precision_means = backend.workspace[
            "visible_ray_precision_means"
        ].index_select(0, visible_indices)
        intrinsics = cameras.intrinsics[0]
        ray_x = (pixel_x - intrinsics[0, 2]) / intrinsics[0, 0]
        ray_y = (pixel_y - intrinsics[1, 2]) / intrinsics[1, 1]
        ray_precision_ray = (
            precisions[:, 0] * ray_x.square()
            + 2.0 * precisions[:, 1] * ray_x * ray_y
            + 2.0 * precisions[:, 2] * ray_x
            + precisions[:, 3] * ray_y.square()
            + 2.0 * precisions[:, 4] * ray_y
            + precisions[:, 5]
        )
        ray_precision_mean = (
            precision_means[:, 0] * ray_x
            + precision_means[:, 1] * ray_y
            + precision_means[:, 2]
        )
        mahalanobis_squared = (
            precision_means[:, 3]
            - ray_precision_mean.square() / ray_precision_ray
        ).clamp_min(0.0)
        splat_alpha = opacities * torch.exp(
            -0.5 * mahalanobis_squared
        )
        splat_alpha = torch.where(
            mahalanobis_squared
            <= args.support_sigma * args.support_sigma,
            splat_alpha,
            torch.zeros_like(splat_alpha),
        )
    else:
        means2d = backend.workspace["visible_means2d"].index_select(
            0,
            visible_indices,
        )
        conics = backend.workspace["visible_conics"].index_select(
            0,
            visible_indices,
        )
        delta_x = pixel_x - means2d[:, 0]
        delta_y = pixel_y - means2d[:, 1]
        power = -0.5 * (
            conics[:, 0] * delta_x.square()
            + 2.0 * conics[:, 1] * delta_x * delta_y
            + conics[:, 2] * delta_y.square()
        )
        splat_alpha = torch.minimum(
            torch.full_like(opacities, 0.99),
            opacities * torch.exp(power),
        )
        splat_alpha = torch.where(
            (power <= 0.0)
            & (power >= -20.0)
            & (splat_alpha >= 1.0 / 255.0),
            splat_alpha,
            torch.zeros_like(splat_alpha),
        )
    alpha_cpu = splat_alpha.detach().cpu().numpy()
    semantic_cpu = semantic_ids.detach().cpu().numpy()
    if (
        int(np.min(semantic_cpu)) < 0
        or int(np.max(semantic_cpu)) >= args.semantic_classes
    ):
        raise RuntimeError(
            "Intersection semantic IDs exceed the configured class range."
        )
    native_semantic = (
        rendered["semantic_id"][0, ..., 0]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    pixel_count = args.height * args.width
    individual_semantic = np.full(pixel_count, -1, dtype=np.int64)
    aggregate_semantic = np.full(pixel_count, -1, dtype=np.int64)
    first_semantic = np.full(pixel_count, -1, dtype=np.int64)
    last_semantic = np.full(pixel_count, -1, dtype=np.int64)
    maximum_alpha_semantic = np.full(
        pixel_count,
        -1,
        dtype=np.int64,
    )
    median_opacity_semantic = np.full(
        pixel_count,
        -1,
        dtype=np.int64,
    )
    aggregate_top_probability = np.zeros(pixel_count, dtype=np.float64)
    aggregate_second_probability = np.zeros(pixel_count, dtype=np.float64)
    native_probability = np.zeros(pixel_count, dtype=np.float64)
    native_competitor_probability = np.zeros(pixel_count, dtype=np.float64)
    accumulated_alpha = np.zeros(pixel_count, dtype=np.float64)
    omitted_probability_upper_bound = np.zeros(
        pixel_count,
        dtype=np.float64,
    )

    for pixel_id, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if end <= start:
            continue
        transmittance = np.float32(1.0)
        best_weight = np.float32(-1.0)
        best_label = -1
        best_alpha = np.float32(-1.0)
        best_alpha_label = -1
        probabilities: dict[int, float] = {}
        compositor_truncated = False
        for index in range(int(start), int(end)):
            alpha = np.float32(alpha_cpu[index])
            if not alpha > 0:
                continue
            weight = np.float32(transmittance * alpha)
            label = int(semantic_cpu[index])
            if first_semantic[pixel_id] < 0:
                first_semantic[pixel_id] = label
            last_semantic[pixel_id] = label
            probabilities[label] = probabilities.get(label, 0.0) + float(
                weight
            )
            if alpha > best_alpha:
                best_alpha = alpha
                best_alpha_label = label
            if weight > best_weight:
                best_weight = weight
                best_label = label
            transmittance = np.float32(
                transmittance * np.float32(1.0 - alpha)
            )
            if (
                median_opacity_semantic[pixel_id] < 0
                and transmittance <= np.float32(0.5)
            ):
                median_opacity_semantic[pixel_id] = label
            if transmittance <= np.float32(1.0e-4):
                compositor_truncated = index + 1 < int(end)
                break
        individual_semantic[pixel_id] = best_label
        maximum_alpha_semantic[pixel_id] = best_alpha_label
        accumulated_alpha[pixel_id] = 1.0 - float(transmittance)
        ordered = sorted(
            probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        )
        aggregate_semantic[pixel_id] = ordered[0][0]
        aggregate_top_probability[pixel_id] = ordered[0][1]
        if len(ordered) > 1:
            aggregate_second_probability[pixel_id] = ordered[1][1]
        native_label = int(native_semantic[pixel_id])
        native_probability[pixel_id] = probabilities.get(
            native_label,
            0.0,
        )
        native_competitor_probability[pixel_id] = max(
            (
                probability
                for label, probability in probabilities.items()
                if label != native_label
            ),
            default=0.0,
        )
        depth_cutoff_truncated = int(depth_cutoff[pixel_id]) < 1023
        if compositor_truncated or depth_cutoff_truncated:
            omitted_probability_upper_bound[pixel_id] = float(
                transmittance
            )

    candidate_semantic = None
    if args.candidate is not None:
        candidate = load_render_output(args.candidate)
        if (
            candidate.semantic.shape
            != (1, args.height, args.width)
        ):
            raise ValueError(
                "Candidate semantic tensor does not match the requested view."
            )
        candidate_semantic = candidate.semantic.reshape(-1)

    foreground = aggregate_semantic >= 0
    aggregate_margin = (
        aggregate_top_probability - aggregate_second_probability
    )
    retained_top_certified_against_omitted_tail = (
        foreground
        & (aggregate_margin > omitted_probability_upper_bound)
    )
    native_retained_gap = (
        aggregate_top_probability - native_probability
    )
    certified_native_full_aggregate_mismatch = (
        foreground
        & (native_semantic != aggregate_semantic)
        & (native_retained_gap > omitted_probability_upper_bound)
    )
    certified_native_full_aggregate_match = (
        foreground
        & (native_semantic == aggregate_semantic)
        & retained_top_certified_against_omitted_tail
    )
    predictions = {}
    for frames in args.prediction_frames:
        mismatch_probability = normal_loss_probability(
            target_probability=native_probability,
            competitor_probability=native_competitor_probability,
            accumulated_alpha=accumulated_alpha,
            frames=frames,
        )
        mismatch_probability[~foreground] = 0.0
        predictions[str(frames)] = {
            "expected_mismatch_pixels_normal_approx": float(
                mismatch_probability.sum()
            ),
            "expected_semantic_agreement_normal_approx": float(
                1.0 - mismatch_probability.mean()
            ),
            "pixels_with_mismatch_probability_at_least_0_5": int(
                np.count_nonzero(mismatch_probability >= 0.5)
            ),
            "pixels_with_mismatch_probability_at_least_0_1": int(
                np.count_nonzero(mismatch_probability >= 0.1)
            ),
            "pixels_with_mismatch_probability_at_least_0_01": int(
                np.count_nonzero(mismatch_probability >= 0.01)
            ),
        }

    result = {
        "schema_version": "ovrtx-sparse-semantic-model/v1",
        "diagnostic_only": True,
        "scene_id": args.scene,
        "scene": scene_manifest,
        "scene_loading": scene_loading,
        "camera_manifest": cameras.manifest,
        "gaussian_count": gaussian_count,
        "support_sigma": args.support_sigma,
        "model": args.model,
        "semantic_classes": args.semantic_classes,
        "projection_counters": counters,
        "foreground_pixels": int(np.count_nonzero(foreground)),
        "native_vs_individual_mismatch_pixels": int(
            np.count_nonzero(native_semantic != individual_semantic)
        ),
        "individual_vs_aggregate_mismatch_pixels": int(
            np.count_nonzero(
                individual_semantic != aggregate_semantic
            )
        ),
        "native_vs_aggregate_mismatch_pixels": int(
            np.count_nonzero(native_semantic != aggregate_semantic)
        ),
        "native_vs_aggregate_semantic_agreement": float(
            np.mean(native_semantic == aggregate_semantic)
        ),
        "native_mismatch_partition": {
            "native_differs_individual_but_individual_matches_aggregate": int(
                np.count_nonzero(
                    (native_semantic != individual_semantic)
                    & (individual_semantic == aggregate_semantic)
                )
            ),
            "native_matches_individual_but_differs_aggregate": int(
                np.count_nonzero(
                    (native_semantic == individual_semantic)
                    & (individual_semantic != aggregate_semantic)
                )
            ),
            "all_three_labels_differ": int(
                np.count_nonzero(
                    (native_semantic != individual_semantic)
                    & (native_semantic != aggregate_semantic)
                    & (individual_semantic != aggregate_semantic)
                )
            ),
        },
        "retained_probability_model": {
            "depth_cutoff_transmittance_threshold": 1.0e-4,
            "maximum_omitted_probability_upper_bound": float(
                np.max(omitted_probability_upper_bound)
            ),
            "retained_top_certified_pixel_count": int(
                np.count_nonzero(
                    retained_top_certified_against_omitted_tail
                )
            ),
            "certified_native_full_aggregate_match_pixels": int(
                np.count_nonzero(
                    certified_native_full_aggregate_match
                )
            ),
            "certified_native_full_aggregate_mismatch_pixels": int(
                np.count_nonzero(
                    certified_native_full_aggregate_mismatch
                )
            ),
            "foreground_pixels_not_certified_against_omitted_tail": int(
                np.count_nonzero(
                    foreground
                    & ~retained_top_certified_against_omitted_tail
                )
            ),
        },
        "candidate": str(args.candidate) if args.candidate else None,
        "candidate_vs_aggregate_mismatch_pixels": (
            int(
                np.count_nonzero(
                    candidate_semantic != aggregate_semantic
                )
            )
            if candidate_semantic is not None
            else None
        ),
        "candidate_vs_aggregate_semantic_agreement": (
            float(np.mean(candidate_semantic == aggregate_semantic))
            if candidate_semantic is not None
            else None
        ),
        "candidate_rule_comparison": (
            {
                name: {
                    "mismatch_pixels": int(
                        np.count_nonzero(candidate_semantic != semantic)
                    ),
                    "semantic_agreement": float(
                        np.mean(candidate_semantic == semantic)
                    ),
                }
                for name, semantic in (
                    ("maximum_weight", individual_semantic),
                    ("aggregate_label_weight", aggregate_semantic),
                    ("first_contributor", first_semantic),
                    ("last_contributor", last_semantic),
                    ("maximum_alpha", maximum_alpha_semantic),
                    (
                        "median_opacity_crossing",
                        median_opacity_semantic,
                    ),
                )
            }
            if candidate_semantic is not None
            else None
        ),
        "native_target_convergence_prediction": predictions,
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
