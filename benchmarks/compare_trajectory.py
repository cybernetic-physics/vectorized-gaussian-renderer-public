#!/usr/bin/env python3
"""Compare complete renderer trajectories with spatial and temporal metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaacsim_gaussian_renderer.evaluation.trajectory_contract import load_trajectory
from isaacsim_gaussian_renderer.fidelity.outputs import RenderOutput, load_render_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--reference-equation", required=True)
    parser.add_argument("--candidate-equation", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opacity-thresholds", default="0.01,0.1,0.5,0.9")
    parser.add_argument("--reprojection-stride", type=int, default=4)
    parser.add_argument("--min-psnr-db", type=float, default=40.0)
    parser.add_argument("--min-ssim", type=float, default=0.995)
    parser.add_argument("--max-lpips", type=float, default=0.01)
    parser.add_argument("--max-alpha-mae", type=float, default=0.005)
    parser.add_argument("--max-depth-relative-error", type=float, default=0.01)
    parser.add_argument("--min-semantic-agreement", type=float, default=0.999)
    parser.add_argument(
        "--ovrtx-convergence",
        action="append",
        default=[],
        metavar="SAMPLES=OUTPUT",
        help="Optional OVRTX temporal accumulation outputs at 1,4,... samples.",
    )
    return parser.parse_args()


def _equation_family(label: str) -> str:
    normalized = label.lower()
    if "ewa" in normalized or "screen-space" in normalized:
        return "deterministic-screen-space-ewa"
    if "ovrtx" in normalized or "perspective" in normalized or "3d-ray" in normalized:
        return "stochastic-perspective-3d-ray"
    return normalized


def _scalar(array: np.ndarray) -> np.ndarray:
    return array[..., 0] if array.ndim >= 4 and array.shape[-1] == 1 else array


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((candidate.astype(np.float64) - reference.astype(np.float64)) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def _ssim_global(reference: np.ndarray, candidate: np.ndarray) -> float:
    values = []
    for channel in range(3):
        x = reference[..., channel].astype(np.float64)
        y = candidate[..., channel].astype(np.float64)
        mux, muy = float(x.mean()), float(y.mean())
        varx, vary = float(x.var()), float(y.var())
        cov = float(np.mean((x - mux) * (y - muy)))
        values.append(
            ((2 * mux * muy + 0.01**2) * (2 * cov + 0.03**2))
            / ((mux**2 + muy**2 + 0.01**2) * (varx + vary + 0.03**2))
        )
    return float(np.mean(values))


def _boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[..., 1:, :] |= mask[..., 1:, :] != mask[..., :-1, :]
    boundary[..., :-1, :] |= mask[..., 1:, :] != mask[..., :-1, :]
    boundary[..., :, 1:] |= mask[..., :, 1:] != mask[..., :, :-1]
    boundary[..., :, :-1] |= mask[..., :, 1:] != mask[..., :, :-1]
    return boundary


def _f1(reference: np.ndarray, candidate: np.ndarray) -> float:
    true_positive = int(np.count_nonzero(reference & candidate))
    false_positive = int(np.count_nonzero(~reference & candidate))
    false_negative = int(np.count_nonzero(reference & ~candidate))
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2.0 * true_positive / denominator


@lru_cache(maxsize=2)
def _lpips_model(device_name: str):
    import lpips

    return lpips.LPIPS(net="alex").to(device_name).eval()


def _lpips(reference: np.ndarray, candidate: np.ndarray) -> tuple[float | None, str | None]:
    if np.array_equal(reference, candidate):
        return 0.0, None
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _lpips_model(str(device))
        ref = torch.from_numpy(reference.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        cand = torch.from_numpy(candidate.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            return float(model(ref * 2 - 1, cand * 2 - 1).item()), None
    except Exception as exc:
        return None, str(exc)


def frame_metrics(
    reference: RenderOutput,
    candidate: RenderOutput,
    index: int,
    thresholds: list[float],
) -> dict[str, Any]:
    ref_rgb, cand_rgb = reference.rgb[index], candidate.rgb[index]
    ref_alpha, cand_alpha = _scalar(reference.alpha)[index], _scalar(candidate.alpha)[index]
    ref_depth, cand_depth = _scalar(reference.depth)[index], _scalar(candidate.depth)[index]
    ref_sem, cand_sem = _scalar(reference.semantic)[index], _scalar(candidate.semantic)[index]
    valid_depth = np.isfinite(ref_depth) & np.isfinite(cand_depth) & (ref_alpha > 0.01) & (ref_depth > 0)
    depth_relative = (
        np.abs(cand_depth[valid_depth] - ref_depth[valid_depth]) / np.maximum(np.abs(ref_depth[valid_depth]), 1e-8)
    )
    ref_foreground, cand_foreground = ref_alpha > 0.01, cand_alpha > 0.01
    union = np.count_nonzero(ref_foreground | cand_foreground)
    foreground_iou = (
        1.0 if union == 0 else np.count_nonzero(ref_foreground & cand_foreground) / union
    )
    luma = 0.2126 * ref_rgb[..., 0] + 0.7152 * ref_rgb[..., 1] + 0.0722 * ref_rgb[..., 2]
    grad_y, grad_x = np.gradient(luma.astype(np.float64))
    weights = 1.0 + np.hypot(grad_x, grad_y)
    rgb_abs = np.mean(np.abs(cand_rgb.astype(np.float64) - ref_rgb.astype(np.float64)), axis=-1)
    epsilon = 1.0e-6
    ref_tau = -np.log(np.maximum(1.0 - np.clip(ref_alpha, 0, 1), epsilon))
    cand_tau = -np.log(np.maximum(1.0 - np.clip(cand_alpha, 0, 1), epsilon))
    lpips_value, lpips_error = _lpips(ref_rgb, cand_rgb)
    pairs = Counter(zip(ref_sem.reshape(-1).tolist(), cand_sem.reshape(-1).tolist(), strict=True))
    confusion = [
        {"reference": int(pair[0]), "candidate": int(pair[1]), "count": int(count)}
        for pair, count in pairs.most_common(32)
        if pair[0] != pair[1]
    ]
    return {
        "frame_index": index,
        "rgb_psnr_db": _psnr(ref_rgb, cand_rgb),
        "rgb_ssim": _ssim_global(ref_rgb, cand_rgb),
        "lpips": lpips_value,
        "lpips_error": lpips_error,
        "alpha_mae": float(np.mean(np.abs(cand_alpha - ref_alpha))),
        "alpha_boundary_f1": {
            str(threshold): _f1(_boundary(ref_alpha >= threshold), _boundary(cand_alpha >= threshold))
            for threshold in thresholds
        },
        "foreground_iou": float(foreground_iou),
        "depth_relative_error_median": float(np.median(depth_relative)) if depth_relative.size else None,
        "depth_relative_error_mean": float(np.mean(depth_relative)) if depth_relative.size else None,
        "depth_relative_error_p95": float(np.percentile(depth_relative, 95)) if depth_relative.size else None,
        "valid_depth_pixels": int(depth_relative.size),
        "edge_weighted_rgb_error": float(np.sum(rgb_abs * weights) / np.sum(weights)),
        "optical_depth_mae": float(np.mean(np.abs(cand_tau - ref_tau))),
        "semantic_agreement": float(np.mean(ref_sem == cand_sem)),
        "semantic_confusion_counts": confusion,
    }


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "worst": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "worst": float(array.max()),
    }


def temporal_metrics(output: RenderOutput, *, timesteps: int, batch: int) -> dict[str, np.ndarray]:
    rgb = output.rgb.reshape(timesteps, batch, *output.rgb.shape[1:])
    alpha = _scalar(output.alpha).reshape(timesteps, batch, *output.alpha.shape[1:])
    semantic = _scalar(output.semantic).reshape(timesteps, batch, *output.semantic.shape[1:])
    return {
        "rgb_difference": np.diff(rgb, axis=0),
        "alpha_difference": np.diff(alpha, axis=0),
        "semantic_flip": semantic[1:] != semantic[:-1],
        "static_rgb_variance": np.var(rgb, axis=0),
        "static_alpha_variance": np.var(alpha, axis=0),
    }


def compare_temporal(reference: RenderOutput, candidate: RenderOutput, *, timesteps: int, batch: int) -> dict[str, float]:
    ref = temporal_metrics(reference, timesteps=timesteps, batch=batch)
    cand = temporal_metrics(candidate, timesteps=timesteps, batch=batch)
    return {
        "difference_of_frame_differences_rgb_mae": float(np.mean(np.abs(cand["rgb_difference"] - ref["rgb_difference"]))),
        "difference_of_frame_differences_alpha_mae": float(np.mean(np.abs(cand["alpha_difference"] - ref["alpha_difference"]))),
        "static_camera_rgb_variance_mean": float(np.mean(cand["static_rgb_variance"])),
        "static_camera_alpha_variance_mean": float(np.mean(cand["static_alpha_variance"])),
        "reprojected_semantic_flip_rate_diagnostic": float(np.mean(cand["semantic_flip"] != ref["semantic_flip"])),
    }


def depth_reprojection_consistency(
    reference: RenderOutput,
    candidate: RenderOutput,
    trajectory,
    *,
    stride: int,
) -> dict[str, float | int | None]:
    if stride <= 0:
        raise ValueError("reprojection stride must be positive.")
    timesteps, batch = trajectory.timesteps, trajectory.batch
    height, width = trajectory.height, trajectory.width
    ref_rgb = reference.rgb.reshape(timesteps, batch, height, width, 3)
    cand_rgb = candidate.rgb.reshape(timesteps, batch, height, width, 3)
    ref_depth = _scalar(reference.depth).reshape(timesteps, batch, height, width)
    ref_alpha = _scalar(reference.alpha).reshape(timesteps, batch, height, width)
    rgb_reference_errors: list[float] = []
    rgb_candidate_errors: list[float] = []
    depth_errors: list[float] = []
    sample_count = 0
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    pixels = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
    for time_index in range(timesteps - 1):
        for batch_index in range(batch):
            depth = ref_depth[time_index, batch_index, ::stride, ::stride].reshape(-1)
            alpha = ref_alpha[time_index, batch_index, ::stride, ::stride].reshape(-1)
            valid = np.isfinite(depth) & (depth > trajectory.near_plane) & (alpha > 0.01)
            if not np.any(valid):
                continue
            uv = pixels[valid]
            z = depth[valid]
            intrinsics = trajectory.intrinsics[time_index, batch_index].astype(np.float64)
            x_camera = (uv[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0] * z
            y_camera = (uv[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1] * z
            homogeneous = np.stack((x_camera, y_camera, z, np.ones_like(z)), axis=-1)
            camera_to_world = np.linalg.inv(trajectory.viewmats[time_index, batch_index].astype(np.float64))
            world = homogeneous @ camera_to_world.T
            next_camera = world @ trajectory.viewmats[time_index + 1, batch_index].astype(np.float64).T
            next_z = next_camera[:, 2]
            next_intrinsics = trajectory.intrinsics[time_index + 1, batch_index].astype(np.float64)
            next_u = np.rint(next_intrinsics[0, 0] * next_camera[:, 0] / next_z + next_intrinsics[0, 2]).astype(np.int64)
            next_v = np.rint(next_intrinsics[1, 1] * next_camera[:, 1] / next_z + next_intrinsics[1, 2]).astype(np.int64)
            visible = (
                np.isfinite(next_z)
                & (next_z > trajectory.near_plane)
                & (next_u >= 0)
                & (next_u < width)
                & (next_v >= 0)
                & (next_v < height)
            )
            if not np.any(visible):
                continue
            uv = uv[visible]
            next_u, next_v, next_z = next_u[visible], next_v[visible], next_z[visible]
            ref_current = ref_rgb[time_index, batch_index, uv[:, 1], uv[:, 0]]
            ref_next = ref_rgb[time_index + 1, batch_index, next_v, next_u]
            cand_current = cand_rgb[time_index, batch_index, uv[:, 1], uv[:, 0]]
            cand_next = cand_rgb[time_index + 1, batch_index, next_v, next_u]
            next_depth = ref_depth[time_index + 1, batch_index, next_v, next_u]
            rgb_reference_errors.extend(np.mean(np.abs(ref_next - ref_current), axis=-1).tolist())
            rgb_candidate_errors.extend(np.mean(np.abs(cand_next - cand_current), axis=-1).tolist())
            valid_next_depth = np.isfinite(next_depth) & (next_depth > trajectory.near_plane)
            depth_errors.extend(
                (
                    np.abs(next_depth[valid_next_depth] - next_z[valid_next_depth])
                    / np.maximum(np.abs(next_z[valid_next_depth]), 1.0e-8)
                ).tolist()
            )
            sample_count += int(np.count_nonzero(visible))
    return {
        "sample_stride": stride,
        "sample_count": sample_count,
        "reference_rgb_reprojection_mae": float(np.mean(rgb_reference_errors)) if rgb_reference_errors else None,
        "candidate_rgb_reprojection_mae": float(np.mean(rgb_candidate_errors)) if rgb_candidate_errors else None,
        "candidate_minus_reference_rgb_reprojection_mae": (
            float(np.mean(rgb_candidate_errors) - np.mean(rgb_reference_errors))
            if rgb_candidate_errors and rgb_reference_errors
            else None
        ),
        "reference_depth_reprojection_relative_error_median": float(np.median(depth_errors)) if depth_errors else None,
        "reference_depth_reprojection_relative_error_p95": float(np.percentile(depth_errors, 95)) if depth_errors else None,
    }


def _load_ovrtx_curve(items: list[str], candidate: RenderOutput, trajectory) -> list[dict[str, Any]]:
    curve = []
    for item in items:
        sample_text, separator, output_path = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid --ovrtx-convergence value {item!r}.")
        samples = int(sample_text)
        output = load_render_output(output_path)
        per_frame = [frame_metrics(output, candidate, index, [0.01, 0.5]) for index in range(output.rgb.shape[0])]
        curve.append(
            {
                "samples": samples,
                "rgb_psnr_db_p50": float(np.percentile([row["rgb_psnr_db"] for row in per_frame], 50)),
                "alpha_mae_p50": float(np.percentile([row["alpha_mae"] for row in per_frame], 50)),
                "semantic_agreement_p50": float(np.percentile([row["semantic_agreement"] for row in per_frame], 50)),
            }
        )
    return sorted(curve, key=lambda row: row["samples"])


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.opacity_thresholds.split(",")]
    trajectory = load_trajectory(args.trajectory)
    reference = load_render_output(args.reference)
    candidate = load_render_output(args.candidate)
    expected_views = trajectory.timesteps * trajectory.batch
    reference.validate(expected_views=expected_views, height=trajectory.height, width=trajectory.width)
    candidate.validate(expected_views=expected_views, height=trajectory.height, width=trajectory.width)
    rows = [frame_metrics(reference, candidate, index, thresholds) for index in range(expected_views)]
    lpips_missing = any(row["lpips"] is None for row in rows if not np.array_equal(reference.rgb[row["frame_index"]], candidate.rgb[row["frame_index"]]))
    temporal = compare_temporal(reference, candidate, timesteps=trajectory.timesteps, batch=trajectory.batch)
    temporal["depth_based_reprojection"] = depth_reprojection_consistency(
        reference,
        candidate,
        trajectory,
        stride=args.reprojection_stride,
    )
    distributions = {
        "rgb_psnr_db": _distribution([-row["rgb_psnr_db"] for row in rows]),
        "alpha_mae": _distribution([row["alpha_mae"] for row in rows]),
        "depth_relative_error_p95": _distribution(
            [row["depth_relative_error_p95"] for row in rows if row["depth_relative_error_p95"] is not None]
        ),
        "semantic_disagreement": _distribution([1.0 - row["semantic_agreement"] for row in rows]),
    }
    distributions["rgb_psnr_db"] = {
        "p50": float(np.percentile([row["rgb_psnr_db"] for row in rows], 50)),
        "p95": float(np.percentile([row["rgb_psnr_db"] for row in rows], 5)),
        "worst": float(min(row["rgb_psnr_db"] for row in rows)),
    }
    acceptance_thresholds = {
        "min_psnr_db": args.min_psnr_db,
        "min_ssim": args.min_ssim,
        "max_lpips": args.max_lpips,
        "max_alpha_mae": args.max_alpha_mae,
        "max_depth_relative_error": args.max_depth_relative_error,
        "min_semantic_agreement": args.min_semantic_agreement,
    }
    for row in rows:
        row["acceptance_pass"] = bool(
            row["rgb_psnr_db"] >= args.min_psnr_db
            and row["rgb_ssim"] >= args.min_ssim
            and row["lpips"] is not None
            and row["lpips"] <= args.max_lpips
            and row["alpha_mae"] <= args.max_alpha_mae
            and row["depth_relative_error_mean"] is not None
            and row["depth_relative_error_mean"] <= args.max_depth_relative_error
            and row["semantic_agreement"] >= args.min_semantic_agreement
        )
    failed_frames = [int(row["frame_index"]) for row in rows if not row["acceptance_pass"]]
    reference_family = _equation_family(args.reference_equation)
    candidate_family = _equation_family(args.candidate_equation)
    summary = {
        "schema_version": "trajectory-fidelity-comparison-v1",
        "trajectory_id": trajectory.trajectory_id,
        "rendering_equations": {
            "reference": args.reference_equation,
            "candidate": args.candidate_equation,
            "reference_family": reference_family,
            "candidate_family": candidate_family,
            "same_equation": reference_family == candidate_family,
        },
        "per_frame_distributions": distributions,
        "temporal": temporal,
        "strict_semantic_agreement": float(np.mean(reference.semantic == candidate.semantic)),
        "tie_aware_semantics": {
            "status": "diagnostic-only",
            "available": False,
            "reason": "no probability/tie-margin tensors supplied",
        },
        "lpips_required_and_available": not lpips_missing,
        "acceptance": {
            "thresholds": acceptance_thresholds,
            "every_frame_required": True,
            "failed_frame_count": len(failed_frames),
            "failed_frames": failed_frames,
        },
        "ovrtx_convergence": _load_ovrtx_curve(args.ovrtx_convergence, candidate, trajectory),
        "pass": not lpips_missing and not failed_frames,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per-frame.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "frame_index",
            "rgb_psnr_db",
            "rgb_ssim",
            "lpips",
            "alpha_mae",
            "foreground_iou",
            "depth_relative_error_median",
            "depth_relative_error_mean",
            "depth_relative_error_p95",
            "edge_weighted_rgb_error",
            "optical_depth_mae",
            "semantic_agreement",
            "acceptance_pass",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    (args.output_dir / "per-frame.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("TRAJECTORY_COMPARISON_OK " + json.dumps({"summary": str(args.output_dir / "summary.json"), "pass": summary["pass"]}, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
