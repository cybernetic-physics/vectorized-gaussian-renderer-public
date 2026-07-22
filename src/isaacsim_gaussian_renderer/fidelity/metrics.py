"""Renderer fidelity metrics, thresholding, and artifact generation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

from .camera_bundle import CameraBundle
from .outputs import RenderOutput

FIDELITY_THRESHOLDS = {
    "rgb_psnr_db": 40.0,
    "rgb_ssim": 0.995,
    "lpips": 0.01,
    "alpha_mae": 0.005,
    "depth_rel_error": 0.01,
    "semantic_foreground_agreement": 0.999,
}

HIGHER_IS_BETTER = {
    "rgb_psnr_db",
    "rgb_ssim",
    "semantic_foreground_agreement",
}


def compare_render_outputs(
    *,
    reference: RenderOutput,
    candidate: RenderOutput,
    camera_bundle: CameraBundle,
    output_dir: str | Path | None = None,
    config_id: str | None = None,
    require_lpips: bool = True,
    max_artifact_views: int | None = None,
) -> dict[str, Any]:
    """Compare reference and candidate tensors and optionally write report artifacts."""
    camera_bundle.validate()
    reference.validate(
        expected_views=len(camera_bundle.cameras),
        height=camera_bundle.height,
        width=camera_bundle.width,
    )
    candidate.validate(
        expected_views=len(camera_bundle.cameras),
        height=camera_bundle.height,
        width=camera_bundle.width,
    )
    _validate_contract(reference, candidate, camera_bundle)

    lpips_values, lpips_backend, lpips_error = _lpips_per_view(reference.rgb, candidate.rgb, require=require_lpips)
    per_view = []
    for index, camera in enumerate(camera_bundle.cameras):
        lpips = lpips_values[index] if lpips_values is not None else None
        metrics = _metrics_for_view(reference, candidate, index, lpips)
        threshold_status = _threshold_status(metrics, lpips_backend=lpips_backend)
        per_view.append(
            {
                "index": index,
                "view_id": camera.view_id,
                "scene_id": camera.scene_id,
                "metrics": metrics,
                "thresholds": threshold_status,
                "pass": all(item["pass"] for item in threshold_status.values()),
                "worst_metric": _worst_metric(metrics, threshold_status),
            }
        )

    worst_view = min(per_view, key=lambda item: _view_margin(item["metrics"], item["thresholds"]))
    aggregate = _aggregate(per_view, lpips_backend=lpips_backend, lpips_error=lpips_error)
    report: dict[str, Any] = {
        "schema_version": "fidelity-report-v1",
        "config_id": config_id or camera_bundle.bundle_id,
        "camera_bundle_id": camera_bundle.bundle_id,
        "contract": {
            "width": camera_bundle.width,
            "height": camera_bundle.height,
            "num_views": len(camera_bundle.cameras),
            "background": list(camera_bundle.background),
            "color_space": camera_bundle.color_space,
            "coordinate_system": camera_bundle.coordinate_system,
            "scene_checksum": camera_bundle.scene_checksum,
        },
        "thresholds": FIDELITY_THRESHOLDS,
        "lpips_backend": lpips_backend,
        "lpips_error": lpips_error,
        "per_view": per_view,
        "aggregate": aggregate,
        "worst_view": {
            "index": worst_view["index"],
            "view_id": worst_view["view_id"],
            "scene_id": worst_view["scene_id"],
            "worst_metric": worst_view["worst_metric"],
        },
        "pass": aggregate["pass"],
    }

    if output_dir is not None:
        artifact_dir = Path(output_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_reports(report, artifact_dir)
        artifact_view_indices = _artifact_view_indices(
            per_view,
            max_views=max_artifact_views,
        )
        report["artifact_view_indices"] = artifact_view_indices
        _write_artifacts(
            reference,
            candidate,
            camera_bundle,
            artifact_dir,
            artifact_view_indices,
        )
        # The report is rewritten after artifact selection so the immutable JSON
        # says exactly which worst-case views have visual evidence.
        _write_reports(report, artifact_dir)

    return report


def validate_fidelity_report(report: dict[str, Any]) -> None:
    """Recompute all threshold/aggregate decisions from stored per-view metrics."""

    if report.get("schema_version") != "fidelity-report-v1":
        raise ValueError("Unexpected fidelity-report schema.")
    if report.get("thresholds") != FIDELITY_THRESHOLDS:
        raise ValueError("Fidelity-report thresholds differ from the repository gate.")
    per_view = report.get("per_view")
    if not isinstance(per_view, list) or not per_view:
        raise ValueError("Fidelity report has no per-view metrics.")
    if (report.get("contract") or {}).get("num_views") != len(per_view):
        raise ValueError("Fidelity-report view count differs from its contract.")
    lpips_backend = str(report.get("lpips_backend", "unavailable"))
    normalized_views: list[dict[str, Any]] = []
    for expected_index, view in enumerate(per_view):
        if view.get("index") != expected_index:
            raise ValueError("Fidelity-report view indices are not contiguous.")
        metrics = {
            name: _metric_from_json(value)
            for name, value in (view.get("metrics") or {}).items()
        }
        for name in FIDELITY_THRESHOLDS:
            if name not in metrics:
                raise ValueError(f"Fidelity-report metric {name} is missing.")
        recomputed_thresholds = _threshold_status(
            metrics,
            lpips_backend=lpips_backend,
        )
        if view.get("thresholds") != _json_safe(recomputed_thresholds):
            raise ValueError(
                f"Fidelity-report thresholds differ for view {expected_index}."
            )
        recomputed_pass = all(
            item["pass"] for item in recomputed_thresholds.values()
        )
        if view.get("pass") is not recomputed_pass:
            raise ValueError(
                f"Fidelity-report pass differs for view {expected_index}."
            )
        if view.get("worst_metric") != _worst_metric(
            metrics,
            recomputed_thresholds,
        ):
            raise ValueError(
                f"Fidelity-report worst metric differs for view {expected_index}."
            )
        normalized_views.append(
            {
                **view,
                "metrics": metrics,
                "thresholds": recomputed_thresholds,
            }
        )
    recomputed_aggregate = _aggregate(
        normalized_views,
        lpips_backend=lpips_backend,
        lpips_error=report.get("lpips_error"),
    )
    if report.get("aggregate") != _json_safe(recomputed_aggregate):
        raise ValueError("Fidelity-report aggregate differs from per-view metrics.")
    worst_view = min(
        normalized_views,
        key=lambda item: _view_margin(item["metrics"], item["thresholds"]),
    )
    expected_worst = {
        "index": worst_view["index"],
        "view_id": worst_view["view_id"],
        "scene_id": worst_view["scene_id"],
        "worst_metric": worst_view["worst_metric"],
    }
    if report.get("worst_view") != expected_worst:
        raise ValueError("Fidelity-report worst-view record differs.")
    if report.get("pass") is not bool(recomputed_aggregate["pass"]):
        raise ValueError("Fidelity-report top-level pass differs.")


def _metric_from_json(value: Any) -> Any:
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    return value


def _artifact_view_indices(
    per_view: list[dict[str, Any]],
    *,
    max_views: int | None,
) -> list[int]:
    ordered = sorted(
        per_view,
        key=lambda item: (
            _view_margin(item["metrics"], item["thresholds"]),
            int(item["index"]),
        ),
    )
    count = len(ordered) if max_views is None else min(max_views, len(ordered))
    return [int(item["index"]) for item in ordered[:count]]


def _validate_contract(reference: RenderOutput, candidate: RenderOutput, camera_bundle: CameraBundle) -> None:
    for name, output in (("reference", reference), ("candidate", candidate)):
        if output.camera_bundle_id is not None and output.camera_bundle_id != camera_bundle.bundle_id:
            raise ValueError(f"{name} camera_bundle_id does not match the supplied camera bundle.")
        if output.color_space is not None and output.color_space != camera_bundle.color_space:
            raise ValueError(f"{name} color_space {output.color_space!r} does not match {camera_bundle.color_space!r}.")
        if output.background is not None and tuple(output.background) != tuple(camera_bundle.background):
            raise ValueError(f"{name} background {output.background!r} does not match {camera_bundle.background!r}.")
    if (
        reference.color_space is not None
        and candidate.color_space is not None
        and reference.color_space != candidate.color_space
    ):
        raise ValueError("Reference and candidate color spaces differ.")
    if (
        reference.background is not None
        and candidate.background is not None
        and reference.background != candidate.background
    ):
        raise ValueError("Reference and candidate backgrounds differ.")


def _metrics_for_view(
    reference: RenderOutput,
    candidate: RenderOutput,
    view_index: int,
    lpips: float | None,
) -> dict[str, float | None]:
    ref_rgb = np.clip(reference.rgb[view_index], 0.0, 1.0)
    cand_rgb = np.clip(candidate.rgb[view_index], 0.0, 1.0)
    ref_alpha = reference.alpha[view_index]
    cand_alpha = candidate.alpha[view_index]
    ref_depth = reference.depth[view_index]
    cand_depth = candidate.depth[view_index]
    ref_semantic = reference.semantic[view_index]
    cand_semantic = candidate.semantic[view_index]

    valid_depth = _valid_depth_mask(reference, candidate, view_index)
    semantic_foreground = (ref_alpha >= 0.01) | (cand_alpha >= 0.01)
    if semantic_foreground.any():
        semantic_foreground_agreement = float(
            np.mean(
                cand_semantic[semantic_foreground]
                == ref_semantic[semantic_foreground]
            )
        )
    else:
        semantic_foreground_agreement = 1.0

    if valid_depth.any():
        depth_rel_error = float(
            np.mean(np.abs(cand_depth[valid_depth] - ref_depth[valid_depth]) / np.abs(ref_depth[valid_depth]))
        )
    else:
        depth_rel_error = None

    return {
        "rgb_psnr_db": _psnr(ref_rgb, cand_rgb),
        "rgb_ssim": _rgb_ssim(ref_rgb, cand_rgb),
        "lpips": lpips,
        "alpha_mae": float(np.mean(np.abs(cand_alpha - ref_alpha))),
        "depth_rel_error": depth_rel_error,
        "valid_depth_pixels": int(valid_depth.sum()),
        "semantic_agreement": float(np.mean(cand_semantic == ref_semantic)),
        "semantic_foreground_agreement": semantic_foreground_agreement,
        "semantic_foreground_pixels": int(semantic_foreground.sum()),
        "semantic_mismatch_pixels": int(np.count_nonzero(cand_semantic != ref_semantic)),
    }


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((candidate.astype(np.float64) - reference.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return float(10.0 * math.log10(1.0 / mse))


def _rgb_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean([_ssim_channel(reference[..., channel], candidate[..., channel]) for channel in range(3)]))


def _ssim_channel(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    height, width = ref.shape
    window = min(11, height, width)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return _global_ssim_channel(ref.reshape(-1), cand.reshape(-1))

    pad = window // 2
    ref_windows = sliding_window_view(np.pad(ref, pad, mode="reflect"), (window, window))
    cand_windows = sliding_window_view(np.pad(cand, pad, mode="reflect"), (window, window))
    mux = ref_windows.mean(axis=(-2, -1))
    muy = cand_windows.mean(axis=(-2, -1))
    varx = ((ref_windows - mux[..., None, None]) ** 2).mean(axis=(-2, -1))
    vary = ((cand_windows - muy[..., None, None]) ** 2).mean(axis=(-2, -1))
    cov = ((ref_windows - mux[..., None, None]) * (cand_windows - muy[..., None, None])).mean(axis=(-2, -1))
    c1 = 0.01**2
    c2 = 0.03**2
    denominator = (mux**2 + muy**2 + c1) * (varx + vary + c2)
    ssim_map = ((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / denominator
    return float(np.clip(ssim_map.mean(), -1.0, 1.0))


def _global_ssim_channel(reference: np.ndarray, candidate: np.ndarray) -> float:
    mux = float(reference.mean())
    muy = float(candidate.mean())
    varx = float(((reference - mux) ** 2).mean())
    vary = float(((candidate - muy) ** 2).mean())
    cov = float(((reference - mux) * (candidate - muy)).mean())
    c1 = 0.01**2
    c2 = 0.03**2
    denominator = (mux**2 + muy**2 + c1) * (varx + vary + c2)
    return float(np.clip(((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / denominator, -1.0, 1.0))


def _lpips_per_view(
    reference_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    *,
    require: bool,
) -> tuple[list[float] | None, str, str | None]:
    if np.array_equal(reference_rgb, candidate_rgb):
        return [0.0 for _ in range(reference_rgb.shape[0])], "exact_identity", None
    if min(reference_rgb.shape[1:3]) < 64:
        message = (
            "LPIPS AlexNet requires images at least 64 pixels on each side; "
            f"got {reference_rgb.shape[2]}x{reference_rgb.shape[1]}."
        )
        return None, "unavailable" if require else "skipped", message
    try:
        import torch
        import lpips
    except Exception as exc:
        message = f"LPIPS package unavailable for non-identical images: {exc}"
        if require:
            return None, "unavailable", message
        return None, "skipped", message

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = lpips.LPIPS(net="alex").to(device).eval()
        values = []
        with torch.inference_mode():
            for view in range(reference_rgb.shape[0]):
                ref = torch.from_numpy(reference_rgb[view].transpose(2, 0, 1))
                cand = torch.from_numpy(candidate_rgb[view].transpose(2, 0, 1))
                ref = ref.unsqueeze(0).to(device=device, dtype=torch.float32)
                cand = cand.unsqueeze(0).to(device=device, dtype=torch.float32)
                values.append(float(model(ref * 2.0 - 1.0, cand * 2.0 - 1.0).item()))
    except Exception as exc:
        message = f"LPIPS evaluation failed: {exc}"
        return None, "unavailable" if require else "skipped", message
    return values, "lpips-alex", None


def _threshold_status(metrics: dict[str, float | None], *, lpips_backend: str) -> dict[str, dict[str, Any]]:
    status = {}
    for name, threshold in FIDELITY_THRESHOLDS.items():
        value = metrics.get(name)
        if value is None:
            passed = False
        elif name in HIGHER_IS_BETTER:
            passed = float(value) >= threshold
        else:
            passed = float(value) <= threshold
        if name == "lpips" and lpips_backend in {"unavailable", "skipped"}:
            passed = False
        status[name] = {"value": value, "threshold": threshold, "pass": bool(passed)}
    return status


def _worst_metric(metrics: dict[str, float | None], status: dict[str, dict[str, Any]]) -> str:
    metric_names = [name for name in FIDELITY_THRESHOLDS if not (name == "lpips" and metrics.get(name) is None)]
    if not metric_names:
        metric_names = list(FIDELITY_THRESHOLDS)
    return min(metric_names, key=lambda name: _metric_margin(name, metrics.get(name), status[name]["threshold"]))


def _view_margin(metrics: dict[str, float | None], status: dict[str, dict[str, Any]]) -> float:
    margins = [
        _metric_margin(name, metrics.get(name), status[name]["threshold"])
        for name in FIDELITY_THRESHOLDS
        if not (name == "lpips" and metrics.get(name) is None)
    ]
    return min(margins) if margins else -math.inf


def _metric_margin(name: str, value: float | None, threshold: float) -> float:
    if value is None:
        return -math.inf
    if name in HIGHER_IS_BETTER:
        return float(value) - threshold
    return threshold - float(value)


def _aggregate(per_view: list[dict[str, Any]], *, lpips_backend: str, lpips_error: str | None) -> dict[str, Any]:
    metrics: dict[str, dict[str, float | None]] = {}
    aggregate_names = (*FIDELITY_THRESHOLDS, "semantic_agreement")
    for name in aggregate_names:
        values = [view["metrics"][name] for view in per_view if view["metrics"][name] is not None]
        if values:
            metrics[name] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            metrics[name] = {"mean": None, "min": None, "max": None}
    metrics["valid_depth_pixels"] = {
        "sum": int(sum(view["metrics"]["valid_depth_pixels"] for view in per_view)),
    }
    metrics["semantic_mismatch_pixels"] = {
        "sum": int(sum(view["metrics"]["semantic_mismatch_pixels"] for view in per_view)),
    }
    metrics["semantic_foreground_pixels"] = {
        "sum": int(sum(view["metrics"]["semantic_foreground_pixels"] for view in per_view)),
    }

    threshold_status = {}
    for name, threshold in FIDELITY_THRESHOLDS.items():
        per_view_pass = all(view["thresholds"][name]["pass"] for view in per_view)
        aggregate_value = metrics[name]["min"] if name in HIGHER_IS_BETTER else metrics[name]["max"]
        threshold_status[name] = {
            "value": aggregate_value,
            "threshold": threshold,
            "pass": bool(per_view_pass),
            "rule": "all_views",
        }
    return {
        "metrics": metrics,
        "thresholds": threshold_status,
        "lpips_backend": lpips_backend,
        "lpips_error": lpips_error,
        "pass": all(item["pass"] for item in threshold_status.values()),
    }


def _write_reports(report: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "fidelity_report.json").write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")
    with (output_dir / "fidelity_report.csv").open("w", newline="") as handle:
        fieldnames = [
            "config_id",
            "camera_bundle_id",
            "view_id",
            "scene_id",
            "pass",
            "worst_metric",
            *FIDELITY_THRESHOLDS.keys(),
            "semantic_agreement",
            "valid_depth_pixels",
            "semantic_foreground_pixels",
            "semantic_mismatch_pixels",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for view in report["per_view"]:
            row = {
                "config_id": report["config_id"],
                "camera_bundle_id": report["camera_bundle_id"],
                "view_id": view["view_id"],
                "scene_id": view["scene_id"],
                "pass": view["pass"],
                "worst_metric": view["worst_metric"],
            }
            row.update(view["metrics"])
            writer.writerow(row)


def _write_artifacts(
    reference: RenderOutput,
    candidate: RenderOutput,
    camera_bundle: CameraBundle,
    output_dir: Path,
    view_indices: list[int],
) -> None:
    for index in view_indices:
        stem = f"{index:04d}_{camera_bundle.cameras[index].view_id}"
        ref_rgb = np.clip(reference.rgb[index], 0.0, 1.0)
        cand_rgb = np.clip(candidate.rgb[index], 0.0, 1.0)
        rgb_diff = np.abs(cand_rgb - ref_rgb)
        side = np.concatenate((ref_rgb, cand_rgb, np.clip(rgb_diff * 8.0, 0.0, 1.0)), axis=1)
        _write_rgb(output_dir / f"{stem}_rgb_side_by_side.png", side)
        _write_rgb(output_dir / f"{stem}_rgb_absdiff.png", np.clip(rgb_diff * 8.0, 0.0, 1.0))

        alpha_diff = np.abs(candidate.alpha[index] - reference.alpha[index])
        alpha_image = np.clip(alpha_diff / FIDELITY_THRESHOLDS["alpha_mae"], 0.0, 1.0)
        _write_gray(output_dir / f"{stem}_alpha_absdiff.png", alpha_image)

        depth_den = np.maximum(np.abs(reference.depth[index]), 1.0e-8)
        depth_rel = np.abs(candidate.depth[index] - reference.depth[index]) / depth_den
        valid = _valid_depth_mask(reference, candidate, index)
        depth_image = np.where(valid, np.clip(depth_rel / FIDELITY_THRESHOLDS["depth_rel_error"], 0.0, 1.0), 0.0)
        _write_gray(output_dir / f"{stem}_depth_relerr.png", depth_image)

        semantic_mismatch = candidate.semantic[index] != reference.semantic[index]
        semantic_rgb = np.zeros((*semantic_mismatch.shape, 3), dtype=np.float32)
        semantic_rgb[..., 0] = semantic_mismatch.astype(np.float32)
        _write_rgb(output_dir / f"{stem}_semantic_mismatch.png", semantic_rgb)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def _write_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def _valid_depth_mask(reference: RenderOutput, candidate: RenderOutput, view_index: int) -> np.ndarray:
    ref_depth = reference.depth[view_index]
    cand_depth = candidate.depth[view_index]
    valid_depth = np.isfinite(ref_depth) & np.isfinite(cand_depth) & (np.abs(ref_depth) > 1.0e-8)
    if reference.valid_depth is not None:
        valid_depth &= reference.valid_depth[view_index]
    if candidate.valid_depth is not None:
        valid_depth &= candidate.valid_depth[view_index]
    return valid_depth


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
