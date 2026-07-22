"""Sweep custom Gaussian models against a saved temporal OVRTX output."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.run_ovrtx import (
    PLY_SCENES,
    SYNTHETIC_SCENES,
    load_scene_and_cameras,
)
from experiments.ovrtx_parameter_sweep import metrics
from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    compact_semantic_ids,
    sha256_json,
    spatial_semantic_ids,
    spatial_semantic_manifest,
)
from isaacsim_gaussian_renderer.fidelity import load_render_output


def float_list(value: str) -> list[float]:
    values = [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of finite floats."
        )
    return values


def string_list(value: str) -> list[str]:
    values = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]
    if not values:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of values."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
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
    parser.add_argument(
        "--semantic-scheme",
        choices=("index-modulo", "spatial-grid"),
        default="index-modulo",
    )
    parser.add_argument(
        "--semantic-grid",
        type=lambda value: tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ),
        default=(2, 2, 2),
    )
    parser.add_argument(
        "--support-sigmas",
        type=float_list,
        default=float_list("2.5,2.7,3.0,3.33"),
    )
    parser.add_argument(
        "--covariance-epsilons",
        type=float_list,
        default=float_list("0,0.3"),
    )
    parser.add_argument(
        "--models",
        choices=("screen", "ray", "both"),
        default="both",
    )
    parser.add_argument(
        "--semantic-min-alpha",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--orientation-modes",
        type=string_list,
        default=["wxyz"],
        help=(
            "Comma-separated quaternion interpretations: wxyz, conjugate, "
            "xyzw-memory, xyzw-memory-conjugate."
        ),
    )
    parser.add_argument(
        "--scale-permutations",
        type=string_list,
        default=["xyz"],
        help=(
            "Comma-separated axis permutations drawn from xyz,xzy,yxz,yzx,zxy,zyx."
        ),
    )
    parser.add_argument("--visible-capacity", type=int, default=None)
    parser.add_argument(
        "--intersection-capacity",
        type=int,
        default=2_000_000,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def candidate_arrays(path: Path) -> dict[str, np.ndarray]:
    output = load_render_output(path)
    return {
        "rgb": output.rgb,
        "alpha": output.alpha,
        "depth": output.depth,
        "semantic": output.semantic,
    }


def transformed_rotations(
    rotations: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "wxyz":
        return rotations
    if mode == "conjugate":
        transformed = rotations.clone()
        transformed[:, 1:].neg_()
        return transformed.contiguous()
    if mode in {"xyzw-memory", "xyzw-memory-conjugate"}:
        transformed = rotations[:, [3, 0, 1, 2]].contiguous()
        if mode.endswith("conjugate"):
            transformed[:, 1:].neg_()
        return transformed
    raise ValueError(f"Unsupported orientation mode: {mode}")


def transformed_scales(
    scales: torch.Tensor,
    permutation: str,
) -> torch.Tensor:
    permutations = {
        "xyz": (0, 1, 2),
        "xzy": (0, 2, 1),
        "yxz": (1, 0, 2),
        "yzx": (1, 2, 0),
        "zxy": (2, 0, 1),
        "zyx": (2, 1, 0),
    }
    if permutation not in permutations:
        raise ValueError(
            f"Unsupported scale permutation: {permutation}"
        )
    indices = permutations[permutation]
    if indices == (0, 1, 2):
        return scales
    return scales[:, indices].contiguous()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.focal_scale <= 0
        or args.home_scan_fit_margin <= 0
        or len(args.semantic_grid) != 3
        or min(args.semantic_grid) <= 0
        or min(args.support_sigmas) <= 0
        or min(args.covariance_epsilons) < 0
        or not math.isfinite(args.semantic_min_alpha)
        or args.semantic_min_alpha < 0
        or args.semantic_min_alpha > 1
        or args.intersection_capacity <= 0
        or (
            args.visible_capacity is not None
            and args.visible_capacity <= 0
        )
    ):
        raise ValueError("Geometry and capacities must be positive.")

    candidate = candidate_arrays(args.candidate)
    expected_shape = (1, args.height, args.width)
    if candidate["alpha"].shape != expected_shape:
        raise ValueError(
            "Candidate resolution does not match the requested sweep."
        )

    device = torch.device("cuda:0")
    args.batch_size = 1
    scene, scene_manifest, cameras, scene_loading = (
        load_scene_and_cameras(args)
    )
    if args.semantic_scheme == "spatial-grid":
        raw_semantic_ids = spatial_semantic_ids(
            scene["means"],
            grid=args.semantic_grid,
            dtype=torch.int64,
        )
        scene["semantic_ids"], occupied_source_ids = compact_semantic_ids(
            raw_semantic_ids,
            dtype=torch.int64,
        )
        scene_manifest = dict(scene_manifest)
        scene_manifest["semantic_id_rule"] = spatial_semantic_manifest(
            int(scene["means"].shape[0]),
            grid=args.semantic_grid,
            position_min=scene["means"].amin(dim=0),
            position_max=scene["means"].amax(dim=0),
            occupied_source_ids=occupied_source_ids,
        )
        scene_manifest["checksum_sha256"] = sha256_json(
            {
                key: value
                for key, value in scene_manifest.items()
                if key != "checksum_sha256"
            }
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
    models = (
        (False, True)
        if args.models == "both"
        else (args.models == "ray",)
    )
    rows: list[dict[str, Any]] = []

    for orientation_mode in args.orientation_modes:
        rotations = transformed_rotations(
            scene["quats"],
            orientation_mode,
        )
        for scale_permutation in args.scale_permutations:
            scales = transformed_scales(
                scene["scales"],
                scale_permutation,
            )
            for ray_gaussian_evaluation in models:
                for covariance_epsilon in args.covariance_epsilons:
                    for support_sigma in args.support_sigmas:
                        backend = CustomCudaBackend(
                            max_visible_records=visible_capacity,
                            max_intersections=args.intersection_capacity,
                            near_plane=float(
                                cameras.manifest["near_plane"]
                            ),
                            far_plane=float(
                                cameras.manifest["far_plane"]
                            ),
                            gaussian_support_sigma=support_sigma,
                            covariance_epsilon=covariance_epsilon,
                            semantic_min_alpha=args.semantic_min_alpha,
                            ray_gaussian_evaluation=(
                                ray_gaussian_evaluation
                            ),
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
                            scales=scales,
                            rotations=rotations,
                            opacities=scene["opacities"],
                            features=scene["colors"],
                            semantic_ids=(
                                scene["semantic_ids"].to(torch.int64)
                            ),
                        )
                        rendered = service.render(
                            cameras.viewmats,
                            cameras.intrinsics,
                            torch.tensor(
                                [101],
                                device=device,
                                dtype=torch.int64,
                            ),
                        )
                        service.synchronize()
                        counters = backend.check_capacity(
                            synchronize=False
                        )
                        custom = {
                            "rgb": (
                                rendered["rgb"].detach().cpu().numpy()
                            ),
                            "alpha": (
                                rendered["alpha"][..., 0]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            "depth": (
                                rendered["depth"][..., 0]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            "semantic": (
                                rendered["semantic_id"][..., 0]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                        }
                        row_metrics = metrics(
                            custom=custom,
                            candidate=candidate,
                        )
                        row = {
                            "config_id": (
                                f"{orientation_mode}-"
                                f"scales-{scale_permutation}-"
                                f"{'ray' if ray_gaussian_evaluation else 'screen'}-"
                                f"sigma-{support_sigma:g}-"
                                f"epsilon-{covariance_epsilon:g}"
                            ),
                            "orientation_mode": orientation_mode,
                            "scale_permutation": scale_permutation,
                            "ray_gaussian_evaluation": (
                                ray_gaussian_evaluation
                            ),
                            "support_sigma": support_sigma,
                            "covariance_epsilon": covariance_epsilon,
                            "semantic_min_alpha": args.semantic_min_alpha,
                            "counters": counters,
                            "metrics": row_metrics,
                        }
                        rows.append(row)
                        print(
                            "OVRTX_CUSTOM_MODEL_SWEEP "
                            + json.dumps(row, sort_keys=True),
                            flush=True,
                        )
                        service.shutdown()
                        del custom
                        del rendered
                        del service
                        del backend
                        gc.collect()
                        torch.cuda.empty_cache()
            if scales is not scene["scales"]:
                del scales
        if rotations is not scene["quats"]:
            del rotations

    rows.sort(key=lambda row: row["metrics"]["threshold_distance"])
    result = {
        "schema_version": "ovrtx-custom-model-sweep/v1",
        "diagnostic_only": True,
        "candidate": str(args.candidate),
        "scene_id": args.scene,
        "scene": scene_manifest,
        "scene_loading": scene_loading,
        "camera_manifest": cameras.manifest,
        "gaussian_count": gaussian_count,
        "semantic_scheme": args.semantic_scheme,
        "semantic_grid": list(args.semantic_grid),
        "semantic_min_alpha": args.semantic_min_alpha,
        "row_count": len(rows),
        "best_row": rows[0],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
