"""Measure temporal convergence of stochastic OVRTX Gaussian rendering."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import ovrtx
import torch

from benchmarks.run_ovrtx import (
    PLY_SCENES,
    PROJECTION_ACTIVATION_SCENE,
    PROJECTION_MODES,
    SORTING_MODES,
    SYNTHETIC_SCENES,
    allocate_contract_outputs,
    build_stage,
    copy_mapped_outputs_to_contract,
    git_commit,
    implementation_commit,
    load_scene_and_cameras,
    product_paths,
    semantic_lut_from_outputs,
    splat_paths,
    splat_prim_semantic_ids,
    require_token_attribute_readback,
    upload_scene,
)
from isaacsim_gaussian_renderer.benchmark_manifest import (
    compact_semantic_ids,
    sha256_json,
    spatial_semantic_ids,
    spatial_semantic_manifest,
    scene_tensors_sha256,
)
from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.fidelity import (
    RenderOutput,
    bundle_from_tensors,
    compare_render_outputs,
    write_camera_bundle,
)


OVRTX_TEMPORAL_CLEANUP_MARKER = "OVRTX_TEMPORAL_CLEANUP_OK"
OVRTX_TEMPORAL_SUCCESS_MARKER = "OVRTX_TEMPORAL_CAPTURE_OK"
_RendererT = TypeVar("_RendererT")


@dataclass(frozen=True)
class _RendererCleanupResult:
    renderer_was_created: bool
    python_renderer_released: bool
    collected_objects: int


class _OvrtxLifecycle:
    """Own OVRTX objects until the capture stack has completely unwound."""

    def __init__(self) -> None:
        self._renderer: object | None = None
        self._upload_keepalive: object | None = None
        self._renderer_was_created = False

    def own_renderer(self, renderer: _RendererT) -> _RendererT:
        if self._renderer is not None:
            raise RuntimeError("OVRTX lifecycle already owns a renderer.")
        self._renderer = renderer
        self._renderer_was_created = True
        return renderer

    def retain_upload(self, upload_keepalive: object) -> None:
        if self._renderer is None:
            raise RuntimeError("Cannot retain OVRTX upload buffers without a renderer.")
        self._upload_keepalive = upload_keepalive

    def release(self) -> _RendererCleanupResult:
        renderer = self._renderer
        renderer_ref = weakref.ref(renderer) if renderer is not None else None

        # OVRTX tensor writes use ASYNC access. Drop their source buffers only
        # after the capture body has returned (or unwound), and before dropping
        # the renderer that owns the native system.
        self._upload_keepalive = None
        self._renderer = None
        del renderer
        collected_objects = gc.collect()

        return _RendererCleanupResult(
            renderer_was_created=self._renderer_was_created,
            python_renderer_released=(renderer_ref is None or renderer_ref() is None),
            collected_objects=collected_objects,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        choices=sorted((*SYNTHETIC_SCENES, PROJECTION_ACTIVATION_SCENE, *PLY_SCENES)),
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
    parser.add_argument("--focal-scale", type=float, default=0.9)
    parser.add_argument(
        "--home-scan-fit-margin",
        type=float,
        default=1.15,
    )
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--custom-support-sigma", type=float, default=3.0)
    parser.add_argument(
        "--custom-covariance-epsilon",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--custom-ray-gaussian-evaluation",
        action="store_true",
    )
    parser.add_argument(
        "--custom-semantic-min-alpha",
        type=float,
        default=0.01,
    )
    parser.add_argument("--custom-visible-capacity", type=int)
    parser.add_argument("--custom-intersection-capacity", type=int)
    parser.add_argument("--custom-tile-size", type=int)
    parser.add_argument("--custom-depth-bucket-count", type=int)
    parser.add_argument("--custom-depth-bucket-group-size", type=int)
    parser.add_argument(
        "--custom-deterministic",
        action="store_true",
        help=(
            "Enable deterministic secondary ordering for the custom reference. "
            "This disables the dense pixel depth-cutoff optimization."
        ),
    )
    parser.add_argument(
        "--projection-mode",
        choices=PROJECTION_MODES,
        default="perspective",
    )
    parser.add_argument(
        "--sorting-mode",
        choices=SORTING_MODES,
        default="zDepth",
    )
    parser.add_argument(
        "--semantic-scheme",
        choices=("index-modulo", "spatial-grid"),
        default="index-modulo",
    )
    parser.add_argument(
        "--semantic-grid",
        type=lambda value: tuple(int(item.strip()) for item in value.split(",") if item.strip()),
        default=(2, 2, 2),
        help="XYZ grid dimensions for --semantic-scheme spatial-grid.",
    )
    parser.add_argument(
        "--save-checkpoint-outputs",
        action="store_true",
        help="Save canonical OVRTX output tensors at every power-of-four checkpoint.",
    )
    parser.add_argument(
        "--final-report-only",
        action="store_true",
        help="Skip fidelity reports at intermediate temporal checkpoints.",
    )
    parser.add_argument(
        "--resume-accumulators",
        type=Path,
        default=None,
        help=(
            "Resume temporal sums and semantic vote counts from a prior "
            "accumulators-final.npz; --frames is the new total target."
        ),
    )
    parser.add_argument(
        "--save-final-accumulators",
        action="store_true",
        help="Save resumable temporal sums and semantic counts at completion.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=4096,
        help="Write progress.json and a log line every N accumulated frames.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def as_render_output(
    *,
    rgb: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    semantic: np.ndarray,
    camera_bundle_id: str,
    source: str,
    valid_depth: np.ndarray | None = None,
) -> RenderOutput:
    output = RenderOutput(
        rgb=rgb.astype(np.float32, copy=False),
        alpha=alpha.astype(np.float32, copy=False),
        depth=depth.astype(np.float32, copy=False),
        semantic=semantic.astype(np.int64, copy=False),
        valid_depth=(alpha > 0 if valid_depth is None else valid_depth.astype(bool, copy=False)),
        color_space="display_srgb",
        background=(0.0, 0.0, 0.0),
        camera_bundle_id=camera_bundle_id,
        source=source,
    )
    output.validate()
    return output


def save_render_output(path: Path, output: RenderOutput) -> None:
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


@torch.inference_mode()
def _run_temporal_capture(lifecycle: _OvrtxLifecycle) -> None:
    args = parse_args()
    if (
        args.frames < 1
        or args.warmup < 0
        or args.progress_every <= 0
        or args.focal_scale <= 0
        or args.home_scan_fit_margin <= 0
        or args.custom_support_sigma <= 0
        or len(args.semantic_grid) != 3
        or min(args.semantic_grid) <= 0
        or not np.isfinite(args.custom_covariance_epsilon)
        or args.custom_covariance_epsilon < 0
        or not np.isfinite(args.custom_semantic_min_alpha)
        or args.custom_semantic_min_alpha < 0
        or args.custom_semantic_min_alpha > 1
        or (args.custom_visible_capacity is not None and args.custom_visible_capacity <= 0)
        or (args.custom_intersection_capacity is not None and args.custom_intersection_capacity <= 0)
        or (
            args.custom_tile_size is not None
            and (
                args.custom_tile_size <= 0
                or args.custom_tile_size > 16
                or args.custom_tile_size & (args.custom_tile_size - 1)
            )
        )
        or (args.custom_depth_bucket_count is not None and args.custom_depth_bucket_count <= 0)
        or (args.custom_depth_bucket_group_size is not None and args.custom_depth_bucket_group_size <= 0)
    ):
        raise ValueError(
            "Expected positive frames/support sigma, non-negative warmup, and finite non-negative covariance epsilon."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    args.batch_size = 1
    scene, scene_manifest, cameras, scene_loading = load_scene_and_cameras(args)
    if args.semantic_scheme == "spatial-grid":
        raw_semantic_ids = spatial_semantic_ids(
            scene["means"],
            grid=args.semantic_grid,
            dtype=torch.int64,
        )
        semantic_ids, occupied_source_ids = compact_semantic_ids(
            raw_semantic_ids,
            dtype=torch.int64,
        )
        scene["semantic_ids"] = semantic_ids
        scene_manifest = dict(scene_manifest)
        scene_manifest["semantic_id_rule"] = spatial_semantic_manifest(
            int(scene["means"].shape[0]),
            grid=args.semantic_grid,
            position_min=scene["means"].amin(dim=0),
            position_max=scene["means"].amax(dim=0),
            occupied_source_ids=occupied_source_ids,
        )
        scene_manifest["checksum_sha256"] = sha256_json(
            {key: value for key, value in scene_manifest.items() if key != "checksum_sha256"}
        )
    gaussian_count = int(scene["means"].shape[0])
    scene_tensor_checksum = scene_tensors_sha256(scene)
    semantic_group_count = int(scene["semantic_ids"].max().item()) + 1
    semantic_counts = (
        torch.bincount(
            scene["semantic_ids"].to(torch.int64),
            minlength=semantic_group_count,
        )
        .to(device="cpu")
        .tolist()
    )
    prim_semantic_ids = splat_prim_semantic_ids(semantic_counts)
    scene_ids = torch.tensor([101], device=device, dtype=torch.int64)
    fidelity_bundle = bundle_from_tensors(
        viewmats=cameras.viewmats,
        intrinsics=cameras.intrinsics,
        width=args.width,
        height=args.height,
        color_space="display_srgb",
        scene_ids=scene_ids,
        scene_checksum=scene_manifest["checksum_sha256"],
    )
    write_camera_bundle(fidelity_bundle, args.output / "camera-bundle.json")

    real_scene = args.scene in PLY_SCENES
    custom_visible_capacity = args.custom_visible_capacity or (
        gaussian_count if real_scene else (5_000 if args.custom_ray_gaussian_evaluation else 650)
    )
    custom_intersection_capacity = args.custom_intersection_capacity or (
        500_000 if real_scene or args.custom_ray_gaussian_evaluation else 2_600
    )
    custom_tile_size = args.custom_tile_size or (1 if real_scene else 16)
    custom_depth_bucket_count = args.custom_depth_bucket_count or (1024 if real_scene else 4096)
    custom_depth_bucket_group_size = args.custom_depth_bucket_group_size or (32 if real_scene else 64)
    if custom_depth_bucket_group_size > custom_depth_bucket_count:
        raise ValueError("custom depth bucket group size cannot exceed bucket count.")
    custom_backend = CustomCudaBackend(
        max_visible_records=custom_visible_capacity,
        max_intersections=custom_intersection_capacity,
        near_plane=float(cameras.manifest["near_plane"]),
        far_plane=float(cameras.manifest["far_plane"]),
        gaussian_support_sigma=args.custom_support_sigma,
        covariance_epsilon=args.custom_covariance_epsilon,
        semantic_min_alpha=args.custom_semantic_min_alpha,
        ray_gaussian_evaluation=args.custom_ray_gaussian_evaluation,
        tile_size=custom_tile_size,
        depth_bucket_count=custom_depth_bucket_count,
        depth_bucket_group_size=custom_depth_bucket_group_size,
        output_srgb=False,
        deterministic=args.custom_deterministic,
    )
    custom_service = RendererService(
        custom_backend,
        height=args.height,
        width=args.width,
        max_views=1,
    )
    custom_service.initialize(stage=None, device=device)
    custom_service.load_scene(
        101,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )
    custom_tensors = custom_service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
    )
    custom_service.synchronize()
    custom_counters = custom_backend.check_capacity(synchronize=False)
    reference = as_render_output(
        rgb=custom_tensors["rgb"].detach().cpu().numpy(),
        alpha=custom_tensors["alpha"][..., 0].detach().cpu().numpy(),
        depth=custom_tensors["depth"][..., 0].detach().cpu().numpy(),
        semantic=custom_tensors["semantic_id"][..., 0].detach().cpu().numpy(),
        camera_bundle_id=fidelity_bundle.bundle_id,
        source="custom-cuda-authored-display-rgb",
    )
    save_render_output(args.output / "reference-custom.npz", reference)
    custom_service.shutdown()
    del custom_tensors
    del custom_service
    del custom_backend
    torch.cuda.empty_cache()

    stage = build_stage(
        1,
        args.width,
        args.height,
        args.focal_scale,
        cameras.viewmats,
        "none",
        float(cameras.manifest["near_plane"]),
        float(cameras.manifest["far_plane"]),
        semantic_group_count,
        splat_prim_semantic_ids_override=prim_semantic_ids,
        projection_mode=args.projection_mode,
        sorting_mode=args.sorting_mode,
        fractional_opacity=True,
    )
    stage_path = args.output / "scene.usda"
    stage_path.write_text(stage, encoding="utf-8")
    paths = product_paths(1)

    renderer = lifecycle.own_renderer(
        ovrtx.Renderer(
            ovrtx.RendererConfig(
                # OVRTX 0.3 Linux-headless gate lanes retain the native system
                # until process exit. The bounded parent process, not this
                # Python-handle cleanup marker, proves global shutdown.
                keep_system_alive=True,
                log_level="warning",
                active_cuda_gpus="0",
                use_vulkan=True,
            )
        )
    )
    renderer.open_usd_from_string(stage)
    splat_prim_paths = splat_paths(len(prim_semantic_ids))
    runtime_readback = {
        "projection_mode": require_token_attribute_readback(
            renderer,
            attribute_name="projectionModeHint",
            prim_paths=splat_prim_paths,
            expected=args.projection_mode,
        ),
        "sorting_mode": require_token_attribute_readback(
            renderer,
            attribute_name="sortingModeHint",
            prim_paths=splat_prim_paths,
            expected=args.sorting_mode,
        ),
    }
    _upload_contract, upload_keepalive = upload_scene(
        renderer,
        scene,
        semantic_group_count=semantic_group_count,
    )
    lifecycle.retain_upload(upload_keepalive)
    contract, workspace = allocate_contract_outputs(1, args.height, args.width)
    first = renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
    _semantic_map, _semantic_lut_cpu, semantic_lut = semantic_lut_from_outputs(
        first,
        paths[0],
        expected_group_count=semantic_group_count,
    )
    copy_mapped_outputs_to_contract(
        first,
        paths,
        contract,
        workspace,
        semantic_lut,
    )
    del first
    first_labels = contract["semantic_id"]
    invalid_first_labels = (first_labels < -1) | (first_labels >= semantic_group_count)
    if bool(invalid_first_labels.any().item()):
        raise ValueError("OVRTX produced a semantic ID outside the requested class range.")
    del first_labels
    del invalid_first_labels

    for _ in range(args.warmup):
        outputs = renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
        copy_mapped_outputs_to_contract(
            outputs,
            paths,
            contract,
            workspace,
            semantic_lut,
        )
        del outputs

    pixel_count = args.height * args.width
    resumed_from_frames = 0
    if args.resume_accumulators is None:
        rgb_sum = torch.zeros_like(contract["rgb"])
        alpha_sum = torch.zeros_like(contract["alpha"])
        depth_numerator = torch.zeros_like(contract["depth"])
        depth_weight = torch.zeros_like(contract["depth"])
        semantic_counts = torch.zeros(
            (pixel_count, semantic_group_count),
            device=device,
            dtype=torch.int32,
        )
    else:
        with np.load(
            args.resume_accumulators,
            allow_pickle=False,
        ) as accumulators:
            required = {
                "frames",
                "rgb_sum",
                "alpha_sum",
                "depth_numerator",
                "depth_weight",
                "semantic_counts",
                "width",
                "height",
                "camera_bundle_id",
                "scene_checksum",
                "projection_mode",
                "sorting_mode",
            }
            missing = required - set(accumulators.files)
            if missing:
                raise ValueError(f"Resume accumulator is missing arrays: {sorted(missing)}.")
            resumed_from_frames = int(np.asarray(accumulators["frames"]).item())
            if (
                int(np.asarray(accumulators["width"]).item()) != args.width
                or int(np.asarray(accumulators["height"]).item()) != args.height
            ):
                raise ValueError("Resume accumulator resolution does not match.")
            if str(np.asarray(accumulators["camera_bundle_id"]).item()) != fidelity_bundle.bundle_id:
                raise ValueError("Resume accumulator camera bundle does not match.")
            if str(np.asarray(accumulators["scene_checksum"]).item()) != scene_manifest["checksum_sha256"]:
                raise ValueError("Resume accumulator scene checksum does not match.")
            if (
                str(np.asarray(accumulators["projection_mode"]).item()) != args.projection_mode
                or str(np.asarray(accumulators["sorting_mode"]).item()) != args.sorting_mode
            ):
                raise ValueError("Resume accumulator OVRTX projection/sorting configuration does not match.")
            rgb_sum = torch.from_numpy(np.asarray(accumulators["rgb_sum"])).to(
                device=device, dtype=contract["rgb"].dtype
            )
            alpha_sum = torch.from_numpy(np.asarray(accumulators["alpha_sum"])).to(
                device=device, dtype=contract["alpha"].dtype
            )
            depth_numerator = torch.from_numpy(np.asarray(accumulators["depth_numerator"])).to(
                device=device, dtype=contract["depth"].dtype
            )
            depth_weight = torch.from_numpy(np.asarray(accumulators["depth_weight"])).to(
                device=device, dtype=contract["depth"].dtype
            )
            semantic_counts = torch.from_numpy(np.asarray(accumulators["semantic_counts"])).to(
                device=device, dtype=torch.int32
            )
        if resumed_from_frames <= 0:
            raise ValueError("Resume accumulator frame count must be positive.")
        if resumed_from_frames >= args.frames:
            raise ValueError("--frames must exceed the resumed frame count.")
        expected_shapes = {
            "rgb_sum": tuple(contract["rgb"].shape),
            "alpha_sum": tuple(contract["alpha"].shape),
            "depth_numerator": tuple(contract["depth"].shape),
            "depth_weight": tuple(contract["depth"].shape),
            "semantic_counts": (
                pixel_count,
                semantic_group_count,
            ),
        }
        actual_tensors = {
            "rgb_sum": rgb_sum,
            "alpha_sum": alpha_sum,
            "depth_numerator": depth_numerator,
            "depth_weight": depth_weight,
            "semantic_counts": semantic_counts,
        }
        for name, expected_shape in expected_shapes.items():
            if tuple(actual_tensors[name].shape) != expected_shape:
                raise ValueError(
                    f"Resume {name} has shape {tuple(actual_tensors[name].shape)}, expected {expected_shape}."
                )
    resume_advance_seconds = 0.0
    if resumed_from_frames:
        advance_start = time.perf_counter()
        for skipped_frame in range(1, resumed_from_frames + 1):
            skipped_outputs = renderer.step(
                render_products=set(paths),
                delta_time=1.0 / 60.0,
            )
            del skipped_outputs
            if skipped_frame % args.progress_every == 0 or skipped_frame == resumed_from_frames:
                elapsed = time.perf_counter() - advance_start
                progress = {
                    "schema_version": "ovrtx-temporal-resume-advance/v1",
                    "advanced_frames": skipped_frame,
                    "target_advance_frames": resumed_from_frames,
                    "frames_per_second": (skipped_frame / elapsed if elapsed > 0 else None),
                }
                print(
                    "OVRTX_TEMPORAL_RESUME_ADVANCE " + json.dumps(progress, sort_keys=True),
                    flush=True,
                )
        resume_advance_seconds = time.perf_counter() - advance_start
    pixel_ids = torch.arange(pixel_count, device=device)
    checkpoints = {args.frames}
    checkpoint = 1
    while checkpoint <= args.frames:
        checkpoints.add(checkpoint)
        checkpoint *= 4
    checkpoints = sorted(checkpoints)
    reports = {}
    final_candidate = None
    temporal_start = time.perf_counter()

    for frame_index in range(
        resumed_from_frames + 1,
        args.frames + 1,
    ):
        outputs = renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
        copy_mapped_outputs_to_contract(
            outputs,
            paths,
            contract,
            workspace,
            semantic_lut,
        )
        del outputs

        rgb_sum.add_(contract["rgb"])
        alpha_sum.add_(contract["alpha"])
        valid_depth = torch.isfinite(contract["depth"]) & (contract["depth"] > 0) & (contract["alpha"] > 0)
        frame_depth_weight = torch.where(
            valid_depth,
            contract["alpha"],
            torch.zeros_like(contract["alpha"]),
        )
        depth_numerator.add_(
            torch.where(
                valid_depth,
                contract["depth"] * contract["alpha"],
                torch.zeros_like(contract["depth"]),
            )
        )
        depth_weight.add_(frame_depth_weight)
        labels = contract["semantic_id"].view(-1)
        valid = labels >= 0
        semantic_counts[pixel_ids[valid], labels[valid]] += 1

        if frame_index % args.progress_every == 0 or frame_index == args.frames:
            elapsed_seconds = time.perf_counter() - temporal_start
            new_frames = frame_index - resumed_from_frames
            frames_per_second = new_frames / elapsed_seconds if elapsed_seconds > 0 else None
            remaining_frames = args.frames - frame_index
            progress = {
                "schema_version": "ovrtx-temporal-progress/v1",
                "scene": args.scene,
                "frames": frame_index,
                "target_frames": args.frames,
                "resumed_from_frames": resumed_from_frames,
                "elapsed_seconds_this_run": elapsed_seconds,
                "frames_per_second_this_run": frames_per_second,
                "eta_seconds": (remaining_frames / frames_per_second if frames_per_second else None),
            }
            (args.output / "progress.json").write_text(
                json.dumps(progress, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                "OVRTX_TEMPORAL_PROGRESS " + json.dumps(progress, sort_keys=True),
                flush=True,
            )

        if frame_index not in checkpoints:
            continue
        alpha = alpha_sum / frame_index
        rgb = rgb_sum / frame_index
        depth = depth_numerator / depth_weight.clamp_min(torch.finfo(depth_numerator.dtype).eps)
        accumulated_valid_depth = depth_weight[..., 0] > 0
        depth[..., 0] = torch.where(
            accumulated_valid_depth,
            depth[..., 0],
            torch.full_like(depth[..., 0], float("inf")),
        )
        semantic_vote_counts = semantic_counts.sum(dim=1)
        semantic = semantic_counts.argmax(dim=1).view(
            1,
            args.height,
            args.width,
        )
        semantic = semantic.to(torch.int64)
        semantic[
            (alpha[..., 0] == 0)
            | (
                semantic_vote_counts.view(
                    1,
                    args.height,
                    args.width,
                )
                == 0
            )
        ] = -1
        candidate = as_render_output(
            rgb=rgb.detach().cpu().numpy(),
            alpha=alpha[..., 0].detach().cpu().numpy(),
            depth=depth[..., 0].detach().cpu().numpy(),
            semantic=semantic.detach().cpu().numpy(),
            camera_bundle_id=fidelity_bundle.bundle_id,
            source=f"ovrtx-temporal-average-{frame_index}",
            valid_depth=(accumulated_valid_depth.detach().cpu().numpy()),
        )
        if args.save_checkpoint_outputs:
            save_render_output(
                args.output / f"candidate-ovrtx-temporal-{frame_index:06d}.npz",
                candidate,
            )
        if not args.final_report_only or frame_index == args.frames:
            report = compare_render_outputs(
                reference=reference,
                candidate=candidate,
                camera_bundle=fidelity_bundle,
                output_dir=(args.output / f"report-{frame_index:06d}" if frame_index == args.frames else None),
                config_id=(f"ovrtx-{args.projection_mode}-{args.sorting_mode}-temporal-{frame_index}-vs-custom"),
                require_lpips=True,
                max_artifact_views=1,
            )
            reports[str(frame_index)] = {
                "pass": report["pass"],
                "aggregate": report["aggregate"],
            }
        final_candidate = candidate

    assert final_candidate is not None
    save_render_output(args.output / "candidate-ovrtx-temporal.npz", final_candidate)
    accumulator_path = None
    if args.save_final_accumulators:
        accumulator_path = args.output / "accumulators-final.npz"
        np.savez_compressed(
            accumulator_path,
            frames=np.asarray(args.frames, dtype=np.int64),
            width=np.asarray(args.width, dtype=np.int64),
            height=np.asarray(args.height, dtype=np.int64),
            camera_bundle_id=np.asarray(fidelity_bundle.bundle_id),
            scene_checksum=np.asarray(scene_manifest["checksum_sha256"]),
            projection_mode=np.asarray(args.projection_mode),
            sorting_mode=np.asarray(args.sorting_mode),
            semantic_group_count=np.asarray(
                semantic_group_count,
                dtype=np.int64,
            ),
            semantic_scheme=np.asarray(args.semantic_scheme),
            semantic_grid=np.asarray(args.semantic_grid, dtype=np.int64),
            rgb_sum=rgb_sum.detach().cpu().numpy(),
            alpha_sum=alpha_sum.detach().cpu().numpy(),
            depth_numerator=(depth_numerator.detach().cpu().numpy()),
            depth_weight=depth_weight.detach().cpu().numpy(),
            semantic_counts=semantic_counts.detach().cpu().numpy(),
        )
    summary = {
        "schema_version": "ovrtx-temporal-fidelity/v1",
        "frames": args.frames,
        "resumed_from_frames": resumed_from_frames,
        "resume_sample_sequence_advanced_frames": resumed_from_frames,
        "resume_sample_sequence_advance_seconds": resume_advance_seconds,
        "resume_accumulators": (str(args.resume_accumulators) if args.resume_accumulators is not None else None),
        "final_accumulators": (str(accumulator_path) if accumulator_path is not None else None),
        "warmup": args.warmup,
        "scene": scene_manifest,
        "scene_loading": scene_loading,
        "scene_id": args.scene,
        "gaussian_count": gaussian_count,
        "scene_tensor_sha256": scene_tensor_checksum,
        "camera_bundle_id": fidelity_bundle.bundle_id,
        "ovrtx_fractional_opacity": True,
        "ovrtx_keep_system_alive": True,
        "ovrtx_cleanup_scope": "python_renderer_handle_and_async_upload_buffers",
        "ovrtx_projection_mode_hint": args.projection_mode,
        "ovrtx_projection_mode_hint_is_advisory": True,
        "ovrtx_projection_mode_observed": runtime_readback["projection_mode"]["observed"][0],
        "ovrtx_sorting_mode": args.sorting_mode,
        "ovrtx_sorting_mode_observed": runtime_readback["sorting_mode"]["observed"][0],
        "runtime_token_readback": runtime_readback,
        "semantic_scheme": args.semantic_scheme,
        "semantic_group_count": semantic_group_count,
        "semantic_grid": list(args.semantic_grid),
        "ovrtx_color_space": "display_srgb",
        "custom_color_transform": "identity_authored_display_rgb",
        "custom_gaussian_support_sigma": args.custom_support_sigma,
        "custom_covariance_epsilon": args.custom_covariance_epsilon,
        "custom_semantic_min_alpha": args.custom_semantic_min_alpha,
        "custom_ray_gaussian_evaluation": (args.custom_ray_gaussian_evaluation),
        "custom_deterministic": args.custom_deterministic,
        "custom_tile_size": custom_tile_size,
        "custom_visible_capacity": custom_visible_capacity,
        "custom_intersection_capacity": custom_intersection_capacity,
        "custom_depth_bucket_count": custom_depth_bucket_count,
        "custom_depth_bucket_group_size": custom_depth_bucket_group_size,
        "custom_counters": custom_counters,
        "checkpoints": checkpoints,
        "checkpoint_outputs_saved": args.save_checkpoint_outputs,
        "final_report_only": args.final_report_only,
        "progress_every": args.progress_every,
        "reports": reports,
        "final_pass": reports[str(args.frames)]["pass"],
        "provenance": {
            "project_commit": git_commit(),
            "ovrtx_commit": implementation_commit(),
            "ovrtx_version": ovrtx.__version__,
            "renderer_version": list(renderer.version),
            "command": [sys.executable, *sys.argv],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "stage_sha256": hashlib.sha256(stage.encode("utf-8")).hexdigest(),
            "stage_file": str(stage_path),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    lifecycle = _OvrtxLifecycle()
    try:
        _run_temporal_capture(lifecycle)
    finally:
        cleanup = lifecycle.release()

    # This executes only after a successful capture body. On an exception, the
    # traceback may still retain frame locals; never replace that primary
    # failure with a secondary weak-reference assertion from the finally block.
    if not cleanup.renderer_was_created or not cleanup.python_renderer_released:
        raise RuntimeError("OVRTX temporal capture completed without releasing its Python renderer handle.")
    print(
        "OVRTX_TEMPORAL_CLEANUP_DETAILS "
        "python_renderer_released=true keep_system_alive=true "
        f"gc_collected={cleanup.collected_objects}",
        flush=True,
    )
    print(OVRTX_TEMPORAL_CLEANUP_MARKER, flush=True)
    print(OVRTX_TEMPORAL_SUCCESS_MARKER, flush=True)


if __name__ == "__main__":
    main()
