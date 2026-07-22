#!/usr/bin/env python3
"""Run cache-explicit spatial or temporal Gaussian-renderer trajectories.

Measured timing covers renderer submission through GPU completion. Scene load,
JIT/shader warmup, CPU readback, image encoding, video encoding, and artifact
hashing are reported separately and never included in renderer throughput.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    file_sha256,
    spatial_semantic_ids,
)
from isaacsim_gaussian_renderer.evaluation.artifacts import EvaluationArtifactBundle
from isaacsim_gaussian_renderer.evaluation.scenarios import ScenarioSpec, build_scenario
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (
    CameraTrajectory,
    canonical_json_bytes,
    load_trajectory,
    save_trajectory,
)
from isaacsim_gaussian_renderer.ply_loader import canonicalize_3dgs_scene, load_ply_to_gaussians

HOME_SCAN_COUNT = 21_497_908
HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
SCENE_ID = 404
SEMANTIC_MIN_ALPHA = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", type=Path)
    source.add_argument("--route", type=Path)
    parser.add_argument("--renderer", choices=("custom", "gsplat", "ovrtx", "isaac-fabric"), default="custom")
    parser.add_argument(
        "--scenario",
        choices=(
            "contract-exact",
            "static-repeat",
            "appearance-updates",
            "sequential-flyby",
            "phase-offset-vectorized-flyby",
            "mixed-motion",
            "intrinsics-sweep",
            "environment-transform-motion",
            "teleport",
            "active-subset-cadence",
            "multi-scene-trajectory",
        ),
        required=True,
    )
    parser.add_argument("--scene-path", type=Path, default=Path("/workspace/datasets/home-scan-lod0.ply"))
    parser.add_argument("--scene-sha256", default=HOME_SCAN_SHA256)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--focal-scale", type=float, default=0.72)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--minimum-duration-seconds",
        type=float,
        default=0.0,
        help="Continue complete repetitions until this measured wall duration is reached.",
    )
    parser.add_argument("--moved-fraction", type=float, default=1.0)
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--semantic-topology",
        choices=("spatial-8", "interleaved-1024"),
        default="interleaved-1024",
    )
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--capture-full-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist every rendered view in canonical render-output-v1 NPZ form for fidelity/route analysis.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/evals"))
    parser.add_argument("--run-id")
    parser.add_argument("--visible-per-view", type=int, default=500_000)
    parser.add_argument("--intersections-per-view-at-128", type=int, default=2_400_000)
    parser.add_argument(
        "--materialize-projected-records",
        action="store_true",
        help=(
            "Reuse compact screen-space projection records for intersection "
            "emission and compositing in the custom CUDA renderer."
        ),
    )
    parser.add_argument(
        "--max-workspace-gib",
        type=float,
        help=(
            "Bound custom-renderer workspace allocation. Projected-record "
            "reuse falls back to the recompute path if it cannot fit."
        ),
    )
    return parser.parse_args()


def _route_hash(payload: dict[str, Any]) -> str:
    content = dict(payload)
    expected = str(content.pop("route_sha256"))
    # Validation evidence is appendable metadata, not a trajectory-defining
    # input. Keep the route identity stable as measured validation improves.
    content.pop("validation", None)
    actual = __import__("hashlib").sha256(canonical_json_bytes(content)).hexdigest()
    if actual != expected:
        raise ValueError(f"Route SHA-256 mismatch: {actual} != {expected}.")
    return actual


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise ValueError("Route produces a zero-length look direction.")
    return vector / norm


def trajectory_from_route(args: argparse.Namespace) -> CameraTrajectory:
    route = json.loads(args.route.read_text(encoding="utf-8"))
    route_sha256 = _route_hash(route)
    points_xz = np.asarray(route["world_path_xz"], dtype=np.float64)
    segment = np.linalg.norm(np.diff(points_xz, axis=0), axis=1)
    keep = np.concatenate(([True], segment > 1.0e-8))
    points_xz = points_xz[keep]
    segment = np.linalg.norm(np.diff(points_xz, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    distance = np.linspace(0.0, cumulative[-1], args.frames)
    positions_xz = np.stack(
        [np.interp(distance, cumulative, points_xz[:, axis]) for axis in range(2)],
        axis=-1,
    )
    smoothing_window = int(route.get("position_smoothing_window", 1))
    if smoothing_window > 1:
        if smoothing_window % 2 == 0:
            raise ValueError("position_smoothing_window must be odd.")
        smoothing_window = min(smoothing_window, args.frames if args.frames % 2 else args.frames - 1)
        kernel = np.hanning(smoothing_window)
        kernel /= kernel.sum()
        radius = smoothing_window // 2
        unsmoothed = positions_xz.copy()
        positions_xz = np.stack(
            [
                np.convolve(
                    np.pad(unsmoothed[:, axis], (radius, radius), mode="edge"),
                    kernel,
                    mode="valid",
                )
                for axis in range(2)
            ],
            axis=-1,
        )
    centers = np.empty((args.frames, 3), dtype=np.float64)
    centers[:, 0] = positions_xz[:, 0]
    centers[:, 1] = float(route["eye_height"])
    centers[:, 2] = positions_xz[:, 1]
    lookahead = max(1, min(6, args.frames - 1))
    forward = np.arange(args.frames) + lookahead
    backward = np.arange(args.frames) - 1
    forward = np.minimum(forward, args.frames - 1)
    backward = np.maximum(backward, 0)
    direction = _normalize(centers[forward] - centers[backward])
    world_up = np.asarray([0.0, -1.0, 0.0])
    right = _normalize(np.cross(direction, world_up[None, :]))
    down = _normalize(np.cross(direction, right))
    camera_to_world = np.stack((right, down, direction), axis=-1)
    world_to_camera = np.swapaxes(camera_to_world, -1, -2)
    viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], args.frames, axis=0)
    viewmats[:, :3, :3] = world_to_camera.astype(np.float32)
    viewmats[:, :3, 3] = -np.einsum("tij,tj->ti", world_to_camera, centers).astype(np.float32)
    intrinsics = np.zeros((args.frames, 3, 3), dtype=np.float32)
    intrinsics[:, 0, 0] = args.focal_scale * args.width
    intrinsics[:, 1, 1] = args.focal_scale * args.height
    intrinsics[:, 0, 2] = args.width * 0.5
    intrinsics[:, 1, 2] = args.height * 0.5
    intrinsics[:, 2, 2] = 1.0
    return CameraTrajectory(
        viewmats=viewmats[:, None],
        intrinsics=intrinsics[:, None],
        scene_ids=np.asarray([SCENE_ID], dtype=np.int64),
        scene_id_broadcast="time",
        width=args.width,
        height=args.height,
        fps=args.fps,
        seed=args.seed,
        scene_sha256=args.scene_sha256,
        route_sha256=route_sha256,
        motion_classification=str(route.get("motion_classification", "moving-camera")),
        expected_cache_events=("miss",) * args.frames,
        near_plane=0.01,
        far_plane=100.0,
        provenance={"route_path": f"benchmarks/camera_paths/{args.route.name}"},
        route_validation=route.get("validation", {}),
    )


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            **{
                name: math.nan
                for name in (
                    "mean",
                    "stddev",
                    "p50",
                    "p95",
                    "p99",
                    "max",
                    "ci95",
                )
            },
            "count": 0,
            "samples": [],
        }
    array = np.asarray(values, dtype=np.float64)
    stddev = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "stddev": stddev,
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "ci95": float(1.96 * stddev / math.sqrt(array.size)),
        "samples": [float(value) for value in values],
    }


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def semantic_output_from_top_contributors(
    contributor_ids: torch.Tensor,
    alpha: torch.Tensor,
    scene_semantic_ids: torch.Tensor,
    *,
    semantic_min_alpha: float = SEMANTIC_MIN_ALPHA,
) -> torch.Tensor:
    """Apply the production alpha/background policy to gsplat semantics."""
    leading_id = contributor_ids[..., 0]
    semantic = torch.full(
        (*leading_id.shape, 1),
        -1,
        device=contributor_ids.device,
        dtype=torch.int64,
    )
    valid = (leading_id >= 0) & (alpha[..., 0] >= semantic_min_alpha)
    semantic[..., 0][valid] = scene_semantic_ids[leading_id[valid]]
    return semantic


@dataclass
class RenderSample:
    outputs: dict[str, torch.Tensor]
    gpu_ms: float
    wall_ms: float
    cache_event: str
    rendered_views: int
    native_submissions: int
    renderer_executions: int


class RendererAdapter(Protocol):
    name: str

    def render_step(self, step: int, scenario: ScenarioSpec) -> RenderSample: ...
    def synchronize(self) -> None: ...
    def counters(self) -> dict[str, Any]: ...
    def memory(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class CustomAdapter:
    name = "custom-cuda"

    def __init__(self, args: argparse.Namespace, scenario: ScenarioSpec, scene: dict[str, torch.Tensor]):
        resolution_scale = args.width * args.height / float(128 * 128)
        self.backend = CustomCudaBackend(
            max_visible_records=scenario.batch * args.visible_per_view,
            max_intersections=math.ceil(
                scenario.batch * args.intersections_per_view_at_128 * resolution_scale
            ),
            near_plane=0.01,
            far_plane=100.0,
            gaussian_support_sigma=3.0,
            covariance_epsilon=0.0,
            semantic_min_alpha=SEMANTIC_MIN_ALPHA,
            tile_size=1,
            depth_bucket_count=128,
            depth_bucket_group_size=8,
            compact_projection_cache=True,
            materialize_projected_records=(
                args.materialize_projected_records
            ),
            enable_projection_cache=args.cache,
            output_srgb=False,
            deterministic=False,
            max_workspace_bytes=(
                None
                if args.max_workspace_gib is None
                else int(args.max_workspace_gib * 1024**3)
            ),
        )
        self.service = RendererService(
            self.backend,
            height=args.height,
            width=args.width,
            max_views=scenario.batch,
        )
        self.service.initialize(stage=None, device="cuda")
        scene_id_values = (
            sorted(int(value) for value in np.unique(scenario.scene_ids))
            if scenario.name == "multi-scene-trajectory"
            else [SCENE_ID]
        )
        for scene_id in scene_id_values:
            self.service.load_scene(
                scene_id,
                means=scene["means"],
                scales=scene["scales"],
                rotations=scene["quats"],
                opacities=scene["opacities"],
                features=scene["colors"],
                semantic_ids=scene["semantic_ids"],
            )
        self._camera_inputs: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = []
        previous: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None
        for step in range(scenario.timesteps):
            if scenario.expected_cache_events[step] == "hit" and previous is not None:
                current = previous
            else:
                active = scenario.active_camera_ids[step] if scenario.active_camera_ids else None
                current = (
                    torch.from_numpy(scenario.viewmats[step]).to("cuda"),
                    torch.from_numpy(scenario.intrinsics[step]).to("cuda"),
                    torch.from_numpy(scenario.scene_ids[step]).to("cuda"),
                    torch.from_numpy(active).to("cuda") if active is not None else None,
                )
            self._camera_inputs.append(current)
            previous = current
        self._environment_transforms = (
            [
                torch.from_numpy(scenario.environment_transforms[step]).to("cuda")
                for step in range(scenario.timesteps)
            ]
            if scenario.environment_transforms is not None
            else None
        )
        self.outputs: dict[str, torch.Tensor] | None = None

    def render_step(self, step: int, scenario: ScenarioSpec) -> RenderSample:
        if scenario.expected_cache_events[step] == "skip":
            if self.outputs is None:
                raise RuntimeError("A cadence scenario cannot skip before its first render.")
            return RenderSample(self.outputs, 0.0, 0.0, "skip", 0, 0, 0)
        viewmats, intrinsics, scene_ids, active_ids = self._camera_inputs[step]
        if self._environment_transforms is not None:
            self.backend.update_scene_transforms(scene_ids, self._environment_transforms[step])
        if step in scenario.appearance_update_steps:
            packed = self.backend._packed_scene  # audited mutation target; projection topology is unchanged.
            packed["features"].add_(1.0 / 255.0).remainder_(1.0)
            packed["semantic_ids"].add_(1).remainder_(1024)
        before = self.backend.projection_cache_stats.copy()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        torch.cuda.nvtx.range_push(f"trajectory/{scenario.name}/step-{step:06d}")
        start_event.record()
        self.outputs = self.service.render(
            viewmats,
            intrinsics,
            scene_ids,
            active_camera_ids=active_ids,
            outputs=self.outputs,
        )
        end_event.record()
        self.service.synchronize()
        torch.cuda.nvtx.range_pop()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        after = self.backend.projection_cache_stats
        event = "hit" if int(after["hits"]) > int(before["hits"]) else "miss"
        rendered_views = int(active_ids.numel()) if active_ids is not None else scenario.batch
        return RenderSample(
            self.outputs,
            float(start_event.elapsed_time(end_event)),
            wall_ms,
            event,
            rendered_views,
            1,
            1,
        )

    def synchronize(self) -> None:
        self.service.synchronize()

    def counters(self) -> dict[str, Any]:
        # Preserve a complete failed bundle when capacity is exceeded; the
        # manifest gate below marks either overflow as a failure.
        return {
            **self.backend.read_counters(synchronize=False),
            **self.backend.projection_cache_stats,
            "capacity_adaptation": self.backend.capacity_stats,
        }

    def memory(self) -> dict[str, Any]:
        return {
            "scene_bytes": self.backend.scene_bytes,
            "workspace_bytes": self.backend.workspace_bytes,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "driver_process_memory": command_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
            ),
        }

    def close(self) -> None:
        self.service.shutdown()


class GsplatAdapter:
    """Pinned conventional EWA control with all required outputs enabled."""

    name = "pinned-gsplat"

    def __init__(self, args: argparse.Namespace, scenario: ScenarioSpec, scene: dict[str, torch.Tensor]):
        from gsplat.cuda._wrapper import rasterize_top_contributing_gaussian_ids
        from gsplat.rendering import rasterization

        self.args = args
        self.scene = scene
        self.rasterization = rasterization
        self.top_contributors = rasterize_top_contributing_gaussian_ids
        self.outputs: dict[str, torch.Tensor] | None = None
        self._executions = 0
        self._camera_inputs = [
            (
                torch.from_numpy(scenario.viewmats[step]).to("cuda"),
                torch.from_numpy(scenario.intrinsics[step]).to("cuda"),
                (
                    torch.from_numpy(scenario.active_camera_ids[step]).to("cuda")
                    if scenario.active_camera_ids and scenario.active_camera_ids[step] is not None
                    else None
                ),
            )
            for step in range(scenario.timesteps)
        ]

    def render_step(self, step: int, scenario: ScenarioSpec) -> RenderSample:
        if scenario.expected_cache_events[step] == "skip":
            if self.outputs is None:
                raise RuntimeError("A cadence scenario cannot skip before its first render.")
            return RenderSample(self.outputs, 0.0, 0.0, "skip", 0, 0, 0)
        viewmats, intrinsics, active_ids = self._camera_inputs[step]
        if active_ids is not None:
            viewmats = viewmats.index_select(0, active_ids)
            intrinsics = intrinsics.index_select(0, active_ids)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        torch.cuda.nvtx.range_push(f"trajectory/gsplat/{scenario.name}/step-{step:06d}")
        start_event.record()
        rgbd, alpha, metadata = self.rasterization(
            self.scene["means"],
            self.scene["quats"],
            self.scene["scales"],
            self.scene["opacities"],
            self.scene["colors"],
            viewmats,
            intrinsics,
            self.args.width,
            self.args.height,
            near_plane=0.01,
            far_plane=100.0,
            eps2d=0.0,
            radius_clip=0.0,
            packed=False,
            tile_size=16,
            render_mode="RGB+ED",
            rasterize_mode="classic",
            global_z_order=True,
        )
        contributor_ids, _ = self.top_contributors(
            metadata["means2d"],
            metadata["conics"],
            metadata["opacities"],
            metadata["isect_offsets"],
            metadata["flatten_ids"],
            self.args.width,
            self.args.height,
            metadata["tile_size"],
            1,
        )
        semantic = semantic_output_from_top_contributors(
            contributor_ids,
            alpha,
            self.scene["semantic_ids"],
        )
        self.outputs = {
            "rgb": rgbd[..., :3].contiguous(),
            "depth": rgbd[..., 3:4].contiguous(),
            "alpha": alpha.contiguous(),
            "semantic_id": semantic.contiguous(),
        }
        end_event.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        self._executions += 1
        rendered = int(viewmats.shape[0])
        return RenderSample(
            self.outputs,
            float(start_event.elapsed_time(end_event)),
            (time.perf_counter() - wall_start) * 1000.0,
            "disabled",
            rendered,
            1,
            1,
        )

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def counters(self) -> dict[str, Any]:
        return {"projection_cache_enabled": False, "renderer_executions": self._executions}

    def memory(self) -> dict[str, Any]:
        return {
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }

    def close(self) -> None:
        self.outputs = None


class UnsupportedIntegratedAdapter:
    """Fail closed when a dedicated Vulkan/SimulationApp runner was not used."""

    def __init__(self, renderer: str):
        raise RuntimeError(
            f"{renderer} trajectory execution requires its dedicated lifecycle runner; "
            "run_trajectory.py refuses to relabel CUDA-event timing as Vulkan or integrated Isaac timing."
        )


def load_scene(path: Path, expected_sha256: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    start = time.perf_counter()
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Scene SHA-256 mismatch: {actual_sha256} != {expected_sha256}.")
    raw = load_ply_to_gaussians(path)
    if expected_sha256 == HOME_SCAN_SHA256 and raw.count != HOME_SCAN_COUNT:
        raise ValueError(f"Home Scan count mismatch: {raw.count} != {HOME_SCAN_COUNT}.")
    canonical = canonicalize_3dgs_scene(raw, device="cuda")
    scene = {
        "means": canonical.means,
        "quats": canonical.rotations,
        "scales": canonical.scales,
        "opacities": canonical.opacities,
        "colors": canonical.features[:, :3].contiguous(),
        "semantic_ids": torch.arange(canonical.count, device="cuda", dtype=torch.int64).remainder_(1024),
    }
    torch.cuda.synchronize()
    return scene, {
        "sha256": actual_sha256,
        "gaussian_count": canonical.count,
        "load_and_upload_seconds": time.perf_counter() - start,
    }


def save_frame(path: Path, rgb: torch.Tensor) -> dict[str, float]:
    image = np.clip(rgb.detach().cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path, compress_level=4)
    normalized = image.astype(np.float64) / 255.0
    return {
        "nonblack_fraction": float(np.any(image > 2, axis=-1).mean()),
        "clipped_fraction": float(np.any(image >= 254, axis=-1).mean()),
        "mean": float(normalized.mean()),
        "std": float(normalized.std()),
    }


# Keep camera/scene tensor version counters available to the projection and
# capacity caches. Gradients remain disabled, while genuinely unversioned
# inference-mode inputs continue to fail closed inside the backend.
@torch.no_grad()
def main() -> None:
    args = parse_args()
    if (
        min(args.frames, args.batch, args.width, args.height, args.repetitions) <= 0
        or args.warmup < 0
        or args.minimum_duration_seconds < 0
    ):
        raise ValueError("Frames, batch, resolution, and repetitions must be positive; warmup must be nonnegative.")
    if args.max_workspace_gib is not None and args.max_workspace_gib <= 0:
        raise ValueError("--max-workspace-gib must be positive.")
    if args.materialize_projected_records and args.renderer != "custom":
        raise ValueError(
            "--materialize-projected-records is only valid with "
            "--renderer custom."
        )
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{args.renderer}-{args.scenario}-b{args.batch}"
    )
    bundle = EvaluationArtifactBundle.create(args.output_root / run_id)
    trajectory = load_trajectory(args.trajectory) if args.trajectory else trajectory_from_route(args)
    if args.scenario == "contract-exact":
        if args.trajectory is None:
            raise ValueError("contract-exact requires --trajectory; a route is not an exact measured contract.")
        exact_mismatches = {
            name: (actual, expected)
            for name, actual, expected in (
                ("frames", args.frames, trajectory.timesteps),
                ("batch", args.batch, trajectory.batch),
                ("width", args.width, trajectory.width),
                ("height", args.height, trajectory.height),
                ("fps", args.fps, trajectory.fps),
            )
            if actual != expected
        }
        if exact_mismatches:
            details = ", ".join(
                f"{name}={actual!r} (contract {expected!r})"
                for name, (actual, expected) in exact_mismatches.items()
            )
            raise ValueError(f"contract-exact CLI/trajectory mismatch: {details}.")
    scenario = build_scenario(
        trajectory,
        args.scenario,
        batch=args.batch,
        moved_fraction=args.moved_fraction,
        render_every_n=args.render_every_n,
    )
    active_camera_ids = None
    if any(item is not None for item in scenario.active_camera_ids):
        width = max((len(item) for item in scenario.active_camera_ids if item is not None), default=0)
        active_camera_ids = np.full((scenario.timesteps, width), -1, dtype=np.int64)
        for step, item in enumerate(scenario.active_camera_ids):
            if item is not None:
                active_camera_ids[step, : len(item)] = item
    measured_trajectory = CameraTrajectory(
        viewmats=scenario.viewmats,
        intrinsics=scenario.intrinsics,
        scene_ids=scenario.scene_ids,
        environment_transforms=scenario.environment_transforms,
        active_camera_ids=active_camera_ids,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seed=args.seed,
        scene_sha256=trajectory.scene_sha256,
        route_sha256=trajectory.route_sha256,
        motion_classification=scenario.name,
        expected_cache_events=scenario.expected_cache_events,
        near_plane=trajectory.near_plane,
        far_plane=trajectory.far_plane,
        near_far_policy=trajectory.near_far_policy,
        environment_phase_offsets=scenario.environment_phase_offsets,
        provenance=trajectory.provenance,
        route_validation=trajectory.route_validation,
    )
    save_trajectory(measured_trajectory, bundle.root / "trajectory.json")
    scene, scene_metadata = load_scene(args.scene_path, trajectory.scene_sha256)
    if args.semantic_topology == "spatial-8":
        scene["semantic_ids"] = spatial_semantic_ids(
            scene["means"],
            grid=(2, 2, 2),
            dtype=torch.int64,
        )
        semantic_rule = "axis-aligned-spatial-grid-2x2x2"
    else:
        semantic_rule = "source-index-modulo-1024"
    occupied_semantics = torch.unique(scene["semantic_ids"], sorted=True)
    scene_metadata["semantics"] = {
        "topology": args.semantic_topology,
        "rule": semantic_rule,
        "class_count": int(occupied_semantics.numel()),
        "minimum_id": int(occupied_semantics[0].item()),
        "maximum_id": int(occupied_semantics[-1].item()),
    }
    if args.renderer == "custom":
        adapter: RendererAdapter = CustomAdapter(args, scenario, scene)
    elif args.renderer == "gsplat":
        adapter = GsplatAdapter(args, scenario, scene)
    else:
        adapter = UnsupportedIntegratedAdapter(args.renderer)

    for _ in range(args.warmup):
        adapter.render_step(0, scenario)
    adapter.synchronize()
    # Warmup is outside the measured contract. Start the measured stream from a
    # known cache state so static-repeat records its required cold miss before
    # any hits, rather than inheriting a warmup hit.
    if isinstance(adapter, CustomAdapter):
        adapter.backend.invalidate_projection_cache()
    torch.cuda.reset_peak_memory_stats()
    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    cache_events: list[dict[str, Any]] = []
    frame_stats: list[dict[str, float]] = []
    full_outputs: dict[str, list[np.ndarray]] = {
        "rgb": [],
        "alpha": [],
        "depth": [],
        "semantic": [],
    }
    rendered_views = native_submissions = renderer_executions = 0
    artifact_start = time.perf_counter()
    last_outputs: dict[str, torch.Tensor] | None = None
    repetition = 0
    measurement_start = time.perf_counter()
    memory_checkpoints: list[dict[str, int]] = []
    while repetition < args.repetitions or (
        args.minimum_duration_seconds > 0
        and time.perf_counter() - measurement_start < args.minimum_duration_seconds
    ):
        for step in range(scenario.timesteps):
            sample = adapter.render_step(step, scenario)
            last_outputs = sample.outputs
            if sample.cache_event != "skip":
                gpu_samples.append(sample.gpu_ms)
                wall_samples.append(sample.wall_ms)
            rendered_views += sample.rendered_views
            native_submissions += sample.native_submissions
            renderer_executions += sample.renderer_executions
            expected = scenario.expected_cache_events[step]
            if repetition > 0 and args.scenario in {"static-repeat", "appearance-updates"} and step == 0:
                # Repetitions are one continuous renderer session. A fixed
                # camera remains coherent across repetition boundaries.
                expected = "hit"
            cache_events.append(
                {
                    "repetition": repetition,
                    "step": step,
                    "expected": expected,
                    "actual": sample.cache_event,
                    "pass": sample.cache_event == expected or sample.cache_event == "disabled",
                }
            )
            if args.save_frames and repetition == 0 and sample.rendered_views:
                frame_stats.append(
                    save_frame(bundle.root / "frames" / f"frame-{step:06d}.png", sample.outputs["rgb"][0])
                )
            if args.capture_full_output and repetition == 0 and sample.rendered_views:
                full_outputs["rgb"].append(sample.outputs["rgb"].detach().cpu().numpy())
                full_outputs["alpha"].append(sample.outputs["alpha"].detach().cpu().numpy())
                full_outputs["depth"].append(sample.outputs["depth"].detach().cpu().numpy())
                full_outputs["semantic"].append(sample.outputs["semantic_id"].detach().cpu().numpy())
        memory_checkpoints.append(
            {
                "repetition": repetition,
                "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
                "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
            }
        )
        repetition += 1
    adapter.synchronize()
    artifact_seconds = time.perf_counter() - artifact_start - sum(wall_samples) / 1000.0
    if last_outputs is None:
        raise RuntimeError("Scenario executed no renderer calls.")
    tensor_start = time.perf_counter()
    np.savez_compressed(
        bundle.root / "tensors" / "last-output.npz",
        **{name: value.detach().cpu().numpy() for name, value in last_outputs.items()},
    )
    complete_output_path = None
    if args.capture_full_output:
        if len(full_outputs["rgb"]) != scenario.rendered_requests:
            raise RuntimeError(
                "Full-output capture requires one tensor set per rendered request; "
                f"captured {len(full_outputs['rgb'])}, expected {scenario.rendered_requests}."
            )
        complete_output_path = "tensors/trajectory-output.npz"
        np.savez_compressed(
            bundle.root / complete_output_path,
            **{
                name: np.concatenate(values, axis=0)
                for name, values in full_outputs.items()
            },
            valid_depth=np.concatenate(full_outputs["alpha"], axis=0) > 0.01,
            color_space=np.asarray("linear-rgb"),
            camera_bundle_id=np.asarray(measured_trajectory.trajectory_id),
        )
    tensor_readback_encode_seconds = time.perf_counter() - tensor_start
    video_encoding_seconds = 0.0
    video_status = "not-requested"
    if args.save_frames and len(frame_stats) == scenario.timesteps:
        video_start = time.perf_counter()
        video_path = bundle.root / "videos" / "trajectory.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                str(bundle.root / "frames" / "frame-%06d.png"),
                "-frames:v",
                str(len(frame_stats)),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            check=True,
        )
        video_encoding_seconds = time.perf_counter() - video_start
        video_status = "encoded"
    elif args.save_frames:
        video_status = "skipped-noncontiguous-cadence"
    gpu = stats(gpu_samples)
    wall = stats(wall_samples)
    total_pixels = rendered_views * args.width * args.height
    timing = {
        "schema_version": "trajectory-timing-v1",
        "gpu_ms": gpu,
        "synchronized_wall_ms": wall,
        "images_per_second": rendered_views / (sum(gpu_samples) / 1000.0),
        "megapixels_per_second": total_pixels / 1.0e6 / (sum(gpu_samples) / 1000.0),
        "scene_loading_seconds": scene_metadata["load_and_upload_seconds"],
        "warmup_iterations": args.warmup,
        "cpu_readback_tensor_encoding_seconds": tensor_readback_encode_seconds,
        "frame_artifact_generation_seconds_approx": max(0.0, artifact_seconds),
        "video_encoding_seconds": video_encoding_seconds,
        "rendered_views": rendered_views,
        "api_requests": len(cache_events),
        "native_submissions": native_submissions,
        "renderer_executions": renderer_executions,
        "measured_repetitions": repetition,
        "measured_duration_seconds": time.perf_counter() - measurement_start,
    }
    counters = adapter.counters()
    memory = adapter.memory()
    adapter.close()
    bundle.write_json("timing.json", timing)
    bundle.write_json("cache-events.json", {"schema_version": "trajectory-cache-events-v1", "events": cache_events})
    bundle.write_json(
        "metrics/summary.json",
        {"schema_version": "trajectory-metrics-v1", "status": "not-compared", "reason": "single-renderer-run"},
    )
    manifest = {
        "schema_version": "trajectory-evaluation-run-v1",
        "run_id": run_id,
        "renderer": args.renderer,
        "renderer_configuration": {
            "materialize_projected_records_requested": (
                args.materialize_projected_records
            ),
            "max_workspace_gib": args.max_workspace_gib,
            "semantic_topology": args.semantic_topology,
        },
        "scenario": args.scenario,
        "execution_semantics": scenario.execution_semantics,
        "input_trajectory_id": trajectory.trajectory_id,
        "trajectory_id": measured_trajectory.trajectory_id,
        "scene": scene_metadata,
        "frame_count": len(frame_stats),
        "expected_rendered_frame_count": scenario.rendered_requests,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "frame_statistics": frame_stats,
        "video_status": video_status,
        "max_explained_duplicate_run": 1 if args.scenario != "static-repeat" else args.frames,
        "cited_frames": (
            ["frame-000000.png", sorted((bundle.root / "frames").glob("*.png"))[-1].name]
            if frame_stats
            else []
        ),
        "timing": "timing.json",
        "cache_events": "cache-events.json",
        "complete_render_output": complete_output_path,
        "memory": memory,
        "memory_checkpoints": memory_checkpoints,
        "allocation_growth_bytes": {
            "allocated": (
                memory_checkpoints[-1]["torch_allocated_bytes"]
                - memory_checkpoints[0]["torch_allocated_bytes"]
                if memory_checkpoints
                else 0
            ),
            "reserved": (
                memory_checkpoints[-1]["torch_reserved_bytes"]
                - memory_checkpoints[0]["torch_reserved_bytes"]
                if memory_checkpoints
                else 0
            ),
        },
        "counters": counters,
        "cache_expectations_pass": all(item["pass"] for item in cache_events),
        "environment": {
            "git_head": os.environ.get("SOURCE_GIT_COMMIT") or command_output(["git", "rev-parse", "HEAD"]),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "dependency_commits": {
                "gsplat": command_output(
                    ["git", "-C", os.environ.get("GSPLAT_SOURCE_PATH", "/workspace/src/gsplat"), "rev-parse", "HEAD"]
                ),
                "ovrtx_public_source": command_output(
                    ["git", "-C", "/workspace/src/ovrtx", "rev-parse", "HEAD"]
                ),
                "kit_app_template": command_output(
                    ["git", "-C", "/workspace/src/kit-app-template", "rev-parse", "HEAD"]
                ),
            },
        },
        "pass": all(item["pass"] for item in cache_events)
        and not counters.get("visible_overflow")
        and not counters.get("intersection_overflow"),
    }
    bundle.write_json("manifest.json", manifest)
    bundle.write_hash_manifest()
    print("TRAJECTORY_EVALUATION_OK " + json.dumps({"run_id": run_id, "manifest": str(bundle.root / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
