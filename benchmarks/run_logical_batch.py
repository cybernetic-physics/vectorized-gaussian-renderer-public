#!/usr/bin/env python3
# ruff: noqa: E402
"""Measure a frozen logical robotics batch through bounded physical microbatches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_trajectory import load_scene, semantic_output_from_top_contributors
from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import spatial_semantic_ids
from isaacsim_gaussian_renderer.evaluation.logical_batch import (
    audit_logical_camera_contract,
    audit_microbatch_schedule,
    deterministic_camera_order,
    stratified_source_indices,
)
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (
    CameraTrajectory,
    load_trajectory,
)

SCENE_ID = 404
SEMANTIC_MIN_ALPHA = 0.01
FULL_OUTPUTS = ("rgb", "depth", "alpha", "semantic_id")
GSPLAT_CORE_OUTPUTS = ("rgb", "depth", "alpha")


class LogicalBatchValidationError(AssertionError):
    """Keep mandatory validation evidence attached to a failed run."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument(
        "--renderer",
        choices=("custom", "gsplat", "gsplat-core"),
        required=True,
    )
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--scene-sha256", required=True)
    parser.add_argument("--physical-batch", type=int, required=True)
    parser.add_argument(
        "--semantic-topology",
        choices=("spatial-8", "interleaved-1024"),
        default="spatial-8",
    )
    parser.add_argument(
        "--sample-gaussians",
        type=int,
        default=0,
        help=(
            "Keep a deterministic stratified sample of this many real Gaussians; "
            "zero keeps the full scene."
        ),
    )
    parser.add_argument("--camera-order-seed", type=int, default=5601024)
    parser.add_argument("--warmup-logical-batches", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--materialize-projected-records",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-workspace-gib", type=float, default=20.0)
    parser.add_argument(
        "--visible-per-view",
        type=int,
        default=100_000,
        help="Initial adaptive visible-record capacity per physical view.",
    )
    parser.add_argument(
        "--intersections-per-view-at-128",
        type=int,
        default=170_000,
        help=(
            "Initial adaptive intersection capacity per physical view at 128x128; "
            "scaled by pixel count."
        ),
    )
    parser.add_argument(
        "--gsplat-source",
        type=Path,
        default=Path(os.environ.get("GSPLAT_SOURCE_PATH", "/workspace/src/gsplat")),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sample_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "stddev": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
            "ci95": math.nan,
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
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit(path: Path) -> str:
    return command_output(["git", "-C", str(path), "rev-parse", "HEAD"]) or "unknown"


def driver_process_memory_bytes() -> int | None:
    output = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    pid = str(os.getpid())
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0] == pid:
            return int(fields[1]) * 1024 * 1024
    return None


def deterministic_scene_subset(
    scene: dict[str, torch.Tensor],
    *,
    target_count: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source_count = int(scene["means"].shape[0])
    if target_count > source_count:
        raise ValueError("--sample-gaussians cannot exceed the source scene count.")
    if target_count == 0 or target_count == source_count:
        selection_sha256 = hashlib.sha256(
            f"all:{source_count}".encode("ascii")
        ).hexdigest()
        return scene, {
            "policy": "full-scene",
            "source_gaussians": source_count,
            "selected_gaussians": source_count,
            "selected_indices_sha256": selection_sha256,
        }
    selected_host = stratified_source_indices(source_count, target_count)
    selected = torch.from_numpy(selected_host).to(scene["means"].device)
    selected_hash = hashlib.sha256(
        np.ascontiguousarray(selected_host).tobytes()
    ).hexdigest()
    subset = {
        name: tensor.index_select(0, selected).contiguous()
        for name, tensor in scene.items()
    }
    metadata = {
        "policy": "deterministic-stratified-source-index-v1",
        "source_gaussians": source_count,
        "selected_gaussians": target_count,
        "selected_indices_sha256": selected_hash,
        "source_index_range": {
            "minimum": int(selected_host[0]),
            "maximum": int(selected_host[-1]),
        },
    }
    del selected
    torch.cuda.empty_cache()
    return subset, metadata


def apply_semantic_topology(
    scene: dict[str, torch.Tensor],
    topology: str,
) -> dict[str, Any]:
    if topology == "spatial-8":
        semantic_ids = spatial_semantic_ids(
            scene["means"],
            grid=(2, 2, 2),
            dtype=torch.int64,
        )
        rule = "axis-aligned-spatial-grid-2x2x2"
    elif topology == "interleaved-1024":
        semantic_ids = scene["semantic_ids"].to(torch.int64).contiguous()
        rule = "source-index-modulo-1024"
    else:  # pragma: no cover - argparse keeps callers honest.
        raise ValueError(f"Unknown semantic topology {topology!r}.")
    scene["semantic_ids"] = semantic_ids
    occupied = torch.unique(semantic_ids, sorted=True)
    return {
        "topology": topology,
        "rule": rule,
        "class_count": int(occupied.numel()),
        "minimum_id": int(occupied[0].item()),
        "maximum_id": int(occupied[-1].item()),
    }


class Adapter(Protocol):
    name: str
    required_outputs: tuple[str, ...]

    def render(
        self,
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...

    def synchronize(self) -> None: ...

    def counters(self) -> dict[str, Any]: ...

    def memory(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class CustomAdapter:
    name = "custom-cuda"
    required_outputs = FULL_OUTPUTS

    def __init__(
        self,
        args: argparse.Namespace,
        trajectory: CameraTrajectory,
        scene: dict[str, torch.Tensor],
    ) -> None:
        resolution_scale = trajectory.width * trajectory.height / float(128 * 128)
        self.backend = CustomCudaBackend(
            max_visible_records=args.physical_batch * args.visible_per_view,
            max_intersections=math.ceil(
                args.physical_batch
                * args.intersections_per_view_at_128
                * resolution_scale
            ),
            near_plane=trajectory.near_plane,
            far_plane=trajectory.far_plane,
            gaussian_support_sigma=3.0,
            covariance_epsilon=0.0,
            semantic_min_alpha=SEMANTIC_MIN_ALPHA,
            tile_size=1,
            depth_bucket_count=128,
            depth_bucket_group_size=8,
            compact_projection_cache=True,
            materialize_projected_records=args.materialize_projected_records,
            enable_projection_cache=False,
            output_srgb=False,
            deterministic=False,
            max_workspace_bytes=int(args.max_workspace_gib * 1024**3),
        )
        self.service = RendererService(
            self.backend,
            height=trajectory.height,
            width=trajectory.width,
            max_views=args.physical_batch,
        )
        self.service.initialize(stage=None, device="cuda")
        self.service.load_scene(
            SCENE_ID,
            means=scene["means"],
            scales=scene["scales"],
            rotations=scene["quats"],
            opacities=scene["opacities"],
            features=scene["colors"],
            semantic_ids=scene["semantic_ids"],
        )
        self.scene_ids = torch.full(
            (args.physical_batch,),
            SCENE_ID,
            device="cuda",
            dtype=torch.int64,
        )
        self.outputs: dict[str, torch.Tensor] | None = None

    def render(
        self,
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.outputs = self.service.render(
            viewmats,
            intrinsics,
            self.scene_ids,
            outputs=self.outputs,
        )
        return self.outputs

    def synchronize(self) -> None:
        self.service.synchronize()

    def counters(self) -> dict[str, Any]:
        return {
            **self.backend.read_counters(synchronize=False),
            "projection_cache": self.backend.projection_cache_stats,
            "capacity_adaptation": self.backend.capacity_stats,
        }

    def memory(self) -> dict[str, Any]:
        return {
            "scene_bytes": self.backend.scene_bytes,
            "workspace_bytes": self.backend.workspace_bytes,
            "workspace_logical_bytes_by_tensor": (
                self.backend.workspace_logical_bytes_by_tensor
            ),
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "driver_process_memory_bytes": driver_process_memory_bytes(),
        }

    def close(self) -> None:
        self.service.shutdown()


class GsplatAdapter:
    def __init__(
        self,
        args: argparse.Namespace,
        trajectory: CameraTrajectory,
        scene: dict[str, torch.Tensor],
    ) -> None:
        from gsplat.rendering import rasterization

        self.args = args
        self.trajectory = trajectory
        self.scene = scene
        self.rasterization = rasterization
        self.include_semantic = args.renderer == "gsplat"
        if self.include_semantic:
            from gsplat.cuda._wrapper import (
                rasterize_top_contributing_gaussian_ids,
            )

            self.top_contributors = rasterize_top_contributing_gaussian_ids
            self.name = "pinned-gsplat"
            self.required_outputs = FULL_OUTPUTS
        else:
            self.top_contributors = None
            self.name = "pinned-gsplat-rgb-depth-alpha"
            self.required_outputs = GSPLAT_CORE_OUTPUTS
        self.outputs: dict[str, torch.Tensor] | None = None

    def render(
        self,
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rgbd, alpha, metadata = self.rasterization(
            self.scene["means"],
            self.scene["quats"],
            self.scene["scales"],
            self.scene["opacities"],
            self.scene["colors"],
            viewmats,
            intrinsics,
            self.trajectory.width,
            self.trajectory.height,
            near_plane=self.trajectory.near_plane,
            far_plane=self.trajectory.far_plane,
            eps2d=0.0,
            radius_clip=0.0,
            packed=False,
            tile_size=16,
            render_mode="RGB+ED",
            rasterize_mode="classic",
            global_z_order=True,
        )
        self.outputs = {
            "rgb": rgbd[..., :3].contiguous(),
            "depth": rgbd[..., 3:4].contiguous(),
            "alpha": alpha.contiguous(),
        }
        if self.include_semantic:
            if self.top_contributors is None:  # pragma: no cover - constructor invariant.
                raise AssertionError("gsplat semantic adapter is incomplete.")
            contributor_ids, _ = self.top_contributors(
                metadata["means2d"],
                metadata["conics"],
                metadata["opacities"],
                metadata["isect_offsets"],
                metadata["flatten_ids"],
                self.trajectory.width,
                self.trajectory.height,
                metadata["tile_size"],
                1,
            )
            semantic = semantic_output_from_top_contributors(
                contributor_ids,
                alpha,
                self.scene["semantic_ids"],
            )
            self.outputs["semantic_id"] = semantic.contiguous()
        return self.outputs

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def counters(self) -> dict[str, Any]:
        return {
            "projection_cache_enabled": False,
            "capacity_overflow_interface": "not-exposed-by-pinned-gsplat",
        }

    def memory(self) -> dict[str, Any]:
        return {
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "driver_process_memory_bytes": driver_process_memory_bytes(),
        }

    def close(self) -> None:
        self.outputs = None


def output_summary(
    outputs: dict[str, torch.Tensor],
    *,
    required_outputs: tuple[str, ...] = FULL_OUTPUTS,
) -> dict[str, torch.Tensor]:
    if set(outputs) != set(required_outputs):
        raise AssertionError(f"Incomplete output contract: {sorted(outputs)}")
    rgb = outputs["rgb"]
    depth = outputs["depth"][..., 0]
    alpha = outputs["alpha"][..., 0]
    foreground = alpha > 0
    batch = int(alpha.shape[0])

    def flatten(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, -1)

    summary = {
        "foreground_pixels": flatten(foreground).sum(dim=1),
        "rgb_finite": flatten(torch.isfinite(rgb)).all(dim=1),
        "alpha_finite": flatten(torch.isfinite(alpha)).all(dim=1),
        "alpha_in_range": flatten((alpha >= 0) & (alpha <= 1)).all(dim=1),
        "foreground_depth_finite": flatten(
            torch.isfinite(depth) | ~foreground
        ).all(dim=1),
        "alpha_min": alpha.amin().reshape(1),
        "alpha_max": alpha.amax().reshape(1),
    }
    if "semantic_id" in required_outputs:
        semantic = outputs["semantic_id"][..., 0]
        semantic_foreground = alpha >= SEMANTIC_MIN_ALPHA
        summary.update(
            {
                "foreground_semantic_valid": flatten(
                    (semantic >= 0) | ~semantic_foreground
                ).all(dim=1),
                "background_semantic_valid": flatten(
                    (semantic == -1) | semantic_foreground
                ).all(dim=1),
            }
        )
    return summary


def execute_logical_batch(
    adapter: Adapter,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    physical_batch: int,
    collect_validation: bool,
) -> list[dict[str, torch.Tensor]]:
    summaries: list[dict[str, torch.Tensor]] = []
    for start in range(0, int(viewmats.shape[0]), physical_batch):
        outputs = adapter.render(
            viewmats[start : start + physical_batch],
            intrinsics[start : start + physical_batch],
        )
        if collect_validation:
            summaries.append(
                output_summary(
                    outputs,
                    required_outputs=adapter.required_outputs,
                )
            )
    return summaries


def combine_output_audit(
    summaries: list[dict[str, torch.Tensor]],
    camera_order: np.ndarray,
    *,
    required_outputs: tuple[str, ...] = FULL_OUTPUTS,
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("Output validation produced no summaries.")
    per_camera_names = [
        "foreground_pixels",
        "rgb_finite",
        "alpha_finite",
        "alpha_in_range",
        "foreground_depth_finite",
    ]
    if "semantic_id" in required_outputs:
        per_camera_names.extend(("foreground_semantic_valid", "background_semantic_valid"))
    combined = {
        name: torch.cat([summary[name] for summary in summaries]).cpu().numpy()
        for name in per_camera_names
    }
    if any(values.shape != (camera_order.size,) for values in combined.values()):
        raise AssertionError("Output audit did not cover the exact logical batch.")
    failed_camera_ids: dict[str, list[int]] = {}
    failed_camera_counts: dict[str, int] = {}
    for name in per_camera_names[1:]:
        failed = np.flatnonzero(~combined[name].astype(bool))
        failed_camera_counts[name] = int(failed.size)
        failed_camera_ids[name] = [
            int(value) for value in camera_order[failed[:32]]
        ]
    empty = np.flatnonzero(combined["foreground_pixels"] <= 0)
    failed_camera_counts["empty_foreground"] = int(empty.size)
    failed_camera_ids["empty_foreground"] = [
        int(value) for value in camera_order[empty[:32]]
    ]
    alpha_min = min(float(summary["alpha_min"].item()) for summary in summaries)
    alpha_max = max(float(summary["alpha_max"].item()) for summary in summaries)
    passed = not empty.size and all(
        bool(values.astype(bool).all())
        for name, values in combined.items()
        if name != "foreground_pixels"
    )
    return {
        "schema_version": "logical-output-audit/v1",
        "required_outputs": list(required_outputs),
        "validated_camera_count": int(camera_order.size),
        "minimum_foreground_pixels_per_camera": int(
            combined["foreground_pixels"].min()
        ),
        "maximum_foreground_pixels_per_camera": int(
            combined["foreground_pixels"].max()
        ),
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "failed_camera_counts": failed_camera_counts,
        "failed_camera_ids": failed_camera_ids,
        "pass": bool(passed),
    }


def custom_overflow_free(counters: dict[str, Any]) -> bool:
    return (
        int(counters.get("visible_overflow", 0)) == 0
        and int(counters.get("intersection_overflow", 0)) == 0
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(
        args.physical_batch,
        args.repetitions,
        args.visible_per_view,
        args.intersections_per_view_at_128,
    ) <= 0:
        raise ValueError("All batch, repetition, and capacity counts must be positive.")
    if args.warmup_logical_batches < 0 or args.max_workspace_gib <= 0:
        raise ValueError("Warmup must be nonnegative and workspace must be positive.")
    if args.sample_gaussians < 0:
        raise ValueError("--sample-gaussians cannot be negative.")
    if args.renderer.startswith("gsplat") and not args.materialize_projected_records:
        raise ValueError(
            "Projected-record selection is a custom-renderer variant and must not alter gsplat."
        )

    trajectory = load_trajectory(args.trajectory)
    if trajectory.scene_sha256 != args.scene_sha256:
        raise ValueError("Trajectory and CLI scene checksums differ.")
    camera_audit = audit_logical_camera_contract(trajectory)
    if not camera_audit["pass"]:
        raise AssertionError(f"Camera contract failed: {camera_audit}")
    camera_order, order_sha256 = deterministic_camera_order(
        trajectory.batch,
        seed=args.camera_order_seed,
    )
    schedule_audit = audit_microbatch_schedule(
        camera_order,
        physical_batch=args.physical_batch,
    )
    if not schedule_audit["pass"]:
        raise AssertionError(f"Microbatch schedule failed: {schedule_audit}")
    if order_sha256 != schedule_audit["camera_order_sha256"]:
        raise AssertionError("Camera order checksums disagree.")

    load_start = time.perf_counter()
    scene, source_scene_metadata = load_scene(args.scene_path, args.scene_sha256)
    scene, subset_metadata = deterministic_scene_subset(
        scene,
        target_count=args.sample_gaussians,
    )
    semantic_metadata = apply_semantic_topology(scene, args.semantic_topology)
    torch.cuda.synchronize()
    scene_preparation_seconds = time.perf_counter() - load_start
    scene_identity_payload = {
        "identity_policy": "content-and-selection-only-v1",
        "source_sha256": args.scene_sha256,
        "source_gaussian_count": source_scene_metadata["gaussian_count"],
        "active_gaussian_count": int(scene["means"].shape[0]),
        "selection": subset_metadata,
        "semantics": semantic_metadata,
    }
    scene_identity = {
        **scene_identity_payload,
        "derived_identity_sha256": hashlib.sha256(
            json.dumps(scene_identity_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "preparation_seconds": scene_preparation_seconds,
    }

    order_tensor = torch.from_numpy(camera_order).to("cuda")
    viewmats = torch.from_numpy(trajectory.viewmats).to("cuda").index_select(
        1,
        order_tensor,
    ).contiguous()
    intrinsics = torch.from_numpy(trajectory.intrinsics).to("cuda").index_select(
        1,
        order_tensor,
    ).contiguous()
    del order_tensor

    adapter: Adapter | None = None
    try:
        if args.renderer == "custom":
            adapter = CustomAdapter(args, trajectory, scene)
        else:
            adapter = GsplatAdapter(args, trajectory, scene)

        for _ in range(args.warmup_logical_batches):
            execute_logical_batch(
                adapter,
                viewmats[0],
                intrinsics[0],
                physical_batch=args.physical_batch,
                collect_validation=False,
            )
            adapter.synchronize()

        validation_steps: list[dict[str, Any]] = []
        for step in range(trajectory.timesteps):
            summaries = execute_logical_batch(
                adapter,
                viewmats[step],
                intrinsics[step],
                physical_batch=args.physical_batch,
                collect_validation=True,
            )
            adapter.synchronize()
            output_audit = combine_output_audit(
                summaries,
                camera_order,
                required_outputs=adapter.required_outputs,
            )
            counters = adapter.counters()
            zero_overflow = (
                custom_overflow_free(counters)
                if args.renderer == "custom"
                else True
            )
            validation_steps.append(
                {
                    "step": step,
                    "output_audit": output_audit,
                    "zero_overflow": zero_overflow,
                    "pass": output_audit["pass"] and zero_overflow,
                }
            )
        validation_pass = all(item["pass"] for item in validation_steps)
        if not validation_pass:
            raise LogicalBatchValidationError(
                "One or more mandatory logical output audits failed.",
                {
                    "camera_contract": {
                        "audit": camera_audit,
                        "schedule": schedule_audit,
                    },
                    "scene": scene_identity,
                    "validation": {
                        "all_steps_pass": False,
                        "steps": validation_steps,
                    },
                },
            )

        torch.cuda.reset_peak_memory_stats()
        gpu_samples_ms: list[float] = []
        wall_samples_ms: list[float] = []
        measured_counters: list[dict[str, Any]] = []
        cuda_profiler_range = os.environ.get("VGR_CUDA_PROFILER_RANGE") == "1"
        if cuda_profiler_range:
            torch.cuda.profiler.start()
        try:
            for repetition in range(args.repetitions):
                for step in range(trajectory.timesteps):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    wall_start = time.perf_counter()
                    torch.cuda.nvtx.range_push(
                        f"logical-batch/{args.renderer}/p{args.physical_batch}/step-{step}"
                    )
                    start.record()
                    execute_logical_batch(
                        adapter,
                        viewmats[step],
                        intrinsics[step],
                        physical_batch=args.physical_batch,
                        collect_validation=False,
                    )
                    end.record()
                    end.synchronize()
                    torch.cuda.nvtx.range_pop()
                    gpu_samples_ms.append(float(start.elapsed_time(end)))
                    wall_samples_ms.append((time.perf_counter() - wall_start) * 1000.0)
                    counters = adapter.counters()
                    if args.renderer == "custom" and not custom_overflow_free(counters):
                        raise RuntimeError(f"Custom capacity overflow: {counters}")
                    measured_counters.append(counters)
        finally:
            if cuda_profiler_range:
                torch.cuda.profiler.stop()

        gpu_stats = sample_stats(gpu_samples_ms)
        wall_stats = sample_stats(wall_samples_ms)
        logical_batch = trajectory.batch
        final_counters = adapter.counters()
        memory = adapter.memory()
        result = {
            "schema_version": "logical-robotics-batch/v1",
            "pass": (
                camera_audit["pass"]
                and schedule_audit["pass"]
                and validation_pass
                and (
                    custom_overflow_free(final_counters)
                    if args.renderer == "custom"
                    else True
                )
            ),
            "environment": {
                "project_commit": os.environ.get("SOURCE_GIT_COMMIT")
                or git_commit(PROJECT_ROOT),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "driver": command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ]
                ),
                "gsplat_commit": (
                    git_commit(args.gsplat_source)
                    if args.renderer.startswith("gsplat")
                    else None
                ),
            },
            "configuration": {
                "renderer": adapter.name,
                "logical_batch": logical_batch,
                "physical_batch": args.physical_batch,
                "microbatches_per_logical_batch": logical_batch
                // args.physical_batch,
                "width": trajectory.width,
                "height": trajectory.height,
                "outputs": list(adapter.required_outputs),
                "comparison_contract": (
                    "full-output-matched"
                    if adapter.required_outputs == FULL_OUTPUTS
                    else "unmatched-rgb-depth-alpha-only-control"
                ),
                "semantic_min_alpha": (
                    SEMANTIC_MIN_ALPHA
                    if "semantic_id" in adapter.required_outputs
                    else None
                ),
                "projection_cache_enabled": False,
                "materialize_projected_records": (
                    args.materialize_projected_records
                    if args.renderer == "custom"
                    else None
                ),
                "warmup_logical_batches": args.warmup_logical_batches,
                "repetitions": args.repetitions,
                "measured_camera_steps": trajectory.timesteps,
                "cuda_profiler_range": cuda_profiler_range,
                "max_workspace_gib": (
                    args.max_workspace_gib
                    if args.renderer == "custom"
                    else None
                ),
                "initial_visible_records_per_view": (
                    args.visible_per_view
                    if args.renderer == "custom"
                    else None
                ),
                "initial_intersections_per_view_at_128": (
                    args.intersections_per_view_at_128
                    if args.renderer == "custom"
                    else None
                ),
            },
            "camera_contract": {
                "trajectory_id": trajectory.trajectory_id,
                "trajectory_path": args.trajectory.name,
                "audit": camera_audit,
                "schedule": schedule_audit,
                "camera_order_seed": args.camera_order_seed,
            },
            "scene": scene_identity,
            "validation": {
                "all_steps_pass": validation_pass,
                "steps": validation_steps,
            },
            "timing": {
                "contract": (
                    "CUDA events and synchronized wall time around one complete logical "
                    "B1024 observation; all physical microbatches and required outputs included"
                ),
                "logical_batch_gpu_ms": gpu_stats,
                "logical_batch_wall_ms": wall_stats,
                "gpu_images_per_second": logical_batch * 1000.0
                / gpu_stats["mean"],
                "wall_images_per_second": logical_batch * 1000.0
                / wall_stats["mean"],
                "gpu_ms_per_image": gpu_stats["mean"] / logical_batch,
                "wall_ms_per_image": wall_stats["mean"] / logical_batch,
                "native_submissions_per_logical_batch": logical_batch
                // args.physical_batch,
            },
            "counters": {
                "final": final_counters,
                "per_logical_batch": measured_counters,
            },
            "memory": memory,
            "claim_boundary": (
                "Direct standalone renderer with an immutable robotics camera contract; "
                "not end-to-end Isaac, Kit, RTX, or OVRTX timing"
            ),
        }
        return result
    finally:
        if adapter is not None:
            adapter.close()


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except BaseException as error:
        diagnostics = getattr(error, "diagnostics", None)
        failure = {
            "schema_version": "logical-robotics-batch/v1",
            "pass": False,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                **(
                    {"diagnostics": diagnostics}
                    if diagnostics is not None
                    else {}
                ),
            },
            "configuration": {
                "renderer": args.renderer,
                "physical_batch": args.physical_batch,
                "semantic_topology": args.semantic_topology,
                "sample_gaussians": args.sample_gaussians,
                "materialize_projected_records": args.materialize_projected_records,
                "initial_visible_records_per_view": args.visible_per_view,
                "initial_intersections_per_view_at_128": (
                    args.intersections_per_view_at_128
                ),
            },
            "environment": {
                "project_commit": os.environ.get("SOURCE_GIT_COMMIT")
                or git_commit(PROJECT_ROOT),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "torch_allocated_bytes": (
                    torch.cuda.memory_allocated()
                    if torch.cuda.is_available()
                    else None
                ),
                "torch_reserved_bytes": (
                    torch.cuda.memory_reserved()
                    if torch.cuda.is_available()
                    else None
                ),
                "driver_process_memory_bytes": driver_process_memory_bytes(),
            },
        }
        write_result(args.output, failure)
        print("LOGICAL_ROBOTICS_BATCH_FAILED", json.dumps(failure, sort_keys=True))
        raise
    write_result(args.output, result)
    print("LOGICAL_ROBOTICS_BATCH_OK", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
