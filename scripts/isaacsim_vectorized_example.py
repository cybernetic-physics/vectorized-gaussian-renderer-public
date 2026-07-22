"""Headless Isaac Sim physics + Fabric + vectorized Gaussian renderer example."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import fmean

if __package__:
    from ._isaac_launch import close_simulation_app, isaac_app_config
else:
    from _isaac_launch import close_simulation_app, isaac_app_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--gaussians", type=int, default=10_000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--physics-steps", type=int, default=12)
    parser.add_argument(
        "--motion-mode",
        choices=("static", "fully-moving", "mixed-motion"),
        default="fully-moving",
    )
    parser.add_argument("--moving-fraction", type=float, default=0.5)
    parser.add_argument("--deadline-ms", type=float, default=16.6666667)
    parser.add_argument(
        "--visible-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--gaussian-support-sigma", type=float, default=2.0)
    parser.add_argument("--semantic-min-alpha", type=float, default=0.01)
    parser.add_argument("--depth-bucket-count", type=int, default=32)
    parser.add_argument("--depth-bucket-group-size", type=int, default=8)
    parser.add_argument(
        "--covariance-epsilon",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument(
        "--ray-gaussian-evaluation",
        action="store_true",
    )
    parser.add_argument(
        "--enable-projection-cache",
        action="store_true",
    )
    parser.add_argument(
        "--compact-projection-cache",
        action="store_true",
    )
    parser.add_argument(
        "--materialize-projected-records",
        action="store_true",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--camera-contract",
        type=Path,
        help=(
            "Published Gaussian flyby camera NPZ. Supplying this enables the "
            "real-scene Isaac/Fabric trajectory evaluator."
        ),
    )
    parser.add_argument("--camera-manifest", type=Path)
    parser.add_argument("--scene-path", type=Path)
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--scene-id-label")
    parser.add_argument("--scene-author")
    parser.add_argument("--scene-license")
    parser.add_argument("--scene-source")
    parser.add_argument("--trajectory-frames", type=int)
    parser.add_argument("--trajectory-frame-stride", type=int, default=1)
    parser.add_argument("--trajectory-start-frame", type=int, default=0)
    parser.add_argument(
        "--trajectory-layout",
        choices=("phase-offset", "view-population-pack"),
        default="phase-offset",
        help=(
            "phase-offset models independently moving environments; "
            "view-population-pack groups one unique source-view population "
            "into batches for controlled vectorization scaling."
        ),
    )
    parser.add_argument("--kit-settle-updates", type=int, default=8)
    parser.add_argument(
        "--trajectory-capacity-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--visible-per-view", type=int, default=100_000)
    parser.add_argument(
        "--intersections-per-view-at-128",
        type=int,
        default=170_000,
    )
    parser.add_argument("--max-workspace-bytes", type=int)
    parser.add_argument(
        "--save-trajectory-video",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/isaacsim-integration/vectorized-example.json"),
    )
    if argv is not None:
        return parser.parse_args(argv)
    args, kit_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *kit_args]
    return args


def resolve_covariance_epsilon(
    requested: float | None,
    *,
    rasterize_mode: str,
    ray_gaussian_evaluation: bool,
    compact_projection_cache: bool,
) -> float:
    """Resolve the effective screen-space regularizer without disabling AA."""
    if rasterize_mode == "antialiased" and ray_gaussian_evaluation:
        raise ValueError(
            "--rasterize-mode antialiased is incompatible with "
            "--ray-gaussian-evaluation."
        )
    if requested is None:
        if ray_gaussian_evaluation:
            return 0.0
        if rasterize_mode == "antialiased":
            return 0.3
        return 0.0 if compact_projection_cache else 0.3
    if not math.isfinite(requested) or requested < 0:
        raise ValueError(
            "--covariance-epsilon must be finite and non-negative."
        )
    return float(requested)


def renderer_configuration_contract(
    backend: object,
    *,
    rasterize_mode: str,
    covariance_epsilon: float,
) -> dict[str, object]:
    """Fail closed if Kit constructed a renderer with different settings."""
    actual_mode = getattr(backend, "rasterize_mode", None)
    actual_epsilon = getattr(backend, "covariance_epsilon", None)
    epsilon_matches = isinstance(actual_epsilon, (int, float)) and math.isclose(
        float(actual_epsilon),
        covariance_epsilon,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    mode_matches = actual_mode == rasterize_mode
    if not mode_matches or not epsilon_matches:
        raise AssertionError(
            "Renderer configuration mismatch: requested "
            f"rasterize_mode={rasterize_mode!r}, "
            f"covariance_epsilon={covariance_epsilon}; got "
            f"rasterize_mode={actual_mode!r}, "
            f"covariance_epsilon={actual_epsilon!r}."
        )
    return {
        "requested_rasterize_mode": rasterize_mode,
        "actual_rasterize_mode": actual_mode,
        "requested_covariance_epsilon": covariance_epsilon,
        "actual_covariance_epsilon": float(actual_epsilon),
        "matches": True,
    }


def mean(values: list[float]) -> float:
    return float(fmean(values)) if values else math.nan


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {name: math.nan for name in ("mean", "p50", "p95", "p99", "max")}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return float(ordered[lower])
        alpha = index - lower
        return float(ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha)

    return {
        "mean": mean(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": float(max(values)),
    }


def event_samples(
    starts: list,
    ends: list,
) -> list[float]:
    return [
        float(start.elapsed_time(end))
        for start, end in zip(starts, ends, strict=True)
    ]


def cache_delta(
    before: dict[str, int | bool],
    after: dict[str, int | bool],
) -> dict[str, int]:
    """Return projection-cache counter changes for one measured phase."""
    return {
        name: int(after[name]) - int(before[name])
        for name in ("hits", "misses")
    }


def output_contract(
    outputs: dict,
    *,
    semantic_min_alpha: float,
    expected_shape: tuple[int, int, int],
    require_cuda: bool = True,
) -> dict[str, object]:
    """Validate every production renderer output without a CPU copy."""
    import torch

    expected_specs = {
        "rgb": ((*expected_shape, 3), torch.float32),
        "depth": ((*expected_shape, 1), torch.float32),
        "alpha": ((*expected_shape, 1), torch.float32),
        "semantic_id": ((*expected_shape, 1), torch.int64),
    }
    output_names_match = set(outputs) == set(expected_specs)
    if not output_names_match:
        return {
            "valid": False,
            "output_names_match": False,
            "expected_output_names": sorted(expected_specs),
            "actual_output_names": sorted(outputs),
        }
    output_shapes = {
        name: list(tensor.shape) for name, tensor in outputs.items()
    }
    output_dtypes = {
        name: str(tensor.dtype) for name, tensor in outputs.items()
    }
    shapes_match = all(
        tuple(outputs[name].shape) == shape
        for name, (shape, _dtype) in expected_specs.items()
    )
    dtypes_match = all(
        outputs[name].dtype == dtype
        for name, (_shape, dtype) in expected_specs.items()
    )
    contiguous = all(tensor.is_contiguous() for tensor in outputs.values())
    alpha = outputs["alpha"][..., 0]
    depth = outputs["depth"][..., 0]
    semantic = outputs["semantic_id"][..., 0]
    foreground = alpha > 0
    background = ~foreground
    semantic_foreground = (
        foreground
        if semantic_min_alpha == 0
        else alpha >= semantic_min_alpha
    )
    semantic_background = ~semantic_foreground
    foreground_pixels = int(torch.count_nonzero(foreground).item())
    semantic_foreground_pixels = int(
        torch.count_nonzero(semantic_foreground).item()
    )
    output_devices = {
        name: str(tensor.device)
        for name, tensor in outputs.items()
    }
    gpu_resident = all(tensor.is_cuda for tensor in outputs.values())
    residency_valid = gpu_resident if require_cuda else True
    rgb_finite = bool(torch.isfinite(outputs["rgb"]).all().item())
    alpha_finite = bool(torch.isfinite(alpha).all().item())
    alpha_in_range = bool(
        ((alpha >= -1.0e-6) & (alpha <= 1.0 + 1.0e-6)).all().item()
    )
    foreground_depth_finite = (
        foreground_pixels > 0
        and bool(torch.isfinite(depth[foreground]).all().item())
    )
    background_depth_valid = bool(
        (
            torch.isfinite(depth[background])
            | (
                torch.isinf(depth[background])
                & (depth[background] > 0)
            )
        )
        .all()
        .item()
    )
    foreground_semantic_valid = (
        semantic_foreground_pixels > 0
        and bool(
            (semantic[semantic_foreground] >= 0).all().item()
        )
    )
    background_semantic_valid = bool(
        (semantic[semantic_background] == -1).all().item()
    )
    valid = (
        output_names_match
        and shapes_match
        and dtypes_match
        and contiguous
        and residency_valid
        and rgb_finite
        and alpha_finite
        and alpha_in_range
        and foreground_depth_finite
        and background_depth_valid
        and foreground_semantic_valid
        and background_semantic_valid
    )
    return {
        "valid": valid,
        "output_names_match": output_names_match,
        "shapes_match": shapes_match,
        "dtypes_match": dtypes_match,
        "contiguous": contiguous,
        "output_shapes": output_shapes,
        "output_dtypes": output_dtypes,
        "expected_shape": list(expected_shape),
        "outputs_gpu_resident": gpu_resident,
        "cuda_residency_required": require_cuda,
        "output_devices": output_devices,
        "rgb_finite": rgb_finite,
        "alpha_finite": alpha_finite,
        "alpha_in_range": alpha_in_range,
        "foreground_pixel_count": foreground_pixels,
        "foreground_depth_finite": foreground_depth_finite,
        "background_depth_valid": background_depth_valid,
        "foreground_semantic_valid": foreground_semantic_valid,
        "background_semantic_valid": background_semantic_valid,
        "semantic_foreground_pixel_count": (
            semantic_foreground_pixels
        ),
        "semantic_min_alpha": semantic_min_alpha,
        "depth_background_policy": (
            "finite_or_positive_infinity_when_alpha_zero"
        ),
    }


def driver_process_memory() -> str | None:
    """Return raw PID/MiB rows without claiming allocator equivalence."""
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.materialize_projected_records and not (
        args.compact_projection_cache or args.camera_contract is not None
    ):
        raise ValueError(
            "--materialize-projected-records requires the compact projection "
            "path. Pass --compact-projection-cache outside trajectory mode."
        )
    trajectory_mode = args.camera_contract is not None
    trajectory_required = {
        "--camera-manifest": args.camera_manifest,
        "--scene-path": args.scene_path,
        "--scene-manifest": args.scene_manifest,
        "--scene-id-label": args.scene_id_label,
        "--scene-author": args.scene_author,
        "--scene-license": args.scene_license,
        "--scene-source": args.scene_source,
        "--max-workspace-bytes": args.max_workspace_bytes,
    }
    if trajectory_mode:
        missing = [name for name, value in trajectory_required.items() if value is None]
        if missing:
            raise ValueError(
                "Real-scene trajectory mode requires explicit provenance and a "
                f"bounded workspace; missing {', '.join(missing)}."
            )
        if min(
            args.trajectory_frame_stride,
            args.visible_per_view,
            args.intersections_per_view_at_128,
            args.max_workspace_bytes,
        ) <= 0:
            raise ValueError("Trajectory stride, capacities, and workspace bound must be positive.")
        if args.trajectory_frames is not None and args.trajectory_frames <= 0:
            raise ValueError("--trajectory-frames must be positive when supplied.")
        if args.trajectory_start_frame < 0:
            raise ValueError("--trajectory-start-frame must be nonnegative.")
        if args.kit_settle_updates < 0:
            raise ValueError("--kit-settle-updates must be nonnegative.")
        if args.save_trajectory_video:
            missing_media_tools = [
                tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None
            ]
            if missing_media_tools:
                raise RuntimeError(
                    "Video evidence requires installed ffmpeg and ffprobe; missing "
                    + ", ".join(missing_media_tools)
                    + "."
                )
    elif any(value is not None for value in trajectory_required.values()):
        raise ValueError("Real-scene trajectory arguments require --camera-contract.")
    args.covariance_epsilon = resolve_covariance_epsilon(
        args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        compact_projection_cache=(
            args.compact_projection_cache or trajectory_mode
        ),
    )
    if trajectory_mode and args.ray_gaussian_evaluation:
        raise ValueError(
            "Real-scene trajectory mode uses the compact screen-space path "
            "and does not support --ray-gaussian-evaluation."
        )
    os.environ.setdefault(
        "WARP_CACHE_PATH",
        str(args.repo_root / "build" / "warp-cache"),
    )
    if min(
        args.batch,
        args.gaussians,
        args.width,
        args.height,
        args.warmup,
        args.iterations,
        args.physics_steps,
    ) <= 0:
        raise ValueError("All numeric arguments must be positive.")
    if not 0.0 < args.visible_fraction <= 1.0:
        raise ValueError("--visible-fraction must be in (0, 1].")
    if not 0.0 <= args.moving_fraction <= 1.0:
        raise ValueError("--moving-fraction must be in [0, 1].")
    if args.deadline_ms <= 0:
        raise ValueError("--deadline-ms must be positive.")
    if not 0 <= args.semantic_min_alpha <= 1:
        raise ValueError("--semantic-min-alpha must be in [0, 1].")
    if args.compact_projection_cache and args.deterministic:
        raise ValueError(
            "Compact projection cache requires --no-deterministic."
        )
    if args.compact_projection_cache and args.tile_size != 1:
        raise ValueError(
            "Compact projection cache requires --tile-size 1."
        )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        isaac_app_config(
            renderer="MinimalRendering",
            disable_viewport_updates=True,
        )
    )
    failed = False
    service = None
    extension = None
    try:
        if trajectory_mode:
            from isaacsim_gaussian_renderer.evaluation.isaac_fabric_trajectory import (
                run_isaac_fabric_trajectory,
            )

            result = run_isaac_fabric_trajectory(
                args,
                simulation_app=simulation_app,
                distribution=distribution,
                output_contract=output_contract,
                driver_process_memory=driver_process_memory,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                "ISAACSIM_FABRIC_TRAJECTORY_RESULT",
                json.dumps(result, sort_keys=True),
            )
            if not result["pass"]:
                raise AssertionError(
                    "Isaac/Fabric trajectory result failed its acceptance contract."
                )
            print("ISAACSIM_FABRIC_TRAJECTORY_OK", flush=True)
            return

        import carb
        import omni.kit.app
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf, UsdGeom, UsdPhysics

        from isaacsim_gaussian_renderer.benchmark_manifest import (
            synthetic_scene_tensors,
        )

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.add_path(str(args.repo_root / "exts"))
        manager.set_extension_enabled_immediate("isaacsim.gaussian_renderer", True)
        settings = carb.settings.get_settings()
        settings.set("/app/useFabricSceneDelegate", True)
        fabric_delegate_extension_loaded = manager.is_extension_enabled(
            "omni.hydra.usdrt_delegate"
        )
        fabric_scene_delegate_enabled = settings.get_as_bool(
            "/app/useFabricSceneDelegate"
        )
        if not (
            fabric_delegate_extension_loaded
            and fabric_scene_delegate_enabled
        ):
            raise RuntimeError(
                "Isaac Sim Fabric Scene Delegate is not active."
            )

        from isaacsim_gaussian_renderer_extension import extension as extension_module

        extension = extension_module.STARTUP_INSTANCE
        if extension is None:
            raise RuntimeError("Kit did not start isaacsim.gaussian_renderer.")

        create_new_stage()
        world = World(stage_units_in_meters=1.0)
        world.set_simulation_dt(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(9.81)

        camera_paths: list[str] = []
        grid_width = math.ceil(math.sqrt(args.batch))
        if args.motion_mode == "static":
            moving_count = 0
        elif args.motion_mode == "fully-moving":
            moving_count = args.batch
        else:
            moving_count = math.ceil(args.batch * args.moving_fraction)
        for index in range(args.batch):
            grid_x = index % grid_width
            grid_y = index // grid_width
            x = (grid_x - (grid_width - 1) * 0.5) * 0.12
            y = (grid_y - (grid_width - 1) * 0.5) * 0.12
            phase = 2.0 * math.pi * index / args.batch
            z = 1.5 + 0.05 * math.sin(phase)
            body_path = f"/World/CameraBodies/Body_{index:04d}"
            cube = UsdGeom.Cube.Define(stage, body_path)
            cube.CreateSizeAttr(0.05)
            UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            if index < moving_count:
                UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.1)

            camera_path = f"{body_path}/Camera"
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.CreateFocalLengthAttr(18.0)
            camera.CreateHorizontalApertureAttr(36.0)
            camera.CreateVerticalApertureAttr(
                36.0 * float(args.height) / float(args.width)
            )
            camera_paths.append(camera_path)

        world.reset()
        simulation_app.update()

        device = torch.device("cuda:0")
        camera_path_tuple = tuple(camera_paths)
        camera_source = extension.create_camera_source(
            stage=stage,
            stage_id=omni.usd.get_context().get_stage_id(),
            camera_paths=camera_path_tuple,
            device=str(device),
            update_world_xforms=True,
        )
        intrinsics = camera_source.get_batched_intrinsics(
            camera_path_tuple,
            width=args.width,
            height=args.height,
            device=device,
        )

        visible_capacity = max(
            1,
            math.ceil(
                args.batch
                * args.gaussians
                * args.visible_fraction
            ),
        )
        intersection_factor = (
            max(
                32,
                math.ceil(
                    32
                    * (
                        max(args.width, args.height)
                        / 128.0
                    )
                    ** 2
                ),
            )
            if args.ray_gaussian_evaluation
            else 8
        )
        service = extension.create_service(
            stage=stage,
            device=str(device),
            height=args.height,
            width=args.width,
            max_views=args.batch,
            max_visible_records=visible_capacity,
            max_intersections=(
                visible_capacity * intersection_factor
            ),
            gaussian_support_sigma=args.gaussian_support_sigma,
            covariance_epsilon=args.covariance_epsilon,
            rasterize_mode=args.rasterize_mode,
            semantic_min_alpha=args.semantic_min_alpha,
            ray_gaussian_evaluation=args.ray_gaussian_evaluation,
            tile_size=args.tile_size,
            depth_bucket_count=args.depth_bucket_count,
            depth_bucket_group_size=args.depth_bucket_group_size,
            compact_projection_cache=args.compact_projection_cache,
            materialize_projected_records=(
                args.materialize_projected_records
            ),
            enable_projection_cache=args.enable_projection_cache,
            output_srgb=False,
            deterministic=args.deterministic,
        )
        backend = service.backend
        renderer_configuration = renderer_configuration_contract(
            backend,
            rasterize_mode=args.rasterize_mode,
            covariance_epsilon=args.covariance_epsilon,
        )

        scene = synthetic_scene_tensors(
            args.gaussians,
            seed=919,
            device=device,
        )
        means = scene["means"].clone()
        means[:, 0:2] *= 0.35
        means[:, 2] = -scene["means"][:, 2]
        service.load_scene(
            77,
            means=means.contiguous(),
            scales=scene["scales"],
            rotations=scene["quats"],
            opacities=scene["opacities"],
            features=scene["colors"],
            semantic_ids=scene["semantic_ids"].to(torch.int64),
        )
        scene_ids = torch.full(
            (args.batch,),
            77,
            device=device,
            dtype=torch.int64,
        )

        initial_viewmats = camera_source.read_transforms(
            camera_path_tuple,
            device=device,
        ).clone()
        service.synchronize()
        time_before_physics = float(world.current_time)
        for _ in range(4):
            world.step(render=False)
        time_after_physics = float(world.current_time)
        moved_viewmats = camera_source.read_transforms(
            camera_path_tuple,
            device=device,
        ).clone()
        service.synchronize()
        if not time_after_physics > time_before_physics:
            raise AssertionError("Isaac physics time did not advance.")
        moved_mask = torch.any(initial_viewmats != moved_viewmats, dim=(1, 2))
        expected_moved_mask = torch.arange(args.batch, device=device) < moving_count
        if not torch.equal(moved_mask, expected_moved_mask):
            raise AssertionError(
                "Fabric camera motion mask mismatch: "
                f"expected {expected_moved_mask.tolist()}, got {moved_mask.tolist()}."
            )

        simulation_time_before_render = float(world.current_time)
        first_start = torch.cuda.Event(enable_timing=True)
        first_end = torch.cuda.Event(enable_timing=True)
        first_cache_before = backend.projection_cache_stats
        first_wall_start = time.perf_counter()
        first_start.record()
        outputs = service.render(moved_viewmats, intrinsics, scene_ids)
        first_end.record()
        service.synchronize()
        first_wall_ms = (
            time.perf_counter() - first_wall_start
        ) * 1_000.0
        first_gpu_ms = float(first_start.elapsed_time(first_end))
        first_cache_after = backend.projection_cache_stats
        simulation_time_after_render = float(world.current_time)
        if simulation_time_after_render != simulation_time_before_render:
            raise AssertionError("Renderer work advanced Isaac simulation time.")
        first_counters = backend.check_capacity(synchronize=False)
        if (
            first_counters["visible_overflow"]
            or first_counters["intersection_overflow"]
        ):
            raise AssertionError(
                f"Initial render overflowed capacity: {first_counters}."
            )
        first_capacity = backend.capacity_stats
        initial_output_contract = output_contract(
            outputs,
            semantic_min_alpha=args.semantic_min_alpha,
            expected_shape=(args.batch, args.height, args.width),
        )
        if not initial_output_contract["valid"]:
            raise AssertionError(
                "Initial renderer output contract failed: "
                f"{initial_output_contract}"
            )

        deterministic_reference = {
            name: tensor.clone()
            for name, tensor in outputs.items()
        }
        service.render(
            moved_viewmats,
            intrinsics,
            scene_ids,
            outputs=outputs,
        )
        service.synchronize()
        deterministic_equal = {
            name: bool(torch.equal(deterministic_reference[name], outputs[name]))
            for name in outputs
        }
        if not all(deterministic_equal.values()):
            raise AssertionError(
                f"Fixed-state deterministic render mismatch: {deterministic_equal}"
            )

        fabric_invalidation = {
            "tested": False,
            "reused_cuda_tensor": False,
            "version_counter_bypassed": False,
            "explicit_invalidation_forced_miss": False,
            "motion_reached_outputs": False,
        }
        if args.enable_projection_cache and moving_count > 0:
            cached_viewmats = camera_source.read_transforms(
                camera_path_tuple,
                device=device,
            )
            service.render(cached_viewmats, intrinsics, scene_ids, outputs=outputs)
            service.synchronize()
            cached_outputs = {name: value.clone() for name, value in outputs.items()}
            cached_pointer = cached_viewmats.data_ptr()
            try:
                cached_version = int(cached_viewmats._version)
            except RuntimeError:
                cached_version = -1
            world.step(render=False)
            updated_viewmats = camera_source.read_transforms(
                camera_path_tuple,
                device=device,
            )
            service.synchronize()
            try:
                updated_version = int(updated_viewmats._version)
            except RuntimeError:
                updated_version = -2
            before_invalidation_stats = backend.projection_cache_stats
            backend.invalidate_projection_cache()
            service.render(updated_viewmats, intrinsics, scene_ids, outputs=outputs)
            service.synchronize()
            after_invalidation_stats = backend.projection_cache_stats
            fabric_invalidation = {
                "tested": True,
                "reused_cuda_tensor": updated_viewmats.data_ptr() == cached_pointer,
                "version_counter_bypassed": updated_version == cached_version,
                "explicit_invalidation_forced_miss": (
                    int(after_invalidation_stats["misses"])
                    == int(before_invalidation_stats["misses"]) + 1
                ),
                "motion_reached_outputs": any(
                    not torch.equal(cached_outputs[name], outputs[name])
                    for name in outputs
                ),
            }
            if not (
                fabric_invalidation["reused_cuda_tensor"]
                and fabric_invalidation["explicit_invalidation_forced_miss"]
                and fabric_invalidation["motion_reached_outputs"]
            ):
                raise AssertionError(
                    f"Fabric external-write invalidation failed: {fabric_invalidation}"
                )

        for _ in range(args.warmup):
            viewmats = camera_source.read_transforms(
                camera_path_tuple,
                device=device,
            )
            service.render(
                viewmats,
                intrinsics,
                scene_ids,
                outputs=outputs,
            )
        service.synchronize()

        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
        static_cache_before = backend.projection_cache_stats
        static_starts = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(args.iterations)
        ]
        static_after_sources = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(args.iterations)
        ]
        static_ends = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(args.iterations)
        ]
        static_source_submit_us: list[float] = []
        static_render_submit_us: list[float] = []
        static_wall_ms: list[float] = []
        for start, after_source, end in zip(
            static_starts,
            static_after_sources,
            static_ends,
            strict=True,
        ):
            wall_start = time.perf_counter()
            start.record()
            submit_start = time.perf_counter_ns()
            viewmats = camera_source.read_transforms(
                camera_path_tuple,
                device=device,
            )
            static_source_submit_us.append(
                (time.perf_counter_ns() - submit_start) / 1_000.0
            )
            after_source.record()
            submit_start = time.perf_counter_ns()
            service.render(
                viewmats,
                intrinsics,
                scene_ids,
                outputs=outputs,
            )
            static_render_submit_us.append(
                (time.perf_counter_ns() - submit_start) / 1_000.0
            )
            end.record()
            service.synchronize()
            static_wall_ms.append(
                (time.perf_counter() - wall_start) * 1_000.0
            )
        static_source_gpu_ms = event_samples(
            static_starts,
            static_after_sources,
        )
        static_render_gpu_ms = event_samples(
            static_after_sources,
            static_ends,
        )
        static_integrated_gpu_ms = event_samples(
            static_starts,
            static_ends,
        )
        static_cache_after = backend.projection_cache_stats
        static_output_contract = output_contract(
            outputs,
            semantic_min_alpha=args.semantic_min_alpha,
            expected_shape=(args.batch, args.height, args.width),
        )
        if not static_output_contract["valid"]:
            raise AssertionError(
                "Fixed-camera output contract failed: "
                f"{static_output_contract}"
            )
        static_counters = backend.check_capacity(synchronize=False)
        if (
            static_counters["visible_overflow"]
            or static_counters["intersection_overflow"]
        ):
            raise AssertionError(
                f"Fixed-camera phase overflowed capacity: {static_counters}."
            )

        invalidated_cache_before = backend.projection_cache_stats
        invalidated_starts = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(args.iterations)
        ]
        invalidated_after_sources = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(args.iterations)
        ]
        invalidated_ends = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(args.iterations)
        ]
        invalidated_source_submit_us: list[float] = []
        invalidated_render_submit_us: list[float] = []
        invalidated_wall_ms: list[float] = []
        for start, after_source, end in zip(
            invalidated_starts,
            invalidated_after_sources,
            invalidated_ends,
            strict=True,
        ):
            wall_start = time.perf_counter()
            start.record()
            submit_start = time.perf_counter_ns()
            viewmats = camera_source.read_transforms(
                camera_path_tuple,
                device=device,
            )
            if args.enable_projection_cache:
                backend.invalidate_projection_cache()
            invalidated_source_submit_us.append(
                (time.perf_counter_ns() - submit_start) / 1_000.0
            )
            after_source.record()
            submit_start = time.perf_counter_ns()
            service.render(
                viewmats,
                intrinsics,
                scene_ids,
                outputs=outputs,
            )
            invalidated_render_submit_us.append(
                (time.perf_counter_ns() - submit_start) / 1_000.0
            )
            end.record()
            service.synchronize()
            invalidated_wall_ms.append(
                (time.perf_counter() - wall_start) * 1_000.0
            )
        invalidated_source_gpu_ms = event_samples(
            invalidated_starts,
            invalidated_after_sources,
        )
        invalidated_render_gpu_ms = event_samples(
            invalidated_after_sources,
            invalidated_ends,
        )
        invalidated_integrated_gpu_ms = event_samples(
            invalidated_starts,
            invalidated_ends,
        )
        invalidated_cache_after = backend.projection_cache_stats
        invalidated_output_contract = output_contract(
            outputs,
            semantic_min_alpha=args.semantic_min_alpha,
            expected_shape=(args.batch, args.height, args.width),
        )
        if not invalidated_output_contract["valid"]:
            raise AssertionError(
                "Invalidated-camera output contract failed: "
                f"{invalidated_output_contract}"
            )
        invalidated_counters = backend.check_capacity(synchronize=False)
        if (
            invalidated_counters["visible_overflow"]
            or invalidated_counters["intersection_overflow"]
        ):
            raise AssertionError(
                "Invalidated-camera phase overflowed capacity: "
                f"{invalidated_counters}."
            )

        physics_cache_before = backend.projection_cache_stats
        physics_wall_ms: list[float] = []
        physics_observation_wall_ms: list[float] = []
        physics_source_gpu_ms: list[float] = []
        physics_render_gpu_ms: list[float] = []
        physics_integrated_gpu_ms: list[float] = []
        physics_output_contracts: list[dict[str, object]] = []
        physics_counters: list[dict[str, int]] = []
        deadline_misses = 0
        for step in range(args.physics_steps):
            observation_start = time.perf_counter()
            physics_start = time.perf_counter()
            world.step(render=False)
            physics_wall_ms.append(
                (time.perf_counter() - physics_start) * 1_000.0
            )
            source_start = torch.cuda.Event(enable_timing=True)
            source_end = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            source_start.record()
            viewmats = camera_source.read_transforms(
                camera_path_tuple,
                device=device,
            )
            if args.enable_projection_cache:
                backend.invalidate_projection_cache()
            source_end.record()
            service.render(
                viewmats,
                intrinsics,
                scene_ids,
                outputs=outputs,
            )
            render_end.record()
            service.synchronize()
            physics_source_gpu_ms.append(
                float(source_start.elapsed_time(source_end))
            )
            physics_render_gpu_ms.append(
                float(source_end.elapsed_time(render_end))
            )
            physics_integrated_gpu_ms.append(
                float(source_start.elapsed_time(render_end))
            )
            physics_observation_wall_ms.append(
                (time.perf_counter() - observation_start) * 1_000.0
            )
            if physics_observation_wall_ms[-1] > args.deadline_ms:
                deadline_misses += 1
            contract = output_contract(
                outputs,
                semantic_min_alpha=args.semantic_min_alpha,
                expected_shape=(args.batch, args.height, args.width),
            )
            contract["step"] = step
            physics_output_contracts.append(contract)
            step_counters = backend.check_capacity(synchronize=False)
            physics_counters.append(step_counters)
            if not contract["valid"]:
                raise AssertionError(
                    "Physics-coupled output contract failed at step "
                    f"{step}: {contract}"
                )
            if (
                step_counters["visible_overflow"]
                or step_counters["intersection_overflow"]
            ):
                raise AssertionError(
                    "Physics-coupled render overflowed at step "
                    f"{step}: {step_counters}"
                )

        physics_cache_after = backend.projection_cache_stats
        final_counters = backend.check_capacity(synchronize=False)
        final_capacity = backend.capacity_stats
        allocated_after = int(torch.cuda.memory_allocated())
        memory = {
            "torch_allocated_bytes": allocated_after,
            "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
            "torch_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "allocation_delta_bytes": (
                allocated_after - allocated_before
            ),
            "backend_scene_bytes": int(backend.scene_bytes),
            "backend_workspace_bytes": int(backend.workspace_bytes),
            "driver_compute_process_rows_pid_used_mib": (
                driver_process_memory()
            ),
        }
        static_integrated_mean_ms = mean(static_integrated_gpu_ms)
        invalidated_integrated_mean_ms = mean(
            invalidated_integrated_gpu_ms
        )
        static_cache_changes = cache_delta(
            static_cache_before,
            static_cache_after,
        )
        invalidated_cache_changes = cache_delta(
            invalidated_cache_before,
            invalidated_cache_after,
        )
        physics_cache_changes = cache_delta(
            physics_cache_before,
            physics_cache_after,
        )
        cache_expectations = {
            "enabled": args.enable_projection_cache,
            "fixed_camera": {
                "expected_hits": (
                    args.iterations
                    if args.enable_projection_cache
                    else 0
                ),
                "expected_misses": 0,
                **static_cache_changes,
            },
            "fixed_camera_explicit_invalidation": {
                "expected_hits": 0,
                "expected_misses": (
                    args.iterations
                    if args.enable_projection_cache
                    else 0
                ),
                **invalidated_cache_changes,
            },
            "physics_coupled_explicit_invalidation": {
                "expected_hits": 0,
                "expected_misses": (
                    args.physics_steps
                    if args.enable_projection_cache
                    else 0
                ),
                **physics_cache_changes,
            },
        }
        cache_expectations_pass = all(
            phase["hits"] == phase["expected_hits"]
            and phase["misses"] == phase["expected_misses"]
            for phase in cache_expectations.values()
            if isinstance(phase, dict)
        )
        output_assertions_pass = bool(
            initial_output_contract["valid"]
            and static_output_contract["valid"]
            and invalidated_output_contract["valid"]
            and all(
                contract["valid"]
                for contract in physics_output_contracts
            )
        )
        zero_overflow = bool(
            not first_counters["visible_overflow"]
            and not first_counters["intersection_overflow"]
            and not static_counters["visible_overflow"]
            and not static_counters["intersection_overflow"]
            and not invalidated_counters["visible_overflow"]
            and not invalidated_counters["intersection_overflow"]
            and not final_counters["visible_overflow"]
            and not final_counters["intersection_overflow"]
            and all(
                not counters["visible_overflow"]
                and not counters["intersection_overflow"]
                for counters in physics_counters
            )
        )
        fabric_invalidation_pass = bool(
            not (args.enable_projection_cache and moving_count > 0)
            or (
                fabric_invalidation["tested"]
                and fabric_invalidation[
                    "explicit_invalidation_forced_miss"
                ]
                and fabric_invalidation["motion_reached_outputs"]
            )
        )
        result = {
            "schema_version": "isaacsim-vectorized-example/v2",
            "runtime": "Isaac Sim SimulationApp headless",
            "simulation_app": {
                "renderer": "MinimalRendering",
                "viewport_updates_disabled": True,
            },
            "isaac_lab_required": False,
            "fabric_scene_delegate": {
                "enabled": fabric_scene_delegate_enabled,
                "extension": "omni.hydra.usdrt_delegate",
                "extension_loaded": (
                    fabric_delegate_extension_loaded
                ),
            },
            "batch": args.batch,
            "motion_mode": args.motion_mode,
            "moving_camera_count": moving_count,
            "moving_fraction": moving_count / args.batch,
            "gaussians": args.gaussians,
            "width": args.width,
            "height": args.height,
            "camera_ingestion": {
                "source": "USDRT SelectPrims CUDA Fabric selection",
                "attribute": "omni:fabric:worldMatrix",
                "runtime_camera_loop": False,
                "gpu_conversion_launches_per_batch": 1,
                "torch_output_zero_copy": True,
                "python_loop_over_cameras": False,
                "read_calls": extension.fabric_transform_source.read_calls,
                "topology_rebuilds": extension.fabric_transform_source.topology_rebuilds,
            },
            "first_render": {
                "gpu_ms": first_gpu_ms,
                "wall_ms": first_wall_ms,
                "cache": cache_delta(
                    first_cache_before,
                    first_cache_after,
                ),
                "counters": first_counters,
                "capacity_adaptation": first_capacity,
                "output_contract": initial_output_contract,
            },
            "renderer": {
                "backend": backend.__class__.__name__,
                "deterministic": backend.deterministic,
                "gaussian_evaluation_model": (
                    "exact-ray-3d"
                    if backend.ray_gaussian_evaluation
                    else "screen-space-conic"
                ),
                "covariance_epsilon": backend.covariance_epsilon,
                "rasterize_mode": backend.rasterize_mode,
                "configuration_contract": renderer_configuration,
                "semantic_min_alpha": backend.semantic_min_alpha,
                "tile_size": backend.tile_size,
                "pipeline": backend.pipeline_name,
                "compact_projection_cache": (
                    backend.compact_projection_cache
                ),
                "projected_record_reuse": backend.capacity_stats[
                    "projected_record_reuse"
                ],
                "projection_cache_enabled": (
                    backend.enable_projection_cache
                ),
                "initial_visible_capacity": visible_capacity,
                "final_visible_capacity": (
                    backend.current_visible_capacity
                ),
                "visible_storage_capacity": (
                    backend.visible_storage_capacity
                ),
                "initial_intersection_capacity": (
                    visible_capacity * intersection_factor
                ),
                "final_intersection_capacity": (
                    backend.current_intersection_capacity
                ),
                "visible_fraction": args.visible_fraction,
                "runtime_camera_loop": False,
                "batched_native_submissions_per_render": 1,
                "outputs": {
                    name: {
                        "shape": list(tensor.shape),
                        "device": str(tensor.device),
                    }
                    for name, tensor in outputs.items()
                },
                "counters": final_counters,
                "capacity_adaptation": final_capacity,
                "projection_cache": (
                    backend.projection_cache_stats
                ),
                "deterministic_bitwise_equal": deterministic_equal,
                "render_does_not_advance_physics": (
                    simulation_time_after_render == simulation_time_before_render
                ),
                "fabric_external_write_invalidation": fabric_invalidation,
                "output_assertions_pass": output_assertions_pass,
                "zero_overflow": zero_overflow,
            },
            "static_state_performance": {
                "cache_semantics": (
                    "fixed camera, no explicit invalidation"
                ),
                "warmup_iterations": args.warmup,
                "iterations": args.iterations,
                "fabric_source_gpu_ms": distribution(
                    static_source_gpu_ms
                ),
                "custom_render_gpu_ms": distribution(
                    static_render_gpu_ms
                ),
                "integrated_gpu_ms": distribution(
                    static_integrated_gpu_ms
                ),
                "synchronized_wall_ms": distribution(
                    static_wall_ms
                ),
                "integration_overhead_gpu_ms_mean": (
                    static_integrated_mean_ms
                    - mean(static_render_gpu_ms)
                ),
                "fabric_source_submit_us": distribution(
                    static_source_submit_us
                ),
                "custom_render_submit_us": distribution(
                    static_render_submit_us
                ),
                "gpu_images_per_second": (
                    args.batch * 1_000.0
                    / static_integrated_mean_ms
                ),
                "wall_images_per_second": (
                    args.batch * 1_000.0 / mean(static_wall_ms)
                ),
                "cache": static_cache_changes,
                "counters": static_counters,
                "output_contract": static_output_contract,
            },
            "fixed_invalidated_performance": {
                "cache_semantics": (
                    "fixed camera, explicit invalidation before every render"
                ),
                "iterations": args.iterations,
                "fabric_source_gpu_ms": distribution(
                    invalidated_source_gpu_ms
                ),
                "custom_render_gpu_ms": distribution(
                    invalidated_render_gpu_ms
                ),
                "integrated_gpu_ms": distribution(
                    invalidated_integrated_gpu_ms
                ),
                "synchronized_wall_ms": distribution(
                    invalidated_wall_ms
                ),
                "fabric_source_submit_us": distribution(
                    invalidated_source_submit_us
                ),
                "custom_render_submit_us": distribution(
                    invalidated_render_submit_us
                ),
                "gpu_images_per_second": (
                    args.batch * 1_000.0
                    / invalidated_integrated_mean_ms
                ),
                "wall_images_per_second": (
                    args.batch * 1_000.0
                    / mean(invalidated_wall_ms)
                ),
                "cache": invalidated_cache_changes,
                "counters": invalidated_counters,
                "output_contract": invalidated_output_contract,
            },
            "physics_coupled_performance": {
                "steps": args.physics_steps,
                "physics_step_wall_ms_mean": mean(physics_wall_ms),
                "fabric_source_gpu_ms_mean": mean(physics_source_gpu_ms),
                "custom_render_gpu_ms_mean": mean(physics_render_gpu_ms),
                "observation_wall_ms_mean": mean(physics_observation_wall_ms),
                "physics_wall_ms": distribution(physics_wall_ms),
                "fabric_source_gpu_ms": distribution(physics_source_gpu_ms),
                "custom_render_gpu_ms": distribution(physics_render_gpu_ms),
                "integrated_gpu_ms": distribution(
                    physics_integrated_gpu_ms
                ),
                "integrated_observation_wall_ms": distribution(physics_observation_wall_ms),
                "deadline_ms": args.deadline_ms,
                "deadline_misses": deadline_misses,
                "deadline_miss_rate": deadline_misses / args.physics_steps,
                "observation_images_per_second": (
                    args.batch * 1_000.0 / mean(physics_observation_wall_ms)
                ),
                "simulation_time_before": time_before_physics,
                "simulation_time_after": float(world.current_time),
                "fabric_reflected_physics_motion": bool(moving_count > 0),
                "cache": physics_cache_changes,
                "output_contracts": physics_output_contracts,
                "capacity_counters": physics_counters,
            },
            "cache_expectations": cache_expectations,
            "cache_expectations_pass": cache_expectations_pass,
            "memory": memory,
            "environment": {
                "source_commit": os.environ.get(
                    "SOURCE_GIT_COMMIT",
                    "unknown",
                ),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "pass": (
                output_assertions_pass
                and zero_overflow
                and cache_expectations_pass
                and fabric_invalidation_pass
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "ISAACSIM_VECTORIZED_EXAMPLE_RESULT",
            json.dumps(result, sort_keys=True),
        )
        if not result["pass"]:
            raise AssertionError(
                "Isaac vectorized example failed its acceptance contract."
            )
        print("ISAACSIM_VECTORIZED_EXAMPLE_OK", flush=True)
    except BaseException:
        failed = True
        traceback.print_exc()
        raise
    finally:
        try:
            try:
                if service is not None:
                    service.shutdown()
            finally:
                if (
                    extension is not None
                    and extension.fabric_transform_source is not None
                ):
                    extension.fabric_transform_source.close()
        except BaseException:
            failed = True
            traceback.print_exc()
            raise
        finally:
            close_simulation_app(simulation_app, failed=failed)


if __name__ == "__main__":
    main()
