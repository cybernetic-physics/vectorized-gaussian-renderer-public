"""Finite Isaac Sim headless smoke test for the Gaussian renderer extension."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


SUCCESS_MARKER = "ISAACSIM_GAUSSIAN_RENDERER_SMOKE_OK"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--covariance-epsilon", type=float, default=None)
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument("--ray-gaussian-evaluation", action="store_true")
    parser.add_argument("--compact-projection-cache", action="store_true")
    parser.add_argument("--enable-projection-cache", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/isaacsim-integration/extension-smoke.json"),
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


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def output_contract(
    outputs: dict[str, Any],
    *,
    semantic_min_alpha: float,
    expected_shape: tuple[int, int, int],
    require_cuda: bool,
) -> dict[str, Any]:
    """Validate all production outputs without copying them off the device."""
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

    shapes_match = all(
        tuple(outputs[name].shape) == shape
        for name, (shape, _dtype) in expected_specs.items()
    )
    dtypes_match = all(
        outputs[name].dtype == dtype
        for name, (_shape, dtype) in expected_specs.items()
    )
    contiguous = all(tensor.is_contiguous() for tensor in outputs.values())
    gpu_resident = all(tensor.is_cuda for tensor in outputs.values())
    residency_valid = gpu_resident if require_cuda else True
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
            | (torch.isinf(depth[background]) & (depth[background] > 0))
        )
        .all()
        .item()
    )
    foreground_semantic_valid = (
        semantic_foreground_pixels > 0
        and bool((semantic[semantic_foreground] >= 0).all().item())
    )
    background_semantic_valid = bool(
        (semantic[semantic_background] == -1).all().item()
    )
    valid = bool(
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
        "output_shapes": {
            name: list(tensor.shape) for name, tensor in outputs.items()
        },
        "output_dtypes": {
            name: str(tensor.dtype) for name, tensor in outputs.items()
        },
        "expected_shape": list(expected_shape),
        "outputs_gpu_resident": gpu_resident,
        "cuda_residency_required": require_cuda,
        "output_devices": {
            name: str(tensor.device) for name, tensor in outputs.items()
        },
        "rgb_finite": rgb_finite,
        "alpha_finite": alpha_finite,
        "alpha_in_range": alpha_in_range,
        "foreground_pixel_count": foreground_pixels,
        "foreground_depth_finite": foreground_depth_finite,
        "background_depth_valid": background_depth_valid,
        "foreground_semantic_valid": foreground_semantic_valid,
        "background_semantic_valid": background_semantic_valid,
        "semantic_foreground_pixel_count": semantic_foreground_pixels,
        "semantic_min_alpha": semantic_min_alpha,
        "depth_background_policy": (
            "finite_or_positive_infinity_when_alpha_zero"
        ),
    }

if __package__:
    from ._isaac_launch import close_simulation_app, isaac_app_config
else:
    from _isaac_launch import close_simulation_app, isaac_app_config


def main() -> None:
    args = parse_args()
    if (
        args.tile_size <= 0
        or args.tile_size > 16
        or args.tile_size & (args.tile_size - 1)
    ):
        raise ValueError("--tile-size must be a power of two in [1, 16].")
    if args.compact_projection_cache and (
        args.tile_size != 1 or args.ray_gaussian_evaluation
    ):
        raise ValueError(
            "--compact-projection-cache requires --tile-size 1 and "
            "screen-space evaluation."
        )
    covariance_epsilon = resolve_covariance_epsilon(
        args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        compact_projection_cache=args.compact_projection_cache,
    )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        isaac_app_config(
            renderer="MinimalRendering",
            disable_viewport_updates=True,
        )
    )
    service = None
    failed = True
    result: dict[str, Any] = {
        "schema_version": "isaacsim-gaussian-renderer-extension-smoke/v2",
        "pass": False,
        "configuration": {
            "rasterize_mode": args.rasterize_mode,
            "covariance_epsilon": covariance_epsilon,
            "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
            "compact_projection_cache": args.compact_projection_cache,
            "projection_cache_enabled": args.enable_projection_cache,
            "tile_size": args.tile_size,
        },
    }
    try:
        import carb
        import omni.kit.app
        import omni.usd
        import torch
        from isaacsim.core.utils.stage import create_new_stage

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.add_path(str(args.repo_root / "exts"))
        manager.set_extension_enabled_immediate(
            "isaacsim.gaussian_renderer",
            True,
        )
        settings = carb.settings.get_settings()
        settings.set("/app/useFabricSceneDelegate", True)
        if not manager.is_extension_enabled("isaacsim.gaussian_renderer"):
            raise AssertionError(
                "Kit did not enable isaacsim.gaussian_renderer."
            )
        if not manager.is_extension_enabled("omni.hydra.usdrt_delegate"):
            raise AssertionError(
                "Kit did not load the Fabric Scene Delegate extension."
            )
        if not settings.get_as_bool("/app/useFabricSceneDelegate"):
            raise AssertionError("Fabric Scene Delegate setting is disabled.")

        from isaacsim_gaussian_renderer import (
            DEFAULT_GAUSSIAN_SUPPORT_SIGMA,
            DeterministicFakeBackend,
        )
        from isaacsim_gaussian_renderer_extension import extension as extension_module

        if extension_module.STARTUP_EXT_ID is None:
            raise AssertionError(
                "Kit enabled the package without invoking its IExt lifecycle."
            )
        extension = extension_module.STARTUP_INSTANCE
        if extension is None:
            raise AssertionError(
                "Kit lifecycle did not retain the extension instance."
            )
        print(
            "ISAACSIM_GAUSSIAN_RENDERER_EXTENSION_READY",
            json.dumps(
                {
                    "extension_id": extension_module.STARTUP_EXT_ID,
                    "fabric_scene_delegate": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        create_new_stage()
        stage = omni.usd.get_context().get_stage()
        requested_device = torch.device(args.device)
        device = (
            requested_device
            if requested_device.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        allow_cpu = device.type == "cpu"
        if allow_cpu and args.rasterize_mode == "antialiased":
            raise RuntimeError(
                "The antialiased extension smoke requires CUDA so it can "
                "verify the selected mode reached CustomCudaBackend."
            )

        service = extension.create_service(
            stage=stage,
            device=str(device),
            height=4,
            width=5,
            max_views=2,
            max_visible_records=64,
            max_intersections=4096,
            covariance_epsilon=covariance_epsilon,
            rasterize_mode=args.rasterize_mode,
            semantic_min_alpha=0.01,
            ray_gaussian_evaluation=args.ray_gaussian_evaluation,
            tile_size=args.tile_size,
            depth_bucket_count=32,
            depth_bucket_group_size=8,
            compact_projection_cache=args.compact_projection_cache,
            enable_projection_cache=args.enable_projection_cache,
            output_srgb=False,
            deterministic=not args.compact_projection_cache,
            allow_cpu_for_tests=allow_cpu,
        )
        backend = service.backend
        using_fake_backend = isinstance(backend, DeterministicFakeBackend)
        if using_fake_backend:
            renderer_configuration = {
                "asserted": False,
                "reason": "CPU lifecycle-only fake backend",
                "matches": None,
            }
        else:
            actual_mode = getattr(backend, "rasterize_mode", None)
            actual_epsilon = getattr(backend, "covariance_epsilon", None)
            actual_support = getattr(backend, "gaussian_support_sigma", None)
            renderer_configuration = {
                "asserted": True,
                "requested_rasterize_mode": args.rasterize_mode,
                "actual_rasterize_mode": actual_mode,
                "requested_covariance_epsilon": covariance_epsilon,
                "actual_covariance_epsilon": actual_epsilon,
                "requested_gaussian_support_sigma": (
                    DEFAULT_GAUSSIAN_SUPPORT_SIGMA
                ),
                "actual_gaussian_support_sigma": actual_support,
                "matches": bool(
                    actual_mode == args.rasterize_mode
                    and isinstance(actual_epsilon, (int, float))
                    and math.isclose(
                        float(actual_epsilon),
                        covariance_epsilon,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    and isinstance(actual_support, (int, float))
                    and math.isclose(
                        float(actual_support),
                        DEFAULT_GAUSSIAN_SUPPORT_SIGMA,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ),
            }
            if not renderer_configuration["matches"]:
                raise AssertionError(
                    "The rasterize mode, covariance epsilon, or support "
                    "cutoff did not reach "
                    f"the custom backend: {renderer_configuration}."
                )

        # WXYZ quaternions: anisotropy plus rotations around different axes
        # make this fixture sensitive to layout/convention mistakes.
        scene = {
            "means": torch.tensor(
                [[-0.30, -0.05, 3.5], [0.05, 0.10, 4.0], [0.35, -0.10, 4.5]],
                device=device,
            ),
            "scales": torch.tensor(
                [[0.40, 0.12, 0.20], [0.14, 0.35, 0.18], [0.22, 0.13, 0.38]],
                device=device,
            ),
            "rotations": torch.tensor(
                [
                    [0.9238795, 0.0, 0.0, 0.3826834],
                    [0.9659258, 0.0, 0.2588190, 0.0],
                    [0.9807853, 0.1950903, 0.0, 0.0],
                ],
                device=device,
            ),
            "opacities": torch.tensor([0.95, 0.85, 0.75], device=device),
            "features": torch.tensor(
                [[1.0, 0.2, 0.1], [0.1, 1.0, 0.2], [0.2, 0.1, 1.0]],
                device=device,
            ),
            "semantic_ids": torch.tensor(
                [7, 11, 23],
                dtype=torch.int64,
                device=device,
            ),
        }
        service.load_scene(7, **scene)

        camera_transforms = torch.eye(4, device=device).repeat(2, 1, 1)
        camera_transforms[1, 0, 3] = 0.20
        camera_transforms = camera_transforms.contiguous()
        intrinsics = torch.tensor(
            [[4.5, 0.0, 2.5], [0.0, 4.5, 2.0], [0.0, 0.0, 1.0]],
            device=device,
        ).repeat(2, 1, 1).contiguous()
        scene_ids = torch.tensor([7, 7], dtype=torch.int64, device=device)
        active = torch.tensor([1], dtype=torch.int64, device=device)
        outputs = service.render(
            camera_transforms,
            intrinsics,
            scene_ids,
            active_camera_ids=active,
        )
        service.synchronize()

        contract = output_contract(
            outputs,
            semantic_min_alpha=0.01,
            expected_shape=(2, 4, 5),
            require_cuda=not using_fake_backend,
        )
        if not contract["valid"]:
            raise AssertionError(f"Renderer output contract failed: {contract}.")
        inactive_row_defined = bool(
            torch.count_nonzero(outputs["rgb"][0]).item() == 0
            and torch.count_nonzero(outputs["alpha"][0]).item() == 0
            and torch.all(outputs["semantic_id"][0] == -1).item()
            and torch.all(
                torch.isinf(outputs["depth"][0])
                & (outputs["depth"][0] > 0)
            ).item()
        )
        if not inactive_row_defined:
            raise AssertionError(
                "An inactive active-subset camera exposed stale output values."
            )

        capacity_counters = None
        zero_overflow = True
        if using_fake_backend:
            if backend.render_calls != 1:
                raise AssertionError("Fake backend render count mismatch.")
        else:
            if backend.__class__.__name__ != "CustomCudaBackend":
                raise AssertionError(
                    f"Unexpected CUDA backend {backend.__class__.__name__}."
                )
            capacity_counters = backend.check_capacity(synchronize=False)
            zero_overflow = bool(
                capacity_counters["visible_overflow"] == 0
                and capacity_counters["intersection_overflow"] == 0
            )
            if not zero_overflow:
                raise AssertionError(
                    f"Smoke fixture overflowed capacity: {capacity_counters}."
                )

        before = getattr(backend, "render_calls", None)
        service.synchronize()
        if before is not None and backend.render_calls != before:
            raise AssertionError(
                "synchronize() hid a render or simulation step."
            )
        service.shutdown()
        service = None

        result.update(
            {
                "runtime": "Isaac Sim SimulationApp headless",
                "simulation_app": {
                    "renderer": "MinimalRendering",
                    "viewport_updates_disabled": True,
                },
                "backend": backend.__class__.__name__,
                "device": str(device),
                "fabric_scene_delegate": True,
                "extension_started": True,
                "renderer_configuration": renderer_configuration,
                "fixture": {
                    "gaussians": 3,
                    "anisotropic": True,
                    "non_identity_wxyz_rotations": True,
                    "distinct_semantic_ids": [7, 11, 23],
                    "off_axis_camera": True,
                    "active_camera_ids": [1],
                },
                "output_contract": contract,
                "inactive_row_defined": inactive_row_defined,
                "capacity_counters": capacity_counters,
                "zero_overflow": zero_overflow,
                "environment": {
                    "source_commit": (
                        os.environ.get("SOURCE_GIT_COMMIT")
                        or _command_output(["git", "rev-parse", "HEAD"])
                        or "unknown"
                    ),
                    "command": sys.argv,
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "gpu": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else None
                    ),
                    "isaacsim": _package_version("isaacsim"),
                },
                "pass": True,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "ISAACSIM_GAUSSIAN_RENDERER_SMOKE_RESULT",
            json.dumps(result, sort_keys=True),
            flush=True,
        )
        print(SUCCESS_MARKER, flush=True)
        failed = False
    except BaseException as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        traceback.print_exc()
        raise
    finally:
        try:
            if service is not None:
                service.shutdown()
        except BaseException:
            failed = True
            traceback.print_exc()
            raise
        finally:
            close_simulation_app(simulation_app, failed=failed)


if __name__ == "__main__":
    main()
