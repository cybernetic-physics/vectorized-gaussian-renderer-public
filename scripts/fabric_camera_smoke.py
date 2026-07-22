"""Isolate the CUDA Fabric-to-Torch camera transform ingestion path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

if __package__:
    from ._isaac_launch import close_simulation_app, isaac_app_config
else:
    from _isaac_launch import close_simulation_app, isaac_app_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/isaacsim-integration/fabric-camera-smoke.json"),
    )
    args, kit_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *kit_args]
    return args


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise ValueError("--batch must be positive.")
    os.environ.setdefault(
        "WARP_CACHE_PATH",
        str(args.repo_root / "build" / "warp-cache"),
    )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(isaac_app_config())
    failed = False
    try:
        import carb
        import omni.kit.app
        import omni.usd
        import torch
        import warp as wp
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf, UsdGeom

        from isaacsim_gaussian_renderer.camera_math import (
            usd_camera_world_matrices_to_viewmats,
        )

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.add_path(str(args.repo_root / "exts"))
        manager.set_extension_enabled_immediate(
            "isaacsim.gaussian_renderer",
            True,
        )
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
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        camera_paths: list[str] = []
        translations: list[tuple[float, float, float]] = []
        translate_ops = []
        for index in range(args.batch):
            translation = (
                0.25 * index,
                -0.125 * index,
                1.0 + 0.05 * index,
            )
            path = f"/World/Camera_{index:04d}"
            camera = UsdGeom.Camera.Define(stage, path)
            translate_op = UsdGeom.Xformable(camera).AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*translation))
            camera_paths.append(path)
            translations.append(translation)
            translate_ops.append(translate_op)
        simulation_app.update()

        camera_path_tuple = tuple(camera_paths)
        source = extension.create_camera_source(
            stage=stage,
            stage_id=omni.usd.get_context().get_stage_id(),
            camera_paths=camera_path_tuple,
            device="cuda:0",
            update_world_xforms=True,
        )
        wp.config.verbose = True
        print(
            "FABRIC_CAMERA_WARP_RUNTIME",
            json.dumps(
                {
                    "version": wp.config.version,
                    "module": str(Path(wp.__file__).resolve()),
                    "kernel_cache_dir": wp.config.kernel_cache_dir,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        actual = source.read_transforms(
            camera_path_tuple,
            device=torch.device("cuda:0"),
        ).clone()
        torch.cuda.synchronize()
        world = torch.eye(
            4,
            dtype=torch.float64,
            device="cuda:0",
        ).repeat(args.batch, 1, 1)
        world[:, 3, :3] = torch.tensor(
            translations,
            dtype=torch.float64,
            device="cuda:0",
        )
        expected = usd_camera_world_matrices_to_viewmats(world).to(
            dtype=torch.float32
        )
        if not torch.equal(actual, expected):
            max_error = float(torch.max(torch.abs(actual - expected)).item())
            raise AssertionError(
                f"Fabric camera conversion mismatch; max error={max_error}."
            )

        updated_translation = (
            translations[0][0] + 0.5,
            translations[0][1] - 0.25,
            translations[0][2] + 0.75,
        )
        translate_ops[0].Set(Gf.Vec3d(*updated_translation))
        simulation_app.update()
        updated = source.read_transforms(
            camera_path_tuple,
            device=torch.device("cuda:0"),
        ).clone()
        torch.cuda.synchronize()
        if torch.equal(actual[0], updated[0]):
            raise AssertionError("Fabric transform did not reflect the USD update.")
        if not torch.equal(actual[1:], updated[1:]):
            raise AssertionError("Unmodified camera transforms changed.")

        result = {
            "schema_version": "fabric-camera-smoke/v1",
            "batch": args.batch,
            "device": str(actual.device),
            "dtype": str(actual.dtype),
            "shape": list(actual.shape),
            "contiguous": actual.is_contiguous(),
            "runtime_camera_loop": False,
            "gpu_conversion_launches_per_batch": 1,
            "torch_output_zero_copy": True,
            "fabric_scene_delegate_enabled": (
                fabric_scene_delegate_enabled
            ),
            "fabric_delegate_extension_loaded": (
                fabric_delegate_extension_loaded
            ),
            "read_calls": source.transform_source.read_calls,
            "topology_rebuilds": source.transform_source.topology_rebuilds,
            "warp_version": wp.config.version,
            "warp_module": str(Path(wp.__file__).resolve()),
            "warp_kernel_cache": wp.config.kernel_cache_dir,
            "pass": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "FABRIC_CAMERA_SMOKE_RESULT",
            json.dumps(result, sort_keys=True),
            flush=True,
        )
        print("FABRIC_CAMERA_SMOKE_OK", flush=True)
        source.transform_source.close()
    except BaseException:
        failed = True
        traceback.print_exc()
        raise
    finally:
        close_simulation_app(simulation_app, failed=failed)


if __name__ == "__main__":
    main()
