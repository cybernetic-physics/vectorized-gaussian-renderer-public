"""Finite headless Isaac Sim lifecycle smoke for GaussianLidarService."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lidar/isaacsim-lidar-smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WARP_CACHE_PATH", str(args.repo_root / "build" / "warp-cache"))
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    service = None
    result: dict[str, object] = {
        "schema_version": "isaacsim-gaussian-lidar-smoke/v1",
        "pass": False,
    }
    try:
        import carb
        import omni.kit.app
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf, UsdGeom, UsdPhysics

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.add_path(str(args.repo_root / "exts"))
        manager.set_extension_enabled_immediate("isaacsim.gaussian_renderer", True)
        settings = carb.settings.get_settings()
        settings.set("/app/useFabricSceneDelegate", True)
        from isaacsim_gaussian_renderer_extension import extension as extension_module

        extension = extension_module.STARTUP_INSTANCE
        if extension is None:
            raise RuntimeError("Kit did not start isaacsim.gaussian_renderer.")
        create_new_stage()
        world = World(stage_units_in_meters=1.0)
        world.set_simulation_dt(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        sensor = UsdGeom.Xform.Define(stage, "/World/GaussianLidar")
        sensor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        world.reset()
        simulation_app.update()

        device = torch.device("cuda:0")
        service = extension.create_lidar_service(
            stage=stage,
            device=str(device),
            max_sensors=2,
            max_rays=4,
            returns=2,
        )
        half = math.sqrt(0.5)
        service.load_scene(
            41,
            means=torch.tensor(
                [[5.0, 0.0, 0.0], [5.01, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
                device=device,
                dtype=torch.float32,
            ),
            scales=torch.tensor([[0.01, 1.0, 1.0]] * 4, device=device, dtype=torch.float32),
            rotations=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [half, 0.0, 0.0, half],
                ],
                device=device,
                dtype=torch.float32,
            ),
            opacities=torch.ones((4,), device=device, dtype=torch.float32),
            semantic_ids=torch.tensor([9, 3, 11, 21], device=device, dtype=torch.int64),
            reflectivity=torch.tensor([0.2, 0.6, 0.8, 0.7], device=device, dtype=torch.float32),
        )
        service.synchronize()
        directions = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        offsets = torch.tensor([0, 100, 200], device=device, dtype=torch.int64)
        sensor_to_world = torch.eye(4, device=device).repeat(2, 1, 1).contiguous()
        sensor_to_world[1, 2, 3] = 0.5
        scene_ids = torch.tensor([41, 41], device=device, dtype=torch.int64)
        simulation_time_before = float(world.current_time)
        outputs = service.render_lidar(directions, offsets, sensor_to_world, scene_ids)
        service.synchronize()
        simulation_time_after = float(world.current_time)
        counters = service.backend.check_errors(synchronize=False)
        if simulation_time_after != simulation_time_before:
            raise AssertionError("render_lidar advanced Isaac physics time.")
        if not all(tensor.is_cuda for tensor in outputs.values()):
            raise AssertionError("LiDAR outputs must remain CUDA-resident.")
        if outputs["valid"][:, 0, 0].sum().item() != 2:
            raise AssertionError("Expected both sensors to see the front wall.")
        valid = outputs["valid"]
        if not torch.isfinite(outputs["range_m"][valid]).all().item():
            raise AssertionError("Valid ranges must be finite.")
        if not torch.isfinite(outputs["position_world_m"][valid]).all().item():
            raise AssertionError("Valid positions must be finite.")
        if not torch.isfinite(outputs["intensity"][valid]).all().item():
            raise AssertionError("Valid intensity must be finite.")
        result.update(
            {
                "pass": True,
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "extension_started": True,
                "fabric_scene_delegate_enabled": settings.get_as_bool("/app/useFabricSceneDelegate"),
                "physics_time_before": simulation_time_before,
                "physics_time_after": simulation_time_after,
                "outputs_cuda_resident": True,
                "valid_returns": int(valid.sum().item()),
                "counters": counters,
                "dynamic_geometry_supported": False,
                "motion_distortion_supported": False,
            }
        )
        print("ISAACSIM_GAUSSIAN_LIDAR_SMOKE_OK")
    finally:
        if service is not None:
            service.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        simulation_app.close()


if __name__ == "__main__":
    main()
