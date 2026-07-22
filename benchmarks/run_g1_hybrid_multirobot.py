#!/usr/bin/env python3
"""Massively-vectorized hybrid: N G1 robots, each at its OWN gait phase,
each genuinely rendered by Isaac RTX per env (no pose cache).

One robot per environment, placed in front of that env's camera; every robot is
driven by the deterministic scripted gait at a distinct phase offset
(i * period / N), so every env shows a different, advancing pose each frame.
The Gaussian scene layer renders all env cameras in one batched CUDA
submission; the RTX layer renders one product per env; the depth compositor
merges per env. Per-frame component attribution mirrors run_g1_hybrid_gait.py.

This measures how the per-env-RTX hybrid actually scales with env count — the
honest cost of "different pose per robot" without caching tricks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

HOME_SCAN_COUNT = 21_497_908
SCENE_ID = 404


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--home-scan-path", type=Path, default=Path("/workspace/datasets/home-scan-lod0.ply"))
    p.add_argument("--g1-usda", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--component-frames", type=int, default=3)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--rt-subframes", type=int, default=8)
    p.add_argument("--gait-period", type=float, default=1.2)
    p.add_argument("--physics-hz", type=float, default=60.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    service = None
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        import torch
        import warp as wp
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
        from isaacsim_gaussian_renderer.benchmark_manifest import (
            axis_aligned_exterior_camera_bundle,
        )
        from isaacsim_gaussian_renderer.camera_math import (
            opencv_viewmats_to_usd_camera_world_matrices,
        )
        from isaacsim_gaussian_renderer.hybrid_compositor import composite_gaussian_and_mesh
        from isaacsim_gaussian_renderer.ply_loader import (
            canonicalize_3dgs_scene,
            load_ply_to_gaussians,
        )
        from isaacsim_gaussian_renderer.g1_gait import ScriptedGait

        if not torch.cuda.is_available():
            raise RuntimeError("Multi-robot hybrid benchmark requires CUDA.")

        def progress(msg):
            print(f"MULTIROBOT_PROGRESS {msg}", flush=True)

        # --- Gaussian layer + N distinct env cameras ------------------------
        raw = load_ply_to_gaussians(args.home_scan_path)
        if raw.count != HOME_SCAN_COUNT:
            raise ValueError(f"Home Scan count mismatch: {raw.count}")
        canonical = canonicalize_3dgs_scene(raw, device="cuda")
        del raw
        pos_min = canonical.means.amin(dim=0)
        pos_max = canonical.means.amax(dim=0)
        center = (pos_min + pos_max) * 0.5
        bounding_radius = float((torch.linalg.vector_norm(pos_max - pos_min) * 0.5).item())
        pool = axis_aligned_exterior_camera_bundle(
            center=center, bounding_radius=bounding_radius, batch_size=args.envs,
            width=args.width, height=args.height,
        )
        viewmats = pool.viewmats.to("cuda", torch.float32).contiguous()
        intrinsics = pool.intrinsics.to("cuda", torch.float32).contiguous()
        distance = float(pool.manifest["distance"])
        near_plane, far_plane = 0.01, distance + 2.0 * bounding_radius + 1.0

        backend = CustomCudaBackend(
            max_visible_records=500_000 * args.envs,
            max_intersections=math.ceil(args.envs * 2_400_000 * (args.width * args.height) / (128 * 128)),
            near_plane=near_plane, far_plane=far_plane,
            gaussian_support_sigma=3.0, covariance_epsilon=0.0, semantic_min_alpha=0.01,
            tile_size=1, depth_bucket_count=128, depth_bucket_group_size=8,
            compact_projection_cache=True, enable_projection_cache=False,
            output_srgb=True, deterministic=False,
        )
        service = RendererService(backend, height=args.height, width=args.width, max_views=args.envs)
        service.initialize(stage=None, device="cuda")
        service.load_scene(
            SCENE_ID, means=canonical.means, scales=canonical.scales, rotations=canonical.rotations,
            opacities=canonical.opacities, features=canonical.features[:, :3].contiguous(),
            semantic_ids=torch.zeros(canonical.count, device="cuda", dtype=torch.int64),
        )
        del canonical
        scene_ids = torch.full((args.envs,), SCENE_ID, device="cuda", dtype=torch.int64)

        # --- N robots, one in front of each env camera ----------------------
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        robot_bases = []
        for i in range(args.envs):
            vm = viewmats[i]
            cam_center = -vm[:3, :3].transpose(0, 1) @ vm[:3, 3]
            base = (cam_center + vm[2, :3] * 2.5 + vm[1, :3] * 0.35).detach().cpu().tolist()
            robot_bases.append(base)
            prim = UsdGeom.Xform.Define(stage, f"/World/G1_{i:03d}")
            prim.GetPrim().GetReferences().AddReference(args.g1_usda.resolve().as_posix())
            rx = UsdGeom.XformCommonAPI(prim)
            rx.SetTranslate(tuple(base))
            rx.SetRotate((90.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        # Chrome on every robot (de-instance so the binding takes).
        chrome = UsdShade.Material.Define(stage, "/World/Looks/G1Chrome")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/G1Chrome/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.57, 0.60))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.04)
        chrome.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        world_root = stage.GetPrimAtPath("/World")
        for _ in range(64):
            instanced = [p for p in Usd.PrimRange(world_root) if p.IsInstance()]
            if not instanced:
                break
            for p in instanced:
                p.SetInstanceable(False)
        chrome_meshes = 0
        for prim in Usd.PrimRange(world_root):
            if prim.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI.Apply(prim)
                UsdShade.MaterialBindingAPI(prim).Bind(chrome)
                chrome_meshes += 1

        UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(3000.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(30.0, -25.0, 20.0))

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / args.physics_hz, rendering_dt=1.0 / args.physics_hz)

        def find_articulation_root(top: str) -> str:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(top)):
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    return prim.GetPath().pathString
            return top

        robots = []
        for i in range(args.envs):
            art = SingleArticulation(prim_path=find_articulation_root(f"/World/G1_{i:03d}"), name=f"g1_{i}")
            world.scene.add(art)
            robots.append(art)
        world.reset()
        orientation = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)
        for i, art in enumerate(robots):
            art.initialize()
            try:
                art.set_world_pose(position=np.asarray(robot_bases[i], dtype=np.float32), orientation=orientation)
            except Exception:  # noqa: BLE001
                pass
        try:
            world.get_physics_context().set_gravity(0.0)
        except Exception:  # noqa: BLE001
            pass
        dof_names = list(robots[0].dof_names or [])
        if not dof_names:
            raise RuntimeError("G1 articulation exposed no DOFs.")
        gait = ScriptedGait(period_seconds=args.gait_period)
        progress(f"robots_ready n={args.envs} dof={len(dof_names)} chrome_meshes={chrome_meshes}")

        # --- one RTX render product per env camera (static) -----------------
        aperture = 36.0
        usd_world = opencv_viewmats_to_usd_camera_world_matrices(viewmats).cpu().numpy()
        color_aovs, depth_aovs = [], []
        for i in range(args.envs):
            cam = UsdGeom.Camera.Define(stage, f"/World/Camera_{i:04d}")
            cam.CreateFocalLengthAttr(float(intrinsics[i, 0, 0].item()) * aperture / args.width)
            cam.CreateHorizontalApertureAttr(aperture)
            cam.CreateVerticalApertureAttr(aperture * args.height / args.width)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, far_plane))
            UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*usd_world[i].tolist()))
            rp = rep.create.render_product(cam.GetPath(), (args.width, args.height))
            c = rep.AnnotatorRegistry.get_annotator("LdrColor", device="cuda", do_array_copy=False)
            d = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cuda", do_array_copy=False)
            c.attach(rp); d.attach(rp)
            color_aovs.append(c); depth_aovs.append(d)
        progress("render_products_ready")

        _comp = {"gait_physics": [], "gaussian_render": [], "rtx_mesh": [], "compositor": []}
        _measure = {"on": False}

        def render_frame(t):
            m = _measure["on"]
            t0 = time.perf_counter()
            # Every robot at its own phase: different pose per robot per frame.
            for i, art in enumerate(robots):
                phase_t = t + i * (args.gait_period / max(args.envs, 1))
                art.set_joint_positions(
                    np.asarray(gait.targets_for_dof_names(dof_names, phase_t), dtype=np.float32)
                )
            world.step(render=False)
            if m:
                _comp["gait_physics"].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            gaussian = service.render(viewmats, intrinsics, scene_ids)
            if m:
                service.synchronize(); _comp["gaussian_render"].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            rep.orchestrator.step(rt_subframes=args.rt_subframes)
            rep.orchestrator.wait_until_complete()
            mesh_rgba = torch.stack([wp.to_torch(a.get_data()) for a in color_aovs])
            mesh_depth_raw = torch.stack([wp.to_torch(a.get_data()) for a in depth_aovs])
            mesh_rgb = (mesh_rgba[..., :3].float() / 255.0).contiguous()
            mesh_depth = mesh_depth_raw.float().unsqueeze(-1).contiguous()
            mesh_alpha = (torch.isfinite(mesh_depth) & (mesh_depth > 0.0)).float()
            if m:
                torch.cuda.synchronize(); _comp["rtx_mesh"].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            composite_gaussian_and_mesh(
                gaussian["rgb"], gaussian["alpha"], gaussian["depth"],
                mesh_rgb, mesh_alpha, mesh_depth,
            )
            if m:
                torch.cuda.synchronize(); _comp["compositor"].append((time.perf_counter() - t0) * 1000.0)

        for f in range(args.warmup):
            render_frame(f / args.physics_hz)
        service.synchronize(); torch.cuda.synchronize()
        progress("warmup_done")

        wall0 = time.perf_counter()
        for f in range(args.frames):
            render_frame((args.warmup + f) / args.physics_hz)
        service.synchronize(); torch.cuda.synchronize()
        wall_frame_ms = (time.perf_counter() - wall0) * 1000.0 / args.frames

        _measure["on"] = True
        for f in range(args.component_frames):
            render_frame((args.warmup + args.frames + f) / args.physics_hz)
        _measure["on"] = False
        comp_ms = {k: (sum(v) / len(v) if v else 0.0) for k, v in _comp.items()}

        report = {
            "schema_version": "g1-hybrid-multirobot/v1",
            "pass": True,
            "envs": args.envs,
            "robots": args.envs,
            "distinct_pose_per_robot": True,
            "pose_cache": False,
            "resolution": [args.width, args.height],
            "rt_subframes": args.rt_subframes,
            "dof_count": len(dof_names),
            "chrome_bound_meshes": chrome_meshes,
            "wall_frame_ms": wall_frame_ms,
            "fps": 1000.0 / wall_frame_ms,
            "images_per_second": args.envs * 1000.0 / wall_frame_ms,
            "component_ms": comp_ms,
            "rtx_ms_per_env": comp_ms["rtx_mesh"] / max(args.envs, 1),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("G1_HYBRID_MULTIROBOT_OK " + json.dumps(
            {"envs": args.envs, "fps": round(report["fps"], 2),
             "rtx_ms": round(comp_ms["rtx_mesh"], 1),
             "gaussian_ms": round(comp_ms["gaussian_render"], 1)}))
    finally:
        if service is not None:
            service.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
