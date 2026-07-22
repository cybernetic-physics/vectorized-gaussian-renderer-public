#!/usr/bin/env python3
"""Validate that a scan-derived GLB can become a static Isaac PhysX collider.

The Gaussian PLY stays visual-only.  This narrow smoke checks the separately
generated coarse collision proxy at the USD/PhysX boundary before it is shared
across vectorized environments with robot articulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-glb", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--contact-steps", type=int, default=180)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.contact_steps <= 0:
        raise ValueError("updates and contact-steps must be positive.")
    if not args.collision_glb.is_file():
        raise FileNotFoundError(args.collision_glb)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        import omni.timeline
        import omni.usd
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics.CreateGravityMagnitudeAttr(9.81)
        proxy = UsdGeom.Xform.Define(stage, "/World/HomeScanCollisionProxy")
        proxy.GetPrim().GetReferences().AddReference(
            args.collision_glb.resolve().as_posix()
        )
        # The Gaussian camera contract is Y-down.  Rotate the proxy so its
        # visual vertical (-Y) is Isaac's +Z physical vertical.
        UsdGeom.XformCommonAPI(proxy).SetRotate(
            (-90.0, 0.0, 0.0),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        for _ in range(args.updates):
            simulation_app.update()

        prefix = str(proxy.GetPath()) + "/"
        mesh_prims = [
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(prefix) and prim.IsA(UsdGeom.Mesh)
        ]
        if not mesh_prims:
            raise RuntimeError("Collision GLB reference exposed no USD mesh prims.")
        collision_paths: list[str] = []
        for prim in mesh_prims:
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set(
                UsdPhysics.Tokens.none
            )
            if (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                and prim.HasAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI)
            ):
                collision_paths.append(str(prim.GetPath()))
        if len(collision_paths) != len(mesh_prims):
            raise AssertionError("Not every collision-proxy mesh accepted CollisionAPI.")
        for _ in range(args.updates):
            simulation_app.update()

        support_mesh = UsdGeom.Mesh(mesh_prims[0])
        local_points = support_mesh.GetPointsAttr().Get()
        if not local_points:
            raise RuntimeError("Collision-proxy mesh contains no points.")
        world_from_mesh = UsdGeom.Xformable(mesh_prims[0]).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        support_point = max(
            (world_from_mesh.Transform(point) for point in local_points),
            key=lambda point: point[2],
        )
        # Drop onto a mesh vertex that is already known to be on the imported
        # proxy. This proves static triangle collision without assuming which
        # Home Scan coordinate is the interior floor.
        probe = UsdGeom.Sphere.Define(stage, "/World/ContactProbe")
        probe.CreateRadiusAttr(0.10)
        probe.AddTranslateOp().Set(
            Gf.Vec3d(support_point[0], support_point[1], support_point[2] + 0.5)
        )
        UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
        UsdPhysics.MassAPI.Apply(probe.GetPrim()).CreateMassAttr(1.0)
        xform_cache = UsdGeom.XformCache()
        initial_probe_z = float(
            xform_cache.GetLocalToWorldTransform(probe.GetPrim())
            .ExtractTranslation()[2]
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(args.contact_steps):
            simulation_app.update()
        timeline.stop()
        xform_cache.Clear()
        final_probe_z = float(
            xform_cache.GetLocalToWorldTransform(probe.GetPrim())
            .ExtractTranslation()[2]
        )
        if final_probe_z >= initial_probe_z - 0.25:
            raise AssertionError("Contact probe did not advance under gravity.")
        if final_probe_z <= -1.0:
            raise AssertionError(
                f"Contact probe fell through the scan proxy: z={final_probe_z}."
            )

        report = {
            "schema_version": "isaacsim-scan-collision-proxy-import-smoke/v1",
            "pass": True,
            "scope": (
                "Static scan-proxy authoring plus a dynamic sphere contact probe. "
                "It does not claim Gaussian collision or complete robot contact dynamics."
            ),
            "collision_glb": str(args.collision_glb.resolve()),
            "collision_glb_sha256": sha256(args.collision_glb),
            "collision_glb_bytes": args.collision_glb.stat().st_size,
            "source_axis_conversion": "Gaussian Y-down to Isaac Z-up via proxy RotateX(-90).",
            "mesh_prim_count": len(mesh_prims),
            "collision_prim_count": len(collision_paths),
            "collision_prim_examples": collision_paths[:12],
            "contact_probe": {
                "support_surface_point": [
                    float(support_point[0]),
                    float(support_point[1]),
                    float(support_point[2]),
                ],
                "radius": 0.10,
                "steps": args.contact_steps,
                "initial_z": initial_probe_z,
                "final_z": final_probe_z,
            },
        }
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("ISAACSIM_SCAN_COLLISION_PROXY_IMPORT_OK " + json.dumps(report, sort_keys=True))
    except Exception as error:
        args.report.write_text(
            json.dumps(
                {
                    "schema_version": "isaacsim-scan-collision-proxy-import-smoke/v1",
                    "pass": False,
                    "failure": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
