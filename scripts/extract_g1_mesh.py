#!/usr/bin/env python3
"""Extract the G1 robot's render meshes to a .npz for the custom mesh tracer.

One-time offline step: composes the G1 physics USD, gathers every renderable
UsdGeom.Mesh (purpose != guide, collision scopes skipped), fan-triangulates,
bakes points into the robot root frame, and saves merged vertices/triangles
plus per-link ranges so the tracer can later instance links individually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--g1-usda", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import numpy as np
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(args.g1_usda.resolve().as_posix())
        if stage is None:
            raise RuntimeError(f"cannot open {args.g1_usda}")

        xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
        vertices_list, tris_list = [], []
        link_names, link_tri_offsets = [], []
        skipped = []
        vert_base = 0
        tri_count = 0
        # Instanceable payloads (e.g. Isaac's URDF importer output) hide their
        # meshes behind instance proxies, which plain Traverse() skips.
        prim_range = Usd.PrimRange.Stage(
            stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        )
        for prim in prim_range:
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = prim.GetPath().pathString
            img = UsdGeom.Imageable(prim)
            purpose = img.ComputePurpose()
            lowered = path.lower()
            if purpose == UsdGeom.Tokens.guide or "collision" in lowered:
                skipped.append(path)
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            if points.size == 0 or counts.size == 0:
                skipped.append(path + " (empty)")
                continue
            # Bake into the robot root frame.
            m = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)
            homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
            world = (homo @ m)[:, :3]
            # Fan-triangulate arbitrary polygons.
            tris = []
            cursor = 0
            for c in counts:
                poly = indices[cursor:cursor + c]
                for k in range(1, c - 1):
                    tris.append((poly[0], poly[k], poly[k + 1]))
                cursor += c
            tris = np.asarray(tris, dtype=np.int64) + vert_base
            vertices_list.append(world.astype(np.float32))
            tris_list.append(tris.astype(np.int32))
            link_names.append(path)
            link_tri_offsets.append(tri_count)
            vert_base += len(world)
            tri_count += len(tris)

        if not tris_list:
            raise RuntimeError("no renderable meshes found")
        vertices = np.concatenate(vertices_list)
        triangles = np.concatenate(tris_list)
        bb_min, bb_max = vertices.min(axis=0), vertices.max(axis=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            vertices=vertices,
            triangles=triangles,
            link_names=np.asarray(link_names),
            link_tri_offsets=np.asarray(link_tri_offsets + [tri_count], dtype=np.int64),
            bbox_min=bb_min,
            bbox_max=bb_max,
        )
        summary = {
            "meshes": len(link_names),
            "skipped": len(skipped),
            "vertices": int(len(vertices)),
            "triangles": int(len(triangles)),
            "bbox_min": bb_min.tolist(),
            "bbox_max": bb_max.tolist(),
        }
        print("G1_MESH_EXTRACT_OK " + json.dumps(summary))
    finally:
        app.close()


if __name__ == "__main__":
    main()
