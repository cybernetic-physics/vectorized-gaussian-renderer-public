#!/usr/bin/env python3
"""GO/NO-GO GATE: pure vectorized G1 mesh trace cost, no renderer wrapper.

Thesis under test (BOTTLENECK_REPORT.md): the Isaac-RTX robot layer's measured
36.6 ms/env is framework tax, not ray-tracing silicon. This traces the actual
G1 triangles with a Warp CUDA software BVH — ONE launch covering
envs × H × W rays — with per-env rigid robot transforms applied in ray space
(software instancing: zero per-frame geometry rebuild, transforms animated
every frame so nothing can cache). Depth + lambert land directly in torch
tensors, ready for the existing hybrid compositor.

If even a software tracer hits <= ~4 ms/env (Gaussian-layer parity), the
custom-tracer bet is proven with margin, and OptiX RT cores remain as pure
upside. Requires warp (Isaac ships it in extscache; see run wrapper).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh-npz", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--synthetic-tris",
        type=int,
        default=0,
        help="Replace the mesh with a dense robot-sized ellipsoid of ~N triangles "
        "(stress case for production-quality meshes; full silhouette).",
    )
    p.add_argument("--envs-list", default="1,2,4,8,16,32")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--per-link", action="store_true",
        help="Articulated mode: every link independently posed per env per "
        "frame (per-env mesh copy, vertex transform + BVH refit each frame). "
        "Requires the real G1 npz (link_tri_offsets). Link pivots are bbox "
        "centroids, not URDF joints — this measures the articulation COST "
        "honestly; joint-accurate poses need pivots from the URDF (follow-up).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    import warp as wp

    wp.init()
    device = "cuda:0"

    if args.synthetic_tris > 0:
        # Dense robot-sized ellipsoid (semi-axes 0.10, 0.10, 0.35 m): full
        # silhouette, ~N triangles — the production-mesh stress case.
        rows = max(8, int(math.sqrt(args.synthetic_tris / 2.0)))
        cols = 2 * rows
        lat = np.linspace(0.0, math.pi, rows + 1)
        lon = np.linspace(0.0, 2.0 * math.pi, cols, endpoint=False)
        grid_lat, grid_lon = np.meshgrid(lat, lon, indexing="ij")
        pts = np.stack(
            [
                0.10 * np.sin(grid_lat) * np.cos(grid_lon),
                0.10 * np.sin(grid_lat) * np.sin(grid_lon),
                0.35 * np.cos(grid_lat),
            ],
            axis=-1,
        ).reshape(-1, 3).astype(np.float32)
        faces = []
        for r in range(rows):
            for c in range(cols):
                a = r * cols + c
                b = r * cols + (c + 1) % cols
                cc = (r + 1) * cols + c
                dd = (r + 1) * cols + (c + 1) % cols
                faces.append((a, b, dd))
                faces.append((a, dd, cc))
        vertices = pts
        triangles = np.asarray(faces, dtype=np.int32)
        bb_min, bb_max = vertices.min(axis=0), vertices.max(axis=0)
    else:
        data = np.load(args.mesh_npz, allow_pickle=False)
        vertices = data["vertices"].astype(np.float32)
        triangles = data["triangles"].astype(np.int32)
        bb_min, bb_max = data["bbox_min"], data["bbox_max"]
    center = (bb_min + bb_max) * 0.5
    radius = float(np.linalg.norm(bb_max - bb_min)) * 0.5

    build_start = time.perf_counter()
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(triangles.reshape(-1), dtype=wp.int32, device=device),
    )
    wp.synchronize()
    bvh_build_ms = (time.perf_counter() - build_start) * 1000.0

    @wp.kernel
    def trace(
        mesh_id: wp.uint64,
        cam_origins: wp.array(dtype=wp.vec3),
        robot_rot: wp.array(dtype=wp.mat33),   # world->robot rotation per env
        robot_pos: wp.array(dtype=wp.vec3),    # robot origin per env
        mesh_center: wp.vec3,                  # mesh-frame pivot (bbox center)
        fx: float, fy: float, cx: float, cy: float,
        width: int, height: int, max_t: float,
        depth: wp.array3d(dtype=float),
        shade: wp.array3d(dtype=float),
    ):
        e, y, x = wp.tid()
        # OpenCV pinhole: camera looks +z, axis-aligned (identity rotation),
        # matching the benchmark camera-bundle convention.
        d_world = wp.normalize(
            wp.vec3((float(x) + 0.5 - cx) / fx, (float(y) + 0.5 - cy) / fy, 1.0)
        )
        o_world = cam_origins[e]
        # Rigid instancing in ray space: transform the ray into the robot's
        # local frame instead of ever touching the geometry/BVH.
        rot = robot_rot[e]
        o_local = rot @ (o_world - robot_pos[e]) + mesh_center
        d_local = wp.normalize(rot @ d_world)
        q = wp.mesh_query_ray(mesh_id, o_local, d_local, max_t)
        if q.result:
            depth[e, y, x] = q.t
            n = wp.mesh_eval_face_normal(mesh_id, q.face)
            shade[e, y, x] = wp.abs(wp.dot(n, d_local))
        else:
            depth[e, y, x] = 0.0
            shade[e, y, x] = 0.0

    # ---- optional per-link articulation setup -------------------------------
    link_ids_np = link_pivots_np = None
    n_links = 0
    if args.per_link:
        if args.synthetic_tris > 0:
            raise SystemExit("--per-link requires the real G1 npz")
        data = np.load(args.mesh_npz, allow_pickle=False)
        if "link_tri_offsets" not in data:
            raise SystemExit("npz lacks link_tri_offsets; re-run extract_g1_mesh.py")
        offsets = data["link_tri_offsets"].astype(np.int64)
        n_links = len(offsets) - 1
        link_ids_np = np.zeros(len(vertices), dtype=np.int32)
        link_pivots_np = np.zeros((n_links, 3), dtype=np.float32)
        for L in range(n_links):
            vidx = np.unique(triangles[offsets[L]:offsets[L + 1]].reshape(-1))
            link_ids_np[vidx] = L
            link_pivots_np[L] = vertices[vidx].mean(axis=0)

    @wp.kernel
    def articulate(
        base_pts: wp.array(dtype=wp.vec3),
        link_id: wp.array(dtype=wp.int32),
        link_rot: wp.array2d(dtype=wp.mat33),  # [envs, links]
        link_piv: wp.array(dtype=wp.vec3),
        env: int,
        out_pts: wp.array(dtype=wp.vec3),
    ):
        v = wp.tid()
        L = link_id[v]
        p = base_pts[v]
        out_pts[v] = link_rot[env, L] @ (p - link_piv[L]) + link_piv[L]

    @wp.kernel
    def trace_ml(
        mesh_ids: wp.array(dtype=wp.uint64),   # one articulated mesh per env
        cam_origins: wp.array(dtype=wp.vec3),
        robot_rot: wp.array(dtype=wp.mat33),
        robot_pos: wp.array(dtype=wp.vec3),
        mesh_center: wp.vec3,
        fx: float, fy: float, cx: float, cy: float,
        width: int, height: int, max_t: float,
        depth: wp.array3d(dtype=float),
        shade: wp.array3d(dtype=float),
    ):
        e, y, x = wp.tid()
        d_world = wp.normalize(
            wp.vec3((float(x) + 0.5 - cx) / fx, (float(y) + 0.5 - cy) / fy, 1.0)
        )
        o_world = cam_origins[e]
        rot = robot_rot[e]
        o_local = rot @ (o_world - robot_pos[e]) + mesh_center
        d_local = wp.normalize(rot @ d_world)
        q = wp.mesh_query_ray(mesh_ids[e], o_local, d_local, max_t)
        if q.result:
            depth[e, y, x] = q.t
            n = wp.mesh_eval_face_normal(mesh_ids[e], q.face)
            shade[e, y, x] = wp.abs(wp.dot(n, d_local))
        else:
            depth[e, y, x] = 0.0
            shade[e, y, x] = 0.0

    focal = 0.9 * float(args.width)
    fy = focal * float(args.height) / float(args.width)
    cx, cy = args.width * 0.5, args.height * 0.5
    # Frame the robot so it fills a realistic footprint (the honest workload:
    # a large fraction of rays must actually traverse the BVH and hit).
    cam_distance = 2.6 * radius
    max_t = cam_distance + 4.0 * radius + 1.0
    # Base orientation (world->robot): stand the z-up robot upright in the
    # OpenCV camera frame (screen up = world -y maps to robot +z).
    base_rot = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )

    points = []
    results = {}
    for envs in [int(v) for v in args.envs_list.split(",") if v.strip()]:
        # Robots stand at the world origin of each env; cameras back off along -z.
        origins_np = np.zeros((envs, 3), dtype=np.float32)
        origins_np[:, 0] = np.linspace(-0.1 * radius, 0.1 * radius, envs)
        origins_np[:, 2] = -cam_distance
        cam_origins = wp.array(origins_np, dtype=wp.vec3, device=device)
        robot_rot = wp.zeros(envs, dtype=wp.mat33, device=device)
        robot_pos = wp.zeros(envs, dtype=wp.vec3, device=device)

        depth_t = torch.zeros((envs, args.height, args.width), device=device)
        shade_t = torch.zeros((envs, args.height, args.width), device=device)
        depth_w = wp.from_torch(depth_t, dtype=wp.float32)
        shade_w = wp.from_torch(shade_t, dtype=wp.float32)

        env_meshes = mesh_ids_w = base_pts_w = link_ids_w = link_piv_w = None
        rng_axes = None
        articulate_ms: list[float] = []
        if args.per_link:
            base_pts_w = wp.array(vertices, dtype=wp.vec3, device=device)
            link_ids_w = wp.array(link_ids_np, dtype=wp.int32, device=device)
            link_piv_w = wp.array(link_pivots_np, dtype=wp.vec3, device=device)
            env_meshes = [
                wp.Mesh(
                    points=wp.array(vertices, dtype=wp.vec3, device=device),
                    indices=wp.array(triangles.reshape(-1), dtype=wp.int32,
                                     device=device),
                )
                for _ in range(envs)
            ]
            mesh_ids_w = wp.array(
                np.array([m.id for m in env_meshes], dtype=np.uint64),
                dtype=wp.uint64, device=device,
            )
            rng = np.random.default_rng(11)
            rng_axes = rng.normal(size=(n_links, 3))
            rng_axes /= np.linalg.norm(rng_axes, axis=1, keepdims=True)

        def link_rots_for(frame: int) -> np.ndarray:
            # Per-env, per-link sinusoidal rotation about a fixed random axis
            # through the link centroid — every link independently posed.
            phase = (
                frame * 0.03
                + np.arange(n_links)[None, :] / max(n_links, 1)
                + np.arange(envs)[:, None] / max(envs, 1)
            )
            ang = 0.3 * np.sin(2.0 * math.pi * phase)  # [envs, links]
            k = rng_axes[None, :, :]  # [1, links, 3]
            K = np.zeros((1, n_links, 3, 3), dtype=np.float64)
            K[0, :, 0, 1] = -k[0, :, 2]; K[0, :, 0, 2] = k[0, :, 1]
            K[0, :, 1, 0] = k[0, :, 2];  K[0, :, 1, 2] = -k[0, :, 0]
            K[0, :, 2, 0] = -k[0, :, 1]; K[0, :, 2, 1] = k[0, :, 0]
            s = np.sin(ang)[..., None, None]
            c = np.cos(ang)[..., None, None]
            eye = np.eye(3)[None, None]
            R = eye + s * K + (1.0 - c) * (K @ K)
            return R.astype(np.float32)

        def set_poses(frame: int) -> None:
            # Per-env, per-frame rigid pose (distinct heading per robot, advancing
            # every frame — mimics desynchronized walkers; BVH is never rebuilt).
            angles = (
                np.arange(envs) * (2.0 * math.pi / max(envs, 1))
                + frame * 0.05
            )
            # Heading about the robot's vertical (mesh z), composed onto the
            # stand-upright base orientation; advancing every frame.
            cos, sin = np.cos(angles), np.sin(angles)
            rz = np.zeros((envs, 3, 3), dtype=np.float32)
            rz[:, 0, 0] = cos
            rz[:, 0, 1] = -sin
            rz[:, 1, 0] = sin
            rz[:, 1, 1] = cos
            rz[:, 2, 2] = 1.0
            rots = np.einsum("eij,jk->eik", rz, base_rot)
            wp.copy(robot_rot, wp.array(np.ascontiguousarray(rots), dtype=wp.mat33, device=device))
            # robot_pos stays at the origin (already zero-initialized).

        def frame(i: int) -> None:
            set_poses(i)
            mc = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
            if args.per_link:
                t0 = time.perf_counter()
                rots = wp.array2d(link_rots_for(i), dtype=wp.mat33, device=device)
                for e, m in enumerate(env_meshes):
                    wp.launch(
                        articulate, dim=len(vertices),
                        inputs=[base_pts_w, link_ids_w, rots, link_piv_w, e],
                        outputs=[m.points], device=device,
                    )
                    m.refit()
                wp.synchronize()
                articulate_ms.append((time.perf_counter() - t0) * 1000.0)
                wp.launch(
                    trace_ml,
                    dim=(envs, args.height, args.width),
                    inputs=[mesh_ids_w, cam_origins, robot_rot, robot_pos, mc,
                            focal, fy, cx, cy, args.width, args.height, max_t],
                    outputs=[depth_w, shade_w],
                    device=device,
                )
            else:
                wp.launch(
                    trace,
                    dim=(envs, args.height, args.width),
                    inputs=[mesh.id, cam_origins, robot_rot, robot_pos, mc,
                            focal, fy, cx, cy, args.width, args.height, max_t],
                    outputs=[depth_w, shade_w],
                    device=device,
                )

        for i in range(args.warmup):
            frame(i)
        wp.synchronize()
        samples = []
        for i in range(args.frames):
            t0 = time.perf_counter()
            frame(args.warmup + i)
            wp.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        hit_px = int((depth_t > 0).sum().item())
        mean_ms = sum(samples) / len(samples)
        art_ms = (
            sum(articulate_ms[-args.frames:]) / args.frames
            if args.per_link and articulate_ms else 0.0
        )
        points.append({
            "envs": envs,
            "mean_frame_ms": mean_ms,
            "ms_per_env": mean_ms / envs,
            "images_per_second": envs * 1000.0 / mean_ms,
            "hit_pixels_total": hit_px,
            "hit_pixels_per_env": hit_px / envs,
            "articulate_refit_ms": art_ms,
        })
        art_str = f" artic+refit={art_ms:.2f}" if args.per_link else ""
        print(f"WARP_TRACE envs={envs} mean_ms={mean_ms:.2f} ms/env={mean_ms/envs:.3f} "
              f"hit_px/env={hit_px/envs:.0f}{art_str}", flush=True)
        if hit_px == 0:
            raise AssertionError("tracer hit zero pixels — camera framing bug")

    xs = [p["envs"] for p in points]
    ys = [p["mean_frame_ms"] for p in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    b = my - a * mx
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))

    results = {
        "schema_version": "warp-mesh-tracer-gate/v2",
        "triangles": int(len(triangles)),
        "vertices": int(len(vertices)),
        "bvh_build_ms_once": bvh_build_ms,
        "per_frame_geometry_rebuild": bool(args.per_link),
        "per_link_articulation": bool(args.per_link),
        "links": n_links,
        "resolution": [args.width, args.height],
        "points": points,
        "fit": {
            "ms_per_env": a,
            "intercept_ms": b,
            "r_squared": (1.0 - ssr / sst) if sst else None,
        },
        "gate_target_ms_per_env": 4.0,
        "gate_pass": a <= 4.0,
        "isaac_rtx_reference_ms_per_env": 36.6,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("WARP_MESH_TRACER_OK " + json.dumps(
        {"ms_per_env": round(a, 3), "intercept_ms": round(b, 2),
         "gate_pass": results["gate_pass"], "triangles": results["triangles"]}))


if __name__ == "__main__":
    main()
