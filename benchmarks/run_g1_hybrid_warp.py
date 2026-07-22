#!/usr/bin/env python3
"""Tier-1 custom hybrid: batched Gaussian layer + vectorized Warp mesh tracer
+ CUDA depth compositor. No Isaac RTX, no OVRTX, no SimulationApp.

Replaces the Isaac-RTX robot layer (measured 36.6 ms/env, wrapper-bound) with a
single Warp launch tracing every env's robot (per-env rigid pose applied in ray
space; per-frame pose animation; zero geometry/BVH rebuild). Mesh depth is
written as z-depth so the existing hybrid compositor composites it directly
against the Gaussian layer's depth.

The scaling target: match or beat the unified-OVRTX lane (~221–256 images/s at
256² with robots) while keeping the custom path's full output contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
HOME_SCAN_COUNT = 21_497_908
SCENE_ID = 404


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--home-scan-path", type=Path, default=Path("/workspace/datasets/home-scan-lod0.ply"))
    p.add_argument("--mesh-npz", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--envs-list", default="1,2,4,8,16,32")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--synthetic-tris", type=int, default=0,
                   help="Use a dense robot-sized ellipsoid instead of the G1 npz.")
    p.add_argument(
        "--reflections", action="store_true",
        help="Chrome robot: trace reflection rays at robot-hit pixels through "
        "the OptiX gaussian tracer (zero-copy, in-process) and Fresnel-blend "
        "them into the robot shading before compositing.",
    )
    p.add_argument("--gaussians-dump", type=Path,
                   default=Path("/workspace/datasets/gaussians.bin"))
    p.add_argument("--fresnel-f0", type=float, default=0.9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    import warp as wp

    from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
    from isaacsim_gaussian_renderer.benchmark_manifest import (
        axis_aligned_exterior_camera_bundle,
        file_sha256,
    )
    from isaacsim_gaussian_renderer.hybrid_compositor import composite_gaussian_and_mesh
    from isaacsim_gaussian_renderer.ply_loader import (
        canonicalize_3dgs_scene,
        load_ply_to_gaussians,
    )

    wp.init()
    device = "cuda:0"

    # ---- robot mesh (real G1 npz or dense synthetic stress body) ------------
    if args.synthetic_tris > 0:
        rows = max(8, int(math.sqrt(args.synthetic_tris / 2.0)))
        cols = 2 * rows
        lat = np.linspace(0.0, math.pi, rows + 1)
        lon = np.linspace(0.0, 2.0 * math.pi, cols, endpoint=False)
        gl, gn = np.meshgrid(lat, lon, indexing="ij")
        verts = np.stack(
            [0.10 * np.sin(gl) * np.cos(gn), 0.10 * np.sin(gl) * np.sin(gn), 0.35 * np.cos(gl)],
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
        tris = np.asarray(faces, dtype=np.int32)
        mesh_source = f"synthetic-ellipsoid-{len(tris)}tris"
    else:
        data = np.load(args.mesh_npz, allow_pickle=False)
        verts = data["vertices"].astype(np.float32)
        tris = data["triangles"].astype(np.int32)
        mesh_source = str(args.mesh_npz)
    mesh_center = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
    mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(tris.reshape(-1), dtype=wp.int32, device=device),
    )

    @wp.kernel
    def trace(
        mesh_id: wp.uint64,
        cam_origins: wp.array(dtype=wp.vec3),
        robot_rot: wp.array(dtype=wp.mat33),
        robot_pos: wp.array(dtype=wp.vec3),
        mesh_c: wp.vec3,
        fx: float, fy: float, cx: float, cy: float,
        max_t: float,
        mesh_rgb: wp.array4d(dtype=wp.float32),
        mesh_alpha: wp.array4d(dtype=wp.float32),
        mesh_depth: wp.array4d(dtype=wp.float32),
    ):
        e, y, x = wp.tid()
        d_world = wp.normalize(
            wp.vec3((float(x) + 0.5 - cx) / fx, (float(y) + 0.5 - cy) / fy, 1.0)
        )
        rot = robot_rot[e]
        o_local = rot @ (cam_origins[e] - robot_pos[e]) + mesh_c
        d_local = wp.normalize(rot @ d_world)
        q = wp.mesh_query_ray(mesh_id, o_local, d_local, max_t)
        if q.result:
            n = wp.mesh_eval_face_normal(mesh_id, q.face)
            s = 0.15 + 0.85 * wp.abs(wp.dot(n, d_local))
            mesh_rgb[e, y, x, 0] = 0.58 * s
            mesh_rgb[e, y, x, 1] = 0.60 * s
            mesh_rgb[e, y, x, 2] = 0.63 * s
            mesh_alpha[e, y, x, 0] = 1.0
            # z-depth (distance to image plane) for the depth compositor.
            mesh_depth[e, y, x, 0] = q.t * d_world[2]
        else:
            mesh_rgb[e, y, x, 0] = 0.0
            mesh_rgb[e, y, x, 1] = 0.0
            mesh_rgb[e, y, x, 2] = 0.0
            mesh_alpha[e, y, x, 0] = 0.0
            mesh_depth[e, y, x, 0] = 0.0

    @wp.kernel
    def trace_refl(
        mesh_id: wp.uint64,
        cam_origins: wp.array(dtype=wp.vec3),
        robot_rot: wp.array(dtype=wp.mat33),
        robot_pos: wp.array(dtype=wp.vec3),
        mesh_c: wp.vec3,
        fx: float, fy: float, cx: float, cy: float,
        max_t: float,
        f0: float,
        mesh_rgb: wp.array4d(dtype=wp.float32),
        mesh_alpha: wp.array4d(dtype=wp.float32),
        mesh_depth: wp.array4d(dtype=wp.float32),
        refl_org: wp.array4d(dtype=wp.float32),
        refl_dir: wp.array4d(dtype=wp.float32),
        fresnel: wp.array4d(dtype=wp.float32),
    ):
        # Chrome variant of `trace`: same hit logic, plus world-space
        # reflection ray + Fresnel weight per robot-hit pixel for the sparse
        # OptiX gaussian pass.
        e, y, x = wp.tid()
        d_world = wp.normalize(
            wp.vec3((float(x) + 0.5 - cx) / fx, (float(y) + 0.5 - cy) / fy, 1.0)
        )
        rot = robot_rot[e]
        o_local = rot @ (cam_origins[e] - robot_pos[e]) + mesh_c
        d_local = wp.normalize(rot @ d_world)
        q = wp.mesh_query_ray(mesh_id, o_local, d_local, max_t)
        if q.result:
            n = wp.mesh_eval_face_normal(mesh_id, q.face)
            s = 0.15 + 0.85 * wp.abs(wp.dot(n, d_local))
            mesh_rgb[e, y, x, 0] = 0.58 * s
            mesh_rgb[e, y, x, 1] = 0.60 * s
            mesh_rgb[e, y, x, 2] = 0.63 * s
            mesh_alpha[e, y, x, 0] = 1.0
            mesh_depth[e, y, x, 0] = q.t * d_world[2]
            n_w = wp.normalize(wp.transpose(rot) @ n)
            if wp.dot(n_w, d_world) > 0.0:
                n_w = -n_w
            cos_i = wp.min(wp.max(-wp.dot(d_world, n_w), 0.0), 1.0)
            r = wp.normalize(d_world - 2.0 * wp.dot(d_world, n_w) * n_w)
            hit_w = cam_origins[e] + d_world * q.t + n_w * 1.0e-3
            refl_org[e, y, x, 0] = hit_w[0]
            refl_org[e, y, x, 1] = hit_w[1]
            refl_org[e, y, x, 2] = hit_w[2]
            refl_dir[e, y, x, 0] = r[0]
            refl_dir[e, y, x, 1] = r[1]
            refl_dir[e, y, x, 2] = r[2]
            fresnel[e, y, x, 0] = f0 + (1.0 - f0) * wp.pow(1.0 - cos_i, 5.0)
        else:
            mesh_rgb[e, y, x, 0] = 0.0
            mesh_rgb[e, y, x, 1] = 0.0
            mesh_rgb[e, y, x, 2] = 0.0
            mesh_alpha[e, y, x, 0] = 0.0
            mesh_depth[e, y, x, 0] = 0.0
            fresnel[e, y, x, 0] = 0.0

    # ---- Gaussian scene -----------------------------------------------------
    checksum = file_sha256(args.home_scan_path)
    if checksum != HOME_SCAN_SHA256:
        raise ValueError(f"Home Scan SHA-256 mismatch: {checksum}")
    raw = load_ply_to_gaussians(args.home_scan_path)
    if raw.count != HOME_SCAN_COUNT:
        raise ValueError(f"count mismatch: {raw.count}")
    canonical = canonicalize_3dgs_scene(raw, device="cuda")
    del raw
    pos_min = canonical.means.amin(dim=0)
    pos_max = canonical.means.amax(dim=0)
    scene_center = (pos_min + pos_max) * 0.5
    bounding_radius = float((torch.linalg.vector_norm(pos_max - pos_min) * 0.5).item())

    base_rot = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32
    )

    env_counts = [int(v) for v in args.envs_list.split(",") if v.strip()]
    max_envs = max(env_counts)
    pool = axis_aligned_exterior_camera_bundle(
        center=scene_center, bounding_radius=bounding_radius, batch_size=max_envs,
        width=args.width, height=args.height,
    )
    pool_viewmats = pool.viewmats.to("cuda", torch.float32).contiguous()
    pool_intrinsics = pool.intrinsics.to("cuda", torch.float32).contiguous()
    distance = float(pool.manifest["distance"])
    near_plane, far_plane = 0.01, distance + 2.0 * bounding_radius + 1.0
    fx = float(pool_intrinsics[0, 0, 0].item())
    fyv = float(pool_intrinsics[0, 1, 1].item())
    cxv = float(pool_intrinsics[0, 0, 2].item())
    cyv = float(pool_intrinsics[0, 1, 2].item())
    max_t = far_plane

    tracer = None
    if args.reflections:
        tracer_dir = REPO_ROOT / "experiments" / "optix_tracer"
        sys.path.insert(0, str(tracer_dir))
        from gs_tracer_torch import GsTracer

        tracer = GsTracer(args.gaussians_dump)
        print(
            f"reflections: OptiX tracer up, {tracer.count} gaussians, "
            f"gas {tracer.gas_build_ms:.1f} ms",
            flush=True,
        )

    points = []
    service = None
    for envs in env_counts:
        backend = CustomCudaBackend(
            max_visible_records=500_000 * envs,
            max_intersections=math.ceil(envs * 2_400_000 * (args.width * args.height) / (128 * 128)),
            near_plane=near_plane, far_plane=far_plane,
            gaussian_support_sigma=3.0, covariance_epsilon=0.0, semantic_min_alpha=0.01,
            tile_size=1, depth_bucket_count=128, depth_bucket_group_size=8,
            compact_projection_cache=True, enable_projection_cache=False,
            output_srgb=True, deterministic=False,
        )
        service = RendererService(backend, height=args.height, width=args.width, max_views=envs)
        service.initialize(stage=None, device="cuda")
        service.load_scene(
            SCENE_ID, means=canonical.means, scales=canonical.scales,
            rotations=canonical.rotations, opacities=canonical.opacities,
            features=canonical.features[:, :3].contiguous(),
            semantic_ids=torch.zeros(canonical.count, device="cuda", dtype=torch.int64),
        )
        scene_ids = torch.full((envs,), SCENE_ID, device="cuda", dtype=torch.int64)
        viewmats = pool_viewmats[:envs].contiguous()
        intrinsics = pool_intrinsics[:envs].contiguous()

        # Robot in front of each env camera; camera origins for the tracer.
        cam_centers = np.zeros((envs, 3), dtype=np.float32)
        robot_bases = np.zeros((envs, 3), dtype=np.float32)
        for i in range(envs):
            vm = pool_viewmats[i]
            c = (-vm[:3, :3].transpose(0, 1) @ vm[:3, 3]).detach().cpu().numpy()
            cam_centers[i] = c
            robot_bases[i] = c + vm[2, :3].cpu().numpy() * 2.5 + vm[1, :3].cpu().numpy() * 0.35
        cam_origins = wp.array(cam_centers, dtype=wp.vec3, device=device)
        robot_pos = wp.array(robot_bases, dtype=wp.vec3, device=device)
        robot_rot = wp.zeros(envs, dtype=wp.mat33, device=device)

        mesh_rgb = torch.zeros((envs, args.height, args.width, 3), device="cuda")
        mesh_alpha = torch.zeros((envs, args.height, args.width, 1), device="cuda")
        mesh_depth = torch.zeros((envs, args.height, args.width, 1), device="cuda")
        rgb_w = wp.from_torch(mesh_rgb, dtype=wp.float32)
        alpha_w = wp.from_torch(mesh_alpha, dtype=wp.float32)
        depth_w = wp.from_torch(mesh_depth, dtype=wp.float32)

        refl_org = refl_dir = fresnel = None
        if args.reflections:
            refl_org = torch.zeros((envs, args.height, args.width, 3), device="cuda")
            refl_dir = torch.zeros((envs, args.height, args.width, 3), device="cuda")
            fresnel = torch.zeros((envs, args.height, args.width, 1), device="cuda")
            refl_org_w = wp.from_torch(refl_org, dtype=wp.float32)
            refl_dir_w = wp.from_torch(refl_dir, dtype=wp.float32)
            fresnel_w = wp.from_torch(fresnel, dtype=wp.float32)

        comp = {"gaussian": [], "mesh": [], "compositor": []}
        if args.reflections:
            comp["reflection"] = []
        refl_rays = [0]

        def frame(i: int, measure: bool) -> None:
            angles = np.arange(envs) * (2.0 * math.pi / max(envs, 1)) + i * 0.05
            cos, sin = np.cos(angles), np.sin(angles)
            rz = np.zeros((envs, 3, 3), dtype=np.float32)
            rz[:, 0, 0] = cos; rz[:, 0, 1] = -sin
            rz[:, 1, 0] = sin; rz[:, 1, 1] = cos
            rz[:, 2, 2] = 1.0
            rots = np.einsum("eij,jk->eik", rz, base_rot)
            wp.copy(robot_rot, wp.array(np.ascontiguousarray(rots), dtype=wp.mat33, device=device))

            t0 = time.perf_counter()
            gaussian = service.render(viewmats, intrinsics, scene_ids)
            if measure:
                service.synchronize(); comp["gaussian"].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            mc = wp.vec3(float(mesh_center[0]), float(mesh_center[1]), float(mesh_center[2]))
            if args.reflections:
                wp.launch(
                    trace_refl, dim=(envs, args.height, args.width),
                    inputs=[mesh.id, cam_origins, robot_rot, robot_pos, mc,
                            fx, fyv, cxv, cyv, max_t, args.fresnel_f0],
                    outputs=[rgb_w, alpha_w, depth_w,
                             refl_org_w, refl_dir_w, fresnel_w],
                    device=device,
                )
            else:
                wp.launch(
                    trace, dim=(envs, args.height, args.width),
                    inputs=[mesh.id, cam_origins, robot_rot, robot_pos, mc,
                            fx, fyv, cxv, cyv, max_t],
                    outputs=[rgb_w, alpha_w, depth_w], device=device,
                )
            if measure:
                wp.synchronize(); comp["mesh"].append((time.perf_counter() - t0) * 1000.0)
            if args.reflections:
                # Sparse secondary rays at robot-hit pixels, traced through
                # the gaussian field zero-copy, Fresnel-blended into the
                # chrome shading before the depth composite.
                t0 = time.perf_counter()
                wp.synchronize()
                mask = mesh_alpha[..., 0] > 0.0
                nrays = int(mask.sum().item())
                refl_rays[0] = nrays
                if nrays > 0:
                    origins = refl_org[mask].contiguous()
                    dirs = refl_dir[mask].contiguous()
                    res = tracer.render_sparse(origins, dirs)
                    fr = fresnel[mask]
                    mesh_rgb[mask] = (
                        fr * res["rgba"][:, :3] + (1.0 - fr) * mesh_rgb[mask]
                    )
                if measure:
                    torch.cuda.synchronize()
                    comp["reflection"].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            composite_gaussian_and_mesh(
                gaussian["rgb"], gaussian["alpha"], gaussian["depth"],
                mesh_rgb, mesh_alpha, mesh_depth,
            )
            if measure:
                torch.cuda.synchronize(); comp["compositor"].append((time.perf_counter() - t0) * 1000.0)

        for i in range(args.warmup):
            frame(i, False)
        service.synchronize(); wp.synchronize(); torch.cuda.synchronize()
        wall0 = time.perf_counter()
        for i in range(args.frames):
            frame(args.warmup + i, False)
        service.synchronize(); wp.synchronize(); torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall0) * 1000.0 / args.frames
        for i in range(3):
            frame(args.warmup + args.frames + i, True)
        hit_px = int((mesh_alpha > 0).sum().item())
        comp_ms = {k: sum(v) / len(v) for k, v in comp.items()}
        points.append({
            "envs": envs,
            "wall_frame_ms": wall_ms,
            "fps": 1000.0 / wall_ms,
            "images_per_second": envs * 1000.0 / wall_ms,
            "component_ms": comp_ms,
            "mesh_hit_px_per_env": hit_px / envs,
            "reflection_rays_per_frame": refl_rays[0] if args.reflections else 0,
        })
        refl_str = (
            f" refl={comp_ms['reflection']:.2f} refl_rays={refl_rays[0]}"
            if args.reflections else ""
        )
        print(f"WARP_HYBRID envs={envs} frame_ms={wall_ms:.1f} img/s={envs*1000.0/wall_ms:.1f} "
              f"gauss={comp_ms['gaussian']:.1f} mesh={comp_ms['mesh']:.2f} "
              f"comp={comp_ms['compositor']:.2f} hit_px/env={hit_px/envs:.0f}"
              f"{refl_str}", flush=True)
        service.shutdown()
        service = None
        del backend
        torch.cuda.empty_cache()

    xs = [p["envs"] for p in points]
    ys = [p["wall_frame_ms"] for p in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    b = my - a * mx

    report = {
        "schema_version": "g1-hybrid-warp/v2",
        "reflections": bool(args.reflections),
        "mesh_source": mesh_source,
        "triangles": int(len(tris)),
        "resolution": [args.width, args.height],
        "points": points,
        "fit_frame_ms": {"ms_per_env": a, "intercept_ms": b},
        "ovrtx_unified_reference": {"ms_per_env": 4.45, "images_per_second_ceiling": 256},
        "isaac_rtx_reference_ms_per_env": 36.6,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best = max(points, key=lambda p: p["images_per_second"])
    print("WARP_HYBRID_OK " + json.dumps(
        {"ms_per_env_fit": round(a, 3), "best_images_per_second": round(best["images_per_second"], 1),
         "best_envs": best["envs"]}))


if __name__ == "__main__":
    main()
