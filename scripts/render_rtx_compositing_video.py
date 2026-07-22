#!/usr/bin/env python3
"""Render demo videos for the RTX Gaussian compositing path.

Three clips, all real renders on the Home Scan (no post-processing beyond a
display gamma):

  flythrough   interior camera path through the scan rendered by the OptiX
               RT-core tracer via the zero-copy torch API, presented as an
               RGB | depth | semantic triptych — the full output contract on
               screen.
  chrome       the tier-1 hybrid: raster gaussian layer + Warp-traced chrome
               G1 with live reflection rays through the gaussian field
               (sparse zero-copy pass) + CUDA depth compositor.
  perlink      per-link articulated G1 (per-env mesh + BVH refit) shaded by
               the Warp tracer — every link independently posed per frame.

Frames are written as PNGs and encoded with ffmpeg (H.264, yuv420p).

Run:
  /isaac-sim/python.sh scripts/render_rtx_compositing_video.py \
      --ply /workspace/datasets/home-scan-lod0.ply \
      --dump /workspace/datasets/gaussians.bin \
      --mesh-npz /workspace/outputs/g1_mesh.npz \
      --out-dir /workspace/outputs/videos
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src"),
              str(REPO_ROOT / "experiments" / "optix_tracer")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

GAMMA = 1.0 / 2.2
SEMANTIC_PALETTE = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--mesh-npz", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--frames", type=int, default=192)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--clips", default="flythrough,chrome,perlink")
    p.add_argument("--parity-frames", type=int, default=192)
    return p.parse_args()


def to8(img):
    import numpy as np

    return (np.clip(img, 0.0, 1.0) ** GAMMA * 255.0 + 0.5).astype("uint8")


def save_png(path: Path, arr) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


def encode(frame_dir: Path, out_mp4: Path, fps: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(frame_dir / "f_%04d.png"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "27", str(out_mp4)],
        check=True,
    )
    print(f"VIDEO_OK {out_mp4} bytes={out_mp4.stat().st_size}", flush=True)


# The Home Scan's canonical world is y-DOWN (gravity along +y); the ground
# plane is x-z. Every camera path here uses up = -y.
UP_W = (0.0, -1.0, 0.0)


def look_viewmat(pos, look):
    import numpy as np

    fwd = np.asarray(look, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    up_w = np.array(UP_W)
    right = np.cross(fwd, up_w); right /= np.linalg.norm(right)
    down = np.cross(fwd, right); down /= np.linalg.norm(down)
    R = np.stack([right, down, fwd])
    vm = np.eye(4, dtype=np.float64)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ np.asarray(pos, dtype=np.float64)
    return vm


def make_K(n, width, height):
    import numpy as np
    import torch

    K = np.zeros((n, 3, 3), dtype=np.float64)
    K[:, 0, 0] = 0.85 * width
    K[:, 1, 1] = 0.85 * width
    K[:, 0, 2] = width * 0.5
    K[:, 1, 2] = height * 0.5
    K[:, 2, 2] = 1.0
    return torch.as_tensor(K, dtype=torch.float64)


def probe_scene(tracer, center, radius):
    """Find a well-covered interior viewpoint and the floor height, using the
    tracer itself: coarse x-z orbit for coverage, one downward ray for the
    floor."""
    import numpy as np
    import torch

    n = 48
    pw, ph = 96, 72
    cams = []
    for i in range(n):
        th = 2.0 * math.pi * i / n
        pos = center + np.array(
            [0.15 * radius * math.cos(th), 0.0, 0.15 * radius * math.sin(th)]
        )
        look = center + np.array(
            [0.04 * radius * math.cos(th + 2.4), 0.0,
             0.04 * radius * math.sin(th + 2.4)]
        )
        cams.append((pos, look))
    vm = torch.as_tensor(
        np.stack([look_viewmat(p, l) for p, l in cams]), dtype=torch.float64
    )
    out = tracer.render_dense(vm, make_K(n, pw, ph), width=pw, height=ph)
    alpha = out["rgba"][..., 3]
    cover = (alpha > 0.3).float().mean(dim=(1, 2)).cpu().numpy()
    depth = out["depth"].cpu().numpy()
    best = int(np.argmax(cover))
    pos, look = cams[best]
    d = depth[best]
    finite = np.isfinite(d) & (alpha[best].cpu().numpy() > 0.3)
    med_depth = float(np.median(d[finite])) if finite.any() else 3.0
    fwd = (np.asarray(look) - np.asarray(pos))
    fwd /= np.linalg.norm(fwd)
    anchor = np.asarray(pos) + fwd * float(np.clip(0.5 * med_depth, 1.5, 3.5))

    down_o = torch.tensor([[anchor[0], anchor[1], anchor[2]]],
                          dtype=torch.float32, device="cuda")
    down_d = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device="cuda")
    res = tracer.render_sparse(down_o, down_d)
    d_floor = float(res["depth"][0].item())
    if not math.isfinite(d_floor) or d_floor > 5.0:
        d_floor = 1.4
    floor_y = anchor[1] + d_floor
    print(
        f"probe: best_cam={best} coverage={cover[best]:.2f} "
        f"median_depth={med_depth:.2f} anchor=({anchor[0]:.2f},{anchor[1]:.2f},"
        f"{anchor[2]:.2f}) floor_y={floor_y:.2f}",
        flush=True,
    )
    return anchor, floor_y, cover


def interior_path_cams(center, radius, eye_y, n, width, height):
    """Slow interior orbit in the ground (x-z) plane, level horizon."""
    import numpy as np
    import torch

    viewmats = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        th = 2.0 * math.pi * i / n
        pos = np.array(
            [center[0] + 0.15 * radius * math.cos(th),
             eye_y + 0.01 * radius * math.sin(2.0 * th),
             center[2] + 0.15 * radius * math.sin(th)]
        )
        look = np.array(
            [center[0] + 0.04 * radius * math.cos(th + 2.4),
             eye_y,
             center[2] + 0.04 * radius * math.sin(th + 2.4)]
        )
        viewmats[i] = look_viewmat(pos, look)
    return (
        torch.as_tensor(viewmats, dtype=torch.float64),
        make_K(n, width, height),
    )


def clip_flythrough(args, tracer, center, radius, eye_y) -> None:
    import numpy as np
    import torch

    fdir = args.out_dir / "frames_flythrough"
    fdir.mkdir(parents=True, exist_ok=True)
    vm, K = interior_path_cams(
        center, radius, eye_y, args.frames, args.width, args.height
    )
    batch = 24
    fi = 0
    for s in range(0, args.frames, batch):
        e = min(s + batch, args.frames)
        out = tracer.render_dense(vm[s:e], K[s:e], width=args.width, height=args.height)
        rgba = out["rgba"].detach().cpu().numpy()
        depth = out["depth"].detach().cpu().numpy()
        sem = out["semantic"].detach().cpu().numpy()
        for b in range(e - s):
            rgb8 = to8(rgba[b, ..., :3])
            d = depth[b]
            finite = np.isfinite(d)
            if finite.any():
                lo, hi = np.percentile(d[finite], [2, 98])
                dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            else:
                dn = np.zeros_like(d)
            dn[~finite] = 1.0
            # Inverse-depth grayscale with a warm tint (near = bright).
            inv = 1.0 - dn
            depth8 = np.stack(
                [inv, inv * 0.85, inv * 0.65], axis=-1
            )
            depth8 = (np.clip(depth8, 0, 1) * 255.0 + 0.5).astype("uint8")
            pal = np.array(SEMANTIC_PALETTE, dtype="uint8")
            sem_img = np.zeros((args.height, args.width, 3), dtype="uint8")
            labeled = sem[b] >= 0
            sem_img[labeled] = pal[sem[b][labeled] % len(pal)]
            # Modulate the semantic flats with luminance so structure reads.
            lum = (rgba[b, ..., :3].mean(axis=-1, keepdims=True) ** GAMMA)
            sem_img = (sem_img * (0.25 + 0.75 * np.clip(lum, 0, 1))).astype("uint8")
            frame = np.concatenate([rgb8, depth8, sem_img], axis=1)
            save_png(fdir / f"f_{fi:04d}.png", frame)
            fi += 1
    encode(fdir, args.out_dir / "tracer_flythrough_rgb_depth_semantic.mp4", args.fps)


def clip_parity(args, tracer, canonical, center, radius, eye_y) -> None:
    """Raster vs RT-core tracer side-by-side on IDENTICAL cameras — the
    ray-matched comparison as footage. Both render linear RGB; same gamma."""
    import numpy as np
    import torch

    from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService

    fdir = args.out_dir / "frames_parity"
    fdir.mkdir(parents=True, exist_ok=True)
    w, h = args.width, args.height
    n = args.parity_frames
    vm, K = interior_path_cams(center, radius, eye_y, n, w, h)

    backend = CustomCudaBackend(
        max_visible_records=500_000 * 8,
        max_intersections=math.ceil(8 * 2_400_000 * (w * h) / (128 * 128)),
        near_plane=0.01, far_plane=6.0 * radius,
        gaussian_support_sigma=3.0, covariance_epsilon=0.0,
        semantic_min_alpha=0.01, tile_size=1, depth_bucket_count=128,
        depth_bucket_group_size=8, compact_projection_cache=True,
        enable_projection_cache=False, output_srgb=False, deterministic=False,
    )
    service = RendererService(backend, height=h, width=w, max_views=8)
    service.initialize(stage=None, device="cuda")
    service.load_scene(
        98, means=canonical.means, scales=canonical.scales,
        rotations=canonical.rotations, opacities=canonical.opacities,
        features=canonical.features[:, :3].contiguous(),
        semantic_ids=torch.zeros(canonical.count, device="cuda", dtype=torch.int64),
    )

    fi = 0
    for s in range(0, n, 8):
        e = min(s + 8, n)
        batch = e - s
        scene_ids = torch.full((batch,), 98, device="cuda", dtype=torch.int64)
        raster = service.render(
            vm[s:e].to("cuda", torch.float32),
            K[s:e].to("cuda", torch.float32), scene_ids,
        )
        rrgb = raster["rgb"].clamp(0, 1).detach().cpu().numpy()
        tout = tracer.render_dense(vm[s:e], K[s:e], width=w, height=h)
        trgb = tout["rgba"][..., :3].clamp(0, 1).detach().cpu().numpy()
        for b in range(batch):
            frame = np.concatenate([to8(rrgb[b]), to8(trgb[b])], axis=1)
            save_png(fdir / f"f_{fi:04d}.png", frame)
            fi += 1
    service.shutdown()
    encode(fdir, args.out_dir / "raster_vs_tracer_same_cameras.mp4", args.fps)


def clip_chrome(args, tracer, canonical, center, radius, anchor, floor_y) -> None:
    import numpy as np
    import torch
    import warp as wp

    from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
    from isaacsim_gaussian_renderer.hybrid_compositor import (
        composite_gaussian_and_mesh,
    )

    fdir = args.out_dir / "frames_chrome"
    fdir.mkdir(parents=True, exist_ok=True)
    w, h = args.width, args.height

    data = np.load(args.mesh_npz, allow_pickle=False)
    verts = data["vertices"].astype(np.float32)
    tris = data["triangles"].astype(np.int32)
    mesh_center = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
    mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device="cuda:0"),
        indices=wp.array(tris.reshape(-1), dtype=wp.int32, device="cuda:0"),
    )

    @wp.kernel
    def trace_refl(
        mesh_id: wp.uint64,
        cam_o: wp.vec3,
        cam_rot: wp.mat33,          # camera->world
        robot_rot: wp.mat33,        # world->robot
        robot_pos: wp.vec3,
        mesh_c: wp.vec3,
        fx: float, fy: float, cx: float, cy: float,
        max_t: float, f0: float,
        mesh_rgb: wp.array3d(dtype=wp.float32),
        mesh_alpha: wp.array2d(dtype=wp.float32),
        mesh_depth: wp.array2d(dtype=wp.float32),
        refl_org: wp.array3d(dtype=wp.float32),
        refl_dir: wp.array3d(dtype=wp.float32),
        fresnel: wp.array2d(dtype=wp.float32),
    ):
        y, x = wp.tid()
        d_cam = wp.normalize(
            wp.vec3((float(x) + 0.5 - cx) / fx, (float(y) + 0.5 - cy) / fy, 1.0)
        )
        d_world = wp.normalize(cam_rot @ d_cam)
        o_local = robot_rot @ (cam_o - robot_pos) + mesh_c
        d_local = wp.normalize(robot_rot @ d_world)
        q = wp.mesh_query_ray(mesh_id, o_local, d_local, max_t)
        if q.result:
            n = wp.mesh_eval_face_normal(mesh_id, q.face)
            s = 0.15 + 0.85 * wp.abs(wp.dot(n, d_local))
            mesh_rgb[y, x, 0] = 0.58 * s
            mesh_rgb[y, x, 1] = 0.60 * s
            mesh_rgb[y, x, 2] = 0.63 * s
            mesh_alpha[y, x] = 1.0
            mesh_depth[y, x] = q.t * d_cam[2]
            n_w = wp.normalize(wp.transpose(robot_rot) @ n)
            if wp.dot(n_w, d_world) > 0.0:
                n_w = -n_w
            cos_i = wp.min(wp.max(-wp.dot(d_world, n_w), 0.0), 1.0)
            r = wp.normalize(d_world - 2.0 * wp.dot(d_world, n_w) * n_w)
            hit = cam_o + d_world * q.t + n_w * 1.0e-3
            refl_org[y, x, 0] = hit[0]
            refl_org[y, x, 1] = hit[1]
            refl_org[y, x, 2] = hit[2]
            refl_dir[y, x, 0] = r[0]
            refl_dir[y, x, 1] = r[1]
            refl_dir[y, x, 2] = r[2]
            fresnel[y, x] = f0 + (1.0 - f0) * wp.pow(1.0 - cos_i, 5.0)
        else:
            mesh_rgb[y, x, 0] = 0.0
            mesh_rgb[y, x, 1] = 0.0
            mesh_rgb[y, x, 2] = 0.0
            mesh_alpha[y, x] = 0.0
            mesh_depth[y, x] = 0.0
            fresnel[y, x] = 0.0

    backend = CustomCudaBackend(
        max_visible_records=500_000,
        max_intersections=math.ceil(2_400_000 * (w * h) / (128 * 128)),
        near_plane=0.01, far_plane=6.0 * radius,
        gaussian_support_sigma=3.0, covariance_epsilon=0.0,
        semantic_min_alpha=0.01, tile_size=1, depth_bucket_count=128,
        depth_bucket_group_size=8, compact_projection_cache=True,
        enable_projection_cache=False, output_srgb=False, deterministic=False,
    )
    service = RendererService(backend, height=h, width=w, max_views=1)
    service.initialize(stage=None, device="cuda")
    service.load_scene(
        99, means=canonical.means, scales=canonical.scales,
        rotations=canonical.rotations, opacities=canonical.opacities,
        features=canonical.features[:, :3].contiguous(),
        semantic_ids=torch.zeros(canonical.count, device="cuda", dtype=torch.int64),
    )
    scene_ids = torch.full((1,), 99, device="cuda", dtype=torch.int64)

    # Robot standing on the probed floor at the probed anchor; camera orbits
    # it at eye height. y-down world: feet (robot z = -0.79) sit at
    # base_y + 0.79, so base_y = floor_y - 0.79.
    robot_base = np.array([anchor[0], floor_y - 0.79, anchor[2]], dtype=np.float64)
    fx = fy = 0.85 * w
    cx, cy = w * 0.5, h * 0.5
    K = torch.zeros((1, 3, 3), dtype=torch.float64)
    K[0, 0, 0], K[0, 1, 1] = fx, fy
    K[0, 0, 2], K[0, 1, 2] = cx, cy
    K[0, 2, 2] = 1.0

    mesh_rgb = torch.zeros((h, w, 3), device="cuda")
    mesh_alpha = torch.zeros((h, w), device="cuda")
    mesh_depth = torch.zeros((h, w), device="cuda")
    refl_org = torch.zeros((h, w, 3), device="cuda")
    refl_dir = torch.zeros((h, w, 3), device="cuda")
    fresnel = torch.zeros((h, w), device="cuda")
    wraps = [wp.from_torch(t, dtype=wp.float32) for t in
             (mesh_rgb, mesh_alpha, mesh_depth, refl_org, refl_dir, fresnel)]

    B = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    for i in range(args.frames):
        th = 2.0 * math.pi * i / args.frames
        # Orbit in the ground (x-z) plane at chest height (y-down world).
        cam_pos = robot_base + np.array(
            [2.0 * math.cos(th), -0.45, 2.0 * math.sin(th)]
        )
        look = robot_base + np.array([0.0, -0.35, 0.0])
        vm_np = look_viewmat(cam_pos, look)
        Rwc = vm_np[:3, :3]
        vm = torch.as_tensor(vm_np, dtype=torch.float64)[None]

        gaussian = service.render(
            vm.to("cuda", torch.float32), K.to("cuda", torch.float32), scene_ids
        )

        # y-down world: B stands the z-up G1 upright (robot +z = world -y);
        # heading is a rotation about the world up axis (y).
        yaw = 0.6 * math.sin(2.0 * math.pi * i / args.frames * 2.0) + th + math.pi
        Ry = np.array(
            [[math.cos(yaw), 0.0, math.sin(yaw)],
             [0.0, 1.0, 0.0],
             [-math.sin(yaw), 0.0, math.cos(yaw)]]
        )
        Rrob = (B @ Ry).astype(np.float32)  # world->robot
        wp.launch(
            trace_refl, dim=(h, w),
            inputs=[mesh.id,
                    wp.vec3(*[float(v) for v in cam_pos]),
                    wp.mat33(*Rwc.T.astype(np.float32).reshape(-1)),
                    wp.mat33(*Rrob.reshape(-1)),
                    wp.vec3(*[float(v) for v in robot_base]),
                    wp.vec3(*[float(v) for v in mesh_center]),
                    float(fx), float(fy), float(cx), float(cy),
                    float(6.0 * radius), 0.9],
            outputs=wraps, device="cuda:0",
        )
        wp.synchronize()
        mask = mesh_alpha > 0.0
        n_hit = int(mask.sum().item())
        if n_hit > 0:
            res = tracer.render_sparse(
                refl_org[mask].contiguous(), refl_dir[mask].contiguous()
            )
            fr = fresnel[mask][:, None]
            mesh_rgb[mask] = fr * res["rgba"][:, :3] + (1.0 - fr) * mesh_rgb[mask]
        out = composite_gaussian_and_mesh(
            gaussian["rgb"], gaussian["alpha"], gaussian["depth"],
            mesh_rgb[None], mesh_alpha[None, ..., None],
            mesh_depth[None, ..., None],
        )
        frame = to8(out.rgb[0].detach().cpu().numpy())
        save_png(fdir / f"f_{i:04d}.png", frame)
        if i % 48 == 0:
            print(f"chrome frame {i}/{args.frames} refl_rays={n_hit}", flush=True)
    service.shutdown()
    encode(fdir, args.out_dir / "hybrid_chrome_reflections.mp4", args.fps)


def clip_perlink(args, center, radius) -> None:
    import numpy as np
    import torch
    import warp as wp

    fdir = args.out_dir / "frames_perlink"
    fdir.mkdir(parents=True, exist_ok=True)
    w, h = args.width, args.height

    data = np.load(args.mesh_npz, allow_pickle=False)
    verts = data["vertices"].astype(np.float32)
    tris = data["triangles"].astype(np.int32)
    offsets = data["link_tri_offsets"].astype(np.int64)
    n_links = len(offsets) - 1
    link_ids = np.zeros(len(verts), dtype=np.int32)
    pivots = np.zeros((n_links, 3), dtype=np.float32)
    for L in range(n_links):
        vidx = np.unique(tris[offsets[L]:offsets[L + 1]].reshape(-1))
        link_ids[vidx] = L
        pivots[L] = verts[vidx].mean(axis=0)
    mesh_center = (verts.min(axis=0) + verts.max(axis=0)) * 0.5

    mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device="cuda:0"),
        indices=wp.array(tris.reshape(-1), dtype=wp.int32, device="cuda:0"),
    )
    base_pts = wp.array(verts, dtype=wp.vec3, device="cuda:0")
    link_ids_w = wp.array(link_ids, dtype=wp.int32, device="cuda:0")
    pivots_w = wp.array(pivots, dtype=wp.vec3, device="cuda:0")

    @wp.kernel
    def articulate(
        base: wp.array(dtype=wp.vec3),
        lid: wp.array(dtype=wp.int32),
        rot: wp.array(dtype=wp.mat33),
        piv: wp.array(dtype=wp.vec3),
        out_pts: wp.array(dtype=wp.vec3),
    ):
        v = wp.tid()
        L = lid[v]
        p = base[v]
        out_pts[v] = rot[L] @ (p - piv[L]) + piv[L]

    @wp.kernel
    def shade(
        mesh_id: wp.uint64,
        cam_o: wp.vec3,
        robot_rot: wp.mat33,
        mesh_c: wp.vec3,
        fx: float, fy: float, cx: float, cy: float, max_t: float,
        img: wp.array3d(dtype=wp.float32),
    ):
        y, x = wp.tid()
        d_world = wp.normalize(
            wp.vec3((float(x) + 0.5 - cx) / fx, (float(y) + 0.5 - cy) / fy, 1.0)
        )
        o_local = robot_rot @ cam_o + mesh_c
        d_local = wp.normalize(robot_rot @ d_world)
        q = wp.mesh_query_ray(mesh_id, o_local, d_local, max_t)
        if q.result:
            n = wp.mesh_eval_face_normal(mesh_id, q.face)
            s = 0.2 + 0.8 * wp.abs(wp.dot(n, d_local))
            img[y, x, 0] = 0.75 * s
            img[y, x, 1] = 0.78 * s
            img[y, x, 2] = 0.85 * s
        else:
            v = 0.12 + 0.05 * float(y) / float(cy * 2.0)
            img[y, x, 0] = v
            img[y, x, 1] = v
            img[y, x, 2] = v + 0.02

    rng = np.random.default_rng(11)
    axes = rng.normal(size=(n_links, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    img_t = torch.zeros((h, w, 3), device="cuda")
    img_w = wp.from_torch(img_t, dtype=wp.float32)
    fx = fy = 0.9 * w
    cx, cy = w * 0.5, h * 0.5
    B = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

    for i in range(args.frames):
        phase = i * 0.05 + np.arange(n_links) / n_links
        ang = 0.28 * np.sin(2.0 * math.pi * phase)
        K = np.zeros((n_links, 3, 3))
        k = axes
        K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
        K[:, 1, 0] = k[:, 2]; K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
        s = np.sin(ang)[:, None, None]
        c = np.cos(ang)[:, None, None]
        R = np.eye(3)[None] + s * K + (1.0 - c) * (K @ K)
        rots = wp.array(R.astype(np.float32), dtype=wp.mat33, device="cuda:0")
        wp.launch(articulate, dim=len(verts),
                  inputs=[base_pts, link_ids_w, rots, pivots_w],
                  outputs=[mesh.points], device="cuda:0")
        mesh.refit()
        th = 2.0 * math.pi * i / args.frames
        yaw = th + math.pi
        Rz = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0.0],
             [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
        )
        Rrob = (Rz @ B).astype(np.float32)
        cam_o = np.array([0.0, 0.05, -1.6], dtype=np.float64)
        wp.launch(shade, dim=(h, w),
                  inputs=[mesh.id, wp.vec3(*[float(v) for v in cam_o]),
                          wp.mat33(*Rrob.reshape(-1)),
                          wp.vec3(*[float(v) for v in mesh_center]),
                          float(fx), float(fy), float(cx), float(cy), 30.0],
                  outputs=[img_w], device="cuda:0")
        wp.synchronize()
        save_png(fdir / f"f_{i:04d}.png", to8(img_t.detach().cpu().numpy()))
    encode(fdir, args.out_dir / "perlink_articulation.mp4", args.fps)


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch

    from gs_tracer_torch import GsTracer
    from isaacsim_gaussian_renderer.ply_loader import (
        canonicalize_3dgs_scene,
        load_ply_to_gaussians,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips = {c.strip() for c in args.clips.split(",")}

    tracer = GsTracer(args.dump)
    center = np.array(tracer.center)
    radius = tracer.radius
    print(f"tracer up: {tracer.count} gaussians", flush=True)

    anchor, floor_y, _cover = probe_scene(tracer, center, radius)
    eye_y = floor_y - 1.5  # 1.5 m above the floor in the y-down world

    if "flythrough" in clips:
        clip_flythrough(args, tracer, center, radius, eye_y)
    canonical = None
    if "chrome" in clips or "parity" in clips:
        raw = load_ply_to_gaussians(args.ply)
        canonical = canonicalize_3dgs_scene(raw, device="cuda")
        del raw
    if "parity" in clips:
        clip_parity(args, tracer, canonical, center, radius, eye_y)
    if "chrome" in clips:
        clip_chrome(args, tracer, canonical, center, radius, anchor, floor_y)
    if "perlink" in clips:
        clip_perlink(args, center, radius)
    tracer.close()
    print("RENDER_VIDEOS_OK", flush=True)


if __name__ == "__main__":
    main()
