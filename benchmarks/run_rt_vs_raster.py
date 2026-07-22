#!/usr/bin/env python3
"""Ray-matched head-to-head: OptiX RT-core tracer vs the custom tile raster.

Both renderers consume the SAME per-env cameras (same origins, rotations, and
intrinsics) on the same checksummed scene, in one process invocation each,
with identical warmup/frames/sync discipline and N repeats:

  - "exterior": the repo's standard `axis_aligned_exterior_camera_bundle`
    (the cameras every raster/OVRTX lane uses).
  - "interior": ego-view cameras inside the scene (near-1 hit fraction, the
    RL-observation shape and the k-buffer's worst case).

The tracer receives the cameras via `gs_tracer --cams <file>` (int64 envs,
float32 fx,fy,cx,cy, then per env float32 origin[3] + camera->world R[9]
row-major). Per camera set the harness also PSNRs the two env-0 images to
prove both systems drew the same picture before comparing their times.

Known asymmetry (stated in the output): the raster pays the full output
contract (rgb+depth+alpha+semantic) as the benchmark protocol requires; the
tracer writes RGBA only. Its full output contract is an explicit follow-up.

Run:
  /isaac-sim/python.sh benchmarks/run_rt_vs_raster.py \
      --ply /workspace/datasets/home-scan-lod0.ply \
      --dump /workspace/datasets/gaussians.bin \
      --tracer-bin experiments/optix_tracer/gs_tracer \
      --output outputs/rt_vs_raster.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
SCENE_ID = 11


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=Path, default=Path("/workspace/datasets/home-scan-lod0.ply"))
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--tracer-bin", type=Path, required=True)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--camera-sets", default="exterior,interior")
    p.add_argument(
        "--tracer-extra", default="",
        help="extra args appended to every gs_tracer invocation, e.g. "
        "'--tmin 0.05 --ptx gs_tracer_device_k16.ptx'",
    )
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def write_cams_bin(path: Path, viewmats, intrinsics) -> None:
    """viewmats: [B,4,4] OpenCV world->camera; intrinsics: [B,3,3]."""
    import numpy as np

    vm = np.asarray(viewmats, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    envs = vm.shape[0]
    with path.open("wb") as f:
        f.write(struct.pack("<q", envs))
        f.write(struct.pack("<4f", K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]))
        for e in range(envs):
            R = vm[e, :3, :3]
            t = vm[e, :3, 3]
            origin = -R.T @ t
            r_c2w = R.T  # camera->world, row-major
            f.write(struct.pack("<3f", *origin.astype(np.float32)))
            f.write(struct.pack("<9f", *r_c2w.astype(np.float32).reshape(-1)))


def read_ppm(path: Path):
    import numpy as np

    with path.open("rb") as f:
        assert f.readline().strip() == b"P6"
        w, h = (int(v) for v in f.readline().split())
        assert int(f.readline()) == 255
        return np.frombuffer(f.read(w * h * 3), dtype=np.uint8).reshape(h, w, 3)


def psnr8(a, b) -> float:
    import numpy as np

    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch

    from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
    from isaacsim_gaussian_renderer.benchmark_manifest import (
        axis_aligned_exterior_camera_bundle,
        file_sha256,
    )
    from isaacsim_gaussian_renderer.ply_loader import (
        canonicalize_3dgs_scene,
        load_ply_to_gaussians,
    )

    checksum = file_sha256(args.ply)
    if checksum != HOME_SCAN_SHA256:
        raise ValueError(f"Home Scan SHA-256 mismatch: {checksum}")
    raw = load_ply_to_gaussians(args.ply)
    canonical = canonicalize_3dgs_scene(raw, device="cuda")
    del raw
    pos_min = canonical.means.amin(dim=0)
    pos_max = canonical.means.amax(dim=0)
    center_t = (pos_min + pos_max) * 0.5
    center = center_t.cpu().numpy().astype(np.float64)
    radius = float((torch.linalg.vector_norm(pos_max - pos_min) * 0.5).item())

    bundle = axis_aligned_exterior_camera_bundle(
        center=center_t, bounding_radius=radius,
        batch_size=args.envs, width=args.width, height=args.height,
    )
    ext_viewmats = bundle.viewmats.to("cuda", torch.float32).contiguous()
    ext_intrinsics = bundle.intrinsics.to("cuda", torch.float32).contiguous()

    # Interior ego-view set: cameras inside the scene volume, per-env yaw
    # sweep, level horizon, same intrinsics as the exterior bundle.
    K0 = ext_intrinsics[0].detach().cpu().numpy().astype(np.float64)
    int_viewmats = np.zeros((args.envs, 4, 4), dtype=np.float64)
    for e in range(args.envs):
        yaw = 2.0 * math.pi * e / args.envs
        pos = center + np.array(
            [0.15 * radius * math.cos(yaw), 0.15 * radius * math.sin(yaw), 0.0]
        )
        fwd = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])  # look inward
        up_w = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_w); right /= np.linalg.norm(right)
        down = np.cross(fwd, right); down /= np.linalg.norm(down)
        R = np.stack([right, down, fwd])  # world->camera rows (OpenCV x,y,z)
        int_viewmats[e, :3, :3] = R
        int_viewmats[e, :3, 3] = -R @ pos
        int_viewmats[e, 3, 3] = 1.0
    int_viewmats_t = torch.as_tensor(int_viewmats, device="cuda", dtype=torch.float32).contiguous()
    int_intrinsics = ext_intrinsics.clone()

    sets = {
        "exterior": (ext_viewmats, ext_intrinsics),
        "interior": (int_viewmats_t, int_intrinsics),
    }

    near_plane, far_plane = 0.01, 6.0 * radius
    backend = CustomCudaBackend(
        max_visible_records=500_000 * args.envs,
        max_intersections=math.ceil(
            args.envs * 2_400_000 * (args.width * args.height) / (128 * 128)
        ),
        near_plane=near_plane, far_plane=far_plane,
        gaussian_support_sigma=3.0, covariance_epsilon=0.0,
        semantic_min_alpha=0.01, tile_size=1, depth_bucket_count=128,
        depth_bucket_group_size=8, compact_projection_cache=True,
        enable_projection_cache=False, output_srgb=False, deterministic=False,
    )
    service = RendererService(backend, height=args.height, width=args.width, max_views=args.envs)
    service.initialize(stage=None, device="cuda")
    service.load_scene(
        SCENE_ID, means=canonical.means, scales=canonical.scales,
        rotations=canonical.rotations, opacities=canonical.opacities,
        features=canonical.features[:, :3].contiguous(),
        semantic_ids=torch.zeros(canonical.count, device="cuda", dtype=torch.int64),
    )
    scene_ids = torch.full((args.envs,), SCENE_ID, device="cuda", dtype=torch.int64)

    report = {
        "schema_version": "rt-vs-raster/v1",
        "envs": args.envs, "width": args.width, "height": args.height,
        "frames": args.frames, "warmup": args.warmup, "repeats": args.repeats,
        "home_scan_sha256": checksum,
        "gaussians": int(canonical.count),
        "output_contract_note": (
            "raster renders rgb+depth+alpha+semantic (protocol contract); "
            "tracer writes RGBA only — tracer full contract is a follow-up"
        ),
        "camera_sets": {},
    }

    tmpdir = Path(tempfile.mkdtemp(prefix="rt-vs-raster-"))
    for name in [s.strip() for s in args.camera_sets.split(",") if s.strip()]:
        viewmats, intrinsics = sets[name]

        # ---- raster ---------------------------------------------------------
        raster_means = []
        last_rgb = None
        for _ in range(args.repeats):
            for _ in range(args.warmup):
                service.render(viewmats, intrinsics, scene_ids)
            service.synchronize(); torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.frames):
                out = service.render(viewmats, intrinsics, scene_ids)
            service.synchronize(); torch.cuda.synchronize()
            raster_means.append((time.perf_counter() - t0) * 1000.0 / args.frames)
            last_rgb = out["rgb"]
        raster_env0 = (
            last_rgb[0].clamp(0, 1).mul(255.0).add(0.5).to(torch.uint8).cpu().numpy()
        )

        # ---- tracer (same cameras via --cams) -------------------------------
        cams_bin = tmpdir / f"cams_{name}.bin"
        write_cams_bin(cams_bin, viewmats.cpu().numpy(), intrinsics.cpu().numpy())
        tracer_means, tracer_meta = [], {}
        tracer_ppm = tmpdir / f"tracer_{name}_env0.ppm"
        for r in range(args.repeats):
            out_json = tmpdir / f"tracer_{name}_{r}.json"
            cmd = [
                str(args.tracer_bin.resolve()),
                "--dump", str(args.dump), "--mode", "dense",
                "--cams", str(cams_bin),
                "--width", str(args.width), "--height", str(args.height),
                "--frames", str(args.frames), "--kbuffer", "1",
                "--out-json", str(out_json),
            ] + args.tracer_extra.split() + (
                ["--out-ppm", str(tracer_ppm)] if r == 0 else []
            )
            subprocess.run(
                cmd, check=True, cwd=str(args.tracer_bin.resolve().parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            data = json.loads(out_json.read_text())
            tracer_means.append(data["mean_launch_ms"])
            tracer_meta = {
                "hit_fraction": data["hit_fraction"],
                "avg_traces_per_ray": data["avg_traces_per_ray"],
                "mrays_per_second": data["mrays_per_second"],
            }

        parity_psnr = psnr8(raster_env0, read_ppm(tracer_ppm))
        entry = {
            "raster_ms": raster_means,
            "raster_mean_ms": sum(raster_means) / len(raster_means),
            "tracer_ms": tracer_means,
            "tracer_mean_ms": sum(tracer_means) / len(tracer_means),
            "speedup_raster_over_tracer": (
                (sum(raster_means) / len(raster_means))
                / (sum(tracer_means) / len(tracer_means))
            ),
            "tracer": tracer_meta,
            "env0_parity_psnr_db": parity_psnr,
        }
        report["camera_sets"][name] = entry
        print(
            f"RT_VS_RASTER {name} raster={entry['raster_mean_ms']:.2f}ms "
            f"tracer={entry['tracer_mean_ms']:.2f}ms "
            f"speedup={entry['speedup_raster_over_tracer']:.2f}x "
            f"hit_frac={tracer_meta['hit_fraction']:.3f} "
            f"parity_psnr={parity_psnr:.1f}dB",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RT_VS_RASTER_OK " + json.dumps(
        {k: round(v["speedup_raster_over_tracer"], 2) for k, v in report["camera_sets"].items()}
    ))


if __name__ == "__main__":
    main()
