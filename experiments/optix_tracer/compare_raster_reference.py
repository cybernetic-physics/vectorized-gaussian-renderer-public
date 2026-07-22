#!/usr/bin/env python3
"""Full output-contract parity: OptiX RT-core tracer vs the tile raster.

Reconstructs the tracer's dense-mode env-0 camera exactly (identity rotation,
origin from the dump header's center/radius, fx = 0.9*W, fy = 0.9*H, principal
point at the image center), renders the same Home Scan scene through
``CustomCudaBackend`` with the same spatial semantic ids the dump carries, and
compares every contract output against the tracer's ``--out-raw`` dumps:

  rgb      PSNR floor over the 8-bit linear image
  alpha    mean-absolute-error ceiling
  depth    mean relative error ceiling over jointly-valid pixels
           (both alphas above the validity threshold)
  semantic agreement-fraction floor over jointly-labeled pixels

Exact equality is not expected — the tracer composites with alpha_min=1/255 /
transmittance_min=0.03 and per-ray ordering, the raster with its tile
pipeline — so the gates are tolerances, not bitwise checks.

Run (after `bash build.sh` and an export via scripts/export_gaussians_bin.py):
  ./gs_tracer --dump gaussians.bin --mode dense --envs 32 --width 256 \
      --height 256 --kbuffer 1 --out-raw tracer_out
  /isaac-sim/python.sh experiments/optix_tracer/compare_raster_reference.py \
      --ply /workspace/datasets/home-scan-lod0.ply --dump gaussians.bin \
      --tracer-raw tracer_out --envs 32 --width 256 --height 256 \
      --output parity.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=Path, default=Path("/workspace/datasets/home-scan-lod0.ply"))
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument(
        "--tracer-raw", type=Path, required=True,
        help="gs_tracer --out-raw prefix (reads .rgba.f32/.depth.f32/.semantic.i32)",
    )
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--min-psnr-db", type=float, default=25.0)
    p.add_argument("--max-alpha-mae", type=float, default=0.03)
    p.add_argument("--max-depth-rel-err", type=float, default=0.05)
    p.add_argument("--min-semantic-agreement", type=float, default=0.90)
    p.add_argument("--valid-alpha", type=float, default=0.5)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def read_dump_header(path: Path) -> tuple[int, list[float], float]:
    with path.open("rb") as f:
        raw = f.read(8 + 4 * 4)
    count = struct.unpack("<q", raw[:8])[0]
    center = list(struct.unpack("<3f", raw[8:20]))
    radius = struct.unpack("<f", raw[20:24])[0]
    return count, center, radius


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch

    from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
    from isaacsim_gaussian_renderer.benchmark_manifest import (
        compact_semantic_ids,
        file_sha256,
        spatial_semantic_ids,
    )
    from isaacsim_gaussian_renderer.ply_loader import (
        canonicalize_3dgs_scene,
        load_ply_to_gaussians,
    )

    checksum = file_sha256(args.ply)
    if checksum != HOME_SCAN_SHA256:
        raise ValueError(f"Home Scan SHA-256 mismatch: {checksum}")

    _, center, radius = read_dump_header(args.dump)
    hw = args.height * args.width

    # Tracer env-0 outputs from the raw contract dumps.
    raw_prefix = str(args.tracer_raw)
    tr_rgba = np.fromfile(
        raw_prefix + ".rgba.f32", dtype=np.float32, count=hw * 4
    ).reshape(args.height, args.width, 4)
    tr_depth = np.fromfile(
        raw_prefix + ".depth.f32", dtype=np.float32, count=hw
    ).reshape(args.height, args.width)
    tr_sem = np.fromfile(
        raw_prefix + ".semantic.i32", dtype=np.int32, count=hw
    ).reshape(args.height, args.width)

    # Tracer dense-mode env-0 camera: identity rotation, y-down/z-forward.
    frac = 0.5 if args.envs <= 1 else 0.0
    origin = np.array(
        [
            center[0] + (frac - 0.5) * 0.2 * radius,
            center[1],
            center[2] - 2.2 * radius,
        ],
        dtype=np.float64,
    )
    viewmat = np.eye(4, dtype=np.float64)
    viewmat[:3, 3] = -origin  # world->camera with R = I
    fx = 0.9 * args.width
    fy = 0.9 * args.height
    intrinsics = np.array(
        [
            [fx, 0.0, args.width * 0.5],
            [0.0, fy, args.height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    raw = load_ply_to_gaussians(args.ply)
    canonical = canonicalize_3dgs_scene(raw, device="cuda")
    del raw
    semantic_ids = compact_semantic_ids(
        spatial_semantic_ids(canonical.means, grid=(2, 2, 2), dtype=torch.int64),
        dtype=torch.int64,
    )[0]

    near_plane = 0.01
    far_plane = 6.0 * radius  # tracer t_max
    backend = CustomCudaBackend(
        max_visible_records=500_000,
        max_intersections=math.ceil(
            2_400_000 * (args.width * args.height) / (128 * 128)
        ),
        near_plane=near_plane,
        far_plane=far_plane,
        gaussian_support_sigma=3.0,
        covariance_epsilon=0.0,
        semantic_min_alpha=0.01,
        tile_size=1,
        depth_bucket_count=128,
        depth_bucket_group_size=8,
        compact_projection_cache=True,
        enable_projection_cache=False,
        output_srgb=False,  # tracer output is linear
        deterministic=False,
    )
    service = RendererService(
        backend, height=args.height, width=args.width, max_views=1
    )
    service.initialize(stage=None, device="cuda")
    SCENE_ID = 7
    service.load_scene(
        SCENE_ID,
        means=canonical.means,
        scales=canonical.scales,
        rotations=canonical.rotations,
        opacities=canonical.opacities,
        features=canonical.features[:, :3].contiguous(),
        semantic_ids=semantic_ids,
    )
    out = service.render(
        torch.as_tensor(viewmat, device="cuda", dtype=torch.float32)[None],
        torch.as_tensor(intrinsics, device="cuda", dtype=torch.float32)[None],
        torch.full((1,), SCENE_ID, device="cuda", dtype=torch.int64),
    )
    rs_rgb = out["rgb"][0].clamp(0.0, 1.0).detach().cpu().numpy()
    rs_alpha = out["alpha"][0, ..., 0].detach().cpu().numpy()
    rs_depth = out["depth"][0, ..., 0].detach().cpu().numpy()
    rs_sem = out["semantic_id"][0, ..., 0].detach().cpu().numpy()

    # ---- rgb ---------------------------------------------------------------
    ref8 = (rs_rgb * 255.0 + 0.5).astype(np.uint8).astype(np.float64)
    tr8 = (np.clip(tr_rgba[..., :3], 0, 1) * 255.0 + 0.5).astype(np.uint8).astype(np.float64)
    mse = float(np.mean((ref8 - tr8) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)

    # ---- alpha -------------------------------------------------------------
    alpha_mae = float(np.mean(np.abs(rs_alpha - tr_rgba[..., 3])))

    # ---- depth (jointly-valid pixels) --------------------------------------
    valid = (rs_alpha > args.valid_alpha) & (tr_rgba[..., 3] > args.valid_alpha)
    valid &= np.isfinite(rs_depth) & np.isfinite(tr_depth)
    if valid.sum() == 0:
        depth_rel_err = float("inf")
    else:
        depth_rel_err = float(
            np.mean(
                np.abs(tr_depth[valid] - rs_depth[valid])
                / np.maximum(np.abs(rs_depth[valid]), 1e-3)
            )
        )

    # ---- semantic (jointly-labeled pixels) ---------------------------------
    labeled = (rs_sem >= 0) & (tr_sem >= 0)
    semantic_agreement = (
        float(np.mean(rs_sem[labeled] == tr_sem[labeled]))
        if labeled.sum() > 0
        else 0.0
    )

    report = {
        "schema_version": "optix-tracer-raster-parity/v2",
        "pass": bool(
            psnr >= args.min_psnr_db
            and alpha_mae <= args.max_alpha_mae
            and depth_rel_err <= args.max_depth_rel_err
            and semantic_agreement >= args.min_semantic_agreement
        ),
        "rgb_psnr_db": psnr,
        "alpha_mae": alpha_mae,
        "depth_rel_err": depth_rel_err,
        "depth_valid_px": int(valid.sum()),
        "semantic_agreement": semantic_agreement,
        "semantic_labeled_px": int(labeled.sum()),
        "gates": {
            "min_psnr_db": args.min_psnr_db,
            "max_alpha_mae": args.max_alpha_mae,
            "max_depth_rel_err": args.max_depth_rel_err,
            "min_semantic_agreement": args.min_semantic_agreement,
        },
        "envs": args.envs,
        "width": args.width,
        "height": args.height,
        "home_scan_sha256": checksum,
        "camera": {
            "origin": origin.tolist(),
            "fx": fx,
            "fy": fy,
            "rotation": "identity",
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("OPTIX_RASTER_PARITY " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
