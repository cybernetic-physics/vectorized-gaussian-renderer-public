#!/usr/bin/env python3
"""Export the canonical Gaussian scene to a flat binary for the OptiX tracer.

Layout (little-endian):
  header: int64 count, float32 center[3], float32 radius
  means      float32 [N,3]
  precision  float32 [N,6]   (xx,xy,xz,yy,yz,zz of the symmetric square-root
                              precision R diag(1/scale) R^T)
  rgb        float32 [N,3]
  opacity    float32 [N]
  aabb       float32 [N,6]   (min xyz, max xyz; opacity-adaptive by default)
  semantics  int32   [N]     (compact spatial semantic ids — the same 2x2x2
                              grid rule the OVRTX lane uses; optional trailing
                              section, older dumps without it still load)
  tag         char     [8]    ("GST3SQRT", marks the precision representation)
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=Path, default=Path("/workspace/datasets/home-scan-lod0.ply"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument(
        "--adaptive-aabb", action=argparse.BooleanOptionalAction, default=True,
        help="shrink each AABB to the radius where the gaussian's response "
        "can still reach --alpha-min (r = sqrt(2 ln(opacity/alpha_min)) "
        "sigmas, capped at --sigma), and drop gaussians whose opacity is "
        "at or below --alpha-min entirely; measured 15-19%% faster tracing "
        "at identical parity PSNR (default on, --no-adaptive-aabb for the "
        "fixed-sigma layout)",
    )
    # 2/255 measured 14-16% faster tracing than 1/255 at equal-or-better
    # full-contract parity (PSNR 36.34, semantic agreement 99.4% — RESULTS.md).
    p.add_argument("--alpha-min", type=float, default=2.0 / 255.0)
    args = p.parse_args()

    import torch
    from isaacsim_gaussian_renderer.ply_loader import (
        canonicalize_3dgs_scene,
        load_ply_to_gaussians,
    )

    raw = load_ply_to_gaussians(args.ply)
    c = canonicalize_3dgs_scene(raw, device="cuda")
    del raw
    means = c.means            # [N,3]
    scales = c.scales          # [N,3]
    quats = c.rotations        # [N,4] wxyz
    opac = c.opacities         # [N]
    rgb = c.features[:, :3].contiguous()
    n = means.shape[0]

    # Rotation matrices from wxyz quaternions.
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(n, 3, 3)

    sqrt_inv_cov = torch.einsum(
        "nij,nj,nkj->nik", R, 1.0 / scales.clamp_min(1.0e-12), R
    )
    precision6 = torch.stack([
        sqrt_inv_cov[:, 0, 0], sqrt_inv_cov[:, 0, 1],
        sqrt_inv_cov[:, 0, 2], sqrt_inv_cov[:, 1, 1],
        sqrt_inv_cov[:, 1, 2], sqrt_inv_cov[:, 2, 2],
    ], dim=1)

    # World-axis extents: e_i = r_sigma * ||row_i(R diag(s))||.
    RS = R * scales.unsqueeze(1)  # [N,3,3] rows scaled
    if args.adaptive_aabb:
        keep = opac > args.alpha_min
        dropped = int((~keep).sum().item())
        means, precision6, rgb, opac = (
            means[keep], precision6[keep], rgb[keep], opac[keep]
        )
        RS = RS[keep]
        n = means.shape[0]
        # Beyond r sigmas the response opacity*exp(-r^2/2) is below alpha_min,
        # so the intersection program rejects every hit there anyway.
        r_sigma = torch.sqrt(2.0 * torch.log(opac / args.alpha_min)).clamp(
            min=0.1, max=args.sigma
        )
        print(
            f"adaptive aabb: dropped {dropped} gaussians <= alpha_min, "
            f"mean radius {float(r_sigma.mean()):.2f} sigma "
            f"(fixed would be {args.sigma:.1f})"
        )
        ext = r_sigma.unsqueeze(1) * torch.linalg.vector_norm(RS, dim=2)
    else:
        ext = args.sigma * torch.linalg.vector_norm(RS, dim=2)  # [N,3]
    aabb = torch.cat([means - ext, means + ext], dim=1)

    bb_min = means.amin(dim=0)
    bb_max = means.amax(dim=0)
    center = ((bb_min + bb_max) * 0.5).cpu().numpy()
    radius = float((torch.linalg.vector_norm(bb_max - bb_min) * 0.5).item())

    from isaacsim_gaussian_renderer.benchmark_manifest import (
        compact_semantic_ids,
        spatial_semantic_ids,
    )

    semantics = compact_semantic_ids(
        spatial_semantic_ids(means, grid=(2, 2, 2), dtype=torch.int64),
        dtype=torch.int64,
    )[0].to(torch.int32)
    groups = int(semantics.max().item()) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(struct.pack("<q", n))
        f.write(struct.pack("<3f", *center.tolist()))
        f.write(struct.pack("<f", radius))
        for tensor in (means, precision6, rgb, opac, aabb):
            f.write(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
        f.write(semantics.detach().cpu().numpy().tobytes())
        f.write(b"GST3SQRT")
    print(
        f"GAUSSIAN_EXPORT_OK count={n} radius={radius:.2f} "
        f"semantic_groups={groups} stable_precision_count={n} "
        f"stable_precision=1 bytes={args.output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
