#!/usr/bin/env python3
"""Render the imported Unitree G1 through Isaac Sim RTX as a finite smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-usda", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--gaussians-dump",
        type=Path,
        help=(
            "Optionally keep the reusable OptiX Gaussian tracer alive across "
            "the real RTX frame and verify a CUDA sparse-ray probe before/after."
        ),
    )
    parser.add_argument("--tracer-probe-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.tracer_probe_size <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not args.g1_usda.is_file():
        raise FileNotFoundError(args.g1_usda)
    if args.gaussians_dump is not None and not args.gaussians_dump.is_file():
        raise FileNotFoundError(args.gaussians_dump)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": True, "renderer": "RayTracedLighting"}
    )
    tracer = None
    try:
        import omni.replicator.core as rep
        import omni.usd
        import torch
        import warp as wp
        from pxr import Gf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        robot = UsdGeom.Xform.Define(stage, "/World/G1")
        robot.GetPrim().GetReferences().AddReference(
            args.g1_usda.resolve().as_posix()
        )
        UsdGeom.XformCommonAPI(robot).SetTranslate((0.0, 0.0, 0.0))

        floor = UsdGeom.Cube.Define(stage, "/World/Floor")
        floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.04))
        floor.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.04))

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1200.0)
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(2500.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(35.0, -20.0, 30.0))

        camera = rep.create.camera(
            position=(1.7, -1.7, 1.3),
            look_at=(0.0, 0.0, 0.75),
            focal_length=35.0,
        )
        render_product = rep.create.render_product(
            camera, (args.width, args.height)
        )
        # These annotators expose native GPU buffers. Keep their Warp views
        # until after conversion to Torch so this smoke rejects host readback.
        color_aov = rep.AnnotatorRegistry.get_annotator(
            "LdrColor", device="cuda", do_array_copy=False
        )
        depth_aov = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane", device="cuda", do_array_copy=False
        )
        color_aov.attach(render_product)
        depth_aov.attach(render_product)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(args.output_dir), rgb=True)
        writer.attach([render_product])

        tracer_before = None
        tracer_probe = None
        if args.gaussians_dump is not None:
            tracer_dir = REPO_ROOT / "experiments" / "optix_tracer"
            if str(tracer_dir) not in sys.path:
                sys.path.insert(0, str(tracer_dir))
            from gs_tracer_torch import GsTracer

            tracer = GsTracer(args.gaussians_dump)
            probe_size = args.tracer_probe_size
            y, x = torch.meshgrid(
                torch.arange(probe_size, device="cuda", dtype=torch.float32),
                torch.arange(probe_size, device="cuda", dtype=torch.float32),
                indexing="ij",
            )
            focal = 0.9 * probe_size
            directions = torch.stack(
                (
                    (x + 0.5 - probe_size * 0.5) / focal,
                    (y + 0.5 - probe_size * 0.5) / focal,
                    torch.ones_like(x),
                ),
                dim=-1,
            ).reshape(-1, 3).contiguous()
            origin = torch.tensor(
                (
                    tracer.center[0],
                    tracer.center[1],
                    tracer.center[2] - 2.2 * tracer.radius,
                ),
                device="cuda",
                dtype=torch.float32,
            )
            origins = origin.expand(directions.shape[0], 3).contiguous()
            tracer_probe = (origins, directions)
            tracer_before = tracer.render_sparse(origins, directions)
            if not torch.isfinite(tracer_before["rgba"]).all():
                raise RuntimeError("OptiX sparse probe produced non-finite RGBA.")
            if not (tracer_before["rgba"][:, 3] >= 0.01).any():
                raise RuntimeError("OptiX sparse probe did not hit the Gaussian map.")

        rep.orchestrator.step(rt_subframes=8)
        rep.orchestrator.wait_until_complete()
        writer.detach()
        color = wp.to_torch(color_aov.get_data())
        depth = wp.to_torch(depth_aov.get_data())
        color_aov.detach()
        depth_aov.detach()
        if color.device.type != "cuda" or depth.device.type != "cuda":
            raise RuntimeError("Isaac RTX AOV bridge returned a host tensor.")
        if color.shape[:2] != (args.height, args.width):
            raise RuntimeError(f"Unexpected RTX color AOV shape: {color.shape}")
        if depth.shape[:2] != (args.height, args.width):
            raise RuntimeError(f"Unexpected RTX depth AOV shape: {depth.shape}")

        tracer_report = None
        if tracer is not None and tracer_probe is not None and tracer_before is not None:
            tracer_after = tracer.render_sparse(*tracer_probe)
            bitwise_stable = all(
                torch.equal(tracer_before[name], tracer_after[name])
                for name in ("rgba", "depth", "semantic")
            )
            if not bitwise_stable:
                raise RuntimeError("Isaac RTX frame perturbed the OptiX sparse probe.")
            alpha = tracer_after["rgba"][:, 3]
            depth_mask_valid = torch.equal(torch.isfinite(tracer_after["depth"]), alpha > 1.0e-8)
            background_semantic_valid = bool(
                (tracer_after["semantic"][alpha < 0.01] == -1).all().item()
            )
            if not depth_mask_valid or not background_semantic_valid:
                raise RuntimeError("OptiX sparse probe output contract failed.")
            tracer_report = {
                "dump": str(args.gaussians_dump.resolve()),
                "gaussians": tracer.count,
                "gas_build_ms": tracer.gas_build_ms,
                "probe_rays": int(alpha.numel()),
                "foreground_rays": int((alpha >= 0.01).sum().item()),
                "device": str(alpha.device),
                "bitwise_stable_across_rtx_frame": bitwise_stable,
                "depth_finite_mask_matches_alpha": depth_mask_valid,
                "background_semantic_valid": background_semantic_valid,
            }

        images = sorted(args.output_dir.glob("rgb_*.png"))
        if len(images) != 1:
            raise RuntimeError(f"Expected one RTX RGB image, found {images}")
        report = {
            "schema_version": "isaacsim-g1-rtx-render-smoke/v2",
            "pass": True,
            "scope": (
                "Isaac RTX mesh render with CUDA AOVs and, when requested, a "
                "persistent sparse OptiX Gaussian probe. It does not compose "
                "the layers or validate contact physics."
            ),
            "g1_usda": str(args.g1_usda.resolve()),
            "rgb_png": str(images[0].resolve()),
            "resolution": [args.width, args.height],
            "cuda_aovs": {
                "color_shape": list(color.shape),
                "color_dtype": str(color.dtype),
                "depth_shape": list(depth.shape),
                "depth_dtype": str(depth.dtype),
                "device": str(color.device),
            },
            "optix_sparse_coexistence": tracer_report,
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("ISAACSIM_G1_RTX_RENDER_OK " + json.dumps(report, sort_keys=True))
    finally:
        if tracer is not None:
            tracer.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
