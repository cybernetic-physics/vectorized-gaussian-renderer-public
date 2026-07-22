#!/usr/bin/env python3
"""Render one real CUDA-tensor composite of Home Scan and a Unitree G1 mesh.

The static Home Scan layer is rendered by the project-owned Gaussian kernel.
Isaac Sim RTX renders only the imported G1 mesh, and Replicator exposes its
LDR colour and image-plane depth as CUDA-backed Warp/Torch tensors.  The two
layers are then depth-composited by :mod:`hybrid_compositor` before the three
PNG artifacts are deliberately copied to the host for review.

This is a fixed-state integration smoke.  ``--batch`` repeats one camera/robot
state through the GPU batch boundary, establishing the rendering bridge without
claiming independent multi-environment throughput, collision physics, or a
matched HDR colour pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


HOME_SCAN_COUNT = 21_497_908
HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
SCENE_ID = 404


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--home-scan-path", type=Path, default=Path("/workspace/datasets/home-scan-lod0.ply")
    )
    parser.add_argument("--g1-usda", type=Path, required=True)
    parser.add_argument(
        "--camera-contract",
        type=Path,
        default=Path("outputs/flyby/home-scan-v1/camera-path.npz"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Repeat one fixed camera/robot state through the GPU batch boundary.",
    )
    parser.add_argument("--rt-subframes", type=int, default=8)
    parser.add_argument("--min-mesh-pixels", type=int, default=128)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_png(path: Path, rgb) -> None:
    """Serialize an already display-referred RGB CUDA tensor for visual QA."""
    from PIL import Image
    import torch

    image = (rgb.clamp(0.0, 1.0) * 255.0 + 0.5).to(dtype=torch.uint8)
    image = image.detach().cpu().numpy()
    Image.fromarray(image).save(path, compress_level=4)


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.rt_subframes <= 0 or args.batch <= 0:
        raise ValueError("Image dimensions, batch, and rt-subframes must be positive.")
    if args.min_mesh_pixels <= 0:
        raise ValueError("min-mesh-pixels must be positive.")
    for path in (args.home_scan_path, args.g1_usda, args.camera_contract):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing evidence: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    # Isaac/Kit imports must occur after SimulationApp has initialized.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    service = None
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        import torch
        import warp as wp
        from pxr import Gf, UsdGeom, UsdLux

        from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
        from isaacsim_gaussian_renderer.camera_math import (
            opencv_viewmats_to_usd_camera_world_matrices,
        )
        from isaacsim_gaussian_renderer.hybrid_compositor import (
            composite_gaussian_and_mesh,
        )
        from isaacsim_gaussian_renderer.ply_loader import (
            canonicalize_3dgs_scene,
            load_ply_to_gaussians,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("This integration smoke requires CUDA.")
        scene_sha = sha256(args.home_scan_path)
        if scene_sha != HOME_SCAN_SHA256:
            raise ValueError(f"Home Scan SHA-256 mismatch: {scene_sha}")
        with np.load(args.camera_contract, allow_pickle=False) as contract:
            viewmats_np = np.asarray(contract["viewmats"], dtype=np.float32)
            intrinsics_np = np.asarray(contract["intrinsics"], dtype=np.float32)
            camera_manifest = json.loads(str(np.asarray(contract["manifest_json"]).item()))
        if not 0 <= args.camera_index < len(viewmats_np):
            raise IndexError(f"camera-index {args.camera_index} is outside the contract")
        contract_width = int(camera_manifest["width"])
        contract_height = int(camera_manifest["height"])
        scale_x = args.width / float(contract_width)
        scale_y = args.height / float(contract_height)
        if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=1.0e-7):
            raise ValueError("This smoke requires aspect-ratio-preserving resolution scaling.")
        viewmat = torch.from_numpy(viewmats_np[args.camera_index : args.camera_index + 1]).to(
            device="cuda", dtype=torch.float32
        ).repeat(args.batch, 1, 1).contiguous()
        intrinsics = torch.from_numpy(
            intrinsics_np[args.camera_index : args.camera_index + 1].copy()
        ).to(device="cuda", dtype=torch.float32).repeat(args.batch, 1, 1)
        intrinsics[:, 0, :] *= scale_x
        intrinsics[:, 1, :] *= scale_y
        intrinsics[:, 2, :] = torch.tensor((0.0, 0.0, 1.0), device="cuda")
        intrinsics = intrinsics.contiguous()

        raw = load_ply_to_gaussians(args.home_scan_path)
        if raw.count != HOME_SCAN_COUNT:
            raise ValueError(f"Home Scan count mismatch: {raw.count}")
        canonical = canonicalize_3dgs_scene(raw, device="cuda")
        del raw
        backend = CustomCudaBackend(
            max_visible_records=500_000 * args.batch,
            max_intersections=math.ceil(
                args.batch * 2_400_000 * (args.width * args.height) / (128 * 128)
            ),
            near_plane=float(camera_manifest["near_plane"]),
            far_plane=float(camera_manifest["far_plane"]),
            gaussian_support_sigma=3.0,
            covariance_epsilon=0.0,
            semantic_min_alpha=0.01,
            tile_size=1,
            depth_bucket_count=128,
            depth_bucket_group_size=8,
            compact_projection_cache=True,
            enable_projection_cache=False,
            output_srgb=True,
            deterministic=False,
        )
        service = RendererService(
            backend,
            height=args.height,
            width=args.width,
            max_views=args.batch,
        )
        service.initialize(stage=None, device="cuda")
        service.load_scene(
            SCENE_ID,
            means=canonical.means,
            scales=canonical.scales,
            rotations=canonical.rotations,
            opacities=canonical.opacities,
            features=canonical.features[:, :3].contiguous(),
            semantic_ids=torch.zeros(canonical.count, device="cuda", dtype=torch.int64),
        )
        del canonical
        gaussian = service.render(
            viewmat,
            intrinsics,
            torch.full((args.batch,), SCENE_ID, device="cuda", dtype=torch.int64),
        )
        service.synchronize()
        counters = backend.check_capacity(synchronize=False)
        if counters["visible_overflow"] or counters["intersection_overflow"]:
            raise RuntimeError(f"Custom Gaussian capacity overflow: {counters}")
        gaussian_rgb = gaussian["rgb"]
        gaussian_alpha = gaussian["alpha"]
        gaussian_depth = gaussian["depth"]
        if not torch.isfinite(gaussian_rgb).all() or not (gaussian_alpha > 0.0).any():
            raise RuntimeError("Custom Gaussian layer is not a valid foreground image.")

        # This inverse is already unit-tested in camera_math.  It preserves the
        # exact OpenCV camera contract when authoring a USD camera transform.
        usd_world = opencv_viewmats_to_usd_camera_world_matrices(viewmat)[0].cpu().numpy()
        camera_center = -viewmat[0, :3, :3].transpose(0, 1) @ viewmat[0, :3, 3]
        forward = viewmat[0, 2, :3]
        down = viewmat[0, 1, :3]
        robot_base = camera_center + forward * 2.5 + down * 0.35

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        robot = UsdGeom.Xform.Define(stage, "/World/G1")
        robot.GetPrim().GetReferences().AddReference(args.g1_usda.resolve().as_posix())
        robot_xform = UsdGeom.XformCommonAPI(robot)
        robot_xform.SetTranslate(tuple(float(value) for value in robot_base.detach().cpu()))
        # G1 is Z-up; the Home Scan camera contract is Y-down.  Map G1 +Z to
        # Home -Y so the default standing axis is upright in this scene.
        robot_xform.SetRotate((90.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1500.0)
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(3000.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(30.0, -25.0, 20.0))

        aperture = 36.0
        color_aovs = []
        depth_aovs = []
        for index in range(args.batch):
            camera = UsdGeom.Camera.Define(stage, f"/World/Camera_{index:04d}")
            camera.CreateFocalLengthAttr(
                float(intrinsics[index, 0, 0].item()) * aperture / args.width
            )
            camera.CreateHorizontalApertureAttr(aperture)
            camera.CreateVerticalApertureAttr(aperture * args.height / args.width)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, float(camera_manifest["far_plane"])))
            UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d(*usd_world.tolist()))
            render_product = rep.create.render_product(camera.GetPath(), (args.width, args.height))
            color_aov = rep.AnnotatorRegistry.get_annotator(
                "LdrColor", device="cuda", do_array_copy=False
            )
            depth_aov = rep.AnnotatorRegistry.get_annotator(
                "distance_to_image_plane", device="cuda", do_array_copy=False
            )
            color_aov.attach(render_product)
            depth_aov.attach(render_product)
            color_aovs.append(color_aov)
            depth_aovs.append(depth_aov)
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
        rep.orchestrator.wait_until_complete()
        mesh_rgba = torch.stack([wp.to_torch(aov.get_data()) for aov in color_aovs])
        mesh_depth_raw = torch.stack([wp.to_torch(aov.get_data()) for aov in depth_aovs])
        for aov in (*color_aovs, *depth_aovs):
            aov.detach()
        if mesh_rgba.device.type != "cuda" or mesh_depth_raw.device.type != "cuda":
            raise RuntimeError("Isaac RTX AOV bridge returned a host tensor.")
        if tuple(mesh_rgba.shape) != (args.batch, args.height, args.width, 4):
            raise RuntimeError(f"Unexpected RTX colour AOV shape: {mesh_rgba.shape}")
        if tuple(mesh_depth_raw.shape) != (args.batch, args.height, args.width):
            raise RuntimeError(f"Unexpected RTX depth AOV shape: {mesh_depth_raw.shape}")
        mesh_rgb = (mesh_rgba[..., :3].to(dtype=torch.float32) / 255.0).contiguous()
        mesh_depth = mesh_depth_raw.to(dtype=torch.float32).unsqueeze(-1).contiguous()
        mesh_alpha = (torch.isfinite(mesh_depth) & (mesh_depth > 0.0)).to(dtype=torch.float32)
        mesh_pixels_per_view = mesh_alpha.sum(dim=(1, 2, 3))
        if int(mesh_pixels_per_view.min().item()) < args.min_mesh_pixels:
            raise RuntimeError(f"RTX G1 foreground too small: {mesh_pixels_per_view.tolist()}")
        composite = composite_gaussian_and_mesh(
            gaussian_rgb, gaussian_alpha, gaussian_depth, mesh_rgb, mesh_alpha, mesh_depth
        )
        torch.cuda.synchronize()
        if any(tensor.device.type != "cuda" for tensor in (
            gaussian_rgb, gaussian_alpha, gaussian_depth, mesh_rgb, mesh_alpha, mesh_depth,
            composite.rgb, composite.alpha, composite.depth,
        )):
            raise RuntimeError("Hybrid rendering copied a tensor off CUDA before serialization.")
        front_pixels_per_view = composite.mesh_in_front.sum(dim=(1, 2, 3))
        if int(front_pixels_per_view.min().item()) < args.min_mesh_pixels:
            raise RuntimeError(f"G1 is not in front of the Gaussian layer: {front_pixels_per_view.tolist()}")

        write_png(args.output_dir / "custom-home-scan.png", gaussian_rgb[0])
        write_png(args.output_dir / "g1-rtx-matte.png", mesh_rgb[0] * mesh_alpha[0])
        write_png(args.output_dir / "composite.png", composite.rgb[0])
        report = {
            "schema_version": "isaacsim-home-g1-hybrid-composite-smoke/v1",
            "pass": True,
            "scope": (
                "CUDA-resident custom-Gaussian plus Isaac RTX G1 mesh composite over a "
                "fixed-state camera batch. Not a contact-physics, HDR-colour, or "
                "independent-environment throughput acceptance result."
            ),
            "scene": {
                "path": str(args.home_scan_path.resolve()),
                "sha256": scene_sha,
                "gaussian_count": HOME_SCAN_COUNT,
            },
            "g1_usda": str(args.g1_usda.resolve()),
            "camera_contract": str(args.camera_contract.resolve()),
            "camera_contract_sha256": sha256(args.camera_contract),
            "camera_index": args.camera_index,
            "batch": args.batch,
            "batch_contract": "Repeated fixed camera/robot state across CUDA batch entries.",
            "resolution": [args.width, args.height],
            "rt_subframes": args.rt_subframes,
            "custom_counters": {name: int(value) for name, value in counters.items()},
            "cuda_aovs": {
                "color_shape": list(mesh_rgba.shape), "color_dtype": str(mesh_rgba.dtype),
                "depth_shape": list(mesh_depth_raw.shape), "depth_dtype": str(mesh_depth_raw.dtype),
                "device": str(mesh_rgba.device),
            },
            "mesh_foreground_pixels_per_view": [int(value) for value in mesh_pixels_per_view.cpu()],
            "mesh_in_front_pixels_per_view": [int(value) for value in front_pixels_per_view.cpu()],
            "artifacts": {
                "custom": "custom-home-scan.png", "mesh": "g1-rtx-matte.png", "composite": "composite.png",
            },
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("ISAACSIM_HOME_G1_HYBRID_COMPOSITE_OK " + json.dumps(report, sort_keys=True))
    finally:
        if service is not None:
            service.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
