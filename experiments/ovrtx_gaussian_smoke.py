#!/usr/bin/env python3
"""Headless OVRTX smoke test for the OpenUSD Gaussian-splat schema.

This intentionally uses only the public OVRTX Python API and the concrete
``ParticleField3DGaussianSplat`` prim type bundled with OVRTX 0.3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ovrtx
from PIL import Image


RENDER_PRODUCT = "/Render/GaussianCamera"


def render_product_paths(camera_count: int, layout: str) -> list[str]:
    if layout == "tiled":
        return [RENDER_PRODUCT]
    return [f"{RENDER_PRODUCT}_{index}" for index in range(camera_count)]


def build_scene(
    width: int,
    height: int,
    color_aov: str = "both",
    camera_count: int = 1,
    layout: str = "tiled",
) -> str:
    sh_white = 0.5 / 0.28209479177387814
    color_names = {
        "ldr": ["LdrColor"],
        "hdr": ["HdrColor"],
        "both": ["LdrColor", "HdrColor"],
    }[color_aov]
    ordered_vars = ",\n            ".join(
        f"<{name}>" for name in [*color_names, "DepthSD", "DistanceToImagePlaneSD"]
    )
    render_var_defs = "\n\n".join(
        f"""        def RenderVar "{name}"
        {{
            string sourceName = "{name}"
        }}"""
        for name in [*color_names, "DepthSD", "DistanceToImagePlaneSD"]
    )
    camera_defs = "\n\n".join(
        f"""    def Camera "Camera_{index}" (
        prepend apiSchemas = [
            "OmniRtxCameraAutoExposureAPI_1",
            "OmniRtxCameraExposureAPI_1"
        ]
    )
    {{
        float2 clippingRange = (0.01, 100)
        float exposure:fStop = 5
        float exposure:responsivity = 1.1026709
        float exposure:time = 0.02
        float focalLength = 50
        float focusDistance = 5
        float fStop = 0
        float horizontalAperture = 36
        bool omni:rtx:autoExposure:enabled = 0
        float verticalAperture = 36
        token projection = "perspective"
        double3 xformOp:translate = (0, 0, 5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}"""
        for index in range(camera_count)
    )
    camera_paths = [f"</World/Camera_{index}>" for index in range(camera_count)]
    product_specs = (
        [("GaussianCamera", camera_paths)]
        if layout == "tiled"
        else [
            (f"GaussianCamera_{index}", [camera_path])
            for index, camera_path in enumerate(camera_paths)
        ]
    )
    render_product_defs = "\n\n".join(
        f"""    def RenderProduct "{product_name}"
    {{
        rel camera = [
            {", ".join(product_cameras)}
        ]
        int2 resolution = ({width}, {height})
        token omni:rtx:rendermode = "RealTimePathTracing"
        token omni:rtx:background:source:type = "color"
        rel orderedVars = [
            {ordered_vars}
        ]

{render_var_defs}
    }}"""
        for product_name, product_cameras in product_specs
    )
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "World"
{{
    def ParticleField3DGaussianSplat "Splats"
    {{
        point3f[] positions = [
            (-1.05, 0, 0),
            (0, 0, 0),
            (1.05, 0, 0),
            (0, 1.05, 0)
        ]
        float3[] scales = [
            (0.30, 0.30, 0.30),
            (0.30, 0.30, 0.30),
            (0.30, 0.30, 0.30),
            (0.30, 0.30, 0.30)
        ]
        quatf[] orientations = [
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 0)
        ]
        float[] opacities = [0.95, 0.95, 0.95, 0.95]
        uniform int radiance:sphericalHarmonicsDegree = 0
        float3[] radiance:sphericalHarmonicsCoefficients = [
            ({sh_white}, {-sh_white}, {-sh_white}),
            ({-sh_white}, {sh_white}, {-sh_white}),
            ({-sh_white}, {-sh_white}, {sh_white}),
            ({sh_white}, {sh_white}, {sh_white})
        ] (
            elementSize = 1
            interpolation = "vertex"
        )
        float3[] extent = [(-2, -1, -1), (2, 2, 1)]
        uniform token projectionModeHint = "perspective"
        uniform token sortingModeHint = "zDepth"
    }}

{camera_defs}
}}

def Scope "Render"
{{
{render_product_defs}
}}
"""


def copy_cpu_render_var(frame: object, name: str) -> np.ndarray:
    mapping = frame.render_vars[name].map(device=ovrtx.Device.CPU)
    try:
        return np.from_dlpack(mapping).copy()
    finally:
        mapping.unmap()


def collect_render_var(
    products: object,
    product_paths: list[str],
    name: str,
    layout: str,
) -> np.ndarray:
    arrays = [
        copy_cpu_render_var(products[path].frames[0], name)
        for path in product_paths
    ]
    return arrays[0] if layout == "tiled" else np.stack(arrays, axis=0)


def save_ldr(output_dir: Path, ldr: np.ndarray, layout: str) -> None:
    if layout == "tiled":
        Image.fromarray(ldr).save(output_dir / "ldr.png")
        return

    np.save(output_dir / "ldr.npy", ldr)
    images = [Image.fromarray(image) for image in ldr]
    for index, image in enumerate(images):
        image.save(output_dir / f"ldr_{index:04d}.png")
    montage = Image.new("RGBA", (sum(image.width for image in images), images[0].height))
    x_offset = 0
    for image in images:
        montage.paste(image, (x_offset, 0))
        x_offset += image.width
    montage.save(output_dir / "ldr_montage.png")


def save_hdr_preview(output_dir: Path, hdr_preview: np.ndarray, layout: str) -> None:
    if layout == "tiled":
        Image.fromarray((hdr_preview * 255.0).astype(np.uint8)).save(
            output_dir / "hdr_preview.png"
        )
        return
    previews = [
        Image.fromarray((image * 255.0).astype(np.uint8))
        for image in hdr_preview
    ]
    for index, image in enumerate(previews):
        image.save(output_dir / f"hdr_preview_{index:04d}.png")
    montage = Image.new("RGBA", (sum(image.width for image in previews), previews[0].height))
    x_offset = 0
    for image in previews:
        montage.paste(image, (x_offset, 0))
        x_offset += image.width
    montage.save(output_dir / "hdr_preview_montage.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ovrtx-gaussian-smoke"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--cameras", type=int, default=1)
    parser.add_argument("--layout", choices=("tiled", "products"), default="tiled")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--color-aov", choices=("ldr", "hdr", "both"), default="ldr")
    parser.add_argument(
        "--skip-tonemapping",
        choices=("default", "enabled", "disabled"),
        default="default",
    )
    args = parser.parse_args()
    if args.cameras <= 0:
        parser.error("--cameras must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene(
        args.width,
        args.height,
        args.color_aov,
        args.cameras,
        args.layout,
    )
    (args.output_dir / "scene.usda").write_text(scene, encoding="utf-8")
    product_paths = render_product_paths(args.cameras, args.layout)

    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            keep_system_alive=True,
            log_level="info",
            active_cuda_gpus="0",
            use_vulkan=True,
        )
    )
    renderer.open_usd_from_string(scene)
    if args.skip_tonemapping != "default":
        renderer.write_attribute(
            prim_paths=product_paths,
            attribute_name="omni:rtx:rtpt:gaussian:skipTonemapping:enabled",
            tensor=np.full(
                (len(product_paths),),
                args.skip_tonemapping == "enabled",
                dtype=np.bool_,
            ),
            prim_mode=ovrtx.PrimMode.CREATE_NEW,
        )
        renderer.reset()

    for _ in range(args.warmup):
        renderer.step(render_products=set(product_paths), delta_time=1.0 / 60.0)

    products = renderer.step(render_products=set(product_paths), delta_time=1.0 / 60.0)
    ldr = (
        collect_render_var(products, product_paths, "LdrColor", args.layout)
        if args.color_aov in {"ldr", "both"}
        else None
    )
    hdr = (
        collect_render_var(products, product_paths, "HdrColor", args.layout)
        if args.color_aov in {"hdr", "both"}
        else None
    )
    depth = collect_render_var(products, product_paths, "DepthSD", args.layout)
    distance_to_image_plane = collect_render_var(
        products,
        product_paths,
        "DistanceToImagePlaneSD",
        args.layout,
    )

    if ldr is not None:
        save_ldr(args.output_dir, ldr, args.layout)
    if hdr is not None:
        np.save(args.output_dir / "hdr.npy", hdr)
    np.save(args.output_dir / "depth.npy", depth)
    np.save(args.output_dir / "distance_to_image_plane.npy", distance_to_image_plane)

    hdr_rgb = hdr[..., :3].astype(np.float32) if hdr is not None else None
    if hdr_rgb is not None:
        finite_hdr_rgb = np.where(np.isfinite(hdr_rgb), hdr_rgb, 0.0)
        hdr_preview = np.clip(
            finite_hdr_rgb / (1.0 + np.maximum(finite_hdr_rgb, 0.0)),
            0.0,
            1.0,
        )
        save_hdr_preview(args.output_dir, hdr_preview, args.layout)

    rgb = ldr[..., :3] if ldr is not None else None
    alpha = ldr[..., 3] if ldr is not None else None
    finite_depth = depth[np.isfinite(depth) & (depth > 0)]
    finite_distance = distance_to_image_plane[
        np.isfinite(distance_to_image_plane) & (distance_to_image_plane > 0)
    ]
    finite_hdr_pixels = np.all(np.isfinite(hdr_rgb), axis=-1) if hdr_rgb is not None else None
    finite_nonzero_hdr_pixels = (
        finite_hdr_pixels & np.any(hdr_rgb != 0, axis=-1)
        if hdr_rgb is not None and finite_hdr_pixels is not None
        else None
    )
    summary = {
        "ovrtx_version": ovrtx.__version__,
        "renderer_version": list(renderer.version),
        "resolution": [args.width, args.height],
        "camera_count": args.cameras,
        "layout": args.layout,
        "render_product_count": len(product_paths),
        "single_host_submission": True,
        "warmup_frames": args.warmup,
        "color_aov": args.color_aov,
        "skip_tonemapping": args.skip_tonemapping,
        "sh_convention": "rgb = clamp(0.5 + 0.28209479177387814 * sh_dc)",
        "ldr_shape": list(ldr.shape) if ldr is not None else None,
        "ldr_dtype": str(ldr.dtype) if ldr is not None else None,
        "hdr_shape": list(hdr.shape) if hdr is not None else None,
        "hdr_dtype": str(hdr.dtype) if hdr is not None else None,
        "hdr_finite_pixels": int(np.count_nonzero(finite_hdr_pixels))
        if finite_hdr_pixels is not None
        else None,
        "hdr_finite_nonzero_pixels": int(np.count_nonzero(finite_nonzero_hdr_pixels))
        if finite_nonzero_hdr_pixels is not None
        else None,
        "hdr_nonfinite_values": int(np.count_nonzero(~np.isfinite(hdr))) if hdr is not None else None,
        "hdr_finite_min": float(hdr_rgb[np.isfinite(hdr_rgb)].min())
        if hdr_rgb is not None and np.any(np.isfinite(hdr_rgb))
        else None,
        "hdr_finite_max": float(hdr_rgb[np.isfinite(hdr_rgb)].max())
        if hdr_rgb is not None and np.any(np.isfinite(hdr_rgb))
        else None,
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "distance_to_image_plane_shape": list(distance_to_image_plane.shape),
        "distance_to_image_plane_dtype": str(distance_to_image_plane.dtype),
        "rgb_nonzero_pixels": int(np.count_nonzero(np.any(rgb != 0, axis=-1)))
        if rgb is not None
        else None,
        "rgb_unique_colors": int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0])
        if rgb is not None
        else None,
        "rgb_max": int(rgb.max()) if rgb is not None else None,
        "alpha_nonzero_pixels": int(np.count_nonzero(alpha)) if alpha is not None else None,
        "alpha_min": int(alpha.min()) if alpha is not None else None,
        "alpha_max": int(alpha.max()) if alpha is not None else None,
        "ldr_alpha_contract": (
            "composited_output_alpha_not_gaussian_coverage"
            if alpha is not None
            else None
        ),
        "finite_positive_depth_pixels": int(finite_depth.size),
        "finite_positive_depth_min": float(finite_depth.min()) if finite_depth.size else None,
        "finite_positive_depth_max": float(finite_depth.max()) if finite_depth.size else None,
        "finite_positive_distance_pixels": int(finite_distance.size),
        "finite_positive_distance_min": float(finite_distance.min()) if finite_distance.size else None,
        "finite_positive_distance_max": float(finite_distance.max()) if finite_distance.size else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    del products, renderer
    valid_ldr = summary["rgb_nonzero_pixels"] is not None and summary["rgb_nonzero_pixels"] > 0
    valid_hdr = (
        summary["hdr_finite_nonzero_pixels"] is not None
        and summary["hdr_finite_nonzero_pixels"] > 0
        and summary["hdr_nonfinite_values"] == 0
    )
    return 0 if (valid_ldr or valid_hdr) and summary["finite_positive_distance_pixels"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
