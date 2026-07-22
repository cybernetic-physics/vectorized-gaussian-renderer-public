#!/usr/bin/env python3
"""Verify semantic segmentation labels on Gaussian ParticleField prims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ovrtx
from PIL import Image

from isaacsim_gaussian_renderer.semantic_mapping import (
    decode_semantic_id_map,
    renderer_semantic_lut,
)


SH_C0 = 0.28209479177387814
PRODUCT = "/Render/GaussianCamera"


def particle(index: int, position: tuple[float, float, float], rgb: tuple[float, float, float]) -> str:
    sh = tuple((channel - 0.5) / SH_C0 for channel in rgb)
    return f"""    def ParticleField3DGaussianSplat "Splats_{index}" (
        prepend apiSchemas = ["SemanticsAPI:class"]
    )
    {{
        string semantic:class:params:semanticData = "{index}"
        string semantic:class:params:semanticType = "class"
        point3f[] positions = [{position}]
        float3[] scales = [(0.30, 0.30, 0.30)]
        quatf[] orientations = [(1, 0, 0, 0)]
        float[] opacities = [0.95]
        uniform int radiance:sphericalHarmonicsDegree = 0
        float3[] radiance:sphericalHarmonicsCoefficients = [{sh}] (
            elementSize = 1
            interpolation = "vertex"
        )
        float3[] extent = [(-2, -2, -1), (2, 2, 1)]
        uniform token projectionModeHint = "perspective"
        uniform token sortingModeHint = "zDepth"
    }}"""


def build_stage() -> str:
    particles = "\n\n".join(
        [
            particle(0, (-1.05, 0.0, 0.0), (1.0, 0.0, 0.0)),
            particle(1, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            particle(2, (1.05, 0.0, 0.0), (0.0, 0.0, 1.0)),
            particle(3, (0.0, 1.05, 0.0), (1.0, 1.0, 1.0)),
        ]
    )
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "World"
{{
{particles}

    def Camera "Camera"
    {{
        float2 clippingRange = (0.01, 100)
        float focalLength = 50
        float horizontalAperture = 36
        float verticalAperture = 36
        token projection = "perspective"
        double3 xformOp:translate = (0, 0, 5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}

def Scope "Render"
{{
    def RenderProduct "GaussianCamera"
    {{
        rel camera = </World/Camera>
        int2 resolution = (512, 512)
        token omni:rtx:rendermode = "RealTimePathTracing"
        token omni:rtx:background:source:type = "color"
        bool omni:rtx:post:aa:limitedOps = 0
        token omni:rtx:post:aa:op = "none"
        rel orderedVars = [
            <LdrColor>,
            <SemanticSegmentation>,
            <SemanticIdMap>
        ]

        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}

        def RenderVar "SemanticSegmentation"
        {{
            string sourceName = "SemanticSegmentation"
        }}

        def RenderVar "SemanticIdMap"
        {{
            string sourceName = "SemanticIdMap"
        }}
    }}
}}
"""


def copy_render_var(frame: object, name: str) -> np.ndarray:
    mapping = frame.render_vars[name].map(device=ovrtx.Device.CPU)
    try:
        return np.from_dlpack(mapping).copy()
    finally:
        mapping.unmap()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage = build_stage()
    (args.output_dir / "scene.usda").write_text(stage, encoding="utf-8")
    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            keep_system_alive=True,
            log_level="info",
            active_cuda_gpus="0",
            use_vulkan=True,
        )
    )
    renderer.open_usd_from_string(stage)
    for _ in range(8):
        renderer.step(render_products={PRODUCT}, delta_time=1.0 / 60.0)
    outputs = renderer.step(render_products={PRODUCT}, delta_time=1.0 / 60.0)
    frame = outputs[PRODUCT].frames[0]
    ldr = copy_render_var(frame, "LdrColor")
    semantic = copy_render_var(frame, "SemanticSegmentation")
    id_map = decode_semantic_id_map(copy_render_var(frame, "SemanticIdMap"))
    semantic_lut = renderer_semantic_lut(id_map, expected_group_count=4)
    Image.fromarray(ldr).save(args.output_dir / "ldr.png")
    np.save(args.output_dir / "semantic.npy", semantic)

    unique, counts = np.unique(semantic, return_counts=True)
    pixels = {
        str(int(identifier)): {
            "label": id_map.get(int(identifier)),
            "pixels": int(count),
        }
        for identifier, count in zip(unique, counts, strict=True)
    }
    summary = {
        "id_map": {str(identifier): label for identifier, label in id_map.items()},
        "pixels": pixels,
        "renderer_id_to_semantic_id": {
            str(renderer_id): int(semantic_id)
            for renderer_id, semantic_id in enumerate(semantic_lut)
            if semantic_id >= 0
        },
        "all_labels_present": set(semantic_lut[semantic_lut >= 0].tolist())
        == set(range(4)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_labels_present"] and len(unique) >= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
