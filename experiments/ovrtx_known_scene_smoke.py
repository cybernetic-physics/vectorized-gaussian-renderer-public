#!/usr/bin/env python3
"""Render an OVRTX-bundled mesh scene to validate the camera output pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ovrtx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("/workspace/tools/ovrtx-0.3.0/tests/data/simple_scene.usda"),
    )
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    scene = args.scene.resolve()
    usda = f"""#usda 1.0
(
    subLayers = [@{scene}@]
)

def Scope "RenderSmoke"
{{
    def RenderProduct "Camera"
    {{
        int2 resolution = (256, 144)
        rel camera = </Camera0>
        token omni:rtx:rendermode = "RealTimePathTracing"
        rel orderedVars = [<LdrColor>, <HdrColor>, <DistanceToImagePlaneSD>]

        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}

        def RenderVar "HdrColor"
        {{
            string sourceName = "HdrColor"
        }}

        def RenderVar "DistanceToImagePlaneSD"
        {{
            string sourceName = "DistanceToImagePlaneSD"
        }}
    }}
}}
"""

    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            keep_system_alive=True,
            active_cuda_gpus="0",
            use_vulkan=True,
            log_level="warn",
        )
    )
    renderer.open_usd_from_string(usda)
    products = None
    for _ in range(args.warmup):
        products = renderer.step({"/RenderSmoke/Camera"}, 1.0 / 60.0)

    assert products is not None
    frame = products["/RenderSmoke/Camera"].frames[0]
    summary: dict[str, object] = {}
    for name in ("LdrColor", "HdrColor", "DistanceToImagePlaneSD"):
        mapping = frame.render_vars[name].map(device=ovrtx.Device.CPU)
        try:
            array = np.from_dlpack(mapping).copy()
        finally:
            mapping.unmap()
        summary[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite_values": int(np.count_nonzero(np.isfinite(array))),
            "nonzero_values": int(np.count_nonzero(array)),
            "min": float(np.nanmin(array)),
            "max": float(np.nanmax(array)),
        }

    print(json.dumps(summary, indent=2, sort_keys=True))
    ldr_nonzero = int(summary["LdrColor"]["nonzero_values"])  # type: ignore[index]
    hdr_finite = int(summary["HdrColor"]["finite_values"])  # type: ignore[index]
    hdr_nonzero = int(summary["HdrColor"]["nonzero_values"])  # type: ignore[index]
    del products, frame, renderer
    return 0 if ldr_nonzero > 0 and hdr_finite > 0 and hdr_nonzero > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
