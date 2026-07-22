#!/usr/bin/env python3
"""Probe OVRTX compositing settings for a Gaussian-coverage alpha channel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ovrtx

from ovrtx_gaussian_smoke import (
    build_scene,
    copy_cpu_render_var,
    render_product_paths,
)


def write_bool(renderer: ovrtx.Renderer, path: str, name: str, value: bool) -> None:
    renderer.write_attribute(
        prim_paths=[path],
        attribute_name=name,
        tensor=np.array([value], dtype=np.bool_),
        prim_mode=ovrtx.PrimMode.CREATE_NEW,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stage = build_scene(256, 256, color_aov="ldr", camera_count=1, layout="products")
    product = render_product_paths(1, "products")[0]
    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            keep_system_alive=True,
            log_level="info",
            active_cuda_gpus="0",
            use_vulkan=True,
        )
    )
    renderer.open_usd_from_string(stage)
    renderer.write_attribute(
        prim_paths=[product],
        attribute_name="omni:rtx:post:aa:op",
        tensor=["none"],
        prim_mode=ovrtx.PrimMode.CREATE_NEW,
    )

    results: list[dict[str, object]] = []
    cases = [
        {
            "enabled": enabled,
            "output_alpha": output_alpha,
            "do_composite": do_composite,
            "black_background": black_background,
            "premultiply": premultiply,
        }
        for enabled in (False, True)
        for output_alpha in (False, True)
        for do_composite in (False, True)
        for black_background in (False, True)
        for premultiply in (False, True)
        if enabled or not any(
            (output_alpha, do_composite, black_background, premultiply)
        )
    ]
    for case in cases:
        write_bool(
            renderer,
            product,
            "omni:rtx:post:compositing:enabled",
            case["enabled"],
        )
        write_bool(
            renderer,
            product,
            "omni:rtx:post:compositing:outputAlpha",
            case["output_alpha"],
        )
        write_bool(
            renderer,
            product,
            "omni:rtx:post:compositing:doComposite",
            case["do_composite"],
        )
        write_bool(
            renderer,
            product,
            "omni:rtx:post:compositing:blackBackground",
            case["black_background"],
        )
        write_bool(
            renderer,
            product,
            "omni:rtx:post:compositing:premultiply",
            case["premultiply"],
        )
        renderer.reset()
        for _ in range(2):
            renderer.step(render_products={product}, delta_time=1.0 / 60.0)
        outputs = renderer.step(render_products={product}, delta_time=1.0 / 60.0)
        ldr = copy_cpu_render_var(outputs[product].frames[0], "LdrColor")
        alpha = ldr[..., 3]
        rgb = ldr[..., :3]
        results.append(
            {
                **case,
                "alpha_min": int(alpha.min()),
                "alpha_max": int(alpha.max()),
                "alpha_zero_pixels": int(np.count_nonzero(alpha == 0)),
                "alpha_partial_pixels": int(
                    np.count_nonzero((alpha > 0) & (alpha < 255))
                ),
                "alpha_opaque_pixels": int(np.count_nonzero(alpha == 255)),
                "rgb_nonzero_pixels": int(
                    np.count_nonzero(np.any(rgb != 0, axis=-1))
                ),
            }
        )

    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if any(result["alpha_partial_pixels"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
