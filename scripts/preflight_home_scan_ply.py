"""Inspect the canonical Home Scan PLY without allocating CUDA memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("/workspace/datasets/home-scan-lod0.ply"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/home-scan/preflight.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--sample-count", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.sample_count <= 0:
        raise ValueError("Chunk and sample counts must be positive.")
    if not args.path.is_file():
        raise FileNotFoundError(args.path)

    start_time = time.perf_counter()
    vertex = PlyData.read(str(args.path), mmap="r").elements[0]
    count = len(vertex)
    fields = [item.name for item in vertex.properties]
    expected = {
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(expected - set(fields))
    if missing:
        raise ValueError(f"Home Scan PLY is missing properties: {missing}")

    position_min = np.full(3, np.inf)
    position_max = np.full(3, -np.inf)
    raw_scale_min = np.full(3, np.inf)
    raw_scale_max = np.full(3, -np.inf)
    opacity_min = np.inf
    opacity_max = -np.inf
    rotation_norm_min = np.inf
    rotation_norm_max = -np.inf
    all_finite = True

    for start in range(0, count, args.chunk_size):
        stop = min(count, start + args.chunk_size)
        positions = np.stack(
            [
                np.asarray(vertex[name][start:stop])
                for name in ("x", "y", "z")
            ],
            axis=1,
        )
        scales = np.stack(
            [
                np.asarray(vertex[name][start:stop])
                for name in ("scale_0", "scale_1", "scale_2")
            ],
            axis=1,
        )
        rotations = np.stack(
            [
                np.asarray(vertex[name][start:stop])
                for name in ("rot_0", "rot_1", "rot_2", "rot_3")
            ],
            axis=1,
        )
        opacities = np.asarray(vertex["opacity"][start:stop])
        all_finite = all_finite and bool(
            np.isfinite(positions).all()
            and np.isfinite(scales).all()
            and np.isfinite(rotations).all()
            and np.isfinite(opacities).all()
        )
        position_min = np.minimum(position_min, positions.min(axis=0))
        position_max = np.maximum(position_max, positions.max(axis=0))
        raw_scale_min = np.minimum(raw_scale_min, scales.min(axis=0))
        raw_scale_max = np.maximum(raw_scale_max, scales.max(axis=0))
        opacity_min = min(opacity_min, float(opacities.min()))
        opacity_max = max(opacity_max, float(opacities.max()))
        rotation_norms = np.linalg.norm(rotations, axis=1)
        rotation_norm_min = min(
            rotation_norm_min,
            float(rotation_norms.min()),
        )
        rotation_norm_max = max(
            rotation_norm_max,
            float(rotation_norms.max()),
        )

    stride = max(1, count // args.sample_count)
    sample = slice(None, None, stride)
    sample_scale = np.stack(
        [
            np.asarray(vertex[name][sample])
            for name in ("scale_0", "scale_1", "scale_2")
        ],
        axis=1,
    )
    sample_opacity = np.asarray(vertex["opacity"][sample])
    sample_dc = np.stack(
        [
            np.asarray(vertex[name][sample])
            for name in ("f_dc_0", "f_dc_1", "f_dc_2")
        ],
        axis=1,
    )
    activated_scale = np.exp(sample_scale)
    activated_opacity = 1.0 / (1.0 + np.exp(-sample_opacity))
    linear_rgb = np.clip(
        0.5 + 0.28209479177387814 * sample_dc,
        0.0,
        1.0,
    )
    quantiles = (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)
    extent = position_max - position_min
    result = {
        "schema_version": "home-scan-preflight/v1",
        "path": str(args.path),
        "count": count,
        "fields": fields,
        "row_bytes": sum(
            np.dtype(vertex[name].dtype).itemsize
            for name in fields
        ),
        "file_bytes": args.path.stat().st_size,
        "all_finite": all_finite,
        "position_min": position_min.tolist(),
        "position_max": position_max.tolist(),
        "center": ((position_min + position_max) * 0.5).tolist(),
        "extent": extent.tolist(),
        "bounding_radius": float(np.linalg.norm(extent) * 0.5),
        "raw_scale_min": raw_scale_min.tolist(),
        "raw_scale_max": raw_scale_max.tolist(),
        "activated_scale_quantiles": {
            str(value): np.quantile(
                activated_scale,
                value,
                axis=0,
            ).tolist()
            for value in quantiles
        },
        "raw_opacity_min": opacity_min,
        "raw_opacity_max": opacity_max,
        "activated_opacity_quantiles": {
            str(value): float(np.quantile(activated_opacity, value))
            for value in quantiles
        },
        "rotation_norm_min": rotation_norm_min,
        "rotation_norm_max": rotation_norm_max,
        "linear_rgb_quantiles": {
            str(value): np.quantile(
                linear_rgb,
                value,
                axis=0,
            ).tolist()
            for value in quantiles
        },
        "sample_count": int(activated_opacity.size),
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HOME_SCAN_PREFLIGHT_OK", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
