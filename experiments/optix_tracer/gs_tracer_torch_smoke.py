#!/usr/bin/env python3
"""Torch integration contract for the reusable OptiX tracer context.

Renders the tracer's built-in dense env-0..N camera set through
gs_tracer_torch (outputs landing in torch CUDA tensors) and compares every
contract output against the CLI's --out-raw dumps of the same configuration.
Since both paths run the identical PTX on identical inputs, the gate is exact
equality (bitwise for semantic, exact float compare for rgba/depth).

The smoke also exercises CUDA-resident cameras, heterogeneous per-camera
intrinsics, a non-default torch stream, input validation, and resource
reclamation. Those checks cover the integration properties needed by a
long-lived, vectorized Isaac process; CLI equality alone does not.

Run:
  ./gs_tracer --dump D.bin --mode dense --envs 8 --width 256 --height 256 \
      --frames 3 --out-raw cli_out
  /isaac-sim/python.sh experiments/optix_tracer/gs_tracer_torch_smoke.py \
      --dump D.bin --cli-raw cli_out --envs 8 --width 256 --height 256 \
      --output smoke.json
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--cli-raw", type=Path, required=True)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--inside-isaac",
        action="store_true",
        help=(
            "Initialize a headless Isaac SimulationApp before exercising the "
            "tracer, proving that the reusable OptiX context can coexist with "
            "Isaac RTX in the same process."
        ),
    )
    return p.parse_args()


def run_smoke(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from gs_tracer_torch import GsTracer

    with args.dump.open("rb") as f:
        raw = f.read(24)
    _, cx_, cy_, cz_, radius = struct.unpack("<q4f", raw)
    center = (cx_, cy_, cz_)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_before_create, total_memory = torch.cuda.mem_get_info()
    tracer = GsTracer(args.dump)
    assert tracer.count > 0
    torch.cuda.synchronize()
    free_after_create, _ = torch.cuda.mem_get_info()

    # Rebuild the CLI's built-in dense cameras as viewmats (identity rotation).
    # All origin arithmetic in float32, mirroring the CLI's C float math
    # exactly — a float64 derivation differs in the last ulp and flips
    # borderline hits, which the bitwise gate would (correctly) reject.
    envs, w, h = args.envs, args.width, args.height
    f32 = np.float32
    viewmats = torch.zeros((envs, 4, 4), dtype=torch.float64)
    for e in range(envs):
        frac = f32(e) / f32(envs - 1) if envs > 1 else f32(0.5)
        origin = np.array(
            [
                f32(center[0]) + (frac - f32(0.5)) * f32(0.2) * f32(radius),
                f32(center[1]),
                f32(center[2]) - f32(2.2) * f32(radius),
            ],
            dtype=np.float32,
        )
        viewmats[e] = torch.eye(4, dtype=torch.float64)
        viewmats[e, :3, 3] = -torch.as_tensor(origin, dtype=torch.float64)
    intrinsics = torch.zeros((envs, 3, 3), dtype=torch.float64)
    intrinsics[:, 0, 0] = 0.9 * w
    intrinsics[:, 1, 1] = 0.9 * w * (h / w)
    intrinsics[:, 0, 2] = w * 0.5
    intrinsics[:, 1, 2] = h * 0.5
    intrinsics[:, 2, 2] = 1.0

    out = tracer.render_dense(viewmats, intrinsics, width=w, height=h)

    # The ordinary vectorized Isaac path already owns its camera state on the
    # GPU. It must agree exactly with the CPU staging path for these cameras.
    viewmats_cuda = viewmats.to(device="cuda", dtype=torch.float32)
    intrinsics_cuda = intrinsics.to(device="cuda", dtype=torch.float32)
    device_out = tracer.render_dense(
        viewmats_cuda, intrinsics_cuda, width=w, height=h
    )
    device_camera_equal = all(
        torch.equal(out[name], device_out[name])
        for name in ("rgba", "depth", "semantic")
    )
    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        default_dtype_out = tracer.render_dense(
            viewmats_cuda, intrinsics_cuda, width=w, height=h
        )
    finally:
        torch.set_default_dtype(previous_default_dtype)
    explicit_output_dtypes = (
        default_dtype_out["rgba"].dtype == torch.float32
        and default_dtype_out["depth"].dtype == torch.float32
        and default_dtype_out["semantic"].dtype == torch.int32
    )
    explicit_default_range = tracer.render_dense(
        viewmats_cuda,
        intrinsics_cuda,
        width=w,
        height=h,
        near_plane=1.0e-4,
        far_plane=6.0 * tracer.radius,
    )
    explicit_default_range_equal = all(
        torch.equal(device_out[name], explicit_default_range[name])
        for name in ("rgba", "depth", "semantic")
    )
    far_clipped = tracer.render_dense(
        viewmats_cuda,
        intrinsics_cuda,
        width=w,
        height=h,
        far_plane=tracer.radius,
    )
    far_clip_reduces_foreground = bool(
        (far_clipped["rgba"][..., 3] >= 0.01).sum()
        < (device_out["rgba"][..., 3] >= 0.01).sum()
    )
    invalid_trace_range_rejected = False
    try:
        tracer.render_dense(
            viewmats_cuda,
            intrinsics_cuda,
            width=w,
            height=h,
            near_plane=2.0,
            far_plane=1.0,
        )
    except ValueError:
        invalid_trace_range_rejected = True

    # A batched launch must use each camera's K, rather than silently reusing
    # K[0]. Compare camera 1 in a mixed batch with an independent launch.
    hetero_viewmats = viewmats_cuda[:1].repeat(2, 1, 1).contiguous()
    hetero_intrinsics = intrinsics_cuda[:1].repeat(2, 1, 1).contiguous()
    hetero_intrinsics[1, 0, 0] *= 0.73
    hetero_intrinsics[1, 1, 1] *= 1.17
    hetero_intrinsics[1, 0, 2] += 3.25
    hetero_intrinsics[1, 1, 2] -= 2.5
    hetero_batch = tracer.render_dense(
        hetero_viewmats, hetero_intrinsics, width=w, height=h
    )
    hetero_single = tracer.render_dense(
        hetero_viewmats[1:2].contiguous(),
        hetero_intrinsics[1:2].contiguous(),
        width=w,
        height=h,
    )
    heterogeneous_intrinsics_equal = all(
        torch.equal(hetero_batch[name][1], hetero_single[name][0])
        for name in ("rgba", "depth", "semantic")
    )

    # Delay and poison caller-owned outputs on a non-default stream before the
    # launch. A stream-0 launch races this fill; a current-stream launch is
    # ordered after it and reproduces the reference.
    stream = torch.cuda.Stream()
    stream_out = {
        "rgba": torch.empty_like(device_out["rgba"]),
        "depth": torch.empty_like(device_out["depth"]),
        "semantic": torch.empty_like(device_out["semantic"]),
    }
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        stream_out["rgba"].fill_(-123.0)
        stream_out["depth"].fill_(-123.0)
        stream_out["semantic"].fill_(-123)
        tracer.render_dense(
            viewmats_cuda,
            intrinsics_cuda,
            width=w,
            height=h,
            out_rgba=stream_out["rgba"],
            out_depth=stream_out["depth"],
            out_semantic=stream_out["semantic"],
        )
    stream.synchronize()
    nondefault_stream_equal = all(
        torch.equal(stream_out[name], device_out[name])
        for name in ("rgba", "depth", "semantic")
    )

    sparse_mismatch_rejected = False
    try:
        tracer.render_sparse(
            torch.empty((2, 3), device="cuda"),
            torch.empty((3, 3), device="cuda"),
        )
    except ValueError:
        sparse_mismatch_rejected = True
    empty_sparse = tracer.render_sparse(
        torch.empty((0, 3), device="cuda"),
        torch.empty((0, 3), device="cuda"),
    )
    empty_sparse_valid = all(t.numel() == 0 for t in empty_sparse.values())
    # In-process launch timing (synchronous call includes the device sync).
    for _ in range(3):
        tracer.render_dense(
            viewmats, intrinsics, width=w, height=h,
            out_rgba=out["rgba"], out_depth=out["depth"],
            out_semantic=out["semantic"],
        )
    t0 = time.perf_counter()
    for _ in range(args.frames):
        tracer.render_dense(
            viewmats, intrinsics, width=w, height=h,
            out_rgba=out["rgba"], out_depth=out["depth"],
            out_semantic=out["semantic"],
        )
    lib_ms = (time.perf_counter() - t0) * 1000.0 / args.frames

    n = envs * h * w
    cli_rgba = np.fromfile(f"{args.cli_raw}.rgba.f32", dtype=np.float32, count=n * 4)
    cli_depth = np.fromfile(f"{args.cli_raw}.depth.f32", dtype=np.float32, count=n)
    cli_sem = np.fromfile(f"{args.cli_raw}.semantic.i32", dtype=np.int32, count=n)

    lib_rgba = out["rgba"].detach().cpu().numpy().reshape(-1)
    lib_depth = out["depth"].detach().cpu().numpy().reshape(-1)
    lib_sem = out["semantic"].detach().cpu().numpy().reshape(-1)

    finite = np.isfinite(cli_depth) & np.isfinite(lib_depth)
    rgba_max_abs = float(np.max(np.abs(lib_rgba - cli_rgba)))
    depth_max_abs = float(
        np.max(np.abs(lib_depth[finite] - cli_depth[finite]))
        if finite.any() else 0.0
    )
    inf_agree = bool(np.array_equal(np.isinf(cli_depth), np.isinf(lib_depth)))
    sem_equal = bool(np.array_equal(lib_sem, cli_sem))

    torch.cuda.synchronize()
    free_before_close, _ = torch.cuda.mem_get_info()
    tracer.close()
    torch.cuda.synchronize()
    free_after_close, _ = torch.cuda.mem_get_info()
    live_context_bytes = max(0, free_before_create - free_after_create)
    reclaimed_on_close_bytes = max(0, free_after_close - free_before_close)
    retained_after_close_bytes = max(
        0, live_context_bytes - reclaimed_on_close_bytes
    )
    close_reclaims_context = retained_after_close_bytes <= max(
        128 * 1024**2, int(live_context_bytes * 0.1)
    )
    closed_render_rejected = False
    try:
        tracer.render_dense(viewmats, intrinsics, width=w, height=h)
    except RuntimeError:
        closed_render_rejected = True
    malformed_dump_rejected = False
    with tempfile.NamedTemporaryFile(suffix=".bin") as malformed:
        malformed.write(b"not-a-gaussian-dump")
        malformed.flush()
        try:
            GsTracer(malformed.name)
        except RuntimeError:
            malformed_dump_rejected = True

    report = {
        "schema_version": "gs-tracer-torch-smoke/v2",
        "pass": bool(
            rgba_max_abs == 0.0
            and depth_max_abs == 0.0
            and inf_agree
            and sem_equal
            and device_camera_equal
            and explicit_output_dtypes
            and explicit_default_range_equal
            and far_clip_reduces_foreground
            and invalid_trace_range_rejected
            and heterogeneous_intrinsics_equal
            and nondefault_stream_equal
            and sparse_mismatch_rejected
            and empty_sparse_valid
            and close_reclaims_context
            and closed_render_rejected
            and malformed_dump_rejected
        ),
        "rgba_max_abs_diff": rgba_max_abs,
        "depth_max_abs_diff": depth_max_abs,
        "depth_inf_mask_equal": inf_agree,
        "semantic_bitwise_equal": sem_equal,
        "device_camera_bitwise_equal": device_camera_equal,
        "default_float64_still_allocates_float32_outputs": explicit_output_dtypes,
        "explicit_default_trace_range_bitwise_equal": explicit_default_range_equal,
        "far_clip_reduces_foreground": far_clip_reduces_foreground,
        "invalid_trace_range_rejected": invalid_trace_range_rejected,
        "heterogeneous_intrinsics_bitwise_equal": heterogeneous_intrinsics_equal,
        "nondefault_stream_bitwise_equal": nondefault_stream_equal,
        "sparse_mismatch_rejected": sparse_mismatch_rejected,
        "empty_sparse_valid": empty_sparse_valid,
        "closed_render_rejected": closed_render_rejected,
        "malformed_dump_rejected_without_process_exit": malformed_dump_rejected,
        "memory": {
            "device_total_bytes": total_memory,
            "live_context_bytes": live_context_bytes,
            "reclaimed_on_close_bytes": reclaimed_on_close_bytes,
            "retained_after_close_bytes": retained_after_close_bytes,
            "close_reclaims_context": close_reclaims_context,
        },
        "in_process_launch_ms": lib_ms,
        "gas_build_ms": tracer.gas_build_ms,
        "envs": envs,
        "width": w,
        "height": h,
        "has_semantics": tracer.has_semantics,
        "inside_isaac_simulation_app": args.inside_isaac,
        "zero_copy": "outputs are caller-owned torch CUDA tensors (data_ptr)",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("GS_TRACER_TORCH_SMOKE " + json.dumps(report, sort_keys=True))


def main() -> None:
    args = parse_args()
    simulation_app = None
    if args.inside_isaac:
        # Isaac/Kit must own application startup before importing torch or the
        # tracer library. This is deliberately the existing integration gate,
        # not a second reduced harness with weaker assertions.
        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {"headless": True, "renderer": "RayTracedLighting"}
        )
    try:
        run_smoke(args)
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
