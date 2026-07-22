"""Separately measured benchmark for the hybrid robot/mesh tensor compositor.

HYBRID_PHYSICS.md fixes the rendering boundary: the Gaussian renderer and the
mesh renderer are separate layers with separately measured timing, joined by a
depth-aware GPU compositor that never copies either layer through the CPU.
This runner measures `composite_gaussian_and_mesh` alone — no Gaussian
renderer, no RTX mesh renderer — over the protocol batch/resolution grid with
CUDA-event timing, a synchronized wall clock, steady-state allocation checks,
and fail-closed output validity flags.

The input layers are deterministic synthetic tensors whose depth/alpha layout
exercises every compositor branch: mesh strictly in front, mesh strictly
behind, interleaved and near-epsilon depth ties, and invalid regions (zero
alpha, non-finite depth) on either or both layers.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from isaacsim_gaussian_renderer.benchmark_manifest import sha256_json, stats
from isaacsim_gaussian_renderer.benchmark_schema import (
    SCHEMA_VERSION,
    write_result_json,
    write_results_csv,
)
from isaacsim_gaussian_renderer.hybrid_compositor import (
    composite_gaussian_and_mesh,
)


IMPLEMENTATION_NAME = "hybrid-compositor"
SCENE_ID = "synthetic-hybrid-layers"
DEFAULT_BATCHES = [1, 8, 32, 64, 128, 256]
DEFAULT_RESOLUTIONS = [(128, 128), (256, 256), (512, 512)]
PROTOCOL_MIN_ITERATIONS = 50
LAYER_KEYS = (
    "gaussian_rgb",
    "gaussian_alpha",
    "gaussian_depth",
    "mesh_rgb",
    "mesh_alpha",
    "mesh_depth",
)
COVERAGE_LAYOUT = (
    "row-band-0 mesh strictly in front; row-band-1 mesh strictly behind; "
    "row-band-2 checkerboard interleave of mesh-front and within-epsilon "
    "depth ties; row-band-3 invalid regions by column: gaussian zero-alpha "
    "(both-invalid on checker), gaussian infinite depth, mesh zero-alpha, "
    "mesh NaN depth"
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_resolutions(value: str) -> list[tuple[int, int]]:
    resolutions: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        width, height = item.split("x", 1)
        resolutions.append((int(width), int(height)))
    return resolutions


def hybrid_config_id(batch: int, width: int, height: int) -> str:
    return f"cfg-v1-hybrid-compositor-b{batch}-{width}x{height}"


def case_seed(base_seed: int, batch: int, width: int, height: int) -> int:
    """Stable per-case seed so every grid case uses distinct deterministic data."""
    return (base_seed * 1_000_003 + batch * 8_191 + width * 131 + height) % (2**31)


def build_case_grid(
    batches: list[int],
    resolutions: list[tuple[int, int]],
    *,
    base_seed: int,
) -> list[dict[str, Any]]:
    if not batches or not resolutions:
        raise ValueError("The case grid requires at least one batch and one resolution.")
    if min(batches) <= 0:
        raise ValueError("Batch sizes must be positive.")
    if min(min(width, height) for width, height in resolutions) < 8:
        raise ValueError("Resolutions must be at least 8x8 to hold every coverage band.")
    cases: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for width, height in resolutions:
        for batch in batches:
            key = (batch, width, height)
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                {
                    "batch": batch,
                    "width": width,
                    "height": height,
                    "config_id": hybrid_config_id(batch, width, height),
                    "seed": case_seed(base_seed, batch, width, height),
                }
            )
    return cases


def synthesize_hybrid_layers(
    batch: int,
    height: int,
    width: int,
    *,
    seed: int,
    depth_epsilon: float = 1.0e-4,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Deterministic CPU gaussian/mesh layer pair with full branch coverage.

    Generated on CPU with a seeded generator so the same seed produces
    bitwise-identical inputs on every host; the benchmark uploads the result
    to CUDA once, outside the measured loop.
    """
    if min(batch, height, width) <= 0:
        raise ValueError("Batch, height, and width must be positive.")
    if height < 8 or width < 8:
        # Height >= 8 keeps at least two rows in every band so the band-3
        # checkerboard holds both parities; width >= 8 keeps at least two
        # columns in every invalid-column phase.
        raise ValueError("Coverage layout requires height >= 8 and width >= 8.")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=generator, dtype=dtype)

    gaussian_rgb = rand(batch, height, width, 3)
    mesh_rgb = rand(batch, height, width, 3)
    gaussian_alpha = 0.10 + 0.85 * rand(batch, height, width, 1)
    mesh_alpha = 0.10 + 0.85 * rand(batch, height, width, 1)
    gaussian_depth = 2.0 + 6.0 * rand(batch, height, width, 1)

    row = torch.arange(height).reshape(1, height, 1, 1)
    col = torch.arange(width).reshape(1, 1, width, 1)
    band = (row * 4) // height
    checker = (row + col) % 2 == 0

    # Band 0: mesh strictly in front. Band 1: mesh strictly behind.
    mesh_depth = torch.where(band == 0, gaussian_depth - 1.0, gaussian_depth + 1.0)
    # Band 2: checkerboard interleave of mesh-front and within-epsilon ties
    # (a tie must premultiplied-alpha blend with the gaussian layer in front).
    mesh_depth = torch.where(
        (band == 2) & checker,
        gaussian_depth - 0.5,
        mesh_depth,
    )
    mesh_depth = torch.where(
        (band == 2) & ~checker,
        gaussian_depth + depth_epsilon * 0.25,
        mesh_depth,
    )

    # Band 3: invalid-layer regions selected by column phase.
    phase = col % 4
    gaussian_invalid_alpha = (band == 3) & (phase == 0)
    gaussian_invalid_depth = (band == 3) & (phase == 1)
    mesh_invalid_alpha = ((band == 3) & (phase == 2)) | (gaussian_invalid_alpha & checker)
    mesh_invalid_depth = (band == 3) & (phase == 3)
    gaussian_alpha = torch.where(
        gaussian_invalid_alpha,
        torch.zeros_like(gaussian_alpha),
        gaussian_alpha,
    )
    gaussian_depth = torch.where(
        gaussian_invalid_depth,
        torch.full_like(gaussian_depth, float("inf")),
        gaussian_depth,
    )
    mesh_alpha = torch.where(
        mesh_invalid_alpha,
        torch.zeros_like(mesh_alpha),
        mesh_alpha,
    )
    mesh_depth = torch.where(
        mesh_invalid_depth,
        torch.full_like(mesh_depth, float("nan")),
        mesh_depth,
    )
    return {
        "gaussian_rgb": gaussian_rgb.contiguous(),
        "gaussian_alpha": gaussian_alpha.contiguous(),
        "gaussian_depth": gaussian_depth.contiguous(),
        "mesh_rgb": mesh_rgb.contiguous(),
        "mesh_alpha": mesh_alpha.contiguous(),
        "mesh_depth": mesh_depth.contiguous(),
    }


def branch_coverage(
    layers: Mapping[str, torch.Tensor],
    *,
    min_alpha: float = 1.0e-4,
    depth_epsilon: float = 1.0e-4,
) -> dict[str, Any]:
    """Count the pixels reaching each compositor branch, using its selection rule."""
    gaussian_valid = (layers["gaussian_alpha"] > min_alpha) & torch.isfinite(
        layers["gaussian_depth"]
    )
    mesh_valid = (layers["mesh_alpha"] > min_alpha) & torch.isfinite(
        layers["mesh_depth"]
    )
    mesh_in_front = mesh_valid & (
        ~gaussian_valid
        | (layers["mesh_depth"] + depth_epsilon < layers["gaussian_depth"])
    )
    both_valid = gaussian_valid & mesh_valid
    near_tie = both_valid & (
        (layers["mesh_depth"] - layers["gaussian_depth"]).abs() <= depth_epsilon
    )
    counts = {
        "mesh_front_both_valid": int((both_valid & mesh_in_front).sum().item()),
        "gaussian_front_blend": int((both_valid & ~mesh_in_front).sum().item()),
        "near_tie_both_valid": int(near_tie.sum().item()),
        "mesh_only": int((mesh_valid & ~gaussian_valid).sum().item()),
        "gaussian_only": int((gaussian_valid & ~mesh_valid).sum().item()),
        "both_invalid": int((~gaussian_valid & ~mesh_valid).sum().item()),
    }
    return {
        **counts,
        "coverage_complete": all(count > 0 for count in counts.values()),
    }


@contextlib.contextmanager
def forbid_host_transfers() -> Iterator[None]:
    """Fail fast if the measured loop syncs or copies anything to the host.

    Patches the tensor host-transfer entry points and the device-wide
    synchronize so any `.cpu()`, `.item()`, `.numpy()`, `.tolist()`, or
    `torch.cuda.synchronize()` inside the guarded region raises instead of
    silently serializing the hot path.
    """

    def _forbidden(name: str):
        def _raise(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                f"{name} is forbidden inside the measured hybrid compositor "
                "loop; time with CUDA events and read statistics afterwards."
            )

        return _raise

    patched = {
        "cpu": torch.Tensor.cpu,
        "item": torch.Tensor.item,
        "numpy": torch.Tensor.numpy,
        "tolist": torch.Tensor.tolist,
    }
    original_synchronize = torch.cuda.synchronize
    try:
        for name in patched:
            setattr(torch.Tensor, name, _forbidden(f"torch.Tensor.{name}"))
        torch.cuda.synchronize = _forbidden("torch.cuda.synchronize")
        yield
    finally:
        for name, original in patched.items():
            setattr(torch.Tensor, name, original)
        torch.cuda.synchronize = original_synchronize


def tensor_map_bytes(tensors: Mapping[str, torch.Tensor]) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors.values()))


def git_commit() -> str:
    if os.environ.get("SOURCE_GIT_COMMIT"):
        return os.environ["SOURCE_GIT_COMMIT"]
    try:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            env=env,
        ).strip()
    except Exception:
        return "unknown"


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    driver = command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "driver_version": driver.splitlines()[0] if driver else None,
        "git_commit": git_commit(),
    }


def driver_process_memory_bytes() -> int | None:
    output = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    pid = str(os.getpid())
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == pid:
            return int(parts[1]) * 1024 * 1024
    return None


def assemble_case_result(
    case: Mapping[str, Any],
    *,
    environment_info: Mapping[str, Any],
    flags: Mapping[str, Any],
    warmup_iterations: int,
    measured_iterations: int,
    cold_start_ms: float,
    upload_seconds: float,
    gpu_samples_ms: list[float],
    wall_loop_seconds: float,
    memory: Mapping[str, Any],
    coverage: Mapping[str, Any],
    checks: Mapping[str, bool],
    notes: str = "",
) -> dict[str, Any]:
    """Assemble one protocol-shaped result JSON for a measured compositor case."""
    batch = int(case["batch"])
    width = int(case["width"])
    height = int(case["height"])
    pixels = batch * width * height
    gpu_stats = stats(gpu_samples_ms)
    mean_seconds = gpu_stats["mean"] / 1_000.0
    passed = bool(checks) and all(bool(value) for value in checks.values())
    dataset_spec = {
        "scene_id": SCENE_ID,
        "generator": "benchmarks/run_hybrid_compositor.synthesize_hybrid_layers",
        "seed": int(case["seed"]),
        "batch": batch,
        "width": width,
        "height": height,
        "depth_epsilon": flags.get("depth_epsilon"),
        "coverage_layout": COVERAGE_LAYOUT,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "config_id": case["config_id"],
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": dict(environment_info),
        "implementation": {
            "name": IMPLEMENTATION_NAME,
            "commit": environment_info.get("git_commit"),
            "flags": dict(flags),
        },
        "configuration": {
            "batch_size": batch,
            "width": width,
            "height": height,
            "scene_mode": "synthetic-hybrid-layers",
            "outputs": ["rgb", "alpha", "depth", "mesh_in_front"],
            "semantic_contract": (
                "compositor-scope-rgb-alpha-depth-selection-only; semantic "
                "passthrough is outside composite_gaussian_and_mesh"
            ),
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "measured_scope": (
                "composite_gaussian_and_mesh only; no Gaussian or mesh "
                "renderer inside the timed region"
            ),
        },
        "dataset": {
            **dataset_spec,
            "checksum_sha256": sha256_json(dataset_spec),
            "gaussian_count": None,
        },
        "timing": {
            "cold_start_ms": cold_start_ms,
            "upload_seconds": upload_seconds,
            "gpu_ms": gpu_stats,
            "wall_ms": {
                "mean": (
                    wall_loop_seconds * 1_000.0 / measured_iterations
                    if measured_iterations > 0
                    else None
                ),
                "loop_seconds": wall_loop_seconds,
                "sample_scope": "synchronized_whole_measured_loop",
            },
            "images_per_second": (
                batch / mean_seconds if mean_seconds > 0 else None
            ),
            "megapixels_per_second": (
                pixels / mean_seconds / 1.0e6 if mean_seconds > 0 else None
            ),
            "gpu_samples_ms": list(gpu_samples_ms),
            "sample_scope": "individual_cuda_event_pairs_synchronized_after_loop",
        },
        "memory": dict(memory),
        "work": {
            "pixels": pixels,
            "cameras": batch,
            "visible_gaussians": None,
            "tile_intersections": None,
            "branch_coverage": dict(coverage),
        },
        "fidelity": {
            "rgb_psnr": None,
            "rgb_ssim": None,
            "lpips": None,
            "alpha_mae": None,
            "depth_relative_error": None,
            "semantic_id_agreement": None,
            "note": (
                "Compositor correctness is unit-tested against exact blend "
                "expectations; renderer-vs-renderer fidelity is out of scope "
                "for this timing-only benchmark."
            ),
        },
        "checks": dict(checks),
        "status": {
            "verdict": "PASS" if passed else "FAIL",
            "pass": passed,
            "notes": notes,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        default=",".join(str(batch) for batch in DEFAULT_BATCHES),
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--resolutions",
        default=",".join(f"{w}x{h}" for w, h in DEFAULT_RESOLUTIONS),
        help="Comma-separated WxH resolutions.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iterations",
        type=int,
        default=PROTOCOL_MIN_ITERATIONS,
        help=(
            "Measured iterations per case; the protocol requires at least "
            f"{PROTOCOL_MIN_ITERATIONS} and shorter runs are marked FAIL."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--min-alpha", type=float, default=1.0e-4)
    parser.add_argument("--depth-epsilon", type=float, default=1.0e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hybrid-compositor"),
    )
    return parser.parse_args()


@torch.inference_mode()
def run_case(args: argparse.Namespace, case: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    host_layers = synthesize_hybrid_layers(
        case["batch"],
        case["height"],
        case["width"],
        seed=case["seed"],
        depth_epsilon=args.depth_epsilon,
    )
    upload_start = time.perf_counter()
    layers = {
        name: tensor.to(device=device).contiguous()
        for name, tensor in host_layers.items()
    }
    torch.cuda.synchronize()
    upload_seconds = time.perf_counter() - upload_start
    del host_layers

    coverage = branch_coverage(
        layers,
        min_alpha=args.min_alpha,
        depth_epsilon=args.depth_epsilon,
    )

    def composite() -> Any:
        return composite_gaussian_and_mesh(
            layers["gaussian_rgb"],
            layers["gaussian_alpha"],
            layers["gaussian_depth"],
            layers["mesh_rgb"],
            layers["mesh_alpha"],
            layers["mesh_depth"],
            min_alpha=args.min_alpha,
            depth_epsilon=args.depth_epsilon,
        )

    # Cold call, timed and synchronized separately per the protocol.
    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    output = composite()
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - cold_start) * 1_000.0

    for _ in range(args.warmup):
        output = composite()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    wall_start = time.perf_counter()
    with forbid_host_transfers():
        for start, end in zip(starts, ends, strict=True):
            start.record()
            output = composite()
            end.record()
    torch.cuda.synchronize()
    wall_loop_seconds = time.perf_counter() - wall_start
    allocated_after = torch.cuda.memory_allocated()
    gpu_samples_ms = [
        float(start.elapsed_time(end))
        for start, end in zip(starts, ends, strict=True)
    ]

    outputs = {
        "rgb": output.rgb,
        "alpha": output.alpha,
        "depth": output.depth,
        "mesh_in_front": output.mesh_in_front,
    }
    outputs_gpu_resident = all(tensor.is_cuda for tensor in outputs.values())
    rgb_finite = bool(torch.isfinite(output.rgb).all().item())
    alpha_finite = bool(torch.isfinite(output.alpha).all().item())
    alpha_in_unit_range = bool(
        ((output.alpha >= 0.0) & (output.alpha <= 1.0 + 1.0e-5)).all().item()
    )
    depth_valid = bool(
        (
            torch.isfinite(output.depth)
            | (torch.isinf(output.depth) & (output.depth > 0))
        )
        .all()
        .item()
    )
    infinite_depth_pixels = int(torch.isinf(output.depth).sum().item())
    depth_matches_invalid_regions = (
        infinite_depth_pixels == coverage["both_invalid"]
    )
    mesh_in_front_pixels = int(output.mesh_in_front.sum().item())
    expected_mesh_front = (
        coverage["mesh_front_both_valid"] + coverage["mesh_only"]
    )

    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "persistent_scene_bytes": tensor_map_bytes(layers),
        "reusable_workspace_bytes": tensor_map_bytes(outputs),
        "temporary_allocation_delta_bytes": int(allocated_after - allocated_before),
        "driver_process_memory_bytes": driver_process_memory_bytes(),
    }
    checks = {
        "outputs_gpu_resident": outputs_gpu_resident,
        "rgb_finite": rgb_finite,
        "alpha_finite": alpha_finite,
        "alpha_in_unit_range": alpha_in_unit_range,
        "depth_finite_or_positive_infinity": depth_valid,
        "infinite_depth_only_where_both_layers_invalid": depth_matches_invalid_regions,
        "mesh_in_front_matches_expected_selection": (
            mesh_in_front_pixels == expected_mesh_front
        ),
        "branch_coverage_complete": bool(coverage["coverage_complete"]),
        "no_measured_loop_allocation_growth": allocated_after == allocated_before,
        "no_host_transfer_in_measured_loop": True,
        "meets_protocol_iteration_minimum": (
            args.iterations >= PROTOCOL_MIN_ITERATIONS
        ),
    }
    result = assemble_case_result(
        case,
        environment_info=environment(),
        flags={
            "min_alpha": args.min_alpha,
            "depth_epsilon": args.depth_epsilon,
            "dtype": "float32",
            "inference_mode": True,
            "host_transfer_guard": True,
        },
        warmup_iterations=args.warmup,
        measured_iterations=args.iterations,
        cold_start_ms=cold_start_ms,
        upload_seconds=upload_seconds,
        gpu_samples_ms=gpu_samples_ms,
        wall_loop_seconds=wall_loop_seconds,
        memory=memory,
        coverage={
            **coverage,
            "mesh_in_front_pixels": mesh_in_front_pixels,
            "expected_mesh_in_front_pixels": expected_mesh_front,
        },
        checks=checks,
        notes=(
            "Compositor-only measurement per HYBRID_PHYSICS.md rendering "
            "boundary; both renderers are excluded from the timed region."
        ),
    )
    del output, outputs, layers
    return result


def main() -> None:
    args = parse_args()
    if min(args.warmup, args.iterations) <= 0:
        raise ValueError("Warmup and iterations must be positive.")
    if not 0.0 <= args.min_alpha <= 1.0:
        raise ValueError("--min-alpha must be in [0, 1].")
    if args.depth_epsilon < 0.0:
        raise ValueError("--depth-epsilon must be non-negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the hybrid compositor benchmark.")

    cases = build_case_grid(
        parse_csv_ints(args.batches),
        parse_resolutions(args.resolutions),
        base_seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for case in cases:
        try:
            result = run_case(args, case)
        except Exception as error:  # noqa: BLE001 - record the failed case and continue
            torch.cuda.empty_cache()
            errors.append({"config_id": case["config_id"], "error": repr(error)})
            continue
        results.append(result)
        write_result_json(result, args.output / f"{result['run_id']}.json")
        torch.cuda.empty_cache()

    write_results_csv(results, args.output / "hybrid-compositor-results.csv")
    all_passed = bool(results) and not errors and all(
        result["status"]["pass"] for result in results
    )
    manifest = {
        "schema_version": "hybrid-compositor-benchmark-manifest/v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "case_count": len(cases),
        "result_count": len(results),
        "results": [result["run_id"] for result in results],
        "errors": errors,
        "pass": all_passed,
    }
    (args.output / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker = (
        "HYBRID_COMPOSITOR_BENCHMARK_OK"
        if all_passed
        else "HYBRID_COMPOSITOR_BENCHMARK_FAILED"
    )
    print(
        marker,
        json.dumps(
            {
                "cases": len(cases),
                "passed": sum(
                    1 for result in results if result["status"]["pass"]
                ),
                "errors": len(errors),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
    )
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
