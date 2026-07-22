"""Run reproducible baseline measurements and emit JSON/CSV results."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .renderer_options import (
        GSPLAT_RENDERED_OUTPUTS,
        gsplat_config_id,
        rasterize_mode_qualified,
    )
else:
    from renderer_options import (
        GSPLAT_RENDERED_OUTPUTS,
        gsplat_config_id,
        rasterize_mode_qualified,
    )

from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle,
    file_sha256,
    stats,
    synthetic_scene_manifest,
    synthetic_scene_tensors,
    tensor_bytes,
)
from isaacsim_gaussian_renderer.benchmark_schema import (
    SCHEMA_VERSION,
    write_json_schema,
    write_result_json,
    write_results_csv,
)


SYNTHETIC_SCENES = {
    "synthetic-small": 10_000,
    "synthetic-medium": 100_000,
}

GSPAT_OUTPUT_CONTRACT = "rgb_depth_alpha_native_semantic_id_unavailable"


def gsplat_output_contract() -> dict[str, Any]:
    """Describe only the AOVs produced by gsplat's public rasterizer."""
    return {
        "outputs": list(GSPLAT_RENDERED_OUTPUTS),
        "output_contract": GSPAT_OUTPUT_CONTRACT,
        "output_match_scope": "rgb_depth_alpha_only",
        "semantic_contract": "unavailable_not_measured",
    }


def gsplat_output_bytes(batch_size: int, width: int, height: int) -> int:
    """Return bytes written for float32 RGB, depth, and alpha outputs."""
    return batch_size * width * height * (3 + 1 + 1) * 4


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


def git_commit() -> str:
    if os.environ.get("SOURCE_GIT_COMMIT"):
        return os.environ["SOURCE_GIT_COMMIT"]
    try:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, env=env).strip()
    except Exception:
        return "unknown"


def command_output(args: list[str], *, clean_git_env: bool = False) -> str | None:
    try:
        env = None
        if clean_git_env:
            env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, env=env).strip()
    except Exception:
        return None


def environment() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    driver = command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
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
    pid = os.getpid()
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            return int(parts[1]) * 1024 * 1024
    return None


def implementation_commit(name: str) -> str:
    if name == "gsplat":
        gsplat_root = Path("/workspace/src/gsplat")
        if gsplat_root.exists():
            commit = command_output(["git", "-C", str(gsplat_root), "rev-parse", "HEAD"], clean_git_env=True)
            if commit:
                return commit
        return command_output([sys.executable, "-c", "import gsplat, pathlib; print(pathlib.Path(gsplat.__file__).resolve())"]) or "unknown"
    if name.startswith("ovrtx"):
        ovrtx_root = Path("/workspace/src/ovrtx")
        if ovrtx_root.exists():
            return command_output(["git", "-C", str(ovrtx_root), "rev-parse", "HEAD"], clean_git_env=True) or "unknown"
    return "unknown"


def load_scene(scene_id: str, scene_path: Path | None, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if scene_id in SYNTHETIC_SCENES:
        count = SYNTHETIC_SCENES[scene_id]
        seed = 42 if scene_id == "synthetic-small" else 43
        tensors = synthetic_scene_tensors(count, seed=seed, device=device)
        tensors.pop("semantic_ids", None)
        manifest = synthetic_scene_manifest(scene_id, count, seed)
        return tensors, manifest

    if scene_path is None:
        raise ValueError(f"Scene {scene_id!r} requires --scene-path.")
    from isaacsim_gaussian_renderer.ply_loader import load_ply_to_gaussians

    splats = load_ply_to_gaussians(scene_path)
    means = splats.means.to(device=device, dtype=torch.float32).contiguous()
    quats = torch.nn.functional.normalize(
        splats.rotations.to(device=device, dtype=torch.float32),
        dim=-1,
    ).contiguous()
    scales = splats.scales.to(device=device, dtype=torch.float32).exp().contiguous()
    opacities = splats.opacities.to(device=device, dtype=torch.float32).sigmoid().contiguous()
    colors = (
        splats.features.reshape(means.shape[0], -1, 3)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    manifest = {
        "version": "benchmark-manifest/v1",
        "scene_id": scene_id,
        "scene_type": "3dgs-ply",
        "path": str(scene_path),
        "checksum_sha256": file_sha256(scene_path),
        "gaussian_count": int(means.shape[0]),
        "attributes": ["means", "quats", "scales", "opacities", "sh"],
    }
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }, manifest


@torch.inference_mode()
def run_gsplat_case(args: argparse.Namespace, scene_id: str, width: int, height: int, batch_size: int) -> dict[str, Any]:
    from gsplat.rendering import rasterization

    device = torch.device("cuda")
    scene_path = Path(args.scene_path) if args.scene_path else None
    scene, scene_manifest = load_scene(scene_id, scene_path, device)
    cameras = camera_bundle(batch_size, width, height, device=device)
    backgrounds = torch.tensor(args.background, device=device, dtype=torch.float32).expand(batch_size, 3)
    sh_degree = None if scene["colors"].ndim == 2 else args.sh_degree

    def render_once() -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        rendered, alpha, meta = rasterization(
            scene["means"],
            scene["quats"],
            scene["scales"],
            scene["opacities"],
            scene["colors"],
            cameras.viewmats,
            cameras.intrinsics,
            width,
            height,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            radius_clip=args.radius_clip,
            packed=args.packed,
            backgrounds=backgrounds,
            render_mode="RGB+D",
            rasterize_mode=args.rasterize_mode,
            eps2d=args.eps2d,
            sh_degree=sh_degree,
        )
        return rendered, alpha, meta

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    persistent_scene_bytes = tensor_bytes(scene)

    cold_start = time.perf_counter()
    render_once()
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - cold_start) * 1000.0

    for _ in range(args.warmup):
        render_once()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    wall_values: list[float] = []
    gpu_values: list[float] = []
    last_meta: dict[str, Any] = {}
    for _ in range(args.iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        _rendered, _alpha, meta = render_once()
        end_event.record()
        torch.cuda.synchronize()
        wall_values.append((time.perf_counter() - wall_start) * 1000.0)
        gpu_values.append(float(start_event.elapsed_time(end_event)))
        last_meta = meta

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    allocated_after = torch.cuda.memory_allocated()
    gpu_stats = stats(gpu_values)
    wall_stats = stats(wall_values)
    mean_seconds = gpu_stats["mean"] / 1000.0
    pixels = batch_size * width * height
    run_id = str(uuid.uuid4())
    config_id = gsplat_config_id(
        scene_id=scene_id,
        scene_mode=args.scene_mode,
        batch_size=batch_size,
        width=width,
        height=height,
        rasterize_mode=args.rasterize_mode,
    )

    visible = None
    intersections = None
    if isinstance(last_meta, dict):
        for key in ("radii", "tiles_per_gauss"):
            value = last_meta.get(key)
            if torch.is_tensor(value):
                if key == "radii":
                    visible = int((value > 0).sum().item())
                elif key == "tiles_per_gauss":
                    intersections = int(value.sum().item())

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_id": config_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": environment(),
        "implementation": {
            "name": "gsplat",
            "commit": implementation_commit("gsplat"),
            "flags": {
                "packed": args.packed,
                "rasterize_mode": args.rasterize_mode,
                "eps2d": args.eps2d,
                "render_mode": "RGB+D",
                "sh_degree": sh_degree,
                "inference_mode": True,
            },
        },
        "configuration": {
            "batch_size": batch_size,
            "width": width,
            "height": height,
            "scene_mode": args.scene_mode,
            **gsplat_output_contract(),
            "eps2d": args.eps2d,
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "near_plane": args.near_plane,
            "far_plane": args.far_plane,
            "background": args.background,
            "camera_manifest": cameras.manifest,
        },
        "dataset": scene_manifest,
        "timing": {
            "cold_start_ms": cold_start_ms,
            "gpu_ms": gpu_stats,
            "wall_ms": wall_stats,
            "images_per_second": batch_size / mean_seconds if mean_seconds > 0 else None,
            "megapixels_per_second": pixels / mean_seconds / 1.0e6 if mean_seconds > 0 else None,
            "gpu_samples_ms": gpu_values,
            "wall_samples_ms": wall_values,
        },
        "memory": {
            "peak_allocated_bytes": int(peak_allocated),
            "peak_reserved_bytes": int(peak_reserved),
            "persistent_scene_bytes": int(persistent_scene_bytes),
            "reusable_workspace_bytes": None,
            "temporary_allocation_delta_bytes": int(max(0, allocated_after - allocated_before)),
            "driver_process_memory_bytes": driver_process_memory_bytes(),
        },
        "work": {
            "gaussians": int(scene["means"].shape[0]),
            "cameras": batch_size,
            "pixels": pixels,
            "output_bytes": gsplat_output_bytes(
                batch_size,
                width,
                height,
            ),
            "visible_gaussians": visible,
            "tile_intersections": intersections,
        },
        "fidelity": {
            "reference": None,
            "rgb_psnr": None,
            "rgb_ssim": None,
            "lpips": None,
            "alpha_mae": None,
            "depth_valid_relative_error": None,
            "semantic_id_agreement": None,
        },
        "status": {
            "verdict": "MEASURED_CORE_OUTPUTS_ONLY",
            "pass": False,
            "notes": "The public gsplat path renders RGB/depth/alpha only. No fake semantic AOV is allocated or timed, so this result is excluded from all-output and semantic-parity claims.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("BENCHMARK_PROTOCOL.md"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--implementations", default="gsplat")
    parser.add_argument("--scenes", default="synthetic-small")
    parser.add_argument("--scene-path")
    parser.add_argument("--scene-mode", choices=["shared-scene", "batched-scene-ids"], default="shared-scene")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--resolutions", default="128x128")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument("--eps2d", type=float, default=0.3)
    parser.add_argument("--sh-degree", type=int, default=None)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=100.0)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    args = parser.parse_args()

    if not args.protocol.is_file():
        raise FileNotFoundError(f"Benchmark protocol not found: {args.protocol}")
    if args.scene_mode != "shared-scene":
        raise ValueError(
            "The current gsplat control runner implements shared-scene batching only; "
            "multi-scene IDs must not be mislabeled."
        )
    if not math.isfinite(args.eps2d) or args.eps2d < 0:
        raise ValueError("--eps2d must be finite and non-negative.")

    args.output.mkdir(parents=True, exist_ok=True)
    write_json_schema(args.output / "baseline-result.schema.json")
    implementations = {item.strip() for item in args.implementations.split(",") if item.strip()}
    scenes = [item.strip() for item in args.scenes.split(",") if item.strip()]
    batches = parse_csv_ints(args.batches)
    resolutions = parse_resolutions(args.resolutions)

    results: list[dict[str, Any]] = []
    if "gsplat" in implementations:
        for scene_id in scenes:
            for width, height in resolutions:
                for batch_size in batches:
                    result = run_gsplat_case(args, scene_id, width, height, batch_size)
                    results.append(result)
                    write_result_json(result, args.output / f"{result['run_id']}.json")

    unsupported = implementations - {"gsplat"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Unsupported implementation(s): {names}. "
            "Use the dedicated OVRTX runner for the one-step multi-RenderProduct baseline."
        )

    results_stem = rasterize_mode_qualified(
        "baseline-results",
        args.rasterize_mode,
    )
    write_results_csv(results, args.output / f"{results_stem}.csv")
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "rasterize_mode": args.rasterize_mode,
        "eps2d": args.eps2d,
        **gsplat_output_contract(),
        "result_count": len(results),
        "results": [result["run_id"] for result in results],
    }
    manifest_stem = rasterize_mode_qualified(
        "run-manifest",
        args.rasterize_mode,
    )
    (args.output / f"{manifest_stem}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
