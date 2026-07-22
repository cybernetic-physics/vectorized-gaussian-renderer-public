"""Compare a converged OVRTX PathTracing Gaussian reference to custom CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ovrtx
import torch

from benchmarks.run_ovrtx import (
    allocate_contract_outputs,
    build_stage,
    copy_mapped_outputs_to_contract,
    product_paths,
    semantic_lut_from_outputs,
    upload_scene,
)
from experiments.ovrtx_temporal_fidelity import as_render_output, save_render_output
from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle,
    synthetic_scene_manifest,
    synthetic_scene_tensors,
)
from isaacsim_gaussian_renderer.fidelity import (
    bundle_from_tensors,
    compare_render_outputs,
    write_camera_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-pixel", type=int, default=4096)
    parser.add_argument("--samples-per-iteration", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--custom-support-sigma", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if (
        args.samples_per_pixel <= 0
        or args.samples_per_iteration <= 0
        or args.custom_support_sigma <= 0
    ):
        raise ValueError("Path-tracing sample counts must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    gaussian_count = 10_000
    seed = 42
    scene = synthetic_scene_tensors(gaussian_count, seed=seed, device=device)
    scene_manifest = synthetic_scene_manifest("synthetic-small", gaussian_count, seed)
    cameras = camera_bundle(1, args.width, args.height, device=device)
    scene_ids = torch.tensor([101], device=device, dtype=torch.int64)
    fidelity_bundle = bundle_from_tensors(
        viewmats=cameras.viewmats,
        intrinsics=cameras.intrinsics,
        width=args.width,
        height=args.height,
        color_space="display_srgb",
        scene_ids=scene_ids,
        scene_checksum=scene_manifest["checksum_sha256"],
    )
    write_camera_bundle(fidelity_bundle, args.output / "camera-bundle.json")

    custom_backend = CustomCudaBackend(
        max_visible_records=650,
        max_intersections=2_600,
        gaussian_support_sigma=args.custom_support_sigma,
        output_srgb=False,
    )
    custom_service = RendererService(
        custom_backend,
        height=args.height,
        width=args.width,
        max_views=1,
    )
    custom_service.initialize(stage=None, device=device)
    custom_service.load_scene(
        101,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )
    custom_tensors = custom_service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
    )
    custom_service.synchronize()
    custom_backend.check_capacity(synchronize=False)
    reference = as_render_output(
        rgb=custom_tensors["rgb"].detach().cpu().numpy(),
        alpha=custom_tensors["alpha"][..., 0].detach().cpu().numpy(),
        depth=custom_tensors["depth"][..., 0].detach().cpu().numpy(),
        semantic=custom_tensors["semantic_id"][..., 0].detach().cpu().numpy(),
        camera_bundle_id=fidelity_bundle.bundle_id,
        source="custom-cuda-identity-display-srgb",
    )
    save_render_output(args.output / "reference-custom.npz", reference)
    custom_service.shutdown()
    torch.cuda.empty_cache()

    stage = build_stage(
        1,
        args.width,
        args.height,
        0.9,
        cameras.viewmats,
        "none",
    )
    stage = stage.replace(
        'token omni:rtx:rendermode = "RealTimePathTracing"',
        'token omni:rtx:rendermode = "PathTracing"\n'
        f"        int omni:rtx:pt:samplesPerPixel = {args.samples_per_pixel}\n"
        f"        int omni:rtx:pt:samplesPerIteration = {args.samples_per_iteration}\n"
        "        bool omni:rtx:pt:adaptiveSampling:enabled = 0\n"
        "        bool omni:rtx:pt:denoising:enabled = 0\n"
        "        bool omni:rtx:pt:denoising:optix:temporal = 0\n"
        "        bool omni:rtx:pt:fractionalCutoutOpacity = 1",
    )
    (args.output / "scene.usda").write_text(stage, encoding="utf-8")
    paths = product_paths(1)

    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            keep_system_alive=True,
            log_level="warning",
            active_cuda_gpus="0",
            use_vulkan=True,
        )
    )
    renderer.open_usd_from_string(stage)
    _upload_contract, upload_keepalive = upload_scene(renderer, scene)
    contract, workspace = allocate_contract_outputs(1, args.height, args.width)

    start = time.perf_counter()
    outputs = renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
    render_wall_seconds = time.perf_counter() - start
    _semantic_map, _semantic_lut_cpu, semantic_lut = semantic_lut_from_outputs(
        outputs,
        paths[0],
    )
    copy_mapped_outputs_to_contract(
        outputs,
        paths,
        contract,
        workspace,
        semantic_lut,
    )
    del outputs

    candidate = as_render_output(
        rgb=contract["rgb"].detach().cpu().numpy(),
        alpha=contract["alpha"][..., 0].detach().cpu().numpy(),
        depth=contract["depth"][..., 0].detach().cpu().numpy(),
        semantic=contract["semantic_id"][..., 0].detach().cpu().numpy(),
        camera_bundle_id=fidelity_bundle.bundle_id,
        source=f"ovrtx-pathtracing-{args.samples_per_pixel}spp",
    )
    save_render_output(args.output / "candidate-ovrtx-pathtraced.npz", candidate)
    report = compare_render_outputs(
        reference=reference,
        candidate=candidate,
        camera_bundle=fidelity_bundle,
        output_dir=args.output / "report",
        config_id=f"ovrtx-pathtracing-{args.samples_per_pixel}spp-vs-custom",
        require_lpips=True,
        max_artifact_views=1,
    )
    summary = {
        "schema_version": "ovrtx-pathtraced-fidelity/v1",
        "samples_per_pixel": args.samples_per_pixel,
        "samples_per_iteration": args.samples_per_iteration,
        "adaptive_sampling": False,
        "denoising": False,
        "temporal_denoising": False,
        "fractional_cutout_opacity": True,
        "render_wall_seconds": render_wall_seconds,
        "scene": scene_manifest,
        "camera_bundle_id": fidelity_bundle.bundle_id,
        "custom_gaussian_support_sigma": args.custom_support_sigma,
        "aggregate": report["aggregate"],
        "pass": report["pass"],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    del upload_keepalive
    del renderer


if __name__ == "__main__":
    main()
