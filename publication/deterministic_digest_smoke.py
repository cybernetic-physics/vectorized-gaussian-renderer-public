#!/usr/bin/env python3
"""Render the deterministic CUDA fixture and emit content-addressed outputs.

This is the worker used by the publication cross-process replay gate.  It
keeps the frozen renderer checkout read-only while adding the output digests
that the older in-process smoke intentionally did not expose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import camera_bundle


SCHEMA_VERSION = "publication-deterministic-digest-smoke-v1"
OUTPUTS = ("rgb", "alpha", "depth", "semantic_id")
PRODUCTION_ARGUMENTS = {
    "batch": 4,
    "gaussians": 1_024,
    "height": 64,
    "iterations": 32,
    "ray_gaussian_evaluation": False,
    "tile_size": 1,
    "width": 64,
}
CLAIM_SCOPE = {
    **PRODUCTION_ARGUMENTS,
    "camera_configuration": "identical-centered-cameras",
    "claim": "synthetic-equal-depth-deterministic-mode-replay-only",
    "gaussian_configuration": "isotropic-identity-rotation-equal-depth",
    "renderer_mode": "deterministic",
}
COMPARISON_METHOD = "sha256(dtype,shape,raw-c-order-bytes)"
NATIVE_BUILD_FIELDS = {
    "build_directory",
    "cuda_flags",
    "cxx_flags",
    "module_name",
    "sources",
    "torch_cuda_arch_list",
}
EXPECTED_CXX_FLAGS = ["-O3", "-std=c++17"]
EXPECTED_CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-std=c++17",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-smoke", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--gaussians", type=int, default=1_024)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=1)
    parser.add_argument("--ray-gaussian-evaluation", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path_value: Path) -> dict[str, Any]:
    if path_value.is_symlink():
        raise ValueError(f"Reference artifact may not be a symlink: {path_value}")
    path = path_value.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Reference artifact is not a regular file: {path}")
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": file_sha256(path)}


def arguments_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch": args.batch,
        "gaussians": args.gaussians,
        "height": args.height,
        "iterations": args.iterations,
        "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
        "tile_size": args.tile_size,
        "width": args.width,
    }


def loaded_native_evidence(backend: CustomCudaBackend) -> tuple[dict[str, Any], dict[str, Any]]:
    native = getattr(backend, "_native", None)
    native_path = getattr(native, "__file__", None)
    raw_contract = getattr(native, "__vgr_build_contract__", None)
    if not isinstance(native_path, str) or not native_path:
        raise RuntimeError("Loaded native extension has no module file")
    if not isinstance(raw_contract, dict) or set(raw_contract) != NATIVE_BUILD_FIELDS:
        raise RuntimeError("Loaded native extension has no exact build contract")
    contract = json.loads(json.dumps(raw_contract, sort_keys=True))
    if (
        not isinstance(contract.get("module_name"), str)
        or not contract["module_name"].startswith("isaacsim_gaussian_renderer_cuda_")
        or contract.get("cxx_flags") != EXPECTED_CXX_FLAGS
        or contract.get("cuda_flags") != EXPECTED_CUDA_FLAGS
        or not isinstance(contract.get("torch_cuda_arch_list"), str)
        or not contract["torch_cuda_arch_list"]
        or not isinstance(contract.get("sources"), list)
        or [Path(value).name for value in contract["sources"]]
        != ["renderer.cpp", "renderer_cuda.cu"]
        or not all(isinstance(value, str) and value for value in contract["sources"])
        or not isinstance(contract.get("build_directory"), str)
        or not contract["build_directory"]
    ):
        raise RuntimeError("Loaded native extension build contract is malformed")
    for source in contract["sources"]:
        artifact_record(Path(source))
    native_record = artifact_record(Path(native_path))
    if Path(native_record["path"]).parent != Path(contract["build_directory"]).resolve(
        strict=True
    ):
        raise RuntimeError("Loaded native extension is outside its recorded build directory")
    if not Path(native_record["path"]).name.startswith(contract["module_name"]):
        raise RuntimeError("Loaded native extension filename differs from its module name")
    return native_record, contract


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    contiguous = tensor.detach().contiguous().cpu()
    payload = contiguous.numpy().tobytes(order="C")
    contract = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(contract + b"\0" + payload).hexdigest()
    return {
        "bytes": len(payload),
        "dtype": str(contiguous.dtype),
        "sha256": digest,
        "shape": list(contiguous.shape),
    }


def output_contract(
    outputs: dict[str, torch.Tensor],
    *,
    batch: int,
    height: int,
    width: int,
    semantic_min_alpha: float,
) -> dict[str, Any]:
    expected = {
        "rgb": ((batch, height, width, 3), torch.float32),
        "alpha": ((batch, height, width, 1), torch.float32),
        "depth": ((batch, height, width, 1), torch.float32),
        "semantic_id": ((batch, height, width, 1), torch.int64),
    }
    output_names_match = set(outputs) == set(expected)
    if not output_names_match:
        raise AssertionError(
            f"Renderer output set differs: expected={sorted(expected)}, actual={sorted(outputs)}"
        )
    shapes_match = all(tuple(outputs[name].shape) == shape for name, (shape, _) in expected.items())
    dtypes_match = all(outputs[name].dtype == dtype for name, (_, dtype) in expected.items())
    contiguous = all(outputs[name].is_contiguous() for name in OUTPUTS)
    cuda_resident = all(outputs[name].is_cuda for name in OUTPUTS)
    output_devices = {name: str(outputs[name].device) for name in OUTPUTS}
    single_cuda_device = len(set(output_devices.values())) == 1 and cuda_resident

    alpha = outputs["alpha"][..., 0]
    depth = outputs["depth"][..., 0]
    semantic = outputs["semantic_id"][..., 0]
    foreground = alpha > 0.0
    background = ~foreground
    semantic_foreground = alpha >= semantic_min_alpha
    semantic_background = ~semantic_foreground
    foreground_pixels = int(torch.count_nonzero(foreground).item())
    background_pixels = int(torch.count_nonzero(background).item())
    semantic_foreground_pixels = int(torch.count_nonzero(semantic_foreground).item())
    semantic_background_pixels = int(torch.count_nonzero(semantic_background).item())

    finite_rgb = bool(torch.isfinite(outputs["rgb"]).all().item())
    finite_alpha = bool(torch.isfinite(alpha).all().item())
    alpha_in_range = bool(((alpha >= 0.0) & (alpha <= 1.0)).all().item())
    finite_foreground_depth = bool(
        foreground_pixels > 0 and torch.isfinite(depth[foreground]).all().item()
    )
    valid_background_depth = bool(
        (
            torch.isfinite(depth[background])
            | (torch.isinf(depth[background]) & (depth[background] > 0.0))
        )
        .all()
        .item()
    )
    valid_foreground_semantics = bool(
        semantic_foreground_pixels > 0
        and (semantic[semantic_foreground] >= 0).all().item()
    )
    valid_background_semantics = bool(
        (semantic[semantic_background] == -1).all().item()
    )
    valid = bool(
        output_names_match
        and shapes_match
        and dtypes_match
        and contiguous
        and cuda_resident
        and single_cuda_device
        and finite_rgb
        and finite_alpha
        and alpha_in_range
        and finite_foreground_depth
        and valid_background_depth
        and valid_foreground_semantics
        and valid_background_semantics
    )
    return {
        "alpha_in_range": alpha_in_range,
        "background_pixel_count": background_pixels,
        "contiguous": contiguous,
        "cuda_resident": cuda_resident,
        "finite_alpha": finite_alpha,
        "finite_foreground_depth": finite_foreground_depth,
        "finite_rgb": finite_rgb,
        "foreground_pixel_count": foreground_pixels,
        "output_devices": output_devices,
        "output_names_match": output_names_match,
        "semantic_background_pixel_count": semantic_background_pixels,
        "semantic_foreground_pixel_count": semantic_foreground_pixels,
        "shapes_match": shapes_match,
        "single_cuda_device": single_cuda_device,
        "dtypes_match": dtypes_match,
        "valid": valid,
        "valid_background_depth": valid_background_depth,
        "valid_background_semantics": valid_background_semantics,
        "valid_foreground_semantics": valid_foreground_semantics,
    }


def fixture_sha256(tensors: dict[str, torch.Tensor], arguments: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(arguments, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    for name in sorted(tensors):
        record = tensor_record(tensors[name])
        digest.update(name.encode("utf-8"))
        digest.update(record["sha256"].encode("ascii"))
    return digest.hexdigest()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    arguments = arguments_record(args)
    if arguments != PRODUCTION_ARGUMENTS:
        raise ValueError(
            "Publication deterministic replay uses one fixed scope: "
            + json.dumps(PRODUCTION_ARGUMENTS, sort_keys=True)
        )

    source_smoke = artifact_record(args.source_smoke)
    device = torch.device("cuda")
    count = args.gaussians
    means = torch.zeros((count, 3), device=device, dtype=torch.float32)
    means[:, 2] = 5.0
    scales = torch.full((count, 3), 0.03, device=device, dtype=torch.float32)
    rotations = torch.zeros((count, 4), device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.full((count,), 0.2, device=device, dtype=torch.float32)
    features = torch.zeros((count, 3), device=device, dtype=torch.float32)
    features[:, 0] = torch.linspace(1.0, 0.0, count, device=device)
    features[:, 2] = torch.linspace(0.0, 1.0, count, device=device)
    semantic_ids = torch.arange(1_000, 1_000 + count, device=device, dtype=torch.int64)

    one_camera = camera_bundle(1, args.width, args.height, device=device)
    centered_viewmat = torch.eye(4, device=device, dtype=torch.float32)
    centered_viewmat[2, 3] = -4.0
    viewmats = centered_viewmat.repeat(args.batch, 1, 1).contiguous()
    intrinsics = one_camera.intrinsics.repeat(args.batch, 1, 1).contiguous()
    scene_ids = torch.full((args.batch,), 73, device=device, dtype=torch.int64)
    fixture = fixture_sha256(
        {
            "features": features,
            "intrinsics": intrinsics,
            "means": means,
            "opacities": opacities,
            "rotations": rotations,
            "scales": scales,
            "scene_ids": scene_ids,
            "semantic_ids": semantic_ids,
            "viewmats": viewmats,
        },
        arguments,
    )

    max_intersections = args.batch * count * (256 if args.tile_size == 1 else 64)
    backend = CustomCudaBackend(
        max_visible_records=args.batch * count,
        max_intersections=max_intersections,
        gaussian_support_sigma=3.0,
        covariance_epsilon=0.0 if args.ray_gaussian_evaluation else 0.3,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        tile_size=args.tile_size,
        output_srgb=False,
        deterministic=True,
    )
    service = RendererService(backend, height=args.height, width=args.width, max_views=args.batch)
    tight_service: RendererService | None = None
    try:
        service.initialize(stage=None, device=device)
        native_extension, native_build_contract = loaded_native_evidence(backend)
        service.load_scene(
            73,
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            features=features,
            semantic_ids=semantic_ids,
        )
        outputs = service.render(viewmats, intrinsics, scene_ids)
        service.synchronize()
        contract = output_contract(
            outputs,
            batch=args.batch,
            height=args.height,
            width=args.width,
            semantic_min_alpha=backend.semantic_min_alpha,
        )
        if not contract["valid"]:
            raise AssertionError(f"Deterministic output contract failed: {contract}")
        reference = {name: tensor.clone() for name, tensor in outputs.items()}
        for _ in range(args.iterations - 1):
            service.render(viewmats, intrinsics, scene_ids, outputs=outputs)
        service.synchronize()
        bitwise_equal = {name: bool(torch.equal(reference[name], outputs[name])) for name in OUTPUTS}
        if not all(bitwise_equal.values()):
            raise AssertionError(f"Repeated outputs were not bitwise equal: {bitwise_equal}")

        tiles_x = (args.width + args.tile_size - 1) // args.tile_size
        center_tile = ((args.height // 2) // args.tile_size) * tiles_x + ((args.width // 2) // args.tile_size)
        start = int(backend.workspace["tile_starts"][center_tile].item())
        end = int(backend.workspace["tile_ends"][center_tile].item())
        visible_indices = backend.workspace["values_out"][start:end].to(torch.int64)
        sorted_ids = backend.workspace["visible_gaussian_ids"].index_select(0, visible_indices)
        expected_ids = torch.arange(count, device=device, dtype=torch.int32)
        if not torch.equal(sorted_ids, expected_ids):
            raise AssertionError("Equal-depth records are not ordered by ascending global Gaussian ID")
        center_semantic = int(outputs["semantic_id"][0, args.height // 2, args.width // 2, 0].item())
        if center_semantic != 1_000:
            raise AssertionError(f"Expected semantic ID 1000 at the center; got {center_semantic}")

        counters = backend.check_capacity(synchronize=False)
        tight_backend = CustomCudaBackend(
            max_visible_records=counters["visible_gaussians"],
            max_intersections=counters["tile_intersections"],
            gaussian_support_sigma=3.0,
            covariance_epsilon=0.0 if args.ray_gaussian_evaluation else 0.3,
            ray_gaussian_evaluation=args.ray_gaussian_evaluation,
            tile_size=args.tile_size,
            output_srgb=False,
            deterministic=True,
        )
        tight_service = RendererService(
            tight_backend, height=args.height, width=args.width, max_views=args.batch
        )
        tight_service.initialize(stage=None, device=device)
        tight_native_extension, tight_native_build_contract = loaded_native_evidence(tight_backend)
        if (
            tight_native_extension != native_extension
            or tight_native_build_contract != native_build_contract
        ):
            raise AssertionError("Wide and exact-capacity services loaded different native code")
        tight_service.load_scene(
            73,
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            features=features,
            semantic_ids=semantic_ids,
        )
        tight_outputs = tight_service.render(viewmats, intrinsics, scene_ids)
        tight_service.synchronize()
        tight_counters = tight_backend.check_capacity(synchronize=False)
        capacity_equal = {
            name: bool(torch.equal(reference[name], tight_outputs[name])) for name in OUTPUTS
        }
        if not all(capacity_equal.values()) or tight_counters != counters:
            raise AssertionError("Exact-capacity and wide-capacity deterministic renders differ")
        if counters.get("intersection_overflow") != 0 or counters.get("visible_overflow") != 0:
            raise AssertionError(f"Deterministic replay overflowed: {counters}")

        result = {
            "arguments": arguments,
            "bitwise_equal": bitwise_equal,
            "capacity_invariant_bitwise_equal": capacity_equal,
            "center_semantic": center_semantic,
            "claim_scope": CLAIM_SCOPE,
            "comparison_method": COMPARISON_METHOD,
            "counters": counters,
            "equal_depth_gaussian_id_order": "ascending",
            "equal_depth_records_checked": int(sorted_ids.numel()),
            "fixture_sha256": fixture,
            "native_build_contract": native_build_contract,
            "native_extension": native_extension,
            "output_contract": contract,
            "output_digests": {name: tensor_record(reference[name]) for name in OUTPUTS},
            "pass": True,
            "renderer_configuration": {
                "adaptive_capacity": backend.adaptive_capacity,
                "covariance_epsilon": backend.covariance_epsilon,
                "deterministic": backend.deterministic,
                "fixed_capacity_sort": backend.fixed_capacity_sort,
                "gaussian_support_sigma": backend.gaussian_support_sigma,
                "output_srgb": backend.output_srgb,
                "ray_gaussian_evaluation": backend.ray_gaussian_evaluation,
                "semantic_min_alpha": backend.semantic_min_alpha,
                "tile_size": backend.tile_size,
            },
            "schema_version": SCHEMA_VERSION,
            "source_smoke": source_smoke,
        }
        print("DETERMINISTIC_DIGEST_SMOKE_OK " + json.dumps(result, sort_keys=True))
    finally:
        try:
            if tight_service is not None:
                tight_service.shutdown()
        finally:
            service.shutdown()


if __name__ == "__main__":
    main()
