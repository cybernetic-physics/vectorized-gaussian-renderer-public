#!/usr/bin/env python3
"""Render selected matched cameras with the pinned gsplat correctness oracle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.benchmark_manifest import (  # noqa: E402
    file_sha256,
    sha256_json,
)
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    GSPLAT_BOUNDARY_PROBE_EXPECTED_RADII,
    GSPLAT_BUILD_ATTESTATION_SCHEMA,
    GSPLAT_ORACLE_SCHEMA,
    GSPLAT_SUPPORT_PROBE_EXPECTED_RADII,
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PINNED_GSPLAT_COMMIT,
    PINNED_GSPLAT_PATCHED_UTILS_SHA256,
    PINNED_GSPLAT_PATCH_SHA256,
    active_cuda_device_uuid,
    artifact_record,
    load_verified_gsplat_build_attestation,
    load_verified_source_manifest,
    primary_fidelity_selection,
    same_artifact,
)
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import load_trajectory  # noqa: E402
from isaacsim_gaussian_renderer.evaluation.matched_semantics import (  # noqa: E402
    REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    SEMANTIC_TOPOLOGIES,
    matched_semantic_ids,
)
from benchmarks.flashgs_matched_occupancy import (  # noqa: E402
    ComputeProcessSampler,
    capture_node_snapshot,
    ensure_cooperative_executor_lock,
    occupancy_failures,
)
from isaacsim_gaussian_renderer.fidelity.camera_bundle import (  # noqa: E402
    bundle_from_tensors,
    write_camera_bundle,
)
from isaacsim_gaussian_renderer.ply_loader import (  # noqa: E402
    canonicalize_3dgs_scene,
    load_ply_to_gaussians,
)

HOME_SCAN_COUNT = 21_497_908
HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--primary-fidelity-suite", action="store_true")
    parser.add_argument("--gsplat-source", type=Path, default=Path("/workspace/src/gsplat"))
    parser.add_argument(
        "--semantic-topology",
        choices=SEMANTIC_TOPOLOGIES,
        default=REPRESENTATIVE_SEMANTIC_TOPOLOGY,
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--gsplat-build-attestation",
        type=Path,
        required=True,
    )
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def current_process_gpu_uuid() -> str | None:
    property_uuid = active_cuda_device_uuid(torch.cuda)
    if property_uuid:
        return property_uuid
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[1] == str(os.getpid()):
            return fields[0]
    return None


def verify_gsplat_compatibility_patch(source: Path) -> dict[str, str]:
    """Require only the repository's recorded CUDA event API correction."""
    patch = PROJECT_ROOT / "patches/gsplat-cuda-event-flags.patch"
    patch_sha256 = file_sha256(patch)
    if patch_sha256 != PINNED_GSPLAT_PATCH_SHA256:
        raise RuntimeError(
            "gsplat compatibility patch checksum mismatch: "
            f"{patch_sha256} != {PINNED_GSPLAT_PATCH_SHA256}."
        )
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--short", "--untracked-files=all"],
        text=True,
    ).strip()
    if status != "M gsplat/cuda/csrc/Utils.cpp":
        raise RuntimeError(
            "Pinned gsplat must contain only the recorded CUDA event API patch; "
            f"status was {status!r}."
        )
    reverse_check = subprocess.run(
        ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if reverse_check.returncode:
        raise RuntimeError(
            "Pinned gsplat does not contain the expected compatibility patch: "
            + reverse_check.stdout.strip()
        )
    patched_utils = source / "gsplat" / "cuda" / "csrc" / "Utils.cpp"
    patched_utils_sha256 = file_sha256(patched_utils)
    if patched_utils_sha256 != PINNED_GSPLAT_PATCHED_UTILS_SHA256:
        raise RuntimeError(
            "Pinned gsplat patched Utils.cpp checksum mismatch: "
            f"{patched_utils_sha256} != "
            f"{PINNED_GSPLAT_PATCHED_UTILS_SHA256}."
        )
    return {
        "path": str(patch),
        "sha256": patch_sha256,
        "patched_utils_sha256": patched_utils_sha256,
        "scope": (
            "Two cudaEventCreateWithFlags API-name corrections in Utils.cpp; "
            "no image-formation or rasterization change."
        ),
    }


def verify_gsplat_support_contract(source: Path) -> dict[str, object]:
    """Bind the oracle to the pinned gsplat Gaussian support macro."""

    header = source / "gsplat" / "cuda" / "include" / "Common.h"
    contents = header.read_text(encoding="utf-8")
    match = re.search(
        r"^\s*#define\s+GAUSSIAN_EXTEND\s+([0-9]+(?:\.[0-9]+)?)f\s*$",
        contents,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("Pinned gsplat GAUSSIAN_EXTEND macro is unavailable.")
    value = float(match.group(1))
    if value != MATCHED_GAUSSIAN_SUPPORT_SIGMA:
        raise RuntimeError(
            "Pinned gsplat Gaussian support differs from the matched contract: "
            f"{value} != {MATCHED_GAUSSIAN_SUPPORT_SIGMA}."
        )
    return {
        "macro": "GAUSSIAN_EXTEND",
        "value": value,
        "header_artifact": artifact_record(header),
    }


def gsplat_source_input_records(source: Path) -> dict[str, dict[str, object]]:
    """Record every pinned gsplat CUDA/JIT input, including vendored headers."""

    roots = [
        source / "gsplat" / "cuda",
        source / "gsplat" / "_lazy_backend.py",
    ]
    paths: set[Path] = set()
    for root in roots:
        if root.is_file():
            paths.add(root)
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            ):
                paths.add(path)
    if not paths:
        raise RuntimeError("Pinned gsplat CUDA build inputs are unavailable.")
    return {
        str(path.relative_to(source)): artifact_record(path)
        for path in sorted(paths)
    }


def expected_gsplat_build_directory() -> Path:
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not root:
        raise RuntimeError(
            "TORCH_EXTENSIONS_DIR must name a fresh, dedicated build root for "
            "the matched gsplat oracle."
        )
    return (Path(root).resolve() / "gsplat_cuda").resolve()


def preflight_gsplat_build_attestation(
    *,
    source: Path,
    attestation_path: Path,
) -> tuple[Path, dict[str, object] | None]:
    """Reject AOT/stale binaries before importing gsplat's native backend."""

    build_directory = expected_gsplat_build_directory()
    source_local_extensions = sorted(
        path
        for pattern in ("csrc*.so", "csrc*.pyd")
        for path in (source / "gsplat").glob(pattern)
    )
    if source_local_extensions:
        raise RuntimeError(
            "Pinned gsplat contains a source-local/AOT native extension; the "
            "matched oracle requires a fresh JIT build: "
            + ", ".join(str(path) for path in source_local_extensions)
        )
    if attestation_path.is_file():
        attestation = load_verified_gsplat_build_attestation(
            attestation_path,
            gsplat_source=source,
        )
        if Path(str(attestation.get("build_directory", ""))).resolve() != (
            build_directory
        ):
            raise RuntimeError(
                "TORCH_EXTENSIONS_DIR differs from the recorded gsplat build "
                "attestation."
            )
        return build_directory, attestation
    if build_directory.exists():
        raise RuntimeError(
            "The gsplat JIT build directory pre-existed without an attestation; "
            "use a new empty TORCH_EXTENSIONS_DIR: "
            f"{build_directory}."
        )
    return build_directory, None


def gsplat_behavior_probe(fully_fused_projection: object) -> dict[str, object]:
    """Prove that the loaded binary implements 3.33/opacity and edge rules."""

    device = torch.device("cuda")
    viewmats = torch.eye(4, device=device, dtype=torch.float32)[None]
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )
    means = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    scales = torch.full((2, 3), 0.1, device=device, dtype=torch.float32)
    opacities = torch.tensor([1.0, 0.1], device=device, dtype=torch.float32)
    radii = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        intrinsics,
        128,
        128,
        eps2d=0.0,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=0.0,
        packed=False,
        opacities=opacities,
    )[0]
    boundary_radii = fully_fused_projection(
        means[:1],
        None,
        quats[:1],
        scales[:1],
        viewmats,
        intrinsics,
        128,
        128,
        eps2d=0.0,
        near_plane=1.0,
        far_plane=1.0,
        radius_clip=0.0,
        packed=False,
        opacities=opacities[:1],
    )[0]
    torch.cuda.synchronize()
    observed = radii.reshape(-1, 2).detach().cpu().tolist()
    boundary_observed = boundary_radii.reshape(-1, 2).detach().cpu().tolist()
    expected_from_equation = [
        [
            math.ceil(
                min(
                    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
                    math.sqrt(2.0 * math.log(opacity / (1.0 / 255.0))),
                )
                * 10.0
            )
        ]
        * 2
        for opacity in (1.0, 0.1)
    ]
    passed = bool(
        expected_from_equation == GSPLAT_SUPPORT_PROBE_EXPECTED_RADII
        and observed == GSPLAT_SUPPORT_PROBE_EXPECTED_RADII
        and boundary_observed == GSPLAT_BOUNDARY_PROBE_EXPECTED_RADII
    )
    result = {
        "pass": passed,
        "support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        "alpha_threshold": 1.0 / 255.0,
        "opacities": [1.0, 0.1],
        "projected_standard_deviation_pixels": 10.0,
        "expected_radii": GSPLAT_SUPPORT_PROBE_EXPECTED_RADII,
        "observed_radii": observed,
        "boundary_near_plane": 1.0,
        "boundary_far_plane": 1.0,
        "boundary_camera_z": 1.0,
        "boundary_expected_radii": GSPLAT_BOUNDARY_PROBE_EXPECTED_RADII,
        "boundary_observed_radii": boundary_observed,
    }
    if not passed:
        raise RuntimeError(
            "Loaded gsplat native extension failed the matched support/boundary "
            f"probe: {result}."
        )
    return result


def attest_loaded_gsplat_build(
    *,
    source: Path,
    support_contract: dict[str, object],
    build_directory: Path,
    existing_attestation: dict[str, object] | None,
    attestation_path: Path,
    native_path: Path,
    behavior_probe: dict[str, object],
) -> dict[str, object]:
    """Create once, then strictly verify, the fresh gsplat JIT build."""

    if native_path.parent.resolve() != build_directory:
        raise RuntimeError(
            "Loaded gsplat native extension is not the dedicated JIT output: "
            f"{native_path}."
        )
    native_artifact = artifact_record(native_path)
    if existing_attestation is not None:
        if not same_artifact(
            existing_attestation.get("native_extension"),
            native_artifact,
        ):
            raise RuntimeError(
                "Loaded gsplat native extension differs from its attestation."
            )
        if existing_attestation.get("behavior_probe") != behavior_probe:
            raise RuntimeError("Loaded gsplat behavior probe changed.")
        return load_verified_gsplat_build_attestation(
            attestation_path,
            gsplat_source=source,
        )

    source_inputs = gsplat_source_input_records(source)
    normalized_inputs = {
        relative: {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for relative, record in source_inputs.items()
    }
    build_ninja = build_directory / "build.ninja"
    build_parameters = build_directory / "build_params.json"
    payload: dict[str, object] = {
        "schema_version": GSPLAT_BUILD_ATTESTATION_SCHEMA,
        "gsplat_commit": PINNED_GSPLAT_COMMIT,
        "gsplat_source_root": str(source),
        "source_inputs": source_inputs,
        "source_input_tree_sha256": sha256_json(normalized_inputs),
        "support_contract": support_contract,
        "projection_contract": MATCHED_PROJECTION_RULES,
        "build_directory": str(build_directory),
        "build_ninja": artifact_record(build_ninja),
        "build_parameters": artifact_record(build_parameters),
        "native_extension": native_artifact,
        "build_environment": {
            name: os.environ.get(name)
            for name in (
                "TORCH_EXTENSIONS_DIR",
                "TORCH_CUDA_ARCH_LIST",
                "CUDA_HOME",
                "DEBUG",
                "FAST_MATH",
                "WITH_SYMBOLS",
                "NVCC_FLAGS",
                "MAX_JOBS",
                "BUILD_3DGS",
                "NUM_CHANNELS",
            )
        },
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "behavior_probe": behavior_probe,
        "fresh_build_directory_required": True,
    }
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    attestation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return load_verified_gsplat_build_attestation(
        attestation_path,
        gsplat_source=source,
    )


def semantic_from_top_contributor(
    contributor_ids: torch.Tensor,
    alpha: torch.Tensor,
    semantic_ids: torch.Tensor,
) -> torch.Tensor:
    leading_id = contributor_ids[..., 0]
    output = torch.full(
        (*leading_id.shape, 1),
        -1,
        device=leading_id.device,
        dtype=torch.int64,
    )
    valid = (leading_id >= 0) & (alpha[..., 0] >= 0.01)
    output[..., 0][valid] = semantic_ids[leading_id[valid]]
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    executor_lock = ensure_cooperative_executor_lock()
    occupancy_path = args.output.with_suffix(".node-occupancy.json")
    occupancy_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_snapshot = capture_node_snapshot()
    preflight_failures = occupancy_failures(
        preflight_snapshot,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=False,
    )
    occupancy_payload = {
        "schema_version": "flashgs-matched-node-occupancy-v2",
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "executor_control": {
            "cooperative_node_wide_lock": executor_lock,
            "scope": "all-visible-gpus",
            "limitation": (
                "The lock coordinates participating repository runners; "
                "periodic NVIDIA process samples detect but cannot make a "
                "continuous absence claim about uncooperative processes."
            ),
        },
        "preflight": preflight_snapshot,
        "preflight_failures": preflight_failures,
        "sampled_compute_process_telemetry": None,
        "postflight": None,
        "postflight_failures": None,
        "pass": False,
    }
    occupancy_path.write_text(
        json.dumps(occupancy_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if preflight_failures:
        raise RuntimeError(
            "Shared RTX node occupancy preflight failed: "
            + "; ".join(preflight_failures)
        )
    occupancy_sampler = ComputeProcessSampler(
        expected_gpu_uuid=args.expected_gpu_uuid,
        allowed_pids={os.getpid()},
    ).start()
    source_provenance = load_verified_source_manifest(
        args.source_manifest,
        project_root=PROJECT_ROOT,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("The gsplat oracle requires CUDA.")
    torch.cuda.init()
    gpu_uuid = current_process_gpu_uuid()
    if gpu_uuid != args.expected_gpu_uuid:
        raise RuntimeError(
            "Oracle GPU UUID differs from the explicit contract: "
            f"{gpu_uuid!r} != {args.expected_gpu_uuid!r}."
        )
    if args.max_views <= 0:
        raise ValueError("max_views must be positive.")
    commit = git_commit(args.gsplat_source)
    if commit != PINNED_GSPLAT_COMMIT:
        raise RuntimeError(
            f"gsplat commit mismatch: {commit} != {PINNED_GSPLAT_COMMIT}."
        )
    gsplat_patch = verify_gsplat_compatibility_patch(args.gsplat_source)
    gsplat_support_contract = verify_gsplat_support_contract(
        args.gsplat_source
    )
    gsplat_source = args.gsplat_source.resolve()
    sys.path = [entry for entry in sys.path if entry != str(gsplat_source)]
    sys.path.insert(0, str(gsplat_source))
    gsplat_spec = importlib.util.find_spec("gsplat")
    if gsplat_spec is None or gsplat_spec.origin is None:
        raise RuntimeError("Pinned gsplat package cannot be resolved.")
    if not Path(gsplat_spec.origin).resolve().is_relative_to(gsplat_source):
        raise RuntimeError(
            "Resolved gsplat package is outside --gsplat-source: "
            f"{gsplat_spec.origin}."
        )
    attestation_path = args.gsplat_build_attestation.resolve()
    build_directory, existing_attestation = preflight_gsplat_build_attestation(
        source=gsplat_source,
        attestation_path=attestation_path,
    )
    from gsplat.cuda._backend import _C as gsplat_native
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        rasterize_top_contributing_gaussian_ids,
    )
    from gsplat.rendering import rasterization
    import gsplat

    gsplat_python_artifact = artifact_record(Path(gsplat.__file__))
    gsplat_native_path_value = getattr(gsplat_native, "__file__", None)
    if not gsplat_native_path_value:
        raise RuntimeError("Loaded gsplat CUDA extension has no file identity.")
    gsplat_native_path = Path(gsplat_native_path_value).resolve()
    behavior_probe = gsplat_behavior_probe(fully_fused_projection)
    build_attestation = attest_loaded_gsplat_build(
        source=gsplat_source,
        support_contract=gsplat_support_contract,
        build_directory=build_directory,
        existing_attestation=existing_attestation,
        attestation_path=attestation_path,
        native_path=gsplat_native_path,
        behavior_probe=behavior_probe,
    )
    gsplat_native_artifact = artifact_record(gsplat_native_path)
    if not same_artifact(
        build_attestation.get("native_extension"),
        gsplat_native_artifact,
    ):
        raise RuntimeError("Loaded gsplat binary differs from its attestation.")
    trajectory = load_trajectory(args.trajectory)
    if args.primary_fidelity_suite:
        selection = primary_fidelity_selection(
            trajectory.batch,
            trajectory_timesteps=trajectory.timesteps,
        )
    else:
        step = args.step if args.step >= 0 else trajectory.timesteps + args.step
        if step < 0 or step >= trajectory.timesteps:
            raise ValueError(f"step {args.step} is outside the trajectory.")
        view_count = min(args.max_views, trajectory.batch)
        camera_indices = np.unique(
            np.linspace(0, trajectory.batch - 1, view_count, dtype=np.int64)
        )
        selection = tuple((step, int(index)) for index in camera_indices)
    steps = np.asarray([step for step, _index in selection], dtype=np.int64)
    camera_indices = np.asarray(
        [index for _step, index in selection], dtype=np.int64
    )
    if file_sha256(args.scene_path) != HOME_SCAN_SHA256:
        raise ValueError("Home Scan SHA-256 mismatch.")
    raw = load_ply_to_gaussians(args.scene_path)
    if raw.count != HOME_SCAN_COUNT:
        raise ValueError(f"Home Scan count mismatch: {raw.count}.")
    canonical = canonicalize_3dgs_scene(raw, device="cuda")
    means = canonical.means
    rotations = canonical.rotations
    scales = canonical.scales
    opacities = canonical.opacities
    colors = canonical.features[:, :3].contiguous()
    semantic_ids = matched_semantic_ids(
        means,
        args.semantic_topology,
    )
    # The production equation uses canonical degree-zero RGB only. Release the
    # unused higher-order SH tensor before gsplat allocates its O(G) projection
    # metadata on the 24 GB headline GPU.
    del canonical, raw
    torch.cuda.empty_cache()
    rgb_values: list[torch.Tensor] = []
    alpha_values: list[torch.Tensor] = []
    depth_values: list[torch.Tensor] = []
    semantic_values: list[torch.Tensor] = []
    started = time.perf_counter()
    for step, camera_index in selection:
        viewmat = torch.from_numpy(
            trajectory.viewmats[step, camera_index : camera_index + 1]
        ).to("cuda")
        intrinsic = torch.from_numpy(
            trajectory.intrinsics[step, camera_index : camera_index + 1]
        ).to("cuda")
        rgbd, alpha, metadata = rasterization(
            means,
            rotations,
            scales,
            opacities,
            colors,
            viewmat,
            intrinsic,
            trajectory.width,
            trajectory.height,
            near_plane=trajectory.near_plane,
            far_plane=trajectory.far_plane,
            eps2d=0.0,
            radius_clip=0.0,
            packed=False,
            tile_size=16,
            render_mode="RGB+ED",
            rasterize_mode="classic",
            global_z_order=True,
        )
        contributor_ids, _ = rasterize_top_contributing_gaussian_ids(
            metadata["means2d"],
            metadata["conics"],
            metadata["opacities"],
            metadata["isect_offsets"],
            metadata["flatten_ids"],
            trajectory.width,
            trajectory.height,
            metadata["tile_size"],
            1,
        )
        semantic = semantic_from_top_contributor(
            contributor_ids, alpha, semantic_ids
        )
        expected_depth = torch.where(
            alpha > 1.0e-8,
            rgbd[..., 3:4],
            torch.full_like(rgbd[..., 3:4], float("inf")),
        )
        rgb_values.append(rgbd[..., :3].contiguous())
        depth_values.append(expected_depth.contiguous())
        alpha_values.append(alpha.contiguous())
        semantic_values.append(semantic.contiguous())
    torch.cuda.synchronize()
    outputs = {
        "rgb": torch.cat(rgb_values, dim=0),
        "alpha": torch.cat(alpha_values, dim=0),
        "depth": torch.cat(depth_values, dim=0),
        "semantic": torch.cat(semantic_values, dim=0),
    }
    bundle = bundle_from_tensors(
        viewmats=trajectory.viewmats[steps, camera_indices],
        intrinsics=trajectory.intrinsics[steps, camera_indices],
        scene_ids=np.full((camera_indices.size,), 404, dtype=np.int64),
        width=trajectory.width,
        height=trajectory.height,
        background=(0.0, 0.0, 0.0),
        color_space="linear_rgb",
        scene_checksum=HOME_SCAN_SHA256,
        view_ids=[
            f"t{step:03d}-b{camera_index:04d}"
            for step, camera_index in selection
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **{name: tensor.detach().cpu().numpy() for name, tensor in outputs.items()},
        valid_depth=(outputs["alpha"].detach().cpu().numpy() > 0.01),
        color_space=np.asarray("linear_rgb"),
        background=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        camera_bundle_id=np.asarray(bundle.bundle_id),
        camera_indices=camera_indices,
        steps=steps,
        trajectory_id=np.asarray(trajectory.trajectory_id),
        semantic_topology=np.asarray(args.semantic_topology),
        gaussian_support_sigma=np.asarray(
            MATCHED_GAUSSIAN_SUPPORT_SIGMA, dtype=np.float32
        ),
    )
    camera_bundle_path = args.output.with_suffix(".camera-bundle.json")
    write_camera_bundle(bundle, camera_bundle_path)
    sampled_compute_process_telemetry = occupancy_sampler.stop()
    postflight_snapshot = capture_node_snapshot()
    postflight_failures = occupancy_failures(
        postflight_snapshot,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=True,
    )
    occupancy_payload.update(
        {
            "postflight": postflight_snapshot,
            "postflight_failures": postflight_failures,
            "sampled_compute_process_telemetry": (
                sampled_compute_process_telemetry
            ),
            "pass": bool(
                not postflight_failures
                and sampled_compute_process_telemetry["pass"]
                and executor_lock["pass"]
            ),
        }
    )
    occupancy_path.write_text(
        json.dumps(occupancy_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not occupancy_payload["pass"]:
        raise RuntimeError(
            "Shared RTX node occupancy evidence failed: "
            + "; ".join(
                [
                    *postflight_failures,
                    *sampled_compute_process_telemetry["failures"],
                ]
            )
        )
    manifest = {
        "schema_version": GSPLAT_ORACLE_SCHEMA,
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "camera_bundle": str(camera_bundle_path),
        "camera_bundle_artifact": artifact_record(camera_bundle_path),
        "camera_bundle_id": bundle.bundle_id,
        "camera_indices": camera_indices.tolist(),
        "steps": steps.tolist(),
        "selection_pairs": [list(pair) for pair in selection],
        "selection_profile": (
            "primary-fidelity-suite"
            if args.primary_fidelity_suite
            else "diagnostic-single-step"
        ),
        "trajectory_id": trajectory.trajectory_id,
        "scene_sha256": HOME_SCAN_SHA256,
        "gaussian_count": HOME_SCAN_COUNT,
        "semantic_topology": args.semantic_topology,
        "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        "gsplat_commit": commit,
        "gsplat_compatibility_patch": gsplat_patch,
        "gsplat_support_contract": gsplat_support_contract,
        "projection_contract": MATCHED_PROJECTION_RULES,
        "gsplat_build_attestation": artifact_record(attestation_path),
        "gsplat_behavior_probe": behavior_probe,
        "gsplat_python_artifact": gsplat_python_artifact,
        "gsplat_native_extension": gsplat_native_artifact,
        "node_occupancy": artifact_record(occupancy_path),
        "source_provenance": source_provenance,
        "render_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": gpu_uuid,
        "driver": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                gpu_uuid,
            ],
            text=True,
        ).strip(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "pass": True,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "FLASHGS_GSPLAT_ORACLE_OK "
        + json.dumps({"manifest": str(manifest_path)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
