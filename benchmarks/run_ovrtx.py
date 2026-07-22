#!/usr/bin/env python3
"""Measure the OVRTX Gaussian control with one batched host submission.

OVRTX currently returns valid Gaussian color for multiple single-camera
RenderProducts submitted in one ``Renderer.step()``. Its native tiled
single-RenderProduct path returns black color for the same scene, so this
runner labels the control as ``multi-product`` rather than ``tiled``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import ovrtx
import torch
from PIL import Image

from isaacsim_gaussian_renderer.benchmark_manifest import (
    compact_semantic_ids,
    CameraBundle,
    STRESS_SCENES,
    axis_aligned_exterior_camera_bundle,
    camera_bundle,
    deterministic_semantic_ids,
    deterministic_semantic_manifest,
    file_sha256,
    projection_activation_scene_manifest,
    projection_activation_scene_tensors,
    sha256_json,
    spatial_semantic_ids,
    spatial_semantic_manifest,
    stats,
    stress_scene_manifest,
    stress_scene_tensors,
    synthetic_scene_manifest,
    synthetic_scene_tensors,
    tensor_bytes,
)
from isaacsim_gaussian_renderer.ply_loader import (
    canonicalize_3dgs_scene,
    load_ply_to_gaussians,
)
from isaacsim_gaussian_renderer.quaternion import quaternion_wxyz_to_xyzw
from isaacsim_gaussian_renderer.camera_math import (
    opencv_viewmats_to_usd_camera_world_matrices,
)
from isaacsim_gaussian_renderer.benchmark_schema import (
    SCHEMA_VERSION,
    fabric_scene_delegate_record,
    write_json_schema,
    write_result_json,
    write_results_csv,
)
from isaacsim_gaussian_renderer.semantic_mapping import (
    BACKGROUND_SEMANTIC_ID,
    decode_semantic_id_map,
    remap_renderer_semantics,
    renderer_semantic_lut,
    semantic_group_counts,
    semantic_label_grouping,
    semantic_group_permutation,
)


SH_C0 = 0.28209479177387814
SEMANTIC_GROUP_COUNT = 1024
OVRTX_MAX_GAUSSIANS_PER_PRIM = (1 << 24) - 1
REQUIRED_OUTPUTS = ["rgb", "depth", "alpha", "semantic_id"]
MAPPED_AOVS = ["LdrColor", "DistanceToImagePlaneSD", "SemanticSegmentation"]
OVRTX_OUTPUT_CONTRACT = (
    "rgb_srgb_float32_0_1_from_ldrcolor;"
    "metric_distance_to_image_plane_float32;"
    "generated_coverage_alpha_float32_0_1;"
    "semantic_class_id_int64_background_minus_one;"
    "all_outputs_preallocated_cuda"
)
SYNTHETIC_SCENES = {
    "synthetic-small": (10_000, 42),
    "synthetic-medium": (100_000, 43),
    **STRESS_SCENES,
}
HOME_SCAN_SCENE = "home-scan-lod0"
HOME_SCAN_COUNT = 21_497_908
HOME_SCAN_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
PUBLIC_SCENE = "voxel51-train-30000"
PUBLIC_SCENE_COUNT = 1_074_761
PUBLIC_SCENE_SHA256 = "a4c1906ce0256f5cb2255aa078d18b468dc01d8604f6eda1aaf22b235147a7a1"
PROJECTION_ACTIVATION_SCENE = "projection-activation"
PROJECTION_MODES = ("perspective", "tangential")
SORTING_MODES = ("zDepth", "cameraDistance", "rayHitDistance")
PLY_SCENES = {
    HOME_SCAN_SCENE: {
        "count": HOME_SCAN_COUNT,
        "sha256": HOME_SCAN_SHA256,
        "manifest": "datasets/home-scan-lod0.manifest.json",
        "source_sh_degree": 0,
        "camera_bounds_quantiles": None,
    },
    PUBLIC_SCENE: {
        "count": PUBLIC_SCENE_COUNT,
        "sha256": PUBLIC_SCENE_SHA256,
        "manifest": "datasets/voxel51-train-30000.manifest.json",
        "source_sh_degree": 3,
        "camera_bounds_quantiles": (0.1, 0.9),
    },
}


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_commit() -> str:
    if os.environ.get("SOURCE_GIT_COMMIT"):
        return os.environ["SOURCE_GIT_COMMIT"]
    return command_output(["git", "rev-parse", "HEAD"]) or "unknown"


def implementation_commit() -> str:
    return command_output(["git", "-C", "/workspace/src/ovrtx", "rev-parse", "HEAD"]) or "unknown"


def environment() -> dict[str, Any]:
    driver = command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "driver_version": driver.splitlines()[0] if driver else None,
        "git_commit": git_commit(),
    }


def process_gpu_memory_bytes(pid: int) -> int | None:
    output = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if output:
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 2 and fields[0] == str(pid):
                return int(fields[1]) * 1024 * 1024
    pmon = command_output(["nvidia-smi", "pmon", "-s", "m", "-c", "1"])
    if not pmon:
        return None
    for line in pmon.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 4 and fields[1] == str(pid) and fields[3] != "-":
            return int(fields[3]) * 1024 * 1024
    return None


def total_gpu_memory_bytes() -> int | None:
    output = command_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if not output:
        return None
    return int(output.splitlines()[0].strip()) * 1024 * 1024


class GpuMemorySampler:
    def __init__(self, pid: int, interval_seconds: float = 0.05):
        self._pid = pid
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []
        self.total_samples: list[int] = []

    def _sample(self) -> None:
        while not self._stop.is_set():
            value = process_gpu_memory_bytes(self._pid)
            if value is not None:
                self.samples.append(value)
            total = total_gpu_memory_bytes()
            if total is not None:
                self.total_samples.append(total)
            self._stop.wait(self._interval_seconds)

    def __enter__(self) -> "GpuMemorySampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        value = process_gpu_memory_bytes(self._pid)
        if value is not None:
            self.samples.append(value)
        total = total_gpu_memory_bytes()
        if total is not None:
            self.total_samples.append(total)

    @property
    def peak_bytes(self) -> int | None:
        return max(self.samples) if self.samples else None

    @property
    def peak_total_bytes(self) -> int | None:
        return max(self.total_samples) if self.total_samples else None


def product_paths(batch_size: int) -> list[str]:
    return [f"/Render/GaussianCamera_{index}" for index in range(batch_size)]


def splat_paths(group_count: int = SEMANTIC_GROUP_COUNT) -> list[str]:
    return [f"/World/Splats_{index:04d}" for index in range(group_count)]


def require_token_attribute_readback(
    renderer: ovrtx.Renderer,
    *,
    attribute_name: str,
    prim_paths: list[str],
    expected: str,
) -> dict[str, Any]:
    """Read one composed token per prim and fail unless all match."""
    token_ids = np.asarray(
        np.from_dlpack(
            renderer.read_attribute(
                attribute_name=attribute_name,
                prim_paths=prim_paths,
            )
        )
    ).reshape(-1)
    if token_ids.size != len(prim_paths):
        raise RuntimeError(f"{attribute_name} readback returned {token_ids.size} tokens for {len(prim_paths)} prims.")
    path_dictionary = renderer._get_path_dict()
    observed = [path_dictionary.token_to_string(int(token_id)) for token_id in token_ids]
    unique_observed = sorted(set(observed))
    if unique_observed != [expected]:
        raise RuntimeError(f"{attribute_name} readback failed: expected {expected!r}, observed {unique_observed!r}.")
    return {
        "attribute": attribute_name,
        "requested": expected,
        "observed": unique_observed,
        "prim_count": len(prim_paths),
        "all_match": True,
    }


def splat_prim_semantic_ids(
    semantic_counts: list[int],
    *,
    max_gaussians_per_prim: int = OVRTX_MAX_GAUSSIANS_PER_PRIM,
) -> list[int]:
    if max_gaussians_per_prim <= 0:
        raise ValueError("max_gaussians_per_prim must be positive.")
    prim_semantic_ids: list[int] = []
    for semantic_id, count in enumerate(semantic_counts):
        if count <= 0:
            raise ValueError(f"Semantic group {semantic_id} must contain Gaussians.")
        shard_count = (count + max_gaussians_per_prim - 1) // max_gaussians_per_prim
        prim_semantic_ids.extend([semantic_id] * shard_count)
    return prim_semantic_ids


def particle_field_definition(
    prim_index: int,
    semantic_id: int | None = None,
    *,
    projection_mode: str = "perspective",
    sorting_mode: str = "zDepth",
) -> str:
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"Unsupported projection mode: {projection_mode}")
    if sorting_mode not in SORTING_MODES:
        raise ValueError(f"Unsupported sorting mode: {sorting_mode}")
    semantic_id = prim_index if semantic_id is None else semantic_id
    return f"""    def ParticleField3DGaussianSplat "Splats_{prim_index:04d}" (
        prepend apiSchemas = ["SemanticsAPI:class"]
    )
    {{
        string semantic:class:params:semanticData = "{semantic_id}"
        string semantic:class:params:semanticType = "class"
        point3f[] positions = [(0, 0, 5)]
        float3[] scales = [(0.01, 0.01, 0.01)]
        quatf[] orientations = [(1, 0, 0, 0)]
        float[] opacities = [0]
        uniform int radiance:sphericalHarmonicsDegree = 0
        float3[] radiance:sphericalHarmonicsCoefficients = [(0, 0, 0)] (
            elementSize = 1
            interpolation = "vertex"
        )
        float3[] extent = [(-32, -32, -32), (32, 32, 64)]
        uniform token projectionModeHint = "{projection_mode}"
        uniform token sortingModeHint = "{sorting_mode}"
    }}"""


def build_stage(
    batch_size: int,
    width: int,
    height: int,
    focal_scale: float,
    viewmats: torch.Tensor,
    aa_op: str,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    semantic_group_count: int = SEMANTIC_GROUP_COUNT,
    color_only: bool = False,
    splat_prim_semantic_ids_override: list[int] | None = None,
    projection_mode: str = "perspective",
    sorting_mode: str = "zDepth",
    fractional_opacity: bool = False,
    world_prims: str = "",
) -> str:
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"Unsupported projection mode: {projection_mode}")
    if sorting_mode not in SORTING_MODES:
        raise ValueError(f"Unsupported sorting mode: {sorting_mode}")
    focal_length = 36.0 * focal_scale
    camera_world_matrices = (
        opencv_viewmats_to_usd_camera_world_matrices(viewmats).to(device="cpu", dtype=torch.float64).tolist()
    )

    def matrix_literal(matrix: list[list[float]]) -> str:
        rows = ", ".join("(" + ", ".join(f"{value:.17g}" for value in row) + ")" for row in matrix)
        return f"({rows})"

    camera_defs = "\n\n".join(
        f"""    def Camera "Camera_{index}" (
        prepend apiSchemas = [
            "OmniRtxCameraAutoExposureAPI_1",
            "OmniRtxCameraExposureAPI_1"
        ]
    )
    {{
        float2 clippingRange = ({near_plane}, {far_plane})
        float exposure:fStop = 5
        float exposure:responsivity = 1.1026709
        float exposure:time = 0.02
        float focalLength = {focal_length}
        float focusDistance = 5
        float fStop = 0
        float horizontalAperture = 36
        bool omni:rtx:autoExposure:enabled = 0
        float verticalAperture = 36
        token projection = "perspective"
        matrix4d xformOp:transform = {matrix_literal(matrix)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}"""
        for index, matrix in enumerate(camera_world_matrices)
    )
    ordered_vars = (
        "<LdrColor>"
        if color_only
        else (
            "<LdrColor>,\n"
            "            <DistanceToImagePlaneSD>,\n"
            "            <SemanticSegmentation>,\n"
            "            <SemanticIdMap>"
        )
    )
    auxiliary_render_vars = (
        ""
        if color_only
        else """

        def RenderVar "DistanceToImagePlaneSD"
        {
            string sourceName = "DistanceToImagePlaneSD"
        }

        def RenderVar "SemanticSegmentation"
        {
            string sourceName = "SemanticSegmentation"
        }

        def RenderVar "SemanticIdMap"
        {
            string sourceName = "SemanticIdMap"
        }"""
    )
    fractional_opacity_settings = (
        "\n        bool omni:rtx:pt:fractionalCutoutOpacity = 1\n        bool omni:rtx:rt:fractionalOpacity = 1"
        if fractional_opacity
        else ""
    )
    product_defs = "\n\n".join(
        f"""    def RenderProduct "GaussianCamera_{index}"
    {{
        rel camera = </World/Camera_{index}>
        int2 resolution = ({width}, {height})
        token omni:rtx:rendermode = "RealTimePathTracing"{fractional_opacity_settings}
        token omni:rtx:background:source:type = "color"
        bool omni:rtx:dlss:frameGeneration = 0
        bool omni:rtx:post:aa:limitedOps = 0
        token omni:rtx:post:aa:op = "{aa_op}"
        bool omni:rtx:post:compositing:blackBackground = 0
        bool omni:rtx:post:compositing:doComposite = 0
        bool omni:rtx:post:compositing:enabled = 1
        bool omni:rtx:post:compositing:outputAlpha = 1
        bool omni:rtx:post:compositing:premultiply = 0
        rel orderedVars = [
            {ordered_vars}
        ]

        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}{auxiliary_render_vars}
    }}"""
        for index in range(batch_size)
    )
    prim_semantic_ids = (
        list(range(semantic_group_count))
        if splat_prim_semantic_ids_override is None
        else splat_prim_semantic_ids_override
    )
    if not prim_semantic_ids or min(prim_semantic_ids) < 0 or max(prim_semantic_ids) >= semantic_group_count:
        raise ValueError("Splat prim semantic IDs are outside the stage range.")
    particle_defs = "\n\n".join(
        particle_field_definition(
            prim_index,
            semantic_id,
            projection_mode=projection_mode,
            sorting_mode=sorting_mode,
        )
        for prim_index, semantic_id in enumerate(prim_semantic_ids)
    )
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "World"
{{
{particle_defs}

{world_prims}

{camera_defs}
}}

def Scope "Render"
{{
{product_defs}
}}
"""


def upload_scene(
    renderer: ovrtx.Renderer,
    scene: dict[str, torch.Tensor],
    *,
    semantic_group_count: int | None = None,
    max_gaussians_per_prim: int = OVRTX_MAX_GAUSSIANS_PER_PRIM,
) -> tuple[dict[str, Any], dict[str, Any]]:
    colors = scene["colors"]
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError("The initial OVRTX runner supports degree-zero RGB scenes only.")
    gaussian_count = int(colors.shape[0])
    semantic_group_count = (
        int(scene["semantic_ids"].max().item()) + 1 if semantic_group_count is None else semantic_group_count
    )
    if gaussian_count < semantic_group_count:
        raise ValueError(
            f"OVRTX semantic parity requires at least {semantic_group_count} Gaussians; got {gaussian_count}."
        )

    sh_dc = ((colors - 0.5) / SH_C0).contiguous()
    arrays = {
        "positions": scene["means"],
        # OVRTX/Fabric quaternion tensors use (i, j, k, real) lane order,
        # while canonical 3DGS tensors use (real, i, j, k).
        "orientations": quaternion_wxyz_to_xyzw(scene["quats"]),
        "scales": scene["scales"],
        "opacities": scene["opacities"],
        "radiance:sphericalHarmonicsCoefficients": sh_dc,
    }
    modulo_ids = torch.arange(
        gaussian_count,
        device=colors.device,
        dtype=scene["semantic_ids"].dtype,
    ).remainder_(semantic_group_count)
    if torch.equal(scene["semantic_ids"], modulo_ids):
        counts = semantic_group_counts(
            gaussian_count,
            semantic_group_count,
        )
        permutation = semantic_group_permutation(
            gaussian_count,
            semantic_group_count,
            device=colors.device,
        )
        grouping_rule = "semantic_id[i] = i % semantic_group_count"
    else:
        counts, permutation = semantic_label_grouping(
            scene["semantic_ids"],
            group_count=semantic_group_count,
        )
        grouping_rule = "stable label-major argsort"
    del modulo_ids
    prim_semantic_ids = splat_prim_semantic_ids(
        counts,
        max_gaussians_per_prim=max_gaussians_per_prim,
    )
    paths = splat_paths(len(prim_semantic_ids))
    grouped_buffers: dict[str, torch.Tensor] = {}
    grouped_views: dict[str, list[torch.Tensor]] = {}

    torch.cuda.synchronize()
    for attribute_name, tensor in arrays.items():
        grouped = tensor.index_select(0, permutation).contiguous()
        semantic_views = list(grouped.split(counts))
        views = [shard for semantic_view in semantic_views for shard in semantic_view.split(max_gaussians_per_prim)]
        if len(views) != len(paths) or not all(view.is_contiguous() for view in views):
            raise RuntimeError(f"Failed to build contiguous {attribute_name} groups.")
        renderer.write_array_attribute(
            prim_paths=paths,
            attribute_name=attribute_name,
            tensors=views,
            data_access=ovrtx.DataAccess.ASYNC,
        )
        grouped_buffers[attribute_name] = grouped
        grouped_views[attribute_name] = views

    renderer.reset()
    grouped_source_bytes = tensor_bytes(grouped_buffers)
    contract = {
        "scene_arrays": {
            name: {
                "grouped_shape": list(grouped_buffers[name].shape),
                "dtype": str(grouped_buffers[name].dtype),
                "device": str(grouped_buffers[name].device),
                "group_count": len(grouped_views[name]),
            }
            for name in arrays
        },
        "semantic_group_count": semantic_group_count,
        "semantic_group_counts_min": min(counts),
        "semantic_group_counts_max": max(counts),
        "semantic_group_rule": grouping_rule,
        "semantic_authoring": ("one or more SemanticsAPI-labeled ParticleFields per class"),
        "splat_prim_count": len(paths),
        "splat_prim_semantic_ids": prim_semantic_ids,
        "splat_prim_counts": [int(view.shape[0]) for view in grouped_views["positions"]],
        "max_gaussians_per_splat_prim": max_gaussians_per_prim,
        "scene_sharded_for_ovrtx_limit": len(paths) > semantic_group_count,
        "camera_transform_source": ("authored-usd-matrix4d-from-opencv-viewmat"),
        "scene_upload_device": "cuda:0",
        "scene_upload_access": "ovrtx.DataAccess.ASYNC",
        "scene_quaternion_order": "wxyz",
        "ovrtx_fabric_quaternion_lane_order": "xyzw",
        "quaternion_upload_conversion": "wxyz-to-xyzw",
        "async_source_buffers_kept_alive": True,
        "grouped_source_buffer_bytes": grouped_source_bytes,
    }
    keepalive = {
        "buffers": grouped_buffers,
        "views": grouped_views,
    }
    return contract, keepalive


def allocate_contract_outputs(
    batch_size: int,
    height: int,
    width: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    outputs = {
        "rgb": torch.empty(
            (batch_size, height, width, 3),
            device="cuda",
            dtype=torch.float32,
        ),
        "depth": torch.empty(
            (batch_size, height, width, 1),
            device="cuda",
            dtype=torch.float32,
        ),
        "alpha": torch.empty(
            (batch_size, height, width, 1),
            device="cuda",
            dtype=torch.float32,
        ),
        "semantic_id": torch.empty(
            (batch_size, height, width, 1),
            device="cuda",
            dtype=torch.int64,
        ),
    }
    workspace = {
        "native_semantic_ids": torch.empty(
            (batch_size, height, width, 1),
            device="cuda",
            dtype=torch.int64,
        )
    }
    return outputs, workspace


def output_contract_metadata(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "contiguous": tensor.is_contiguous(),
        }
        for name, tensor in outputs.items()
    }


def copy_cpu_render_var(frame: object, name: str) -> np.ndarray:
    mapping = frame.render_vars[name].map(device=ovrtx.Device.CPU)
    try:
        return np.from_dlpack(mapping).copy()
    finally:
        mapping.unmap()


def semantic_lut_from_outputs(
    outputs: object,
    first_path: str,
    *,
    expected_group_count: int = SEMANTIC_GROUP_COUNT,
) -> tuple[dict[int, str], np.ndarray, torch.Tensor]:
    frame = outputs[first_path].frames[0]
    id_map = decode_semantic_id_map(copy_cpu_render_var(frame, "SemanticIdMap"))
    lut_cpu = renderer_semantic_lut(
        id_map,
        expected_group_count=expected_group_count,
    )
    lut_cuda = torch.from_numpy(lut_cpu).to(device="cuda", non_blocking=False)
    return id_map, lut_cpu, lut_cuda


def copy_mapped_outputs_to_contract(
    outputs: object,
    paths: list[str],
    contract_outputs: dict[str, torch.Tensor],
    workspace: dict[str, torch.Tensor],
    semantic_lut: torch.Tensor,
) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    mappings: list[object] = []
    mapped_tensors: list[torch.Tensor] = []
    try:
        for index, path in enumerate(paths):
            frame = outputs[path].frames[0]
            tensors: dict[str, torch.Tensor] = {}
            for name in MAPPED_AOVS:
                mapping = frame.render_vars[name].map(device=ovrtx.Device.CUDA)
                tensor = torch.from_dlpack(mapping)
                if not tensor.is_cuda:
                    raise RuntimeError(f"{path} {name} did not map to CUDA.")
                shapes.setdefault(name, list(tensor.shape))
                mappings.append(mapping)
                mapped_tensors.append(tensor)
                tensors[name] = tensor

            ldr = tensors["LdrColor"]
            distance = tensors["DistanceToImagePlaneSD"]
            native_semantic = tensors["SemanticSegmentation"]
            if ldr.shape[-1] != 4:
                raise RuntimeError(f"Expected RGBA LdrColor; got {tuple(ldr.shape)}.")

            contract_outputs["rgb"][index].copy_(ldr[..., :3]).mul_(1.0 / 255.0)
            contract_outputs["depth"][index].copy_(distance)
            contract_outputs["alpha"][index, ..., 0].copy_(ldr[..., 3]).mul_(1.0 / 255.0)
            semantic_indices = workspace["native_semantic_ids"][index]
            semantic_indices.copy_(native_semantic)
            torch.take(
                semantic_lut,
                semantic_indices,
                out=contract_outputs["semantic_id"][index],
            )
        torch.cuda.synchronize()
    finally:
        mapped_tensors.clear()
        for mapping in reversed(mappings):
            mapping.unmap()
    return shapes


def render_once(
    renderer: ovrtx.Renderer,
    paths: list[str],
    contract_outputs: dict[str, torch.Tensor],
    workspace: dict[str, torch.Tensor],
    semantic_lut: torch.Tensor,
) -> dict[str, list[int]]:
    outputs = renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
    shapes = copy_mapped_outputs_to_contract(
        outputs,
        paths,
        contract_outputs,
        workspace,
        semantic_lut,
    )
    del outputs
    return shapes


def save_reference_outputs(
    renderer: ovrtx.Renderer,
    paths: list[str],
    output_dir: Path,
    semantic_lut: np.ndarray,
) -> dict[str, Any]:
    outputs = renderer.step(render_products=set(paths), delta_time=1.0 / 60.0)
    ldr_images: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    native_semantics: list[np.ndarray] = []
    for path in paths:
        frame = outputs[path].frames[0]
        ldr_images.append(copy_cpu_render_var(frame, "LdrColor"))
        distances.append(copy_cpu_render_var(frame, "DistanceToImagePlaneSD"))
        native_semantics.append(copy_cpu_render_var(frame, "SemanticSegmentation"))

    ldr = np.stack(ldr_images)
    distance = np.stack(distances)
    native_semantic = np.stack(native_semantics)
    semantic = remap_renderer_semantics(native_semantic, semantic_lut)
    rgb = ldr[..., :3].astype(np.float32) / 255.0
    alpha = ldr[..., 3:4].astype(np.float32) / 255.0
    np.save(output_dir / "reference_ldr.npy", ldr)
    np.save(output_dir / "reference_rgb.npy", rgb)
    np.save(output_dir / "reference_distance_to_image_plane.npy", distance)
    np.save(output_dir / "reference_depth.npy", distance)
    np.save(output_dir / "reference_alpha.npy", alpha)
    np.save(output_dir / "reference_semantic_native.npy", native_semantic)
    np.save(output_dir / "reference_semantic.npy", semantic)
    for index, image in enumerate(ldr):
        Image.fromarray(image).save(output_dir / f"reference_ldr_{index:04d}.png")

    rgb_u8 = ldr[..., :3]
    alpha_u8 = ldr[..., 3]
    finite_distance = distance[np.isfinite(distance) & (distance > 0)]
    foreground_semantic = semantic[semantic != BACKGROUND_SEMANTIC_ID]
    return {
        "artifacts_saved": True,
        "ldr_shape": list(ldr.shape),
        "rgb_shape": list(rgb.shape),
        "distance_shape": list(distance.shape),
        "alpha_shape": list(alpha.shape),
        "semantic_shape": list(semantic.shape),
        "rgb_nonzero_pixels": int(np.count_nonzero(np.any(rgb_u8 != 0, axis=-1))),
        "alpha_min": int(alpha_u8.min()),
        "alpha_max": int(alpha_u8.max()),
        "alpha_zero_pixels": int(np.count_nonzero(alpha_u8 == 0)),
        "alpha_partial_pixels": int(np.count_nonzero((alpha_u8 > 0) & (alpha_u8 < 255))),
        "alpha_opaque_pixels": int(np.count_nonzero(alpha_u8 == 255)),
        "finite_positive_distance_pixels": int(finite_distance.size),
        "semantic_background_id": BACKGROUND_SEMANTIC_ID,
        "semantic_foreground_pixels": int(foreground_semantic.size),
        "semantic_unique_ids": np.unique(semantic).astype(np.int64).tolist(),
        "native_semantic_unique_ids": np.unique(native_semantic).astype(np.uint32).tolist(),
    }


def summarize_contract_outputs(
    contract_outputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Validate resident CUDA outputs without writing large CPU artifacts."""
    rgb = contract_outputs["rgb"]
    depth = contract_outputs["depth"]
    alpha = contract_outputs["alpha"]
    semantic = contract_outputs["semantic_id"]
    positive_depth = torch.isfinite(depth) & (depth > 0)
    foreground_semantic = semantic != BACKGROUND_SEMANTIC_ID
    return {
        "artifacts_saved": False,
        "ldr_shape": None,
        "rgb_shape": list(rgb.shape),
        "distance_shape": list(depth.shape),
        "alpha_shape": list(alpha.shape),
        "semantic_shape": list(semantic.shape),
        "rgb_nonzero_pixels": int(torch.count_nonzero(torch.any(rgb != 0, dim=-1)).item()),
        "alpha_min": float(alpha.min().item()),
        "alpha_max": float(alpha.max().item()),
        "alpha_zero_pixels": int(torch.count_nonzero(alpha == 0).item()),
        "alpha_partial_pixels": int(torch.count_nonzero((alpha > 0) & (alpha < 1)).item()),
        "alpha_opaque_pixels": int(torch.count_nonzero(alpha == 1).item()),
        "finite_positive_distance_pixels": int(torch.count_nonzero(positive_depth).item()),
        "semantic_background_id": BACKGROUND_SEMANTIC_ID,
        "semantic_foreground_pixels": int(torch.count_nonzero(foreground_semantic).item()),
        "semantic_unique_ids": None,
        "native_semantic_unique_ids": None,
    }


def contract_reference_has_signal(reference: dict[str, Any]) -> bool:
    return (
        reference["rgb_nonzero_pixels"] > 0
        and reference["alpha_max"] > 0
        and reference["finite_positive_distance_pixels"] > 0
        and reference["semantic_foreground_pixels"] > 0
    )


def nullable_stats() -> dict[str, None]:
    return {
        "mean": None,
        "std": None,
        "p50": None,
        "p95": None,
        "ci95": None,
    }


def load_scene_and_cameras(
    args: argparse.Namespace,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, Any],
    CameraBundle,
    dict[str, Any],
]:
    if args.scene == PROJECTION_ACTIVATION_SCENE:
        scene = projection_activation_scene_tensors(device="cuda")
        cameras = camera_bundle(
            args.batch_size,
            args.width,
            args.height,
            device="cuda",
            focal_scale=args.focal_scale,
        )
        return (
            scene,
            projection_activation_scene_manifest(),
            cameras,
            {
                "source": "controlled_projection_activation_fixture",
                "host_ply_seconds": 0.0,
                "canonicalize_upload_seconds": 0.0,
            },
        )
    if args.scene in SYNTHETIC_SCENES:
        gaussian_count, seed = SYNTHETIC_SCENES[args.scene]
        if args.scene in STRESS_SCENES:
            scene = stress_scene_tensors(
                args.scene,
                gaussian_count,
                seed=seed,
                device="cuda",
            )
            scene_manifest = stress_scene_manifest(
                args.scene,
                gaussian_count,
                seed,
            )
        else:
            scene = synthetic_scene_tensors(
                gaussian_count,
                seed=seed,
                device="cuda",
            )
            scene_manifest = synthetic_scene_manifest(
                args.scene,
                gaussian_count,
                seed,
            )
        cameras = camera_bundle(
            args.batch_size,
            args.width,
            args.height,
            device="cuda",
            focal_scale=args.focal_scale,
        )
        return (
            scene,
            scene_manifest,
            cameras,
            {
                "source": "deterministic_gpu_generation",
                "host_ply_seconds": 0.0,
                "canonicalize_upload_seconds": 0.0,
            },
        )

    if args.scene not in PLY_SCENES:
        raise ValueError(f"Unsupported scene: {args.scene}")
    spec = PLY_SCENES[args.scene]
    scene_path = args.home_scan_path if args.scene == HOME_SCAN_SCENE else args.public_scene_path
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    checksum = file_sha256(scene_path)
    expected_sha256 = str(spec["sha256"])
    if checksum != expected_sha256:
        raise ValueError(f"{args.scene} SHA-256 mismatch: expected {expected_sha256}, got {checksum}.")

    host_start = time.perf_counter()
    raw_scene = load_ply_to_gaussians(scene_path)
    host_ply_seconds = time.perf_counter() - host_start
    expected_count = int(spec["count"])
    if raw_scene.count != expected_count:
        raise ValueError(f"{args.scene} Gaussian-count mismatch: expected {expected_count}, got {raw_scene.count}.")

    upload_start = time.perf_counter()
    canonical = canonicalize_3dgs_scene(
        raw_scene,
        device="cuda",
    )
    semantic_ids = deterministic_semantic_ids(
        canonical.count,
        device="cuda",
        dtype=torch.int64,
    )
    scene = {
        "means": canonical.means,
        "quats": canonical.rotations,
        "scales": canonical.scales,
        "opacities": canonical.opacities,
        "colors": canonical.features[:, :3].contiguous(),
        "semantic_ids": semantic_ids,
    }
    del canonical
    del raw_scene
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    canonicalize_upload_seconds = time.perf_counter() - upload_start

    position_min = scene["means"].amin(dim=0)
    position_max = scene["means"].amax(dim=0)
    extent = position_max - position_min
    camera_quantiles = spec["camera_bounds_quantiles"]
    if camera_quantiles is None:
        camera_min = position_min
        camera_max = position_max
        bounds_policy = "full-componentwise-min-max"
    else:
        quantiles = torch.tensor(
            camera_quantiles,
            device="cuda",
            dtype=scene["means"].dtype,
        )
        camera_min, camera_max = torch.quantile(
            scene["means"],
            quantiles,
            dim=0,
        )
        bounds_policy = f"componentwise-quantile-{camera_quantiles[0]}-{camera_quantiles[1]}"
    center = (camera_min + camera_max) * 0.5
    camera_extent = camera_max - camera_min
    bounding_radius = float((torch.linalg.vector_norm(camera_extent) * 0.5).item())
    cameras = axis_aligned_exterior_camera_bundle(
        center=center,
        bounding_radius=bounding_radius,
        batch_size=args.batch_size,
        width=args.width,
        height=args.height,
        focal_scale=args.focal_scale,
        fit_margin=args.home_scan_fit_margin,
        camera_path=f"{args.scene}-axis-aligned-exterior",
        bounds_policy=bounds_policy,
    )
    manifest = {
        "version": "benchmark-manifest/v1",
        "scene_id": args.scene,
        "scene_type": "canonical-real-world-ply",
        "gaussian_count": expected_count,
        "path": str(scene_path),
        "file_sha256": checksum,
        "source_manifest": str(spec["manifest"]),
        "source_spherical_harmonics_degree": int(spec["source_sh_degree"]),
        "spherical_harmonics_degree": 0,
        "spherical_harmonics_policy": ("degree-zero DC reconstruction shared by OVRTX and custom"),
        "attributes": [
            "means",
            "quats",
            "scales",
            "opacities",
            "colors",
            "semantic_ids",
        ],
        "semantic_id_rule": deterministic_semantic_manifest(expected_count),
        "position_min": position_min.detach().cpu().tolist(),
        "position_max": position_max.detach().cpu().tolist(),
        "center": center.detach().cpu().tolist(),
        "extent": extent.detach().cpu().tolist(),
        "camera_bounds_min": camera_min.detach().cpu().tolist(),
        "camera_bounds_max": camera_max.detach().cpu().tolist(),
        "camera_bounds_policy": bounds_policy,
        "bounding_radius": bounding_radius,
    }
    manifest["checksum_sha256"] = sha256_json(manifest)
    return (
        scene,
        manifest,
        cameras,
        {
            "source": "canonical_3dgs_ply",
            "host_ply_seconds": host_ply_seconds,
            "canonicalize_upload_seconds": (canonicalize_upload_seconds),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("BENCHMARK_PROTOCOL.md"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene",
        choices=sorted((*SYNTHETIC_SCENES, PROJECTION_ACTIVATION_SCENE, *PLY_SCENES)),
        default="synthetic-small",
    )
    parser.add_argument(
        "--home-scan-path",
        type=Path,
        default=Path("/workspace/datasets/home-scan-lod0.ply"),
    )
    parser.add_argument(
        "--public-scene-path",
        type=Path,
        default=Path(
            "/workspace/datasets/public-gaussian/"
            "Voxel51_gaussian_splatting/FO_dataset/train/"
            "point_cloud/iteration_30000/point_cloud.ply"
        ),
    )
    parser.add_argument("--home-scan-fit-margin", type=float, default=1.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--focal-scale", type=float, default=0.9)
    parser.add_argument(
        "--projection-mode",
        choices=PROJECTION_MODES,
        default="perspective",
    )
    parser.add_argument(
        "--sorting-mode",
        choices=SORTING_MODES,
        default="zDepth",
    )
    parser.add_argument(
        "--fractional-opacity",
        action="store_true",
        help="Enable OVRTX fractional Gaussian opacity settings.",
    )
    parser.add_argument(
        "--semantic-scheme",
        choices=("index-modulo", "spatial-grid"),
        default="index-modulo",
    )
    parser.add_argument(
        "--semantic-grid",
        type=lambda value: tuple(int(item.strip()) for item in value.split(",") if item.strip()),
        default=(2, 2, 2),
        help="XYZ grid dimensions for --semantic-scheme spatial-grid.",
    )
    parser.add_argument(
        "--aa-op",
        choices=("none", "taa", "fxaa", "dlss", "rtxaa"),
        default="none",
    )
    parser.add_argument(
        "--skip-reference-artifacts",
        action="store_true",
        help=("Validate resident CUDA outputs without an extra render, CPU mapping, NPY files, or per-camera PNGs."),
    )
    args = parser.parse_args()

    if not args.protocol.is_file():
        raise FileNotFoundError(f"Benchmark protocol not found: {args.protocol}")
    if args.batch_size <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("Batch size and resolution must be positive.")
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    if args.home_scan_fit_margin <= 0:
        parser.error("--home-scan-fit-margin must be positive.")
    if len(args.semantic_grid) != 3 or min(args.semantic_grid) <= 0:
        parser.error("--semantic-grid must contain three positive dimensions.")

    args.output.mkdir(parents=True, exist_ok=True)
    write_json_schema(args.output / "baseline-result.schema.json")
    scene, scene_manifest, cameras, scene_loading = load_scene_and_cameras(args)
    if args.semantic_scheme == "spatial-grid":
        raw_semantic_ids = spatial_semantic_ids(
            scene["means"],
            grid=args.semantic_grid,
            dtype=torch.int64,
        )
        scene["semantic_ids"], occupied_source_ids = compact_semantic_ids(
            raw_semantic_ids,
            dtype=torch.int64,
        )
        scene_manifest = dict(scene_manifest)
        scene_manifest["semantic_id_rule"] = spatial_semantic_manifest(
            int(scene["means"].shape[0]),
            grid=args.semantic_grid,
            position_min=scene["means"].amin(dim=0),
            position_max=scene["means"].amax(dim=0),
            occupied_source_ids=occupied_source_ids,
        )
        scene_manifest["checksum_sha256"] = sha256_json(
            {key: value for key, value in scene_manifest.items() if key != "checksum_sha256"}
        )
    gaussian_count = int(scene["means"].shape[0])
    semantic_group_count = int(scene["semantic_ids"].max().item()) + 1
    semantic_counts = (
        torch.bincount(
            scene["semantic_ids"],
            minlength=semantic_group_count,
        )
        .to(device="cpu")
        .tolist()
    )
    prim_semantic_ids = splat_prim_semantic_ids(semantic_counts)
    canonical_scene_bytes = tensor_bytes(scene)
    near_plane = float(cameras.manifest["near_plane"])
    far_plane = float(cameras.manifest["far_plane"])
    stage = build_stage(
        args.batch_size,
        args.width,
        args.height,
        args.focal_scale,
        cameras.viewmats,
        args.aa_op,
        near_plane,
        far_plane,
        semantic_group_count,
        splat_prim_semantic_ids_override=prim_semantic_ids,
        projection_mode=args.projection_mode,
        sorting_mode=args.sorting_mode,
        fractional_opacity=args.fractional_opacity,
    )
    stage_path = args.output / "scene.usda"
    stage_path.write_text(stage, encoding="utf-8")
    paths = product_paths(args.batch_size)

    create_start = time.perf_counter()
    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            keep_system_alive=True,
            log_level="info",
            active_cuda_gpus="0",
            use_vulkan=True,
        )
    )
    renderer_create_ms = (time.perf_counter() - create_start) * 1000.0

    load_start = time.perf_counter()
    renderer.open_usd_from_string(stage)
    splat_prim_paths = splat_paths(len(prim_semantic_ids))
    runtime_readback = {
        "projection_mode": require_token_attribute_readback(
            renderer,
            attribute_name="projectionModeHint",
            prim_paths=splat_prim_paths,
            expected=args.projection_mode,
        ),
        "sorting_mode": require_token_attribute_readback(
            renderer,
            attribute_name="sortingModeHint",
            prim_paths=splat_prim_paths,
            expected=args.sorting_mode,
        ),
    }
    upload_contract, upload_keepalive = upload_scene(
        renderer,
        scene,
        semantic_group_count=semantic_group_count,
    )
    stage_load_upload_ms = (time.perf_counter() - load_start) * 1000.0
    del scene
    torch.cuda.empty_cache()

    contract_outputs, output_workspace = allocate_contract_outputs(
        args.batch_size,
        args.height,
        args.width,
    )

    first_start = time.perf_counter()
    first_outputs = renderer.step(
        render_products=set(paths),
        delta_time=1.0 / 60.0,
    )
    semantic_id_map, semantic_lut_cpu, semantic_lut_cuda = semantic_lut_from_outputs(
        first_outputs,
        paths[0],
        expected_group_count=semantic_group_count,
    )
    output_shapes = copy_mapped_outputs_to_contract(
        first_outputs,
        paths,
        contract_outputs,
        output_workspace,
        semantic_lut_cuda,
    )
    del first_outputs
    first_render_wall_ms = (time.perf_counter() - first_start) * 1000.0
    (args.output / "semantic-id-map.json").write_text(
        json.dumps(
            {
                "background_semantic_id": BACKGROUND_SEMANTIC_ID,
                "renderer_labels": {str(renderer_id): label for renderer_id, label in semantic_id_map.items()},
                "renderer_id_to_semantic_id": {
                    str(renderer_id): int(semantic_id)
                    for renderer_id, semantic_id in enumerate(semantic_lut_cpu)
                    if semantic_id != BACKGROUND_SEMANTIC_ID
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    warmup_samples: list[float] = []
    for _ in range(args.warmup):
        start = time.perf_counter()
        output_shapes = render_once(
            renderer,
            paths,
            contract_outputs,
            output_workspace,
            semantic_lut_cuda,
        )
        warmup_samples.append((time.perf_counter() - start) * 1000.0)

    wall_samples: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        output_shapes = render_once(
            renderer,
            paths,
            contract_outputs,
            output_workspace,
            semantic_lut_cuda,
        )
        wall_samples.append((time.perf_counter() - start) * 1000.0)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocator_before_memory_pass = torch.cuda.memory_allocated()
    with GpuMemorySampler(os.getpid()) as memory_sampler:
        for _ in range(min(3, args.iterations)):
            render_once(
                renderer,
                paths,
                contract_outputs,
                output_workspace,
                semantic_lut_cuda,
            )
    torch.cuda.synchronize()
    torch_peak_allocated = torch.cuda.max_memory_allocated()
    torch_peak_reserved = torch.cuda.max_memory_reserved()
    allocator_after_memory_pass = torch.cuda.memory_allocated()

    wall_stats = stats(wall_samples)
    mean_seconds = wall_stats["mean"] / 1000.0
    pixels = args.batch_size * args.width * args.height
    validation_renders = 0
    if args.skip_reference_artifacts:
        reference = summarize_contract_outputs(contract_outputs)
        while not contract_reference_has_signal(reference) and validation_renders < 256:
            render_once(
                renderer,
                paths,
                contract_outputs,
                output_workspace,
                semantic_lut_cuda,
            )
            validation_renders += 1
            reference = summarize_contract_outputs(contract_outputs)
    else:
        reference = save_reference_outputs(
            renderer,
            paths,
            args.output,
            semantic_lut_cpu,
        )
    run_id = str(uuid.uuid4())
    config_id = (
        f"cfg-v1-ovrtx-multi-product-{args.scene}-shared-scene-"
        f"b{args.batch_size}-{args.width}x{args.height}-aa-{args.aa_op}-"
        f"projection-{args.projection_mode}-sorting-{args.sorting_mode}-"
        f"fractional-opacity-{int(args.fractional_opacity)}-"
        f"semantic-{args.semantic_scheme}-{semantic_group_count}-"
        "rgb-depth-alpha-semantic"
    )
    output_bytes = tensor_bytes(contract_outputs)
    reusable_workspace_bytes = (
        tensor_bytes(output_workspace) + semantic_lut_cuda.numel() * semantic_lut_cuda.element_size()
    )
    output_contract = output_contract_metadata(contract_outputs)
    semantic_contract_valid = len(semantic_id_map) == semantic_group_count and set(
        semantic_lut_cpu[semantic_lut_cpu != BACKGROUND_SEMANTIC_ID].tolist()
    ) == set(range(semantic_group_count))
    valid = (
        contract_reference_has_signal(reference)
        and semantic_contract_valid
        and all(shape[0] == args.height and shape[1] == args.width for shape in output_shapes.values())
        and all(tensor.is_cuda for tensor in contract_outputs.values())
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_id": config_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": environment(),
        "implementation": {
            "name": "ovrtx-multi-product",
            "commit": implementation_commit(),
            "version": ovrtx.__version__,
            "renderer_version": list(renderer.version),
            "flags": {
                "layout": "one-render-product-per-camera",
                "single_host_submission": True,
                "host_output_mapping_loop": True,
                "render_mode": "RealTimePathTracing",
                "color_output": "OVRTX LdrColor normalized to sRGB float32 [0,1]",
                "anti_aliasing": args.aa_op,
                "projection_mode": args.projection_mode,
                "sorting_mode": args.sorting_mode,
                "fractional_opacity": args.fractional_opacity,
                "dlss_frame_generation": False,
                "compositing_alpha": True,
                "background_composite": False,
                "output_device": "cuda",
                "mapped_aovs": MAPPED_AOVS,
                "semantic_fields": semantic_group_count,
                "semantic_scheme": args.semantic_scheme,
                "semantic_grid": list(args.semantic_grid),
                "semantic_gpu_remap": "preallocated int64 indices plus torch.take",
                "reference_artifacts_saved": not args.skip_reference_artifacts,
                "untimed_validation_renders": validation_renders,
            },
        },
        "configuration": {
            "batch_size": args.batch_size,
            "width": args.width,
            "height": args.height,
            "scene_mode": "shared-scene",
            "outputs": REQUIRED_OUTPUTS,
            "semantic_contract": OVRTX_OUTPUT_CONTRACT,
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "background": [0.0, 0.0, 0.0],
            "semantic_background_id": BACKGROUND_SEMANTIC_ID,
            "semantic_group_count": semantic_group_count,
            "semantic_scheme": args.semantic_scheme,
            "semantic_grid": list(args.semantic_grid),
            "projection_mode": args.projection_mode,
            "sorting_mode": args.sorting_mode,
            "fractional_opacity": args.fractional_opacity,
            "projection_mode_hint_is_advisory": True,
            "fabric_scene_delegate": fabric_scene_delegate_record(),
            "runtime_token_readback": runtime_readback,
            "camera_manifest": cameras.manifest,
            "output_contract": output_contract,
        },
        "dataset": {
            **scene_manifest,
            "loading": scene_loading,
        },
        "timing": {
            "renderer_create_ms": renderer_create_ms,
            "stage_load_upload_ms": stage_load_upload_ms,
            "first_render_wall_ms": first_render_wall_ms,
            "warmup_wall_ms": stats(warmup_samples) if warmup_samples else None,
            "gpu_ms": nullable_stats(),
            "gpu_timing_note": (
                "OVRTX renders through Vulkan/RTX and does not expose its internal "
                "GPU stream to CUDA events; synchronized end-to-end wall time is "
                "primary and includes CUDA mapping, dtype/color conversion, depth "
                "copy, alpha extraction, and semantic-ID remapping."
            ),
            "wall_ms": wall_stats,
            "images_per_second": (args.batch_size / mean_seconds if mean_seconds > 0 else None),
            "megapixels_per_second": (pixels / mean_seconds / 1.0e6 if mean_seconds > 0 else None),
            "wall_samples_ms": wall_samples,
            "untimed_validation_renders": validation_renders,
        },
        "memory": {
            "peak_allocated_bytes": torch_peak_allocated,
            "peak_reserved_bytes": torch_peak_reserved,
            "persistent_scene_bytes": upload_contract["grouped_source_buffer_bytes"],
            "canonical_scene_bytes": canonical_scene_bytes,
            "reusable_workspace_bytes": reusable_workspace_bytes,
            "output_bytes": output_bytes,
            "temporary_allocation_delta_bytes": max(
                0,
                torch_peak_allocated - allocator_before_memory_pass,
            ),
            "allocator_before_memory_pass_bytes": allocator_before_memory_pass,
            "allocator_after_memory_pass_bytes": allocator_after_memory_pass,
            "driver_process_memory_bytes": (memory_sampler.peak_bytes or memory_sampler.peak_total_bytes),
            "driver_process_memory_source": (
                "nvidia-smi-pmon-process" if memory_sampler.peak_bytes is not None else "nvidia-smi-total-dedicated-gpu"
            ),
            "driver_total_memory_bytes": memory_sampler.peak_total_bytes,
            "memory_note": (
                "OVRTX allocations are outside the PyTorch allocator; peak process "
                "memory is sampled in a separate warmed render pass so the sampler "
                "does not perturb timing. Total GPU use is the fallback because "
                "OVRTX can appear as a graphics rather than compute process. "
                "PyTorch figures cover persistent async upload buffers, contract "
                "outputs, semantic LUT/workspace, and conversion/remap work."
            ),
        },
        "work": {
            "gaussians": gaussian_count,
            "cameras": args.batch_size,
            "render_products": len(paths),
            "semantic_particle_fields": semantic_group_count,
            "pixels": pixels,
            "output_bytes": output_bytes,
            "visible_gaussians": None,
            "tile_intersections": None,
            "mapped_output_shapes": output_shapes,
        },
        "fidelity": {
            "reference": reference,
            "rgb_psnr": None,
            "rgb_ssim": None,
            "lpips": None,
            "alpha_mae": None,
            "depth_valid_relative_error": None,
            "semantic_id_agreement": None,
        },
        "upload": upload_contract,
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "stage_sha256": file_sha256(stage_path),
            "stage_file": str(stage_path),
        },
        "runtime_token_readback": runtime_readback,
        "semantic_mapping": {
            "id_map_entries": len(semantic_id_map),
            "lut_entries": int(semantic_lut_cpu.size),
            "all_requested_classes_present": semantic_contract_valid,
            "id_map_file": "semantic-id-map.json",
        },
        "status": {
            "verdict": "MEASURED" if valid else "INVALID_OUTPUT",
            "pass": valid,
            "notes": (
                "One OVRTX step renders every camera. RGB, metric depth, generated "
                "coverage alpha, and deterministic per-Gaussian-sidecar semantic "
                "IDs are converted/remapped into preallocated CUDA tensors before "
                "the synchronized timing boundary."
            ),
        },
    }
    write_result_json(result, args.output / f"{run_id}.json")
    write_results_csv([result], args.output / "baseline-results.csv")
    (args.output / "run-manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "result_count": 1,
                "results": [run_id],
                "fabric_scene_delegate": fabric_scene_delegate_record(),
                "projection_mode": args.projection_mode,
                "sorting_mode": args.sorting_mode,
                "fractional_opacity": args.fractional_opacity,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    del renderer
    del upload_keepalive
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
