"""Deterministic benchmark camera and scene manifests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


MANIFEST_VERSION = "benchmark-manifest/v1"
SEMANTIC_RULE_VERSION = "semantic-index-modulo/v1"
SPATIAL_SEMANTIC_RULE_VERSION = "semantic-spatial-grid/v1"
STRESS_SCENE_VERSION = "stress-scene/v1"


STRESS_SCENES = {
    "stress-anisotropic": (100_000, 1_043),
    "stress-overdraw": (100_000, 2_043),
}


def synthetic_exact_intersections_per_visible(
    *,
    width: int,
    height: int,
    tile_size: int,
) -> int:
    """Return the synthetic exact-ray capacity budget per visible record.

    Scale by the actual ceiling-rounded scheduling grid so non-power-of-two
    resolutions cannot under-allocate when they cross a tile boundary. Exact
    16x16 raster tiles use the production 32x32 macro-bin grid.
    """
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("width, height, and tile_size must be positive.")
    bin_tile_size = 32 if tile_size == 16 else tile_size
    reference_tiles = math.ceil(128 / bin_tile_size) ** 2
    image_tiles = math.ceil(width / bin_tile_size) * math.ceil(height / bin_tile_size)
    reference_budget = 64.0 if tile_size == 1 else (8.0 if tile_size == 16 else 32.0)
    return math.ceil(reference_budget * image_tiles / reference_tiles)


@dataclass(frozen=True)
class CameraBundle:
    viewmats: torch.Tensor
    intrinsics: torch.Tensor
    manifest: dict[str, Any]


def resolve_output_color_contract(
    *,
    linear_output: bool,
    authored_display_output: bool,
) -> dict[str, Any]:
    """Resolve the renderer transfer function and declared output color space."""
    if linear_output and authored_display_output:
        raise ValueError("linear_output and authored_display_output are mutually exclusive.")
    if authored_display_output:
        return {
            "output_srgb": False,
            "color_space": "display_srgb",
            "color_transform": "identity_authored_display_rgb",
        }
    if linear_output:
        return {
            "output_srgb": False,
            "color_space": "linear_rgb",
            "color_transform": "identity_linear_rgb",
        }
    return {
        "output_srgb": True,
        "color_space": "srgb",
        "color_transform": "linear_to_srgb",
    }


def canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def scene_tensors_sha256(
    tensors: dict[str, torch.Tensor],
    *,
    keys: tuple[str, ...] = (
        "means",
        "quats",
        "scales",
        "opacities",
        "colors",
        "semantic_ids",
    ),
) -> str:
    """Hash canonical tensor metadata and bytes after all scene transforms."""
    digest = hashlib.sha256()
    for key in keys:
        if key not in tensors:
            raise KeyError(f"Scene tensor bundle is missing {key!r}.")
        array = tensors[key].detach().contiguous().to(device="cpu").numpy()
        metadata = {
            "key": key,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        encoded = canonical_json_bytes(metadata)
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: str | Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_scene_manifest(scene_id: str, gaussian_count: int, seed: int) -> dict[str, Any]:
    manifest = {
        "version": MANIFEST_VERSION,
        "scene_id": scene_id,
        "scene_type": "deterministic-synthetic",
        "gaussian_count": gaussian_count,
        "seed": seed,
        "attributes": ["means", "quats", "scales", "opacities", "colors", "semantic_ids"],
        "generation": {
            "means_xy_scale": 3.0,
            "means_z_abs_scale": 1.5,
            "means_z_offset": 4.0,
            "scale": 0.025,
            "opacity": 0.65,
            "semantic_id_rule": deterministic_semantic_manifest(gaussian_count),
        },
    }
    manifest["checksum_sha256"] = sha256_json(manifest)
    return manifest


def stress_scene_manifest(scene_id: str, gaussian_count: int, seed: int) -> dict[str, Any]:
    """Describe deterministic scenes designed to invalidate easy synthetic cases."""
    if scene_id not in STRESS_SCENES:
        raise ValueError(f"Unsupported stress scene: {scene_id!r}.")
    if gaussian_count <= 0:
        raise ValueError("gaussian_count must be positive.")
    generation = {
        "stress-anisotropic": {
            "means_xy_std": 1.8,
            "means_z_abs_std": 1.25,
            "means_z_offset": 4.0,
            "quaternion": "normalized_random_wxyz",
            "scale_axes": [0.0125, 0.05, 0.18],
            "scale_log_jitter_std": 0.35,
            "opacity_range": [0.35, 0.95],
        },
        "stress-overdraw": {
            "means_xy_std": 0.35,
            "means_z_range": [4.25, 6.25],
            "quaternion": "normalized_random_wxyz",
            "scale_min": [0.08, 0.08, 0.015],
            "scale_max": [0.30, 0.30, 0.05],
            "opacity_range": [0.80, 0.99],
        },
    }[scene_id]
    manifest = {
        "version": MANIFEST_VERSION,
        "stress_version": STRESS_SCENE_VERSION,
        "scene_id": scene_id,
        "scene_type": "deterministic-adversarial-synthetic",
        "gaussian_count": gaussian_count,
        "seed": seed,
        "attributes": ["means", "quats", "scales", "opacities", "colors", "semantic_ids"],
        "generation": {
            **generation,
            "semantic_id_rule": spatial_semantic_manifest(
                gaussian_count,
                grid=(2, 2, 2),
                position_min=torch.tensor([-1.0, -1.0, -1.0]),
                position_max=torch.tensor([1.0, 1.0, 1.0]),
            )["formula"],
            "semantic_grid": [2, 2, 2],
        },
    }
    manifest["checksum_sha256"] = sha256_json(manifest)
    return manifest


def deterministic_semantic_manifest(gaussian_count: int, modulus: int = 1024) -> dict[str, Any]:
    """Describe the shared label rule for assets without source semantics."""
    manifest = {
        "version": SEMANTIC_RULE_VERSION,
        "gaussian_count": gaussian_count,
        "modulus": modulus,
        "formula": "semantic_id[i] = i % modulus",
    }
    manifest["checksum_sha256"] = sha256_json(manifest)
    return manifest


def deterministic_semantic_ids(
    gaussian_count: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.int64,
    modulus: int = 1024,
) -> torch.Tensor:
    """Generate the shared deterministic semantic sidecar on the target device."""
    if gaussian_count < 0:
        raise ValueError("gaussian_count must be non-negative.")
    if modulus <= 0:
        raise ValueError("modulus must be positive.")
    return torch.arange(gaussian_count, device=device, dtype=dtype).remainder_(modulus)


def spatial_semantic_manifest(
    gaussian_count: int,
    *,
    grid: tuple[int, int, int],
    position_min: torch.Tensor,
    position_max: torch.Tensor,
    occupied_source_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Describe a deterministic axis-aligned spatial semantic sidecar."""
    if gaussian_count < 0:
        raise ValueError("gaussian_count must be non-negative.")
    if len(grid) != 3 or min(grid) <= 0:
        raise ValueError("grid must contain three positive dimensions.")
    if position_min.shape != (3,) or position_max.shape != (3,):
        raise ValueError("position bounds must have shape [3].")
    grid_class_count = math.prod(grid)
    if occupied_source_ids is not None:
        if (
            not occupied_source_ids
            or occupied_source_ids != sorted(set(occupied_source_ids))
            or min(occupied_source_ids) < 0
            or max(occupied_source_ids) >= grid_class_count
        ):
            raise ValueError("occupied_source_ids must be unique, sorted, and within the grid.")
    manifest = {
        "version": SPATIAL_SEMANTIC_RULE_VERSION,
        "gaussian_count": gaussian_count,
        "grid": list(grid),
        "grid_class_count": grid_class_count,
        "class_count": (grid_class_count if occupied_source_ids is None else len(occupied_source_ids)),
        "formula": ("componentwise normalized position bins in XYZ-major order"),
        "position_min": position_min.detach().cpu().tolist(),
        "position_max": position_max.detach().cpu().tolist(),
    }
    if occupied_source_ids is not None:
        manifest["dense_ordinal_to_grid_class"] = occupied_source_ids
        manifest["formula"] += "; occupied classes compacted to dense ordinals"
    manifest["checksum_sha256"] = sha256_json(manifest)
    return manifest


def spatial_semantic_ids(
    means: torch.Tensor,
    *,
    grid: tuple[int, int, int] = (2, 2, 2),
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Assign coherent semantic IDs from an axis-aligned XYZ grid."""
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape [G, 3].")
    if len(grid) != 3 or min(grid) <= 0:
        raise ValueError("grid must contain three positive dimensions.")
    if means.shape[0] == 0:
        return torch.empty(
            (0,),
            device=means.device,
            dtype=dtype,
        )
    position_min = means.amin(dim=0)
    position_max = means.amax(dim=0)
    extent = (position_max - position_min).clamp_min(torch.finfo(means.dtype).eps)
    grid_tensor = torch.tensor(
        grid,
        device=means.device,
        dtype=means.dtype,
    )
    bins = torch.floor((means - position_min) / extent * grid_tensor).to(torch.int64)
    maxima = (
        torch.tensor(
            grid,
            device=means.device,
            dtype=torch.int64,
        )
        - 1
    )
    bins = torch.minimum(torch.maximum(bins, torch.zeros_like(bins)), maxima)
    semantic_ids = bins[:, 0] + grid[0] * (bins[:, 1] + grid[1] * bins[:, 2])
    return semantic_ids.to(dtype=dtype).contiguous()


def compact_semantic_ids(
    semantic_ids: torch.Tensor,
    *,
    dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, list[int]]:
    """Compact occupied non-negative labels to dense deterministic ordinals."""
    if semantic_ids.ndim != 1:
        raise ValueError("semantic_ids must have shape [G].")
    if semantic_ids.numel() == 0:
        return (
            torch.empty(
                (0,),
                device=semantic_ids.device,
                dtype=dtype,
            ),
            [],
        )
    if bool((semantic_ids < 0).any().item()):
        raise ValueError("semantic_ids must be non-negative.")
    occupied = torch.unique(semantic_ids.to(torch.int64), sorted=True)
    dense = torch.searchsorted(
        occupied,
        semantic_ids.to(torch.int64),
    ).to(dtype=dtype)
    return dense.contiguous(), occupied.detach().cpu().tolist()


def camera_bundle(
    batch_size: int,
    width: int,
    height: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    radius: float = 2.25,
    z_offset: float = 4.0,
    focal_scale: float = 0.9,
) -> CameraBundle:
    """Create deterministic OpenCV world-to-camera matrices and intrinsics."""
    device = torch.device(device)
    angles = torch.linspace(-0.45, 0.45, batch_size, device=device, dtype=dtype)
    viewmats = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
    viewmats[:, 0, 3] = -torch.sin(angles) * radius
    viewmats[:, 1, 3] = -torch.cos(angles) * 0.25
    viewmats[:, 2, 3] = -z_offset

    focal = focal_scale * float(width)
    intrinsics = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal * float(height) / float(width)
    intrinsics[:, 0, 2] = float(width) * 0.5
    intrinsics[:, 1, 2] = float(height) * 0.5
    intrinsics[:, 2, 2] = 1.0

    manifest = {
        "version": MANIFEST_VERSION,
        "camera_model": "opencv-pinhole-world-to-camera",
        "batch_size": batch_size,
        "width": width,
        "height": height,
        "radius": radius,
        "z_offset": z_offset,
        "focal_scale": focal_scale,
        "angle_start_rad": -0.45,
        "angle_end_rad": 0.45,
        "angle_step_rad": 0.0 if batch_size == 1 else 0.9 / (batch_size - 1),
        "near_plane": 0.01,
        "far_plane": 100.0,
    }
    manifest["checksum_sha256"] = sha256_json(manifest)
    return CameraBundle(viewmats=viewmats.contiguous(), intrinsics=intrinsics.contiguous(), manifest=manifest)


def axis_aligned_exterior_camera_bundle(
    *,
    center: torch.Tensor,
    bounding_radius: float,
    batch_size: int,
    width: int,
    height: int,
    focal_scale: float = 0.9,
    fit_margin: float = 1.15,
    camera_path: str = "axis-aligned-exterior",
    bounds_policy: str = "full-componentwise-min-max",
) -> CameraBundle:
    """Create an axis-aligned exterior bundle shared by custom and OVRTX.

    OpenCV cameras look along positive camera Z. The camera centers are placed
    behind the scene along world negative Z, and the fixed orientation keeps
    the USD/OVRTX camera conversion unambiguous.
    """
    if center.shape != (3,):
        raise ValueError(f"center must have shape [3], got {tuple(center.shape)}.")
    if bounding_radius <= 0 or batch_size <= 0 or width <= 0 or height <= 0 or focal_scale <= 0 or fit_margin <= 0:
        raise ValueError("Camera geometry and dimensions must be positive.")

    device = center.device
    dtype = center.dtype
    focal_x = focal_scale * float(width)
    focal_y = focal_x * float(height) / float(width)
    half_fov_x = math.atan(float(width) / (2.0 * focal_x))
    half_fov_y = math.atan(float(height) / (2.0 * focal_y))
    distance = fit_margin * bounding_radius / min(math.tan(half_fov_x), math.tan(half_fov_y))
    offsets = (
        torch.zeros((1,), device=device, dtype=dtype)
        if batch_size == 1
        else torch.linspace(
            -0.10 * bounding_radius,
            0.10 * bounding_radius,
            batch_size,
            device=device,
            dtype=dtype,
        )
    )
    camera_centers = center.repeat(batch_size, 1)
    camera_centers[:, 0] += offsets
    camera_centers[:, 2] -= distance

    viewmats = torch.eye(
        4,
        device=device,
        dtype=dtype,
    ).repeat(batch_size, 1, 1)
    viewmats[:, :3, 3] = -camera_centers
    intrinsics = torch.zeros(
        (batch_size, 3, 3),
        device=device,
        dtype=dtype,
    )
    intrinsics[:, 0, 0] = focal_x
    intrinsics[:, 1, 1] = focal_y
    intrinsics[:, 0, 2] = float(width) * 0.5
    intrinsics[:, 1, 2] = float(height) * 0.5
    intrinsics[:, 2, 2] = 1.0
    far_plane = max(
        100.0,
        distance + bounding_radius * 2.0,
    )
    manifest = {
        "version": MANIFEST_VERSION,
        "camera_model": "opencv-pinhole-world-to-camera",
        "camera_path": camera_path,
        "bounds_policy": bounds_policy,
        "batch_size": batch_size,
        "width": width,
        "height": height,
        "focal_scale": focal_scale,
        "fit_margin": fit_margin,
        "center": center.detach().cpu().tolist(),
        "bounding_radius": bounding_radius,
        "distance": distance,
        "x_offset_start": float(offsets[0].item()),
        "x_offset_end": float(offsets[-1].item()),
        "near_plane": 0.01,
        "far_plane": far_plane,
    }
    manifest["checksum_sha256"] = sha256_json(manifest)
    return CameraBundle(
        viewmats=viewmats.contiguous(),
        intrinsics=intrinsics.contiguous(),
        manifest=manifest,
    )


def home_scan_axis_camera_bundle(
    *,
    center: torch.Tensor,
    bounding_radius: float,
    batch_size: int,
    width: int,
    height: int,
    focal_scale: float = 0.9,
    fit_margin: float = 1.15,
) -> CameraBundle:
    """Compatibility wrapper for the immutable Home Scan camera path."""
    return axis_aligned_exterior_camera_bundle(
        center=center,
        bounding_radius=bounding_radius,
        batch_size=batch_size,
        width=width,
        height=height,
        focal_scale=focal_scale,
        fit_margin=fit_margin,
        camera_path="home-scan-axis-aligned-exterior",
        bounds_policy="full-componentwise-min-max",
    )


def synthetic_scene_tensors(
    gaussian_count: int,
    *,
    seed: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    means = torch.randn((gaussian_count, 3), generator=generator, device=device)
    means[:, 0] *= 3.0
    means[:, 1] *= 3.0
    means[:, 2] = means[:, 2].abs() * 1.5 + 4.0
    quats = torch.zeros((gaussian_count, 4), device=device)
    quats[:, 0] = 1.0
    scales = torch.full((gaussian_count, 3), 0.025, device=device)
    opacities = torch.full((gaussian_count,), 0.65, device=device)
    colors = torch.rand((gaussian_count, 3), generator=generator, device=device)
    semantic_ids = deterministic_semantic_ids(
        gaussian_count,
        device=device,
        dtype=torch.int32,
    )
    return {
        "means": means.contiguous(),
        "quats": quats.contiguous(),
        "scales": scales.contiguous(),
        "opacities": opacities.contiguous(),
        "colors": colors.contiguous(),
        "semantic_ids": semantic_ids.contiguous(),
    }


def projection_activation_scene_tensors(
    *,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Return an in-frame one-ParticleField projection-sensitivity scene."""
    device = torch.device(device)
    values = {
        "means": [
            [-0.37867245, -0.39488822, 5.92],
            [-1.52867245, -0.07488822, 6.20],
            [-0.52867245, 0.77511178, 6.50],
            [-1.57867245, 0.72511178, 6.80],
        ],
        "quats": [
            [0.92387953, 0.0, 0.38268343, 0.0],
            [0.96592583, 0.25881905, 0.0, 0.0],
            [0.86602540, 0.0, 0.0, 0.5],
            [0.75, 0.25, 0.5, 0.35355339],
        ],
        "scales": [
            [0.10, 0.025, 0.18],
            [0.03, 0.12, 0.05],
            [0.05, 0.03, 0.16],
            [0.13, 0.03, 0.06],
        ],
        "opacities": [0.70, 0.80, 0.65, 0.75],
        "colors": [
            [0.95, 0.15, 0.10],
            [0.10, 0.90, 0.20],
            [0.15, 0.25, 0.95],
            [0.95, 0.80, 0.10],
        ],
    }
    return {
        "means": torch.tensor(values["means"], device=device, dtype=torch.float32).contiguous(),
        "quats": torch.tensor(values["quats"], device=device, dtype=torch.float32).contiguous(),
        "scales": torch.tensor(values["scales"], device=device, dtype=torch.float32).contiguous(),
        "opacities": torch.tensor(values["opacities"], device=device, dtype=torch.float32).contiguous(),
        "colors": torch.tensor(values["colors"], device=device, dtype=torch.float32).contiguous(),
        # One semantic class deliberately produces one ParticleField.  The
        # projection oracle must not depend on multi-prim ingestion behavior.
        "semantic_ids": torch.zeros(4, device=device, dtype=torch.int32),
    }


def projection_activation_scene_manifest() -> dict[str, Any]:
    tensor_checksum = scene_tensors_sha256(projection_activation_scene_tensors(device="cpu"))
    manifest = {
        "version": MANIFEST_VERSION,
        "scene_id": "projection-activation",
        "scene_type": "controlled-projection-activation",
        "gaussian_count": 4,
        "attributes": [
            "means",
            "quats",
            "scales",
            "opacities",
            "colors",
            "semantic_ids",
        ],
        "requirements": {
            "anisotropic_scales": True,
            "nonidentity_wxyz_rotations": True,
            "off_axis_camera": True,
            "fully_in_front": True,
            "conventional_three_sigma_support_in_frame": True,
            "single_particle_field": True,
            "distinct_depth_color_opacity": True,
        },
        "semantic_id_rule": {
            "formula": "semantic_id[i] = 0",
            "class_count": 1,
        },
        "tensor_checksum_sha256": tensor_checksum,
    }
    manifest["checksum_sha256"] = sha256_json(manifest)
    return manifest


def stress_scene_tensors(
    scene_id: str,
    gaussian_count: int,
    *,
    seed: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Generate rotated, anisotropic scenes that are intentionally hard to rasterize."""
    if scene_id not in STRESS_SCENES:
        raise ValueError(f"Unsupported stress scene: {scene_id!r}.")
    if gaussian_count <= 0:
        raise ValueError("gaussian_count must be positive.")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    quats = torch.randn((gaussian_count, 4), generator=generator, device=device)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    # q and -q encode the same rotation. Fixing the sign makes byte-level
    # artifacts stable if the generator implementation is held constant.
    quats = torch.where(quats[:, :1] < 0, -quats, quats)

    if scene_id == "stress-anisotropic":
        means = torch.randn((gaussian_count, 3), generator=generator, device=device)
        means[:, :2] *= 1.8
        means[:, 2] = means[:, 2].abs() * 1.25 + 4.0
        base_scales = torch.tensor(
            [0.0125, 0.05, 0.18],
            device=device,
            dtype=torch.float32,
        )
        axis_roll = torch.arange(gaussian_count, device=device).remainder(3)
        axis_indices = (torch.arange(3, device=device).unsqueeze(0) - axis_roll.unsqueeze(1)).remainder(3)
        scales = base_scales[axis_indices]
        scales *= torch.exp(
            0.35
            * torch.randn(
                (gaussian_count, 1),
                generator=generator,
                device=device,
            )
        )
        opacities = 0.35 + 0.60 * torch.rand(
            (gaussian_count,),
            generator=generator,
            device=device,
        )
    else:
        means = torch.empty((gaussian_count, 3), device=device)
        means[:, :2] = 0.35 * torch.randn(
            (gaussian_count, 2),
            generator=generator,
            device=device,
        )
        means[:, 2] = 4.25 + 2.0 * torch.rand(
            (gaussian_count,),
            generator=generator,
            device=device,
        )
        scale_min = torch.tensor(
            [0.08, 0.08, 0.015],
            device=device,
            dtype=torch.float32,
        )
        scale_max = torch.tensor(
            [0.30, 0.30, 0.05],
            device=device,
            dtype=torch.float32,
        )
        scales = scale_min + (scale_max - scale_min) * torch.rand(
            (gaussian_count, 3),
            generator=generator,
            device=device,
        )
        opacities = 0.80 + 0.19 * torch.rand(
            (gaussian_count,),
            generator=generator,
            device=device,
        )

    colors = torch.rand((gaussian_count, 3), generator=generator, device=device)
    semantic_ids = spatial_semantic_ids(
        means,
        grid=(2, 2, 2),
        dtype=torch.int32,
    )
    return {
        "means": means.contiguous(),
        "quats": quats.contiguous(),
        "scales": scales.contiguous(),
        "opacities": opacities.contiguous(),
        "colors": colors.contiguous(),
        "semantic_ids": semantic_ids.contiguous(),
    }


def tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors.values()))


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan, "p50": math.nan, "p95": math.nan, "ci95": math.nan}
    sorted_values = sorted(values)
    n = len(sorted_values)
    mean = sum(sorted_values) / n
    variance = sum((value - mean) ** 2 for value in sorted_values) / max(1, n - 1)
    std = math.sqrt(variance)

    def percentile(p: float) -> float:
        if n == 1:
            return sorted_values[0]
        index = (n - 1) * p
        lo = math.floor(index)
        hi = math.ceil(index)
        if lo == hi:
            return sorted_values[lo]
        return sorted_values[lo] * (hi - index) + sorted_values[hi] * (index - lo)

    return {
        "mean": mean,
        "std": std,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "ci95": 1.96 * std / math.sqrt(n),
    }
