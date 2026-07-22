"""Loader for the standard INRIA 3D Gaussian Splatting PLY layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from .scene import GaussianScene

SH_C0 = 0.28209479177387814


def ply_vertex_property_names(path: str | Path) -> tuple[str, ...]:
    """Read vertex property names from a PLY header without loading its body."""
    names: list[str] = []
    in_vertex_element = False
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("PLY header must begin with the 'ply' magic line.")
        while stream.tell() <= 1024 * 1024:
            raw_line = stream.readline()
            if not raw_line:
                break
            try:
                parts = raw_line.decode("ascii").strip().split()
            except UnicodeDecodeError as error:
                raise ValueError("PLY header is not ASCII.") from error
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "end_header":
                return tuple(names)
            if parts[0] == "element":
                in_vertex_element = len(parts) >= 2 and parts[1] == "vertex"
                continue
            if parts[0] == "property" and in_vertex_element:
                if len(parts) == 3:
                    names.append(parts[2])
                elif len(parts) == 5 and parts[1] == "list":
                    names.append(parts[4])
                else:
                    raise ValueError(f"Malformed PLY property line: {raw_line!r}")
        raise ValueError("PLY header lacks end_header within the first MiB.")


def load_ply_to_gaussians(path: str | Path) -> GaussianScene:
    """Load raw 3DGS parameters from a PLY file into CPU tensors."""
    vertex = PlyData.read(str(path)).elements[0]
    count = len(vertex)

    def stack(names: list[str]) -> np.ndarray:
        return np.stack([np.asarray(vertex[name]) for name in names], axis=1)

    property_names = [prop.name for prop in vertex.properties]
    means = stack(["x", "y", "z"])
    opacities = np.asarray(vertex["opacity"])
    sh0 = stack(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]

    rest_names = sorted(
        (name for name in property_names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(rest_names) % 3:
        raise ValueError(f"Expected an RGB-multiple number of f_rest fields, got {len(rest_names)}.")
    if rest_names:
        shn = stack(rest_names).reshape(count, 3, len(rest_names) // 3).swapaxes(1, 2)
    else:
        shn = np.zeros((count, 0, 3), dtype=np.float32)

    scale_names = sorted(
        (name for name in property_names if name.startswith("scale_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    rotation_names = sorted(
        (name for name in property_names if name.startswith("rot_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    semantic = np.asarray(vertex["semantic_id"]) if "semantic_id" in property_names else np.zeros((count,), dtype=np.int64)
    features = np.concatenate((sh0, shn), axis=1).reshape(count, -1)

    return GaussianScene(
        means=torch.from_numpy(np.ascontiguousarray(means)).float(),
        scales=torch.from_numpy(np.ascontiguousarray(stack(scale_names))).float(),
        rotations=torch.from_numpy(np.ascontiguousarray(stack(rotation_names))).float(),
        opacities=torch.from_numpy(np.ascontiguousarray(opacities)).float(),
        features=torch.from_numpy(np.ascontiguousarray(features)).float(),
        semantic_ids=torch.from_numpy(np.ascontiguousarray(semantic)).long(),
    )


def canonicalize_3dgs_scene(
    scene: GaussianScene,
    *,
    device: torch.device | str | None = None,
) -> GaussianScene:
    """Activate raw INRIA 3DGS parameters for inference/rendering.

    The returned scene uses positive scales, opacity probabilities, normalized
    WXYZ quaternions, and linear RGB in the first three feature channels. Any
    higher-order SH coefficients remain in their original trailing channels.
    """
    target = torch.device(device) if device is not None else scene.means.device
    means = scene.means.to(device=target, dtype=torch.float32).contiguous()
    scales = scene.scales.to(device=target, dtype=torch.float32).exp().contiguous()
    opacities = scene.opacities.to(device=target, dtype=torch.float32).sigmoid().contiguous()

    rotations = scene.rotations.to(device=target, dtype=torch.float32)
    norms = torch.linalg.vector_norm(rotations, dim=-1, keepdim=True)
    rotations = rotations / norms.clamp_min(torch.finfo(torch.float32).eps)
    invalid = norms[..., 0] <= torch.finfo(torch.float32).eps
    if invalid.any():
        rotations = rotations.clone()
        rotations[invalid] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=target,
            dtype=torch.float32,
        )
    rotations = rotations.contiguous()

    features = scene.features.to(device=target, dtype=torch.float32).clone()
    if features.shape[1] < 3:
        raise ValueError("Raw 3DGS scene must contain the three DC SH channels.")
    features[:, :3] = (0.5 + SH_C0 * features[:, :3]).clamp_(0.0, 1.0)
    features = features.contiguous()

    return GaussianScene(
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        features=features,
        semantic_ids=scene.semantic_ids.to(device=target, dtype=torch.int64).contiguous(),
    )


def load_ply_to_canonical_gaussians(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> GaussianScene:
    """Load a standard 3DGS PLY and return renderer-ready canonical tensors."""
    return canonicalize_3dgs_scene(load_ply_to_gaussians(path), device=device)
