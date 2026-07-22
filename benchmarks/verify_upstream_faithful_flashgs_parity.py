#!/usr/bin/env python3
"""Untimed parity check against byte-exact pinned public FlashGS.

This is a bounded correctness gate, not benchmark evidence. It deliberately
uses an off-axis, rotated-anisotropic degree-zero fixture that can expose
camera, quaternion, covariance, support, compositing, and RGB-conversion drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer import (  # noqa: E402
    RendererService,
    UpstreamFaithfulFlashGSBackend,
)
from isaacsim_gaussian_renderer.ply_loader import SH_C0  # noqa: E402
from isaacsim_gaussian_renderer.upstream_faithful_flashgs_native_loader import (  # noqa: E402
    load_pinned_upstream_flashgs_reference_extension,
)
from isaacsim_gaussian_renderer.upstream_faithful_flashgs_sources import (  # noqa: E402
    FLASHGS_UPSTREAM_COMMIT,
)


WIDTH = 32
HEIGHT = 32
SCENE_ID = 909


def upstream_camera() -> tuple[torch.Tensor, torch.Tensor]:
    angle = np.float32(0.17)
    cosine = np.float32(np.cos(angle))
    sine = np.float32(np.sin(angle))
    position = torch.tensor([0.14, -0.06, 0.11], dtype=torch.float32)
    # Public FlashGS consumes the cameras.json camera-to-world rotation.
    rotation = torch.tensor(
        [
            [cosine, 0.0, -sine],
            [0.0, 1.0, 0.0],
            [sine, 0.0, cosine],
        ],
        dtype=torch.float32,
    )
    return position, rotation


def service_viewmat_for_upstream_camera() -> torch.Tensor:
    position, camera_to_world = upstream_camera()
    world_to_camera = camera_to_world.transpose(0, 1)
    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[:3, :3] = world_to_camera
    viewmat[:3, 3] = -(world_to_camera @ position)
    return viewmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--expected-gpu-uuid", required=True)
    return parser.parse_args()


def fixture() -> dict[str, torch.Tensor]:
    means = torch.tensor(
        [
            [-0.42, -0.18, 2.4],
            [0.37, 0.24, 2.9],
            [0.03, -0.47, 3.3],
            [0.56, -0.11, 3.8],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    scales = torch.tensor(
        [
            [0.22, 0.08, 0.05],
            [0.07, 0.19, 0.11],
            [0.13, 0.06, 0.24],
            [0.09, 0.16, 0.04],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    rotations = torch.tensor(
        [
            [0.9238795, 0.0, 0.3826834, 0.0],
            [0.9659258, 0.2588190, 0.0, 0.0],
            [0.8660254, 0.0, 0.0, 0.5],
            [0.8923991, 0.2391176, 0.3696438, 0.0990458],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    colors = torch.tensor(
        [
            [0.82, 0.17, 0.26],
            [0.13, 0.74, 0.31],
            [0.21, 0.32, 0.88],
            [0.66, 0.53, 0.12],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    return {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "opacities": torch.tensor(
            [0.72, 0.61, 0.83, 0.54], dtype=torch.float32, device="cuda"
        ),
        "features": colors,
        "semantic_ids": torch.arange(4, dtype=torch.int64, device="cuda"),
    }


def upstream_covariances(
    scales: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Independent CPU transcription of public pybind.cpp computeCov3D."""
    scale_np = scales.detach().cpu().numpy().astype(np.float32, copy=False)
    rotation_np = rotations.detach().cpu().numpy().astype(np.float32, copy=False)
    result = np.empty((scale_np.shape[0], 6), dtype=np.float32)
    for index, (scale, quaternion) in enumerate(
        zip(scale_np, rotation_np, strict=True)
    ):
        r, x, y, z = (np.float32(value) for value in quaternion)
        one = np.float32(1.0)
        two = np.float32(2.0)
        # Written row-major, then transposed to match GLM's scalar constructor,
        # whose arguments populate columns.
        scalar_order = np.asarray(
            [
                one - two * (y * y + z * z),
                two * (x * y - r * z),
                two * (x * z + r * y),
                two * (x * y + r * z),
                one - two * (x * x + z * z),
                two * (y * z - r * x),
                two * (x * z - r * y),
                two * (y * z + r * x),
                one - two * (x * x + y * y),
            ],
            dtype=np.float32,
        ).reshape(3, 3)
        rotation = scalar_order.T
        matrix = np.diag(scale) @ rotation
        covariance = matrix.T @ matrix
        result[index] = (
            covariance[0, 0],
            covariance[0, 1],
            covariance[0, 2],
            covariance[1, 1],
            covariance[1, 2],
            covariance[2, 2],
        )
    return torch.from_numpy(result).to(device=scales.device)


def render_exact_upstream(
    native: Any,
    scene: dict[str, torch.Tensor],
    capacity: int,
) -> tuple[torch.Tensor, int]:
    sh = torch.zeros(
        (scene["means"].shape[0], 48), dtype=torch.float32, device="cuda"
    )
    sh[:, :3] = (scene["features"] - 0.5) / SH_C0
    covariance = upstream_covariances(scene["scales"], scene["rotations"])
    points_xy = torch.empty((scene["means"].shape[0], 2), device="cuda")
    rgb_depth = torch.empty((scene["means"].shape[0], 4), device="cuda")
    conic_opacity = torch.empty((scene["means"].shape[0], 4), device="cuda")
    keys_unsorted = torch.empty((capacity,), dtype=torch.int64, device="cuda")
    values_unsorted = torch.empty((capacity,), dtype=torch.int32, device="cuda")
    keys_sorted = torch.empty((capacity,), dtype=torch.int64, device="cuda")
    values_sorted = torch.empty((capacity,), dtype=torch.int32, device="cuda")
    sort_temp = torch.empty(
        (int(native.ops.get_sort_buffer_size(capacity)),),
        dtype=torch.int8,
        device="cuda",
    )
    ranges = torch.empty(
        ((WIDTH // 16) * (HEIGHT // 16), 2),
        dtype=torch.int32,
        device="cuda",
    )
    offset = torch.zeros((1,), dtype=torch.int32, device="cuda")
    position, rotation = upstream_camera()
    native.ops.preprocess(
        scene["means"],
        sh,
        scene["opacities"],
        covariance,
        WIDTH,
        HEIGHT,
        16,
        16,
        position,
        rotation,
        24.0,
        24.0,
        100.0,
        0.01,
        points_xy,
        rgb_depth,
        conic_opacity,
        keys_unsorted,
        values_unsorted,
        offset,
    )
    count = int(offset.cpu()[0])
    if count <= 0 or count >= capacity:
        raise RuntimeError(f"Parity fixture emitted invalid count {count}/{capacity}.")
    native.ops.sort_gaussian(
        count,
        WIDTH,
        HEIGHT,
        16,
        16,
        sort_temp,
        keys_unsorted,
        values_unsorted,
        keys_sorted,
        values_sorted,
    )
    output = torch.empty(
        (HEIGHT, WIDTH, 3), dtype=torch.int8, device="cuda"
    )
    native.ops.render_16x16(
        count,
        WIDTH,
        HEIGHT,
        points_xy,
        rgb_depth,
        conic_opacity,
        keys_sorted,
        values_sorted,
        ranges,
        torch.zeros((3,), dtype=torch.float32),
        output,
    )
    torch.cuda.synchronize()
    return output.view(torch.uint8), count


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def file_artifact(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def main() -> None:
    args = parse_args()
    if args.capacity <= 0:
        raise ValueError("capacity must be positive.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.expected_gpu_uuid:
        raise RuntimeError(
            "Parity requires CUDA_VISIBLE_DEVICES to be the one exact expected "
            f"GPU UUID {args.expected_gpu_uuid!r}."
        )
    scene = fixture()
    reference_native = load_pinned_upstream_flashgs_reference_extension()
    expected_u8, reference_intersections = render_exact_upstream(
        reference_native,
        scene,
        args.capacity,
    )

    backend = UpstreamFaithfulFlashGSBackend(max_intersections=args.capacity)
    service = RendererService(
        backend,
        height=HEIGHT,
        width=WIDTH,
        outputs=("rgb",),
        max_views=1,
    )
    service.initialize(stage=None, device="cuda")
    service.load_scene(SCENE_ID, **scene)
    viewmat = service_viewmat_for_upstream_camera().to(device="cuda").unsqueeze(0)
    intrinsic = torch.tensor(
        [[24.0, 0.0, 16.0], [0.0, 24.0, 16.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)
    outputs = service.render(
        viewmat,
        intrinsic,
        torch.tensor([SCENE_ID], dtype=torch.int64, device="cuda"),
    )
    service.synchronize()
    counters = backend.check_capacity(synchronize=False)
    actual = outputs["rgb"][0]
    expected = expected_u8.to(dtype=torch.float32) * (1.0 / 255.0)
    max_abs = float((actual - expected).abs().max().item())
    reconstructed_u8 = torch.round(actual * 255.0).to(dtype=torch.uint8)
    mismatch_count = int(torch.count_nonzero(reconstructed_u8 != expected_u8).item())
    passed = bool(
        mismatch_count == 0
        and max_abs <= 1.0e-7
        and counters["intersection_overflow"] == 0
        and counters["camera_contract_errors"] == 0
        and counters["generated_intersections"] == reference_intersections
    )
    result = {
        "schema_version": "upstream-faithful-flashgs-parity-v1",
        "pass": passed,
        "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
        "gpu_uuid": args.expected_gpu_uuid,
        "fixture": {
            "gaussians": int(scene["means"].shape[0]),
            "width": WIDTH,
            "height": HEIGHT,
            "anisotropic": True,
            "non_identity_rotations": True,
            "off_axis": True,
            "non_identity_camera": True,
            "degree": 0,
        },
        "reference": {
            "implementation": "byte-exact-pinned-public-FlashGS-extension",
            "native_build_contract": reference_native.__vgr_build_contract__,
            "native_extension": file_artifact(Path(reference_native.__file__)),
            "intersections": reference_intersections,
            "rgb_u8_sha256": tensor_sha256(expected_u8),
        },
        "candidate": {
            "implementation": "UpstreamFaithfulFlashGSBackend",
            "native_build_contract": backend._native.__vgr_build_contract__,
            "native_extension": file_artifact(Path(backend._native.__file__)),
            "execution_stats": backend.execution_stats,
            "counters": counters,
            "rgb_float32_sha256": tensor_sha256(actual),
        },
        "comparison": {
            "reconstructed_u8_mismatch_count": mismatch_count,
            "float32_max_abs_error": max_abs,
            "float32_tolerance": 1.0e-7,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    service.shutdown()
    print("UPSTREAM_FAITHFUL_FLASHGS_PARITY " + json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
