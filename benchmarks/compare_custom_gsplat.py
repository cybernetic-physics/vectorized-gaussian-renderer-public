"""Reproducible all-output fidelity control: custom CUDA renderer vs gsplat."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    STRESS_SCENES,
    camera_bundle,
    stress_scene_manifest,
    stress_scene_tensors,
    stats,
    synthetic_scene_manifest,
    synthetic_scene_tensors,
)
from isaacsim_gaussian_renderer.fidelity import (
    RenderOutput,
    bundle_from_tensors,
    compare_render_outputs,
    write_camera_bundle,
)


SYNTHETIC_SCENES = {
    "synthetic-small": (10_000, 42),
    "synthetic-medium": (100_000, 43),
    **STRESS_SCENES,
}
PINNED_GSPLAT_COMMIT = "77ab983ffe43420b2131669cb35776b883ca4c3c"
PINNED_GSPLAT_COMPATIBILITY_PATCH_SHA256 = (
    "ea30120f728d5c728e082fec686f15056de5813a4e8fcc2c6b4e68aa324ca36d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(SYNTHETIC_SCENES), default="synthetic-small")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument(
        "--visible-capacity",
        type=int,
        help=(
            "Explicit custom visible-record capacity. Defaults to batch * "
            "Gaussian count. Use this to separate capacity sensitivity from "
            "scene work in performance audits."
        ),
    )
    parser.add_argument(
        "--intersection-capacity",
        type=int,
        help=(
            "Explicit custom tile-intersection capacity. Defaults to batch * "
            "Gaussian count * --intersections-per-visible. Storage is reserved "
            "to this capacity, while emitted-prefix ordering processes only "
            "intersections emitted by the current render."
        ),
    )
    parser.add_argument("--intersections-per-visible", type=float, default=8.0)
    parser.add_argument(
        "--capacity-regime",
        choices=(
            "production-default",
            "conservative-non-oracle",
            "hindsight-oracle-diagnostic",
            "explicit-diagnostic",
        ),
        default="explicit-diagnostic",
        help=(
            "Required provenance for the custom workspace reservation. "
            "Only production-default is eligible for production acceptance."
        ),
    )
    parser.add_argument("--semantic-min-alpha", type=float, default=0.01)
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Alternating-order timed iterations per implementation; zero disables timing.",
    )
    parser.add_argument("--require-lpips", action="store_true")
    parser.add_argument(
        "--verify-antialias-fixture",
        action="store_true",
        help=(
            "Also run a one-Gaussian anisotropic subpixel fixture and check "
            "the projected opacity against the analytic Mip-Splatting value "
            "and pinned gsplat. Requires --rasterize-mode antialiased."
        ),
    )
    parser.add_argument(
        "--gsplat-compatibility-patch",
        type=Path,
        help=(
            "Allow the pinned gsplat checkout to contain exactly the known "
            "CUDA-event compatibility patch. Arbitrary dirty, staged, "
            "untracked, or submodule state remains an error."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _render_output(
    *,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    semantic: torch.Tensor,
    camera_bundle_id: str,
    source: str,
) -> RenderOutput:
    alpha_3d = alpha[..., 0] if alpha.ndim == 4 else alpha
    depth_3d = depth[..., 0] if depth.ndim == 4 else depth
    semantic_3d = semantic[..., 0] if semantic.ndim == 4 else semantic
    output = RenderOutput(
        rgb=_to_numpy(rgb).astype(np.float32, copy=False),
        alpha=_to_numpy(alpha_3d).astype(np.float32, copy=False),
        depth=_to_numpy(depth_3d).astype(np.float32, copy=False),
        semantic=_to_numpy(semantic_3d).astype(np.int64, copy=False),
        valid_depth=_to_numpy(alpha_3d > 0),
        color_space="linear_rgb",
        background=(0.0, 0.0, 0.0),
        camera_bundle_id=camera_bundle_id,
        source=source,
    )
    output.validate()
    return output


def _save_output(path: Path, output: RenderOutput) -> None:
    np.savez_compressed(
        path,
        rgb=output.rgb,
        alpha=output.alpha,
        depth=output.depth,
        semantic=output.semantic,
        valid_depth=output.valid_depth,
        color_space=np.asarray(output.color_space),
        background=np.asarray(output.background, dtype=np.float32),
        camera_bundle_id=np.asarray(output.camera_bundle_id),
    )


def _zero_capacity_overflow(counters: dict[str, Any]) -> bool:
    """Return whether both production capacity counters stayed at zero."""
    return bool(
        counters.get("visible_overflow") == 0
        and counters.get("intersection_overflow") == 0
    )


def _comparison_pass(
    *,
    fidelity_pass: bool,
    antialias_fixture: dict[str, Any] | None,
    custom_counters: dict[str, Any],
) -> bool:
    """Gate the comparison verdict on fidelity, fixtures, and no overflow."""
    return bool(
        fidelity_pass
        and (
            antialias_fixture is None
            or antialias_fixture.get("pass") is True
        )
        and _zero_capacity_overflow(custom_counters)
    )


def _require_passing_summary(summary: dict[str, Any]) -> None:
    """Make a failing persisted verdict observable to shell automation."""
    if summary.get("pass") is not True:
        raise SystemExit(1)


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _isolated_git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    ):
        environment.pop(name, None)
    return environment


def _git_command(
    path: Path,
    *arguments: str,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env if env is not None else _isolated_git_environment(),
    )
    return completed.stdout.strip("\n")


def _temporary_index_tree(
    source_path: Path,
    *,
    compatibility_patch: Path | None,
) -> str:
    with tempfile.TemporaryDirectory(prefix="vgr-gsplat-index-") as directory:
        index_path = Path(directory) / "index"
        env = _isolated_git_environment()
        env["GIT_INDEX_FILE"] = str(index_path)
        _git_command(source_path, "read-tree", "HEAD", env=env)
        if compatibility_patch is None:
            _git_command(source_path, "add", "-u", "--", ".", env=env)
        else:
            _git_command(
                source_path,
                "apply",
                "--cached",
                "--whitespace=nowarn",
                str(compatibility_patch),
                env=env,
            )
        return _git_command(source_path, "write-tree", env=env)


def _validate_pinned_gsplat_checkout(
    source_path: Path,
    *,
    expected_commit: str,
    compatibility_patch: Path | None,
    expected_compatibility_patch_sha256: str,
) -> dict[str, Any]:
    commit = _git_command(source_path, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(
            "The gsplat parity control must use pinned commit "
            f"{expected_commit}; got {commit!r} from {source_path}."
        )

    status_output = _git_command(
        source_path,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    status_lines = status_output.splitlines() if status_output else []
    submodule_output = _git_command(
        source_path,
        "submodule",
        "status",
        "--recursive",
    )
    submodule_lines = submodule_output.splitlines() if submodule_output else []
    invalid_submodules = [
        line
        for line in submodule_lines
        if not line.startswith(" ")
    ]
    if invalid_submodules:
        raise RuntimeError(
            "The pinned gsplat checkout has an uninitialized, conflicted, or "
            "wrong-commit submodule: "
            + "; ".join(invalid_submodules)
        )
    try:
        _git_command(
            source_path,
            "submodule",
            "foreach",
            "--recursive",
            "--quiet",
            (
                "test -z \"$(git status --porcelain=v1 "
                "--untracked-files=all)\""
            ),
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "The pinned gsplat checkout contains a dirty submodule."
        ) from error

    patch_record: dict[str, Any] | None = None
    if compatibility_patch is not None:
        compatibility_patch = compatibility_patch.resolve()
        if not compatibility_patch.is_file():
            raise RuntimeError(
                "The configured gsplat compatibility patch does not exist: "
                f"{compatibility_patch}."
            )
        patch_sha256 = _sha256_file(compatibility_patch)
        if patch_sha256 != expected_compatibility_patch_sha256:
            raise RuntimeError(
                "The configured gsplat compatibility patch hash does not "
                "match the project allowlist; expected "
                f"{expected_compatibility_patch_sha256}, got {patch_sha256}."
            )
        expected_tree = _temporary_index_tree(
            source_path,
            compatibility_patch=compatibility_patch,
        )
        patch_record = {
            "path": str(compatibility_patch),
            "sha256": patch_sha256,
            "expected_tree": expected_tree,
            "applied": bool(status_lines),
        }

    if status_lines:
        if compatibility_patch is None:
            raise RuntimeError(
                "The pinned gsplat checkout is dirty and no allowed "
                "compatibility patch was declared: "
                + "; ".join(status_lines)
            )
        if any(not line.startswith(" M ") for line in status_lines):
            raise RuntimeError(
                "The allowed gsplat compatibility state may contain only "
                "unstaged modifications to tracked files; got: "
                + "; ".join(status_lines)
            )
        actual_tree = _temporary_index_tree(
            source_path,
            compatibility_patch=None,
        )
        assert patch_record is not None
        patch_record["actual_tree"] = actual_tree
        if actual_tree != patch_record["expected_tree"]:
            raise RuntimeError(
                "The dirty gsplat checkout does not exactly match the "
                "allowlisted compatibility patch."
            )

    return {
        "commit": commit,
        "source_path": str(source_path),
        "worktree_clean": not status_lines,
        "worktree_status": status_lines,
        "submodule_status": submodule_lines,
        "compatibility_patch": patch_record,
    }


def _mip_splat_compensated_opacity(
    *,
    opacity: float,
    covariance_00: float,
    covariance_01: float,
    covariance_11: float,
    covariance_epsilon: float,
) -> tuple[float, float]:
    determinant = (
        covariance_00 * covariance_11
        - covariance_01 * covariance_01
    )
    blurred_determinant = (
        (covariance_00 + covariance_epsilon)
        * (covariance_11 + covariance_epsilon)
        - covariance_01 * covariance_01
    )
    compensation = max(
        0.005 * 0.005,
        determinant / blurred_determinant,
    ) ** 0.5
    return opacity * compensation, compensation


def _rotation_matrix_wxyz(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    values /= np.linalg.norm(values)
    w, x, y, z = values
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _projected_covariance_for_fixture(
    *,
    mean: tuple[float, float, float],
    scales: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    focal_x: float,
    focal_y: float,
) -> tuple[float, float, float]:
    mean_array = np.asarray(mean, dtype=np.float64)
    if mean_array[2] <= 0:
        raise ValueError("Fixture mean must be in front of the camera.")
    rotation = _rotation_matrix_wxyz(quaternion)
    covariance_3d = (
        rotation
        @ np.diag(np.square(np.asarray(scales, dtype=np.float64)))
        @ rotation.T
    )
    inverse_z = 1.0 / mean_array[2]
    jacobian = np.asarray(
        [
            [
                focal_x * inverse_z,
                0.0,
                -focal_x * mean_array[0] * inverse_z * inverse_z,
            ],
            [
                0.0,
                focal_y * inverse_z,
                -focal_y * mean_array[1] * inverse_z * inverse_z,
            ],
        ],
        dtype=np.float64,
    )
    covariance_2d = jacobian @ covariance_3d @ jacobian.T
    return (
        float(covariance_2d[0, 0]),
        float(covariance_2d[0, 1]),
        float(covariance_2d[1, 1]),
    )


@torch.inference_mode()
def _render_native_single_gaussian(
    *,
    device: torch.device,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    mean: tuple[float, float, float],
    scales: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    opacity: float,
    covariance_epsilon: float,
    rasterize_mode: str,
) -> dict[str, Any]:
    backend = CustomCudaBackend(
        max_visible_records=4,
        max_intersections=max(64, width * height),
        gaussian_support_sigma=3.33,
        covariance_epsilon=covariance_epsilon,
        rasterize_mode=rasterize_mode,  # type: ignore[arg-type]
        output_srgb=False,
        adaptive_capacity=False,
        deterministic=True,
    )
    service = RendererService(
        backend,
        height=height,
        width=width,
        max_views=1,
    )
    means = torch.tensor([mean], device=device, dtype=torch.float32)
    scale_tensor = torch.tensor([scales], device=device, dtype=torch.float32)
    quaternions = torch.tensor(
        [quaternion],
        device=device,
        dtype=torch.float32,
    )
    opacities = torch.tensor([opacity], device=device, dtype=torch.float32)
    colors = torch.tensor(
        [[0.8, 0.3, 0.1]],
        device=device,
        dtype=torch.float32,
    )
    viewmats = torch.eye(
        4,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).contiguous()
    intrinsics = torch.tensor(
        [
            [focal_x, 0.0, width / 2.0],
            [0.0, focal_y, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).contiguous()
    try:
        service.initialize(stage=None, device=device)
        service.load_scene(
            1,
            means=means,
            scales=scale_tensor,
            rotations=quaternions,
            opacities=opacities,
            features=colors,
            semantic_ids=torch.tensor(
                [7],
                device=device,
                dtype=torch.int64,
            ),
        )
        outputs = service.render(
            viewmats,
            intrinsics,
            torch.tensor([1], device=device, dtype=torch.int64),
        )
        service.synchronize()
        counters = backend.check_capacity(synchronize=False)
        if counters["visible_gaussians"] != 1:
            raise RuntimeError(
                "Single-Gaussian antialias fixture expected one visible "
                f"record; got {counters['visible_gaussians']}."
            )
        if counters["visible_overflow"] or counters["intersection_overflow"]:
            raise RuntimeError(
                "Single-Gaussian antialias fixture overflowed its workspace."
            )
        conic = backend.workspace["visible_conics"][0].detach().cpu().numpy()
        inverse_determinant = conic[0] * conic[2] - conic[1] * conic[1]
        blurred_covariance = (
            float(conic[2] / inverse_determinant),
            float(-conic[1] / inverse_determinant),
            float(conic[0] / inverse_determinant),
        )
        alpha = outputs["alpha"][..., 0]
        covered = alpha > 0.0
        foreground = alpha >= 0.01
        depth = outputs["depth"][..., 0]
        semantic_id = outputs["semantic_id"][..., 0]
        output_contract = {
            "shapes": (
                tuple(outputs["rgb"].shape) == (1, height, width, 3)
                and tuple(outputs["depth"].shape) == (1, height, width, 1)
                and tuple(outputs["alpha"].shape) == (1, height, width, 1)
                and tuple(outputs["semantic_id"].shape)
                == (1, height, width, 1)
            ),
            "dtypes": (
                outputs["rgb"].dtype == torch.float32
                and outputs["depth"].dtype == torch.float32
                and outputs["alpha"].dtype == torch.float32
                and outputs["semantic_id"].dtype == torch.int64
            ),
            "cuda_resident": all(
                tensor.is_cuda for tensor in outputs.values()
            ),
            "finite_rgb": bool(torch.isfinite(outputs["rgb"]).all().item()),
            "finite_alpha": bool(torch.isfinite(alpha).all().item()),
            "finite_covered_depth": bool(
                covered.any().item()
                and torch.isfinite(depth[covered]).all().item()
            ),
            "alpha_bounds": bool(
                ((alpha >= 0.0) & (alpha <= 1.0)).all().item()
            ),
            "semantic_threshold_respected": bool(
                (semantic_id[foreground] == 7).all().item()
                and (semantic_id[~foreground] == -1).all().item()
            ),
        }
        if not all(output_contract.values()):
            raise RuntimeError(
                "Single-Gaussian antialias fixture violated the output "
                f"contract: {output_contract}."
            )
        return {
            "projected_opacity": float(
                backend.workspace["visible_opacities"][0]
            ),
            "blurred_covariance": blurred_covariance,
            "alpha_max": float(alpha.max()),
            "alpha_sum": float(alpha.sum()),
            "counters": counters,
            "output_contract": output_contract,
        }
    finally:
        service.shutdown()


@torch.inference_mode()
def _render_gsplat_single_gaussian(
    *,
    device: torch.device,
    rasterization: Any,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    mean: tuple[float, float, float],
    scales: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    opacity: float,
    covariance_epsilon: float,
) -> dict[str, float]:
    means = torch.tensor([mean], device=device, dtype=torch.float32)
    quaternions = torch.tensor(
        [quaternion],
        device=device,
        dtype=torch.float32,
    )
    scale_tensor = torch.tensor([scales], device=device, dtype=torch.float32)
    opacities = torch.tensor([opacity], device=device, dtype=torch.float32)
    colors = torch.tensor(
        [[0.8, 0.3, 0.1]],
        device=device,
        dtype=torch.float32,
    )
    viewmats = torch.eye(
        4,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).contiguous()
    intrinsics = torch.tensor(
        [
            [focal_x, 0.0, width / 2.0],
            [0.0, focal_y, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).contiguous()
    _rgbd, alpha, metadata = rasterization(
        means,
        quaternions,
        scale_tensor,
        opacities,
        colors,
        viewmats,
        intrinsics,
        width,
        height,
        near_plane=0.01,
        far_plane=100.0,
        eps2d=covariance_epsilon,
        radius_clip=0.0,
        packed=False,
        tile_size=16,
        render_mode="RGB+ED",
        rasterize_mode="antialiased",
        global_z_order=True,
    )
    return {
        "projected_opacity": float(metadata["opacities"].reshape(-1)[0]),
        "alpha_max": float(alpha.max()),
        "alpha_sum": float(alpha.sum()),
    }


@torch.inference_mode()
def _verify_antialias_fixture(
    *,
    device: torch.device,
    rasterization: Any,
) -> dict[str, Any]:
    covariance_epsilon = 0.3
    diagonal_case = {
        "width": 32,
        "height": 32,
        "focal_x": 64.0,
        "focal_y": 64.0,
        "mean": (0.0, 0.0, 2.0),
        "scales": (0.00625, 0.0125, 0.01),
        "quaternion": (1.0, 0.0, 0.0, 0.0),
        "opacity": 0.8,
    }
    raw_covariance_00, raw_covariance_01, raw_covariance_11 = (
        _projected_covariance_for_fixture(
            mean=diagonal_case["mean"],
            scales=diagonal_case["scales"],
            quaternion=diagonal_case["quaternion"],
            focal_x=diagonal_case["focal_x"],
            focal_y=diagonal_case["focal_y"],
        )
    )
    expected_opacity, expected_compensation = (
        _mip_splat_compensated_opacity(
            opacity=diagonal_case["opacity"],
            covariance_00=raw_covariance_00,
            covariance_01=raw_covariance_01,
            covariance_11=raw_covariance_11,
            covariance_epsilon=covariance_epsilon,
        )
    )
    diagonal_native = _render_native_single_gaussian(
        device=device,
        covariance_epsilon=covariance_epsilon,
        rasterize_mode="antialiased",
        **diagonal_case,
    )
    diagonal_gsplat = _render_gsplat_single_gaussian(
        device=device,
        rasterization=rasterization,
        covariance_epsilon=covariance_epsilon,
        **diagonal_case,
    )
    for actual in (
        diagonal_native["projected_opacity"],
        diagonal_gsplat["projected_opacity"],
    ):
        np.testing.assert_allclose(
            actual,
            expected_opacity,
            rtol=2.0e-5,
            atol=1.0e-6,
        )

    rotated_case = {
        "width": 64,
        "height": 48,
        "focal_x": 96.0,
        "focal_y": 88.0,
        "mean": (0.18, -0.12, 3.0),
        "scales": (0.006, 0.014, 0.01),
        "quaternion": (
            0.9659258262890683,
            0.0,
            0.0,
            0.25881904510252074,
        ),
        "opacity": 0.8,
    }
    rotated_covariance = _projected_covariance_for_fixture(
        mean=rotated_case["mean"],
        scales=rotated_case["scales"],
        quaternion=rotated_case["quaternion"],
        focal_x=rotated_case["focal_x"],
        focal_y=rotated_case["focal_y"],
    )
    if abs(rotated_covariance[1]) < 1.0e-4:
        raise RuntimeError(
            "Rotated off-axis fixture failed to produce a nonzero projected "
            "covariance cross-term."
        )
    rotated_expected_opacity, rotated_expected_compensation = (
        _mip_splat_compensated_opacity(
            opacity=rotated_case["opacity"],
            covariance_00=rotated_covariance[0],
            covariance_01=rotated_covariance[1],
            covariance_11=rotated_covariance[2],
            covariance_epsilon=covariance_epsilon,
        )
    )
    rotated_native = _render_native_single_gaussian(
        device=device,
        covariance_epsilon=covariance_epsilon,
        rasterize_mode="antialiased",
        **rotated_case,
    )
    rotated_gsplat = _render_gsplat_single_gaussian(
        device=device,
        rasterization=rasterization,
        covariance_epsilon=covariance_epsilon,
        **rotated_case,
    )
    for actual in (
        rotated_native["projected_opacity"],
        rotated_gsplat["projected_opacity"],
    ):
        np.testing.assert_allclose(
            actual,
            rotated_expected_opacity,
            rtol=2.0e-5,
            atol=1.0e-6,
        )
    expected_blurred_covariance = np.asarray(
        [
            rotated_covariance[0] + covariance_epsilon,
            rotated_covariance[1],
            rotated_covariance[2] + covariance_epsilon,
        ]
    )
    np.testing.assert_allclose(
        rotated_native["blurred_covariance"],
        expected_blurred_covariance,
        rtol=3.0e-4,
        atol=2.0e-6,
    )

    floor_case = {
        "width": 32,
        "height": 32,
        "focal_x": 64.0,
        "focal_y": 64.0,
        "mean": (0.015625, 0.015625, 2.0),
        "scales": (1.0e-6, 1.5e-6, 1.0e-6),
        "quaternion": (1.0, 0.0, 0.0, 0.0),
        "opacity": 0.99,
    }
    floor_covariance = _projected_covariance_for_fixture(
        mean=floor_case["mean"],
        scales=floor_case["scales"],
        quaternion=floor_case["quaternion"],
        focal_x=floor_case["focal_x"],
        focal_y=floor_case["focal_y"],
    )
    floor_expected_opacity, floor_compensation = (
        _mip_splat_compensated_opacity(
            opacity=floor_case["opacity"],
            covariance_00=floor_covariance[0],
            covariance_01=floor_covariance[1],
            covariance_11=floor_covariance[2],
            covariance_epsilon=covariance_epsilon,
        )
    )
    if floor_compensation != 0.005:
        raise RuntimeError(
            "Native compensation-floor fixture did not analytically select "
            f"the 0.005 floor; got {floor_compensation}."
        )
    floor_native = _render_native_single_gaussian(
        device=device,
        covariance_epsilon=covariance_epsilon,
        rasterize_mode="antialiased",
        **floor_case,
    )
    floor_gsplat = _render_gsplat_single_gaussian(
        device=device,
        rasterization=rasterization,
        covariance_epsilon=covariance_epsilon,
        **floor_case,
    )
    for actual in (
        floor_native["projected_opacity"],
        floor_gsplat["projected_opacity"],
    ):
        np.testing.assert_allclose(
            actual,
            floor_expected_opacity,
            rtol=2.0e-5,
            atol=1.0e-6,
        )

    resolution_records: list[dict[str, Any]] = []
    trend_mean = (0.15, -0.10, 3.0)
    trend_scales = (0.012, 0.020, 0.010)
    trend_quaternion = rotated_case["quaternion"]
    trend_opacity = 0.8
    for width, height in ((32, 24), (64, 48), (128, 96)):
        trend_case = {
            "width": width,
            "height": height,
            "focal_x": 1.5 * width,
            "focal_y": 1.5 * width,
            "mean": trend_mean,
            "scales": trend_scales,
            "quaternion": trend_quaternion,
            "opacity": trend_opacity,
        }
        covariance = _projected_covariance_for_fixture(
            mean=trend_mean,
            scales=trend_scales,
            quaternion=trend_quaternion,
            focal_x=trend_case["focal_x"],
            focal_y=trend_case["focal_y"],
        )
        expected_trend_opacity, compensation = (
            _mip_splat_compensated_opacity(
                opacity=trend_opacity,
                covariance_00=covariance[0],
                covariance_01=covariance[1],
                covariance_11=covariance[2],
                covariance_epsilon=covariance_epsilon,
            )
        )
        antialiased_native = _render_native_single_gaussian(
            device=device,
            covariance_epsilon=covariance_epsilon,
            rasterize_mode="antialiased",
            **trend_case,
        )
        classic_native = _render_native_single_gaussian(
            device=device,
            covariance_epsilon=covariance_epsilon,
            rasterize_mode="classic",
            **trend_case,
        )
        antialiased_gsplat = _render_gsplat_single_gaussian(
            device=device,
            rasterization=rasterization,
            covariance_epsilon=covariance_epsilon,
            **trend_case,
        )
        for actual in (
            antialiased_native["projected_opacity"],
            antialiased_gsplat["projected_opacity"],
        ):
            np.testing.assert_allclose(
                actual,
                expected_trend_opacity,
                rtol=2.0e-5,
                atol=1.0e-6,
            )
        np.testing.assert_allclose(
            classic_native["projected_opacity"],
            trend_opacity,
            rtol=2.0e-5,
            atol=1.0e-6,
        )
        resolution_records.append(
            {
                "resolution": [width, height],
                "raw_projected_covariance": list(covariance),
                "compensation": compensation,
                "expected_compensated_opacity": expected_trend_opacity,
                "custom_antialiased_opacity": antialiased_native[
                    "projected_opacity"
                ],
                "custom_classic_opacity": classic_native[
                    "projected_opacity"
                ],
                "gsplat_antialiased_opacity": antialiased_gsplat[
                    "projected_opacity"
                ],
                "custom_antialiased_alpha_max": antialiased_native[
                    "alpha_max"
                ],
                "custom_antialiased_alpha_sum": antialiased_native[
                    "alpha_sum"
                ],
            }
        )
    compensated_opacities = [
        record["custom_antialiased_opacity"]
        for record in resolution_records
    ]
    darkening = [
        trend_opacity - opacity
        for opacity in compensated_opacities
    ]
    if not all(
        compensated_opacities[index] < compensated_opacities[index + 1]
        for index in range(len(compensated_opacities) - 1)
    ):
        raise RuntimeError(
            "Antialiased projected opacity must increase as a fixed-FOV "
            "fixture gains image resolution."
        )
    if not all(
        darkening[index] > darkening[index + 1]
        for index in range(len(darkening) - 1)
    ):
        raise RuntimeError(
            "Mip-Splat darkening must decrease as the projected Gaussian "
            "gains pixel support."
        )

    return {
        "pass": True,
        "fixture": "one-anisotropic-subpixel-gaussian",
        "raw_projected_covariance": [
            raw_covariance_00,
            raw_covariance_01,
            raw_covariance_11,
        ],
        "covariance_epsilon": covariance_epsilon,
        "raw_opacity": diagonal_case["opacity"],
        "expected_compensation": expected_compensation,
        "expected_compensated_opacity": expected_opacity,
        "custom_projected_opacity": diagonal_native["projected_opacity"],
        "gsplat_projected_opacity": diagonal_gsplat["projected_opacity"],
        "counters": diagonal_native["counters"],
        "rotated_off_axis": {
            "pass": True,
            "raw_projected_covariance": list(rotated_covariance),
            "expected_compensation": rotated_expected_compensation,
            "expected_compensated_opacity": rotated_expected_opacity,
            "custom_projected_opacity": rotated_native["projected_opacity"],
            "gsplat_projected_opacity": rotated_gsplat["projected_opacity"],
            "custom_blurred_covariance": list(
                rotated_native["blurred_covariance"]
            ),
            "counters": rotated_native["counters"],
        },
        "compensation_floor": {
            "pass": True,
            "raw_projected_covariance": list(floor_covariance),
            "expected_compensation": floor_compensation,
            "expected_compensated_opacity": floor_expected_opacity,
            "custom_projected_opacity": floor_native["projected_opacity"],
            "gsplat_projected_opacity": floor_gsplat["projected_opacity"],
            "custom_alpha_max": floor_native["alpha_max"],
            "counters": floor_native["counters"],
        },
        "multi_resolution_darkening": {
            "pass": True,
            "fixed_fov": True,
            "records": resolution_records,
        },
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("batch and resolution must be positive.")
    if args.visible_capacity is not None and args.visible_capacity <= 0:
        raise ValueError("visible-capacity must be positive when provided.")
    if args.intersection_capacity is not None and args.intersection_capacity <= 0:
        raise ValueError("intersection-capacity must be positive when provided.")
    if not 0.0 <= args.semantic_min_alpha <= 1.0:
        raise ValueError("semantic-min-alpha must be in [0, 1].")
    if args.warmup < 0 or args.iterations < 0:
        raise ValueError("warmup and iterations must be non-negative.")
    if args.verify_antialias_fixture and args.rasterize_mode != "antialiased":
        raise ValueError(
            "--verify-antialias-fixture requires "
            "--rasterize-mode antialiased."
        )

    configured_gsplat_source_path = Path(
        os.environ.get("GSPLAT_SOURCE_PATH", "/workspace/src/gsplat")
    ).resolve()
    gsplat_spec = importlib.util.find_spec("gsplat")
    if gsplat_spec is None or gsplat_spec.origin is None:
        raise RuntimeError("Could not resolve the configured gsplat package.")
    discovered_gsplat_module_path = Path(gsplat_spec.origin).resolve()
    discovered_gsplat_source_path = discovered_gsplat_module_path.parent.parent
    if discovered_gsplat_source_path != configured_gsplat_source_path:
        raise RuntimeError(
            "The gsplat import spec must come from GSPLAT_SOURCE_PATH before "
            "the package is imported; resolved "
            f"{discovered_gsplat_module_path}, configured "
            f"{configured_gsplat_source_path}."
        )
    gsplat_provenance = _validate_pinned_gsplat_checkout(
        configured_gsplat_source_path,
        expected_commit=PINNED_GSPLAT_COMMIT,
        compatibility_patch=args.gsplat_compatibility_patch,
        expected_compatibility_patch_sha256=(
            PINNED_GSPLAT_COMPATIBILITY_PATCH_SHA256
        ),
    )

    import gsplat
    import gsplat.rendering as gsplat_rendering
    from gsplat.cuda._wrapper import rasterize_top_contributing_gaussian_ids
    from gsplat.rendering import rasterization

    gsplat_module_path = Path(gsplat.__file__ or "").resolve()
    gsplat_rendering_path = Path(gsplat_rendering.__file__ or "").resolve()
    gsplat_source_path = gsplat_module_path.parent.parent
    if gsplat_source_path != configured_gsplat_source_path:
        raise RuntimeError(
            "The imported gsplat module must come from GSPLAT_SOURCE_PATH; "
            f"imported {gsplat_module_path}, configured "
            f"{configured_gsplat_source_path}."
        )
    try:
        gsplat_rendering_path.relative_to(gsplat_module_path.parent)
    except ValueError as error:
        raise RuntimeError(
            "The imported gsplat.rendering module must share the pinned "
            f"package root; imported {gsplat_rendering_path}, package "
            f"{gsplat_module_path.parent}."
        ) from error
    gsplat_commit = gsplat_provenance["commit"]
    device = torch.device("cuda")
    gaussian_count, seed = SYNTHETIC_SCENES[args.scene]
    if args.scene in STRESS_SCENES:
        scene = stress_scene_tensors(
            args.scene,
            gaussian_count,
            seed=seed,
            device=device,
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
            device=device,
        )
        scene_manifest = synthetic_scene_manifest(
            args.scene,
            gaussian_count,
            seed,
        )
    cameras = camera_bundle(args.batch, args.width, args.height, device=device)
    scene_id = 101
    scene_ids = torch.full((args.batch,), scene_id, device=device, dtype=torch.int64)
    semantic_ids = scene["semantic_ids"].to(torch.int64)
    visible_capacity = args.visible_capacity or args.batch * gaussian_count
    intersection_capacity = args.intersection_capacity or int(
        args.batch * gaussian_count * args.intersections_per_visible
    )

    backend = CustomCudaBackend(
        max_visible_records=visible_capacity,
        max_intersections=intersection_capacity,
        gaussian_support_sigma=3.33,
        rasterize_mode=args.rasterize_mode,
        semantic_min_alpha=args.semantic_min_alpha,
        output_srgb=False,
    )
    service = RendererService(
        backend,
        height=args.height,
        width=args.width,
        max_views=args.batch,
    )
    service.initialize(stage=None, device=device)
    service.load_scene(
        scene_id,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=semantic_ids,
    )
    custom = service.render(cameras.viewmats, cameras.intrinsics, scene_ids)
    service.synchronize()
    custom_counters = backend.check_capacity(synchronize=False)
    custom_zero_capacity_overflow = _zero_capacity_overflow(custom_counters)

    reference_rgbd, reference_alpha, meta = rasterization(
        scene["means"],
        scene["quats"],
        scene["scales"],
        scene["opacities"],
        scene["colors"],
        cameras.viewmats,
        cameras.intrinsics,
        args.width,
        args.height,
        near_plane=0.01,
        far_plane=100.0,
        eps2d=0.3,
        radius_clip=0.0,
        packed=False,
        tile_size=16,
        render_mode="RGB+ED",
        rasterize_mode=args.rasterize_mode,
        global_z_order=True,
    )
    contributor_ids, _contributor_weights = rasterize_top_contributing_gaussian_ids(
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        meta["isect_offsets"],
        meta["flatten_ids"],
        args.width,
        args.height,
        meta["tile_size"],
        1,
    )
    reference_semantic = torch.full(
        contributor_ids.shape[:-1],
        -1,
        device=device,
        dtype=torch.int64,
    )
    valid_contributor = contributor_ids[..., 0] >= 0
    reference_semantic[valid_contributor] = semantic_ids[contributor_ids[..., 0][valid_contributor]]
    reference_semantic[reference_alpha[..., 0] < args.semantic_min_alpha] = -1
    torch.cuda.synchronize()

    timing = None
    if args.iterations:
        custom_gpu_ms: list[float] = []
        custom_wall_ms: list[float] = []
        gsplat_core_gpu_ms: list[float] = []
        gsplat_full_gpu_ms: list[float] = []
        gsplat_wall_ms: list[float] = []

        def render_custom() -> None:
            service.render(
                cameras.viewmats,
                cameras.intrinsics,
                scene_ids,
                outputs=custom,
            )

        def render_gsplat() -> None:
            rgbd, alpha, render_meta = rasterization(
                scene["means"],
                scene["quats"],
                scene["scales"],
                scene["opacities"],
                scene["colors"],
                cameras.viewmats,
                cameras.intrinsics,
                args.width,
                args.height,
                near_plane=0.01,
                far_plane=100.0,
                eps2d=0.3,
                radius_clip=0.0,
                packed=False,
                tile_size=16,
                render_mode="RGB+ED",
                rasterize_mode=args.rasterize_mode,
                global_z_order=True,
            )
            ids, _weights = rasterize_top_contributing_gaussian_ids(
                render_meta["means2d"],
                render_meta["conics"],
                render_meta["opacities"],
                render_meta["isect_offsets"],
                render_meta["flatten_ids"],
                args.width,
                args.height,
                render_meta["tile_size"],
                1,
            )
            semantic = torch.full(
                ids.shape[:-1],
                -1,
                device=device,
                dtype=torch.int64,
            )
            valid = ids[..., 0] >= 0
            semantic[valid] = semantic_ids[ids[..., 0][valid]]
            semantic[alpha[..., 0] < args.semantic_min_alpha] = -1
            # Retain the operations until they have been enqueued. The values
            # are intentionally not copied to the host inside measured ranges.
            _ = rgbd, semantic

        for index in range(args.warmup):
            if index % 2 == 0:
                render_custom()
                render_gsplat()
            else:
                render_gsplat()
                render_custom()
        torch.cuda.synchronize()

        for index in range(args.iterations):
            order = ("custom", "gsplat") if index % 2 == 0 else ("gsplat", "custom")
            for implementation in order:
                start = torch.cuda.Event(enable_timing=True)
                core_end = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                if implementation == "custom":
                    render_custom()
                    end.record()
                    end.synchronize()
                    elapsed = float(start.elapsed_time(end))
                    custom_gpu_ms.append(elapsed)
                    custom_wall_ms.append((time.perf_counter() - wall_start) * 1_000.0)
                else:
                    rgbd, alpha, render_meta = rasterization(
                        scene["means"],
                        scene["quats"],
                        scene["scales"],
                        scene["opacities"],
                        scene["colors"],
                        cameras.viewmats,
                        cameras.intrinsics,
                        args.width,
                        args.height,
                        near_plane=0.01,
                        far_plane=100.0,
                        eps2d=0.3,
                        radius_clip=0.0,
                        packed=False,
                        tile_size=16,
                        render_mode="RGB+ED",
                        rasterize_mode=args.rasterize_mode,
                        global_z_order=True,
                    )
                    core_end.record()
                    ids, _weights = rasterize_top_contributing_gaussian_ids(
                        render_meta["means2d"],
                        render_meta["conics"],
                        render_meta["opacities"],
                        render_meta["isect_offsets"],
                        render_meta["flatten_ids"],
                        args.width,
                        args.height,
                        render_meta["tile_size"],
                        1,
                    )
                    semantic = torch.full(
                        ids.shape[:-1],
                        -1,
                        device=device,
                        dtype=torch.int64,
                    )
                    valid = ids[..., 0] >= 0
                    semantic[valid] = semantic_ids[ids[..., 0][valid]]
                    semantic[alpha[..., 0] < args.semantic_min_alpha] = -1
                    end.record()
                    end.synchronize()
                    _ = rgbd, semantic
                    gsplat_core_gpu_ms.append(float(start.elapsed_time(core_end)))
                    gsplat_full_gpu_ms.append(float(start.elapsed_time(end)))
                    gsplat_wall_ms.append((time.perf_counter() - wall_start) * 1_000.0)

        custom_stats = stats(custom_gpu_ms)
        gsplat_core_stats = stats(gsplat_core_gpu_ms)
        gsplat_full_stats = stats(gsplat_full_gpu_ms)
        timing = {
            "measurement": (
                "CUDA events with per-call synchronization; implementation order alternates "
                "every iteration; warmup is excluded"
            ),
            "warmup_iterations_per_implementation": args.warmup,
            "measured_iterations_per_implementation": args.iterations,
            "custom_all_output_gpu_ms": {
                **custom_stats,
                "samples": custom_gpu_ms,
            },
            "custom_all_output_wall_ms": {
                **stats(custom_wall_ms),
                "samples": custom_wall_ms,
            },
            "gsplat_rgb_depth_alpha_gpu_ms": {
                **gsplat_core_stats,
                "samples": gsplat_core_gpu_ms,
            },
            "gsplat_all_output_gpu_ms": {
                **gsplat_full_stats,
                "samples": gsplat_full_gpu_ms,
            },
            "gsplat_all_output_wall_ms": {
                **stats(gsplat_wall_ms),
                "samples": gsplat_wall_ms,
            },
            "custom_speedup_vs_gsplat_rgb_depth_alpha": (gsplat_core_stats["mean"] / custom_stats["mean"]),
            "custom_speedup_vs_gsplat_all_output": (gsplat_full_stats["mean"] / custom_stats["mean"]),
            "note": (
                "The gsplat core figure omits semantic-ID selection and is not output-matched; "
                "the all-output figure includes its public top-contributor API and ID mapping."
            ),
        }

    fidelity_bundle = bundle_from_tensors(
        viewmats=cameras.viewmats,
        intrinsics=cameras.intrinsics,
        width=args.width,
        height=args.height,
        color_space="linear_rgb",
        scene_ids=scene_ids,
        scene_checksum=scene_manifest["checksum_sha256"],
    )
    reference = _render_output(
        rgb=reference_rgbd[..., :3],
        depth=reference_rgbd[..., 3:4],
        alpha=reference_alpha,
        semantic=reference_semantic,
        camera_bundle_id=fidelity_bundle.bundle_id,
        source="gsplat",
    )
    candidate = _render_output(
        rgb=custom["rgb"],
        depth=custom["depth"],
        alpha=custom["alpha"],
        semantic=custom["semantic_id"],
        camera_bundle_id=fidelity_bundle.bundle_id,
        source="custom-cuda",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_camera_bundle(fidelity_bundle, args.output / "camera-bundle.json")
    _save_output(args.output / "reference-gsplat.npz", reference)
    _save_output(args.output / "candidate-custom.npz", candidate)
    report = compare_render_outputs(
        reference=reference,
        candidate=candidate,
        camera_bundle=fidelity_bundle,
        output_dir=args.output / "report",
        config_id=(
            f"custom-vs-gsplat-{args.rasterize_mode}-{args.scene}-"
            f"b{args.batch}-{args.width}x{args.height}-all-output"
        ),
        require_lpips=args.require_lpips,
        max_artifact_views=min(args.batch, 8),
    )
    antialias_fixture = (
        _verify_antialias_fixture(
            device=device,
            rasterization=rasterization,
        )
        if args.verify_antialias_fixture
        else None
    )
    summary: dict[str, Any] = {
        "schema_version": "custom-gsplat-fidelity-control/v1",
        "environment": {
            "project_commit": os.environ.get("SOURCE_GIT_COMMIT") or _git_commit(Path.cwd()),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "configuration": {
            "scene": args.scene,
            "batch": args.batch,
            "width": args.width,
            "height": args.height,
            "outputs": ["rgb", "depth", "alpha", "semantic_id"],
            "semantic_min_alpha": args.semantic_min_alpha,
            "covariance_epsilon": 0.3,
            "rasterize_mode": args.rasterize_mode,
            "warmup_iterations_per_implementation": args.warmup,
            "measured_iterations_per_implementation": args.iterations,
            "capacity_regime": args.capacity_regime,
            "production_acceptance_eligible": (args.capacity_regime == "production-default"),
        },
        "reference": {
            "implementation": "gsplat",
            "commit": gsplat_commit,
            "source_path": str(gsplat_source_path),
            "module_path": str(gsplat_module_path),
            "rendering_module_path": str(gsplat_rendering_path),
            "provenance": gsplat_provenance,
            "eps2d": 0.3,
            "semantic_rule": "maximum_alpha_transmittance_contribution",
            "semantic_api": "rasterize_top_contributing_gaussian_ids",
            "rasterize_mode": args.rasterize_mode,
        },
        "candidate": {
            "implementation": "custom-cuda",
            "pipeline": backend.pipeline_name,
            "gaussian_support_sigma": 3.33,
            "covariance_epsilon": 0.3,
            "semantic_min_alpha": args.semantic_min_alpha,
            "rasterize_mode": args.rasterize_mode,
            "capacity": {
                "visible_records": visible_capacity,
                "tile_intersections": intersection_capacity,
                "final_visible_records": backend.current_visible_capacity,
                "final_tile_intersections": (
                    backend.current_intersection_capacity
                ),
                "regime": args.capacity_regime,
                "production_acceptance_eligible": (args.capacity_regime == "production-default"),
                "allocation_scope": "reserved_intersection_capacity",
                "sort_scope": "emitted_prefix_global_radix_after_host_count_sync",
                "zero_overflow": custom_zero_capacity_overflow,
            },
            "capacity_adaptation": backend.capacity_stats,
            "counters": custom_counters,
        },
        "scene": scene_manifest,
        "camera_bundle_id": fidelity_bundle.bundle_id,
        "fidelity_report": "report/fidelity_report.json",
        "antialias_fixture": antialias_fixture,
        "timing": timing,
        "zero_capacity_overflow": custom_zero_capacity_overflow,
        "pass": _comparison_pass(
            fidelity_pass=bool(report["pass"]),
            antialias_fixture=antialias_fixture,
            custom_counters=custom_counters,
        ),
        "acceptance_scope": (
            "Pinned-gsplat all-output correctness control only; this is not the RTX/OVRTX acceptance reference."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    service.shutdown()
    _require_passing_summary(summary)


if __name__ == "__main__":
    main()
