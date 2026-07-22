#!/usr/bin/env python3
"""Finite production-CUDA acceptance for repaired FlashGS compositor tails.

The fixture creates four horizontal 16x16 tiles whose sorted Gaussian ranges
have lengths 0, 1, 2, and 4.  Both production compositor specializations are
executed twice.  The full-sensor result is checked against a small independent
CPU implementation of the compositor equation, while the RGB-only result is
required to be bitwise identical to the full-sensor RGB output.

This is a correctness and Compute Sanitizer target.  It intentionally records
no performance timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
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

from benchmarks.flashgs_matched_occupancy import (  # noqa: E402
    ComputeProcessSampler,
    capture_node_snapshot,
    ensure_cooperative_executor_lock,
    occupancy_failures,
)
from isaacsim_gaussian_renderer import FlashGSBackend, RendererService  # noqa: E402
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    active_cuda_device_uuid,
    artifact_record,
    load_verified_source_manifest,
)
from isaacsim_gaussian_renderer.flashgs_debug import (  # noqa: E402
    load_verified_flashgs_adapter_attestation,
)


SCHEMA_VERSION = "flashgs-short-tail-synthetic-verification-v1"
SUCCESS_SENTINEL = "FLASHGS_SHORT_TAIL_SYNTHETIC_OK"
WIDTH = 64
HEIGHT = 16
TILE_SIZE = 16
SCENE_ID = 901
EXPECTED_RANGE_LENGTHS = (0, 1, 2, 4)
EXPECTED_RANGE_GAUSSIAN_IDS = ((), (1,), (2, 3), (0, 4, 5, 6))
FULL_OUTPUTS = ("rgb", "depth", "alpha", "semantic_id")
RGB_OUTPUTS = ("rgb",)
RGB_ATOL = 3.0e-6
RGB_RTOL = 3.0e-5
ALPHA_ATOL = 3.0e-6
ALPHA_RTOL = 3.0e-5
DEPTH_ATOL = 3.0e-6
DEPTH_RTOL = 3.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded, untimed production FlashGS short-tail acceptance suitable for compute-sanitizer memcheck."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--flashgs-adapter-attestation",
        type=Path,
        required=True,
    )
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    metadata = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(metadata).to_bytes(8, "little", signed=False))
    digest.update(metadata)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def synthetic_fixture_arrays() -> dict[str, np.ndarray]:
    """Return the canonical seven-Gaussian CPU fixture.

    Gaussian zero deliberately belongs to the four-entry control tile.  The
    one- and two-entry short tails therefore cannot accidentally pass by
    reading element zero when they should read their logical Gaussian.
    """

    projected_x = np.asarray([56.5, 24.5, 40.5, 40.5, 56.5, 56.5, 56.5], dtype=np.float32)
    depths = np.asarray([3.0, 4.0, 3.4, 4.4, 3.6, 4.2, 4.8], dtype=np.float32)
    means = np.zeros((7, 3), dtype=np.float32)
    means[:, 0] = (projected_x - 32.0) * depths / 32.0
    means[:, 1] = (np.float32(8.5) - np.float32(8.0)) * depths / np.float32(32.0)
    means[:, 2] = depths
    return {
        "means": means,
        "scales": np.full((7, 3), 0.025, dtype=np.float32),
        "rotations": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (7, 1),
        ),
        "opacities": np.asarray(
            [0.08, 0.80, 0.35, 0.95, 0.25, 0.55, 0.90],
            dtype=np.float32,
        ),
        "features": np.asarray(
            [
                [1.00, 1.00, 0.00],
                [1.00, 0.00, 0.00],
                [0.00, 1.00, 0.00],
                [0.00, 0.00, 1.00],
                [1.00, 0.00, 1.00],
                [0.00, 1.00, 1.00],
                [0.50, 0.25, 1.00],
            ],
            dtype=np.float32,
        ),
        "semantic_ids": np.asarray(
            [900, 101, 201, 202, 301, 302, 303],
            dtype=np.int64,
        ),
    }


def fixture_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in (
        "means",
        "scales",
        "rotations",
        "opacities",
        "features",
        "semantic_ids",
    ):
        digest.update(name.encode("utf-8"))
        digest.update(_sha256_array(arrays[name]).encode("ascii"))
    return digest.hexdigest()


def validate_tile_layout(
    ranges: np.ndarray,
    point_list: np.ndarray,
) -> tuple[list[int], list[list[int]]]:
    """Validate the exact 0/1/2/4 sorted-range contract."""

    ranges = np.asarray(ranges)
    point_list = np.asarray(point_list)
    if ranges.shape != (4, 2) or not np.issubdtype(ranges.dtype, np.integer):
        raise ValueError("Synthetic FlashGS ranges must be integer [4, 2].")
    if point_list.ndim != 1 or not np.issubdtype(point_list.dtype, np.integer):
        raise ValueError("Synthetic FlashGS point list must be one-dimensional integers.")
    lengths: list[int] = []
    gaussian_ids: list[list[int]] = []
    previous_end = 0
    for tile, (start_raw, end_raw) in enumerate(ranges.tolist()):
        start = int(start_raw)
        end = int(end_raw)
        if start < 0 or end < start or end > point_list.size:
            raise ValueError(f"Tile {tile} has an invalid sorted range [{start}, {end}).")
        if tile and start < previous_end:
            raise ValueError("Synthetic FlashGS ranges overlap or regress.")
        previous_end = end
        lengths.append(end - start)
        gaussian_ids.append([int(value) for value in point_list[start:end]])
    if tuple(lengths) != EXPECTED_RANGE_LENGTHS:
        raise ValueError(f"Synthetic FlashGS range lengths {lengths} != {list(EXPECTED_RANGE_LENGTHS)}.")
    if tuple(tuple(ids) for ids in gaussian_ids) != EXPECTED_RANGE_GAUSSIAN_IDS:
        raise ValueError(
            "Synthetic FlashGS sorted Gaussian IDs differ: "
            f"{gaussian_ids} != {[list(ids) for ids in EXPECTED_RANGE_GAUSSIAN_IDS]}."
        )
    return lengths, gaussian_ids


def compose_cpu_oracle(
    *,
    width: int,
    height: int,
    ranges: np.ndarray,
    point_list: np.ndarray,
    points_xy: np.ndarray,
    rgb_depth: np.ndarray,
    conic_opacity: np.ndarray,
    semantic_ids: np.ndarray,
    semantic_min_alpha: float = 0.01,
) -> dict[str, np.ndarray]:
    """Evaluate the production FlashGS compositor equation on the CPU."""

    if width <= 0 or height <= 0 or width % TILE_SIZE or height % TILE_SIZE:
        raise ValueError("CPU oracle requires positive dimensions divisible by 16.")
    tile_count = (width // TILE_SIZE) * (height // TILE_SIZE)
    ranges = np.asarray(ranges)
    point_list = np.asarray(point_list)
    points_xy = np.asarray(points_xy, dtype=np.float32)
    rgb_depth = np.asarray(rgb_depth, dtype=np.float32)
    conic_opacity = np.asarray(conic_opacity, dtype=np.float32)
    semantic_ids = np.asarray(semantic_ids, dtype=np.int64)
    gaussian_count = points_xy.shape[0]
    if ranges.shape != (tile_count, 2):
        raise ValueError("CPU oracle range shape differs from the tile grid.")
    if (
        points_xy.shape != (gaussian_count, 2)
        or rgb_depth.shape != (gaussian_count, 4)
        or conic_opacity.shape != (gaussian_count, 4)
        or semantic_ids.shape != (gaussian_count,)
    ):
        raise ValueError("CPU oracle Gaussian workspace shapes differ.")

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.full((height, width, 1), np.inf, dtype=np.float32)
    alpha_out = np.zeros((height, width, 1), dtype=np.float32)
    semantic = np.full((height, width, 1), -1, dtype=np.int64)
    x_tiles = width // TILE_SIZE
    for tile in range(tile_count):
        tile_y, tile_x = divmod(tile, x_tiles)
        start, end = (int(value) for value in ranges[tile])
        if start < 0 or end < start or end > point_list.size:
            raise ValueError(f"CPU oracle tile {tile} has an invalid range.")
        for y in range(tile_y * TILE_SIZE, (tile_y + 1) * TILE_SIZE):
            for x in range(tile_x * TILE_SIZE, (tile_x + 1) * TILE_SIZE):
                transmittance = np.float32(1.0)
                color = np.zeros((3,), dtype=np.float32)
                depth_sum = np.float32(0.0)
                best_weight = np.float32(-1.0)
                best_gaussian = -1
                for sorted_index in range(start, end):
                    gaussian = int(point_list[sorted_index])
                    if gaussian < 0 or gaussian >= gaussian_count:
                        raise ValueError("CPU oracle point list has an invalid Gaussian ID.")
                    dx = np.float32(points_xy[gaussian, 0] - np.float32(x + 0.5))
                    dy = np.float32(points_xy[gaussian, 1] - np.float32(y + 0.5))
                    conic = conic_opacity[gaussian]
                    quadratic = np.float32(
                        conic[0] * dx * dx + np.float32(2.0) * conic[1] * dx * dy + conic[2] * dy * dy
                    )
                    power = np.float32(-0.5) * quadratic
                    if power > 0.0 or power < -20.0:
                        continue
                    contributor_alpha = np.float32(min(0.99, float(conic[3] * np.exp(power, dtype=np.float32))))
                    if contributor_alpha < np.float32(1.0 / 255.0):
                        continue
                    next_transmittance = np.float32(transmittance * np.float32(1.0 - contributor_alpha))
                    if next_transmittance <= np.float32(1.0e-4):
                        break
                    weight = np.float32(transmittance * contributor_alpha)
                    color = np.asarray(
                        color + rgb_depth[gaussian, :3] * weight,
                        dtype=np.float32,
                    )
                    depth_sum = np.float32(depth_sum + rgb_depth[gaussian, 3] * weight)
                    if weight > best_weight:
                        best_weight = weight
                        best_gaussian = gaussian
                    transmittance = next_transmittance
                accumulated_alpha = np.float32(1.0 - transmittance)
                rgb[y, x] = color
                alpha_out[y, x, 0] = accumulated_alpha
                if accumulated_alpha > np.float32(1.0e-8):
                    depth[y, x, 0] = np.float32(depth_sum / accumulated_alpha)
                if best_gaussian >= 0 and accumulated_alpha >= semantic_min_alpha:
                    semantic[y, x, 0] = semantic_ids[best_gaussian]
    return {
        "rgb": rgb,
        "depth": depth,
        "alpha": alpha_out,
        "semantic_id": semantic,
    }


def validate_result_payload(payload: dict[str, Any]) -> list[str]:
    """Return fail-closed schema errors for a completed acceptance artifact."""

    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version differs")
    if payload.get("pass") is not True:
        errors.append("pass is not true")
    fixture = payload.get("fixture") or {}
    if fixture.get("range_lengths") != list(EXPECTED_RANGE_LENGTHS):
        errors.append("range_lengths differ")
    if fixture.get("range_gaussian_ids") != [list(values) for values in EXPECTED_RANGE_GAUSSIAN_IDS]:
        errors.append("range_gaussian_ids differ")
    lanes = payload.get("specializations") or {}
    if set(lanes) != {"full_sensor", "rgb_only"}:
        errors.append("specialization inventory differs")
    for lane in ("full_sensor", "rgb_only"):
        record = lanes.get(lane) or {}
        if record.get("repeat_bitwise_equal") is not True:
            errors.append(f"{lane} repeat was not bitwise equal")
        counters = record.get("counters") or {}
        if counters.get("intersection_overflow") != 0:
            errors.append(f"{lane} overflow is nonzero")
        if counters.get("visible_gaussians") != 7:
            errors.append(f"{lane} visible count differs")
        if counters.get("generated_intersections") != 7:
            errors.append(f"{lane} intersection count differs")
    checks = payload.get("checks") or {}
    for name in (
        "full_sensor_matches_cpu_oracle",
        "rgb_only_matches_cpu_oracle",
        "rgb_specializations_bitwise_equal",
        "workspace_layouts_equal",
        "output_contracts_valid",
        "occupancy_pass",
    ):
        if checks.get(name) is not True:
            errors.append(f"check failed: {name}")
    environment = payload.get("environment") or {}
    if not environment.get("gpu_uuid"):
        errors.append("GPU UUID is missing")
    if not (environment.get("native_extension") or {}).get("sha256"):
        errors.append("native extension artifact is missing")
    return errors


def _to_device_fixture(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(array).to(device=device).contiguous()
        for name, array in synthetic_fixture_arrays().items()
    }


def _camera(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    intrinsic = torch.tensor(
        [[32.0, 0.0, 32.0], [0.0, 32.0, 8.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    scene_ids = torch.full((1,), SCENE_ID, device=device, dtype=torch.int64)
    return viewmat.contiguous(), intrinsic.contiguous(), scene_ids


def _tensor_contract_errors(
    outputs: dict[str, torch.Tensor],
    *,
    output_names: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    if tuple(outputs) != output_names:
        errors.append("output names differ")
    expected = {
        "rgb": ((1, HEIGHT, WIDTH, 3), torch.float32),
        "depth": ((1, HEIGHT, WIDTH, 1), torch.float32),
        "alpha": ((1, HEIGHT, WIDTH, 1), torch.float32),
        "semantic_id": ((1, HEIGHT, WIDTH, 1), torch.int64),
    }
    for name, tensor in outputs.items():
        shape, dtype = expected[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            errors.append(f"{name} shape/dtype differs")
        if not tensor.is_cuda or not tensor.is_contiguous():
            errors.append(f"{name} is not contiguous CUDA storage")
    return errors


def _run_specialization(
    *,
    device: torch.device,
    output_names: tuple[str, ...],
) -> dict[str, Any]:
    backend = FlashGSBackend(
        max_intersections=16,
        near_plane=0.01,
        far_plane=100.0,
        gaussian_support_sigma=3.0,
        covariance_epsilon=0.0,
        semantic_min_alpha=0.01,
        tile_size=16,
    )
    service = RendererService(
        backend,
        height=HEIGHT,
        width=WIDTH,
        outputs=output_names,
        max_views=1,
    )
    try:
        service.initialize(stage=None, device=device)
        scene = _to_device_fixture(device)
        service.load_scene(SCENE_ID, **scene)
        camera = _camera(device)
        outputs = service.prepare_outputs(1)
        service.render(*camera, outputs=outputs)
        service.synchronize()
        first = {name: value.detach().clone() for name, value in outputs.items()}
        service.render(*camera, outputs=outputs)
        service.synchronize()
        counters = backend.check_capacity(synchronize=False)
        contract_errors = _tensor_contract_errors(outputs, output_names=output_names)
        repeat_bitwise_equal = all(torch.equal(first[name], outputs[name]) for name in output_names)
        if not repeat_bitwise_equal:
            raise AssertionError(f"{output_names} production render was not bitwise repeatable.")
        if contract_errors:
            raise AssertionError("; ".join(contract_errors))
        native = backend._native
        if native is None or not getattr(native, "__file__", None):
            raise RuntimeError("Production FlashGS native extension has no binary path.")
        arrays = {name: value.detach().cpu().numpy().copy() for name, value in outputs.items()}
        workspace = {
            "ranges": backend.workspace["ranges"].detach().cpu().numpy().copy(),
            "point_list": backend.workspace["values_sorted"].detach().cpu().numpy().copy(),
            "points_xy": backend.workspace["points_xy"].detach().cpu().numpy().copy(),
            "rgb_depth": backend.workspace["rgb_depth"].detach().cpu().numpy().copy(),
            "conic_opacity": backend.workspace["conic_opacity"].detach().cpu().numpy().copy(),
        }
        return {
            "arrays": arrays,
            "workspace": workspace,
            "counters": counters,
            "repeat_bitwise_equal": repeat_bitwise_equal,
            "native_extension": artifact_record(native.__file__),
            "native_build_contract": dict(getattr(native, "__vgr_build_contract__", {})),
        }
    finally:
        service.shutdown()


def _max_finite_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[mask] - expected[mask]))) if np.any(mask) else 0.0


def _json_sample(arrays: dict[str, np.ndarray], *, x: int, y: int) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "pixel_xy": [x, y],
        "rgb": [float(value) for value in arrays["rgb"][y, x]],
    }
    if "alpha" in arrays:
        depth = float(arrays["depth"][y, x, 0])
        sample.update(
            {
                "alpha": float(arrays["alpha"][y, x, 0]),
                "depth": depth if math.isfinite(depth) else None,
                "semantic_id": int(arrays["semantic_id"][y, x, 0]),
            }
        )
    return sample


@torch.no_grad()
def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite synthetic evidence: {args.output}.")
    source_provenance = load_verified_source_manifest(
        args.source_manifest,
        project_root=PROJECT_ROOT,
    )
    if source_provenance.get("dirty") is not False:
        raise RuntimeError("Synthetic FlashGS acceptance requires clean source provenance.")
    adapter = load_verified_flashgs_adapter_attestation(
        args.flashgs_adapter_attestation,
        source_provenance=source_provenance,
        project_root=PROJECT_ROOT,
    )
    executor_lock = ensure_cooperative_executor_lock()
    preflight = capture_node_snapshot()
    preflight_errors = occupancy_failures(
        preflight,
        expected_gpu_uuid=args.expected_gpu_uuid,
        allow_current_gpu_process=False,
    )
    if preflight_errors:
        raise RuntimeError("Shared RTX node occupancy preflight failed: " + "; ".join(preflight_errors))
    sampler = ComputeProcessSampler(
        expected_gpu_uuid=args.expected_gpu_uuid,
        allowed_pids={os.getpid()},
        interval_seconds=0.25,
    ).start()
    sampling: dict[str, Any] | None = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Synthetic FlashGS acceptance requires CUDA.")
        torch.cuda.init()
        device = torch.device("cuda", torch.cuda.current_device())
        gpu_uuid = active_cuda_device_uuid(torch.cuda)
        if gpu_uuid != args.expected_gpu_uuid:
            raise RuntimeError(f"Active CUDA UUID {gpu_uuid!r} != {args.expected_gpu_uuid!r}.")
        full = _run_specialization(device=device, output_names=FULL_OUTPUTS)
        rgb_only = _run_specialization(device=device, output_names=RGB_OUTPUTS)
        sampling = sampler.stop()
        sampler = None
        postflight = capture_node_snapshot()
        postflight_errors = occupancy_failures(
            postflight,
            expected_gpu_uuid=args.expected_gpu_uuid,
            allow_current_gpu_process=True,
        )
        occupancy_pass = not preflight_errors and not postflight_errors and (sampling.get("pass") is True)
        if not occupancy_pass:
            raise RuntimeError(
                "Shared RTX node occupancy sampling failed: "
                + "; ".join([*postflight_errors, *sampling.get("failures", [])])
            )

        full_workspace = full["workspace"]
        rgb_workspace = rgb_only["workspace"]
        lengths, gaussian_ids = validate_tile_layout(
            full_workspace["ranges"],
            full_workspace["point_list"],
        )
        rgb_lengths, rgb_gaussian_ids = validate_tile_layout(
            rgb_workspace["ranges"],
            rgb_workspace["point_list"],
        )
        workspace_layouts_equal = (
            lengths == rgb_lengths
            and gaussian_ids == rgb_gaussian_ids
            and np.array_equal(full_workspace["ranges"], rgb_workspace["ranges"])
        )
        if not workspace_layouts_equal:
            raise AssertionError("Full and RGB-only sorted tile layouts differ.")
        if full["counters"] != {
            "visible_gaussians": 7,
            "generated_intersections": 7,
            "intersection_overflow": 0,
        } or rgb_only["counters"] != {
            "visible_gaussians": 7,
            "generated_intersections": 7,
            "intersection_overflow": 0,
        }:
            raise AssertionError("Synthetic FlashGS production counters differ.")

        fixture = synthetic_fixture_arrays()
        oracle = compose_cpu_oracle(
            width=WIDTH,
            height=HEIGHT,
            ranges=full_workspace["ranges"],
            point_list=full_workspace["point_list"],
            points_xy=full_workspace["points_xy"],
            rgb_depth=full_workspace["rgb_depth"],
            conic_opacity=full_workspace["conic_opacity"],
            semantic_ids=fixture["semantic_ids"],
        )
        full_arrays = {name: array[0] for name, array in full["arrays"].items()}
        rgb_array = rgb_only["arrays"]["rgb"][0]
        np.testing.assert_allclose(
            full_arrays["rgb"],
            oracle["rgb"],
            rtol=RGB_RTOL,
            atol=RGB_ATOL,
        )
        np.testing.assert_allclose(
            full_arrays["alpha"],
            oracle["alpha"],
            rtol=ALPHA_RTOL,
            atol=ALPHA_ATOL,
        )
        actual_depth = full_arrays["depth"]
        oracle_depth = oracle["depth"]
        if not np.array_equal(np.isfinite(actual_depth), np.isfinite(oracle_depth)):
            raise AssertionError("Full-sensor depth foreground mask differs from the CPU oracle.")
        finite_depth = np.isfinite(oracle_depth)
        np.testing.assert_allclose(
            actual_depth[finite_depth],
            oracle_depth[finite_depth],
            rtol=DEPTH_RTOL,
            atol=DEPTH_ATOL,
        )
        if not np.array_equal(full_arrays["semantic_id"], oracle["semantic_id"]):
            raise AssertionError("Full-sensor semantics differ from the CPU oracle.")
        np.testing.assert_allclose(
            rgb_array,
            oracle["rgb"],
            rtol=RGB_RTOL,
            atol=RGB_ATOL,
        )
        rgb_specializations_equal = np.array_equal(
            rgb_array,
            full_arrays["rgb"],
        )
        if not rgb_specializations_equal:
            raise AssertionError("RGB-only and full-sensor RGB are not bitwise equal.")
        if (
            not np.all(np.isfinite(full_arrays["rgb"]))
            or not np.all(np.isfinite(rgb_array))
            or not np.all((full_arrays["alpha"] >= 0.0) & (full_arrays["alpha"] <= 1.0))
            or not np.any(full_arrays["alpha"] > 0.0)
        ):
            raise AssertionError("Synthetic output tensor contract failed.")
        center_semantics = [int(full_arrays["semantic_id"][8, x, 0]) for x in (8, 24, 40, 56)]
        if center_semantics != [-1, 101, 202, 302]:
            raise AssertionError(f"Synthetic center semantics differ: {center_semantics} != [-1, 101, 202, 302].")
        if full["native_extension"] != rgb_only["native_extension"]:
            raise AssertionError("Specializations loaded different native binaries.")

        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "bounded untimed production-CUDA correctness and Compute Sanitizer target; no performance claim"
            ),
            "fixture": {
                "fixture_sha256": fixture_hash(fixture),
                "gaussian_count": 7,
                "width": WIDTH,
                "height": HEIGHT,
                "tile_size": TILE_SIZE,
                "range_lengths": lengths,
                "range_gaussian_ids": gaussian_ids,
                "center_semantic_ids": center_semantics,
                "camera": {
                    "viewmat": np.eye(4, dtype=np.float32).tolist(),
                    "intrinsics": [
                        [32.0, 0.0, 32.0],
                        [0.0, 32.0, 8.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            },
            "numeric_tolerances": {
                "rgb": {"rtol": RGB_RTOL, "atol": RGB_ATOL},
                "alpha": {"rtol": ALPHA_RTOL, "atol": ALPHA_ATOL},
                "depth": {"rtol": DEPTH_RTOL, "atol": DEPTH_ATOL},
                "semantic_id": "exact",
                "repeatability": "bitwise",
                "rgb_specialization_parity": "bitwise",
            },
            "specializations": {
                "full_sensor": {
                    "outputs": list(FULL_OUTPUTS),
                    "counters": full["counters"],
                    "repeat_bitwise_equal": full["repeat_bitwise_equal"],
                    "tensor_sha256": {name: _sha256_array(array) for name, array in full["arrays"].items()},
                },
                "rgb_only": {
                    "outputs": list(RGB_OUTPUTS),
                    "counters": rgb_only["counters"],
                    "repeat_bitwise_equal": rgb_only["repeat_bitwise_equal"],
                    "tensor_sha256": {
                        "rgb": _sha256_array(rgb_only["arrays"]["rgb"]),
                    },
                },
            },
            "oracle": {
                "equation": "independent CPU implementation of production compositor",
                "max_abs_error": {
                    "rgb": _max_finite_abs_error(full_arrays["rgb"], oracle["rgb"]),
                    "alpha": _max_finite_abs_error(full_arrays["alpha"], oracle["alpha"]),
                    "depth": _max_finite_abs_error(full_arrays["depth"], oracle["depth"]),
                },
                "semantic_exact": True,
                "samples": [_json_sample(full_arrays, x=x, y=8) for x in (8, 24, 40, 56)],
            },
            "checks": {
                "full_sensor_matches_cpu_oracle": True,
                "rgb_only_matches_cpu_oracle": True,
                "rgb_specializations_bitwise_equal": rgb_specializations_equal,
                "workspace_layouts_equal": workspace_layouts_equal,
                "output_contracts_valid": True,
                "occupancy_pass": occupancy_pass,
            },
            "environment": {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_uuid": gpu_uuid,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "source_manifest": artifact_record(args.source_manifest),
                "source_identity": source_provenance,
                "flashgs_adapter_attestation": artifact_record(args.flashgs_adapter_attestation),
                "adapter_validation": adapter,
                "native_extension": full["native_extension"],
                "native_build_contract": full["native_build_contract"],
            },
            "node_occupancy": {
                "executor_control": executor_lock,
                "preflight": preflight,
                "preflight_failures": preflight_errors,
                "sampled_compute_process_telemetry": sampling,
                "postflight": postflight,
                "postflight_failures": postflight_errors,
                "pass": occupancy_pass,
            },
            "pass": True,
        }
        errors = validate_result_payload(result)
        if errors:
            raise AssertionError("Synthetic evidence schema failed: " + "; ".join(errors))
        return result
    finally:
        if sampler is not None:
            sampler.stop()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_acceptance(args)
    except BaseException as error:
        if not args.output.exists():
            failure = {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pass": False,
                "failure_reason": f"{type(error).__name__}: {error}",
            }
            args.output.write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        raise
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        SUCCESS_SENTINEL
        + " "
        + json.dumps(
            {
                "gpu_uuid": result["environment"]["gpu_uuid"],
                "output": str(args.output.resolve()),
                "range_lengths": result["fixture"]["range_lengths"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
