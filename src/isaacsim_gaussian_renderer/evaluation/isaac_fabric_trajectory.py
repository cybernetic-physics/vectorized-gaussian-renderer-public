"""Real-scene camera-contract evaluation through Isaac Sim and Fabric.

This module is imported only after ``SimulationApp`` starts.  The public entry
point remains ``scripts/isaacsim_vectorized_example.py`` so workstation setup,
extension lifecycle, and evidence collection do not fork into a second test
harness.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class PublishedCameraContract:
    """Checksum-validated flyby arrays and their immutable metadata."""

    viewmats: np.ndarray
    intrinsics: np.ndarray
    manifest: dict[str, Any]
    npz_sha256: str
    manifest_sha256: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_contract_hash(
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
    manifest_without_contract_hash: dict[str, Any],
) -> str:
    """Reproduce the published flyby contract's canonical SHA-256."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(viewmats).tobytes())
    digest.update(np.ascontiguousarray(intrinsics).tobytes())
    digest.update(
        json.dumps(
            manifest_without_contract_hash,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def load_published_camera_contract(
    npz_path: str | Path,
    manifest_path: str | Path,
) -> PublishedCameraContract:
    """Load both published camera files and fail closed on any mismatch."""
    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"viewmats", "intrinsics", "manifest_json"}:
            raise ValueError(
                "Published camera NPZ must contain only viewmats, intrinsics, "
                "and manifest_json."
            )
        viewmats = np.asarray(archive["viewmats"], dtype=np.float32)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float32)
        embedded_manifest = json.loads(str(np.asarray(archive["manifest_json"]).item()))
    external_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if embedded_manifest != external_manifest:
        raise ValueError("Camera NPZ and camera JSON carry different manifests.")

    manifest = dict(embedded_manifest)
    expected_hash = str(manifest.pop("camera_contract_sha256"))
    actual_hash = camera_contract_hash(viewmats, intrinsics, manifest)
    manifest["camera_contract_sha256"] = expected_hash
    if actual_hash != expected_hash:
        raise ValueError(
            f"Camera contract SHA-256 mismatch: {actual_hash} != {expected_hash}."
        )
    if viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4):
        raise ValueError(f"Camera viewmats have invalid shape {viewmats.shape}.")
    if intrinsics.shape != (viewmats.shape[0], 3, 3):
        raise ValueError(f"Camera intrinsics have invalid shape {intrinsics.shape}.")
    if int(manifest["frame_count"]) != viewmats.shape[0]:
        raise ValueError("Camera manifest frame count does not match its arrays.")
    if not np.isfinite(viewmats).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Camera contract contains non-finite values.")
    if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise ValueError("Camera contract focal lengths must be positive.")
    return PublishedCameraContract(
        viewmats=np.ascontiguousarray(viewmats),
        intrinsics=np.ascontiguousarray(intrinsics),
        manifest=manifest,
        npz_sha256=file_sha256(npz_path),
        manifest_sha256=file_sha256(manifest_path),
    )


def resolve_camera_route_sha256(
    manifest: dict[str, Any],
) -> tuple[str, str]:
    """Return the strongest available immutable route identity.

    Clearance-planned walkthroughs publish a dedicated ``route_sha256``.
    Procedural paths such as the cinematic figure eight instead publish a
    checksum of their path parameters.  The complete camera-contract digest is
    a final, still immutable fallback for older valid contracts.
    """
    for field in (
        "route_sha256",
        "checksum_sha256",
        "camera_contract_sha256",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            continue
        try:
            int(value, 16)
        except ValueError:
            continue
        return value, field
    raise ValueError(
        "Camera manifest lacks a 64-character route, path-parameter, or "
        "camera-contract SHA-256 digest."
    )


def phase_offset_contract(
    contract: PublishedCameraContract,
    *,
    batch: int,
    frames: int | None,
    stride: int,
    start_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Create a time-major B-camera contract from one published walkthrough."""
    if batch <= 0 or stride <= 0 or start_frame < 0:
        raise ValueError("Batch/stride must be positive and start_frame nonnegative.")
    source_frames = int(contract.viewmats.shape[0])
    selected_frames = source_frames if frames is None else frames
    if selected_frames <= 0 or selected_frames > source_frames:
        raise ValueError(
            f"Requested {selected_frames} trajectory frames from {source_frames}."
        )
    offsets_array = np.floor(
        np.arange(batch, dtype=np.float64) * source_frames / batch
    ).astype(np.int64)
    base = (start_frame + np.arange(selected_frames, dtype=np.int64) * stride) % source_frames
    source_indices = (base[:, None] + offsets_array[None, :]) % source_frames
    return (
        np.ascontiguousarray(contract.viewmats[source_indices]),
        np.ascontiguousarray(contract.intrinsics[source_indices]),
        np.ascontiguousarray(source_indices),
        tuple(int(value) for value in offsets_array),
    )


def view_population_pack_contract(
    contract: PublishedCameraContract,
    *,
    batch: int,
    frames: int | None,
    stride: int,
    start_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Pack one non-repeating source-view population into ``[T,B]`` batches."""
    if batch <= 0 or stride <= 0 or start_frame < 0:
        raise ValueError("Batch/stride must be positive and start_frame nonnegative.")
    source_frames = int(contract.viewmats.shape[0])
    if frames is None:
        if source_frames % batch:
            raise ValueError(
                "view-population-pack requires an explicit frame count when "
                "the source frame count is not divisible by batch."
            )
        selected_frames = source_frames // batch
    else:
        selected_frames = frames
    if selected_frames <= 0:
        raise ValueError("Requested trajectory frame count must be positive.")
    linear = start_frame + np.arange(
        selected_frames * batch,
        dtype=np.int64,
    ) * stride
    source_indices = np.mod(linear, source_frames).reshape(selected_frames, batch)
    if np.unique(source_indices).size != source_indices.size:
        raise ValueError(
            "view-population-pack must not repeat source views; reduce frames/batch "
            "or choose a stride coprime to the source frame count."
        )
    return (
        np.ascontiguousarray(contract.viewmats[source_indices]),
        np.ascontiguousarray(contract.intrinsics[source_indices]),
        np.ascontiguousarray(source_indices),
        tuple(int(value) for value in source_indices[0]),
    )


def validate_scene_release(
    scene_path: str | Path,
    scene_manifest_path: str | Path,
    *,
    scene_id_label: str,
    camera_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify the exact PLY bytes/count contract before parsing or CUDA use."""
    scene_path = Path(scene_path)
    scene_manifest_path = Path(scene_manifest_path)
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if not scene_manifest_path.is_file():
        raise FileNotFoundError(scene_manifest_path)
    release = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    if str(release.get("dataset_id")) != scene_id_label:
        raise ValueError(
            f"Scene label {scene_id_label!r} does not match manifest "
            f"{release.get('dataset_id')!r}."
        )
    canonical = release.get("canonical_ply")
    selection = release.get("selection")
    if not isinstance(canonical, dict):
        raise ValueError("Scene manifest lacks canonical_ply metadata.")
    expected_sha256 = str(canonical["sha256"])
    expected_bytes = int(canonical["byte_count"])
    if isinstance(selection, dict) and "record_count" in selection:
        expected_count = int(selection["record_count"])
        count_source = "selection.record_count"
    elif "gaussian_count" in canonical:
        expected_count = int(canonical["gaussian_count"])
        count_source = "canonical_ply.gaussian_count"
    else:
        raise ValueError(
            "Scene manifest must declare selection.record_count or "
            "canonical_ply.gaussian_count."
        )
    actual_bytes = scene_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"Scene byte count mismatch: {actual_bytes} != {expected_bytes}.")
    actual_sha256 = file_sha256(scene_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Scene SHA-256 mismatch: {actual_sha256} != {expected_sha256}."
        )
    if str(camera_manifest["scene_sha256"]) != actual_sha256:
        raise ValueError("Camera contract and scene release name different PLY bytes.")
    if int(camera_manifest["scene_gaussian_count"]) != expected_count:
        raise ValueError("Camera and scene manifests disagree on Gaussian count.")
    return {
        "release": release,
        "sha256": actual_sha256,
        "file_bytes": actual_bytes,
        "gaussian_count": expected_count,
        "gaussian_count_source": count_source,
        "manifest_sha256": file_sha256(scene_manifest_path),
    }


def _cache_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(after[name]) - int(before[name])
        for name in ("hits", "misses")
    }


def _event_samples(starts: list[Any], ends: list[Any]) -> list[float]:
    return [
        float(start.elapsed_time(end))
        for start, end in zip(starts, ends, strict=True)
    ]


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_source_commit() -> str:
    """Resolve provenance from an override or the checkout running the test."""
    override = os.environ.get("SOURCE_GIT_COMMIT", "").strip()
    if override:
        return override
    return _command_output(["git", "rev-parse", "HEAD"]) or "unknown"


def renderer_configuration_contract(
    backend: object,
    *,
    rasterize_mode: str,
    covariance_epsilon: float,
) -> dict[str, Any]:
    """Prove that the Kit extension instantiated the requested renderer."""
    actual_mode = getattr(backend, "rasterize_mode", None)
    actual_epsilon = getattr(backend, "covariance_epsilon", None)
    epsilon_matches = isinstance(actual_epsilon, (int, float)) and math.isclose(
        float(actual_epsilon),
        covariance_epsilon,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    mode_matches = actual_mode == rasterize_mode
    if not mode_matches or not epsilon_matches:
        raise AssertionError(
            "Isaac/Fabric renderer configuration mismatch: requested "
            f"rasterize_mode={rasterize_mode!r}, "
            f"covariance_epsilon={covariance_epsilon}; got "
            f"rasterize_mode={actual_mode!r}, "
            f"covariance_epsilon={actual_epsilon!r}."
        )
    return {
        "requested_rasterize_mode": rasterize_mode,
        "actual_rasterize_mode": actual_mode,
        "requested_covariance_epsilon": covariance_epsilon,
        "actual_covariance_epsilon": float(actual_epsilon),
        "matches": True,
    }


def _write_video(
    frames: list[np.ndarray],
    *,
    output_json: Path,
    fps: float,
) -> dict[str, Any]:
    """Encode captured RGB after measurement and verify it with ffprobe."""
    from PIL import Image

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path is None or ffprobe_path is None:
        raise RuntimeError("Video evidence requires installed ffmpeg and ffprobe.")

    artifact_root = output_json.with_suffix("")
    frames_dir = artifact_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(frames_dir / f"frame-{index:06d}.png")
    video_path = artifact_root / "trajectory.mp4"
    encode_start = time.perf_counter()
    subprocess.run(
        [
            ffmpeg_path,
            "-v",
            "error",
            "-y",
            "-framerate",
            f"{fps:.9g}",
            "-i",
            str(frames_dir / "frame-%06d.png"),
            "-frames:v",
            str(len(frames)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    probe = json.loads(
        subprocess.check_output(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,avg_frame_rate,pix_fmt",
                "-of",
                "json",
                str(video_path),
            ],
            text=True,
        )
    )
    stream = probe["streams"][0]
    verified = (
        stream.get("codec_name") == "h264"
        and int(stream.get("nb_frames", -1)) == len(frames)
    )
    return {
        "status": "verified" if verified else "invalid",
        "path": str(video_path),
        "sha256": file_sha256(video_path),
        "file_bytes": video_path.stat().st_size,
        "frame_count": len(frames),
        "fps": fps,
        "ffmpeg_path": ffmpeg_path,
        "ffmpeg_version": _command_output([ffmpeg_path, "-version"]),
        "ffprobe_path": ffprobe_path,
        "ffprobe_version": _command_output([ffprobe_path, "-version"]),
        "ffprobe": stream,
        "encoding_seconds": time.perf_counter() - encode_start,
        "pass": verified,
    }


def run_isaac_fabric_trajectory(
    args: Any,
    *,
    simulation_app: Any,
    distribution: Callable[[list[float]], dict[str, float]],
    output_contract: Callable[..., dict[str, object]],
    driver_process_memory: Callable[[], str | None],
) -> dict[str, Any]:
    """Execute fixed controls and a moving published trajectory in one Kit session."""
    import carb
    import omni.kit.app
    import omni.usd
    import torch
    from isaacsim.core.utils.stage import create_new_stage
    from pxr import Gf, UsdGeom

    from isaacsim_gaussian_renderer.camera_math import (
        opencv_viewmats_to_usd_camera_world_matrices,
    )
    from isaacsim_gaussian_renderer.ply_loader import (
        canonicalize_3dgs_scene,
        load_ply_to_gaussians,
        ply_vertex_property_names,
    )

    camera_contract = load_published_camera_contract(
        args.camera_contract,
        args.camera_manifest,
    )
    manifest = camera_contract.manifest
    camera_route_sha256, camera_route_sha256_source = (
        resolve_camera_route_sha256(manifest)
    )
    if int(manifest["width"]) != args.width or int(manifest["height"]) != args.height:
        raise ValueError("CLI resolution does not match the immutable camera contract.")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (
            args.scene_id_label,
            args.scene_author,
            args.scene_license,
            args.scene_source,
        )
    ):
        raise ValueError("Scene ID, author, license, and source must be nonempty strings.")
    release_validation_start = time.perf_counter()
    scene_release = validate_scene_release(
        args.scene_path,
        args.scene_manifest,
        scene_id_label=args.scene_id_label,
        camera_manifest=manifest,
    )
    release_validation_seconds = time.perf_counter() - release_validation_start
    contract_builder = (
        phase_offset_contract
        if args.trajectory_layout == "phase-offset"
        else view_population_pack_contract
    )
    scenario_viewmats, scenario_intrinsics, source_indices, phase_offsets = (
        contract_builder(
            camera_contract,
            batch=args.batch,
            frames=args.trajectory_frames,
            stride=args.trajectory_frame_stride,
            start_frame=args.trajectory_start_frame,
        )
    )
    timesteps = int(scenario_viewmats.shape[0])
    from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (
        CameraTrajectory,
        save_trajectory,
    )

    measured_contract = CameraTrajectory(
        viewmats=scenario_viewmats,
        intrinsics=scenario_intrinsics,
        scene_ids=np.full((timesteps, args.batch), 404, dtype=np.int64),
        width=args.width,
        height=args.height,
        fps=float(manifest["fps"]),
        seed=0,
        scene_sha256=scene_release["sha256"],
        route_sha256=camera_route_sha256,
        motion_classification=(
            f"{args.trajectory_layout}-published-walkthrough"
            if args.trajectory_frame_stride == 1
            else f"{args.trajectory_layout}-strided-walkthrough-stress"
        ),
        expected_cache_events=("miss",) * timesteps,
        near_plane=float(manifest["near_plane"]),
        far_plane=float(manifest["far_plane"]),
        environment_phase_offsets=phase_offsets,
        provenance={
            "source_camera_contract_sha256": manifest["camera_contract_sha256"],
            "source_camera_npz_sha256": camera_contract.npz_sha256,
            "source_camera_manifest_sha256": camera_contract.manifest_sha256,
            "source_indices_sha256": hashlib.sha256(source_indices.tobytes()).hexdigest(),
            "trajectory_layout": args.trajectory_layout,
        },
        route_validation=manifest.get("route_planner", {}),
    )
    measured_contract_json, measured_contract_npz = save_trajectory(
        measured_contract,
        args.output.with_suffix(".trajectory.json"),
    )
    measured_contract_files = {
        "trajectory_id": measured_contract.trajectory_id,
        "json_path": str(measured_contract_json),
        "json_sha256": file_sha256(measured_contract_json),
        "npz_path": str(measured_contract_npz),
        "npz_sha256": file_sha256(measured_contract_npz),
    }
    if not np.allclose(
        scenario_intrinsics,
        scenario_intrinsics[:1],
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "This Fabric evaluator requires constant USD camera intrinsics; "
            "the supplied contract varies them over time."
        )
    expected_cx = args.width * 0.5
    expected_cy = args.height * 0.5
    if not (
        np.allclose(scenario_intrinsics[..., 0, 2], expected_cx, atol=1.0e-5)
        and np.allclose(scenario_intrinsics[..., 1, 2], expected_cy, atol=1.0e-5)
        and np.allclose(scenario_intrinsics[..., 0, 1], 0.0, atol=1.0e-7)
        and np.allclose(scenario_intrinsics[..., 1, 0], 0.0, atol=1.0e-7)
    ):
        raise ValueError("USD camera ingestion currently requires centered, zero-skew intrinsics.")

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.add_path(str(args.repo_root / "exts"))
    manager.set_extension_enabled_immediate("isaacsim.gaussian_renderer", True)
    settings = carb.settings.get_settings()
    settings.set("/app/useFabricSceneDelegate", True)
    delegate_extension_loaded = manager.is_extension_enabled("omni.hydra.usdrt_delegate")
    delegate_enabled = settings.get_as_bool("/app/useFabricSceneDelegate")
    if not delegate_extension_loaded or not delegate_enabled:
        raise RuntimeError("Isaac Sim Fabric Scene Delegate is not active.")

    from isaacsim_gaussian_renderer_extension import extension as extension_module

    extension = extension_module.STARTUP_INSTANCE
    if extension is None:
        raise RuntimeError("Kit did not start isaacsim.gaussian_renderer.")

    service = None
    camera_source = None
    try:
        create_new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/TrajectoryCameras")

        flat_viewmats = torch.from_numpy(
            scenario_viewmats.reshape(-1, 4, 4)
        )
        world_matrices = (
            opencv_viewmats_to_usd_camera_world_matrices(flat_viewmats)
            .numpy()
            .reshape(timesteps, args.batch, 4, 4)
            .astype(np.float64)
        )
        camera_paths: list[str] = []
        camera_ops: list[Any] = []
        aperture = 36.0
        for index in range(args.batch):
            path = f"/World/TrajectoryCameras/Camera_{index:04d}"
            camera = UsdGeom.Camera.Define(stage, path)
            intrinsic = scenario_intrinsics[0, index]
            focal_length = float(intrinsic[0, 0]) * aperture / args.width
            vertical_aperture = focal_length * args.height / float(intrinsic[1, 1])
            camera.CreateFocalLengthAttr(focal_length)
            camera.CreateHorizontalApertureAttr(aperture)
            camera.CreateVerticalApertureAttr(vertical_aperture)
            camera.CreateClippingRangeAttr(
                Gf.Vec2f(float(manifest["near_plane"]), float(manifest["far_plane"]))
            )
            op = UsdGeom.Xformable(camera).AddTransformOp(
                precision=UsdGeom.XformOp.PrecisionDouble
            )
            op.Set(Gf.Matrix4d(*world_matrices[0, index].tolist()))
            camera_paths.append(path)
            camera_ops.append(op)
        camera_path_tuple = tuple(camera_paths)
        simulation_app.update()

        device = torch.device("cuda:0")
        camera_source = extension.create_camera_source(
            stage=stage,
            stage_id=omni.usd.get_context().get_stage_id(),
            camera_paths=camera_path_tuple,
            device=str(device),
            update_world_xforms=True,
        )
        usd_intrinsics = camera_source.get_batched_intrinsics(
            camera_path_tuple,
            width=args.width,
            height=args.height,
            device=device,
        )
        expected_intrinsics = torch.from_numpy(scenario_intrinsics[0]).to(device)
        intrinsics_max_abs_error = float(
            torch.max(torch.abs(usd_intrinsics - expected_intrinsics)).item()
        )
        if intrinsics_max_abs_error > 1.0e-4:
            raise AssertionError(
                f"USD camera intrinsics differ from contract by {intrinsics_max_abs_error}."
            )

        resolution_scale = args.width * args.height / float(128 * 128)
        initial_visible_capacity = args.batch * args.visible_per_view
        initial_intersection_capacity = math.ceil(
            args.batch * args.intersections_per_view_at_128 * resolution_scale
        )
        service = extension.create_service(
            stage=stage,
            device=str(device),
            height=args.height,
            width=args.width,
            max_views=args.batch,
            max_visible_records=initial_visible_capacity,
            max_intersections=initial_intersection_capacity,
            gaussian_support_sigma=args.gaussian_support_sigma,
            covariance_epsilon=args.covariance_epsilon,
            rasterize_mode=args.rasterize_mode,
            semantic_min_alpha=args.semantic_min_alpha,
            tile_size=1,
            depth_bucket_count=args.depth_bucket_count,
            depth_bucket_group_size=args.depth_bucket_group_size,
            compact_projection_cache=True,
            materialize_projected_records=(
                args.materialize_projected_records
            ),
            enable_projection_cache=True,
            output_srgb=False,
            deterministic=False,
            adaptive_capacity=True,
            max_workspace_bytes=args.max_workspace_bytes,
        )
        backend = service.backend
        renderer_configuration = renderer_configuration_contract(
            backend,
            rasterize_mode=args.rasterize_mode,
            covariance_epsilon=args.covariance_epsilon,
        )

        host_load_start = time.perf_counter()
        source_has_semantics = (
            "semantic_id" in ply_vertex_property_names(args.scene_path)
        )
        raw_scene = load_ply_to_gaussians(args.scene_path)
        host_load_seconds = time.perf_counter() - host_load_start
        if raw_scene.count != scene_release["gaussian_count"]:
            raise ValueError(
                f"PLY contains {raw_scene.count} Gaussians, expected "
                f"{scene_release['gaussian_count']}."
            )
        upload_start = time.perf_counter()
        canonical = canonicalize_3dgs_scene(raw_scene, device=device)
        semantic_ids = canonical.semantic_ids
        if not source_has_semantics:
            semantic_ids = torch.arange(
                canonical.count,
                device=device,
                dtype=torch.int64,
            ).remainder_(1024)
        service.load_scene(
            404,
            means=canonical.means,
            scales=canonical.scales,
            rotations=canonical.rotations,
            opacities=canonical.opacities,
            features=canonical.features[:, :3].contiguous(),
            semantic_ids=semantic_ids,
        )
        service.synchronize()
        upload_seconds = time.perf_counter() - upload_start
        del raw_scene
        del canonical
        gc.collect()
        torch.cuda.empty_cache()

        expected_viewmats = torch.from_numpy(scenario_viewmats).to(device)
        scene_ids = torch.full(
            (args.batch,),
            404,
            device=device,
            dtype=torch.int64,
        )

        def author_step(step: int) -> None:
            for camera_index, op in enumerate(camera_ops):
                op.Set(Gf.Matrix4d(*world_matrices[step, camera_index].tolist()))

        author_step(0)
        simulation_app.update()
        first_viewmats = camera_source.read_transforms(camera_path_tuple, device=device)
        service.synchronize()
        first_camera_error = float(
            torch.max(torch.abs(first_viewmats - expected_viewmats[0])).item()
        )
        if first_camera_error > 1.0e-4:
            raise AssertionError(
                f"Fabric camera readback differs from contract by {first_camera_error}."
            )

        first_start = torch.cuda.Event(enable_timing=True)
        first_end = torch.cuda.Event(enable_timing=True)
        first_cache_before = backend.projection_cache_stats
        first_wall_start = time.perf_counter()
        first_start.record()
        outputs = service.render(first_viewmats, usd_intrinsics, scene_ids)
        first_end.record()
        service.synchronize()
        first_wall_ms = (time.perf_counter() - first_wall_start) * 1_000.0
        first_gpu_ms = float(first_start.elapsed_time(first_end))
        first_cache_after = backend.projection_cache_stats
        first_counters = backend.check_capacity(synchronize=False)
        first_zero_overflow = bool(
            first_counters["visible_overflow"] == 0
            and first_counters["intersection_overflow"] == 0
        )
        if not first_zero_overflow:
            raise AssertionError(
                f"First trajectory render overflowed capacity: {first_counters}."
            )
        first_capacity = backend.capacity_stats
        first_output_contract = output_contract(
            outputs,
            semantic_min_alpha=args.semantic_min_alpha,
            expected_shape=(args.batch, args.height, args.width),
        )
        if not first_output_contract["valid"]:
            raise AssertionError(f"First output contract failed: {first_output_contract}")

        for _ in range(args.warmup):
            viewmats = camera_source.read_transforms(camera_path_tuple, device=device)
            service.render(viewmats, usd_intrinsics, scene_ids, outputs=outputs)
        service.synchronize()

        def fixed_phase(*, invalidate: bool) -> dict[str, Any]:
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
            source_ends = [
                torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)
            ]
            render_ends = [
                torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)
            ]
            walls: list[float] = []
            cache_before = backend.projection_cache_stats
            for start, source_end, render_end in zip(
                starts,
                source_ends,
                render_ends,
                strict=True,
            ):
                wall_start = time.perf_counter()
                start.record()
                viewmats = camera_source.read_transforms(camera_path_tuple, device=device)
                source_end.record()
                if invalidate:
                    backend.invalidate_projection_cache()
                service.render(viewmats, usd_intrinsics, scene_ids, outputs=outputs)
                render_end.record()
                service.synchronize()
                walls.append((time.perf_counter() - wall_start) * 1_000.0)
            cache_after = backend.projection_cache_stats
            source_gpu = _event_samples(starts, source_ends)
            render_gpu = _event_samples(source_ends, render_ends)
            integrated_gpu = _event_samples(starts, render_ends)
            contract = output_contract(
                outputs,
                semantic_min_alpha=args.semantic_min_alpha,
                expected_shape=(args.batch, args.height, args.width),
            )
            counters = backend.check_capacity(synchronize=False)
            zero_overflow = bool(
                counters["visible_overflow"] == 0
                and counters["intersection_overflow"] == 0
            )
            if not contract["valid"]:
                raise AssertionError(
                    f"Fixed trajectory output contract failed: {contract}."
                )
            if not zero_overflow:
                raise AssertionError(
                    f"Fixed trajectory phase overflowed capacity: {counters}."
                )
            cache = _cache_delta(cache_before, cache_after)
            expected_cache = {
                "hits": 0 if invalidate else args.iterations,
                "misses": args.iterations if invalidate else 0,
            }
            return {
                "iterations": args.iterations,
                "cache_semantics": (
                    "fixed camera, explicit invalidation"
                    if invalidate
                    else "fixed camera, no invalidation"
                ),
                "fabric_source_gpu_ms": distribution(source_gpu),
                "custom_render_gpu_ms": distribution(render_gpu),
                "integrated_gpu_ms": distribution(integrated_gpu),
                "synchronized_wall_ms": distribution(walls),
                "samples_ms": {
                    "fabric_source_gpu": source_gpu,
                    "custom_render_gpu": render_gpu,
                    "integrated_gpu": integrated_gpu,
                    "synchronized_wall": walls,
                },
                "wall_images_per_second": args.batch * 1_000.0 / float(np.mean(walls)),
                "cache": cache,
                "expected_cache": expected_cache,
                "cache_pass": cache == expected_cache,
                "output_contract": contract,
                "counters": counters,
                "zero_overflow": zero_overflow,
            }

        fixed_hits = fixed_phase(invalidate=False)
        fixed_invalidated = fixed_phase(invalidate=True)

        capacity_warmup_seconds = 0.0
        capacity_warmup_contracts_pass = True
        capacity_warmup_zero_overflow = True
        if args.trajectory_capacity_warmup:
            capacity_start = time.perf_counter()
            for step in range(timesteps):
                author_step(step)
                simulation_app.update()
                viewmats = camera_source.read_transforms(camera_path_tuple, device=device)
                backend.invalidate_projection_cache()
                service.render(viewmats, usd_intrinsics, scene_ids, outputs=outputs)
                service.synchronize()
                counters = backend.check_capacity(synchronize=False)
                contract = output_contract(
                    outputs,
                    semantic_min_alpha=args.semantic_min_alpha,
                    expected_shape=(args.batch, args.height, args.width),
                )
                capacity_warmup_contracts_pass &= bool(contract["valid"])
                capacity_warmup_zero_overflow &= not bool(
                    counters["visible_overflow"] or counters["intersection_overflow"]
                )
            capacity_warmup_seconds = time.perf_counter() - capacity_start

        kit_settle_start = time.perf_counter()
        for _ in range(args.kit_settle_updates):
            simulation_app.update()
        kit_settle_seconds = time.perf_counter() - kit_settle_start

        previous_rgb = torch.empty_like(outputs["rgb"])
        previous_rgb_valid = False
        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
        reserved_before = int(torch.cuda.memory_reserved())
        author_wall: list[float] = []
        kit_update_wall: list[float] = []
        fabric_gpu: list[float] = []
        render_gpu: list[float] = []
        integrated_gpu: list[float] = []
        observation_wall: list[float] = []
        validation_wall: list[float] = []
        camera_errors: list[float] = []
        output_contracts: list[dict[str, object]] = []
        capacity_counters: list[dict[str, int]] = []
        cache_events: list[dict[str, Any]] = []
        allocation_checkpoints: list[int] = []
        input_changed_steps = 0
        output_changed_steps = 0
        deadline_ms = 1_000.0 / float(manifest["fps"])
        deadline_misses = 0
        captured_frames: list[np.ndarray] = []

        for step in range(timesteps):
            observation_start = time.perf_counter()
            author_start = time.perf_counter()
            author_step(step)
            author_wall.append((time.perf_counter() - author_start) * 1_000.0)
            update_start = time.perf_counter()
            simulation_app.update()
            kit_update_wall.append((time.perf_counter() - update_start) * 1_000.0)

            source_start = torch.cuda.Event(enable_timing=True)
            source_end = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            source_start.record()
            viewmats = camera_source.read_transforms(camera_path_tuple, device=device)
            source_end.record()
            cache_before = backend.projection_cache_stats
            backend.invalidate_projection_cache()
            service.render(viewmats, usd_intrinsics, scene_ids, outputs=outputs)
            render_end.record()
            service.synchronize()
            elapsed_wall = (time.perf_counter() - observation_start) * 1_000.0
            observation_wall.append(elapsed_wall)
            if elapsed_wall > deadline_ms:
                deadline_misses += 1
            fabric_gpu.append(float(source_start.elapsed_time(source_end)))
            render_gpu.append(float(source_end.elapsed_time(render_end)))
            integrated_gpu.append(float(source_start.elapsed_time(render_end)))
            cache_after = backend.projection_cache_stats

            validation_start = time.perf_counter()
            camera_error = float(
                torch.max(torch.abs(viewmats - expected_viewmats[step])).item()
            )
            camera_errors.append(camera_error)
            contract = output_contract(
                outputs,
                semantic_min_alpha=args.semantic_min_alpha,
                expected_shape=(args.batch, args.height, args.width),
            )
            contract["step"] = step
            output_contracts.append(contract)
            counters = backend.check_capacity(synchronize=False)
            capacity_counters.append(counters)
            cache = _cache_delta(cache_before, cache_after)
            cache_events.append(
                {
                    "step": step,
                    "expected": "miss",
                    **cache,
                    "pass": cache["hits"] == 0 and cache["misses"] >= 1,
                }
            )
            if step > 0:
                input_changed = not torch.equal(
                    expected_viewmats[step],
                    expected_viewmats[step - 1],
                )
                output_changed = not torch.equal(previous_rgb, outputs["rgb"])
                input_changed_steps += int(input_changed)
                output_changed_steps += int(input_changed and output_changed)
            previous_rgb.copy_(outputs["rgb"])
            previous_rgb_valid = True
            if args.save_trajectory_video:
                frame = outputs["rgb"][0].detach().cpu().numpy()
                captured_frames.append(
                    np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
                )
            allocation_checkpoints.append(int(torch.cuda.memory_allocated()))
            validation_wall.append((time.perf_counter() - validation_start) * 1_000.0)

        if not previous_rgb_valid:
            raise AssertionError("Trajectory executed no frames.")
        final_counters = backend.check_capacity(synchronize=False)
        final_capacity = backend.capacity_stats
        allocated_after = int(torch.cuda.memory_allocated())
        reserved_after = int(torch.cuda.memory_reserved())
        video = {
            "status": "not-requested",
            "pass": True,
        }
        if args.save_trajectory_video:
            video = _write_video(
                captured_frames,
                output_json=args.output,
                fps=float(manifest["fps"]) / args.trajectory_frame_stride,
            )

        camera_max_error = max(camera_errors)
        output_assertions_pass = all(bool(item["valid"]) for item in output_contracts)
        zero_overflow = bool(
            first_zero_overflow
            and fixed_hits["zero_overflow"]
            and fixed_invalidated["zero_overflow"]
            and capacity_warmup_zero_overflow
            and final_counters["visible_overflow"] == 0
            and final_counters["intersection_overflow"] == 0
            and all(
                not item["visible_overflow"]
                and not item["intersection_overflow"]
                for item in capacity_counters
            )
        )
        moving_cache_pass = all(item["pass"] for item in cache_events)
        output_motion_pass = output_changed_steps == input_changed_steps
        allocation_delta = allocated_after - allocated_before
        allocation_pass = allocation_delta == 0
        source_commit = resolve_source_commit()
        source_index_sha256 = hashlib.sha256(source_indices.tobytes()).hexdigest()
        result = {
            "schema_version": "isaacsim-fabric-trajectory/v1",
            "runtime": "Isaac Sim SimulationApp headless",
            "simulation_app": {
                "renderer": "MinimalRendering",
                "viewport_updates_disabled": True,
            },
            "fabric_scene_delegate": {
                "enabled": delegate_enabled,
                "extension": "omni.hydra.usdrt_delegate",
                "extension_loaded": delegate_extension_loaded,
            },
            "scene": {
                "dataset_id": args.scene_id_label,
                "author": args.scene_author,
                "license": args.scene_license,
                "source": args.scene_source,
                "path": str(Path(args.scene_path).resolve()),
                "manifest_path": str(Path(args.scene_manifest).resolve()),
                "manifest_sha256": scene_release["manifest_sha256"],
                "sha256": scene_release["sha256"],
                "file_bytes": scene_release["file_bytes"],
                "gaussian_count": scene_release["gaussian_count"],
                "gaussian_count_source": scene_release["gaussian_count_source"],
                "semantic_source": (
                    "PLY semantic_id"
                    if source_has_semantics
                    else "deterministic Gaussian-index modulo 1024 sidecar"
                ),
                "release_checksum_validation_seconds": release_validation_seconds,
                "host_load_seconds": host_load_seconds,
                "canonicalize_register_seconds": upload_seconds,
            },
            "camera_contract": {
                "schema_version": manifest["schema_version"],
                "camera_contract_sha256": manifest["camera_contract_sha256"],
                "camera_npz_sha256": camera_contract.npz_sha256,
                "camera_manifest_sha256": camera_contract.manifest_sha256,
                "route_sha256": camera_route_sha256,
                "route_sha256_source": camera_route_sha256_source,
                "source_frames": int(camera_contract.viewmats.shape[0]),
                "measured_frames": timesteps,
                "batch": args.batch,
                "phase_offsets": list(phase_offsets),
                "layout": args.trajectory_layout,
                "total_batched_views": int(source_indices.size),
                "unique_source_views": int(np.unique(source_indices).size),
                "covers_full_source_population": bool(
                    np.unique(source_indices).size
                    == camera_contract.viewmats.shape[0]
                ),
                "source_indices_sha256": source_index_sha256,
                "measured_contract": measured_contract_files,
                "start_frame": args.trajectory_start_frame,
                "stride": args.trajectory_frame_stride,
                "width": args.width,
                "height": args.height,
                "fps": float(manifest["fps"]),
                "fabric_viewmat_max_abs_error": camera_max_error,
                "usd_intrinsics_max_abs_error": intrinsics_max_abs_error,
            },
            "camera_ingestion": {
                "source": "USDRT SelectPrims CUDA Fabric selection",
                "attribute": "omni:fabric:worldMatrix",
                "runtime_camera_loop": False,
                "gpu_conversion_launches_per_batch": 1,
                "torch_output_zero_copy": True,
                "usd_authoring_loop_over_cameras": True,
                "read_calls": extension.fabric_transform_source.read_calls,
                "topology_rebuilds": extension.fabric_transform_source.topology_rebuilds,
            },
            "renderer": {
                "backend": backend.__class__.__name__,
                "pipeline": backend.pipeline_name,
                "rasterize_mode": backend.rasterize_mode,
                "covariance_epsilon": backend.covariance_epsilon,
                "configuration_contract": renderer_configuration,
                "tile_size": backend.tile_size,
                "compact_projection_cache": backend.compact_projection_cache,
                "projected_record_reuse": backend.capacity_stats[
                    "projected_record_reuse"
                ],
                "projection_cache_enabled": backend.enable_projection_cache,
                "initial_visible_capacity": initial_visible_capacity,
                "initial_intersection_capacity": initial_intersection_capacity,
                "final_visible_capacity": backend.current_visible_capacity,
                "final_intersection_capacity": backend.current_intersection_capacity,
                "max_workspace_bytes": args.max_workspace_bytes,
                "capacity_adaptation": final_capacity,
                "counters": final_counters,
                "zero_overflow": zero_overflow,
                "output_assertions_pass": output_assertions_pass,
            },
            "first_render": {
                "gpu_ms": first_gpu_ms,
                "wall_ms": first_wall_ms,
                "cache": _cache_delta(first_cache_before, first_cache_after),
                "camera_max_abs_error": first_camera_error,
                "counters": first_counters,
                "capacity_adaptation": first_capacity,
                "output_contract": first_output_contract,
            },
            "fixed_camera_cache_hit_control": fixed_hits,
            "fixed_camera_invalidated_control": fixed_invalidated,
            "capacity_warmup": {
                "enabled": args.trajectory_capacity_warmup,
                "frames": timesteps if args.trajectory_capacity_warmup else 0,
                "seconds": capacity_warmup_seconds,
                "output_contracts_pass": capacity_warmup_contracts_pass,
                "zero_overflow": capacity_warmup_zero_overflow,
            },
            "kit_settle": {
                "updates": args.kit_settle_updates,
                "seconds": kit_settle_seconds,
            },
            "moving_trajectory": {
                "frames": timesteps,
                "batch": args.batch,
                "usd_author_wall_ms": distribution(author_wall),
                "kit_update_wall_ms": distribution(kit_update_wall),
                "fabric_source_gpu_ms": distribution(fabric_gpu),
                "custom_render_gpu_ms": distribution(render_gpu),
                "fabric_and_render_gpu_ms": distribution(integrated_gpu),
                "integrated_observation_wall_ms": distribution(observation_wall),
                "validation_and_artifact_wall_ms": distribution(validation_wall),
                "samples_ms": {
                    "usd_author_wall": author_wall,
                    "kit_update_wall": kit_update_wall,
                    "fabric_source_gpu": fabric_gpu,
                    "custom_render_gpu": render_gpu,
                    "fabric_and_render_gpu": integrated_gpu,
                    "integrated_observation_wall": observation_wall,
                    "validation_and_artifact_wall": validation_wall,
                },
                "wall_images_per_second": (
                    args.batch * 1_000.0 / float(np.mean(observation_wall))
                ),
                "deadline_ms": deadline_ms,
                "deadline_misses": deadline_misses,
                "deadline_miss_rate": deadline_misses / timesteps,
                "cache_events": cache_events,
                "cache_pass": moving_cache_pass,
                "input_changed_steps": input_changed_steps,
                "output_changed_steps": output_changed_steps,
                "output_motion_pass": output_motion_pass,
                "camera_max_abs_errors": camera_errors,
                "output_contracts": output_contracts,
                "capacity_counters": capacity_counters,
            },
            "memory": {
                "allocated_before_bytes": allocated_before,
                "allocated_after_bytes": allocated_after,
                "allocation_delta_bytes": allocation_delta,
                "allocation_checkpoints_bytes": allocation_checkpoints,
                "reserved_before_bytes": reserved_before,
                "reserved_after_bytes": reserved_after,
                "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "backend_scene_bytes": int(backend.scene_bytes),
                "backend_workspace_bytes": int(backend.workspace_bytes),
                "backend_workspace_bytes_by_tensor": (
                    backend.workspace_bytes_by_tensor
                ),
                "backend_workspace_logical_bytes_by_tensor": (
                    backend.workspace_logical_bytes_by_tensor
                ),
                "backend_workspace_storage_aliases": (
                    backend.workspace_storage_aliases
                ),
                "driver_compute_process_rows_pid_used_mib": driver_process_memory(),
                "steady_state_allocation_pass": allocation_pass,
            },
            "video": video,
            "environment": {
                "source_commit": source_commit,
                "command": sys.argv,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "driver": _command_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
                ),
                "isaacsim": importlib.metadata.version("isaacsim"),
            },
            "realtime_deadline_pass": deadline_misses == 0,
            "pass": bool(
                first_output_contract["valid"]
                and fixed_hits["cache_pass"]
                and fixed_hits["output_contract"]["valid"]
                and fixed_invalidated["cache_pass"]
                and fixed_invalidated["output_contract"]["valid"]
                and capacity_warmup_contracts_pass
                and capacity_warmup_zero_overflow
                and camera_max_error <= 1.0e-4
                and output_assertions_pass
                and zero_overflow
                and moving_cache_pass
                and output_motion_pass
                and allocation_pass
                and video["pass"]
                and source_commit != "unknown"
            ),
        }
        return result
    finally:
        try:
            if service is not None:
                service.shutdown()
        finally:
            if (
                camera_source is not None
                and extension.fabric_transform_source is not None
            ):
                extension.fabric_transform_source.close()
