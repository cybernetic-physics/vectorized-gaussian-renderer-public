"""Deterministic camera paths for renderer demonstrations and visual QA."""

from __future__ import annotations

import math
from typing import Any

import torch

from .benchmark_manifest import CameraBundle, sha256_json


def _normalize(vectors: torch.Tensor, *, name: str) -> torch.Tensor:
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    if bool((norms <= torch.finfo(vectors.dtype).eps).any().item()):
        raise ValueError(f"{name} contains a zero-length direction.")
    return vectors / norms


def look_at_opencv_viewmats(
    camera_centers: torch.Tensor,
    targets: torch.Tensor,
    *,
    world_up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build OpenCV world-to-camera matrices for batched look-at poses.

    The OpenCV camera frame is ``+X`` right, ``+Y`` down, and ``+Z`` forward.
    ``world_up`` is only a roll hint; the function keeps every output rotation
    proper and orthonormal.
    """
    if (
        camera_centers.ndim != 2
        or camera_centers.shape[-1] != 3
        or targets.shape != camera_centers.shape
    ):
        raise ValueError(
            "camera_centers and targets must both have shape [B, 3]."
        )
    if not camera_centers.is_floating_point():
        raise ValueError("Camera positions must use a floating-point dtype.")
    if world_up is None:
        world_up = torch.tensor(
            [0.0, 1.0, 0.0],
            device=camera_centers.device,
            dtype=camera_centers.dtype,
        )
    if world_up.shape != (3,):
        raise ValueError("world_up must have shape [3].")
    world_up = world_up.to(
        device=camera_centers.device,
        dtype=camera_centers.dtype,
    )

    forward = _normalize(targets - camera_centers, name="look-at forward")
    up = world_up.expand_as(forward)
    right = torch.linalg.cross(forward, up, dim=-1)
    nearly_parallel = (
        torch.linalg.vector_norm(right, dim=-1)
        <= 1.0e-5
    )
    if bool(nearly_parallel.any().item()):
        fallback_up = torch.tensor(
            [0.0, 0.0, 1.0],
            device=camera_centers.device,
            dtype=camera_centers.dtype,
        ).expand_as(forward)
        fallback_right = torch.linalg.cross(
            forward,
            fallback_up,
            dim=-1,
        )
        right = torch.where(
            nearly_parallel[:, None],
            fallback_right,
            right,
        )
    right = _normalize(right, name="look-at right")
    down = _normalize(
        torch.linalg.cross(forward, right, dim=-1),
        name="look-at down",
    )
    camera_to_world = torch.stack((right, down, forward), dim=-1)
    world_to_camera = camera_to_world.transpose(-1, -2)

    viewmats = torch.eye(
        4,
        device=camera_centers.device,
        dtype=camera_centers.dtype,
    ).repeat(camera_centers.shape[0], 1, 1)
    viewmats[:, :3, :3] = world_to_camera
    viewmats[:, :3, 3] = -torch.bmm(
        world_to_camera,
        camera_centers.unsqueeze(-1),
    ).squeeze(-1)
    return viewmats.contiguous()


def cinematic_orbit_camera_bundle(
    *,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    frame_count: int,
    width: int,
    height: int,
    focal_scale: float = 0.72,
    orbit_margin: float = 2.0,
    elevation_scale: float = 0.58,
    start_angle_degrees: float = -125.0,
    camera_path: str = "cinematic-robust-bounds-orbit",
    bounds_policy: str = "deterministic-sampled-0.5-to-99.5-percentile",
) -> CameraBundle:
    """Create a smooth elevated orbit that exposes a flat room-scale scan.

    The path is closed: frame ``N`` would equal frame zero. The endpoint is
    omitted so encoded videos can loop without a duplicated final frame.
    """
    if bounds_min.shape != (3,) or bounds_max.shape != (3,):
        raise ValueError("bounds_min and bounds_max must have shape [3].")
    if (
        frame_count <= 0
        or width <= 0
        or height <= 0
        or focal_scale <= 0
        or orbit_margin < 0
        or elevation_scale <= 0
    ):
        raise ValueError("Flyby dimensions and path scales must be valid.")
    if not bool(torch.all(bounds_max > bounds_min).item()):
        raise ValueError("Every robust bound maximum must exceed its minimum.")

    device = bounds_min.device
    dtype = bounds_min.dtype
    center = (bounds_min + bounds_max) * 0.5
    extent = bounds_max - bounds_min
    half_extent = extent * 0.5
    horizontal_radius = torch.linalg.vector_norm(
        half_extent[[0, 2]]
    )
    orbit_radius_x = float(
        half_extent[0].item() * 1.12 + orbit_margin
    )
    orbit_radius_z = float(
        half_extent[2].item() * 1.28 + orbit_margin
    )
    elevation = max(
        float(extent[1].item()) * 1.25,
        float(horizontal_radius.item()) * elevation_scale,
    )

    start_angle = math.radians(start_angle_degrees)
    phase = (
        torch.arange(frame_count, device=device, dtype=dtype)
        * (2.0 * math.pi / frame_count)
        + start_angle
    )
    camera_centers = center.repeat(frame_count, 1)
    camera_centers[:, 0] += orbit_radius_x * torch.cos(phase)
    camera_centers[:, 2] += orbit_radius_z * torch.sin(phase)
    camera_centers[:, 1] += elevation * (
        1.0 + 0.08 * torch.sin(phase * 2.0)
    )

    targets = center.repeat(frame_count, 1)
    targets[:, 0] += (
        half_extent[0] * 0.14 * torch.sin(phase - 0.35)
    )
    targets[:, 2] += (
        half_extent[2] * 0.10 * torch.cos(phase * 2.0)
    )
    targets[:, 1] += extent[1] * 0.08
    viewmats = look_at_opencv_viewmats(camera_centers, targets)

    focal_x = focal_scale * float(width)
    focal_y = focal_x * float(height) / float(width)
    intrinsics = torch.zeros(
        (frame_count, 3, 3),
        device=device,
        dtype=dtype,
    )
    intrinsics[:, 0, 0] = focal_x
    intrinsics[:, 1, 1] = focal_y
    intrinsics[:, 0, 2] = float(width) * 0.5
    intrinsics[:, 1, 2] = float(height) * 0.5
    intrinsics[:, 2, 2] = 1.0

    camera_distances = torch.linalg.vector_norm(
        camera_centers - targets,
        dim=-1,
    )
    scene_radius = float(
        torch.linalg.vector_norm(half_extent).item()
    )
    far_plane = max(
        100.0,
        float(camera_distances.max().item()) + 2.0 * scene_radius,
    )
    manifest: dict[str, Any] = {
        "version": "benchmark-camera-manifest/v1",
        "camera_model": "opencv-pinhole-world-to-camera",
        "camera_path": camera_path,
        "bounds_policy": bounds_policy,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "focal_scale": focal_scale,
        "start_angle_degrees": start_angle_degrees,
        "closed_loop_endpoint_omitted": True,
        "center": center.detach().cpu().tolist(),
        "bounds_min": bounds_min.detach().cpu().tolist(),
        "bounds_max": bounds_max.detach().cpu().tolist(),
        "extent": extent.detach().cpu().tolist(),
        "orbit_radius_x": orbit_radius_x,
        "orbit_radius_z": orbit_radius_z,
        "elevation": elevation,
        "camera_distance_min": float(camera_distances.min().item()),
        "camera_distance_max": float(camera_distances.max().item()),
        "near_plane": 0.01,
        "far_plane": far_plane,
    }
    manifest["checksum_sha256"] = sha256_json(manifest)
    return CameraBundle(
        viewmats=viewmats,
        intrinsics=intrinsics.contiguous(),
        manifest=manifest,
    )


def cinematic_walkthrough_camera_bundle(
    *,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    frame_count: int,
    width: int,
    height: int,
    focal_scale: float = 0.72,
    path_radius_x_fraction: float = 0.58,
    path_radius_z_fraction: float = 0.52,
    eye_height_fraction: float = 0.60,
    vertical_bob_fraction: float = 0.025,
    lookahead_phase: float = 0.12,
    start_angle_degrees: float = -90.0,
    camera_path: str = "cinematic-interior-figure-eight-walkthrough",
    bounds_policy: str = "deterministic-sampled-0.5-to-99.5-percentile",
) -> CameraBundle:
    """Create a smooth eye-level figure-eight path through a room-scale scan.

    Positions stay inside the robust horizontal bounds and use a vertical
    fraction measured from the robust minimum, approximating a human eye
    height without assuming metric scene units. Cameras look ahead along the
    path instead of at the scene center, so the result reads as a walkthrough
    rather than an exterior orbit.
    """
    if bounds_min.shape != (3,) or bounds_max.shape != (3,):
        raise ValueError("bounds_min and bounds_max must have shape [3].")
    if (
        frame_count <= 0
        or width <= 0
        or height <= 0
        or focal_scale <= 0
        or path_radius_x_fraction <= 0
        or path_radius_z_fraction <= 0
        or lookahead_phase <= 0
    ):
        raise ValueError("Walkthrough dimensions and path scales must be valid.")
    if not (
        path_radius_x_fraction < 1.0
        and path_radius_z_fraction < 1.0
        and 0.0 < eye_height_fraction < 1.0
        and 0.0 <= vertical_bob_fraction < 0.25
    ):
        raise ValueError("Walkthrough path fractions are outside valid ranges.")
    if not bool(torch.all(bounds_max > bounds_min).item()):
        raise ValueError("Every robust bound maximum must exceed its minimum.")

    device = bounds_min.device
    dtype = bounds_min.dtype
    center = (bounds_min + bounds_max) * 0.5
    extent = bounds_max - bounds_min
    half_extent = extent * 0.5
    radius_x = float(half_extent[0].item()) * path_radius_x_fraction
    radius_z = float(half_extent[2].item()) * path_radius_z_fraction
    eye_y = float(
        (
            bounds_min[1]
            + extent[1] * eye_height_fraction
        ).item()
    )
    vertical_bob = float(extent[1].item()) * vertical_bob_fraction

    start_angle = math.radians(start_angle_degrees)
    phase = (
        torch.arange(frame_count, device=device, dtype=dtype)
        * (2.0 * math.pi / frame_count)
        + start_angle
    )

    def positions_at(path_phase: torch.Tensor) -> torch.Tensor:
        positions = center.repeat(path_phase.shape[0], 1)
        positions[:, 0] += radius_x * torch.sin(path_phase)
        positions[:, 2] += radius_z * torch.sin(path_phase * 2.0)
        positions[:, 1] = (
            eye_y
            + vertical_bob
            * torch.sin(path_phase * 2.0 + 0.4)
        )
        return positions

    camera_centers = positions_at(phase)
    targets = positions_at(phase + lookahead_phase)
    targets[:, 1] -= float(extent[1].item()) * 0.015
    viewmats = look_at_opencv_viewmats(camera_centers, targets)

    focal_x = focal_scale * float(width)
    focal_y = focal_x * float(height) / float(width)
    intrinsics = torch.zeros(
        (frame_count, 3, 3),
        device=device,
        dtype=dtype,
    )
    intrinsics[:, 0, 0] = focal_x
    intrinsics[:, 1, 1] = focal_y
    intrinsics[:, 0, 2] = float(width) * 0.5
    intrinsics[:, 1, 2] = float(height) * 0.5
    intrinsics[:, 2, 2] = 1.0

    target_distances = torch.linalg.vector_norm(
        targets - camera_centers,
        dim=-1,
    )
    scene_radius = float(torch.linalg.vector_norm(half_extent).item())
    far_plane = max(100.0, scene_radius * 4.0)
    manifest: dict[str, Any] = {
        "version": "benchmark-camera-manifest/v1",
        "camera_model": "opencv-pinhole-world-to-camera",
        "camera_path": camera_path,
        "bounds_policy": bounds_policy,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "focal_scale": focal_scale,
        "start_angle_degrees": start_angle_degrees,
        "closed_loop_endpoint_omitted": True,
        "center": center.detach().cpu().tolist(),
        "bounds_min": bounds_min.detach().cpu().tolist(),
        "bounds_max": bounds_max.detach().cpu().tolist(),
        "extent": extent.detach().cpu().tolist(),
        "path_radius_x": radius_x,
        "path_radius_z": radius_z,
        "path_radius_x_fraction": path_radius_x_fraction,
        "path_radius_z_fraction": path_radius_z_fraction,
        "eye_height": eye_y,
        "eye_height_fraction": eye_height_fraction,
        "vertical_bob": vertical_bob,
        "vertical_bob_fraction": vertical_bob_fraction,
        "lookahead_phase": lookahead_phase,
        "target_distance_min": float(target_distances.min().item()),
        "target_distance_max": float(target_distances.max().item()),
        "near_plane": 0.01,
        "far_plane": far_plane,
    }
    manifest["checksum_sha256"] = sha256_json(manifest)
    return CameraBundle(
        viewmats=viewmats,
        intrinsics=intrinsics.contiguous(),
        manifest=manifest,
    )


def scripted_walkthrough_camera_bundle(
    *,
    path_points: torch.Tensor,
    frame_count: int,
    width: int,
    height: int,
    focal_scale: float = 0.72,
    lookahead_frames: int = 6,
    pitch_down_fraction: float = 0.015,
    world_up: torch.Tensor | None = None,
    route_sha256: str | None = None,
    camera_path: str = "scripted-clearance-aware-walkthrough",
) -> CameraBundle:
    """Resample an open world-space path into smooth forward-looking cameras."""
    if (
        path_points.ndim != 2
        or path_points.shape[1] != 3
        or path_points.shape[0] < 2
    ):
        raise ValueError("path_points must have shape [N, 3], N >= 2.")
    if (
        frame_count < 2
        or width <= 0
        or height <= 0
        or focal_scale <= 0
        or lookahead_frames <= 0
        or pitch_down_fraction < 0
    ):
        raise ValueError("Scripted walkthrough parameters are invalid.")
    if not path_points.is_floating_point():
        raise ValueError("path_points must use a floating-point dtype.")

    segment_vectors = torch.diff(path_points, dim=0)
    segment_lengths = torch.linalg.vector_norm(segment_vectors, dim=-1)
    if bool((segment_lengths <= 1.0e-7).any().item()):
        keep = torch.cat(
            (
                torch.ones(
                    1,
                    device=path_points.device,
                    dtype=torch.bool,
                ),
                segment_lengths > 1.0e-7,
            )
        )
        path_points = path_points[keep]
        if path_points.shape[0] < 2:
            raise ValueError("Scripted path has no nonzero-length segments.")
        segment_vectors = torch.diff(path_points, dim=0)
        segment_lengths = torch.linalg.vector_norm(
            segment_vectors,
            dim=-1,
        )

    cumulative = torch.cat(
        (
            torch.zeros(
                1,
                device=path_points.device,
                dtype=path_points.dtype,
            ),
            torch.cumsum(segment_lengths, dim=0),
        )
    )
    total_length = cumulative[-1]
    sample_distances = torch.linspace(
        0.0,
        float(total_length.item()),
        frame_count,
        device=path_points.device,
        dtype=path_points.dtype,
    )
    segment_indices = torch.searchsorted(
        cumulative[1:],
        sample_distances,
        right=False,
    ).clamp_max(path_points.shape[0] - 2)
    local_distances = sample_distances - cumulative[segment_indices]
    alpha = (
        local_distances / segment_lengths[segment_indices]
    ).unsqueeze(-1)
    camera_centers = (
        path_points[segment_indices]
        + alpha * segment_vectors[segment_indices]
    )

    indices = torch.arange(frame_count, device=path_points.device)
    forward_indices = (indices + lookahead_frames).clamp_max(
        frame_count - 1
    )
    backward_indices = (indices - 1).clamp_min(0)
    directions = (
        camera_centers[forward_indices]
        - camera_centers[backward_indices]
    )
    directions = _normalize(directions, name="scripted path direction")
    target_distance = max(
        float(total_length.item()) / frame_count * lookahead_frames,
        0.5,
    )
    targets = camera_centers + directions * target_distance
    vertical_extent = float(
        path_points[:, 1].max().item()
        - path_points[:, 1].min().item()
    )
    pitch_scale = max(vertical_extent, 1.0)
    if world_up is None:
        world_up = torch.tensor(
            [0.0, 1.0, 0.0],
            device=path_points.device,
            dtype=path_points.dtype,
        )
    world_up = world_up.to(
        device=path_points.device,
        dtype=path_points.dtype,
    )
    targets = (
        targets
        - world_up.unsqueeze(0)
        * (pitch_scale * pitch_down_fraction)
    )
    viewmats = look_at_opencv_viewmats(
        camera_centers,
        targets,
        world_up=world_up,
    )

    focal_x = focal_scale * float(width)
    focal_y = focal_x * float(height) / float(width)
    intrinsics = torch.zeros(
        (frame_count, 3, 3),
        device=path_points.device,
        dtype=path_points.dtype,
    )
    intrinsics[:, 0, 0] = focal_x
    intrinsics[:, 1, 1] = focal_y
    intrinsics[:, 0, 2] = float(width) * 0.5
    intrinsics[:, 1, 2] = float(height) * 0.5
    intrinsics[:, 2, 2] = 1.0

    spatial_extent = path_points.max(dim=0).values - path_points.min(
        dim=0
    ).values
    far_plane = max(
        100.0,
        float(torch.linalg.vector_norm(spatial_extent).item()) * 4.0,
    )
    manifest: dict[str, Any] = {
        "version": "benchmark-camera-manifest/v1",
        "camera_model": "opencv-pinhole-world-to-camera",
        "camera_path": camera_path,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "focal_scale": focal_scale,
        "closed_loop_endpoint_omitted": False,
        "path_endpoint_included": True,
        "source_path_point_count": int(path_points.shape[0]),
        "path_length": float(total_length.item()),
        "lookahead_frames": lookahead_frames,
        "pitch_down_fraction": pitch_down_fraction,
        "world_up": world_up.detach().cpu().tolist(),
        "path_min": path_points.min(dim=0).values.detach().cpu().tolist(),
        "path_max": path_points.max(dim=0).values.detach().cpu().tolist(),
        "near_plane": 0.01,
        "far_plane": far_plane,
    }
    if route_sha256 is not None:
        manifest["route_sha256"] = route_sha256
    manifest["checksum_sha256"] = sha256_json(manifest)
    return CameraBundle(
        viewmats=viewmats.contiguous(),
        intrinsics=intrinsics.contiguous(),
        manifest=manifest,
    )
