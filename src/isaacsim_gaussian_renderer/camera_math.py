"""Camera convention conversion utilities."""

from __future__ import annotations

import torch


def quaternion_xyzw_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert normalized or unnormalized XYZW quaternions to rotation matrices."""
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    x, y, z, w = q.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def isaac_world_cameras_to_viewmats(positions: torch.Tensor, orientations_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert Isaac world-camera poses to renderer world-to-camera matrices.

    Isaac's camera frame uses +X forward and +Z up. The renderer contract uses
    OpenCV camera axes: +X right, +Y down, +Z forward.
    """
    world_from_isaac = quaternion_xyzw_to_matrix(orientations_xyzw)
    isaac_from_opencv = torch.tensor(
        (
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ),
        device=positions.device,
        dtype=positions.dtype,
    )
    world_from_opencv = world_from_isaac @ isaac_from_opencv

    viewmats = torch.eye(4, device=positions.device, dtype=positions.dtype).expand(positions.shape[0], 4, 4).clone()
    rotation = world_from_opencv.transpose(-1, -2)
    viewmats[:, :3, :3] = rotation
    viewmats[:, :3, 3] = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)
    return viewmats.contiguous()


def usd_camera_world_matrices_to_viewmats(world_matrices: torch.Tensor) -> torch.Tensor:
    """Convert Fabric/Gf row-vector USD camera matrices to OpenCV view matrices.

    ``omni:fabric:worldMatrix`` follows the Gf row-vector convention, with
    translation in row three. USD cameras look down local ``-Z`` with ``+Y``
    up; the renderer uses OpenCV ``+Z`` forward with ``+Y`` down.
    """
    if world_matrices.ndim != 3 or world_matrices.shape[1:] != (4, 4):
        raise ValueError(
            "world_matrices must have shape [B, 4, 4]; "
            f"got {tuple(world_matrices.shape)}."
        )

    rotation = world_matrices[:, :3, :3].clone()
    rotation[:, 1:3, :] = -rotation[:, 1:3, :]
    translation = world_matrices[:, 3, :3]
    viewmats = torch.eye(
        4,
        device=world_matrices.device,
        dtype=world_matrices.dtype,
    ).repeat(world_matrices.shape[0], 1, 1)
    viewmats[:, :3, :3] = rotation
    viewmats[:, :3, 3] = -torch.bmm(
        rotation,
        translation.unsqueeze(-1),
    ).squeeze(-1)
    return viewmats.contiguous()


def opencv_viewmats_to_usd_camera_world_matrices(
    viewmats: torch.Tensor,
) -> torch.Tensor:
    """Convert OpenCV view matrices to Gf row-vector USD camera matrices.

    This is the inverse of :func:`usd_camera_world_matrices_to_viewmats`.
    The returned matrices use translation in row three and can be written to
    OVRTX's ``omni:xform`` attribute with
    ``Semantic.XFORM_MAT4x4``.
    """
    if viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4):
        raise ValueError(
            "viewmats must have shape [B, 4, 4]; "
            f"got {tuple(viewmats.shape)}."
        )

    rotation = viewmats[:, :3, :3]
    camera_centers = -torch.bmm(
        rotation.transpose(-1, -2),
        viewmats[:, :3, 3].unsqueeze(-1),
    ).squeeze(-1)
    world_matrices = torch.eye(
        4,
        device=viewmats.device,
        dtype=viewmats.dtype,
    ).repeat(viewmats.shape[0], 1, 1)
    world_matrices[:, :3, :3] = rotation
    world_matrices[:, 1:3, :3] = -world_matrices[:, 1:3, :3]
    world_matrices[:, 3, :3] = camera_centers
    return world_matrices.contiguous()
