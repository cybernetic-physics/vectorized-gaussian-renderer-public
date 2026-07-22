import sys

import torch

from scripts import run_projection_mode_activation_gate as projection_gate
from isaacsim_gaussian_renderer.camera_math import quaternion_xyzw_to_matrix
from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle,
    compact_semantic_ids,
    home_scan_axis_camera_bundle,
    projection_activation_scene_manifest,
    projection_activation_scene_tensors,
    resolve_output_color_contract,
    scene_tensors_sha256,
    spatial_semantic_ids,
    spatial_semantic_manifest,
    synthetic_exact_intersections_per_visible,
    stress_scene_manifest,
    stress_scene_tensors,
)
from isaacsim_gaussian_renderer.quaternion import quaternion_wxyz_to_xyzw


def test_projection_activation_scene_is_anisotropic_and_rotated() -> None:
    scene = projection_activation_scene_tensors(device="cpu")
    manifest = projection_activation_scene_manifest()

    assert scene["means"].shape == (4, 3)
    assert torch.allclose(
        torch.linalg.vector_norm(scene["quats"], dim=1),
        torch.ones(4),
        atol=1.0e-6,
    )
    assert torch.all(scene["quats"][:, 1:].abs().sum(dim=1) > 0)
    assert torch.all(scene["scales"].amax(dim=1) > scene["scales"].amin(dim=1))
    assert scene["semantic_ids"].tolist() == [0, 0, 0, 0]
    assert manifest["requirements"]["off_axis_camera"] is True
    assert manifest["requirements"]["single_particle_field"] is True
    assert manifest["tensor_checksum_sha256"] == scene_tensors_sha256(scene)
    assert len(manifest["checksum_sha256"]) == 64


def test_projection_activation_scene_is_in_frame_for_gate_and_control(
    tmp_path,
    monkeypatch,
) -> None:
    scene = projection_activation_scene_tensors(device="cpu")
    rotations = quaternion_xyzw_to_matrix(quaternion_wxyz_to_xyzw(scene["quats"]))
    covariance_world = rotations @ torch.diag_embed(scene["scales"].square()) @ rotations.transpose(-1, -2)

    monkeypatch.setattr(
        sys,
        "argv",
        ["gate", "--output-root", str(tmp_path / "activation")],
    )
    gate_args = projection_gate.parse_args()
    measured_focal_scales = [lane.focal_scale for lane in projection_gate.plan_lanes(gate_args) if lane.measured]
    assert [round(value, 10) for value in measured_focal_scales] == [
        0.945,
        0.9,
        0.9,
        0.9,
        0.9,
        0.945,
    ]

    for focal_scale in sorted(set(measured_focal_scales)):
        cameras = camera_bundle(
            1,
            gate_args.width,
            gate_args.height,
            device="cpu",
            focal_scale=focal_scale,
        )
        homogeneous_means = torch.cat(
            (scene["means"], torch.ones((scene["means"].shape[0], 1))),
            dim=1,
        )
        means_camera = (cameras.viewmats[0] @ homogeneous_means.transpose(0, 1)).transpose(0, 1)[:, :3]
        camera_rotation = cameras.viewmats[0, :3, :3]
        covariance_camera = (
            camera_rotation.unsqueeze(0) @ covariance_world @ camera_rotation.transpose(0, 1).unsqueeze(0)
        )
        x, y, z = means_camera.unbind(dim=1)
        focal_x = cameras.intrinsics[0, 0, 0]
        focal_y = cameras.intrinsics[0, 1, 1]
        jacobian = torch.zeros((scene["means"].shape[0], 2, 3))
        jacobian[:, 0, 0] = focal_x / z
        jacobian[:, 0, 2] = -focal_x * x / z.square()
        jacobian[:, 1, 1] = focal_y / z
        jacobian[:, 1, 2] = -focal_y * y / z.square()
        covariance_2d = jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
        centers = torch.stack(
            (
                focal_x * x / z + cameras.intrinsics[0, 0, 2],
                focal_y * y / z + cameras.intrinsics[0, 1, 2],
            ),
            dim=1,
        )
        three_sigma_radius = 3.0 * torch.sqrt(torch.diagonal(covariance_2d, dim1=-2, dim2=-1))
        lower = centers - three_sigma_radius
        upper = centers + three_sigma_radius

        assert torch.all(z - 3.0 * scene["scales"].amax(dim=1) > 0.01)
        assert torch.all(lower >= 3.0)
        assert torch.all(upper <= 253.0)
        off_axis_angle = torch.atan2(
            torch.linalg.vector_norm(means_camera[:, :2], dim=1),
            z,
        )
        assert torch.rad2deg(off_axis_angle).max() >= 20.0


def test_home_scan_axis_camera_bundle_centers_scan_in_front() -> None:
    center = torch.tensor([-3.9, -5.65, -6.11])
    cameras = home_scan_axis_camera_bundle(
        center=center,
        bounding_radius=16.8,
        batch_size=8,
        width=128,
        height=128,
    )
    homogeneous_center = torch.cat(
        (
            center.expand(8, 3),
            torch.ones((8, 1)),
        ),
        dim=1,
    )
    camera_center = torch.bmm(
        cameras.viewmats,
        homogeneous_center.unsqueeze(-1),
    ).squeeze(-1)

    assert cameras.viewmats.shape == (8, 4, 4)
    assert cameras.intrinsics.shape == (8, 3, 3)
    assert torch.all(camera_center[:, 2] > 0)
    assert torch.all(camera_center[:, 0].abs() <= 0.10 * 16.8 + 1.0e-5)
    assert cameras.manifest["near_plane"] == 0.01
    assert cameras.manifest["far_plane"] > cameras.manifest["distance"]


def test_authored_display_output_uses_identity_transfer() -> None:
    contract = resolve_output_color_contract(
        linear_output=False,
        authored_display_output=True,
    )

    assert contract == {
        "output_srgb": False,
        "color_space": "display_srgb",
        "color_transform": "identity_authored_display_rgb",
    }


def test_exact_intersection_budget_uses_ceiling_rounded_bin_grid() -> None:
    assert (
        synthetic_exact_intersections_per_visible(
            width=128,
            height=128,
            tile_size=16,
        )
        == 8
    )
    assert (
        synthetic_exact_intersections_per_visible(
            width=130,
            height=129,
            tile_size=16,
        )
        == 13
    )
    assert (
        synthetic_exact_intersections_per_visible(
            width=256,
            height=256,
            tile_size=16,
        )
        == 32
    )


def test_spatial_semantic_ids_are_coherent_and_xyz_major() -> None:
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    semantic_ids = spatial_semantic_ids(means, grid=(2, 2, 2))

    torch.testing.assert_close(
        semantic_ids,
        torch.tensor([0, 1, 2, 7]),
    )
    manifest = spatial_semantic_manifest(
        means.shape[0],
        grid=(2, 2, 2),
        position_min=means.amin(dim=0),
        position_max=means.amax(dim=0),
    )
    assert manifest["class_count"] == 8
    assert manifest["grid"] == [2, 2, 2]
    assert len(manifest["checksum_sha256"]) == 64


def test_compact_semantic_ids_records_occupied_spatial_classes() -> None:
    source = torch.tensor([0, 7, 2, 7, 0], dtype=torch.int64)

    dense, occupied = compact_semantic_ids(source, dtype=torch.int32)

    torch.testing.assert_close(
        dense,
        torch.tensor([0, 2, 1, 2, 0], dtype=torch.int32),
    )
    assert occupied == [0, 2, 7]

    manifest = spatial_semantic_manifest(
        source.shape[0],
        grid=(2, 2, 2),
        position_min=torch.zeros(3),
        position_max=torch.ones(3),
        occupied_source_ids=occupied,
    )
    assert manifest["grid_class_count"] == 8
    assert manifest["class_count"] == 3
    assert manifest["dense_ordinal_to_grid_class"] == [0, 2, 7]


def test_anisotropic_stress_scene_is_deterministic_rotated_and_nonisotropic() -> None:
    first = stress_scene_tensors(
        "stress-anisotropic",
        1_024,
        seed=1_043,
        device="cpu",
    )
    second = stress_scene_tensors(
        "stress-anisotropic",
        1_024,
        seed=1_043,
        device="cpu",
    )

    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(first["quats"], dim=-1),
        torch.ones(1_024),
    )
    assert torch.all(first["quats"][:, 0] >= 0)
    assert torch.count_nonzero(first["quats"][:, 1:]) > 0
    assert torch.all(first["scales"].amax(dim=1) / first["scales"].amin(dim=1) > 10)
    assert set(first["semantic_ids"].tolist()) == set(range(8))

    manifest = stress_scene_manifest(
        "stress-anisotropic",
        100_000,
        1_043,
    )
    assert manifest["scene_type"] == "deterministic-adversarial-synthetic"
    assert len(manifest["checksum_sha256"]) == 64


def test_overdraw_stress_scene_is_dense_opaque_and_anisotropic() -> None:
    scene = stress_scene_tensors(
        "stress-overdraw",
        4_096,
        seed=2_043,
        device="cpu",
    )

    assert float(scene["means"][:, :2].std().item()) < 0.5
    assert float(scene["means"][:, 2].amin().item()) >= 4.25
    assert float(scene["means"][:, 2].amax().item()) <= 6.25
    assert float(scene["opacities"].amin().item()) >= 0.80
    assert float(scene["opacities"].amax().item()) <= 0.99
    assert torch.all(scene["scales"][:, :2] > scene["scales"][:, 2:])
