import torch

from isaacsim_gaussian_renderer.camera_paths import (
    cinematic_orbit_camera_bundle,
    cinematic_walkthrough_camera_bundle,
    look_at_opencv_viewmats,
    scripted_walkthrough_camera_bundle,
)


def test_look_at_places_targets_on_positive_camera_z() -> None:
    centers = torch.tensor(
        [[0.0, 5.0, -10.0], [8.0, 4.0, 0.0]],
        dtype=torch.float32,
    )
    targets = torch.zeros_like(centers)

    viewmats = look_at_opencv_viewmats(centers, targets)
    homogeneous_targets = torch.cat(
        (targets, torch.ones((2, 1))),
        dim=1,
    )
    camera_targets = torch.bmm(
        viewmats,
        homogeneous_targets.unsqueeze(-1),
    ).squeeze(-1)

    assert torch.all(camera_targets[:, 2] > 0)
    torch.testing.assert_close(
        camera_targets[:, :2],
        torch.zeros((2, 2)),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    rotations = viewmats[:, :3, :3]
    torch.testing.assert_close(
        torch.bmm(rotations, rotations.transpose(-1, -2)),
        torch.eye(3).repeat(2, 1, 1),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        torch.linalg.det(rotations),
        torch.ones(2),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_cinematic_orbit_is_deterministic_and_closed() -> None:
    bounds_min = torch.tensor([-16.3, -2.62, -14.12])
    bounds_max = torch.tensor([9.0, 0.07, 1.45])
    first = cinematic_orbit_camera_bundle(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        frame_count=24,
        width=512,
        height=512,
    )
    second = cinematic_orbit_camera_bundle(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        frame_count=24,
        width=512,
        height=512,
    )

    torch.testing.assert_close(first.viewmats, second.viewmats)
    torch.testing.assert_close(first.intrinsics, second.intrinsics)
    assert first.manifest == second.manifest
    assert first.manifest["closed_loop_endpoint_omitted"]
    assert first.viewmats.shape == (24, 4, 4)
    assert first.intrinsics.shape == (24, 3, 3)
    camera_centers = torch.linalg.inv(first.viewmats)[:, :3, 3]
    assert torch.all(
        camera_centers[:, 1]
        > bounds_max[1]
    )


def test_cinematic_walkthrough_is_eye_level_and_looks_forward() -> None:
    bounds_min = torch.tensor([-16.3, -2.62, -14.12])
    bounds_max = torch.tensor([9.0, 0.07, 1.45])
    first = cinematic_walkthrough_camera_bundle(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        frame_count=24,
        width=512,
        height=512,
    )
    second = cinematic_walkthrough_camera_bundle(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        frame_count=24,
        width=512,
        height=512,
    )

    torch.testing.assert_close(first.viewmats, second.viewmats)
    torch.testing.assert_close(first.intrinsics, second.intrinsics)
    assert first.manifest == second.manifest
    assert first.manifest["camera_path"].endswith("walkthrough")
    camera_centers = torch.linalg.inv(first.viewmats)[:, :3, 3]
    assert torch.all(camera_centers[:, 0] > bounds_min[0])
    assert torch.all(camera_centers[:, 0] < bounds_max[0])
    assert torch.all(camera_centers[:, 1] > bounds_min[1])
    assert torch.all(camera_centers[:, 1] < bounds_max[1])
    assert torch.all(camera_centers[:, 2] > bounds_min[2])
    assert torch.all(camera_centers[:, 2] < bounds_max[2])
    camera_forward = torch.linalg.inv(first.viewmats)[:, :3, 2]
    movement = torch.roll(camera_centers, shifts=-1, dims=0) - camera_centers
    assert torch.all(
        torch.sum(camera_forward * movement, dim=-1) > 0
    )


def test_scripted_walkthrough_resamples_open_path() -> None:
    path_points = torch.tensor(
        [
            [0.0, 1.5, 0.0],
            [2.0, 1.5, 0.0],
            [2.0, 1.5, 3.0],
        ],
        dtype=torch.float32,
    )
    bundle = scripted_walkthrough_camera_bundle(
        path_points=path_points,
        frame_count=20,
        width=320,
        height=240,
        lookahead_frames=3,
        route_sha256="route-test",
    )

    centers = torch.linalg.inv(bundle.viewmats)[:, :3, 3]
    torch.testing.assert_close(centers[0], path_points[0])
    torch.testing.assert_close(centers[-1], path_points[-1])
    assert bundle.manifest["path_endpoint_included"]
    assert not bundle.manifest["closed_loop_endpoint_omitted"]
    assert bundle.manifest["route_sha256"] == "route-test"
    assert bundle.manifest["world_up"] == [0.0, 1.0, 0.0]
    assert bundle.manifest["path_length"] == 5.0
    forward = torch.linalg.inv(bundle.viewmats)[:, :3, 2]
    movement = centers[1:] - centers[:-1]
    assert torch.all(
        torch.sum(forward[:-1] * movement, dim=-1) > 0
    )
