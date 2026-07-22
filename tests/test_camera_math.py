import torch

from isaacsim_gaussian_renderer.camera_math import (
    isaac_world_cameras_to_viewmats,
    opencv_viewmats_to_usd_camera_world_matrices,
    quaternion_xyzw_to_matrix,
    usd_camera_world_matrices_to_viewmats,
)


def test_identity_quaternion() -> None:
    matrix = quaternion_xyzw_to_matrix(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(matrix, torch.eye(3).unsqueeze(0))


def test_viewmat_shape_and_origin() -> None:
    positions = torch.zeros((4, 3))
    orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(4, 1)
    viewmats = isaac_world_cameras_to_viewmats(positions, orientations)
    assert viewmats.shape == (4, 4, 4)
    torch.testing.assert_close(viewmats[:, :3, 3], torch.zeros((4, 3)))


def test_usd_camera_world_matrix_to_opencv_viewmat() -> None:
    world = torch.eye(4).repeat(2, 1, 1)
    world[0, 3, :3] = torch.tensor([1.0, 2.0, 3.0])
    world[1, 3, :3] = torch.tensor([-2.0, 0.5, 4.0])

    viewmats = usd_camera_world_matrices_to_viewmats(world)

    expected_rotation = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    torch.testing.assert_close(
        viewmats[:, :3, :3],
        expected_rotation.repeat(2, 1, 1),
    )
    torch.testing.assert_close(
        viewmats[:, :3, 3],
        torch.tensor([[-1.0, 2.0, 3.0], [2.0, 0.5, 4.0]]),
    )


def test_usd_camera_world_matrix_requires_batch_shape() -> None:
    try:
        usd_camera_world_matrices_to_viewmats(torch.eye(4))
    except ValueError as error:
        assert "[B, 4, 4]" in str(error)
    else:
        raise AssertionError("Expected invalid world-matrix shape to fail.")


def test_opencv_usd_camera_conversion_round_trip() -> None:
    viewmats = torch.eye(4).repeat(3, 1, 1)
    viewmats[0, :3, 3] = torch.tensor([-1.0, 2.0, -3.0])
    viewmats[1, :3, :3] = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    viewmats[1, :3, 3] = torch.tensor([2.0, -0.5, 4.0])
    viewmats[2, :3, :3] = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    viewmats[2, :3, 3] = torch.tensor([-4.0, 1.0, 0.25])

    usd_world = opencv_viewmats_to_usd_camera_world_matrices(
        viewmats
    )
    round_trip = usd_camera_world_matrices_to_viewmats(usd_world)

    torch.testing.assert_close(round_trip, viewmats)
    camera_centers = torch.linalg.inv(viewmats)[:, :3, 3]
    torch.testing.assert_close(usd_world[:, 3, :3], camera_centers)


def test_opencv_to_usd_camera_requires_batch_shape() -> None:
    try:
        opencv_viewmats_to_usd_camera_world_matrices(torch.eye(4))
    except ValueError as error:
        assert "[B, 4, 4]" in str(error)
    else:
        raise AssertionError("Expected invalid view-matrix shape to fail.")
