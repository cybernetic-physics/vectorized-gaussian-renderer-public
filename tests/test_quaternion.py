import pytest
import torch

from isaacsim_gaussian_renderer.quaternion import (
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_wxyz,
)


def test_wxyz_xyzw_round_trip_preserves_anisotropic_rotations() -> None:
    wxyz = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, -0.25, 0.75, -0.125],
        ],
        dtype=torch.float32,
    )

    xyzw = quaternion_wxyz_to_xyzw(wxyz)

    torch.testing.assert_close(
        xyzw,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [-0.25, 0.75, -0.125, 0.5],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(quaternion_xyzw_to_wxyz(xyzw), wxyz)
    assert xyzw.is_contiguous()


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (3,),
        (2, 5),
    ],
)
def test_quaternion_order_conversion_rejects_invalid_shapes(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="final dimension"):
        quaternion_wxyz_to_xyzw(torch.empty(shape))
