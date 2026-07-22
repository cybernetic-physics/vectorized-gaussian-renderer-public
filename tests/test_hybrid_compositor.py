import pytest
import torch

from isaacsim_gaussian_renderer.hybrid_compositor import (
    composite_gaussian_and_mesh,
)


def _layer(rgb, alpha, depth):
    return (
        torch.tensor(rgb, dtype=torch.float32).reshape(1, 1, 1, 3),
        torch.tensor([alpha], dtype=torch.float32).reshape(1, 1, 1, 1),
        torch.tensor([depth], dtype=torch.float32).reshape(1, 1, 1, 1),
    )


def test_mesh_in_front_depth_composites_over_gaussian() -> None:
    gaussian = _layer((1.0, 0.0, 0.0), 1.0, 3.0)
    mesh = _layer((0.0, 0.0, 1.0), 0.5, 2.0)
    output = composite_gaussian_and_mesh(*gaussian, *mesh)
    torch.testing.assert_close(output.rgb, torch.tensor([[[[0.5, 0.0, 0.5]]]]))
    torch.testing.assert_close(output.alpha, torch.ones((1, 1, 1, 1)))
    torch.testing.assert_close(output.depth, torch.full((1, 1, 1, 1), 2.0))
    assert bool(output.mesh_in_front.item())


def test_gaussian_in_front_occludes_mesh_by_depth() -> None:
    gaussian = _layer((1.0, 0.0, 0.0), 1.0, 1.0)
    mesh = _layer((0.0, 0.0, 1.0), 1.0, 2.0)
    output = composite_gaussian_and_mesh(*gaussian, *mesh)
    torch.testing.assert_close(output.rgb, torch.tensor([[[[1.0, 0.0, 0.0]]]]))
    assert not bool(output.mesh_in_front.item())


def test_empty_gaussian_pixel_selects_mesh_even_with_infinite_depth() -> None:
    gaussian = _layer((0.0, 0.0, 0.0), 0.0, float("inf"))
    mesh = _layer((0.0, 1.0, 0.0), 1.0, 4.0)
    output = composite_gaussian_and_mesh(*gaussian, *mesh)
    torch.testing.assert_close(output.rgb, torch.tensor([[[[0.0, 1.0, 0.0]]]]))
    torch.testing.assert_close(output.depth, torch.full((1, 1, 1, 1), 4.0))
    assert bool(output.mesh_in_front.item())


def test_exact_depth_tie_keeps_gaussian_front_and_premultiplied_blends() -> None:
    gaussian = _layer((1.0, 0.0, 0.0), 0.5, 2.0)
    mesh = _layer((0.0, 0.0, 1.0), 0.5, 2.0)
    output = composite_gaussian_and_mesh(*gaussian, *mesh)
    assert not bool(output.mesh_in_front.item())
    torch.testing.assert_close(output.alpha, torch.full((1, 1, 1, 1), 0.75))
    torch.testing.assert_close(
        output.rgb,
        torch.tensor([[[[2.0 / 3.0, 0.0, 1.0 / 3.0]]]]),
    )
    torch.testing.assert_close(output.depth, torch.full((1, 1, 1, 1), 2.0))


def test_nan_mesh_depth_is_invalid_and_gaussian_wins() -> None:
    gaussian = _layer((1.0, 0.0, 0.0), 1.0, 3.0)
    mesh = _layer((0.0, 0.0, 1.0), 1.0, float("nan"))
    output = composite_gaussian_and_mesh(*gaussian, *mesh)
    assert not bool(output.mesh_in_front.item())
    torch.testing.assert_close(output.rgb, torch.tensor([[[[1.0, 0.0, 0.0]]]]))
    torch.testing.assert_close(output.depth, torch.full((1, 1, 1, 1), 3.0))
    assert bool(torch.isfinite(output.rgb).all().item())


def test_invalid_mesh_depth_cannot_alpha_blend_behind_valid_gaussian() -> None:
    gaussian = _layer((1.0, 0.0, 0.0), 0.5, 3.0)
    mesh = _layer((0.0, 0.0, 1.0), 1.0, float("nan"))

    output = composite_gaussian_and_mesh(*gaussian, *mesh)

    torch.testing.assert_close(output.rgb, torch.tensor([[[[1.0, 0.0, 0.0]]]]))
    torch.testing.assert_close(output.alpha, torch.full((1, 1, 1, 1), 0.5))
    torch.testing.assert_close(output.depth, torch.full((1, 1, 1, 1), 3.0))


def test_invalid_gaussian_depth_cannot_alpha_blend_behind_valid_mesh() -> None:
    gaussian = _layer((1.0, 0.0, 0.0), 1.0, float("nan"))
    mesh = _layer((0.0, 0.0, 1.0), 0.5, 2.0)

    output = composite_gaussian_and_mesh(*gaussian, *mesh)

    torch.testing.assert_close(output.rgb, torch.tensor([[[[0.0, 0.0, 1.0]]]]))
    torch.testing.assert_close(output.alpha, torch.full((1, 1, 1, 1), 0.5))
    torch.testing.assert_close(output.depth, torch.full((1, 1, 1, 1), 2.0))


def test_both_layers_invalid_yields_empty_pixel_with_infinite_depth() -> None:
    gaussian = _layer((0.3, 0.3, 0.3), 0.0, 3.0)
    mesh = _layer((0.7, 0.7, 0.7), 0.0, 2.0)
    output = composite_gaussian_and_mesh(*gaussian, *mesh)
    torch.testing.assert_close(output.rgb, torch.zeros((1, 1, 1, 3)))
    torch.testing.assert_close(output.alpha, torch.zeros((1, 1, 1, 1)))
    assert bool(torch.isinf(output.depth).item())
    assert bool((output.depth > 0).item())
    assert not bool(output.mesh_in_front.item())


def test_layer_shape_mismatch_raises() -> None:
    gaussian = _layer((1.0, 0.0, 0.0), 1.0, 3.0)
    mesh_rgb = torch.zeros((1, 1, 2, 3))
    mesh_alpha = torch.zeros((1, 1, 2, 1))
    mesh_depth = torch.zeros((1, 1, 2, 1))
    with pytest.raises(ValueError, match="equal shapes"):
        composite_gaussian_and_mesh(*gaussian, mesh_rgb, mesh_alpha, mesh_depth)


def test_invalid_blend_parameters_raise() -> None:
    gaussian = _layer((1.0, 0.0, 0.0), 1.0, 3.0)
    mesh = _layer((0.0, 0.0, 1.0), 1.0, 2.0)
    with pytest.raises(ValueError, match="min_alpha"):
        composite_gaussian_and_mesh(*gaussian, *mesh, min_alpha=1.5)
    with pytest.raises(ValueError, match="depth_epsilon"):
        composite_gaussian_and_mesh(*gaussian, *mesh, depth_epsilon=-1.0)
