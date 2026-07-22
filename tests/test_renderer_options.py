from __future__ import annotations

import math

import pytest

from benchmarks.renderer_options import (
    GSPLAT_RENDERED_OUTPUTS,
    gsplat_config_id,
    rasterize_mode_qualified,
    resolve_covariance_epsilon,
)


@pytest.mark.parametrize(
    (
        "rasterize_mode",
        "ray_gaussian_evaluation",
        "compact_projection_cache",
        "expected",
    ),
    [
        ("classic", True, False, 0.0),
        ("classic", False, True, 0.0),
        ("classic", False, False, 0.3),
        ("antialiased", False, False, 0.3),
        ("antialiased", False, True, 0.3),
    ],
)
def test_resolve_covariance_epsilon_mode_aware_defaults(
    rasterize_mode: str,
    ray_gaussian_evaluation: bool,
    compact_projection_cache: bool,
    expected: float,
) -> None:
    assert (
        resolve_covariance_epsilon(
            None,
            rasterize_mode=rasterize_mode,
            ray_gaussian_evaluation=ray_gaussian_evaluation,
            compact_projection_cache=compact_projection_cache,
        )
        == expected
    )


@pytest.mark.parametrize("explicit", [0.0, 0.125])
def test_resolve_covariance_epsilon_preserves_explicit_values(
    explicit: float,
) -> None:
    assert (
        resolve_covariance_epsilon(
            explicit,
            rasterize_mode="antialiased",
            ray_gaussian_evaluation=False,
            compact_projection_cache=True,
        )
        == explicit
    )


@pytest.mark.parametrize("invalid", [-0.1, math.inf, math.nan])
def test_resolve_covariance_epsilon_rejects_invalid_values(
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        resolve_covariance_epsilon(
            invalid,
            rasterize_mode="classic",
            ray_gaussian_evaluation=False,
            compact_projection_cache=False,
        )


def test_resolve_covariance_epsilon_rejects_antialiased_exact_ray() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        resolve_covariance_epsilon(
            None,
            rasterize_mode="antialiased",
            ray_gaussian_evaluation=True,
            compact_projection_cache=False,
        )


def test_rasterize_mode_qualified_is_distinct_and_idempotent() -> None:
    classic = rasterize_mode_qualified("case", "classic")
    antialiased = rasterize_mode_qualified("case", "antialiased")

    assert classic == "case-rasterize-classic"
    assert antialiased == "case-rasterize-antialiased"
    assert classic != antialiased
    assert rasterize_mode_qualified(classic, "classic") == classic


def test_renderer_options_reject_unknown_rasterize_mode() -> None:
    with pytest.raises(ValueError, match="rasterize_mode"):
        resolve_covariance_epsilon(
            None,
            rasterize_mode="typo",
            ray_gaussian_evaluation=False,
            compact_projection_cache=False,
        )


def test_gsplat_config_id_is_mode_qualified_and_core_output_only() -> None:
    antialiased = gsplat_config_id(
        scene_id="synthetic-small",
        scene_mode="shared-scene",
        batch_size=8,
        width=128,
        height=128,
        rasterize_mode="antialiased",
    )
    classic = gsplat_config_id(
        scene_id="synthetic-small",
        scene_mode="shared-scene",
        batch_size=8,
        width=128,
        height=128,
        rasterize_mode="classic",
    )

    assert GSPLAT_RENDERED_OUTPUTS == ("rgb", "depth", "alpha")
    assert "semantic" not in antialiased
    assert antialiased.endswith("-rasterize-antialiased")
    assert classic.endswith("-rasterize-classic")
    assert classic != antialiased
