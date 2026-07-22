import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.isaacsim_vectorized_example import (
    cache_delta,
    close_simulation_app,
    distribution,
    output_contract,
    parse_args,
    renderer_configuration_contract,
    resolve_covariance_epsilon,
)


def test_vectorized_example_forwards_unknown_kit_arguments(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isaacsim_vectorized_example.py",
            "--rasterize-mode",
            "antialiased",
            "--portable-root",
            "/tmp/portable",
        ],
    )

    args = parse_args()

    assert args.rasterize_mode == "antialiased"
    assert sys.argv == [
        "isaacsim_vectorized_example.py",
        "--portable-root",
        "/tmp/portable",
    ]


def test_cache_delta_reports_only_measured_phase_changes() -> None:
    assert cache_delta(
        {"enabled": True, "hits": 7, "misses": 3},
        {"enabled": True, "hits": 12, "misses": 5},
    ) == {"hits": 5, "misses": 2}


def test_distribution_reports_tail_latency() -> None:
    result = distribution([1.0, 2.0, 3.0, 10.0])

    assert result["mean"] == 4.0
    assert result["p50"] == 2.5
    assert result["p95"] == pytest.approx(8.95)
    assert result["max"] == 10.0


@pytest.mark.parametrize(("failed", "expected"), [(False, 0), (True, 1)])
def test_simulation_app_close_preserves_process_status(failed, expected) -> None:
    class FakeSimulationApp:
        exit_code = None
        wait_for_replicator = None

        def close(self, *, wait_for_replicator, exit_code):
            self.wait_for_replicator = wait_for_replicator
            self.exit_code = exit_code

    app = FakeSimulationApp()
    close_simulation_app(app, failed=failed)

    assert app.exit_code == expected
    assert app.wait_for_replicator is False


def test_antialias_resolves_nonzero_epsilon_on_compact_path() -> None:
    assert resolve_covariance_epsilon(
        None,
        rasterize_mode="antialiased",
        ray_gaussian_evaluation=False,
        compact_projection_cache=True,
    ) == 0.3
    assert resolve_covariance_epsilon(
        None,
        rasterize_mode="classic",
        ray_gaussian_evaluation=False,
        compact_projection_cache=True,
    ) == 0.0


def test_antialias_rejects_exact_ray_before_isaac_startup() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        resolve_covariance_epsilon(
            None,
            rasterize_mode="antialiased",
            ray_gaussian_evaluation=True,
            compact_projection_cache=False,
        )


def test_renderer_configuration_contract_fails_closed() -> None:
    backend = SimpleNamespace(
        rasterize_mode="antialiased",
        covariance_epsilon=0.3,
    )
    assert renderer_configuration_contract(
        backend,
        rasterize_mode="antialiased",
        covariance_epsilon=0.3,
    )["matches"]

    backend.covariance_epsilon = 0.0
    with pytest.raises(AssertionError, match="configuration mismatch"):
        renderer_configuration_contract(
            backend,
            rasterize_mode="antialiased",
            covariance_epsilon=0.3,
        )


def test_output_contract_checks_all_layouts_and_values_on_cpu_fixture() -> None:
    outputs = {
        "rgb": torch.zeros((1, 2, 3, 3), dtype=torch.float32),
        "depth": torch.full((1, 2, 3, 1), float("inf"), dtype=torch.float32),
        "alpha": torch.zeros((1, 2, 3, 1), dtype=torch.float32),
        "semantic_id": torch.full((1, 2, 3, 1), -1, dtype=torch.int64),
    }
    outputs["depth"][0, 0, 0, 0] = 2.0
    outputs["alpha"][0, 0, 0, 0] = 0.5
    outputs["semantic_id"][0, 0, 0, 0] = 17

    contract = output_contract(
        outputs,
        semantic_min_alpha=0.01,
        expected_shape=(1, 2, 3),
        require_cuda=False,
    )

    assert contract["valid"]
    assert contract["shapes_match"]
    assert contract["dtypes_match"]
    assert contract["contiguous"]
    assert contract["foreground_pixel_count"] == 1


def test_output_contract_rejects_wrong_channel_layout() -> None:
    outputs = {
        "rgb": torch.zeros((1, 2, 3, 2), dtype=torch.float32),
        "depth": torch.ones((1, 2, 3, 1), dtype=torch.float32),
        "alpha": torch.ones((1, 2, 3, 1), dtype=torch.float32),
        "semantic_id": torch.zeros((1, 2, 3, 1), dtype=torch.int64),
    }

    contract = output_contract(
        outputs,
        semantic_min_alpha=0.01,
        expected_shape=(1, 2, 3),
        require_cuda=False,
    )

    assert not contract["valid"]
    assert not contract["shapes_match"]
