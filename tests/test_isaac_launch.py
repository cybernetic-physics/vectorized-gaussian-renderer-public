from __future__ import annotations

import pytest

from scripts._isaac_launch import (
    close_simulation_app,
    isaac_app_config,
    selected_experience,
    selected_vulkan_gpu,
)


def test_launch_config_is_single_gpu_and_uses_preflight_index(monkeypatch) -> None:
    monkeypatch.setenv("VGR_ISAAC_ACTIVE_GPU", "3")
    monkeypatch.setenv("VGR_ISAAC_REQUIRE_EXPLICIT_GPU", "1")

    assert isaac_app_config(
        renderer="MinimalRendering",
        disable_viewport_updates=True,
    ) == {
        "headless": True,
        "physics_gpu": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "active_gpu": 3,
        "renderer": "MinimalRendering",
        "disable_viewport_updates": True,
    }


def test_required_gpu_index_must_be_explicit(monkeypatch) -> None:
    monkeypatch.delenv("VGR_ISAAC_ACTIVE_GPU", raising=False)
    monkeypatch.setenv("VGR_ISAAC_REQUIRE_EXPLICIT_GPU", "true")

    with pytest.raises(RuntimeError, match="requires VGR_ISAAC_ACTIVE_GPU"):
        selected_vulkan_gpu()


def test_launch_config_uses_explicit_test_experience(monkeypatch, tmp_path) -> None:
    experience = tmp_path / "test.kit"
    experience.touch()
    monkeypatch.setenv("VGR_ISAAC_EXPERIENCE", str(experience))
    monkeypatch.setenv("VGR_ISAAC_REQUIRE_EXPERIENCE", "1")

    assert isaac_app_config()["experience"] == str(experience.absolute())


def test_required_experience_must_exist(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "missing.kit"
    monkeypatch.setenv("VGR_ISAAC_EXPERIENCE", str(missing))

    with pytest.raises(RuntimeError, match="does not name a file"):
        selected_experience()


@pytest.mark.parametrize("value", ["-1", "gpu0", "1.5"])
def test_gpu_index_rejects_invalid_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("VGR_ISAAC_ACTIVE_GPU", value)

    with pytest.raises(RuntimeError, match="non-negative integer"):
        selected_vulkan_gpu()


def test_close_preserves_failure_and_skips_unused_replicator_wait() -> None:
    calls = []

    class App:
        def close(self, **kwargs) -> None:
            calls.append(kwargs)

    close_simulation_app(App(), failed=True)

    assert calls == [{"wait_for_replicator": False, "exit_code": 1}]
