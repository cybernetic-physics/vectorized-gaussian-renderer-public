from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaacsim_gaussian_renderer import camera_source


def test_fabric_module_loader_initializes_warp_before_use(monkeypatch) -> None:
    calls: list[str] = []
    usdrt = SimpleNamespace()
    warp = SimpleNamespace(init=lambda: calls.append("warp.init"))
    kernels = SimpleNamespace(fabric_world_to_opencv_viewmats=object())
    modules = {
        "usdrt": usdrt,
        "warp": warp,
        "isaacsim_gaussian_renderer.fabric_camera_kernels": kernels,
    }
    monkeypatch.setattr(
        camera_source.importlib,
        "import_module",
        lambda name: modules[name],
    )

    assert camera_source._load_fabric_modules() == (usdrt, warp, kernels)
    assert calls == ["warp.init"]


def test_fabric_module_loader_explains_missing_isaac_runtime(monkeypatch) -> None:
    def missing(_name: str):
        raise ImportError("missing")

    monkeypatch.setattr(camera_source.importlib, "import_module", missing)

    with pytest.raises(RuntimeError, match="must run inside Isaac Sim"):
        camera_source._load_fabric_modules()
