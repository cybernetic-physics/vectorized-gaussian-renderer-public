"""Native-free contract tests for the finite Isaac lifecycle smoke script."""

from __future__ import annotations

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "isaac_headless_smoke.py"
SUCCESS_MARKER = "ISAAC_HEADLESS_SMOKE_OK"


def _stub_module(name: str, attributes: dict[str, object] | None = None) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in (attributes or {}).items():
        setattr(module, attribute, value)
    return module


def _run_smoke(
    monkeypatch: pytest.MonkeyPatch,
    *,
    renderer: str | None,
    fail_at: str | None = None,
    events: list[tuple[Any, ...]] | None = None,
) -> list[tuple[Any, ...]]:
    if events is None:
        events = []

    class FakeSimulationApp:
        def __init__(self, config: dict[str, object]) -> None:
            events.append(("app_init", dict(config), tuple(sys.argv)))

        def close(self, *, wait_for_replicator: bool, exit_code: int) -> None:
            events.append(("close", wait_for_replicator, exit_code))

    class FakeScene:
        def add_default_ground_plane(self) -> None:
            events.append(("add_default_ground_plane",))

    class FakeWorld:
        def __init__(self) -> None:
            self.scene = FakeScene()
            self._step_count = 0
            events.append(("world_init",))

        def reset(self) -> None:
            events.append(("reset",))

        def step(self, *, render: bool) -> None:
            self._step_count += 1
            events.append(("step", self._step_count, render))
            if fail_at == "step" and self._step_count == 3:
                raise RuntimeError("synthetic step failure")

        def stop(self) -> None:
            events.append(("stop",))
            if fail_at == "stop":
                raise RuntimeError("synthetic stop failure")

    isaacsim = _stub_module("isaacsim", {"SimulationApp": FakeSimulationApp})
    isaacsim.__path__ = []  # type: ignore[attr-defined]
    core = _stub_module("isaacsim.core")
    core.__path__ = []  # type: ignore[attr-defined]
    core_api = _stub_module("isaacsim.core.api", {"World": FakeWorld})
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "isaacsim.core", core)
    monkeypatch.setitem(sys.modules, "isaacsim.core.api", core_api)
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.delenv("VGR_ISAAC_ACTIVE_GPU", raising=False)
    monkeypatch.delenv("VGR_ISAAC_REQUIRE_EXPLICIT_GPU", raising=False)
    monkeypatch.delenv("VGR_ISAAC_EXPERIENCE", raising=False)
    monkeypatch.delenv("VGR_ISAAC_REQUIRE_EXPERIENCE", raising=False)
    renderer_args = [] if renderer is None else ["--renderer", renderer]
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *renderer_args, "--/app/testForwardedArg=1"])

    real_print = builtins.print

    def tracked_print(*values: object, **kwargs: object) -> None:
        if values == (SUCCESS_MARKER,):
            events.append(("success_marker",))
        real_print(*values, **kwargs)

    monkeypatch.setattr(builtins, "print", tracked_print)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return events


@pytest.mark.parametrize(
    ("renderer", "expected_config"),
    [
        (
            None,
            {
                "headless": True,
                "physics_gpu": 0,
                "multi_gpu": False,
                "max_gpu_count": 1,
            },
        ),
        (
            "minimal",
            {
                "headless": True,
                "physics_gpu": 0,
                "multi_gpu": False,
                "max_gpu_count": 1,
                "renderer": "MinimalRendering",
            },
        ),
        (
            "default",
            {
                "headless": True,
                "physics_gpu": 0,
                "multi_gpu": False,
                "max_gpu_count": 1,
            },
        ),
    ],
)
def test_smoke_success_has_finite_ordered_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    renderer: str | None,
    expected_config: dict[str, object],
) -> None:
    events = _run_smoke(monkeypatch, renderer=renderer)

    assert events[0] == (
        "app_init",
        expected_config,
        (str(SCRIPT), "--/app/testForwardedArg=1"),
    )
    steps = [event for event in events if event[0] == "step"]
    assert steps == [("step", index, True) for index in range(1, 11)]
    assert events.index(("reset",)) < events.index(("step", 1, True))
    assert events.index(("step", 10, True)) < events.index(("stop",))
    assert events.index(("stop",)) < events.index(("success_marker",))
    assert events.index(("success_marker",)) < events.index(("close", False, 0))
    assert events[-1] == ("close", False, 0)


@pytest.mark.parametrize(
    ("fail_at", "message"),
    [
        ("step", "synthetic step failure"),
        ("stop", "synthetic stop failure"),
    ],
)
def test_smoke_failure_closes_nonzero_without_success_marker(
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str,
    message: str,
) -> None:
    events: list[tuple[Any, ...]] = []
    with pytest.raises(RuntimeError, match=message):
        _run_smoke(monkeypatch, renderer="minimal", fail_at=fail_at, events=events)

    assert ("success_marker",) not in events
    assert events[-1] == ("close", False, 1)
