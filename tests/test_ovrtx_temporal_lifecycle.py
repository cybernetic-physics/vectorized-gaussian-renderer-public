from __future__ import annotations

import ast
import gc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "ovrtx_temporal_fidelity.py"


def _stub_module(name: str, attributes: dict[str, object] | None = None) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in (attributes or {}).items():
        setattr(module, attribute, value)
    return module


def _load_temporal_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    benchmarks = _stub_module("benchmarks")
    benchmarks.__path__ = []  # type: ignore[attr-defined]
    benchmark_names = {
        "PLY_SCENES": {},
        "PROJECTION_ACTIVATION_SCENE": "projection-activation",
        "PROJECTION_MODES": ("perspective", "tangential"),
        "SORTING_MODES": ("zDepth",),
        "SYNTHETIC_SCENES": {},
        **{
            name: object()
            for name in (
                "allocate_contract_outputs",
                "build_stage",
                "copy_mapped_outputs_to_contract",
                "git_commit",
                "implementation_commit",
                "load_scene_and_cameras",
                "product_paths",
                "semantic_lut_from_outputs",
                "splat_paths",
                "splat_prim_semantic_ids",
                "require_token_attribute_readback",
                "upload_scene",
            )
        },
    }
    run_ovrtx = _stub_module("benchmarks.run_ovrtx", benchmark_names)

    renderer_package = _stub_module(
        "isaacsim_gaussian_renderer",
        {"CustomCudaBackend": object, "RendererService": object},
    )
    renderer_package.__path__ = []  # type: ignore[attr-defined]
    manifest = _stub_module(
        "isaacsim_gaussian_renderer.benchmark_manifest",
        {
            name: object()
            for name in (
                "compact_semantic_ids",
                "sha256_json",
                "spatial_semantic_ids",
                "spatial_semantic_manifest",
                "scene_tensors_sha256",
            )
        },
    )
    fidelity = _stub_module(
        "isaacsim_gaussian_renderer.fidelity",
        {
            name: object()
            for name in (
                "RenderOutput",
                "bundle_from_tensors",
                "compare_render_outputs",
                "write_camera_bundle",
            )
        },
    )

    for name, module in {
        "ovrtx": _stub_module("ovrtx"),
        "benchmarks": benchmarks,
        "benchmarks.run_ovrtx": run_ovrtx,
        "isaacsim_gaussian_renderer": renderer_package,
        "isaacsim_gaussian_renderer.benchmark_manifest": manifest,
        "isaacsim_gaussian_renderer.fidelity": fidelity,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_ovrtx_temporal_fidelity_lifecycle_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _TrackedObject:
    def __init__(self, label: str, events: list[str]) -> None:
        self._label = label
        self._events = events

    def __del__(self) -> None:
        self._events.append(f"{self._label}_released")


def test_success_markers_follow_upload_and_renderer_release(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    temporal = _load_temporal_module(monkeypatch)
    events: list[str] = []

    def fake_capture(lifecycle: object) -> None:
        renderer = _TrackedObject("renderer", events)
        upload = _TrackedObject("upload", events)
        lifecycle.own_renderer(renderer)
        lifecycle.retain_upload(upload)
        events.append("capture_complete")

    monkeypatch.setattr(temporal, "_run_temporal_capture", fake_capture)

    temporal.main()

    assert events == ["capture_complete", "upload_released", "renderer_released"]
    marker_lines = [line for line in capsys.readouterr().out.splitlines() if "OVRTX_TEMPORAL_" in line]
    assert marker_lines[0].startswith(
        "OVRTX_TEMPORAL_CLEANUP_DETAILS python_renderer_released=true keep_system_alive=true gc_collected="
    )
    assert marker_lines[1:] == ["OVRTX_TEMPORAL_CLEANUP_OK", "OVRTX_TEMPORAL_CAPTURE_OK"]


def test_capture_exception_releases_objects_without_success_markers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    temporal = _load_temporal_module(monkeypatch)
    events: list[str] = []

    def failing_capture(lifecycle: object) -> None:
        renderer = _TrackedObject("renderer", events)
        upload = _TrackedObject("upload", events)
        lifecycle.own_renderer(renderer)
        lifecycle.retain_upload(upload)
        raise ValueError("capture failed")

    monkeypatch.setattr(temporal, "_run_temporal_capture", failing_capture)

    with pytest.raises(ValueError, match="capture failed"):
        temporal.main()

    # The active exception traceback retains both frame locals until the
    # primary error is observed, so their final destructor order is not part
    # of the failure-path contract.
    assert sorted(events) == ["renderer_released", "upload_released"]
    output = capsys.readouterr().out
    assert "OVRTX_TEMPORAL_CLEANUP_OK" not in output
    assert "OVRTX_TEMPORAL_CAPTURE_OK" not in output


def test_retained_renderer_fails_before_success_markers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    temporal = _load_temporal_module(monkeypatch)
    retained: list[object] = []

    def retaining_capture(lifecycle: object) -> None:
        renderer = _TrackedObject("renderer", [])
        lifecycle.own_renderer(renderer)
        retained.append(renderer)

    monkeypatch.setattr(temporal, "_run_temporal_capture", retaining_capture)

    with pytest.raises(RuntimeError, match="without releasing its Python renderer handle"):
        temporal.main()

    output = capsys.readouterr().out
    assert "OVRTX_TEMPORAL_CLEANUP_OK" not in output
    assert "OVRTX_TEMPORAL_CAPTURE_OK" not in output
    retained.clear()
    gc.collect()


def test_temporal_renderer_retains_native_system_for_bounded_process_exit() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    renderer_configs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "RendererConfig"
    ]

    assert len(renderer_configs) == 1
    keep_alive = [keyword.value for keyword in renderer_configs[0].keywords if keyword.arg == "keep_system_alive"]
    assert len(keep_alive) == 1
    assert isinstance(keep_alive[0], ast.Constant)
    assert keep_alive[0].value is True
