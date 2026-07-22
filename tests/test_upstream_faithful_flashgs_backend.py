from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from benchmarks.verify_upstream_faithful_flashgs_parity import (
    service_viewmat_for_upstream_camera,
    upstream_camera,
    upstream_covariances,
)
from benchmarks.run_upstream_faithful_flashgs_control import (
    summarize_counter_samples,
)
from isaacsim_gaussian_renderer import (
    RendererService,
    UpstreamFaithfulFlashGSBackend,
)
from isaacsim_gaussian_renderer.upstream_faithful_flashgs_sources import (
    PINNED_GENERATED_PREPROCESS_SHA256,
    PINNED_PREPROCESS_DIFF,
    PINNED_SOURCE_SHA256,
    PRESERVED_PREPROCESS_FUNCTIONS,
    transform_preprocess_source,
    verify_pinned_source_files,
)
from isaacsim_gaussian_renderer.upstream_faithful_flashgs_native_loader import (
    UPSTREAM_FAITHFUL_CUDA_FLAGS,
    _require_live_cuda_architecture,
)


class FakeUpstreamFaithfulNative:
    def __init__(self) -> None:
        self.precompute_calls = 0
        self.render_calls = 0
        self.last_workspace: list[torch.Tensor] | None = None
        self.last_outputs: list[torch.Tensor] | None = None
        self.__vgr_build_contract__ = {
            "lane": "upstream-faithful-flashgs-rgb"
        }

    def get_sort_buffer_size(self, capacity: int) -> int:
        assert capacity == 32
        return 64

    def precompute_covariances(
        self,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        covariances: torch.Tensor,
    ) -> None:
        self.precompute_calls += 1
        assert scales.shape == (3, 3)
        assert rotations.shape == (3, 4)
        covariances.fill_(0.125)

    def render_batch(
        self,
        scene: list[torch.Tensor],
        cameras: list[torch.Tensor],
        outputs: list[torch.Tensor],
        workspace: list[torch.Tensor],
        height: int,
        width: int,
        near_plane: float,
        far_plane: float,
        registered_scene_id: int,
    ) -> None:
        self.render_calls += 1
        self.last_workspace = workspace
        self.last_outputs = outputs
        assert len(scene) == 4
        assert len(cameras) == 4
        assert len(outputs) == 1
        assert len(workspace) == 12
        assert (height, width) == (16, 32)
        assert near_plane == 0.01
        assert far_plane == 100.0
        assert registered_scene_id == 404
        outputs[0].fill_(64.0 / 255.0)
        batch = cameras[0].shape[0]
        workspace[11][:batch].copy_(
            torch.tensor([[11, 0, 0]] * batch, dtype=torch.int32)
        )


def scene_tensors() -> dict[str, torch.Tensor]:
    return {
        "means": torch.zeros((3, 3), dtype=torch.float32),
        "scales": torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.2, 0.1]],
            dtype=torch.float32,
        ),
        "rotations": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9238795, 0.3826834, 0.0, 0.0],
                [0.9659258, 0.0, 0.2588190, 0.0],
            ],
            dtype=torch.float32,
        ),
        "opacities": torch.full((3,), 0.5, dtype=torch.float32),
        "features": torch.tensor(
            [[0.2, 0.3, 0.4], [0.8, 0.1, 0.2], [0.4, 0.7, 0.2]],
            dtype=torch.float32,
        ),
        "semantic_ids": torch.tensor([2, 4, 6], dtype=torch.int64),
    }


def make_service(
    native: Any,
    *,
    outputs: tuple[str, ...] = ("rgb",),
) -> tuple[UpstreamFaithfulFlashGSBackend, RendererService]:
    backend = UpstreamFaithfulFlashGSBackend(
        max_intersections=32,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=16,
        width=32,
        outputs=outputs,
        max_views=2,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(404, **scene_tensors())
    return backend, service


def camera_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [[24.0, 0.0, 16.0], [0.0, 24.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).repeat(2, 1, 1)
    return (
        torch.eye(4, dtype=torch.float32).repeat(2, 1, 1).contiguous(),
        intrinsics.contiguous(),
        torch.tensor([404, 404], dtype=torch.int64),
    )


def test_upstream_faithful_backend_is_separate_rgb_only_service_lane() -> None:
    native = FakeUpstreamFaithfulNative()
    backend, service = make_service(native)

    outputs = service.render(*camera_batch())

    assert native.precompute_calls == 1
    assert native.render_calls == 1
    assert tuple(outputs) == ("rgb",)
    assert torch.all(outputs["rgb"] == 64.0 / 255.0)
    assert backend.check_capacity() == {
        "generated_intersections": 22,
        "intersection_overflow": 0,
        "camera_contract_errors": 0,
    }
    assert backend.execution_stats == {
        "lane": "upstream-faithful-flashgs-rgb",
        "upstream_commit": "cdfc4e4002318423eda356eed02df8e01fa32cb6",
        "render_requests": 1,
        "native_camera_executions": 2,
        "true_native_batch": False,
        "fixed_capacity_sort": True,
        "max_intersections_per_camera": 32,
        "output_contract": "rgb-only",
        "native_output": "uint8-rgb",
        "service_output": "float32-rgb",
        "conversion_in_measured_render": True,
        "full_sensor_output": False,
        "equation_contract": "public-flashgs-native-rgb-degree-zero-specialized",
        "byte_exact_public_preprocess": False,
        "byte_exact_public_sort_and_render": True,
        "performance_identical_to_unmodified_public_binary": False,
    }


def test_upstream_faithful_workspace_has_one_safe_sentinel_tile() -> None:
    native = FakeUpstreamFaithfulNative()
    backend, _service = make_service(native)

    assert backend.workspace["ranges"].shape == (3, 2)
    assert backend.workspace["rgb_u8"].shape == (16, 32, 3)
    assert backend.workspace["camera_state"].numel() == 39
    assert native.last_workspace is None


def test_upstream_faithful_backend_rejects_full_sensor_contract() -> None:
    native = FakeUpstreamFaithfulNative()
    _backend, service = make_service(
        native,
        outputs=("rgb", "depth", "alpha", "semantic_id"),
    )
    with pytest.raises(ValueError, match="supports RGB only"):
        service.render(*camera_batch())


def test_upstream_faithful_backend_rejects_non_degree_zero_scene() -> None:
    native = FakeUpstreamFaithfulNative()
    backend = UpstreamFaithfulFlashGSBackend(
        max_intersections=32,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=16,
        width=32,
        outputs=("rgb",),
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    scene = scene_tensors()
    scene["features"] = torch.zeros((3, 48), dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly three canonical"):
        service.load_scene(404, **scene)


def test_source_guard_rejects_any_unpinned_preprocess_text() -> None:
    with pytest.raises(RuntimeError, match="unpinned"):
        transform_preprocess_source("// source drift\n")


def test_source_tree_guard_rejects_pinned_file_drift(tmp_path: Path) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text("# changed build policy\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source drift"):
        verify_pinned_source_files(tmp_path)


def test_pinned_source_manifest_covers_all_algorithm_translation_units() -> None:
    assert set(PINNED_SOURCE_SHA256) == {
        "setup.py",
        "csrc/ops.h",
        "csrc/pybind.cpp",
        "csrc/cuda_rasterizer/preprocess.cu",
        "csrc/cuda_rasterizer/render.cu",
        "csrc/cuda_rasterizer/sort.cu",
    }
    assert all(len(value) == 64 for value in PINNED_SOURCE_SHA256.values())
    assert "-O1" in UPSTREAM_FAITHFUL_CUDA_FLAGS
    assert "--use_fast_math" not in UPSTREAM_FAITHFUL_CUDA_FLAGS


def test_upstream_faithful_build_rejects_wrong_cuda_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 6))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9")

    with pytest.raises(RuntimeError, match="match the live GPU exactly"):
        _require_live_cuda_architecture()


def test_upstream_faithful_build_pins_live_cuda_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 6))
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

    assert _require_live_cuda_architecture() == "8.6"


def test_checked_out_upstream_transform_preserves_equation_functions() -> None:
    project_root = Path(__file__).resolve().parents[1]
    upstream = project_root.parent / "FlashGS"
    if not upstream.is_dir():
        pytest.skip("Pinned sibling FlashGS checkout is unavailable.")
    source_audit = verify_pinned_source_files(upstream)
    source = (upstream / "csrc/cuda_rasterizer/preprocess.cu").read_text(
        encoding="utf-8"
    )
    result = transform_preprocess_source(source)

    assert source_audit["pass"] is True
    assert result.audit["pass"] is True
    assert set(result.audit["preserved_function_sha256"]) == set(
        PRESERVED_PREPROCESS_FUNCTIONS
    )
    assert result.audit["exact_upstream_translation_units"] == [
        "csrc/cuda_rasterizer/sort.cu",
        "csrc/cuda_rasterizer/render.cu",
    ]
    assert result.audit["generated_sha256"] == PINNED_GENERATED_PREPROCESS_SHA256
    assert result.audit["diff"] == PINNED_PREPROCESS_DIFF
    assert "computeColorFromSH" in result.source
    assert "computeColorFromDegreeZero" in result.source
    assert "output_offset < capacity" in result.source
    assert "camera_state->viewmatrix" in result.source


def test_independent_cpu_reference_camera_and_covariance_contract() -> None:
    position, _rotation = upstream_camera()
    viewmat = service_viewmat_for_upstream_camera()
    camera_space_center = viewmat[:3, :3] @ position + viewmat[:3, 3]
    assert torch.allclose(camera_space_center, torch.zeros(3), atol=1.0e-7)
    covariance = upstream_covariances(
        torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        covariance,
        torch.tensor([[0.04, 0.0, 0.0, 0.09, 0.0, 0.16]]),
        atol=1.0e-7,
    )


def test_counter_audit_cannot_hide_an_earlier_overflow() -> None:
    audit = summarize_counter_samples(
        [
            {
                "generated_intersections": 19,
                "intersection_overflow": 3,
                "camera_contract_errors": 0,
            },
            {
                "generated_intersections": 11,
                "intersection_overflow": 0,
                "camera_contract_errors": 0,
            },
        ]
    )

    assert audit["pass"] is False
    assert audit["sample_count"] == 2
    assert audit["maxima"]["intersection_overflow"] == 3
    assert audit["totals"]["generated_intersections"] == 30
