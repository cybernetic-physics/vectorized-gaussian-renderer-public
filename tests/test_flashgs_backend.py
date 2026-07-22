from __future__ import annotations

from typing import Any

import pytest
import torch

from benchmarks.run_flashgs_matched import current_device_counter_snapshots
from isaacsim_gaussian_renderer import FlashGSBackend, RendererService


class FakeFlashGSNative:
    def __init__(self) -> None:
        self.precompute_calls = 0
        self.render_calls = 0
        self.last_full_sensor_output: bool | None = None
        self.last_scene: list[torch.Tensor] | None = None
        self.last_cameras: list[torch.Tensor] | None = None
        self.last_workspace: list[torch.Tensor] | None = None

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
        support_sigma: float,
        covariance_epsilon: float,
        semantic_min_alpha: float,
        full_sensor_output: bool,
        registered_scene_id: int,
    ) -> None:
        self.render_calls += 1
        self.last_full_sensor_output = full_sensor_output
        self.last_scene = scene
        self.last_cameras = cameras
        self.last_workspace = workspace
        assert (height, width) == (4, 5)
        assert near_plane == 0.01
        assert far_plane == 100.0
        assert support_sigma == 3.0
        assert covariance_epsilon == 0.0
        assert semantic_min_alpha == 0.01
        assert registered_scene_id == 404
        outputs[0].fill_(0.25)
        if full_sensor_output:
            outputs[1].fill_(2.0)
            outputs[2].fill_(0.75)
            outputs[3].fill_(9)
        batch = cameras[0].shape[0]
        workspace[9][:batch].copy_(
            torch.tensor([[3, 5, 0]] * batch, dtype=torch.int64)
        )


def scene_tensors() -> dict[str, torch.Tensor]:
    return {
        "means": torch.zeros((3, 3), dtype=torch.float32),
        "scales": torch.full((3, 3), 0.1, dtype=torch.float32),
        "rotations": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 3,
            dtype=torch.float32,
        ),
        "opacities": torch.full((3,), 0.5, dtype=torch.float32),
        "features": torch.full((3, 3), 0.2, dtype=torch.float32),
        "semantic_ids": torch.tensor([2, 4, 6], dtype=torch.int64),
    }


def make_service(
    native: Any,
    *,
    outputs: tuple[str, ...] = ("rgb", "depth", "alpha", "semantic_id"),
) -> tuple[FlashGSBackend, RendererService]:
    backend = FlashGSBackend(
        max_intersections=32,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        outputs=outputs,
        max_views=2,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(404, **scene_tensors())
    return backend, service


def camera_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.eye(4).repeat(2, 1, 1).contiguous(),
        torch.eye(3).repeat(2, 1, 1).contiguous(),
        torch.tensor([404, 404], dtype=torch.int64),
    )


def test_flashgs_backend_full_sensor_contract_and_counters() -> None:
    native = FakeFlashGSNative()
    backend, service = make_service(native)

    outputs = service.render(*camera_batch())

    assert native.precompute_calls == 1
    assert native.render_calls == 1
    assert native.last_full_sensor_output is True
    assert torch.all(outputs["rgb"] == 0.25)
    assert torch.all(outputs["depth"] == 2.0)
    assert torch.all(outputs["alpha"] == 0.75)
    assert torch.all(outputs["semantic_id"] == 9)
    assert backend.check_capacity() == {
        "visible_gaussians": 6,
        "generated_intersections": 10,
        "intersection_overflow": 0,
    }
    assert backend.execution_stats["native_camera_executions"] == 2
    assert backend.execution_stats["true_native_batch"] is False
    assert native.last_scene is not None
    assert torch.all(native.last_scene[1] == 0.125)
    assert native.last_workspace is not None
    assert len(native.last_workspace) == 10


def test_matched_flashgs_capacity_is_maximum_per_camera_not_batch_sum() -> None:
    native = FakeFlashGSNative()
    backend, service = make_service(native)
    service.render(*camera_batch())
    backend.workspace["counters"][:2].copy_(
        torch.tensor([[3, 5, 0], [7, 11, 0]], dtype=torch.int64)
    )

    batch_totals, capacity_demand = current_device_counter_snapshots(backend)

    assert batch_totals.tolist() == [10, 16, 0]
    assert capacity_demand.tolist() == [7, 11, 0]


def test_flashgs_backend_rgb_only_uses_one_element_output_sinks() -> None:
    native = FakeFlashGSNative()
    backend, service = make_service(native, outputs=("rgb",))

    outputs = service.render(*camera_batch())

    assert tuple(outputs) == ("rgb",)
    assert native.last_full_sensor_output is False
    assert torch.all(outputs["rgb"] == 0.25)
    assert backend.workspace["dummy_depth"].numel() == 1
    assert backend.workspace["dummy_alpha"].numel() == 1
    assert backend.workspace["dummy_semantic"].numel() == 1


def test_flashgs_backend_rejects_second_scene() -> None:
    native = FakeFlashGSNative()
    _backend, service = make_service(native)
    with pytest.raises(ValueError, match="one shared resident scene"):
        service.load_scene(405, **scene_tensors())


def test_flashgs_backend_rejects_partial_output_contract() -> None:
    native = FakeFlashGSNative()
    _backend, service = make_service(native, outputs=("rgb", "alpha"))
    with pytest.raises(ValueError, match="supports exactly"):
        service.render(*camera_batch())
