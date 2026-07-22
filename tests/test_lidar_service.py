from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any

import pytest
import torch

from isaacsim_gaussian_renderer import GaussianLidarService, LidarRenderRequest
from isaacsim_gaussian_renderer.cuda_lidar_backend import CudaLidarBackend
from isaacsim_gaussian_renderer.lidar_native_loader import is_lidar_native_loaded


def scene_tensors(count: int = 2) -> dict[str, torch.Tensor]:
    return {
        "means": torch.tensor([[5.0, 0.0, 0.0]] * count, dtype=torch.float32),
        "scales": torch.tensor([[0.01, 1.0, 1.0]] * count, dtype=torch.float32),
        "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count, dtype=torch.float32),
        "opacities": torch.ones((count,), dtype=torch.float32),
        "semantic_ids": torch.arange(count, dtype=torch.int64),
        "reflectivity": torch.full((count,), 0.4, dtype=torch.float32),
    }


class FakeLidarBackend:
    def __init__(self) -> None:
        self.render_calls = 0
        self.registered: list[int] = []
        self.revisions: list[int] = []
        self.shutdown_called = False

    def initialize(self, stage: Any, device: torch.device) -> None:
        self.device = device

    def allocate_workspace(self, *, max_sensors: int, max_rays: int, device: torch.device):
        return {
            "identity_scene_to_world": torch.eye(4, device=device).repeat(max_sensors, 1, 1).contiguous()
        }

    def register_scene(self, scene_id: int, scene: Any) -> None:
        self.registered.append(scene_id)

    def revise_scene(self, scene_id: int, scene: Any) -> None:
        self.revisions.append(scene.revision)

    def render_lidar(self, request: LidarRenderRequest) -> None:
        self.render_calls += 1
        for name, tensor in request.outputs.items():
            if name == "range_m":
                tensor.fill_(4.0 + self.render_calls)
            elif name == "position_world_m":
                tensor.fill_(2.0)
            elif name == "intensity":
                tensor.fill_(0.25)
            elif name == "semantic_id":
                tensor.fill_(7)
            elif name == "valid":
                tensor.fill_(True)
            elif name == "time_offset_ns":
                tensor.copy_(request.time_offsets_ns.to(torch.int64)[None, :, None].expand_as(tensor))
            elif name == "return_count":
                tensor.fill_(request.returns)

    def synchronize(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_called = True


def initialized_service(backend: Any | None = None) -> tuple[GaussianLidarService, Any]:
    backend = backend or FakeLidarBackend()
    service = GaussianLidarService(
        backend,
        max_sensors=4,
        max_rays=8,
        returns=2,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=object(), device="cpu")
    service.load_scene(17, **scene_tensors())
    return service, backend


def request_tensors(batch: int = 2, rays: int = 3):
    directions = torch.tensor([[1.0, 0.0, 0.0]] * rays, dtype=torch.float32)
    times = torch.arange(rays, dtype=torch.int32) * 100
    transforms = torch.eye(4).repeat(batch, 1, 1).contiguous()
    ids = torch.full((batch,), 17, dtype=torch.int64)
    return directions, times, transforms, ids


def test_lidar_service_rewrites_every_call_and_has_no_cadence() -> None:
    service, backend = initialized_service()
    assert service.mode == "gaussian_surface"
    inputs = request_tensors()
    outputs = service.render_lidar(*inputs)
    first_range = outputs["range_m"].clone()
    service.render_lidar(*inputs, outputs=outputs)

    assert backend.render_calls == 2
    assert torch.all(first_range == 5.0)
    assert torch.all(outputs["range_m"] == 6.0)
    assert outputs["range_m"].shape == (2, 3, 2)
    assert outputs["position_world_m"].shape == (2, 3, 2, 3)
    assert outputs["semantic_id"].dtype == torch.int64
    assert outputs["valid"].dtype == torch.bool
    assert outputs["return_count"].dtype == torch.int32
    torch.testing.assert_close(outputs["time_offset_ns"][0, :, 0], inputs[1].to(torch.int64))


def test_lidar_contract_rejects_non_normalized_cpu_rays_and_bad_outputs() -> None:
    service, _ = initialized_service()
    directions, times, transforms, ids = request_tensors()
    directions[0, 0] = 2.0
    with pytest.raises(ValueError, match="normalized"):
        service.render_lidar(directions, times, transforms, ids)

    directions[0, 0] = 1.0
    outputs = service.render_lidar(directions, times, transforms, ids)
    outputs.pop("intensity")
    with pytest.raises(ValueError, match="Missing LiDAR output"):
        service.render_lidar(directions, times, transforms, ids, outputs=outputs)
    with pytest.raises(ValueError, match="Missing LiDAR output"):
        service.render_lidar(directions, times, transforms, ids, outputs={})


def test_scene_revision_is_explicit_and_strictly_increasing() -> None:
    service, backend = initialized_service()
    service.registry.scene(17).means[0, 0] = 6.0
    service.revise_scene(17, revision=1)
    assert backend.revisions == [1]
    with pytest.raises(ValueError, match="must increase"):
        service.revise_scene(17, revision=1)


def test_invalid_scene_revision_is_rejected_before_registry_commit() -> None:
    service, backend = initialized_service()
    scene = service.registry.scene(17)
    scene.scales[0, 0] = float("nan")
    with pytest.raises(ValueError, match="scales must be finite"):
        service.revise_scene(17, revision=1)
    assert service.registry.scene(17).revision == 0
    assert backend.revisions == []


def test_negative_scene_semantic_is_reserved_for_no_hit() -> None:
    backend = FakeLidarBackend()
    service = GaussianLidarService(
        backend,
        max_sensors=1,
        max_rays=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(None, "cpu")
    scene = scene_tensors(1)
    scene["semantic_ids"][0] = -1
    with pytest.raises(ValueError, match="reserved for no-hit"):
        service.load_scene(1, **scene)


def test_camera_import_and_service_do_not_load_lidar_native_extension() -> None:
    from isaacsim_gaussian_renderer import DeterministicFakeBackend, RendererService

    assert not is_lidar_native_loaded()
    camera = RendererService(
        DeterministicFakeBackend(),
        height=1,
        width=1,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    camera.initialize(stage=None, device="cpu")
    camera.load_scene(3, features=torch.ones((2, 3)), **{
        key: value for key, value in scene_tensors().items()
        if key in {"means", "scales", "rotations", "opacities", "semantic_ids"}
    })
    assert not is_lidar_native_loaded()
    camera.shutdown()


def test_fresh_camera_process_does_not_import_or_compile_lidar_native(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import pathlib
        import sys
        import torch
        from isaacsim_gaussian_renderer import DeterministicFakeBackend, RendererService

        camera = RendererService(
            DeterministicFakeBackend(),
            height=1,
            width=1,
            max_views=1,
            allow_cpu_for_tests=True,
        )
        camera.initialize(None, torch.device("cpu"))
        if "isaacsim_gaussian_renderer.lidar_native_loader" in sys.modules:
            raise AssertionError("camera-only process imported LiDAR native loader")
        build_root = pathlib.Path(__import__("os").environ["VGR_LIDAR_NATIVE_BUILD_ROOT"])
        if build_root.exists() and any(build_root.rglob("*lidar*")):
            raise AssertionError("camera-only process created a LiDAR build artifact")
        camera.shutdown()
        """
    )
    environment = os.environ.copy()
    environment["VGR_LIDAR_NATIVE_BUILD_ROOT"] = str(tmp_path / "lidar-extensions")
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


class FakeLidarNative:
    def __init__(self) -> None:
        self.build_calls = 0
        self.render_calls = 0

    def sort_temp_bytes(self, num_items: int) -> int:
        return max(1, num_items * 2)

    def build_scene_lbvh(self, scene, build, packet_size, leaf_count, leaf_capacity, support, planarity):
        self.build_calls += 1
        build[3].copy_(torch.arange(scene[0].shape[0], dtype=torch.int32))
        build[6].zero_()

    def pack_scene_descriptor(self, descriptors, slot, scene_id, scene, leaf_count, leaf_capacity):
        descriptors[slot, 0] = scene_id
        descriptors[slot, 1] = scene[0].shape[0]

    def render_lidar(self, descriptors, scene_count, inputs, outputs, counters, returns, *config):
        self.render_calls += 1
        outputs[0].fill_(3.0)
        outputs[1].fill_(1.0)
        outputs[2].fill_(0.5)
        outputs[3].fill_(2)
        outputs[4].fill_(True)
        outputs[5].copy_(inputs[1].to(torch.int64)[None, :, None].expand_as(outputs[5]))
        outputs[6].fill_(returns)
        counters[0] += 1
        counters[1] += 1
        counters[2] += inputs[0].shape[0] * inputs[2].shape[0]


def test_cuda_backend_fake_native_keeps_one_bvh_independent_of_batch() -> None:
    native = FakeLidarNative()
    backend = CudaLidarBackend(
        native_module=native,
        allow_cpu_for_tests=True,
        max_scenes=2,
    )
    service, _ = initialized_service(backend)
    bvh_bytes = backend.bvh_bytes
    directions, times, transforms, ids = request_tensors(batch=4, rays=8)
    service.render_lidar(directions, times, transforms, ids)
    assert native.build_calls == 1
    assert native.render_calls == 1
    assert backend.bvh_bytes == bvh_bytes
    assert backend.workspace["scene_descriptors"].shape == (2, 14)
    assert backend.read_counters()["rays_traced"] == 32
