import pytest
import torch

from isaacsim_gaussian_renderer import DeterministicFakeBackend, RendererService


def make_scene(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "means": torch.zeros((2, 3), device=device),
        "scales": torch.ones((2, 3), device=device),
        "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device),
        "opacities": torch.ones((2,), device=device),
        "features": torch.ones((2, 3), device=device),
        "semantic_ids": torch.tensor([3, 3], dtype=torch.int64, device=device),
    }


def initialized_service() -> tuple[RendererService, DeterministicFakeBackend, torch.device]:
    device = torch.device("cpu")
    backend = DeterministicFakeBackend()
    service = RendererService(backend, height=3, width=4, max_views=4, allow_cpu_for_tests=True)
    service.initialize(stage=object(), device=device)
    service.load_scene(3, **make_scene(device))
    return service, backend, device


def test_renderer_service_lifecycle_and_owned_outputs() -> None:
    service, backend, device = initialized_service()
    camera_transforms = torch.eye(4, device=device).repeat(2, 1, 1).contiguous()
    intrinsics = torch.eye(3, device=device).repeat(2, 1, 1).contiguous()
    scene_ids = torch.tensor([3, 3], dtype=torch.int64, device=device)

    outputs = service.render(camera_transforms, intrinsics, scene_ids)
    service.synchronize()

    assert backend.render_calls == 1
    assert outputs["rgb"].shape == (2, 3, 4, 3)
    assert outputs["depth"].device == device
    torch.testing.assert_close(outputs["depth"], torch.full((2, 3, 4, 1), 4.0))
    assert torch.all(outputs["semantic_id"] == 3)

    service.shutdown()
    assert backend.shutdown_called


def test_prepare_outputs_reuses_full_logical_storage() -> None:
    service, _, _ = initialized_service()

    first = service.prepare_outputs(4)
    second = service.prepare_outputs(4)

    assert second is first
    assert first["rgb"].shape == (4, 3, 4, 3)
    with pytest.raises(ValueError, match="must be positive"):
        service.prepare_outputs(0)
    with pytest.raises(ValueError, match="max_views"):
        service.prepare_outputs(5)


def test_caller_owned_active_subset_defines_inactive_defaults() -> None:
    service, backend, device = initialized_service()
    camera_transforms = torch.eye(4, device=device).repeat(2, 1, 1).contiguous()
    intrinsics = torch.eye(3, device=device).repeat(2, 1, 1).contiguous()
    scene_ids = torch.tensor([3, 3], dtype=torch.int64, device=device)
    outputs = {
        "rgb": torch.full((2, 3, 4, 3), 0.75, device=device),
        "depth": torch.full((2, 3, 4, 1), 123.0, device=device),
        "alpha": torch.ones((2, 3, 4, 1), device=device),
        "semantic_id": torch.full((2, 3, 4, 1), 99, dtype=torch.int64, device=device),
    }

    service.render(
        camera_transforms,
        intrinsics,
        scene_ids,
        outputs=outputs,
        active_camera_ids=torch.tensor([1], dtype=torch.int64, device=device),
    )

    assert backend.render_calls == 1
    assert torch.all(outputs["rgb"][0] == 0)
    assert torch.all(torch.isposinf(outputs["depth"][0]))
    assert torch.all(outputs["alpha"][0] == 0)
    assert torch.all(outputs["semantic_id"][0] == -1)
    assert torch.all(outputs["alpha"][1] == 1)


def test_owned_active_subset_has_defined_inactive_defaults() -> None:
    service, _, device = initialized_service()
    camera_transforms = torch.eye(4, device=device).repeat(2, 1, 1).contiguous()
    intrinsics = torch.eye(3, device=device).repeat(2, 1, 1).contiguous()
    scene_ids = torch.tensor([3, 3], dtype=torch.int64, device=device)

    outputs = service.render(
        camera_transforms,
        intrinsics,
        scene_ids,
        active_camera_ids=torch.tensor([1], dtype=torch.int64, device=device),
    )

    assert torch.all(outputs["rgb"][0] == 0)
    assert torch.all(torch.isposinf(outputs["depth"][0]))
    assert torch.all(outputs["alpha"][0] == 0)
    assert torch.all(outputs["semantic_id"][0] == -1)


def test_contract_rejects_wrong_shape_and_unregistered_scene() -> None:
    service, _, device = initialized_service()
    camera_transforms = torch.eye(4, device=device).repeat(2, 1, 1).contiguous()
    intrinsics = torch.eye(3, device=device).repeat(1, 1, 1).contiguous()

    with pytest.raises(ValueError, match="intrinsics dimension 0"):
        service.render(camera_transforms, intrinsics, torch.tensor([3, 3], dtype=torch.int64, device=device))

    intrinsics = torch.eye(3, device=device).repeat(2, 1, 1).contiguous()
    with pytest.raises(ValueError, match="Unregistered scene"):
        service.render(camera_transforms, intrinsics, torch.tensor([3, 9], dtype=torch.int64, device=device))


def test_default_service_requires_cuda() -> None:
    service = RendererService(DeterministicFakeBackend(), height=1, width=1)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        service.initialize(stage=object(), device="cpu")


def test_render_cadence_reuses_last_outputs_without_backend_call() -> None:
    device = torch.device("cpu")
    backend = DeterministicFakeBackend()
    service = RendererService(
        backend,
        height=2,
        width=2,
        max_views=1,
        render_every_n=2,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=object(), device=device)
    service.load_scene(3, **make_scene(device))
    camera_transforms = torch.eye(4, device=device).repeat(1, 1, 1).contiguous()
    intrinsics = torch.eye(3, device=device).repeat(1, 1, 1).contiguous()
    scene_ids = torch.tensor([3], dtype=torch.int64, device=device)

    first = service.render(camera_transforms, intrinsics, scene_ids)
    second = service.render(camera_transforms, intrinsics, scene_ids)

    assert backend.render_calls == 1
    assert second is first


def test_scene_offsets_are_reused_between_renders() -> None:
    service, backend, device = initialized_service()
    camera_transforms = torch.eye(4, device=device).repeat(1, 1, 1).contiguous()
    intrinsics = torch.eye(3, device=device).repeat(1, 1, 1).contiguous()
    scene_ids = torch.tensor([3], dtype=torch.int64, device=device)

    service.render(camera_transforms, intrinsics, scene_ids)
    assert backend.last_request is not None
    first_offsets = backend.last_request.scene_offsets

    service.render(camera_transforms, intrinsics, scene_ids)
    assert backend.last_request is not None
    assert backend.last_request.scene_offsets is first_offsets
