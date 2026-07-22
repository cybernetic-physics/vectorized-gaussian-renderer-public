"""GPU smoke for packed scene IDs, transforms, and active camera subsets."""

from __future__ import annotations

import torch

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle,
    synthetic_scene_tensors,
)


def _service(
    *,
    max_views: int,
    height: int,
    width: int,
    max_gaussians: int,
) -> tuple[RendererService, CustomCudaBackend]:
    backend = CustomCudaBackend(
        max_visible_records=max_views * max_gaussians,
        max_intersections=max_views * max_gaussians * 8,
        output_srgb=False,
    )
    service = RendererService(
        backend,
        height=height,
        width=width,
        max_views=max_views,
    )
    service.initialize(stage=None, device="cuda")
    return service, backend


def _load(
    service: RendererService,
    scene_id: int,
    scene: dict[str, torch.Tensor],
) -> None:
    service.load_scene(
        scene_id,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda")
    height = 128
    width = 128
    scene_a = synthetic_scene_tensors(5_000, seed=71, device=device)
    scene_b = synthetic_scene_tensors(7_000, seed=72, device=device)
    scene_b["colors"] = (scene_b["colors"] * 0.5).contiguous()
    scene_b["semantic_ids"] = (scene_b["semantic_ids"] + 2_000).contiguous()

    cameras = camera_bundle(4, width, height, device=device)
    scene_ids = torch.tensor([17, 901, 17, 901], device=device, dtype=torch.int64)
    env_xforms = torch.eye(4, device=device).repeat(4, 1, 1).contiguous()
    env_xforms[1, 0, 3] = 0.20
    env_xforms[2, 1, 3] = -0.15
    env_xforms[3, 0, 3] = -0.25

    service, backend = _service(
        max_views=4,
        height=height,
        width=width,
        max_gaussians=7_000,
    )
    _load(service, 17, scene_a)
    _load(service, 901, scene_b)
    service.update_scene_transforms(scene_ids, env_xforms)
    batched = service.render(cameras.viewmats, cameras.intrinsics, scene_ids)
    service.synchronize()
    full_outputs = {name: tensor.clone() for name, tensor in batched.items()}
    counters = backend.check_capacity(synchronize=False)

    for expected_scene_id, scene, indices in (
        (17, scene_a, torch.tensor([0, 2], device=device, dtype=torch.int64)),
        (901, scene_b, torch.tensor([1, 3], device=device, dtype=torch.int64)),
    ):
        reference_service, reference_backend = _service(
            max_views=2,
            height=height,
            width=width,
            max_gaussians=scene["means"].shape[0],
        )
        _load(reference_service, expected_scene_id, scene)
        reference_scene_ids = torch.full(
            (2,),
            expected_scene_id,
            device=device,
            dtype=torch.int64,
        )
        reference_service.update_scene_transforms(
            reference_scene_ids,
            env_xforms.index_select(0, indices).contiguous(),
        )
        reference = reference_service.render(
            cameras.viewmats.index_select(0, indices).contiguous(),
            cameras.intrinsics.index_select(0, indices).contiguous(),
            reference_scene_ids,
        )
        reference_service.synchronize()
        reference_backend.check_capacity(synchronize=False)
        for name in ("rgb", "depth", "alpha"):
            torch.testing.assert_close(
                full_outputs[name].index_select(0, indices),
                reference[name],
                rtol=2e-5,
                atol=2e-6,
            )
        torch.testing.assert_close(
            full_outputs["semantic_id"].index_select(0, indices),
            reference["semantic_id"],
        )
        reference_service.shutdown()

    active = torch.tensor([1, 2], device=device, dtype=torch.int64)
    subset = service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        active_camera_ids=active,
    )
    service.synchronize()
    backend.check_capacity(synchronize=False)
    torch.testing.assert_close(
        subset["rgb"].index_select(0, active),
        full_outputs["rgb"].index_select(0, active),
    )
    inactive = torch.tensor([0, 3], device=device, dtype=torch.int64)
    inactive_rgb = subset["rgb"].index_select(0, inactive)
    inactive_depth = subset["depth"].index_select(0, inactive)
    inactive_alpha = subset["alpha"].index_select(0, inactive)
    inactive_semantic = subset["semantic_id"].index_select(0, inactive)
    if not torch.all(inactive_rgb == 0):
        raise AssertionError("Inactive RGB outputs were not reset to background.")
    if not torch.all(torch.isposinf(inactive_depth)):
        raise AssertionError("Inactive depth outputs were not reset to +inf.")
    if not torch.all(inactive_alpha == 0):
        raise AssertionError("Inactive alpha outputs were not reset to background.")
    if not torch.all(inactive_semantic == -1):
        raise AssertionError("Inactive semantic outputs were not reset to -1.")

    print(
        "MULTI_SCENE_CUDA_SMOKE_OK",
        {
            "scene_ids": scene_ids.tolist(),
            "active_camera_ids": active.tolist(),
            "inactive_output_policy": {
                "rgb": 0,
                "depth": "+inf",
                "alpha": 0,
                "semantic_id": -1,
            },
            "packed_gaussians": 12_000,
            **counters,
        },
    )
    service.shutdown()


if __name__ == "__main__":
    main()
