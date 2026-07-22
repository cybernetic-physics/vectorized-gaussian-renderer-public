"""Finite CUDA smoke test for the project-owned renderer backend."""

from __future__ import annotations

import torch

from isaacsim_gaussian_renderer import CustomCudaBackend, RendererService
from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle,
    synthetic_scene_tensors,
)


def _load_synthetic_scene(
    service: RendererService,
    *,
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


def _cuda_allocator_state() -> dict[str, int]:
    stats = torch.cuda.memory_stats()
    names = (
        "allocation.all.allocated",
        "allocation.all.freed",
        "allocated_bytes.all.allocated",
        "allocated_bytes.all.freed",
        "segment.all.allocated",
        "segment.all.freed",
        "reserved_bytes.all.allocated",
        "reserved_bytes.all.freed",
    )
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        **{name: int(stats[name]) for name in names},
    }


@torch.no_grad()
def _chunked_submission_smoke(
    scene: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, object]:
    """Exercise real chunk helper kernels against an unchunked oracle."""

    logical_batch = 5
    physical_batch = 2
    height = 64
    width = 64
    scene_id = 173
    count = int(scene["means"].shape[0])
    cameras = camera_bundle(logical_batch, width, height, device=device)
    scene_ids = torch.full(
        (logical_batch,),
        scene_id,
        device=device,
        dtype=torch.int64,
    )
    active_ids = torch.tensor([0, 2, 4], device=device, dtype=torch.int64)

    reference_backend = CustomCudaBackend(
        max_visible_records=logical_batch * count,
        max_intersections=logical_batch * count * 64,
        fixed_capacity_sort=True,
        adaptive_capacity=False,
        output_srgb=False,
    )
    reference_service = RendererService(
        reference_backend,
        height=height,
        width=width,
        max_views=logical_batch,
    )
    reference_service.initialize(stage=None, device=device)
    _load_synthetic_scene(
        reference_service,
        scene_id=scene_id,
        scene=scene,
    )
    reference_outputs = reference_service.prepare_outputs(logical_batch)
    reference_service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=reference_outputs,
    )
    reference_service.synchronize()
    reference_counters = reference_backend.check_capacity(synchronize=False)

    chunked_backend = CustomCudaBackend(
        max_visible_records=physical_batch * count,
        # The logical reference total is a conservative upper bound for every
        # physical chunk in this finite functional smoke.
        max_intersections=max(1, reference_counters["tile_intersections"]),
        fixed_capacity_sort=True,
        max_physical_views=physical_batch,
        adaptive_capacity=False,
        output_srgb=False,
    )
    chunked_service = RendererService(
        chunked_backend,
        height=height,
        width=width,
        max_views=logical_batch,
    )
    chunked_service.initialize(stage=None, device=device)
    _load_synthetic_scene(
        chunked_service,
        scene_id=scene_id,
        scene=scene,
    )
    chunked_outputs = chunked_service.prepare_outputs(logical_batch)
    chunked_service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=chunked_outputs,
    )
    chunked_service.synchronize()

    output_pointers = {
        name: int(tensor.data_ptr())
        for name, tensor in chunked_outputs.items()
    }
    workspace_storage_pointers = {
        int(tensor.untyped_storage().data_ptr())
        for tensor in chunked_service.workspace.values()
    }
    allocator_before = _cuda_allocator_state()
    chunked_service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=chunked_outputs,
    )
    chunked_service.synchronize()
    allocator_after = _cuda_allocator_state()
    assert allocator_after == allocator_before
    assert output_pointers == {
        name: int(tensor.data_ptr())
        for name, tensor in chunked_outputs.items()
    }
    assert workspace_storage_pointers == {
        int(tensor.untyped_storage().data_ptr())
        for tensor in chunked_service.workspace.values()
    }

    chunked_counters = chunked_backend.check_capacity(synchronize=False)
    physical_counters = chunked_backend.read_physical_capacity_counters(
        synchronize=False
    )
    assert chunked_counters == reference_counters
    assert physical_counters["visible_gaussians"] <= (
        physical_batch * count
    )
    assert physical_counters["tile_intersections"] <= max(
        1, reference_counters["tile_intersections"]
    )
    for name in ("rgb", "depth", "alpha"):
        torch.testing.assert_close(
            chunked_outputs[name],
            reference_outputs[name],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        chunked_outputs["semantic_id"],
        reference_outputs["semantic_id"],
        rtol=0,
        atol=0,
    )

    reference_service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=reference_outputs,
        active_camera_ids=active_ids,
    )
    chunked_service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=chunked_outputs,
        active_camera_ids=active_ids,
    )
    reference_service.synchronize()
    chunked_service.synchronize()
    for name in ("rgb", "depth", "alpha", "semantic_id"):
        torch.testing.assert_close(
            chunked_outputs[name],
            reference_outputs[name],
            rtol=0,
            atol=0,
        )

    result: dict[str, object] = {
        "logical_batch": logical_batch,
        "physical_batch": physical_batch,
        "chunks": 3,
        "logical_counters": chunked_counters,
        "physical_capacity_counters": physical_counters,
        "allocator_state_unchanged": True,
    }
    chunked_service.shutdown()
    reference_service.shutdown()
    return result


# Preserve tensor version counters so this smoke exercises the documented
# validated-input reuse path. Inference-mode tensors intentionally fail closed
# because their in-place mutation cannot be detected.
@torch.no_grad()
def main() -> None:
    device = torch.device("cuda")
    batch = 2
    height = 64
    width = 64
    count = 1_000
    scene = synthetic_scene_tensors(count, seed=7, device=device)
    cameras = camera_bundle(batch, width, height, device=device)
    scene_ids = torch.full((batch,), 73, device=device, dtype=torch.int64)
    backend = CustomCudaBackend(
        max_visible_records=batch * count,
        max_intersections=batch * count * 8,
        output_srgb=False,
    )
    service = RendererService(
        backend,
        height=height,
        width=width,
        max_views=batch,
    )
    service.initialize(stage=None, device=device)
    assert backend.workspace_storage_aliases == {
        "depth_bucket_tau": "sort_temp",
    }
    assert (
        backend.workspace["depth_bucket_tau"].untyped_storage().data_ptr()
        == backend.workspace["sort_temp"].untyped_storage().data_ptr()
    )
    service.load_scene(
        73,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )
    outputs = service.render(cameras.viewmats, cameras.intrinsics, scene_ids)
    service.synchronize()
    counters = backend.check_capacity(synchronize=False)

    assert outputs["rgb"].is_cuda
    assert outputs["depth"].is_cuda
    assert outputs["alpha"].is_cuda
    assert outputs["semantic_id"].is_cuda
    assert torch.isfinite(outputs["rgb"]).all()
    assert outputs["alpha"].min() >= 0
    assert outputs["alpha"].max() <= 1
    assert counters["visible_gaussians"] > 0
    assert counters["tile_intersections"] > 0

    reference_outputs = {
        name: tensor.clone() for name, tensor in outputs.items()
    }
    adaptive_backend = CustomCudaBackend(
        max_visible_records=max(1, counters["visible_gaussians"] // 8),
        max_intersections=max(1, counters["tile_intersections"] // 8),
        enable_projection_cache=True,
        output_srgb=False,
    )
    adaptive_service = RendererService(
        adaptive_backend,
        height=height,
        width=width,
        max_views=batch,
    )
    adaptive_service.initialize(stage=None, device=device)
    adaptive_service.load_scene(
        73,
        means=scene["means"],
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )
    adaptive_outputs = adaptive_service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
    )
    adaptive_service.synchronize()
    adaptive_counters = adaptive_backend.check_capacity(synchronize=False)
    first_adaptation = adaptive_backend.capacity_stats
    assert first_adaptation["totals"]["growth_events"] > 0
    assert first_adaptation["totals"]["retries"] > 0
    assert first_adaptation["last_render"]["status"] == "success"
    assert adaptive_counters["visible_overflow"] == 0
    assert adaptive_counters["intersection_overflow"] == 0
    assert adaptive_backend.workspace_storage_aliases == {
        "depth_bucket_tau": "sort_temp",
    }
    alias_savings = first_adaptation["last_allocation"][
        "workspace_alias_savings_bytes"
    ]
    assert alias_savings > 0
    assert first_adaptation["last_allocation"][
        "requested_workspace_unaliased_bytes"
    ] == (
        first_adaptation["last_allocation"]["requested_workspace_bytes"]
        + alias_savings
    )
    for name in ("rgb", "depth", "alpha"):
        torch.testing.assert_close(
            adaptive_outputs[name],
            reference_outputs[name],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
    torch.testing.assert_close(
        adaptive_outputs["semantic_id"],
        reference_outputs["semantic_id"],
        rtol=0,
        atol=0,
    )
    output_pointers = {
        name: tensor.data_ptr()
        for name, tensor in adaptive_outputs.items()
    }
    adaptive_service.render(
        cameras.viewmats,
        cameras.intrinsics,
        scene_ids,
        outputs=adaptive_outputs,
    )
    adaptive_service.synchronize()
    repeated_adaptation = adaptive_backend.capacity_stats
    assert repeated_adaptation["last_render"]["validation"] == (
        "validated-identical-input"
    )
    assert repeated_adaptation["totals"]["growth_events"] == (
        first_adaptation["totals"]["growth_events"]
    )
    assert output_pointers == {
        name: tensor.data_ptr()
        for name, tensor in adaptive_outputs.items()
    }
    adaptive_service.shutdown()

    far_means = scene["means"].clone()
    far_means[:, 2] = backend.far_plane + 1_000.0
    service.load_scene(
        74,
        means=far_means,
        scales=scene["scales"],
        rotations=scene["quats"],
        opacities=scene["opacities"],
        features=scene["colors"],
        semantic_ids=scene["semantic_ids"].to(torch.int64),
    )
    empty_scene_ids = torch.full(
        (batch,),
        74,
        device=device,
        dtype=torch.int64,
    )
    service.render(
        cameras.viewmats,
        cameras.intrinsics,
        empty_scene_ids,
        outputs=outputs,
    )
    service.synchronize()
    empty_counters = backend.check_capacity(synchronize=False)
    assert empty_counters == {
        "visible_gaussians": 0,
        "tile_intersections": 0,
        "visible_overflow": 0,
        "intersection_overflow": 0,
        "active_tiles": 0,
    }
    assert int(torch.count_nonzero(outputs["alpha"]).item()) == 0

    chunked = _chunked_submission_smoke(scene, device=device)

    print(
        "CUSTOM_CUDA_BACKEND_SMOKE_OK",
        {
            "nonempty": counters,
            "empty": empty_counters,
            "adaptive": first_adaptation,
            "workspace_storage_aliases": (
                backend.workspace_storage_aliases
            ),
            "chunked": chunked,
        },
    )
    service.shutdown()


if __name__ == "__main__":
    main()
