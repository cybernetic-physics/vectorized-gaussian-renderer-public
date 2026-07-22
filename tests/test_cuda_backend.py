from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from isaacsim_gaussian_renderer import (
    CustomCudaBackend,
    RendererCapacityError,
    RendererService,
)
from isaacsim_gaussian_renderer.cuda_backend import (
    _emitted_sort_key_end_bit,
)


class FakeNativeModule:
    def __init__(
        self,
        *,
        expected_covariance_epsilon: float = 0.3,
        expected_antialiased: bool = False,
        expected_semantic_min_alpha: float = 0.01,
        expected_ray_gaussian_evaluation: bool = False,
        expected_compact_projection_cache: bool | None = False,
        expected_materialize_projected_records: bool | None = False,
        pointer_sort_temp_bytes: int = 64,
        double_buffer_sort_temp_bytes: int = 64,
        sorted_buffer_selectors: list[int] | None = None,
    ) -> None:
        self.render_calls = 0
        self.last_scene: list[torch.Tensor] | None = None
        self.last_cameras: list[torch.Tensor] | None = None
        self.last_workspace: list[torch.Tensor] | None = None
        self.last_deterministic: bool | None = None
        self.last_tile_size: int | None = None
        self.last_covariance_epsilon: float | None = None
        self.last_antialiased: bool | None = None
        self.last_semantic_min_alpha: float | None = None
        self.last_ray_gaussian_evaluation: bool | None = None
        self.last_full_sensor_output: bool | None = None
        self.last_fixed_capacity_sort: bool | None = None
        self.reuse_projection_flags: list[bool] = []
        self.compact_projection_cache_flags: list[bool] = []
        self.materialize_projected_records_flags: list[bool] = []
        self.expected_covariance_epsilon = expected_covariance_epsilon
        self.expected_antialiased = expected_antialiased
        self.expected_semantic_min_alpha = expected_semantic_min_alpha
        self.expected_ray_gaussian_evaluation = (
            expected_ray_gaussian_evaluation
        )
        self.expected_compact_projection_cache = (
            expected_compact_projection_cache
        )
        self.expected_materialize_projected_records = (
            expected_materialize_projected_records
        )
        self.pointer_sort_temp_bytes = pointer_sort_temp_bytes
        self.double_buffer_sort_temp_bytes = double_buffer_sort_temp_bytes
        self.sorted_buffer_selectors = sorted_buffer_selectors or [1]
        self.pointer_sort_temp_queries: list[tuple[int, int]] = []
        self.double_buffer_sort_temp_queries: list[tuple[int, int]] = []
        self.rendered_camera_batches: list[int] = []
        self.sort_buffer_pointers: list[tuple[int, int, int, int]] = []

    def sort_temp_bytes(self, num_items: int, num_segments: int) -> int:
        assert num_items > 0
        assert num_segments > 0
        self.pointer_sort_temp_queries.append((num_items, num_segments))
        return self.pointer_sort_temp_bytes

    def radix_sort_double_buffer_temp_bytes(
        self,
        num_items: int,
        num_segments: int,
    ) -> int:
        assert num_items > 0
        assert num_segments > 0
        self.double_buffer_sort_temp_queries.append(
            (num_items, num_segments)
        )
        return self.double_buffer_sort_temp_bytes

    def render(
        self,
        scene: list[torch.Tensor],
        cameras: list[torch.Tensor],
        outputs: list[torch.Tensor],
        workspace: list[torch.Tensor],
        height: int,
        width: int,
        max_scene_gaussians: int,
        near_plane: float,
        far_plane: float,
        gaussian_support_sigma: float,
        covariance_epsilon: float,
        antialiased: bool,
        semantic_min_alpha: float,
        ray_gaussian_evaluation: bool,
        tile_size: int,
        depth_bucket_count: int,
        depth_bucket_group_size: int,
        compact_projection_cache: bool,
        materialize_projected_records: bool,
        reuse_projection: bool,
        output_srgb: bool,
        deterministic: bool,
        full_sensor_output: bool,
        fixed_capacity_sort: bool,
    ) -> int:
        self.render_calls += 1
        self.rendered_camera_batches.append(int(cameras[0].shape[0]))
        self.last_scene = scene
        self.last_cameras = cameras
        self.last_workspace = workspace
        self.sort_buffer_pointers.append(
            tuple(int(workspace[index].data_ptr()) for index in (7, 8, 9, 10))
        )
        self.last_deterministic = deterministic
        self.last_tile_size = tile_size
        self.last_covariance_epsilon = covariance_epsilon
        self.last_antialiased = antialiased
        self.last_semantic_min_alpha = semantic_min_alpha
        self.last_ray_gaussian_evaluation = ray_gaussian_evaluation
        self.compact_projection_cache_flags.append(
            compact_projection_cache
        )
        self.materialize_projected_records_flags.append(
            materialize_projected_records
        )
        self.last_full_sensor_output = full_sensor_output
        self.last_fixed_capacity_sort = fixed_capacity_sort
        self.reuse_projection_flags.append(reuse_projection)
        outputs[0].fill_(0.25)
        outputs[1].fill_(2.0)
        outputs[2].fill_(0.75)
        outputs[3].fill_(9)
        workspace[13].copy_(torch.tensor([3, 5, 0, 0, 2], dtype=torch.int64))
        assert (height, width) == (4, 5)
        assert max_scene_gaussians == 3
        assert near_plane == 0.01
        assert far_plane == 100.0
        assert gaussian_support_sigma == 3.33
        assert covariance_epsilon == self.expected_covariance_epsilon
        assert antialiased is self.expected_antialiased
        assert semantic_min_alpha == self.expected_semantic_min_alpha
        assert (
            ray_gaussian_evaluation
            is self.expected_ray_gaussian_evaluation
        )
        assert depth_bucket_count == 4096
        assert depth_bucket_group_size == 64
        if self.expected_compact_projection_cache is not None:
            assert (
                compact_projection_cache
                is self.expected_compact_projection_cache
            )
        if self.expected_materialize_projected_records is not None:
            assert (
                materialize_projected_records
                is self.expected_materialize_projected_records
            )
        assert output_srgb
        assert len(scene) == 7
        assert scene[1].shape[1] == 6
        return self.sorted_buffer_selectors[
            min(self.render_calls - 1, len(self.sorted_buffer_selectors) - 1)
        ]

    def prepare_chunked_active_ids(
        self,
        active_camera_ids: torch.Tensor,
        active_mask: torch.Tensor,
        chunk_active_ids: torch.Tensor,
        logical_batch: int,
        physical_batch: int,
    ) -> None:
        active_mask[:logical_batch].zero_()
        valid = active_camera_ids[
            (active_camera_ids >= 0) & (active_camera_ids < logical_batch)
        ]
        if valid.numel():
            active_mask.index_fill_(0, valid, 1)
        chunk_active_ids.fill_(-1)
        for camera in range(logical_batch):
            if int(active_mask[camera]):
                chunk_active_ids[
                    camera // physical_batch,
                    camera % physical_batch,
                ] = camera % physical_batch

    def aggregate_chunk_counters(
        self,
        chunk_counters: torch.Tensor,
        counters: torch.Tensor,
        physical_capacity_counters: torch.Tensor,
        chunk_count: int,
    ) -> None:
        torch.sum(chunk_counters[:chunk_count], dim=0, out=counters)
        torch.amax(
            chunk_counters[:chunk_count],
            dim=0,
            out=physical_capacity_counters,
        )


class CapacitySequenceNative(FakeNativeModule):
    def __init__(
        self,
        counters: list[dict[str, int]],
        **native_kwargs: Any,
    ) -> None:
        super().__init__(**native_kwargs)
        self.counter_sequence = counters
        self.capacities: list[tuple[int, int]] = []
        self.output_pointers: list[tuple[int, ...]] = []

    def render(self, *args: Any) -> int:
        sorted_buffer_selector = super().render(*args)
        outputs = args[2]
        workspace = args[3]
        counters = self.counter_sequence[
            min(self.render_calls - 1, len(self.counter_sequence) - 1)
        ]
        workspace[13].copy_(
            torch.tensor(
                [
                    counters["visible_gaussians"],
                    counters["tile_intersections"],
                    counters["visible_overflow"],
                    counters["intersection_overflow"],
                    counters["active_tiles"],
                ],
                dtype=torch.int64,
            )
        )
        outputs[0].fill_(self.render_calls / 10.0)
        self.capacities.append(
            (int(workspace[2].numel()), int(workspace[7].numel()))
        )
        self.output_pointers.append(
            tuple(int(tensor.data_ptr()) for tensor in outputs)
        )
        return sorted_buffer_selector


class ChunkAwareFakeNative(FakeNativeModule):
    """CPU double that makes camera order and transform alignment observable."""

    def render(self, *args: Any) -> int:
        sorted_buffer_selector = super().render(*args)
        cameras = args[1]
        outputs = args[2]
        workspace = args[3]
        batch = int(cameras[0].shape[0])
        active_ids = cameras[4]
        valid_ids = (
            list(range(batch))
            if active_ids.numel() == 0
            else [
                int(value)
                for value in active_ids.tolist()
                if 0 <= int(value) < batch
            ]
        )
        valid_set = set(valid_ids)
        for camera in range(batch):
            if camera not in valid_set:
                outputs[0][camera].zero_()
                outputs[1][camera].fill_(float("inf"))
                outputs[2][camera].zero_()
                outputs[3][camera].fill_(-1)
                continue
            value = (
                cameras[0][camera, 0, 3]
                + cameras[2][camera, 0, 3]
            )
            outputs[0][camera].fill_(value)
            outputs[1][camera].fill_(value + 100.0)
            outputs[2][camera].fill_(0.5)
            outputs[3][camera].fill_(int(value.item()))
        active_count = len(valid_set)
        workspace[13].copy_(
            torch.tensor(
                [
                    active_count * 3,
                    active_count * 5,
                    0,
                    0,
                    active_count,
                ],
                dtype=torch.int64,
            )
        )
        return sorted_buffer_selector


def scene_tensors(count: int, semantic_id: int) -> dict[str, torch.Tensor]:
    return {
        "means": torch.zeros((count, 3), dtype=torch.float32),
        "scales": torch.full((count, 3), 0.1, dtype=torch.float32),
        "rotations": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count,
            dtype=torch.float32,
        ),
        "opacities": torch.full((count,), 0.5, dtype=torch.float32),
        "features": torch.full((count, 3), 0.2, dtype=torch.float32),
        "semantic_ids": torch.full(
            (count,),
            semantic_id,
            dtype=torch.int64,
        ),
    }


def make_backend(native: Any) -> CustomCudaBackend:
    return CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        native_module=native,
        allow_cpu_for_tests=True,
    )


def make_chunked_service(
    native: Any,
    *,
    outputs: tuple[str, ...] = ("rgb", "depth", "alpha", "semantic_id"),
) -> tuple[RendererService, CustomCudaBackend]:
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        fixed_capacity_sort=True,
        max_physical_views=2,
        adaptive_capacity=False,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        outputs=outputs,
        max_views=5,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    return service, backend


def test_chunked_rgb_preserves_outputs_across_double_buffer_swaps() -> None:
    native = FakeNativeModule(sorted_buffer_selectors=[0, 1, 0])
    service, backend = make_chunked_service(native, outputs=("rgb",))
    initial_pointers = tuple(
        int(backend.workspace[name].data_ptr())
        for name in ("keys_in", "keys_out", "values_in", "values_out")
    )

    outputs = service.render(
        torch.eye(4).repeat(5, 1, 1).contiguous(),
        torch.eye(3).repeat(5, 1, 1).contiguous(),
        torch.full((5,), 17, dtype=torch.int64),
    )

    assert tuple(outputs) == ("rgb",)
    assert torch.all(outputs["rgb"] == 0.25)
    assert native.rendered_camera_batches == [2, 2, 1]
    assert native.sort_buffer_pointers[0] == initial_pointers
    assert native.sort_buffer_pointers[1] == (
        initial_pointers[1],
        initial_pointers[0],
        initial_pointers[3],
        initial_pointers[2],
    )
    assert native.sort_buffer_pointers[2] == native.sort_buffer_pointers[1]
    assert tuple(
        int(backend.workspace[name].data_ptr())
        for name in ("keys_in", "keys_out", "values_in", "values_out")
    ) == initial_pointers


def test_custom_backend_packs_arbitrary_scene_ids_and_submits_once() -> None:
    native = FakeNativeModule()
    backend = make_backend(native)
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=2,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=object(), device="cpu")
    service.load_scene(17, **scene_tensors(2, 4))
    service.load_scene(901, **scene_tensors(3, 8))

    viewmats = torch.eye(4).repeat(2, 1, 1).contiguous()
    intrinsics = torch.eye(3).repeat(2, 1, 1).contiguous()
    scene_ids = torch.tensor([901, 17], dtype=torch.int64)
    outputs = service.render(viewmats, intrinsics, scene_ids)

    assert native.render_calls == 1
    assert native.rendered_camera_batches == [2]
    assert "logical_chunk_counters" not in backend.workspace
    assert native.last_scene is not None
    torch.testing.assert_close(
        native.last_scene[5],
        torch.tensor([17, 901], dtype=torch.int64),
    )
    torch.testing.assert_close(
        native.last_scene[6],
        torch.tensor([0, 2, 5], dtype=torch.int64),
    )
    assert native.last_scene[0].shape == (5, 3)
    assert native.last_deterministic is False
    assert native.last_tile_size == 16
    assert torch.all(outputs["rgb"] == 0.25)
    assert torch.all(outputs["semantic_id"] == 9)
    assert backend.check_capacity() == {
        "visible_gaussians": 3,
        "tile_intersections": 5,
        "visible_overflow": 0,
        "intersection_overflow": 0,
        "active_tiles": 2,
    }


def test_chunked_logical_batch_preserves_order_transforms_and_tail() -> None:
    native = ChunkAwareFakeNative()
    service, backend = make_chunked_service(native)
    viewmats = torch.eye(4).repeat(5, 1, 1).contiguous()
    viewmats[:, 0, 3] = torch.arange(5, dtype=torch.float32)
    intrinsics = torch.eye(3).repeat(5, 1, 1).contiguous()
    scene_ids = torch.full((5,), 17, dtype=torch.int64)
    env_xforms = torch.eye(4).repeat(5, 1, 1).contiguous()
    env_xforms[:, 0, 3] = 10.0 * torch.arange(5, dtype=torch.float32)
    service.update_scene_transforms(scene_ids, env_xforms)
    outputs = service.prepare_outputs(5)
    output_pointers = {
        name: int(tensor.data_ptr()) for name, tensor in outputs.items()
    }
    workspace_pointers = {
        name: int(tensor.data_ptr())
        for name, tensor in service.workspace.items()
    }

    returned = service.render(
        viewmats,
        intrinsics,
        scene_ids,
        outputs=outputs,
    )

    assert returned is outputs
    assert native.rendered_camera_batches == [2, 2, 1]
    expected = 11.0 * torch.arange(5, dtype=torch.float32)
    torch.testing.assert_close(outputs["rgb"][:, 0, 0, 0], expected)
    torch.testing.assert_close(outputs["depth"][:, 0, 0, 0], expected + 100.0)
    torch.testing.assert_close(
        outputs["semantic_id"][:, 0, 0, 0],
        expected.to(torch.int64),
    )
    assert backend.read_counters() == {
        "visible_gaussians": 15,
        "tile_intersections": 25,
        "visible_overflow": 0,
        "intersection_overflow": 0,
        "active_tiles": 5,
    }
    assert backend.read_physical_capacity_counters() == {
        "visible_gaussians": 6,
        "tile_intersections": 10,
        "visible_overflow": 0,
        "intersection_overflow": 0,
        "active_tiles": 2,
    }
    assert output_pointers == {
        name: int(tensor.data_ptr()) for name, tensor in outputs.items()
    }
    assert workspace_pointers == {
        name: int(tensor.data_ptr())
        for name, tensor in service.workspace.items()
    }


def test_chunked_logical_batch_maps_active_ids_across_boundaries() -> None:
    native = ChunkAwareFakeNative()
    service, backend = make_chunked_service(native)
    viewmats = torch.eye(4).repeat(5, 1, 1).contiguous()
    viewmats[:, 0, 3] = torch.arange(5, dtype=torch.float32)
    intrinsics = torch.eye(3).repeat(5, 1, 1).contiguous()
    scene_ids = torch.full((5,), 17, dtype=torch.int64)
    active_ids = torch.tensor([0, 2, 4], dtype=torch.int64)

    outputs = service.render(
        viewmats,
        intrinsics,
        scene_ids,
        outputs=service.prepare_outputs(5),
        active_camera_ids=active_ids,
    )

    assert native.rendered_camera_batches == [2, 2, 1]
    torch.testing.assert_close(
        outputs["rgb"][:, 0, 0, 0],
        torch.tensor([0.0, 0.0, 2.0, 0.0, 4.0]),
    )
    assert torch.all(torch.isposinf(outputs["depth"][[1, 3]]))
    assert torch.all(outputs["alpha"][[1, 3]] == 0)
    assert torch.all(outputs["semantic_id"][[1, 3]] == -1)
    assert backend.read_counters() == {
        "visible_gaussians": 9,
        "tile_intersections": 15,
        "visible_overflow": 0,
        "intersection_overflow": 0,
        "active_tiles": 3,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fixed_capacity_sort": False}, "fixed_capacity_sort=True"),
        ({"adaptive_capacity": True}, "adaptive_capacity=False"),
        ({"enable_projection_cache": True}, "enable_projection_cache=False"),
    ],
)
def test_chunked_logical_batch_rejects_sync_or_cache_modes(
    kwargs: dict[str, bool],
    message: str,
) -> None:
    options = {
        "fixed_capacity_sort": True,
        "adaptive_capacity": False,
        "enable_projection_cache": False,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        CustomCudaBackend(
            max_physical_views=128,
            native_module=FakeNativeModule(),
            allow_cpu_for_tests=True,
            **options,
        )


def test_custom_backend_rgb_only_uses_sink_outputs_and_fixed_sort() -> None:
    native = FakeNativeModule()
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        fixed_capacity_sort=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        outputs=("rgb",),
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))

    outputs = service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    assert tuple(outputs) == ("rgb",)
    assert torch.all(outputs["rgb"] == 0.25)
    assert native.last_full_sensor_output is False
    assert native.last_fixed_capacity_sort is True
    assert native.last_workspace is not None
    assert backend.workspace["dummy_depth"].numel() == 1
    assert backend.workspace["dummy_alpha"].numel() == 1
    assert backend.workspace["dummy_semantic"].numel() == 1


def test_custom_backend_rejects_partial_sensor_contract() -> None:
    native = FakeNativeModule()
    backend = make_backend(native)
    service = RendererService(
        backend,
        height=4,
        width=5,
        outputs=("rgb", "alpha"),
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))

    with pytest.raises(ValueError, match="supports exactly"):
        service.render(
            torch.eye(4).unsqueeze(0).contiguous(),
            torch.eye(3).unsqueeze(0).contiguous(),
            torch.tensor([17], dtype=torch.int64),
        )


def test_custom_backend_rejects_fixed_sort_with_deterministic_mode() -> None:
    with pytest.raises(ValueError, match="fixed_capacity_sort"):
        CustomCudaBackend(
            deterministic=True,
            fixed_capacity_sort=True,
            native_module=FakeNativeModule(),
            allow_cpu_for_tests=True,
        )


def test_custom_backend_uses_aligned_environment_transforms_and_active_ids() -> None:
    native = FakeNativeModule()
    backend = make_backend(native)
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=2,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    scene_ids = torch.tensor([17, 17], dtype=torch.int64)
    env_xforms = torch.eye(4).repeat(2, 1, 1).contiguous()
    env_xforms[1, 0, 3] = 2.0
    service.update_scene_transforms(scene_ids, env_xforms)

    viewmats = torch.eye(4).repeat(2, 1, 1).contiguous()
    intrinsics = torch.eye(3).repeat(2, 1, 1).contiguous()
    active = torch.tensor([1], dtype=torch.int64)
    service.render(
        viewmats,
        intrinsics,
        scene_ids,
        active_camera_ids=active,
    )

    assert native.last_cameras is not None
    assert native.last_cameras[2] is env_xforms
    assert native.last_cameras[4] is active
    assert len(native.last_workspace or []) == 24


def test_workspace_is_fixed_and_reports_bytes() -> None:
    native = FakeNativeModule()
    backend = make_backend(native)
    backend.initialize(stage=None, device=torch.device("cpu"))
    workspace = backend.allocate_workspace(
        max_views=2,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert workspace["visible_depths"].shape == (16,)
    assert workspace["keys_in"].shape == (32,)
    assert workspace["sort_temp"].shape == (64,)
    assert (
        workspace["depth_bucket_tau"].untyped_storage().data_ptr()
        == workspace["sort_temp"].untyped_storage().data_ptr()
    )
    assert workspace["tile_starts"].dtype == torch.int32
    assert workspace["tile_ends"].dtype == torch.int32
    assert workspace["identity_env_xforms"].shape == (2, 4, 4)
    assert backend.workspace_bytes > 0
    assert sum(backend.workspace_bytes_by_tensor.values()) == (
        backend.workspace_bytes
    )
    assert backend.workspace_bytes_by_tensor["depth_bucket_tau"] == 0
    assert backend.workspace_storage_aliases == {
        "depth_bucket_tau": "sort_temp",
    }


def test_workspace_uses_mode_specific_sort_storage_query() -> None:
    fast_native = FakeNativeModule(
        pointer_sort_temp_bytes=128,
        double_buffer_sort_temp_bytes=32,
    )
    fast_backend = make_backend(fast_native)
    fast_backend.initialize(stage=None, device=torch.device("cpu"))
    fast_workspace = fast_backend.allocate_workspace(
        max_views=2,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert fast_workspace["sort_temp"].shape == (32,)
    assert fast_native.double_buffer_sort_temp_queries == [(32, 2)]
    assert fast_native.pointer_sort_temp_queries == []
    assert fast_backend.capacity_stats["sort_temp_strategy"] == (
        "double-buffer-global-radix"
    )
    assert fast_backend.capacity_stats["last_allocation"][
        "sort_temp_bytes"
    ] == 32
    assert fast_backend.capacity_stats["last_allocation"][
        "sort_key_begin_bit"
    ] == 0
    assert fast_backend.capacity_stats["last_allocation"][
        "sort_key_end_bit"
    ] == 33
    assert fast_backend.capacity_stats["last_allocation"][
        "sort_key_bits"
    ] == 33
    assert fast_backend.capacity_stats["last_allocation"][
        "sort_radix_digit_bits_assumption"
    ] == 8
    assert fast_backend.capacity_stats["last_allocation"][
        "sort_radix_passes_expected"
    ] == 5

    deterministic_native = FakeNativeModule(
        pointer_sort_temp_bytes=128,
        double_buffer_sort_temp_bytes=32,
    )
    deterministic_backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        deterministic=True,
        native_module=deterministic_native,
        allow_cpu_for_tests=True,
    )
    deterministic_backend.initialize(stage=None, device=torch.device("cpu"))
    deterministic_workspace = deterministic_backend.allocate_workspace(
        max_views=2,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert deterministic_workspace["sort_temp"].shape == (128,)
    assert deterministic_native.pointer_sort_temp_queries == [(32, 2)]
    assert deterministic_native.double_buffer_sort_temp_queries == []
    assert deterministic_backend.capacity_stats["sort_temp_strategy"] == (
        "pointer-api-global-segmented-scan"
    )
    assert deterministic_backend.capacity_stats["last_allocation"][
        "sort_key_end_bit"
    ] == 64
    assert deterministic_backend.capacity_stats["last_allocation"][
        "sort_radix_passes_expected"
    ] == 8


@pytest.mark.parametrize(
    ("global_tile_capacity", "expected_end_bit"),
    [
        (1, 32),
        (2, 33),
        (3, 34),
        (4, 34),
        (1 << 20, 52),
        ((1 << 20) + 1, 53),
        (2_147_483_647, 63),
    ],
)
def test_emitted_sort_key_end_bit_boundaries(
    global_tile_capacity: int,
    expected_end_bit: int,
) -> None:
    assert (
        _emitted_sort_key_end_bit(global_tile_capacity)
        == expected_end_bit
    )


@pytest.mark.parametrize("global_tile_capacity", [-1, 0, 2_147_483_648])
def test_emitted_sort_key_end_bit_rejects_invalid_capacity(
    global_tile_capacity: int,
) -> None:
    with pytest.raises(ValueError, match="tile capacity"):
        _emitted_sort_key_end_bit(global_tile_capacity)


def test_depth_bucket_tau_reuses_larger_phase_scratch_storage() -> None:
    native = FakeNativeModule()
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        tile_size=1,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    backend.initialize(stage=None, device=torch.device("cpu"))
    workspace = backend.allocate_workspace(
        max_views=2,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_depth_shape = (40, backend.depth_bucket_group_size)
    expected_shared_bytes = 40 * backend.depth_bucket_group_size * 4
    assert workspace["depth_bucket_tau"].shape == expected_depth_shape
    assert workspace["depth_bucket_tau"].dtype == torch.float32
    assert workspace["depth_bucket_tau"].is_contiguous()
    assert workspace["sort_temp"].shape == (expected_shared_bytes,)
    assert workspace["sort_temp"].dtype == torch.uint8
    assert workspace["sort_temp"].is_contiguous()
    assert (
        workspace["depth_bucket_tau"].untyped_storage().data_ptr()
        == workspace["sort_temp"].untyped_storage().data_ptr()
    )
    assert backend.workspace_bytes_by_tensor["sort_temp"] == (
        expected_shared_bytes
    )
    assert backend.workspace_bytes_by_tensor["depth_bucket_tau"] == 0
    assert backend.workspace_logical_bytes_by_tensor[
        "depth_bucket_tau"
    ] == expected_shared_bytes
    assert sum(backend.workspace_bytes_by_tensor.values()) == (
        backend.workspace_bytes
    )
    assert sum(backend.workspace_logical_bytes_by_tensor.values()) == (
        backend.workspace_bytes + expected_shared_bytes
    )


def test_tile_size_controls_workspace_and_native_dispatch() -> None:
    native = FakeNativeModule()
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        tile_size=1,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    assert backend.workspace["tile_starts"].shape == (20,)
    assert backend.workspace["depth_bucket_tau"].shape == (20, 64)
    assert backend.workspace["depth_bucket_counts"].shape == (4096,)
    assert backend.workspace["depth_bucket_offsets"].shape == (4097,)
    assert backend.workspace["depth_ordered_visible_indices"].shape == (16,)
    assert backend.workspace["visible_ray_precisions"].shape == (1, 6)
    assert backend.workspace["visible_ray_precision_means"].shape == (1, 4)
    assert native.last_tile_size == 1


def test_deterministic_mode_is_forwarded_to_native_renderer() -> None:
    native = FakeNativeModule()
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        deterministic=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    assert native.last_deterministic is True


def test_covariance_epsilon_is_forwarded_to_native_renderer() -> None:
    native = FakeNativeModule(expected_covariance_epsilon=0.0)
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        covariance_epsilon=0.0,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    assert native.last_covariance_epsilon == 0.0


def test_antialiased_rasterize_mode_is_validated_and_forwarded() -> None:
    native = FakeNativeModule(expected_antialiased=True)
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        rasterize_mode="antialiased",
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    assert native.last_antialiased is True
    assert backend.capacity_stats["parameters"]["rasterize_mode"] == (
        "antialiased"
    )
    assert backend.capacity_stats["parameters"]["covariance_epsilon"] == 0.3

    with pytest.raises(ValueError, match="rasterize_mode"):
        CustomCudaBackend(rasterize_mode="unsupported")  # type: ignore[arg-type]


def test_antialiased_mode_rejects_exact_ray_evaluation() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        CustomCudaBackend(
            rasterize_mode="antialiased",
            ray_gaussian_evaluation=True,
        )


def test_rasterize_mode_participates_in_projection_cache_token() -> None:
    native = FakeNativeModule()
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        enable_projection_cache=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    viewmats = torch.eye(4).unsqueeze(0).contiguous()
    intrinsics = torch.eye(3).unsqueeze(0).contiguous()
    scene_ids = torch.tensor([17], dtype=torch.int64)

    outputs = service.render(viewmats, intrinsics, scene_ids)
    service.render(viewmats, intrinsics, scene_ids, outputs=outputs)
    backend.rasterize_mode = "antialiased"
    assert backend._projection_cache_token is None
    assert backend._capacity_validation_token is None
    native.expected_antialiased = True
    service.render(viewmats, intrinsics, scene_ids, outputs=outputs)

    assert native.reuse_projection_flags == [False, True, False]

    with pytest.raises(ValueError, match="rasterize_mode"):
        backend.rasterize_mode = "antialaised"  # type: ignore[assignment]
    assert backend.rasterize_mode == "antialiased"

    backend.rasterize_mode = "classic"
    native.expected_antialiased = False
    service.render(viewmats, intrinsics, scene_ids, outputs=outputs)
    assert native.reuse_projection_flags == [False, True, False, False]


def test_runtime_antialias_mode_mutation_rejects_exact_ray_backend() -> None:
    backend = CustomCudaBackend(
        ray_gaussian_evaluation=True,
        native_module=FakeNativeModule(
            expected_ray_gaussian_evaluation=True,
        ),
        allow_cpu_for_tests=True,
    )

    with pytest.raises(ValueError, match="incompatible"):
        backend.rasterize_mode = "antialiased"
    assert backend.rasterize_mode == "classic"


def test_semantic_min_alpha_is_validated_and_forwarded() -> None:
    native = FakeNativeModule(expected_semantic_min_alpha=0.125)
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        semantic_min_alpha=0.125,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    assert native.last_semantic_min_alpha == 0.125

    for invalid in (-0.01, 1.01, float("nan")):
        try:
            CustomCudaBackend(semantic_min_alpha=invalid)
        except ValueError as error:
            assert "semantic_min_alpha" in str(error)
        else:
            raise AssertionError(
                f"semantic_min_alpha={invalid!r} should be rejected."
            )


def test_exact_ray_gaussian_evaluation_allocates_and_forwards_workspace() -> None:
    native = FakeNativeModule(expected_ray_gaussian_evaluation=True)
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        ray_gaussian_evaluation=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    assert native.last_ray_gaussian_evaluation is True
    assert backend.workspace["visible_ray_precisions"].shape == (16, 6)
    assert backend.workspace["visible_ray_precision_means"].shape == (16, 4)


def test_exact_ray_uses_32_pixel_macro_bins_for_16_pixel_raster_tiles() -> None:
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        ray_gaussian_evaluation=True,
        tile_size=16,
        native_module=FakeNativeModule(
            expected_ray_gaussian_evaluation=True,
        ),
        allow_cpu_for_tests=True,
    )
    backend.initialize(stage=None, device=torch.device("cpu"))
    workspace = backend.allocate_workspace(
        max_views=2,
        height=64,
        width=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert backend.bin_tile_size == 32
    assert workspace["tile_starts"].shape == (8,)
    assert workspace["tile_ends"].shape == (8,)
    assert "macrobin2" in backend.pipeline_name


def test_exact_ray_gaussian_evaluation_allocates_dense_pixel_workspace() -> None:
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        ray_gaussian_evaluation=True,
        tile_size=1,
        native_module=FakeNativeModule(
            expected_ray_gaussian_evaluation=True,
        ),
        allow_cpu_for_tests=True,
    )
    backend.initialize(stage=None, device=torch.device("cpu"))
    workspace = backend.allocate_workspace(
        max_views=1,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert workspace["depth_bucket_tau"].shape == (20, 64)
    assert workspace["depth_cutoff"].shape == (20,)
    assert workspace["visible_ray_precisions"].shape == (16, 6)
    assert workspace["visible_ray_precision_means"].shape == (16, 4)


def test_compact_projection_cache_uses_full_bucket_grid_and_small_auxiliary_buffers() -> None:
    native = FakeNativeModule()
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        tile_size=1,
        depth_bucket_count=128,
        compact_projection_cache=True,
        enable_projection_cache=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    backend.initialize(stage=None, device=torch.device("cpu"))
    workspace = backend.allocate_workspace(
        max_views=2,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert workspace["depth_bucket_tau"].shape == (40, 128)
    assert workspace["visible_depths"].shape == (1,)
    assert workspace["depth_bucket_counts"].shape == (1,)
    assert workspace["depth_bucket_offsets"].shape == (1,)
    assert workspace["depth_ordered_visible_indices"].shape == (1,)
    assert backend.visible_storage_capacity == 1
    assert backend.pipeline_name == (
        "compact-project-tau-cutoff-direct-intersections-"
        "emitted-count-sync-double-buffer-global-radix-linear-ranges-"
        "fused-reproject-rgba-depth-semantic"
    )
    assert "sorted Gaussian-ID intersections" in (
        backend.projection_cache_scope
    )
    assert "output" in backend.projection_cache_scope


def test_compact_projected_records_materialize_only_when_requested() -> None:
    native = FakeNativeModule()
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        tile_size=1,
        depth_bucket_count=128,
        compact_projection_cache=True,
        materialize_projected_records=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    backend.initialize(stage=None, device=torch.device("cpu"))
    workspace = backend.allocate_workspace(
        max_views=2,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert workspace["visible_means2d"].shape == (16, 2)
    assert workspace["visible_conics"].shape == (16, 3)
    assert workspace["visible_depths"].shape == (16,)
    assert workspace["visible_camera_ids"].shape == (16,)
    assert workspace["visible_radii"].shape == (16, 2)
    assert backend.visible_storage_capacity == 16
    assert backend.materialize_projected_records_active is True
    assert backend.capacity_stats["projected_record_reuse"] == {
        "requested": True,
        "active": True,
        "fallback_reason": None,
        "bytes_per_record": 44,
    }
    assert "project-once-candidate" in backend.pipeline_name
    assert "first-pass screen-space candidates" in (
        backend.projection_cache_scope
    )


def test_compact_projected_records_fall_back_at_workspace_ceiling() -> None:
    direct = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        tile_size=1,
        depth_bucket_count=128,
        compact_projection_cache=True,
        native_module=FakeNativeModule(),
        allow_cpu_for_tests=True,
    )
    direct.initialize(stage=None, device=torch.device("cpu"))
    direct.allocate_workspace(
        max_views=2,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        tile_size=1,
        depth_bucket_count=128,
        compact_projection_cache=True,
        materialize_projected_records=True,
        max_workspace_bytes=direct.workspace_bytes,
        native_module=FakeNativeModule(),
        allow_cpu_for_tests=True,
    )
    backend.initialize(stage=None, device=torch.device("cpu"))
    workspace = backend.allocate_workspace(
        max_views=2,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert workspace["visible_depths"].shape == (1,)
    assert backend.workspace_bytes == direct.workspace_bytes
    assert backend.materialize_projected_records_active is False
    assert backend.capacity_stats["projected_record_reuse"] == {
        "requested": True,
        "active": False,
        "fallback_reason": "workspace-memory-ceiling",
        "bytes_per_record": 44,
    }
    allocation = backend.capacity_stats["last_allocation"]
    assert allocation["projected_record_reuse"]["fallback_reason"] == (
        "workspace-memory-ceiling"
    )
    assert (
        allocation["projected_record_reuse"][
            "materialized_workspace_bytes"
        ]
        > direct.workspace_bytes
    )


def test_compact_projected_records_are_forwarded_to_native() -> None:
    native = FakeNativeModule(
        expected_compact_projection_cache=True,
        expected_materialize_projected_records=True,
    )
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        tile_size=1,
        compact_projection_cache=True,
        materialize_projected_records=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=2,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))

    service.render(
        torch.eye(4).repeat(2, 1, 1).contiguous(),
        torch.eye(3).repeat(2, 1, 1).contiguous(),
        torch.tensor([17, 17], dtype=torch.int64),
    )

    assert native.render_calls == 1


def test_materialized_projected_records_require_compact_path() -> None:
    with pytest.raises(
        ValueError,
        match="materialize_projected_records requires compact_projection_cache",
    ):
        CustomCudaBackend(materialize_projected_records=True)


def test_projected_record_growth_falls_back_without_failing_render() -> None:
    native = CapacitySequenceNative(
        [
            {
                "visible_gaussians": 3,
                "tile_intersections": 5,
                "visible_overflow": 1,
                "intersection_overflow": 0,
                "active_tiles": 2,
            },
            {
                "visible_gaussians": 3,
                "tile_intersections": 5,
                "visible_overflow": 0,
                "intersection_overflow": 0,
                "active_tiles": 2,
            },
        ],
        expected_compact_projection_cache=True,
        expected_materialize_projected_records=None,
    )
    backend = CustomCudaBackend(
        max_visible_records=2,
        max_intersections=32,
        tile_size=1,
        compact_projection_cache=True,
        materialize_projected_records=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=2,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    initial_workspace_bytes = backend.workspace_bytes
    backend.max_workspace_bytes = initial_workspace_bytes
    service.load_scene(17, **scene_tensors(3, 4))

    service.render(
        torch.eye(4).repeat(2, 1, 1).contiguous(),
        torch.eye(3).repeat(2, 1, 1).contiguous(),
        torch.tensor([17, 17], dtype=torch.int64),
    )

    assert native.materialize_projected_records_flags == [True, False]
    assert native.capacities == [(2, 32), (1, 33)]
    assert backend.materialize_projected_records_active is False
    assert backend.capacity_stats["last_render"]["retry_count"] == 1
    assert backend.capacity_stats["projected_record_reuse"][
        "fallback_reason"
    ] == "workspace-memory-ceiling"


def test_pipeline_metadata_discloses_intersection_ordering_sync() -> None:
    fast_backend = CustomCudaBackend(allow_cpu_for_tests=True)
    deterministic_backend = CustomCudaBackend(
        deterministic=True,
        allow_cpu_for_tests=True,
    )

    assert "emitted-count-sync-double-buffer-global-radix" in (
        fast_backend.pipeline_name
    )
    assert "synchronize the current CUDA stream" in (
        fast_backend.capacity_stats["intersection_ordering_synchronization"]
    )
    assert "device-tile-segmented-radix" in (
        deterministic_backend.pipeline_name
    )
    assert "device-only" in (
        deterministic_backend.capacity_stats[
            "intersection_ordering_synchronization"
        ]
    )


def test_compact_projection_cache_rejects_unsupported_modes() -> None:
    for kwargs in (
        {"tile_size": 16},
        {"tile_size": 1, "ray_gaussian_evaluation": True},
        {"tile_size": 1, "deterministic": True},
    ):
        try:
            CustomCudaBackend(
                compact_projection_cache=True,
                **kwargs,
            )
        except ValueError as error:
            assert "compact_projection_cache" in str(error)
        else:
            raise AssertionError(f"Expected rejection for {kwargs!r}")


def test_workspace_rejects_capacities_beyond_native_int32_limit() -> None:
    backend = CustomCudaBackend(
        max_visible_records=1,
        max_intersections=2_147_483_648,
        compact_projection_cache=True,
        tile_size=1,
        native_module=FakeNativeModule(),
        allow_cpu_for_tests=True,
    )
    backend.initialize(stage=None, device=torch.device("cpu"))
    try:
        backend.allocate_workspace(
            max_views=1,
            height=1,
            width=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    except ValueError as error:
        assert "int32 implementation limit" in str(error)
    else:
        raise AssertionError("Expected int32 capacity rejection.")


def test_projection_cache_reuses_only_coherent_inputs() -> None:
    native = FakeNativeModule()
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        enable_projection_cache=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    scene = scene_tensors(3, 4)
    service.load_scene(17, **scene)
    viewmats = torch.eye(4).unsqueeze(0).contiguous()
    intrinsics = torch.eye(3).unsqueeze(0).contiguous()
    scene_ids = torch.tensor([17], dtype=torch.int64)

    outputs = service.render(viewmats, intrinsics, scene_ids)
    service.render(
        viewmats,
        intrinsics,
        scene_ids,
        outputs=outputs,
    )
    viewmats = viewmats.clone()
    intrinsics = intrinsics.clone()
    scene_ids = scene_ids.clone()
    service.render(
        viewmats,
        intrinsics,
        scene_ids,
        outputs=outputs,
    )
    scene["features"].mul_(0.5)
    service.render(
        viewmats,
        intrinsics,
        scene_ids,
        outputs=outputs,
    )
    viewmats[0, 0, 3] = 1.0
    service.render(
        viewmats,
        intrinsics,
        scene_ids,
        outputs=outputs,
    )
    backend.invalidate_projection_cache()
    service.render(
        viewmats,
        intrinsics,
        scene_ids,
        outputs=outputs,
    )

    assert native.reuse_projection_flags == [
        False,
        True,
        False,
        True,
        False,
        False,
    ]
    assert backend.projection_cache_stats == {
        "enabled": True,
        "hits": 2,
        "misses": 4,
    }


def test_inference_tensor_inputs_fail_closed_for_cache_and_capacity() -> None:
    native = FakeNativeModule()
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        enable_projection_cache=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    with torch.inference_mode():
        viewmats = torch.eye(4).unsqueeze(0).contiguous()
        intrinsics = torch.eye(3).unsqueeze(0).contiguous()
        scene_ids = torch.tensor([17], dtype=torch.int64)

    outputs = service.render(viewmats, intrinsics, scene_ids)
    service.render(viewmats, intrinsics, scene_ids, outputs=outputs)
    with torch.inference_mode():
        viewmats[0, 0, 3] = 1.0
    service.render(viewmats, intrinsics, scene_ids, outputs=outputs)

    assert native.reuse_projection_flags == [False, False, False]
    assert backend.projection_cache_stats == {
        "enabled": True,
        "hits": 0,
        "misses": 3,
    }
    assert backend.capacity_stats["totals"] == {
        "device_validations": 3,
        "validated_input_reuses": 0,
        "growth_events": 0,
        "retries": 0,
    }


def test_double_buffer_selector_normalizes_cached_sorted_storage() -> None:
    native = FakeNativeModule(sorted_buffer_selectors=[0, 1, 0, 1])
    backend = CustomCudaBackend(
        max_visible_records=16,
        max_intersections=32,
        enable_projection_cache=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    viewmats = torch.eye(4).unsqueeze(0).contiguous()
    intrinsics = torch.eye(3).unsqueeze(0).contiguous()
    scene_ids = torch.tensor([17], dtype=torch.int64)
    original_pointers = {
        name: int(backend.workspace[name].data_ptr())
        for name in ("keys_in", "keys_out", "values_in", "values_out")
    }

    outputs = service.render(viewmats, intrinsics, scene_ids)

    normalized_pointers = {
        name: int(backend.workspace[name].data_ptr())
        for name in original_pointers
    }
    assert normalized_pointers == {
        "keys_in": original_pointers["keys_out"],
        "keys_out": original_pointers["keys_in"],
        "values_in": original_pointers["values_out"],
        "values_out": original_pointers["values_in"],
    }

    service.render(viewmats, intrinsics, scene_ids, outputs=outputs)

    assert {
        name: int(backend.workspace[name].data_ptr())
        for name in original_pointers
    } == normalized_pointers

    viewmats[0, 0, 3] = 0.25
    service.render(viewmats, intrinsics, scene_ids, outputs=outputs)
    assert {
        name: int(backend.workspace[name].data_ptr())
        for name in original_pointers
    } == original_pointers

    service.render(viewmats, intrinsics, scene_ids, outputs=outputs)
    assert {
        name: int(backend.workspace[name].data_ptr())
        for name in original_pointers
    } == original_pointers
    assert native.reuse_projection_flags == [False, True, False, True]


def test_adaptive_capacity_grows_retries_and_reuses_validated_state() -> None:
    native = CapacitySequenceNative(
        [
            {
                "visible_gaussians": 12,
                "tile_intersections": 40,
                "visible_overflow": 8,
                "intersection_overflow": 32,
                "active_tiles": 2,
            },
            {
                "visible_gaussians": 12,
                "tile_intersections": 40,
                "visible_overflow": 0,
                "intersection_overflow": 0,
                "active_tiles": 2,
            },
        ]
    )
    backend = CustomCudaBackend(
        max_visible_records=4,
        max_intersections=8,
        enable_projection_cache=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    workspace_identity = id(service.workspace)
    viewmats = torch.eye(4).unsqueeze(0).contiguous()
    intrinsics = torch.eye(3).unsqueeze(0).contiguous()
    scene_ids = torch.tensor([17], dtype=torch.int64)

    outputs = service.render(viewmats, intrinsics, scene_ids)

    assert native.render_calls == 2
    assert native.capacities == [(4, 8), (15, 150)]
    assert native.output_pointers[0] == native.output_pointers[1]
    assert id(service.workspace) == workspace_identity
    assert service.workspace is backend.workspace
    assert torch.all(outputs["rgb"] == 0.2)
    stats = backend.capacity_stats
    assert stats["initial_capacity"] == {
        "visible_records": 4,
        "tile_intersections": 8,
    }
    assert stats["current_capacity"]["visible_records"] == 15
    assert stats["current_capacity"]["tile_intersections"] == 150
    assert stats["totals"] == {
        "device_validations": 1,
        "validated_input_reuses": 0,
        "growth_events": 1,
        "retries": 1,
    }
    assert stats["last_render"]["retry_count"] == 1
    assert stats["last_growth"]["retry_count"] == 1
    assert stats["last_allocation"]["allocation_strategy"] == (
        "release-old-workspace-first"
    )
    assert stats["last_allocation"]["transient_logical_bytes"] == (
        stats["last_allocation"]["requested_workspace_bytes"]
    )
    assert stats["last_allocation"]["workspace_storage_aliases"] == {
        "depth_bucket_tau": "sort_temp",
    }
    assert stats["last_allocation"][
        "installed_workspace_storage_aliases"
    ] == {
        "depth_bucket_tau": "sort_temp",
    }
    assert stats["last_allocation"]["workspace_alias_savings_bytes"] > 0
    assert stats["last_allocation"][
        "requested_workspace_unaliased_bytes"
    ] == (
        stats["last_allocation"]["requested_workspace_bytes"]
        + stats["last_allocation"]["workspace_alias_savings_bytes"]
    )
    assert len(stats["last_growth"]["attempts"]) == 2
    assert stats["last_render"]["final_counters"] == {
        "visible_gaussians": 12,
        "tile_intersections": 40,
        "visible_overflow": 0,
        "intersection_overflow": 0,
        "active_tiles": 2,
    }
    json.dumps(stats)

    reused = service.render(
        viewmats,
        intrinsics,
        scene_ids,
        outputs=outputs,
    )

    assert reused is outputs
    assert native.render_calls == 3
    assert native.capacities[-1] == (15, 150)
    assert native.output_pointers[2] == native.output_pointers[0]
    assert backend.capacity_stats["last_render"]["validation"] == (
        "validated-identical-input"
    )
    assert backend.capacity_stats["last_growth"]["retry_count"] == 1
    assert backend.capacity_stats["totals"]["validated_input_reuses"] == 1
    assert native.reuse_projection_flags == [False, False, True]

    viewmats[0, 0, 3] = 1.0
    service.render(viewmats, intrinsics, scene_ids, outputs=outputs)

    assert native.render_calls == 4
    assert backend.capacity_stats["last_render"]["validation"] == (
        "device-counters"
    )
    assert backend.capacity_stats["totals"]["device_validations"] == 2
    assert backend.capacity_stats["totals"]["growth_events"] == 1
    assert native.reuse_projection_flags[-1] is False


def test_compact_candidate_growth_extrapolates_intersection_count() -> None:
    native = CapacitySequenceNative(
        [
            {
                "visible_gaussians": 12,
                "tile_intersections": 40,
                "visible_overflow": 8,
                "intersection_overflow": 32,
                "active_tiles": 2,
            },
            {
                "visible_gaussians": 12,
                "tile_intersections": 40,
                "visible_overflow": 0,
                "intersection_overflow": 0,
                "active_tiles": 2,
            },
        ],
        expected_compact_projection_cache=True,
        expected_materialize_projected_records=True,
    )
    backend = CustomCudaBackend(
        max_visible_records=4,
        max_intersections=8,
        tile_size=1,
        compact_projection_cache=True,
        materialize_projected_records=True,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))

    service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    # Only four of twelve first-pass candidates fit, so forty intersections
    # are an incomplete sample. Extrapolate to 120, then apply 1.25x headroom.
    assert native.capacities == [(4, 8), (15, 150)]


def test_default_capacity_growth_does_not_double_a_near_full_workspace() -> None:
    native = CapacitySequenceNative(
        [
            {
                "visible_gaussians": 80,
                "tile_intersections": 101,
                "visible_overflow": 0,
                "intersection_overflow": 1,
                "active_tiles": 2,
            },
            {
                "visible_gaussians": 80,
                "tile_intersections": 101,
                "visible_overflow": 0,
                "intersection_overflow": 0,
                "active_tiles": 2,
            },
        ]
    )
    backend = CustomCudaBackend(
        max_visible_records=100,
        max_intersections=100,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))

    service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    assert native.capacities == [(100, 100), (100, 127)]
    stats = backend.capacity_stats
    assert stats["parameters"]["growth_factor"] == 1.25
    assert stats["last_growth"]["attempts"][0]["growth_request"] == {
        "visible_records": 100,
        "tile_intersections": 127,
    }


def test_workspace_ceiling_relaxes_headroom_before_failing() -> None:
    probe = CustomCudaBackend(
        max_visible_records=100,
        max_intersections=100,
        native_module=FakeNativeModule(),
        allow_cpu_for_tests=True,
    )
    probe.initialize(stage=None, device=torch.device("cpu"))
    probe.allocate_workspace(
        max_views=1,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    exact_demand_bytes = int(
        probe._workspace_plan(
            visible_capacity=100,
            intersection_capacity=101,
        )["workspace_bytes"]
    )
    probe.shutdown()

    native = CapacitySequenceNative(
        [
            {
                "visible_gaussians": 80,
                "tile_intersections": 101,
                "visible_overflow": 0,
                "intersection_overflow": 1,
                "active_tiles": 2,
            },
            {
                "visible_gaussians": 80,
                "tile_intersections": 101,
                "visible_overflow": 0,
                "intersection_overflow": 0,
                "active_tiles": 2,
            },
        ]
    )
    backend = CustomCudaBackend(
        max_visible_records=100,
        max_intersections=100,
        max_workspace_bytes=exact_demand_bytes,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))

    service.render(
        torch.eye(4).unsqueeze(0).contiguous(),
        torch.eye(3).unsqueeze(0).contiguous(),
        torch.tensor([17], dtype=torch.int64),
    )

    assert native.capacities == [(100, 100), (100, 101)]
    adjustment = backend.capacity_stats["last_growth"]["attempts"][0][
        "workspace_ceiling_adjustment"
    ]
    assert adjustment["policy"] == "relax-headroom-to-observed-demand"
    assert adjustment["status"] == "headroom-relaxed"
    assert adjustment["target_capacity"] == {
        "visible_records": 100,
        "tile_intersections": 127,
    }
    assert adjustment["minimum_observed_demand_capacity"] == {
        "visible_records": 100,
        "tile_intersections": 101,
    }
    assert adjustment["minimum_observed_demand_workspace_bytes"] == (
        exact_demand_bytes
    )


def test_adaptive_capacity_memory_ceiling_fails_closed() -> None:
    probe = CustomCudaBackend(
        max_visible_records=4,
        max_intersections=8,
        native_module=FakeNativeModule(),
        allow_cpu_for_tests=True,
    )
    probe.initialize(stage=None, device=torch.device("cpu"))
    probe.allocate_workspace(
        max_views=1,
        height=4,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    initial_workspace_bytes = probe.workspace_bytes
    probe.shutdown()

    native = CapacitySequenceNative(
        [
            {
                "visible_gaussians": 12,
                "tile_intersections": 40,
                "visible_overflow": 8,
                "intersection_overflow": 32,
                "active_tiles": 2,
            }
        ]
    )
    backend = CustomCudaBackend(
        max_visible_records=4,
        max_intersections=8,
        max_workspace_bytes=initial_workspace_bytes,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    outputs = {
        "rgb": torch.zeros((1, 4, 5, 3), dtype=torch.float32),
        "depth": torch.zeros((1, 4, 5, 1), dtype=torch.float32),
        "alpha": torch.zeros((1, 4, 5, 1), dtype=torch.float32),
        "semantic_id": torch.zeros((1, 4, 5, 1), dtype=torch.int64),
    }

    with pytest.raises(RendererCapacityError) as raised:
        service.render(
            torch.eye(4).unsqueeze(0).contiguous(),
            torch.eye(3).unsqueeze(0).contiguous(),
            torch.tensor([17], dtype=torch.int64),
            outputs=outputs,
        )

    assert native.render_calls == 1
    assert backend.current_visible_capacity == 4
    assert backend.current_intersection_capacity == 8
    assert backend.workspace_bytes == initial_workspace_bytes
    assert torch.isnan(outputs["rgb"]).all()
    assert torch.isnan(outputs["depth"]).all()
    assert torch.isnan(outputs["alpha"]).all()
    assert torch.all(outputs["semantic_id"] == -1)
    telemetry = raised.value.telemetry
    assert telemetry["reason"] == "growth-allocation-failed"
    growth_failure = telemetry["capacity"]["last_render"]["attempts"][0][
        "growth_failure"
    ]
    assert growth_failure["failure"] == "workspace-memory-limit"
    assert growth_failure["max_workspace_bytes"] == initial_workspace_bytes
    assert backend.capacity_stats["totals"]["growth_events"] == 0
    json.dumps(telemetry)


def test_allocator_failure_restores_released_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = CapacitySequenceNative(
        [
            {
                "visible_gaussians": 12,
                "tile_intersections": 40,
                "visible_overflow": 8,
                "intersection_overflow": 32,
                "active_tiles": 2,
            }
        ]
    )
    backend = CustomCudaBackend(
        max_visible_records=4,
        max_intersections=8,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))
    workspace_identity = id(service.workspace)
    initial_workspace_bytes = backend.workspace_bytes
    real_empty = torch.empty
    failed = False

    def fail_new_intersection_workspace(
        shape: tuple[int, ...],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        nonlocal failed
        if shape == (150,) and not failed:
            failed = True
            raise torch.OutOfMemoryError("injected allocation failure")
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_new_intersection_workspace)

    with pytest.raises(RendererCapacityError) as raised:
        service.render(
            torch.eye(4).unsqueeze(0).contiguous(),
            torch.eye(3).unsqueeze(0).contiguous(),
            torch.tensor([17], dtype=torch.int64),
        )

    assert failed is True
    assert id(service.workspace) == workspace_identity
    assert service.workspace is backend.workspace
    assert backend.current_visible_capacity == 4
    assert backend.current_intersection_capacity == 8
    assert backend.workspace_bytes == initial_workspace_bytes
    allocation = raised.value.telemetry["capacity"]["last_allocation"]
    assert allocation["allocation_strategy"] == (
        "release-old-workspace-first"
    )
    assert allocation["failure"] == "allocator-out-of-memory"
    assert allocation["restoration"] == {
        "status": "restored-old-capacity",
        "workspace_usable": True,
        "workspace_bytes": initial_workspace_bytes,
    }


def test_adaptive_capacity_retry_limit_is_bounded_and_fails_closed() -> None:
    native = CapacitySequenceNative(
        [
            {
                "visible_gaussians": 12,
                "tile_intersections": 40,
                "visible_overflow": 8,
                "intersection_overflow": 32,
                "active_tiles": 2,
            },
            {
                "visible_gaussians": 20,
                "tile_intersections": 200,
                "visible_overflow": 5,
                "intersection_overflow": 50,
                "active_tiles": 2,
            },
        ]
    )
    backend = CustomCudaBackend(
        max_visible_records=4,
        max_intersections=8,
        max_capacity_retries=1,
        native_module=native,
        allow_cpu_for_tests=True,
    )
    service = RendererService(
        backend,
        height=4,
        width=5,
        max_views=1,
        allow_cpu_for_tests=True,
    )
    service.initialize(stage=None, device="cpu")
    service.load_scene(17, **scene_tensors(3, 4))

    with pytest.raises(RendererCapacityError) as raised:
        service.render(
            torch.eye(4).unsqueeze(0).contiguous(),
            torch.eye(3).unsqueeze(0).contiguous(),
            torch.tensor([17], dtype=torch.int64),
        )

    assert native.render_calls == 2
    assert raised.value.telemetry["reason"] == "retry-limit"
    stats = backend.capacity_stats
    assert stats["last_render"]["status"] == "failure"
    assert stats["last_render"]["retry_count"] == 1
    assert stats["totals"]["growth_events"] == 1
    assert stats["totals"]["retries"] == 1


def test_adaptive_capacity_parameters_are_strictly_validated() -> None:
    for kwargs in (
        {"capacity_growth_factor": 1.0},
        {"capacity_growth_factor": float("nan")},
        {"capacity_headroom": 0.99},
        {"max_capacity_retries": -1},
        {"max_capacity_retries": 1.5},
        {"max_workspace_bytes": 0},
    ):
        with pytest.raises(ValueError):
            CustomCudaBackend(**kwargs)
