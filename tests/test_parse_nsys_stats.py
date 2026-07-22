import pytest

from scripts.parse_nsys_stats import classify_kernel


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("count_intersections_by_tile_kernel", "tile_grouping"),
        ("initialize_tile_scatter_kernel", "tile_grouping"),
        ("scatter_intersections_by_tile_kernel", "tile_grouping"),
        ("mark_sorted_tile_ranges_kernel", "tile_grouping"),
        (
            "emit_cached_projected_intersections_kernel",
            "projected_intersection_emission",
        ),
        ("cub::DeviceScanKernel", "offsets"),
        ("cub::DeviceSegmentedRadixSortKernel", "sorting"),
        ("cub::DeviceRadixSortKernel", "sorting"),
    ),
)
def test_custom_segmented_sort_kernels_have_explicit_stages(
    name: str,
    expected: str,
) -> None:
    assert classify_kernel(name) == expected
