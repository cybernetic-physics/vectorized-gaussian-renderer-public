import numpy as np
import pytest
import torch

from isaacsim_gaussian_renderer.semantic_mapping import (
    parse_numeric_class_label,
    remap_renderer_semantics,
    renderer_semantic_lut,
    semantic_group_counts,
    semantic_label_grouping,
    semantic_group_permutation,
)


def test_semantic_label_grouping_handles_arbitrary_labels_stably() -> None:
    semantic_ids = torch.tensor([2, 0, 1, 2, 1, 0], dtype=torch.int64)

    counts, permutation = semantic_label_grouping(semantic_ids)

    assert counts == [2, 2, 2]
    torch.testing.assert_close(
        permutation,
        torch.tensor([1, 5, 2, 4, 0, 3]),
    )
    assert [
        part.tolist()
        for part in semantic_ids.index_select(0, permutation).split(counts)
    ] == [[0, 0], [1, 1], [2, 2]]


def test_semantic_label_grouping_rejects_empty_groups() -> None:
    with pytest.raises(ValueError, match="Every semantic group"):
        semantic_label_grouping(
            torch.tensor([0, 2], dtype=torch.int32),
            group_count=3,
        )


def test_semantic_group_permutation_is_stable_and_label_major() -> None:
    counts = semantic_group_counts(10, 4)
    permutation = semantic_group_permutation(10, 4, device="cpu")

    assert counts == [3, 3, 2, 2]
    torch.testing.assert_close(
        permutation,
        torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, 3, 7]),
    )
    labels = torch.arange(10).remainder(4).index_select(0, permutation)
    assert [part.tolist() for part in labels.split(counts)] == [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2],
        [3, 3],
    ]


def test_semantic_group_permutation_handles_more_groups_than_values() -> None:
    assert semantic_group_counts(3, 5) == [1, 1, 1, 0, 0]
    torch.testing.assert_close(
        semantic_group_permutation(3, 5, device="cpu"),
        torch.tensor([0, 1, 2]),
    )


def test_renderer_semantic_lut_parses_ovrtx_labels_and_background() -> None:
    lut = renderer_semantic_lut(
        {
            2: "class: 3;",
            3: "class: 2;",
            4: "class: 1;",
            5: "class: 0;",
        },
        expected_group_count=4,
    )
    assert lut.tolist() == [-1, -1, 3, 2, 1, 0]
    remapped = remap_renderer_semantics(
        np.array([[[0], [2], [5]]], dtype=np.uint32),
        lut,
    )
    assert remapped.tolist() == [[[-1], [3], [0]]]


def test_numeric_class_label_can_coexist_with_other_semantics() -> None:
    assert parse_numeric_class_label("label: left; class: 17;") == 17


def test_renderer_semantic_lut_rejects_missing_classes() -> None:
    with pytest.raises(ValueError, match="missing classes"):
        renderer_semantic_lut({2: "class: 0;"}, expected_group_count=2)
