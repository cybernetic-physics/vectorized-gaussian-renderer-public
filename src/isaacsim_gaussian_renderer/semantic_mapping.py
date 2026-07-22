"""Shared semantic grouping and OVRTX identifier-map helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


BACKGROUND_SEMANTIC_ID = -1


def semantic_label_grouping(
    semantic_ids: torch.Tensor,
    *,
    group_count: int | None = None,
) -> tuple[list[int], torch.Tensor]:
    """Return label-major counts and a stable permutation for arbitrary IDs."""
    if semantic_ids.ndim != 1:
        raise ValueError("semantic_ids must be one-dimensional.")
    if semantic_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("semantic_ids must use an integer dtype.")
    if group_count is not None and group_count <= 0:
        raise ValueError("group_count must be positive.")
    if semantic_ids.numel() == 0:
        if group_count is None:
            raise ValueError(
                "group_count is required when semantic_ids is empty."
            )
        return (
            [0] * group_count,
            torch.empty(
                (0,),
                device=semantic_ids.device,
                dtype=torch.int64,
            ),
        )

    minimum = int(semantic_ids.min().item())
    maximum = int(semantic_ids.max().item())
    if minimum < 0:
        raise ValueError("semantic_ids must be non-negative.")
    resolved_group_count = (
        maximum + 1
        if group_count is None
        else group_count
    )
    if maximum >= resolved_group_count:
        raise ValueError(
            "semantic_ids contain a value outside group_count."
        )

    labels = semantic_ids.to(torch.int64)
    counts_tensor = torch.bincount(
        labels,
        minlength=resolved_group_count,
    )
    counts = [int(value) for value in counts_tensor.cpu().tolist()]
    if any(count == 0 for count in counts):
        raise ValueError(
            "Every semantic group must contain at least one Gaussian."
        )
    permutation = torch.argsort(labels, stable=True)
    return counts, permutation.contiguous()


def semantic_group_counts(gaussian_count: int, group_count: int) -> list[int]:
    """Return counts for ``semantic_id[i] = i % group_count`` in label order."""
    if gaussian_count < 0:
        raise ValueError("gaussian_count must be non-negative.")
    if group_count <= 0:
        raise ValueError("group_count must be positive.")
    quotient, remainder = divmod(gaussian_count, group_count)
    return [
        quotient + (1 if semantic_id < remainder else 0)
        for semantic_id in range(group_count)
    ]


def semantic_group_permutation(
    gaussian_count: int,
    group_count: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the stable label-major permutation without sorting semantic IDs."""
    counts = semantic_group_counts(gaussian_count, group_count)
    if gaussian_count == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)

    quotient, remainder = divmod(gaussian_count, group_count)
    positions = torch.arange(gaussian_count, dtype=torch.int64, device=device)
    first_region = remainder * (quotient + 1)

    if quotient == 0:
        labels = positions
        local_indices = torch.zeros_like(positions)
    else:
        in_first_region = positions < first_region
        labels = torch.where(
            in_first_region,
            positions.div(quotient + 1, rounding_mode="floor"),
            remainder
            + (positions - first_region).div(quotient, rounding_mode="floor"),
        )
        local_indices = torch.where(
            in_first_region,
            positions.remainder(quotient + 1),
            (positions - first_region).remainder(quotient),
        )

    permutation = labels + local_indices * group_count
    if sum(counts) != gaussian_count:
        raise AssertionError("Semantic group counts do not cover the scene.")
    return permutation.contiguous()


def decode_semantic_id_map(tensor: np.ndarray) -> dict[int, str]:
    """Decode OVRTX's packed ``SemanticIdMap`` RenderVar."""
    data = np.ascontiguousarray(tensor).view(np.uint8).reshape(-1)
    if data.size < 4:
        return {}
    entry_dtype = np.dtype(
        [("id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")]
    )
    count = int.from_bytes(data[-4:].tobytes(), byteorder="little")
    table_bytes = count * entry_dtype.itemsize
    if table_bytes > data.size - 4:
        raise ValueError("SemanticIdMap entry table exceeds the mapped buffer.")
    entries = data[:table_bytes].view(entry_dtype).reshape(count)
    result: dict[int, str] = {}
    for entry in entries:
        offset = int(entry["label_offset"])
        length = int(entry["label_length"])
        if offset < table_bytes or offset + length > data.size - 4:
            raise ValueError("SemanticIdMap label points outside the mapped buffer.")
        label = (
            data[offset : offset + length]
            .tobytes()
            .decode("utf-8")
            .rstrip("\x00")
            .rstrip()
        )
        result[int(entry["id"][0])] = label
    return result


def parse_numeric_class_label(label: str) -> int:
    """Extract an integer from an OVRTX label such as ``class: 17;``."""
    for segment in label.split(";"):
        key, separator, value = segment.partition(":")
        if separator and key.strip() == "class":
            return int(value.strip())
    raise ValueError(f"Semantic label has no numeric class field: {label!r}")


def renderer_semantic_lut(
    id_map: Mapping[int, str],
    *,
    expected_group_count: int,
    background_semantic_id: int = BACKGROUND_SEMANTIC_ID,
) -> np.ndarray:
    """Build a dense renderer-ID to requested semantic-ID lookup table."""
    if expected_group_count <= 0:
        raise ValueError("expected_group_count must be positive.")
    if not id_map:
        raise ValueError("SemanticIdMap is empty.")

    max_renderer_id = max(id_map)
    lut = np.full(
        (max_renderer_id + 1,),
        background_semantic_id,
        dtype=np.int64,
    )
    found: set[int] = set()
    for renderer_id, label in id_map.items():
        semantic_id = parse_numeric_class_label(label)
        if semantic_id < 0 or semantic_id >= expected_group_count:
            raise ValueError(
                f"Semantic class {semantic_id} is outside "
                f"[0, {expected_group_count})."
            )
        if semantic_id in found:
            raise ValueError(f"Duplicate semantic class in SemanticIdMap: {semantic_id}")
        lut[renderer_id] = semantic_id
        found.add(semantic_id)

    expected = set(range(expected_group_count))
    missing = sorted(expected - found)
    if missing:
        preview = ", ".join(str(value) for value in missing[:16])
        suffix = "..." if len(missing) > 16 else ""
        raise ValueError(f"SemanticIdMap is missing classes: {preview}{suffix}")
    return lut


def remap_renderer_semantics(
    renderer_ids: np.ndarray,
    lut: np.ndarray,
    *,
    background_semantic_id: int = BACKGROUND_SEMANTIC_ID,
) -> np.ndarray:
    """Apply a dense OVRTX semantic lookup table on CPU reference output."""
    renderer_ids = np.asarray(renderer_ids)
    result = np.full(renderer_ids.shape, background_semantic_id, dtype=np.int64)
    valid = renderer_ids < lut.size
    result[valid] = lut[renderer_ids[valid]]
    return result
