from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.run_logical_batch import (
    FULL_OUTPUTS,
    GSPLAT_CORE_OUTPUTS,
    combine_output_audit,
    output_summary,
)


def render_outputs(*, include_semantic: bool) -> dict[str, torch.Tensor]:
    outputs = {
        "rgb": torch.zeros((2, 2, 2, 3), dtype=torch.float32),
        "depth": torch.ones((2, 2, 2, 1), dtype=torch.float32),
        "alpha": torch.ones((2, 2, 2, 1), dtype=torch.float32),
    }
    if include_semantic:
        outputs["semantic_id"] = torch.full(
            (2, 2, 2, 1),
            3,
            dtype=torch.int64,
        )
    return outputs


def test_core_output_audit_does_not_pretend_to_validate_semantics() -> None:
    summary = output_summary(
        render_outputs(include_semantic=False),
        required_outputs=GSPLAT_CORE_OUTPUTS,
    )
    audit = combine_output_audit(
        [summary],
        np.asarray([4, 9], dtype=np.int64),
        required_outputs=GSPLAT_CORE_OUTPUTS,
    )

    assert audit["pass"] is True
    assert audit["required_outputs"] == ["rgb", "depth", "alpha"]
    assert "foreground_semantic_valid" not in audit["failed_camera_counts"]
    assert "background_semantic_valid" not in audit["failed_camera_counts"]


def test_full_output_audit_still_requires_semantics() -> None:
    summary = output_summary(
        render_outputs(include_semantic=True),
        required_outputs=FULL_OUTPUTS,
    )
    audit = combine_output_audit(
        [summary],
        np.asarray([4, 9], dtype=np.int64),
        required_outputs=FULL_OUTPUTS,
    )

    assert audit["pass"] is True
    assert audit["required_outputs"] == [
        "rgb",
        "depth",
        "alpha",
        "semantic_id",
    ]
    assert audit["failed_camera_counts"]["foreground_semantic_valid"] == 0


def test_full_output_audit_rejects_a_core_only_result() -> None:
    with pytest.raises(AssertionError, match="Incomplete output contract"):
        output_summary(
            render_outputs(include_semantic=False),
            required_outputs=FULL_OUTPUTS,
        )
