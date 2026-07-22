from __future__ import annotations

import json

import numpy as np
from PIL import Image

from isaacsim_gaussian_renderer.evaluation.artifacts import (
    EvaluationArtifactBundle,
    audit_artifact_bundle,
)


def _make_bundle(tmp_path, frames: list[np.ndarray]):
    bundle = EvaluationArtifactBundle.create(tmp_path / "run")
    stats = []
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(bundle.root / "frames" / f"frame-{index:06d}.png")
        rgb = frame.astype(np.float64) / 255.0
        stats.append(
            {
                "nonblack_fraction": float(np.any(rgb > (2 / 255), axis=-1).mean()),
                "clipped_fraction": float(np.any(rgb >= (254 / 255), axis=-1).mean()),
                "mean": float(rgb.mean()),
                "std": float(rgb.std()),
            }
        )
    bundle.write_json(
        "manifest.json",
        {
            "schema_version": "trajectory-evaluation-run-v1",
            "frame_count": len(frames),
            "width": frames[0].shape[1],
            "height": frames[0].shape[0],
            "frame_statistics": stats,
            "max_explained_duplicate_run": 1,
            "cited_frames": ["frame-000000.png"],
        },
    )
    bundle.write_hash_manifest()
    return bundle


def test_artifact_auditor_accepts_reopened_frames(tmp_path) -> None:
    frames = [
        np.full((8, 8, 3), 20 + index, dtype=np.uint8)
        for index in range(3)
    ]
    result = audit_artifact_bundle(_make_bundle(tmp_path, frames).root)
    assert result["pass"] is True


def test_artifact_auditor_rejects_black_and_duplicate_regression(tmp_path) -> None:
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    bundle = _make_bundle(tmp_path, [black, black.copy(), black.copy()])
    result = audit_artifact_bundle(bundle.root)
    assert result["pass"] is False
    assert any(item.startswith("black:") for item in result["failures"])
    assert any(item.startswith("unexplained-frozen-run:") for item in result["failures"])
