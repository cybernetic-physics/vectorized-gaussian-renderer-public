from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import torch

from benchmarks.compare_custom_gsplat import (
    _comparison_pass,
    _mip_splat_compensated_opacity,
    _projected_covariance_for_fixture,
    _require_passing_summary,
    _validate_pinned_gsplat_checkout,
    _zero_capacity_overflow,
)
from scripts.projection_cache_cuda_smoke import (
    _output_contract,
    _resolve_covariance_epsilon,
)


def test_mip_splat_compensation_matches_subpixel_fixture() -> None:
    opacity, compensation = _mip_splat_compensated_opacity(
        opacity=0.8,
        covariance_00=0.04,
        covariance_01=0.0,
        covariance_11=0.16,
        covariance_epsilon=0.3,
    )

    expected = (0.04 * 0.16 / (0.34 * 0.46)) ** 0.5
    assert compensation == pytest.approx(expected)
    assert opacity == pytest.approx(0.8 * expected)


def test_mip_splat_compensation_applies_degenerate_floor() -> None:
    opacity, compensation = _mip_splat_compensated_opacity(
        opacity=0.8,
        covariance_00=0.0,
        covariance_01=0.0,
        covariance_11=0.0,
        covariance_epsilon=0.3,
    )

    assert compensation == pytest.approx(0.005)
    assert opacity == pytest.approx(0.004)


def test_rotated_off_axis_fixture_has_nonzero_cross_covariance() -> None:
    covariance = _projected_covariance_for_fixture(
        mean=(0.18, -0.12, 3.0),
        scales=(0.006, 0.014, 0.01),
        quaternion=(
            0.9659258262890683,
            0.0,
            0.0,
            0.25881904510252074,
        ),
        focal_x=96.0,
        focal_y=88.0,
    )

    assert covariance[0] > 0
    assert covariance[2] > 0
    assert covariance[0] * covariance[2] - covariance[1] ** 2 > 0
    assert abs(covariance[1]) > 1.0e-4


def test_antialias_darkening_decreases_with_resolution() -> None:
    compensated_opacities = []
    for width in (32, 64, 128):
        covariance = _projected_covariance_for_fixture(
            mean=(0.15, -0.10, 3.0),
            scales=(0.012, 0.020, 0.010),
            quaternion=(
                0.9659258262890683,
                0.0,
                0.0,
                0.25881904510252074,
            ),
            focal_x=1.5 * width,
            focal_y=1.5 * width,
        )
        opacity, _compensation = _mip_splat_compensated_opacity(
            opacity=0.8,
            covariance_00=covariance[0],
            covariance_01=covariance[1],
            covariance_11=covariance[2],
            covariance_epsilon=0.3,
        )
        compensated_opacities.append(opacity)

    assert compensated_opacities[0] < compensated_opacities[1]
    assert compensated_opacities[1] < compensated_opacities[2]
    assert 0.8 - compensated_opacities[0] > 0.8 - compensated_opacities[1]
    assert 0.8 - compensated_opacities[1] > 0.8 - compensated_opacities[2]


def test_projection_smoke_defaults_antialias_compact_to_nonzero_filter() -> None:
    assert _resolve_covariance_epsilon(
        requested=None,
        rasterize_mode="antialiased",
        ray_gaussian_evaluation=False,
        compact_projection_cache=True,
    ) == pytest.approx(0.3)
    assert _resolve_covariance_epsilon(
        requested=None,
        rasterize_mode="classic",
        ray_gaussian_evaluation=False,
        compact_projection_cache=True,
    ) == pytest.approx(0.0)
    assert _resolve_covariance_epsilon(
        requested=0.0,
        rasterize_mode="antialiased",
        ray_gaussian_evaluation=False,
        compact_projection_cache=True,
    ) == pytest.approx(0.0)


def test_comparison_verdict_requires_zero_capacity_overflow() -> None:
    counters = {"visible_overflow": 0, "intersection_overflow": 0}
    assert _zero_capacity_overflow(counters)
    assert _comparison_pass(
        fidelity_pass=True,
        antialias_fixture={"pass": True},
        custom_counters=counters,
    )

    for overflow_name in ("visible_overflow", "intersection_overflow"):
        overflowing = {**counters, overflow_name: 1}
        assert not _zero_capacity_overflow(overflowing)
        assert not _comparison_pass(
            fidelity_pass=True,
            antialias_fixture={"pass": True},
            custom_counters=overflowing,
        )


def test_failing_comparison_summary_exits_nonzero() -> None:
    _require_passing_summary({"pass": True})
    with pytest.raises(SystemExit) as error:
        _require_passing_summary({"pass": False})
    assert error.value.code == 1


def test_projection_smoke_rejects_low_alpha_semantic_id() -> None:
    outputs = {
        "rgb": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "depth": torch.ones((1, 1, 2, 1), dtype=torch.float32),
        "alpha": torch.tensor(
            [[[[0.5], [0.005]]]],
            dtype=torch.float32,
        ),
        "semantic_id": torch.tensor(
            [[[[3], [7]]]],
            dtype=torch.int64,
        ),
    }

    invalid = _output_contract(
        outputs,
        height=1,
        width=2,
        semantic_min_alpha=0.01,
    )
    assert invalid["valid_foreground_semantics"]
    assert not invalid["background_semantics"]

    outputs["semantic_id"][0, 0, 1, 0] = -1
    valid = _output_contract(
        outputs,
        height=1,
        width=2,
        semantic_min_alpha=0.01,
    )
    assert valid["background_semantics"]


def _git(path: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _fixture_repository(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repository = tmp_path / "gsplat"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Antialias Test")
    _git(repository, "config", "user.email", "antialias@example.invalid")
    source = repository / "gsplat" / "cuda" / "csrc" / "Utils.cpp"
    source.parent.mkdir(parents=True)
    original = "event=create(default);\nmerge=create(default);\n"
    source.write_text(original, encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    _git(repository, "rev-parse", "HEAD")
    source.write_text(
        "event=create(disable_timing);\nmerge=create(disable_timing);\n",
        encoding="utf-8",
    )
    patch = tmp_path / "compatibility.patch"
    patch.write_text(_git(repository, "diff", "--binary"), encoding="utf-8")
    patch_sha256 = hashlib.sha256(patch.read_bytes()).hexdigest()
    return repository, patch, patch_sha256, original


def test_pinned_gsplat_provenance_accepts_clean_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _patch, _patch_sha256, original = _fixture_repository(tmp_path)
    source = repository / "gsplat" / "cuda" / "csrc" / "Utils.cpp"
    source.write_text(original, encoding="utf-8")
    commit = _git(repository, "rev-parse", "HEAD").strip()
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong-git-dir"))

    record = _validate_pinned_gsplat_checkout(
        repository,
        expected_commit=commit,
        compatibility_patch=None,
        expected_compatibility_patch_sha256="unused",
    )

    assert record["worktree_clean"] is True
    assert record["compatibility_patch"] is None


def test_pinned_gsplat_provenance_accepts_only_exact_patch(
    tmp_path: Path,
) -> None:
    repository, patch, patch_sha256, _original = _fixture_repository(tmp_path)
    commit = _git(repository, "rev-parse", "HEAD").strip()

    record = _validate_pinned_gsplat_checkout(
        repository,
        expected_commit=commit,
        compatibility_patch=patch,
        expected_compatibility_patch_sha256=patch_sha256,
    )

    assert record["worktree_clean"] is False
    assert record["compatibility_patch"]["applied"] is True
    assert record["compatibility_patch"]["sha256"] == patch_sha256
    assert (
        record["compatibility_patch"]["actual_tree"]
        == record["compatibility_patch"]["expected_tree"]
    )

    source = repository / "gsplat" / "cuda" / "csrc" / "Utils.cpp"
    source.write_text("arbitrary dirty state\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not exactly match"):
        _validate_pinned_gsplat_checkout(
            repository,
            expected_commit=commit,
            compatibility_patch=patch,
            expected_compatibility_patch_sha256=patch_sha256,
        )


def test_pinned_gsplat_provenance_rejects_untracked_and_staged_state(
    tmp_path: Path,
) -> None:
    repository, patch, patch_sha256, _original = _fixture_repository(tmp_path)
    commit = _git(repository, "rev-parse", "HEAD").strip()
    (repository / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unstaged modifications"):
        _validate_pinned_gsplat_checkout(
            repository,
            expected_commit=commit,
            compatibility_patch=patch,
            expected_compatibility_patch_sha256=patch_sha256,
        )

    (repository / "unexpected.txt").unlink()
    _git(repository, "add", "gsplat/cuda/csrc/Utils.cpp")
    with pytest.raises(RuntimeError, match="unstaged modifications"):
        _validate_pinned_gsplat_checkout(
            repository,
            expected_commit=commit,
            compatibility_patch=patch,
            expected_compatibility_patch_sha256=patch_sha256,
        )
