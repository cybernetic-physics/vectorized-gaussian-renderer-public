import ast
import sys
from pathlib import Path

import pytest
import torch

from scripts.isaacsim_extension_smoke import (
    output_contract,
    parse_args,
    resolve_covariance_epsilon,
)
from isaacsim_gaussian_renderer import DEFAULT_GAUSSIAN_SUPPORT_SIGMA


def test_kit_extension_uses_the_production_support_cutoff_default() -> None:
    extension_path = (
        Path(__file__).parents[1]
        / "exts"
        / "isaacsim.gaussian_renderer"
        / "isaacsim_gaussian_renderer_extension"
        / "extension.py"
    )
    tree = ast.parse(extension_path.read_text(encoding="utf-8"))
    extension_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "IsaacSimGaussianRendererExtension"
    )
    create_service = next(
        node
        for node in extension_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_service"
    )
    keyword_defaults = dict(
        zip(
            (argument.arg for argument in create_service.args.kwonlyargs),
            create_service.args.kw_defaults,
            strict=True,
        )
    )
    default = keyword_defaults["gaussian_support_sigma"]

    assert isinstance(default, ast.Name)
    assert default.id == "DEFAULT_GAUSSIAN_SUPPORT_SIGMA"
    assert DEFAULT_GAUSSIAN_SUPPORT_SIGMA == 3.33


def test_kit_extension_exposes_bounded_physical_batch_configuration() -> None:
    extension_path = (
        Path(__file__).parents[1]
        / "exts"
        / "isaacsim.gaussian_renderer"
        / "isaacsim_gaussian_renderer_extension"
        / "extension.py"
    )
    tree = ast.parse(extension_path.read_text(encoding="utf-8"))
    extension_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "IsaacSimGaussianRendererExtension"
    )
    create_service = next(
        node
        for node in extension_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_service"
    )
    keyword_defaults = dict(
        zip(
            (argument.arg for argument in create_service.args.kwonlyargs),
            create_service.args.kw_defaults,
            strict=True,
        )
    )

    assert isinstance(keyword_defaults["fixed_capacity_sort"], ast.Constant)
    assert keyword_defaults["fixed_capacity_sort"].value is False
    assert isinstance(keyword_defaults["max_physical_views"], ast.Constant)
    assert keyword_defaults["max_physical_views"].value is None


def test_extension_smoke_exposes_antialias_configuration(tmp_path) -> None:
    output = tmp_path / "result.json"
    args = parse_args(
        [
            "--rasterize-mode",
            "antialiased",
            "--compact-projection-cache",
            "--tile-size",
            "1",
            "--output",
            str(output),
        ]
    )

    assert args.rasterize_mode == "antialiased"
    assert resolve_covariance_epsilon(
        args.covariance_epsilon,
        rasterize_mode=args.rasterize_mode,
        ray_gaussian_evaluation=args.ray_gaussian_evaluation,
        compact_projection_cache=args.compact_projection_cache,
    ) == 0.3
    assert args.output == output


def test_extension_smoke_forwards_unknown_kit_arguments(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isaacsim_extension_smoke.py",
            "--rasterize-mode",
            "antialiased",
            "--portable-root",
            "/tmp/portable",
        ],
    )

    args = parse_args()

    assert args.rasterize_mode == "antialiased"
    assert sys.argv == [
        "isaacsim_extension_smoke.py",
        "--portable-root",
        "/tmp/portable",
    ]


def test_extension_smoke_rejects_antialias_exact_ray() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        resolve_covariance_epsilon(
            None,
            rasterize_mode="antialiased",
            ray_gaussian_evaluation=True,
            compact_projection_cache=False,
        )


def test_extension_smoke_output_contract_accepts_defined_cpu_fixture() -> None:
    outputs = {
        "rgb": torch.zeros((2, 2, 2, 3), dtype=torch.float32),
        "depth": torch.full((2, 2, 2, 1), float("inf"), dtype=torch.float32),
        "alpha": torch.zeros((2, 2, 2, 1), dtype=torch.float32),
        "semantic_id": torch.full((2, 2, 2, 1), -1, dtype=torch.int64),
    }
    outputs["depth"][1, 0, 0, 0] = 3.0
    outputs["alpha"][1, 0, 0, 0] = 0.75
    outputs["semantic_id"][1, 0, 0, 0] = 23

    contract = output_contract(
        outputs,
        semantic_min_alpha=0.01,
        expected_shape=(2, 2, 2),
        require_cuda=False,
    )

    assert contract["valid"]
    assert contract["foreground_pixel_count"] == 1
    assert not contract["outputs_gpu_resident"]
