"""Build the pinned upstream-equation FlashGS RGB control extension."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

from .upstream_faithful_flashgs_sources import (
    FLASHGS_UPSTREAM_COMMIT,
    PINNED_SOURCE_SHA256,
    verify_pinned_source_files,
    write_generated_preprocess,
)


# These match the public FlashGS setup.py optimization policy. The C++17 flag
# is required by current PyTorch extension headers but does not alter kernels.
UPSTREAM_FAITHFUL_CXX_FLAGS = ("-g", "-O1", "-std=c++17")
UPSTREAM_FAITHFUL_CUDA_FLAGS = ("-O1", "-Xptxas=-O1", "-std=c++17")
_LOADED_MODULE: ModuleType | None = None
_LOADED_REFERENCE_MODULE: ModuleType | None = None


def _require_live_cuda_architecture() -> str:
    """Bind the content-addressed build to the GPU executing the control."""
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    expected = f"{major}.{minor}"
    configured = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if configured is not None and configured != expected:
        raise RuntimeError(
            "The upstream-faithful control requires TORCH_CUDA_ARCH_LIST to "
            f"match the live GPU exactly: {configured!r} != {expected!r}."
        )
    os.environ["TORCH_CUDA_ARCH_LIST"] = expected
    return expected


def _source_path(project_root: Path) -> Path:
    configured = os.environ.get("FLASHGS_SOURCE_PATH")
    candidates = [
        Path(configured) if configured else None,
        project_root.parent / "FlashGS",
        Path("/workspace/src/FlashGS"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "csrc/cuda_rasterizer/render.cu").is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise RuntimeError(
        "Pinned FlashGS source is required for the upstream-faithful control; "
        f"searched {searched}. Set FLASHGS_SOURCE_PATH."
    )


def _verify_clean_commit(source_path: Path) -> dict[str, str | bool]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(source_path),
                "status",
                "--short",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot audit FlashGS source at {source_path}.") from error
    if commit != FLASHGS_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"FlashGS source commit mismatch: {commit} != {FLASHGS_UPSTREAM_COMMIT}."
        )
    if status:
        raise RuntimeError(
            "The upstream-faithful control requires a clean FlashGS checkout; "
            f"status was {status!r}."
        )
    return {"commit": commit, "clean": True}


def load_upstream_faithful_flashgs_native_extension(
    *, verbose: bool | None = None
) -> ModuleType:
    """Compile the exact-source control once per content-addressed build."""
    global _LOADED_MODULE
    if _LOADED_MODULE is not None:
        return _LOADED_MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("The upstream-faithful FlashGS extension requires CUDA.")

    project_root = Path(__file__).resolve().parents[2]
    source_path = _source_path(project_root)
    git_audit = _verify_clean_commit(source_path)
    source_audit = verify_pinned_source_files(source_path)
    cuda_architecture = _require_live_cuda_architecture()
    native_root = Path(__file__).resolve().parent / "native/flashgs_upstream_faithful"
    native_sources = [native_root / "adapter.cpp", native_root / "glue.cu"]
    native_header = native_root / "faithful_glue.h"
    missing = [
        str(path)
        for path in [*native_sources, native_header]
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"Upstream-faithful FlashGS glue is missing: {missing}.")

    digest = hashlib.sha256(FLASHGS_UPSTREAM_COMMIT.encode("ascii"))
    digest.update(
        json.dumps(PINNED_SOURCE_SHA256, sort_keys=True).encode("utf-8")
    )
    transform_source = Path(__file__).with_name(
        "upstream_faithful_flashgs_sources.py"
    )
    for path in [transform_source, native_header, *native_sources]:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(repr(UPSTREAM_FAITHFUL_CXX_FLAGS).encode("ascii"))
    digest.update(repr(UPSTREAM_FAITHFUL_CUDA_FLAGS).encode("ascii"))
    digest.update(cuda_architecture.encode("ascii"))
    digest.update(torch.__version__.encode("utf-8"))
    digest.update(str(torch.version.cuda).encode("ascii"))
    module_name = f"isaacsim_flashgs_upstream_faithful_{digest.hexdigest()[:12]}"
    build_root = Path(
        os.environ.get(
            "VGR_NATIVE_BUILD_ROOT",
            project_root / "build/torch_extensions",
        )
    ).resolve()
    build_directory = build_root / module_name
    build_directory.mkdir(parents=True, exist_ok=True)
    generated_preprocess = build_directory / "generated/preprocess.cu"
    transform_audit_path = build_directory / "generated/preprocess-audit.json"
    transform_audit = write_generated_preprocess(
        source_path,
        generated_preprocess,
        transform_audit_path,
    )
    build_audit_path = build_directory / "upstream-faithful-source-audit.json"
    build_audit = {
        "schema_version": "upstream-faithful-flashgs-build-audit-v1",
        "upstream": {**git_audit, **source_audit},
        "preprocess_transform": transform_audit,
        "algorithm_translation_units": {
            "preprocess": str(generated_preprocess),
            "sort": str(source_path / "csrc/cuda_rasterizer/sort.cu"),
            "render": str(source_path / "csrc/cuda_rasterizer/render.cu"),
        },
        "classification": {
            "upstream_faithful": True,
            "scope": "rgb-only-degree-zero",
            "exact_upstream_sort": True,
            "exact_upstream_render_and_tile_shader": True,
            "preprocess_equation_functions_byte_preserved": True,
            "byte_exact_upstream_preprocess_translation_unit": False,
            "degree_zero_color_specialization": True,
            "performance_identical_to_unmodified_public_binary": False,
            "integration_changes": [
                "gpu-camera-state",
                "canonical-degree-zero-color-input",
                "bounded-fixed-capacity-key-emission",
                "out-of-range-sentinel-tile",
                "active-pytorch-stream",
                "preallocated-reusable-storage",
                "measured-u8-to-float32-conversion",
            ],
            "full_sensor_output": False,
        },
        "flags": {
            "cxx": list(UPSTREAM_FAITHFUL_CXX_FLAGS),
            "cuda": list(UPSTREAM_FAITHFUL_CUDA_FLAGS),
            "torch_cuda_arch_list": cuda_architecture,
        },
        "pass": True,
    }
    build_audit_path.write_text(
        json.dumps(build_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if verbose is None:
        verbose = os.environ.get("VGR_NATIVE_VERBOSE", "0") == "1"
    sources = [
        native_sources[0],
        native_sources[1],
        generated_preprocess,
        source_path / "csrc/cuda_rasterizer/sort.cu",
        source_path / "csrc/cuda_rasterizer/render.cu",
    ]
    _LOADED_MODULE = load(
        name=module_name,
        sources=[str(path) for path in sources],
        extra_include_paths=[str(native_root), str(source_path / "csrc")],
        build_directory=str(build_directory),
        extra_cflags=list(UPSTREAM_FAITHFUL_CXX_FLAGS),
        extra_cuda_cflags=list(UPSTREAM_FAITHFUL_CUDA_FLAGS),
        with_cuda=True,
        is_python_module=True,
        verbose=verbose,
    )
    _LOADED_MODULE.__vgr_build_contract__ = {
        "lane": "upstream-faithful-flashgs-rgb",
        "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
        "module_name": module_name,
        "build_directory": str(build_directory),
        "source_audit": str(build_audit_path),
        "source_audit_sha256": hashlib.sha256(build_audit_path.read_bytes()).hexdigest(),
        "cxx_flags": list(UPSTREAM_FAITHFUL_CXX_FLAGS),
        "cuda_flags": list(UPSTREAM_FAITHFUL_CUDA_FLAGS),
        "torch_cuda_arch_list": cuda_architecture,
        "sources": [str(path) for path in sources],
    }
    return _LOADED_MODULE


def load_pinned_upstream_flashgs_reference_extension(
    *, verbose: bool | None = None
) -> ModuleType:
    """Build byte-exact public FlashGS for untimed output-parity checks only."""
    global _LOADED_REFERENCE_MODULE
    if _LOADED_REFERENCE_MODULE is not None:
        return _LOADED_REFERENCE_MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("The pinned FlashGS reference extension requires CUDA.")
    project_root = Path(__file__).resolve().parents[2]
    source_path = _source_path(project_root)
    _verify_clean_commit(source_path)
    source_audit = verify_pinned_source_files(source_path)
    cuda_architecture = _require_live_cuda_architecture()
    digest = hashlib.sha256(b"pinned-upstream-flashgs-reference")
    for relative, record in sorted(source_audit["files"].items()):
        digest.update(relative.encode("utf-8"))
        digest.update(str(record["sha256"]).encode("ascii"))
    digest.update(repr(UPSTREAM_FAITHFUL_CXX_FLAGS).encode("ascii"))
    digest.update(repr(UPSTREAM_FAITHFUL_CUDA_FLAGS).encode("ascii"))
    digest.update(cuda_architecture.encode("ascii"))
    digest.update(torch.__version__.encode("utf-8"))
    digest.update(str(torch.version.cuda).encode("ascii"))
    module_name = f"flash_gaussian_splatting_reference_{digest.hexdigest()[:12]}"
    build_root = Path(
        os.environ.get(
            "VGR_NATIVE_BUILD_ROOT",
            project_root / "build/torch_extensions",
        )
    ).resolve()
    build_directory = build_root / module_name
    build_directory.mkdir(parents=True, exist_ok=True)
    if verbose is None:
        verbose = os.environ.get("VGR_NATIVE_VERBOSE", "0") == "1"
    sources = [
        source_path / "csrc/pybind.cpp",
        source_path / "csrc/cuda_rasterizer/preprocess.cu",
        source_path / "csrc/cuda_rasterizer/sort.cu",
        source_path / "csrc/cuda_rasterizer/render.cu",
    ]
    _LOADED_REFERENCE_MODULE = load(
        name=module_name,
        sources=[str(path) for path in sources],
        extra_include_paths=[str(source_path / "csrc")],
        build_directory=str(build_directory),
        extra_cflags=list(UPSTREAM_FAITHFUL_CXX_FLAGS),
        extra_cuda_cflags=list(UPSTREAM_FAITHFUL_CUDA_FLAGS),
        with_cuda=True,
        is_python_module=True,
        verbose=verbose,
    )
    _LOADED_REFERENCE_MODULE.__vgr_build_contract__ = {
        "lane": "byte-exact-pinned-upstream-parity-reference",
        "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
        "module_name": module_name,
        "build_directory": str(build_directory),
        "sources": [str(path) for path in sources],
        "source_sha256": {
            relative: record["sha256"]
            for relative, record in source_audit["files"].items()
        },
        "cxx_flags": list(UPSTREAM_FAITHFUL_CXX_FLAGS),
        "cuda_flags": list(UPSTREAM_FAITHFUL_CUDA_FLAGS),
        "torch_cuda_arch_list": cuda_architecture,
        "benchmark_eligible": False,
        "purpose": "untimed-output-parity-only",
    }
    return _LOADED_REFERENCE_MODULE
