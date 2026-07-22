"""Build a separate, non-production FlashGS pipeline diagnostic extension."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

from .flashgs_native_loader import (
    FLASHGS_CUDA_FLAGS,
    FLASHGS_CXX_FLAGS,
    _flashgs_source_path,
    _verify_flashgs_commit,
)


_LOADED_DEBUG_MODULE: ModuleType | None = None


def load_flashgs_debug_extension(*, verbose: bool | None = None) -> ModuleType:
    """Compile diagnostics without changing the production adapter module.

    The module digest binds both debug sources and the production FlashGS
    projection/range/compositor sources whose intermediate buffers it reads.
    This prevents a trace built for one adapter revision from being silently
    presented as evidence for another.
    """

    global _LOADED_DEBUG_MODULE
    if _LOADED_DEBUG_MODULE is not None:
        return _LOADED_DEBUG_MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("The FlashGS pipeline diagnostic requires CUDA.")

    project_root = Path(__file__).resolve().parents[2]
    flashgs_source = _flashgs_source_path(project_root)
    flashgs_commit = _verify_flashgs_commit(flashgs_source)
    native_root = Path(__file__).resolve().parent / "native" / "flashgs"
    sources = [
        native_root / "debug_adapter.cpp",
        native_root / "debug_trace.cu",
    ]
    headers = [native_root / "debug_ops.h", native_root / "ops.h"]
    production_sources = [
        native_root / "adapter.cpp",
        native_root / "preprocess.cu",
        native_root / "sort.cu",
        native_root / "render.cu",
    ]
    inputs = [*sources, *headers, *production_sources]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"FlashGS diagnostic build inputs are missing: {missing}.")

    digest = hashlib.sha256(flashgs_commit.encode("ascii"))
    input_sha256: dict[str, str] = {}
    for path in inputs:
        contents = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(contents)
        input_sha256[path.name] = hashlib.sha256(contents).hexdigest()
    module_name = f"isaacsim_flashgs_debug_{digest.hexdigest()[:12]}"
    build_root = Path(
        os.environ.get(
            "VGR_NATIVE_BUILD_ROOT",
            project_root / "build" / "torch_extensions",
        )
    )
    build_directory = build_root / module_name
    build_directory.mkdir(parents=True, exist_ok=True)
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    if verbose is None:
        verbose = os.environ.get("VGR_NATIVE_VERBOSE", "0") == "1"

    _LOADED_DEBUG_MODULE = load(
        name=module_name,
        sources=[str(source) for source in sources],
        extra_include_paths=[str(native_root), str(flashgs_source / "csrc")],
        build_directory=str(build_directory),
        extra_cflags=list(FLASHGS_CXX_FLAGS),
        extra_cuda_cflags=list(FLASHGS_CUDA_FLAGS),
        with_cuda=True,
        is_python_module=True,
        verbose=verbose,
    )
    _LOADED_DEBUG_MODULE.__vgr_build_contract__ = {
        "purpose": "debug-only-flashgs-pipeline-trace",
        "upstream_commit": flashgs_commit,
        "module_name": module_name,
        "build_directory": str(build_directory),
        "cxx_flags": list(FLASHGS_CXX_FLAGS),
        "cuda_flags": list(FLASHGS_CUDA_FLAGS),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "compiled_sources": [str(source) for source in sources],
        "bound_production_sources": [str(source) for source in production_sources],
        "input_sha256": input_sha256,
    }
    return _LOADED_DEBUG_MODULE
