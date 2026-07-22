"""Build and cache the project-owned PyTorch C++/CUDA extension."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


NATIVE_CXX_FLAGS = ("-O3", "-std=c++17")
NATIVE_CUDA_FLAGS = (
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-std=c++17",
)
_LOADED_MODULE: ModuleType | None = None


def load_native_extension(*, verbose: bool | None = None) -> ModuleType:
    """Compile once per source hash and return the loaded native module."""
    global _LOADED_MODULE
    if _LOADED_MODULE is not None:
        return _LOADED_MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("The custom Gaussian renderer native extension requires CUDA.")

    native_root = Path(__file__).resolve().parent / "native"
    sources = [native_root / "renderer.cpp", native_root / "renderer_cuda.cu"]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    module_name = f"isaacsim_gaussian_renderer_cuda_{digest.hexdigest()[:12]}"

    project_root = Path(__file__).resolve().parents[2]
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

    _LOADED_MODULE = load(
        name=module_name,
        sources=[str(source) for source in sources],
        build_directory=str(build_directory),
        extra_cflags=list(NATIVE_CXX_FLAGS),
        extra_cuda_cflags=list(NATIVE_CUDA_FLAGS),
        with_cuda=True,
        is_python_module=True,
        verbose=verbose,
    )
    _LOADED_MODULE.__vgr_build_contract__ = {
        "module_name": module_name,
        "build_directory": str(build_directory),
        "cxx_flags": list(NATIVE_CXX_FLAGS),
        "cuda_flags": list(NATIVE_CUDA_FLAGS),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "sources": [str(source) for source in sources],
    }
    return _LOADED_MODULE
