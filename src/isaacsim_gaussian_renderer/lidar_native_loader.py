"""Separately lazy-load the project-owned Gaussian LiDAR extension."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_LOADED_MODULE: ModuleType | None = None


def is_lidar_native_loaded() -> bool:
    return _LOADED_MODULE is not None


def load_lidar_native_extension(*, verbose: bool | None = None) -> ModuleType:
    """Compile only after an explicit LiDAR backend is initialized."""

    global _LOADED_MODULE
    if _LOADED_MODULE is not None:
        return _LOADED_MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("The Gaussian LiDAR native extension requires CUDA.")
    native_root = Path(__file__).resolve().parent / "native"
    sources = [native_root / "lidar_renderer.cpp", native_root / "lidar_renderer_cuda.cu"]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    module_name = f"isaacsim_gaussian_lidar_cuda_{digest.hexdigest()[:12]}"
    project_root = Path(__file__).resolve().parents[2]
    build_root = Path(
        os.environ.get("VGR_LIDAR_NATIVE_BUILD_ROOT", project_root / "build" / "lidar_torch_extensions")
    )
    build_directory = build_root / module_name
    build_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    if verbose is None:
        verbose = os.environ.get("VGR_LIDAR_NATIVE_VERBOSE", "0") == "1"
    _LOADED_MODULE = load(
        name=module_name,
        sources=[str(source) for source in sources],
        build_directory=str(build_directory),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"],
        with_cuda=True,
        is_python_module=True,
        verbose=verbose,
    )
    return _LOADED_MODULE
