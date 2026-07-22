"""Build the pinned FlashGS-derived matched-contract port."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


FLASHGS_UPSTREAM_COMMIT = "cdfc4e4002318423eda356eed02df8e01fa32cb6"
FLASHGS_CXX_FLAGS = ("-O3", "-std=c++17")
FLASHGS_CUDA_FLAGS = (
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-std=c++17",
)
_LOADED_MODULE: ModuleType | None = None


def _flashgs_source_path(project_root: Path) -> Path:
    configured = os.environ.get("FLASHGS_SOURCE_PATH")
    candidates = [
        Path(configured) if configured else None,
        project_root.parent / "FlashGS",
        Path("/workspace/src/FlashGS"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "csrc" / "glm" / "glm.hpp").is_file():
            return candidate.resolve()
    searched = ", ".join(str(item) for item in candidates if item is not None)
    raise RuntimeError(
        "Pinned FlashGS source is required to supply its public GLM headers; "
        f"searched {searched}. Set FLASHGS_SOURCE_PATH."
    )


def _verify_flashgs_commit(source_path: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Cannot resolve FlashGS provenance at {source_path}."
        ) from error
    if commit != FLASHGS_UPSTREAM_COMMIT:
        raise RuntimeError(
            "FlashGS source commit mismatch: "
            f"{commit} != pinned {FLASHGS_UPSTREAM_COMMIT}."
        )
    try:
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
        raise RuntimeError(
            f"Cannot audit FlashGS working tree at {source_path}."
        ) from error
    if status:
        raise RuntimeError(
            "Pinned FlashGS source must be a clean checkout; status was "
            f"{status!r}."
        )
    return commit


def load_flashgs_native_extension(*, verbose: bool | None = None) -> ModuleType:
    """Compile once per adaptation hash and return its PyTorch extension."""
    global _LOADED_MODULE
    if _LOADED_MODULE is not None:
        return _LOADED_MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("The FlashGS-derived matched-port extension requires CUDA.")

    project_root = Path(__file__).resolve().parents[2]
    flashgs_source = _flashgs_source_path(project_root)
    flashgs_commit = _verify_flashgs_commit(flashgs_source)
    native_root = Path(__file__).resolve().parent / "native" / "flashgs"
    sources = [
        native_root / "adapter.cpp",
        native_root / "preprocess.cu",
        native_root / "sort.cu",
        native_root / "render.cu",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"FlashGS-derived matched-port sources are missing: {missing}.")

    digest = hashlib.sha256(flashgs_commit.encode("ascii"))
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    header = native_root / "ops.h"
    digest.update(header.read_bytes())
    module_name = f"isaacsim_flashgs_adapter_{digest.hexdigest()[:12]}"
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
        extra_include_paths=[str(flashgs_source / "csrc")],
        build_directory=str(build_directory),
        extra_cflags=list(FLASHGS_CXX_FLAGS),
        extra_cuda_cflags=list(FLASHGS_CUDA_FLAGS),
        with_cuda=True,
        is_python_module=True,
        verbose=verbose,
    )
    _LOADED_MODULE.__vgr_build_contract__ = {
        "upstream_commit": flashgs_commit,
        "module_name": module_name,
        "build_directory": str(build_directory),
        "cxx_flags": list(FLASHGS_CXX_FLAGS),
        "cuda_flags": list(FLASHGS_CUDA_FLAGS),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "sources": [str(source) for source in sources],
        "header": str(header),
    }
    return _LOADED_MODULE
