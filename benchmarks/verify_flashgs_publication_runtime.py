#!/usr/bin/env python3
"""Fail fast when the matched benchmark runtime mixes incompatible CUDA stacks."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    active_cuda_device_uuid,
    artifact_record,
    load_verified_source_manifest,
    source_identity,
)
from isaacsim_gaussian_renderer.fidelity.metrics import _lpips_per_view  # noqa: E402


RUNTIME_PREFLIGHT_SCHEMA = "flashgs-publication-runtime-preflight-v1"
_HOST_USER_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"file://(?:localhost)?/(?:Users|home)/[^/\s:]+/|"
    r"/(?:Users|home)/[^/\s:]+/|"
    r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s:]+[\\/]"
    r")"
)
_CUDA_LIBRARY_NAMES = (
    "libcublas",
    "libcuda",
    "libcudart",
    "libcudnn",
    "libnvrtc",
    "libtorch_cuda",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_portable_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if _HOST_USER_PATH_RE.search(str(resolved)):
        raise RuntimeError(f"{label} contains a host-user identity: {resolved}")
    return resolved


def module_record(module: ModuleType) -> dict[str, Any]:
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str):
        raise RuntimeError(f"Module {module.__name__} has no filesystem origin.")
    origin = require_portable_path(Path(raw_origin), label=f"{module.__name__} origin")
    return {
        "name": module.__name__,
        "origin": str(origin),
        "version": str(getattr(module, "__version__", "unknown")),
    }


def loaded_cuda_libraries() -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    libraries: set[str] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        candidate = line.rsplit(maxsplit=1)[-1]
        name = Path(candidate).name.lower()
        if candidate.startswith("/") and any(token in name for token in _CUDA_LIBRARY_NAMES):
            libraries.add(
                str(require_portable_path(Path(candidate), label="loaded CUDA library"))
            )
    return sorted(libraries)


def require_weight(path: Path, *, label: str) -> dict[str, Any]:
    portable = require_portable_path(path, label=label)
    return artifact_record(portable)


def main() -> None:
    args = parse_args()

    import lpips
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the publication runtime.")
    torch.cuda.init()
    observed_gpu_uuid = active_cuda_device_uuid(torch.cuda)
    if observed_gpu_uuid != args.expected_gpu_uuid:
        raise RuntimeError(
            f"Publication runtime resolved {observed_gpu_uuid}, expected {args.expected_gpu_uuid}."
        )

    # Exercise cuBLAS and convolution before any long renderer work. The prior
    # mixed Isaac/venv runtime aborted only when LPIPS first loaded cublasLt.
    torch.manual_seed(20_260_722)
    left = torch.linspace(-1.0, 1.0, 256 * 128, device="cuda").reshape(256, 128)
    right = torch.linspace(1.0, -1.0, 128 * 64, device="cuda").reshape(128, 64)
    matrix = left @ right
    image = torch.linspace(-1.0, 1.0, 3 * 64 * 64, device="cuda").reshape(1, 3, 64, 64)
    kernel = torch.linspace(-0.5, 0.5, 8 * 3 * 3 * 3, device="cuda").reshape(8, 3, 3, 3)
    convolution = torch.nn.functional.conv2d(image, kernel, padding=1)
    if not bool(torch.isfinite(matrix).all().item()):
        raise RuntimeError("CUDA matrix multiplication produced a non-finite value.")
    if not bool(torch.isfinite(convolution).all().item()):
        raise RuntimeError("CUDA convolution produced a non-finite value.")

    reference = np.zeros((1, 64, 64, 3), dtype=np.float32)
    candidate = reference.copy()
    candidate[:, 12:52, 16:48, :] = 0.25
    lpips_values, lpips_backend, lpips_error = _lpips_per_view(
        reference,
        candidate,
        require=True,
    )
    if (
        lpips_backend != "lpips-alex"
        or lpips_error is not None
        or lpips_values is None
        or len(lpips_values) != 1
        or not math.isfinite(lpips_values[0])
    ):
        raise RuntimeError(
            "Required LPIPS-Alex CUDA preflight failed: "
            f"backend={lpips_backend!r}, error={lpips_error!r}."
        )
    torch.cuda.synchronize()

    source_provenance = load_verified_source_manifest(
        args.source_manifest,
        project_root=PROJECT_ROOT,
    )
    torch_hub_checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    )
    lpips_linear_weights = (
        Path(lpips.__file__).resolve().parent / "weights/v0.1/alex.pth"
    )
    result = {
        "schema_version": RUNTIME_PREFLIGHT_SCHEMA,
        "pass": True,
        "gpu_uuid": observed_gpu_uuid,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "source_identity": source_identity(source_provenance),
        "source_manifest": artifact_record(args.source_manifest),
        "modules": {
            module.__name__: module_record(module)
            for module in (torch, torchvision, lpips)
        },
        "operations": {
            "cuda_matmul_finite": True,
            "cuda_convolution_finite": True,
            "lpips_backend": lpips_backend,
            "lpips_finite": True,
        },
        "lpips_weights": {
            "alexnet_imagenet": require_weight(
                torch_hub_checkpoint,
                label="Torch Hub AlexNet checkpoint",
            ),
            "lpips_alex_v0_1": require_weight(
                lpips_linear_weights,
                label="LPIPS Alex v0.1 linear weights",
            ),
        },
        "loaded_cuda_libraries": loaded_cuda_libraries(),
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(
        "FLASHGS_PUBLICATION_RUNTIME_OK "
        + json.dumps(
            {
                "gpu_uuid": observed_gpu_uuid,
                "lpips_backend": lpips_backend,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
