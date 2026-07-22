"""Emit the CUDA identity used to map Torch cuda:0 to Vulkan safely."""

from __future__ import annotations

import json
import platform
import re
import sys
from importlib import metadata


UUID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def normalize_gpu_uuid(value: object) -> str:
    """Normalize NVIDIA/Torch/Vulkan UUID spellings to 32 lowercase hex digits."""

    normalized = str(value).strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "")
    if not UUID_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid GPU UUID: {value!r}")
    return normalized


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access CUDA.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Torch reports no CUDA devices.")

    properties = torch.cuda.get_device_properties(0)
    raw_uuid = getattr(properties, "uuid", None)
    if raw_uuid is None:
        raise RuntimeError("This Torch build does not expose cuda:0's UUID; refusing an index-only match.")
    result = {
        "schema_version": "isaac-gpu-identity/v1",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "isaacsim": metadata.version("isaacsim"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_uuid_raw": str(raw_uuid),
        "cuda_device_uuid": normalize_gpu_uuid(raw_uuid),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    print("ISAAC_GPU_IDENTITY " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
