from __future__ import annotations

import pytest

from scripts.isaac_gpu_identity import normalize_gpu_uuid


@pytest.mark.parametrize(
    "value",
    [
        "GPU-1da70b0f-240d-f25d-77a0-8a437043f744",
        "1da70b0f240df25d77a08a437043f744",
        "1DA70B0F-240D-F25D-77A0-8A437043F744",
    ],
)
def test_normalize_gpu_uuid_accepts_nvidia_and_vulkan_spellings(value: str) -> None:
    assert normalize_gpu_uuid(value) == "1da70b0f240df25d77a08a437043f744"


@pytest.mark.parametrize("value", ["", "GPU-1234", "not-a-uuid"])
def test_normalize_gpu_uuid_rejects_partial_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid GPU UUID"):
        normalize_gpu_uuid(value)
