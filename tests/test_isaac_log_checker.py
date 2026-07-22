"""Regression tests for the shell-level Isaac/OVRTX log acceptance check."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKER = PROJECT_ROOT / ".agents/skills/test-isaac-graphics/scripts/check-isaac-log.sh"
SUCCESS_MARKER = "RENDER_COMPLETE_OK"


def _check(
    tmp_path: Path,
    text: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log_path = tmp_path / "combined.log"
    log_path.write_text(text, encoding="utf-8")
    return subprocess.run(
        [str(CHECKER), str(log_path), SUCCESS_MARKER],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_checker_accepts_clean_marker_and_benign_headless_warning(tmp_path: Path) -> None:
    completed = _check(
        tmp_path,
        f"[Warning] GLFW initialization failed because DISPLAY is unset\n{SUCCESS_MARKER}\n",
    )

    assert completed.returncode == 0, completed.stderr
    assert "ISAAC_LOG_OK" in completed.stdout


def test_checker_falls_back_to_grep_when_ripgrep_is_absent(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("bash", "grep"):
        executable = shutil.which(name)
        assert executable is not None
        (tools / name).symlink_to(Path(executable).resolve())
    env = dict(os.environ, PATH=str(tools))

    completed = _check(tmp_path, f"{SUCCESS_MARKER}\n", env=env)

    assert completed.returncode == 0, completed.stderr
    assert "ISAAC_LOG_OK" in completed.stdout


@pytest.mark.parametrize(
    "fatal_line",
    [
        "TaskGroup::~TaskGroup(): Assertion (empty()) failed: Destroying busy TaskGroup!",
        "Fatal Python error: Aborted",
        "Aborted",
        "Aborted (core dumped)",
        "terminated by SIGABRT",
        "process exited from signal 6",
        "timeout: monitored command dumped core",
        "terminate called after throwing an instance of std::runtime_error",
        "Assertion Failed in renderer shutdown",
        "Exception during renderer cleanup in __del__",
        "Warning: Failed to shutdown ovrtx library: internal error",
        "Warning: Exception during ovrtx library cleanup: native failure",
        "CUDA error: an illegal memory access was encountered",
        "Segmentation fault",
        "[Error] [omni.rtx] VkResult: ERROR_INCOMPATIBLE_DRIVER",
        "[Error] [omni.rtx] vkCreateInstance failed. Vulkan 1.1 is not supported",
        "[Error] [omni.rtx] carb::graphics::createInstance failed.",
        "Failed to create any GPU devices, including compatibility mode",
        "GPU Foundation is not initialized!",
        "unable to get a valid CUDA device id from the renderer",
        "error while loading shared libraries: libXt.so.6: cannot open shared object file",
    ],
)
def test_checker_rejects_native_failure_even_after_success_marker(
    tmp_path: Path,
    fatal_line: str,
) -> None:
    completed = _check(tmp_path, f"{SUCCESS_MARKER}\n{fatal_line}\n")

    assert completed.returncode == 1
    assert "ISAAC_LOG_INVALID fatal_pattern_found" in completed.stderr


def test_checker_rejects_missing_success_marker(tmp_path: Path) -> None:
    completed = _check(tmp_path, "render stopped without its completion marker\n")

    assert completed.returncode == 1
    assert "ISAAC_LOG_INVALID missing_marker" in completed.stderr


@pytest.mark.parametrize(
    "near_match",
    [
        f"prefix-{SUCCESS_MARKER}",
        f"{SUCCESS_MARKER}-suffix",
        f" {SUCCESS_MARKER}",
        f"{SUCCESS_MARKER} ",
    ],
)
def test_checker_rejects_success_marker_as_a_substring(
    tmp_path: Path,
    near_match: str,
) -> None:
    completed = _check(tmp_path, f"{near_match}\n")

    assert completed.returncode == 1
    assert "ISAAC_LOG_INVALID missing_marker" in completed.stderr
