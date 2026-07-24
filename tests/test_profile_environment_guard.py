from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

SOURCE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/profile_flashgs_matched_nsys.sh"
)


def run_environment_guard(*, torch_home: str, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as raw:
        project = Path(raw) / "project"
        scripts = project / "scripts"
        scripts.mkdir(parents=True)
        copied = scripts / SOURCE_SCRIPT.name
        shutil.copy2(SOURCE_SCRIPT, copied)
        environment = {
            "HOME": "/workspace/runtime/home",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "MATCHED_BENCHMARK_PYTHON": "/usr/bin/python3",
            "MATCHED_CUDA_HOME": "/workspace/cuda",
            "MATCHED_PUBLICATION_RUNTIME": "1",
            "PATH": "/usr/bin:/bin",
            "PROFILE_MEASURED_FRAMES": "4",
            "TORCH_HOME": torch_home,
            "USER": "vgr-publication",
        }
        environment.update(extra or {})
        return subprocess.run(
            [
                "/bin/bash",
                str(copied),
                "custom",
                "full",
                "1",
                "missing-run.json",
                "missing-trajectory.json",
                "guard-only",
            ],
            cwd=project,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )


def test_portable_nested_home_component_is_not_a_host_identity() -> None:
    result = run_environment_guard(
        torch_home="/workspace/runtime/home/torch-cache",
    )

    assert result.returncode == 2
    assert "PUBLICATION_PROFILE_ENVIRONMENT_SAFE" in result.stdout
    assert "PROFILE_MEASURED_FRAMES is frozen at 3" in result.stdout
    assert "Host-user path is forbidden" not in result.stdout


def test_actual_linux_user_home_is_rejected_before_profile_setup() -> None:
    result = run_environment_guard(torch_home="/home/example/.cache/torch")

    assert result.returncode == 2
    assert "Host-user path is forbidden during publication profiling: TORCH_HOME" in result.stdout
    assert "PUBLICATION_PROFILE_ENVIRONMENT_SAFE" not in result.stdout


def test_credential_shaped_environment_is_still_rejected() -> None:
    result = run_environment_guard(
        torch_home="/workspace/runtime/home/torch-cache",
        extra={"EXAMPLE_API_KEY": "not-a-real-secret"},
    )

    assert result.returncode == 2
    assert "Credential-shaped environment variable is forbidden" in result.stdout
    assert "PUBLICATION_PROFILE_ENVIRONMENT_SAFE" not in result.stdout
