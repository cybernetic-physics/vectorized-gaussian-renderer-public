import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.flashgs_matched_occupancy import (
    _probe,
    compute_sample_failures,
    occupancy_failures,
)


GPU_UUID = "GPU-397e1e47-c44e-debe-5e89-fe81ef1b3647"


def snapshot() -> dict:
    return {
        "pid": 100,
        "ancestor_pids": [100, 50, 1],
        "ancestry_complete": True,
        "probe_status": {
            name: {"success": True}
            for name in (
                "gpu_inventory",
                "compute_apps",
                "process_table",
                "tmux_sessions",
            )
        },
        "gpus": [{"uuid": GPU_UUID}, {"uuid": "GPU-other"}],
        "compute_apps": [],
        "relevant_processes": [
            {"pid": 100, "command": "python benchmark.py"},
            {"pid": 50, "command": "python matrix.py"},
        ],
        "tmux_sessions": ["owned"],
        "current_tmux_session": "owned",
    }


def test_empty_preflight_is_launch_safe() -> None:
    assert occupancy_failures(
        snapshot(),
        expected_gpu_uuid=GPU_UUID,
        allow_current_gpu_process=False,
    ) == []


def test_preflight_rejects_any_cuda_process_on_either_gpu() -> None:
    state = snapshot()
    state["compute_apps"] = [
        {
            "gpu_uuid": "GPU-other",
            "pid": "202",
            "process_name": "python",
        }
    ]

    failures = occupancy_failures(
        state,
        expected_gpu_uuid=GPU_UUID,
        allow_current_gpu_process=False,
    )

    assert any("unfamiliar CUDA compute process" in item for item in failures)


def test_postflight_allows_only_the_current_gpu_process() -> None:
    state = snapshot()
    state["compute_apps"] = [
        {
            "gpu_uuid": GPU_UUID,
            "pid": "100",
            "process_name": "python",
        }
    ]

    assert occupancy_failures(
        state,
        expected_gpu_uuid=GPU_UUID,
        allow_current_gpu_process=True,
    ) == []


def test_unowned_tmux_or_relevant_process_fails() -> None:
    state = snapshot()
    state["tmux_sessions"].append("other-agent")
    state["relevant_processes"].append(
        {"pid": 999, "command": "nsys profile another-run"}
    )

    failures = occupancy_failures(
        state,
        expected_gpu_uuid=GPU_UUID,
        allow_current_gpu_process=False,
    )

    assert any("unfamiliar tmux" in item for item in failures)
    assert any("unfamiliar relevant process" in item for item in failures)


def test_unknown_telemetry_fails_closed() -> None:
    state = snapshot()
    state["probe_status"]["compute_apps"] = {
        "success": False,
        "returncode": 1,
    }

    failures = occupancy_failures(
        state,
        expected_gpu_uuid=GPU_UUID,
        allow_current_gpu_process=False,
    )

    assert "occupancy telemetry probe failed: compute_apps" in failures


def test_tmux_missing_socket_is_a_known_empty_state(monkeypatch) -> None:
    def missing_tmux_socket(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr=(
                "error connecting to /tmp/tmux-1001/default "
                "(No such file or directory)\n"
            ),
        )

    monkeypatch.setattr(subprocess, "run", missing_tmux_socket)
    output, status = _probe(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        accepted_nonzero_stderr=(
            "no server running",
            "failed to connect",
            "no such file or directory",
        ),
    )

    assert output == ""
    assert status["returncode"] == 1
    assert status["success"] is True


def test_periodic_sample_allows_only_current_job_on_expected_gpu() -> None:
    sample = {
        "probe_status": {"success": True},
        "compute_apps": [
            {"gpu_uuid": GPU_UUID, "pid": "100", "process_name": "python"}
        ],
    }
    assert compute_sample_failures(
        sample,
        expected_gpu_uuid=GPU_UUID,
        allowed_pids={100},
    ) == []

    sample["compute_apps"].append(
        {"gpu_uuid": "GPU-other", "pid": "999", "process_name": "python"}
    )
    failures = compute_sample_failures(
        sample,
        expected_gpu_uuid=GPU_UUID,
        allowed_pids={100},
    )
    assert any("sampled unfamiliar CUDA" in item for item in failures)


def test_periodic_sample_probe_failure_fails_closed() -> None:
    failures = compute_sample_failures(
        {"probe_status": {"success": False}, "compute_apps": []},
        expected_gpu_uuid=GPU_UUID,
        allowed_pids={100},
    )
    assert failures == ["sampled compute-process telemetry probe failed"]


def test_cooperative_lock_is_inherited_only_through_verified_ancestry(
    tmp_path,
) -> None:
    lock_path = tmp_path / "executor.lock"
    project_root = os.fspath(Path(__file__).resolve().parents[1])
    child_code = (
        "import json; "
        "from benchmarks.flashgs_matched_occupancy import "
        "ensure_cooperative_executor_lock; "
        "print(json.dumps(ensure_cooperative_executor_lock()))"
    )
    parent_code = """
import json
import os
import subprocess
import sys
from benchmarks.flashgs_matched_occupancy import (
    EXECUTOR_LOCK_OWNER_ENV,
    ensure_cooperative_executor_lock,
)

owned = ensure_cooperative_executor_lock()
verified = json.loads(subprocess.check_output(
    [sys.executable, "-c", sys.argv[1]], text=True, env=os.environ
))
unowned_environment = dict(os.environ)
unowned_environment.pop(EXECUTOR_LOCK_OWNER_ENV, None)
blocked = subprocess.run(
    [sys.executable, "-c", sys.argv[1]],
    text=True,
    env=unowned_environment,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
print(json.dumps({
    "owned": owned,
    "verified": verified,
    "blocked_returncode": blocked.returncode,
}))
"""
    environment = dict(os.environ)
    environment["VGR_GPU_EXECUTOR_LOCK_PATH"] = str(lock_path)
    environment["PYTHONPATH"] = project_root
    output = subprocess.check_output(
        [sys.executable, "-c", parent_code, child_code],
        text=True,
        env=environment,
        cwd=project_root,
    )
    result = json.loads(output)
    assert result["owned"]["mode"] == "direct-owner"
    assert result["verified"]["mode"] == "verified-ancestor-owner"
    assert result["blocked_returncode"] != 0
