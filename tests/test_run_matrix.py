from __future__ import annotations

import io
import signal
import subprocess
from pathlib import Path

import pytest

import benchmarks.run_matrix as run_matrix
from benchmarks.run_matrix import (
    child_environment,
    gsplat_rasterization_arguments,
    load_passing_result,
    matrix_pass,
    mode_result_filename,
    remove_stale_files,
    run_bounded_child,
)


def test_child_environment_prioritizes_current_checkout(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/stale/editable/src")

    environment = child_environment()
    first_entry = environment["PYTHONPATH"].split(":", 1)[0]

    expected = Path(__file__).resolve().parents[1] / "src"
    assert Path(first_entry) == expected


def test_gsplat_matrix_arguments_match_custom_covariance_epsilon() -> None:
    assert gsplat_rasterization_arguments(
        rasterize_mode="antialiased",
        covariance_epsilon=0.3,
    ) == [
        "--rasterize-mode",
        "antialiased",
        "--eps2d",
        "0.3",
    ]
    assert (
        gsplat_rasterization_arguments(
            rasterize_mode="classic",
            covariance_epsilon=0.0,
        )[-1]
        == "0.0"
    )


def test_matrix_result_filename_cannot_collide_across_modes() -> None:
    classic = mode_result_filename(
        "synthetic-small-b8-128x128",
        "classic",
    )
    antialiased = mode_result_filename(
        "synthetic-small-b8-128x128",
        "antialiased",
    )

    assert classic == ("synthetic-small-b8-128x128-rasterize-classic.json")
    assert antialiased == ("synthetic-small-b8-128x128-rasterize-antialiased.json")
    assert classic != antialiased


def test_load_passing_result_rejects_failed_child(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text('{"pass": false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="did not pass"):
        load_passing_result(result_path)

    result_path.write_text('{"pass": true}\n', encoding="utf-8")
    assert load_passing_result(result_path)["pass"] is True


def test_matrix_pass_requires_every_expected_passing_result() -> None:
    assert matrix_pass(
        expected_result_count=1,
        results=[{"pass": True}],
        failures=[],
    )
    assert not matrix_pass(
        expected_result_count=2,
        results=[{"pass": True}],
        failures=[],
    )
    assert not matrix_pass(
        expected_result_count=1,
        results=[{"pass": False}],
        failures=[],
    )
    assert not matrix_pass(
        expected_result_count=0,
        results=[],
        failures=[],
    )


def test_remove_stale_files_requires_current_run_artifacts(tmp_path: Path) -> None:
    stale = tmp_path / "stale.json"
    stale.write_text('{"pass": true}\n', encoding="utf-8")

    remove_stale_files(stale, tmp_path / "missing.json")

    assert not stale.exists()


def test_run_bounded_child_terminates_owned_group_on_timeout(monkeypatch) -> None:
    class FakeProcess:
        pid = 12345
        returncode: int | None = None

        def wait(self, timeout: float) -> int:
            raise subprocess.TimeoutExpired(["child"], timeout)

    process = FakeProcess()
    launch: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        launch["command"] = command
        launch.update(kwargs)
        return process

    def fake_terminate(target) -> None:
        assert target is process
        process.returncode = -15

    monkeypatch.setattr(run_matrix.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_matrix, "terminate_process_group", fake_terminate)

    returncode, timed_out = run_bounded_child(
        ["child"],
        log_file=io.StringIO(),
        timeout_seconds=1.0,
    )

    assert timed_out
    assert returncode == -15
    assert launch["start_new_session"] is True


def test_terminate_process_group_kills_survivors_after_parent_exits(
    monkeypatch,
) -> None:
    class FakeProcess:
        pid = 12345

        def wait(self, timeout: float | None = None) -> int:
            return -15

    signals: list[int] = []
    monkeypatch.setattr(
        run_matrix.os,
        "killpg",
        lambda _pid, value: signals.append(value),
    )

    run_matrix.terminate_process_group(FakeProcess())  # type: ignore[arg-type]

    assert signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.parametrize(
    ("extra_args", "expected_eps2d"),
    [
        ([], "0.3"),
        (["--covariance-epsilon", "0"], "0.0"),
    ],
)
def test_matrix_antialiased_compact_passes_resolved_eps2d_to_gsplat(
    monkeypatch,
    tmp_path: Path,
    extra_args: list[str],
    expected_eps2d: str,
) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("test protocol\n", encoding="utf-8")
    commands: list[list[str]] = []

    def capture_command(command: list[str], **_kwargs) -> None:
        commands.append(command)

    monkeypatch.setattr(run_matrix.subprocess, "check_call", capture_command)
    monkeypatch.setattr(
        run_matrix.sys,
        "argv",
        [
            "run_matrix.py",
            "--protocol",
            str(protocol),
            "--output",
            str(tmp_path / "output"),
            "--quick",
            "--implementations",
            "gsplat",
            "--compact-projection-cache",
            "--tile-size",
            "1",
            "--rasterize-mode",
            "antialiased",
            *extra_args,
        ],
    )

    run_matrix.main()

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--eps2d") + 1] == expected_eps2d
    assert command[command.index("--rasterize-mode") + 1] == "antialiased"
