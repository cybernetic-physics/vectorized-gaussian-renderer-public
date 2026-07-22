from __future__ import annotations

import json
import signal

import pytest

from benchmarks.run_ovrtx_matrix import (
    fabric_scene_delegate_record,
    reusable_cases,
    write_manifest,
)
import benchmarks.run_ovrtx_matrix as run_ovrtx_matrix


def test_reusable_cases_keeps_only_verified_measured_rows(tmp_path) -> None:
    output = tmp_path / "matrix"
    case_dir = output / "synthetic-small-perspective-b1-128x128"
    case_dir.mkdir(parents=True)
    (case_dir / "result.json").write_text(
        json.dumps({"status": {"pass": True}}),
        encoding="utf-8",
    )
    (case_dir / "run-manifest.json").write_text(
        json.dumps({"result_count": 1, "results": ["result"]}),
        encoding="utf-8",
    )
    configuration = {"projection_mode": "perspective"}
    manifest = {
        "configuration": configuration,
        "cases": [
            {
                "case": "synthetic-small-perspective-b1-128x128",
                "status": "MEASURED",
            },
            {
                "case": "synthetic-small-perspective-b8-128x128",
                "status": "FAILED",
            },
        ],
    }
    manifest_path = output / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reusable = reusable_cases(manifest_path, configuration=configuration)

    assert list(reusable) == ["synthetic-small-perspective-b1-128x128"]


def test_matrix_manifest_records_fabric_scene_delegate(tmp_path) -> None:
    manifest_path = tmp_path / "run-manifest.json"
    record = fabric_scene_delegate_record()
    write_manifest(
        manifest_path,
        created_utc="2026-07-17T00:00:00+00:00",
        cases=[
            {
                "case": "synthetic-small-perspective-b1-128x128",
                "status": "MEASURED",
                "fabric_scene_delegate": record,
            }
        ],
        complete=True,
        configuration={
            "fabric_scene_delegate": record,
            "projection_mode": "perspective",
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for payload in (
        manifest["configuration"]["fabric_scene_delegate"],
        manifest["cases"][0]["fabric_scene_delegate"],
    ):
        assert payload["state"] == "not_applicable"
        assert payload["citable_as_fsd_evidence"] is False


def test_resume_rejects_manifest_without_fsd_record(tmp_path) -> None:
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "configuration": {"projection_mode": "perspective"},
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configuration to match"):
        reusable_cases(
            manifest_path,
            configuration={
                "fabric_scene_delegate": fabric_scene_delegate_record(),
                "projection_mode": "perspective",
            },
        )


def test_timeout_cleanup_kills_surviving_process_group_members(monkeypatch) -> None:
    class FakeProcess:
        pid = 12345

        def wait(self, timeout: float | None = None) -> int:
            return -15

    signals: list[int] = []
    monkeypatch.setattr(
        run_ovrtx_matrix.os,
        "killpg",
        lambda _pid, value: signals.append(value),
    )

    run_ovrtx_matrix.terminate_process_group(  # type: ignore[arg-type]
        FakeProcess()
    )

    assert signals == [signal.SIGTERM, signal.SIGKILL]
