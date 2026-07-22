import json
import subprocess
import sys
from pathlib import Path

import pytest

from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (
    SOURCE_MANIFEST_SCHEMA,
    artifact_record,
    load_verified_source_manifest,
    verify_node_occupancy_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_source_manifest_binds_head_status_and_file_set(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.com")
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(repo, "add", "source.py")
    git(repo, "commit", "-m", "fixture")
    manifest_path = tmp_path / "source-manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/write_source_manifest.py"),
            "--repo-root",
            str(repo),
            "--output",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SOURCE_MANIFEST_SCHEMA
    verified = load_verified_source_manifest(
        manifest_path,
        project_root=repo,
    )
    assert verified["source_tree_sha256"] == manifest["source_tree_sha256"]

    (repo / "shadow.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        load_verified_source_manifest(manifest_path, project_root=repo)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/write_source_manifest.py"),
            "--repo-root",
            str(repo),
            "--output",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dirty_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert dirty_manifest["dirty"] is True
    load_verified_source_manifest(manifest_path, project_root=repo)

    dirty_manifest["dirty"] = False
    manifest_path.write_text(json.dumps(dirty_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dirty flag"):
        load_verified_source_manifest(manifest_path, project_root=repo)

    dirty_manifest["dirty"] = True
    dirty_manifest["diff_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(dirty_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="diff fingerprint"):
        load_verified_source_manifest(manifest_path, project_root=repo)


def test_node_occupancy_evidence_fails_closed(tmp_path):
    path = tmp_path / "occupancy.json"
    payload = {
        "schema_version": "flashgs-matched-node-occupancy-v2",
        "expected_gpu_uuid": "GPU-test",
        "executor_control": {
            "scope": "all-visible-gpus",
            "cooperative_node_wide_lock": {
                "schema_version": "vgr-cooperative-gpu-lock-v1",
                "lock_observed_held": True,
                "pass": True,
            },
        },
        "sampled_compute_process_telemetry": {
            "schema_version": (
                "flashgs-matched-compute-process-sampling-v1"
            ),
            "coverage": "periodic-samples-not-continuous-observation",
            "sample_count": 2,
            "pass": True,
        },
        "pass": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    record = artifact_record(path)

    assert verify_node_occupancy_evidence(
        record, expected_gpu_uuid="GPU-test"
    ) == payload

    payload["pass"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded hash"):
        verify_node_occupancy_evidence(record, expected_gpu_uuid="GPU-test")

    record = artifact_record(path)
    with pytest.raises(ValueError, match="did not pass"):
        verify_node_occupancy_evidence(record, expected_gpu_uuid="GPU-test")
