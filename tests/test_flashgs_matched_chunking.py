from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.run_flashgs_gsplat_oracle import (
    current_process_gpu_uuid as oracle_process_gpu_uuid,
    verify_gsplat_support_contract,
)
from benchmarks.run_flashgs_matched import (
    build_backend,
    create_timed_capacity_audit,
    current_device_counter_snapshots,
    current_process_gpu_uuid,
    distribution,
    intersection_capacity_demand,
    finalize_timed_capacity_audit,
    update_timed_capacity_audit,
    visible_capacity_demand,
)
from benchmarks.run_flashgs_matched_matrix import (
    FLASHGS_UPSTREAM_COMMIT,
    PROJECT_ROOT as MATRIX_PROJECT_ROOT,
    RUNTIME_PREFLIGHT_SCHEMA,
    SRC_ROOT as MATRIX_SRC_ROOT,
    capacity_calibration_schedule,
    custom_physical_view_args,
    publication_child_environment,
    require_publication_safe_environment,
    require_publication_safe_paths,
    require_publication_working_directory,
    require_runtime_preflight_identity,
    require_adapter_attestation_identity,
    require_b128_repeat_identity,
    require_capacity_calibration_identity,
    require_direct_timed_capacity_verification,
    require_flashgs_demand_survey_identity,
    require_profile_identity,
    require_publication_source_manifest_location,
    require_run_identity,
    result_root_has_compatible_invocation_or_is_pristine,
    write_or_validate_matrix_invocation,
)
from isaacsim_gaussian_renderer import CustomCudaBackend, FlashGSBackend
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (
    CAPACITY_CALIBRATION_SCHEMA,
    CAPACITY_CONSUMPTION_SCHEMA,
    FLASHGS_DEMAND_SURVEY_SCHEMA,
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PRIMARY_BATCHES,
    PRIMARY_TRAJECTORY_IDS,
    audit_flashgs_demand_counter_source,
    artifact_record,
    derive_flashgs_prefix_capacities,
    primary_fidelity_selection,
    primary_execution_schedule_failures,
    primary_max_physical_views,
    prove_trajectory_prefixes,
    source_identity,
    trajectory_artifacts,
    verify_flashgs_demand_counter_source_audit,
    verify_trajectory_prefix_proof,
)
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (
    CameraTrajectory,
    save_trajectory,
)
from isaacsim_gaussian_renderer.flashgs_baseline_contract import (
    FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
    FLASHGS_MATCHED_PORT_CLASSIFICATION,
)


def runner_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "renderer": "custom",
        "initial_visible_per_view": 170_000,
        "initial_intersections_per_view": 200_000,
        "flashgs_initial_intersections": 200_000,
        "custom_max_physical_views": 128,
        "custom_depth_buckets": 128,
        "custom_depth_bucket_group": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runner_sizes_initial_custom_workspace_from_physical_batch() -> None:
    backend = build_backend(
        runner_args(),
        batch=1024,
        near_plane=0.01,
        far_plane=100.0,
    )

    assert isinstance(backend, CustomCudaBackend)
    assert backend.max_physical_views == 128
    assert backend.max_visible_records == 128 * 170_000
    assert backend.max_intersections == 128 * 200_000
    assert backend.gaussian_support_sigma == MATCHED_GAUSSIAN_SUPPORT_SIGMA


def test_runner_starts_at_calibrated_capacity_without_reservation() -> None:
    backend = build_backend(
        runner_args(),
        batch=1024,
        near_plane=0.01,
        far_plane=100.0,
        installed_capacity={
            "visible_records": 12_345,
            "intersections": 67_890,
        },
    )

    assert isinstance(backend, CustomCudaBackend)
    assert backend.max_visible_records == 12_345
    assert backend.max_intersections == 67_890


def test_runner_matches_pinned_gsplat_support_for_both_backends() -> None:
    custom = build_backend(
        runner_args(custom_max_physical_views=None),
        batch=1,
        near_plane=0.01,
        far_plane=100.0,
    )
    flashgs = build_backend(
        runner_args(renderer="flashgs"),
        batch=1,
        near_plane=0.01,
        far_plane=100.0,
    )

    assert custom.gaussian_support_sigma == MATCHED_GAUSSIAN_SUPPORT_SIGMA
    assert flashgs.gaussian_support_sigma == MATCHED_GAUSSIAN_SUPPORT_SIGMA


def test_runner_separates_logical_work_from_physical_capacity() -> None:
    backend = CustomCudaBackend(
        fixed_capacity_sort=True,
        max_physical_views=128,
        adaptive_capacity=False,
        native_module=object(),
        allow_cpu_for_tests=True,
    )
    logical = torch.tensor([800, 1600, 0, 0, 80], dtype=torch.int64)
    physical = torch.tensor([120, 250, 0, 0, 10], dtype=torch.int64)
    backend.workspace = {
        "counters": logical,
        "physical_capacity_counters": physical,
    }

    work, capacity = current_device_counter_snapshots(backend)
    counters = {
        "max_visible_gaussians": int(work[0]),
        "max_generated_intersections": int(work[1]),
        "max_visible_gaussians_per_physical_chunk": int(capacity[0]),
        "max_generated_intersections_per_physical_chunk": int(capacity[1]),
    }

    assert work is logical
    assert capacity is physical
    assert visible_capacity_demand(backend, counters) == 120
    assert intersection_capacity_demand(backend, counters) == 250


def test_primary_physical_schedule_is_exact() -> None:
    assert primary_max_physical_views("custom", 1024) == 128
    assert primary_max_physical_views("custom", 512) == 128
    assert primary_max_physical_views("custom", 256) is None
    assert primary_max_physical_views("flashgs", 1024) is None


def test_flashgs_execution_and_capacity_submission_counts_are_cross_bound() -> None:
    execution = {
        "logical_batch": 512,
        "physical_batch": 1,
        "native_submissions_per_logical_batch": 512,
    }
    capacity = dict(execution)

    assert primary_execution_schedule_failures("flashgs", 512, execution, capacity) == []
    capacity["native_submissions_per_logical_batch"] = 4
    assert primary_execution_schedule_failures("flashgs", 512, execution, capacity) == [
        "flashgs capacity native submission count is inconsistent"
    ]


def test_capacity_calibrations_fail_fast_on_memory_risk_rows() -> None:
    schedule = capacity_calibration_schedule((1, 8, 32, 64, 128, 256, 512, 1024))

    assert schedule[:2] == ((1024, "custom"), (512, "custom"))
    assert len(schedule) == 8
    assert len(set(schedule)) == 8
    assert set(schedule) == {(batch, "custom") for batch in (1, 8, 32, 64, 128, 256, 512, 1024)}


def test_matrix_command_uses_frozen_physical_schedule() -> None:
    assert custom_physical_view_args("custom", 256, 128) == []
    assert custom_physical_view_args("custom", 512, 128) == [
        "--custom-max-physical-views",
        "128",
    ]
    assert custom_physical_view_args("custom", 1024, 128) == [
        "--custom-max-physical-views",
        "128",
    ]
    assert custom_physical_view_args("flashgs", 1024, 128) == []
    with pytest.raises(ValueError, match="requires P128"):
        custom_physical_view_args("custom", 512, 512)


def test_matrix_result_root_guard_and_invocation_are_immutable(tmp_path: Path) -> None:
    output_root = tmp_path / "result"
    source_manifest = output_root / "provenance/source-manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text("{}\n", encoding="utf-8")
    assert (
        result_root_has_compatible_invocation_or_is_pristine(
            output_root,
            source_manifest=source_manifest,
        )
        is False
    )

    invocation_path = output_root / "provenance/matrix-invocation.json"
    document = {
        "schema_version": "flashgs-matched-matrix-invocation-v1",
        "created_at": "2026-07-22T12:00:00+00:00",
        "argv": ["python", "matrix.py"],
    }
    write_or_validate_matrix_invocation(invocation_path, document, resume=False)
    original_bytes = invocation_path.read_bytes()
    assert (
        result_root_has_compatible_invocation_or_is_pristine(
            output_root,
            source_manifest=source_manifest,
        )
        is True
    )
    resumed = {**document, "created_at": "2026-07-22T13:00:00+00:00"}
    write_or_validate_matrix_invocation(invocation_path, resumed, resume=True)
    assert invocation_path.read_bytes() == original_bytes

    mismatched = {**resumed, "argv": ["python", "other.py"]}
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_or_validate_matrix_invocation(invocation_path, mismatched, resume=True)
    assert invocation_path.read_bytes() == original_bytes


def test_matrix_requires_result_owned_source_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "result"
    expected = output_root / "provenance/source-manifest.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("{}\n", encoding="utf-8")
    require_publication_source_manifest_location(output_root, expected)
    with pytest.raises(ValueError, match="OUTPUT_ROOT/provenance/source-manifest.json"):
        require_publication_source_manifest_location(
            output_root,
            tmp_path / "external-source-manifest.json",
        )

    external = tmp_path / "external-source-manifest.json"
    external.write_text("{}\n", encoding="utf-8")
    expected.unlink()
    expected.symlink_to(external)
    with pytest.raises(ValueError, match="non-symlink"):
        require_publication_source_manifest_location(output_root, expected)


def test_matrix_result_root_guard_rejects_mixed_or_symlinked_roots(tmp_path: Path) -> None:
    output_root = tmp_path / "mixed"
    output_root.mkdir()
    (output_root / "old.log").write_text("old evidence\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-pristine"):
        result_root_has_compatible_invocation_or_is_pristine(
            output_root,
            source_manifest=tmp_path / "source-manifest.json",
        )

    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="may not be a symlink"):
        result_root_has_compatible_invocation_or_is_pristine(
            alias,
            source_manifest=tmp_path / "source-manifest.json",
        )


def test_reused_adapter_attestation_must_match_source_identity(tmp_path: Path) -> None:
    adapter_diff = tmp_path / "adapter.patch"
    adapter_diff.write_text("matched adapter diff\n", encoding="utf-8")
    render_source = tmp_path / "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
    render_source.parent.mkdir(parents=True)
    render_source.write_text("repaired render source\n", encoding="utf-8")
    provenance = {
        "manifest": {"sha256": "a" * 64},
        "source_tree_sha256": "b" * 64,
        "head": "c" * 40,
        "dirty": False,
        "diff_sha256": "d" * 64,
    }
    attestation = {
        "schema_version": FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
        "pass": True,
        "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
        "upstream_clean": True,
        "source_provenance": provenance,
        "adapter_diff": artifact_record(adapter_diff),
        "baseline_classification": FLASHGS_MATCHED_PORT_CLASSIFICATION,
        "correctness_repair_audit": {
            "render_source": artifact_record(render_source),
            "checks": {
                "repaired_slot_0_predicate_exactly_once": True,
                "repaired_slot_1_predicate_exactly_once": True,
                "buggy_slot_0_predicate_absent": True,
                "buggy_slot_1_predicate_absent": True,
                "slot_0_load_guard_present": True,
                "slot_1_load_guard_present": True,
            },
            "pass": True,
        },
    }

    require_adapter_attestation_identity(
        attestation,
        expected_source_identity=source_identity(provenance),
        project_root=tmp_path,
    )

    stale = {
        **attestation,
        "source_provenance": {**provenance, "head": "e" * 40},
    }
    with pytest.raises(RuntimeError, match="invalid or stale"):
        require_adapter_attestation_identity(
            stale,
            expected_source_identity=source_identity(provenance),
            project_root=tmp_path,
        )


def test_reused_profile_must_use_fixed_three_frame_control(tmp_path: Path) -> None:
    unprofiled = tmp_path / "unprofiled.json"
    unprofiled.write_text("{}\n", encoding="utf-8")
    profile = {
        "renderer": "custom",
        "output_contract": "rgb",
        "batch": 1,
        "captured_frames": 3,
        "unprofiled": {"run_artifact": artifact_record(unprofiled)},
        "profiled_control": {
            "nvtx_measured_frame_calls": 3,
            "no_measured_range_allocator_calls": True,
        },
    }

    require_profile_identity(
        profile,
        renderer="custom",
        output_contract="rgb",
        batch=1,
        unprofiled_run=unprofiled,
    )

    profile["captured_frames"] = 100
    with pytest.raises(RuntimeError, match="fixed three-frame"):
        require_profile_identity(
            profile,
            renderer="custom",
            output_contract="rgb",
            batch=1,
            unprofiled_run=unprofiled,
        )


PUBLICATION_ENTRYPOINTS = (
    "benchmarks/compare_flashgs_matched_fidelity.py",
    "benchmarks/debug_flashgs_pipeline.py",
    "benchmarks/generate_flashgs_dynamic_contract.py",
    "benchmarks/run_flashgs_gsplat_oracle.py",
    "benchmarks/run_flashgs_matched.py",
    "benchmarks/run_flashgs_matched_matrix.py",
    "benchmarks/run_upstream_faithful_flashgs_control.py",
    "benchmarks/summarize_flashgs_matched.py",
    "benchmarks/summarize_flashgs_profile.py",
    "benchmarks/verify_upstream_faithful_flashgs_parity.py",
    "scripts/write_flashgs_adapter_attestation.py",
    "scripts/write_flashgs_profile_wrapper_evidence.py",
)


@pytest.mark.parametrize("relative_path", PUBLICATION_ENTRYPOINTS)
def test_publication_entrypoint_defeats_hostile_pythonpath(
    tmp_path: Path,
    relative_path: str,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    hostile_root = tmp_path / "hostile"
    hostile_package = hostile_root / "isaacsim_gaussian_renderer"
    hostile_package.mkdir(parents=True)
    hostile_package.joinpath("__init__.py").write_text(
        "raise RuntimeError('hostile stale checkout imported')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(hostile_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / relative_path),
            "--help",
        ],
        cwd=project_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_matrix_child_environment_force_prefixes_current_checkout() -> None:
    environment = publication_child_environment(
        {
            "PYTHONPATH": os.pathsep.join(("/stale/repository/src", str(MATRIX_SRC_ROOT))),
            "PROJECT_ROOT": "/stale/repository",
            "VGR_PROJECT_ROOT": "/stale/repository",
        }
    )

    python_paths = environment["PYTHONPATH"].split(os.pathsep)
    assert python_paths[:2] == [str(MATRIX_SRC_ROOT), str(MATRIX_PROJECT_ROOT)]
    assert python_paths.count(str(MATRIX_SRC_ROOT)) == 1
    assert environment["PROJECT_ROOT"] == str(MATRIX_PROJECT_ROOT)
    assert environment["VGR_PROJECT_ROOT"] == str(MATRIX_PROJECT_ROOT)


def test_publication_environment_rejects_secrets_and_host_user_paths() -> None:
    require_publication_safe_environment(
        {
            "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
            "HOME": "/workspace/publication-home",
        }
    )
    with pytest.raises(ValueError, match="credential-shaped.*JUPYTER_TOKEN"):
        require_publication_safe_environment(
            {"JUPYTER_TOKEN": "A" * 64, "PATH": "/usr/bin"}
        )
    with pytest.raises(ValueError, match="host-user paths.*PATH"):
        require_publication_safe_environment(
            {"PATH": "/home/alice/private/bin:/usr/bin"}
        )
    with pytest.raises(ValueError, match="host-user paths.*HOME"):
        require_publication_safe_environment({"HOME": "/home/alice"})


def test_publication_launch_paths_reject_host_user_identity() -> None:
    require_publication_safe_paths(
        {"source": "/workspace/source", "output": "/workspace/output"}
    )
    with pytest.raises(ValueError, match="host-user identities.*python"):
        require_publication_safe_paths(
            {"python": "/Users/alice/private/venv/bin/python"}
        )
    with pytest.raises(ValueError, match="host-user identities.*cwd"):
        require_publication_safe_paths({"cwd": "/Users/alice"})


def test_publication_working_directory_must_be_frozen_checkout() -> None:
    require_publication_working_directory(MATRIX_PROJECT_ROOT)
    with pytest.raises(ValueError, match="working directory differs"):
        require_publication_working_directory(MATRIX_PROJECT_ROOT.parent)


def test_publication_runtime_preflight_is_content_bound(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    source = {
        "manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "head": "c" * 40,
        "dirty": False,
        "diff_sha256": None,
    }
    modules = {}
    for name in ("torch", "torchvision", "lpips"):
        path = tmp_path / f"{name}.py"
        path.write_text(f"# {name}\n", encoding="utf-8")
        modules[name] = {"name": name, "origin": str(path), "version": "1"}
    weights = {}
    for name in ("alexnet_imagenet", "lpips_alex_v0_1"):
        path = tmp_path / f"{name}.pth"
        path.write_bytes(name.encode("ascii"))
        weights[name] = artifact_record(path)
    preflight = {
        "schema_version": RUNTIME_PREFLIGHT_SCHEMA,
        "pass": True,
        "gpu_uuid": "GPU-test",
        "source_identity": source,
        "source_manifest": artifact_record(source_manifest),
        "modules": modules,
        "operations": {
            "cuda_convolution_finite": True,
            "cuda_matmul_finite": True,
            "lpips_backend": "lpips-alex",
            "lpips_finite": True,
        },
        "lpips_weights": weights,
        "loaded_cuda_libraries": ["/usr/lib/libcublas.so.12"],
    }

    require_runtime_preflight_identity(
        preflight,
        expected_gpu_uuid="GPU-test",
        expected_source_identity=source,
        source_manifest=source_manifest,
    )
    Path(weights["lpips_alex_v0_1"]["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="weight lpips_alex_v0_1"):
        require_runtime_preflight_identity(
            preflight,
            expected_gpu_uuid="GPU-test",
            expected_source_identity=source,
            source_manifest=source_manifest,
        )


def test_profile_wrapper_rebinds_project_root_before_remote_environment() -> None:
    script = Path(__file__).resolve().parents[1] / ("scripts/profile_flashgs_matched_nsys.sh")
    text = script.read_text(encoding="utf-8")

    assignment = text.index('PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"')
    source = text.index('source "$(dirname "$0")/remote_env.sh"')
    assert assignment < source
    assert 'export VGR_PROJECT_ROOT="$PROJECT_ROOT"' in text[:source]
    assert 'profile_root="$expected_profile_root"' in text
    assert 'export PROFILE_ROOT="$profile_root"' in text
    assert "measured_frames=3" in text
    assert 'export PROFILE_MEASURED_FRAMES="$measured_frames"' in text
    assert 'runner_args+=(--flashgs-demand-survey "$capacity_artifact")' in text
    assert 'runner_args+=(--capacity-calibration "$capacity_artifact")' in text
    assert "flashgs-matched-flashgs-demand-survey-v1" in text
    assert "PUBLICATION_PROFILE_ENVIRONMENT_SAFE" in text
    assert 'if [[ "${MATCHED_PUBLICATION_RUNTIME:-0}" == "1" ]]' in text


def test_publication_launcher_uses_empty_explicit_environment() -> None:
    script = Path(__file__).resolve().parents[1] / (
        "scripts/launch_flashgs_publication_matrix.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert 'exec env -i "${safe_environment[@]}"' in text
    assert '"HOME=$safe_home"' in text
    assert '"USER=vgr-publication"' in text
    assert '"PYTHONNOUSERSITE=1"' in text
    assert '"TMPDIR=$safe_tmp"' in text
    assert '"XDG_RUNTIME_DIR=$safe_xdg"' in text
    assert '"MATCHED_PUBLICATION_RUNTIME=1"' in text
    assert 'safe_pythonpath="$PROJECT_ROOT/src:$gsplat_source"' in text
    assert 'cd "$PROJECT_ROOT"' in text
    assert text.index('cd "$PROJECT_ROOT"') < text.index(
        'exec env -i "${safe_environment[@]}"'
    )
    assert "LD_LIBRARY_PATH=" not in text
    assert "ISAACSIM_ML_PREBUNDLE" not in text
    assert "OVRTX_ROOT" not in text
    assert "PUBLICATION_MATRIX_ENVIRONMENT_SAFE" in text


def test_primary_fidelity_selection_covers_steps_and_chunk_boundaries() -> None:
    selection = primary_fidelity_selection(1024)
    selected_steps = {step for step, _camera in selection}
    selected_cameras = {camera for _step, camera in selection}

    assert selected_steps == {8, 57, 107}
    assert len(selection) == 66
    assert {0, 1023}.issubset(selected_cameras)
    for boundary in range(128, 1024, 128):
        assert {boundary - 1, boundary}.issubset(selected_cameras)
    for step in selected_steps:
        assert {camera for selected_step, camera in selection if selected_step == step} == selected_cameras

    assert primary_fidelity_selection(1) == ((8, 0), (57, 0), (107, 0))
    assert {step for step, _camera in primary_fidelity_selection(64)} == {107}

    b512_selection = primary_fidelity_selection(512)
    assert {step for step, _camera in b512_selection} == {8, 57, 107}
    b512_cameras = {camera for _step, camera in b512_selection}
    for boundary in range(128, 512, 128):
        assert {boundary - 1, boundary}.issubset(b512_cameras)


def test_oracle_binds_pinned_gsplat_support_macro(tmp_path) -> None:
    header = tmp_path / "gsplat/cuda/include/Common.h"
    header.parent.mkdir(parents=True)
    header.write_text("#define GAUSSIAN_EXTEND 3.33f\n", encoding="utf-8")

    contract = verify_gsplat_support_contract(tmp_path)

    assert contract["macro"] == "GAUSSIAN_EXTEND"
    assert contract["value"] == MATCHED_GAUSSIAN_SUPPORT_SIGMA

    header.write_text("#define GAUSSIAN_EXTEND 3.0f\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Gaussian support differs"):
        verify_gsplat_support_contract(tmp_path)


def test_gpu_uuid_resolves_before_first_cuda_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(uuid="397e1e47-c44e-debe-5e89-fe81ef1b3647"),
    )

    assert current_process_gpu_uuid() == ("GPU-397e1e47-c44e-debe-5e89-fe81ef1b3647")
    assert oracle_process_gpu_uuid() == ("GPU-397e1e47-c44e-debe-5e89-fe81ef1b3647")


def test_primary_matrix_distribution_preserves_ordered_raw_samples() -> None:
    values = [float(index) for index in range(100)]
    summary = distribution(values)

    assert summary["count"] == 100
    assert summary["samples"] == values
    assert "ci95" not in summary
    assert "primary_matrix_familywise95_component" not in summary


def _run_identity_fixture(
    tmp_path: Path,
    *,
    max_physical_views: int | None,
) -> tuple[dict, dict]:
    provenance = {
        "manifest": {"sha256": "manifest"},
        "source_tree_sha256": "tree",
        "head": "head",
        "dirty": False,
        "diff_sha256": "diff",
    }
    runtime = {
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "gpu_uuid": "GPU-test",
        "driver": "test-driver",
        "torch": "test-torch",
        "cuda_runtime": "test-cuda",
        "compute_capability": "8.6",
        "torch_cuda_arch_list": "8.6",
    }
    setup = {
        "initial_workspace_from_calibration": True,
        "prepare_outputs_calls": 1,
        "warmup_frames": 8,
        "trajectory_preflight_frames": 0,
        "explicit_capacity_reservation_calls": 0,
        "empty_cache_calls": 0,
    }
    installed = {"visible_records": 10, "intersections": 20}
    verified = {
        "max_visible_overflow": 0,
        "max_intersection_overflow": 0,
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": CAPACITY_CALIBRATION_SCHEMA,
                "pass": True,
                "environment": {
                    "native_extension": artifact_record(Path(__file__)),
                },
                "capacity": {
                    "installed": installed,
                    "verified_preflight": verified,
                    "intersection_capacity_scope": "batch-global-workspace",
                },
            }
        ),
        encoding="utf-8",
    )
    calibration_artifact = artifact_record(calibration_path)
    run = {
        "renderer": "custom",
        "output_contract": "full",
        "camera_contract": {"batch": 8, "trajectory_id": "trajectory"},
        "equation_contract": {
            "semantic_topology": "spatial-octants-8",
            "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            **MATCHED_PROJECTION_RULES,
        },
        "scene": {"sha256": "scene"},
        "environment": {
            **runtime,
            "source_provenance": provenance,
            "native_extension": artifact_record(Path(__file__)),
            "node_occupancy": artifact_record(Path(__file__).parent / "fixtures" / "passing_node_occupancy.json"),
            "flashgs_adapter_attestation": artifact_record(Path(__file__)),
            "flashgs_upstream_commit": ("cdfc4e4002318423eda356eed02df8e01fa32cb6"),
        },
        "runner_config": {
            "warmup_frames": 8,
            "measured_frames": 100,
            "capacity_headroom": 1.05,
            "initial_visible_per_view": 170_000,
            "initial_intersections_per_view": 200_000,
            "flashgs_initial_intersections": 200_000,
            "custom_depth_buckets": 128,
            "custom_depth_bucket_group": 8,
            "custom_max_physical_views": max_physical_views,
            "semantic_topology": "spatial-octants-8",
            "capture_last_output": True,
            "independent_trial": 1,
            "profile_control": False,
            "fixed_visible_capacity": None,
            "fixed_intersection_capacity": None,
            "capacity_calibration": calibration_artifact,
            "flashgs_demand_survey": None,
            "capacity_calibration_only": False,
            "flashgs_demand_survey_only": False,
            "premeasurement_schedule": setup,
        },
        "capacity": {
            "schema_version": CAPACITY_CONSUMPTION_SCHEMA,
            "capacity_source": "hash-bound-calibration-artifact",
            "calibration_artifact": calibration_artifact,
            "installed": installed,
            "verified_preflight": verified,
            "intersection_capacity_scope": "batch-global-workspace",
            "logical_batch": 8,
            "physical_batch": 8,
            "native_submissions_per_logical_batch": 1,
            "timed_setup": setup,
            "timed_verification": {
                "schema_version": "flashgs-matched-timed-capacity-verification-v1",
                "pass": True,
                "source": "renderer-device-counters",
                "frames_observed": 108,
                "warmup_frames_observed": 8,
                "measured_frames_observed": 100,
                "device_max_updates": 108,
                "cpu_readbacks_in_measured_loop": 0,
                "final_cpu_readback_after_measurement": True,
                "audit_updates_in_cuda_event_timing": False,
                "audit_updates_in_wall_samples": False,
                "observed": {
                    "max_visible_gaussians": 9,
                    "max_generated_intersections": 19,
                    "max_visible_overflow": 0,
                    "max_intersection_overflow": 0,
                },
            },
        },
        "backend_execution": {
            "logical_batch": 8,
            "physical_batch": 8,
            "native_submissions_per_logical_batch": 1,
        },
    }
    return run, runtime


def _require_fixture_identity(run: dict, runtime: dict) -> None:
    require_run_identity(
        run,
        renderer="custom",
        output_contract="full",
        batch=8,
        trajectory_id="trajectory",
        semantic_topology="spatial-octants-8",
        custom_chunked_physical_views=128,
        expected_gpu_uuid="GPU-test",
        expected_source_identity=source_identity(run["environment"]["source_provenance"]),
        expected_scene_sha256="scene",
        expected_runtime_identity=runtime,
        expected_flashgs_adapter_attestation=artifact_record(Path(__file__)),
        expected_capacity_calibration=run["runner_config"]["capacity_calibration"],
    )


def test_run_identity_accepts_direct_path_for_smaller_primary_batch(
    tmp_path: Path,
) -> None:
    run, runtime = _run_identity_fixture(tmp_path, max_physical_views=None)
    _require_fixture_identity(run, runtime)


def test_run_identity_rejects_chunk_flag_for_smaller_primary_batch(
    tmp_path: Path,
) -> None:
    run, runtime = _run_identity_fixture(tmp_path, max_physical_views=8)

    with pytest.raises(RuntimeError, match="internal camera schedule"):
        _require_fixture_identity(run, runtime)


def test_run_identity_rejects_non_gsplat_support_cutoff(
    tmp_path: Path,
) -> None:
    run, runtime = _run_identity_fixture(tmp_path, max_physical_views=None)
    run["equation_contract"]["gaussian_support_sigma"] = 3.0

    with pytest.raises(RuntimeError, match="support cutoff"):
        _require_fixture_identity(run, runtime)


def test_run_identity_rejects_non_gsplat_projection_edge_rule(
    tmp_path: Path,
) -> None:
    run, runtime = _run_identity_fixture(tmp_path, max_physical_views=None)
    run["equation_contract"]["frustum_depth_interval"] = "exclusive"

    with pytest.raises(RuntimeError, match="projection rule"):
        _require_fixture_identity(run, runtime)


def test_b128_repeat_rejects_timed_setup_tampering(tmp_path: Path) -> None:
    run, runtime = _run_identity_fixture(tmp_path, max_physical_views=None)
    run["camera_contract"] = {
        "batch": 128,
        "trajectory_id": PRIMARY_TRAJECTORY_IDS[128],
    }
    run["runner_config"] = {
        **run["runner_config"],
        "capture_last_output": False,
        "independent_trial": 2,
    }
    run["fidelity_capture"] = None
    run["memory"] = {"steady_state_fairness": {"pass": True}}
    run["pass"] = True

    require_b128_repeat_identity(
        run,
        renderer="custom",
        output_contract="full",
        trial=2,
        trajectory_id=PRIMARY_TRAJECTORY_IDS[128],
        semantic_topology="spatial-octants-8",
        expected_gpu_uuid="GPU-test",
        expected_source_identity=source_identity(run["environment"]["source_provenance"]),
        expected_scene_sha256="scene",
        expected_runtime_identity=runtime,
        expected_flashgs_adapter_attestation=artifact_record(Path(__file__)),
        expected_capacity_calibration=run["runner_config"]["capacity_calibration"],
    )

    run["capacity"]["timed_setup"] = {
        **run["capacity"]["timed_setup"],
        "empty_cache_calls": 1,
    }
    with pytest.raises(RuntimeError, match="timed setup"):
        require_b128_repeat_identity(
            run,
            renderer="custom",
            output_contract="full",
            trial=2,
            trajectory_id=PRIMARY_TRAJECTORY_IDS[128],
            semantic_topology="spatial-octants-8",
            expected_gpu_uuid="GPU-test",
            expected_source_identity=source_identity(run["environment"]["source_provenance"]),
            expected_scene_sha256="scene",
            expected_runtime_identity=runtime,
            expected_flashgs_adapter_attestation=artifact_record(Path(__file__)),
            expected_capacity_calibration=run["runner_config"]["capacity_calibration"],
        )


def test_native_matched_projection_edges_follow_pinned_gsplat() -> None:
    project_root = Path(__file__).resolve().parents[1]
    custom = (project_root / "src/isaacsim_gaussian_renderer/native/renderer_cuda.cu").read_text(encoding="utf-8")
    flashgs = (project_root / "src/isaacsim_gaussian_renderer/native/flashgs/preprocess.cu").read_text(encoding="utf-8")

    for source in (custom, flashgs):
        assert "camera_z < near_plane || camera_z > far_plane" in source
        assert "camera_z > near_plane && camera_z < far_plane" not in source
    assert custom.count("if (!(determinant > 0.0f))") >= 2
    assert "if (!(determinant > 0.0f))" in flashgs
    assert custom.count("radius_x <= 0 && radius_y <= 0") >= 2
    assert "*radius_x <= 0 && *radius_y <= 0" in flashgs
    assert "sqrtf(fmaxf(covariance_2d_00, 1.0e-12f))" not in custom
    assert "sqrtf(fmaxf(cov2d_00, 1.0e-12f))" not in flashgs


def test_capacity_calibration_requires_frozen_chunked_zero_overflow_path(
    tmp_path: Path,
) -> None:
    run, runtime = _run_identity_fixture(tmp_path, max_physical_views=None)
    calibration = {
        "schema_version": CAPACITY_CALIBRATION_SCHEMA,
        "mode": "capacity-calibration-only",
        "renderer": "custom",
        "calibration_output_contract": "full",
        "pass": True,
        "camera_contract": {
            "batch": 512,
            "trajectory_id": PRIMARY_TRAJECTORY_IDS[512],
            "timesteps": 108,
            "width": 128,
            "height": 128,
            "measured_start": 8,
        },
        "scene": {"sha256": "scene"},
        "environment": run["environment"],
        "equation_contract": run["equation_contract"],
        "calibration_config": {
            "capacity_headroom": 1.05,
            "initial_visible_per_view": 170_000,
            "initial_intersections_per_view": 200_000,
            "flashgs_initial_intersections": 200_000,
            "custom_depth_buckets": 128,
            "custom_depth_bucket_group": 8,
            "custom_max_physical_views": 128,
        },
        "capacity": {
            "schema_version": "flashgs-matched-capacity-v1",
            "headroom": 1.05,
            "capacity_source": "full-trajectory-device-counter-calibration",
            "preflight_frames_per_pass": 108,
            "preflight_measured_frames_per_pass": 100,
            "intersection_capacity_scope": "physical-camera-chunk-reused-workspace",
            "initial_preflight": {"max_visible_gaussians": 1},
            "calibration_attempts": [{"installed_visible_records": 1}],
            "preflight_passes": 2,
            "installed": {"visible_records": 1, "intersections": 1},
            "verified_preflight": {
                "max_visible_overflow": 0,
                "max_intersection_overflow": 0,
            },
            "logical_batch": 512,
            "physical_batch": 128,
            "native_submissions_per_logical_batch": 4,
        },
        "output_validation": {"pass": True},
    }

    require_capacity_calibration_identity(
        calibration,
        renderer="custom",
        batch=512,
        trajectory_id=PRIMARY_TRAJECTORY_IDS[512],
        trajectory_timesteps=108,
        semantic_topology="spatial-octants-8",
        expected_gpu_uuid="GPU-test",
        expected_source_identity=source_identity(run["environment"]["source_provenance"]),
        expected_scene_sha256="scene",
        expected_runtime_identity=runtime,
        expected_flashgs_adapter_attestation=artifact_record(Path(__file__)),
    )

    calibration["capacity"]["physical_batch"] = 512
    with pytest.raises(RuntimeError, match="physical_batch"):
        require_capacity_calibration_identity(
            calibration,
            renderer="custom",
            batch=512,
            trajectory_id=PRIMARY_TRAJECTORY_IDS[512],
            trajectory_timesteps=108,
            semantic_topology="spatial-octants-8",
            expected_gpu_uuid="GPU-test",
            expected_source_identity=source_identity(run["environment"]["source_provenance"]),
            expected_scene_sha256="scene",
            expected_runtime_identity=runtime,
            expected_flashgs_adapter_attestation=artifact_record(Path(__file__)),
        )

    calibration["capacity"]["physical_batch"] = 128
    calibration["capacity"]["capacity_source"] = "reused-full-sensor-run"
    with pytest.raises(RuntimeError, match="capacity_source"):
        require_capacity_calibration_identity(
            calibration,
            renderer="custom",
            batch=512,
            trajectory_id=PRIMARY_TRAJECTORY_IDS[512],
            trajectory_timesteps=108,
            semantic_topology="spatial-octants-8",
            expected_gpu_uuid="GPU-test",
            expected_source_identity=source_identity(run["environment"]["source_provenance"]),
            expected_scene_sha256="scene",
            expected_runtime_identity=runtime,
            expected_flashgs_adapter_attestation=artifact_record(Path(__file__)),
        )


def _write_prefix_trajectory(
    path: Path,
    *,
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    batch = int(viewmats.shape[1])
    save_trajectory(
        CameraTrajectory(
            viewmats=viewmats,
            intrinsics=intrinsics,
            scene_ids=np.full((batch,), 404, dtype=np.int64),
            scene_id_broadcast="time",
            width=128,
            height=128,
            fps=60.0,
            seed=1,
            scene_sha256="a" * 64,
            route_sha256="b" * 64,
            motion_classification="independently-changing-camera-batch",
            expected_cache_events=("disabled",) * int(viewmats.shape[0]),
        ),
        path,
    )


def test_trajectory_prefix_proof_is_exact_and_artifact_bound(tmp_path: Path) -> None:
    viewmats = np.broadcast_to(
        np.eye(4, dtype=np.float32),
        (3, 8, 4, 4),
    ).copy()
    viewmats[:, :, 0, 3] = np.arange(8, dtype=np.float32)[None, :]
    intrinsics = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (3, 8, 3, 3),
    ).copy()
    paths = {batch: tmp_path / f"b{batch}.json" for batch in (1, 2, 4, 8)}
    for batch, path in paths.items():
        _write_prefix_trajectory(
            path,
            viewmats=np.ascontiguousarray(viewmats[:, :batch]),
            intrinsics=np.ascontiguousarray(intrinsics[:, :batch]),
        )

    proof = prove_trajectory_prefixes(
        paths,
        expected_batches=(1, 2, 4, 8),
        expected_timesteps=3,
    )
    assert verify_trajectory_prefix_proof(proof) == proof

    paths[2].with_suffix(".npz").write_bytes(paths[2].with_suffix(".npz").read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="changed|mismatch"):
        verify_trajectory_prefix_proof(proof)


def test_trajectory_prefix_proof_rejects_nonprefix_camera_bits(tmp_path: Path) -> None:
    viewmats = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 2, 4, 4)).copy()
    intrinsics = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 2, 3, 3)).copy()
    b1 = tmp_path / "b1.json"
    b2 = tmp_path / "b2.json"
    _write_prefix_trajectory(b1, viewmats=viewmats[:, :1], intrinsics=intrinsics[:, :1])
    _write_prefix_trajectory(b2, viewmats=viewmats, intrinsics=intrinsics)
    tampered = intrinsics[:, :1].copy()
    tampered[1, 0, 0, 0] = np.nextafter(tampered[1, 0, 0, 0], np.float32(2.0))
    _write_prefix_trajectory(b1, viewmats=viewmats[:, :1], intrinsics=tampered)

    with pytest.raises(ValueError, match="intrinsics are not an exact"):
        prove_trajectory_prefixes(
            {1: b1, 2: b2},
            expected_batches=(1, 2),
            expected_timesteps=2,
        )


def test_trajectory_prefix_proof_rejects_signed_zero_bit_tamper(tmp_path: Path) -> None:
    viewmats = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 2, 4, 4)).copy()
    intrinsics = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 2, 3, 3)).copy()
    b1 = tmp_path / "b1.json"
    b2 = tmp_path / "b2.json"
    tampered = intrinsics[:, :1].copy()
    tampered[0, 0, 0, 1] = np.float32(-0.0)
    assert np.array_equal(tampered, intrinsics[:, :1])
    _write_prefix_trajectory(b1, viewmats=viewmats[:, :1], intrinsics=tampered)
    _write_prefix_trajectory(b2, viewmats=viewmats, intrinsics=intrinsics)

    with pytest.raises(ValueError, match="intrinsics are not an exact"):
        prove_trajectory_prefixes(
            {1: b1, 2: b2},
            expected_batches=(1, 2),
            expected_timesteps=2,
        )


def test_flashgs_counter_source_audit_fails_when_demand_atomic_moves(tmp_path: Path) -> None:
    relative = Path("src/isaacsim_gaussian_renderer/native/flashgs/preprocess.cu")
    source = Path(__file__).resolve().parents[1] / relative
    copied = tmp_path / relative
    copied.parent.mkdir(parents=True)
    copied.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    audit = audit_flashgs_demand_counter_source(tmp_path)
    verify_flashgs_demand_counter_source_audit(audit, project_root=tmp_path)

    copied.write_text(
        copied.read_text(encoding="utf-8").replace(
            "reinterpret_cast<unsigned long long*>(counters + 1)",
            "reinterpret_cast<unsigned long long*>(counters + 0)",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source audit failed"):
        verify_flashgs_demand_counter_source_audit(audit, project_root=tmp_path)


def test_flashgs_prefix_capacities_apply_headroom_per_batch() -> None:
    derived = derive_flashgs_prefix_capacities(
        [10] * 8,
        [100, 80, 200, 90, 400, 120, 110, 105],
        batches=(1, 2, 4, 8),
        headroom=1.05,
    )

    assert derived["1"]["installed_intersections_per_camera"] == 105
    assert derived["2"]["installed_intersections_per_camera"] == 105
    assert derived["4"]["installed_intersections_per_camera"] == 210
    assert derived["8"]["installed_intersections_per_camera"] == 420
    assert derived["4"]["prefix_camera_range"] == [0, 4]
    assert derived["4"]["installed_capacity_is_batch_specific"] is True
    with pytest.raises(ValueError, match="1.05 headroom"):
        derive_flashgs_prefix_capacities([1], [1], batches=(1,), headroom=1.1)


def test_flashgs_demand_survey_identity_rejects_derived_capacity_tamper(
    tmp_path: Path,
) -> None:
    contracts_root = tmp_path / "contracts"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "benchmarks/generate_flashgs_dynamic_contract.py"),
            "--output-root",
            str(contracts_root),
            "--max-batch",
            "1024",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    contract_paths = {batch: contracts_root / f"b{batch}.json" for batch in PRIMARY_BATCHES}
    proof = prove_trajectory_prefixes(contract_paths)
    assert {
        batch: proof["contracts"][str(batch)]["trajectory_id"] for batch in PRIMARY_BATCHES
    } == PRIMARY_TRAJECTORY_IDS
    visible = [100 + index for index in range(PRIMARY_BATCHES[-1])]
    generated = [1_000 + index for index in range(PRIMARY_BATCHES[-1])]
    overflow = [max(0, value - 200_000) for value in generated]
    derived = derive_flashgs_prefix_capacities(visible, generated)
    run, runtime = _run_identity_fixture(tmp_path, max_physical_views=None)
    survey = {
        "schema_version": FLASHGS_DEMAND_SURVEY_SCHEMA,
        "mode": "flashgs-demand-survey-only",
        "renderer": "flashgs",
        "pass": True,
        "timing_valid": False,
        "render_outputs_valid": False,
        "camera_contract": {
            "batch": 1024,
            "trajectory_id": PRIMARY_TRAJECTORY_IDS[1024],
            "timesteps": 108,
            "width": 128,
            "height": 128,
            "artifacts": trajectory_artifacts(contract_paths[1024]),
        },
        "scene": {"sha256": "scene"},
        "environment": run["environment"],
        "equation_contract": run["equation_contract"],
        "survey_config": {
            "capacity_headroom": 1.05,
            "initial_visible_per_view": 170_000,
            "initial_intersections_per_view": 200_000,
            "flashgs_initial_intersections": 200_000,
            "custom_depth_buckets": 128,
            "custom_depth_bucket_group": 8,
            "custom_max_physical_views": None,
            "installed_probe_intersections_per_camera": 200_000,
            "workspace_reuse": "per-camera-reused-workspace",
            "output_contract_exercised_but_invalid": "full",
        },
        "counter_source_audit": audit_flashgs_demand_counter_source(Path(__file__).resolve().parents[1]),
        "trajectory_prefix_proof": proof,
        "demand_observation": {
            "schema_version": "flashgs-matched-flashgs-demand-observation-v1",
            "counter_storage": "cuda-int64-per-camera",
            "frames_observed": 108,
            "camera_count": 1024,
            "device_max_updates": 108,
            "cpu_readbacks": 1,
            "per_camera_max_visible_gaussians": visible,
            "per_camera_max_generated_intersections": generated,
            "per_camera_max_intersection_overflow": overflow,
        },
        "derived_batch_capacities": derived,
    }

    require_flashgs_demand_survey_identity(
        survey,
        semantic_topology="spatial-octants-8",
        expected_gpu_uuid="GPU-test",
        expected_source_identity=source_identity(run["environment"]["source_provenance"]),
        expected_scene_sha256="scene",
        expected_runtime_identity=runtime,
        expected_flashgs_adapter_attestation=artifact_record(Path(__file__)),
        expected_contract_paths=contract_paths,
    )

    survey["derived_batch_capacities"]["8"] = dict(survey["derived_batch_capacities"]["1024"])
    with pytest.raises(RuntimeError, match="derived capacities"):
        require_flashgs_demand_survey_identity(
            survey,
            semantic_topology="spatial-octants-8",
            expected_gpu_uuid="GPU-test",
            expected_source_identity=source_identity(run["environment"]["source_provenance"]),
            expected_scene_sha256="scene",
            expected_runtime_identity=runtime,
            expected_flashgs_adapter_attestation=artifact_record(Path(__file__)),
            expected_contract_paths=contract_paths,
        )


def test_timed_device_counter_audit_covers_all_frames_and_detects_overflow() -> None:
    backend = FlashGSBackend(
        native_module=object(),
        allow_cpu_for_tests=True,
    )
    backend.initialize(stage=None, device=torch.device("cpu"))
    backend._max_views = 2
    backend.workspace["counters"] = torch.zeros((2, 3), dtype=torch.int64)
    state = create_timed_capacity_audit(
        backend,
        expected_frames=2,
        measured_start=1,
    )
    backend.workspace["counters"].copy_(torch.tensor([[3, 8, 0], [2, 5, 0]], dtype=torch.int64))
    update_timed_capacity_audit(state, backend, measured=False)
    backend.workspace["counters"].copy_(torch.tensor([[4, 9, 0], [1, 7, 2]], dtype=torch.int64))
    update_timed_capacity_audit(state, backend, measured=True)
    verification = finalize_timed_capacity_audit(state, backend)

    assert verification["frames_observed"] == 2
    assert verification["cpu_readbacks_in_measured_loop"] == 0
    assert verification["observed"]["max_generated_intersections_per_camera"] == 9
    assert verification["observed"]["max_intersection_overflow"] == 2
    assert verification["pass"] is False
    capacity = {
        "installed": {"visible_records": None, "intersections": 10},
        "timed_verification": verification,
    }
    with pytest.raises(RuntimeError, match="direct zero-overflow"):
        require_direct_timed_capacity_verification(
            capacity,
            renderer="flashgs",
            warmup_frames=1,
            measured_frames=1,
        )
