import json
import sys
from pathlib import Path

import pytest

from benchmarks import summarize_flashgs_profile
from scripts.parse_nsys_stats import (
    numeric,
    parse_api_rows,
    parse_kernel_rows,
    parse_nvtx_rows,
)
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    RENDERER_RUN_SCHEMA,
    artifact_record,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def passing_node_occupancy(tmp_path, name="run-occupancy.json"):
    path = tmp_path / name
    write_json(
        path,
        {
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
                "schema_version": ("flashgs-matched-compute-process-sampling-v1"),
                "coverage": "periodic-samples-not-continuous-observation",
                "sample_count": 2,
                "pass": True,
            },
            "pass": True,
        },
    )
    return artifact_record(path)


def write_wrapper_evidence(
    tmp_path,
    *,
    profile_path,
    native_path,
    renderer,
    contract,
    batch,
    trajectory,
    capacity_calibration=None,
):
    occupancy_path = tmp_path / "wrapper-occupancy.json"
    write_json(
        occupancy_path,
        {
            "schema_version": "flashgs-matched-occupancy-preflight-v2",
            "expected_gpu_uuid": "GPU-test",
            "executor_control": {
                "scope": "all-visible-gpus",
                "cooperative_node_wide_lock": {"pass": True},
            },
            "pass": True,
        },
    )
    exit_path = tmp_path / "profile-exit-status.txt"
    exit_path.write_text("0\n", encoding="utf-8")
    version_path = tmp_path / "nsys-version.txt"
    version_path.write_text("NVIDIA Nsight Systems 2024.6.2\n", encoding="utf-8")
    source_script = Path(__file__).resolve().parents[1] / ("scripts/profile_flashgs_matched_nsys.sh")
    capture = f"flashgs-matched-capture/{renderer}/{contract}/b{batch}"
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--capture-range=nvtx",
        "--capture-range-end=stop",
        f"--nvtx-capture={capture}",
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "benchmarks/run_flashgs_matched.py"),
        "--renderer",
        renderer,
        "--output-contract",
        contract,
        "--trajectory",
        trajectory,
        "--expected-gpu-uuid",
        "GPU-test",
        ("--capacity-calibration" if renderer == "custom" else "--flashgs-demand-survey"),
        str(capacity_calibration or ""),
        "--warmup-frames",
        "8",
        "--measured-frames",
        "3",
        "--profile-control",
        "--no-capture-last-output",
        "--output",
        str(profile_path),
    ]
    wrapper_path = tmp_path / "wrapper-evidence.json"
    write_json(
        wrapper_path,
        {
            "schema_version": "flashgs-profile-wrapper-evidence-v1",
            "pass": True,
            "command": command,
            "environment": {
                "PROJECT_ROOT": str(Path(__file__).resolve().parents[1]),
                "PROFILE_MEASURED_FRAMES": "3",
                "PROFILE_ROOT": str(profile_path.resolve().parent),
                "TORCH_CUDA_ARCH_LIST": "8.6",
                "VGR_NATIVE_BUILD_ROOT": str(native_path.resolve().parent.parent),
                "VGR_PROJECT_ROOT": str(Path(__file__).resolve().parents[1]),
            },
            "nsys_version": "NVIDIA Nsight Systems 2024.6.2",
            "artifacts": {
                "occupancy_preflight": artifact_record(occupancy_path),
                "profile_exit_status": artifact_record(exit_path),
                "profile_control": artifact_record(profile_path),
                "nsys_version": artifact_record(version_path),
                "source_script": artifact_record(source_script),
            },
        },
    )
    return wrapper_path


@pytest.mark.parametrize("renderer", ("custom", "flashgs"))
def test_nonheadline_control_can_produce_valid_profile_evidence(
    tmp_path,
    monkeypatch,
    renderer,
):
    native_path = tmp_path / "renderer.so"
    native_path.write_bytes(b"native")
    native = artifact_record(native_path)
    attestation_path = tmp_path / "flashgs-adapter-attestation.json"
    attestation_path.write_text("{}\n", encoding="utf-8")
    attestation = artifact_record(attestation_path)
    occupancy = passing_node_occupancy(tmp_path)
    installed = {"visible_records": 10, "intersections": 20}
    calibration_path = tmp_path / "capacity-artifact.json"
    write_json(
        calibration_path,
        {
            "schema_version": (
                "flashgs-matched-capacity-calibration-v1"
                if renderer == "custom"
                else "flashgs-matched-flashgs-demand-survey-v1"
            ),
            "pass": True,
        },
    )
    calibration = artifact_record(calibration_path)
    common = {
        "schema_version": RENDERER_RUN_SCHEMA,
        "renderer": renderer,
        "output_contract": "rgb",
        "pass": True,
        "headline_eligible": False,
        "scene": {"sha256": "scene"},
        "camera_contract": {
            "path": "trajectory.json",
            "trajectory_id": "trajectory",
            "batch": 32,
            "measured_frames": 3,
        },
        "environment": {
            "gpu_name": "NVIDIA GeForce RTX 3090",
            "gpu_uuid": "GPU-test",
            "driver": "test-driver",
            "torch": "test-torch",
            "cuda_runtime": "test-cuda",
            "compute_capability": [8, 6],
            "torch_cuda_arch_list": "8.6",
            "source_git_commit": "source",
            "source_provenance": {
                "manifest": {"sha256": "manifest"},
                "source_tree_sha256": "tree",
                "head": "source",
                "dirty": False,
                "diff_sha256": "diff",
            },
            "native_extension": native,
            "flashgs_adapter_attestation": attestation,
            "node_occupancy": occupancy,
        },
        "equation_contract": {
            "precision": "float32",
            "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            **MATCHED_PROJECTION_RULES,
        },
        "runner_config": {
            "custom_depth_buckets": 128,
            "custom_depth_bucket_group": 8,
            "custom_max_physical_views": None,
            "semantic_topology": "spatial-octants-8",
            ("capacity_calibration" if renderer == "custom" else "flashgs_demand_survey"): calibration,
        },
        "capacity": {
            "installed": installed,
            ("calibration_artifact" if renderer == "custom" else "demand_survey_artifact"): calibration,
        },
        "timing": {
            "gpu_batch_ms": {"mean": 10.0},
            "host_submission_ms": {"mean": 1.0},
        },
        "memory": {
            "driver_process_memory_growth_mib": 2,
            "allocation_growth_bytes": 0,
            "reservation_growth_bytes": 0,
        },
    }
    control = {**common, "profile_control": False}
    profile = {
        **common,
        "profile_control": True,
        "timing": {
            "gpu_batch_ms": {"mean": 12.0},
            "host_submission_ms": {"mean": 1.5},
        },
    }
    nsys = {
        "schema_version": 1,
        "limitations": [],
        "kernel_summary": {
            "total_kernel_time_ms": 30.0,
            "stages": {},
            "top_kernels": [],
        },
        "cuda_api_summary": {
            "totals": {"calls": 3.0, "time_ns": 1.0, "time_ms": 0.0},
            "kernel_launch": {
                "calls": 3.0,
                "time_ns": 1.0,
                "time_ms": 0.0,
            },
        },
        "allocator_classification": "cuda-runtime-driver-vmm-v2",
        "nvtx_summary": {
            "ranges": [
                {
                    "name": f"flashgs-matched-capture/{renderer}/rgb/b32",
                    "calls": 1,
                },
                {
                    "name": f"flashgs-matched/{renderer}/rgb/b32/frame-000",
                    "calls": 3,
                },
            ]
        },
    }
    control_path = tmp_path / "control.json"
    profile_path = tmp_path / "profile.json"
    nsys_path = tmp_path / "nsys.json"
    report_path = tmp_path / "capture.nsys-rep"
    sqlite_path = tmp_path / "capture.sqlite"
    stats_dir = tmp_path / "stats"
    output_path = tmp_path / "summary.json"
    stats_dir.mkdir()
    write_json(control_path, control)
    write_json(profile_path, profile)
    wrapper_path = write_wrapper_evidence(
        tmp_path,
        profile_path=profile_path,
        native_path=native_path,
        renderer=renderer,
        contract="rgb",
        batch=32,
        trajectory="trajectory.json",
        capacity_calibration=calibration_path,
    )
    write_json(nsys_path, nsys)
    report_path.write_bytes(b"report")
    sqlite_path.write_bytes(b"sqlite")
    (stats_dir / "cuda_api_sum.csv").write_text("Name,Calls\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_flashgs_profile.py",
            "--unprofiled-run",
            str(control_path),
            "--profile-run",
            str(profile_path),
            "--nsys-summary",
            str(nsys_path),
            "--nsys-report",
            str(report_path),
            "--nsys-sqlite",
            str(sqlite_path),
            "--stats-dir",
            str(stats_dir),
            "--wrapper-evidence",
            str(wrapper_path),
            "--output",
            str(output_path),
        ],
    )

    summarize_flashgs_profile.main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["pass"] is True
    assert result["hardware_scope"]["unprofiled_headline_eligible"] is False
    assert result["profiled_control"]["measured_range_allocator_calls"] == 0
    assert result["profiled_control"]["no_measured_range_allocator_calls"] is True
    assert result["profiled_control"]["total_cuda_api_calls"] == 3
    assert result["profiled_control"]["kernel_launch_api_calls"] == 3

    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    exit_status_path = Path(wrapper["artifacts"]["profile_exit_status"]["path"])
    exit_status_path.write_text("7\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        summarize_flashgs_profile.main()
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    assert "profile_exit_status" in " ".join(failed["failures"])
    exit_status_path.write_text("0\n", encoding="utf-8")

    original_command = list(wrapper["command"])
    wrapper["command"].remove("--capture-range=nvtx")
    write_json(wrapper_path, wrapper)
    with pytest.raises(SystemExit):
        summarize_flashgs_profile.main()
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    assert "lacks required arguments" in " ".join(failed["failures"])
    wrapper["command"] = original_command
    write_json(wrapper_path, wrapper)

    wrapper["environment"]["PROFILE_MEASURED_FRAMES"] = "4"
    write_json(wrapper_path, wrapper)
    with pytest.raises(SystemExit):
        summarize_flashgs_profile.main()
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    assert "measured-frame control" in " ".join(failed["failures"])
    wrapper["environment"]["PROFILE_MEASURED_FRAMES"] = "3"
    write_json(wrapper_path, wrapper)

    measured_index = wrapper["command"].index("--measured-frames") + 1
    wrapper["command"][measured_index] = "4"
    write_json(wrapper_path, wrapper)
    with pytest.raises(SystemExit):
        summarize_flashgs_profile.main()
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    assert "--measured-frames differs" in " ".join(failed["failures"])
    wrapper["command"] = original_command
    write_json(wrapper_path, wrapper)

    nsys["cuda_api_summary"]["allocator"] = {
        "calls": 1.0,
        "time_ns": 10.0,
        "time_ms": 0.00001,
    }
    write_json(nsys_path, nsys)
    with pytest.raises(SystemExit):
        summarize_flashgs_profile.main()
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    assert failed["pass"] is False
    assert "allocator calls" in " ".join(failed["failures"])

    del nsys["cuda_api_summary"]["allocator"]
    write_json(nsys_path, nsys)
    profile["runner_config"] = {
        **profile["runner_config"],
        "custom_depth_buckets": 64,
    }
    write_json(profile_path, profile)
    with pytest.raises(SystemExit):
        summarize_flashgs_profile.main()
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    assert "kernel_runner_config" in " ".join(failed["failures"])

    profile["runner_config"] = control["runner_config"]
    control["equation_contract"] = {
        **control["equation_contract"],
        "gaussian_support_sigma": 3.0,
    }
    profile["equation_contract"] = control["equation_contract"]
    write_json(control_path, control)
    write_json(profile_path, profile)
    with pytest.raises(SystemExit):
        summarize_flashgs_profile.main()
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    assert "pinned gsplat" in " ".join(failed["failures"])

    control["equation_contract"] = profile["equation_contract"] = {
        **control["equation_contract"],
        "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    }
    profile["environment"] = {
        **profile["environment"],
        "flashgs_adapter_attestation": artifact_record(native_path),
    }
    write_json(control_path, control)
    write_json(profile_path, profile)
    with pytest.raises(SystemExit):
        summarize_flashgs_profile.main()
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    assert "adapter attestation" in " ".join(failed["failures"])


def test_cuda_vmm_apis_are_allocator_activity():
    summary = parse_api_rows(
        [
            {
                "Name": "cuMemCreate",
                "Total Time (ns)": "11",
                "Instances": "1",
            },
            {
                "Name": "cuMemMap",
                "Total Time (ns)": "12",
                "Instances": "2",
            },
            {
                "Name": "cuMemUnmap",
                "Total Time (ns)": "13",
                "Instances": "3",
            },
            {
                "Name": "cuMemPoolTrimTo",
                "Total Time (ns)": "14",
                "Instances": "4",
            },
            {
                "Name": "cuArrayCreate",
                "Total Time (ns)": "15",
                "Instances": "5",
            },
        ]
    )
    assert summary["allocator"]["calls"] == 15
    assert summary["totals"]["calls"] == 15


def test_default_domain_nvtx_prefix_is_normalized():
    summary = parse_nvtx_rows(
        [
            {
                "Range": ":flashgs-matched-capture/custom/full/b1024",
                "Instances": "1",
                "Total Time (ns)": "10",
            }
        ]
    )

    assert summary["ranges"][0]["name"] == ("flashgs-matched-capture/custom/full/b1024")


@pytest.mark.parametrize("value", [None, "", "not-a-number", "nan", "inf", "-1"])
def test_required_nsys_numeric_values_fail_closed(value):
    with pytest.raises(ValueError):
        numeric(value, field="test counter")


def test_malformed_required_nsys_columns_fail_closed():
    with pytest.raises(ValueError, match="total time"):
        parse_api_rows([{"Name": "cudaLaunchKernel", "Instances": "1"}])
    with pytest.raises(ValueError, match="call count"):
        parse_kernel_rows([{"Kernel Name": "render_kernel", "Total Time (ns)": "10"}])
    with pytest.raises(ValueError, match="required name"):
        parse_nvtx_rows([{"Range": "", "Instances": "1", "Total Time (ns)": "10"}])


def test_profile_rejects_missing_cuda_activity(tmp_path, monkeypatch):
    """A syntactically present Nsight summary cannot pass with zero activity."""

    native_path = tmp_path / "renderer.so"
    native_path.write_bytes(b"native")
    native = artifact_record(native_path)
    attestation_path = tmp_path / "flashgs-adapter-attestation.json"
    attestation_path.write_text("{}\n", encoding="utf-8")
    attestation = artifact_record(attestation_path)
    occupancy = passing_node_occupancy(tmp_path)
    run = {
        "schema_version": RENDERER_RUN_SCHEMA,
        "renderer": "custom",
        "output_contract": "rgb",
        "pass": True,
        "headline_eligible": False,
        "profile_control": False,
        "scene": {"sha256": "scene"},
        "camera_contract": {
            "path": "trajectory.json",
            "trajectory_id": "trajectory",
            "batch": 1,
            "measured_frames": 1,
        },
        "environment": {
            "gpu_name": "NVIDIA GeForce RTX 3090",
            "gpu_uuid": "GPU-test",
            "driver": "driver",
            "torch": "torch",
            "cuda_runtime": "cuda",
            "compute_capability": [8, 6],
            "torch_cuda_arch_list": "8.6",
            "source_git_commit": "source",
            "source_provenance": {
                "manifest": {"sha256": "manifest"},
                "source_tree_sha256": "tree",
                "head": "source",
                "dirty": False,
                "diff_sha256": "diff",
            },
            "native_extension": native,
            "flashgs_adapter_attestation": attestation,
            "node_occupancy": occupancy,
        },
        "equation_contract": {
            "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            **MATCHED_PROJECTION_RULES,
        },
        "runner_config": {
            "custom_depth_buckets": 128,
            "custom_depth_bucket_group": 8,
            "custom_max_physical_views": None,
            "semantic_topology": "spatial-octants-8",
        },
        "capacity": {"installed": {"visible_records": 1, "intersections": 1}},
        "backend_execution": {
            "physical_batch": 1,
            "native_submissions_per_logical_batch": 1,
        },
        "timing": {
            "gpu_batch_ms": {"mean": 1.0},
            "host_submission_ms": {"mean": 0.1},
        },
        "memory": {
            "driver_process_memory_growth_mib": 0,
            "allocation_growth_bytes": 0,
            "reservation_growth_bytes": 0,
        },
    }
    control_path = tmp_path / "control.json"
    profile_path = tmp_path / "profile.json"
    nsys_path = tmp_path / "nsys.json"
    report_path = tmp_path / "capture.nsys-rep"
    sqlite_path = tmp_path / "capture.sqlite"
    stats_dir = tmp_path / "stats"
    output_path = tmp_path / "summary.json"
    stats_dir.mkdir()
    write_json(control_path, run)
    write_json(profile_path, {**run, "profile_control": True})
    wrapper_path = write_wrapper_evidence(
        tmp_path,
        profile_path=profile_path,
        native_path=native_path,
        renderer="custom",
        contract="rgb",
        batch=1,
        trajectory="trajectory.json",
    )
    write_json(
        nsys_path,
        {
            "schema_version": 1,
            "limitations": [],
            "kernel_summary": {
                "total_kernel_time_ms": 1.0,
                "stages": {},
                "top_kernels": [],
            },
            "cuda_api_summary": {"totals": {"calls": 0, "time_ns": 0, "time_ms": 0}},
            "allocator_classification": "cuda-runtime-driver-vmm-v2",
            "nvtx_summary": {
                "ranges": [
                    {
                        "name": "flashgs-matched-capture/custom/rgb/b1",
                        "calls": 1,
                    },
                    {
                        "name": "flashgs-matched/custom/rgb/b1/frame-000",
                        "calls": 1,
                    },
                ]
            },
        },
    )
    report_path.write_bytes(b"report")
    sqlite_path.write_bytes(b"sqlite")
    (stats_dir / "cuda_api_sum.csv").write_text("Name,Calls,Total Time (ns)\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_flashgs_profile.py",
            "--unprofiled-run",
            str(control_path),
            "--profile-run",
            str(profile_path),
            "--nsys-summary",
            str(nsys_path),
            "--nsys-report",
            str(report_path),
            "--nsys-sqlite",
            str(sqlite_path),
            "--stats-dir",
            str(stats_dir),
            "--wrapper-evidence",
            str(wrapper_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit):
        summarize_flashgs_profile.main()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["pass"] is False
    assert "no CUDA API calls" in " ".join(result["failures"])
    assert "no CUDA kernel-launch API calls" in " ".join(result["failures"])
