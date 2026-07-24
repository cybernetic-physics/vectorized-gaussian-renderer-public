from __future__ import annotations

import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_isaac_graphics_gate import (
    GateFailure,
    OVERLAY_REQUIREMENTS,
    _terminate_process_group,
    assert_no_residual_processes,
    assert_rasterize_mode_result,
    assert_single_active_gpu,
    lane_plan,
    match_vulkan_device,
    make_environment,
    parse_torch_identity,
    parse_nvidia_gpu_inventory,
    parse_vulkan_summary,
    prepare_experience_overlay,
    resolve_vulkan_icd,
    runtime_fingerprint,
    source_fingerprint,
)


UUID = "1da70b0f240df25d77a08a437043f744"


def test_vulkan_icd_auto_resolves_grid_runfile_layout(tmp_path: Path) -> None:
    runfile = tmp_path / "etc/vulkan/icd.d/nvidia_icd.json"
    distro = tmp_path / "usr/share/vulkan/icd.d/nvidia_icd.json"
    runfile.parent.mkdir(parents=True)
    runfile.write_text("{}\n", encoding="utf-8")

    assert resolve_vulkan_icd(None, candidates=(runfile, distro)) == runfile.resolve()


def test_vulkan_icd_auto_resolution_rejects_distinct_ambiguous_manifests(
    tmp_path: Path,
) -> None:
    runfile = tmp_path / "etc/nvidia_icd.json"
    distro = tmp_path / "usr/nvidia_icd.json"
    runfile.parent.mkdir()
    distro.parent.mkdir()
    runfile.write_text("{\"driver\": \"runfile\"}\n", encoding="utf-8")
    distro.write_text("{\"driver\": \"distro\"}\n", encoding="utf-8")

    with pytest.raises(GateFailure, match="Multiple distinct"):
        resolve_vulkan_icd(None, candidates=(runfile, distro))


def test_vulkan_icd_explicit_path_records_resolved_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(manifest)

    assert resolve_vulkan_icd(alias) == manifest.resolve()


def test_dependency_overlay_pins_direct_and_transitive_missing_modules() -> None:
    assert OVERLAY_REQUIREMENTS == (
        "plyfile==1.1.3",
        "lpips==0.1.4",
        "tqdm==4.67.1",
    )


def test_vulkan_uuid_is_mapped_to_torch_cuda_zero() -> None:
    summary = f"""
Vulkan Instance Version: 1.4.312

GPU0:
    deviceName        = llvmpipe
    deviceUUID        = 00000000000000000000000000000000
GPU2:
    deviceName        = NVIDIA L40
    deviceUUID        = {UUID}
"""
    devices = parse_vulkan_summary(summary)
    identity = parse_torch_identity(
        'ISAAC_GPU_IDENTITY {"cuda_device_uuid": "GPU-1da70b0f-240d-f25d-77a0-8a437043f744"}\n'
    )

    assert match_vulkan_device(devices, identity["cuda_device_uuid"])["index"] == 2


def test_nvidia_smi_inventory_preserves_driver_and_full_uuid() -> None:
    devices = parse_nvidia_gpu_inventory("0, NVIDIA L40, GPU-1da70b0f-240d-f25d-77a0-8a437043f744, 580.159.03, 46068\n")

    assert devices == [
        {
            "index": 0,
            "name": "NVIDIA L40",
            "uuid_raw": "GPU-1da70b0f-240d-f25d-77a0-8a437043f744",
            "uuid": UUID,
            "driver": "580.159.03",
            "memory_mib": 46068,
        }
    ]


def test_uuid_mapping_rejects_index_only_or_ambiguous_selection() -> None:
    device = {"index": 0, "name": "NVIDIA L40", "uuid": UUID}

    with pytest.raises(GateFailure, match="matched 0"):
        match_vulkan_device([device], "0" * 32)
    with pytest.raises(GateFailure, match="matched 2"):
        match_vulkan_device([device, dict(device, index=1)], UUID)


def test_gpu_foundation_contract_requires_only_matched_active_gpu() -> None:
    log = """
Starting kit application with the following args: ['--/renderer/activeGpu=2', '--/renderer/multiGpu/enabled=False', '--/renderer/multiGpu/maxGpuCount=1', '--/physics/cudaDevice=0']
| 0   | NVIDIA L4  | No     |     |
| 2   | NVIDIA L40 | Yes: 0 |     |
Simulation App Startup Complete
Simulation App Shutting Down
"""

    assert assert_single_active_gpu(log, 2)["active_rows"][0]["index"] == "2"


def test_gpu_foundation_contract_rejects_second_active_gpu() -> None:
    log = """
--/renderer/activeGpu=2 --/renderer/multiGpu/enabled=False --/renderer/multiGpu/maxGpuCount=1 --/physics/cudaDevice=0
| 1 | NVIDIA L4 | Yes: 1 | |
| 2 | NVIDIA L40 | Yes: 0 | |
Simulation App Startup Complete
Simulation App Shutting Down
"""
    with pytest.raises(GateFailure, match="exactly"):
        assert_single_active_gpu(log, 2)


def test_gpu_foundation_contract_rejects_extension_registry_fallback() -> None:
    log = """
--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=False --/renderer/multiGpu/maxGpuCount=1 --/physics/cudaDevice=0
| 0 | NVIDIA L4 | Yes: 0 | |
Simulation App Startup Complete
Failed to solve some dependencies locally, syncing with extension registry...
Simulation App Shutting Down
"""
    with pytest.raises(GateFailure, match="moving extension registry"):
        assert_single_active_gpu(log, 0)


def test_gpu_foundation_contract_requires_marker_before_shutdown() -> None:
    log = """
--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=False --/renderer/multiGpu/maxGpuCount=1 --/physics/cudaDevice=0
| 0 | NVIDIA L4 | Yes: 0 | |
Simulation App Startup Complete
Simulation App Shutting Down
LANE_OK
"""
    with pytest.raises(GateFailure, match="outside the startup/close lifecycle"):
        assert_single_active_gpu(log, 0, success_marker="LANE_OK")


def test_gpu_foundation_contract_requires_isolated_experience(tmp_path: Path) -> None:
    experience = tmp_path / "test.kit"
    experience.touch()
    log = """
--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=False --/renderer/multiGpu/maxGpuCount=1 --/physics/cudaDevice=0
| 0 | NVIDIA L4 | Yes: 0 | |
Simulation App Startup Complete
Simulation App Shutting Down
"""
    with pytest.raises(GateFailure, match="isolated Kit experience"):
        assert_single_active_gpu(log, 0, expected_experience=experience)


def test_runtime_fingerprint_detects_metadata_changes(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    package = runtime / "package.py"
    package.write_text("value = 1\n", encoding="utf-8")
    before = runtime_fingerprint(runtime)
    package.write_text("value = 22\n", encoding="utf-8")

    assert runtime_fingerprint(runtime) != before


def test_runtime_fingerprint_ignores_directory_mtime_only(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    package = runtime / "package"
    package.mkdir(parents=True)
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    before = runtime_fingerprint(runtime)
    stat = package.stat()
    os.utime(package, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert runtime_fingerprint(runtime) == before


def test_source_fingerprint_detects_same_size_content_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    module = source / "module.py"
    module.write_text("value=1\n", encoding="utf-8")
    before = source_fingerprint(source)
    stat = module.stat()
    module.write_text("value=2\n", encoding="utf-8")
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert source_fingerprint(source) != before


def test_residual_process_check_rejects_new_cpu_only_python() -> None:
    baseline = {
        "gpu_compute": {"empty": True, "rows": []},
        "host": {"relevant_rows": [{"pid": 10, "arguments": "gate.py"}]},
    }
    current = {
        "gpu_compute": {"empty": True, "rows": []},
        "host": {
            "relevant_rows": [
                {"pid": 10, "arguments": "gate.py"},
                {"pid": 20, "arguments": "python escaped_child.py"},
            ]
        },
    }

    with pytest.raises(GateFailure, match="host processes survived"):
        assert_no_residual_processes(baseline, current)


def test_timeout_cleanup_escalates_even_if_direct_child_exits(monkeypatch) -> None:
    signals = []

    class Process:
        pid = 123

        @staticmethod
        def wait(timeout=None):
            return 0

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(
        os,
        "killpg",
        lambda group_id, sent_signal: signals.append((group_id, sent_signal)),
    )

    _terminate_process_group(Process())

    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]


def test_lane_plan_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.touch()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(target)
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    args = SimpleNamespace(
        python=venv_python,
        source_root=source,
        output_root=output,
        rasterize_mode="antialiased",
    )

    lanes = lane_plan(args, tmp_path / "scratch")
    assert lanes[0].command[0] == str(venv_python.absolute())
    extension = lanes[1]
    assert extension.command[extension.command.index("--rasterize-mode") + 1] == (
        "antialiased"
    )
    assert extension.command[extension.command.index("--output") + 1] == str(
        output / "results" / "extension.json"
    )
    assert extension.result_json == output / "results" / "extension.json"
    assert extension.expected_rasterize_mode == "antialiased"


def test_rasterize_mode_evidence_requires_cuda_backend_match(
    tmp_path: Path,
) -> None:
    result = tmp_path / "extension.json"
    result.write_text(
        """{
          "configuration": {"rasterize_mode": "antialiased"},
          "renderer_configuration": {
            "asserted": true,
            "requested_rasterize_mode": "antialiased",
            "actual_rasterize_mode": "antialiased",
            "matches": true
          }
        }\n""",
        encoding="utf-8",
    )

    acceptance = assert_rasterize_mode_result(result, "antialiased")

    assert acceptance["renderer_actual_rasterize_mode"] == "antialiased"


def test_rasterize_mode_evidence_rejects_classic_false_green(
    tmp_path: Path,
) -> None:
    result = tmp_path / "extension.json"
    result.write_text(
        """{
          "configuration": {"rasterize_mode": "classic"},
          "renderer_configuration": {
            "asserted": true,
            "requested_rasterize_mode": "classic",
            "actual_rasterize_mode": "classic",
            "matches": true
          }
        }\n""",
        encoding="utf-8",
    )

    with pytest.raises(GateFailure, match="did not prove"):
        assert_rasterize_mode_result(result, "antialiased")


def test_experience_overlay_links_canonical_extensions(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    isaac = runtime / "lib" / "python3.12" / "site-packages" / "isaacsim"
    for name in ("exts", "extscache", "extsDeprecated"):
        (isaac / name).mkdir(parents=True)
    source = tmp_path / "source"
    (source / "apps").mkdir(parents=True)
    (source / "exts").mkdir()
    (source / "apps" / "vgr.graphical_test.kit").write_text("[package]\n")

    record = prepare_experience_overlay(runtime, source, tmp_path / "scratch")

    experience = Path(record["experience"])
    assert experience.read_text() == "[package]\n"
    for name in ("exts", "extscache", "extsDeprecated"):
        link = experience.parents[1] / name
        assert link.is_symlink()
        assert link.resolve() == (isaac / name).resolve()
    assert (experience.parents[1] / "extsUser").resolve() == (source / "exts").resolve()


def test_runtime_environment_prefers_disposable_dependency_overlay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    overlay = tmp_path / "overlay"
    icd = tmp_path / "nvidia.json"
    monkeypatch.setenv("PYTHONPATH", "/untrusted/inherited/path")
    env = make_environment(
        source,
        tmp_path / "scratch",
        0,
        icd,
        python_overlay=overlay,
    )

    assert env["PYTHONPATH"].split(":")[:2] == [
        str(overlay),
        str(source / "src"),
    ]
    assert "/untrusted/inherited/path" not in env["PYTHONPATH"]
