from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaacsim_gaussian_renderer.fidelity import bundle_from_tensors, write_camera_bundle


def _load_gate_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_projection_mode_activation_gate.py"
    spec = importlib.util.spec_from_file_location("run_projection_mode_activation_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def gate_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=tmp_path,
        python="/isaac-sim/python.sh",
        width=gate.GATE_WIDTH,
        height=gate.GATE_HEIGHT,
        warmup=gate.MIN_WARMUP_FRAMES,
        frames=gate.MIN_TEMPORAL_FRAMES,
        prime_frames=16,
        focal_scale=gate.GATE_FOCAL_SCALE,
        per_lane_timeout_seconds=gate.DEFAULT_PER_LANE_TIMEOUT_SECONDS,
        outer_timeout_seconds=gate.DEFAULT_OUTER_TIMEOUT_SECONDS,
    )


def test_cli_defaults_are_the_pinned_fixture_contract(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gate"])

    args = gate.parse_args()

    assert (args.width, args.height) == (gate.GATE_WIDTH, gate.GATE_HEIGHT) == (256, 256)
    assert args.focal_scale == gate.GATE_FOCAL_SCALE == pytest.approx(0.9)
    assert args.frames == gate.MIN_TEMPORAL_FRAMES == 256
    assert args.warmup == gate.MIN_WARMUP_FRAMES == 8
    assert gate.CONTROL_FOCAL_SCALE_FACTOR == pytest.approx(1.05)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--width", "128"],
        ["--height", "128"],
        ["--focal-scale", "1.0"],
        ["--frames", "255"],
        ["--warmup", "7"],
    ],
)
def test_cli_rejects_fixture_contract_drift(monkeypatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["gate", *arguments])

    with pytest.raises(SystemExit):
        gate.parse_args()


def test_lane_plan_is_discarded_prime_then_bracketed_cpttpc(
    tmp_path: Path,
) -> None:
    lanes = gate.plan_lanes(gate_args(tmp_path))

    assert [lane.name for lane in lanes] == [
        "prime",
        "positive-control-candidate",
        "perspective-candidate",
        "tangential-candidate",
        "tangential-repeat",
        "perspective-repeat",
        "positive-control-repeat",
    ]
    assert [lane.mode for lane in lanes[2:6]] == [
        "perspective",
        "tangential",
        "tangential",
        "perspective",
    ]
    assert lanes[0].measured is False
    assert [lane.frames for lane in lanes] == [16, 256, 256, 256, 256, 256, 256]
    assert [lane.focal_scale for lane in lanes if lane.measured] == pytest.approx([0.945, 0.9, 0.9, 0.9, 0.9, 0.945])


def test_lane_command_pins_gate_camera_and_temporal_capture(
    tmp_path: Path,
) -> None:
    args = gate_args(tmp_path)
    for lane in gate.plan_lanes(args):
        lane_dir = tmp_path / lane.name
        command = gate.lane_command(args, lane, lane_dir)

        assert command[0] == "/isaac-sim/python.sh"
        assert command[1].endswith("experiments/ovrtx_temporal_fidelity.py")
        assert command[command.index("--scene") + 1] == "projection-activation"
        assert command[command.index("--projection-mode") + 1] == lane.mode
        assert command[command.index("--sorting-mode") + 1] == gate.GATE_SORTING_MODE
        assert command[command.index("--width") + 1] == "256"
        assert command[command.index("--height") + 1] == "256"
        assert command[command.index("--frames") + 1] == str(lane.frames)
        assert command[command.index("--warmup") + 1] == "8"
        assert command[command.index("--focal-scale") + 1] == f"{lane.focal_scale:.10g}"
        assert command[command.index("--semantic-scheme") + 1] == "index-modulo"
        assert command[command.index("--output") + 1] == str(lane_dir)
        assert "--final-report-only" in command


def test_outer_timeout_must_cover_every_lane_budget(monkeypatch) -> None:
    minimum = gate.required_outer_timeout_seconds(1800)
    assert minimum == 14400

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate",
            "--per-lane-timeout-seconds",
            "1800",
            "--outer-timeout-seconds",
            str(minimum - 1),
        ],
    )
    with pytest.raises(SystemExit):
        gate.parse_args()


def write_temporal_lane(
    lane_dir: Path,
    *,
    mode: str,
    frames: int,
    warmup: int,
    final_pass: bool,
    focal_scale: float = 0.9,
    write_execution_evidence: bool = True,
) -> list[str]:
    lane_dir.mkdir(parents=True, exist_ok=True)
    width = gate.GATE_WIDTH
    height = gate.GATE_HEIGHT
    rgb = np.zeros((1, height, width, 3), dtype=np.float32)
    alpha = np.zeros((1, height, width), dtype=np.float32)
    depth = np.full((1, height, width), np.inf, dtype=np.float32)
    semantic = np.full((1, height, width), -1, dtype=np.int64)
    valid_depth = np.zeros((1, height, width), dtype=np.bool_)
    rgb[:, 96:160, 96:160, :] = 0.25
    alpha[:, 96:160, 96:160] = 0.5
    depth[:, 96:160, 96:160] = 2.0
    semantic[:, 96:160, 96:160] = 0
    valid_depth[:, 96:160, 96:160] = True

    scene_manifest = gate.projection_activation_scene_manifest()
    cameras = gate.fixture_camera_bundle(
        1,
        width,
        height,
        device="cpu",
        focal_scale=focal_scale,
    )
    fidelity_bundle = bundle_from_tensors(
        viewmats=cameras.viewmats,
        intrinsics=cameras.intrinsics,
        width=width,
        height=height,
        color_space="display_srgb",
        scene_ids=torch.tensor([101], dtype=torch.int64),
        scene_checksum=scene_manifest["checksum_sha256"],
    )
    write_camera_bundle(fidelity_bundle, lane_dir / gate.CAMERA_BUNDLE_FILE)
    np.savez(
        lane_dir / gate.CANDIDATE_NPZ,
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        semantic=semantic,
        valid_depth=valid_depth,
        color_space=np.asarray("display_srgb"),
        background=np.zeros(3, dtype=np.float32),
        camera_bundle_id=np.asarray(fidelity_bundle.bundle_id),
    )
    stage_path = lane_dir / gate.STAGE_FILE
    stage_path.write_text(
        "#usda 1.0\n"
        'def ParticleField3DGaussianSplat "Splats_0000" {\n'
        f' uniform token projectionModeHint = "{mode}"\n'
        f' uniform token sortingModeHint = "{gate.GATE_SORTING_MODE}"\n'
        "}\n"
        'def Camera "Camera_0" {\n'
        f" float focalLength = {36.0 * focal_scale}\n"
        "}\n",
        encoding="utf-8",
    )
    projection_readback = {
        "attribute": "projectionModeHint",
        "requested": mode,
        "observed": [mode],
        "prim_count": 1,
        "all_match": True,
    }
    sorting_readback = {
        "attribute": "sortingModeHint",
        "requested": gate.GATE_SORTING_MODE,
        "observed": [gate.GATE_SORTING_MODE],
        "prim_count": 1,
        "all_match": True,
    }
    command = [
        "/isaac-sim/python.sh",
        str(gate.TEMPORAL_SCRIPT),
        "--scene",
        "projection-activation",
        "--projection-mode",
        mode,
        "--sorting-mode",
        gate.GATE_SORTING_MODE,
        "--frames",
        str(frames),
        "--warmup",
        str(warmup),
        "--width",
        str(width),
        "--height",
        str(height),
        "--focal-scale",
        f"{focal_scale:.10g}",
        "--semantic-scheme",
        "index-modulo",
        "--final-report-only",
        "--output",
        str(lane_dir),
    ]
    summary = {
        "schema_version": "ovrtx-temporal-fidelity/v1",
        "frames": frames,
        "warmup": warmup,
        "resumed_from_frames": 0,
        "resume_sample_sequence_advanced_frames": 0,
        "resume_accumulators": None,
        "final_report_only": True,
        "scene_id": "projection-activation",
        "scene": scene_manifest,
        "scene_tensor_sha256": scene_manifest["tensor_checksum_sha256"],
        "gaussian_count": 4,
        "semantic_scheme": "index-modulo",
        "semantic_group_count": 1,
        "camera_bundle_id": fidelity_bundle.bundle_id,
        "ovrtx_fractional_opacity": True,
        "ovrtx_color_space": "display_srgb",
        "ovrtx_projection_mode_hint": mode,
        "ovrtx_projection_mode_observed": mode,
        "ovrtx_sorting_mode": gate.GATE_SORTING_MODE,
        "ovrtx_sorting_mode_observed": gate.GATE_SORTING_MODE,
        "runtime_token_readback": {
            "projection_mode": projection_readback,
            "sorting_mode": sorting_readback,
        },
        "final_pass": final_pass,
        "provenance": {
            "project_commit": "1" * 40,
            "ovrtx_commit": "2" * 40,
            "ovrtx_version": "0.3.0",
            "renderer_version": [0, 3, 0],
            "command": command,
            "script_sha256": gate.file_sha256(gate.TEMPORAL_SCRIPT),
            "stage_sha256": gate.file_sha256(stage_path),
            "stage_file": str(stage_path),
        },
    }
    (lane_dir / gate.SUMMARY_JSON).write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    if write_execution_evidence:
        log_path = lane_dir / "run.log"
        log_path.write_text(
            f"{gate.TEMPORAL_CLEANUP_SUCCESS_MARKER}\n{gate.TEMPORAL_SUCCESS_MARKER}\n",
            encoding="utf-8",
        )
        gate.write_process_receipt(
            lane_dir,
            command=command,
            returncode=0,
            timed_out=False,
            timeout_seconds=gate.DEFAULT_PER_LANE_TIMEOUT_SECONDS,
            duration_seconds=1.0,
            log_path=log_path,
        )
    return command


def test_lane_validation_ignores_custom_fidelity_verdict(
    tmp_path: Path,
) -> None:
    lane = gate.LaneSpec(
        "perspective-candidate",
        "perspective",
        256,
        0.9,
        True,
    )
    lane_dir = tmp_path / lane.name
    command = write_temporal_lane(
        lane_dir,
        mode=lane.mode,
        frames=lane.frames,
        warmup=8,
        final_pass=False,
    )

    evidence = gate.validate_lane_artifacts(
        lane_dir,
        lane,
        expected_command=command,
        warmup=8,
    )

    assert evidence["frames"] == 256
    assert evidence["custom_fidelity_final_pass_observed"] is False
    assert evidence["custom_fidelity_final_pass_consumed"] is False
    assert set(evidence["candidate_shapes"]) == gate.REQUIRED_CANDIDATE_ARRAYS


def test_lane_validation_fails_closed_on_wrong_frame_count(
    tmp_path: Path,
) -> None:
    lane = gate.LaneSpec(
        "tangential-candidate",
        "tangential",
        256,
        0.9,
        True,
    )
    lane_dir = tmp_path / lane.name
    command = write_temporal_lane(
        lane_dir,
        mode=lane.mode,
        frames=64,
        warmup=8,
        final_pass=False,
    )

    with pytest.raises(ValueError, match="frame count mismatch"):
        gate.validate_lane_artifacts(
            lane_dir,
            lane,
            expected_command=command,
            warmup=8,
        )


def test_seven_realistic_lane_artifacts_satisfy_and_enforce_cross_lane_contract(
    tmp_path: Path,
) -> None:
    args = gate_args(tmp_path)
    evidence = {}
    for lane in gate.plan_lanes(args):
        lane_dir = tmp_path / lane.name
        command = write_temporal_lane(
            lane_dir,
            mode=lane.mode,
            frames=lane.frames,
            warmup=args.warmup,
            final_pass=False,
            focal_scale=lane.focal_scale,
        )
        evidence[lane.name] = gate.validate_lane_artifacts(
            lane_dir,
            lane,
            expected_command=command,
            warmup=args.warmup,
        )

    assert gate.validate_cross_lane_contract(evidence)["pass"] is True

    tangential_stage = tmp_path / "tangential-candidate" / gate.STAGE_FILE
    tangential_stage.write_text(
        tangential_stage.read_text(encoding="utf-8") + "custom int unrelatedSetting = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Tangential candidate/repeat stages differ"):
        gate.validate_cross_lane_contract(evidence)


def test_launch_lane_fails_closed_on_child_timeout_or_bad_log(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = gate_args(tmp_path / "timeout")
    lane = gate.plan_lanes(args)[0]

    def timeout_run(_command, **_kwargs):
        raise subprocess.TimeoutExpired(["temporal"], 1)

    monkeypatch.setattr(gate, "run_bounded_process", timeout_run)
    with pytest.raises(subprocess.TimeoutExpired):
        gate.launch_lane(
            args,
            lane,
            outer_deadline=gate.time.perf_counter() + 60,
        )
    timeout_receipt = json.loads((args.output_root / lane.name / gate.PROCESS_RECEIPT_FILE).read_text(encoding="utf-8"))
    assert timeout_receipt["command"] == gate.lane_command(
        args,
        lane,
        args.output_root / lane.name,
    )
    assert timeout_receipt["returncode"] is None
    assert timeout_receipt["signal"] is None
    assert timeout_receipt["timeout"]["expired"] is True
    assert timeout_receipt["log"]["sha256"] == gate.file_sha256(args.output_root / lane.name / "run.log")

    log_args = gate_args(tmp_path / "bad-log")
    log_lane = gate.plan_lanes(log_args)[0]
    calls = 0

    def bad_log_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0 if calls == 1 else 1)

    monkeypatch.setattr(gate, "run_bounded_process", bad_log_run)
    with pytest.raises(RuntimeError, match="log contract failed"):
        gate.launch_lane(
            log_args,
            log_lane,
            outer_deadline=gate.time.perf_counter() + 60,
        )


def test_launch_lane_requires_exact_post_cleanup_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = gate_args(tmp_path)
    lane = gate.plan_lanes(args)[0]
    calls = 0

    def missing_cleanup_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            kwargs["stdout"].write(f"{gate.TEMPORAL_SUCCESS_MARKER}\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(gate, "run_bounded_process", missing_cleanup_run)
    with pytest.raises(RuntimeError, match="exactly once"):
        gate.launch_lane(
            args,
            lane,
            outer_deadline=gate.time.perf_counter() + 60,
        )
    assert calls == 2


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (
            [
                gate.TEMPORAL_CLEANUP_SUCCESS_MARKER,
                gate.TEMPORAL_CLEANUP_SUCCESS_MARKER,
                gate.TEMPORAL_SUCCESS_MARKER,
            ],
            "exactly once",
        ),
        (
            [
                gate.TEMPORAL_CLEANUP_SUCCESS_MARKER,
                gate.TEMPORAL_SUCCESS_MARKER,
                gate.TEMPORAL_SUCCESS_MARKER,
            ],
            "exactly once",
        ),
        (
            [
                gate.TEMPORAL_SUCCESS_MARKER,
                gate.TEMPORAL_CLEANUP_SUCCESS_MARKER,
            ],
            "required order",
        ),
        (
            [
                f"INFO {gate.TEMPORAL_CLEANUP_SUCCESS_MARKER}",
                gate.TEMPORAL_SUCCESS_MARKER,
            ],
            "exactly once",
        ),
        (
            [
                gate.TEMPORAL_CLEANUP_SUCCESS_MARKER,
                f"[stdout] {gate.TEMPORAL_SUCCESS_MARKER}",
            ],
            "exactly once",
        ),
    ],
    ids=[
        "duplicate-cleanup",
        "duplicate-capture",
        "reversed",
        "prefixed-cleanup",
        "prefixed-capture",
    ],
)
def test_exact_lifecycle_marker_contract_rejects_malformed_logs(
    tmp_path: Path,
    lines: list[str],
    message: str,
) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        gate.require_exact_log_markers(
            log_path,
            gate.TEMPORAL_CLEANUP_SUCCESS_MARKER,
            gate.TEMPORAL_SUCCESS_MARKER,
        )


def test_successful_lane_records_standalone_log_checker_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = gate_args(tmp_path)
    lane = gate.plan_lanes(args)[0]
    lane_dir = args.output_root / lane.name
    calls: list[list[str]] = []

    def successful_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            write_temporal_lane(
                lane_dir,
                mode=lane.mode,
                frames=lane.frames,
                warmup=args.warmup,
                final_pass=False,
                focal_scale=lane.focal_scale,
                write_execution_evidence=False,
            )
            kwargs["stdout"].write(
                f"standalone OVRTX output\n{gate.TEMPORAL_CLEANUP_SUCCESS_MARKER}\n{gate.TEMPORAL_SUCCESS_MARKER}\n"
            )
        return subprocess.CompletedProcess(command, 0, stdout="checker ok", stderr="")

    monkeypatch.setattr(gate, "run_bounded_process", successful_run)
    _lane_dir, _duration, evidence = gate.launch_lane(
        args,
        lane,
        outer_deadline=gate.time.perf_counter() + 60,
    )

    contract = evidence["log_checker_contract"]
    assert len(calls) == 2
    assert contract["command"] == [
        str(gate.LOG_CHECKER),
        str(lane_dir / "run.log"),
        gate.TEMPORAL_SUCCESS_MARKER,
    ]
    assert contract["returncode"] == 0
    assert contract["required_exact_marker_order"] == [
        gate.TEMPORAL_CLEANUP_SUCCESS_MARKER,
        gate.TEMPORAL_SUCCESS_MARKER,
    ]
    assert contract["observed_line_numbers"] == {
        gate.TEMPORAL_CLEANUP_SUCCESS_MARKER: 2,
        gate.TEMPORAL_SUCCESS_MARKER: 3,
    }
    assert contract["each_exactly_once"] is True
    assert contract["ordered"] is True
    assert contract["log_sha256"] == evidence["log_sha256"]
    assert contract["log_scope"] == "standalone_ovrtx_stdout_stderr"
    assert contract["separate_kit_internal_log_expected"] is False


@pytest.mark.skipif(not hasattr(signal, "SIGABRT"), reason="SIGABRT is unavailable")
def test_native_abort_after_success_text_cannot_reach_artifact_or_audit_acceptance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = gate_args(tmp_path)
    args.per_lane_timeout_seconds = 30
    lane = gate.plan_lanes(args)[0]
    lane_dir = args.output_root / lane.name
    child_code = (
        "import os, resource, signal; "
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
        f"print({gate.TEMPORAL_CLEANUP_SUCCESS_MARKER!r}, flush=True); "
        f"print({gate.TEMPORAL_SUCCESS_MARKER!r}, flush=True); "
        "os.kill(os.getpid(), signal.SIGABRT)"
    )
    abort_command = [sys.executable, "-c", child_code]
    monkeypatch.setattr(
        gate,
        "lane_command",
        lambda _args, _lane, _lane_dir: abort_command,
    )

    with pytest.raises(RuntimeError, match="SIGABRT"):
        gate.launch_lane(
            args,
            lane,
            outer_deadline=gate.time.perf_counter() + 60,
        )

    log_path = lane_dir / "run.log"
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        gate.TEMPORAL_CLEANUP_SUCCESS_MARKER,
        gate.TEMPORAL_SUCCESS_MARKER,
    ]
    receipt = json.loads((lane_dir / gate.PROCESS_RECEIPT_FILE).read_text(encoding="utf-8"))
    assert receipt["command"] == abort_command
    assert receipt["returncode"] == -signal.SIGABRT
    assert receipt["signal"] == {
        "number": signal.SIGABRT,
        "name": "SIGABRT",
    }
    assert receipt["timeout"]["expired"] is False
    assert receipt["log"]["sha256"] == gate.file_sha256(log_path)

    # Even complete, syntactically valid child artifacts cannot be consumed
    # after the parent observed a native abort.
    write_temporal_lane(
        lane_dir,
        mode=lane.mode,
        frames=lane.frames,
        warmup=args.warmup,
        final_pass=True,
        focal_scale=lane.focal_scale,
        write_execution_evidence=False,
    )
    with pytest.raises(ValueError, match="SIGABRT"):
        gate.validate_lane_artifacts(
            lane_dir,
            lane,
            expected_command=abort_command,
            warmup=args.warmup,
        )
    assert not (args.output_root / "gate-execution.json").exists()
    assert not (args.output_root / "projection-mode-audit.json").exists()
    assert not (args.output_root / "activation-gate-report.json").exists()


def test_cold_suspect_detection_requires_complete_measured_set() -> None:
    durations = {
        "prime": 700.0,
        "positive-control-candidate": 79.0,
        "perspective-candidate": 80.0,
        "tangential-candidate": 82.0,
        "tangential-repeat": 81.0,
        "perspective-repeat": 240.0,
        "positive-control-repeat": 80.0,
    }

    median, suspects = gate.cold_suspect_lanes(durations)

    assert median == 80.5
    assert suspects == ["perspective-repeat"]

    with pytest.raises(ValueError, match="every measured lane"):
        gate.cold_suspect_lanes({"prime": 1.0})


def test_incomplete_lane_report_cannot_be_mistaken_for_a_verdict(
    tmp_path: Path,
) -> None:
    path = gate.write_incomplete_report(
        tmp_path,
        completed_lanes=["prime", "perspective-candidate"],
        error=TimeoutError("bounded lane timeout"),
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["pass"] is False
    assert report["classification"] == "INCOMPLETE_LANE_SET"
    assert report["phase"] == "lane_execution"
    assert report["required_lane_count"] == 7


def valid_audit_report(*, passed: bool) -> dict:
    return {
        "schema_version": gate.AUDIT_SCHEMA_VERSION,
        "pass": passed,
        "classification": ("OBSERVABLE_MODE_EFFECT" if passed else "NO_OBSERVABLE_MODE_EFFECT"),
        "activation_proof_required": True,
        "activation_proof_valid": passed,
        "token_proof_valid": True,
        "behavioral_activation_valid": passed,
        "candidate_outputs_valid": True,
        "matched_lane_contract_valid": True,
        "stage_authoring_valid": True,
        "observably_different": passed,
        "repeat_controls_present": True,
        "effect_above_repeat_noise": passed,
        "runtime_readback_valid": True,
        "positive_control_required": True,
        "positive_control_present": True,
        "positive_control_stable": True,
        "positive_control_detected": True,
        "positive_control_valid": True,
    }


def test_audit_report_validation_accepts_complete_positive_and_negative() -> None:
    positive = valid_audit_report(passed=True)
    negative = valid_audit_report(passed=False)

    assert gate.validate_audit_report(positive) is positive
    assert gate.validate_audit_report(negative) is negative


def test_audit_report_validation_rejects_truthy_non_boolean_pass() -> None:
    report = valid_audit_report(passed=False)
    report["pass"] = "false"

    with pytest.raises(ValueError, match="boolean fields"):
        gate.validate_audit_report(report)


def test_audit_report_validation_rejects_bare_or_inconsistent_success() -> None:
    with pytest.raises(ValueError, match="boolean fields"):
        gate.validate_audit_report(
            {
                "schema_version": gate.AUDIT_SCHEMA_VERSION,
                "pass": True,
                "classification": "OBSERVABLE_MODE_EFFECT",
            }
        )

    report = valid_audit_report(passed=True)
    report["positive_control_valid"] = False
    with pytest.raises(ValueError, match="positive control fields"):
        gate.validate_audit_report(report)


def test_audit_report_validation_rejects_mislabeled_negative() -> None:
    report = valid_audit_report(passed=False)
    report["classification"] = "MODE_EFFECT_WITHIN_REPEAT_NOISE"

    with pytest.raises(ValueError, match="classification disagrees"):
        gate.validate_audit_report(report)


def test_audit_report_validation_rejects_inconsistent_control_validity() -> None:
    report = valid_audit_report(passed=True)
    report["positive_control_valid"] = False
    report["activation_proof_valid"] = False
    report["pass"] = False

    with pytest.raises(ValueError, match="positive control fields"):
        gate.validate_audit_report(report)


def test_incomplete_audit_report_names_the_failed_phase(tmp_path: Path) -> None:
    path = gate.write_incomplete_report(
        tmp_path,
        completed_lanes=[lane.name for lane in gate.plan_lanes(gate_args(tmp_path))],
        error=TimeoutError("bounded auditor timeout"),
        phase="audit_execution",
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["pass"] is False
    assert report["classification"] == "INCOMPLETE_AUDIT"
    assert report["phase"] == "audit_execution"
    assert len(report["completed_lanes"]) == gate.LANE_COUNT


def install_main_harness(
    monkeypatch,
    tmp_path: Path,
    *,
    audit_report: dict,
    audit_exit_code: int,
    durations: dict[str, float] | None = None,
    lane_error: tuple[str, Exception] | None = None,
    audit_error: Exception | None = None,
) -> tuple[SimpleNamespace, list[str], list[list[str]]]:
    args = gate_args(tmp_path / "gate-output")
    monkeypatch.setattr(gate, "parse_args", lambda: args)
    launched: list[str] = []
    auditor_commands: list[list[str]] = []

    def fake_launch_lane(_args, lane, *, outer_deadline):
        del outer_deadline
        launched.append(lane.name)
        if lane_error is not None and lane.name == lane_error[0]:
            raise lane_error[1]
        lane_dir = args.output_root / lane.name
        duration = 10.0 if durations is None else durations[lane.name]
        return lane_dir, duration, {"lane": lane.name}

    def fake_cross_lane_contract(evidence):
        assert set(evidence) == {lane.name for lane in gate.plan_lanes(args)}
        return {"pass": True, "mocked_contract": True}

    def fake_run(command, **kwargs):
        auditor_commands.append(command)
        if audit_error is not None:
            raise audit_error
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(audit_report), encoding="utf-8")
        if kwargs.get("stdout") is not None:
            kwargs["stdout"].write("mock auditor\n")
        return subprocess.CompletedProcess(command, audit_exit_code)

    monkeypatch.setattr(gate, "launch_lane", fake_launch_lane)
    monkeypatch.setattr(gate, "validate_cross_lane_contract", fake_cross_lane_contract)
    monkeypatch.setattr(gate, "run_bounded_process", fake_run)
    return args, launched, auditor_commands


def test_main_mocked_success_runs_seven_lanes_and_preserves_raw_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_report = valid_audit_report(passed=True)
    args, launched, commands = install_main_harness(
        monkeypatch,
        tmp_path,
        audit_report=raw_report,
        audit_exit_code=0,
    )

    assert gate.main() == 0
    assert launched == [lane.name for lane in gate.plan_lanes(args)]
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--positive-control-candidate") + 1].endswith(
        "positive-control-candidate/candidate-ovrtx-temporal.npz"
    )
    assert command[command.index("--positive-control-repeat-candidate") + 1].endswith(
        "positive-control-repeat/candidate-ovrtx-temporal.npz"
    )
    assert json.loads((args.output_root / "projection-mode-audit.json").read_text()) == raw_report
    gate_report = json.loads((args.output_root / "activation-gate-report.json").read_text())
    assert gate_report["schema_version"] == "ovrtx-projection-mode-activation-gate/v1"
    assert gate_report["pass"] is True
    assert gate_report["audit_report"] == raw_report


def test_main_honest_negative_preserves_audit_classification(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_report = valid_audit_report(passed=False)
    args, _launched, _commands = install_main_harness(
        monkeypatch,
        tmp_path,
        audit_report=raw_report,
        audit_exit_code=1,
    )

    assert gate.main() == 1
    gate_report = json.loads((args.output_root / "activation-gate-report.json").read_text())
    assert gate_report["pass"] is False
    assert gate_report["classification"] == "NO_OBSERVABLE_MODE_EFFECT"
    assert gate_report["audit_report"] == raw_report


def test_main_lane_timeout_stops_before_auditor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args, launched, commands = install_main_harness(
        monkeypatch,
        tmp_path,
        audit_report=valid_audit_report(passed=True),
        audit_exit_code=0,
        lane_error=(
            "tangential-candidate",
            subprocess.TimeoutExpired(["temporal"], 1),
        ),
    )

    assert gate.main() == 1
    assert launched[-1] == "tangential-candidate"
    assert "tangential-repeat" not in launched
    assert commands == []
    incomplete = json.loads((args.output_root / "activation-gate-incomplete.json").read_text())
    assert incomplete["classification"] == "INCOMPLETE_LANE_SET"
    assert incomplete["completed_lanes"] == [
        "prime",
        "positive-control-candidate",
        "perspective-candidate",
    ]


def test_main_malformed_or_timed_out_auditor_never_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    malformed = valid_audit_report(passed=False)
    malformed["pass"] = "false"
    malformed_args, _launched, _commands = install_main_harness(
        monkeypatch,
        tmp_path / "malformed",
        audit_report=malformed,
        audit_exit_code=1,
    )
    assert gate.main() == 1
    malformed_incomplete = json.loads((malformed_args.output_root / "activation-gate-incomplete.json").read_text())
    assert malformed_incomplete["classification"] == "INCOMPLETE_AUDIT"

    timeout_args, _launched, _commands = install_main_harness(
        monkeypatch,
        tmp_path / "timeout",
        audit_report=valid_audit_report(passed=True),
        audit_exit_code=0,
        audit_error=subprocess.TimeoutExpired(["audit"], 1),
    )
    assert gate.main() == 1
    timeout_incomplete = json.loads((timeout_args.output_root / "activation-gate-incomplete.json").read_text())
    assert timeout_incomplete["classification"] == "INCOMPLETE_AUDIT"
    assert len(timeout_incomplete["completed_lanes"]) == gate.LANE_COUNT


def test_main_exit_mismatch_and_cold_lane_override_cannot_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mismatch_args, _launched, _commands = install_main_harness(
        monkeypatch,
        tmp_path / "mismatch",
        audit_report=valid_audit_report(passed=True),
        audit_exit_code=1,
    )
    assert gate.main() == 1
    mismatch = json.loads((mismatch_args.output_root / "activation-gate-report.json").read_text())
    assert mismatch["classification"] == "AUDITOR_EXECUTION_INVALID"
    assert mismatch["audit_report"]["pass"] is True

    durations = {lane.name: 10.0 for lane in gate.plan_lanes(gate_args(tmp_path))}
    durations["perspective-repeat"] = 25.0
    cold_args, _launched, _commands = install_main_harness(
        monkeypatch,
        tmp_path / "cold",
        audit_report=valid_audit_report(passed=True),
        audit_exit_code=0,
        durations=durations,
    )
    assert gate.main() == 1
    cold = json.loads((cold_args.output_root / "activation-gate-report.json").read_text())
    assert cold["classification"] == "COLD_LANE_CONTAMINATION"
    assert cold["audit_report"]["pass"] is True


def test_main_deadline_expiry_before_verdict_writes_no_passing_gate_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args, _launched, _commands = install_main_harness(
        monkeypatch,
        tmp_path,
        audit_report=valid_audit_report(passed=True),
        audit_exit_code=0,
    )
    original_bounded_timeout = gate.bounded_timeout_seconds

    def expire_at_verdict(deadline, requested_seconds, *, phase):
        if phase == "final gate verdict":
            raise TimeoutError("outer deadline expired")
        return original_bounded_timeout(
            deadline,
            requested_seconds,
            phase=phase,
        )

    monkeypatch.setattr(gate, "bounded_timeout_seconds", expire_at_verdict)

    assert gate.main() == 1
    assert not (args.output_root / "activation-gate-report.json").exists()
    incomplete = json.loads((args.output_root / "activation-gate-incomplete.json").read_text())
    assert incomplete["classification"] == "INCOMPLETE_AUDIT"
