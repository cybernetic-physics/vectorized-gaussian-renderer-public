"""Run the fail-closed OVRTX projection-mode activation experiment.

The evidence lanes use ``experiments/ovrtx_temporal_fidelity.py`` because it
explicitly accumulates RGB, alpha, depth, and semantic outputs.  Its custom
renderer fidelity verdict is intentionally ignored here: this gate consumes
only the OVRTX temporal candidate, composed-token readback, and authored USD
stage.

Execution order is a discarded cache prime followed by a bracketed
``control``, ``perspective``, ``tangential``, ``tangential``, ``perspective``,
``control`` sequence.  Both focal-control comparisons must clear all three
same-configuration repeat envelopes.  The control proves that matched camera
authoring and output capture can detect a modest geometric change; it does not
prove that ParticleField hints reach or are consumed by a render kernel.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from isaacsim_gaussian_renderer.benchmark_manifest import (
    camera_bundle as fixture_camera_bundle,
    projection_activation_scene_manifest,
)
from isaacsim_gaussian_renderer.fidelity import load_camera_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_SCRIPT = PROJECT_ROOT / "experiments" / "ovrtx_temporal_fidelity.py"
AUDIT_SCRIPT = PROJECT_ROOT / "benchmarks" / "audit_ovrtx_projection_modes.py"
LOG_CHECKER = PROJECT_ROOT / ".agents" / "skills" / "test-isaac-graphics" / "scripts" / "check-isaac-log.sh"
TEMPORAL_SUCCESS_MARKER = "OVRTX_TEMPORAL_CAPTURE_OK"
TEMPORAL_CLEANUP_SUCCESS_MARKER = "OVRTX_TEMPORAL_CLEANUP_OK"
CANDIDATE_NPZ = "candidate-ovrtx-temporal.npz"
SUMMARY_JSON = "summary.json"
STAGE_FILE = "scene.usda"
CAMERA_BUNDLE_FILE = "camera-bundle.json"
PROCESS_RECEIPT_FILE = "process-receipt.json"
PROCESS_RECEIPT_SCHEMA_VERSION = "ovrtx-projection-mode-child-process/v1"
REQUIRED_CANDIDATE_ARRAYS = {
    "rgb",
    "alpha",
    "depth",
    "semantic",
    "valid_depth",
}
CONTROL_FOCAL_SCALE_FACTOR = 1.05
GATE_SORTING_MODE = "zDepth"
GATE_WIDTH = 256
GATE_HEIGHT = 256
GATE_FOCAL_SCALE = 0.9
MIN_TEMPORAL_FRAMES = 256
MIN_WARMUP_FRAMES = 8
COLD_LANE_WALL_RATIO = 2.0
OUTER_TIMEOUT_OVERHEAD_SECONDS = 1800
DEFAULT_PER_LANE_TIMEOUT_SECONDS = 1800
LANE_COUNT = 7
DEFAULT_OUTER_TIMEOUT_SECONDS = LANE_COUNT * DEFAULT_PER_LANE_TIMEOUT_SECONDS + OUTER_TIMEOUT_OVERHEAD_SECONDS
MIN_FOREGROUND_PIXELS = 4
PROJECTION_MODE_PATTERN = re.compile(r'(projectionModeHint\s*=\s*)"(perspective|tangential)"')
FOCAL_LENGTH_PATTERN = re.compile(r"(focalLength\s*=\s*)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
SORTING_MODE_PATTERN = re.compile(r'(sortingModeHint\s*=\s*)"(zDepth|distanceToCamera|none)"')
AUDIT_SCHEMA_VERSION = "ovrtx-projection-mode-audit/v3"
AUDIT_CLASSIFICATIONS = {
    "INVALID_MODE_AUTHORING",
    "INVALID_RUNTIME_MODE_READBACK",
    "MISSING_REPEAT_CONTROLS",
    "MISSING_POSITIVE_CONTROL",
    "INVALID_CANDIDATE_OUTPUT",
    "INVALID_MATCHED_LANE_CONTRACT",
    "POSITIVE_CONTROL_UNSTABLE",
    "POSITIVE_CONTROL_NOT_DETECTED",
    "NO_OBSERVABLE_MODE_EFFECT",
    "MODE_EFFECT_WITHIN_REPEAT_NOISE",
    "OBSERVABLE_MODE_EFFECT",
}
AUDIT_BOOLEAN_FIELDS = {
    "pass",
    "activation_proof_required",
    "activation_proof_valid",
    "token_proof_valid",
    "behavioral_activation_valid",
    "candidate_outputs_valid",
    "matched_lane_contract_valid",
    "stage_authoring_valid",
    "observably_different",
    "repeat_controls_present",
    "effect_above_repeat_noise",
    "runtime_readback_valid",
    "positive_control_required",
    "positive_control_present",
    "positive_control_stable",
    "positive_control_detected",
    "positive_control_valid",
}


@dataclass(frozen=True)
class LaneSpec:
    name: str
    mode: str
    frames: int
    focal_scale: float
    measured: bool


class GateTerminationRequested(RuntimeError):
    """Raised so an external TERM still reaps the active owned process group."""


class LaneEvidenceInvalid(RuntimeError):
    """Raised when a completed lane violates its artifact/evidence contract."""


def _raise_on_termination(signum: int, _frame: Any) -> None:
    raise GateTerminationRequested(f"Activation gate received signal {signum}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(PROJECT_ROOT / "outputs" / "projection-mode-activation-temporal"),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Isaac/OVRTX Python interpreter used for every child lane.",
    )
    parser.add_argument("--width", type=int, default=GATE_WIDTH)
    parser.add_argument("--height", type=int, default=GATE_HEIGHT)
    parser.add_argument("--warmup", type=int, default=MIN_WARMUP_FRAMES)
    parser.add_argument(
        "--frames",
        type=int,
        default=MIN_TEMPORAL_FRAMES,
        help="Explicitly accumulated temporal samples in every evidence lane.",
    )
    parser.add_argument(
        "--prime-frames",
        type=int,
        default=16,
        help="Samples in the discarded shader/pipeline cache-prime lane.",
    )
    parser.add_argument("--focal-scale", type=float, default=GATE_FOCAL_SCALE)
    parser.add_argument(
        "--per-lane-timeout-seconds",
        type=int,
        default=DEFAULT_PER_LANE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--outer-timeout-seconds",
        type=int,
        default=DEFAULT_OUTER_TIMEOUT_SECONDS,
        help=(
            "Enforced wall-time budget for the complete gate. It must cover "
            "every per-lane timeout plus the fixed audit/evidence margin."
        ),
    )
    args = parser.parse_args()
    if min(args.width, args.height, args.frames, args.prime_frames) <= 0:
        parser.error("Resolution and frame counts must be positive.")
    if args.warmup < 0 or args.focal_scale <= 0:
        parser.error("Warmup must be non-negative and focal scale positive.")
    if args.per_lane_timeout_seconds <= 0:
        parser.error("--per-lane-timeout-seconds must be positive.")
    if (args.width, args.height) != (GATE_WIDTH, GATE_HEIGHT):
        parser.error(f"The activation fixture contract is pinned to {GATE_WIDTH}x{GATE_HEIGHT}.")
    if not np.isclose(args.focal_scale, GATE_FOCAL_SCALE):
        parser.error(f"The activation fixture contract is pinned to focal scale {GATE_FOCAL_SCALE}.")
    if args.frames < MIN_TEMPORAL_FRAMES:
        parser.error(f"--frames must be at least {MIN_TEMPORAL_FRAMES} for the activation evidence contract.")
    if args.warmup < MIN_WARMUP_FRAMES:
        parser.error(f"--warmup must be at least {MIN_WARMUP_FRAMES} for the activation evidence contract.")
    required_outer = required_outer_timeout_seconds(args.per_lane_timeout_seconds)
    if args.outer_timeout_seconds < required_outer:
        parser.error(
            "--outer-timeout-seconds must be at least "
            f"{required_outer} for {LANE_COUNT} lanes at "
            f"{args.per_lane_timeout_seconds}s each plus "
            f"{OUTER_TIMEOUT_OVERHEAD_SECONDS}s overhead."
        )
    return args


def required_outer_timeout_seconds(per_lane_timeout_seconds: int) -> int:
    return LANE_COUNT * per_lane_timeout_seconds + OUTER_TIMEOUT_OVERHEAD_SECONDS


def bounded_timeout_seconds(
    deadline: float,
    requested_seconds: float,
    *,
    phase: str,
) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError(f"Activation gate outer deadline expired before {phase}.")
    return min(float(requested_seconds), remaining)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    stdout: Any = None,
    stderr: Any = None,
    capture_output: bool = False,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run one owned process group and reap all children on timeout/error."""
    if capture_output:
        if stdout is not None or stderr is not None:
            raise ValueError("capture_output cannot be combined with stdout/stderr.")
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        text=text,
        start_new_session=True,
    )
    try:
        output, error = process.communicate(timeout=timeout)
    except BaseException as exc:
        _terminate_process_group(process)
        # Preserve the reaped child's status for the parent-owned lane receipt.
        # TimeoutExpired and signal-handler exceptions do not expose it by
        # default, even though the process has been reaped at this point.
        exc.child_returncode = process.returncode  # type: ignore[attr-defined]
        raise
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=output,
        stderr=error,
    )


def plan_lanes(args: argparse.Namespace) -> list[LaneSpec]:
    return [
        LaneSpec(
            "prime",
            "perspective",
            args.prime_frames,
            args.focal_scale,
            False,
        ),
        LaneSpec(
            "positive-control-candidate",
            "perspective",
            args.frames,
            args.focal_scale * CONTROL_FOCAL_SCALE_FACTOR,
            True,
        ),
        LaneSpec(
            "perspective-candidate",
            "perspective",
            args.frames,
            args.focal_scale,
            True,
        ),
        LaneSpec(
            "tangential-candidate",
            "tangential",
            args.frames,
            args.focal_scale,
            True,
        ),
        LaneSpec(
            "tangential-repeat",
            "tangential",
            args.frames,
            args.focal_scale,
            True,
        ),
        LaneSpec(
            "perspective-repeat",
            "perspective",
            args.frames,
            args.focal_scale,
            True,
        ),
        LaneSpec(
            "positive-control-repeat",
            "perspective",
            args.frames,
            args.focal_scale * CONTROL_FOCAL_SCALE_FACTOR,
            True,
        ),
    ]


def lane_command(
    args: argparse.Namespace,
    lane: LaneSpec,
    lane_dir: Path,
) -> list[str]:
    return [
        args.python,
        str(TEMPORAL_SCRIPT),
        "--scene",
        "projection-activation",
        "--projection-mode",
        lane.mode,
        "--sorting-mode",
        GATE_SORTING_MODE,
        "--frames",
        str(lane.frames),
        "--warmup",
        str(args.warmup),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--focal-scale",
        f"{lane.focal_scale:.10g}",
        "--semantic-scheme",
        "index-modulo",
        "--final-report-only",
        "--output",
        str(lane_dir),
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signal_receipt(returncode: int | None) -> dict[str, Any] | None:
    if returncode is None or returncode >= 0:
        return None
    number = -returncode
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = f"SIGNAL_{number}"
    return {"number": number, "name": name}


def write_process_receipt(
    lane_dir: Path,
    *,
    command: list[str],
    returncode: int | None,
    timed_out: bool,
    timeout_seconds: float,
    duration_seconds: float,
    log_path: Path,
    error: BaseException | None = None,
) -> Path:
    """Write parent-observed child termination evidence for one lane."""
    receipt = {
        "schema_version": PROCESS_RECEIPT_SCHEMA_VERSION,
        "parent_owned": True,
        "command": command,
        "returncode": returncode,
        "signal": _signal_receipt(returncode),
        "timeout": {
            "expired": timed_out,
            "seconds": timeout_seconds,
        },
        "duration_seconds": duration_seconds,
        "log": {
            "path": str(log_path),
            "sha256": file_sha256(log_path),
        },
        "error": (
            None
            if error is None
            else {
                "type": type(error).__name__,
                "message": str(error),
            }
        ),
    }
    receipt_path = lane_dir / PROCESS_RECEIPT_FILE
    temporary_path = receipt_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(receipt_path)
    return receipt_path


def validate_process_receipt(
    lane_dir: Path,
    *,
    expected_command: list[str],
) -> dict[str, Any]:
    """Require a clean parent-observed exit before trusting child artifacts."""
    receipt_path = lane_dir / PROCESS_RECEIPT_FILE
    log_path = lane_dir / "run.log"
    if not receipt_path.is_file() or receipt_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required lane process receipt is missing: {receipt_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Required lane process log is missing: {log_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema_version") != PROCESS_RECEIPT_SCHEMA_VERSION:
        raise ValueError(f"Unexpected lane process receipt schema in {receipt_path}")
    if receipt.get("parent_owned") is not True:
        raise ValueError(f"Lane process receipt is not parent-owned in {receipt_path}")
    if receipt.get("command") != expected_command:
        raise ValueError(f"Lane process receipt command mismatch in {receipt_path}")

    timeout = receipt.get("timeout")
    if not isinstance(timeout, dict) or type(timeout.get("expired")) is not bool:
        raise ValueError(f"Lane process timeout evidence is malformed in {receipt_path}")
    timeout_seconds = timeout.get("seconds")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError(f"Lane process timeout budget is invalid in {receipt_path}")
    duration_seconds = receipt.get("duration_seconds")
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(float(duration_seconds))
        or duration_seconds < 0
    ):
        raise ValueError(f"Lane process duration is invalid in {receipt_path}")

    log = receipt.get("log")
    if not isinstance(log, dict) or log.get("path") != str(log_path):
        raise ValueError(f"Lane process log provenance is invalid in {receipt_path}")
    expected_log_sha256 = file_sha256(log_path)
    if log.get("sha256") != expected_log_sha256:
        raise ValueError(f"Lane process log checksum mismatch in {receipt_path}")

    returncode = receipt.get("returncode")
    if type(returncode) is not int:
        raise ValueError(f"Lane child has no trustworthy return code in {receipt_path}")
    expected_signal = _signal_receipt(returncode)
    if receipt.get("signal") != expected_signal:
        raise ValueError(f"Lane child signal evidence is inconsistent in {receipt_path}")
    if timeout["expired"] is True:
        raise ValueError(f"Lane child timed out according to {receipt_path}")
    if returncode != 0:
        signal_name = expected_signal["name"] if expected_signal is not None else None
        suffix = f" ({signal_name})" if signal_name is not None else ""
        raise ValueError(f"Lane child returncode was {returncode}{suffix} according to {receipt_path}")
    if receipt.get("signal") is not None or receipt.get("error") is not None:
        raise ValueError(f"Lane child clean-exit evidence is inconsistent in {receipt_path}")
    return receipt


def require_exact_log_markers(log_path: Path, *markers: str) -> dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    positions = {marker: [index for index, line in enumerate(lines) if line == marker] for marker in markers}
    invalid_counts = {marker: len(indices) for marker, indices in positions.items() if len(indices) != 1}
    if invalid_counts:
        raise RuntimeError(
            f"Lane log must contain every lifecycle marker exactly once; observed counts={invalid_counts}: {log_path}"
        )
    observed_order = [positions[marker][0] for marker in markers]
    if observed_order != sorted(observed_order):
        raise RuntimeError(f"Lane lifecycle markers do not appear in the required order {list(markers)}: {log_path}")
    return {
        "required_exact_marker_order": list(markers),
        "observed_line_numbers": {marker: positions[marker][0] + 1 for marker in markers},
        "each_exactly_once": True,
        "ordered": True,
    }


def _required_scalar(array: np.ndarray, *, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"Candidate metadata {name!r} must be scalar, got {value.shape}.")
    return value.item()


def _validate_token_readback(
    summary: dict[str, Any],
    *,
    key: str,
    attribute: str,
    expected: str,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    top_level = summary.get("runtime_token_readback")
    if isinstance(top_level, dict) and key in top_level:
        sources["runtime_token_readback"] = top_level[key]
    configuration = summary.get("configuration")
    if isinstance(configuration, dict):
        duplicate = configuration.get("runtime_token_readback")
        if isinstance(duplicate, dict) and key in duplicate:
            sources["configuration.runtime_token_readback"] = duplicate[key]
    if "runtime_token_readback" not in sources:
        raise ValueError(f"Canonical {key} token readback is missing.")
    expected_detail = {
        "attribute": attribute,
        "requested": expected,
        "observed": [expected],
        "prim_count": 1,
        "all_match": True,
    }
    for source, detail in sources.items():
        if detail != expected_detail:
            raise ValueError(f"Invalid {key} token readback in {source}: expected {expected_detail!r}, got {detail!r}.")
    if len(sources) > 1 and any(detail != sources["runtime_token_readback"] for detail in sources.values()):
        raise ValueError(f"Duplicate {key} token readbacks disagree.")
    return expected_detail


def _command_value(command: list[Any], flag: str) -> str:
    if command.count(flag) != 1:
        raise ValueError(f"Provenance command must contain {flag} exactly once.")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"Provenance command has no value for {flag}.")
    return str(command[index + 1])


def validate_lane_artifacts(
    lane_dir: Path,
    lane: LaneSpec,
    *,
    expected_command: list[str],
    warmup: int,
    width: int = GATE_WIDTH,
    height: int = GATE_HEIGHT,
) -> dict[str, Any]:
    process_receipt = validate_process_receipt(
        lane_dir,
        expected_command=expected_command,
    )
    process_receipt_path = lane_dir / PROCESS_RECEIPT_FILE
    candidate_path = lane_dir / CANDIDATE_NPZ
    summary_path = lane_dir / SUMMARY_JSON
    stage_path = lane_dir / STAGE_FILE
    camera_bundle_path = lane_dir / CAMERA_BUNDLE_FILE
    for path in (candidate_path, summary_path, stage_path, camera_bundle_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required lane artifact is missing: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "ovrtx-temporal-fidelity/v1":
        raise ValueError(f"Unexpected temporal schema in {summary_path}")
    if summary.get("frames") != lane.frames:
        raise ValueError(f"Temporal frame count mismatch in {summary_path}")
    if summary.get("warmup") != warmup:
        raise ValueError(f"Temporal warmup mismatch in {summary_path}")
    if (
        summary.get("resumed_from_frames") != 0
        or summary.get("resume_sample_sequence_advanced_frames") != 0
        or summary.get("resume_accumulators") is not None
        or summary.get("final_report_only") is not True
    ):
        raise ValueError(f"Temporal lane must be a fresh final-only capture in {summary_path}")
    if summary.get("scene_id") != "projection-activation":
        raise ValueError(f"Temporal scene mismatch in {summary_path}")
    if summary.get("ovrtx_projection_mode_hint") != lane.mode:
        raise ValueError(f"Temporal projection hint mismatch in {summary_path}")
    if summary.get("ovrtx_projection_mode_observed") != lane.mode:
        raise ValueError(f"Temporal projection observed scalar mismatch in {summary_path}")
    if summary.get("ovrtx_sorting_mode") != GATE_SORTING_MODE:
        raise ValueError(f"Temporal sorting hint mismatch in {summary_path}")
    if summary.get("ovrtx_sorting_mode_observed") != GATE_SORTING_MODE:
        raise ValueError(f"Temporal sorting observed scalar mismatch in {summary_path}")
    if summary.get("ovrtx_fractional_opacity") is not True or summary.get("ovrtx_color_space") != "display_srgb":
        raise ValueError(f"Temporal OVRTX output contract mismatch in {summary_path}")
    projection_readback = _validate_token_readback(
        summary,
        key="projection_mode",
        attribute="projectionModeHint",
        expected=lane.mode,
    )
    sorting_readback = _validate_token_readback(
        summary,
        key="sorting_mode",
        attribute="sortingModeHint",
        expected=GATE_SORTING_MODE,
    )
    scene = summary.get("scene")
    expected_scene = projection_activation_scene_manifest()
    if scene != expected_scene:
        raise ValueError(f"Temporal projection fixture manifest mismatch in {summary_path}")
    scene_checksum = scene["checksum_sha256"]
    scene_tensor_checksum = scene["tensor_checksum_sha256"]
    if summary.get("scene_tensor_sha256") != scene_tensor_checksum:
        raise ValueError(f"Uploaded scene tensor checksum mismatch in {summary_path}")
    if summary.get("gaussian_count") != 4:
        raise ValueError(f"Projection fixture must contain four Gaussians in {summary_path}")
    if summary.get("semantic_scheme") != "index-modulo" or summary.get("semantic_group_count") != 1:
        raise ValueError(f"Projection fixture must produce one ParticleField in {summary_path}")

    loaded_camera_bundle = load_camera_bundle(camera_bundle_path)
    camera_bundle = json.loads(camera_bundle_path.read_text(encoding="utf-8"))
    if camera_bundle.get("schema_version") != "camera-bundle-v1":
        raise ValueError(f"Unexpected camera bundle schema in {camera_bundle_path}")
    if (camera_bundle.get("width"), camera_bundle.get("height")) != (width, height):
        raise ValueError(f"Camera bundle resolution mismatch in {camera_bundle_path}")
    if camera_bundle.get("scene_checksum") != scene_checksum:
        raise ValueError(f"Camera bundle scene checksum mismatch in {camera_bundle_path}")
    camera_bundle_id = summary.get("camera_bundle_id")
    if (
        not isinstance(camera_bundle_id, str)
        or camera_bundle.get("bundle_id") != camera_bundle_id
        or loaded_camera_bundle.bundle_id != camera_bundle_id
    ):
        raise ValueError(f"Camera bundle ID mismatch in {camera_bundle_path}")
    if (
        loaded_camera_bundle.coordinate_system != "opencv_world_to_camera"
        or loaded_camera_bundle.color_space != "display_srgb"
        or loaded_camera_bundle.background != (0.0, 0.0, 0.0)
    ):
        raise ValueError(f"Camera bundle rendering contract mismatch in {camera_bundle_path}")
    cameras = camera_bundle.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != 1:
        raise ValueError(f"Activation gate requires exactly one camera in {camera_bundle_path}")
    intrinsics = np.asarray(cameras[0].get("intrinsics"), dtype=np.float64)
    viewmat = np.asarray(cameras[0].get("viewmat"), dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"Invalid camera intrinsics in {camera_bundle_path}")
    if (
        viewmat.shape != (4, 4)
        or not np.isfinite(viewmat).all()
        or cameras[0].get("view_id") != "view_000000"
        or cameras[0].get("scene_id") != 101
    ):
        raise ValueError(f"Invalid camera record in {camera_bundle_path}")
    expected_fixture_camera = fixture_camera_bundle(
        1,
        width,
        height,
        device="cpu",
        focal_scale=lane.focal_scale,
    )
    expected_viewmat = expected_fixture_camera.viewmats[0].numpy().astype(np.float64)
    expected_intrinsics = expected_fixture_camera.intrinsics[0].numpy().astype(np.float64)
    if not np.allclose(viewmat, expected_viewmat, rtol=0, atol=5.0e-7):
        raise ValueError(f"Camera view matrix mismatch in {camera_bundle_path}")
    if not np.allclose(intrinsics, expected_intrinsics, rtol=0, atol=2.0e-5):
        raise ValueError(f"Camera intrinsic matrix mismatch in {camera_bundle_path}")
    expected_focal_x = float(np.float32(lane.focal_scale * width))
    expected_focal_y = float(np.float32(lane.focal_scale * height))
    if not (
        math.isclose(float(intrinsics[0, 0]), expected_focal_x, rel_tol=0, abs_tol=2.0e-5)
        and math.isclose(float(intrinsics[1, 1]), expected_focal_y, rel_tol=0, abs_tol=2.0e-5)
    ):
        raise ValueError(f"Camera focal scale mismatch in {camera_bundle_path}")

    stage_text = stage_path.read_text(encoding="utf-8")
    stage_modes = [match.group(2) for match in PROJECTION_MODE_PATTERN.finditer(stage_text)]
    if stage_modes != [lane.mode]:
        raise ValueError(f"Stage projection authoring mismatch in {stage_path}")
    sorting_modes = [match.group(2) for match in SORTING_MODE_PATTERN.finditer(stage_text)]
    if sorting_modes != [GATE_SORTING_MODE]:
        raise ValueError(f"Stage sorting authoring mismatch in {stage_path}")
    particle_field_count = len(re.findall(r'\bdef\s+ParticleField3DGaussianSplat\s+"', stage_text))
    if particle_field_count != 1:
        raise ValueError(f"Stage must contain exactly one ParticleField in {stage_path}")
    focal_values = [float(match.group(2)) for match in FOCAL_LENGTH_PATTERN.finditer(stage_text)]
    expected_stage_focal = 36.0 * lane.focal_scale
    if len(focal_values) != 1 or not math.isclose(
        focal_values[0],
        expected_stage_focal,
        rel_tol=0,
        abs_tol=1.0e-6,
    ):
        raise ValueError(f"Stage focalLength mismatch in {stage_path}")

    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Temporal provenance is missing in {summary_path}")
    if provenance.get("script_sha256") != file_sha256(TEMPORAL_SCRIPT):
        raise ValueError(f"Temporal script checksum mismatch in {summary_path}")
    if provenance.get("stage_sha256") != file_sha256(stage_path):
        raise ValueError(f"Temporal stage checksum mismatch in {summary_path}")
    command = provenance.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError(f"Temporal provenance command is malformed in {summary_path}")
    expected_command_values = {
        "--scene": "projection-activation",
        "--projection-mode": lane.mode,
        "--sorting-mode": GATE_SORTING_MODE,
        "--frames": str(lane.frames),
        "--warmup": str(warmup),
        "--width": str(width),
        "--height": str(height),
        "--semantic-scheme": "index-modulo",
    }
    for flag, expected in expected_command_values.items():
        if _command_value(command, flag) != expected:
            raise ValueError(f"Temporal provenance command mismatch for {flag} in {summary_path}")
    command_focal_scale = float(_command_value(command, "--focal-scale"))
    if not math.isclose(command_focal_scale, lane.focal_scale, rel_tol=0, abs_tol=1.0e-9):
        raise ValueError(f"Temporal provenance focal scale mismatch in {summary_path}")
    if command.count("--final-report-only") != 1:
        raise ValueError(f"Temporal provenance lacks --final-report-only in {summary_path}")
    forbidden_flags = {
        "--resume-accumulators",
        "--save-checkpoint-outputs",
        "--save-final-accumulators",
        "--custom-ray-gaussian-evaluation",
        "--custom-deterministic",
    }
    present_forbidden = sorted(flag for flag in forbidden_flags if flag in command)
    if present_forbidden:
        raise ValueError(f"Temporal provenance contains forbidden gate flags {present_forbidden} in {summary_path}")
    for required_provenance in (
        "project_commit",
        "ovrtx_commit",
        "ovrtx_version",
        "renderer_version",
    ):
        if provenance.get(required_provenance) in (None, "", []):
            raise ValueError(f"Temporal provenance lacks {required_provenance} in {summary_path}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(provenance["project_commit"])):
        raise ValueError(f"Temporal project commit is not a full Git SHA in {summary_path}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(provenance["ovrtx_commit"])):
        raise ValueError(f"Temporal OVRTX commit is not a full Git SHA in {summary_path}")
    if provenance["ovrtx_version"] != "0.3.0" or provenance["renderer_version"] != [0, 3, 0]:
        raise ValueError(f"Unexpected OVRTX runtime version in {summary_path}")

    with np.load(candidate_path, allow_pickle=False) as candidate:
        required_candidate_keys = REQUIRED_CANDIDATE_ARRAYS | {
            "color_space",
            "background",
            "camera_bundle_id",
        }
        missing = required_candidate_keys - set(candidate.files)
        if missing:
            raise ValueError(f"{candidate_path} is missing candidate arrays: {sorted(missing)}")
        extras = set(candidate.files) - required_candidate_keys
        if extras:
            raise ValueError(f"{candidate_path} has unexpected candidate arrays: {sorted(extras)}")
        arrays = {key: np.asarray(candidate[key]) for key in REQUIRED_CANDIDATE_ARRAYS}
        expected_shapes = {
            "rgb": (1, height, width, 3),
            "alpha": (1, height, width),
            "depth": (1, height, width),
            "semantic": (1, height, width),
            "valid_depth": (1, height, width),
        }
        expected_dtypes = {
            "rgb": np.dtype(np.float32),
            "alpha": np.dtype(np.float32),
            "depth": np.dtype(np.float32),
            "semantic": np.dtype(np.int64),
            "valid_depth": np.dtype(np.bool_),
        }
        for key in REQUIRED_CANDIDATE_ARRAYS:
            if arrays[key].shape != expected_shapes[key]:
                raise ValueError(
                    f"Candidate {key} shape mismatch in {candidate_path}: "
                    f"expected {expected_shapes[key]}, got {arrays[key].shape}."
                )
            if arrays[key].dtype != expected_dtypes[key]:
                raise ValueError(
                    f"Candidate {key} dtype mismatch in {candidate_path}: "
                    f"expected {expected_dtypes[key]}, got {arrays[key].dtype}."
                )
        rgb = arrays["rgb"]
        alpha = arrays["alpha"]
        depth = arrays["depth"]
        semantic = arrays["semantic"]
        valid_depth = arrays["valid_depth"]
        if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError(f"Candidate RGB contract failed in {candidate_path}")
        if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
            raise ValueError(f"Candidate alpha contract failed in {candidate_path}")
        expected_valid_depth = np.isfinite(depth) & (depth > 0)
        if not np.array_equal(valid_depth, expected_valid_depth):
            raise ValueError(f"Candidate valid_depth contract failed in {candidate_path}")
        if not np.isposinf(depth[~valid_depth]).all():
            raise ValueError(f"Candidate invalid depth must be positive infinity in {candidate_path}")
        if np.any((depth[valid_depth] < 0.01) | (depth[valid_depth] > 100.0)):
            raise ValueError(f"Candidate valid depth exceeds clipping range in {candidate_path}")
        if np.any(valid_depth & (alpha <= 0)):
            raise ValueError(f"Candidate valid depth lacks alpha coverage in {candidate_path}")
        if np.any((semantic < -1) | (semantic > 0)):
            raise ValueError(f"Candidate semantic range failed in {candidate_path}")
        if np.any((semantic >= 0) & (alpha <= 0)):
            raise ValueError(f"Candidate semantic foreground lacks alpha in {candidate_path}")
        foreground_counts = {
            "alpha": int(np.count_nonzero(alpha > 0)),
            "partial_alpha": int(np.count_nonzero((alpha > 0) & (alpha < 1))),
            "background_alpha": int(np.count_nonzero(alpha == 0)),
            "depth": int(np.count_nonzero(valid_depth)),
            "background_depth": int(np.count_nonzero(~valid_depth)),
            "semantic": int(np.count_nonzero(semantic == 0)),
            "background_semantic": int(np.count_nonzero(semantic == -1)),
            "nonzero_rgb": int(np.count_nonzero(rgb > 0)),
        }
        if min(foreground_counts.values()) < MIN_FOREGROUND_PIXELS:
            raise ValueError(f"Candidate lacks required foreground signal in {candidate_path}: {foreground_counts}.")
        if np.any(rgb[alpha == 0] != 0):
            raise ValueError(f"Candidate background RGB must be black in {candidate_path}")
        candidate_color_space = str(_required_scalar(candidate["color_space"], name="color_space"))
        if candidate_color_space != "display_srgb":
            raise ValueError(f"Candidate color space mismatch in {candidate_path}")
        candidate_background = np.asarray(candidate["background"])
        if (
            candidate_background.shape != (3,)
            or candidate_background.dtype != np.dtype(np.float32)
            or not np.isfinite(candidate_background).all()
            or not np.array_equal(candidate_background, np.zeros(3, dtype=np.float32))
        ):
            raise ValueError(f"Candidate background contract failed in {candidate_path}")
        candidate_camera_bundle_id = str(_required_scalar(candidate["camera_bundle_id"], name="camera_bundle_id"))
        if candidate_camera_bundle_id != camera_bundle_id:
            raise ValueError(f"Candidate camera bundle ID mismatch in {candidate_path}")
        shapes = {key: list(arrays[key].shape) for key in sorted(REQUIRED_CANDIDATE_ARRAYS)}
        dtypes = {key: str(arrays[key].dtype) for key in sorted(REQUIRED_CANDIDATE_ARRAYS)}

    return {
        "schema_version": summary["schema_version"],
        "frames": summary["frames"],
        "warmup": summary["warmup"],
        "mode": lane.mode,
        "focal_scale": lane.focal_scale,
        "candidate": str(candidate_path),
        "candidate_sha256": file_sha256(candidate_path),
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "stage": str(stage_path),
        "stage_sha256": file_sha256(stage_path),
        "stage_text": stage_text,
        "camera_bundle": str(camera_bundle_path),
        "camera_bundle_sha256": file_sha256(camera_bundle_path),
        "camera_bundle_payload": camera_bundle,
        "camera_bundle_id": camera_bundle_id,
        "scene_checksum": scene_checksum,
        "scene_tensor_checksum": scene_tensor_checksum,
        "projection_readback": projection_readback,
        "sorting_readback": sorting_readback,
        "provenance_contract": {
            key: provenance[key]
            for key in (
                "project_commit",
                "ovrtx_commit",
                "ovrtx_version",
                "renderer_version",
                "script_sha256",
            )
        },
        "candidate_shapes": shapes,
        "candidate_dtypes": dtypes,
        "foreground_counts": foreground_counts,
        "custom_fidelity_final_pass_observed": summary.get("final_pass"),
        "custom_fidelity_final_pass_consumed": False,
        "process_receipt": str(process_receipt_path),
        "process_receipt_sha256": file_sha256(process_receipt_path),
        "process_execution": process_receipt,
    }


def _normalized_stage_text(
    stage_text: str,
    *,
    normalize_projection: bool,
    normalize_focal: bool,
) -> str:
    normalized = stage_text
    if normalize_projection:
        normalized = PROJECTION_MODE_PATTERN.sub(r'\1"__PROJECTION_MODE__"', normalized)
    if normalize_focal:
        normalized = FOCAL_LENGTH_PATTERN.sub(r"\1__FOCAL_LENGTH__", normalized)
    return normalized


def _normalized_camera_bundle(camera_bundle: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(camera_bundle)
    normalized.pop("bundle_id", None)
    cameras = normalized.get("cameras", [])
    for camera in cameras:
        intrinsics = camera.get("intrinsics")
        if isinstance(intrinsics, list) and len(intrinsics) == 3:
            intrinsics[0][0] = "__FOCAL_X__"
            intrinsics[1][1] = "__FOCAL_Y__"
    return normalized


def validate_cross_lane_contract(
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_names = {
        "prime",
        "positive-control-candidate",
        "perspective-candidate",
        "tangential-candidate",
        "tangential-repeat",
        "perspective-repeat",
        "positive-control-repeat",
    }
    if set(evidence) != expected_names:
        raise ValueError("Cross-lane validation requires the exact seven-lane evidence set.")
    ordered_names = [
        "prime",
        "positive-control-candidate",
        "perspective-candidate",
        "tangential-candidate",
        "tangential-repeat",
        "perspective-repeat",
        "positive-control-repeat",
    ]
    scene_checksums = {evidence[name]["scene_checksum"] for name in ordered_names}
    if len(scene_checksums) != 1:
        raise ValueError("Projection-mode lanes used different scene checksums.")
    scene_tensor_checksums = {evidence[name]["scene_tensor_checksum"] for name in ordered_names}
    if len(scene_tensor_checksums) != 1:
        raise ValueError("Projection-mode lanes uploaded different scene tensors.")
    provenance_contracts = {json.dumps(evidence[name]["provenance_contract"], sort_keys=True) for name in ordered_names}
    if len(provenance_contracts) != 1:
        raise ValueError("Projection-mode lanes used different source/runtime provenance.")

    base_names = [
        "prime",
        "perspective-candidate",
        "tangential-candidate",
        "tangential-repeat",
        "perspective-repeat",
    ]
    control_names = ["positive-control-candidate", "positive-control-repeat"]
    base_camera_ids = {evidence[name]["camera_bundle_id"] for name in base_names}
    control_camera_ids = {evidence[name]["camera_bundle_id"] for name in control_names}
    if len(base_camera_ids) != 1 or len(control_camera_ids) != 1:
        raise ValueError("Same-camera candidate/repeat lanes do not share camera bundle IDs.")
    if base_camera_ids == control_camera_ids:
        raise ValueError("Focal positive control did not change the camera bundle ID.")
    normalized_cameras = {
        json.dumps(
            _normalized_camera_bundle(evidence[name]["camera_bundle_payload"]),
            sort_keys=True,
        )
        for name in ordered_names
    }
    if len(normalized_cameras) != 1:
        raise ValueError("Lane camera bundles differ in more than focal intrinsics.")

    stage_texts = {name: Path(evidence[name]["stage"]).read_text(encoding="utf-8") for name in ordered_names}
    if stage_texts["prime"] != stage_texts["perspective-candidate"]:
        raise ValueError("Discarded prime stage differs from the base perspective stage.")
    if stage_texts["perspective-candidate"] != stage_texts["perspective-repeat"]:
        raise ValueError("Perspective candidate/repeat stages differ.")
    if stage_texts["tangential-candidate"] != stage_texts["tangential-repeat"]:
        raise ValueError("Tangential candidate/repeat stages differ.")
    if stage_texts["positive-control-candidate"] != stage_texts["positive-control-repeat"]:
        raise ValueError("Positive-control candidate/repeat stages differ.")
    projection_normalized = {
        _normalized_stage_text(
            stage_texts[name],
            normalize_projection=True,
            normalize_focal=False,
        )
        for name in (
            "perspective-candidate",
            "tangential-candidate",
            "tangential-repeat",
            "perspective-repeat",
        )
    }
    if len(projection_normalized) != 1:
        raise ValueError("Mode stages differ in more than projectionModeHint.")
    fully_normalized = {
        _normalized_stage_text(
            stage_texts[name],
            normalize_projection=True,
            normalize_focal=True,
        )
        for name in ordered_names
    }
    if len(fully_normalized) != 1:
        raise ValueError("Control stages differ from mode stages beyond focalLength/projectionModeHint.")
    return {
        "pass": True,
        "scene_checksum": next(iter(scene_checksums)),
        "scene_tensor_checksum": next(iter(scene_tensor_checksums)),
        "base_camera_bundle_id": next(iter(base_camera_ids)),
        "control_camera_bundle_id": next(iter(control_camera_ids)),
        "source_runtime_provenance_equal": True,
        "mode_stages_differ_only_in_projection_mode_hint": True,
        "control_stages_differ_only_in_focal_length": True,
        "camera_bundles_differ_only_in_focal_intrinsics": True,
        "same_configuration_repeats_identical": True,
    }


def launch_lane(
    args: argparse.Namespace,
    lane: LaneSpec,
    *,
    outer_deadline: float,
) -> tuple[Path, float, dict[str, Any]]:
    lane_dir = args.output_root / lane.name
    lane_dir.mkdir(parents=True, exist_ok=False)
    command = lane_command(args, lane, lane_dir)
    log_path = lane_dir / "run.log"
    start = time.perf_counter()
    lane_timeout_seconds = bounded_timeout_seconds(
        outer_deadline,
        args.per_lane_timeout_seconds,
        phase=f"lane {lane.name}",
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = run_bounded_process(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=lane_timeout_seconds,
            )
    except BaseException as exc:
        duration_seconds = time.perf_counter() - start
        write_process_receipt(
            lane_dir,
            command=command,
            returncode=getattr(exc, "child_returncode", None),
            timed_out=isinstance(exc, subprocess.TimeoutExpired),
            timeout_seconds=lane_timeout_seconds,
            duration_seconds=duration_seconds,
            log_path=log_path,
            error=exc,
        )
        raise
    duration_seconds = time.perf_counter() - start
    receipt_path = write_process_receipt(
        lane_dir,
        command=command,
        returncode=completed.returncode,
        timed_out=False,
        timeout_seconds=lane_timeout_seconds,
        duration_seconds=duration_seconds,
        log_path=log_path,
    )
    bounded_timeout_seconds(
        outer_deadline,
        1,
        phase=f"post-process checkpoint for lane {lane.name}",
    )
    if completed.returncode != 0:
        signal_receipt = _signal_receipt(completed.returncode)
        signal_suffix = f" ({signal_receipt['name']})" if signal_receipt is not None else ""
        raise RuntimeError(
            f"Temporal lane {lane.name} failed with exit {completed.returncode}{signal_suffix}; see {receipt_path}."
        )
    validate_process_receipt(
        lane_dir,
        expected_command=command,
    )
    log_checker_command = [str(LOG_CHECKER), str(log_path), TEMPORAL_SUCCESS_MARKER]
    checked = run_bounded_process(
        log_checker_command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=bounded_timeout_seconds(
            outer_deadline,
            30,
            phase=f"log validation for lane {lane.name}",
        ),
    )
    bounded_timeout_seconds(
        outer_deadline,
        1,
        phase=f"post-log checkpoint for lane {lane.name}",
    )
    if checked.returncode != 0:
        raise RuntimeError(f"Temporal lane log contract failed for {lane.name}: {checked.stdout}{checked.stderr}")
    marker_contract = require_exact_log_markers(
        log_path,
        TEMPORAL_CLEANUP_SUCCESS_MARKER,
        TEMPORAL_SUCCESS_MARKER,
    )
    try:
        evidence = validate_lane_artifacts(
            lane_dir,
            lane,
            expected_command=command,
            warmup=args.warmup,
            width=args.width,
            height=args.height,
        )
    except Exception as exc:
        raise LaneEvidenceInvalid(f"Lane {lane.name} produced invalid evidence: {exc}") from exc
    bounded_timeout_seconds(
        outer_deadline,
        1,
        phase=f"post-validation checkpoint for lane {lane.name}",
    )
    log_sha256 = file_sha256(log_path)
    evidence["duration_seconds"] = duration_seconds
    evidence["log"] = str(log_path)
    evidence["log_sha256"] = log_sha256
    evidence["log_checker_contract"] = {
        "command": log_checker_command,
        "returncode": checked.returncode,
        **marker_contract,
        "log": str(log_path),
        "log_sha256": log_sha256,
        # The temporal lane launches ovrtx.Renderer directly, not Kit's
        # SimulationApp, so run.log is the complete process log. There is no
        # separate Kit-internal log to discover or validate for this gate.
        "log_scope": "standalone_ovrtx_stdout_stderr",
        "separate_kit_internal_log_expected": False,
    }
    return lane_dir, duration_seconds, evidence


def cold_suspect_lanes(durations: dict[str, float]) -> tuple[float, list[str]]:
    measured = [duration for name, duration in durations.items() if name != "prime"]
    if len(measured) != LANE_COUNT - 1:
        raise ValueError("Cold-lane analysis requires every measured lane.")
    median = float(statistics.median(measured))
    suspects = sorted(
        name for name, duration in durations.items() if name != "prime" and duration > COLD_LANE_WALL_RATIO * median
    )
    return median, suspects


def write_incomplete_report(
    output_root: Path,
    *,
    completed_lanes: list[str],
    error: BaseException,
    phase: str = "lane_execution",
) -> Path:
    classifications = {
        "lane_execution": "INCOMPLETE_LANE_SET",
        "evidence_validation": "INVALID_LANE_EVIDENCE",
        "audit_execution": "INCOMPLETE_AUDIT",
    }
    if phase not in classifications:
        raise ValueError(f"Unknown incomplete gate phase: {phase!r}.")
    report = {
        "schema_version": "ovrtx-projection-mode-gate-execution/v1",
        "pass": False,
        "classification": classifications[phase],
        "phase": phase,
        "completed_lanes": completed_lanes,
        "required_lane_count": LANE_COUNT,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    path = output_root / "activation-gate-incomplete.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_audit_report(report: Any) -> dict[str, Any]:
    """Reject malformed auditor output before it can influence gate success."""
    if not isinstance(report, dict):
        raise ValueError("Projection-mode auditor report must be a JSON object.")
    if report.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(
            "Projection-mode auditor schema mismatch: "
            f"expected {AUDIT_SCHEMA_VERSION!r}, got {report.get('schema_version')!r}."
        )
    classification = report.get("classification")
    if classification not in AUDIT_CLASSIFICATIONS:
        raise ValueError(f"Unknown projection-mode audit classification: {classification!r}.")
    invalid_boolean_fields = sorted(field for field in AUDIT_BOOLEAN_FIELDS if type(report.get(field)) is not bool)
    if invalid_boolean_fields:
        raise ValueError(f"Projection-mode auditor boolean fields are missing or invalid: {invalid_boolean_fields}.")
    if report["activation_proof_required"] is not True:
        raise ValueError("Gate auditor did not acknowledge --require-activation-proof.")
    if report["positive_control_required"] is not True:
        raise ValueError("Gate auditor did not acknowledge --require-positive-control.")
    if report["repeat_controls_present"] is not True:
        raise ValueError("Gate auditor did not consume both required repeat controls.")
    if report["positive_control_present"] is not True:
        raise ValueError("Gate auditor did not consume the complete positive control.")
    expected_token_proof = (
        report["stage_authoring_valid"] and report["runtime_readback_valid"] and report["matched_lane_contract_valid"]
    )
    if report["token_proof_valid"] is not expected_token_proof:
        raise ValueError("Gate auditor token proof fields are internally inconsistent.")
    expected_behavioral_proof = (
        report["observably_different"]
        and report["repeat_controls_present"]
        and report["effect_above_repeat_noise"]
        and report["candidate_outputs_valid"]
        and report["matched_lane_contract_valid"]
    )
    if report["behavioral_activation_valid"] is not expected_behavioral_proof:
        raise ValueError("Gate auditor behavioral proof fields are internally inconsistent.")
    expected_positive_control_valid = (
        report["matched_lane_contract_valid"]
        and report["positive_control_stable"]
        and report["positive_control_detected"]
    )
    if report["positive_control_valid"] is not expected_positive_control_valid:
        raise ValueError("Gate auditor positive control fields are internally inconsistent.")
    expected_activation_proof = (
        report["token_proof_valid"] and report["behavioral_activation_valid"] and report["positive_control_valid"]
    )
    if report["activation_proof_valid"] is not expected_activation_proof:
        raise ValueError("Gate auditor activation proof fields are internally inconsistent.")
    if report["pass"] is not expected_activation_proof:
        raise ValueError("Gate auditor pass disagrees with its required proof fields.")
    if report["positive_control_valid"] and not report["positive_control_detected"]:
        raise ValueError("Valid positive control must have a detected geometry effect.")
    if report["positive_control_valid"] and not report["positive_control_stable"]:
        raise ValueError("Valid positive control must have a stable repeat.")

    if not report["repeat_controls_present"]:
        expected_classification = "MISSING_REPEAT_CONTROLS"
    elif not report["positive_control_present"]:
        expected_classification = "MISSING_POSITIVE_CONTROL"
    elif not report["candidate_outputs_valid"]:
        expected_classification = "INVALID_CANDIDATE_OUTPUT"
    elif not report["stage_authoring_valid"]:
        expected_classification = "INVALID_MODE_AUTHORING"
    elif not report["runtime_readback_valid"]:
        expected_classification = "INVALID_RUNTIME_MODE_READBACK"
    elif not report["matched_lane_contract_valid"]:
        expected_classification = "INVALID_MATCHED_LANE_CONTRACT"
    elif not report["positive_control_stable"]:
        expected_classification = "POSITIVE_CONTROL_UNSTABLE"
    elif not report["positive_control_detected"]:
        expected_classification = "POSITIVE_CONTROL_NOT_DETECTED"
    elif not report["observably_different"]:
        expected_classification = "NO_OBSERVABLE_MODE_EFFECT"
    elif not report["effect_above_repeat_noise"]:
        expected_classification = "MODE_EFFECT_WITHIN_REPEAT_NOISE"
    else:
        expected_classification = "OBSERVABLE_MODE_EFFECT"
    if classification != expected_classification:
        raise ValueError(
            "Projection-mode auditor classification disagrees with its proof fields: "
            f"expected {expected_classification!r}, got {classification!r}."
        )

    if report["pass"] is True:
        required_success_fields = {
            "token_proof_valid",
            "behavioral_activation_valid",
            "stage_authoring_valid",
            "observably_different",
            "effect_above_repeat_noise",
            "runtime_readback_valid",
            "positive_control_detected",
            "positive_control_valid",
        }
        failed_success_fields = sorted(field for field in required_success_fields if report[field] is not True)
        if failed_success_fields:
            raise ValueError(f"Passing projection-mode audit has false proof fields: {failed_success_fields}.")
        if classification != "OBSERVABLE_MODE_EFFECT":
            raise ValueError("Passing projection-mode audit must be classified OBSERVABLE_MODE_EFFECT.")
    elif classification == "OBSERVABLE_MODE_EFFECT":
        raise ValueError("Failed projection-mode audit cannot claim OBSERVABLE_MODE_EFFECT.")
    return report


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, _raise_on_termination)
    gate_start = time.perf_counter()
    outer_deadline = gate_start + args.outer_timeout_seconds
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to mix activation evidence in non-empty {args.output_root}.")
    args.output_root.mkdir(parents=True, exist_ok=True)

    lane_specs = plan_lanes(args)
    if len(lane_specs) != LANE_COUNT:
        raise RuntimeError("Lane plan does not match the timeout contract.")
    lane_dirs: dict[str, Path] = {}
    durations: dict[str, float] = {}
    evidence: dict[str, dict[str, Any]] = {}
    completed_lanes: list[str] = []
    try:
        for lane in lane_specs:
            print(
                "ACTIVATION_GATE_LANE "
                f"name={lane.name} mode={lane.mode} frames={lane.frames} "
                f"focal_scale={lane.focal_scale:.10g}",
                flush=True,
            )
            lane_dir, duration, lane_evidence = launch_lane(
                args,
                lane,
                outer_deadline=outer_deadline,
            )
            lane_dirs[lane.name] = lane_dir
            durations[lane.name] = duration
            evidence[lane.name] = lane_evidence
            completed_lanes.append(lane.name)
    except Exception as exc:
        phase = "evidence_validation" if isinstance(exc, LaneEvidenceInvalid) else "lane_execution"
        incomplete_path = write_incomplete_report(
            args.output_root,
            completed_lanes=completed_lanes,
            error=exc,
            phase=phase,
        )
        print(
            f"PROJECTION_MODE_ACTIVATION_GATE_INCOMPLETE report={incomplete_path}",
            flush=True,
        )
        return 1

    try:
        cross_lane_contract = validate_cross_lane_contract(evidence)
        bounded_timeout_seconds(
            outer_deadline,
            1,
            phase="cross-lane evidence validation",
        )
    except Exception as exc:
        incomplete_path = write_incomplete_report(
            args.output_root,
            completed_lanes=completed_lanes,
            error=exc,
            phase="evidence_validation",
        )
        print(
            f"PROJECTION_MODE_ACTIVATION_GATE_INCOMPLETE report={incomplete_path}",
            flush=True,
        )
        return 1

    median_duration, cold_suspects = cold_suspect_lanes(durations)
    execution = {
        "schema_version": "ovrtx-projection-mode-gate-execution/v1",
        "complete": len(completed_lanes) == LANE_COUNT,
        "lane_order": completed_lanes,
        "lane_specs": [asdict(lane) for lane in lane_specs],
        "lane_evidence": evidence,
        "durations_seconds": durations,
        "median_measured_duration_seconds": median_duration,
        "cold_lane_wall_ratio": COLD_LANE_WALL_RATIO,
        "cold_suspect_lanes": cold_suspects,
        "cross_lane_contract": cross_lane_contract,
        "per_lane_timeout_seconds": args.per_lane_timeout_seconds,
        "declared_outer_timeout_seconds": args.outer_timeout_seconds,
        "required_outer_timeout_seconds": required_outer_timeout_seconds(args.per_lane_timeout_seconds),
        "outer_deadline_enforced": True,
        "gate_elapsed_before_audit_seconds": time.perf_counter() - gate_start,
        "temporal_capture": {
            "width": args.width,
            "height": args.height,
            "warmup": args.warmup,
            "frames": args.frames,
            "prime_frames": args.prime_frames,
            "color_source": "OVRTX LdrColor temporally averaged in float32",
            "ldr_input_quantization_step": 1.0 / 255.0,
            "custom_fidelity_final_pass_consumed": False,
        },
        "positive_control_scope": (
            "camera focalLength authoring and output capture; ParticleField "
            "projectionModeHint composition is proven separately by runtime "
            "readback, while kernel consumption remains the behavior under test"
        ),
    }
    execution_path = args.output_root / "gate-execution.json"
    execution_path.write_text(
        json.dumps(execution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    raw_audit_report_path = args.output_root / "projection-mode-audit.json"
    gate_report_path = args.output_root / "activation-gate-report.json"
    audit_command = [
        args.python,
        str(AUDIT_SCRIPT),
        "--perspective-candidate",
        str(lane_dirs["perspective-candidate"] / CANDIDATE_NPZ),
        "--tangential-candidate",
        str(lane_dirs["tangential-candidate"] / CANDIDATE_NPZ),
        "--perspective-repeat",
        str(lane_dirs["perspective-repeat"] / CANDIDATE_NPZ),
        "--tangential-repeat",
        str(lane_dirs["tangential-repeat"] / CANDIDATE_NPZ),
        "--perspective-stage",
        str(lane_dirs["perspective-candidate"] / STAGE_FILE),
        "--tangential-stage",
        str(lane_dirs["tangential-candidate"] / STAGE_FILE),
        "--perspective-summary",
        str(lane_dirs["perspective-candidate"] / SUMMARY_JSON),
        "--tangential-summary",
        str(lane_dirs["tangential-candidate"] / SUMMARY_JSON),
        "--positive-control-candidate",
        str(lane_dirs["positive-control-candidate"] / CANDIDATE_NPZ),
        "--positive-control-stage",
        str(lane_dirs["positive-control-candidate"] / STAGE_FILE),
        "--positive-control-summary",
        str(lane_dirs["positive-control-candidate"] / SUMMARY_JSON),
        "--positive-control-repeat-candidate",
        str(lane_dirs["positive-control-repeat"] / CANDIDATE_NPZ),
        "--positive-control-repeat-stage",
        str(lane_dirs["positive-control-repeat"] / STAGE_FILE),
        "--positive-control-repeat-summary",
        str(lane_dirs["positive-control-repeat"] / SUMMARY_JSON),
        "--expected-width",
        str(args.width),
        "--expected-height",
        str(args.height),
        "--expected-control-focal-ratio",
        str(CONTROL_FOCAL_SCALE_FACTOR),
        "--perspective-repeat-stage",
        str(lane_dirs["perspective-repeat"] / STAGE_FILE),
        "--perspective-repeat-summary",
        str(lane_dirs["perspective-repeat"] / SUMMARY_JSON),
        "--tangential-repeat-stage",
        str(lane_dirs["tangential-repeat"] / STAGE_FILE),
        "--tangential-repeat-summary",
        str(lane_dirs["tangential-repeat"] / SUMMARY_JSON),
        "--require-positive-control",
        "--require-activation-proof",
        "--output",
        str(raw_audit_report_path),
    ]
    audit_log_path = args.output_root / "audit.log"
    try:
        with audit_log_path.open("w", encoding="utf-8") as audit_log:
            completed = run_bounded_process(
                audit_command,
                cwd=PROJECT_ROOT,
                stdout=audit_log,
                stderr=subprocess.STDOUT,
                timeout=bounded_timeout_seconds(
                    outer_deadline,
                    300,
                    phase="projection-mode audit",
                ),
            )
        bounded_timeout_seconds(
            outer_deadline,
            1,
            phase="post-auditor process checkpoint",
        )
        if not raw_audit_report_path.is_file():
            raise RuntimeError(f"Projection-mode auditor wrote no report; see {audit_log_path}.")
        report = validate_audit_report(json.loads(raw_audit_report_path.read_text(encoding="utf-8")))
        bounded_timeout_seconds(
            outer_deadline,
            1,
            phase="post-audit validation checkpoint",
        )
    except Exception as exc:
        incomplete_path = write_incomplete_report(
            args.output_root,
            completed_lanes=completed_lanes,
            error=exc,
            phase="audit_execution",
        )
        print(
            f"PROJECTION_MODE_ACTIVATION_GATE_INCOMPLETE report={incomplete_path}",
            flush=True,
        )
        return 1
    expected_audit_code = 0 if report["pass"] is True else 1
    gate_pass = report["pass"] is True
    gate_classification = report["classification"]
    if completed.returncode != expected_audit_code:
        gate_classification = "AUDITOR_EXECUTION_INVALID"
        gate_pass = False
    elif cold_suspects:
        gate_classification = "COLD_LANE_CONTAMINATION"
        gate_pass = False
    try:
        bounded_timeout_seconds(
            outer_deadline,
            1,
            phase="final gate verdict",
        )
    except TimeoutError as exc:
        incomplete_path = write_incomplete_report(
            args.output_root,
            completed_lanes=completed_lanes,
            error=exc,
            phase="audit_execution",
        )
        print(
            f"PROJECTION_MODE_ACTIVATION_GATE_INCOMPLETE report={incomplete_path}",
            flush=True,
        )
        return 1
    gate_report = {
        "schema_version": "ovrtx-projection-mode-activation-gate/v1",
        "pass": gate_pass,
        "classification": gate_classification,
        "activation_proof_valid": gate_pass,
        "audit_classification": report["classification"],
        "audit_report": report,
        "audit_artifact": {
            "path": str(raw_audit_report_path),
            "sha256": file_sha256(raw_audit_report_path),
            "exit_code": completed.returncode,
            "expected_exit_code": expected_audit_code,
            "log": str(audit_log_path),
            "log_sha256": file_sha256(audit_log_path),
        },
        "gate_execution": execution,
    }
    gate_report_path.write_text(
        json.dumps(gate_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if gate_report["pass"] is True:
        print(
            f"PROJECTION_MODE_ACTIVATION_GATE_PASS report={gate_report_path}",
            flush=True,
        )
        return 0
    print(
        "PROJECTION_MODE_ACTIVATION_GATE_BLOCKED "
        f"classification={gate_report['classification']} report={gate_report_path}",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
