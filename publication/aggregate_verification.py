#!/usr/bin/env python3
"""Strict, publication-local aggregate verification.

This module intentionally lives outside the frozen renderer and benchmark
trees.  It writes and verifies ``publication/verification.json`` from a
declarative specification, and can resolve every referenced artifact either
from its measurement-host path or from a relocated evidence bundle's
content-addressed object store.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


SPEC_SCHEMA = "publication-verification-spec-v1"
VERIFICATION_SCHEMA = "publication-verification-v1"
GATE_RESULT_SCHEMA = "publication-gate-result-v1"
BUNDLE_SCHEMA = "flashgs-matched-evidence-bundle-v1"
VERIFICATION_LOGICAL_PATH = "publication/verification.json"
ABLATION_LOGICAL_PATH = "publication/evidence/custom-vectorization-ablation.json"
MACHINE_PROVENANCE_LOGICAL_PATH = "publication/machine-provenance.json"
MACHINE_PROVENANCE_SCHEMA = "publication-machine-provenance-v1"
PRIMARY_BATCHES = (1, 8, 32, 64, 128, 256, 512, 1024)
OUTPUTS = ("rgb", "alpha", "depth", "semantic_id")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
MACHINE_ENVIRONMENT_FIELDS = (
    "compiler_versions",
    "compute_capability",
    "cuda_runtime",
    "driver",
    "gpu_name",
    "gpu_uuid",
    "torch",
    "torch_cuda_arch_list",
)

REQUIRED_GATE_IDS = (
    "flashgs-b64-repair",
    "repair-compute-sanitizer",
    "python-unit-contract",
    "native-cuda-smoke",
    "native-determinism",
    "native-multi-scene",
    "isaac-graphics",
    "all-output-fidelity",
    "custom-vectorization-ablation",
    "deterministic-replay",
    "compute-sanitizer-memcheck",
    "bounded-stability-soak",
)
GPU_REQUIRED = {
    gate_id: gate_id != "python-unit-contract"
    for gate_id in REQUIRED_GATE_IDS
}
NATIVE_ROLE = {
    gate_id: (
        "flashgs"
        if gate_id in {"flashgs-b64-repair", "repair-compute-sanitizer"}
        else "custom"
    )
    for gate_id in REQUIRED_GATE_IDS
}
NATIVE_SENTINELS = {
    "native-cuda-smoke": "CUSTOM_CUDA_BACKEND_SMOKE_OK",
    "native-determinism": "DETERMINISTIC_CUDA_SMOKE_OK",
    "native-multi-scene": "MULTI_SCENE_CUDA_SMOKE_OK",
}
ISAAC_LANES = {
    "lifecycle-default": "ISAAC_HEADLESS_SMOKE_OK",
    "extension": "ISAACSIM_GAUSSIAN_RENDERER_SMOKE_OK",
    "fabric-camera": "FABRIC_CAMERA_SMOKE_OK",
}

COMMAND_SCHEMA = "publication-command-record-v1"
RENDERER_RUN_SCHEMA = "flashgs-matched-renderer-run-v4"
FIDELITY_SCHEMA = "flashgs-matched-fidelity-v4"
SOURCE_MANIFEST_SCHEMA = "renderer-source-manifest-v2"
B64_REPAIR_SCHEMA = "flashgs-b64-repair-verification-v1"
B64_ORACLE_REPORT_SCHEMA = "flashgs-b64-repair-gsplat-all-pixel-v2"
B64_ORACLE_MANIFEST_SCHEMA = "flashgs-matched-gsplat-oracle-v4"
B64_KNOWN_FAILURE_SCHEMA = "flashgs-b64-known-failure-cases-v1"
B64_DIAGNOSIS_LOCK_SCHEMA = "flashgs-b64-diagnosis-lock-v1"
B64_DIAGNOSIS_INDEX_SCHEMA = "flashgs-b64-diagnosis-index-v1"
B64_PRIVACY_REDACTION_SCHEMA = "flashgs-b64-privacy-redaction-v1"
B64_PRIVACY_REDACTION_INVENTORY_SCHEMA = (
    "flashgs-b64-privacy-redaction-inventory-v1"
)
BOUNDED_WRAPPER_SCHEMA = "publication-bounded-gate-wrapper-v1"
NODE_OCCUPANCY_SCHEMA = "flashgs-matched-node-occupancy-v2"
B64_TRANSITIVE_EVIDENCE_KEYS = frozenset(
    {
        "repair_report",
        "repeat_repair_report",
        "primary_repair_full_raw",
        "primary_repair_rgb_only_raw",
        "repeat_repair_full_raw",
        "repeat_repair_rgb_only_raw",
        "oracle_report",
        "oracle",
        "oracle_manifest",
        "oracle_camera_bundle",
        "oracle_node_occupancy",
        "diagnosis_index",
        "diagnosis_lock",
        "diagnosis_privacy_redaction_inventory",
        "diagnosis_privacy_redaction_manifest",
        "diagnosis_privacy_redaction_tool",
        "known_failure_manifest",
        *(
            f"{prefix}_support_{suffix}"
            for prefix in ("primary", "repeat", "oracle")
            for suffix in ("wrapper", "command", "log", "exit_status", "occupancy")
        ),
    }
)
REPLAY_WORKER_SCHEMA = "publication-deterministic-digest-smoke-v1"
REPLAY_SCHEMA = "publication-deterministic-cross-process-replay-v1"
REPLAY_EVIDENCE_KEYS = frozenset(
    {
        "process_1",
        "process_2",
        "raw_result",
        "runner_script",
        "source_smoke",
        "worker_script",
    }
)
REPLAY_ARGUMENTS = {
    "batch": 4,
    "gaussians": 1_024,
    "height": 64,
    "iterations": 32,
    "ray_gaussian_evaluation": False,
    "tile_size": 1,
    "width": 64,
}
REPLAY_CLAIM_SCOPE = {
    **REPLAY_ARGUMENTS,
    "camera_configuration": "identical-centered-cameras",
    "claim": "synthetic-equal-depth-deterministic-mode-replay-only",
    "gaussian_configuration": "isotropic-identity-rotation-equal-depth",
    "renderer_mode": "deterministic",
}
REPLAY_COMPARISON_METHOD = "sha256(dtype,shape,raw-c-order-bytes)"
REPLAY_RENDERER_CONFIGURATION = {
    "adaptive_capacity": True,
    "covariance_epsilon": 0.3,
    "deterministic": True,
    "fixed_capacity_sort": False,
    "gaussian_support_sigma": 3.0,
    "output_srgb": False,
    "ray_gaussian_evaluation": False,
    "semantic_min_alpha": 0.01,
    "tile_size": 1,
}
REPLAY_WORKER_FIELDS = frozenset(
    {
        "arguments",
        "bitwise_equal",
        "capacity_invariant_bitwise_equal",
        "center_semantic",
        "claim_scope",
        "comparison_method",
        "counters",
        "equal_depth_gaussian_id_order",
        "equal_depth_records_checked",
        "fixture_sha256",
        "native_build_contract",
        "native_extension",
        "output_contract",
        "output_digests",
        "pass",
        "renderer_configuration",
        "schema_version",
        "source_smoke",
    }
)
REPLAY_RESULT_FIELDS = frozenset(
    {
        "claim_scope",
        "comparison_method",
        "cross_process_output_hashes_equal",
        "fixture_sha256",
        "fresh_processes",
        "native_build_contract",
        "native_extension",
        "output_contract",
        "output_digests",
        "pass",
        "process_payloads_equal",
        "processes",
        "replays_per_process",
        "renderer_configuration",
        "runner_script",
        "schema_version",
        "source_smoke",
        "within_process_bitwise_equal",
        "worker_script",
        "zero_overflow",
    }
)
REPLAY_COUNTER_FIELDS = frozenset(
    {
        "active_tiles",
        "intersection_overflow",
        "tile_intersections",
        "visible_gaussians",
        "visible_overflow",
    }
)
REPLAY_OUTPUT_BOOLEAN_FIELDS = frozenset(
    {
        "alpha_in_range",
        "contiguous",
        "cuda_resident",
        "dtypes_match",
        "finite_alpha",
        "finite_foreground_depth",
        "finite_rgb",
        "output_names_match",
        "shapes_match",
        "single_cuda_device",
        "valid",
        "valid_background_depth",
        "valid_background_semantics",
        "valid_foreground_semantics",
    }
)
REPLAY_OUTPUT_COUNT_FIELDS = frozenset(
    {
        "background_pixel_count",
        "foreground_pixel_count",
        "semantic_background_pixel_count",
        "semantic_foreground_pixel_count",
    }
)
REPLAY_NATIVE_BUILD_FIELDS = frozenset(
    {
        "build_directory",
        "cuda_flags",
        "cxx_flags",
        "module_name",
        "sources",
        "torch_cuda_arch_list",
    }
)
PRIMARY_TRAJECTORY_IDS = {
    1: "a901ed9a3d7a62bffc2860ecefa2cdf8eaf2196085bca067215ac0469fbc8dfe",
    8: "39ab99be70fef112dd7f82636f28d61e2b67764b7f7f7eb6d1e6f401bddb3ca3",
    32: "86febe653241edbbd94d321ad22ed3214c6ff09130434a3c3f5c587107cd5d15",
    64: "e30050bf2d873825cf7cdebbe799c911ddaee5cc15d90e1c0dd8adc2c2e62cc0",
    128: "c9a7ef7727761865263c3432954b259c54e2065dfd2326e65494583083704925",
    256: "f0a68d425ca12f38b9d62c6fd445caae9d75472bdf9bacc29f563b93817233f4",
    512: "f750294341d523b1b46f626b609fbd5e2a0a43d28f0aa35a4614d6f0f6ed8c7d",
    1024: "375a6711e86333621448c94ca1ad7985bd2687c4b585f68c823624725c223910",
}
MATCHED_PROJECTION_RULES = {
    "frustum_depth_interval": "inclusive-near-and-far",
    "invalid_covariance_2d": "determinant<=0",
    "projected_radius_covariance_diagonal_floor": None,
    "projected_radius_min_pixels": 0,
    "projected_radius_clip": "cull-only-when-both-radii<=0",
}

# Every command record must name the real bounded entry point for its gate.
# Tokens are matched against individual argv entries (or their basename), not
# against an unstructured shell string.
COMMAND_TOKEN_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "flashgs-b64-repair": (("compare_flashgs_b64_repair_oracle.py",),),
    "repair-compute-sanitizer": (
        ("compute-sanitizer",),
        ("--tool",),
        ("memcheck",),
        ("verify_flashgs_short_tail_synthetic.py",),
    ),
    "python-unit-contract": (("pytest",), ("tests",)),
    "native-cuda-smoke": (("custom_backend_smoke.py",),),
    "native-determinism": (("deterministic_cuda_smoke.py",),),
    "native-multi-scene": (("multi_scene_cuda_smoke.py",),),
    "isaac-graphics": (("run_isaac_graphics_gate.py",),),
    "all-output-fidelity": (
        ("summarize_flashgs_matched.py",),
        ("--verify-existing",),
    ),
    "custom-vectorization-ablation": (("run_flashgs_matched.py",),),
    "deterministic-replay": (
        ("run_deterministic_replay.py",),
        ("deterministic_digest_smoke.py",),
        ("deterministic_cuda_smoke.py",),
    ),
    "compute-sanitizer-memcheck": (
        ("compute-sanitizer",),
        ("--tool",),
        ("memcheck",),
        ("custom_backend_smoke.py",),
    ),
    "bounded-stability-soak": (("run_home_scan.py", "run_soak.py"),),
}


class VerificationError(ValueError):
    """Raised when a publication verification gate is not independently proven."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving any symlink component."""

    return path if path.is_absolute() else Path.cwd() / path


def _reject_symlink_components(path: Path, *, boundary: Path, label: str) -> None:
    """Reject a leaf or parent symlink and require lexical root containment."""

    raw = _lexical_absolute(path)
    root = boundary.absolute()
    try:
        raw.relative_to(root)
    except ValueError as error:
        raise VerificationError(f"{label} escapes the verification root: {raw}.") from error
    current = raw
    while True:
        if current.is_symlink():
            raise VerificationError(f"{label} crosses a symlink: {current}.")
        if current == root:
            return
        if current.parent == current:
            raise VerificationError(f"{label} does not descend from {root}.")
        current = current.parent


def _regular_within(path: Path, *, boundary: Path, label: str) -> Path:
    _reject_symlink_components(path, boundary=boundary, label=label)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise VerificationError(f"{label} is missing: {path}.") from error
    try:
        resolved.relative_to(boundary)
    except ValueError as error:
        raise VerificationError(f"{label} resolves outside the verification root: {path}.") from error
    if not resolved.is_file():
        raise VerificationError(f"{label} is not a regular file: {path}.")
    return resolved


def artifact_record(path: str | Path) -> dict[str, Any]:
    raw = Path(path)
    if raw.is_symlink():
        raise VerificationError(f"Artifact may not be a symlink: {raw}.")
    candidate = raw.resolve(strict=True)
    if not candidate.is_file():
        raise VerificationError(f"Artifact is not a regular file: {candidate}.")
    return {
        "bytes": candidate.stat().st_size,
        "path": str(candidate),
        "sha256": file_sha256(candidate),
    }


def _identity(value: Any, *, label: str) -> tuple[str, int]:
    if not isinstance(value, dict):
        raise VerificationError(f"{label} is not an artifact record.")
    sha256 = value.get("sha256")
    byte_count = value.get("bytes")
    if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
        raise VerificationError(f"{label} has an invalid SHA-256.")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise VerificationError(f"{label} has an invalid byte count.")
    return sha256, byte_count


def _same_artifact(left: Any, right: Any) -> bool:
    try:
        return _identity(left, label="left artifact") == _identity(right, label="right artifact")
    except VerificationError:
        return False


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise VerificationError(f"{label} is not a safe logical path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise VerificationError(f"{label} is not a safe relative path: {value!r}.")
    return path.as_posix()


def _parse_time(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} is missing.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise VerificationError(f"{label} is not ISO-8601: {value!r}.") from error
    if parsed.tzinfo is None:
        raise VerificationError(f"{label} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"{label} is not valid UTF-8 JSON.") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be a JSON object.")
    return value


class EvidenceRoot:
    """Resolve logical publication files and artifact records in two modes."""

    def __init__(self, root: Path, manifest: dict[str, Any] | None) -> None:
        self.root = root
        self.manifest = manifest
        self.mode = "bundle" if manifest is not None else "staging"
        self.logical: dict[str, dict[str, Any]] = {}
        if manifest is not None:
            for item in manifest.get("publication_artifacts", []):
                if not isinstance(item, dict):
                    raise VerificationError("Bundle publication artifact is malformed.")
                logical = _safe_relative(item.get("logical_path"), label="bundle logical path")
                if logical in self.logical:
                    raise VerificationError(f"Duplicate bundle logical path: {logical}.")
                self.logical[logical] = item

    @classmethod
    def open(cls, value: str | Path) -> "EvidenceRoot":
        raw_root = Path(value)
        if raw_root.is_symlink():
            raise VerificationError(f"Verification root may not be a symlink: {raw_root}.")
        root = raw_root.resolve(strict=True)
        if not root.is_dir():
            raise VerificationError(f"Verification root is not a regular directory: {root}.")
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink():
            raise VerificationError("Bundle manifest may not be a symlink.")
        if not manifest_path.is_file():
            return cls(root, None)
        manifest = _read_json(manifest_path, label="bundle manifest")
        if manifest.get("schema_version") != BUNDLE_SCHEMA:
            raise VerificationError(f"Unexpected bundle schema: {manifest.get('schema_version')!r}.")
        return cls(root, manifest)

    def resolve_record(self, value: Any, *, label: str) -> Path:
        sha256, byte_count = _identity(value, label=label)
        if self.mode == "bundle":
            path = self.root / "objects" / "sha256" / sha256[:2] / sha256
        else:
            raw = value.get("path")
            if not isinstance(raw, str) or not raw:
                raise VerificationError(f"{label} has no staging path.")
            path = Path(raw)
            if not path.is_absolute():
                path = self.root / path
        resolved = _regular_within(path, boundary=self.root, label=label)
        if resolved.stat().st_size != byte_count or file_sha256(resolved) != sha256:
            raise VerificationError(f"{label} differs from its bytes/SHA-256 record.")
        return resolved

    def logical_path(self, logical_path: str) -> Path:
        logical = _safe_relative(logical_path, label="verification logical path")
        if self.mode == "staging":
            return _regular_within(
                self.root.joinpath(*PurePosixPath(logical).parts),
                boundary=self.root,
                label=f"Logical artifact {logical}",
            )
        item = self.logical.get(logical)
        if item is None:
            raise VerificationError(f"Bundle has no publication artifact {logical!r}.")
        return self.resolve_record(item, label=logical)


def _artifact_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        for child in value.values():
            yield from _artifact_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_records(child)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _contains(value: Any, needle: Any) -> bool:
    if value == needle:
        return True
    if isinstance(value, dict):
        return any(_contains(child, needle) for child in value.values())
    if isinstance(value, list):
        return any(_contains(child, needle) for child in value)
    return False


def _load_evidence_json(
    evidence_root: EvidenceRoot,
    evidence: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    record = evidence.get(name)
    path = evidence_root.resolve_record(record, label=f"gate evidence {name}")
    return _read_json(path, label=f"gate evidence {name}")


def _finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{label} is not numeric.")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0.0):
        raise VerificationError(f"{label} is not finite{' and positive' if positive else ''}.")
    return converted


def _close(actual: Any, expected: float, *, label: str) -> None:
    converted = _finite_number(actual, label=label)
    _require(
        math.isclose(converted, expected, rel_tol=1.0e-12, abs_tol=1.0e-12),
        f"{label} differs from raw samples: {converted!r} != {expected!r}.",
    )


def _timing_distribution(run: dict[str, Any], key: str, *, label: str) -> tuple[list[float], float]:
    timing = run.get("timing")
    _require(isinstance(timing, dict), f"{label} has no timing object.")
    distribution = timing.get(key)
    _require(isinstance(distribution, dict), f"{label} has no {key} distribution.")
    samples = distribution.get("samples")
    _require(isinstance(samples, list) and len(samples) == 100, f"{label} must contain 100 {key} samples.")
    converted = [
        _finite_number(value, label=f"{label} {key} sample", positive=True)
        for value in samples
    ]
    mean = statistics.fmean(converted)
    _require(distribution.get("count") == 100, f"{label} {key} count differs.")
    _close(distribution.get("mean"), mean, label=f"{label} {key} mean")
    return converted, mean


def _validate_output_contract(run: dict[str, Any], *, contract: str, label: str) -> None:
    output = run.get("output_validation")
    _require(isinstance(output, dict) and output.get("pass") is True, f"{label} output validation failed.")
    required_true = ("rgb_cuda", "rgb_contiguous", "rgb_finite")
    for field in required_true:
        _require(output.get(field) is True, f"{label} output predicate {field} failed.")
    _require(output.get("rgb_dtype") == "torch.float32", f"{label} RGB dtype differs.")
    if contract == "full":
        for field in (
            "alpha_finite",
            "foreground_depth_finite",
            "background_depth_inf",
            "foreground_semantic_nonnegative",
            "background_semantic_minus_one",
        ):
            _require(output.get(field) is True, f"{label} output predicate {field} failed.")
        _require(output.get("alpha_dtype") == "torch.float32", f"{label} alpha dtype differs.")
        _require(output.get("depth_dtype") == "torch.float32", f"{label} depth dtype differs.")
        _require(output.get("semantic_dtype") == "torch.int64", f"{label} semantic dtype differs.")
        _require(
            _finite_number(output.get("foreground_fraction"), label=f"{label} foreground fraction") > 0.0,
            f"{label} has no foreground coverage.",
        )


def _validate_matrix_run(
    evidence_root: EvidenceRoot,
    *,
    run_record: Any,
    fidelity_record: Any,
    record: dict[str, Any],
    renderer: str,
    contract: str,
    batch: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"{renderer} {contract} B{batch}"
    run = _read_json(
        evidence_root.resolve_record(run_record, label=f"{label} raw run"),
        label=f"{label} raw run",
    )
    _require(run.get("schema_version") == RENDERER_RUN_SCHEMA, f"{label} raw-run schema differs.")
    _require(run.get("pass") is True and run.get("headline_eligible") is True, f"{label} raw run did not pass.")
    _require(run.get("primary_workload_eligible") is True, f"{label} is not the primary workload.")
    _require(run.get("profile_control") is False, f"{label} is a profiler control, not a timed row.")
    _require(run.get("renderer") == renderer, f"{label} renderer identity differs.")
    _require(run.get("output_contract") == contract, f"{label} output contract differs.")
    camera = run.get("camera_contract")
    _require(isinstance(camera, dict), f"{label} camera contract is absent.")
    _require(camera.get("batch") == batch and not isinstance(camera.get("batch"), bool), f"{label} batch differs.")
    _require(camera.get("trajectory_id") == PRIMARY_TRAJECTORY_IDS[batch], f"{label} trajectory differs.")
    _require(camera.get("width") == 128 and camera.get("height") == 128, f"{label} resolution differs.")
    _require(camera.get("warmup_frames") == 8 and camera.get("measured_frames") == 100, f"{label} frame schedule differs.")
    _require(camera.get("motion_classification") == "independently-changing-camera-batch", f"{label} motion contract differs.")
    _require(camera.get("unchanged_measured_camera_frame_pairs") == 0, f"{label} contains unchanged cameras.")
    _require(run.get("scene", {}).get("sha256") == record["matrix"]["scene_sha256"], f"{label} scene differs.")
    equation = run.get("equation_contract")
    _require(isinstance(equation, dict), f"{label} equation contract is absent.")
    _require(equation.get("gaussian_support_sigma") == 3.33, f"{label} Gaussian support differs.")
    _require(equation.get("semantic_topology") == "spatial-octants-8", f"{label} semantic topology differs.")
    for field, expected in MATCHED_PROJECTION_RULES.items():
        _require(equation.get(field) == expected, f"{label} equation field {field} differs.")
    environment = run.get("environment")
    _require(isinstance(environment, dict), f"{label} environment is absent.")
    provenance = environment.get("source_provenance")
    _require(isinstance(provenance, dict), f"{label} source provenance is absent.")
    _require(environment.get("gpu_uuid") == record["matrix"]["gpu_uuid"], f"{label} GPU differs.")
    _require(provenance.get("head") == record["benchmark"]["commit"] and provenance.get("dirty") is False, f"{label} source differs.")
    _require(_same_artifact(provenance.get("manifest"), record["matrix"]["source_manifest"]), f"{label} source manifest differs.")
    _require(
        _same_artifact(environment.get("native_extension"), record["matrix"][f"{renderer}_native_extension"]),
        f"{label} native extension differs.",
    )
    execution = run.get("backend_execution")
    _require(isinstance(execution, dict), f"{label} execution schedule is absent.")
    expected_physical = (128 if renderer == "custom" and batch in (512, 1024) else batch) if renderer == "custom" else 1
    expected_submissions = (batch + expected_physical - 1) // expected_physical
    _require(execution.get("logical_batch") == batch, f"{label} logical batch differs.")
    _require(execution.get("physical_batch") == expected_physical, f"{label} physical batch differs.")
    _require(execution.get("native_submissions_per_logical_batch") == expected_submissions, f"{label} submission count differs.")
    capacity = run.get("capacity")
    _require(isinstance(capacity, dict), f"{label} capacity record is absent.")
    observed = (capacity.get("timed_verification") or {}).get("observed") or {}
    _require(observed.get("measured_max_intersection_overflow") == 0, f"{label} intersection capacity overflowed.")
    if renderer == "custom":
        _require(observed.get("measured_max_visible_overflow") == 0, f"{label} visible capacity overflowed.")
    _validate_output_contract(run, contract=contract, label=label)

    fidelity = _read_json(
        evidence_root.resolve_record(fidelity_record, label=f"{label} fidelity"),
        label=f"{label} fidelity",
    )
    _require(fidelity.get("schema_version") == FIDELITY_SCHEMA and fidelity.get("pass") is True, f"{label} fidelity failed.")
    for field, expected in (
        ("renderer", renderer),
        ("output_contract", contract),
        ("batch", batch),
        ("trajectory_id", PRIMARY_TRAJECTORY_IDS[batch]),
    ):
        _require(fidelity.get(field) == expected, f"{label} fidelity {field} differs.")
    return run, fidelity


def _validate_matrix(
    evidence_root: EvidenceRoot,
    record: dict[str, Any],
    source_manifest: dict[str, Any],
) -> None:
    matrix = record["matrix"]
    summary = _read_json(
        evidence_root.resolve_record(matrix["summary"], label="matrix summary"),
        label="matrix summary",
    )
    invocation = _read_json(
        evidence_root.resolve_record(matrix["invocation"], label="matrix invocation"),
        label="matrix invocation",
    )
    _require(summary.get("schema_version") == "flashgs-matched-summary-v4", "Matrix summary schema differs.")
    _require(summary.get("pass") is True, "Matrix summary pass is not true.")
    _require(summary.get("scientific_pass") is True, "Matrix scientific gate did not pass.")
    _require(summary.get("headline_eligible") is True, "Matrix is not headline eligible.")
    _require(summary.get("primary_contract_eligible") is True, "Matrix is not the complete primary contract.")
    _require(summary.get("matrix_fairness_failures") == [], "Matrix has fairness failures.")
    run_identities: set[tuple[str, int]] = set()
    fidelity_identities: set[tuple[str, int]] = set()
    for table_name, contract in (
        ("primary_full_sensor_dynamic_table", "full"),
        ("rgb_only_dynamic_table", "rgb"),
    ):
        table = summary.get(table_name)
        _require(isinstance(table, list) and len(table) == len(PRIMARY_BATCHES), f"{table_name} is incomplete.")
        _require(all(isinstance(row, dict) for row in table), f"{table_name} contains a malformed row.")
        batches = tuple(row.get("batch") for row in table)
        _require(batches == PRIMARY_BATCHES, f"{table_name} batches differ from the frozen matrix.")
        for row in table:
            batch = row["batch"]
            _require(isinstance(batch, int) and not isinstance(batch, bool), f"{table_name} batch is not an integer.")
            _require(row.get("pass") is True, f"{table_name} contains a failed row.")
            _require(row.get("fidelity_pass") is True, f"{table_name} contains a failed fidelity row.")
            _require(row.get("fairness_pass") is True, f"{table_name} contains a failed fairness row.")
            _require(row.get("fairness_failures") == [], f"{table_name} row has fairness failures.")
            artifacts = row.get("artifacts")
            _require(isinstance(artifacts, dict), f"{table_name} row has no raw-run bindings.")
            _require(
                set(artifacts) == {"custom_run", "flashgs_run", "custom_fidelity", "flashgs_fidelity"},
                f"{table_name} row artifact set differs.",
            )
            runs: dict[str, dict[str, Any]] = {}
            for renderer in ("custom", "flashgs"):
                run_record = artifacts.get(f"{renderer}_run")
                fidelity_record = artifacts.get(f"{renderer}_fidelity")
                run_identity = _identity(run_record, label=f"{table_name} B{batch} {renderer} run")
                fidelity_identity = _identity(fidelity_record, label=f"{table_name} B{batch} {renderer} fidelity")
                _require(run_identity not in run_identities, "A matrix raw run was reused for more than one row.")
                _require(fidelity_identity not in fidelity_identities, "A matrix fidelity artifact was reused for more than one row.")
                run_identities.add(run_identity)
                fidelity_identities.add(fidelity_identity)
                run, _fidelity = _validate_matrix_run(
                    evidence_root,
                    run_record=run_record,
                    fidelity_record=fidelity_record,
                    record=record,
                    renderer=renderer,
                    contract=contract,
                    batch=batch,
                )
                runs[renderer] = run
            custom_gpu, custom_mean = _timing_distribution(runs["custom"], "gpu_batch_ms", label=f"Custom {contract} B{batch}")
            flash_gpu, flash_mean = _timing_distribution(runs["flashgs"], "gpu_batch_ms", label=f"FlashGS {contract} B{batch}")
            custom_wall, custom_wall_mean = _timing_distribution(runs["custom"], "synchronized_wall_batch_ms", label=f"Custom {contract} B{batch}")
            flash_wall, flash_wall_mean = _timing_distribution(runs["flashgs"], "synchronized_wall_batch_ms", label=f"FlashGS {contract} B{batch}")
            custom_host, custom_host_mean = _timing_distribution(runs["custom"], "host_submission_ms", label=f"Custom {contract} B{batch}")
            flash_host, flash_host_mean = _timing_distribution(runs["flashgs"], "host_submission_ms", label=f"FlashGS {contract} B{batch}")
            for field, expected in (
                ("custom_ms", custom_mean),
                ("flashgs_ms", flash_mean),
                ("custom_synchronized_wall_ms", custom_wall_mean),
                ("flashgs_synchronized_wall_ms", flash_wall_mean),
                ("custom_host_submission_ms", custom_host_mean),
                ("flashgs_host_submission_ms", flash_host_mean),
                ("speedup_custom_over_flashgs", flash_mean / custom_mean),
                ("gpu_speedup_ratio_of_mean_latency", flash_mean / custom_mean),
                ("wall_speedup_ratio_of_mean_latency", flash_wall_mean / custom_wall_mean),
            ):
                _close(row.get(field), expected, label=f"{table_name} B{batch} {field}")
            expected_samples = {
                "custom": {
                    "gpu_batch_ms": custom_gpu,
                    "synchronized_wall_batch_ms": custom_wall,
                    "host_submission_ms": custom_host,
                },
                "flashgs": {
                    "gpu_batch_ms": flash_gpu,
                    "synchronized_wall_batch_ms": flash_wall,
                    "host_submission_ms": flash_host,
                },
            }
            _require(row.get("timing_samples") == expected_samples, f"{table_name} B{batch} timing samples differ from raw runs.")
    _require(invocation.get("schema_version") == "flashgs-matched-matrix-invocation-v1", "Matrix invocation schema differs.")
    argv = invocation.get("argv")
    _require(isinstance(argv, list) and any(Path(item).name == "run_flashgs_matched_matrix.py" for item in argv if isinstance(item, str)), "Matrix invocation does not name the frozen runner.")
    parsed_arguments = invocation.get("parsed_arguments")
    _require(isinstance(parsed_arguments, dict) and parsed_arguments, "Matrix invocation has no parsed arguments.")
    _require(parsed_arguments.get("expected_gpu_uuid") == matrix["gpu_uuid"], "Matrix invocation GPU UUID differs.")
    _require(parsed_arguments.get("batches") == ",".join(str(batch) for batch in PRIMARY_BATCHES), "Matrix invocation batches differ.")
    _require(parsed_arguments.get("custom_chunked_physical_views") == 128, "Matrix invocation chunk schedule differs.")
    _require(parsed_arguments.get("allow_nonheadline_gpu") is False, "Matrix invocation used a non-headline-GPU override.")
    _require(_contains(summary, matrix["gpu_uuid"]), "Matrix summary does not bind the declared GPU UUID.")
    _require(_contains(summary, record["benchmark"]["commit"]), "Matrix summary does not bind the benchmark commit.")
    _require(_contains(summary, matrix["source_manifest"]["sha256"]), "Matrix summary does not bind the source manifest.")
    _require(source_manifest.get("schema_version") == SOURCE_MANIFEST_SCHEMA, "Source manifest schema differs.")
    _require(source_manifest.get("head") == record["benchmark"]["commit"], "Source manifest head differs from benchmark commit.")
    _require(source_manifest.get("dirty") is False, "Source manifest is dirty.")


def _record_path(record: Any, *, label: str) -> str:
    _identity(record, label=label)
    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise VerificationError(f"{label} has no recorded path.")
    return path


def _exact_flag(argv: list[str], name: str, *, label: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == name]
    _require(
        len(positions) == 1 and positions[0] + 1 < len(argv),
        f"{label} must contain exactly one {name} value.",
    )
    return argv[positions[0] + 1]


def _validate_node_occupancy(
    evidence_root: EvidenceRoot,
    record: Any,
    *,
    expected_gpu_uuid: str,
    label: str,
) -> dict[str, Any]:
    payload = _read_json(
        evidence_root.resolve_record(record, label=label),
        label=label,
    )
    executor = payload.get("executor_control") or {}
    lock = executor.get("cooperative_node_wide_lock") or {}
    sampling = payload.get("sampled_compute_process_telemetry") or {}
    _require(payload.get("schema_version") == NODE_OCCUPANCY_SCHEMA, f"{label} schema differs.")
    _require(payload.get("pass") is True, f"{label} failed.")
    _require(payload.get("expected_gpu_uuid") == expected_gpu_uuid, f"{label} GPU UUID differs.")
    _require(executor.get("scope") == "all-visible-gpus", f"{label} scope differs.")
    _require(
        lock.get("pass") is True and lock.get("lock_observed_held") is True,
        f"{label} cooperative GPU lock failed.",
    )
    _require(
        sampling.get("pass") is True
        and isinstance(sampling.get("sample_count"), int)
        and not isinstance(sampling.get("sample_count"), bool)
        and sampling["sample_count"] >= 2,
        f"{label} sampling is insufficient.",
    )
    return payload


def _validate_repair_raw_path(
    *,
    report_record: dict[str, Any],
    embedded_record: Any,
    evidence_record: Any,
    label: str,
) -> None:
    _require(_same_artifact(embedded_record, evidence_record), f"{label} identity differs.")
    relative = _safe_relative(
        _record_path(embedded_record, label=f"{label} embedded artifact"),
        label=f"{label} report-relative path",
    )
    parsed = PurePosixPath(relative)
    report_path = Path(_record_path(report_record, label=f"{label} report"))
    expected_path = report_path.parent.joinpath(*parsed.parts)
    recorded_path = Path(_record_path(evidence_record, label=f"{label} normalized artifact"))
    _require(expected_path == recorded_path, f"{label} does not resolve beside its repair report.")


def _validate_b64_support_producer(
    evidence_root: EvidenceRoot,
    evidence: dict[str, Any],
    *,
    prefix: str,
    gate_id: str,
    script_name: str,
    output_record: dict[str, Any],
    result: dict[str, Any],
    aggregate_record: dict[str, Any],
) -> dict[str, Any]:
    records = {
        suffix: evidence[f"{prefix}_support_{suffix}"]
        for suffix in ("wrapper", "command", "log", "exit_status", "occupancy")
    }
    paths = {
        suffix: evidence_root.resolve_record(record, label=f"B64 {prefix} support {suffix}")
        for suffix, record in records.items()
    }
    wrapper = _read_json(paths["wrapper"], label=f"B64 {prefix} support wrapper")
    expected_wrapper_fields = {
        "schema_version",
        "gate_id",
        "started_at",
        "completed_at",
        "timeout_seconds",
        "timed_out",
        "returncode",
        "command_record",
        "log",
        "exit_status",
        "occupancy",
        "result",
        "pass",
    }
    _require(set(wrapper) == expected_wrapper_fields, f"B64 {prefix} support wrapper fields differ.")
    _require(wrapper.get("schema_version") == BOUNDED_WRAPPER_SCHEMA, f"B64 {prefix} support wrapper schema differs.")
    _require(wrapper.get("gate_id") == gate_id, f"B64 {prefix} support gate ID differs.")
    _require(wrapper.get("pass") is True, f"B64 {prefix} support wrapper failed.")
    _require(wrapper.get("returncode") == 0 and wrapper.get("timed_out") is False, f"B64 {prefix} support process failed.")
    _require(wrapper.get("timeout_seconds") == 7_200, f"B64 {prefix} support timeout differs.")
    _require(wrapper.get("result") is None, f"B64 {prefix} support unexpectedly has a normalized result.")
    for wrapper_field, suffix in (
        ("command_record", "command"),
        ("log", "log"),
        ("exit_status", "exit_status"),
        ("occupancy", "occupancy"),
    ):
        _require(
            wrapper.get(wrapper_field) == _record_path(records[suffix], label=f"B64 {prefix} support {suffix}"),
            f"B64 {prefix} support wrapper {wrapper_field} binding differs.",
        )

    command = _read_json(paths["command"], label=f"B64 {prefix} support command")
    _require(
        set(command) == {"command", "gate_id", "schema_version", "started_at", "subcommands", "timeout_seconds"},
        f"B64 {prefix} support command fields differ.",
    )
    _require(command.get("schema_version") == COMMAND_SCHEMA, f"B64 {prefix} support command schema differs.")
    _require(command.get("gate_id") == gate_id, f"B64 {prefix} support command gate ID differs.")
    _require(command.get("started_at") == wrapper.get("started_at"), f"B64 {prefix} support start timestamp differs.")
    _require(command.get("timeout_seconds") == 7_200, f"B64 {prefix} support command timeout differs.")
    _require(command.get("subcommands") == [], f"B64 {prefix} support command unexpectedly has subcommands.")
    argv = command.get("command")
    _require(
        isinstance(argv, list) and argv and all(isinstance(token, str) and token for token in argv),
        f"B64 {prefix} support argv is malformed.",
    )
    _require(
        sum(Path(token).name == script_name for token in argv) == 1,
        f"B64 {prefix} support uses the wrong producer.",
    )
    _require(
        _exact_flag(argv, "--output", label=f"B64 {prefix} support")
        == _record_path(output_record, label=f"B64 {prefix} producer output"),
        f"B64 {prefix} support output differs.",
    )
    _require(
        _exact_flag(argv, "--expected-gpu-uuid", label=f"B64 {prefix} support")
        == result["identity"]["gpu_uuid"],
        f"B64 {prefix} support GPU UUID differs.",
    )
    _require(
        _exact_flag(argv, "--source-manifest", label=f"B64 {prefix} support")
        == _record_path(aggregate_record["benchmark"]["source_manifest"], label="benchmark source manifest"),
        f"B64 {prefix} support source manifest differs.",
    )
    _require(
        Path(_exact_flag(argv, "--trajectory", label=f"B64 {prefix} support")).name == "b64.json",
        f"B64 {prefix} support trajectory differs.",
    )
    _exact_flag(argv, "--scene-path", label=f"B64 {prefix} support")
    if prefix in {"primary", "repeat"}:
        _exact_flag(argv, "--flashgs-adapter-attestation", label=f"B64 {prefix} support")
        _require(
            _exact_flag(argv, "--scene-sha256", label=f"B64 {prefix} support")
            == result["identity"]["scene_sha256"],
            f"B64 {prefix} support scene differs.",
        )
        _require(
            _exact_flag(argv, "--semantic-topology", label=f"B64 {prefix} support")
            == "spatial-octants-8",
            f"B64 {prefix} support semantic topology differs.",
        )
        capacity = _exact_flag(argv, "--intersection-capacity", label=f"B64 {prefix} support")
        _require(capacity.isdigit() and int(capacity) > 0, f"B64 {prefix} support capacity is invalid.")
        for flag, key in (
            ("--known-failure-manifest", "known_failure_manifest"),
            ("--diagnosis-index", "diagnosis_index"),
            ("--diagnosis-lock", "diagnosis_lock"),
        ):
            _require(
                _exact_flag(argv, flag, label=f"B64 {prefix} support")
                == _record_path(evidence[key], label=f"B64 {key}"),
                f"B64 {prefix} support {flag} differs.",
            )
        artifact_root = Path(
            _exact_flag(argv, "--diagnosis-artifact-root", label=f"B64 {prefix} support")
        )
        diagnosis_paths = [
            Path(_record_path(evidence[key], label=f"B64 {key}"))
            for key in ("known_failure_manifest", "diagnosis_index", "diagnosis_lock")
        ]
        _require(
            artifact_root == Path(os.path.commonpath([str(path) for path in diagnosis_paths])),
            f"B64 {prefix} support diagnosis artifact root differs.",
        )
    else:
        _require(_exact_flag(argv, "--step", label="B64 oracle support") == "107", "B64 oracle support step differs.")
        _require(_exact_flag(argv, "--max-views", label="B64 oracle support") == "64", "B64 oracle support view count differs.")
        _require(
            _exact_flag(argv, "--semantic-topology", label="B64 oracle support") == "spatial-octants-8",
            "B64 oracle support semantic topology differs.",
        )
        _exact_flag(argv, "--gsplat-source", label="B64 oracle support")
        _exact_flag(argv, "--gsplat-build-attestation", label="B64 oracle support")

    exit_text = paths["exit_status"].read_text(encoding="utf-8").strip()
    _require(exit_text == "0", f"B64 {prefix} support exit status is nonzero.")
    log_text = paths["log"].read_text(encoding="utf-8", errors="strict")
    sentinel = (
        "FLASHGS_GSPLAT_ORACLE_OK "
        if prefix == "oracle"
        else "FLASHGS_B64_REPAIR_VERIFICATION "
    )
    _require(
        sum(line.startswith(sentinel) for line in log_text.splitlines()) == 1,
        f"B64 {prefix} support log lacks exactly one producer success record.",
    )
    _validate_node_occupancy(
        evidence_root,
        records["occupancy"],
        expected_gpu_uuid=result["identity"]["gpu_uuid"],
        label=f"B64 {prefix} support occupancy",
    )
    start = _parse_time(wrapper.get("started_at"), label=f"B64 {prefix} support started_at")
    completed = _parse_time(wrapper.get("completed_at"), label=f"B64 {prefix} support completed_at")
    result_time = _parse_time(result.get("created_at"), label="B64 repair result created_at")
    _require(start <= completed <= result_time, f"B64 {prefix} support timestamps are out of order.")
    return wrapper


def _validate_b64_repair_report(
    report: dict[str, Any],
    *,
    label: str,
    report_record: dict[str, Any],
    full_record: dict[str, Any],
    rgb_record: dict[str, Any],
    evidence: dict[str, Any],
    aggregate_record: dict[str, Any],
) -> None:
    _require(report.get("schema_version") == B64_REPAIR_SCHEMA, f"{label} schema differs.")
    _require(report.get("pass") is True, f"{label} failed.")
    tool = report.get("tool_integrity") or {}
    tool_checks = tool.get("checks") or {}
    _require(
        tool.get("pass") is True
        and isinstance(tool_checks, dict)
        and bool(tool_checks)
        and all(value is True for value in tool_checks.values()),
        f"{label} tool-integrity checks failed.",
    )
    _require(report.get("aggregate", {}).get("case_count") == 5, f"{label} case count differs.")
    _require(report.get("aggregate", {}).get("passed_case_count") == 5, f"{label} did not pass five cases.")
    _require(
        report.get("historical_mismatch_repair", {}).get("pixel_count") == 92
        and report.get("historical_mismatch_repair", {}).get("passed_pixel_count") == 92,
        f"{label} did not pass 92 historical pixels.",
    )
    rgb = report.get("rgb_only_specialization", {}).get("result", {})
    _require(
        rgb.get("pixel_count") == 92 and rgb.get("passed_pixel_count") == 92,
        f"{label} RGB-only repair did not pass 92 pixels.",
    )
    raw_outputs = report.get("post_fix_raw_outputs")
    _require(
        isinstance(raw_outputs, dict) and set(raw_outputs) == {"full", "rgb_only"},
        f"{label} raw-output bindings differ.",
    )
    _validate_repair_raw_path(
        report_record=report_record,
        embedded_record=raw_outputs["full"],
        evidence_record=full_record,
        label=f"{label} full raw output",
    )
    _validate_repair_raw_path(
        report_record=report_record,
        embedded_record=raw_outputs["rgb_only"],
        evidence_record=rgb_record,
        label=f"{label} RGB-only raw output",
    )
    for field, key in (
        ("known_failure_manifest", "known_failure_manifest"),
        ("diagnosis_index", "diagnosis_index"),
        ("diagnosis_lock", "diagnosis_lock"),
    ):
        _require(
            _same_artifact(report.get(field), evidence[key]),
            f"{label} {field} binding differs.",
        )
    _require(
        report.get("scene", {}).get("sha256") == aggregate_record["matrix"]["scene_sha256"],
        f"{label} scene differs.",
    )
    _require(
        report.get("environment", {}).get("gpu_uuid") == aggregate_record["matrix"]["gpu_uuid"],
        f"{label} GPU differs.",
    )
    _require(report.get("camera", {}).get("step") == 107, f"{label} diagnostic step differs.")
    _require(
        report.get("equation", {}).get("semantic_topology") == "spatial-octants-8",
        f"{label} semantic topology differs.",
    )
    _require(
        _same_artifact(
            (report.get("source_provenance") or {}).get("manifest"),
            aggregate_record["benchmark"]["source_manifest"],
        ),
        f"{label} source manifest differs.",
    )
    _require(
        _same_artifact(
            (report.get("production_adapter") or {}).get("native_extension"),
            aggregate_record["matrix"]["flashgs_native_extension"],
        ),
        f"{label} native extension differs.",
    )
    _parse_time(report.get("created_at"), label=f"{label} created_at")


def _validate_b64_privacy_redaction(
    evidence_root: EvidenceRoot,
    evidence: dict[str, Any],
) -> None:
    """Bind every public B64 derivative byte to its disclosed transformation."""

    redaction = _load_evidence_json(
        evidence_root,
        evidence,
        "diagnosis_privacy_redaction_manifest",
    )
    inventory = _load_evidence_json(
        evidence_root,
        evidence,
        "diagnosis_privacy_redaction_inventory",
    )
    tool_path = evidence_root.resolve_record(
        evidence["diagnosis_privacy_redaction_tool"],
        label="B64 diagnosis privacy-redaction tool",
    )
    tool_record = artifact_record(tool_path)
    expected_contract = {
        "artifact_hash_rebinding": "lowercase-sha256-same-length-substitution",
        "path_redaction": "deterministic-same-length-private-root-substitution",
        "payload_sizes_unchanged": True,
        "reversibility": "every-public-file-roundtrips-byte-for-byte",
    }
    _require(
        set(redaction)
        == {
            "artifact_hash_rebindings",
            "file_count",
            "files",
            "generator",
            "known_failure_manifest",
            "pass",
            "path_prefix_replacements",
            "private_roots",
            "schema_version",
            "transformation_contract",
        }
        and redaction.get("schema_version") == B64_PRIVACY_REDACTION_SCHEMA
        and redaction.get("pass") is True
        and redaction.get("transformation_contract") == expected_contract,
        "B64 diagnosis privacy-redaction contract differs.",
    )
    generator = redaction.get("generator")
    _require(
        isinstance(generator, dict)
        and set(generator) == {"bytes", "logical_path", "sha256"}
        and generator.get("logical_path")
        == "publication/redact_b64_diagnosis.py"
        and _same_artifact(generator, tool_record),
        "B64 diagnosis privacy-redaction generator differs.",
    )
    known_binding = redaction.get("known_failure_manifest")
    _require(
        isinstance(known_binding, dict)
        and set(known_binding) == {"bytes", "logical_path", "sha256"}
        and known_binding.get("logical_path")
        == "experiments/flashgs_matched/B64_KNOWN_FAILURE_CASES.json"
        and _same_artifact(
            known_binding,
            evidence["known_failure_manifest"],
        ),
        "B64 privacy-redaction known-failure binding differs.",
    )

    raw_files = redaction.get("files")
    _require(
        isinstance(raw_files, list)
        and len(raw_files) == 30
        and redaction.get("file_count") == 30,
        "B64 diagnosis privacy-redaction file set is incomplete.",
    )
    files_by_label: dict[str, dict[str, Any]] = {}
    path_replacements = 0
    hash_rebindings = 0
    for raw_file in raw_files:
        _require(
            isinstance(raw_file, dict)
            and set(raw_file)
            == {
                "artifact_hash_rebindings",
                "bytes",
                "logical_path",
                "path_prefix_replacements",
                "public_sha256",
                "reversible",
                "source_sha256",
            },
            "B64 diagnosis privacy-redaction file record is malformed.",
        )
        label = _safe_relative(
            raw_file.get("logical_path"),
            label="B64 privacy-redaction logical path",
        )
        _require(
            label not in files_by_label,
            "B64 privacy-redaction file labels repeat.",
        )
        path_count = raw_file.get("path_prefix_replacements")
        hash_count = raw_file.get("artifact_hash_rebindings")
        source_sha256 = raw_file.get("source_sha256")
        public_sha256 = raw_file.get("public_sha256")
        _require(
            isinstance(raw_file.get("bytes"), int)
            and not isinstance(raw_file.get("bytes"), bool)
            and raw_file["bytes"] >= 0
            and isinstance(path_count, int)
            and not isinstance(path_count, bool)
            and path_count >= 0
            and isinstance(hash_count, int)
            and not isinstance(hash_count, bool)
            and hash_count >= 0
            and raw_file.get("reversible") is True
            and isinstance(source_sha256, str)
            and SHA256_RE.fullmatch(source_sha256) is not None
            and isinstance(public_sha256, str)
            and SHA256_RE.fullmatch(public_sha256) is not None
            and ((source_sha256 != public_sha256) == (path_count + hash_count > 0)),
            f"B64 privacy-redaction file record differs: {label}.",
        )
        files_by_label[label] = raw_file
        path_replacements += path_count
        hash_rebindings += hash_count
    required_labels = {
        "diagnosis-index.json",
        "diagnosis-lock.json",
        "known-failure-manifest.json",
    }
    historical_labels = {
        label for label in files_by_label if label.startswith("historical/")
    }
    _require(
        set(files_by_label) == required_labels | historical_labels
        and len(historical_labels) == 27,
        "B64 privacy-redaction logical file set differs.",
    )
    _require(
        redaction.get("path_prefix_replacements") == path_replacements
        and path_replacements > 0
        and redaction.get("artifact_hash_rebindings") == hash_rebindings
        and hash_rebindings > 0,
        "B64 privacy-redaction aggregate counts differ.",
    )

    roots = redaction.get("private_roots")
    _require(
        isinstance(roots, list)
        and bool(roots)
        and sum(
            item.get("occurrences", 0)
            for item in roots
            if isinstance(item, dict)
        )
        == path_replacements,
        "B64 privacy-redaction root commitments differ.",
    )
    root_hashes: set[str] = set()
    replacements: set[str] = set()
    for root in roots:
        _require(
            isinstance(root, dict)
            and set(root)
            == {"bytes", "occurrences", "replacement", "source_sha256"}
            and isinstance(root.get("bytes"), int)
            and not isinstance(root.get("bytes"), bool)
            and root["bytes"] > 0
            and isinstance(root.get("occurrences"), int)
            and not isinstance(root.get("occurrences"), bool)
            and root["occurrences"] > 0
            and isinstance(root.get("replacement"), str)
            and root["replacement"].isascii()
            and re.fullmatch(r"/work/[0-9A-Za-z_]+", root["replacement"])
            is not None
            and len(root["replacement"].encode("ascii")) == root["bytes"]
            and isinstance(root.get("source_sha256"), str)
            and SHA256_RE.fullmatch(root["source_sha256"]) is not None,
            "B64 privacy-redaction root commitment is malformed.",
        )
        root_hashes.add(root["source_sha256"])
        replacements.add(root["replacement"])
    _require(
        len(root_hashes) == len(roots) and len(replacements) == len(roots),
        "B64 privacy-redaction root commitments repeat.",
    )

    _require(
        set(inventory)
        == {"file_count", "files", "manifest", "pass", "schema_version"}
        and inventory.get("schema_version")
        == B64_PRIVACY_REDACTION_INVENTORY_SCHEMA
        and inventory.get("pass") is True
        and inventory.get("file_count") == 30
        and _same_artifact(
            inventory.get("manifest"),
            evidence["diagnosis_privacy_redaction_manifest"],
        ),
        "B64 privacy-redaction inventory contract differs.",
    )
    inventory_files = inventory.get("files")
    _require(
        isinstance(inventory_files, list) and len(inventory_files) == 30,
        "B64 privacy-redaction inventory file set is incomplete.",
    )
    inventory_by_label: dict[str, dict[str, Any]] = {}
    for item in inventory_files:
        _require(
            isinstance(item, dict)
            and set(item) == {"artifact", "logical_path"}
            and isinstance(item.get("artifact"), dict)
            and set(item["artifact"]) == {"bytes", "path", "sha256"},
            "B64 privacy-redaction inventory file record is malformed.",
        )
        label = _safe_relative(
            item.get("logical_path"),
            label="B64 privacy-redaction inventory logical path",
        )
        _require(
            label not in inventory_by_label,
            "B64 privacy-redaction inventory labels repeat.",
        )
        resolved = evidence_root.resolve_record(
            item["artifact"],
            label=f"B64 privacy-redacted artifact {label}",
        )
        _require(
            _identity(
                artifact_record(resolved),
                label=f"B64 privacy-redacted artifact {label}",
            )
            == (
                files_by_label.get(label, {}).get("public_sha256"),
                files_by_label.get(label, {}).get("bytes"),
            ),
            f"B64 privacy-redacted artifact differs: {label}.",
        )
        inventory_by_label[label] = item["artifact"]
    _require(
        set(inventory_by_label) == set(files_by_label),
        "B64 privacy-redaction manifest and inventory file sets differ.",
    )
    for label, evidence_name in (
        ("diagnosis-index.json", "diagnosis_index"),
        ("diagnosis-lock.json", "diagnosis_lock"),
        ("known-failure-manifest.json", "known_failure_manifest"),
    ):
        _require(
            _same_artifact(
                inventory_by_label[label],
                evidence[evidence_name],
            ),
            f"B64 privacy-redaction inventory does not bind {label}.",
        )


def _validate_repair(
    evidence_root: EvidenceRoot,
    result: dict[str, Any],
    aggregate_record: dict[str, Any],
) -> None:
    checks = result["checks"]
    expected = {
        "cases_expected": 5,
        "cases_passed": 5,
        "historical_pixels_expected": 92,
        "historical_pixels_passed": 92,
        "rgb_only_pixels_passed": 92,
        "output_contracts": ["full", "rgb"],
        "primary_full_views_passed": 64,
        "primary_rgb_views_passed": 64,
        "repeat_full_views_passed": 64,
        "repeat_rgb_views_passed": 64,
        "zero_overflow": True,
        "oracle_pinned": True,
    }
    for name, value in expected.items():
        _require(checks.get(name) == value, f"Repair predicate {name} differs.")
    max_ulp = checks.get("max_intermediate_ulp")
    _require(isinstance(max_ulp, int) and not isinstance(max_ulp, bool) and 0 <= max_ulp <= 128, "Repair ULP bound failed.")
    oracle_checks = checks.get("oracle_comparisons")
    required_oracle = {"primary_full", "primary_rgb", "repeat_full", "repeat_rgb"}
    _require(isinstance(oracle_checks, dict) and set(oracle_checks) == required_oracle, "Repair oracle comparison set differs.")
    _require(all(value is True for value in oracle_checks.values()), "A repair oracle comparison failed.")

    evidence = result["evidence"]
    _require(
        set(evidence) == B64_TRANSITIVE_EVIDENCE_KEYS,
        "B64 repair transitive evidence set is incomplete or contains unknown keys.",
    )
    repair_record = evidence["repair_report"]
    repeat_record = evidence["repeat_repair_report"]
    repair = _load_evidence_json(evidence_root, evidence, "repair_report")
    repeat = _load_evidence_json(evidence_root, evidence, "repeat_repair_report")
    oracle = _load_evidence_json(evidence_root, evidence, "oracle_report")
    _require(
        _identity(repair_record, label="primary B64 repair report")
        != _identity(repeat_record, label="repeat B64 repair report"),
        "Primary and repeat B64 repair reports are not distinct artifacts.",
    )
    _require(
        repair.get("created_at") != repeat.get("created_at"),
        "Primary and repeat B64 repair reports have the same creation timestamp.",
    )
    _validate_b64_repair_report(
        repair,
        label="Primary B64 repair report",
        report_record=repair_record,
        full_record=evidence["primary_repair_full_raw"],
        rgb_record=evidence["primary_repair_rgb_only_raw"],
        evidence=evidence,
        aggregate_record=aggregate_record,
    )
    _validate_b64_repair_report(
        repeat,
        label="Repeat B64 repair report",
        report_record=repeat_record,
        full_record=evidence["repeat_repair_full_raw"],
        rgb_record=evidence["repeat_repair_rgb_only_raw"],
        evidence=evidence,
        aggregate_record=aggregate_record,
    )
    primary_trajectory_id = repair.get("camera", {}).get("trajectory_id")
    _require(
        isinstance(primary_trajectory_id, str)
        and bool(primary_trajectory_id)
        and repeat.get("camera", {}).get("trajectory_id") == primary_trajectory_id,
        "Primary and repeat B64 repair trajectories differ.",
    )
    raw_paths = {
        _record_path(evidence[key], label=f"B64 {key}")
        for key in (
            "primary_repair_full_raw",
            "primary_repair_rgb_only_raw",
            "repeat_repair_full_raw",
            "repeat_repair_rgb_only_raw",
        )
    }
    _require(len(raw_paths) == 4, "Primary/repeat B64 repair raw outputs reuse a path.")

    observed_ulps: list[int] = []
    for report in (repair, repeat):
        for case in report.get("cases", []):
            for culprit in case.get("culprits", []):
                trace = culprit.get("trace_float32_contract", {})
                observed_ulps.extend(
                    value
                    for value in (
                        trace.get("target_alpha_ulp_distance"),
                        trace.get("target_power_ulp_distance"),
                    )
                    if isinstance(value, int) and not isinstance(value, bool)
                )
    _require(
        observed_ulps and max(observed_ulps) <= max_ulp,
        "Primary/repeat repair traces do not prove the declared ULP bound.",
    )

    known = _load_evidence_json(evidence_root, evidence, "known_failure_manifest")
    diagnosis_lock = _load_evidence_json(evidence_root, evidence, "diagnosis_lock")
    diagnosis = _load_evidence_json(evidence_root, evidence, "diagnosis_index")
    _validate_b64_privacy_redaction(evidence_root, evidence)
    _require(
        known.get("schema_version") == B64_KNOWN_FAILURE_SCHEMA
        and known.get("diagnostic_only") is True
        and known.get("batch") == 64
        and known.get("step") == 107
        and known.get("trajectory_id") == primary_trajectory_id
        and known.get("scene_sha256") == aggregate_record["matrix"]["scene_sha256"]
        and known.get("total_mismatched_pixels_across_regions") == 92
        and isinstance(known.get("cases"), list)
        and len(known["cases"]) == 5,
        "B64 known-failure manifest contract differs.",
    )
    _require(
        diagnosis_lock.get("schema_version") == B64_DIAGNOSIS_LOCK_SCHEMA
        and _same_artifact(
            diagnosis_lock.get("known_failure_manifest"),
            evidence["known_failure_manifest"],
        ),
        "B64 diagnosis lock is not bound to the known-failure manifest.",
    )
    corpus = diagnosis.get("historical_mismatch_corpus") or {}
    _require(
        diagnosis.get("schema_version") == B64_DIAGNOSIS_INDEX_SCHEMA
        and diagnosis.get("diagnostic_only") is True
        and diagnosis.get("pass") is True
        and diagnosis.get("case_count") == 5
        and _same_artifact(diagnosis.get("known_failure_manifest"), evidence["known_failure_manifest"])
        and _same_artifact(diagnosis.get("diagnosis_lock"), evidence["diagnosis_lock"])
        and corpus.get("mismatch_count") == 92,
        "B64 diagnosis index contract or artifact binding differs.",
    )

    _require(oracle.get("pass") is True, "Raw repair oracle report failed.")
    _require(
        oracle.get("schema_version") == B64_ORACLE_REPORT_SCHEMA,
        "Raw repair oracle schema differs.",
    )
    acceptance = oracle.get("acceptance_policy", {})
    _require(acceptance.get("all_required_oracle_comparisons_pass") is True, "Raw primary/repeat oracle acceptance failed.")
    comparisons = acceptance.get("required_oracle_comparisons", {})
    expected_keys = {"primary_full_sensor", "primary_rgb_only", "repeat_full_sensor", "repeat_rgb_only"}
    _require(isinstance(comparisons, dict) and set(comparisons) == expected_keys, "Raw oracle comparison set differs.")
    _require(all(value is True for value in comparisons.values()), "Raw oracle comparison failed.")
    oracle_internal_checks = oracle.get("checks") or {}
    _require(
        isinstance(oracle_internal_checks, dict)
        and bool(oracle_internal_checks)
        and all(value is True for value in oracle_internal_checks.values()),
        "Raw oracle report producer checks failed.",
    )
    oracle_contract = oracle.get("contract") or {}
    _require(
        oracle_contract.get("batch") == 64
        and oracle_contract.get("width") == 128
        and oracle_contract.get("height") == 128
        and oracle_contract.get("step") == 107
        and oracle_contract.get("camera_indices") == list(range(64))
        and oracle_contract.get("selection_pairs") == [[107, camera] for camera in range(64)]
        and oracle_contract.get("semantic_topology") == "spatial-octants-8"
        and oracle_contract.get("gpu_uuid") == aggregate_record["matrix"]["gpu_uuid"],
        "Raw B64 oracle report contract differs.",
    )
    _require(
        oracle_contract.get("trajectory_id") == primary_trajectory_id,
        "Raw B64 oracle trajectory differs from the repair reports.",
    )
    input_artifacts = oracle.get("input_artifacts")
    required_inputs = {
        "primary_repair_report": "repair_report",
        "primary_repair_full_raw": "primary_repair_full_raw",
        "primary_repair_rgb_only_raw": "primary_repair_rgb_only_raw",
        "repeat_repair_report": "repeat_repair_report",
        "repeat_repair_full_raw": "repeat_repair_full_raw",
        "repeat_repair_rgb_only_raw": "repeat_repair_rgb_only_raw",
        "diagnosis_index": "diagnosis_index",
        "diagnosis_lock": "diagnosis_lock",
        "oracle": "oracle",
        "oracle_manifest": "oracle_manifest",
        "oracle_camera_bundle": "oracle_camera_bundle",
    }
    _require(isinstance(input_artifacts, dict), "Raw B64 oracle report has no input artifacts.")
    for input_name, evidence_name in required_inputs.items():
        _require(
            _same_artifact(input_artifacts.get(input_name), evidence[evidence_name]),
            f"Raw B64 oracle input {input_name} differs from normalized evidence.",
        )

    oracle_manifest = _load_evidence_json(evidence_root, evidence, "oracle_manifest")
    oracle_camera_bundle = _load_evidence_json(evidence_root, evidence, "oracle_camera_bundle")
    _require(
        oracle_manifest.get("schema_version") == B64_ORACLE_MANIFEST_SCHEMA
        and oracle_manifest.get("pass") is True,
        "Fresh B64 gsplat oracle manifest failed or has the wrong schema.",
    )
    _require(
        oracle_manifest.get("output") == _record_path(evidence["oracle"], label="B64 oracle")
        and oracle_manifest.get("output_sha256") == evidence["oracle"]["sha256"],
        "Fresh B64 gsplat oracle output binding differs.",
    )
    _require(
        oracle_manifest.get("camera_bundle")
        == _record_path(evidence["oracle_camera_bundle"], label="B64 oracle camera bundle")
        and _same_artifact(
            oracle_manifest.get("camera_bundle_artifact"),
            evidence["oracle_camera_bundle"],
        ),
        "Fresh B64 gsplat oracle camera-bundle binding differs.",
    )
    bundle_id = oracle_camera_bundle.get("bundle_id")
    bundle_without_id = dict(oracle_camera_bundle)
    bundle_without_id.pop("bundle_id", None)
    calculated_bundle_id = hashlib.sha256(
        json.dumps(bundle_without_id, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cameras = oracle_camera_bundle.get("cameras")
    _require(
        oracle_camera_bundle.get("schema_version") == "camera-bundle-v1"
        and oracle_camera_bundle.get("width") == 128
        and oracle_camera_bundle.get("height") == 128
        and oracle_camera_bundle.get("scene_checksum") == aggregate_record["matrix"]["scene_sha256"]
        and isinstance(cameras, list)
        and len(cameras) == 64
        and [camera.get("view_id") for camera in cameras if isinstance(camera, dict)]
        == [f"t107-b{camera:04d}" for camera in range(64)]
        and bundle_id == calculated_bundle_id
        and oracle_manifest.get("camera_bundle_id") == bundle_id,
        "Fresh B64 gsplat oracle camera-bundle contract differs.",
    )
    _require(
        _same_artifact(oracle_manifest.get("node_occupancy"), evidence["oracle_node_occupancy"]),
        "Fresh B64 gsplat oracle inner occupancy binding differs.",
    )
    _require(
        oracle_manifest.get("gpu_uuid") == aggregate_record["matrix"]["gpu_uuid"]
        and oracle_manifest.get("scene_sha256") == aggregate_record["matrix"]["scene_sha256"]
        and oracle_manifest.get("camera_indices") == list(range(64))
        and oracle_manifest.get("steps") == [107] * 64
        and oracle_manifest.get("selection_pairs") == [[107, camera] for camera in range(64)]
        and oracle_manifest.get("selection_profile") == "diagnostic-single-step"
        and oracle_manifest.get("semantic_topology") == "spatial-octants-8"
        and oracle_manifest.get("trajectory_id") == oracle_contract.get("trajectory_id"),
        "Fresh B64 gsplat oracle selection or identity differs.",
    )
    _require(
        _same_artifact(
            (oracle_manifest.get("source_provenance") or {}).get("manifest"),
            aggregate_record["benchmark"]["source_manifest"],
        ),
        "Fresh B64 gsplat oracle source manifest differs.",
    )
    _validate_node_occupancy(
        evidence_root,
        evidence["oracle_node_occupancy"],
        expected_gpu_uuid=aggregate_record["matrix"]["gpu_uuid"],
        label="Fresh B64 gsplat oracle inner occupancy",
    )

    support_definitions = (
        ("primary", "support-b64-repair-primary", "verify_flashgs_b64_repair.py", repair_record),
        ("repeat", "support-b64-repair-repeat", "verify_flashgs_b64_repair.py", repeat_record),
        ("oracle", "support-b64-gsplat-oracle", "run_flashgs_gsplat_oracle.py", evidence["oracle"]),
    )
    support_wrappers: dict[str, dict[str, Any]] = {}
    for prefix, gate_id, script_name, output_record in support_definitions:
        support_wrappers[prefix] = _validate_b64_support_producer(
            evidence_root,
            evidence,
            prefix=prefix,
            gate_id=gate_id,
            script_name=script_name,
            output_record=output_record,
            result=result,
            aggregate_record=aggregate_record,
        )
    for prefix, report in (("primary", repair), ("repeat", repeat)):
        wrapper = support_wrappers[prefix]
        _require(
            _parse_time(wrapper["started_at"], label=f"B64 {prefix} support started_at")
            <= _parse_time(report["created_at"], label=f"B64 {prefix} report created_at")
            <= _parse_time(wrapper["completed_at"], label=f"B64 {prefix} support completed_at"),
            f"B64 {prefix} repair report was not created during its support run.",
        )
    _require(
        _parse_time(support_wrappers["primary"]["completed_at"], label="B64 primary support completed_at")
        <= _parse_time(support_wrappers["repeat"]["started_at"], label="B64 repeat support started_at")
        and _parse_time(support_wrappers["repeat"]["completed_at"], label="B64 repeat support completed_at")
        <= _parse_time(support_wrappers["oracle"]["started_at"], label="B64 oracle support started_at"),
        "B64 support producers were not executed as three ordered, non-overlapping runs.",
    )
    support_occupancy_paths = {
        _record_path(evidence[f"{prefix}_support_occupancy"], label=f"B64 {prefix} support occupancy")
        for prefix in ("primary", "repeat", "oracle")
    }
    _require(len(support_occupancy_paths) == 3, "B64 support producers reuse occupancy evidence paths.")
    _require(
        _record_path(evidence["oracle_node_occupancy"], label="B64 oracle inner occupancy")
        != _record_path(evidence["oracle_support_occupancy"], label="B64 oracle outer occupancy"),
        "B64 oracle inner and outer occupancy evidence are the same file.",
    )


def _validate_sanitizer(
    gate_id: str,
    evidence_root: EvidenceRoot,
    result: dict[str, Any],
    log_text: str,
) -> None:
    checks = result["checks"]
    _require(checks.get("tool") == "compute-sanitizer", "Sanitizer tool differs.")
    _require(checks.get("error_summary") == 0, "Sanitizer structured error count is nonzero.")
    _require(checks.get("application_exit_code") == 0, "Sanitized application failed.")
    _require(checks.get("bounded") is True, "Sanitizer run was not bounded.")
    zero_lines = re.findall(r"(?m)^.*ERROR SUMMARY:\s*0 errors.*$", log_text)
    _require(len(zero_lines) == 1, "Sanitizer log must contain exactly one literal zero-error summary.")
    _require(re.search(r"ERROR SUMMARY:\s*[1-9][0-9]* errors", log_text) is None, "Sanitizer log contains errors.")
    target = _load_evidence_json(evidence_root, result["evidence"], "target_result")
    if gate_id == "repair-compute-sanitizer":
        _require(
            target.get("schema_version") == "flashgs-short-tail-synthetic-verification-v1"
            and target.get("pass") is True,
            "Repair sanitizer target result did not pass.",
        )
        expected_checks = {
            "full_sensor_matches_cpu_oracle",
            "rgb_only_matches_cpu_oracle",
            "rgb_specializations_bitwise_equal",
            "workspace_layouts_equal",
            "output_contracts_valid",
            "occupancy_pass",
        }
        raw_checks = target.get("checks")
        _require(isinstance(raw_checks, dict) and set(raw_checks) == expected_checks and all(value is True for value in raw_checks.values()), "Repair sanitizer target checks failed.")
        for specialization in (target.get("specializations") or {}).values():
            _zero_native_overflow(specialization.get("counters"), label="repair sanitizer target")
            equal = specialization.get("repeat_bitwise_equal")
            _require(isinstance(equal, dict) and equal and all(value is True for value in equal.values()), "Repair sanitizer repeatability failed.")
    else:
        payload = _sentinel_payload(log_text, "CUSTOM_CUDA_BACKEND_SMOKE_OK", json_payload=False)
        _require(target == payload, "Custom sanitizer target differs from the sole log payload.")
        _zero_native_overflow(payload.get("nonempty"), label="custom sanitizer target")
        _zero_native_overflow(payload.get("empty"), label="custom sanitizer empty target")


def _validate_pytest(
    evidence_root: EvidenceRoot,
    result: dict[str, Any],
    log_text: str,
) -> None:
    checks = result["checks"]
    for name in ("collected", "passed", "failures", "errors", "skipped_required"):
        _require(isinstance(checks.get(name), int) and not isinstance(checks[name], bool), f"Pytest {name} is not an integer.")
    _require(checks["collected"] > 0 and checks["passed"] > 0, "Pytest executed no passing tests.")
    _require(checks["failures"] == 0 and checks["errors"] == 0 and checks["skipped_required"] == 0, "Pytest has failures/errors/required skips.")
    required = checks.get("required_tests")
    _require(isinstance(required, list) and required and all(isinstance(item, str) and item for item in required), "Pytest required-test set is empty.")
    _require(log_text.count("PYTEST_PUBLICATION_OK") == 1, "Pytest sentinel is missing or duplicated.")
    junit_path = evidence_root.resolve_record(result["evidence"].get("junit_xml"), label="pytest JUnit XML")
    try:
        root = ET.parse(junit_path).getroot()
    except ET.ParseError as error:
        raise VerificationError("Pytest JUnit XML is malformed.") from error
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    _require(bool(suites), "Pytest JUnit XML has no test suites.")
    totals = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    _require(totals["tests"] == checks["collected"], "Pytest JUnit collected count differs.")
    _require(totals["failures"] == 0 and totals["errors"] == 0, "Pytest JUnit contains failures/errors.")
    _require(checks["passed"] == totals["tests"] - totals["skipped"], "Pytest passed count differs from JUnit.")
    testcase_ids = {
        f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}".strip(":")
        for case in root.findall(".//testcase")
    }
    _require(set(required).issubset(testcase_ids), "Pytest required tests are not present in JUnit evidence.")


def _sentinel_payload(log_text: str, sentinel: str, *, json_payload: bool) -> dict[str, Any]:
    lines = [line for line in log_text.splitlines() if line.startswith(sentinel + " ")]
    _require(len(lines) == 1, f"{sentinel} must occur once as a complete payload line.")
    encoded = lines[0][len(sentinel) + 1 :]
    try:
        value = json.loads(encoded) if json_payload else ast.literal_eval(encoded)
    except (ValueError, SyntaxError, json.JSONDecodeError) as error:
        raise VerificationError(f"{sentinel} payload is malformed.") from error
    _require(isinstance(value, dict), f"{sentinel} payload is not an object.")
    return value


def _zero_native_overflow(counters: Any, *, label: str) -> None:
    _require(isinstance(counters, dict), f"{label} counters are absent.")
    _require(counters.get("visible_overflow") == 0, f"{label} visible capacity overflowed.")
    intersection = counters.get("intersection_overflow", counters.get("tile_intersection_overflow"))
    _require(intersection == 0, f"{label} intersection capacity overflowed.")


def _validate_native(
    gate_id: str,
    evidence_root: EvidenceRoot,
    result: dict[str, Any],
    log_text: str,
) -> None:
    checks = result["checks"]
    sentinel = NATIVE_SENTINELS[gate_id]
    _require(checks.get("sentinel") == sentinel, f"{gate_id} sentinel declaration differs.")
    _require(log_text.count(sentinel) == 1, f"{gate_id} sentinel is missing or duplicated.")
    for name in ("json_pass", "all_outputs_valid", "zero_overflow", "cuda_resident", "finite_values"):
        _require(checks.get(name) is True, f"{gate_id} predicate {name} failed.")
    if gate_id == "native-determinism":
        _require(checks.get("bitwise_equal") is True, "Native deterministic outputs are not bitwise equal.")
    if gate_id == "native-multi-scene":
        _require(isinstance(checks.get("scene_count"), int) and checks["scene_count"] >= 2, "Multi-scene gate covered fewer than two scenes.")
    raw = _load_evidence_json(evidence_root, result["evidence"], "raw_result")
    payload = _sentinel_payload(
        log_text,
        sentinel,
        json_payload=gate_id == "native-determinism",
    )
    _require(raw == payload, f"{gate_id} raw result differs from the sole log payload.")
    if gate_id == "native-cuda-smoke":
        _zero_native_overflow(payload.get("nonempty"), label=gate_id)
        _zero_native_overflow(payload.get("empty"), label=f"{gate_id} empty-scene")
        _require((payload.get("empty") or {}).get("visible_gaussians") == 0, "Native empty-scene proof is not empty.")
        chunked = payload.get("chunked")
        _require(isinstance(chunked, dict) and chunked, "Native chunked-submission proof is absent.")
    elif gate_id == "native-determinism":
        _require(payload.get("schema_version") == "deterministic-cuda-smoke/v1" and payload.get("pass") is True, "Raw deterministic payload failed.")
        equal = payload.get("bitwise_equal")
        invariant = payload.get("capacity_invariant_bitwise_equal")
        _require(isinstance(equal, dict) and set(equal) == set(OUTPUTS) and all(value is True for value in equal.values()), "Raw deterministic outputs differ.")
        _require(isinstance(invariant, dict) and set(invariant) == set(OUTPUTS) and all(value is True for value in invariant.values()), "Raw deterministic capacity invariant failed.")
        _require(isinstance(payload.get("iterations"), int) and payload["iterations"] >= 2, "Raw deterministic proof has fewer than two replays.")
        _zero_native_overflow(payload.get("counters"), label=gate_id)
    else:
        scene_ids = payload.get("scene_ids")
        _require(isinstance(scene_ids, list) and len(set(scene_ids)) >= 2, "Raw multi-scene proof has fewer than two scenes.")
        _require(
            payload.get("inactive_output_policy")
            == {"rgb": 0, "depth": "+inf", "alpha": 0, "semantic_id": -1},
            "Raw multi-scene inactive-output policy differs.",
        )
        _zero_native_overflow(payload, label=gate_id)
    if gate_id == "native-determinism":
        _require(checks.get("bitwise_equal") is True, "Normalized deterministic predicate differs.")
    if gate_id == "native-multi-scene":
        _require(checks.get("scene_count") == len(set(payload["scene_ids"])), "Normalized multi-scene count differs.")


def _validate_isaac(
    evidence_root: EvidenceRoot,
    result: dict[str, Any],
    log_text: str,
) -> None:
    checks = result["checks"]
    for name in ("source_unchanged", "runtime_unchanged", "simulation_app_closed", "all_outputs_valid", "zero_overflow"):
        _require(checks.get(name) is True, f"Isaac predicate {name} failed.")
    _require(checks.get("residual_process_count") == 0, "Isaac left residual processes.")
    _require(checks.get("active_gpu_count") == 1, "Isaac did not use exactly one active GPU.")
    lanes = checks.get("lanes")
    _require(lanes == ISAAC_LANES, "Isaac lane/sentinel set differs.")
    for sentinel in (*ISAAC_LANES.values(), "ISAAC_GRAPHICS_GATE_OK"):
        _require(log_text.count(sentinel) == 1, f"Isaac log is missing or duplicates {sentinel}.")
    raw = _load_evidence_json(evidence_root, result["evidence"], "isaac_summary")
    _require(raw.get("schema_version") == "isaac-graphics-gate/v1" and raw.get("pass") is True, "Raw Isaac graphics summary failed.")
    _require(raw.get("source_unchanged") is True and raw.get("runtime_unchanged") is True, "Raw Isaac source/runtime changed.")
    _require(raw.get("residual_process_acceptance", {}).get("pass") is True, "Raw Isaac summary has residual processes.")
    raw_lanes = raw.get("lanes")
    _require(isinstance(raw_lanes, list), "Raw Isaac summary has no lanes.")
    lane_names = [lane.get("name") for lane in raw_lanes if isinstance(lane, dict)]
    _require(len(lane_names) == len(ISAAC_LANES) and set(lane_names) == set(ISAAC_LANES), "Raw Isaac lanes differ or repeat.")
    _require(raw.get("processes_before", {}).get("gpu_compute", {}).get("empty") is True, "Isaac preflight found an active GPU process.")
    for lane in raw_lanes:
        _require(lane.get("process", {}).get("returncode") == 0, "An Isaac lane exited nonzero.")
        _require(lane.get("process", {}).get("timed_out") is False, "An Isaac lane timed out.")
        _require(lane.get("log_acceptance", {}).get("returncode") == 0, "An Isaac lane log failed acceptance.")
        _require(lane.get("residual_process_acceptance", {}).get("pass") is True, "An Isaac lane left residuals.")
        active_rows = lane.get("gpu_foundation", {}).get("active_rows")
        _require(isinstance(active_rows, list) and len(active_rows) == 1, "An Isaac lane did not use exactly one GPU.")


def _validate_fidelity(
    evidence_root: EvidenceRoot,
    result: dict[str, Any],
    record: dict[str, Any],
) -> None:
    checks = result["checks"]
    _require(checks.get("output_contracts") == list(OUTPUTS), "Fidelity output set differs.")
    _require(checks.get("all_views_pass") is True, "Fidelity has a failing view.")
    _require(checks.get("finite_values") is True and checks.get("zero_overflow") is True, "Fidelity finite/overflow gate failed.")
    raw_record = result["evidence"].get("raw_result")
    _require(_same_artifact(raw_record, record["matrix"]["summary"]), "Fidelity gate is not bound to the validated matrix summary.")
    raw = _load_evidence_json(evidence_root, result["evidence"], "raw_result")
    _require(raw.get("schema_version") == "flashgs-matched-summary-v4", "Raw fidelity summary schema differs.")
    _require(raw.get("pass") is True and raw.get("scientific_pass") is True, "Raw fidelity matrix failed.")
    rows = [
        *raw.get("primary_full_sensor_dynamic_table", []),
        *raw.get("rgb_only_dynamic_table", []),
    ]
    _require(len(rows) == 16 and all(isinstance(row, dict) and row.get("fidelity_pass") is True for row in rows), "Raw matrix does not prove all 16 fidelity rows.")


def _replay_expected_output_specs() -> dict[str, dict[str, Any]]:
    batch = REPLAY_ARGUMENTS["batch"]
    height = REPLAY_ARGUMENTS["height"]
    width = REPLAY_ARGUMENTS["width"]
    return {
        "rgb": {
            "bytes": batch * height * width * 3 * 4,
            "dtype": "torch.float32",
            "shape": [batch, height, width, 3],
        },
        "alpha": {
            "bytes": batch * height * width * 4,
            "dtype": "torch.float32",
            "shape": [batch, height, width, 1],
        },
        "depth": {
            "bytes": batch * height * width * 4,
            "dtype": "torch.float32",
            "shape": [batch, height, width, 1],
        },
        "semantic_id": {
            "bytes": batch * height * width * 8,
            "dtype": "torch.int64",
            "shape": [batch, height, width, 1],
        },
    }


def _validate_replay_output_digests(value: Any) -> None:
    _require(
        isinstance(value, dict) and set(value) == set(OUTPUTS),
        "Replay output digests do not cover the exact output contract.",
    )
    expected = _replay_expected_output_specs()
    for name in OUTPUTS:
        record = value[name]
        _require(
            isinstance(record, dict)
            and set(record) == {"bytes", "dtype", "sha256", "shape"},
            f"Replay {name} digest record is malformed.",
        )
        _require(
            record.get("bytes") == expected[name]["bytes"]
            and record.get("dtype") == expected[name]["dtype"]
            and record.get("shape") == expected[name]["shape"]
            and isinstance(record.get("sha256"), str)
            and SHA256_RE.fullmatch(record["sha256"]) is not None,
            f"Replay {name} digest contract differs.",
        )


def _validate_replay_output_contract(value: Any) -> None:
    expected_fields = (
        REPLAY_OUTPUT_BOOLEAN_FIELDS
        | REPLAY_OUTPUT_COUNT_FIELDS
        | {"output_devices"}
    )
    _require(
        isinstance(value, dict) and set(value) == expected_fields,
        "Replay output validation has missing or unknown fields.",
    )
    _require(
        all(value.get(field) is True for field in REPLAY_OUTPUT_BOOLEAN_FIELDS),
        "Replay output validation contains a failed predicate.",
    )
    for field in REPLAY_OUTPUT_COUNT_FIELDS:
        _require(
            isinstance(value.get(field), int)
            and not isinstance(value[field], bool)
            and value[field] >= 0,
            f"Replay output validation count {field} is malformed.",
        )
    total = REPLAY_ARGUMENTS["batch"] * REPLAY_ARGUMENTS["height"] * REPLAY_ARGUMENTS["width"]
    _require(
        value["foreground_pixel_count"] > 0
        and value["semantic_foreground_pixel_count"] > 0
        and value["foreground_pixel_count"] + value["background_pixel_count"] == total
        and value["semantic_foreground_pixel_count"]
        + value["semantic_background_pixel_count"]
        == total
        and value["semantic_foreground_pixel_count"] <= value["foreground_pixel_count"],
        "Replay output validation coverage differs from the rendered tensor shape.",
    )
    devices = value.get("output_devices")
    _require(
        isinstance(devices, dict)
        and set(devices) == set(OUTPUTS)
        and all(
            isinstance(device, str)
            and re.fullmatch(r"cuda(?::[0-9]+)?", device) is not None
            for device in devices.values()
        )
        and len(set(devices.values())) == 1,
        "Replay outputs are not bound to one CUDA device.",
    )


def _validate_replay_native(
    evidence_root: EvidenceRoot,
    payload: dict[str, Any],
    *,
    expected_native_sha256: str,
) -> None:
    native = payload.get("native_extension")
    _require(
        isinstance(native, dict) and set(native) == {"bytes", "path", "sha256"},
        "Replay native extension artifact is malformed.",
    )
    _require(
        native.get("sha256") == expected_native_sha256,
        "Replay loaded a native extension different from the normalized gate identity.",
    )
    evidence_root.resolve_record(native, label="replay loaded native extension")
    contract = payload.get("native_build_contract")
    _require(
        isinstance(contract, dict) and set(contract) == REPLAY_NATIVE_BUILD_FIELDS,
        "Replay native build contract is malformed.",
    )
    module_name = contract.get("module_name")
    sources = contract.get("sources")
    _require(
        isinstance(module_name, str)
        and re.fullmatch(r"isaacsim_gaussian_renderer_cuda_[0-9a-f]{12}", module_name)
        is not None
        and contract.get("cxx_flags") == ["-O3", "-std=c++17"]
        and contract.get("cuda_flags")
        == ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"]
        and contract.get("torch_cuda_arch_list") == "8.9"
        and isinstance(contract.get("build_directory"), str)
        and bool(contract["build_directory"])
        and isinstance(sources, list)
        and len(sources) == 2
        and [Path(source).name for source in sources if isinstance(source, str)]
        == ["renderer.cpp", "renderer_cuda.cu"]
        and Path(str(native.get("path"))).parent
        == Path(contract["build_directory"])
        and Path(str(native.get("path"))).name.startswith(module_name),
        "Replay native build identity differs from the frozen custom backend contract.",
    )


def _validate_replay_worker(
    evidence_root: EvidenceRoot,
    payload: Any,
    *,
    expected_source_smoke: dict[str, Any],
    expected_native_sha256: str,
) -> dict[str, Any]:
    _require(
        isinstance(payload, dict) and set(payload) == REPLAY_WORKER_FIELDS,
        "Replay worker payload has missing or unknown fields.",
    )
    _require(
        payload.get("schema_version") == REPLAY_WORKER_SCHEMA
        and payload.get("pass") is True,
        "Replay worker schema/pass differs.",
    )
    _require(payload.get("arguments") == REPLAY_ARGUMENTS, "Replay worker arguments differ.")
    _require(payload.get("claim_scope") == REPLAY_CLAIM_SCOPE, "Replay claim scope differs.")
    _require(
        payload.get("comparison_method") == REPLAY_COMPARISON_METHOD,
        "Replay digest method differs.",
    )
    _require(
        payload.get("renderer_configuration") == REPLAY_RENDERER_CONFIGURATION,
        "Replay renderer configuration differs.",
    )
    _require(
        _same_artifact(payload.get("source_smoke"), expected_source_smoke),
        "Replay worker source-smoke identity differs.",
    )
    evidence_root.resolve_record(payload["source_smoke"], label="replay source smoke")
    fixture = payload.get("fixture_sha256")
    _require(
        isinstance(fixture, str) and SHA256_RE.fullmatch(fixture) is not None,
        "Replay fixture SHA-256 is malformed.",
    )
    for field in ("bitwise_equal", "capacity_invariant_bitwise_equal"):
        equal = payload.get(field)
        _require(
            isinstance(equal, dict)
            and set(equal) == set(OUTPUTS)
            and all(equal[name] is True for name in OUTPUTS),
            f"Replay worker {field} failed.",
        )
    counters = payload.get("counters")
    _require(
        isinstance(counters, dict) and set(counters) == REPLAY_COUNTER_FIELDS,
        "Replay worker counters are malformed.",
    )
    _require(
        all(
            isinstance(counters[field], int)
            and not isinstance(counters[field], bool)
            and counters[field] >= 0
            for field in REPLAY_COUNTER_FIELDS
        )
        and counters["visible_overflow"] == 0
        and counters["intersection_overflow"] == 0
        and counters["visible_gaussians"] > 0
        and counters["tile_intersections"] > 0
        and counters["active_tiles"] > 0,
        "Replay worker counters fail the nonempty zero-overflow contract.",
    )
    _require(
        payload.get("equal_depth_gaussian_id_order") == "ascending"
        and payload.get("equal_depth_records_checked") == REPLAY_ARGUMENTS["gaussians"]
        and payload.get("center_semantic") == 1_000,
        "Replay equal-depth ordering proof differs.",
    )
    _validate_replay_output_digests(payload.get("output_digests"))
    _validate_replay_output_contract(payload.get("output_contract"))
    _validate_replay_native(
        evidence_root,
        payload,
        expected_native_sha256=expected_native_sha256,
    )
    return payload


def _replay_log_payloads(log_text: str, sentinel: str, *, count: int) -> list[dict[str, Any]]:
    prefix = sentinel + " "
    lines = [line[len(prefix) :] for line in log_text.splitlines() if line.startswith(prefix)]
    _require(len(lines) == count, f"{sentinel} must occur exactly {count} times.")
    values: list[dict[str, Any]] = []
    for encoded in lines:
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise VerificationError(f"{sentinel} payload is malformed.") from error
        _require(isinstance(value, dict), f"{sentinel} payload is not an object.")
        values.append(value)
    return values


def _validate_replay(
    evidence_root: EvidenceRoot,
    result: dict[str, Any],
    log_text: str,
    command: dict[str, Any],
) -> None:
    checks = result["checks"]
    _require(
        set(checks)
        == {
            "bitwise_equal",
            "cross_process_output_hashes_equal",
            "fresh_processes",
            "output_contracts",
            "replays_per_process",
            "zero_overflow",
        },
        "Replay normalized checks have missing or unknown fields.",
    )
    _require(checks.get("output_contracts") == list(OUTPUTS), "Replay output set differs.")
    _require(checks.get("fresh_processes") == 2, "Replay did not launch two fresh processes.")
    _require(checks.get("replays_per_process") == 32, "Replay count per process differs.")
    for field in ("bitwise_equal", "cross_process_output_hashes_equal", "zero_overflow"):
        _require(checks.get(field) is True, f"Replay normalized predicate {field} failed.")

    evidence = result["evidence"]
    _require(set(evidence) == REPLAY_EVIDENCE_KEYS, "Replay evidence set differs.")
    raw = _load_evidence_json(evidence_root, evidence, "raw_result")
    _require(set(raw) == REPLAY_RESULT_FIELDS, "Replay parent result has missing or unknown fields.")
    parent_payload = _sentinel_payload(
        log_text,
        "DETERMINISTIC_CROSS_PROCESS_REPLAY_OK",
        json_payload=True,
    )
    _require(raw == parent_payload, "Replay parent result differs from its log payload.")

    runner_record = evidence["runner_script"]
    worker_record = evidence["worker_script"]
    source_record = evidence["source_smoke"]
    for name, record in (
        ("runner_script", runner_record),
        ("worker_script", worker_record),
        ("source_smoke", source_record),
    ):
        _require(
            isinstance(record, dict) and set(record) == {"bytes", "path", "sha256"},
            f"Replay {name} artifact is malformed.",
        )
        evidence_root.resolve_record(record, label=f"replay {name}")
        _require(_same_artifact(raw.get(name), record), f"Replay {name} binding differs.")

    outer_argv = command.get("command")
    _require(isinstance(outer_argv, list), "Replay outer command is absent.")
    for record, label in (
        (runner_record, "runner"),
        (worker_record, "worker"),
        (source_record, "source smoke"),
    ):
        _require(record["path"] in outer_argv, f"Replay command does not bind the {label} path.")
    separators = [index for index, token in enumerate(outer_argv) if token == "--"]
    _require(
        len(separators) == 1 and separators[0] + 2 < len(outer_argv),
        "Replay adapter command has no unique executable child boundary.",
    )
    executed = outer_argv[separators[0] + 1 :]
    _require(
        executed[1] == runner_record["path"]
        and Path(executed[1]).name == "run_deterministic_replay.py",
        "Replay adapter does not execute the bound replay runner.",
    )

    def exact_runner_flag(name: str) -> str:
        positions = [position for position, token in enumerate(executed) if token == name]
        _require(
            len(positions) == 1 and positions[0] + 1 < len(executed),
            f"Replay runner command omits exact {name}.",
        )
        return executed[positions[0] + 1]

    _require(
        exact_runner_flag("--python") == executed[0],
        "Replay runner child-Python identity differs.",
    )
    _require(
        exact_runner_flag("--worker-script") == worker_record["path"],
        "Replay runner does not bind the normalized worker.",
    )
    _require(
        exact_runner_flag("--source-smoke") == source_record["path"],
        "Replay runner does not bind the normalized source smoke.",
    )
    _require(
        exact_runner_flag("--output") == evidence["raw_result"]["path"],
        "Replay runner output differs from the normalized raw result.",
    )

    process_payloads = [
        _load_evidence_json(evidence_root, evidence, "process_1"),
        _load_evidence_json(evidence_root, evidence, "process_2"),
    ]
    log_payloads = _replay_log_payloads(
        log_text,
        "DETERMINISTIC_DIGEST_SMOKE_OK",
        count=2,
    )
    _require(process_payloads == log_payloads, "Replay worker files differ from their log payloads.")
    expected_native_sha256 = result["identity"]["native_extension_sha256"]
    for payload in process_payloads:
        _validate_replay_worker(
            evidence_root,
            payload,
            expected_source_smoke=source_record,
            expected_native_sha256=expected_native_sha256,
        )
    _require(
        process_payloads[0] == process_payloads[1],
        "Replay worker payloads differ across fresh processes.",
    )

    processes = raw.get("processes")
    _require(isinstance(processes, list) and len(processes) == 2, "Replay process records differ.")
    child_commands: list[list[str]] = []
    for index, process in enumerate(processes, start=1):
        _require(
            isinstance(process, dict)
            and set(process) == {"command", "index", "payload", "pid", "returncode"}
            and process.get("index") == index
            and process.get("returncode") == 0
            and isinstance(process.get("pid"), int)
            and not isinstance(process["pid"], bool)
            and process["pid"] > 0,
            f"Replay process {index} record is malformed.",
        )
        _require(
            _same_artifact(process.get("payload"), evidence[f"process_{index}"]),
            f"Replay process {index} payload artifact differs.",
        )
        child = process.get("command")
        _require(
            isinstance(child, list)
            and child
            and all(isinstance(token, str) and token for token in child),
            f"Replay process {index} command is malformed.",
        )
        child_commands.append(child)
    _require(child_commands[0] == child_commands[1], "Replay child commands differ.")
    child = child_commands[0]
    _require(
        len(child) >= 2
        and child[1] == worker_record["path"]
        and Path(child[1]).name == "deterministic_digest_smoke.py",
        "Replay child command does not execute the bound worker.",
    )

    def exact_child_flag(name: str) -> str:
        positions = [position for position, token in enumerate(child) if token == name]
        _require(
            len(positions) == 1 and positions[0] + 1 < len(child),
            f"Replay child command omits exact {name}.",
        )
        return child[positions[0] + 1]

    _require(exact_child_flag("--source-smoke") == source_record["path"], "Replay source path differs.")
    for name, expected in (
        ("--batch", 4),
        ("--gaussians", 1_024),
        ("--iterations", 32),
        ("--width", 64),
        ("--height", 64),
        ("--tile-size", 1),
    ):
        _require(exact_child_flag(name) == str(expected), f"Replay child {name} differs.")
    _require("--ray-gaussian-evaluation" not in child, "Replay unexpectedly used ray evaluation.")

    first = process_payloads[0]
    for name in (
        "claim_scope",
        "comparison_method",
        "fixture_sha256",
        "native_build_contract",
        "native_extension",
        "output_contract",
        "output_digests",
        "renderer_configuration",
    ):
        _require(raw.get(name) == first.get(name), f"Replay parent field {name} differs.")
    _require(
        raw.get("schema_version") == REPLAY_SCHEMA
        and raw.get("pass") is True
        and raw.get("fresh_processes") == 2
        and raw.get("replays_per_process") == 32
        and raw.get("process_payloads_equal") is True
        and raw.get("cross_process_output_hashes_equal") is True
        and raw.get("within_process_bitwise_equal") is True
        and raw.get("zero_overflow") is True,
        "Replay parent acceptance predicates failed.",
    )


def _validate_soak(evidence_root: EvidenceRoot, result: dict[str, Any], log_text: str) -> None:
    checks = result["checks"]
    _require(log_text.count("CUSTOM_CUDA_SOAK_OK") == 1, "Soak sentinel is missing or duplicated.")
    for name in ("requested_duration_seconds", "duration_seconds"):
        value = checks.get(name)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 600.0, f"Soak {name} is shorter than 600 seconds.")
    _require(checks.get("output_contracts") == list(OUTPUTS), "Soak output set differs.")
    for name in ("zero_overflow", "all_outputs_valid", "outputs_gpu_resident", "no_cuda_errors"):
        _require(checks.get(name) is True, f"Soak predicate {name} failed.")
    raw = _load_evidence_json(evidence_root, result["evidence"], "soak_result")
    _require(raw.get("schema_version") == "real-scene-custom-benchmark/v1" and raw.get("pass") is True, "Raw soak is not the Home Scan benchmark.")
    _require(float(raw.get("requested_duration_seconds", 0.0)) >= 600.0, "Raw requested soak is shorter than 600 seconds.")
    _require(float(raw.get("duration_seconds", 0.0)) >= 600.0, "Raw soak is shorter than 600 seconds.")
    _require(raw.get("measurement_mode") == "duration_soak", "Raw soak was not duration bounded.")
    _require(raw.get("batch") == 128 and raw.get("width") == 128 and raw.get("height") == 128, "Raw soak workload differs.")
    _require(raw.get("dataset", {}).get("sha256") == result.get("identity", {}).get("scene_sha256"), "Raw soak scene differs.")
    _require(raw.get("zero_overflow") is True and raw.get("all_output_checks_valid") is True, "Raw soak output/overflow gate failed.")
    _require(raw.get("outputs_gpu_resident") is True and raw.get("outputs_contract_valid") is True, "Raw soak CUDA/output contract failed.")
    _require(raw.get("allocation_delta_bytes") == 0, "Raw soak allocated during steady state.")
    counters = raw.get("measured_counters") or {}
    _require(counters.get("visible_overflow") == 0 and counters.get("intersection_overflow") == 0, "Raw soak counters overflowed.")


def _validate_vectorization_ablation(
    evidence_root: EvidenceRoot,
    result: dict[str, Any],
    record: dict[str, Any],
) -> None:
    raw = _load_evidence_json(evidence_root, result["evidence"], "raw_result")
    expected_fields = {"batch", "pass", "ratios", "run_order", "runs", "schema_version"}
    _require(set(raw) == expected_fields, "Vectorization ablation record has missing or unknown fields.")
    _require(raw.get("schema_version") == "publication-custom-vectorization-ablation-v1", "Vectorization ablation schema differs.")
    _require(raw.get("pass") is True and raw.get("batch") == 128, "Vectorization ablation did not pass B128.")
    entries = raw.get("runs")
    _require(isinstance(entries, list) and len(entries) == 12, "Vectorization ablation must contain twelve runs.")
    by_key: dict[tuple[str, int, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    by_id: dict[str, tuple[str, int, int]] = {}
    run_artifacts: set[tuple[str, int]] = set()
    occupancy_artifacts: set[tuple[str, int]] = set()
    capacity_by_physical: dict[int, tuple[str, int]] = {}

    summary = _read_json(
        evidence_root.resolve_record(record["matrix"]["summary"], label="matrix summary for ablation"),
        label="matrix summary for ablation",
    )
    reference_by_contract: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for table_name, contract in (
        ("primary_full_sensor_dynamic_table", "full"),
        ("rgb_only_dynamic_table", "rgb"),
    ):
        row = next(item for item in summary[table_name] if item["batch"] == 128)
        reference_run = _read_json(
            evidence_root.resolve_record(row["artifacts"]["custom_run"], label=f"matrix B128 {contract} run"),
            label=f"matrix B128 {contract} run",
        )
        reference_fidelity = _read_json(
            evidence_root.resolve_record(row["artifacts"]["custom_fidelity"], label=f"matrix B128 {contract} fidelity"),
            label=f"matrix B128 {contract} fidelity",
        )
        reference_by_contract[contract] = reference_run, reference_fidelity

    expected_entry_fields = {"capacity", "contract", "fidelity", "physical_batch", "run", "run_id", "trial"}
    for entry in entries:
        _require(isinstance(entry, dict) and set(entry) == expected_entry_fields, "Ablation run entry is malformed.")
        contract = entry.get("contract")
        trial = entry.get("trial")
        physical = entry.get("physical_batch")
        run_id = entry.get("run_id")
        _require(contract in {"full", "rgb"} and trial in {1, 2, 3} and physical in {1, 128}, "Ablation run key differs.")
        _require(isinstance(run_id, str) and run_id and run_id not in by_id, "Ablation run ID is missing or duplicated.")
        key = (contract, trial, physical)
        _require(key not in by_key, "Ablation run design cell is duplicated.")
        identity = _identity(entry["run"], label=f"ablation {run_id} run")
        _require(identity not in run_artifacts, "Ablation reused a raw run instead of a fresh process.")
        run_artifacts.add(identity)
        run = _read_json(
            evidence_root.resolve_record(entry["run"], label=f"ablation {run_id} run"),
            label=f"ablation {run_id} run",
        )
        reference_run, reference_fidelity = reference_by_contract[contract]
        _require(run.get("schema_version") == RENDERER_RUN_SCHEMA and run.get("pass") is True, f"Ablation {run_id} failed.")
        _require(run.get("renderer") == "custom" and run.get("output_contract") == contract, f"Ablation {run_id} identity differs.")
        camera = run.get("camera_contract") or {}
        reference_camera = reference_run.get("camera_contract") or {}
        stable_camera_fields = (
            "trajectory_id",
            "batch",
            "width",
            "height",
            "warmup_frames",
            "measured_frames",
            "motion_classification",
            "unchanged_measured_camera_frame_pairs",
            "minimum_consecutive_viewmat_max_abs_delta",
        )
        _require(
            camera.get("batch") == 128
            and all(camera.get(field) == reference_camera.get(field) for field in stable_camera_fields),
            f"Ablation {run_id} camera contract differs from the matrix oracle.",
        )
        upload_seconds = _finite_number(camera.get("camera_upload_seconds"), label=f"Ablation {run_id} camera upload seconds")
        _require(upload_seconds >= 0.0, f"Ablation {run_id} camera upload time is negative.")
        _require(run.get("equation_contract") == reference_run.get("equation_contract"), f"Ablation {run_id} equation differs.")
        _require(run.get("scene", {}).get("sha256") == record["matrix"]["scene_sha256"], f"Ablation {run_id} scene differs.")
        environment = run.get("environment") or {}
        provenance = environment.get("source_provenance") or {}
        _require(environment.get("gpu_uuid") == record["matrix"]["gpu_uuid"], f"Ablation {run_id} GPU differs.")
        _require(provenance.get("head") == record["benchmark"]["commit"] and provenance.get("dirty") is False, f"Ablation {run_id} source differs.")
        _require(_same_artifact(provenance.get("manifest"), record["matrix"]["source_manifest"]), f"Ablation {run_id} source manifest differs.")
        _require(_same_artifact(environment.get("native_extension"), record["matrix"]["custom_native_extension"]), f"Ablation {run_id} native extension differs.")
        config = run.get("runner_config") or {}
        execution = run.get("backend_execution") or {}
        _require(config.get("independent_trial") == trial, f"Ablation {run_id} trial identity differs.")
        _require(config.get("custom_max_physical_views") == physical, f"Ablation {run_id} physical policy differs.")
        _require(execution.get("logical_batch") == 128 and execution.get("physical_batch") == physical, f"Ablation {run_id} execution batch differs.")
        _require(execution.get("native_submissions_per_logical_batch") == 128 // physical, f"Ablation {run_id} submission count differs.")
        _validate_output_contract(run, contract=contract, label=f"Ablation {run_id}")
        observed = ((run.get("capacity") or {}).get("timed_verification") or {}).get("observed") or {}
        _require(observed.get("measured_max_visible_overflow") == 0 and observed.get("measured_max_intersection_overflow") == 0, f"Ablation {run_id} overflowed.")
        gpu_samples, gpu_mean = _timing_distribution(run, "gpu_batch_ms", label=f"Ablation {run_id}")
        wall_samples, wall_mean = _timing_distribution(run, "synchronized_wall_batch_ms", label=f"Ablation {run_id}")
        _require(len(gpu_samples) == len(wall_samples) == 100, f"Ablation {run_id} sample count differs.")

        capacity_record = entry.get("capacity")
        _require(_same_artifact(capacity_record, (run.get("capacity") or {}).get("calibration_artifact")), f"Ablation {run_id} capacity binding differs.")
        capacity_identity = _identity(capacity_record, label=f"ablation {run_id} capacity")
        previous_capacity = capacity_by_physical.setdefault(physical, capacity_identity)
        _require(previous_capacity == capacity_identity, f"Ablation P{physical} capacity artifact changed across runs.")
        capacity = _read_json(
            evidence_root.resolve_record(capacity_record, label=f"ablation P{physical} capacity"),
            label=f"ablation P{physical} capacity",
        )
        _require(capacity.get("schema_version") == "flashgs-matched-capacity-calibration-v1" and capacity.get("pass") is True, f"Ablation P{physical} capacity failed.")
        _require(capacity.get("renderer") == "custom" and capacity.get("camera_contract", {}).get("batch") == 128, f"Ablation P{physical} capacity workload differs.")
        _require(capacity.get("output_validation", {}).get("pass") is True, f"Ablation P{physical} capacity output validation failed.")
        installed = capacity.get("capacity") or {}
        _require(installed.get("logical_batch") == 128 and installed.get("physical_batch") == physical, f"Ablation P{physical} calibrated the wrong physical batch.")
        _require(installed.get("native_submissions_per_logical_batch") == 128 // physical, f"Ablation P{physical} calibration submission count differs.")
        expected_scope = "physical-camera-chunk-reused-workspace" if physical == 1 else "batch-global-workspace"
        _require(installed.get("intersection_capacity_scope") == expected_scope, f"Ablation P{physical} capacity scope differs.")
        capacity_camera = capacity.get("camera_contract") or {}
        _require(capacity_camera.get("timesteps") == 108 and capacity_camera.get("measured_start") == 8, f"Ablation P{physical} capacity frame coverage differs.")
        _require(installed.get("headroom") == 1.05, f"Ablation P{physical} capacity headroom differs.")
        verified = installed.get("verified_preflight") or {}
        _require(verified.get("max_visible_overflow") == 0 and verified.get("max_intersection_overflow") == 0, f"Ablation P{physical} calibration overflowed.")
        capacity_environment = capacity.get("environment") or {}
        capacity_source = capacity_environment.get("source_provenance") or {}
        _require(capacity_environment.get("gpu_uuid") == record["matrix"]["gpu_uuid"], f"Ablation P{physical} capacity GPU differs.")
        _require(_same_artifact(capacity_environment.get("native_extension"), record["matrix"]["custom_native_extension"]), f"Ablation P{physical} capacity native extension differs.")
        _require(_same_artifact(capacity_source.get("manifest"), record["matrix"]["source_manifest"]), f"Ablation P{physical} capacity source differs.")

        occupancy = environment.get("node_occupancy")
        occupancy_identity = _identity(occupancy, label=f"ablation {run_id} occupancy")
        _require(occupancy_identity not in occupancy_artifacts, "Ablation reused occupancy evidence across fresh processes.")
        occupancy_artifacts.add(occupancy_identity)
        evidence_root.resolve_record(occupancy, label=f"ablation {run_id} occupancy")

        fidelity_record = entry.get("fidelity")
        if trial == 1:
            _identity(fidelity_record, label=f"ablation {run_id} fidelity")
            fidelity = _read_json(
                evidence_root.resolve_record(fidelity_record, label=f"ablation {run_id} fidelity"),
                label=f"ablation {run_id} fidelity",
            )
            _require(fidelity.get("schema_version") == FIDELITY_SCHEMA and fidelity.get("pass") is True, f"Ablation {run_id} oracle fidelity failed.")
            _require(fidelity.get("renderer") == "custom" and fidelity.get("output_contract") == contract and fidelity.get("batch") == 128, f"Ablation {run_id} fidelity identity differs.")
            _require(_same_artifact((fidelity.get("input_artifacts") or {}).get("run"), entry["run"]), f"Ablation {run_id} fidelity uses another run.")
            _require(
                _same_artifact(
                    (fidelity.get("input_artifacts") or {}).get("oracle_manifest"),
                    (reference_fidelity.get("input_artifacts") or {}).get("oracle_manifest"),
                ),
                f"Ablation {run_id} does not use the matrix B128 oracle.",
            )
        else:
            _require(fidelity_record is None, f"Ablation {run_id} unexpectedly carries non-trial1 fidelity.")
        by_key[key] = run, {"gpu_mean": gpu_mean, "wall_mean": wall_mean}
        by_id[run_id] = key

    expected_keys = {(contract, trial, physical) for contract in ("full", "rgb") for trial in (1, 2, 3) for physical in (1, 128)}
    _require(set(by_key) == expected_keys, "Vectorization ablation design is incomplete.")
    _require(set(capacity_by_physical) == {1, 128} and capacity_by_physical[1] != capacity_by_physical[128], "P1 and P128 must use distinct exact capacity artifacts.")
    order = raw.get("run_order")
    _require(isinstance(order, list) and len(order) == 12 and set(order) == set(by_id), "Ablation run order differs from its run set.")
    orientations: dict[str, list[tuple[int, int]]] = {"full": [], "rgb": []}
    for contract in ("full", "rgb"):
        for trial in (1, 2, 3):
            positions = sorted((order.index(run_id), physical) for run_id, key in by_id.items() if key[:2] == (contract, trial) for physical in (key[2],))
            _require(len(positions) == 2 and positions[1][0] == positions[0][0] + 1, "Ablation counterbalanced pair is not adjacent.")
            orientations[contract].append((positions[0][1], positions[1][1]))
        _require(orientations[contract][0] == orientations[contract][2] and orientations[contract][0] != orientations[contract][1], f"Ablation {contract} order is not counterbalanced.")
    _require(orientations["full"][0] != orientations["rgb"][0], "Full and RGB ablations must start with opposite order.")

    ratios = raw.get("ratios")
    _require(isinstance(ratios, dict) and set(ratios) == {"full", "rgb"}, "Ablation ratio contracts differ.")
    for contract in ("full", "rgb"):
        rows = ratios[contract]
        _require(isinstance(rows, list) and len(rows) == 3, f"Ablation {contract} ratio trials differ.")
        for expected_trial, ratio in enumerate(rows, 1):
            _require(isinstance(ratio, dict) and set(ratio) == {"cuda_speedup_p128_over_p1", "trial", "wall_speedup_p128_over_p1"}, "Ablation ratio row is malformed.")
            _require(ratio["trial"] == expected_trial, "Ablation ratio trial order differs.")
            p1 = by_key[(contract, expected_trial, 1)][1]
            p128 = by_key[(contract, expected_trial, 128)][1]
            _close(ratio["cuda_speedup_p128_over_p1"], p1["gpu_mean"] / p128["gpu_mean"], label=f"Ablation {contract} trial {expected_trial} CUDA ratio")
            _close(ratio["wall_speedup_p128_over_p1"], p1["wall_mean"] / p128["wall_mean"], label=f"Ablation {contract} trial {expected_trial} wall ratio")


def _validate_command(
    gate_id: str,
    command: dict[str, Any],
    gate: dict[str, Any],
    record: dict[str, Any],
) -> None:
    expected_fields = {"command", "gate_id", "schema_version", "started_at", "subcommands", "timeout_seconds"}
    _require(set(command) == expected_fields, f"{gate_id} command record has missing or unknown fields.")
    _require(command.get("schema_version") == COMMAND_SCHEMA, f"{gate_id} command schema differs.")
    _require(command.get("gate_id") == gate_id, f"{gate_id} command gate identity differs.")
    argv = command.get("command")
    _require(isinstance(argv, list) and argv and all(isinstance(item, str) and item for item in argv), f"{gate_id} command is absent.")
    _require(not any("fixture" in item.lower() or "placeholder" in item.lower() for item in argv), f"{gate_id} command is a placeholder.")
    subcommands = command.get("subcommands")
    _require(
        isinstance(subcommands, list)
        and all(
            isinstance(item, list)
            and item
            and all(isinstance(token, str) and token for token in item)
            for item in subcommands
        ),
        f"{gate_id} subcommand record is malformed.",
    )
    flattened = [*argv, *(token for item in subcommands for token in item)]
    tokens = set(flattened) | {Path(item).name for item in flattened}
    for alternatives in COMMAND_TOKEN_GROUPS[gate_id]:
        _require(any(candidate in tokens for candidate in alternatives), f"{gate_id} command omits required argv token {alternatives!r}.")
    timeout = command.get("timeout_seconds")
    _require(
        isinstance(timeout, int) and not isinstance(timeout, bool) and 1 <= timeout <= 7_200,
        f"{gate_id} command has no finite bounded timeout.",
    )
    if gate_id == "bounded-stability-soak":
        _require(timeout >= 600, "Soak command timeout is shorter than its required duration.")
        _require("--duration-seconds" in tokens and "600" in tokens, "Soak command does not request exactly 600 seconds.")
    if gate_id in {"repair-compute-sanitizer", "compute-sanitizer-memcheck"}:
        _require("--error-exitcode" in tokens and "99" in tokens, f"{gate_id} command lacks fail-closed sanitizer exit handling.")
    if gate_id == "all-output-fidelity":
        def exact_flag(name: str) -> str:
            positions = [index for index, token in enumerate(argv) if token == name]
            _require(
                len(positions) == 1 and positions[0] + 1 < len(argv),
                f"Fidelity reconstruction command omits exact {name} value.",
            )
            return argv[positions[0] + 1]

        _require(
            sum(token == "--verify-existing" for token in argv) == 1,
            "Fidelity reconstruction must use exactly one --verify-existing flag.",
        )
        summary_path = Path(record["matrix"]["summary"]["path"])
        _require(
            Path(exact_flag("--root")).resolve(strict=False)
            == summary_path.parent.resolve(strict=False),
            "Fidelity reconstruction root differs from the canonical matrix result root.",
        )
        _require(
            Path(exact_flag("--output")).resolve(strict=False)
            == summary_path.resolve(strict=False),
            "Fidelity reconstruction output differs from the canonical matrix summary.",
        )
        _require(
            exact_flag("--batches") == ",".join(str(batch) for batch in PRIMARY_BATCHES),
            "Fidelity reconstruction batch matrix differs.",
        )
    if gate_id == "deterministic-replay":
        def replay_flag(name: str) -> str:
            positions = [index for index, token in enumerate(argv) if token == name]
            _require(
                len(positions) == 1 and positions[0] + 1 < len(argv),
                f"Replay command omits exact {name} value.",
            )
            return argv[positions[0] + 1]

        _require(
            Path(replay_flag("--worker-script")).name
            == "deterministic_digest_smoke.py",
            "Replay command uses the wrong digest worker.",
        )
        _require(
            Path(replay_flag("--source-smoke")).name
            == "deterministic_cuda_smoke.py",
            "Replay command binds the wrong frozen smoke source.",
        )
        for flag, expected in (
            ("--batch", "4"),
            ("--gaussians", "1024"),
            ("--iterations", "32"),
            ("--width", "64"),
            ("--height", "64"),
            ("--tile-size", "1"),
        ):
            _require(replay_flag(flag) == expected, f"Replay command {flag} differs.")
        _require(
            "--ray-gaussian-evaluation" not in argv,
            "Replay command unexpectedly uses ray-Gaussian evaluation.",
        )
    if gate_id == "custom-vectorization-ablation":
        _require(len(subcommands) == 12, "Vectorization ablation must bind twelve fresh-process commands.")
        observed: set[tuple[str, int, int]] = set()
        for subcommand in subcommands:
            def flag(name: str) -> str:
                positions = [index for index, token in enumerate(subcommand) if token == name]
                _require(len(positions) == 1 and positions[0] + 1 < len(subcommand), f"Ablation command omits {name}.")
                return subcommand[positions[0] + 1]

            _require(any(Path(token).name == "run_flashgs_matched.py" for token in subcommand), "Ablation subcommand uses the wrong runner.")
            _require(flag("--renderer") == "custom", "Ablation subcommand is not Custom.")
            contract = flag("--output-contract")
            trial = int(flag("--independent-trial"))
            physical = int(flag("--custom-max-physical-views"))
            _require(contract in {"full", "rgb"} and trial in {1, 2, 3} and physical in {1, 128}, "Ablation subcommand contract differs.")
            observed.add((contract, trial, physical))
        expected = {(contract, trial, physical) for contract in ("full", "rgb") for trial in (1, 2, 3) for physical in (1, 128)}
        _require(observed == expected, "Ablation subcommands do not cover the exact 2x2x3 design.")
    else:
        _require(subcommands == [], f"{gate_id} unexpectedly contains subcommands.")
    _require(command.get("started_at") == gate.get("started_at"), f"{gate_id} command start timestamp differs.")


def _validate_machine_provenance(
    evidence_root: EvidenceRoot,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Bind one portable machine record to all 32 timed primary runs."""

    path = evidence_root.logical_path(MACHINE_PROVENANCE_LOGICAL_PATH)
    machine = _read_json(path, label="publication machine provenance")
    expected_fields = {
        "compiler_versions",
        "compute_capability",
        "cuda_runtime",
        "driver",
        "gpu_name",
        "gpu_uuid",
        "pass",
        "primary_run_count",
        "primary_runs",
        "schema_version",
        "source_commit",
        "source_manifest",
        "torch",
        "torch_cuda_arch_list",
    }
    _require(
        set(machine) == expected_fields,
        "Machine provenance has missing or unknown fields.",
    )
    _require(
        machine.get("schema_version") == MACHINE_PROVENANCE_SCHEMA
        and machine.get("pass") is True,
        "Machine provenance is not a passing v1 record.",
    )
    _require(machine.get("gpu_name") == "NVIDIA L4", "Machine provenance GPU name differs.")
    _require(
        machine.get("gpu_uuid") == record["matrix"]["gpu_uuid"],
        "Machine provenance GPU UUID differs.",
    )
    _require(
        machine.get("compute_capability") == [8, 9]
        and machine.get("torch_cuda_arch_list") == "8.9",
        "Machine provenance architecture differs.",
    )
    for field in ("driver", "cuda_runtime", "torch"):
        _require(
            isinstance(machine.get(field), str) and bool(machine[field]),
            f"Machine provenance {field} is missing.",
        )
    compilers = machine.get("compiler_versions")
    _require(
        isinstance(compilers, dict)
        and set(compilers) == {"cxx", "nvcc"}
        and all(isinstance(compilers[field], str) and bool(compilers[field]) for field in compilers),
        "Machine provenance compiler versions differ.",
    )
    _require(
        machine.get("source_commit") == record["benchmark"]["commit"],
        "Machine provenance source commit differs.",
    )
    source_record = machine.get("source_manifest")
    _require(
        isinstance(source_record, dict)
        and set(source_record) == {"bytes", "logical_path", "sha256"}
        and source_record.get("logical_path") == "provenance/source-manifest.json"
        and _same_artifact(source_record, record["matrix"]["source_manifest"]),
        "Machine provenance source-manifest binding differs.",
    )
    _require(
        _same_artifact(
            source_record,
            {
                "bytes": evidence_root.logical_path("provenance/source-manifest.json").stat().st_size,
                "sha256": file_sha256(
                    evidence_root.logical_path("provenance/source-manifest.json")
                ),
            },
        ),
        "Machine provenance canonical source manifest differs.",
    )

    summary = _read_json(
        evidence_root.resolve_record(record["matrix"]["summary"], label="matrix summary"),
        label="matrix summary",
    )
    summary_runs: dict[tuple[int, str, str], Any] = {}
    for table_name, contract in (
        ("primary_full_sensor_dynamic_table", "full"),
        ("rgb_only_dynamic_table", "rgb"),
    ):
        for row in summary[table_name]:
            for renderer in ("custom", "flashgs"):
                summary_runs[(row["batch"], renderer, contract)] = row["artifacts"][
                    f"{renderer}_run"
                ]

    expected_order = [
        (batch, renderer, contract)
        for batch in PRIMARY_BATCHES
        for renderer in ("custom", "flashgs")
        for contract in ("full", "rgb")
    ]
    primary_runs = machine.get("primary_runs")
    _require(
        machine.get("primary_run_count") == len(expected_order)
        and isinstance(primary_runs, list)
        and len(primary_runs) == len(expected_order),
        "Machine provenance does not cover all 32 primary runs.",
    )
    common_environment: dict[str, Any] | None = None
    for item, (batch, renderer, contract) in zip(primary_runs, expected_order):
        logical_path = f"runs/{renderer}/{contract}/b{batch}.json"
        _require(
            isinstance(item, dict)
            and set(item) == {"bytes", "logical_path", "sha256"}
            and item.get("logical_path") == logical_path,
            f"Machine provenance primary-run order differs at {logical_path}.",
        )
        canonical_path = evidence_root.logical_path(logical_path)
        canonical_identity = {
            "bytes": canonical_path.stat().st_size,
            "sha256": file_sha256(canonical_path),
        }
        _require(
            _same_artifact(item, canonical_identity)
            and _same_artifact(item, summary_runs[(batch, renderer, contract)]),
            f"Machine provenance run binding differs at {logical_path}.",
        )
        run = _read_json(canonical_path, label=f"machine-bound {logical_path}")
        environment = run.get("environment")
        _require(isinstance(environment, dict), f"{logical_path} has no environment.")
        normalized = {field: environment.get(field) for field in MACHINE_ENVIRONMENT_FIELDS}
        _require(
            normalized["gpu_name"] == machine["gpu_name"]
            and normalized["gpu_uuid"] == machine["gpu_uuid"]
            and normalized["driver"] == machine["driver"]
            and normalized["cuda_runtime"] == machine["cuda_runtime"]
            and normalized["torch"] == machine["torch"]
            and normalized["compute_capability"] == machine["compute_capability"]
            and normalized["torch_cuda_arch_list"] == machine["torch_cuda_arch_list"]
            and normalized["compiler_versions"] == machine["compiler_versions"],
            f"Machine provenance environment differs at {logical_path}.",
        )
        if common_environment is None:
            common_environment = normalized
        else:
            _require(
                normalized == common_environment,
                f"Primary machine environments disagree at {logical_path}.",
            )
        provenance = environment.get("source_provenance")
        _require(
            environment.get("source_git_commit") == record["benchmark"]["commit"]
            and isinstance(provenance, dict)
            and _same_artifact(provenance.get("manifest"), source_record),
            f"Machine provenance source binding differs at {logical_path}.",
        )

    return {
        "bytes": path.stat().st_size,
        "logical_path": MACHINE_PROVENANCE_LOGICAL_PATH,
        "sha256": file_sha256(path),
    }


def _validate_gate(
    evidence_root: EvidenceRoot,
    gate: dict[str, Any],
    record: dict[str, Any],
) -> None:
    gate_id = gate["gate_id"]
    benchmark_source = record["benchmark"]["source_manifest"]
    native_role = NATIVE_ROLE[gate_id]
    matrix_native = record["matrix"][f"{native_role}_native_extension"]
    _require(gate.get("native_role") == native_role, f"{gate_id} native role differs.")
    _require(_same_artifact(gate["source_manifest"], benchmark_source), f"{gate_id} uses a stale source manifest.")
    _require(_same_artifact(gate["native_extension"], matrix_native), f"{gate_id} uses a stale native extension.")
    _require(gate.get("gpu_required") is GPU_REQUIRED[gate_id], f"{gate_id} GPU requirement differs.")
    if GPU_REQUIRED[gate_id]:
        _require(gate.get("gpu_uuid") == record["matrix"]["gpu_uuid"], f"{gate_id} GPU UUID differs.")
        occupancy = _read_json(
            evidence_root.resolve_record(gate.get("occupancy"), label=f"{gate_id} occupancy"),
            label=f"{gate_id} occupancy",
        )
        _require(occupancy.get("schema_version") == "flashgs-matched-node-occupancy-v2", f"{gate_id} occupancy schema differs.")
        _require(occupancy.get("pass") is True, f"{gate_id} occupancy failed.")
        _require(occupancy.get("expected_gpu_uuid") == record["matrix"]["gpu_uuid"], f"{gate_id} occupancy does not bind the GPU UUID.")
        executor = occupancy.get("executor_control") or {}
        lock = executor.get("cooperative_node_wide_lock") or {}
        sampling = occupancy.get("sampled_compute_process_telemetry") or {}
        _require(executor.get("scope") == "all-visible-gpus", f"{gate_id} occupancy scope differs.")
        _require(lock.get("pass") is True and lock.get("lock_observed_held") is True, f"{gate_id} cooperative GPU lock failed.")
        _require(sampling.get("pass") is True and isinstance(sampling.get("sample_count"), int) and sampling["sample_count"] >= 2, f"{gate_id} occupancy sampling is insufficient.")
    else:
        _require(gate.get("gpu_uuid") is None and gate.get("occupancy") is None, f"{gate_id} unexpectedly claims GPU evidence.")

    exit_path = evidence_root.resolve_record(gate["exit_status"], label=f"{gate_id} exit status")
    exit_text = exit_path.read_text(encoding="utf-8").strip()
    _require(re.fullmatch(r"-?[0-9]+", exit_text) is not None, f"{gate_id} exit status is not numeric.")
    _require(int(exit_text) == gate.get("exit_code") == 0, f"{gate_id} exited nonzero.")
    command = _read_json(
        evidence_root.resolve_record(gate["command"], label=f"{gate_id} command"),
        label=f"{gate_id} command",
    )
    _validate_command(gate_id, command, gate, record)
    log_path = evidence_root.resolve_record(gate["log"], label=f"{gate_id} full log")
    log_text = log_path.read_text(encoding="utf-8", errors="strict")
    _require(bool(log_text.strip()), f"{gate_id} full log is empty.")
    result = _read_json(
        evidence_root.resolve_record(gate["result"], label=f"{gate_id} result"),
        label=f"{gate_id} result",
    )
    _require(result.get("schema_version") == GATE_RESULT_SCHEMA, f"{gate_id} result schema differs.")
    _require(set(result) == {"checks", "created_at", "evidence", "gate_id", "identity", "pass", "schema_version"}, f"{gate_id} result has missing or unknown fields.")
    _require(result.get("gate_id") == gate_id and result.get("pass") is True, f"{gate_id} normalized result failed.")
    _require(isinstance(result.get("checks"), dict) and len(result["checks"]) >= 3, f"{gate_id} is a shallow pass-only result.")
    _require(isinstance(result.get("evidence"), dict) and result["evidence"], f"{gate_id} has no raw evidence bindings.")
    identity = result.get("identity")
    _require(
        isinstance(identity, dict)
        and set(identity) == {"gpu_uuid", "native_extension_sha256", "scene_sha256", "source_manifest_sha256"},
        f"{gate_id} has a malformed result identity.",
    )
    _require(identity.get("source_manifest_sha256") == benchmark_source["sha256"], f"{gate_id} result source identity is stale.")
    _require(identity.get("native_extension_sha256") == matrix_native["sha256"], f"{gate_id} result native identity is stale.")
    expected_gpu = record["matrix"]["gpu_uuid"] if GPU_REQUIRED[gate_id] else None
    _require(identity.get("gpu_uuid") == expected_gpu, f"{gate_id} result GPU identity differs.")
    _require(identity.get("scene_sha256") == record["matrix"]["scene_sha256"], f"{gate_id} result scene identity differs.")
    for nested in _artifact_records(result["evidence"]):
        evidence_root.resolve_record(nested, label=f"{gate_id} nested evidence")

    start = _parse_time(gate.get("started_at"), label=f"{gate_id} started_at")
    result_time = _parse_time(result.get("created_at"), label=f"{gate_id} result created_at")
    completed = _parse_time(gate.get("completed_at"), label=f"{gate_id} completed_at")
    aggregate_time = _parse_time(record.get("created_at"), label="verification created_at")
    _require(gate.get("result_created_at") == result.get("created_at"), f"{gate_id} result timestamp binding differs.")
    _require(start <= result_time <= completed <= aggregate_time, f"{gate_id} has a pre-result or out-of-order timestamp.")

    if gate_id == "flashgs-b64-repair":
        _validate_repair(evidence_root, result, record)
    elif gate_id in {"repair-compute-sanitizer", "compute-sanitizer-memcheck"}:
        _validate_sanitizer(gate_id, evidence_root, result, log_text)
    elif gate_id == "python-unit-contract":
        _validate_pytest(evidence_root, result, log_text)
    elif gate_id in NATIVE_SENTINELS:
        _validate_native(gate_id, evidence_root, result, log_text)
    elif gate_id == "isaac-graphics":
        _validate_isaac(evidence_root, result, log_text)
    elif gate_id == "all-output-fidelity":
        _validate_fidelity(evidence_root, result, record)
    elif gate_id == "custom-vectorization-ablation":
        _validate_vectorization_ablation(evidence_root, result, record)
    elif gate_id == "deterministic-replay":
        _validate_replay(evidence_root, result, log_text, command)
    elif gate_id == "bounded-stability-soak":
        _validate_soak(evidence_root, result, log_text)
    else:  # pragma: no cover - exact gate set makes this unreachable.
        raise VerificationError(f"Unknown verification gate: {gate_id}.")


def verify_aggregate_verification(
    root: str | Path,
    *,
    verification_logical_path: str = VERIFICATION_LOGICAL_PATH,
) -> dict[str, Any]:
    """Verify the aggregate record from staging or a relocated bundle."""

    evidence_root = EvidenceRoot.open(root)
    record_path = evidence_root.logical_path(verification_logical_path)
    record = _read_json(record_path, label="publication verification record")
    expected_top = {
        "benchmark",
        "created_at",
        "gates",
        "matrix",
        "pass",
        "required_gate_ids",
        "schema_version",
    }
    _require(set(record) == expected_top, "Verification record has missing or unknown fields.")
    _require(record.get("schema_version") == VERIFICATION_SCHEMA, "Verification record schema differs.")
    _require(record.get("pass") is True, "Verification record pass is not true.")
    _parse_time(record.get("created_at"), label="verification created_at")
    _require(record.get("required_gate_ids") == list(REQUIRED_GATE_IDS), "Required gate declaration differs.")

    benchmark = record.get("benchmark")
    _require(isinstance(benchmark, dict) and set(benchmark) == {"commit", "source_manifest", "tag"}, "Benchmark identity is malformed.")
    _require(isinstance(benchmark.get("commit"), str) and COMMIT_RE.fullmatch(benchmark["commit"]) is not None, "Benchmark commit is invalid.")
    _require(isinstance(benchmark.get("tag"), str) and benchmark["tag"].strip() == benchmark["tag"] and bool(benchmark["tag"]), "Benchmark tag is invalid.")
    source_path = evidence_root.resolve_record(benchmark["source_manifest"], label="benchmark source manifest")
    source_manifest = _read_json(source_path, label="benchmark source manifest")

    matrix = record.get("matrix")
    expected_matrix = {
        "custom_native_extension",
        "flashgs_native_extension",
        "gpu_uuid",
        "invocation",
        "scene_sha256",
        "source_manifest",
        "summary",
    }
    _require(isinstance(matrix, dict) and set(matrix) == expected_matrix, "Matrix identity is malformed.")
    _require(isinstance(matrix.get("gpu_uuid"), str) and matrix["gpu_uuid"].startswith("GPU-"), "Matrix GPU UUID is invalid.")
    _require(isinstance(matrix.get("scene_sha256"), str) and SHA256_RE.fullmatch(matrix["scene_sha256"]) is not None, "Matrix scene SHA-256 is invalid.")
    _require(_same_artifact(matrix["source_manifest"], benchmark["source_manifest"]), "Matrix and benchmark source manifests differ.")
    evidence_root.resolve_record(matrix["custom_native_extension"], label="matrix Custom native extension")
    evidence_root.resolve_record(matrix["flashgs_native_extension"], label="matrix FlashGS native extension")
    _validate_matrix(evidence_root, record, source_manifest)
    machine_provenance = _validate_machine_provenance(evidence_root, record)

    gates = record.get("gates")
    _require(isinstance(gates, list), "Verification gates must be a list.")
    gate_ids = [gate.get("gate_id") for gate in gates if isinstance(gate, dict)]
    _require(len(gate_ids) == len(gates), "Verification contains a malformed gate.")
    _require(len(gate_ids) == len(set(gate_ids)), "Verification contains duplicate gates.")
    _require(set(gate_ids) == set(REQUIRED_GATE_IDS), "Verification has missing or unknown gates.")
    _require(gate_ids == list(REQUIRED_GATE_IDS), "Verification gates are not in canonical order.")
    expected_gate_fields = {
        "command",
        "completed_at",
        "exit_code",
        "exit_status",
        "gate_id",
        "gpu_required",
        "gpu_uuid",
        "log",
        "native_extension",
        "native_role",
        "occupancy",
        "result",
        "result_created_at",
        "source_manifest",
        "started_at",
    }
    for gate in gates:
        _require(set(gate) == expected_gate_fields, f"Gate {gate.get('gate_id')} has missing or unknown fields.")
        _validate_gate(evidence_root, gate, record)

    ablation_gate = next(gate for gate in gates if gate["gate_id"] == "custom-vectorization-ablation")
    ablation_result = _read_json(
        evidence_root.resolve_record(ablation_gate["result"], label="vectorization ablation gate result"),
        label="vectorization ablation gate result",
    )
    ablation_record = ablation_result["evidence"]["raw_result"]
    ablation_path = evidence_root.resolve_record(
        ablation_record,
        label="aggregate-referenced vectorization ablation",
    )
    canonical_ablation_path = evidence_root.logical_path(ABLATION_LOGICAL_PATH)
    _require(
        canonical_ablation_path.stat().st_size == ablation_record["bytes"]
        and file_sha256(canonical_ablation_path) == ablation_record["sha256"],
        "Canonical publication ablation differs from the aggregate-validated raw result.",
    )

    return {
        "schema_version": "publication-aggregate-verification-receipt-v1",
        "pass": True,
        "root_mode": evidence_root.mode,
        "benchmark_commit": benchmark["commit"],
        "benchmark_tag": benchmark["tag"],
        "gpu_uuid": matrix["gpu_uuid"],
        "scene_sha256": matrix["scene_sha256"],
        "gate_ids": list(REQUIRED_GATE_IDS),
        "verification": {
            "bytes": record_path.stat().st_size,
            "logical_path": VERIFICATION_LOGICAL_PATH,
            "sha256": file_sha256(record_path),
        },
        "summary": {
            "bytes": matrix["summary"]["bytes"],
            "logical_path": "summary.json",
            "sha256": matrix["summary"]["sha256"],
        },
        "ablation": {
            "bytes": ablation_path.stat().st_size,
            "logical_path": ABLATION_LOGICAL_PATH,
            "sha256": file_sha256(ablation_path),
        },
        "machine_provenance": machine_provenance,
    }


def _spec_path(base: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} path is missing.")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise VerificationError(f"{label} is missing: {path}.") from error
    if path.is_symlink() or not resolved.is_file():
        raise VerificationError(f"{label} is not a regular file: {path}.")
    return resolved


def write_verification_from_spec(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    verify_tag: bool = True,
) -> dict[str, Any]:
    """Create a strict aggregate record from a declarative post-result spec."""

    spec_file = Path(spec_path).resolve(strict=True)
    spec = _read_json(spec_file, label="verification specification")
    _require(set(spec) == {"benchmark", "gates", "matrix", "schema_version"}, "Verification spec has missing or unknown fields.")
    _require(spec.get("schema_version") == SPEC_SCHEMA, "Verification spec schema differs.")
    base = spec_file.parent
    benchmark_spec = spec.get("benchmark")
    _require(isinstance(benchmark_spec, dict) and set(benchmark_spec) == {"commit", "repository_root", "source_manifest", "tag"}, "Benchmark spec is malformed.")
    commit = benchmark_spec.get("commit")
    tag = benchmark_spec.get("tag")
    _require(isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None, "Benchmark spec commit is invalid.")
    _require(isinstance(tag, str) and bool(tag.strip()), "Benchmark spec tag is empty.")
    if verify_tag:
        repo = Path(str(benchmark_spec.get("repository_root", "")))
        if not repo.is_absolute():
            repo = base / repo
        repo = repo.resolve(strict=True)
        try:
            tagged_commit = subprocess.check_output(
                ["git", "rev-parse", f"{tag}^{{commit}}"],
                cwd=repo,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise VerificationError(f"Benchmark tag cannot be resolved: {tag!r}.") from error
        _require(tagged_commit == commit, "Benchmark tag does not resolve to the benchmark commit.")
    source_path = _spec_path(base, benchmark_spec["source_manifest"], label="benchmark source manifest")

    matrix_spec = spec.get("matrix")
    _require(
        isinstance(matrix_spec, dict)
        and set(matrix_spec)
        == {
            "custom_native_extension",
            "flashgs_native_extension",
            "gpu_uuid",
            "invocation",
            "scene_sha256",
            "summary",
        },
        "Matrix spec is malformed.",
    )
    custom_native = _spec_path(base, matrix_spec["custom_native_extension"], label="matrix Custom native extension")
    flashgs_native = _spec_path(base, matrix_spec["flashgs_native_extension"], label="matrix FlashGS native extension")
    record: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pass": True,
        "required_gate_ids": list(REQUIRED_GATE_IDS),
        "benchmark": {
            "commit": commit,
            "tag": tag,
            "source_manifest": artifact_record(source_path),
        },
        "matrix": {
            "summary": artifact_record(_spec_path(base, matrix_spec["summary"], label="matrix summary")),
            "invocation": artifact_record(_spec_path(base, matrix_spec["invocation"], label="matrix invocation")),
            "custom_native_extension": artifact_record(custom_native),
            "flashgs_native_extension": artifact_record(flashgs_native),
            "source_manifest": artifact_record(source_path),
            "gpu_uuid": matrix_spec["gpu_uuid"],
            "scene_sha256": matrix_spec["scene_sha256"],
        },
        "gates": [],
    }
    raw_gates = spec.get("gates")
    _require(isinstance(raw_gates, list), "Gate spec must be a list.")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_gates:
        _require(isinstance(raw, dict), "Gate spec contains a malformed gate.")
        expected = {"command", "completed_at", "exit_status", "gate_id", "gpu_uuid", "log", "native_extension", "occupancy", "result", "source_manifest", "started_at"}
        _require(set(raw) == expected, f"Gate spec {raw.get('gate_id')} has missing or unknown fields.")
        gate_id = raw.get("gate_id")
        _require(isinstance(gate_id, str) and gate_id not in by_id, f"Duplicate or invalid gate ID: {gate_id!r}.")
        by_id[gate_id] = raw
    _require(set(by_id) == set(REQUIRED_GATE_IDS), "Gate spec has missing or unknown required gates.")
    for gate_id in REQUIRED_GATE_IDS:
        raw = by_id[gate_id]
        result_path = _spec_path(base, raw["result"], label=f"{gate_id} result")
        result_payload = _read_json(result_path, label=f"{gate_id} result")
        command_path = _spec_path(base, raw["command"], label=f"{gate_id} command")
        command_payload = _read_json(command_path, label=f"{gate_id} command")
        exit_path = _spec_path(base, raw["exit_status"], label=f"{gate_id} exit status")
        exit_text = exit_path.read_text(encoding="utf-8").strip()
        _require(re.fullmatch(r"-?[0-9]+", exit_text) is not None, f"{gate_id} exit status is not numeric.")
        _require(command_payload.get("started_at") == raw["started_at"], f"{gate_id} command timestamp differs from spec.")
        gate = {
            "gate_id": gate_id,
            "result": artifact_record(result_path),
            "command": artifact_record(command_path),
            "log": artifact_record(_spec_path(base, raw["log"], label=f"{gate_id} log")),
            "exit_status": artifact_record(exit_path),
            "exit_code": int(exit_text),
            "source_manifest": artifact_record(_spec_path(base, raw["source_manifest"], label=f"{gate_id} source manifest")),
            "native_extension": artifact_record(_spec_path(base, raw["native_extension"], label=f"{gate_id} native extension")),
            "native_role": NATIVE_ROLE[gate_id],
            "gpu_required": GPU_REQUIRED[gate_id],
            "gpu_uuid": raw["gpu_uuid"],
            "occupancy": (
                artifact_record(_spec_path(base, raw["occupancy"], label=f"{gate_id} occupancy"))
                if raw["occupancy"] is not None
                else None
            ),
            "started_at": raw["started_at"],
            "result_created_at": result_payload.get("created_at"),
            "completed_at": raw["completed_at"],
        }
        record["gates"].append(gate)

    output = Path(output_path).resolve()
    _require(
        output.name == "verification.json" and output.parent.name == "publication",
        "Verification output must be publication/verification.json.",
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite verification record: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(record))
    try:
        verify_aggregate_verification(output.parent.parent)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return record


def parse_verify_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verification-logical-path", default=VERIFICATION_LOGICAL_PATH)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(arguments)


def verify_main(arguments: Iterable[str] | None = None) -> None:
    args = parse_verify_args(arguments)
    receipt = verify_aggregate_verification(
        args.root,
        verification_logical_path=args.verification_logical_path,
    )
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(canonical_json_bytes(receipt))
    print("PUBLICATION_AGGREGATE_VERIFIED " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    verify_main()
