"""Fail-closed evidence contracts for the FlashGS B64 short-tail repair."""

from __future__ import annotations

import json
import hashlib
import math
import difflib
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .benchmark_manifest import sha256_json
from .evaluation.matched_artifacts import (
    HEADLINE_TORCH_CUDA_ARCH_LIST,
    artifact_record,
    same_artifact,
    source_identity,
)
from .fidelity.camera_bundle import load_camera_bundle
from .flashgs_debug import (
    CAUSE_STAGES,
    FLASHGS_B64_KNOWN_FAILURE_CASES_SCHEMA,
    FLASHGS_PIPELINE_DEBUG_SCHEMA,
    MAX_TRACED_CANDIDATES,
    _record_from_rows,
)
from .flashgs_debug_native_loader import load_flashgs_debug_extension
from .flashgs_native_loader import (
    FLASHGS_CUDA_FLAGS,
    FLASHGS_CXX_FLAGS,
    FLASHGS_UPSTREAM_COMMIT,
)


FLASHGS_B64_DIAGNOSIS_INDEX_SCHEMA = "flashgs-b64-diagnosis-index-v1"
FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA = "flashgs-b64-repair-verification-v1"
PRE_FIX_PIPELINE_DEBUG_SCHEMA = "flashgs-pipeline-debug-v3"
PRE_FIX_RENDERER_RUN_SCHEMA = "flashgs-matched-renderer-run-v3"
REPAIR_ALPHA_ATOL = 1.0e-6
REPAIR_RGB_ATOL = 1.0e-5
REPAIR_DEPTH_ATOL = 1.0e-4
# The frozen pre-fix traces were captured on an sm_86 RTX 3090, while the
# publication rerun targets an sm_89 L4.  CUDA implementations of the
# projection/transcendental path are not required to produce bit-identical
# intermediate float32 values across architectures.  Keep that historical
# diagnostic comparison explicitly ULP-bounded; final rendered outputs retain
# the stricter oracle tolerances above.
REPAIR_TRACE_FLOAT32_MAX_ULPS = 128
EXPECTED_CASE_COUNT = 5
EXPECTED_MISMATCH_COUNT = 92
EXPECTED_SELECTED_VIEW_MISMATCH_COUNTS = (0, 0, 2, 0, 87, 1, 1, 1)
FLASHGS_B64_DIAGNOSIS_LOCK_SCHEMA = "flashgs-b64-diagnosis-lock-v1"
FLASHGS_B64_PRE_FIX_COMMIT = "5eca4e640aabd234158d08b4ed61d061e530ee3f"
FLASHGS_B64_PRE_FIX_RENDER_RELATIVE = (
    "src/isaacsim_gaussian_renderer/native/flashgs/repair_reference_render.cu"
)
FLASHGS_B64_PRE_FIX_RENDER_SHA256 = (
    "141ddad0271055b41ef097540a95a9f217cc9e55b52bdec211c4da319486c97a"
)
FLASHGS_PRODUCTION_SOURCE_NAMES = (
    "adapter.cpp",
    "preprocess.cu",
    "sort.cu",
    "render.cu",
)
REPAIRED_SLOT_0_OFFSET_PREDICATE = "else if ((lane & 4) == 0 && point_id + 0 < range.y)"
REPAIRED_SLOT_1_OFFSET_PREDICATE = "else if (point_id + 1 < range.y)"
BUGGY_SLOT_0_OFFSET_PREDICATE = "else if ((lane & 4) == 0 && point_id + 1 < range.y)"
BUGGY_SLOT_1_OFFSET_PREDICATE = "else if (point_id + 2 < range.y)"


def _resolved_artifact_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Artifact root is not a directory: {root}.")
    return root


def _relative_artifact_record(
    path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    root = _resolved_artifact_root(artifact_root)
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Artifact lies outside artifact root: {resolved}.")
    record = artifact_record(resolved)
    relative = resolved.relative_to(root)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative artifact path: {relative}.")
    record["path"] = relative.as_posix()
    return record


def _resolve_relative_artifact(
    record: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    root = _resolved_artifact_root(artifact_root)
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} has no relative path.")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is not safe and relative: {raw_path!r}.")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes artifact root.")
    actual = artifact_record(resolved)
    if not same_artifact(dict(record), actual):
        raise ValueError(f"{label} artifact hash differs.")
    return resolved, actual


def flashgs_adapter_module_name(
    native_root: str | Path,
    *,
    upstream_commit: str = FLASHGS_UPSTREAM_COMMIT,
) -> str:
    """Recompute the production loader's exact source-derived module name."""

    root = Path(native_root).resolve()
    sources = [root / name for name in FLASHGS_PRODUCTION_SOURCE_NAMES]
    header = root / "ops.h"
    missing = [str(path) for path in (*sources, header) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FlashGS production sources are missing: {missing}.")
    digest = hashlib.sha256(upstream_commit.encode("ascii"))
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    digest.update(header.read_bytes())
    return f"isaacsim_flashgs_adapter_{digest.hexdigest()[:12]}"


def audit_repaired_flashgs_production(
    *,
    native_root: str | Path,
    build_contract: Mapping[str, Any] | None,
    repository_root: str | Path,
    pre_fix_source_identity: Mapping[str, Any],
    loaded_module_name: str | None = None,
    loaded_module_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind the loaded binary to the exact two-line production repair."""

    root = Path(native_root).resolve()
    render_path = root / "render.cu"
    repository = Path(repository_root).resolve()
    render_source = render_path.read_text(encoding="utf-8")
    pre_fix_head = pre_fix_source_identity.get("head")
    if not isinstance(pre_fix_head, str) or len(pre_fix_head) != 40:
        raise ValueError("Pre-fix source identity has no full Git commit.")
    render_relative = "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
    pre_fix_render_path = repository / FLASHGS_B64_PRE_FIX_RENDER_RELATIVE
    try:
        pre_fix_render_bytes = pre_fix_render_path.read_bytes()
    except OSError as error:
        raise ValueError(
            "Cannot load the tracked pre-fix render reference at "
            f"{FLASHGS_B64_PRE_FIX_RENDER_RELATIVE}."
        ) from error
    pre_fix_render_sha256 = hashlib.sha256(pre_fix_render_bytes).hexdigest()
    if pre_fix_render_sha256 != FLASHGS_B64_PRE_FIX_RENDER_SHA256:
        raise ValueError("Tracked pre-fix render reference hash differs.")
    try:
        pre_fix_render_source = pre_fix_render_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Frozen pre-fix render source is not UTF-8.") from error
    expected_repaired_source = pre_fix_render_source.replace(
        BUGGY_SLOT_0_OFFSET_PREDICATE,
        REPAIRED_SLOT_0_OFFSET_PREDICATE,
        1,
    ).replace(
        BUGGY_SLOT_1_OFFSET_PREDICATE,
        REPAIRED_SLOT_1_OFFSET_PREDICATE,
        1,
    )
    unified_diff = "\n".join(
        difflib.unified_diff(
            pre_fix_render_source.splitlines(),
            render_source.splitlines(),
            fromfile=f"{pre_fix_head}:{render_relative}",
            tofile=render_relative,
            lineterm="",
        )
    )
    expected_module_name = flashgs_adapter_module_name(root)
    contract = dict(build_contract) if isinstance(build_contract, Mapping) else {}
    expected_sources = [str((root / name).resolve()) for name in FLASHGS_PRODUCTION_SOURCE_NAMES]
    recorded_sources = [str(Path(str(path)).resolve()) for path in contract.get("sources", [])]
    expected_header = str((root / "ops.h").resolve())
    module_path_name_matches = True
    module_artifact: dict[str, Any] | None = None
    if loaded_module_path is not None:
        resolved_module_path = Path(loaded_module_path).resolve()
        module_artifact = artifact_record(resolved_module_path)
        module_path_name_matches = resolved_module_path.name.startswith(expected_module_name)
    checks = {
        "slot_0_repaired_predicate_exactly_once": render_source.count(REPAIRED_SLOT_0_OFFSET_PREDICATE) == 1,
        "slot_1_repaired_predicate_exactly_once": render_source.count(REPAIRED_SLOT_1_OFFSET_PREDICATE) == 1,
        "buggy_slot_0_predicate_absent": BUGGY_SLOT_0_OFFSET_PREDICATE not in render_source,
        "buggy_slot_1_predicate_absent": BUGGY_SLOT_1_OFFSET_PREDICATE not in render_source,
        "slot_0_load_guard_present": ("load_enable = load_enable && point_id + 0 < range.y" in render_source),
        "slot_1_load_guard_present": ("load_enable = load_enable && point_id + 1 < range.y" in render_source),
        "pre_fix_source_identity_clean": pre_fix_source_identity.get("dirty") is False,
        "pre_fix_commit_is_frozen_diagnosis_head": pre_fix_head == FLASHGS_B64_PRE_FIX_COMMIT,
        "pre_fix_reference_matches_frozen_hash": (
            pre_fix_render_sha256 == FLASHGS_B64_PRE_FIX_RENDER_SHA256
        ),
        "pre_fix_has_exact_buggy_predicates": pre_fix_render_source.count(BUGGY_SLOT_0_OFFSET_PREDICATE) == 1
        and pre_fix_render_source.count(BUGGY_SLOT_1_OFFSET_PREDICATE) == 1,
        "production_render_diff_is_exact_two_line_repair": render_source == expected_repaired_source,
        "build_upstream_commit_matches": contract.get("upstream_commit") == FLASHGS_UPSTREAM_COMMIT,
        "build_cxx_flags_match": contract.get("cxx_flags") == list(FLASHGS_CXX_FLAGS),
        "build_cuda_flags_match": contract.get("cuda_flags") == list(FLASHGS_CUDA_FLAGS),
        "build_arch_matches_headline_gpu": (
            contract.get("torch_cuda_arch_list")
            == HEADLINE_TORCH_CUDA_ARCH_LIST
        ),
        "build_module_name_matches_source_digest": contract.get("module_name") == expected_module_name,
        "build_sources_match": recorded_sources == expected_sources,
        "build_header_matches": str(Path(str(contract.get("header", ""))).resolve()) == expected_header,
        "loaded_module_name_matches": loaded_module_name
        in (
            None,
            expected_module_name,
        ),
        "loaded_module_path_name_matches": module_path_name_matches,
    }
    return {
        "expected_module_name": expected_module_name,
        "loaded_module_name": loaded_module_name,
        "loaded_module": module_artifact,
        "render_source": artifact_record(render_path),
        "exact_repair_diff": {
            "pre_fix_commit": pre_fix_head,
            "pre_fix_render_sha256": pre_fix_render_sha256,
            "pre_fix_render_reference": artifact_record(pre_fix_render_path),
            "current_render_sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
            "unified_diff": unified_diff,
        },
        "source_artifacts": [artifact_record(root / name) for name in (*FLASHGS_PRODUCTION_SOURCE_NAMES, "ops.h")],
        "build_contract": contract,
        "checks": checks,
        "pass": all(checks.values()),
    }


def trace_explicit_flashgs_candidates(
    backend: Any,
    *,
    viewmat: torch.Tensor,
    intrinsics: torch.Tensor,
    target_pixel_x: int,
    target_pixel_y: int,
    candidate_ids: Iterable[int],
    expected_semantic_id: int,
    native_module: Any | None = None,
) -> dict[str, Any]:
    """Trace only pinned IDs, avoiding the all-21M-Gaussian discovery scan."""

    scene = getattr(backend, "_scene", None)
    covariances = getattr(backend, "_covariances", None)
    workspace = getattr(backend, "workspace", None)
    if scene is None or covariances is None or not workspace:
        raise RuntimeError("FlashGS backend must retain scene/workspace state.")
    height = int(getattr(backend, "_height", 0))
    width = int(getattr(backend, "_width", 0))
    if not (0 <= target_pixel_x < width and 0 <= target_pixel_y < height):
        raise ValueError("Target pixel lies outside the FlashGS output.")
    ids = [int(value) for value in candidate_ids]
    if (
        not ids
        or len(ids) > MAX_TRACED_CANDIDATES
        or len(ids) != len(set(ids))
        or any(value < 0 or value >= scene.count for value in ids)
    ):
        raise ValueError("Explicit candidate IDs are empty, duplicate, or invalid.")
    viewmat = viewmat.reshape(4, 4).contiguous()
    intrinsics = intrinsics.reshape(3, 3).contiguous()
    if viewmat.device != scene.means.device or intrinsics.device != scene.means.device:
        raise ValueError("Camera and scene tensors must share one device.")
    counters = workspace.get("counters")
    if counters is None or counters.ndim != 2 or counters.shape[1] < 2:
        raise RuntimeError("FlashGS per-camera counters are unavailable.")
    generated_intersections = int(counters[0, 1].item())
    capacity = int(workspace["keys_unsorted"].numel())
    if not 0 <= generated_intersections <= capacity:
        raise RuntimeError("Generated intersections exceed retained capacity.")
    native = native_module or load_flashgs_debug_extension()
    candidate_tensor = torch.tensor(ids, device=scene.means.device, dtype=torch.int64)
    float_trace, int_trace = native.trace_candidates(
        scene.means,
        covariances,
        scene.opacities,
        viewmat,
        intrinsics,
        workspace["keys_unsorted"][:generated_intersections],
        workspace["values_unsorted"][:generated_intersections],
        workspace["keys_sorted"][:generated_intersections],
        workspace["values_sorted"][:generated_intersections],
        workspace["ranges"],
        workspace["points_xy"],
        workspace["conic_opacity"],
        candidate_tensor,
        height,
        width,
        float(backend.near_plane),
        float(backend.far_plane),
        float(backend.gaussian_support_sigma),
        float(backend.covariance_epsilon),
        target_pixel_x,
        target_pixel_y,
    )
    semantic_ids = scene.semantic_ids.index_select(0, candidate_tensor).detach().cpu().tolist()
    records = [
        _record_from_rows(float_row, int_row, semantic_id=int(semantic_id))
        for float_row, int_row, semantic_id in zip(
            float_trace.detach().cpu().tolist(),
            int_trace.detach().cpu().tolist(),
            semantic_ids,
            strict=True,
        )
    ]
    replay_checks: list[bool] = []
    identified_causes: list[dict[str, Any]] = []
    for record in records:
        record["discovery"] = None
        record["discovery_score_bit_exact"] = None
        replay = record["projection"]["retained_workspace_crosscheck"]
        if replay["available"]:
            replay_checks.append(bool(replay["bit_exact_values"]))
        if record["first_loss_stage"] in CAUSE_STAGES:
            identified_causes.append(
                {
                    "gaussian_id": record["gaussian_id"],
                    "first_loss_stage": record["first_loss_stage"],
                }
            )
    return {
        "schema_version": FLASHGS_PIPELINE_DEBUG_SCHEMA,
        "debug_only": True,
        "measured_timing_valid": False,
        "target_pixel_xy": [target_pixel_x, target_pixel_y],
        "target_tile_xy": [target_pixel_x // 16, target_pixel_y // 16],
        "expected_semantic_id_filter": expected_semantic_id,
        "candidate_discovery": {
            "mode": "explicit-only",
            "matched": len(ids),
            "selected": len(ids),
            "truncated": 0,
            "gaussians_scored": 0,
            "valid_intersections_scanned_per_candidate": generated_intersections,
        },
        "explicit_gaussian_ids": sorted(ids),
        "projection_replay_crosscheck": {
            "checked_records": len(replay_checks),
            "all_bit_exact": all(replay_checks) if replay_checks else None,
        },
        "discovery_score_crosscheck": {
            "checked_records": 0,
            "all_bit_exact": None,
            "not_applicable": True,
        },
        "cause_identified": bool(identified_causes),
        "identified_causes": identified_causes,
        "records": records,
        "native_build_contract": getattr(native, "__vgr_build_contract__", None),
    }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array.")
    return value


def _finite_float(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite.")
    return float(value)


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def load_b64_failure_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen five-case historical failure manifest."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != FLASHGS_B64_KNOWN_FAILURE_CASES_SCHEMA
        or payload.get("diagnostic_only") is not True
        or payload.get("batch") != 64
        or payload.get("step") != 107
        or payload.get("width") != 128
        or payload.get("height") != 128
        or payload.get("total_mismatched_pixels_across_regions") != 92
    ):
        raise ValueError("B64 known-failure manifest contract differs.")
    for field in ("scene_sha256", "trajectory_id"):
        if not _valid_sha256(payload.get(field)):
            raise ValueError(f"B64 manifest has an invalid {field}.")
    historical_artifacts = _require_mapping(
        payload.get("historical_artifacts"),
        "historical_artifacts",
    )
    oracle_record = _require_mapping(
        historical_artifacts.get("pinned_gsplat_oracle"),
        "pinned_gsplat_oracle",
    )
    historical_record = _require_mapping(
        historical_artifacts.get("flashgs_full_last_output"),
        "flashgs_full_last_output",
    )
    if not _valid_sha256(oracle_record.get("sha256")):
        raise ValueError("B64 manifest has no pinned oracle hash.")
    if not _valid_sha256(historical_record.get("sha256")):
        raise ValueError("B64 manifest has no historical FlashGS hash.")

    cases = _require_sequence(payload.get("cases"), "cases")
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("B64 failure manifest must contain exactly five cases.")
    case_ids: set[str] = set()
    camera_pixels: set[tuple[int, int, int]] = set()
    normalized_cases: list[dict[str, Any]] = []
    for raw_case in cases:
        case = dict(_require_mapping(raw_case, "case"))
        case_id = case.get("case_id")
        camera_index = case.get("camera_index")
        pixel_xy = case.get("pixel_xy")
        tile_xy = case.get("tile_xy")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("B64 case IDs must be unique non-empty strings.")
        if not isinstance(camera_index, int) or not 0 <= camera_index < 64:
            raise ValueError(f"B64 case {case_id} has an invalid camera.")
        pixel = _require_sequence(pixel_xy, f"{case_id}.pixel_xy")
        tile = _require_sequence(tile_xy, f"{case_id}.tile_xy")
        if (
            len(pixel) != 2
            or len(tile) != 2
            or not all(isinstance(value, int) for value in pixel)
            or not 0 <= int(pixel[0]) < 128
            or not 0 <= int(pixel[1]) < 128
            or list(tile) != [int(pixel[0]) // 16, int(pixel[1]) // 16]
        ):
            raise ValueError(f"B64 case {case_id} has invalid pixel geometry.")
        key = (camera_index, int(pixel[0]), int(pixel[1]))
        if key in camera_pixels:
            raise ValueError("B64 camera/pixel cases must be unique.")
        camera_pixels.add(key)
        _finite_float(case.get("oracle_alpha"), f"{case_id}.oracle_alpha")
        _finite_float(
            case.get("historical_flashgs_alpha"),
            f"{case_id}.historical_flashgs_alpha",
        )
        for field in ("oracle_semantic_id", "historical_flashgs_semantic_id"):
            if not isinstance(case.get(field), int):
                raise ValueError(f"B64 case {case_id} has invalid {field}.")
        case_ids.add(case_id)
        normalized_cases.append(case)

    return {
        "artifact": artifact_record(manifest_path),
        "payload": payload,
        "cases": sorted(normalized_cases, key=lambda item: item["case_id"]),
        "cases_by_id": {case["case_id"]: case for case in normalized_cases},
        "pinned_oracle_sha256": str(oracle_record["sha256"]),
        "historical_flashgs_sha256": str(historical_record["sha256"]),
    }


def _npz_scalar(archive: Any, key: str) -> str | None:
    if key not in archive.files:
        return None
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"NPZ metadata {key} must be scalar.")
    return str(value.item())


def _load_selected_full_output(
    path: Path,
    *,
    semantic_key: str,
) -> dict[str, Any]:
    if path.suffix != ".npz":
        raise ValueError(f"Selected renderer output must be NPZ: {path}.")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "rgb",
            "alpha",
            "depth",
            semantic_key,
            "camera_indices",
            "trajectory_id",
        }
        missing = required - set(archive.files)
        if missing or not ({"step", "steps"} & set(archive.files)):
            if not ({"step", "steps"} & set(archive.files)):
                missing.add("step-or-steps")
            raise ValueError(f"Selected output {path} lacks {sorted(missing)}.")
        rgb = np.asarray(archive["rgb"], dtype=np.float32)
        alpha = _squeeze_scalar_channel(np.asarray(archive["alpha"], dtype=np.float32), "alpha")
        depth = _squeeze_scalar_channel(np.asarray(archive["depth"], dtype=np.float32), "depth")
        semantic = _squeeze_scalar_channel(np.asarray(archive[semantic_key]), semantic_key)
        cameras = np.asarray(archive["camera_indices"], dtype=np.int64).reshape(-1)
        if "steps" in archive.files:
            steps = np.asarray(archive["steps"], dtype=np.int64).reshape(-1)
        else:
            scalar_step = np.asarray(archive["step"], dtype=np.int64)
            if scalar_step.shape != ():
                raise ValueError("Selected output scalar step is not scalar.")
            steps = np.full(
                cameras.shape,
                int(scalar_step.item()),
                dtype=np.int64,
            )
        trajectory_id = _npz_scalar(archive, "trajectory_id")
        semantic_topology = _npz_scalar(archive, "semantic_topology")
        camera_bundle_id = _npz_scalar(archive, "camera_bundle_id")
    if (
        rgb.ndim != 4
        or rgb.shape[-1] != 3
        or alpha.shape != depth.shape
        or alpha.shape != semantic.shape
        or rgb.shape[:3] != alpha.shape
        or alpha.shape[0] != cameras.size
        or alpha.shape[0] != steps.size
        or not np.isfinite(rgb).all()
        or not np.isfinite(alpha).all()
    ):
        raise ValueError(f"Selected output tensor contract differs: {path}.")
    return {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "semantic": semantic,
        "camera_indices": cameras,
        "steps": steps,
        "trajectory_id": trajectory_id,
        "semantic_topology": semantic_topology,
        "camera_bundle_id": camera_bundle_id,
    }


def _load_historical_full_batch_output(path: Path) -> dict[str, np.ndarray]:
    if path.suffix != ".npz":
        raise ValueError("Historical FlashGS full-batch output must be NPZ.")
    with np.load(path, allow_pickle=False) as archive:
        required = {"rgb", "alpha", "depth", "semantic_id"}
        if not required.issubset(archive.files):
            raise ValueError("Historical FlashGS output lacks full-sensor arrays.")
        rgb = np.asarray(archive["rgb"], dtype=np.float32)
        alpha = _squeeze_scalar_channel(np.asarray(archive["alpha"], dtype=np.float32), "alpha")
        depth = _squeeze_scalar_channel(np.asarray(archive["depth"], dtype=np.float32), "depth")
        semantic = _squeeze_scalar_channel(np.asarray(archive["semantic_id"]), "semantic_id")
    if (
        rgb.shape != (64, 128, 128, 3)
        or alpha.shape != (64, 128, 128)
        or depth.shape != alpha.shape
        or semantic.shape != alpha.shape
        or not np.isfinite(rgb).all()
        or not np.isfinite(alpha).all()
    ):
        raise ValueError("Historical FlashGS full-batch tensor contract differs.")
    return {"rgb": rgb, "alpha": alpha, "depth": depth, "semantic": semantic}


def _load_historical_run_binding(
    path: Path,
    *,
    known: Mapping[str, Any],
    historical_output_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera = _require_mapping(payload.get("camera_contract"), "camera_contract")
    equation = _require_mapping(payload.get("equation_contract"), "equation_contract")
    if (
        payload.get("schema_version") != PRE_FIX_RENDERER_RUN_SCHEMA
        or payload.get("renderer") != "flashgs"
        or payload.get("output_contract") != "full"
        or camera.get("trajectory_id") != known["payload"]["trajectory_id"]
        or camera.get("batch") != known["payload"]["batch"]
        or camera.get("width") != known["payload"]["width"]
        or camera.get("height") != known["payload"]["height"]
        or equation.get("semantic_topology") is None
        or not same_artifact(
            payload.get("last_output_capture"),
            dict(historical_output_artifact),
        )
        or _require_mapping(payload.get("runner_config", {}), "runner_config").get("capture_last_output", True)
        is not True
    ):
        raise ValueError("Historical run JSON does not bind the B64 output.")
    return payload


def build_b64_mismatch_corpus(
    *,
    historical_run: str | Path,
    historical_flashgs_output: str | Path,
    pinned_oracle_output: str | Path,
    camera_bundle_path: str | Path,
    known_failure_manifest: str | Path,
    semantic_topology: str,
) -> dict[str, Any]:
    """Derive and freeze the exact 92 historical semantic mismatches."""

    known = load_b64_failure_manifest(known_failure_manifest)
    historical_path = Path(historical_flashgs_output).resolve()
    run_path = Path(historical_run).resolve()
    oracle_path = Path(pinned_oracle_output).resolve()
    bundle_path = Path(camera_bundle_path).resolve()
    historical_artifact = artifact_record(historical_path)
    run_artifact = artifact_record(run_path)
    oracle_artifact = artifact_record(oracle_path)
    bundle_artifact = artifact_record(bundle_path)
    if historical_artifact["sha256"] != known["historical_flashgs_sha256"]:
        raise ValueError("Historical FlashGS output hash differs.")
    if oracle_artifact["sha256"] != known["pinned_oracle_sha256"]:
        raise ValueError("Pinned gsplat oracle hash differs.")
    historical_run_payload = _load_historical_run_binding(
        run_path,
        known=known,
        historical_output_artifact=historical_artifact,
    )
    historical = _load_historical_full_batch_output(historical_path)
    oracle = _load_selected_full_output(oracle_path, semantic_key="semantic")
    bundle = load_camera_bundle(bundle_path)
    oracle_pairs = list(
        zip(
            oracle["steps"].tolist(),
            oracle["camera_indices"].tolist(),
            strict=True,
        )
    )
    expected_pairs = [(known["payload"]["step"], camera) for camera in (0, 9, 18, 27, 36, 45, 54, 63)]
    if (
        oracle_pairs != expected_pairs
        or oracle["trajectory_id"] != known["payload"]["trajectory_id"]
        or oracle["alpha"].shape[1:] != (known["payload"]["height"], known["payload"]["width"])
        or oracle["semantic_topology"] != semantic_topology
        or bundle.width != known["payload"]["width"]
        or bundle.height != known["payload"]["height"]
        or len(bundle.cameras) != len(oracle_pairs)
        or bundle.scene_checksum != known["payload"]["scene_sha256"]
        or oracle["camera_bundle_id"] != bundle.bundle_id
        or _require_mapping(
            historical_run_payload.get("equation_contract"),
            "equation_contract",
        ).get("semantic_topology")
        != semantic_topology
    ):
        raise ValueError("Historical/oracle/camera-bundle contracts differ.")
    historical_selected_semantic = np.stack([historical["semantic"][int(camera)] for _step, camera in oracle_pairs])
    mismatch_mask = historical_selected_semantic != oracle["semantic"]
    per_view_counts = tuple(
        int(np.count_nonzero(mismatch_mask[view_index])) for view_index in range(mismatch_mask.shape[0])
    )
    mismatch_count = int(np.count_nonzero(mismatch_mask))
    if mismatch_count != EXPECTED_MISMATCH_COUNT or per_view_counts != EXPECTED_SELECTED_VIEW_MISMATCH_COUNTS:
        raise ValueError(
            f"Historical semantic mismatch mask differs: total={mismatch_count}, per_view={per_view_counts}."
        )
    pixels: list[dict[str, Any]] = []
    for view_index, (step, camera_index) in enumerate(oracle_pairs):
        coordinates = np.argwhere(mismatch_mask[view_index])
        for y_value, x_value in coordinates.tolist():
            x = int(x_value)
            y = int(y_value)
            oracle_depth = float(oracle["depth"][view_index, y, x])
            if not all(
                math.isfinite(value)
                for value in (
                    oracle_depth,
                    *oracle["rgb"][view_index, y, x].tolist(),
                )
            ):
                raise ValueError("Mismatch corpus contains non-finite values.")
            pixels.append(
                {
                    "view_index": view_index,
                    "step": int(step),
                    "camera_index": int(camera_index),
                    "pixel_xy": [x, y],
                    "historical_flashgs": {
                        "alpha": float(historical["alpha"][int(camera_index), y, x]),
                        "semantic_id": int(historical["semantic"][int(camera_index), y, x]),
                    },
                    "oracle": {
                        "rgb": [float(value) for value in oracle["rgb"][view_index, y, x]],
                        "alpha": float(oracle["alpha"][view_index, y, x]),
                        "depth": oracle_depth,
                        "semantic_id": int(oracle["semantic"][view_index, y, x]),
                    },
                }
            )
    coordinates = [
        {
            "step": pixel["step"],
            "camera_index": pixel["camera_index"],
            "pixel_xy": pixel["pixel_xy"],
        }
        for pixel in pixels
    ]
    oracle_values = [
        {"coordinate": coordinate, "oracle": pixel["oracle"]}
        for coordinate, pixel in zip(coordinates, pixels, strict=True)
    ]
    pixels_by_coordinate = {
        (
            int(pixel["camera_index"]),
            int(pixel["pixel_xy"][0]),
            int(pixel["pixel_xy"][1]),
        ): pixel
        for pixel in pixels
    }
    for case in known["cases"]:
        key = (
            int(case["camera_index"]),
            int(case["pixel_xy"][0]),
            int(case["pixel_xy"][1]),
        )
        pixel = pixels_by_coordinate.get(key)
        if (
            pixel is None
            or pixel["oracle"]["alpha"] != case["oracle_alpha"]
            or pixel["oracle"]["semantic_id"] != case["oracle_semantic_id"]
            or pixel["historical_flashgs"]["alpha"] != case["historical_flashgs_alpha"]
            or pixel["historical_flashgs"]["semantic_id"] != case["historical_flashgs_semantic_id"]
        ):
            raise ValueError(f"Mismatch corpus does not bind representative {case['case_id']}.")
    return {
        "historical_run": run_artifact,
        "historical_flashgs_output": historical_artifact,
        "pinned_oracle_output": oracle_artifact,
        "camera_bundle": bundle_artifact,
        "camera_bundle_id": bundle.bundle_id,
        "trajectory_id": known["payload"]["trajectory_id"],
        "semantic_topology": semantic_topology,
        "selection_pairs": [[int(step), int(camera)] for step, camera in oracle_pairs],
        "per_view_mismatch_counts": list(per_view_counts),
        "mismatch_count": mismatch_count,
        "coordinate_sha256": sha256_json(coordinates),
        "oracle_values_sha256": sha256_json(oracle_values),
        "pixels": pixels,
    }


def _diagnosis_case_key(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    camera = _require_mapping(payload.get("camera"), "diagnosis.camera")
    pixel = _require_sequence(camera.get("target_pixel_xy"), "target_pixel_xy")
    if len(pixel) != 2:
        raise ValueError("Diagnosis target pixel must contain two coordinates.")
    return (
        int(camera.get("camera_index", -1)),
        int(pixel[0]),
        int(pixel[1]),
    )


def _select_case_culprits(
    diagnosis: Mapping[str, Any],
    case: Mapping[str, Any],
) -> list[dict[str, Any]]:
    trace = _require_mapping(diagnosis.get("trace"), "diagnosis.trace")
    records = _require_sequence(trace.get("records"), "trace.records")
    culprits: list[dict[str, Any]] = []
    for raw_record in records:
        record = _require_mapping(raw_record, "trace record")
        compositor = _require_mapping(record.get("compositor"), "record.compositor")
        initial_load = _require_mapping(compositor.get("initial_load"), "compositor.initial_load")
        feature_evidence = _require_mapping(
            compositor.get("feature_load_evidence"),
            "compositor.feature_load_evidence",
        )
        enumeration = _require_mapping(
            record.get("enumeration_sort_range"),
            "record.enumeration_sort_range",
        )
        if (
            record.get("first_loss_stage") != "optimized-compositor-feature-load"
            or record.get("semantic_id") != case["oracle_semantic_id"]
            or initial_load.get("zero_offset_fallback") is not True
            or compositor.get("feature_gaussian_id") != 0
            or compositor.get("feature_load_offset") != 0
            or compositor.get("feature_source_matches_sorted_gaussian") is not False
            or feature_evidence.get("kind") != "source-grounded-debug-model"
            or feature_evidence.get("production_lane_registers_observed") is not False
        ):
            continue

        gaussian_id = record.get("gaussian_id")
        range_start = enumeration.get("range_start")
        range_end = enumeration.get("range_end")
        sorted_position = enumeration.get("sorted_position")
        if not all(isinstance(value, int) for value in (gaussian_id, range_start, range_end, sorted_position)):
            raise ValueError("Culprit ID/range fields must be integers.")
        range_length = int(range_end) - int(range_start)
        relative_slot = int(sorted_position) - int(range_start)
        expected_fault = (range_length, relative_slot) in ((1, 0), (2, 1))
        if (
            not expected_fault
            or initial_load.get("pair_slot") != relative_slot
            or initial_load.get("offset_assignment_guard") is not False
            or initial_load.get("load_enable_guard") is not True
            or enumeration.get("candidate_in_range_count") != 1
        ):
            raise ValueError("Optimized-load culprit does not use the one/two-entry fault.")
        target = _require_mapping(record.get("target_pixel"), "record.target_pixel")
        alpha = _finite_float(
            target.get("reprojected_alpha"),
            "culprit target alpha",
        )
        if alpha <= 0.0:
            raise ValueError("Culprit target alpha must be positive.")
        discovery = _require_mapping(record.get("discovery"), "culprit discovery")
        if discovery.get("gaussian_id") != gaussian_id:
            raise ValueError("Culprit discovery ID differs from its record ID.")
        if float(discovery.get("individual_alpha", float("nan"))) != alpha:
            raise ValueError("Culprit discovery alpha is not bit-exact.")
        logical_workspace = _require_mapping(
            compositor.get("logical_feature_workspace"),
            "logical_feature_workspace",
        )
        loaded_workspace = _require_mapping(
            compositor.get("loaded_feature_workspace"),
            "loaded_feature_workspace",
        )
        if logical_workspace == loaded_workspace:
            raise ValueError("Pre-fix culprit did not load different features.")
        if compositor.get("branch") == "contributed":
            raise ValueError("Pre-fix fallback unexpectedly contributed.")
        culprits.append(
            {
                "gaussian_id": int(gaussian_id),
                "semantic_id": int(record["semantic_id"]),
                "discovery_rank": int(discovery.get("rank", 0)),
                "target_reprojected_power": _finite_float(
                    target.get("reprojected_power"),
                    "culprit target power",
                ),
                "target_reprojected_alpha": alpha,
                "range_start": int(range_start),
                "range_end": int(range_end),
                "range_length": range_length,
                "sorted_position": int(sorted_position),
                "range_relative_slot": relative_slot,
                "pre_fix_feature_source_gaussian_id": 0,
                "pre_fix_feature_load_offset": 0,
                "pre_fix_offset_assignment_guard": False,
                "pre_fix_load_enable_guard": True,
                "pre_fix_zero_offset_fallback": True,
                "pre_fix_compositor_branch": compositor.get("branch"),
                "logical_feature_workspace": dict(logical_workspace),
                "pre_fix_loaded_feature_workspace": dict(loaded_workspace),
            }
        )
    if not culprits:
        raise ValueError(f"Diagnosis for {case['case_id']} has no short-tail culprit.")
    ids = [culprit["gaussian_id"] for culprit in culprits]
    if len(ids) != len(set(ids)):
        raise ValueError("Diagnosis contains duplicate culprit Gaussian IDs.")
    return sorted(culprits, key=lambda item: item["discovery_rank"])


def load_b64_diagnosis_lock(
    path: str | Path,
    *,
    artifact_root: str | Path,
    known_failure_manifest: str | Path,
) -> dict[str, Any]:
    """Resolve and hash-check the independently committed pre-fix lock."""

    lock_path = Path(path).resolve()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FLASHGS_B64_DIAGNOSIS_LOCK_SCHEMA:
        raise ValueError("B64 diagnosis lock schema differs.")
    known_record = _require_mapping(payload.get("known_failure_manifest"), "lock known_failure_manifest")
    locked_known_path, _locked_known_artifact = _resolve_relative_artifact(
        known_record,
        artifact_root=artifact_root,
        label="lock known-failure manifest",
    )
    supplied_known = artifact_record(known_failure_manifest)
    if not same_artifact(known_record, supplied_known) or not same_artifact(
        artifact_record(locked_known_path), supplied_known
    ):
        raise ValueError("Diagnosis lock known-failure manifest differs.")
    source_identity_record = _require_mapping(payload.get("pre_fix_source_identity"), "pre_fix_source_identity")
    if (
        source_identity_record.get("dirty") is not False
        or not _valid_sha256(source_identity_record.get("manifest_sha256"))
        or not _valid_sha256(source_identity_record.get("source_tree_sha256"))
        or not isinstance(source_identity_record.get("head"), str)
        or len(str(source_identity_record.get("head"))) != 40
        or not _valid_sha256(source_identity_record.get("diff_sha256"))
    ):
        raise ValueError("Diagnosis lock source identity is invalid.")
    resolved_artifacts: dict[str, dict[str, Any]] = {}
    for field in (
        "adapter_attestation",
        "production_native_extension",
        "debug_native_extension",
    ):
        record = _require_mapping(payload.get(field), f"lock {field}")
        resolved_path, actual = _resolve_relative_artifact(
            record,
            artifact_root=artifact_root,
            label=f"lock {field}",
        )
        resolved_artifacts[field] = {
            "record": dict(record),
            "resolved_path": str(resolved_path),
            "actual": actual,
        }
    cases = _require_sequence(payload.get("cases"), "diagnosis lock cases")
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("Diagnosis lock must contain exactly five cases.")
    cases_by_id: dict[str, dict[str, Any]] = {}
    for raw_case in cases:
        case = dict(_require_mapping(raw_case, "diagnosis lock case"))
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in cases_by_id:
            raise ValueError("Diagnosis lock case IDs are invalid or duplicate.")
        diagnosis_record = _require_mapping(case.get("diagnosis"), "lock diagnosis artifact")
        diagnosis_path, diagnosis_actual = _resolve_relative_artifact(
            diagnosis_record,
            artifact_root=artifact_root,
            label=f"lock diagnosis {case_id}",
        )
        culprits = _require_sequence(case.get("culprits"), "lock culprits")
        if not culprits:
            raise ValueError(f"Diagnosis lock case {case_id} has no culprits.")
        normalized_culprits: list[dict[str, int]] = []
        for raw_culprit in culprits:
            culprit = _require_mapping(raw_culprit, "lock culprit")
            fields = (
                "gaussian_id",
                "range_start",
                "range_end",
                "sorted_position",
                "range_relative_slot",
            )
            if not all(isinstance(culprit.get(field), int) for field in fields):
                raise ValueError("Diagnosis lock culprit fields must be integers.")
            normalized_culprits.append({field: int(culprit[field]) for field in fields})
        case["diagnosis"] = dict(diagnosis_record)
        case["resolved_diagnosis_path"] = str(diagnosis_path)
        case["resolved_diagnosis_artifact"] = diagnosis_actual
        case["culprits"] = sorted(normalized_culprits, key=lambda item: item["gaussian_id"])
        cases_by_id[case_id] = case
    return {
        "artifact": _relative_artifact_record(lock_path, artifact_root=artifact_root),
        "payload": payload,
        "source_identity": dict(source_identity_record),
        "resolved_artifacts": resolved_artifacts,
        "cases_by_id": cases_by_id,
    }


def build_b64_diagnosis_index(
    known_failure_manifest: str | Path,
    diagnosis_paths: Iterable[str | Path],
    *,
    diagnosis_lock: str | Path,
    artifact_root: str | Path,
    historical_run: str | Path,
    historical_flashgs_output: str | Path,
    pinned_oracle_output: str | Path,
    camera_bundle_path: str | Path,
    semantic_topology: str,
    created_at: str,
) -> dict[str, Any]:
    """Validate five pre-fix traces and build their immutable culprit index."""

    if not isinstance(created_at, str) or not created_at:
        raise ValueError("Diagnosis index requires a creation timestamp.")
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Diagnosis index timestamp is invalid.") from error
    known = load_b64_failure_manifest(known_failure_manifest)
    root = _resolved_artifact_root(artifact_root)
    lock = load_b64_diagnosis_lock(
        diagnosis_lock,
        artifact_root=root,
        known_failure_manifest=known_failure_manifest,
    )
    mismatch_corpus = build_b64_mismatch_corpus(
        historical_run=historical_run,
        historical_flashgs_output=historical_flashgs_output,
        pinned_oracle_output=pinned_oracle_output,
        camera_bundle_path=camera_bundle_path,
        known_failure_manifest=known_failure_manifest,
        semantic_topology=semantic_topology,
    )
    mismatch_corpus.update(
        {
            "historical_run": _relative_artifact_record(historical_run, artifact_root=root),
            "historical_flashgs_output": _relative_artifact_record(historical_flashgs_output, artifact_root=root),
            "pinned_oracle_output": _relative_artifact_record(pinned_oracle_output, artifact_root=root),
            "camera_bundle": _relative_artifact_record(camera_bundle_path, artifact_root=root),
        }
    )
    paths = [Path(path).resolve() for path in diagnosis_paths]
    if len(paths) != EXPECTED_CASE_COUNT or len(set(paths)) != len(paths):
        raise ValueError("Exactly five unique diagnosis artifacts are required.")

    known_by_key = {
        (
            int(case["camera_index"]),
            int(case["pixel_xy"][0]),
            int(case["pixel_xy"][1]),
        ): case
        for case in known["cases"]
    }
    indexed_cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    diagnosis_source_identity: dict[str, Any] | None = None
    diagnosis_hashes: set[str] = set()
    for path in paths:
        artifact = artifact_record(path)
        if artifact["sha256"] in diagnosis_hashes:
            raise ValueError("Diagnosis artifact content hashes must be unique.")
        diagnosis_hashes.add(str(artifact["sha256"]))
        diagnosis = json.loads(path.read_text(encoding="utf-8"))
        trace = _require_mapping(diagnosis.get("trace"), "diagnosis.trace")
        tool_integrity = _require_mapping(diagnosis.get("tool_integrity"), "diagnosis.tool_integrity")
        checks = _require_mapping(tool_integrity.get("checks"), "diagnosis integrity checks")
        capacity = _require_mapping(diagnosis.get("capacity"), "diagnosis.capacity")
        counters = _require_mapping(capacity.get("counters"), "diagnosis capacity counters")
        if (
            diagnosis.get("schema_version") != PRE_FIX_PIPELINE_DEBUG_SCHEMA
            or trace.get("schema_version") != PRE_FIX_PIPELINE_DEBUG_SCHEMA
            or diagnosis.get("debug_only") is not True
            or diagnosis.get("measured_timing_valid") is not False
            or tool_integrity.get("pass") is not True
            or not checks
            or not all(value is True for value in checks.values())
            or checks.get("historical_failure_reproduced") is not True
            or diagnosis.get("cause_identified") is not True
            or diagnosis.get("diagnosis_complete") is not True
            or trace.get("cause_identified") is not True
            or counters.get("intersection_overflow") not in (False, 0)
        ):
            raise ValueError("Pre-fix diagnosis is not complete and fail-closed.")
        key = _diagnosis_case_key(diagnosis)
        case = known_by_key.get(key)
        if case is None or case["case_id"] in seen_case_ids:
            raise ValueError("Diagnosis does not map uniquely to a frozen case.")
        locked_case = lock["cases_by_id"].get(case["case_id"])
        if locked_case is None or not same_artifact(locked_case["diagnosis"], artifact):
            raise ValueError("Diagnosis artifact is not pinned by the lock.")
        camera = _require_mapping(diagnosis.get("camera"), "diagnosis.camera")
        bound_case = _require_mapping(
            camera.get("historical_failure_case"),
            "historical_failure_case",
        )
        if (
            camera.get("trajectory_id") != known["payload"]["trajectory_id"]
            or camera.get("step") != known["payload"]["step"]
            or bound_case.get("case") != case
            or not same_artifact(bound_case.get("manifest"), known["artifact"])
        ):
            raise ValueError("Diagnosis is not bound to the frozen case manifest.")
        pixel = _require_mapping(
            diagnosis.get("rendered_target_pixel"),
            "rendered_target_pixel",
        )
        if (
            pixel.get("alpha") != case["historical_flashgs_alpha"]
            or pixel.get("semantic_id") != case["historical_flashgs_semantic_id"]
        ):
            raise ValueError("Diagnosis did not reproduce the historical pixel.")
        provenance = _require_mapping(diagnosis.get("source_provenance"), "source_provenance")
        identity = source_identity(dict(provenance))
        if identity.get("dirty") is not False:
            raise ValueError("Diagnosis source provenance is not clean.")
        if diagnosis_source_identity is None:
            diagnosis_source_identity = identity
        elif diagnosis_source_identity != identity:
            raise ValueError("All diagnosis artifacts must share one source.")
        if identity != lock["source_identity"]:
            raise ValueError("Diagnosis source identity differs from the lock.")
        diagnosis_attestation = _require_mapping(
            _require_mapping(
                diagnosis.get("flashgs_adapter_attestation"),
                "diagnosis flashgs_adapter_attestation",
            ).get("attestation"),
            "diagnosis adapter attestation artifact",
        )
        if not same_artifact(
            diagnosis_attestation,
            lock["resolved_artifacts"]["adapter_attestation"]["record"],
        ):
            raise ValueError("Diagnosis adapter attestation differs from lock.")
        production_adapter = _require_mapping(diagnosis.get("production_adapter"), "production_adapter")
        production_native = _require_mapping(
            production_adapter.get("native_extension"),
            "production native extension",
        )
        if not same_artifact(
            production_native,
            lock["resolved_artifacts"]["production_native_extension"]["record"],
        ):
            raise ValueError("Diagnosis production binary differs from lock.")
        debug_contract = _require_mapping(trace.get("native_build_contract"), "debug native build contract")
        debug_module_name = debug_contract.get("module_name")
        locked_debug_name = Path(lock["resolved_artifacts"]["debug_native_extension"]["resolved_path"]).name
        if not isinstance(debug_module_name, str) or not locked_debug_name.startswith(debug_module_name):
            raise ValueError("Diagnosis debug binary differs from lock.")
        culprits = _select_case_culprits(diagnosis, case)
        locked_culprits = sorted(locked_case["culprits"], key=lambda item: item["gaussian_id"])
        indexed_culprit_specs = sorted(
            [
                {
                    field: culprit[field]
                    for field in (
                        "gaussian_id",
                        "range_start",
                        "range_end",
                        "sorted_position",
                        "range_relative_slot",
                    )
                }
                for culprit in culprits
            ],
            key=lambda item: item["gaussian_id"],
        )
        if indexed_culprit_specs != locked_culprits:
            raise ValueError("Diagnosis culprits differ from the lock.")
        indexed_cases.append(
            {
                "case_id": case["case_id"],
                "camera_index": case["camera_index"],
                "pixel_xy": case["pixel_xy"],
                "tile_xy": case["tile_xy"],
                "oracle_alpha": case["oracle_alpha"],
                "oracle_semantic_id": case["oracle_semantic_id"],
                "historical_flashgs_alpha": case["historical_flashgs_alpha"],
                "historical_flashgs_semantic_id": case["historical_flashgs_semantic_id"],
                "diagnosis_artifact": _relative_artifact_record(path, artifact_root=root),
                "culprits": culprits,
            }
        )
        seen_case_ids.add(str(case["case_id"]))
    if seen_case_ids != set(known["cases_by_id"]):
        raise ValueError("Diagnosis artifacts do not cover all five cases.")
    culprit_count = sum(len(case["culprits"]) for case in indexed_cases)
    return {
        "schema_version": FLASHGS_B64_DIAGNOSIS_INDEX_SCHEMA,
        "created_at": created_at,
        "diagnostic_only": True,
        "pass": True,
        "case_count": EXPECTED_CASE_COUNT,
        "culprit_count": culprit_count,
        "artifact_path_contract": "relative-to-explicit-artifact-root",
        "known_failure_manifest": _relative_artifact_record(known_failure_manifest, artifact_root=root),
        "diagnosis_lock": lock["artifact"],
        "pinned_oracle_sha256": known["pinned_oracle_sha256"],
        "diagnosis_source_identity": diagnosis_source_identity,
        "historical_mismatch_corpus": mismatch_corpus,
        "cases": sorted(indexed_cases, key=lambda item: item["case_id"]),
    }


def load_verified_b64_diagnosis_index(
    index_path: str | Path,
    *,
    known_failure_manifest: str | Path,
    diagnosis_lock: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Verify an index, all five source artifacts, and its frozen manifest."""

    resolved = Path(index_path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != FLASHGS_B64_DIAGNOSIS_INDEX_SCHEMA
        or payload.get("diagnostic_only") is not True
        or payload.get("pass") is not True
        or payload.get("case_count") != EXPECTED_CASE_COUNT
    ):
        raise ValueError("B64 diagnosis index contract differs.")
    known = load_b64_failure_manifest(known_failure_manifest)
    root = _resolved_artifact_root(artifact_root)
    lock = load_b64_diagnosis_lock(
        diagnosis_lock,
        artifact_root=root,
        known_failure_manifest=known_failure_manifest,
    )
    if (
        not same_artifact(payload.get("known_failure_manifest"), known["artifact"])
        or payload.get("artifact_path_contract") != "relative-to-explicit-artifact-root"
        or not same_artifact(payload.get("diagnosis_lock"), lock["artifact"])
        or payload.get("pinned_oracle_sha256") != known["pinned_oracle_sha256"]
    ):
        raise ValueError("Diagnosis index is not bound to the frozen manifest.")
    cases = _require_sequence(payload.get("cases"), "diagnosis index cases")
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("Diagnosis index must contain exactly five cases.")
    diagnosis_paths: list[Path] = []
    case_ids: set[str] = set()
    for raw_case in cases:
        case = _require_mapping(raw_case, "diagnosis index case")
        case_id = case.get("case_id")
        if case_id not in known["cases_by_id"] or case_id in case_ids:
            raise ValueError("Diagnosis index case IDs are incomplete or duplicate.")
        expected_case = known["cases_by_id"][case_id]
        for field in (
            "camera_index",
            "pixel_xy",
            "tile_xy",
            "oracle_alpha",
            "oracle_semantic_id",
            "historical_flashgs_alpha",
            "historical_flashgs_semantic_id",
        ):
            if case.get(field) != expected_case.get(field):
                raise ValueError(f"Diagnosis index case {case_id} changed {field}.")
        diagnosis_record = _require_mapping(case.get("diagnosis_artifact"), "diagnosis_artifact")
        diagnosis_path, _diagnosis_actual = _resolve_relative_artifact(
            diagnosis_record,
            artifact_root=root,
            label=f"diagnosis artifact {case_id}",
        )
        culprits = _require_sequence(case.get("culprits"), "culprits")
        if not culprits:
            raise ValueError(f"Diagnosis index case {case_id} has no culprits.")
        diagnosis_paths.append(diagnosis_path)
        case_ids.add(str(case_id))
    if case_ids != set(known["cases_by_id"]):
        raise ValueError("Diagnosis index does not cover all frozen cases.")
    rebuilt = build_b64_diagnosis_index(
        known_failure_manifest,
        diagnosis_paths,
        diagnosis_lock=diagnosis_lock,
        artifact_root=root,
        historical_run=Path(
            _resolve_relative_artifact(
                _require_mapping(
                    _require_mapping(
                        payload.get("historical_mismatch_corpus"),
                        "historical_mismatch_corpus",
                    ).get("historical_run"),
                    "historical_run",
                ),
                artifact_root=root,
                label="historical run",
            )[0]
        ),
        historical_flashgs_output=Path(
            _resolve_relative_artifact(
                _require_mapping(
                    _require_mapping(
                        payload.get("historical_mismatch_corpus"),
                        "historical_mismatch_corpus",
                    ).get("historical_flashgs_output"),
                    "historical_flashgs_output",
                ),
                artifact_root=root,
                label="historical FlashGS output",
            )[0]
        ),
        pinned_oracle_output=Path(
            _resolve_relative_artifact(
                _require_mapping(
                    _require_mapping(
                        payload.get("historical_mismatch_corpus"),
                        "historical_mismatch_corpus",
                    ).get("pinned_oracle_output"),
                    "pinned_oracle_output",
                ),
                artifact_root=root,
                label="pinned oracle output",
            )[0]
        ),
        camera_bundle_path=Path(
            _resolve_relative_artifact(
                _require_mapping(
                    _require_mapping(
                        payload.get("historical_mismatch_corpus"),
                        "historical_mismatch_corpus",
                    ).get("camera_bundle"),
                    "camera_bundle",
                ),
                artifact_root=root,
                label="camera bundle",
            )[0]
        ),
        semantic_topology=str(
            _require_mapping(
                payload.get("historical_mismatch_corpus"),
                "historical_mismatch_corpus",
            ).get("semantic_topology", "")
        ),
        created_at=str(payload.get("created_at", "")),
    )
    for field in (
        "schema_version",
        "created_at",
        "diagnostic_only",
        "pass",
        "case_count",
        "culprit_count",
        "artifact_path_contract",
        "pinned_oracle_sha256",
        "diagnosis_source_identity",
        "historical_mismatch_corpus",
        "diagnosis_lock",
        "cases",
    ):
        if payload.get(field) != rebuilt.get(field):
            raise ValueError(f"Diagnosis index field {field} is not reproducible.")
    return {
        "artifact": artifact_record(resolved),
        "payload": payload,
        "cases_by_id": {case["case_id"]: case for case in cases},
    }


def _squeeze_scalar_channel(array: np.ndarray, label: str) -> np.ndarray:
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Oracle {label} must have shape [V,H,W].")
    return array


def load_bound_b64_oracle_pixels(
    oracle_path: str | Path,
    *,
    known_failure_manifest: str | Path,
    semantic_topology: str,
) -> dict[str, Any]:
    """Hash-bind the original oracle and extract all five frozen pixels."""

    known = load_b64_failure_manifest(known_failure_manifest)
    resolved = Path(oracle_path).resolve()
    record = artifact_record(resolved)
    if record["sha256"] != known["pinned_oracle_sha256"]:
        raise ValueError("Pinned gsplat oracle hash differs from the B64 manifest.")
    if resolved.suffix != ".npz":
        raise ValueError("The pinned B64 oracle must be an NPZ artifact.")
    with np.load(resolved, allow_pickle=False) as archive:
        required = {"alpha", "semantic", "camera_indices"}
        if not required.issubset(archive.files) or not ({"step", "steps"} & set(archive.files)):
            raise ValueError("Pinned oracle lacks alpha/semantic selection metadata.")
        alpha = _squeeze_scalar_channel(np.asarray(archive["alpha"], dtype=np.float32), "alpha")
        semantic = _squeeze_scalar_channel(np.asarray(archive["semantic"]), "semantic")
        camera_indices = np.asarray(archive["camera_indices"], dtype=np.int64).reshape(-1)
        if "steps" in archive.files:
            steps = np.asarray(archive["steps"], dtype=np.int64).reshape(-1)
        else:
            scalar_step = np.asarray(archive["step"], dtype=np.int64)
            if scalar_step.shape != ():
                raise ValueError("Pinned oracle scalar step is not scalar.")
            steps = np.full(
                camera_indices.shape,
                int(scalar_step.item()),
                dtype=np.int64,
            )
        trajectory_id = str(np.asarray(archive["trajectory_id"]).item()) if "trajectory_id" in archive.files else None
        archive_topology = (
            str(np.asarray(archive["semantic_topology"]).item()) if "semantic_topology" in archive.files else None
        )
    if (
        alpha.shape != semantic.shape
        or alpha.shape[0] != camera_indices.size
        or alpha.shape[0] != steps.size
        or alpha.shape[1:]
        != (
            known["payload"]["height"],
            known["payload"]["width"],
        )
        or trajectory_id != known["payload"]["trajectory_id"]
        or archive_topology != semantic_topology
    ):
        raise ValueError("Pinned oracle camera/output contract differs.")
    pair_to_view: dict[tuple[int, int], int] = {}
    for view_index, pair in enumerate(zip(steps.tolist(), camera_indices.tolist())):
        normalized_pair = (int(pair[0]), int(pair[1]))
        if normalized_pair in pair_to_view:
            raise ValueError("Pinned oracle has duplicate camera selections.")
        pair_to_view[normalized_pair] = view_index
    pixels: dict[str, dict[str, Any]] = {}
    for case in known["cases"]:
        pair = (known["payload"]["step"], int(case["camera_index"]))
        if pair not in pair_to_view:
            raise ValueError(f"Pinned oracle lacks {case['case_id']}.")
        view_index = pair_to_view[pair]
        x, y = (int(value) for value in case["pixel_xy"])
        oracle_alpha = float(alpha[view_index, y, x])
        oracle_semantic = int(semantic[view_index, y, x])
        if oracle_alpha != case["oracle_alpha"] or oracle_semantic != case["oracle_semantic_id"]:
            raise ValueError(f"Pinned oracle scalar differs for {case['case_id']}.")
        pixels[case["case_id"]] = {
            "view_index": view_index,
            "step": pair[0],
            "camera_index": pair[1],
            "pixel_xy": [x, y],
            "alpha": oracle_alpha,
            "semantic_id": oracle_semantic,
        }
    return {
        "artifact": record,
        "semantic_topology": semantic_topology,
        "trajectory_id": trajectory_id,
        "selection_pairs": [
            [int(step), int(camera)] for step, camera in zip(steps.tolist(), camera_indices.tolist(), strict=True)
        ],
        "pixels": pixels,
    }


def _workspace_bit_equal(
    logical: Any,
    loaded: Any,
) -> bool:
    """Compare the serialized float32 workspace without numeric coercion."""

    if type(logical) is not type(loaded):
        return False
    if isinstance(logical, Mapping):
        return list(logical) == list(loaded) and all(_workspace_bit_equal(logical[key], loaded[key]) for key in logical)
    if isinstance(logical, (list, tuple)):
        return len(logical) == len(loaded) and all(
            _workspace_bit_equal(left, right) for left, right in zip(logical, loaded, strict=True)
        )
    if isinstance(logical, float):
        left_bits = np.asarray(logical, dtype=np.float32).view(np.uint32).item()
        right_bits = np.asarray(loaded, dtype=np.float32).view(np.uint32).item()
        return bool(left_bits == right_bits)
    if isinstance(logical, np.float32):
        return bool(logical.view(np.uint32) == loaded.view(np.uint32))
    return False


def _float32_ulp_distance(left: Any, right: Any) -> int:
    """Return the ordered float32 ULP distance for two finite scalars."""

    left_float = np.float32(_finite_float(left, "left float32 ULP operand"))
    right_float = np.float32(_finite_float(right, "right float32 ULP operand"))

    def ordered(value: np.float32) -> int:
        bits = int(value.view(np.uint32).item())
        magnitude = bits & 0x7FFFFFFF
        return 0x80000000 - magnitude if bits & 0x80000000 else 0x80000000 + magnitude

    return abs(ordered(left_float) - ordered(right_float))


def evaluate_repaired_b64_case(
    *,
    frozen_case: Mapping[str, Any],
    indexed_case: Mapping[str, Any],
    oracle_pixel: Mapping[str, Any],
    rendered_pixel: Mapping[str, Any],
    trace: Mapping[str, Any],
    capacity_counters: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one post-fix trace against its frozen culprit and oracle."""

    case_id = str(frozen_case.get("case_id"))
    if indexed_case.get("case_id") != case_id:
        raise ValueError("Indexed case and frozen case IDs differ.")
    alpha = _finite_float(rendered_pixel.get("alpha"), "rendered alpha")
    oracle_alpha = _finite_float(oracle_pixel.get("alpha"), "oracle alpha")
    alpha_error = abs(alpha - oracle_alpha)
    historical_pair = alpha == frozen_case.get("historical_flashgs_alpha") and rendered_pixel.get(
        "semantic_id"
    ) == frozen_case.get("historical_flashgs_semantic_id")
    records = _require_sequence(trace.get("records"), "repair trace records")
    records_by_id: dict[int, Mapping[str, Any]] = {}
    for raw_record in records:
        record = _require_mapping(raw_record, "repair trace record")
        gaussian_id = record.get("gaussian_id")
        if isinstance(gaussian_id, int):
            if gaussian_id in records_by_id:
                raise ValueError("Repair trace contains duplicate Gaussian IDs.")
            records_by_id[gaussian_id] = record
    culprit_results: list[dict[str, Any]] = []
    culprits = _require_sequence(indexed_case.get("culprits"), "indexed culprits")
    for raw_culprit in culprits:
        culprit = _require_mapping(raw_culprit, "indexed culprit")
        gaussian_id = int(culprit.get("gaussian_id", -1))
        record = records_by_id.get(gaussian_id)
        if record is None:
            culprit_results.append(
                {
                    "gaussian_id": gaussian_id,
                    "checks": {"record_present": False},
                    "pass": False,
                }
            )
            continue
        compositor = _require_mapping(record.get("compositor"), "compositor")
        initial_load = _require_mapping(compositor.get("initial_load"), "initial_load")
        enumeration = _require_mapping(record.get("enumeration_sort_range"), "enumeration_sort_range")
        target = _require_mapping(record.get("target_pixel"), "target_pixel")
        projection = _require_mapping(record.get("projection"), "projection")
        retained = _require_mapping(
            projection.get("retained_workspace_crosscheck"),
            "retained_workspace_crosscheck",
        )
        logical_workspace = compositor.get("logical_feature_workspace")
        loaded_workspace = compositor.get("loaded_feature_workspace")
        feature_evidence = _require_mapping(
            compositor.get("feature_load_evidence"),
            "compositor.feature_load_evidence",
        )
        target_alpha_ulp_distance = _float32_ulp_distance(
            target.get("reprojected_alpha"),
            culprit.get("target_reprojected_alpha"),
        )
        target_power_ulp_distance = _float32_ulp_distance(
            target.get("reprojected_power"),
            culprit.get("target_reprojected_power"),
        )
        culprit_checks = {
            "record_present": True,
            "compositor_seen": compositor.get("seen") is True,
            "feature_evidence_is_source_grounded_model": feature_evidence.get("kind") == "source-grounded-debug-model",
            "production_lane_registers_not_claimed_observed": feature_evidence.get("production_lane_registers_observed")
            is False,
            "semantic_matches": record.get("semantic_id") == culprit.get("semantic_id"),
            "range_start_matches": enumeration.get("range_start") == culprit.get("range_start"),
            "range_end_matches": enumeration.get("range_end") == culprit.get("range_end"),
            "range_length_matches": enumeration.get("range_length") == culprit.get("range_length"),
            "sorted_position_matches": enumeration.get("sorted_position") == culprit.get("sorted_position"),
            "candidate_present_once": enumeration.get("candidate_in_range_count") == 1,
            "pair_slot_matches": initial_load.get("pair_slot") == culprit.get("range_relative_slot"),
            "offset_assignment_guard": initial_load.get("offset_assignment_guard") is True,
            "load_enable_guard": initial_load.get("load_enable_guard") is True,
            "no_zero_offset_fallback": initial_load.get("zero_offset_fallback") is False,
            "feature_source_matches": compositor.get("feature_source_matches_sorted_gaussian") is True,
            "feature_gaussian_id_matches": compositor.get("feature_gaussian_id") == gaussian_id,
            "feature_load_offset_matches": compositor.get("feature_load_offset") == gaussian_id,
            "workspace_bit_equal": _workspace_bit_equal(logical_workspace, loaded_workspace),
            "projection_workspace_replay_bit_exact": retained.get("bit_exact_values") is True,
            "projection_workspace_replay_available": retained.get("available") is True,
            "target_alpha_matches_historical_trace_ulp_bound": target_alpha_ulp_distance
            <= REPAIR_TRACE_FLOAT32_MAX_ULPS,
            "target_power_matches_historical_trace_ulp_bound": target_power_ulp_distance
            <= REPAIR_TRACE_FLOAT32_MAX_ULPS,
            "loaded_alpha_matches_logical": compositor.get("loaded_feature_target_alpha")
            == target.get("reprojected_alpha"),
            "loaded_power_matches_logical": compositor.get("loaded_feature_target_power")
            == target.get("reprojected_power"),
            "compositor_contributed": compositor.get("branch") == "contributed",
            "survived_compositor": record.get("first_loss_stage") == "survived-compositor",
        }
        culprit_results.append(
            {
                "gaussian_id": gaussian_id,
                "historical_trace_float32_consistency": {
                    "maximum_ulp_distance": REPAIR_TRACE_FLOAT32_MAX_ULPS,
                    "target_alpha_ulp_distance": target_alpha_ulp_distance,
                    "target_power_ulp_distance": target_power_ulp_distance,
                    "scope": "cross-architecture-debug-intermediates-only",
                },
                "checks": culprit_checks,
                "pass": all(culprit_checks.values()),
            }
        )
    explicit_ids = sorted(int(value) for value in trace.get("explicit_gaussian_ids", []))
    indexed_ids = sorted(int(culprit["gaussian_id"]) for culprit in culprits)
    culprit_compositors = [
        _require_mapping(records_by_id[gaussian_id].get("compositor"), "compositor")
        for gaussian_id in indexed_ids
        if gaussian_id in records_by_id
    ]
    projection_crosscheck = _require_mapping(
        trace.get("projection_replay_crosscheck"),
        "projection_replay_crosscheck",
    )
    discovery_crosscheck = _require_mapping(
        trace.get("discovery_score_crosscheck"),
        "discovery_score_crosscheck",
    )
    discovery = _require_mapping(trace.get("candidate_discovery"), "candidate_discovery")
    checks = {
        "trace_schema_is_current_v4": trace.get("schema_version") == FLASHGS_PIPELINE_DEBUG_SCHEMA,
        "trace_is_debug_only": trace.get("debug_only") is True,
        "trace_timing_is_invalid": trace.get("measured_timing_valid") is False,
        "oracle_scalar_matches_manifest": oracle_alpha == frozen_case.get("oracle_alpha")
        and oracle_pixel.get("semantic_id") == frozen_case.get("oracle_semantic_id"),
        "rendered_alpha_matches_oracle": alpha_error <= REPAIR_ALPHA_ATOL,
        "rendered_semantic_matches_oracle": rendered_pixel.get("semantic_id") == oracle_pixel.get("semantic_id"),
        "historical_pair_eliminated": not historical_pair,
        "zero_intersection_overflow": capacity_counters.get("intersection_overflow") in (False, 0),
        "trace_target_matches": trace.get("target_pixel_xy") == frozen_case.get("pixel_xy"),
        "trace_semantic_filter_matches": trace.get("expected_semantic_id_filter")
        == frozen_case.get("oracle_semantic_id"),
        "explicit_culprits_match": explicit_ids == indexed_ids,
        "candidate_discovery_mode_explicit_only": discovery.get("mode") == "explicit-only",
        "candidate_discovery_counts_match": discovery.get("matched") == len(indexed_ids)
        and discovery.get("selected") == len(indexed_ids)
        and discovery.get("truncated") == 0,
        "projection_replay_crosscheck": projection_crosscheck.get("all_bit_exact") is True,
        "discovery_score_crosscheck_not_applicable": discovery_crosscheck.get("not_applicable") is True
        and discovery_crosscheck.get("all_bit_exact") is None,
        "all_culprit_records_no_fallback": len(culprit_compositors) == len(indexed_ids)
        and all(
            _require_mapping(compositor.get("initial_load"), "initial_load").get("zero_offset_fallback") is False
            for compositor in culprit_compositors
        ),
        "all_culprit_records_source_match": len(culprit_compositors) == len(indexed_ids)
        and all(compositor.get("feature_source_matches_sorted_gaussian") is True for compositor in culprit_compositors),
        "all_culprits_pass": bool(culprit_results) and all(result["pass"] for result in culprit_results),
    }
    return {
        "case_id": case_id,
        "camera_index": frozen_case.get("camera_index"),
        "pixel_xy": frozen_case.get("pixel_xy"),
        "oracle_pixel": dict(oracle_pixel),
        "rendered_pixel": dict(rendered_pixel),
        "alpha_absolute_error": alpha_error,
        "alpha_absolute_tolerance": REPAIR_ALPHA_ATOL,
        "culprits": culprit_results,
        "checks": checks,
        "pass": all(checks.values()),
    }


def evaluate_repaired_mismatch_corpus(
    *,
    corpus: Mapping[str, Any],
    rendered_by_camera: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Validate RGB/alpha/depth/semantic at all 92 historical mismatches."""

    pixels = _require_sequence(corpus.get("pixels"), "mismatch corpus pixels")
    if (
        corpus.get("mismatch_count") != EXPECTED_MISMATCH_COUNT
        or len(pixels) != EXPECTED_MISMATCH_COUNT
        or tuple(corpus.get("per_view_mismatch_counts", ())) != EXPECTED_SELECTED_VIEW_MISMATCH_COUNTS
    ):
        raise ValueError("Historical mismatch corpus does not contain 92 pixels.")
    coordinates = [
        {
            "step": pixel["step"],
            "camera_index": pixel["camera_index"],
            "pixel_xy": pixel["pixel_xy"],
        }
        for pixel in pixels
    ]
    oracle_values = [
        {"coordinate": coordinate, "oracle": pixel["oracle"]}
        for coordinate, pixel in zip(coordinates, pixels, strict=True)
    ]
    if sha256_json(coordinates) != corpus.get("coordinate_sha256") or sha256_json(oracle_values) != corpus.get(
        "oracle_values_sha256"
    ):
        raise ValueError("Historical mismatch coordinate/value hashes differ.")

    validated_rendered: dict[int, Mapping[str, np.ndarray]] = {}
    for raw_camera_index, raw_rendered in rendered_by_camera.items():
        camera_index = int(raw_camera_index)
        rendered = _require_mapping(raw_rendered, "rendered camera outputs")
        arrays = {name: np.asarray(rendered.get(name)) for name in ("rgb", "alpha", "depth", "semantic_id")}
        expected_contracts = {
            "rgb": ((128, 128, 3), np.dtype(np.float32)),
            "alpha": ((128, 128, 1), np.dtype(np.float32)),
            "depth": ((128, 128, 1), np.dtype(np.float32)),
            "semantic_id": ((128, 128, 1), np.dtype(np.int64)),
        }
        for name, (shape, dtype) in expected_contracts.items():
            if arrays[name].shape != shape or arrays[name].dtype != dtype:
                raise ValueError(
                    f"Rendered {name} for camera {camera_index} must be "
                    f"{shape} {dtype}, got {arrays[name].shape} "
                    f"{arrays[name].dtype}."
                )
        if not np.isfinite(arrays["rgb"]).all() or not np.isfinite(arrays["alpha"]).all():
            raise ValueError("Rendered RGB/alpha values must all be finite.")
        validated_rendered[camera_index] = arrays

    pixel_results: list[dict[str, Any]] = []
    for frozen in pixels:
        camera_index = int(frozen["camera_index"])
        x, y = (int(value) for value in frozen["pixel_xy"])
        rendered = validated_rendered.get(camera_index)
        if rendered is None:
            pixel_results.append(
                {
                    "camera_index": camera_index,
                    "pixel_xy": [x, y],
                    "checks": {"camera_render_present": False},
                    "pass": False,
                }
            )
            continue
        rgb = rendered["rgb"]
        alpha = rendered["alpha"]
        depth = rendered["depth"]
        semantic = rendered["semantic_id"]
        oracle = _require_mapping(frozen.get("oracle"), "mismatch oracle")
        rendered_rgb = [float(value) for value in rgb[y, x].tolist()]
        oracle_rgb = [float(value) for value in oracle["rgb"]]
        rgb_errors = [abs(actual - expected) for actual, expected in zip(rendered_rgb, oracle_rgb, strict=True)]
        rendered_alpha = float(alpha[y, x, 0])
        rendered_depth = float(depth[y, x, 0])
        if not math.isfinite(rendered_depth):
            raise ValueError("Rendered depth at a frozen mismatch is not finite.")
        rendered_semantic = semantic[y, x, 0].item()
        alpha_error = abs(rendered_alpha - float(oracle["alpha"]))
        depth_error = abs(rendered_depth - float(oracle["depth"]))
        checks = {
            "camera_render_present": True,
            "rgb_matches_oracle": max(rgb_errors) <= REPAIR_RGB_ATOL,
            "alpha_matches_oracle": alpha_error <= REPAIR_ALPHA_ATOL,
            "depth_matches_oracle": depth_error <= REPAIR_DEPTH_ATOL,
            "semantic_matches_oracle": rendered_semantic == int(oracle["semantic_id"]),
        }
        pixel_results.append(
            {
                "camera_index": camera_index,
                "pixel_xy": [x, y],
                "rgb_max_absolute_error": max(rgb_errors),
                "alpha_absolute_error": alpha_error,
                "depth_absolute_error": depth_error,
                "rendered_semantic_id": rendered_semantic,
                "oracle_semantic_id": int(oracle["semantic_id"]),
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    checks = {
        "exactly_92_pixels": len(pixel_results) == EXPECTED_MISMATCH_COUNT,
        "all_pixels_pass": bool(pixel_results) and all(result["pass"] for result in pixel_results),
    }
    return {
        "pixel_count": len(pixel_results),
        "passed_pixel_count": sum(result["pass"] for result in pixel_results),
        "tolerances": {
            "rgb_absolute": REPAIR_RGB_ATOL,
            "alpha_absolute": REPAIR_ALPHA_ATOL,
            "depth_absolute": REPAIR_DEPTH_ATOL,
            "semantic": "exact",
        },
        "coordinate_sha256": corpus.get("coordinate_sha256"),
        "oracle_values_sha256": corpus.get("oracle_values_sha256"),
        "pixels": pixel_results,
        "checks": checks,
        "pass": all(checks.values()),
    }


def evaluate_repaired_rgb_only_corpus(
    *,
    corpus: Mapping[str, Any],
    rendered_rgb_by_camera: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """Validate the RGB-only compositor specialization at all 92 pixels."""

    pixels = _require_sequence(corpus.get("pixels"), "mismatch corpus pixels")
    if len(pixels) != EXPECTED_MISMATCH_COUNT:
        raise ValueError("RGB-only corpus does not contain exactly 92 pixels.")
    pixel_results: list[dict[str, Any]] = []
    for frozen in pixels:
        camera_index = int(frozen["camera_index"])
        x, y = (int(value) for value in frozen["pixel_xy"])
        rendered = rendered_rgb_by_camera.get(camera_index)
        if rendered is None:
            pixel_results.append(
                {
                    "camera_index": camera_index,
                    "pixel_xy": [x, y],
                    "rgb_max_absolute_error": None,
                    "pass": False,
                }
            )
            continue
        rgb = np.asarray(rendered)
        if rgb.shape != (128, 128, 3) or rgb.dtype != np.dtype(np.float32):
            raise ValueError("RGB-only output must be [128,128,3] float32.")
        if not np.isfinite(rgb).all():
            raise ValueError("RGB-only output values must all be finite.")
        oracle_rgb = [float(value) for value in frozen["oracle"]["rgb"]]
        rendered_rgb = [float(value) for value in rgb[y, x].tolist()]
        max_error = max(abs(actual - expected) for actual, expected in zip(rendered_rgb, oracle_rgb, strict=True))
        pixel_results.append(
            {
                "camera_index": camera_index,
                "pixel_xy": [x, y],
                "rgb_max_absolute_error": max_error,
                "pass": max_error <= REPAIR_RGB_ATOL,
            }
        )
    checks = {
        "exactly_92_pixels": len(pixel_results) == EXPECTED_MISMATCH_COUNT,
        "all_rgb_pixels_pass": all(result["pass"] for result in pixel_results),
    }
    return {
        "pixel_count": len(pixel_results),
        "passed_pixel_count": sum(result["pass"] for result in pixel_results),
        "rgb_absolute_tolerance": REPAIR_RGB_ATOL,
        "coordinate_sha256": corpus.get("coordinate_sha256"),
        "pixels": pixel_results,
        "checks": checks,
        "pass": all(checks.values()),
    }


def aggregate_b64_repair_cases(
    case_results: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Iterable[str],
) -> dict[str, Any]:
    """Produce the exact-five aggregate verdict used by the GPU verifier."""

    expected = set(expected_case_ids)
    actual_ids = [str(result.get("case_id")) for result in case_results]
    checks = {
        "exactly_five_results": len(case_results) == EXPECTED_CASE_COUNT,
        "unique_case_ids": len(actual_ids) == len(set(actual_ids)),
        "exact_case_coverage": set(actual_ids) == expected and len(expected) == EXPECTED_CASE_COUNT,
        "all_cases_pass": bool(case_results) and all(result.get("pass") is True for result in case_results),
    }
    return {
        "case_count": len(case_results),
        "passed_case_count": sum(result.get("pass") is True for result in case_results),
        "checks": checks,
        "pass": all(checks.values()),
    }


def load_verified_b64_repair_raw_outputs(
    report_path: str | Path,
    *,
    diagnosis_index: str | Path,
    diagnosis_lock: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Relocate, hash-check, and independently re-evaluate the B64 NPZs."""

    resolved_report = Path(report_path).resolve()
    report = json.loads(resolved_report.read_text(encoding="utf-8"))
    tool_integrity = _require_mapping(report.get("tool_integrity"), "repair report tool_integrity")
    if (
        report.get("schema_version") != FLASHGS_B64_REPAIR_VERIFICATION_SCHEMA
        or report.get("pass") is not True
        or tool_integrity.get("pass") is not True
    ):
        raise ValueError("Repair report did not pass its evidence contract.")

    root = _resolved_artifact_root(artifact_root)
    index_path = Path(diagnosis_index).resolve()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    known_path, _known_actual = _resolve_relative_artifact(
        _require_mapping(
            index_payload.get("known_failure_manifest"),
            "diagnosis index known_failure_manifest",
        ),
        artifact_root=root,
        label="diagnosis index known-failure manifest",
    )
    diagnosis = load_verified_b64_diagnosis_index(
        index_path,
        known_failure_manifest=known_path,
        diagnosis_lock=diagnosis_lock,
        artifact_root=root,
    )
    if not same_artifact(
        _require_mapping(report.get("diagnosis_index"), "report diagnosis_index"),
        artifact_record(index_path),
    ) or not same_artifact(
        _require_mapping(report.get("diagnosis_lock"), "report diagnosis_lock"),
        diagnosis["payload"]["diagnosis_lock"],
    ):
        raise ValueError("Repair report diagnosis binding differs.")

    corpus = _require_mapping(
        diagnosis["payload"].get("historical_mismatch_corpus"),
        "historical_mismatch_corpus",
    )
    camera_contract = _require_mapping(report.get("camera"), "report camera")
    equation = _require_mapping(report.get("equation"), "report equation")
    expected_trajectory_id = str(corpus.get("trajectory_id"))
    expected_topology = str(corpus.get("semantic_topology"))
    if (
        camera_contract.get("trajectory_id") != expected_trajectory_id
        or camera_contract.get("step") != 107
        or equation.get("semantic_topology") != expected_topology
    ):
        raise ValueError("Repair report camera/equation contract differs.")

    raw_records = _require_mapping(report.get("post_fix_raw_outputs"), "post_fix_raw_outputs")
    loaded: dict[str, dict[str, Any]] = {}
    for label, required_arrays in (
        ("full", {"rgb", "depth", "alpha", "semantic_id"}),
        ("rgb_only", {"rgb"}),
    ):
        record = _require_mapping(raw_records.get(label), f"raw {label}")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Post-fix raw {label} path is missing.")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Post-fix raw {label} path is not safe and relative.")
        path = (resolved_report.parent / relative).resolve()
        if not path.is_relative_to(resolved_report.parent):
            raise ValueError(f"Post-fix raw {label} path escapes report directory.")
        if not same_artifact(record, artifact_record(path)):
            raise ValueError(f"Post-fix raw {label} artifact hash differs.")
        with np.load(path, allow_pickle=False) as archive:
            metadata = {
                "camera_indices",
                "step",
                "trajectory_id",
                "semantic_topology",
            }
            if not (required_arrays | metadata).issubset(archive.files):
                raise ValueError(f"Post-fix raw {label} NPZ contract differs.")
            arrays = {name: np.asarray(archive[name]).copy() for name in required_arrays}
            cameras = np.asarray(archive["camera_indices"])
            step = np.asarray(archive["step"])
            trajectory_id = str(np.asarray(archive["trajectory_id"]).item())
            semantic_topology = str(np.asarray(archive["semantic_topology"]).item())
        if (
            cameras.shape != (64,)
            or cameras.dtype != np.dtype(np.int64)
            or not np.array_equal(cameras, np.arange(64, dtype=np.int64))
            or step.shape != ()
            or step.dtype != np.dtype(np.int64)
            or int(step.item()) != 107
            or trajectory_id != expected_trajectory_id
            or semantic_topology != expected_topology
            or arrays["rgb"].shape != (64, 128, 128, 3)
            or arrays["rgb"].dtype != np.dtype(np.float32)
            or not np.isfinite(arrays["rgb"]).all()
        ):
            raise ValueError(f"Post-fix raw {label} tensor contract differs.")
        if label == "full" and (
            arrays["depth"].shape != (64, 128, 128, 1)
            or arrays["depth"].dtype != np.dtype(np.float32)
            or not np.logical_or(np.isfinite(arrays["depth"]), np.isposinf(arrays["depth"])).all()
            or arrays["alpha"].shape != (64, 128, 128, 1)
            or arrays["alpha"].dtype != np.dtype(np.float32)
            or not np.isfinite(arrays["alpha"]).all()
            or arrays["semantic_id"].shape != (64, 128, 128, 1)
            or arrays["semantic_id"].dtype != np.dtype(np.int64)
        ):
            raise ValueError("Post-fix raw full-sensor tensor contract differs.")
        loaded[label] = {
            "artifact": dict(record),
            "arrays": arrays,
            "camera_indices": cameras.tolist(),
            "step": int(step.item()),
            "trajectory_id": trajectory_id,
            "semantic_topology": semantic_topology,
        }
    if any(
        loaded["full"][field] != loaded["rgb_only"][field]
        for field in (
            "camera_indices",
            "step",
            "trajectory_id",
            "semantic_topology",
        )
    ):
        raise ValueError("Full and RGB-only raw output metadata differ.")

    rendered_by_camera = {
        camera_index: {name: array[camera_index] for name, array in loaded["full"]["arrays"].items()}
        for camera_index in range(64)
    }
    rendered_rgb_by_camera = {
        camera_index: loaded["rgb_only"]["arrays"]["rgb"][camera_index] for camera_index in range(64)
    }
    recomputed_full = evaluate_repaired_mismatch_corpus(
        corpus=corpus,
        rendered_by_camera=rendered_by_camera,
    )
    recomputed_rgb_only = evaluate_repaired_rgb_only_corpus(
        corpus=corpus,
        rendered_rgb_by_camera=rendered_rgb_by_camera,
    )
    reported_rgb_only = _require_mapping(
        _require_mapping(
            report.get("rgb_only_specialization"),
            "report rgb_only_specialization",
        ).get("result"),
        "report rgb_only result",
    )
    if (
        recomputed_full != report.get("historical_mismatch_repair")
        or recomputed_rgb_only != reported_rgb_only
        or recomputed_full.get("pass") is not True
        or recomputed_rgb_only.get("pass") is not True
    ):
        raise ValueError("Raw B64 outputs do not reproduce reported verdicts.")
    return {
        "report": artifact_record(resolved_report),
        "diagnosis_index": diagnosis["artifact"],
        "full": loaded["full"],
        "rgb_only": loaded["rgb_only"],
        "recomputed_full": recomputed_full,
        "recomputed_rgb_only": recomputed_rgb_only,
    }
