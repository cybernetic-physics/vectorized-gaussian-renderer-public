#!/usr/bin/env python3
"""Compare a matched custom/FlashGS capture with the pinned gsplat oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.fidelity.camera_bundle import load_camera_bundle  # noqa: E402
from isaacsim_gaussian_renderer.fidelity.metrics import compare_render_outputs  # noqa: E402
from isaacsim_gaussian_renderer.fidelity.outputs import RenderOutput, load_render_output  # noqa: E402
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    FIDELITY_SCHEMA,
    GSPLAT_ORACLE_SCHEMA,
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PINNED_GSPLAT_COMMIT,
    PINNED_GSPLAT_PATCHED_UTILS_SHA256,
    PINNED_GSPLAT_PATCH_SHA256,
    RENDERER_RUN_SCHEMA,
    artifact_record,
    same_artifact,
    source_identity,
    verify_gsplat_oracle_support_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--camera-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-artifact-views", type=int, default=2)
    parser.add_argument("--no-require-lpips", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    if run.get("schema_version") != RENDERER_RUN_SCHEMA:
        raise ValueError("Run does not use the matched renderer schema.")
    for field, expected in MATCHED_PROJECTION_RULES.items():
        if run.get("equation_contract", {}).get(field) != expected:
            raise ValueError(f"Run projection rule {field} differs.")
    capture_record = run.get("fidelity_capture")
    if not capture_record:
        raise ValueError("Run has no frozen fidelity-suite capture.")
    capture_path = Path(capture_record["path"])
    run_artifact = artifact_record(args.run)
    capture_artifact = artifact_record(capture_path)
    if not same_artifact(capture_record, capture_artifact):
        raise ValueError("Run capture content does not match its recorded hash.")
    oracle_artifact = artifact_record(args.oracle)
    camera_bundle_artifact = artifact_record(args.camera_bundle)
    oracle_manifest_path = args.oracle.with_suffix(".manifest.json")
    oracle_manifest = json.loads(
        oracle_manifest_path.read_text(encoding="utf-8")
    )
    if oracle_manifest.get("schema_version") != GSPLAT_ORACLE_SCHEMA:
        raise ValueError("Oracle manifest does not use the matched v4 schema.")
    verify_gsplat_oracle_support_evidence(oracle_manifest)
    if oracle_manifest.get("output_sha256") != oracle_artifact["sha256"]:
        raise ValueError("Oracle content does not match its manifest hash.")
    if not same_artifact(
        oracle_manifest.get("camera_bundle_artifact"),
        camera_bundle_artifact,
    ):
        raise ValueError("Camera bundle does not match the oracle manifest.")
    if oracle_manifest.get("gsplat_commit") != PINNED_GSPLAT_COMMIT:
        raise ValueError("Oracle does not use the pinned gsplat commit.")
    if (oracle_manifest.get("gsplat_compatibility_patch") or {}).get(
        "sha256"
    ) != PINNED_GSPLAT_PATCH_SHA256:
        raise ValueError("Oracle does not use the pinned compatibility patch.")
    if (oracle_manifest.get("gsplat_compatibility_patch") or {}).get(
        "patched_utils_sha256"
    ) != PINNED_GSPLAT_PATCHED_UTILS_SHA256:
        raise ValueError("Oracle patched Utils.cpp content differs.")
    if oracle_manifest.get("gaussian_support_sigma") != (
        MATCHED_GAUSSIAN_SUPPORT_SIGMA
    ):
        raise ValueError("Oracle does not use the pinned gsplat support cutoff.")
    support_contract = oracle_manifest.get("gsplat_support_contract") or {}
    if (
        support_contract.get("macro") != "GAUSSIAN_EXTEND"
        or support_contract.get("value") != MATCHED_GAUSSIAN_SUPPORT_SIGMA
    ):
        raise ValueError("Oracle gsplat support contract differs.")
    support_header = support_contract.get("header_artifact") or {}
    support_header_path = Path(str(support_header.get("path", "")))
    if not support_header_path.is_file() or not same_artifact(
        support_header,
        artifact_record(support_header_path),
    ):
        raise ValueError("Oracle gsplat support header content differs.")
    for field in ("gsplat_python_artifact", "gsplat_native_extension"):
        recorded = oracle_manifest.get(field)
        loaded_path = Path(str((recorded or {}).get("path", "")))
        if not loaded_path.is_file() or not same_artifact(
            recorded,
            artifact_record(loaded_path),
        ):
            raise ValueError(f"Oracle {field} content differs from its record.")
    if oracle_manifest.get("gpu_uuid") != run["environment"]["gpu_uuid"]:
        raise ValueError("Run and oracle GPU UUIDs differ.")
    if source_identity(run["environment"]["source_provenance"]) != (
        source_identity(oracle_manifest["source_provenance"])
    ):
        raise ValueError("Run and oracle source provenance differ.")
    oracle = load_render_output(args.oracle)
    camera_bundle = load_camera_bundle(args.camera_bundle)
    with np.load(args.oracle, allow_pickle=False) as oracle_archive:
        camera_indices = np.asarray(
            oracle_archive["camera_indices"], dtype=np.int64
        )
        oracle_steps = np.asarray(oracle_archive["steps"], dtype=np.int64)
        oracle_trajectory_id = str(oracle_archive["trajectory_id"].item())
        oracle_semantic_topology = str(
            oracle_archive["semantic_topology"].item()
        )
        oracle_support_sigma = float(
            oracle_archive["gaussian_support_sigma"].item()
        )
    oracle_selection = [
        [int(step), int(camera)]
        for step, camera in zip(
            oracle_steps.tolist(), camera_indices.tolist(), strict=True
        )
    ]
    if (
        oracle_manifest.get("selection_pairs") != oracle_selection
        or oracle_manifest.get("steps") != oracle_steps.tolist()
        or oracle_manifest.get("camera_indices") != camera_indices.tolist()
    ):
        raise ValueError("Oracle archive and manifest selections differ.")
    if run["camera_contract"]["trajectory_id"] != oracle_trajectory_id:
        raise ValueError("Run and oracle trajectory IDs differ.")
    run_semantic_topology = run["equation_contract"].get(
        "semantic_topology"
    )
    if run_semantic_topology != oracle_semantic_topology:
        raise ValueError(
            "Run and oracle semantic topologies differ: "
            f"{run_semantic_topology!r} != {oracle_semantic_topology!r}."
        )
    run_support_sigma = run["equation_contract"].get(
        "gaussian_support_sigma"
    )
    if not np.isclose(
        oracle_support_sigma,
        MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        rtol=0.0,
        atol=1.0e-6,
    ) or run_support_sigma != MATCHED_GAUSSIAN_SUPPORT_SIGMA:
        raise ValueError(
            "Run and oracle do not use the pinned gsplat support cutoff."
        )
    with np.load(capture_path, allow_pickle=False) as capture:
        capture_indices = np.asarray(capture["camera_indices"], dtype=np.int64)
        capture_steps = np.asarray(capture["steps"], dtype=np.int64)
        capture_trajectory_id = str(capture["trajectory_id"].item())
        if (
            capture_trajectory_id != oracle_trajectory_id
            or not np.array_equal(capture_indices, camera_indices)
            or not np.array_equal(capture_steps, oracle_steps)
            or capture_record.get("selection_pairs") != oracle_selection
        ):
            raise ValueError("Candidate and oracle fidelity selections differ.")
        rgb = np.asarray(capture["rgb"], dtype=np.float32)
        if run["output_contract"] == "full":
            alpha = np.asarray(capture["alpha"], dtype=np.float32)[..., 0]
            depth = np.asarray(capture["depth"], dtype=np.float32)[..., 0]
            semantic = np.asarray(capture["semantic_id"])[..., 0]
            non_rgb_source = "candidate"
        else:
            # The RGB-only contract is judged only on its requested output.
            # Supplying oracle-identical auxiliary arrays lets the repository's
            # standard all-output metric engine retain its strict RGB gates.
            alpha = oracle.alpha.copy()
            depth = oracle.depth.copy()
            semantic = oracle.semantic.copy()
            non_rgb_source = "oracle-not-requested-by-candidate"
    candidate = RenderOutput(
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        semantic=semantic,
        valid_depth=alpha > 0.01,
        color_space="linear_rgb",
        background=(0.0, 0.0, 0.0),
        camera_bundle_id=camera_bundle.bundle_id,
        source=str(capture_path),
    )
    report = compare_render_outputs(
        reference=oracle,
        candidate=candidate,
        camera_bundle=camera_bundle,
        output_dir=args.output_dir,
        config_id=(
            f"{run['renderer']}-{run['output_contract']}-"
            f"b{run['camera_contract']['batch']}-vs-pinned-gsplat"
        ),
        require_lpips=not args.no_require_lpips,
        max_artifact_views=args.max_artifact_views,
    )
    summary = {
        "schema_version": FIDELITY_SCHEMA,
        "run": str(args.run),
        "renderer": run["renderer"],
        "output_contract": run["output_contract"],
        "batch": run["camera_contract"]["batch"],
        "trajectory_id": oracle_trajectory_id,
        "steps": oracle_steps.tolist(),
        "semantic_topology": oracle_semantic_topology,
        "gaussian_support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        "camera_indices": camera_indices.tolist(),
        "selection_pairs": oracle_selection,
        "oracle": str(args.oracle),
        "source_identity": source_identity(
            run["environment"]["source_provenance"]
        ),
        "input_artifacts": {
            "run": run_artifact,
            "capture": capture_artifact,
            "oracle": oracle_artifact,
            "oracle_manifest": artifact_record(oracle_manifest_path),
            "gsplat_build_attestation": artifact_record(
                oracle_manifest["gsplat_build_attestation"]["path"]
            ),
            "camera_bundle": camera_bundle_artifact,
        },
        "non_rgb_candidate_source": non_rgb_source,
        "thresholds": report["thresholds"],
        "aggregate": report["aggregate"],
        "pass": bool(report["pass"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary["report_artifacts"] = {
        "json": artifact_record(args.output_dir / "fidelity_report.json"),
        "csv": artifact_record(args.output_dir / "fidelity_report.csv"),
    }
    summary_path = args.output_dir / "matched-fidelity-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "FLASHGS_MATCHED_FIDELITY_OK "
        + json.dumps({"summary": str(summary_path), "pass": summary["pass"]}, sort_keys=True)
    )
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
