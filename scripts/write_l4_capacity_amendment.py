#!/usr/bin/env python3
"""Freeze the pre-timing direct-B512 OOM that motivated the L4 schedule amendment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(PROJECT_ROOT), str(SRC_ROOT)):
    while import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

from isaacsim_gaussian_renderer.evaluation.evidence_bundle import (  # noqa: E402
    CAPACITY_AMENDMENT_ARTIFACTS,
    CAPACITY_AMENDMENT_SCHEMA,
    capacity_amendment_relative_path,
    file_sha256,
)
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (  # noqa: E402
    artifact_record,
)

SOURCE_PATHS = {
    "failed_source_manifest": Path("evidence/provenance/source-manifest.json"),
    "matrix_invocation": Path("matrix/provenance/matrix-invocation.json"),
    "matrix_launch_occupancy": Path("matrix/provenance/matrix-launch-occupancy.json"),
    "capacity_run_occupancy": Path("matrix/capacity/custom/b512.node-occupancy.json"),
    "capacity_command": Path("matrix/logs/capacity-custom-b512.command.json"),
    "capacity_log": Path("matrix/logs/capacity-custom-b512.log"),
    "matrix_driver_log": Path("matrix/matrix-driver.log"),
    "matrix_exit_code": Path("matrix/matrix-exit-code"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def require_exact_source(path: Path, *, name: str) -> None:
    expected = CAPACITY_AMENDMENT_ARTIFACTS[name]
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Capacity-amendment source must be a regular file: {path}.")
    if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
        raise ValueError(f"Capacity-amendment source differs: {name}.")


def main() -> None:
    args = parse_args()
    failed_root = args.failed_root.resolve()
    matrix_root = failed_root / "matrix"
    if list((matrix_root / "runs").rglob("*.json")) or (matrix_root / "summary.json").exists():
        raise ValueError("Failed matrix root unexpectedly contains a timed result or summary.")

    source_manifest = json.loads((failed_root / SOURCE_PATHS["failed_source_manifest"]).read_text())
    invocation = json.loads((failed_root / SOURCE_PATHS["matrix_invocation"]).read_text())
    parsed = invocation.get("parsed_arguments") or {}
    expected_old_schedule = [
        {"batch": 512, "renderer": "custom"},
        {"batch": 1024, "renderer": "custom"},
        {"batch": 1, "renderer": "custom"},
        {"batch": 8, "renderer": "custom"},
        {"batch": 32, "renderer": "custom"},
        {"batch": 64, "renderer": "custom"},
        {"batch": 128, "renderer": "custom"},
        {"batch": 256, "renderer": "custom"},
    ]
    if (
        source_manifest.get("head") != "025bdaf2258894f477074dbd955653abcbfb3f62"
        or source_manifest.get("dirty") is not False
        or parsed.get("capacity_calibration_schedule") != expected_old_schedule
        or parsed.get("expected_gpu_uuid") != "GPU-ebf6dc95-db46-4e1f-6e95-492c5c787805"
    ):
        raise ValueError("Failed matrix identity differs from the frozen direct-B512 attempt.")
    failure_log = (failed_root / SOURCE_PATHS["capacity_log"]).read_text(encoding="utf-8")
    if "CUDA out of memory" not in failure_log or "workspace growth to 21445449872 bytes" not in failure_log:
        raise ValueError("Direct-B512 allocator failure is absent from the frozen log.")

    output_root = args.output_root.resolve()
    evidence: dict[str, dict[str, object]] = {}
    for name, relative in SOURCE_PATHS.items():
        source = failed_root / relative
        require_exact_source(source, name=name)
        destination = output_root / capacity_amendment_relative_path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require_exact_source(destination, name=name)
        else:
            shutil.copyfile(source, destination)
        evidence[name] = artifact_record(destination)

    record = {
        "schema_version": CAPACITY_AMENDMENT_SCHEMA,
        "pass": True,
        "failed_source_head": "025bdaf2258894f477074dbd955653abcbfb3f62",
        "failed_direct_protocol": {
            "logical_batch": 512,
            "physical_batch": 512,
            "capacity_headroom": 1.05,
            "requested_backend_workspace_bytes": 21_445_449_872,
            "timed_rows_completed": 0,
            "failure_stage": "custom-capacity-calibration-workspace-allocation",
        },
        "replacement_protocol": {
            "capacity_headroom": 1.05,
            "alternative_chunk_sizes_swept": False,
            "timing_observed_before_selection": False,
            "custom_physical_batches": {
                "b1-b256": "direct",
                "b512": 128,
                "b1024": 128,
            },
            "custom_native_submissions": {
                "b1-b256": 1,
                "b512": 4,
                "b1024": 8,
            },
        },
        "evidence": evidence,
    }
    output = output_root / "publication/capacity-amendment.json"
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"Refusing to overwrite a different capacity amendment: {output}.")
    output.write_text(encoded, encoding="utf-8")
    print(json.dumps(artifact_record(output), sort_keys=True))


if __name__ == "__main__":
    main()
