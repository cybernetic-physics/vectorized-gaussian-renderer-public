#!/usr/bin/env python3
# ruff: noqa: E402
"""Build a checksum-frozen, independently moving robotics camera contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_trajectory import trajectory_from_route
from isaacsim_gaussian_renderer.benchmark_manifest import file_sha256
from isaacsim_gaussian_renderer.evaluation.logical_batch import (
    audit_logical_camera_contract,
    build_logical_phase_contract,
    deterministic_camera_order,
    select_logical_cameras,
)
from isaacsim_gaussian_renderer.evaluation.trajectory_contract import (
    save_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--scene-sha256", required=True)
    parser.add_argument("--logical-batch", type=int, default=1024)
    parser.add_argument("--route-samples", dest="frames", type=int, default=1024)
    parser.add_argument("--measured-steps", type=int, default=4)
    parser.add_argument("--step-stride", type=int, default=37)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--focal-scale", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--fidelity-batch", type=int, default=8)
    parser.add_argument("--fidelity-seed", type=int, default=560008)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < args.logical_batch:
        raise ValueError("--route-samples must be at least --logical-batch.")
    source = trajectory_from_route(args)
    contract = build_logical_phase_contract(
        source,
        logical_batch=args.logical_batch,
        measured_steps=args.measured_steps,
        step_stride=args.step_stride,
    )
    json_path, npz_path = save_trajectory(contract, args.output)
    fidelity_order, _ = deterministic_camera_order(
        contract.batch,
        seed=args.fidelity_seed,
    )
    if not 0 < args.fidelity_batch <= contract.batch:
        raise ValueError("--fidelity-batch must be within the logical batch.")
    fidelity_ids = fidelity_order[: args.fidelity_batch]
    fidelity_contract = select_logical_cameras(
        contract,
        fidelity_ids,
        purpose="held-out-fidelity",
    )
    fidelity_json = json_path.with_name(f"{json_path.stem}.fidelity.json")
    fidelity_json, fidelity_npz = save_trajectory(
        fidelity_contract,
        fidelity_json,
    )
    audit = {
        **audit_logical_camera_contract(contract),
        "trajectory_id": contract.trajectory_id,
        "route_sha256": contract.route_sha256,
        "scene_sha256": contract.scene_sha256,
        "json_path": json_path.name,
        "npz_path": npz_path.name,
        "json_sha256": file_sha256(json_path),
        "npz_sha256": file_sha256(npz_path),
        "fidelity": {
            "trajectory_id": fidelity_contract.trajectory_id,
            "batch": fidelity_contract.batch,
            "camera_ids": [int(value) for value in fidelity_ids],
            "json_path": fidelity_json.name,
            "npz_path": fidelity_npz.name,
            "json_sha256": file_sha256(fidelity_json),
            "npz_sha256": file_sha256(fidelity_npz),
            "audit": audit_logical_camera_contract(fidelity_contract),
        },
    }
    audit_path = json_path.with_name(f"{json_path.stem}.audit.json")
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ROBOTICS_BATCH_CONTRACT_OK", json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
