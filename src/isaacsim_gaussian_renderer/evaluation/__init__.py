"""Trajectory-based renderer evaluation contracts and artifact helpers."""

from .artifacts import EvaluationArtifactBundle, audit_artifact_bundle
from .logical_batch import (
    audit_logical_camera_contract,
    audit_microbatch_schedule,
    build_logical_phase_contract,
    deterministic_camera_order,
    select_logical_cameras,
)
from .scenarios import ScenarioSpec, build_scenario
from .trajectory_contract import (
    CAMERA_TRAJECTORY_SCHEMA_VERSION,
    CameraTrajectory,
    load_trajectory,
    save_trajectory,
)

__all__ = [
    "CAMERA_TRAJECTORY_SCHEMA_VERSION",
    "CameraTrajectory",
    "EvaluationArtifactBundle",
    "ScenarioSpec",
    "audit_logical_camera_contract",
    "audit_microbatch_schedule",
    "audit_artifact_bundle",
    "build_logical_phase_contract",
    "build_scenario",
    "deterministic_camera_order",
    "load_trajectory",
    "save_trajectory",
    "select_logical_cameras",
]
