"""CPU tests for the deterministic G1 scripted gait (WS5)."""

from __future__ import annotations

import math

from isaacsim_gaussian_renderer.g1_gait import GAIT_JOINTS, ScriptedGait, static_pose

DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_yaw_joint",  # unmatched -> 0
]


def test_deterministic():
    gait = ScriptedGait()
    a = gait.targets_for_dof_names(DOF_NAMES, 0.37)
    b = gait.targets_for_dof_names(DOF_NAMES, 0.37)
    assert a == b


def test_periodic():
    gait = ScriptedGait(period_seconds=1.2)
    a = gait.targets_for_dof_names(DOF_NAMES, 0.4)
    b = gait.targets_for_dof_names(DOF_NAMES, 0.4 + 1.2)
    for x, y in zip(a, b):
        assert math.isclose(x, y, abs_tol=1e-9)


def test_legs_are_antiphase():
    gait = ScriptedGait()
    # At a phase where left hip is near its positive peak, right hip is negative.
    t = gait.period_seconds * 0.25  # phase = pi/2 -> sin=1 for left
    left = gait.role_target("left_hip_pitch", t)
    right = gait.role_target("right_hip_pitch", t)
    assert left > 0.0 > right


def test_knee_is_one_sided():
    gait = ScriptedGait()
    # Knee target never drops below its bias (flexes only on swing half).
    bias = GAIT_JOINTS["left_knee"][2]
    mins = min(gait.role_target("left_knee", 0.01 * i) for i in range(300))
    assert mins >= bias - 1e-9


def test_targets_bounded_and_unmatched_zero():
    gait = ScriptedGait()
    for i in range(500):
        targets = gait.targets_for_dof_names(DOF_NAMES, 0.013 * i)
        assert all(abs(v) <= math.pi for v in targets)
        assert targets[-1] == 0.0  # waist_yaw unmatched


def test_static_pose_is_zero():
    gait = static_pose()
    assert gait.is_static()
    targets = gait.targets_for_dof_names(DOF_NAMES, 0.5)
    # Only knee bias remains; hips/ankles/arms are zero at amplitude 0.
    assert targets[DOF_NAMES.index("left_hip_pitch_joint")] == 0.0
    assert targets[DOF_NAMES.index("waist_yaw_joint")] == 0.0
