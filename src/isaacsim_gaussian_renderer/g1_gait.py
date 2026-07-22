"""Deterministic scripted gait for the Unitree G1 articulation (WS5).

This is a hand-authored joint-angle trajectory — NOT an RL/locomotion policy and
NOT dependent on Isaac Lab. It produces per-step joint position targets for the
imported G1 URDF/USD articulation so the robot performs a repeatable walking
motion inside the Gaussian scene. The gait is a pure function of phase, so it is
fully deterministic and CPU-testable independent of Isaac Sim.

The G1 leg joints oscillate sinusoidally with the two legs a half-period out of
phase (a symmetric walk); knees flex only during the swing half of each leg's
cycle. Joint roles are matched to the articulation's actual DOF names at runtime
by the benchmark, so this module never hard-codes DOF indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Joint-role -> (amplitude_rad, phase_offset_rad, bias_rad). Roles are matched
# case-insensitively against the articulation's DOF names by substring.
_HIP_A = 0.45
_KNEE_A = 0.75
_ANKLE_A = 0.25

GAIT_JOINTS: dict[str, tuple[float, float, float]] = {
    # left leg (reference phase)
    "left_hip_pitch": (_HIP_A, 0.0, 0.0),
    "left_knee": (_KNEE_A, 0.0, 0.30),
    "left_ankle_pitch": (_ANKLE_A, 0.0, 0.0),
    # right leg (half period out of phase)
    "right_hip_pitch": (_HIP_A, math.pi, 0.0),
    "right_knee": (_KNEE_A, math.pi, 0.30),
    "right_ankle_pitch": (_ANKLE_A, math.pi, 0.0),
    # gentle contralateral arm swing for balance
    "left_shoulder_pitch": (0.20, math.pi, 0.0),
    "right_shoulder_pitch": (0.20, 0.0, 0.0),
}


@dataclass
class ScriptedGait:
    """A deterministic periodic walk parameterized by period and amplitude scale."""

    period_seconds: float = 1.2
    amplitude_scale: float = 1.0
    # Joint position limit used to clamp targets (rad); keeps targets physical.
    joint_limit_rad: float = math.pi
    joints: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: dict(GAIT_JOINTS)
    )

    def phase(self, t_seconds: float) -> float:
        return 2.0 * math.pi * (t_seconds % self.period_seconds) / self.period_seconds

    def role_target(self, role: str, t_seconds: float) -> float:
        spec = self.joints.get(role)
        if spec is None:
            return 0.0
        amp, offset, bias = spec
        phase = self.phase(t_seconds) + offset
        if "knee" in role:
            # Knee flexes only on the swing half of the cycle (one-sided).
            value = bias + self.amplitude_scale * amp * max(0.0, math.sin(phase))
        else:
            value = bias + self.amplitude_scale * amp * math.sin(phase)
        return max(-self.joint_limit_rad, min(self.joint_limit_rad, value))

    def targets_for_dof_names(self, dof_names: list[str], t_seconds: float) -> list[float]:
        """Map each articulation DOF name to a scripted target (0 for unmatched)."""
        targets = []
        for name in dof_names:
            lower = name.lower()
            matched = 0.0
            for role in self.joints:
                if role in lower:
                    matched = self.role_target(role, t_seconds)
                    break
            targets.append(matched)
        return targets

    def is_static(self) -> bool:
        return self.amplitude_scale == 0.0


def static_pose() -> ScriptedGait:
    """The default (non-walking) pose: all scripted targets are zero."""
    return ScriptedGait(amplitude_scale=0.0)
