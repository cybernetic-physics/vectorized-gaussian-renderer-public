"""Publication labels for the distinct FlashGS comparison lanes."""

from __future__ import annotations

from typing import Any


FLASHGS_ADAPTER_ATTESTATION_SCHEMA = "flashgs-adapter-attestation-v3"

FLASHGS_MATCHED_PORT_CLASSIFICATION = {
    "baseline_name": "FlashGS-derived matched-contract port",
    "integration_only": False,
    "upstream_faithful": False,
    "retained_upstream_topology": [
        "warp-cooperative tile-key emission",
        "CUB tile-and-depth radix ordering",
        "sorted tile-range discovery",
        "16x16 tile compositor with cooperative Gaussian loads",
    ],
    "service_integration_changes": [
        "canonical GPU-resident scene and camera tensors",
        "active PyTorch CUDA stream execution",
        "reusable outputs and workspace",
        "fixed-capacity sorting without a measured-region count readback",
        "one native pipeline invocation per camera",
    ],
    "matched_equation_changes": [
        "pinned-gsplat projection and culling conventions",
        "3.33-sigma opacity-contracted support and matched radius conventions",
        "zero covariance epsilon and half-pixel sample centers",
        "degree-zero canonical RGB instead of upstream spherical-harmonic evaluation",
        "matched alpha cap, per-splat rejection, transmittance termination, and background",
    ],
    "full_sensor_extension_changes": [
        "float32 linear RGB output",
        "accumulated alpha",
        "expected camera-space depth",
        "strongest-contributor semantic ID",
    ],
    "correctness_repairs": [
        "aligned cooperative feature-load guards for one- and two-entry tile tails",
    ],
    "allowed_publication_label": "FlashGS-derived matched-contract port",
    "forbidden_publication_labels": [
        "upstream FlashGS",
        "minimally adapted FlashGS",
        "integration-only FlashGS",
    ],
}

_REPAIRED_SHORT_TAIL_PREDICATES = (
    "else if ((lane & 4) == 0 && point_id + 0 < range.y)",
    "else if (point_id + 1 < range.y)",
)
_BUGGY_SHORT_TAIL_PREDICATES = (
    "else if ((lane & 4) == 0 && point_id + 1 < range.y)",
    "else if (point_id + 2 < range.y)",
)


def flashgs_short_tail_repair_checks(render_source: str) -> dict[str, bool]:
    """Return the exact source predicates required by attestation v3."""

    return {
        "repaired_slot_0_predicate_exactly_once": render_source.count(_REPAIRED_SHORT_TAIL_PREDICATES[0]) == 1,
        "repaired_slot_1_predicate_exactly_once": render_source.count(_REPAIRED_SHORT_TAIL_PREDICATES[1]) == 1,
        "buggy_slot_0_predicate_absent": (_BUGGY_SHORT_TAIL_PREDICATES[0] not in render_source),
        "buggy_slot_1_predicate_absent": (_BUGGY_SHORT_TAIL_PREDICATES[1] not in render_source),
        "slot_0_load_guard_present": ("load_enable = load_enable && point_id + 0 < range.y" in render_source),
        "slot_1_load_guard_present": ("load_enable = load_enable && point_id + 1 < range.y" in render_source),
    }


def require_flashgs_short_tail_repair(render_source: str) -> dict[str, bool]:
    """Fail unless the attested production source contains the repair."""

    checks = flashgs_short_tail_repair_checks(render_source)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError("FlashGS short-tail correctness repair is absent or ambiguous: " + ", ".join(failed))
    return checks


def require_flashgs_matched_port_classification(payload: Any) -> dict[str, Any]:
    """Fail unless an attestation preserves the reviewed baseline boundary."""

    if payload != FLASHGS_MATCHED_PORT_CLASSIFICATION:
        raise ValueError("FlashGS adapter semantic classification differs from the reviewed matched-port contract.")
    return FLASHGS_MATCHED_PORT_CLASSIFICATION
