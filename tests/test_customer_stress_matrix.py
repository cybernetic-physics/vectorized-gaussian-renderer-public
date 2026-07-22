from __future__ import annotations

from pathlib import Path

import benchmarks.run_customer_stress_matrix as stress_matrix

from benchmarks.run_customer_stress_matrix import (
    CapacityPlan,
    MatrixCase,
    acceptance_summary,
    cache_comparisons,
    capacity_failure_evidence,
    classify_failure,
    conservative_capacity,
    oracle_capacity,
    production_capacity,
    validate_capacity_adaptation,
    validate_comparison_summary,
    validate_custom_result,
    validate_public_result,
)


def test_comparison_case_declares_repo_owned_gsplat_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(**kwargs: object) -> tuple[int, str, None]:
        captured.update(kwargs)
        return 1, "expected test failure", None

    monkeypatch.setattr(stress_matrix, "_run_command", fake_run_command)
    project_root = tmp_path / "checkout"
    stress_matrix.run_comparison_case(
        case=MatrixCase("synthetic-small", 1, 128, 128),
        capacity=CapacityPlan(
            regime="conservative-non-oracle",
            visible_records=10_000,
            tile_intersections=80_000,
            acceptance_gate=False,
            source="test",
        ),
        output_root=tmp_path / "output",
        project_root=project_root,
        warmup=1,
        iterations=1,
        timeout_seconds=30.0,
        required_evidence=True,
    )

    command = captured["command"]
    assert isinstance(command, list)
    flag_index = command.index("--gsplat-compatibility-patch")
    assert Path(command[flag_index + 1]) == (
        project_root / "patches" / "gsplat-cuda-event-flags.patch"
    )


def test_capacity_regimes_are_explicit_and_oracle_is_non_gating() -> None:
    case = MatrixCase("synthetic-medium", 8, 256, 256)

    production = production_capacity(case)
    conservative = conservative_capacity(case)
    oracle = oracle_capacity(
        counters={
            "visible_gaussians": 42_486,
            "tile_intersections": 242_843,
        },
        headroom=1.25,
        source_case_id="conservative/example",
    )

    assert production.regime == "production-default"
    assert production.acceptance_gate is True
    assert production.visible_records == 52_000
    assert production.tile_intersections == 520_000
    assert conservative.regime == "conservative-non-oracle"
    assert conservative.acceptance_gate is False
    assert conservative.visible_records == 800_000
    assert conservative.tile_intersections == 25_600_000
    assert oracle == CapacityPlan(
        regime="hindsight-oracle-diagnostic",
        visible_records=53_108,
        tile_intersections=303_554,
        acceptance_gate=False,
        source=("1.25x observed demand from conservative/example; never valid for production acceptance"),
    )


def test_failure_classifier_distinguishes_capacity_and_oom() -> None:
    assert classify_failure("workspace overflow", 1) == "CAPACITY_OVERFLOW"
    assert classify_failure("CUDA out of memory", 1) == "OUT_OF_DEVICE_MEMORY"
    assert classify_failure("unknown", -9) == "SIGNAL_9"


def test_capacity_adaptation_validation_requires_successful_final_state() -> None:
    valid = {
        "schema_version": "renderer-capacity/v1",
        "adaptive": True,
        "current_capacity": {
            "visible_records": 16,
            "tile_intersections": 64,
            "workspace_bytes": 4096,
        },
        "last_render": {
            "status": "success",
            "final_counters": {
                "visible_overflow": 0,
                "intersection_overflow": 0,
            },
        },
    }

    assert validate_capacity_adaptation(valid, label="candidate") == []
    assert validate_capacity_adaptation(None, label="candidate") == [
        "candidate is missing adaptive-capacity telemetry"
    ]
    valid["last_render"]["final_counters"]["intersection_overflow"] = 1
    assert "candidate adaptive capacity ended with overflow" in (
        validate_capacity_adaptation(valid, label="candidate")
    )


def test_capacity_failure_evidence_marks_censored_and_exact_counters() -> None:
    censored = capacity_failure_evidence(
        "Custom renderer workspace overflow: visible=9344 intersections=62233.",
        visible_capacity=6_500,
        intersection_capacity=26_000,
    )
    exact = capacity_failure_evidence(
        "{'visible_gaussians': 50184, 'tile_intersections': 242378, "
        "'visible_overflow': 0, 'intersection_overflow': 72378, "
        "'active_tiles': 9346}",
        visible_capacity=150_000,
        intersection_capacity=170_000,
    )

    assert censored is not None
    assert censored["censored"] is True
    assert censored["counters"]["visible_gaussians_lower_bound"] == 15_844
    assert censored["counters"]["tile_intersections_lower_bound"] == 88_233
    assert exact is not None
    assert exact["censored"] is False
    assert exact["counters"]["tile_intersections"] == 242_378
    assert exact["counters"]["intersection_overflow"] == 72_378

    adaptive = capacity_failure_evidence(
        "Last attempt: visible_gaussians=20000 tile_intersections=400000 "
        "visible_capacity=18000 intersection_capacity=350000 "
        "visible_overflow=2000 intersection_overflow=50000.",
        visible_capacity=6_500,
        intersection_capacity=26_000,
    )
    assert adaptive is not None
    assert adaptive["censored"] is False
    assert adaptive["initial_requested_visible_capacity"] == 6_500
    assert adaptive["requested_visible_capacity"] == 18_000
    assert adaptive["requested_intersection_capacity"] == 350_000
    assert adaptive["counters"]["visible_gaussians"] == 20_000


def test_custom_validation_fails_closed_on_overflow() -> None:
    result = {
        "schema_version": "custom-cuda-benchmark/v1",
        "pass": True,
        "outputs": ["rgb", "depth", "alpha", "semantic_id"],
        "outputs_gpu_resident": True,
        "outputs_contract_valid": True,
        "zero_overflow": False,
        "environment": {"project_commit": "a" * 40},
        "measured_counters": {
            "visible_overflow": 0,
            "intersection_overflow": 7,
        },
        "frame_ms": {
            "mean": 1.0,
            "p50": 1.0,
            "p95": 1.1,
            "ci95": 0.1,
        },
        "wall_frame_ms": 1.1,
        "driver_process_memory_bytes": 1024,
        "scene": {"checksum_sha256": "a" * 64},
        "camera_manifest": {"checksum_sha256": "b" * 64},
    }

    errors = validate_custom_result(result)

    assert "custom result reported capacity overflow" in errors
    assert "custom measured counters are not overflow-free" in errors


def test_comparison_validation_requires_capacity_role_and_timing() -> None:
    summary = {
        "schema_version": "custom-gsplat-fidelity-control/v1",
        "pass": True,
        "environment": {"project_commit": "a" * 40},
        "candidate": {
            "capacity": {"regime": "hindsight-oracle-diagnostic"},
            "counters": {
                "visible_overflow": 0,
                "intersection_overflow": 0,
            },
        },
        "timing": None,
        "scene": {"checksum_sha256": "a" * 64},
        "camera_bundle_id": "bundle",
    }

    errors = validate_comparison_summary(
        summary,
        expected_regime="conservative-non-oracle",
    )

    assert "custom-gsplat capacity regime is mislabeled" in errors
    assert "custom-gsplat comparison is missing timing" in errors


def test_public_validation_requires_explicit_invalidation_misses() -> None:
    result = {
        "schema_version": "real-scene-custom-benchmark/v1",
        "pass": True,
        "zero_overflow": True,
        "all_output_checks_valid": True,
        "outputs_gpu_resident": True,
        "outputs": {
            "rgb": {},
            "depth": {},
            "alpha": {},
            "semantic_id": {},
        },
        "environment": {"project_commit": "a" * 40},
        "projection_cache": {
            "enabled": True,
            "explicit_invalidation_before_each_render": True,
            "measured_invalidations": 3,
            "measured_misses": 2,
            "measured_hits": 1,
        },
        "measured_iterations": 3,
        "frame_ms": {
            "mean": 1.0,
            "p50": 1.0,
            "p95": 1.1,
            "ci95": 0.1,
        },
        "wall_frame_ms": 1.1,
        "driver_process_memory_bytes": 1024,
    }

    errors = validate_public_result(
        result,
        expected_cache_enabled=True,
        expected_invalidation=True,
    )

    assert "public-scene invalidation lane did not miss on every measured render" in errors
    assert "public-scene invalidation lane unexpectedly reported cache hits" in errors


def test_oracle_pass_cannot_rescue_failed_production_gate() -> None:
    records = [
        {
            "case_id": "production-default/stress",
            "acceptance_gate": True,
            "required_evidence": True,
            "pass": False,
        },
        {
            "case_id": "hindsight-oracle-diagnostic/stress",
            "acceptance_gate": False,
            "required_evidence": True,
            "pass": True,
        },
    ]

    acceptance = acceptance_summary(records)

    assert acceptance["production_default"]["pass"] is False
    assert acceptance["required_evidence"]["pass"] is False
    assert acceptance["oracle_can_satisfy_production_gate"] is False
    assert acceptance["pass"] is False


def test_cache_comparison_is_explicitly_not_moving_camera() -> None:
    records = [
        {
            "case_id": "cache-hit/b1-128x128",
            "lane": "public-direct-compact",
            "workload": {"batch": 1},
            "pass": True,
            "wall_frame_ms": 0.25,
        },
        {
            "case_id": "cache-miss/b1-128x128",
            "lane": "public-direct-compact",
            "workload": {"batch": 1},
            "pass": True,
            "wall_frame_ms": 5.0,
        },
        {
            "case_id": "cache-invalidated/b1-128x128",
            "lane": "public-direct-compact",
            "workload": {"batch": 1},
            "pass": True,
            "wall_frame_ms": 5.5,
        },
    ]

    comparisons = cache_comparisons(records)

    assert comparisons[0]["cache_benefit_vs_disabled"] == 20.0
    assert comparisons[0]["cache_benefit_vs_invalidated"] == 22.0
    assert comparisons[0]["invalidation_overhead_vs_disabled"] == 1.1
    assert "not a moving-camera result" in comparisons[0]["interpretation"]
