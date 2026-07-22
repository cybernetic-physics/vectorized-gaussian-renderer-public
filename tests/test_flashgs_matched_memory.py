from __future__ import annotations

from benchmarks.flashgs_matched_memory import (
    TORCH_CUMULATIVE_MEMORY_COUNTERS,
    assess_steady_state_memory,
)
from benchmarks.summarize_flashgs_matched import steady_state_fairness_failures


def memory_sample(
    *,
    driver_mib: int | None = 4096,
    allocated_bytes: int = 1024,
    reserved_bytes: int = 2048,
    counter_updates: dict[str, int] | None = None,
) -> dict[str, object]:
    counters = {name: 100 for name in TORCH_CUMULATIVE_MEMORY_COUNTERS}
    counters.update(counter_updates or {})
    return {
        "torch": {
            "allocated_bytes": allocated_bytes,
            "reserved_bytes": reserved_bytes,
            "cumulative_counters": counters,
            "missing_cumulative_counters": [],
        },
        "driver": {
            "source": "nvidia-smi-query-compute-apps",
            "current_process_used_mib": driver_mib,
            "available": driver_mib is not None,
        },
    }


def stable_baseline() -> list[dict[str, object]]:
    return [memory_sample() for _ in range(3)]


def run_with_fairness(fairness: dict[str, object]) -> dict[str, object]:
    return {
        "memory": {
            "allocation_growth_bytes": fairness["allocation_growth_bytes"],
            "reservation_growth_bytes": fairness["reservation_growth_bytes"],
            "driver_process_memory_growth_mib": fairness[
                "driver_process_memory_growth_mib"
            ],
            "torch_cumulative_counter_deltas": fairness[
                "torch_cumulative_counter_deltas"
            ],
            "steady_state_fairness": fairness,
        },
        "validation": {
            "scope": "post-measurement",
            "included_in_cuda_event_timing": False,
            "included_in_memory_snapshots": False,
            "post_measurement_memory_snapshots_completed_before_validation": True,
        },
    }


def test_stable_allocation_free_measurement_passes() -> None:
    assessment = assess_steady_state_memory(stable_baseline(), memory_sample())

    assert assessment["pass"] is True
    assert assessment["failures"] == []
    assert assessment["baseline"]["stable"] is True
    assert all(
        delta == 0
        for delta in assessment["torch_cumulative_counter_deltas"].values()
    )


def test_missing_nvml_sample_fails_closed() -> None:
    baseline = stable_baseline()
    baseline[1] = memory_sample(driver_mib=None)

    assessment = assess_steady_state_memory(baseline, memory_sample())

    assert assessment["pass"] is False
    assert assessment["baseline"]["nvml_available"] is False
    assert any("NVML" in failure for failure in assessment["failures"])


def test_unstable_memory_baseline_fails_closed() -> None:
    baseline = stable_baseline()
    baseline[1] = memory_sample(driver_mib=4097)

    assessment = assess_steady_state_memory(baseline, memory_sample())

    assert assessment["pass"] is False
    assert assessment["baseline"]["driver_memory_stable"] is False
    assert any("baseline is unstable" in failure for failure in assessment["failures"])


def test_cumulative_counters_detect_allocate_then_free_with_zero_net_growth() -> None:
    post_measurement = memory_sample(
        counter_updates={
            "allocation.all.allocated": 101,
            "allocation.all.freed": 101,
            "allocated_bytes.all.allocated": 4196,
            "allocated_bytes.all.freed": 4196,
        }
    )

    assessment = assess_steady_state_memory(
        stable_baseline(),
        post_measurement,
    )

    assert assessment["allocation_growth_bytes"] == 0
    assert assessment["reservation_growth_bytes"] == 0
    assert assessment["driver_process_memory_growth_mib"] == 0
    assert assessment["pass"] is False
    assert assessment["torch_cumulative_counter_deltas"][
        "allocation.all.allocated"
    ] == 1
    assert any(
        "cumulative allocator counter" in failure
        for failure in assessment["failures"]
    )


def test_summarizer_accepts_complete_strict_fairness_evidence() -> None:
    fairness = assess_steady_state_memory(stable_baseline(), memory_sample())

    assert steady_state_fairness_failures(
        "custom", run_with_fairness(fairness)
    ) == []


def test_summarizer_fails_closed_without_strict_fairness_evidence() -> None:
    run = {
        "memory": {
            "allocation_growth_bytes": 0,
            "driver_process_memory_growth_mib": None,
        }
    }

    failures = steady_state_fairness_failures("custom", run)

    assert failures == [
        "custom strict steady-state fairness evidence is unavailable"
    ]


def test_summarizer_requires_validation_to_be_outside_memory_snapshots() -> None:
    fairness = assess_steady_state_memory(stable_baseline(), memory_sample())
    run = run_with_fairness(fairness)
    run["validation"]["included_in_memory_snapshots"] = True

    failures = steady_state_fairness_failures("custom", run)

    assert "custom validation memory exclusion is not established" in failures
