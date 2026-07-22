"""Strict steady-state memory evidence for matched renderer benchmarks."""

from __future__ import annotations

from typing import Any

MEMORY_BASELINE_SAMPLE_COUNT = 3
TORCH_CUMULATIVE_MEMORY_COUNTERS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "allocated_bytes.all.allocated",
    "allocated_bytes.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "reserved_bytes.all.allocated",
    "reserved_bytes.all.freed",
)


def _all_equal(values: list[Any]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def assess_steady_state_memory(
    baseline_samples: list[dict[str, Any]],
    post_measurement_sample: dict[str, Any],
    *,
    required_baseline_samples: int = MEMORY_BASELINE_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Fail-closed assessment of the allocation-free measured region."""

    failures: list[str] = []
    enough_baseline_samples = len(baseline_samples) >= required_baseline_samples
    if not enough_baseline_samples:
        failures.append(
            "fewer than the required stable pre-measurement memory samples"
        )

    baseline_driver_values = [
        sample.get("driver", {}).get("current_process_used_mib")
        for sample in baseline_samples
    ]
    baseline_nvml_available = bool(baseline_samples) and all(
        isinstance(value, int) for value in baseline_driver_values
    )
    if not baseline_nvml_available:
        failures.append("NVML current-process memory unavailable in baseline")
    baseline_driver_stable = (
        enough_baseline_samples
        and baseline_nvml_available
        and _all_equal(baseline_driver_values)
    )
    if baseline_nvml_available and not baseline_driver_stable:
        failures.append("NVML current-process memory baseline is unstable")

    baseline_torch_allocated = [
        sample.get("torch", {}).get("allocated_bytes")
        for sample in baseline_samples
    ]
    baseline_torch_reserved = [
        sample.get("torch", {}).get("reserved_bytes")
        for sample in baseline_samples
    ]
    baseline_torch_current_available = bool(baseline_samples) and all(
        isinstance(value, int)
        for value in baseline_torch_allocated + baseline_torch_reserved
    )
    if not baseline_torch_current_available:
        failures.append("Torch current allocator state unavailable in baseline")

    baseline_counter_values = {
        name: [
            sample.get("torch", {})
            .get("cumulative_counters", {})
            .get(name)
            for sample in baseline_samples
        ]
        for name in TORCH_CUMULATIVE_MEMORY_COUNTERS
    }
    baseline_torch_counters_available = bool(baseline_samples) and all(
        all(isinstance(value, int) for value in values)
        for values in baseline_counter_values.values()
    )
    if not baseline_torch_counters_available:
        failures.append("Torch cumulative allocator counters unavailable in baseline")

    baseline_torch_stable = bool(
        enough_baseline_samples
        and baseline_torch_current_available
        and baseline_torch_counters_available
        and _all_equal(baseline_torch_allocated)
        and _all_equal(baseline_torch_reserved)
        and all(_all_equal(values) for values in baseline_counter_values.values())
    )
    if (
        baseline_torch_current_available
        and baseline_torch_counters_available
        and not baseline_torch_stable
    ):
        failures.append("Torch allocator baseline is unstable")

    post_driver_value = post_measurement_sample.get("driver", {}).get(
        "current_process_used_mib"
    )
    post_nvml_available = isinstance(post_driver_value, int)
    if not post_nvml_available:
        failures.append("NVML current-process memory unavailable after measurement")

    post_torch = post_measurement_sample.get("torch", {})
    post_allocated = post_torch.get("allocated_bytes")
    post_reserved = post_torch.get("reserved_bytes")
    post_torch_current_available = isinstance(post_allocated, int) and isinstance(
        post_reserved, int
    )
    if not post_torch_current_available:
        failures.append("Torch current allocator state unavailable after measurement")
    post_counters = post_torch.get("cumulative_counters", {})
    post_torch_counters_available = all(
        isinstance(post_counters.get(name), int)
        for name in TORCH_CUMULATIVE_MEMORY_COUNTERS
    )
    if not post_torch_counters_available:
        failures.append(
            "Torch cumulative allocator counters unavailable after measurement"
        )

    before_sample = baseline_samples[-1] if baseline_samples else {}
    before_torch = before_sample.get("torch", {})
    before_driver_value = before_sample.get("driver", {}).get(
        "current_process_used_mib"
    )
    before_allocated = before_torch.get("allocated_bytes")
    before_reserved = before_torch.get("reserved_bytes")
    allocation_growth_bytes = (
        post_allocated - before_allocated
        if isinstance(post_allocated, int) and isinstance(before_allocated, int)
        else None
    )
    reservation_growth_bytes = (
        post_reserved - before_reserved
        if isinstance(post_reserved, int) and isinstance(before_reserved, int)
        else None
    )
    driver_growth_mib = (
        post_driver_value - before_driver_value
        if isinstance(post_driver_value, int)
        and isinstance(before_driver_value, int)
        else None
    )
    if allocation_growth_bytes is not None and allocation_growth_bytes != 0:
        failures.append(
            f"Torch allocated memory changed by {allocation_growth_bytes} bytes"
        )
    if reservation_growth_bytes is not None and reservation_growth_bytes != 0:
        failures.append(
            f"Torch reserved memory changed by {reservation_growth_bytes} bytes"
        )
    if driver_growth_mib is not None and driver_growth_mib != 0:
        failures.append(
            f"NVML current-process memory changed by {driver_growth_mib} MiB"
        )

    before_counters = before_torch.get("cumulative_counters", {})
    counter_deltas = {
        name: (
            post_counters[name] - before_counters[name]
            if isinstance(post_counters.get(name), int)
            and isinstance(before_counters.get(name), int)
            else None
        )
        for name in TORCH_CUMULATIVE_MEMORY_COUNTERS
    }
    for name, delta in counter_deltas.items():
        if delta is not None and delta != 0:
            failures.append(
                f"Torch cumulative allocator counter {name} changed by {delta}"
            )

    baseline_stable = baseline_driver_stable and baseline_torch_stable
    return {
        "pass": not failures,
        "failures": failures,
        "baseline": {
            "required_sample_count": required_baseline_samples,
            "sample_count": len(baseline_samples),
            "nvml_available": baseline_nvml_available,
            "torch_current_state_available": baseline_torch_current_available,
            "torch_cumulative_counters_available": (
                baseline_torch_counters_available
            ),
            "driver_memory_stable": baseline_driver_stable,
            "torch_memory_stable": baseline_torch_stable,
            "stable": baseline_stable,
        },
        "post_measurement": {
            "nvml_available": post_nvml_available,
            "torch_current_state_available": post_torch_current_available,
            "torch_cumulative_counters_available": (
                post_torch_counters_available
            ),
        },
        "allocation_growth_bytes": allocation_growth_bytes,
        "reservation_growth_bytes": reservation_growth_bytes,
        "driver_process_memory_growth_mib": driver_growth_mib,
        "torch_cumulative_counter_deltas": counter_deltas,
    }
