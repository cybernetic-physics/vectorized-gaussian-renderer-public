"""Compare matched OVRTX and custom-CUDA matrix evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

if __package__:
    from .renderer_options import rasterize_mode_qualified
else:
    from renderer_options import rasterize_mode_qualified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ovrtx-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--custom-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Flat matched-case CSV; defaults to --output with a .csv suffix.",
    )
    parser.add_argument("--gaussian-support-sigma", type=float, default=2.0)
    parser.add_argument("--covariance-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
    )
    parser.add_argument(
        "--require-compact-projection-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = int(value)
    return converted if converted > 0 else None


def custom_scene_checksum(result: dict[str, Any]) -> str:
    scene = result.get("scene")
    if isinstance(scene, dict):
        return str(scene["checksum_sha256"])
    dataset = result.get("dataset")
    if isinstance(dataset, dict):
        return str(dataset["sha256"])
    raise KeyError("Custom result has no scene checksum.")


def ovrtx_scene_checksum(
    result: dict[str, Any],
    *,
    real_scene: bool,
) -> str:
    dataset = result["dataset"]
    if real_scene:
        return str(dataset["file_sha256"])
    return str(dataset["checksum_sha256"])


def classify_failure(case: dict[str, Any]) -> list[str]:
    if case["status"] == "MEASURED":
        return []
    output_dir = Path(case["output_dir"])
    log_path = output_dir / "run.log"
    log = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else str(case.get("log_tail") or "")
    )
    reasons = []
    if (
        "ERROR_OUT_OF_DEVICE_MEMORY" in log
        or "Out of GPU memory" in log
    ):
        reasons.append("out_of_device_memory")
    if (
        "Too many open files" in log
        or "file descriptors: 0" in log
    ):
        reasons.append("file_descriptor_limit")
    if case.get("timed_out"):
        reasons.append("timeout")
    if not reasons:
        reasons.append("process_failure")
    return reasons


def custom_result_path(
    root: Path,
    *,
    scene: str,
    batch: int,
    width: int,
    height: int,
    rasterize_mode: str,
) -> Path:
    stem = f"{scene}-b{batch}-{width}x{height}"
    return root / f"{rasterize_mode_qualified(stem, rasterize_mode)}.json"


def rasterize_mode_contract_mismatches(
    custom: dict[str, Any],
    *,
    expected: str,
) -> list[str]:
    """Reject custom evidence produced under a different rendering mode."""
    if custom.get("rasterize_mode") == expected:
        return []
    return ["custom_rasterize_mode"]


def comparison_pass(
    *,
    ovrtx_manifest_complete: bool,
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> bool:
    """Return the fail-closed acceptance verdict for the matched matrix."""
    return bool(
        ovrtx_manifest_complete
        and rows
        and not failures
        and all(row["throughput_pass"] for row in rows)
        and all(row["memory_pass"] for row in rows)
        and all(row["zero_steady_state_allocation"] for row in rows)
        and all(row["zero_overflow"] for row in rows)
    )


def require_passing_report(report: dict[str, Any]) -> None:
    """Make a failed persisted report observable to shell automation."""
    if report.get("pass") is not True:
        raise SystemExit(1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scene",
        "batch",
        "width",
        "height",
        "outputs",
        "scene_checksum",
        "camera_manifest_checksum",
        "color_space",
        "color_transform",
        "gaussian_support_sigma",
        "covariance_epsilon",
        "rasterize_mode",
        "ray_gaussian_evaluation",
        "gaussian_evaluation_model",
        "ovrtx_frame_ms_mean",
        "ovrtx_images_per_second",
        "ovrtx_driver_memory_bytes",
        "custom_frame_ms_mean",
        "custom_images_per_second",
        "custom_driver_memory_bytes",
        "custom_peak_allocated_bytes",
        "custom_allocation_delta_bytes",
        "visible_gaussians",
        "tile_intersections",
        "speedup",
        "driver_memory_ratio",
        "throughput_pass",
        "memory_pass",
        "zero_steady_state_allocation",
        "zero_overflow",
        "pass",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            comparison_pass = (
                row["throughput_pass"]
                and row["memory_pass"]
                and row["zero_steady_state_allocation"]
                and row["zero_overflow"]
            )
            writer.writerow(
                {
                    "scene": row["scene"],
                    "batch": row["batch"],
                    "width": row["width"],
                    "height": row["height"],
                    "outputs": "+".join(row["outputs"]),
                    "scene_checksum": row["comparison_contract"][
                        "scene_checksum"
                    ],
                    "camera_manifest_checksum": row[
                        "comparison_contract"
                    ]["camera_manifest_checksum"],
                    "color_space": row["comparison_contract"][
                        "color_space"
                    ],
                    "color_transform": row["comparison_contract"][
                        "color_transform"
                    ],
                    "gaussian_support_sigma": row[
                        "comparison_contract"
                    ]["gaussian_support_sigma"],
                    "covariance_epsilon": row[
                        "comparison_contract"
                    ]["covariance_epsilon"],
                    "rasterize_mode": row["comparison_contract"][
                        "rasterize_mode"
                    ],
                    "ray_gaussian_evaluation": row[
                        "comparison_contract"
                    ]["ray_gaussian_evaluation"],
                    "gaussian_evaluation_model": row[
                        "comparison_contract"
                    ]["gaussian_evaluation_model"],
                    "ovrtx_frame_ms_mean": row["ovrtx"][
                        "frame_ms_mean"
                    ],
                    "ovrtx_images_per_second": row["ovrtx"][
                        "images_per_second"
                    ],
                    "ovrtx_driver_memory_bytes": row["ovrtx"][
                        "driver_process_memory_bytes"
                    ],
                    "custom_frame_ms_mean": row["custom"][
                        "frame_ms_mean"
                    ],
                    "custom_images_per_second": row["custom"][
                        "images_per_second"
                    ],
                    "custom_driver_memory_bytes": row["custom"][
                        "driver_process_memory_bytes"
                    ],
                    "custom_peak_allocated_bytes": row["custom"][
                        "peak_allocated_bytes"
                    ],
                    "custom_allocation_delta_bytes": row["custom"][
                        "allocation_delta_bytes"
                    ],
                    "visible_gaussians": row["custom"][
                        "visible_gaussians"
                    ],
                    "tile_intersections": row["custom"][
                        "tile_intersections"
                    ],
                    "speedup": row["speedup"],
                    "driver_memory_ratio": row["driver_memory_ratio"],
                    "throughput_pass": row["throughput_pass"],
                    "memory_pass": row["memory_pass"],
                    "zero_steady_state_allocation": row[
                        "zero_steady_state_allocation"
                    ],
                    "zero_overflow": row["zero_overflow"],
                    "pass": comparison_pass,
                }
            )


def main() -> None:
    args = parse_args()
    ovrtx_manifest = load_json(args.ovrtx_manifest)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for case in ovrtx_manifest["cases"]:
        scene = str(case["scene"])
        batch = int(case["batch"])
        width = int(case["width"])
        height = int(case["height"])
        custom_path = custom_result_path(
            args.custom_root,
            scene=scene,
            batch=batch,
            width=width,
            height=height,
            rasterize_mode=args.rasterize_mode,
        )
        if case["status"] != "MEASURED":
            failures.append(
                {
                    "implementation": "ovrtx",
                    "scene": scene,
                    "batch": batch,
                    "width": width,
                    "height": height,
                    "status": case["status"],
                    "reasons": classify_failure(case),
                    "elapsed_seconds": case["elapsed_seconds"],
                    "output_dir": case["output_dir"],
                }
            )
            continue
        if not custom_path.is_file():
            failures.append(
                {
                    "implementation": "custom-cuda",
                    "scene": scene,
                    "batch": batch,
                    "width": width,
                    "height": height,
                    "status": "MISSING",
                    "reasons": ["missing_custom_result"],
                    "output_dir": str(custom_path.parent),
                }
            )
            continue

        ovrtx_path = Path(str(case["result_json"]))
        ovrtx = load_json(ovrtx_path)
        custom = load_json(custom_path)
        contract_mismatches = rasterize_mode_contract_mismatches(
            custom,
            expected=args.rasterize_mode,
        )
        real_scene = isinstance(custom.get("dataset"), dict)
        if (
            custom_scene_checksum(custom)
            != ovrtx_scene_checksum(
                ovrtx,
                real_scene=real_scene,
            )
        ):
            contract_mismatches.append("scene_checksum")
        if (
            custom["camera_manifest"]["checksum_sha256"]
            != ovrtx["configuration"]["camera_manifest"][
                "checksum_sha256"
            ]
        ):
            contract_mismatches.append("camera_manifest")
        if set(custom["outputs"]) != set(
            ovrtx["configuration"]["outputs"]
        ):
            contract_mismatches.append("outputs")
        if custom.get("color_space") != "display_srgb":
            contract_mismatches.append("custom_color_space")
        if (
            custom.get("color_transform")
            != "identity_authored_display_rgb"
        ):
            contract_mismatches.append("custom_color_transform")
        if (
            custom.get("gaussian_support_sigma")
            != args.gaussian_support_sigma
        ):
            contract_mismatches.append("custom_gaussian_support_sigma")
        if (
            custom.get("covariance_epsilon")
            != args.covariance_epsilon
        ):
            contract_mismatches.append("custom_covariance_epsilon")
        if custom.get("ray_gaussian_evaluation") is not False:
            contract_mismatches.append(
                "custom_ray_gaussian_evaluation"
            )
        if (
            custom.get("gaussian_evaluation_model")
            != "screen_space_2d_conic"
        ):
            contract_mismatches.append(
                "custom_gaussian_evaluation_model"
            )
        if (
            args.require_compact_projection_cache
            and custom.get("compact_projection_cache") is not True
        ):
            contract_mismatches.append(
                "custom_compact_projection_cache"
            )
        if (
            args.require_compact_projection_cache
            and custom.get("projection_cache", {}).get("enabled")
            is not True
        ):
            contract_mismatches.append("custom_projection_cache")
        if custom.get("outputs_contract_valid") is not True:
            contract_mismatches.append("custom_output_contract")
        if custom.get("pass") is not True:
            contract_mismatches.append("custom_result_pass")
        if contract_mismatches:
            failures.append(
                {
                    "implementation": "comparison-contract",
                    "scene": scene,
                    "batch": batch,
                    "width": width,
                    "height": height,
                    "status": "MISMATCH",
                    "reasons": contract_mismatches,
                    "ovrtx_result_json": str(ovrtx_path),
                    "custom_result_json": str(custom_path),
                }
            )
            continue

        ovrtx_images_per_second = float(
            ovrtx["timing"]["images_per_second"]
        )
        custom_images_per_second = float(
            custom["images_per_second"]
        )
        ovrtx_memory = positive_int_or_none(
            ovrtx["memory"]["driver_process_memory_bytes"]
        )
        custom_memory = positive_int_or_none(
            custom["driver_process_memory_bytes"]
        )
        missing_memory = []
        if ovrtx_memory is None:
            missing_memory.append("missing_ovrtx_driver_memory")
        if custom_memory is None:
            missing_memory.append("missing_custom_driver_memory")
        if missing_memory:
            failures.append(
                {
                    "implementation": "comparison-memory",
                    "scene": scene,
                    "batch": batch,
                    "width": width,
                    "height": height,
                    "status": "MISSING",
                    "reasons": missing_memory,
                    "ovrtx_result_json": str(ovrtx_path),
                    "custom_result_json": str(custom_path),
                }
            )
            continue
        assert ovrtx_memory is not None
        assert custom_memory is not None
        rows.append(
            {
                "scene": scene,
                "batch": batch,
                "width": width,
                "height": height,
                "outputs": ["rgb", "depth", "alpha", "semantic_id"],
                "comparison_contract": {
                    "scene_checksum": custom["scene"][
                        "checksum_sha256"
                    ]
                    if isinstance(custom.get("scene"), dict)
                    else custom["dataset"]["sha256"],
                    "camera_manifest_checksum": custom[
                        "camera_manifest"
                    ]["checksum_sha256"],
                    "color_space": "display_srgb",
                    "color_transform": (
                        "identity_authored_display_rgb"
                    ),
                    "gaussian_support_sigma": (
                        args.gaussian_support_sigma
                    ),
                    "covariance_epsilon": args.covariance_epsilon,
                    "rasterize_mode": args.rasterize_mode,
                    "ray_gaussian_evaluation": False,
                    "gaussian_evaluation_model": "screen_space_2d_conic",
                    "compact_projection_cache": custom.get(
                        "compact_projection_cache"
                    ),
                },
                "ovrtx": {
                    "frame_ms_mean": float(
                        ovrtx["timing"]["wall_ms"]["mean"]
                    ),
                    "images_per_second": ovrtx_images_per_second,
                    "driver_process_memory_bytes": ovrtx_memory,
                    "result_json": str(ovrtx_path),
                },
                "custom": {
                    "frame_ms_mean": float(
                        custom["frame_ms"]["mean"]
                    ),
                    "images_per_second": custom_images_per_second,
                    "driver_process_memory_bytes": custom_memory,
                    "peak_allocated_bytes": int(
                        custom["peak_allocated_bytes"]
                    ),
                    "allocation_delta_bytes": int(
                        custom["allocation_delta_bytes"]
                    ),
                    "visible_gaussians": int(
                        custom["measured_counters"][
                            "visible_gaussians"
                        ]
                    ),
                    "tile_intersections": int(
                        custom["measured_counters"][
                            "tile_intersections"
                        ]
                    ),
                    "result_json": str(custom_path),
                },
                "speedup": (
                    custom_images_per_second
                    / ovrtx_images_per_second
                ),
                "driver_memory_ratio": (
                    custom_memory / ovrtx_memory
                ),
                "throughput_pass": (
                    custom_images_per_second
                    / ovrtx_images_per_second
                    >= 5.0
                ),
                "memory_pass": custom_memory / ovrtx_memory <= 0.8,
                "zero_steady_state_allocation": (
                    int(custom["allocation_delta_bytes"]) == 0
                ),
                "zero_overflow": (
                    int(
                        custom["measured_counters"][
                            "visible_overflow"
                        ]
                    )
                    == 0
                    and int(
                        custom["measured_counters"][
                            "intersection_overflow"
                        ]
                    )
                    == 0
                ),
            }
        )

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["scene"]),
            int(row["width"]),
            int(row["height"]),
        )
        groups.setdefault(key, []).append(row)
    largest_common = []
    for (scene, width, height), group_rows in sorted(groups.items()):
        row = max(group_rows, key=lambda item: int(item["batch"]))
        largest_common.append(
            {
                "scene": scene,
                "width": width,
                "height": height,
                "batch": row["batch"],
                "speedup": row["speedup"],
                "driver_memory_ratio": row[
                    "driver_memory_ratio"
                ],
                "throughput_pass": row["throughput_pass"],
                "memory_pass": row["memory_pass"],
            }
        )

    manifest_complete = bool(ovrtx_manifest.get("complete"))
    report = {
        "schema_version": "ovrtx-custom-matrix-comparison/v1",
        "ovrtx_manifest": str(args.ovrtx_manifest),
        "custom_root": str(args.custom_root),
        "rasterize_mode": args.rasterize_mode,
        "ovrtx_manifest_complete": manifest_complete,
        "matched_case_count": len(rows),
        "failure_count": len(failures),
        "rows": rows,
        "failures": failures,
        "largest_common_successful_batch": largest_common,
        "aggregate": {
            "minimum_speedup": (
                min(row["speedup"] for row in rows)
                if rows
                else None
            ),
            "maximum_driver_memory_ratio": (
                max(row["driver_memory_ratio"] for row in rows)
                if rows
                else None
            ),
            "all_matched_throughput_pass": (
                bool(rows)
                and all(row["throughput_pass"] for row in rows)
            ),
            "all_matched_memory_pass": (
                bool(rows)
                and all(row["memory_pass"] for row in rows)
            ),
            "all_custom_zero_allocation": (
                bool(rows)
                and all(
                    row["zero_steady_state_allocation"]
                    for row in rows
                )
            ),
            "all_custom_zero_overflow": (
                bool(rows)
                and all(row["zero_overflow"] for row in rows)
            ),
        },
    }
    report["pass"] = comparison_pass(
        ovrtx_manifest_complete=manifest_complete,
        rows=rows,
        failures=failures,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        args.csv_output or args.output.with_suffix(".csv"),
        rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    require_passing_report(report)


if __name__ == "__main__":
    main()
