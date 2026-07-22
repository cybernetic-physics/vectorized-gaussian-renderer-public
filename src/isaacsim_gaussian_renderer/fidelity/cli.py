"""Command-line entry point for deterministic renderer fidelity comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .camera_bundle import load_camera_bundle
from .metrics import compare_render_outputs
from .outputs import load_render_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare machine-readable Gaussian render outputs.")
    parser.add_argument("--reference", required=True, help="Reference .npz or render-output-v1 JSON manifest.")
    parser.add_argument("--candidate", required=True, help="Candidate .npz or render-output-v1 JSON manifest.")
    parser.add_argument("--camera-bundle", required=True, help="camera-bundle-v1 JSON file shared by both renders.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV reports and image artifacts.")
    parser.add_argument("--config-id", default=None, help="Immutable benchmark configuration ID.")
    parser.add_argument(
        "--max-artifact-views",
        type=int,
        default=None,
        help="Limit image artifacts to the first N views.",
    )
    parser.add_argument(
        "--skip-lpips-if-unavailable",
        action="store_true",
        help="Record missing LPIPS as a threshold failure instead of raising; acceptance still fails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    camera_bundle = load_camera_bundle(args.camera_bundle)
    report = compare_render_outputs(
        reference=load_render_output(args.reference),
        candidate=load_render_output(args.candidate),
        camera_bundle=camera_bundle,
        output_dir=Path(args.output_dir),
        config_id=args.config_id,
        require_lpips=not args.skip_lpips_if_unavailable,
        max_artifact_views=args.max_artifact_views,
    )
    print(f"fidelity_report={Path(args.output_dir) / 'fidelity_report.json'}")
    print(
        f"pass={report['pass']} "
        f"worst_view={report['worst_view']['view_id']} "
        f"worst_metric={report['worst_view']['worst_metric']}"
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
