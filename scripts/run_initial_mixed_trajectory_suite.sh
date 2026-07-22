#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

output_root="${OUTPUT_ROOT:-/workspace/eval-artifacts}"
route="${ROUTE:-$PROJECT_ROOT/benchmarks/camera_paths/home-scan-representative-v1.json}"
dataset="${HOME_SCAN_PATH:-/workspace/datasets/home-scan-lod0.ply}"

for fraction in 0.01 0.25 0.50 1.00; do
  label="${fraction/./p}"
  run_id="dynamic-mixed-b256-t120-moved-$label"
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_trajectory.py" \
    --route "$route" \
    --scene-path "$dataset" \
    --renderer custom \
    --scenario mixed-motion \
    --moved-fraction "$fraction" \
    --frames 120 \
    --batch 256 \
    --width 128 \
    --height 128 \
    --warmup 4 \
    --visible-per-view 150000 \
    --intersections-per-view-at-128 1600000 \
    --output-root "$output_root" \
    --run-id "$run_id"
done
