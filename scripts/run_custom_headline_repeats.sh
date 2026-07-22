#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

output_root="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/acceptance}"
dataset="${HOME_SCAN_PATH:-/workspace/datasets/home-scan-lod0.ply}"

common_args=(
  --path "$dataset"
  --width 128
  --height 128
  --tile-size 1
  --depth-bucket-count 32
  --depth-bucket-group-size 8
  --gaussian-support-sigma 2
  --covariance-epsilon 0
  --semantic-min-alpha 0.01
  --semantic-scheme spatial-grid
  --semantic-grid 2,1,1
  --authored-display-output
  --tight-depth-range
  --compact-projection-cache
  --projection-cache
)

for run in 1 2 3; do
  run_dir="$output_root/b1024/custom-run$run"
  mkdir -p "$run_dir"
  "$ISAACSIM_PATH/python.sh" \
    "$PROJECT_ROOT/benchmarks/run_home_scan.py" \
    "${common_args[@]}" \
    --batch 1024 \
    --warmup "${CUSTOM_WARMUP:-100}" \
    --iterations "${CUSTOM_ITERATIONS:-200}" \
    --visible-capacity 147000000 \
    --intersection-capacity 168000000 \
    --output "$run_dir/result.json" \
    > "$run_dir/run.log" 2>&1
done

single_dir="$output_root/b1/custom"
mkdir -p "$single_dir"
"$ISAACSIM_PATH/python.sh" \
  "$PROJECT_ROOT/benchmarks/run_home_scan.py" \
  "${common_args[@]}" \
  --batch 1 \
  --warmup "${SINGLE_WARMUP:-100}" \
  --iterations "${SINGLE_ITERATIONS:-500}" \
  --visible-capacity 1000000 \
  --intersection-capacity 1000000 \
  --output "$single_dir/result.json" \
  > "$single_dir/run.log" 2>&1
