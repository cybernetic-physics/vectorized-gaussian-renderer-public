#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

output_root="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/acceptance}"
dataset="${HOME_SCAN_PATH:-/workspace/datasets/home-scan-lod0.ply}"

run_case() {
  local output_dir="$1"
  local batch="$2"
  local warmup="$3"
  local iterations="$4"

  mkdir -p "$output_dir"
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_ovrtx.py" \
    --protocol "$PROJECT_ROOT/BENCHMARK_PROTOCOL.md" \
    --output "$output_dir" \
    --scene home-scan-lod0 \
    --home-scan-path "$dataset" \
    --batch-size "$batch" \
    --width 128 \
    --height 128 \
    --warmup "$warmup" \
    --iterations "$iterations" \
    --semantic-scheme spatial-grid \
    --semantic-grid 2,1,1 \
    --aa-op none \
    --skip-reference-artifacts \
    > "$output_dir/run.log" 2>&1
}

for run in 1 2 3; do
  run_case \
    "$output_root/b1024/ovrtx-run$run" \
    1024 \
    "${OVRTX_WARMUP:-1}" \
    "${OVRTX_ITERATIONS:-3}"
done

run_case \
  "$output_root/b1/ovrtx" \
  1 \
  "${SINGLE_OVRTX_WARMUP:-20}" \
  "${SINGLE_OVRTX_ITERATIONS:-100}"
