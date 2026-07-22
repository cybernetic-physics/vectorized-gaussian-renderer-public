#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

profile_root="${PROFILE_ROOT:-$PROJECT_ROOT/outputs/profiles/matrix}"
mkdir -p "$profile_root"

run_point() {
  local config_id="$1"
  local cameras="$2"
  local gaussians="$3"
  local width="$4"
  local height="$5"
  local iterations="$6"
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/profile_gsplat.py" \
    --config-id "$config_id" \
    --output-dir "$profile_root" \
    --num-cameras "$cameras" \
    --num-gaussians "$gaussians" \
    --width "$width" \
    --height "$height" \
    --warmup "${WARMUP:-10}" \
    --iterations "$iterations" \
    ${TORCH_PROFILER:+--torch-profiler}
}

run_point gsplat-synth-shared-c1-n100k-128 1 100000 128 128 "${ITERATIONS_SMALL:-50}"
run_point gsplat-synth-shared-c8-n100k-128 8 100000 128 128 "${ITERATIONS_SMALL:-50}"
run_point gsplat-synth-shared-c64-n100k-128 64 100000 128 128 "${ITERATIONS_SMALL:-50}"
run_point gsplat-synth-shared-c32-n100k-256 32 100000 256 256 "${ITERATIONS_MEDIUM:-30}"
