#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

run_prefix="${1:-lidar-matrix}"
runs="${RUNS:-3}"
iterations="${ITERATIONS:-5}"
warmup="${WARMUP:-2}"
ring_size="${RING_SIZE:-2}"
timeout_seconds="${TIMEOUT_SECONDS:-1800}"
coordination_root="${RTX_REMOTE_COORDINATION_ROOT:-/workspace}"
lock_file="$coordination_root/.vectorized-gaussian-renderer.gpu.lock"
owner_file="$coordination_root/.vectorized-gaussian-renderer.gpu-owner"
mkdir -p "$PROJECT_ROOT/outputs/logs"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "GPU coordination lock is already held: $lock_file" >&2
  exit 75
fi
printf 'run_id=%s\nhost=%s\npid=%s\nstarted_utc=%s\nroot=%s\n' \
  "$run_prefix" "$(hostname)" "$$" "$(date -u +%FT%TZ)" "$PROJECT_ROOT" > "$owner_file"
trap 'rm -f "$owner_file"' EXIT INT TERM HUP

for scene in ${SCENES:-analytic medium}; do
  for repeat in $(seq 1 "$runs"); do
    run_id="${run_prefix}-${scene}-run${repeat}"
    timeout "$timeout_seconds" \
      "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_lidar.py" \
        --scene "$scene" \
        --warmup "$warmup" \
        --iterations "$iterations" \
        --ring-size "$ring_size" \
        --run-id "$run_id" \
        > "$PROJECT_ROOT/outputs/logs/${run_id}.log" 2>&1
    echo "LIDAR_MATRIX_RUN_OK run_id=$run_id"
  done
done

echo "LIDAR_MATRIX_OK prefix=$run_prefix"
