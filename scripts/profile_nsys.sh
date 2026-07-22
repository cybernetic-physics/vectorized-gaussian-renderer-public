#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

config_id="${1:-gsplat-synth-shared-c64-n100k-128}"
num_cameras="${NUM_CAMERAS:-64}"
num_gaussians="${NUM_GAUSSIANS:-100000}"
width="${WIDTH:-128}"
height="${HEIGHT:-128}"
iterations="${ITERATIONS:-20}"
warmup="${WARMUP:-8}"
profile_root="${PROFILE_ROOT:-$PROJECT_ROOT/outputs/profiles/$config_id}"
nsys_root="$profile_root/nsys"
stats_root="$nsys_root/stats"
mkdir -p "$profile_root" "$nsys_root" "$stats_root"

if [[ "${PROFILE_WORKLOAD:-camera}" == "lidar" ]]; then
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output="$nsys_root/$config_id" \
    "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_lidar.py" \
      --scene "${LIDAR_SCENE:-medium}" \
      --batches "${BATCH:-64}" \
      --rays "${RAYS:-131072}" \
      --returns "${RETURNS:-1}" \
      --modes throughput \
      --transform-mode changing \
      --warmup "${WARMUP:-4}" \
      --iterations "${ITERATIONS:-10}" \
      --ring-size "${RING_SIZE:-4}" \
      --run-id "$config_id" \
      --output-dir "$profile_root/benchmark"

  nsys stats \
    --force-export=true \
    --force-overwrite=true \
    --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    --format csv \
    --output "$stats_root/$config_id" \
    "$nsys_root/$config_id.nsys-rep" || true

  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/scripts/parse_nsys_stats.py" \
    --stats-dir "$stats_root" \
    --output "$profile_root/${config_id}_nsys_summary.json"
  exit 0
fi

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output="$nsys_root/$config_id" \
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/profile_gsplat.py" \
    --config-id "$config_id" \
    --output-dir "$profile_root" \
    --num-cameras "$num_cameras" \
    --num-gaussians "$num_gaussians" \
    --width "$width" \
    --height "$height" \
    --warmup "$warmup" \
    --iterations "$iterations"

nsys stats \
  --force-export=true \
  --force-overwrite=true \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv \
  --output "$stats_root/$config_id" \
  "$nsys_root/$config_id.nsys-rep" || true

"$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/scripts/parse_nsys_stats.py" \
  --stats-dir "$stats_root" \
  --output "$profile_root/${config_id}_nsys_summary.json"
