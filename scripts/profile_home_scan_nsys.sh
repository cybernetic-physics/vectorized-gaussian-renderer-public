#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

config_id="${1:-home-scan-projection-cache}"
dataset="${HOME_SCAN_PATH:-/workspace/datasets/home-scan-lod0.ply}"
profile_root="${PROFILE_ROOT:-$PROJECT_ROOT/outputs/profiles/$config_id}"
nsys_root="$profile_root/nsys"
stats_root="$nsys_root/stats"
batch="${BATCH:-1}"
width="${WIDTH:-128}"
height="${HEIGHT:-128}"
visible_capacity_default=$((batch * 150000))
intersection_capacity_default=$(((
  batch * 170000 * width * height + 128 * 128 - 1
) / (128 * 128)))
if ((visible_capacity_default < 1000000)); then
  visible_capacity_default=1000000
fi
if ((intersection_capacity_default < 1000000)); then
  intersection_capacity_default=1000000
fi
mkdir -p "$profile_root" "$nsys_root" "$stats_root"

benchmark_args=(
  --path "$dataset"
  --batch "$batch"
  --width "$width"
  --height "$height"
  --warmup "${WARMUP:-1}"
  --iterations "${ITERATIONS:-20}"
  --visible-capacity "${VISIBLE_CAPACITY:-$visible_capacity_default}"
  --intersection-capacity \
    "${INTERSECTION_CAPACITY:-$intersection_capacity_default}"
  --gaussian-support-sigma "${GAUSSIAN_SUPPORT_SIGMA:-2.0}"
  --tile-size "${TILE_SIZE:-1}"
  --depth-bucket-count "${DEPTH_BUCKET_COUNT:-32}"
  --depth-bucket-group-size "${DEPTH_BUCKET_GROUP_SIZE:-8}"
  --semantic-min-alpha "${SEMANTIC_MIN_ALPHA:-0.01}"
  --semantic-scheme "${SEMANTIC_SCHEME:-spatial-grid}"
  --semantic-grid "${SEMANTIC_GRID:-2,1,1}"
  --covariance-epsilon "${COVARIANCE_EPSILON:-0}"
  --authored-display-output
  --output "$profile_root/result.json"
)
if [[ "${RAY_GAUSSIAN_EVALUATION:-0}" == "1" ]]; then
  benchmark_args+=(
    --ray-gaussian-evaluation
  )
fi
if [[ "${TIGHT_DEPTH_RANGE:-1}" == "1" ]]; then
  benchmark_args+=(--tight-depth-range)
fi
if [[ "${PROJECTION_CACHE:-1}" == "1" ]]; then
  benchmark_args+=(--projection-cache)
fi
if [[ "${COMPACT_PROJECTION_CACHE:-1}" == "1" ]]; then
  benchmark_args+=(--compact-projection-cache)
fi

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output="$nsys_root/$config_id" \
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_home_scan.py" \
  "${benchmark_args[@]}"

nsys stats \
  --force-export=true \
  --force-overwrite=true \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv \
  --output "$stats_root/$config_id" \
  "$nsys_root/$config_id.nsys-rep"

"$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/scripts/parse_nsys_stats.py" \
  --stats-dir "$stats_root" \
  --output "$profile_root/nsys-summary.json"
