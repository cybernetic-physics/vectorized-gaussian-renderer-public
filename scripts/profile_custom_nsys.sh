#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

config_id="${1:-custom-synth-shared-c64-n100k-128-nsys}"
batch="${BATCH:-64}"
scene="${SCENE:-synthetic-medium}"
width="${WIDTH:-128}"
height="${HEIGHT:-128}"
iterations="${ITERATIONS:-20}"
warmup="${WARMUP:-8}"
profile_root="${PROFILE_ROOT:-$PROJECT_ROOT/outputs/profiles/$config_id}"
nsys_root="$profile_root/nsys"
stats_root="$nsys_root/stats"
mkdir -p "$profile_root" "$nsys_root" "$stats_root"

profile_args=(
  --covariance-epsilon "${COVARIANCE_EPSILON:-0}"
  --depth-bucket-count "${DEPTH_BUCKET_COUNT:-32}"
  --depth-bucket-group-size "${DEPTH_BUCKET_GROUP_SIZE:-8}"
)
if [[ -n "${INTERSECTIONS_PER_VISIBLE:-}" ]]; then
  profile_args+=(--intersections-per-visible "$INTERSECTIONS_PER_VISIBLE")
fi
if [[ -n "${VISIBLE_CAPACITY:-}" ]]; then
  profile_args+=(--visible-capacity "$VISIBLE_CAPACITY")
fi
if [[ -n "${INTERSECTION_CAPACITY:-}" ]]; then
  profile_args+=(--intersection-capacity "$INTERSECTION_CAPACITY")
fi
if [[ "${RAY_GAUSSIAN_EVALUATION:-0}" == "1" ]]; then
  profile_args+=(
    --ray-gaussian-evaluation
  )
fi
if [[ "${COMPACT_PROJECTION_CACHE:-1}" == "1" ]]; then
  profile_args+=(--compact-projection-cache)
fi
if [[ "${PROJECTION_CACHE:-1}" == "1" ]]; then
  profile_args+=(--projection-cache)
fi

# Capture the complete short-lived process. Nsight Systems 2024.6 can finish
# an NVTX capture-range run without emitting the report when Python pops the
# terminal range; the named NVTX ranges are still retained for inspection.
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output="$nsys_root/$config_id" \
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/profile_custom.py" \
    --config-id "$config_id" \
    --output-dir "$profile_root" \
    --scene "$scene" \
    --batch "$batch" \
    --width "$width" \
    --height "$height" \
    --gaussian-support-sigma "${GAUSSIAN_SUPPORT_SIGMA:-3.0}" \
    --tile-size "${TILE_SIZE:-1}" \
    --authored-display-output \
    --warmup "$warmup" \
    --iterations "$iterations" \
    "${profile_args[@]}"

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
