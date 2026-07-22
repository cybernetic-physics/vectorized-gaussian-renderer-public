#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

config_id="${1:-trajectory-static-b256}"
scenario="${2:-${SCENARIO:-static-repeat}}"
profile_root="${PROFILE_ROOT:-$PROJECT_ROOT/outputs/profiles/$config_id}"
route="${ROUTE:-$PROJECT_ROOT/benchmarks/camera_paths/home-scan-representative-v1.json}"
trajectory="${TRAJECTORY:-}"
dataset="${HOME_SCAN_PATH:-/workspace/datasets/home-scan-lod0.ply}"
batch="${BATCH:-256}"
frames="${FRAMES:-10}"
width="${WIDTH:-128}"
height="${HEIGHT:-128}"
fps="${FPS:-60.0}"
python_bin="${PYTHON_BIN:-${ISAACSIM_PATH:-/isaac-sim}/python.sh}"
mkdir -p "$profile_root/nsys/stats" "$profile_root/artifacts"

trajectory_args=(--route "$route")
if [[ -n "$trajectory" ]]; then
  trajectory_args=(--trajectory "$trajectory")
fi

scene_args=()
if [[ -n "${SCENE_SHA256:-}" ]]; then
  scene_args+=(--scene-sha256 "$SCENE_SHA256")
fi

renderer_args=()
if [[ "${MATERIALIZE_PROJECTED_RECORDS:-0}" == "1" ]]; then
  renderer_args+=(--materialize-projected-records)
elif [[ "${MATERIALIZE_PROJECTED_RECORDS:-0}" != "0" ]]; then
  echo "MATERIALIZE_PROJECTED_RECORDS must be 0 or 1." >&2
  exit 2
fi
if [[ -n "${MAX_WORKSPACE_GIB:-}" ]]; then
  renderer_args+=(--max-workspace-gib "$MAX_WORKSPACE_GIB")
fi

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output="$profile_root/nsys/$config_id" \
  "$python_bin" "$PROJECT_ROOT/benchmarks/run_trajectory.py" \
    "${trajectory_args[@]}" \
    --scene-path "$dataset" \
    "${scene_args[@]}" \
    --renderer custom \
    --scenario "$scenario" \
    --frames "$frames" \
    --batch "$batch" \
    --width "$width" \
    --height "$height" \
    --fps "$fps" \
    --warmup "${WARMUP:-2}" \
    --repetitions "${REPETITIONS:-1}" \
    --minimum-duration-seconds "${MINIMUM_DURATION_SECONDS:-0}" \
    --visible-per-view "${VISIBLE_PER_VIEW:-150000}" \
    --intersections-per-view-at-128 "${INTERSECTIONS_PER_VIEW_AT_128:-170000}" \
    "${renderer_args[@]}" \
    --no-save-frames \
    --output-root "$profile_root/artifacts" \
    --run-id "$config_id"

nsys stats \
  --force-export=true \
  --force-overwrite=true \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv \
  --output "$profile_root/nsys/stats/$config_id" \
  "$profile_root/nsys/$config_id.nsys-rep" || true

"$python_bin" "$PROJECT_ROOT/scripts/parse_nsys_stats.py" \
  --stats-dir "$profile_root/nsys/stats" \
  --output "$profile_root/nsys-summary.json"
