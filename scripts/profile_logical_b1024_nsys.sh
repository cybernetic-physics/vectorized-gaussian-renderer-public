#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

config_id="${1:-logical-b1024-home-128-p8}"
python_bin="${PYTHON_BIN:?Set PYTHON_BIN to the prepared CUDA Python interpreter.}"
trajectory="${TRAJECTORY:?Set TRAJECTORY to the frozen logical B1024 contract.}"
scene_path="${HOME_SCAN_PATH:?Set HOME_SCAN_PATH to the canonical Home Scan PLY.}"
scene_sha256="${HOME_SCAN_SHA256:-29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267}"
source_commit="${SOURCE_GIT_COMMIT:?Set SOURCE_GIT_COMMIT to the tested commit.}"
physical_batch="${PHYSICAL_BATCH:-8}"
sample_gaussians="${SAMPLE_GAUSSIANS:-0}"
profile_root="${PROFILE_ROOT:-$PROJECT_ROOT/outputs/profiles/$config_id}"
nsys_root="$profile_root/nsys"
stats_root="$nsys_root/stats"

if [[ -e "$profile_root" ]]; then
  echo "Refusing to overwrite logical B1024 profile evidence: $profile_root" >&2
  exit 2
fi
mkdir -p "$nsys_root" "$stats_root"

SOURCE_GIT_COMMIT="$source_commit" \
VGR_CUDA_PROFILER_RANGE=1 \
PYTHONPATH="$PROJECT_ROOT/src:$GSPLAT_SOURCE_PATH:${PYTHONPATH:-}" \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=false \
  --output="$nsys_root/$config_id" \
  "$python_bin" "$PROJECT_ROOT/benchmarks/run_logical_batch.py" \
    --trajectory "$trajectory" \
    --renderer custom \
    --scene-path "$scene_path" \
    --scene-sha256 "$scene_sha256" \
    --physical-batch "$physical_batch" \
    --semantic-topology spatial-8 \
    --sample-gaussians "$sample_gaussians" \
    --warmup-logical-batches 1 \
    --repetitions 1 \
    --materialize-projected-records \
    --max-workspace-gib 20 \
    --output "$profile_root/result.json"

nsys stats \
  --force-export=true \
  --force-overwrite=true \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv \
  --output "$stats_root/$config_id" \
  "$nsys_root/$config_id.nsys-rep" || true

"$python_bin" "$PROJECT_ROOT/scripts/parse_nsys_stats.py" \
  --stats-dir "$stats_root" \
  --output "$profile_root/nsys-summary.json"

echo "LOGICAL_B1024_NSYS_OK root=$profile_root"
