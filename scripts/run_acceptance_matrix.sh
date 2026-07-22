#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

output_root="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/acceptance-matrix}"
warmup="${WARMUP:-10}"
iterations="${ITERATIONS:-50}"
timeout_seconds="${OVRTX_TIMEOUT_SECONDS:-1800}"
custom_case_timeout_seconds="${CUSTOM_CASE_TIMEOUT_SECONDS:-900}"
batches="${BATCHES:-1,8,32,64,128,256,1024}"
resolutions="${RESOLUTIONS:-128x128,256x256,512x512}"
scenes="${SCENES:-synthetic-small,synthetic-medium,voxel51-train-30000,home-scan-lod0}"
ovrtx_root="$output_root/ovrtx"
custom_root="$output_root/custom"
comparison_root="$output_root/comparison"
custom_rasterize_mode="${CUSTOM_RASTERIZE_MODE:-classic}"
case "$custom_rasterize_mode" in
  classic)
    default_covariance_epsilon=0
    ;;
  antialiased)
    default_covariance_epsilon=0.3
    ;;
  *)
    echo "CUSTOM_RASTERIZE_MODE must be classic or antialiased" >&2
    exit 2
    ;;
esac
custom_covariance_epsilon="${CUSTOM_COVARIANCE_EPSILON:-$default_covariance_epsilon}"
ovrtx_manifest="$ovrtx_root/cases/run-manifest.json"
custom_manifest="$custom_root/custom-cuda/run-manifest-rasterize-$custom_rasterize_mode.json"
multi_manifest="$custom_root/custom-cuda-multiscene/run-manifest-rasterize-$custom_rasterize_mode.json"
comparison_report="$comparison_root/ovrtx-custom-matrix.json"
comparison_csv="$comparison_root/ovrtx-custom-matrix.csv"
log_checker="$PROJECT_ROOT/.agents/skills/test-isaac-graphics/scripts/check-isaac-log.sh"

mkdir -p "$ovrtx_root" "$custom_root" "$comparison_root"
rm -f \
  "$ovrtx_manifest" \
  "$custom_manifest" \
  "$multi_manifest" \
  "$comparison_report" \
  "$comparison_csv"

"$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_ovrtx_matrix.py" \
  --protocol "$PROJECT_ROOT/BENCHMARK_PROTOCOL.md" \
  --output "$ovrtx_root/cases" \
  --warmup "$warmup" \
  --iterations "$iterations" \
  --scenes "$scenes" \
  --batches "$batches" \
  --resolutions "$resolutions" \
  --timeout-seconds "$timeout_seconds" \
  > "$ovrtx_root/matrix.log" 2>&1

"$ISAACSIM_PATH/python.sh" - "$ovrtx_manifest" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
result = json.loads(path.read_text(encoding="utf-8"))
if result.get("complete") is not True or result.get("case_count", 0) <= 0:
    raise SystemExit(f"OVRTX matrix manifest is incomplete: {path}")
PY
printf '%s\n' OVRTX_ACCEPTANCE_MATRIX_OK >> "$ovrtx_root/matrix.log"
bash "$log_checker" \
  "$ovrtx_root/matrix.log" \
  OVRTX_ACCEPTANCE_MATRIX_OK

custom_args=(
  --covariance-epsilon "$custom_covariance_epsilon"
  --rasterize-mode "$custom_rasterize_mode"
  --tile-size "${CUSTOM_TILE_SIZE:-1}"
  --depth-bucket-count "${CUSTOM_DEPTH_BUCKET_COUNT:-32}"
  --depth-bucket-group-size "${CUSTOM_DEPTH_BUCKET_GROUP_SIZE:-8}"
  --compact-projection-cache
  --projection-cache
  --continue-on-error
  --case-timeout-seconds "$custom_case_timeout_seconds"
)

"$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_matrix.py" \
  --protocol "$PROJECT_ROOT/BENCHMARK_PROTOCOL.md" \
  --output "$custom_root" \
  --implementations custom-cuda,custom-cuda-multiscene \
  --scenes "$scenes" \
  --batches "$batches" \
  --resolutions "$resolutions" \
  --gaussian-support-sigma "${GAUSSIAN_SUPPORT_SIGMA:-2.0}" \
  --authored-display-output \
  --warmup "$warmup" \
  --iterations "$iterations" \
  "${custom_args[@]}" \
  > "$custom_root/matrix.log" 2>&1

bash "$log_checker" \
  "$custom_root/matrix.log" \
  CUSTOM_CUDA_MATRIX_OK \
  "$custom_manifest"
bash "$log_checker" \
  "$custom_root/matrix.log" \
  CUSTOM_CUDA_MULTISCENE_MATRIX_OK \
  "$multi_manifest"

"$ISAACSIM_PATH/python.sh" \
  "$PROJECT_ROOT/benchmarks/compare_ovrtx_custom_matrix.py" \
  --ovrtx-manifest "$ovrtx_manifest" \
  --custom-root "$custom_root/custom-cuda" \
  --gaussian-support-sigma "${GAUSSIAN_SUPPORT_SIGMA:-2.0}" \
  --covariance-epsilon "$custom_covariance_epsilon" \
  --rasterize-mode "$custom_rasterize_mode" \
  --output "$comparison_report" \
  --csv-output "$comparison_csv" \
  > "$comparison_root/ovrtx-custom-matrix.log" 2>&1

"$ISAACSIM_PATH/python.sh" - "$comparison_report" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
result = json.loads(path.read_text(encoding="utf-8"))
if result.get("pass") is not True:
    raise SystemExit(f"OVRTX/custom comparison did not pass: {path}")
PY
printf '%s\n' OVRTX_CUSTOM_COMPARISON_OK \
  >> "$comparison_root/ovrtx-custom-matrix.log"
bash "$log_checker" \
  "$comparison_root/ovrtx-custom-matrix.log" \
  OVRTX_CUSTOM_COMPARISON_OK \
  "$comparison_report"
