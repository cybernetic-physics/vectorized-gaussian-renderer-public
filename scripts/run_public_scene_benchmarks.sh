#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

output_root="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/public-scene}"
dataset="${PUBLIC_SCENE_PATH:-/workspace/datasets/public-gaussian/Voxel51_gaussian_splatting/FO_dataset/train/point_cloud/iteration_30000/point_cloud.ply}"
batch="${BATCH:-1}"
width="${WIDTH:-128}"
height="${HEIGHT:-128}"
mkdir -p "$output_root/custom-artifacts" "$output_root/ovrtx"

custom_args=(
  --covariance-epsilon "${COVARIANCE_EPSILON:-0}"
)
if [[ "${RAY_GAUSSIAN_EVALUATION:-0}" == "1" ]]; then
  custom_args+=(
    --ray-gaussian-evaluation
  )
fi
if [[ "${COMPACT_PROJECTION_CACHE:-1}" == "1" ]]; then
  custom_args+=(--compact-projection-cache)
fi
if [[ "${PROJECTION_CACHE:-1}" == "1" ]]; then
  custom_args+=(--projection-cache)
fi

"$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_home_scan.py" \
  --dataset-id voxel51-train-30000 \
  --path "$dataset" \
  --batch "$batch" \
  --width "$width" \
  --height "$height" \
  --warmup "${CUSTOM_WARMUP:-5}" \
  --iterations "${CUSTOM_ITERATIONS:-20}" \
  --intersections-per-visible "${INTERSECTIONS_PER_VISIBLE:-8}" \
  --tile-size "${TILE_SIZE:-1}" \
  --depth-bucket-count "${DEPTH_BUCKET_COUNT:-32}" \
  --depth-bucket-group-size "${DEPTH_BUCKET_GROUP_SIZE:-8}" \
  --gaussian-support-sigma "${GAUSSIAN_SUPPORT_SIGMA:-2.0}" \
  --semantic-min-alpha "${SEMANTIC_MIN_ALPHA:-0.01}" \
  --authored-display-output \
  --output "$output_root/custom-b${batch}-${width}x${height}.json" \
  --artifact-dir "$output_root/custom-artifacts" \
  "${custom_args[@]}"

set +e
timeout "${OVRTX_TIMEOUT_SECONDS:-1200}" \
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_ovrtx.py" \
    --protocol "$PROJECT_ROOT/BENCHMARK_PROTOCOL.md" \
    --output "$output_root/ovrtx" \
    --scene voxel51-train-30000 \
    --public-scene-path "$dataset" \
    --batch-size "$batch" \
    --width "$width" \
    --height "$height" \
    --warmup "${OVRTX_WARMUP:-3}" \
    --iterations "${OVRTX_ITERATIONS:-10}" \
    --aa-op none \
    > "$output_root/ovrtx.log" 2>&1
ovrtx_status=$?
set -e

python3 - "$output_root/ovrtx-status.json" "$ovrtx_status" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
returncode = int(sys.argv[2])
payload = {
    "schema_version": "public-scene-ovrtx-status/v1",
    "returncode": returncode,
    "pass": returncode == 0,
    "log": str(path.with_name("ovrtx.log")),
}
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

exit "$ovrtx_status"
