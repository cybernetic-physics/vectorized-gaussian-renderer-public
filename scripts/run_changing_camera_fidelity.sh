#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

trajectory="${TRAJECTORY:?TRAJECTORY is required}"
scene_path="${SCENE_PATH:?SCENE_PATH is required}"
scene_sha256="${SCENE_SHA256:?SCENE_SHA256 is required}"
output_root="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
case_id="${CASE_ID:?CASE_ID is required}"
batch="${BATCH:?BATCH is required}"
frames="${FRAMES:?FRAMES is required}"
width="${WIDTH:?WIDTH is required}"
height="${HEIGHT:?HEIGHT is required}"
fps="${FPS:?FPS is required}"
warmup="${WARMUP:-2}"

render_root="$output_root/renders"
custom_run_id="$case_id-custom"
gsplat_run_id="$case_id-gsplat"
custom_root="$render_root/$custom_run_id"
gsplat_root="$render_root/$gsplat_run_id"
comparison_root="$output_root/comparison"
custom_output="$custom_root/tensors/trajectory-output.npz"
gsplat_output="$gsplat_root/tensors/trajectory-output.npz"
run_renderers="${RUN_RENDERERS:-1}"

if [[ "$run_renderers" == "1" ]]; then
  for path in "$custom_root" "$gsplat_root"; do
    if [[ -e "$path" ]]; then
      echo "Refusing to overwrite changing-camera fidelity evidence: $path" >&2
      exit 2
    fi
  done
elif [[ "$run_renderers" != "0" ]]; then
  echo "RUN_RENDERERS must be 0 or 1." >&2
  exit 2
fi
if [[ -e "$comparison_root" ]]; then
  echo "Refusing to overwrite changing-camera fidelity evidence: $comparison_root" >&2
  exit 2
fi

run_renderer() {
  local renderer="$1"
  local run_id="$2"
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_trajectory.py" \
    --trajectory "$trajectory" \
    --renderer "$renderer" \
    --scenario contract-exact \
    --scene-path "$scene_path" \
    --scene-sha256 "$scene_sha256" \
    --frames "$frames" \
    --batch "$batch" \
    --width "$width" \
    --height "$height" \
    --fps "$fps" \
    --warmup "$warmup" \
    --visible-per-view "${VISIBLE_PER_VIEW:-100000}" \
    --intersections-per-view-at-128 "${INTERSECTIONS_PER_VIEW_AT_128:-170000}" \
    --no-save-frames \
    --capture-full-output \
    --output-root "$render_root" \
    --run-id "$run_id"
}

if [[ "$run_renderers" == "1" ]]; then
  run_renderer custom "$custom_run_id"
  run_renderer gsplat "$gsplat_run_id"
fi
for path in "$custom_output" "$gsplat_output"; do
  if [[ ! -f "$path" ]]; then
    echo "Required changing-camera fidelity output is missing: $path" >&2
    exit 2
  fi
done

set +e
"$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/compare_trajectory.py" \
  --reference "$gsplat_output" \
  --candidate "$custom_output" \
  --trajectory "$trajectory" \
  --reference-equation "gsplat classic screen-space EWA" \
  --candidate-equation "custom compact-project screen-space EWA" \
  --output-dir "$comparison_root"
comparison_status=$?
set -e

echo "CHANGING_CAMERA_FIDELITY_COMPLETE case=$case_id status=$comparison_status root=$output_root"
exit "$comparison_status"
