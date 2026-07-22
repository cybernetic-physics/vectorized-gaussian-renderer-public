#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

python_bin="${PYTHON_BIN:?Set PYTHON_BIN to the prepared CUDA Python interpreter.}"
scene_path="${HOME_SCAN_PATH:?Set HOME_SCAN_PATH to the canonical Home Scan PLY.}"
scene_sha256="${HOME_SCAN_SHA256:-29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267}"
contract_root="${CONTRACT_ROOT:?Set CONTRACT_ROOT to the matrix contract directory.}"
output_root="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a unique fidelity directory.}"
source_commit="${SOURCE_GIT_COMMIT:?Set SOURCE_GIT_COMMIT to the tested commit.}"
resolutions="${RESOLUTIONS:-128 256}"
frames="${FIDELITY_STEPS:-2}"
batch="${FIDELITY_BATCH:-8}"
warmup="${WARMUP:-2}"

mkdir -p "$output_root"

for resolution in $resolutions; do
  trajectory="$contract_root/home-b1024-${resolution}.fidelity.json"
  case_root="$output_root/home-heldout-${resolution}"
  render_root="$case_root/renders"
  comparison_root="$case_root/comparison"
  if [[ -e "$case_root" ]]; then
    echo "Refusing to overwrite robotics fidelity evidence: $case_root" >&2
    exit 2
  fi
  mkdir -p "$render_root"

  run_renderer() {
    local renderer="$1"
    local run_id="$2"
    local projected_args=()
    if [[ "$renderer" == "custom" ]]; then
      projected_args=(--materialize-projected-records --max-workspace-gib 20)
    fi
    SOURCE_GIT_COMMIT="$source_commit" PYTHONPATH="$PROJECT_ROOT/src:$GSPLAT_SOURCE_PATH:${PYTHONPATH:-}" \
      "$python_bin" "$PROJECT_ROOT/benchmarks/run_trajectory.py" \
        --trajectory "$trajectory" \
        --renderer "$renderer" \
        --scenario contract-exact \
        --scene-path "$scene_path" \
        --scene-sha256 "$scene_sha256" \
        --frames "$frames" \
        --batch "$batch" \
        --width "$resolution" \
        --height "$resolution" \
        --fps 60 \
        --warmup "$warmup" \
        --repetitions 1 \
        --semantic-topology spatial-8 \
        --no-cache \
        --no-save-frames \
        --capture-full-output \
        "${projected_args[@]}" \
        --output-root "$render_root" \
        --run-id "$run_id"
  }

  run_renderer custom custom-record
  run_renderer gsplat gsplat
  custom_output="$render_root/custom-record/tensors/trajectory-output.npz"
  gsplat_output="$render_root/gsplat/tensors/trajectory-output.npz"
  test -f "$custom_output"
  test -f "$gsplat_output"

  "$python_bin" "$PROJECT_ROOT/benchmarks/compare_trajectory.py" \
    --reference "$gsplat_output" \
    --candidate "$custom_output" \
    --trajectory "$trajectory" \
    --reference-equation "gsplat classic screen-space EWA" \
    --candidate-equation "custom projected-record compact screen-space EWA" \
    --output-dir "$comparison_root"
  jq -e '.pass == true' "$comparison_root/summary.json" >/dev/null
  echo "ROBOTICS_B1024_FIDELITY_OK resolution=$resolution root=$case_root"
done
