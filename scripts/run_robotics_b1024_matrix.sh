#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

python_bin="${PYTHON_BIN:?Set PYTHON_BIN to the prepared CUDA Python interpreter.}"
scene_path="${HOME_SCAN_PATH:?Set HOME_SCAN_PATH to the canonical Home Scan PLY.}"
scene_sha256="${HOME_SCAN_SHA256:-29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267}"
output_root="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a unique evidence directory.}"
source_commit="${SOURCE_GIT_COMMIT:?Set SOURCE_GIT_COMMIT to the tested commit.}"
route="${ROBOTICS_ROUTE:-$PROJECT_ROOT/benchmarks/camera_paths/home-scan-walkthrough-v1.json}"
resolutions="${RESOLUTIONS:-128 256}"
sample_counts="${SAMPLE_COUNTS:-1000000 0}"
sample_batches="${SAMPLE_BATCHES:-1 8 32 128}"
full_batches="${FULL_BATCHES:-1 4 8 16 32}"
measured_steps="${MEASURED_STEPS:-2}"
step_stride="${STEP_STRIDE:-37}"
repetitions="${REPETITIONS_PER_PROCESS:-1}"
headline_extra_processes="${HEADLINE_EXTRA_PROCESSES:-2}"
case_timeout="${CASE_TIMEOUT:-45m}"
run_semantic_stress="${RUN_SEMANTIC_STRESS:-1}"

mkdir -p "$output_root/contracts" "$output_root/results" "$output_root/logs"

build_contract() {
  local resolution="$1"
  local contract="$output_root/contracts/home-b1024-${resolution}.json"
  if [[ ! -f "$contract" ]]; then
    SOURCE_GIT_COMMIT="$source_commit" PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" \
      "$python_bin" "$PROJECT_ROOT/benchmarks/build_robotics_batch_contract.py" \
        --route "$route" \
        --scene-sha256 "$scene_sha256" \
        --logical-batch 1024 \
        --route-samples 1024 \
        --measured-steps "$measured_steps" \
        --step-stride "$step_stride" \
        --width "$resolution" \
        --height "$resolution" \
        --fidelity-batch 8 \
        --output "$contract"
  fi
  jq -e '.shape.batch == 1024 and .motion_classification == "logical-batch-independent-moving-camera"' \
    "$contract" >/dev/null
}

run_case() {
  local variant="$1"
  local resolution="$2"
  local sample_count="$3"
  local physical_batch="$4"
  local process_index="$5"
  local semantic_topology="${6:-spatial-8}"
  local sample_label="full"
  if [[ "$sample_count" != "0" ]]; then
    sample_label="sample-${sample_count}"
  fi
  local case_id="${variant}-${sample_label}-${semantic_topology}-${resolution}-p${physical_batch}-r${process_index}"
  local result="$output_root/results/$case_id.json"
  local log="$output_root/logs/$case_id.log"
  local renderer="gsplat"
  local materialize=(--materialize-projected-records)
  if [[ "$variant" == "custom-record" ]]; then
    renderer="custom"
  elif [[ "$variant" == "custom-recompute" ]]; then
    renderer="custom"
    materialize=(--no-materialize-projected-records)
  elif [[ "$variant" != "gsplat" ]]; then
    echo "Unknown renderer variant: $variant" >&2
    return 2
  fi
  if [[ -e "$result" || -e "$log" ]]; then
    echo "Refusing to overwrite robotics B1024 evidence: $case_id" >&2
    return 2
  fi

  set +e
  SOURCE_GIT_COMMIT="$source_commit" PYTHONPATH="$PROJECT_ROOT/src:$GSPLAT_SOURCE_PATH:${PYTHONPATH:-}" \
    timeout --signal=TERM "$case_timeout" \
    "$python_bin" "$PROJECT_ROOT/benchmarks/run_logical_batch.py" \
      --trajectory "$output_root/contracts/home-b1024-${resolution}.json" \
      --renderer "$renderer" \
      --scene-path "$scene_path" \
      --scene-sha256 "$scene_sha256" \
      --physical-batch "$physical_batch" \
      --semantic-topology "$semantic_topology" \
      --sample-gaussians "$sample_count" \
      --warmup-logical-batches 1 \
      --repetitions "$repetitions" \
      --max-workspace-gib 20 \
      "${materialize[@]}" \
      --output "$result" \
      >"$log" 2>&1
  local status=$?
  set -e
  if [[ ! -f "$result" ]]; then
    echo "Case $case_id exited $status without a result artifact." >&2
    return "$status"
  fi
  if jq -e '.pass == true' "$result" >/dev/null; then
    echo "ROBOTICS_B1024_CASE_OK id=$case_id"
    return 0
  fi
  echo "ROBOTICS_B1024_CASE_FAILED id=$case_id status=$status"
  return 0
}

for resolution in $resolutions; do
  build_contract "$resolution"
done

for resolution in $resolutions; do
  for sample_count in $sample_counts; do
    batches="$sample_batches"
    if [[ "$sample_count" == "0" ]]; then
      batches="$full_batches"
    fi
    for physical_batch in $batches; do
      run_case custom-record "$resolution" "$sample_count" "$physical_batch" 1
      run_case gsplat "$resolution" "$sample_count" "$physical_batch" 1
    done
    run_case custom-recompute "$resolution" "$sample_count" 8 1
  done
done

for process_index in $(seq 2 "$((headline_extra_processes + 1))"); do
  for resolution in $resolutions; do
    for sample_count in $sample_counts; do
      run_case custom-record "$resolution" "$sample_count" 8 "$process_index"
      run_case gsplat "$resolution" "$sample_count" 8 "$process_index"
    done
  done
done

if [[ "$run_semantic_stress" == "1" ]]; then
  run_case custom-record 128 1000000 8 1 interleaved-1024
  run_case gsplat 128 1000000 8 1 interleaved-1024
elif [[ "$run_semantic_stress" != "0" ]]; then
  echo "RUN_SEMANTIC_STRESS must be 0 or 1." >&2
  exit 2
fi

echo "ROBOTICS_B1024_MATRIX_COMPLETE root=$output_root"
