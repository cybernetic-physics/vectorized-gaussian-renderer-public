#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

dataset_root="${DATASET_ROOT:-/workspace/datasets}"
output_root="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/changing-camera-acceptance}"
source_commit="${SOURCE_GIT_COMMIT:?Set SOURCE_GIT_COMMIT to the tested renderer commit.}"
scenes="${SCENES:-home-scan-lod0 garage-high-detail}"
batches="${BATCHES:-1 4 8}"
max_workspace_bytes="${MAX_WORKSPACE_BYTES:-12884901888}"
run_custom="${RUN_CUSTOM:-1}"
run_gsplat="${RUN_GSPLAT:-1}"

if [[ ! "$run_custom" =~ ^[01]$ || ! "$run_gsplat" =~ ^[01]$ ]]; then
  echo "RUN_CUSTOM and RUN_GSPLAT must each be 0 or 1." >&2
  exit 2
fi
if [[ "$run_custom" == 0 && "$run_gsplat" == 0 ]]; then
  echo "At least one renderer must be enabled." >&2
  exit 2
fi

run_case() {
  local scene="$1"
  local batch="$2"
  local width height source_frames scene_path scene_manifest camera_contract
  local camera_manifest scene_sha256 scene_author scene_license scene_source

  case "$scene" in
    home-scan-lod0)
      width=384
      height=384
      source_frames=240
      scene_path="$dataset_root/$scene/home-scan-lod0.ply"
      scene_manifest="$dataset_root/$scene/manifest.json"
      camera_contract="$dataset_root/$scene/camera-path.npz"
      camera_manifest="$dataset_root/$scene/camera-path.json"
      scene_sha256="29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
      scene_author="Isaiah Sweeney"
      scene_license="CC BY 4.0"
      scene_source="https://superspl.at/scene/3f89bbd3"
      ;;
    garage-high-detail)
      width=256
      height=256
      source_frames=72
      scene_path="$dataset_root/$scene/garage-high-detail.ply"
      scene_manifest="$dataset_root/$scene/manifest.json"
      camera_contract="$dataset_root/$scene/camera-path.npz"
      camera_manifest="$dataset_root/$scene/camera-path.json"
      scene_sha256="cf8d65138dbc31e3619ce697a875b1a005fe88fa126d5737e66294b9064a171a"
      scene_author="Ethan"
      scene_license="CC BY 4.0"
      scene_source="https://superspl.at/scene/8f4d8e05"
      ;;
    *)
      echo "Unsupported acceptance scene: $scene" >&2
      return 2
      ;;
  esac

  if (( source_frames % batch != 0 )); then
    echo "$scene has $source_frames source views, which is not divisible by B$batch." >&2
    return 2
  fi
  for path in "$scene_path" "$scene_manifest" "$camera_contract" "$camera_manifest"; do
    test -f "$path" || { echo "Missing required input: $path" >&2; return 2; }
  done

  local case_root="$output_root/$scene/b${batch}-${width}x${height}"
  local custom_json="$case_root/custom.json"
  local trajectory_json="$case_root/custom.trajectory.json"
  local gsplat_root="$case_root/gsplat"
  local gsplat_run_id="gsplat-contract-exact-b${batch}-${width}x${height}"
  mkdir -p "$case_root"
  if [[ "$run_custom" == 1 && -e "$custom_json" ]]; then
    echo "Refusing to overwrite custom evidence in $case_root" >&2
    return 2
  fi
  if [[ "$run_gsplat" == 1 && -e "$gsplat_root/$gsplat_run_id" ]]; then
    echo "Refusing to overwrite gsplat evidence in $case_root" >&2
    return 2
  fi

  if [[ "$run_custom" == 1 ]]; then
    SOURCE_GIT_COMMIT="$source_commit" OMNI_KIT_ACCEPT_EULA=YES \
      "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/scripts/isaacsim_vectorized_example.py" \
        --batch "$batch" \
        --width "$width" \
        --height "$height" \
        --warmup 5 \
        --iterations 1 \
        --physics-steps 1 \
        --deadline-ms 41.6666667 \
        --tile-size 1 \
        --semantic-min-alpha 0.01 \
        --enable-projection-cache \
        --compact-projection-cache \
        --no-deterministic \
        --camera-contract "$camera_contract" \
        --camera-manifest "$camera_manifest" \
        --scene-path "$scene_path" \
        --scene-manifest "$scene_manifest" \
        --scene-id-label "$scene" \
        --scene-author "$scene_author" \
        --scene-license "$scene_license" \
        --scene-source "$scene_source" \
        --trajectory-layout view-population-pack \
        --visible-per-view 100000 \
        --intersections-per-view-at-128 170000 \
        --max-workspace-bytes "$max_workspace_bytes" \
        --output "$custom_json"
  fi

  jq -e '.pass == true' "$custom_json" >/dev/null
  test -f "$trajectory_json"

  if [[ "$run_gsplat" == 1 ]]; then
    SOURCE_GIT_COMMIT="$source_commit" \
      "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/run_trajectory.py" \
        --trajectory "$trajectory_json" \
        --renderer gsplat \
        --scenario contract-exact \
        --scene-path "$scene_path" \
        --scene-sha256 "$scene_sha256" \
        --frames "$((source_frames / batch))" \
        --batch "$batch" \
        --width "$width" \
        --height "$height" \
        --fps 24 \
        --warmup 5 \
        --repetitions 1 \
        --no-save-frames \
        --output-root "$gsplat_root" \
        --run-id "$gsplat_run_id"

    jq -e '.pass == true' "$gsplat_root/$gsplat_run_id/manifest.json" >/dev/null
  fi
  echo "CHANGING_CAMERA_ACCEPTANCE_OK scene=$scene batch=$batch root=$case_root"
}

for scene in $scenes; do
  for batch in $batches; do
    run_case "$scene" "$batch"
  done
done
