#!/usr/bin/env bash
set -euo pipefail

# Launch the decisive matrix from a deliberately tiny, portable environment.
# Nsight records target-process environment strings, so inheriting an SSH,
# notebook, cloud, or workstation session is forbidden for public evidence.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
publication_python="${VGR_PUBLICATION_PYTHON:?set VGR_PUBLICATION_PYTHON}"
scene_path="${HOME_SCAN_PATH:?set HOME_SCAN_PATH}"
gsplat_source="${GSPLAT_SOURCE_PATH:?set GSPLAT_SOURCE_PATH}"
flashgs_source="${FLASHGS_SOURCE_PATH:?set FLASHGS_SOURCE_PATH}"
output_root="${PUBLICATION_OUTPUT_ROOT:?set PUBLICATION_OUTPUT_ROOT}"
expected_gpu_uuid="${PUBLICATION_GPU_UUID:?set PUBLICATION_GPU_UUID}"
safe_home="${PUBLICATION_SAFE_HOME:-/workspace/vgr-publication-home}"

for required_path in \
  "$PROJECT_ROOT" \
  "$publication_python" \
  "$scene_path" \
  "$gsplat_source" \
  "$flashgs_source" \
  "$output_root" \
  "$safe_home"; do
  case "$required_path" in
    *"/Users/"*|*"/home/"*)
      echo "Publication path contains a host-user identity: $required_path" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$publication_python" ]]; then
  echo "Publication Python is not executable: $publication_python" >&2
  exit 2
fi
if [[ ! -f "$scene_path" || ! -d "$gsplat_source" || ! -d "$flashgs_source" ]]; then
  echo "Scene, gsplat, or FlashGS input is missing." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
if [[ "$(pwd -P)" != "$PROJECT_ROOT" ]]; then
  echo "Publication launcher did not enter its frozen source root." >&2
  exit 2
fi

cuda_home="${MATCHED_CUDA_HOME:-/usr/local/cuda}"
nsight_compute_home="/opt/nvidia/nsight-compute/2025.1.1"
nsight_systems_home="/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64"
nsight_graphics_home="/opt/nvidia/nsight-graphics-for-linux/nsight-graphics-for-linux-2026.2.0.0/NVIDIA-Nsight-Graphics-2026.2/host/linux-desktop-nomad-x64"
safe_tmp="$output_root/tmp"
safe_xdg="$output_root/xdg-runtime"
source_manifest="$output_root/provenance/source-manifest.json"

for runtime_path in "$cuda_home"; do
  case "$runtime_path" in
    *"/Users/"*|*"/home/"*)
      echo "Publication runtime path contains a host-user identity: $runtime_path" >&2
      exit 2
      ;;
  esac
done

mkdir -p \
  "$safe_home" \
  "$safe_tmp" \
  "$safe_xdg" \
  "$output_root/provenance"
chmod 700 "$safe_home" "$safe_tmp" "$safe_xdg"

safe_path="$(dirname "$publication_python"):$cuda_home/bin:$nsight_compute_home:$nsight_systems_home:$nsight_graphics_home:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
safe_pythonpath="$PROJECT_ROOT/src:$gsplat_source"
safe_environment=(
  "CUDA_HOME=$cuda_home"
  "CUDA_VISIBLE_DEVICES=0"
  "HOME=$safe_home"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "LOGNAME=vgr-publication"
  "MATCHED_CUDA_HOME=$cuda_home"
  "MATCHED_PUBLICATION_RUNTIME=1"
  "NSIGHT_COMPUTE_HOME=$nsight_compute_home"
  "NSIGHT_GRAPHICS_HOME=$nsight_graphics_home"
  "NSIGHT_SYSTEMS_HOME=$nsight_systems_home"
  "PATH=$safe_path"
  "PROJECT_ROOT=$PROJECT_ROOT"
  "PYTHONDONTWRITEBYTECODE=1"
  "PYTHONNOUSERSITE=1"
  "PYTHONPATH=$safe_pythonpath"
  "TMPDIR=$safe_tmp"
  "TORCH_HOME=$safe_home/torch-cache"
  "TORCH_CUDA_ARCH_LIST=8.9"
  "USER=vgr-publication"
  "VGR_GPU_EXECUTOR_LOCK_PATH=/tmp/vgr-publication-gpu-executor.lock"
  "VGR_PROJECT_ROOT=$PROJECT_ROOT"
  "XDG_RUNTIME_DIR=$safe_xdg"
)

env -i "${safe_environment[@]}" \
  "$publication_python" \
  "$PROJECT_ROOT/scripts/write_source_manifest.py" \
  --repo-root "$PROJECT_ROOT" \
  --output "$source_manifest"

echo "PUBLICATION_MATRIX_ENVIRONMENT_SAFE"
exec env -i "${safe_environment[@]}" \
  "$publication_python" \
  "$PROJECT_ROOT/benchmarks/run_flashgs_matched_matrix.py" \
  --scene-path "$scene_path" \
  --source-manifest "$source_manifest" \
  --expected-gpu-uuid "$expected_gpu_uuid" \
  --output-root "$output_root" \
  --python "$publication_python" \
  --gsplat-source "$gsplat_source" \
  --flashgs-source "$flashgs_source"
