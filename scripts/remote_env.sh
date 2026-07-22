#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda-12.8
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  else
    # Preserve the pinned target in diagnostics on an incompletely provisioned
    # node; callers may explicitly override CUDA_HOME for another matched
    # Torch/toolkit pair.
    CUDA_HOME=/usr/local/cuda-12.8
  fi
fi
export CUDA_HOME
export NSIGHT_COMPUTE_HOME=/opt/nvidia/nsight-compute/2025.1.1
export NSIGHT_SYSTEMS_HOME=/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64
export NSIGHT_GRAPHICS_HOME=/opt/nvidia/nsight-graphics-for-linux/nsight-graphics-for-linux-2026.2.0.0/NVIDIA-Nsight-Graphics-2026.2/host/linux-desktop-nomad-x64
export PATH="$CUDA_HOME/bin:$NSIGHT_COMPUTE_HOME:$NSIGHT_SYSTEMS_HOME:$NSIGHT_GRAPHICS_HOME:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ALLOW_ROOT=1
export ISAACSIM_PATH=/isaac-sim
export ISAACSIM_ML_PREBUNDLE=/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle
export OVRTX_ROOT="${OVRTX_ROOT:-/workspace/tools/ovrtx-0.3.0}"
export GSPLAT_SOURCE_PATH="${GSPLAT_SOURCE_PATH:-/workspace/src/gsplat}"
export VGR_PROJECT_ROOT="${VGR_PROJECT_ROOT:-$PROJECT_ROOT}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$VGR_PROJECT_ROOT/build/warp-cache}"
export PYTHONPATH="$OVRTX_ROOT/python:$ISAACSIM_ML_PREBUNDLE:$VGR_PROJECT_ROOT/src:$GSPLAT_SOURCE_PATH:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$OVRTX_ROOT/bin:$LD_LIBRARY_PATH"

# OVRTX/Vulkan needs a valid XDG_RUNTIME_DIR or vkCreateInstance warns and can
# fail on compute-only hosts with no session runtime dir.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime}"
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null || true
chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

# OVRTX keeps multiple Vulkan/renderer descriptors per RenderProduct. Vast
# containers expose a high hard limit but start with a low soft descriptor
# limit, so make large-batch capacity tests reproducible.
ulimit -n 65536 2>/dev/null || true
