#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-/isaac-sim/python.sh}"

cd "${GSPLAT_SOURCE_PATH:-/workspace/src/gsplat}"
git submodule update --init --recursive

compat_patch="$project_root/patches/gsplat-cuda-event-flags.patch"
if git apply --reverse --check "$compat_patch" >/dev/null 2>&1; then
  echo "GSPLAT_COMPAT_PATCH_ALREADY_APPLIED"
elif git apply --check "$compat_patch"; then
  git apply "$compat_patch"
  echo "GSPLAT_COMPAT_PATCH_APPLIED"
else
  echo "GSPLAT_COMPAT_PATCH_REJECTED" >&2
  exit 1
fi

export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS="${MAX_JOBS:-16}"
export BUILD_EXPERIMENTAL=0
export BUILD_3DGS=1
export BUILD_2DGS=0
export BUILD_3DGUT=0
export BUILD_ADAM=0
export BUILD_RELOC=0
export BUILD_LOSSES=0
export WITH_SYMBOLS=1

# Keep Isaac Sim's Torch/NCCL pair intact while installing the runtime
# dependencies omitted by the editable --no-deps build below.
"$python_bin" -m pip install \
  ninja \
  jaxtyping \
  nvtx \
  'rich>=12'

"$python_bin" -m pip install \
  --no-deps \
  --no-build-isolation \
  -e .

echo "GSPLAT_MAIN_BUILD_OK"
