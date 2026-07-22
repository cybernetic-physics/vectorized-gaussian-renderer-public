#!/usr/bin/env bash
# Build the batched OptiX gaussian tracer on the node.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OPTIX_INC="${OPTIX_INC:-/workspace/tools/optix-dev/include}"
CUDA="${CUDA_HOME:-/usr/local/cuda-12.8}"

# NVCC_EXTRA lets sweeps override compile-time knobs, e.g. NVCC_EXTRA=-DKBUF=16
"$CUDA/bin/nvcc" -ptx -arch=sm_89 -O3 --use_fast_math -std=c++17 \
  ${NVCC_EXTRA:-} \
  -I"$OPTIX_INC" -I"$HERE" \
  "$HERE/gs_tracer_device.cu" -o "$HERE/gs_tracer_device.ptx"

g++ -O2 -std=c++17 \
  -I"$OPTIX_INC" -I"$CUDA/include" -I"$HERE" \
  "$HERE/gs_tracer_main.cpp" \
  -L"$CUDA/lib64" -lcudart -ldl -o "$HERE/gs_tracer"

# Zero-copy library for in-process (torch/ctypes) use. NVCC compiles the tiny
# camera-unpack kernel that keeps CUDA view matrices and intrinsics on-device.
"$CUDA/bin/nvcc" -O2 -std=c++17 -shared -Xcompiler=-fPIC -x cu \
  -I"$OPTIX_INC" -I"$CUDA/include" -I"$HERE" \
  "$HERE/gs_tracer_lib.cpp" \
  -L"$CUDA/lib64" -lcudart -ldl -o "$HERE/libgs_tracer.so"

echo "OPTIX_TRACER_BUILD_OK $HERE/gs_tracer $HERE/libgs_tracer.so"
