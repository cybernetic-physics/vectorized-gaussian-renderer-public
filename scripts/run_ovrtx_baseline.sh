#!/usr/bin/env bash
set -euo pipefail

project_root="${VGR_PROJECT_ROOT:-/workspace/vectorized-gaussian-renderer}"
source "$project_root/scripts/remote_env.sh"
cd "$project_root"

if (($# == 0)); then
    set -- \
        --protocol BENCHMARK_PROTOCOL.md \
        --output outputs/baselines/ovrtx-fair-b8-128 \
        --scene synthetic-small \
        --batch-size 8 \
        --width 128 \
        --height 128 \
        --warmup 10 \
        --iterations 50 \
        --aa-op none
fi

exec /isaac-sim/python.sh benchmarks/run_ovrtx.py "$@"
