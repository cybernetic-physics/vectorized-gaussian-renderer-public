#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

output_root="${OUTPUT_ROOT:-/workspace/eval-artifacts/isaac-fabric-b256}"
mkdir -p "$output_root"

for mode in static mixed-motion fully-moving; do
  extra=()
  if [[ "$mode" == "mixed-motion" ]]; then
    extra+=(--moving-fraction 0.5)
  fi
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/scripts/isaacsim_vectorized_example.py" \
    --batch 256 \
    --gaussians 10000 \
    --width 128 \
    --height 128 \
    --warmup 20 \
    --iterations 100 \
    --physics-steps 300 \
    --motion-mode "$mode" \
    --deadline-ms 16.6666667 \
    --tile-size 1 \
    --compact-projection-cache \
    --enable-projection-cache \
    --no-deterministic \
    --output "$output_root/$mode.json" \
    "${extra[@]}"
done
