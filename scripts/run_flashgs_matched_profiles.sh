#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

benchmark_python="${MATCHED_BENCHMARK_PYTHON:-$ISAACSIM_PATH/python.sh}"
matrix_root="${1:?matched matrix output root is required}"
for batch in 1 1024; do
  trajectory="$matrix_root/contracts/b${batch}.json"
  for renderer in custom flashgs; do
    for contract in rgb full; do
      config_id="flashgs-matched-${renderer}-${contract}-b${batch}"
      PROFILE_ROOT="$matrix_root/profiles/$config_id" \
        "$PROJECT_ROOT/scripts/profile_flashgs_matched_nsys.sh" \
        "$renderer" \
        "$contract" \
        "$batch" \
        "$matrix_root/runs/$renderer/$contract/b${batch}.json" \
        "$trajectory" \
        "$config_id"
    done
  done
done

"$benchmark_python" \
  "$PROJECT_ROOT/benchmarks/summarize_flashgs_matched.py" \
  --root "$matrix_root" \
  --batches 1,8,32,64,128,256,512,1024 \
  --output "$matrix_root/summary.json"
