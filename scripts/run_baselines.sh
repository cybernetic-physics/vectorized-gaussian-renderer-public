#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-vast-gsplat-isaac}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/vectorized-gaussian-renderer}"
SOURCE_GIT_COMMIT="$(git -C "$(dirname "$0")/.." rev-parse HEAD)"

ssh "$REMOTE_HOST" bash -s -- "$REMOTE_ROOT" "$SOURCE_GIT_COMMIT" <<'REMOTE'
set -euo pipefail
project_root="$1"
export SOURCE_GIT_COMMIT="$2"
export VGR_PROJECT_ROOT="$project_root"
source "$project_root/scripts/remote_env.sh"
cd "$project_root"

mkdir -p outputs/baselines

if [[ ! -f benchmarks/run_baselines.py ]]; then
  echo "The immutable RTX/gsplat baseline runner is not implemented yet." >&2
  exit 2
fi

/isaac-sim/python.sh benchmarks/run_baselines.py \
  --protocol BENCHMARK_PROTOCOL.md \
  --output outputs/baselines
REMOTE
