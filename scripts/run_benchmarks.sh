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

mkdir -p outputs/benchmarks

if [[ ! -x scripts/run_acceptance_matrix.sh ]]; then
  echo "The complete acceptance matrix runner is unavailable." >&2
  exit 2
fi

OUTPUT_ROOT="$project_root/outputs/acceptance-matrix" \
  "$project_root/scripts/run_acceptance_matrix.sh"
REMOTE
