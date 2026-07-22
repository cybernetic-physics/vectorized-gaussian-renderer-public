#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-vast-gsplat-isaac}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/agent-worktrees/verification}"
EXPECTED_REMOTE_ROOT="/workspace/agent-worktrees/verification"

if [[ "$REMOTE_ROOT" != "$EXPECTED_REMOTE_ROOT" ]]; then
  echo "Independent verification must use REMOTE_ROOT=$EXPECTED_REMOTE_ROOT, got $REMOTE_ROOT." >&2
  exit 2
fi

required=(
  GOALS.md
  ARCHITECTURE.md
  BENCHMARK_PROTOCOL.md
  VERIFICATION_PLAN.md
  TASKS.md
  DECISIONS.md
  RESULTS.md
  scripts/setup.sh
  scripts/run_baselines.sh
  scripts/run_benchmarks.sh
  scripts/verify_results.sh
  src/isaacsim_gaussian_renderer/verification.py
  tests/test_verification.py
)

for path in "${required[@]}"; do
  test -f "$PROJECT_ROOT/$path" || {
    echo "Missing required artifact: $path" >&2
    exit 1
  }
done

if [[ "${VERIFY_LOCAL_ONLY:-0}" == "1" ]]; then
  PYTHONPATH="$PROJECT_ROOT/src" "${PYTHON:-python3}" -m isaacsim_gaussian_renderer.verification \
    --evidence-dir "$PROJECT_ROOT/outputs/verification"
  exit
fi

if [[ "${SYNC_VERIFICATION_WORKTREE:-0}" == "1" ]]; then
  ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT'"
  rsync -az --delete \
    --exclude .git/ \
    --exclude outputs/ \
    "$PROJECT_ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/"
fi

ssh "$REMOTE_HOST" bash -s -- "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
project_root="$1"
if [[ "$project_root" != "/workspace/agent-worktrees/verification" ]]; then
  echo "Independent verification must run from /workspace/agent-worktrees/verification." >&2
  exit 2
fi
source "$project_root/scripts/remote_env.sh"
cd "$project_root"

/isaac-sim/python.sh -m pytest -q tests
grep -q "ISAACSIM_GAUSSIAN_RENDERER_SMOKE_OK" /workspace/isaacsim-gaussian-renderer-smoke.log

/isaac-sim/python.sh -m isaacsim_gaussian_renderer.verification \
  --evidence-dir outputs/verification
REMOTE
