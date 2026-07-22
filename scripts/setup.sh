#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-vast-gsplat-isaac}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/agent-worktrees/verification}"
EXPECTED_REMOTE_ROOT="/workspace/agent-worktrees/verification"

if [[ "$REMOTE_ROOT" != "$EXPECTED_REMOTE_ROOT" ]]; then
  echo "Independent verification setup must use REMOTE_ROOT=$EXPECTED_REMOTE_ROOT, got $REMOTE_ROOT." >&2
  exit 2
fi

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT'"
rsync -az --delete --exclude .git/ --exclude outputs/ "$PROJECT_ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/"

ssh "$REMOTE_HOST" bash -s -- "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
project_root="$1"
export VGR_PROJECT_ROOT="$project_root"

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential cmake git git-lfs htop libxt6 ninja-build nodejs npm ripgrep rsync tmux

test -x /isaac-sim/python.sh
test -x /usr/local/cuda-12.8/bin/nvcc
test -d /workspace/src/gsplat
test -d /workspace/tools/ovrtx-0.3.0
test -d /workspace/src/kit-app-template

chown -R root:root "$project_root"
chmod +x "$project_root"/scripts/*.sh

source "$project_root/scripts/remote_env.sh"

for tool in \
  nsys \
  ncu \
  compute-sanitizer \
  cuda-gdb \
  nvprof; do
  command -v "$tool"
done
test -x "$NSIGHT_GRAPHICS_HOME/ngfx"

/isaac-sim/python.sh - <<'PY'
import torch

assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (8, 9)
print("ISAAC_TORCH_RUNTIME_OK", torch.__version__)
PY

test "$(git -C /workspace/src/gsplat rev-parse HEAD)" = \
  "77ab983ffe43420b2131669cb35776b883ca4c3c"

/isaac-sim/python.sh -m pip install --no-deps \
  "lazy_loader>=0.4" \
  "numpy==1.26.4" \
  "pillow>=10" \
  "plyfile==1.1.3" \
  "setuptools>=69" \
  wheel

/isaac-sim/python.sh -m pip install --no-deps -e "$project_root"
/isaac-sim/python.sh -m pytest -q "$project_root/tests"
/isaac-sim/python.sh "$project_root/scripts/isaacsim_extension_smoke.py" \
  > /workspace/isaacsim-gaussian-renderer-smoke.log

splat-transform --version
REMOTE

echo "Remote setup verified on $REMOTE_HOST"
