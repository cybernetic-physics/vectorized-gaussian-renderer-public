#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 HOST [REMOTE_ROOT]" >&2
  exit 2
fi

host="$1"
remote_root="${2:-/workspace/vectorized-gaussian-renderer}"

if [[ "$host" =~ [[:space:]] || "$remote_root" =~ [[:space:]] ]]; then
  echo "HOST and REMOTE_ROOT must not contain whitespace." >&2
  exit 2
fi

ssh_opts=(
  -o BatchMode=yes
  -o "ConnectTimeout=${RTX_SSH_CONNECT_TIMEOUT:-10}"
  -o "ServerAliveInterval=${RTX_SSH_SERVER_ALIVE_INTERVAL:-15}"
  -o "ServerAliveCountMax=${RTX_SSH_SERVER_ALIVE_COUNT_MAX:-3}"
)

ssh "${ssh_opts[@]}" "$host" bash -s -- "$remote_root" <<'REMOTE'
set -u

root="$1"
node_busy=0

echo "RTX_PREFLIGHT_NODE hostname=$(hostname) user=$(id -un)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader

  compute_processes="$(
    nvidia-smi \
      --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader 2>/dev/null || true
  )"
  if [[ -n "$compute_processes" ]]; then
    node_busy=1
    echo "RTX_PREFLIGHT_BUSY compute_processes_present"
    printf '%s\n' "$compute_processes"
  else
    echo "RTX_PREFLIGHT_IDLE no_compute_processes"
  fi
else
  echo "RTX_PREFLIGHT_MISSING tool=nvidia-smi"
fi

relevant_processes="$(
  ps -eo pid=,user=,comm= |
    awk '$3 ~ /(python|isaac|ovrtx|nsys|ncu|ngfx)/ {print "RTX_PROCESS", $0}' ||
    true
)"
if [[ -n "$relevant_processes" ]]; then
  node_busy=1
  printf '%s\n' "$relevant_processes"
fi
tmux ls 2>/dev/null | sed 's/^/RTX_TMUX /' || true
if [[ -s /workspace/.vectorized-gaussian-renderer.gpu-owner ]]; then
  node_busy=1
  sed 's/^/RTX_LOCK_OWNER /' \
    /workspace/.vectorized-gaussian-renderer.gpu-owner
fi

for path in \
  "$root" \
  "$root/scripts/remote_env.sh" \
  /isaac-sim/python.sh \
  /workspace/src/gsplat \
  /workspace/src/ovrtx \
  /workspace/src/kit-app-template \
  /workspace/tools/ovrtx-0.3.0; do
  if [[ -e "$path" ]]; then
    echo "RTX_PREFLIGHT_PATH ok=$path"
  else
    echo "RTX_PREFLIGHT_PATH missing=$path"
  fi
done

if [[ "$node_busy" -eq 1 ]]; then
  echo "RTX_PREFLIGHT_RUNTIME skipped=node_busy"
elif [[ -f "$root/scripts/remote_env.sh" && -x /isaac-sim/python.sh ]]; then
  (
    set -e
    export VGR_PROJECT_ROOT="$root"
    source "$root/scripts/remote_env.sh"

    for tool in nvcc nsys ncu compute-sanitizer cuda-gdb ffmpeg ffprobe rg jq; do
      if command -v "$tool" >/dev/null 2>&1; then
        echo "RTX_PREFLIGHT_TOOL ok=$tool path=$(command -v "$tool")"
      else
        echo "RTX_PREFLIGHT_TOOL missing=$tool"
      fi
    done

    /isaac-sim/python.sh - <<'PY'
import importlib.util
import torch

print("RTX_PREFLIGHT_RUNTIME", "torch=" + torch.__version__)
print("RTX_PREFLIGHT_RUNTIME", "cuda=" + str(torch.version.cuda))
print("RTX_PREFLIGHT_RUNTIME", "cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("RTX_PREFLIGHT_RUNTIME", "gpu=" + torch.cuda.get_device_name(0))
    print("RTX_PREFLIGHT_RUNTIME", "capability=" + str(torch.cuda.get_device_capability(0)))
for name in ("numpy", "ovrtx", "gsplat", "lpips"):
    print(
        "RTX_PREFLIGHT_MODULE",
        f"name={name}",
        f"available={importlib.util.find_spec(name) is not None}",
    )
PY
  )
else
  echo "RTX_PREFLIGHT_RUNTIME skipped=missing_project_env_or_isaac_python"
fi
REMOTE
