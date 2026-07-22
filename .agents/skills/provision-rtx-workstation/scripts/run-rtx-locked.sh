#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || "$4" != "--" ]]; then
  echo "Usage: $0 HOST REMOTE_ROOT RUN_ID -- COMMAND [ARG ...]" >&2
  exit 2
fi

host="$1"
remote_root="$2"
run_id="$3"
shift 4
coordination_root="${RTX_REMOTE_COORDINATION_ROOT:-/workspace}"

if [[ "$host" =~ [[:space:]] ||
      "$remote_root" =~ [[:space:]] ||
      "$coordination_root" =~ [[:space:]] ]]; then
  echo "HOST, REMOTE_ROOT, and the coordination root must not contain whitespace." >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi

printf -v root_q '%q' "$remote_root"
printf -v run_id_q '%q' "$run_id"
printf -v owner_file_q '%q' \
  "$coordination_root/.vectorized-gaussian-renderer.gpu-owner"
printf -v lock_file_q '%q' \
  "$coordination_root/.vectorized-gaussian-renderer.gpu.lock"
printf -v command_q '%q ' "$@"

remote_script="$(
  printf '%s\n' 'set -euo pipefail'
  printf 'root=%s\n' "$root_q"
  printf 'run_id=%s\n' "$run_id_q"
  printf 'owner_file=%s\n' "$owner_file_q"
  printf '%s\n' \
    'cd "$root"' \
    'test -f scripts/remote_env.sh' \
    'export VGR_PROJECT_ROOT="$root"' \
    'source scripts/remote_env.sh' \
    'cleanup() { rm -f "$owner_file"; }' \
    'trap cleanup EXIT INT TERM HUP' \
    'printf "run_id=%s\nhost=%s\npid=%s\nstarted_utc=%s\nroot=%s\n" "$run_id" "$(hostname)" "$$" "$(date -u +%FT%TZ)" "$root" > "$owner_file"' \
    'echo "RTX_LOCK_ACQUIRED run_id=$run_id root=$root"'
  printf '%s\n' "$command_q"
)"

ssh_opts=(
  -o BatchMode=yes
  -o "ConnectTimeout=${RTX_SSH_CONNECT_TIMEOUT:-10}"
  -o "ServerAliveInterval=${RTX_SSH_SERVER_ALIVE_INTERVAL:-15}"
  -o "ServerAliveCountMax=${RTX_SSH_SERVER_ALIVE_COUNT_MAX:-3}"
)

if [[ "${RTX_REMOTE_DRY_RUN:-0}" == "1" ]]; then
  echo "RTX_REMOTE_DRY_RUN host=$host lock=$coordination_root/.vectorized-gaussian-renderer.gpu.lock"
  printf '%s\n' "$remote_script"
  exit 0
fi

printf '%s\n' "$remote_script" |
  ssh "${ssh_opts[@]}" "$host" \
    "flock -E 75 -n $lock_file_q bash -s"
