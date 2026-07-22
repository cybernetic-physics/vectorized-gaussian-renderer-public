#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

config_id="${1:-custom-synth-shared-ncu-preflight}"
profile_root="${PROFILE_ROOT:-$PROJECT_ROOT/outputs/profiles/$config_id}"
mkdir -p "$profile_root"
evidence="$profile_root/${config_id}_ncu_preflight.json"
ncu_bin="${NSIGHT_COMPUTE_HOME}/ncu"
params="$(cat /proc/driver/nvidia/params 2>/dev/null || true)"
caps="$(capsh --print 2>/dev/null | sed -n 's/^Current: //p' || true)"
ncu_version="$("$ncu_bin" --version 2>&1 || true)"
restricted=0
if grep -q "^RmProfilingAdminOnly: 1" /proc/driver/nvidia/params 2>/dev/null \
  && ! printf '%s\n' "$caps" | grep -q "cap_sys_admin"; then
  restricted=1
fi

python3 - "$evidence" "$restricted" "$ncu_bin" "$ncu_version" "$caps" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
restricted = bool(int(sys.argv[2]))
payload = {
    "schema_version": 1,
    "tool": "Nsight Compute",
    "profile_target": "project-owned custom CUDA renderer",
    "ncu_path": sys.argv[3],
    "ncu_version": sys.argv[4],
    "capabilities": sys.argv[5],
    "nvidia_params_path": "/proc/driver/nvidia/params",
    "nvidia_params": pathlib.Path(
        "/proc/driver/nvidia/params"
    ).read_text(
        encoding="utf-8",
        errors="replace",
    )
    if pathlib.Path("/proc/driver/nvidia/params").exists()
    else "",
    "counter_access_restricted": restricted,
    "status": (
        "blocked_by_host_counter_permissions"
        if restricted
        else "counter_preflight_passed"
    ),
}
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [[ "$restricted" == "1" ]]; then
  echo "Nsight Compute is installed, but this host restricts GPU performance counters; evidence: $evidence" >&2
  exit 2
fi

"$ncu_bin" \
  --set full \
  --target-processes all \
  --force-overwrite \
  --export "$profile_root/$config_id" \
  "$ISAACSIM_PATH/python.sh" "$PROJECT_ROOT/benchmarks/profile_custom.py" \
    --config-id "$config_id" \
    --output-dir "$profile_root" \
    --scene "${SCENE:-synthetic-medium}" \
    --batch "${BATCH:-16}" \
    --width "${WIDTH:-128}" \
    --height "${HEIGHT:-128}" \
    --gaussian-support-sigma "${GAUSSIAN_SUPPORT_SIGMA:-2.0}" \
    --covariance-epsilon "${COVARIANCE_EPSILON:-0}" \
    --tile-size "${TILE_SIZE:-1}" \
    --depth-bucket-count "${DEPTH_BUCKET_COUNT:-32}" \
    --depth-bucket-group-size "${DEPTH_BUCKET_GROUP_SIZE:-8}" \
    --compact-projection-cache \
    --projection-cache \
    --authored-display-output \
    --warmup 1 \
    --iterations 1
