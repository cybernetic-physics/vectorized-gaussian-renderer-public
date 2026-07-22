#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

exec "$ISAACSIM_PATH/python.sh" \
  "$PROJECT_ROOT/benchmarks/run_customer_stress_matrix.py" "$@"
