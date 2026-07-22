#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"

output_root="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/acceptance}"
projection_root="$PROJECT_ROOT/outputs/fidelity/ovrtx-projection-paired-20260716"
projection_audit="$output_root/ovrtx-projection-mode-audit.json"

set +e
"$ISAACSIM_PATH/python.sh" \
  "$PROJECT_ROOT/benchmarks/audit_ovrtx_projection_modes.py" \
  --perspective-candidate \
    "$projection_root/perspective-1024/candidate-ovrtx-temporal.npz" \
  --tangential-candidate \
    "$projection_root/tangential-1024/candidate-ovrtx-temporal.npz" \
  --perspective-stage "$projection_root/perspective-1024/scene.usda" \
  --tangential-stage "$projection_root/tangential-1024/scene.usda" \
  --output "$projection_audit"
projection_audit_status=$?
set -e

if [[ ! -f "$projection_audit" ]]; then
  exit "$projection_audit_status"
fi

"$ISAACSIM_PATH/python.sh" \
  "$PROJECT_ROOT/benchmarks/summarize_headline_acceptance.py" \
  --custom-run "$output_root/b1024/custom-run1/result.json" \
  --custom-run "$output_root/b1024/custom-run2/result.json" \
  --custom-run "$output_root/b1024/custom-run3/result.json" \
  --ovrtx-run "$output_root/b1024/ovrtx-run1" \
  --ovrtx-run "$output_root/b1024/ovrtx-run2" \
  --ovrtx-run "$output_root/b1024/ovrtx-run3" \
  --ewa-gsplat-fidelity \
    "$PROJECT_ROOT/outputs/fidelity/custom-vs-gsplat-b8-128/report/fidelity_report.json" \
  --ovrtx-perspective-fidelity \
    "$PROJECT_ROOT/outputs/fidelity/ovrtx-temporal-authored-16384/report-16384/fidelity_report.json" \
  --ovrtx-exact-ray-diagnostic \
    "$PROJECT_ROOT/outputs/fidelity/ovrtx-exact-ray-prototype-65536-report/fidelity_report.json" \
  --home-finite-frame-control \
    "$PROJECT_ROOT/outputs/compact-direct/home-b1-128/fidelity-vs-ovrtx/fidelity_report.json" \
  --projection-mode-audit "$projection_audit" \
  --custom-single \
    "$PROJECT_ROOT/outputs/compact-direct/home-b1-128/result.json" \
  --ovrtx-single \
    "$PROJECT_ROOT/outputs/performance/home-spatial2-screen-sigma2/ovrtx-b1-clockwarm-run1" \
  --ovrtx-next-batch-failure-log \
    "$PROJECT_ROOT/outputs/compact/ovrtx-home-b2048-128-boundary/run.log" \
  --headline-batch 1024 \
  --next-batch 2048 \
  --output "$output_root/headline-acceptance.json"
