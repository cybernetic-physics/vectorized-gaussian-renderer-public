#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 LOG_PATH SUCCESS_MARKER [RESULT_JSON]" >&2
  exit 2
fi

log_path="$1"
success_marker="$2"
result_json="${3:-}"

if [[ ! -s "$log_path" ]]; then
  echo "ISAAC_LOG_INVALID missing_or_empty=$log_path" >&2
  exit 1
fi

fatal_pattern='Traceback \(most recent call last\)|There was an error running python|Fatal Python error|Segmentation fault|Aborted( \(core dumped\))?|SIGABRT|signal 6|core dumped|dumped core|terminate called|Destroying busy TaskGroup|TaskGroup::~TaskGroup|Assertion( Failed| \([^)]*\) failed)|Exception during renderer cleanup in __del__|Failed to shutdown ovrtx library|Exception during ovrtx library cleanup|CUDA error:|illegal memory access|VkResult: ERROR_|vkCreateInstance failed|carb::graphics::createInstance failed|Failed to create any GPU devices|GPU Foundation is not initialized|unable to get a valid CUDA device id from the renderer|libXt\.so\.6: cannot open shared object file'
if command -v rg >/dev/null 2>&1; then
  fatal_search=(rg -n -m 20)
  marker_search=(rg -F -x -q --)
else
  fatal_search=(grep -E -n -m 20 --)
  marker_search=(grep -F -x -q --)
fi

if "${fatal_search[@]}" "$fatal_pattern" "$log_path"; then
  echo "ISAAC_LOG_INVALID fatal_pattern_found=$log_path" >&2
  exit 1
fi

if ! "${marker_search[@]}" "$success_marker" "$log_path"; then
  echo "ISAAC_LOG_INVALID missing_marker=$success_marker" >&2
  exit 1
fi

if [[ -n "$result_json" ]]; then
  if [[ ! -s "$result_json" ]]; then
    echo "ISAAC_LOG_INVALID missing_result_json=$result_json" >&2
    exit 1
  fi
  jq -e '.pass == true' "$result_json" >/dev/null
fi

echo "ISAAC_LOG_OK log=$log_path marker=$success_marker"
