#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export VGR_PROJECT_ROOT="$PROJECT_ROOT"
matched_cuda_home="${MATCHED_CUDA_HOME:-}"
source "$(dirname "$0")/remote_env.sh"
if [[ -n "$matched_cuda_home" ]]; then
  export CUDA_HOME="$matched_cuda_home"
fi

benchmark_python="${MATCHED_BENCHMARK_PYTHON:-$ISAACSIM_PATH/python.sh}"
allow_nonheadline_gpu="${MATCHED_ALLOW_NONHEADLINE_GPU:-0}"
nsys_importer="${MATCHED_NSYS_IMPORTER:-}"
renderer="${1:?renderer must be custom or flashgs}"
contract="${2:?output contract must be rgb or full}"
batch="${3:?batch is required}"
unprofiled_result="${4:?matching unprofiled result JSON is required}"
trajectory="${5:?matching trajectory JSON is required}"
config_id="${6:-flashgs-matched-${renderer}-${contract}-b${batch}}"
dataset="${HOME_SCAN_PATH:-/workspace/datasets/home-scan-lod0.ply}"
matrix_root="$(cd "$(dirname "$unprofiled_result")/../../.." && pwd)"
expected_profile_root="$matrix_root/profiles/$config_id"
if [[ -n "${PROFILE_ROOT:-}" ]] && \
   [[ "$(realpath -m "$PROFILE_ROOT")" != "$(realpath -m "$expected_profile_root")" ]]; then
  echo "PROFILE_ROOT must match the result-owned profile directory: $expected_profile_root" >&2
  exit 2
fi
profile_root="$expected_profile_root"
export PROFILE_ROOT="$profile_root"
nsys_root="$profile_root/nsys"
stats_root="$nsys_root/stats"
if [[ -n "${PROFILE_MEASURED_FRAMES:-}" && "$PROFILE_MEASURED_FRAMES" != "3" ]]; then
  echo "PROFILE_MEASURED_FRAMES is frozen at 3 for the publication control." >&2
  exit 2
fi
measured_frames=3
export PROFILE_MEASURED_FRAMES="$measured_frames"

if [[ "$renderer" != "custom" && "$renderer" != "flashgs" ]]; then
  echo "renderer must be custom or flashgs" >&2
  exit 2
fi
if [[ ! -x "$benchmark_python" ]]; then
  echo "Matched benchmark Python is not executable: $benchmark_python" >&2
  exit 2
fi
if [[ "$allow_nonheadline_gpu" != "0" && "$allow_nonheadline_gpu" != "1" ]]; then
  echo "MATCHED_ALLOW_NONHEADLINE_GPU must be 0 or 1" >&2
  exit 2
fi
if [[ -n "$nsys_importer" && ! -x "$nsys_importer" ]]; then
  echo "MATCHED_NSYS_IMPORTER is not executable: $nsys_importer" >&2
  exit 2
fi
if [[ "$contract" != "rgb" && "$contract" != "full" ]]; then
  echo "output contract must be rgb or full" >&2
  exit 2
fi
if [[ ! -f "$unprofiled_result" || ! -f "$trajectory" || ! -f "$dataset" ]]; then
  echo "unprofiled result, trajectory, and Home Scan dataset must exist" >&2
  exit 2
fi
mkdir -p "$profile_root" "$nsys_root" "$stats_root"

executor_lock_path="${VGR_GPU_EXECUTOR_LOCK_PATH:-/tmp/vgr-publication-gpu-executor.lock}"
if [[ -z "${VGR_GPU_EXECUTOR_LOCK_OWNER_PID:-}" ]]; then
  exec 9>"$executor_lock_path"
  if ! flock -n 9; then
    echo "Another publication GPU executor holds the node-wide lock." >&2
    exit 3
  fi
  printf '{"schema_version":"vgr-cooperative-gpu-lock-metadata-v1","owner_pid":%s,"command":["profile_flashgs_matched_nsys.sh"]}\n' "$$" >&9
  export VGR_GPU_EXECUTOR_LOCK_PATH="$executor_lock_path"
  export VGR_GPU_EXECUTOR_LOCK_OWNER_PID="$$"
fi

readarray -t run_contract < <(
  "$benchmark_python" -c '
import hashlib, json, sys
from pathlib import Path
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("schema_version") != "flashgs-matched-renderer-run-v4":
    raise SystemExit("unexpected unprofiled result schema")
if payload.get("equation_contract", {}).get("gaussian_support_sigma") != 3.33:
    raise SystemExit("unprofiled result does not use pinned gsplat 3.33-sigma support")
projection_rules = {
    "frustum_depth_interval": "inclusive-near-and-far",
    "invalid_covariance_2d": "determinant<=0",
    "projected_radius_covariance_diagonal_floor": None,
    "projected_radius_min_pixels": 0,
    "projected_radius_clip": "cull-only-when-both-radii<=0",
}
equation = payload.get("equation_contract", {})
for field, expected in projection_rules.items():
    if equation.get(field) != expected:
        raise SystemExit(f"unprofiled projection rule {field} differs")
native = payload.get("environment", {}).get("native_extension", {})
native_path = Path(str(native.get("path", "")))
if not native_path.is_file() or native_path.stat().st_size != native.get("bytes"):
    raise SystemExit("unprofiled native extension is missing or changed size")
digest = hashlib.sha256()
with native_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != native.get("sha256"):
    raise SystemExit("unprofiled native extension hash differs")
attestation = payload.get("environment", {}).get(
    "flashgs_adapter_attestation", {}
)
attestation_path = Path(str(attestation.get("path", "")))
if (
    not attestation_path.is_file()
    or attestation_path.stat().st_size != attestation.get("bytes")
):
    raise SystemExit("unprofiled FlashGS adapter attestation is missing or changed size")
digest = hashlib.sha256()
with attestation_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != attestation.get("sha256"):
    raise SystemExit("unprofiled FlashGS adapter attestation hash differs")
if payload.get("renderer") != sys.argv[2]:
    raise SystemExit("unprofiled renderer mismatch")
if payload.get("output_contract") != sys.argv[3]:
    raise SystemExit("unprofiled output-contract mismatch")
if int(payload["camera_contract"]["batch"]) != int(sys.argv[4]):
    raise SystemExit("unprofiled batch mismatch")
capacity_field = (
    "calibration_artifact" if sys.argv[2] == "custom" else "demand_survey_artifact"
)
config_capacity_field = (
    "capacity_calibration" if sys.argv[2] == "custom" else "flashgs_demand_survey"
)
capacity_artifact = payload.get("capacity", {}).get(capacity_field, {})
capacity_path = Path(str(capacity_artifact.get("path", "")))
if (
    not capacity_path.is_file()
    or capacity_path.stat().st_size != capacity_artifact.get("bytes")
):
    raise SystemExit("unprofiled capacity artifact is missing or changed size")
digest = hashlib.sha256()
with capacity_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != capacity_artifact.get("sha256"):
    raise SystemExit("unprofiled capacity artifact hash differs")
capacity_payload = json.load(open(capacity_path, encoding="utf-8"))
if sys.argv[2] == "custom":
    valid_capacity = (
        capacity_payload.get("schema_version")
        == "flashgs-matched-capacity-calibration-v1"
        and capacity_payload.get("pass") is True
        and capacity_payload.get("renderer") == "custom"
        and capacity_payload.get("mode") == "capacity-calibration-only"
        and int(capacity_payload["camera_contract"]["batch"]) == int(sys.argv[4])
    )
else:
    selected = capacity_payload.get("derived_batch_capacities", {}).get(
        str(int(sys.argv[4])), {}
    )
    valid_capacity = (
        capacity_payload.get("schema_version")
        == "flashgs-matched-flashgs-demand-survey-v1"
        and capacity_payload.get("pass") is True
        and capacity_payload.get("renderer") == "flashgs"
        and capacity_payload.get("mode") == "flashgs-demand-survey-only"
        and capacity_payload.get("timing_valid") is False
        and capacity_payload.get("render_outputs_valid") is False
        and int(capacity_payload["camera_contract"]["batch"]) == 1024
        and payload.get("capacity", {}).get("installed_from_batch_specific_prefix")
        == int(sys.argv[4])
        and payload.get("capacity", {}).get("survey_prefix_demand") == selected
        and payload.get("capacity", {}).get("installed", {}).get("intersections")
        == selected.get("installed_intersections_per_camera")
    )
if not valid_capacity:
    raise SystemExit("capacity artifact contract differs")
config_capacity = payload.get("runner_config", {}).get(config_capacity_field, {})
if (
    config_capacity.get("bytes") != capacity_artifact.get("bytes")
    or config_capacity.get("sha256") != capacity_artifact.get("sha256")
):
    raise SystemExit("run capacity-artifact references disagree")
print(capacity_path)
print(payload["equation_contract"]["semantic_topology"])
print(payload.get("backend_execution", {}).get("physical_batch") or "")
print(payload["environment"]["source_provenance"]["manifest"]["path"])
print(payload["environment"]["gpu_uuid"])
print(payload["environment"].get("torch_cuda_arch_list") or "")
print(attestation_path)
print(native_path.resolve().parent.parent)
' "$unprofiled_result" "$renderer" "$contract" "$batch"
)
capacity_artifact="${run_contract[0]}"
semantic_topology="${run_contract[1]}"
physical_batch="${run_contract[2]}"
source_manifest="${run_contract[3]}"
expected_gpu_uuid="${run_contract[4]}"
expected_cuda_arch="${run_contract[5]}"
flashgs_adapter_attestation="${run_contract[6]}"
native_build_root="${run_contract[7]}"
if [[ ! -f "$source_manifest" ]]; then
  echo "Source manifest recorded by the unprofiled run is missing." >&2
  exit 2
fi
"$benchmark_python" \
  "$PROJECT_ROOT/benchmarks/flashgs_matched_occupancy.py" \
  --expected-gpu-uuid "$expected_gpu_uuid" \
  --output "$profile_root/occupancy-preflight.json"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST="$expected_cuda_arch"
elif [[ "$TORCH_CUDA_ARCH_LIST" != "$expected_cuda_arch" ]]; then
  echo "TORCH_CUDA_ARCH_LIST differs from the unprofiled run." >&2
  exit 2
fi
if [[ -z "${VGR_NATIVE_BUILD_ROOT:-}" ]]; then
  export VGR_NATIVE_BUILD_ROOT="$native_build_root"
elif [[ "$(realpath "$VGR_NATIVE_BUILD_ROOT")" != "$(realpath "$native_build_root")" ]]; then
  echo "VGR_NATIVE_BUILD_ROOT differs from the unprofiled run." >&2
  exit 2
fi

runner_args=(
  --renderer "$renderer"
  --output-contract "$contract"
  --trajectory "$trajectory"
  --scene-path "$dataset"
  --warmup-frames 8
  --measured-frames "$measured_frames"
  --semantic-topology "$semantic_topology"
  --source-manifest "$source_manifest"
  --flashgs-adapter-attestation "$flashgs_adapter_attestation"
  --expected-gpu-uuid "$expected_gpu_uuid"
  --profile-control
  --no-capture-last-output
  --output "$profile_root/profile-control.json"
)
if [[ "$renderer" == "custom" ]]; then
  runner_args+=(--capacity-calibration "$capacity_artifact")
else
  runner_args+=(--flashgs-demand-survey "$capacity_artifact")
fi
if [[ "$renderer" == "custom" ]]; then
  if [[ -n "$physical_batch" && "$physical_batch" -lt "$batch" ]]; then
    runner_args+=(--custom-max-physical-views "$physical_batch")
  fi
fi
if [[ "$allow_nonheadline_gpu" == "1" ]]; then
  runner_args+=(--allow-nonheadline-gpu)
fi

capture_name="flashgs-matched-capture/${renderer}/${contract}/b${batch}"
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
nsys --version > "$profile_root/nsys-version.txt"
profile_command=(
  nsys profile
  --trace=cuda,nvtx,osrt
  --sample=none
  --capture-range=nvtx
  --capture-range-end=stop
  --nvtx-capture="$capture_name"
  --force-overwrite=true
  --output="$nsys_root/$config_id"
  "$benchmark_python"
  "$PROJECT_ROOT/benchmarks/run_flashgs_matched.py"
  "${runner_args[@]}"
)
nsys_profile_status=0
"${profile_command[@]}" || nsys_profile_status=$?
printf '%s\n' "$nsys_profile_status" > "$profile_root/nsys-profile-exit-status.txt"

nsys_report="$nsys_root/$config_id.nsys-rep"
nsys_qdstrm="$nsys_root/$config_id.qdstrm"
if [[ ! -s "$nsys_report" && -n "$nsys_importer" && -s "$nsys_qdstrm" ]]; then
  "$nsys_importer" \
    --input-file "$nsys_qdstrm" \
    --output-file "$nsys_report" \
    --force-overwrite
fi
if [[ ! -s "$nsys_report" ]]; then
  echo "Nsight Systems did not produce a report (profile exit $nsys_profile_status)." >&2
  if [[ "$nsys_profile_status" == "0" ]]; then
    nsys_profile_status=1
  fi
  exit "$nsys_profile_status"
fi
if [[ "$nsys_profile_status" != "0" ]]; then
  echo "Nsight profile target exited nonzero: $nsys_profile_status" >&2
  exit "$nsys_profile_status"
fi
if [[ ! -s "$profile_root/profile-control.json" ]]; then
  echo "Profile target did not write profile-control.json." >&2
  exit 1
fi

"$benchmark_python" \
  "$PROJECT_ROOT/scripts/write_flashgs_profile_wrapper_evidence.py" \
  --occupancy-preflight "$profile_root/occupancy-preflight.json" \
  --profile-exit-status "$profile_root/nsys-profile-exit-status.txt" \
  --profile-control "$profile_root/profile-control.json" \
  --nsys-version "$profile_root/nsys-version.txt" \
  --source-script "$PROJECT_ROOT/scripts/profile_flashgs_matched_nsys.sh" \
  --output "$profile_root/wrapper-evidence.json" \
  -- "${profile_command[@]}"

nsys stats \
  --force-export=true \
  --force-overwrite=true \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv \
  --output "$stats_root/$config_id" \
  "$nsys_report"

nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output "$nsys_root/$config_id.sqlite" \
  "$nsys_report"

"$benchmark_python" "$PROJECT_ROOT/scripts/parse_nsys_stats.py" \
  --stats-dir "$stats_root" \
  --output "$profile_root/nsys-summary.json"

"$benchmark_python" \
  "$PROJECT_ROOT/benchmarks/summarize_flashgs_profile.py" \
  --unprofiled-run "$unprofiled_result" \
  --profile-run "$profile_root/profile-control.json" \
  --nsys-summary "$profile_root/nsys-summary.json" \
  --nsys-report "$nsys_report" \
  --nsys-sqlite "$nsys_root/$config_id.sqlite" \
  --stats-dir "$stats_root" \
  --wrapper-evidence "$profile_root/wrapper-evidence.json" \
  --output "$profile_root/profile-summary.json"
