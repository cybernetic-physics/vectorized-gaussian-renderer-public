#!/usr/bin/env python3
"""Compare content-addressed deterministic renders from two fresh processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


SENTINEL = "DETERMINISTIC_DIGEST_SMOKE_OK"
WORKER_SCHEMA = "publication-deterministic-digest-smoke-v1"
REPLAY_SCHEMA = "publication-deterministic-cross-process-replay-v1"
OUTPUTS = ("rgb", "alpha", "depth", "semantic_id")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PRODUCTION_ARGUMENTS = {
    "batch": 4,
    "gaussians": 1_024,
    "height": 64,
    "iterations": 32,
    "ray_gaussian_evaluation": False,
    "tile_size": 1,
    "width": 64,
}
CLAIM_SCOPE = {
    **PRODUCTION_ARGUMENTS,
    "camera_configuration": "identical-centered-cameras",
    "claim": "synthetic-equal-depth-deterministic-mode-replay-only",
    "gaussian_configuration": "isotropic-identity-rotation-equal-depth",
    "renderer_mode": "deterministic",
}
COMPARISON_METHOD = "sha256(dtype,shape,raw-c-order-bytes)"
RENDERER_CONFIGURATION = {
    "adaptive_capacity": True,
    "covariance_epsilon": 0.3,
    "deterministic": True,
    "fixed_capacity_sort": False,
    "gaussian_support_sigma": 3.0,
    "output_srgb": False,
    "ray_gaussian_evaluation": False,
    "semantic_min_alpha": 0.01,
    "tile_size": 1,
}
COUNTER_FIELDS = {
    "active_tiles",
    "intersection_overflow",
    "tile_intersections",
    "visible_gaussians",
    "visible_overflow",
}
NATIVE_BUILD_FIELDS = {
    "build_directory",
    "cuda_flags",
    "cxx_flags",
    "module_name",
    "sources",
    "torch_cuda_arch_list",
}
EXPECTED_CXX_FLAGS = ["-O3", "-std=c++17"]
EXPECTED_CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-std=c++17",
]
OUTPUT_CONTRACT_BOOLEAN_FIELDS = {
    "alpha_in_range",
    "contiguous",
    "cuda_resident",
    "dtypes_match",
    "finite_alpha",
    "finite_foreground_depth",
    "finite_rgb",
    "output_names_match",
    "shapes_match",
    "single_cuda_device",
    "valid",
    "valid_background_depth",
    "valid_background_semantics",
    "valid_foreground_semantics",
}
OUTPUT_CONTRACT_COUNT_FIELDS = {
    "background_pixel_count",
    "foreground_pixel_count",
    "semantic_background_pixel_count",
    "semantic_foreground_pixel_count",
}
WORKER_FIELDS = {
    "arguments",
    "bitwise_equal",
    "capacity_invariant_bitwise_equal",
    "center_semantic",
    "claim_scope",
    "comparison_method",
    "counters",
    "equal_depth_gaussian_id_order",
    "equal_depth_records_checked",
    "fixture_sha256",
    "native_build_contract",
    "native_extension",
    "output_contract",
    "output_digests",
    "pass",
    "renderer_configuration",
    "schema_version",
    "source_smoke",
}


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--source-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--gaussians", type=int, default=1_024)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=1)
    parser.add_argument("--ray-gaussian-evaluation", action="store_true")
    return parser.parse_args(arguments)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(
    path_value: Path,
    *,
    executable: bool = False,
    allow_symlink: bool = False,
) -> Path:
    if path_value.is_symlink() and not allow_symlink:
        raise ValueError(f"Required path may not be a symlink: {path_value}")
    path = path_value.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Required path is not a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"Required path is not executable: {path}")
    return path


def artifact_record(path_value: Path) -> dict[str, Any]:
    path = require_regular(path_value)
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": file_sha256(path)}


def arguments_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch": args.batch,
        "gaussians": args.gaussians,
        "height": args.height,
        "iterations": args.iterations,
        "ray_gaussian_evaluation": args.ray_gaussian_evaluation,
        "tile_size": args.tile_size,
        "width": args.width,
    }


def same_typed_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            same_typed_value(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_typed_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def require_artifact_record(
    value: Any,
    *,
    label: str,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"bytes", "path", "sha256"}:
        raise ValueError(f"{label} is not an exact artifact record")
    if (
        isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value["bytes"] < 0
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or not isinstance(value.get("sha256"), str)
        or SHA256_RE.fullmatch(value["sha256"]) is None
    ):
        raise ValueError(f"{label} artifact identity is malformed")
    observed = artifact_record(Path(value["path"]))
    if value != observed:
        raise ValueError(f"{label} differs from its bytes/SHA-256 record")
    if expected is not None and value != expected:
        raise ValueError(f"{label} differs from the pinned publication artifact")
    return value


def expected_output_specs(arguments: dict[str, Any]) -> dict[str, dict[str, Any]]:
    batch = arguments["batch"]
    height = arguments["height"]
    width = arguments["width"]
    return {
        "rgb": {
            "bytes": batch * height * width * 3 * 4,
            "dtype": "torch.float32",
            "shape": [batch, height, width, 3],
        },
        "alpha": {
            "bytes": batch * height * width * 4,
            "dtype": "torch.float32",
            "shape": [batch, height, width, 1],
        },
        "depth": {
            "bytes": batch * height * width * 4,
            "dtype": "torch.float32",
            "shape": [batch, height, width, 1],
        },
        "semantic_id": {
            "bytes": batch * height * width * 8,
            "dtype": "torch.int64",
            "shape": [batch, height, width, 1],
        },
    }


def validate_output_digests(value: Any, *, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(OUTPUTS):
        raise ValueError("Worker output digests do not cover the exact output contract")
    expected = expected_output_specs(arguments)
    for name in OUTPUTS:
        record = value[name]
        if not isinstance(record, dict) or set(record) != {"bytes", "dtype", "sha256", "shape"}:
            raise ValueError(f"Worker {name} digest record has missing or unknown fields")
        if (
            isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record.get("bytes") != expected[name]["bytes"]
            or record.get("dtype") != expected[name]["dtype"]
            or not isinstance(record.get("shape"), list)
            or any(isinstance(size, bool) or not isinstance(size, int) for size in record["shape"])
            or record["shape"] != expected[name]["shape"]
            or not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(record["sha256"]) is None
        ):
            raise ValueError(f"Worker {name} digest contract differs")
    return value


def validate_output_contract(
    value: Any,
    *,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    expected_fields = (
        OUTPUT_CONTRACT_BOOLEAN_FIELDS
        | OUTPUT_CONTRACT_COUNT_FIELDS
        | {"output_devices"}
    )
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("Worker output validation has missing or unknown fields")
    if any(value.get(field) is not True for field in OUTPUT_CONTRACT_BOOLEAN_FIELDS):
        raise ValueError("Worker output validation contains a failed predicate")
    for field in OUTPUT_CONTRACT_COUNT_FIELDS:
        if (
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), int)
            or value[field] < 0
        ):
            raise ValueError(f"Worker output validation count {field} is malformed")
    if value["foreground_pixel_count"] <= 0 or value["semantic_foreground_pixel_count"] <= 0:
        raise ValueError("Worker output validation has no foreground coverage")
    total_pixels = arguments["batch"] * arguments["height"] * arguments["width"]
    if (
        value["foreground_pixel_count"] + value["background_pixel_count"] != total_pixels
        or value["semantic_foreground_pixel_count"]
        + value["semantic_background_pixel_count"]
        != total_pixels
        or value["semantic_foreground_pixel_count"] > value["foreground_pixel_count"]
    ):
        raise ValueError("Worker output validation coverage counts differ from the render shape")
    devices = value.get("output_devices")
    if (
        not isinstance(devices, dict)
        or set(devices) != set(OUTPUTS)
        or any(
            not isinstance(device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", device) is None
            for device in devices.values()
        )
        or len(set(devices.values())) != 1
    ):
        raise ValueError("Worker output validation does not bind one CUDA device")
    return value


def validate_native_evidence(payload: dict[str, Any]) -> None:
    native = require_artifact_record(
        payload.get("native_extension"),
        label="Loaded native extension",
    )
    contract = payload.get("native_build_contract")
    if not isinstance(contract, dict) or set(contract) != NATIVE_BUILD_FIELDS:
        raise ValueError("Loaded native build contract has missing or unknown fields")
    module_name = contract.get("module_name")
    sources = contract.get("sources")
    if (
        not isinstance(module_name, str)
        or re.fullmatch(r"isaacsim_gaussian_renderer_cuda_[0-9a-f]{12}", module_name) is None
        or contract.get("cxx_flags") != EXPECTED_CXX_FLAGS
        or contract.get("cuda_flags") != EXPECTED_CUDA_FLAGS
        or contract.get("torch_cuda_arch_list") != "8.9"
        or not isinstance(contract.get("build_directory"), str)
        or not contract["build_directory"]
        or not isinstance(sources, list)
        or len(sources) != 2
        or any(not isinstance(source, str) or not source for source in sources)
        or [Path(source).name for source in sources] != ["renderer.cpp", "renderer_cuda.cu"]
    ):
        raise ValueError("Loaded native build contract differs from the production contract")
    build_directory = Path(contract["build_directory"]).resolve(strict=True)
    if not build_directory.is_dir() or Path(native["path"]).parent != build_directory:
        raise ValueError("Loaded native extension is outside its build directory")
    if not Path(native["path"]).name.startswith(module_name):
        raise ValueError("Loaded native extension filename differs from its module name")
    for source in sources:
        require_regular(Path(source))


def validate_worker_payload(
    payload: Any,
    *,
    expected_arguments: dict[str, Any],
    expected_source_smoke: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != WORKER_FIELDS:
        raise ValueError("Deterministic worker payload has missing or unknown fields")
    if payload.get("schema_version") != WORKER_SCHEMA or payload.get("pass") is not True:
        raise ValueError("Deterministic worker schema/pass differs")
    if not same_typed_value(payload.get("arguments"), expected_arguments) or not same_typed_value(
        expected_arguments,
        PRODUCTION_ARGUMENTS,
    ):
        raise ValueError("Deterministic worker arguments differ from the fixed publication scope")
    if not same_typed_value(payload.get("claim_scope"), CLAIM_SCOPE):
        raise ValueError("Deterministic worker claim scope differs")
    if payload.get("comparison_method") != COMPARISON_METHOD:
        raise ValueError("Deterministic worker comparison method differs")
    if not same_typed_value(payload.get("renderer_configuration"), RENDERER_CONFIGURATION):
        raise ValueError("Deterministic worker renderer configuration differs")
    if payload.get("source_smoke") != expected_source_smoke:
        raise ValueError("Deterministic worker used another source smoke")
    require_artifact_record(
        payload["source_smoke"],
        label="Worker source smoke",
        expected=expected_source_smoke,
    )
    if (
        not isinstance(payload.get("fixture_sha256"), str)
        or SHA256_RE.fullmatch(payload["fixture_sha256"]) is None
    ):
        raise ValueError("Deterministic worker fixture SHA-256 is malformed")
    for field in ("bitwise_equal", "capacity_invariant_bitwise_equal"):
        equal = payload.get(field)
        if (
            not isinstance(equal, dict)
            or set(equal) != set(OUTPUTS)
            or any(equal[name] is not True for name in OUTPUTS)
        ):
            raise ValueError(f"Deterministic worker predicate {field} failed or is malformed")
    counters = payload.get("counters")
    if not isinstance(counters, dict) or set(counters) != COUNTER_FIELDS:
        raise ValueError("Deterministic worker counters have missing or unknown fields")
    if any(
        isinstance(counters[field], bool)
        or not isinstance(counters[field], int)
        or counters[field] < 0
        for field in COUNTER_FIELDS
    ):
        raise ValueError("Deterministic worker counters are malformed")
    if (
        counters["visible_overflow"] != 0
        or counters["intersection_overflow"] != 0
        or counters["visible_gaussians"] <= 0
        or counters["tile_intersections"] <= 0
        or counters["active_tiles"] <= 0
    ):
        raise ValueError("Deterministic worker counters failed the nonempty/overflow contract")
    equal_depth_count = payload.get("equal_depth_records_checked")
    center_semantic = payload.get("center_semantic")
    if (
        payload.get("equal_depth_gaussian_id_order") != "ascending"
        or isinstance(equal_depth_count, bool)
        or not isinstance(equal_depth_count, int)
        or equal_depth_count != expected_arguments["gaussians"]
        or isinstance(center_semantic, bool)
        or not isinstance(center_semantic, int)
        or center_semantic != 1_000
    ):
        raise ValueError("Deterministic worker equal-depth ordering evidence differs")
    validate_output_digests(payload.get("output_digests"), arguments=expected_arguments)
    validate_output_contract(
        payload.get("output_contract"),
        arguments=expected_arguments,
    )
    validate_native_evidence(payload)
    return payload


def run_worker(command: list[str], *, cwd: Path) -> tuple[int, dict[str, Any]]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    payloads: list[str] = []
    prefix = SENTINEL + " "
    with process.stdout:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith(prefix):
                payloads.append(line[len(prefix) :].rstrip("\r\n"))
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    if len(payloads) != 1:
        raise RuntimeError(
            f"Expected one {SENTINEL} payload from child {process.pid}; observed {len(payloads)}"
        )
    try:
        payload = json.loads(payloads[0])
    except json.JSONDecodeError as error:
        raise ValueError("Deterministic worker emitted malformed JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Deterministic worker payload is not an object")
    return process.pid, payload


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    requested_arguments = arguments_record(args)
    if requested_arguments != PRODUCTION_ARGUMENTS:
        raise ValueError(
            "Publication deterministic replay uses one fixed scope: "
            + json.dumps(PRODUCTION_ARGUMENTS, sort_keys=True)
        )
    python = require_regular(args.python, executable=True, allow_symlink=True)
    if args.source_root.is_symlink():
        raise ValueError("Source root may not be a symlink")
    source_root = args.source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("Source root must be a real directory")
    worker = require_regular(args.worker_script)
    source_smoke = require_regular(args.source_smoke)
    runner_script = require_regular(Path(__file__))
    pinned_worker = artifact_record(worker)
    pinned_source_smoke = artifact_record(source_smoke)
    pinned_runner = artifact_record(runner_script)
    output = args.output.absolute()
    process_outputs = [
        output.with_name(f"{output.stem}.process-{index}{output.suffix}")
        for index in (1, 2)
    ]
    for path in (output, *process_outputs):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite deterministic replay evidence: {path}")

    command = [
        str(python),
        str(worker),
        "--source-smoke",
        str(source_smoke),
        "--batch",
        str(args.batch),
        "--gaussians",
        str(args.gaussians),
        "--iterations",
        str(args.iterations),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--tile-size",
        str(args.tile_size),
    ]
    if args.ray_gaussian_evaluation:
        command.append("--ray-gaussian-evaluation")

    process_records: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for index, destination in enumerate(process_outputs, start=1):
        if (
            artifact_record(worker) != pinned_worker
            or artifact_record(source_smoke) != pinned_source_smoke
            or artifact_record(runner_script) != pinned_runner
        ):
            raise RuntimeError(f"Deterministic code changed before process {index}")
        pid, payload = run_worker(command, cwd=source_root)
        if (
            artifact_record(worker) != pinned_worker
            or artifact_record(source_smoke) != pinned_source_smoke
            or artifact_record(runner_script) != pinned_runner
        ):
            raise RuntimeError(f"Deterministic code changed while process {index} ran")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeError(f"Process {index} returned an invalid process ID")
        validate_worker_payload(
            payload,
            expected_arguments=requested_arguments,
            expected_source_smoke=pinned_source_smoke,
        )
        write_new(destination, payload)
        payloads.append(payload)
        process_records.append(
            {
                "command": command,
                "index": index,
                "payload": artifact_record(destination),
                "pid": pid,
                "returncode": 0,
            }
        )

    if (
        artifact_record(worker) != pinned_worker
        or artifact_record(source_smoke) != pinned_source_smoke
        or artifact_record(runner_script) != pinned_runner
    ):
        raise RuntimeError("Deterministic code changed after the replay")
    payloads_equal = payloads[0] == payloads[1]
    digests_equal = payloads[0]["output_digests"] == payloads[1]["output_digests"]
    fixtures_equal = payloads[0]["fixture_sha256"] == payloads[1]["fixture_sha256"]
    within_equal = all(
        isinstance(payload.get("bitwise_equal"), dict)
        and set(payload["bitwise_equal"]) == set(OUTPUTS)
        and all(value is True for value in payload["bitwise_equal"].values())
        and isinstance(payload.get("capacity_invariant_bitwise_equal"), dict)
        and set(payload["capacity_invariant_bitwise_equal"]) == set(OUTPUTS)
        and all(value is True for value in payload["capacity_invariant_bitwise_equal"].values())
        for payload in payloads
    )
    zero_overflow = all(
        payload["counters"]["intersection_overflow"] == 0
        and payload["counters"]["visible_overflow"] == 0
        for payload in payloads
    )
    passed = bool(payloads_equal and digests_equal and fixtures_equal and within_equal and zero_overflow)
    result = {
        "claim_scope": CLAIM_SCOPE,
        "comparison_method": COMPARISON_METHOD,
        "cross_process_output_hashes_equal": digests_equal,
        "fixture_sha256": payloads[0]["fixture_sha256"],
        "fresh_processes": 2,
        "native_build_contract": payloads[0]["native_build_contract"],
        "native_extension": payloads[0]["native_extension"],
        "output_contract": payloads[0]["output_contract"],
        "output_digests": payloads[0]["output_digests"],
        "pass": passed,
        "process_payloads_equal": payloads_equal,
        "processes": process_records,
        "replays_per_process": args.iterations,
        "renderer_configuration": payloads[0]["renderer_configuration"],
        "runner_script": pinned_runner,
        "schema_version": REPLAY_SCHEMA,
        "source_smoke": pinned_source_smoke,
        "within_process_bitwise_equal": within_equal,
        "worker_script": pinned_worker,
        "zero_overflow": zero_overflow,
    }
    if not passed:
        raise AssertionError("Fresh-process deterministic output digests differ")
    write_new(output, result)
    print("DETERMINISTIC_CROSS_PROCESS_REPLAY_OK " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
