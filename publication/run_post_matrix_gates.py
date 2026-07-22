#!/usr/bin/env python3
"""Run the exact-current publication gates after the frozen matrix completes.

This is a publication-only coordinator.  It does not alter the frozen source
checkout or benchmark implementation.  Every GPU command is executed through
``run_bounded_gate.py``, so one cooperative node-wide lock, pre/post occupancy,
periodic descendant-process telemetry, a finite timeout, and complete logs are
preserved for every lane.  The script fails on the first incomplete result and
writes a declarative aggregate-verification specification only after all twelve
required gates have passed their bounded wrappers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


SCENE_SHA256 = "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
BATCHES = "1,8,32,64,128,256,512,1024"
REQUIRED_GATES = (
    "flashgs-b64-repair",
    "repair-compute-sanitizer",
    "python-unit-contract",
    "native-cuda-smoke",
    "native-determinism",
    "native-multi-scene",
    "isaac-graphics",
    "all-output-fidelity",
    "custom-vectorization-ablation",
    "deterministic-replay",
    "compute-sanitizer-memcheck",
    "bounded-stability-soak",
)
FLASHGS_GATES = {"flashgs-b64-repair", "repair-compute-sanitizer"}


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--canonical-runtime", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--gsplat-source", type=Path, required=True)
    parser.add_argument("--diagnosis-index", type=Path, required=True)
    parser.add_argument("--historical-b64-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--benchmark-tag", default="benchmark-gcp-l4-matched-v7")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(arguments)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_new_or_identical(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"Existing immutable file differs: {path}")
        return
    write_new(path, payload)


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or symlinked: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(
    path: Path,
    *,
    label: str,
    executable: bool = False,
    allow_symlink: bool = False,
) -> Path:
    raw = path.absolute()
    if raw.is_symlink() and not allow_symlink:
        raise ValueError(f"{label} may not be a symlink: {raw}")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    if executable and not os.access(raw, os.X_OK):
        raise ValueError(f"{label} is not executable: {raw}")
    return raw


def reject_symlink_components(path: Path, *, boundary: Path, label: str) -> None:
    """Reject a symlinked leaf or parent without requiring the leaf to exist."""

    raw = path.absolute()
    root = boundary.absolute()
    if not raw.is_relative_to(root):
        raise ValueError(f"{label} escapes its boundary: {raw}")
    current = raw
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} crosses a symlink: {current}")
        if current == root:
            return
        current = current.parent


def require_real_directory(path: Path, *, boundary: Path, label: str) -> Path:
    reject_symlink_components(path, boundary=boundary, label=label)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or not resolved.is_relative_to(boundary.resolve(strict=True)):
        raise ValueError(f"{label} is not a contained regular directory: {path}")
    return resolved


def require_resume_ablation_alias(
    *,
    output: Path,
    matrix: Path,
    alias: Path,
) -> None:
    """Bind an existing canonical ablation alias to its completed resume gate."""

    reject_symlink_components(alias, boundary=matrix, label="canonical ablation alias")
    gate_result = output / "gates/custom-vectorization-ablation/result.json"
    if not alias.exists():
        if gate_result.exists() or gate_result.is_symlink():
            raise FileNotFoundError(
                "Resumed vectorization gate has a result but no canonical ablation alias."
            )
        return
    reject_symlink_components(
        gate_result,
        boundary=output,
        label="resumed vectorization gate result",
    )
    result = load_json(gate_result, label="resumed vectorization gate result")
    if result.get("gate_id") != "custom-vectorization-ablation" or result.get("pass") is not True:
        raise RuntimeError("Resumed vectorization gate result is not a passing canonical result.")
    evidence = result.get("evidence")
    raw_record = evidence.get("raw_result") if isinstance(evidence, dict) else None
    raw_path = artifact_path(raw_record, label="resumed canonical ablation evidence")
    if raw_path.resolve(strict=True) != alias.resolve(strict=True):
        raise RuntimeError("Resumed vectorization result is not bound to the canonical alias.")


def artifact_path(record: Any, *, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} artifact record is absent.")
    path = require_regular(Path(str(record.get("path", ""))), label=label)
    if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} artifact content differs from its record.")
    return path


def reject_tree_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Tree is missing or symlinked: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing to copy symlinked historical evidence: {path}")


def copy_file_new_or_identical(source: Path, destination: Path) -> None:
    source = require_regular(source, label="copy source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.stat().st_size != source.stat().st_size
            or sha256_file(destination) != sha256_file(source)
        ):
            raise FileExistsError(f"Existing copied artifact differs: {destination}")
        return
    shutil.copy2(source, destination)


def copy_tree_new_or_identical(source: Path, destination: Path) -> None:
    reject_tree_symlinks(source)
    if not destination.exists():
        shutil.copytree(source, destination)
        return
    reject_tree_symlinks(destination)
    source_files = {
        path.relative_to(source): path
        for path in source.rglob("*")
        if path.is_file()
    }
    destination_files = {
        path.relative_to(destination): path
        for path in destination.rglob("*")
        if path.is_file()
    }
    if set(source_files) != set(destination_files):
        raise FileExistsError(f"Existing copied tree file set differs: {destination}")
    for relative, source_path in source_files.items():
        destination_path = destination_files[relative]
        if (
            source_path.stat().st_size != destination_path.stat().st_size
            or sha256_file(source_path) != sha256_file(destination_path)
        ):
            raise FileExistsError(
                f"Existing copied tree content differs: {destination_path}"
            )


def source_fingerprint(source_root: Path) -> str:
    scripts = source_root / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        module_path = scripts / "run_isaac_graphics_gate.py"
        spec = importlib.util.spec_from_file_location("publication_isaac_gate", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the canonical Isaac gate.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        fingerprint = module.source_fingerprint(source_root)
    finally:
        sys.path.remove(str(scripts))
    value = fingerprint.get("content_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("Canonical Isaac source fingerprint is malformed.")
    return value


def command_adapter(
    *,
    python: Path,
    tools_root: Path,
    gate_id: str,
    checks: Path,
    source_manifest: Path,
    native_extension: Path,
    gpu_uuid: str | None,
    evidence: dict[str, Path],
    result_spec: Path,
    child: list[str],
) -> list[str]:
    command = [
        str(python),
        str(tools_root / "run_gate_and_write_result.py"),
        "--gate-id",
        gate_id,
        "--checks",
        str(checks),
        "--source-manifest",
        str(source_manifest),
        "--native-extension",
        str(native_extension),
        "--scene-sha256",
        SCENE_SHA256,
    ]
    if gpu_uuid is not None:
        command.extend(["--gpu-uuid", gpu_uuid])
    for key, path in sorted(evidence.items()):
        command.extend(["--evidence", f"{key}={path}"])
    command.extend(["--result-spec", str(result_spec), "--", *child])
    return command


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    source_root = args.source_root.resolve(strict=True)
    matrix = args.matrix_root.resolve(strict=True)
    lexical_output = args.output_root.absolute()
    output = lexical_output.resolve(strict=False)
    tools = args.tools_root.resolve(strict=True)
    python = require_regular(
        args.python,
        label="benchmark Python",
        executable=True,
        allow_symlink=True,
    )
    uv = require_regular(args.uv, label="uv", executable=True, allow_symlink=True)
    scene = require_regular(args.scene_path, label="Home Scan scene")
    gsplat_source = args.gsplat_source.resolve(strict=True)
    historical = args.historical_b64_root.resolve(strict=True)
    diagnosis_source = require_regular(args.diagnosis_index, label="B64 diagnosis index")
    if not output.is_relative_to(matrix) or output == matrix:
        raise ValueError(
            "Post-matrix output root must be a fresh child of the canonical matrix root."
        )
    reject_symlink_components(
        lexical_output,
        boundary=matrix,
        label="post-matrix output root",
    )
    if output.exists() and not args.resume:
        raise FileExistsError(f"Post-matrix output root must be absent: {output}")
    if args.resume and (not output.exists() or not output.is_dir()):
        raise FileNotFoundError("--resume requires an existing regular output directory.")
    if sha256_file(scene) != SCENE_SHA256:
        raise ValueError("Home Scan scene SHA-256 differs from the publication contract.")
    if not gsplat_source.is_dir() or gsplat_source.is_symlink():
        raise ValueError("Pinned gsplat source is missing or symlinked.")
    if not args.canonical_runtime.resolve(strict=True).is_dir():
        raise ValueError("Canonical Isaac runtime is missing.")

    summary_path = matrix / "summary.json"
    summary = load_json(summary_path, label="canonical matrix summary")
    if not all(
        summary.get(field) is True
        for field in ("pass", "scientific_pass", "headline_eligible", "primary_contract_eligible")
    ):
        raise RuntimeError("Frozen matrix is not a complete passing headline result.")
    source_manifest = require_regular(
        matrix / "provenance/source-manifest.json",
        label="matrix source manifest",
    )
    source_payload = load_json(source_manifest, label="matrix source manifest")
    commit = source_payload.get("head")
    if source_payload.get("dirty") is not False or not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("Matrix source manifest is not a clean full commit.")
    actual_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    actual_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=source_root,
        text=True,
    ).splitlines()
    if actual_head != commit or actual_status != source_payload.get("status_short"):
        raise RuntimeError("Frozen benchmark checkout differs from the matrix source manifest.")
    try:
        tagged_commit = subprocess.check_output(
            ["git", "rev-parse", f"{args.benchmark_tag}^{{commit}}"],
            cwd=source_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Benchmark tag must resolve before any post-matrix gate runs: {args.benchmark_tag}"
        ) from error
    if tagged_commit != commit:
        raise RuntimeError(
            f"Benchmark tag resolves to {tagged_commit}, not frozen commit {commit}."
        )

    verification_output = matrix / "publication/verification.json"
    ablation_raw = matrix / "publication/evidence/custom-vectorization-ablation.json"
    reject_symlink_components(
        verification_output,
        boundary=matrix,
        label="canonical publication verification",
    )
    reject_symlink_components(
        ablation_raw,
        boundary=matrix,
        label="canonical vectorization ablation",
    )
    if verification_output.exists() or verification_output.is_symlink():
        raise FileExistsError(
            "The post-matrix coordinator requires the canonical verification path "
            "to be absent before any gate runs. Validate an already completed "
            "publication directly with aggregate_verification.py instead."
        )
    if not args.resume and (ablation_raw.exists() or ablation_raw.is_symlink()):
        raise FileExistsError(
            "Fresh post-matrix execution requires an absent canonical ablation alias."
        )

    custom_run = load_json(matrix / "runs/custom/full/b1.json", label="Custom B1 run")
    flashgs_run = load_json(matrix / "runs/flashgs/full/b1.json", label="FlashGS-derived B1 run")
    custom_native = artifact_path(
        (custom_run.get("environment") or {}).get("native_extension"),
        label="Custom matrix native extension",
    )
    flashgs_native = artifact_path(
        (flashgs_run.get("environment") or {}).get("native_extension"),
        label="FlashGS-derived matrix native extension",
    )
    adapter = require_regular(
        matrix / "provenance/flashgs-adapter-attestation.json",
        label="FlashGS adapter attestation",
    )
    gsplat_attestation = require_regular(
        matrix / "provenance/gsplat-build-attestation.json",
        label="gsplat build attestation",
    )

    raw_root = output / "raw"
    specs = output / "result-specs"
    gates_root = output / "gates"
    if args.resume:
        require_real_directory(output, boundary=matrix, label="resumed output root")
        # Resume is allowed only from an entirely real, lane-owned tree.  Some
        # support producers populate nested B64/ablation directories before
        # their bounded wrapper is consulted, so checking only top-level
        # directories would permit writes through a descendant symlink.
        reject_tree_symlinks(output)
        for directory, label in (
            (raw_root, "resumed raw root"),
            (specs, "resumed result-spec root"),
            (gates_root, "resumed gates root"),
        ):
            require_real_directory(directory, boundary=output, label=label)
        for optional, label in (
            (output / "ablation", "resumed ablation root"),
            (
                raw_root / "deterministic-replay-tooling",
                "resumed deterministic replay tooling root",
            ),
        ):
            if optional.exists() or optional.is_symlink():
                require_real_directory(optional, boundary=output, label=label)
        require_resume_ablation_alias(
            output=output,
            matrix=matrix,
            alias=ablation_raw,
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
        raw_root.mkdir()
        specs.mkdir()
        gates_root.mkdir()
    checks_root = tools / "gate-checks"
    bounded_runner = tools / "run_bounded_gate.py"
    capture_runner = tools / "run_capture_sentinel.py"
    success_runner = tools / "run_with_success_sentinel.py"
    ablation_runner = tools / "run_custom_vectorization_ablation.py"
    for required in (
        bounded_runner,
        capture_runner,
        success_runner,
        ablation_runner,
        tools / "run_gate_and_write_result.py",
        tools / "run_deterministic_replay.py",
        tools / "deterministic_digest_smoke.py",
        tools / "run_pytest_gate.py",
        tools / "write_machine_provenance.py",
        tools / "write_verification.py",
        tools / "aggregate_verification.py",
    ):
        require_regular(required, label=f"publication tool {required.name}")
    publication_test_paths = sorted(tools.glob("test_*.py"))
    publication_test_paths.extend(
        [
            tools / "self_test_aggregate_verification.py",
            tools / "self_test_verify_release.py",
        ]
    )
    if not publication_test_paths:
        raise RuntimeError("No publication-tool tests were discovered.")
    for test_path in publication_test_paths:
        require_regular(test_path, label=f"publication test {test_path.name}")

    environment = os.environ.copy()
    environment["VGR_PROJECT_ROOT"] = str(source_root)
    environment["PROJECT_ROOT"] = str(source_root)
    environment["VGR_NATIVE_BUILD_ROOT"] = str(matrix / "provenance/native-extensions")
    environment["TORCH_EXTENSIONS_DIR"] = str(matrix / "provenance/torch-extensions")
    environment["TORCH_CUDA_ARCH_LIST"] = "8.9"
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(tools.parent),
            str(source_root / "src"),
            str(source_root),
            existing_pythonpath,
        )
        if value
    )
    environment["VGR_GPU_EXECUTOR_LOCK_PATH"] = "/tmp/vgr-publication-7fb2981-gpu-executor.lock"
    environment.pop("VGR_GPU_EXECUTOR_LOCK_OWNER_PID", None)
    environment["ACCEPT_EULA"] = "Y"

    wrappers: dict[str, dict[str, Any]] = {}

    def run_bounded(
        gate_id: str,
        command: list[str],
        *,
        timeout: int,
        result_spec: Path | None = None,
        subcommands: Path | None = None,
        required_gate: bool = True,
    ) -> dict[str, Any]:
        directory = gates_root / gate_id
        expected_subcommands: list[list[str]] = []
        if subcommands is not None:
            value = json.loads(subcommands.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise ValueError(f"Subcommand plan is malformed: {subcommands}")
            expected_subcommands = value
        if directory.exists() or directory.is_symlink():
            if not args.resume or directory.is_symlink() or not directory.is_dir():
                raise FileExistsError(f"Refusing to reuse bounded gate directory: {directory}")
            command_record = load_json(
                directory / "command.json",
                label=f"{gate_id} resumed command record",
            )
            expected_command_fields = {
                "schema_version": "publication-command-record-v1",
                "gate_id": gate_id,
                "timeout_seconds": timeout,
                "command": command,
                "subcommands": expected_subcommands,
            }
            for key, expected in expected_command_fields.items():
                if command_record.get(key) != expected:
                    raise RuntimeError(
                        f"Resumed {gate_id} command field {key} differs."
                    )
            wrapper = load_json(
                directory / "wrapper.json",
                label=f"{gate_id} resumed wrapper",
            )
            occupancy = load_json(
                directory / "node-occupancy.json",
                label=f"{gate_id} resumed occupancy",
            )
            exit_status = require_regular(
                directory / "exit-status.txt",
                label=f"{gate_id} resumed exit status",
            ).read_text(encoding="utf-8").strip()
            log_path = require_regular(
                directory / "full.log",
                label=f"{gate_id} resumed log",
            )
            if (
                wrapper.get("pass") is not True
                or wrapper.get("returncode") != 0
                or wrapper.get("timed_out") is not False
                or occupancy.get("pass") is not True
                or exit_status != "0"
                or not log_path.read_text(encoding="utf-8").strip()
            ):
                raise RuntimeError(f"Refusing to resume failed/incomplete gate: {gate_id}")
            if result_spec is not None:
                require_regular(result_spec, label=f"{gate_id} resumed result spec")
                require_regular(
                    directory / "result.json",
                    label=f"{gate_id} resumed normalized result",
                )
                if wrapper.get("result") != str(directory / "result.json"):
                    raise RuntimeError(f"Resumed {gate_id} result path differs.")
            elif wrapper.get("result") is not None:
                raise RuntimeError(f"Resumed support gate unexpectedly has a result: {gate_id}")
            if required_gate:
                wrappers[gate_id] = wrapper
            print(f"POST_MATRIX_GATE_RESUMED {gate_id}", flush=True)
            return wrapper
        wrapper_command = [
            str(python),
            str(bounded_runner),
            "--gate-id",
            gate_id,
            "--source-root",
            str(source_root),
            "--output-dir",
            str(directory),
            "--expected-gpu-uuid",
            args.expected_gpu_uuid,
            "--timeout-seconds",
            str(timeout),
        ]
        if subcommands is not None:
            wrapper_command.extend(["--subcommands-json", str(subcommands)])
        if result_spec is not None:
            wrapper_command.extend(["--result-spec", str(result_spec)])
        wrapper_command.extend(["--", *command])
        subprocess.run(wrapper_command, cwd=source_root, env=environment, check=True)
        wrapper = load_json(directory / "wrapper.json", label=f"{gate_id} wrapper")
        if wrapper.get("pass") is not True or wrapper.get("returncode") != 0:
            raise RuntimeError(f"Bounded gate failed: {gate_id}")
        if result_spec is not None and wrapper.get("result") is None:
            raise RuntimeError(f"Bounded gate produced no normalized result: {gate_id}")
        if required_gate:
            wrappers[gate_id] = wrapper
        return wrapper

    # Start with the cheapest exact-current gates before any large-scene repair,
    # sanitizer, ablation, or soak work.
    junit = raw_root / "pytest.xml"
    unit_spec = specs / "python-unit-contract.json"
    unit_command = [
        str(python),
        str(tools / "run_pytest_gate.py"),
        "--junit-xml",
        str(junit),
        "--result-spec",
        str(unit_spec),
        "--source-manifest",
        str(source_manifest),
        "--native-extension",
        str(custom_native),
        "--scene-sha256",
        SCENE_SHA256,
    ]
    for fragment in (
        "test_bundle_is_relocatable_and_archive_is_deterministic",
        "test_primary_physical_schedule_is_exact",
        "test_verify_existing_cli_does_not_rewrite_summary",
        "test_node_occupancy_evidence_fails_closed",
        "test_repaired_compositor_schedule_preserves_feature_sources",
        "test_publication_code_freeze_rejects_load_bearing_changes",
        "test_two_fresh_equal_processes_are_content_addressed",
        "test_replay_inert_bound_runner_token_fails_with_refreshed_command",
        "test_b64_missing_transitive_evidence_key_fails_with_refreshed_graph",
        "test_s3_put_is_signed_and_conditionally_create_only",
        "test_rejects_non_public_repository_before_network_record",
        "test_inconsistent_primary_run_environment_fails",
    ):
        unit_command.extend(["--required-test-fragment", fragment])
    unit_command.extend(
        [
            "--",
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests",
            *(str(path) for path in publication_test_paths),
            f"--junitxml={junit}",
        ]
    )
    run_bounded(
        "python-unit-contract",
        unit_command,
        timeout=7200,
        result_spec=unit_spec,
    )

    def native_gate(
        gate_id: str,
        script_name: str,
        sentinel: str,
        payload_format: str,
        extra: list[str],
    ) -> None:
        raw = raw_root / f"{gate_id}.json"
        result_spec = specs / f"{gate_id}.json"
        capture = [
            str(python),
            str(capture_runner),
            "--sentinel",
            sentinel,
            "--format",
            payload_format,
            "--output",
            str(raw),
            "--",
            str(python),
            str(source_root / f"scripts/{script_name}"),
            *extra,
        ]
        run_bounded(
            gate_id,
            command_adapter(
                python=python,
                tools_root=tools,
                gate_id=gate_id,
                checks=checks_root / f"{gate_id}.json",
                source_manifest=source_manifest,
                native_extension=custom_native,
                gpu_uuid=args.expected_gpu_uuid,
                evidence={"raw_result": raw},
                result_spec=result_spec,
                child=capture,
            ),
            timeout=7200,
            result_spec=result_spec,
        )

    native_gate(
        "native-cuda-smoke",
        "custom_backend_smoke.py",
        "CUSTOM_CUDA_BACKEND_SMOKE_OK",
        "python",
        [],
    )
    deterministic_arguments = [
        "--batch",
        "4",
        "--gaussians",
        "1024",
        "--iterations",
        "32",
        "--width",
        "64",
        "--height",
        "64",
        "--tile-size",
        "1",
    ]
    native_gate(
        "native-determinism",
        "deterministic_cuda_smoke.py",
        "DETERMINISTIC_CUDA_SMOKE_OK",
        "json",
        deterministic_arguments,
    )
    native_gate(
        "native-multi-scene",
        "multi_scene_cuda_smoke.py",
        "MULTI_SCENE_CUDA_SMOKE_OK",
        "python",
        [],
    )

    # Canonical finite lifecycle/extension/Fabric Isaac gate.
    isaac_output = raw_root / "isaac-graphics"
    isaac_summary = isaac_output / "summary.json"
    isaac_spec = specs / "isaac-graphics.json"
    isaac_child = [
        str(python),
        str(source_root / "scripts/run_isaac_graphics_gate.py"),
        "--python",
        str(python),
        "--uv",
        str(uv),
        "--canonical-runtime",
        str(args.canonical_runtime.resolve()),
        "--source-root",
        str(source_root),
        "--output-root",
        str(isaac_output),
        "--source-commit",
        commit,
        "--source-sha256",
        source_fingerprint(source_root),
        "--timeout-seconds",
        "300",
    ]
    run_bounded(
        "isaac-graphics",
        command_adapter(
            python=python,
            tools_root=tools,
            gate_id="isaac-graphics",
            checks=checks_root / "isaac-graphics.json",
            source_manifest=source_manifest,
            native_extension=custom_native,
            gpu_uuid=args.expected_gpu_uuid,
            evidence={"isaac_summary": isaac_summary},
            result_spec=isaac_spec,
            child=isaac_child,
        ),
        timeout=1800,
        result_spec=isaac_spec,
    )

    # Reconstruct the canonical matrix in read-only mode before later gates.
    fidelity_spec = specs / "all-output-fidelity.json"
    fidelity_child = [
        str(python),
        str(source_root / "benchmarks/summarize_flashgs_matched.py"),
        "--root",
        str(matrix),
        "--batches",
        BATCHES,
        "--output",
        str(summary_path),
        "--verify-existing",
    ]
    run_bounded(
        "all-output-fidelity",
        command_adapter(
            python=python,
            tools_root=tools,
            gate_id="all-output-fidelity",
            checks=checks_root / "all-output-fidelity.json",
            source_manifest=source_manifest,
            native_extension=custom_native,
            gpu_uuid=args.expected_gpu_uuid,
            evidence={"raw_result": summary_path},
            result_spec=fidelity_spec,
            child=fidelity_child,
        ),
        timeout=7200,
        result_spec=fidelity_spec,
    )

    # Exact-current B64 repair and fresh pinned-gsplat oracle.
    b64_artifacts = raw_root / "b64-artifacts"
    historical_target = (
        b64_artifacts / "outputs/publication-readiness/b64-short-tail-pre-fix-5eca4e6"
    )
    historical_target.parent.mkdir(parents=True, exist_ok=True)
    copy_tree_new_or_identical(historical, historical_target)
    tracked_target = b64_artifacts / "experiments/flashgs_matched"
    tracked_target.mkdir(parents=True, exist_ok=True)
    known = tracked_target / "B64_KNOWN_FAILURE_CASES.json"
    lock = tracked_target / "B64_DIAGNOSIS_LOCK.json"
    copy_file_new_or_identical(
        source_root / "experiments/flashgs_matched/B64_KNOWN_FAILURE_CASES.json",
        known,
    )
    copy_file_new_or_identical(
        source_root / "experiments/flashgs_matched/B64_DIAGNOSIS_LOCK.json",
        lock,
    )
    diagnosis = b64_artifacts / "evidence/diagnostics/diagnosis-index.json"
    copy_file_new_or_identical(diagnosis_source, diagnosis)
    repair_root = b64_artifacts / "evidence/repair"
    repair_root.mkdir(parents=True, exist_ok=True)
    survey = load_json(
        matrix / "capacity/flashgs/b1024-demand-survey.json",
        label="FlashGS demand survey",
    )
    b64_capacity = (
        (survey.get("derived_batch_capacities") or {}).get("64") or {}
    ).get("installed_intersections_per_camera")
    if not isinstance(b64_capacity, int) or b64_capacity <= 0:
        raise RuntimeError("FlashGS B64 demand-survey capacity is absent.")
    trajectory_b64 = matrix / "contracts/b64.json"

    def repair_command(destination: Path) -> list[str]:
        return [
            str(python),
            str(source_root / "benchmarks/verify_flashgs_b64_repair.py"),
            "--trajectory",
            str(trajectory_b64),
            "--scene-path",
            str(scene),
            "--scene-sha256",
            SCENE_SHA256,
            "--known-failure-manifest",
            str(known),
            "--diagnosis-lock",
            str(lock),
            "--diagnosis-index",
            str(diagnosis),
            "--diagnosis-artifact-root",
            str(b64_artifacts),
            "--intersection-capacity",
            str(b64_capacity),
            "--semantic-topology",
            "spatial-octants-8",
            "--expected-gpu-uuid",
            args.expected_gpu_uuid,
            "--source-manifest",
            str(source_manifest),
            "--flashgs-adapter-attestation",
            str(adapter),
            "--output",
            str(destination),
        ]

    repair_primary = repair_root / "b64-repair-primary.json"
    repair_repeat = repair_root / "b64-repair-repeat.json"
    run_bounded(
        "support-b64-repair-primary",
        repair_command(repair_primary),
        timeout=7200,
        required_gate=False,
    )
    run_bounded(
        "support-b64-repair-repeat",
        repair_command(repair_repeat),
        timeout=7200,
        required_gate=False,
    )
    oracle = b64_artifacts / "evidence/oracle/b64.npz"
    oracle.parent.mkdir(parents=True, exist_ok=True)
    run_bounded(
        "support-b64-gsplat-oracle",
        [
            str(python),
            str(source_root / "benchmarks/run_flashgs_gsplat_oracle.py"),
            "--trajectory",
            str(trajectory_b64),
            "--scene-path",
            str(scene),
            "--step",
            "107",
            "--max-views",
            "64",
            "--gsplat-source",
            str(gsplat_source),
            "--semantic-topology",
            "spatial-octants-8",
            "--source-manifest",
            str(source_manifest),
            "--gsplat-build-attestation",
            str(gsplat_attestation),
            "--expected-gpu-uuid",
            args.expected_gpu_uuid,
            "--output",
            str(oracle),
        ],
        timeout=7200,
        required_gate=False,
    )
    oracle_report = repair_root / "b64-repair-gsplat-all-pixel.json"
    b64_spec = specs / "flashgs-b64-repair.json"
    primary_support = gates_root / "support-b64-repair-primary"
    repeat_support = gates_root / "support-b64-repair-repeat"
    oracle_support = gates_root / "support-b64-gsplat-oracle"
    b64_evidence = {
        "diagnosis_index": diagnosis,
        "diagnosis_lock": lock,
        "known_failure_manifest": known,
        "oracle": oracle,
        "oracle_camera_bundle": oracle.with_suffix(".camera-bundle.json"),
        "oracle_manifest": oracle.with_suffix(".manifest.json"),
        "oracle_node_occupancy": oracle.with_suffix(".node-occupancy.json"),
        "oracle_report": oracle_report,
        "oracle_support_command": oracle_support / "command.json",
        "oracle_support_exit_status": oracle_support / "exit-status.txt",
        "oracle_support_log": oracle_support / "full.log",
        "oracle_support_occupancy": oracle_support / "node-occupancy.json",
        "oracle_support_wrapper": oracle_support / "wrapper.json",
        "primary_repair_full_raw": repair_primary.with_suffix(".full.npz"),
        "primary_repair_rgb_only_raw": repair_primary.with_suffix(".rgb-only.npz"),
        "primary_support_command": primary_support / "command.json",
        "primary_support_exit_status": primary_support / "exit-status.txt",
        "primary_support_log": primary_support / "full.log",
        "primary_support_occupancy": primary_support / "node-occupancy.json",
        "primary_support_wrapper": primary_support / "wrapper.json",
        "repair_report": repair_primary,
        "repeat_repair_full_raw": repair_repeat.with_suffix(".full.npz"),
        "repeat_repair_report": repair_repeat,
        "repeat_repair_rgb_only_raw": repair_repeat.with_suffix(".rgb-only.npz"),
        "repeat_support_command": repeat_support / "command.json",
        "repeat_support_exit_status": repeat_support / "exit-status.txt",
        "repeat_support_log": repeat_support / "full.log",
        "repeat_support_occupancy": repeat_support / "node-occupancy.json",
        "repeat_support_wrapper": repeat_support / "wrapper.json",
    }
    b64_child = [
        str(python),
        str(source_root / "benchmarks/compare_flashgs_b64_repair_oracle.py"),
        "--repair-report",
        str(repair_primary),
        "--repeat-repair-report",
        str(repair_repeat),
        "--diagnosis-index",
        str(diagnosis),
        "--diagnosis-lock",
        str(lock),
        "--artifact-root",
        str(b64_artifacts),
        "--oracle",
        str(oracle),
        "--oracle-manifest",
        str(oracle.with_suffix(".manifest.json")),
        "--oracle-camera-bundle",
        str(oracle.with_suffix(".camera-bundle.json")),
        "--output",
        str(oracle_report),
    ]
    run_bounded(
        "flashgs-b64-repair",
        command_adapter(
            python=python,
            tools_root=tools,
            gate_id="flashgs-b64-repair",
            checks=checks_root / "flashgs-b64-repair.json",
            source_manifest=source_manifest,
            native_extension=flashgs_native,
            gpu_uuid=args.expected_gpu_uuid,
            evidence=b64_evidence,
            result_spec=b64_spec,
            child=b64_child,
        ),
        timeout=1800,
        result_spec=b64_spec,
    )

    # The repaired FlashGS short-tail path under memcheck.
    repair_sanitizer_raw = raw_root / "repair-short-tail-synthetic.json"
    repair_sanitizer_spec = specs / "repair-compute-sanitizer.json"
    repair_sanitizer_child = [
        "compute-sanitizer",
        "--target-processes",
        "all",
        "--tool",
        "memcheck",
        "--error-exitcode",
        "99",
        str(python),
        str(source_root / "benchmarks/verify_flashgs_short_tail_synthetic.py"),
        "--source-manifest",
        str(source_manifest),
        "--flashgs-adapter-attestation",
        str(adapter),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "--output",
        str(repair_sanitizer_raw),
    ]
    run_bounded(
        "repair-compute-sanitizer",
        command_adapter(
            python=python,
            tools_root=tools,
            gate_id="repair-compute-sanitizer",
            checks=checks_root / "sanitizer.json",
            source_manifest=source_manifest,
            native_extension=flashgs_native,
            gpu_uuid=args.expected_gpu_uuid,
            evidence={"target_result": repair_sanitizer_raw},
            result_spec=repair_sanitizer_spec,
            child=repair_sanitizer_child,
        ),
        timeout=7200,
        result_spec=repair_sanitizer_spec,
    )

    # Exact P1 sequential versus one P128 physical submission.  Plan first so
    # the outer command record can bind all twelve fresh-process subcommands.
    ablation_root = output / "ablation"
    ablation_spec = specs / "custom-vectorization-ablation.json"
    ablation_common = [
        str(python),
        str(ablation_runner),
        "--benchmark-root",
        str(source_root),
        "--python",
        str(python),
        "--scene-path",
        str(scene),
        "--scene-sha256",
        SCENE_SHA256,
        "--source-manifest",
        str(source_manifest),
        "--trajectory",
        str(matrix / "contracts/b128.json"),
        "--p128-capacity",
        str(matrix / "capacity/custom/b128.json"),
        "--matrix-summary",
        str(summary_path),
        "--oracle",
        str(matrix / "oracle/b128.npz"),
        "--camera-bundle",
        str(matrix / "oracle/b128.camera-bundle.json"),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
        "--output-root",
        str(ablation_root),
        "--raw-output",
        str(ablation_raw),
        "--gate-result-spec",
        str(ablation_spec),
        "--semantic-topology",
        "spatial-octants-8",
        "--stage-timeout-seconds",
        "7200",
    ]
    subprocess.run([*ablation_common, "--plan-only"], cwd=source_root, env=environment, check=True)
    run_bounded(
        "custom-vectorization-ablation",
        ablation_common,
        timeout=7200,
        result_spec=ablation_spec,
        subcommands=ablation_root / "timed-subcommands.json",
    )

    replay_raw = raw_root / "deterministic-replay.json"
    replay_process_1 = replay_raw.with_name(
        f"{replay_raw.stem}.process-1{replay_raw.suffix}"
    )
    replay_process_2 = replay_raw.with_name(
        f"{replay_raw.stem}.process-2{replay_raw.suffix}"
    )
    replay_tooling = raw_root / "deterministic-replay-tooling"
    replay_tooling.mkdir(parents=True, exist_ok=args.resume)
    replay_runner = replay_tooling / "run_deterministic_replay.py"
    replay_worker = replay_tooling / "deterministic_digest_smoke.py"
    replay_source_smoke = replay_tooling / "deterministic_cuda_smoke.py"
    copy_file_new_or_identical(tools / "run_deterministic_replay.py", replay_runner)
    copy_file_new_or_identical(tools / "deterministic_digest_smoke.py", replay_worker)
    copy_file_new_or_identical(
        source_root / "scripts/deterministic_cuda_smoke.py",
        replay_source_smoke,
    )
    replay_spec = specs / "deterministic-replay.json"
    replay_child = [
        str(python),
        str(replay_runner),
        "--python",
        str(python),
        "--source-root",
        str(source_root),
        "--worker-script",
        str(replay_worker),
        "--source-smoke",
        str(replay_source_smoke),
        "--output",
        str(replay_raw),
        *deterministic_arguments,
    ]
    run_bounded(
        "deterministic-replay",
        command_adapter(
            python=python,
            tools_root=tools,
            gate_id="deterministic-replay",
            checks=checks_root / "deterministic-replay.json",
            source_manifest=source_manifest,
            native_extension=custom_native,
            gpu_uuid=args.expected_gpu_uuid,
            evidence={
                "process_1": replay_process_1,
                "process_2": replay_process_2,
                "raw_result": replay_raw,
                "runner_script": replay_runner,
                "source_smoke": replay_source_smoke,
                "worker_script": replay_worker,
            },
            result_spec=replay_spec,
            child=replay_child,
        ),
        timeout=7200,
        result_spec=replay_spec,
    )

    # Custom backend under Compute Sanitizer.  The sentinel payload is captured
    # into immutable JSON and must exactly match the sole log payload.
    custom_sanitizer_raw = raw_root / "compute-sanitizer-custom.json"
    custom_sanitizer_spec = specs / "compute-sanitizer-memcheck.json"
    sanitizer_capture = [
        str(python),
        str(capture_runner),
        "--sentinel",
        "CUSTOM_CUDA_BACKEND_SMOKE_OK",
        "--format",
        "python",
        "--output",
        str(custom_sanitizer_raw),
        "--",
        "compute-sanitizer",
        "--target-processes",
        "all",
        "--tool",
        "memcheck",
        "--error-exitcode",
        "99",
        str(python),
        str(source_root / "scripts/custom_backend_smoke.py"),
    ]
    run_bounded(
        "compute-sanitizer-memcheck",
        command_adapter(
            python=python,
            tools_root=tools,
            gate_id="compute-sanitizer-memcheck",
            checks=checks_root / "sanitizer.json",
            source_manifest=source_manifest,
            native_extension=custom_native,
            gpu_uuid=args.expected_gpu_uuid,
            evidence={"target_result": custom_sanitizer_raw},
            result_spec=custom_sanitizer_spec,
            child=sanitizer_capture,
        ),
        timeout=7200,
        result_spec=custom_sanitizer_spec,
    )

    # Fixed-camera stability only; it is not a dynamic-camera performance row.
    b128_capacity_payload = load_json(
        matrix / "capacity/custom/b128.json",
        label="Custom B128 capacity",
    )
    installed = (b128_capacity_payload.get("capacity") or {}).get("installed") or {}
    visible_capacity = installed.get("visible_records")
    intersection_capacity = installed.get("intersections")
    if not all(isinstance(value, int) and value > 0 for value in (visible_capacity, intersection_capacity)):
        raise RuntimeError("Custom B128 installed capacities are missing.")
    soak_raw = raw_root / "bounded-stability-soak.json"
    soak_artifacts = raw_root / "bounded-stability-soak-artifacts"
    soak_spec = specs / "bounded-stability-soak.json"
    soak_child = [
        str(python),
        str(success_runner),
        "--sentinel",
        "CUSTOM_CUDA_SOAK_OK",
        "--",
        str(python),
        str(source_root / "benchmarks/run_home_scan.py"),
        "--path",
        str(scene),
        "--batch",
        "128",
        "--width",
        "128",
        "--height",
        "128",
        "--warmup",
        "8",
        "--iterations",
        "1",
        "--duration-seconds",
        "600",
        "--renders-per-check",
        "10",
        "--visible-capacity",
        str(visible_capacity),
        "--intersection-capacity",
        str(intersection_capacity),
        "--gaussian-support-sigma",
        "3.33",
        "--covariance-epsilon",
        "0",
        "--rasterize-mode",
        "classic",
        "--semantic-min-alpha",
        "0.01",
        "--semantic-scheme",
        "spatial-grid",
        "--semantic-grid",
        "2,2,2",
        "--tile-size",
        "1",
        "--depth-bucket-count",
        "128",
        "--depth-bucket-group-size",
        "8",
        "--compact-projection-cache",
        "--linear-output",
        "--output",
        str(soak_raw),
        "--artifact-dir",
        str(soak_artifacts),
    ]
    run_bounded(
        "bounded-stability-soak",
        command_adapter(
            python=python,
            tools_root=tools,
            gate_id="bounded-stability-soak",
            checks=checks_root / "bounded-stability-soak.json",
            source_manifest=source_manifest,
            native_extension=custom_native,
            gpu_uuid=args.expected_gpu_uuid,
            evidence={"soak_result": soak_raw},
            result_spec=soak_spec,
            child=soak_child,
        ),
        timeout=900,
        result_spec=soak_spec,
    )

    if len(wrappers) != len(REQUIRED_GATES) or set(wrappers) != set(REQUIRED_GATES):
        raise RuntimeError(
            f"Required gate set differs: expected={REQUIRED_GATES}, observed={tuple(wrappers)}"
        )
    gate_specs: list[dict[str, Any]] = []
    for gate_id in REQUIRED_GATES:
        wrapper = wrappers[gate_id]
        native = flashgs_native if gate_id in FLASHGS_GATES else custom_native
        gate_specs.append(
            {
                "gate_id": gate_id,
                "result": wrapper["result"],
                "command": wrapper["command_record"],
                "log": wrapper["log"],
                "exit_status": wrapper["exit_status"],
                "source_manifest": str(source_manifest),
                "native_extension": str(native),
                "gpu_uuid": None if gate_id == "python-unit-contract" else args.expected_gpu_uuid,
                "occupancy": None if gate_id == "python-unit-contract" else wrapper["occupancy"],
                "started_at": wrapper["started_at"],
                "completed_at": wrapper["completed_at"],
            }
        )
    verification_spec = {
        "schema_version": "publication-verification-spec-v1",
        "benchmark": {
            "commit": commit,
            "tag": args.benchmark_tag,
            "repository_root": str(source_root),
            "source_manifest": str(source_manifest),
        },
        "matrix": {
            "summary": str(summary_path),
            "invocation": str(matrix / "provenance/matrix-invocation.json"),
            "custom_native_extension": str(custom_native),
            "flashgs_native_extension": str(flashgs_native),
            "gpu_uuid": args.expected_gpu_uuid,
            "scene_sha256": SCENE_SHA256,
        },
        "gates": gate_specs,
    }
    verification_spec_path = output / "verification-spec.json"
    write_new_or_identical(
        verification_spec_path,
        canonical_json_bytes(verification_spec),
    )
    machine_provenance = matrix / "publication/machine-provenance.json"
    subprocess.run(
        [
            str(python),
            str(tools / "write_machine_provenance.py"),
            "--matrix-root",
            str(matrix),
            "--output",
            str(machine_provenance),
            "--expected-gpu-uuid",
            args.expected_gpu_uuid,
        ],
        cwd=source_root,
        env=environment,
        check=True,
    )
    if verification_output.exists() or verification_output.is_symlink():
        raise FileExistsError(
            "The post-matrix coordinator refuses an existing canonical verification; "
            "this prevents an older verification from being mistaken for the current "
            "verification specification. Validate an already completed publication "
            "directly with aggregate_verification.py instead."
        )
    subprocess.run(
        [
            str(python),
            str(tools / "write_verification.py"),
            "--spec",
            str(verification_spec_path),
            "--output",
            str(verification_output),
        ],
        cwd=source_root,
        env=environment,
        check=True,
    )
    print(
        "POST_MATRIX_PUBLICATION_GATES_OK "
        + json.dumps(
            {
                "gate_count": len(gate_specs),
                "output_root": str(output),
                "verification_spec": str(verification_spec_path),
                "verification": str(verification_output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
