#!/usr/bin/env python3
"""Derive portable machine provenance from every primary matrix run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from verify_claim_ledger import (
    ACCEPTED_HARDWARE_NAME,
    ACCEPTED_HARDWARE_UUID,
    PRIMARY_BATCHES,
)


SCHEMA_VERSION = "publication-machine-provenance-v1"
RUN_SCHEMA = "flashgs-matched-renderer-run-v4"
SOURCE_MANIFEST_SCHEMA = "renderer-source-manifest-v2"
RENDERERS = ("custom", "flashgs")
CONTRACTS = ("full", "rgb")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
ENVIRONMENT_FIELDS = (
    "compiler_versions",
    "compute_capability",
    "cuda_runtime",
    "driver",
    "gpu_name",
    "gpu_uuid",
    "torch",
    "torch_cuda_arch_list",
)


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    return parser.parse_args(arguments)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_symlinks(path: Path, *, boundary: Path, label: str) -> None:
    candidate = path.absolute()
    root = boundary.absolute()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{label} escapes the matrix root: {candidate}")
    current = candidate
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} crosses a symlink: {current}")
        if current == root:
            return
        current = current.parent


def require_regular(path: Path, *, root: Path, label: str) -> Path:
    reject_symlinks(path, boundary=root, label=label)
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"{label} is not a contained regular file: {path}")
    return resolved


def load_json(path: Path, *, root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = require_regular(path, root=root, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return resolved, value


def logical_artifact(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = require_regular(path, root=root, label="publication artifact")
    return {
        "bytes": resolved.stat().st_size,
        "logical_path": resolved.relative_to(root.resolve(strict=True)).as_posix(),
        "sha256": file_sha256(resolved),
    }


def same_identity(record: Any, expected: dict[str, Any]) -> bool:
    return bool(
        isinstance(record, dict)
        and record.get("bytes") == expected["bytes"]
        and record.get("sha256") == expected["sha256"]
    )


def normalized_environment(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} has no environment object")
    environment = {field: value.get(field) for field in ENVIRONMENT_FIELDS}
    if (
        environment["gpu_name"] != ACCEPTED_HARDWARE_NAME
        or environment["gpu_uuid"] != ACCEPTED_HARDWARE_UUID
        or not isinstance(environment["driver"], str)
        or not environment["driver"]
        or not isinstance(environment["cuda_runtime"], str)
        or not environment["cuda_runtime"]
        or not isinstance(environment["torch"], str)
        or not environment["torch"]
        or environment["compute_capability"] != [8, 9]
        or environment["torch_cuda_arch_list"] != "8.9"
        or not isinstance(environment["compiler_versions"], dict)
        or set(environment["compiler_versions"]) != {"cxx", "nvcc"}
        or any(
            not isinstance(environment["compiler_versions"][field], str)
            or not environment["compiler_versions"][field]
            for field in ("cxx", "nvcc")
        )
    ):
        raise ValueError(f"{label} machine environment differs from the fixed L4 contract")
    return environment


def build_machine_provenance(
    matrix_root: Path,
    *,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    if matrix_root.is_symlink():
        raise ValueError("Matrix root may not be a symlink")
    root = matrix_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Matrix root is not a directory")
    if expected_gpu_uuid != ACCEPTED_HARDWARE_UUID:
        raise ValueError("Expected GPU UUID differs from the frozen publication L4")

    source_path, source = load_json(
        root / "provenance/source-manifest.json",
        root=root,
        label="matrix source manifest",
    )
    source_artifact = logical_artifact(source_path, root=root)
    source_commit = source.get("head")
    if (
        source.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or source.get("dirty") is not False
        or source.get("status_short") != []
        or not isinstance(source_commit, str)
        or GIT_SHA_RE.fullmatch(source_commit) is None
    ):
        raise ValueError("Matrix source manifest is not a clean frozen source identity")

    common_environment: dict[str, Any] | None = None
    run_artifacts: list[dict[str, Any]] = []
    for batch in PRIMARY_BATCHES:
        for renderer in RENDERERS:
            for contract in CONTRACTS:
                relative = Path(f"runs/{renderer}/{contract}/b{batch}.json")
                run_path, run = load_json(
                    root / relative,
                    root=root,
                    label=f"primary {renderer}/{contract}/B{batch} run",
                )
                if (
                    run.get("schema_version") != RUN_SCHEMA
                    or run.get("renderer") != renderer
                    or run.get("output_contract") != contract
                    or run.get("pass") is not True
                    or run.get("headline_eligible") is not True
                    or (run.get("capacity") or {}).get("logical_batch") != batch
                ):
                    raise ValueError(f"Primary run identity differs: {relative.as_posix()}")
                environment = run.get("environment")
                normalized = normalized_environment(
                    environment,
                    label=relative.as_posix(),
                )
                if common_environment is None:
                    common_environment = normalized
                elif normalized != common_environment:
                    raise ValueError(
                        f"Primary run machine environments disagree: {relative.as_posix()}"
                    )
                if (
                    not isinstance(environment, dict)
                    or environment.get("source_git_commit") != source_commit
                    or not same_identity(
                        ((environment.get("source_provenance") or {}).get("manifest")),
                        source_artifact,
                    )
                ):
                    raise ValueError(
                        f"Primary run source binding differs: {relative.as_posix()}"
                    )
                run_artifacts.append(logical_artifact(run_path, root=root))

    if common_environment is None:
        raise RuntimeError("No primary machine environments were observed")
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": True,
        "gpu_name": common_environment["gpu_name"],
        "gpu_uuid": common_environment["gpu_uuid"],
        "driver": common_environment["driver"],
        "cuda_runtime": common_environment["cuda_runtime"],
        "torch": common_environment["torch"],
        "compute_capability": common_environment["compute_capability"],
        "torch_cuda_arch_list": common_environment["torch_cuda_arch_list"],
        "compiler_versions": common_environment["compiler_versions"],
        "source_commit": source_commit,
        "source_manifest": source_artifact,
        "primary_run_count": len(run_artifacts),
        "primary_runs": run_artifacts,
    }


def write_new_or_identical(path: Path, payload: bytes, *, root: Path) -> None:
    reject_symlinks(path, boundary=root, label="machine provenance output")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite different machine provenance: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlinks(path, boundary=root, label="machine provenance output")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.matrix_root.is_symlink():
        raise ValueError("Matrix root may not be a symlink")
    root = args.matrix_root.resolve(strict=True)
    expected_output = root / "publication/machine-provenance.json"
    output = args.output.resolve(strict=False)
    if output != expected_output:
        raise ValueError(
            "Machine provenance output must be matrix/publication/machine-provenance.json"
        )
    result = build_machine_provenance(
        root,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    write_new_or_identical(
        output,
        canonical_json_bytes(result),
        root=root,
    )
    print(
        "PUBLICATION_MACHINE_PROVENANCE_OK "
        + json.dumps(
            {
                "output": str(output),
                "primary_run_count": result["primary_run_count"],
                "source_commit": result["source_commit"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
