#!/usr/bin/env python3
"""Create a byte-accounted, public-safe derivative of the B64 diagnosis.

The historical diagnosis is scientifically useful, but some captured files
retain build-machine home-directory paths.  Publishing those bytes verbatim is
unnecessary and violates the repository's public-artifact contract.  This
tool performs two deliberately narrow, length-preserving transformations:

* replace every recognized Linux or macOS host-user root with a deterministic
  same-length token below ``/work``; and
* update SHA-256 strings in JSON artifact records when a referenced file was
  changed by the first transformation or by a transitive hash update.

Every output byte is reversible from the recorded transformation classes.  No
numeric trace, tensor, source identity, command outcome, or culprit record is
edited.  The generated manifest contains hashes of the private roots rather
than the roots themselves, so the manifest is safe to publish too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "flashgs-b64-privacy-redaction-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRIVATE_ROOT_RE = re.compile(rb"/(?:home|Users)/[^/\s\x00:]+(?=/|$|[\"'\s,}\]\x00])")


class RedactionError(RuntimeError):
    """Raised when a diagnosis cannot be transformed without ambiguity."""


@dataclass(frozen=True)
class Node:
    label: str
    source: Path
    output: Path | None
    original: bytes
    original_sha256: str
    dependencies: frozenset[str]


@dataclass(frozen=True)
class TransformedNode:
    node: Node
    payload: bytes
    public_sha256: str
    path_replacements: int
    hash_rebindings: int


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _safe_relative(value: Path) -> str:
    text = value.as_posix()
    parsed = PurePosixPath(text)
    if parsed.is_absolute() or any(part in ("", ".", "..") for part in parsed.parts):
        raise RedactionError(f"Unsafe diagnosis-relative path: {text!r}.")
    return text


def _regular_tree_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise RedactionError(f"Historical diagnosis root is missing or symlinked: {root}.")
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise RedactionError(f"Historical diagnosis contains a symlink: {path}.")
        if path.is_file():
            files.append(path)
    if not files:
        raise RedactionError("Historical diagnosis tree is empty.")
    return files


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RedactionError(f"{label} is missing or symlinked: {path}.")
    return path


def _walk_artifact_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        for child in value.values():
            yield from _walk_artifact_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_artifact_records(child)


def _json_dependencies(payload: bytes, known_hashes: set[str]) -> frozenset[str]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return frozenset()
    dependencies: set[str] = set()
    for record in _walk_artifact_records(value):
        digest = record.get("sha256")
        if isinstance(digest, str) and digest in known_hashes:
            dependencies.add(digest)
    return frozenset(dependencies)


def _replacement_for(root: bytes) -> bytes:
    available = len(root) - len(b"/work/")
    if available < 1:
        raise RedactionError("Private root is too short for a safe same-length token.")
    token = (b"redacted_" + hashlib.sha256(root).hexdigest().encode("ascii"))[:available]
    replacement = b"/work/" + token
    if len(replacement) != len(root) or PRIVATE_ROOT_RE.fullmatch(replacement):
        raise RedactionError("Cannot construct a safe same-length private-root replacement.")
    return replacement


def _private_roots(nodes: Iterable[Node]) -> dict[bytes, bytes]:
    roots = sorted({match.group(0) for node in nodes for match in PRIVATE_ROOT_RE.finditer(node.original)})
    if not roots:
        raise RedactionError("Diagnosis contains no private roots to redact.")
    replacements = {root: _replacement_for(root) for root in roots}
    if len(set(replacements.values())) != len(replacements):
        raise RedactionError("Private-root replacement collision detected.")
    return replacements


def _replace_exact(payload: bytes, replacements: dict[bytes, bytes]) -> tuple[bytes, int]:
    count = 0
    transformed = payload
    for source, target in replacements.items():
        if source in payload and target in payload:
            raise RedactionError("A redaction target already occurs in source bytes.")
    for source, target in sorted(replacements.items(), key=lambda item: item[0]):
        if len(source) != len(target):
            raise RedactionError("A redaction replacement changed byte length.")
        occurrences = transformed.count(source)
        transformed = transformed.replace(source, target)
        count += occurrences
    return transformed, count


def _build_nodes(
    *,
    historical_root: Path,
    diagnosis_index: Path,
    diagnosis_lock: Path,
    known_failure_manifest: Path,
    output_root: Path,
    output_index: Path,
    output_lock: Path,
) -> list[Node]:
    sources: list[tuple[str, Path, Path | None]] = []
    for path in _regular_tree_files(historical_root):
        relative = path.relative_to(historical_root)
        sources.append(
            (
                f"historical/{_safe_relative(relative)}",
                path,
                output_root / relative,
            )
        )
    sources.extend(
        [
            (
                "diagnosis-index.json",
                _regular_file(diagnosis_index, label="Diagnosis index"),
                output_index,
            ),
            (
                "diagnosis-lock.json",
                _regular_file(diagnosis_lock, label="Diagnosis lock"),
                output_lock,
            ),
            (
                "known-failure-manifest.json",
                _regular_file(
                    known_failure_manifest,
                    label="Known-failure manifest",
                ),
                None,
            ),
        ]
    )
    original_by_source: dict[Path, bytes] = {}
    hashes: set[str] = set()
    for _label, source, _output in sources:
        original = source.read_bytes()
        original_by_source[source] = original
        hashes.add(sha256_bytes(original))
    nodes = []
    for label, source, output in sources:
        original = original_by_source[source]
        digest = sha256_bytes(original)
        nodes.append(
            Node(
                label=label,
                source=source,
                output=output,
                original=original,
                original_sha256=digest,
                dependencies=_json_dependencies(original, hashes),
            )
        )
    return nodes


def _transform_nodes(
    nodes: list[Node],
    private_roots: dict[bytes, bytes],
) -> dict[str, TransformedNode]:
    nodes_by_original: dict[str, list[Node]] = {}
    for node in nodes:
        nodes_by_original.setdefault(node.original_sha256, []).append(node)
    transformed_by_original: dict[str, TransformedNode] = {}
    pending = list(nodes)
    while pending:
        progress = False
        for node in list(pending):
            unresolved = node.dependencies - set(transformed_by_original)
            if unresolved:
                continue
            payload, path_count = _replace_exact(node.original, private_roots)
            hash_count = 0
            for original_digest in sorted(node.dependencies):
                public_digest = transformed_by_original[original_digest].public_sha256
                if public_digest == original_digest:
                    continue
                source = original_digest.encode("ascii")
                target = public_digest.encode("ascii")
                occurrences = payload.count(source)
                payload = payload.replace(source, target)
                hash_count += occurrences
            if len(payload) != len(node.original):
                raise RedactionError(f"Transformation changed byte length: {node.label}.")
            transformed = TransformedNode(
                node=node,
                payload=payload,
                public_sha256=sha256_bytes(payload),
                path_replacements=path_count,
                hash_rebindings=hash_count,
            )
            prior = transformed_by_original.get(node.original_sha256)
            if prior is not None and (
                prior.payload != transformed.payload or prior.public_sha256 != transformed.public_sha256
            ):
                raise RedactionError(
                    f"Files with identical source hashes transformed differently: {prior.node.label}, {node.label}."
                )
            transformed_by_original[node.original_sha256] = transformed
            pending.remove(node)
            progress = True
        if not progress:
            labels = ", ".join(sorted(node.label for node in pending))
            raise RedactionError(f"Artifact-reference graph contains a cycle: {labels}.")
    return transformed_by_original


def _assert_reversible(
    node: Node,
    transformed: TransformedNode,
    *,
    private_roots: dict[bytes, bytes],
    transformed_by_original: dict[str, TransformedNode],
) -> None:
    payload = transformed.payload
    reverse_hashes: dict[bytes, bytes] = {}
    for original_digest in sorted(node.dependencies):
        public_digest = transformed_by_original[original_digest].public_sha256
        if public_digest == original_digest:
            continue
        source = public_digest.encode("ascii")
        target = original_digest.encode("ascii")
        prior = reverse_hashes.get(source)
        if prior is not None and prior != target:
            raise RedactionError("Public dependency-hash collision detected.")
        reverse_hashes[source] = target
    for source, target in reverse_hashes.items():
        payload = payload.replace(source, target)
    for source, target in sorted(
        private_roots.items(),
        key=lambda item: (-len(item[1]), item[1]),
    ):
        payload = payload.replace(target, source)
    if payload != node.original:
        raise RedactionError(f"Transformation is not exactly reversible: {node.label}.")


def _write_new(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def verify_public_derivative(
    *,
    historical_root: Path,
    diagnosis_index: Path,
    diagnosis_lock: Path,
    known_failure_manifest: Path,
    manifest_path: Path,
    generator_path: Path | None = None,
) -> dict[str, Any]:
    """Revalidate the public derivative without requiring private source bytes."""

    historical_root = historical_root.resolve(strict=True)
    diagnosis_index = _regular_file(
        diagnosis_index.resolve(strict=True),
        label="Public diagnosis index",
    )
    diagnosis_lock = _regular_file(
        diagnosis_lock.resolve(strict=True),
        label="Public diagnosis lock",
    )
    known_failure_manifest = _regular_file(
        known_failure_manifest.resolve(strict=True),
        label="Known-failure manifest",
    )
    manifest_path = _regular_file(
        manifest_path.resolve(strict=True),
        label="Privacy-redaction manifest",
    )
    generator = _regular_file(
        (generator_path or Path(__file__)).resolve(strict=True),
        label="Privacy-redaction generator",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RedactionError("Privacy-redaction manifest is not valid UTF-8 JSON.") from error
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("pass") is not True:
        raise RedactionError("Privacy-redaction manifest did not pass.")
    contract = manifest.get("transformation_contract")
    expected_contract = {
        "artifact_hash_rebinding": "lowercase-sha256-same-length-substitution",
        "path_redaction": "deterministic-same-length-private-root-substitution",
        "payload_sizes_unchanged": True,
        "reversibility": "every-public-file-roundtrips-byte-for-byte",
    }
    if contract != expected_contract:
        raise RedactionError("Privacy-redaction transformation contract differs.")
    expected_generator = {
        **artifact_identity(generator),
        "logical_path": "publication/redact_b64_diagnosis.py",
    }
    if manifest.get("generator") != expected_generator:
        raise RedactionError("Privacy-redaction generator identity differs.")
    expected_known = {
        **artifact_identity(known_failure_manifest),
        "logical_path": "experiments/flashgs_matched/B64_KNOWN_FAILURE_CASES.json",
    }
    if manifest.get("known_failure_manifest") != expected_known:
        raise RedactionError("Privacy-redaction known-failure identity differs.")

    current: dict[str, Path] = {
        f"historical/{_safe_relative(path.relative_to(historical_root))}": path
        for path in _regular_tree_files(historical_root)
    }
    current.update(
        {
            "diagnosis-index.json": diagnosis_index,
            "diagnosis-lock.json": diagnosis_lock,
            "known-failure-manifest.json": known_failure_manifest,
        }
    )
    raw_records = manifest.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise RedactionError("Privacy-redaction manifest has no file records.")
    records: dict[str, dict[str, Any]] = {}
    path_total = 0
    hash_total = 0
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise RedactionError("Privacy-redaction file record is malformed.")
        label = raw.get("logical_path")
        if not isinstance(label, str) or label in records:
            raise RedactionError("Privacy-redaction file labels are invalid or duplicate.")
        if (
            set(raw)
            != {
                "artifact_hash_rebindings",
                "bytes",
                "logical_path",
                "path_prefix_replacements",
                "public_sha256",
                "reversible",
                "source_sha256",
            }
            or not isinstance(raw.get("bytes"), int)
            or raw["bytes"] < 0
            or not isinstance(raw.get("path_prefix_replacements"), int)
            or raw["path_prefix_replacements"] < 0
            or not isinstance(raw.get("artifact_hash_rebindings"), int)
            or raw["artifact_hash_rebindings"] < 0
            or raw.get("reversible") is not True
            or not isinstance(raw.get("source_sha256"), str)
            or SHA256_RE.fullmatch(raw["source_sha256"]) is None
            or not isinstance(raw.get("public_sha256"), str)
            or SHA256_RE.fullmatch(raw["public_sha256"]) is None
        ):
            raise RedactionError(f"Privacy-redaction file record differs: {label!r}.")
        records[label] = raw
        path_total += raw["path_prefix_replacements"]
        hash_total += raw["artifact_hash_rebindings"]
    if set(records) != set(current) or manifest.get("file_count") != len(current):
        raise RedactionError("Privacy-redaction public file set differs.")
    for label, path in current.items():
        record = records[label]
        identity = artifact_identity(path)
        if identity != {
            "bytes": record["bytes"],
            "sha256": record["public_sha256"],
        }:
            raise RedactionError(f"Privacy-redacted artifact changed: {label}.")
        if PRIVATE_ROOT_RE.search(path.read_bytes()):
            raise RedactionError(f"Private root remains in public derivative: {label}.")
    if (
        manifest.get("path_prefix_replacements") != path_total
        or manifest.get("artifact_hash_rebindings") != hash_total
        or path_total <= 0
    ):
        raise RedactionError("Privacy-redaction aggregate counts differ.")
    roots = manifest.get("private_roots")
    if not isinstance(roots, list) or not roots:
        raise RedactionError("Privacy-redaction root commitments are absent.")
    if sum(item.get("occurrences", 0) for item in roots if isinstance(item, dict)) != path_total:
        raise RedactionError("Privacy-redaction root occurrence count differs.")
    for item in roots:
        if (
            not isinstance(item, dict)
            or set(item) != {"bytes", "occurrences", "replacement", "source_sha256"}
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
            or not isinstance(item.get("occurrences"), int)
            or item["occurrences"] <= 0
            or not isinstance(item.get("replacement"), str)
            or not item["replacement"].startswith("/work/")
            or len(item["replacement"].encode("ascii")) != item["bytes"]
            or not isinstance(item.get("source_sha256"), str)
            or SHA256_RE.fullmatch(item["source_sha256"]) is None
        ):
            raise RedactionError("Privacy-redaction root commitment differs.")
    if PRIVATE_ROOT_RE.search(manifest_path.read_bytes()):
        raise RedactionError("Privacy-redaction manifest contains a private root.")
    return manifest


def redact_diagnosis(
    *,
    historical_root: Path,
    diagnosis_index: Path,
    diagnosis_lock: Path,
    known_failure_manifest: Path,
    output_root: Path,
    output_index: Path,
    output_lock: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    historical_root = historical_root.resolve(strict=True)
    diagnosis_index = diagnosis_index.resolve(strict=True)
    diagnosis_lock = diagnosis_lock.resolve(strict=True)
    known_failure_manifest = known_failure_manifest.resolve(strict=True)
    destinations = (output_root, output_index, output_lock, output_manifest)
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Every redaction output must be absent.")
    source_paths = {
        historical_root,
        diagnosis_index,
        diagnosis_lock,
        known_failure_manifest,
    }
    for destination in destinations:
        absolute = destination.absolute()
        if any(absolute == source or source in absolute.parents for source in source_paths):
            raise RedactionError("Redaction output overlaps a source input.")

    nodes = _build_nodes(
        historical_root=historical_root,
        diagnosis_index=diagnosis_index,
        diagnosis_lock=diagnosis_lock,
        known_failure_manifest=known_failure_manifest,
        output_root=output_root,
        output_index=output_index,
        output_lock=output_lock,
    )
    private_roots = _private_roots(nodes)
    transformed_by_original = _transform_nodes(nodes, private_roots)
    records: list[dict[str, Any]] = []
    total_path_replacements = 0
    total_hash_rebindings = 0
    for node in sorted(nodes, key=lambda item: item.label):
        transformed = transformed_by_original[node.original_sha256]
        _assert_reversible(
            node,
            transformed,
            private_roots=private_roots,
            transformed_by_original=transformed_by_original,
        )
        if node.output is not None:
            _write_new(
                node.output,
                transformed.payload,
                mode=stat.S_IMODE(node.source.stat().st_mode),
            )
            if artifact_identity(node.output) != {
                "bytes": len(transformed.payload),
                "sha256": transformed.public_sha256,
            }:
                raise RedactionError(f"Written output differs: {node.output}.")
        records.append(
            {
                "artifact_hash_rebindings": transformed.hash_rebindings,
                "bytes": len(node.original),
                "logical_path": node.label,
                "path_prefix_replacements": transformed.path_replacements,
                "public_sha256": transformed.public_sha256,
                "reversible": True,
                "source_sha256": node.original_sha256,
            }
        )
        total_path_replacements += transformed.path_replacements
        total_hash_rebindings += transformed.hash_rebindings

    manifest = {
        "artifact_hash_rebindings": total_hash_rebindings,
        "file_count": len(records),
        "files": records,
        "generator": {
            **artifact_identity(Path(__file__).resolve()),
            "logical_path": "publication/redact_b64_diagnosis.py",
        },
        "known_failure_manifest": {
            **artifact_identity(known_failure_manifest),
            "logical_path": "experiments/flashgs_matched/B64_KNOWN_FAILURE_CASES.json",
        },
        "pass": True,
        "path_prefix_replacements": total_path_replacements,
        "private_roots": [
            {
                "bytes": len(source),
                "occurrences": sum(node.original.count(source) for node in nodes),
                "replacement": target.decode("ascii"),
                "source_sha256": sha256_bytes(source),
            }
            for source, target in sorted(private_roots.items(), key=lambda item: item[0])
        ],
        "schema_version": SCHEMA_VERSION,
        "transformation_contract": {
            "artifact_hash_rebinding": "lowercase-sha256-same-length-substitution",
            "path_redaction": "deterministic-same-length-private-root-substitution",
            "payload_sizes_unchanged": True,
            "reversibility": "every-public-file-roundtrips-byte-for-byte",
        },
    }
    if total_path_replacements <= 0:
        raise RedactionError("Redaction manifest contains no path replacements.")
    _write_new(output_manifest, canonical_json_bytes(manifest))
    verify_public_derivative(
        historical_root=output_root,
        diagnosis_index=output_index,
        diagnosis_lock=output_lock,
        known_failure_manifest=known_failure_manifest,
        manifest_path=output_manifest,
    )
    return manifest


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--diagnosis-index", type=Path, required=True)
    parser.add_argument("--diagnosis-lock", type=Path, required=True)
    parser.add_argument("--known-failure-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    manifest = redact_diagnosis(
        historical_root=args.historical_root,
        diagnosis_index=args.diagnosis_index,
        diagnosis_lock=args.diagnosis_lock,
        known_failure_manifest=args.known_failure_manifest,
        output_root=args.output_root,
        output_index=args.output_index,
        output_lock=args.output_lock,
        output_manifest=args.output_manifest,
    )
    print(
        "B64_PRIVACY_REDACTION_OK "
        + json.dumps(
            {
                "file_count": manifest["file_count"],
                "manifest": str(args.output_manifest.resolve()),
                "path_prefix_replacements": manifest["path_prefix_replacements"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
