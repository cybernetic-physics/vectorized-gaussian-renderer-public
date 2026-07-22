#!/usr/bin/env python3
"""Derive and write the local input spec for an external release envelope.

The release spec is deliberately generated from the bytes that exist locally;
callers never supply artifact hashes or sizes.  The spec itself is a local
control-plane input and may contain absolute host paths.  Those paths are
verified again, then removed, by :mod:`release_envelope` before anything is
published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PUBLICATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PUBLICATION_ROOT.parent
for path in (str(PROJECT_ROOT), str(PUBLICATION_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from release_envelope import (  # noqa: E402
    SPEC_SCHEMA,
    ReleaseEnvelopeError,
    _validate_repository_spec,
)


ARTIFACT_ROLES = (
    "article",
    "bundle_archive",
    "bundle_manifest",
    "code_freeze",
    "github_public_verification",
    "r2_public_verification",
    "r2_receipt",
    "readme",
)
_CHUNK_BYTES = 1024 * 1024


class ReleaseSpecError(ValueError):
    """Raised when a release spec cannot be derived without ambiguity."""


@dataclass(frozen=True)
class ArtifactSnapshot:
    """A content identity plus the filesystem state used to derive it."""

    path: Path
    bytes: int
    sha256: str
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    @property
    def record(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes,
            "path": str(self.path),
            "sha256": self.sha256,
        }

    @property
    def file_id(self) -> tuple[int, int]:
        return (self.device, self.inode)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical encoding used by the release-envelope boundary."""

    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _absolute_path(value: str | Path, label: str) -> Path:
    raw = os.fspath(value)
    if not raw or any(ord(character) < 0x20 for character in raw):
        raise ReleaseSpecError(f"{label} contains an empty or control-character path.")
    return Path(os.path.abspath(raw))


def _assert_regular_no_symlink(value: str | Path, label: str) -> Path:
    absolute = _absolute_path(value, label)
    current = absolute
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise ReleaseSpecError(f"{label} is missing: {absolute}.") from None
        except OSError as error:
            raise ReleaseSpecError(f"Cannot inspect {label}: {error}.") from error
        if stat.S_ISLNK(mode):
            raise ReleaseSpecError(f"{label} traverses a symlink: {current}.")
        parent = current.parent
        if parent == current:
            break
        current = parent
    try:
        mode = absolute.lstat().st_mode
    except OSError as error:
        raise ReleaseSpecError(f"Cannot inspect {label}: {error}.") from error
    if not stat.S_ISREG(mode):
        raise ReleaseSpecError(f"{label} must be a regular file: {absolute}.")
    return absolute


def _assert_directory_no_symlink(value: str | Path, label: str) -> Path:
    absolute = _absolute_path(value, label)
    current = absolute
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise ReleaseSpecError(f"{label} is missing: {absolute}.") from None
        except OSError as error:
            raise ReleaseSpecError(f"Cannot inspect {label}: {error}.") from error
        if stat.S_ISLNK(mode):
            raise ReleaseSpecError(f"{label} traverses a symlink: {current}.")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not stat.S_ISDIR(absolute.lstat().st_mode):
        raise ReleaseSpecError(f"{label} must be a directory: {absolute}.")
    return absolute


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _identity_path(value: str | Path, label: str) -> ArtifactSnapshot:
    """Hash one stable regular file without following its final symlink."""

    absolute = _assert_regular_no_symlink(value, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ReleaseSpecError(f"Cannot open {label} without symlink traversal: {error}.") from error
    try:
        before = os.fstat(descriptor)
        path_before = os.lstat(absolute)
        if not stat.S_ISREG(before.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (path_before.st_dev, path_before.st_ino):
            raise ReleaseSpecError(f"{label} changed while it was opened.")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, _CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ReleaseSpecError(f"Cannot hash {label}: {error}.") from error
    finally:
        os.close(descriptor)

    try:
        path_after = os.lstat(absolute)
    except OSError as error:
        raise ReleaseSpecError(f"{label} changed while it was hashed: {error}.") from error
    _assert_regular_no_symlink(absolute, label)
    if (
        _stat_signature(before) != _stat_signature(after)
        or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
        or byte_count != after.st_size
    ):
        raise ReleaseSpecError(f"{label} changed while it was hashed.")
    if byte_count < 1:
        raise ReleaseSpecError(
            f"{label} must contain at least one byte for the release-envelope schema."
        )
    return ArtifactSnapshot(
        path=absolute,
        bytes=byte_count,
        sha256=digest.hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
        mode=after.st_mode,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


def _snapshot_artifacts(
    artifact_paths: Mapping[str, str | Path],
) -> dict[str, ArtifactSnapshot]:
    if not isinstance(artifact_paths, Mapping) or set(artifact_paths) != set(ARTIFACT_ROLES):
        actual = (
            sorted(str(key) for key in artifact_paths)
            if isinstance(artifact_paths, Mapping)
            else type(artifact_paths).__name__
        )
        raise ReleaseSpecError(
            "Artifact roles have missing or unknown fields; "
            f"expected={sorted(ARTIFACT_ROLES)}, actual={actual}."
        )
    snapshots = {
        role: _identity_path(artifact_paths[role], f"{role} artifact")
        for role in ARTIFACT_ROLES
    }
    by_path: dict[Path, str] = {}
    by_file_id: dict[tuple[int, int], str] = {}
    for role, snapshot in snapshots.items():
        if snapshot.path in by_path:
            raise ReleaseSpecError(
                f"Artifact roles {by_path[snapshot.path]} and {role} use the same path."
            )
        if snapshot.file_id in by_file_id:
            raise ReleaseSpecError(
                f"Artifact roles {by_file_id[snapshot.file_id]} and {role} alias one file."
            )
        by_path[snapshot.path] = role
        by_file_id[snapshot.file_id] = role
    return snapshots


def _revalidate_snapshots(snapshots: Mapping[str, ArtifactSnapshot]) -> None:
    for role in ARTIFACT_ROLES:
        expected = snapshots[role]
        observed = _identity_path(expected.path, f"{role} artifact")
        if observed != expected:
            raise ReleaseSpecError(
                f"{role} artifact changed after its release-spec identity was derived."
            )


def _repository_spec(
    *,
    owner: str,
    name: str,
    benchmark_commit: str,
    benchmark_tag: str,
    publication_content_commit: str,
    final_merge_commit: str,
    release_tag: str,
) -> dict[str, Any]:
    if not isinstance(release_tag, str):
        raise ReleaseSpecError("Release tag must be a string.")
    repository_url = f"https://github.com/{owner}/{name}"
    release_url = (
        f"{repository_url}/releases/tag/"
        f"{urllib.parse.quote(release_tag, safe='-._~')}"
    )
    repository = {
        "benchmark_commit": benchmark_commit,
        "benchmark_tag": benchmark_tag,
        "final_merge_commit": final_merge_commit,
        "name": name,
        "owner": owner,
        "publication_content_commit": publication_content_commit,
        "release_tag": release_tag,
        "release_url": release_url,
        "url": repository_url,
        "visibility": "public",
    }
    try:
        return _validate_repository_spec(repository)
    except ReleaseEnvelopeError as error:
        raise ReleaseSpecError(str(error)) from error


def _prepare_release_spec(
    *,
    artifact_paths: Mapping[str, str | Path],
    owner: str,
    name: str,
    benchmark_commit: str,
    benchmark_tag: str,
    publication_content_commit: str,
    final_merge_commit: str,
    release_tag: str,
) -> tuple[dict[str, Any], dict[str, ArtifactSnapshot]]:
    snapshots = _snapshot_artifacts(artifact_paths)
    repository = _repository_spec(
        owner=owner,
        name=name,
        benchmark_commit=benchmark_commit,
        benchmark_tag=benchmark_tag,
        publication_content_commit=publication_content_commit,
        final_merge_commit=final_merge_commit,
        release_tag=release_tag,
    )
    spec = {
        "artifacts": {
            role: snapshots[role].record for role in ARTIFACT_ROLES
        },
        "repository": repository,
        "schema_version": SPEC_SCHEMA,
    }
    return spec, snapshots


def build_release_spec(
    *,
    artifact_paths: Mapping[str, str | Path],
    owner: str,
    name: str,
    benchmark_commit: str,
    benchmark_tag: str,
    publication_content_commit: str,
    final_merge_commit: str,
    release_tag: str,
) -> dict[str, Any]:
    """Build a canonicalizable spec after a closing local-file reread."""

    spec, snapshots = _prepare_release_spec(
        artifact_paths=artifact_paths,
        owner=owner,
        name=name,
        benchmark_commit=benchmark_commit,
        benchmark_tag=benchmark_tag,
        publication_content_commit=publication_content_commit,
        final_merge_commit=final_merge_commit,
        release_tag=release_tag,
    )
    _revalidate_snapshots(snapshots)
    return spec


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    expected_size: int,
    forbidden_file_ids: set[tuple[int, int]],
) -> bytes:
    absolute = _assert_regular_no_symlink(path, label)
    metadata = absolute.lstat()
    if (metadata.st_dev, metadata.st_ino) in forbidden_file_ids:
        raise ReleaseSpecError(f"{label} aliases a release artifact.")
    if metadata.st_size != expected_size:
        return b""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
        try:
            before = os.fstat(descriptor)
            payload = bytearray()
            while len(payload) <= expected_size:
                chunk = os.read(descriptor, min(_CHUNK_BYTES, expected_size + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ReleaseSpecError(f"Cannot read {label}: {error}.") from error
    path_after = absolute.lstat()
    _assert_regular_no_symlink(absolute, label)
    if (
        _stat_signature(before) != _stat_signature(after)
        or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
    ):
        raise ReleaseSpecError(f"{label} changed while it was read.")
    return bytes(payload)


def _write_once(
    path: str | Path,
    payload: bytes,
    *,
    forbidden_file_ids: set[tuple[int, int]],
    before_publish: Callable[[], None],
) -> Path:
    output = _absolute_path(path, "Release-spec output")
    parent = _assert_directory_no_symlink(output.parent, "Release-spec output parent")

    def accept_existing(label: str) -> None:
        if output.is_symlink():
            raise ReleaseSpecError(f"{label} may not be a symlink.")
        observed = _read_regular_bytes(
            output,
            label=label,
            expected_size=len(payload),
            forbidden_file_ids=forbidden_file_ids,
        )
        if observed != payload:
            raise ReleaseSpecError(
                "Existing release spec differs; refusing to overwrite it."
            )

    if output.exists() or output.is_symlink():
        before_publish()
        accept_existing("Existing release-spec output")
        return output

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.tmp-",
            suffix=".json",
            dir=parent,
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        before_publish()
        try:
            os.link(temporary, output)
        except FileExistsError:
            accept_existing("Concurrent release-spec output")
    except OSError as error:
        raise ReleaseSpecError(f"Cannot write release spec: {error}.") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    accept_existing("Written release-spec output")
    return output


def write_release_spec(
    *,
    artifact_paths: Mapping[str, str | Path],
    owner: str,
    name: str,
    benchmark_commit: str,
    benchmark_tag: str,
    publication_content_commit: str,
    final_merge_commit: str,
    release_tag: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Derive, recheck, and atomically write one immutable release spec."""

    spec, snapshots = _prepare_release_spec(
        artifact_paths=artifact_paths,
        owner=owner,
        name=name,
        benchmark_commit=benchmark_commit,
        benchmark_tag=benchmark_tag,
        publication_content_commit=publication_content_commit,
        final_merge_commit=final_merge_commit,
        release_tag=release_tag,
    )
    output = _absolute_path(output_path, "Release-spec output")
    if output in {snapshot.path for snapshot in snapshots.values()}:
        raise ReleaseSpecError("Release-spec output may not replace a release artifact.")
    payload = canonical_json_bytes(spec)
    _write_once(
        output,
        payload,
        forbidden_file_ids={snapshot.file_id for snapshot in snapshots.values()},
        before_publish=lambda: _revalidate_snapshots(snapshots),
    )
    return spec


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a deterministic release-envelope spec from exact local artifact bytes."
        )
    )
    for role in ARTIFACT_ROLES:
        parser.add_argument(f"--{role.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--benchmark-tag", required=True)
    parser.add_argument("--publication-content-commit", required=True)
    parser.add_argument("--final-merge-commit", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    artifact_paths = {
        role: getattr(args, role) for role in ARTIFACT_ROLES
    }
    spec = write_release_spec(
        artifact_paths=artifact_paths,
        owner=args.owner,
        name=args.name,
        benchmark_commit=args.benchmark_commit,
        benchmark_tag=args.benchmark_tag,
        publication_content_commit=args.publication_content_commit,
        final_merge_commit=args.final_merge_commit,
        release_tag=args.release_tag,
        output_path=args.output,
    )
    payload = canonical_json_bytes(spec)
    print(
        "PUBLICATION_RELEASE_SPEC_WRITTEN "
        + json.dumps(
            {
                "bytes": len(payload),
                "pass": True,
                "schema_version": spec["schema_version"],
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
