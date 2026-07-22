#!/usr/bin/env python3
"""Fail-closed privacy checks for files intended to become public.

Secret detection is deliberately delegated to the evidence bundle's canonical
scanner.  Publication scanning enables its assignment detector for every file,
including binary profiler/database containers whose string tables can retain a
captured process environment.  This module also adds checks that the bundle
scanner does not own: host-user paths, credential-bearing URLs, and (when
requested) timestamps in deterministic JSON envelopes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, Iterator


PUBLICATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PUBLICATION_ROOT.parent
EVALUATION_ROOT = (
    PROJECT_ROOT / "src" / "isaacsim_gaussian_renderer" / "evaluation"
)
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

# Reuse the bundle's maintained secret patterns.  Keeping this import explicit
# prevents a second, inevitably divergent token/private-key detector here.
from evidence_bundle import _is_textual, _scan_secret_bytes  # noqa: E402


_HOST_USER_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"file://(?:localhost)?/(?:Users|home)/[^/\s]+/|"
    r"/(?:Users|home)/[^/\s]+/|"
    r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s]+[\\/]"
    r")"
)
_HOST_USER_PATH_BYTES_RE = re.compile(
    rb"(?<![A-Za-z0-9])(?:"
    rb"file://(?:localhost)?/(?:Users|home)/[^/\s\x00]+/|"
    rb"/(?:Users|home)/[^/\s\x00]+/|"
    rb"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s\x00]+[\\/]"
    rb")"
)
_CAPTURED_ENV_SECRET_RE = re.compile(
    rb"(?i)(?:^|[\x00\r\n])(?:[a-z0-9]+[_-])*"
    rb"(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    rb"password|secret[_-]?key|token)\s*[\"']?\s*[:=]\s*"
    rb"[\"']?[A-Za-z0-9_./+=:@-]{12,}"
)
_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:?\d{2})$"
)
_TIMESTAMP_KEYS = frozenset(
    {
        "created_at",
        "generated_at",
        "published_at",
        "timestamp",
        "updated_at",
        "verified_at",
    }
)
_SENSITIVE_QUERY_NAMES = frozenset(
    {
        "access_key",
        "api_key",
        "authorization",
        "credential",
        "password",
        "secret",
        "signature",
        "token",
        "x-amz-credential",
        "x-amz-security-token",
        "x-amz-signature",
    }
)


class PublicArtifactPrivacyError(ValueError):
    """Raised when a proposed public artifact contains private material."""


def _walk_strings(value: Any, pointer: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key in sorted(value):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _walk_strings(value[key], f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{pointer}/{index}")
    elif isinstance(value, str):
        yield pointer, value


def _assert_no_local_paths(value: Any) -> None:
    for pointer, text in _walk_strings(value):
        if _HOST_USER_PATH_RE.search(text):
            raise PublicArtifactPrivacyError(
                f"Host-user path detected at {pointer or '/'}; refusing public artifact."
            )


def _assert_no_embedded_local_paths(path: Path) -> None:
    """Reject printable host-user paths retained inside binary containers."""

    overlap = b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            window = overlap + block
            if _HOST_USER_PATH_BYTES_RE.search(window):
                raise PublicArtifactPrivacyError(
                    "Host-user path detected in artifact bytes; refusing public artifact."
                )
            overlap = window[-512:]


def _assert_no_captured_environment_secrets(path: Path) -> None:
    """Reject nonempty secret-shaped assignments in binary string tables."""

    overlap = b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            window = overlap + block
            if _CAPTURED_ENV_SECRET_RE.search(window):
                raise PublicArtifactPrivacyError(
                    "Captured secret assignment detected; refusing public artifact."
                )
            overlap = window[-512:]


def _assert_no_credential_urls(value: Any) -> None:
    for pointer, text in _walk_strings(value):
        if not text.startswith(("http://", "https://")):
            continue
        parsed = urllib.parse.urlsplit(text)
        if parsed.username is not None or parsed.password is not None:
            raise PublicArtifactPrivacyError(
                f"Credential-bearing URL detected at {pointer or '/'}."
            )
        query_names = {
            name.lower() for name, _ in urllib.parse.parse_qsl(parsed.query)
        }
        if query_names & _SENSITIVE_QUERY_NAMES:
            raise PublicArtifactPrivacyError(
                f"Sensitive URL query detected at {pointer or '/'}; refusing public artifact."
            )


def _assert_no_timestamps(value: Any) -> None:
    def visit(item: Any, pointer: str = "") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                child_pointer = f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}"
                if str(key).lower() in _TIMESTAMP_KEYS:
                    raise PublicArtifactPrivacyError(
                        f"Timestamp field detected at {child_pointer}; deterministic envelope forbids it."
                    )
                visit(child, child_pointer)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{pointer}/{index}")
        elif isinstance(item, str) and _ISO_TIMESTAMP_RE.fullmatch(item):
            raise PublicArtifactPrivacyError(
                f"Timestamp value detected at {pointer or '/'}; deterministic envelope forbids it."
            )

    visit(value)


def _assert_regular_no_symlink(path: Path) -> Path:
    absolute = path.absolute()
    current = absolute
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing public artifact: {absolute}.") from None
        if stat.S_ISLNK(mode):
            raise PublicArtifactPrivacyError(
                f"Symlinks are forbidden for public artifacts: {current}."
            )
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not stat.S_ISREG(absolute.lstat().st_mode):
        raise PublicArtifactPrivacyError("Public artifact must be a regular file.")
    return absolute


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return {"bytes": byte_count, "sha256": digest.hexdigest()}


def scan_public_json_value(
    value: Any,
    *,
    reject_timestamps: bool = False,
) -> None:
    """Apply publication-only structural privacy checks to a JSON value."""

    _assert_no_local_paths(value)
    _assert_no_credential_urls(value)
    if reject_timestamps:
        _assert_no_timestamps(value)


def scan_public_artifact(
    path: str | Path,
    *,
    reject_timestamps: bool = False,
) -> dict[str, Any]:
    """Scan one regular file and return only its public-safe identity."""

    candidate = _assert_regular_no_symlink(Path(path))
    textual = _is_textual(candidate.name)
    # Nsight reports and SQLite exports can embed printable environment
    # assignments despite having binary suffixes.  The canonical scanner's
    # ``textual`` flag controls only its assigned-secret regex, not decoding,
    # so it must be enabled at the public boundary for every media type.
    _scan_secret_bytes(candidate, textual=True)
    _assert_no_captured_environment_secrets(candidate)
    _assert_no_embedded_local_paths(candidate)
    if textual:
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise PublicArtifactPrivacyError(
                "Textual public artifact is not valid UTF-8."
            ) from error
        parsed: Any
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        scan_public_json_value(parsed, reject_timestamps=reject_timestamps)
    return file_identity(candidate)


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a proposed public artifact for secrets and private paths."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--reject-timestamps", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    identity = scan_public_artifact(
        args.artifact,
        reject_timestamps=args.reject_timestamps,
    )
    print("PUBLIC_ARTIFACT_PRIVACY_VERIFIED " + json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
