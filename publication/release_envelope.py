#!/usr/bin/env python3
"""Build a deterministic, offline-verified external publication envelope.

The scientific evidence bundle is immutable before this tool runs.  This
envelope binds that bundle to the later GitHub merge/release and R2
publication without placing mutable network state, host paths, credentials,
or timestamps inside the scientific bundle.

Network access is intentionally absent.  Anonymous GitHub and R2 checks must
be captured as separately hashed, strict JSON records and supplied in the
input spec.  The tool verifies those records against local Git, the bundle,
the archive, the R2 publisher receipt, and the final article bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PUBLICATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PUBLICATION_ROOT.parent
EVALUATION_ROOT = (
    PROJECT_ROOT / "src" / "isaacsim_gaussian_renderer" / "evaluation"
)
for path in (str(PROJECT_ROOT), str(PUBLICATION_ROOT), str(EVALUATION_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evidence_bundle import resolve_artifact_record  # noqa: E402
from publish_immutable_r2 import (  # noqa: E402
    CACHE_CONTROL,
    PART_BYTES,
    RECEIPT_SCHEMA,
    PublicationError,
    _validate_key,
)
from scan_public_artifact import (  # noqa: E402
    PublicArtifactPrivacyError,
    scan_public_artifact,
    scan_public_json_value,
)
from verify_release import verify_publication_release  # noqa: E402
from scripts.verify_publication_code_freeze import (  # noqa: E402
    ALLOWED_PUBLICATION_PATHS,
    ALLOWED_PUBLICATION_PREFIXES,
    LOAD_BEARING_PATHS,
    publication_path_allowed,
)


SPEC_SCHEMA = "publication-external-release-envelope-spec-v1"
ENVELOPE_SCHEMA = "publication-external-release-envelope-v1"
R2_PUBLIC_VERIFICATION_SCHEMA = "publication-r2-public-verification-v1"
GITHUB_PUBLIC_VERIFICATION_SCHEMA = "publication-github-public-verification-v1"
CODE_FREEZE_SCHEMA = "publication-code-freeze-v1"
BUNDLE_SCHEMA = "flashgs-matched-evidence-bundle-v1"
BUNDLED_ARTICLE_LOGICAL_PATH = "publication/article.md"
RELEASE_VERIFICATION_SCHEMA = "publication-release-verification-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SAFE_REPOSITORY_PART_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?")


class ReleaseEnvelopeError(ValueError):
    """Raised when any external-release binding is incomplete or inconsistent."""


@dataclass(frozen=True)
class ResolvedArtifact:
    role: str
    path: Path
    bytes: int
    sha256: str

    @property
    def identity(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "sha256": self.sha256}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ReleaseEnvelopeError(
            f"{label} has missing or unknown fields; expected={sorted(keys)}, actual={actual}."
        )
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseEnvelopeError(f"{label} must be a nonempty string.")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReleaseEnvelopeError(f"{label} must be one lowercase SHA-256.")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise ReleaseEnvelopeError(f"{label} must be one full lowercase Git commit.")
    return value


def _require_tag(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_TAG_RE.fullmatch(value) is None:
        raise ReleaseEnvelopeError(f"{label} must be a simple immutable release tag.")
    return value


def _require_bytes(value: Any, label: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReleaseEnvelopeError(f"{label} must be an integer byte count.")
    minimum = 1 if positive else 0
    if value < minimum:
        raise ReleaseEnvelopeError(f"{label} must be at least {minimum}.")
    return value


def _require_https_url(value: Any, label: str, *, allow_query: bool = False) -> str:
    url = _require_string(value, label)
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (parsed.query and not allow_query)
    ):
        raise ReleaseEnvelopeError(f"{label} must be a credential-free HTTPS URL.")
    return url


def _resolve_artifact(role: str, raw: Any) -> ResolvedArtifact:
    record = _require_exact_keys(raw, {"bytes", "path", "sha256"}, f"{role} artifact")
    path_text = _require_string(record["path"], f"{role} artifact path")
    byte_count = _require_bytes(record["bytes"], f"{role} artifact bytes")
    sha256 = _require_sha256(record["sha256"], f"{role} artifact SHA-256")
    try:
        resolved = resolve_artifact_record(
            {"bytes": byte_count, "path": path_text, "sha256": sha256},
            label=f"{role} artifact",
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ReleaseEnvelopeError(str(error)) from error
    return ResolvedArtifact(
        role=role,
        path=Path(resolved["path"]),
        bytes=byte_count,
        sha256=sha256,
    )


def _assert_regular_no_symlink_path(path: Path, label: str) -> Path:
    absolute = path.absolute()
    current = absolute
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise ReleaseEnvelopeError(f"{label} is missing: {absolute}.") from None
        if stat.S_ISLNK(mode):
            raise ReleaseEnvelopeError(f"{label} traverses a symlink: {current}.")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not stat.S_ISREG(absolute.lstat().st_mode):
        raise ReleaseEnvelopeError(f"{label} must be a regular file.")
    return absolute


def _load_canonical_json(artifact: ResolvedArtifact, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseEnvelopeError(f"{label} is not valid UTF-8 JSON.") from error
    if not isinstance(payload, dict):
        raise ReleaseEnvelopeError(f"{label} must be a JSON object.")
    if artifact.path.read_bytes() != canonical_json_bytes(payload):
        raise ReleaseEnvelopeError(f"{label} is not canonically encoded JSON.")
    try:
        observed = scan_public_artifact(artifact.path)
    except (FileNotFoundError, OSError, PublicArtifactPrivacyError, ValueError) as error:
        raise ReleaseEnvelopeError(f"{label} privacy scan failed: {error}") from error
    if observed != artifact.identity:
        raise ReleaseEnvelopeError(f"{label} changed during privacy scanning.")
    return payload


def _git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseEnvelopeError(
            f"Git command failed ({' '.join(arguments)}): {message or completed.returncode}."
        )
    return completed


def _git_text(repo: Path, *arguments: str) -> str:
    return _git(repo, *arguments).stdout.decode("utf-8").strip()


def _assert_repo_root(repo_root: str | Path) -> Path:
    raw = Path(repo_root).absolute()
    current = raw
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise ReleaseEnvelopeError(f"Repository root is missing: {raw}.") from None
        if stat.S_ISLNK(mode):
            raise ReleaseEnvelopeError(f"Repository root traverses a symlink: {current}.")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not raw.is_dir():
        raise ReleaseEnvelopeError("Repository root must be a directory.")
    observed = Path(_git_text(raw, "rev-parse", "--show-toplevel")).resolve()
    if observed != raw.resolve():
        raise ReleaseEnvelopeError("--repo-root is not the Git worktree root.")
    return raw.resolve()


def _normalize_github_origin(value: str) -> str:
    if value.startswith("git@github.com:"):
        path = value[len("git@github.com:") :]
    else:
        parsed = urllib.parse.urlsplit(value)
        if (
            parsed.scheme not in {"https", "ssh"}
            or parsed.hostname != "github.com"
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (parsed.scheme == "https" and parsed.username is not None)
            or (parsed.scheme == "ssh" and parsed.username not in {None, "git"})
        ):
            raise ReleaseEnvelopeError("Git origin is not a GitHub repository URL.")
        path = parsed.path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        raise ReleaseEnvelopeError("Git origin does not identify one GitHub owner/repository.")
    return f"https://github.com/{parts[0]}/{parts[1]}"


def _resolve_tag(repo: Path, tag: str) -> str:
    return _git_text(repo, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")


def _assert_ancestor(repo: Path, ancestor: str, descendant: str, label: str) -> None:
    completed = _git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if completed.returncode != 0:
        raise ReleaseEnvelopeError(f"Git ancestry check failed: {label}.")


def _git_blob(repo: Path, commit: str, git_path: str) -> bytes:
    if PurePosixPath(git_path).is_absolute() or ".." in PurePosixPath(git_path).parts:
        raise ReleaseEnvelopeError("Git content path is unsafe.")
    return _git(repo, "show", f"{commit}:{git_path}").stdout


def _identity_bytes(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _tree_fingerprint(repo: Path, commit: str) -> dict[str, Any]:
    raw = _git_text(repo, "ls-tree", "-r", commit, "--", *LOAD_BEARING_PATHS)
    records = sorted(line for line in raw.splitlines() if line)
    encoded = "\n".join(records).encode("utf-8")
    return {
        "record_count": len(records),
        "records": records,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _code_freeze_changes(repo: Path, benchmark: str, publication: str) -> list[dict[str, Any]]:
    raw = _git_text(
        repo,
        "diff",
        "--name-status",
        "--find-renames",
        benchmark,
        publication,
    )
    changes: list[dict[str, Any]] = []
    for line in raw.splitlines():
        fields = line.split("\t")
        status = fields[0]
        paths = fields[1:]
        allowed = bool(paths) and all(publication_path_allowed(path) for path in paths)
        changes.append({"allowed": allowed, "paths": paths, "status": status})
    return changes


def _validate_tree_record(value: Any, label: str) -> dict[str, Any]:
    record = _require_exact_keys(value, {"record_count", "records", "sha256"}, label)
    count = record["record_count"]
    records = record["records"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or not isinstance(records, list)
        or any(not isinstance(item, str) or not item for item in records)
        or records != sorted(records)
        or len(records) != count
    ):
        raise ReleaseEnvelopeError(f"{label} is malformed.")
    _require_sha256(record["sha256"], f"{label} SHA-256")
    observed = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    if observed != record["sha256"]:
        raise ReleaseEnvelopeError(f"{label} digest differs from its records.")
    return record


def _validate_code_freeze(
    payload: Any,
    *,
    repo: Path,
    benchmark_commit: str,
    benchmark_tag: str,
    publication_commit: str,
) -> dict[str, Any]:
    keys = {
        "allowed_publication_paths",
        "allowed_publication_prefixes",
        "benchmark_commit",
        "benchmark_load_bearing_tree",
        "benchmark_ref",
        "benchmark_tags",
        "changes",
        "created_at",
        "load_bearing_tree_equal",
        "pass",
        "publication_commit",
        "publication_load_bearing_tree",
        "publication_ref",
        "rejected_paths",
        "schema_version",
        "tag_required",
    }
    record = _require_exact_keys(payload, keys, "code-freeze record")
    if record["schema_version"] != CODE_FREEZE_SCHEMA or record["pass"] is not True:
        raise ReleaseEnvelopeError("Code-freeze record is not a passing v1 record.")
    if record["benchmark_commit"] != benchmark_commit:
        raise ReleaseEnvelopeError("Code-freeze benchmark commit differs.")
    if record["publication_commit"] != publication_commit:
        raise ReleaseEnvelopeError("Code-freeze publication commit differs.")
    if record["benchmark_ref"] != benchmark_tag:
        raise ReleaseEnvelopeError("Code-freeze benchmark ref must be the frozen benchmark tag.")
    if record["publication_ref"] != publication_commit:
        raise ReleaseEnvelopeError(
            "Code-freeze publication ref must be the full immutable content commit."
        )
    try:
        created_at = datetime.fromisoformat(
            _require_string(record["created_at"], "code-freeze created_at")
        )
    except ValueError as error:
        raise ReleaseEnvelopeError("Code-freeze created_at is not ISO-8601.") from error
    if created_at.tzinfo is None:
        raise ReleaseEnvelopeError("Code-freeze created_at must include a timezone.")
    if record["allowed_publication_paths"] != sorted(ALLOWED_PUBLICATION_PATHS):
        raise ReleaseEnvelopeError("Code-freeze allowed publication paths differ.")
    if record["allowed_publication_prefixes"] != list(ALLOWED_PUBLICATION_PREFIXES):
        raise ReleaseEnvelopeError("Code-freeze allowed publication prefixes differ.")
    if record["tag_required"] is not True:
        raise ReleaseEnvelopeError("Publication code freeze must require a benchmark tag.")
    tags = sorted(line for line in _git_text(repo, "tag", "--points-at", benchmark_commit).splitlines() if line)
    if record["benchmark_tags"] != tags or benchmark_tag not in tags:
        raise ReleaseEnvelopeError("Code-freeze benchmark tags differ from Git.")
    expected_changes = _code_freeze_changes(repo, benchmark_commit, publication_commit)
    if record["changes"] != expected_changes:
        raise ReleaseEnvelopeError("Code-freeze change inventory differs from Git.")
    rejected = sorted(
        {path for change in expected_changes if not change["allowed"] for path in change["paths"]}
    )
    if record["rejected_paths"] != rejected or rejected:
        raise ReleaseEnvelopeError("Code-freeze contains rejected load-bearing paths.")
    benchmark_tree = _validate_tree_record(
        record["benchmark_load_bearing_tree"], "benchmark load-bearing tree"
    )
    publication_tree = _validate_tree_record(
        record["publication_load_bearing_tree"], "publication load-bearing tree"
    )
    if benchmark_tree != _tree_fingerprint(repo, benchmark_commit):
        raise ReleaseEnvelopeError("Code-freeze benchmark tree differs from Git.")
    if publication_tree != _tree_fingerprint(repo, publication_commit):
        raise ReleaseEnvelopeError("Code-freeze publication tree differs from Git.")
    if record["load_bearing_tree_equal"] is not True or benchmark_tree != publication_tree:
        raise ReleaseEnvelopeError("Code-freeze load-bearing trees are not identical.")
    return {
        "benchmark_load_bearing_tree_sha256": benchmark_tree["sha256"],
        "publication_load_bearing_tree_sha256": publication_tree["sha256"],
    }


def _validate_bundle(
    *,
    manifest_artifact: ResolvedArtifact,
    archive_artifact: ResolvedArtifact,
    code_freeze_artifact: ResolvedArtifact,
    article_artifact: ResolvedArtifact,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if manifest_artifact.path.name != "manifest.json":
        raise ReleaseEnvelopeError("Bundle manifest input must be named manifest.json.")
    bundle_root = manifest_artifact.path.parent
    try:
        manifest, release_verification = verify_publication_release(
            bundle_root=bundle_root,
            archive_path=archive_artifact.path,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise ReleaseEnvelopeError(
            f"Strict publication release verification failed: {error}"
        ) from error
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ReleaseEnvelopeError("Scientific bundle schema differs.")
    release_verification = _require_exact_keys(
        release_verification,
        {
            "aggregate",
            "archive",
            "claim_ledger",
            "claim_validator",
            "manifest_id",
            "pass",
            "schema_version",
        },
        "strict release-verification receipt",
    )
    if (
        release_verification["schema_version"] != RELEASE_VERIFICATION_SCHEMA
        or release_verification["pass"] is not True
        or release_verification["manifest_id"] != manifest.get("manifest_id")
    ):
        raise ReleaseEnvelopeError(
            "Strict release-verification receipt does not bind the verified manifest."
        )
    aggregate = release_verification["aggregate"]
    claim_ledger = release_verification["claim_ledger"]
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("schema_version")
        != "publication-aggregate-verification-receipt-v1"
        or aggregate.get("pass") is not True
        or aggregate.get("root_mode") != "bundle"
    ):
        raise ReleaseEnvelopeError("Strict aggregate verification receipt differs.")
    if (
        not isinstance(claim_ledger, dict)
        or claim_ledger.get("schema_version")
        != "publication-claim-ledger-verification-v1"
        or claim_ledger.get("pass") is not True
        or claim_ledger.get("root_mode") != "bundle"
    ):
        raise ReleaseEnvelopeError("Strict claim-ledger verification receipt differs.")
    for role in ("verification", "summary", "ablation"):
        if claim_ledger.get(role) != aggregate.get(role):
            raise ReleaseEnvelopeError(
                f"Strict claim ledger and aggregate bind different {role} artifacts."
            )
    claim_validator = _require_exact_keys(
        release_verification["claim_validator"],
        {"bytes", "logical_path", "sha256"},
        "strict claim validator",
    )
    if claim_validator["logical_path"] != "publication/validator/verify_claim_ledger.py":
        raise ReleaseEnvelopeError("Strict claim validator logical path differs.")
    _require_bytes(claim_validator["bytes"], "strict claim validator bytes")
    _require_sha256(claim_validator["sha256"], "strict claim validator SHA-256")
    archive = release_verification["archive"]
    if not isinstance(archive, dict):
        raise ReleaseEnvelopeError("Strict deterministic-archive receipt is malformed.")
    if _identity_bytes(manifest_artifact.path.read_bytes()) != manifest_artifact.identity:
        raise ReleaseEnvelopeError("Verified bundle manifest identity differs.")
    if archive.get("bytes") != archive_artifact.bytes or archive.get("sha256") != archive_artifact.sha256:
        raise ReleaseEnvelopeError("Verified deterministic archive identity differs.")
    if archive.get("manifest_id") != manifest.get("manifest_id"):
        raise ReleaseEnvelopeError("Archive and bundle manifest IDs differ.")
    article_matches = [
        item
        for item in manifest.get("publication_artifacts", [])
        if isinstance(item, dict)
        and item.get("logical_path") == BUNDLED_ARTICLE_LOGICAL_PATH
    ]
    if len(article_matches) != 1:
        raise ReleaseEnvelopeError(
            "Bundle must inventory exactly one publication article."
        )
    bundled_article = article_matches[0]
    if {
        "bytes": bundled_article.get("bytes"),
        "sha256": bundled_article.get("sha256"),
    } != article_artifact.identity:
        raise ReleaseEnvelopeError(
            "Bundled publication article differs from supplied post.md."
        )
    article_object_path = bundled_article.get("object_path")
    if not isinstance(article_object_path, str):
        raise ReleaseEnvelopeError("Bundled publication article object path is malformed.")
    try:
        resolved_article = resolve_artifact_record(
            {
                "bytes": article_artifact.bytes,
                "path": "ignored",
                "sha256": article_artifact.sha256,
            },
            bundle_root=bundle_root,
            label="bundled publication article",
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ReleaseEnvelopeError(str(error)) from error
    if Path(resolved_article["path"]).read_bytes() != article_artifact.path.read_bytes():
        raise ReleaseEnvelopeError(
            "Bundled publication article bytes differ from supplied post.md."
        )
    matches = [
        item
        for item in manifest.get("publication_artifacts", [])
        if isinstance(item, dict)
        and item.get("logical_path") == "publication/code-freeze.json"
        and "code-freeze" in item.get("roles", [])
    ]
    if len(matches) != 1:
        raise ReleaseEnvelopeError("Bundle must inventory exactly one code-freeze record.")
    bundled = matches[0]
    if {
        "bytes": bundled.get("bytes"),
        "sha256": bundled.get("sha256"),
    } != code_freeze_artifact.identity:
        raise ReleaseEnvelopeError("Supplied code-freeze record differs from the bundled record.")
    object_path = bundled.get("object_path")
    if not isinstance(object_path, str):
        raise ReleaseEnvelopeError("Bundled code-freeze object path is malformed.")
    try:
        resolved = resolve_artifact_record(
            {
                "bytes": code_freeze_artifact.bytes,
                "path": "ignored",
                "sha256": code_freeze_artifact.sha256,
            },
            bundle_root=bundle_root,
            label="bundled code-freeze record",
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ReleaseEnvelopeError(str(error)) from error
    if Path(resolved["path"]).read_bytes() != code_freeze_artifact.path.read_bytes():
        raise ReleaseEnvelopeError("Bundled and supplied code-freeze bytes differ.")
    strict_proof = {
        "aggregate": aggregate,
        "claim_ledger": claim_ledger,
        "claim_validator": claim_validator,
        "manifest_id": release_verification["manifest_id"],
        "pass": True,
        "schema_version": RELEASE_VERIFICATION_SCHEMA,
    }
    scan_public_json_value(strict_proof, reject_timestamps=True)
    return manifest, archive, strict_proof


def _validate_r2_object(value: Any, *, base_url: str) -> dict[str, Any]:
    keys = {
        "bytes",
        "cache_control",
        "content_disposition",
        "content_type",
        "key",
        "role",
        "sha256",
        "url",
    }
    item = _require_exact_keys(value, keys, "R2 receipt object")
    key = _require_string(item["key"], "R2 receipt object key")
    try:
        _validate_key(key)
    except PublicationError as error:
        raise ReleaseEnvelopeError(str(error)) from error
    byte_count = _require_bytes(item["bytes"], "R2 receipt object bytes")
    sha256 = _require_sha256(item["sha256"], "R2 receipt object SHA-256")
    role = _require_string(item["role"], "R2 receipt object role")
    if role not in {"artifact", "index", "part"}:
        raise ReleaseEnvelopeError("R2 receipt object has an unknown role.")
    if item["cache_control"] != CACHE_CONTROL:
        raise ReleaseEnvelopeError("R2 receipt object does not use immutable cache control.")
    for field in ("content_disposition", "content_type"):
        text = _require_string(item[field], f"R2 receipt object {field}")
        if any(ord(character) < 32 for character in text):
            raise ReleaseEnvelopeError(f"R2 receipt object {field} contains control bytes.")
    expected_url = f"{base_url}/{urllib.parse.quote(key, safe='/-._~')}"
    if item["url"] != expected_url:
        raise ReleaseEnvelopeError("R2 receipt object URL does not match its key.")
    _require_https_url(item["url"], "R2 receipt object URL")
    return {
        "bytes": byte_count,
        "cache_control": item["cache_control"],
        "content_disposition": item["content_disposition"],
        "content_type": item["content_type"],
        "key": key,
        "role": role,
        "sha256": sha256,
        "url": item["url"],
    }


def _validate_r2_receipt(
    payload: Any,
    *,
    archive: ResolvedArtifact,
) -> dict[str, Any]:
    root_keys = {
        "artifact",
        "bucket",
        "cors_origin",
        "objects",
        "public_base_url",
        "publisher",
        "schema_version",
        "storage",
        "verification",
    }
    receipt = _require_exact_keys(payload, root_keys, "R2 receipt")
    if receipt["schema_version"] != RECEIPT_SCHEMA:
        raise ReleaseEnvelopeError("R2 receipt schema differs.")
    base_url = _require_https_url(receipt["public_base_url"], "R2 public base URL").rstrip("/")
    if receipt["public_base_url"] != base_url:
        raise ReleaseEnvelopeError("R2 public base URL must not have a trailing slash.")
    _require_https_url(receipt["cors_origin"], "R2 CORS origin")
    if urllib.parse.urlsplit(receipt["cors_origin"]).path not in {"", "/"}:
        raise ReleaseEnvelopeError("R2 CORS origin must not contain a path.")
    bucket = _require_string(receipt["bucket"], "R2 bucket")
    if re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,62}", bucket) is None:
        raise ReleaseEnvelopeError("R2 bucket name is malformed.")
    artifact = _require_exact_keys(
        receipt["artifact"],
        {"bytes", "content_disposition", "content_type", "key", "sha256"},
        "R2 receipt artifact",
    )
    artifact_key = _require_string(artifact["key"], "R2 logical artifact key")
    try:
        _validate_key(artifact_key)
    except PublicationError as error:
        raise ReleaseEnvelopeError(str(error)) from error
    if (
        _require_bytes(artifact["bytes"], "R2 logical artifact bytes") != archive.bytes
        or _require_sha256(artifact["sha256"], "R2 logical artifact SHA-256") != archive.sha256
    ):
        raise ReleaseEnvelopeError("R2 receipt does not bind the deterministic archive.")
    if archive.sha256 not in artifact_key:
        raise ReleaseEnvelopeError("R2 logical key does not contain the complete archive SHA-256.")
    for field in ("content_disposition", "content_type"):
        _require_string(artifact[field], f"R2 artifact {field}")
    raw_objects = receipt["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ReleaseEnvelopeError("R2 receipt must contain at least one public object.")
    objects = [_validate_r2_object(item, base_url=base_url) for item in raw_objects]
    keys = [item["key"] for item in objects]
    if len(keys) != len(set(keys)):
        raise ReleaseEnvelopeError("R2 receipt repeats an object key.")
    storage = _require_exact_keys(
        receipt["storage"], {"index_key", "mode", "part_bytes"}, "R2 receipt storage"
    )
    mode = storage["mode"]
    if mode not in {"single-object", "deterministic-parts"}:
        raise ReleaseEnvelopeError("R2 receipt has an unknown storage mode.")
    if mode == "single-object":
        if storage["index_key"] is not None or storage["part_bytes"] is not None:
            raise ReleaseEnvelopeError("Single-object R2 storage has multipart fields.")
        if len(objects) != 1 or objects[0]["role"] != "artifact":
            raise ReleaseEnvelopeError("Single-object R2 receipt must contain one artifact object.")
        if {
            "bytes": objects[0]["bytes"],
            "key": objects[0]["key"],
            "sha256": objects[0]["sha256"],
        } != {"bytes": archive.bytes, "key": artifact_key, "sha256": archive.sha256}:
            raise ReleaseEnvelopeError("Single R2 object identity differs from the archive.")
        if (
            objects[0]["content_disposition"] != artifact["content_disposition"]
            or objects[0]["content_type"] != artifact["content_type"]
        ):
            raise ReleaseEnvelopeError("Single R2 object metadata differs from the artifact.")
        public_entry = objects[0]
    else:
        if storage["part_bytes"] != PART_BYTES:
            raise ReleaseEnvelopeError("Deterministic-parts receipt has an unexpected part size.")
        index_key = _require_string(storage["index_key"], "R2 index key")
        parts = [item for item in objects if item["role"] == "part"]
        indexes = [item for item in objects if item["role"] == "index"]
        if (
            not parts
            or len(indexes) != 1
            or [item["role"] for item in objects] != ["part"] * len(parts) + ["index"]
        ):
            raise ReleaseEnvelopeError("Deterministic-parts receipt shape differs.")
        if indexes[0]["key"] != index_key:
            raise ReleaseEnvelopeError("R2 storage index key differs from its object.")
        if sum(item["bytes"] for item in parts) != archive.bytes:
            raise ReleaseEnvelopeError("R2 part byte counts do not reconstruct the archive.")
        for number, item in enumerate(parts):
            expected_prefix = f"{artifact_key}.part-{number:05d}-"
            if item["key"] != expected_prefix + item["sha256"]:
                raise ReleaseEnvelopeError("R2 part key is not deterministic or contiguous.")
            if number < len(parts) - 1 and item["bytes"] != PART_BYTES:
                raise ReleaseEnvelopeError("Non-final R2 part has an unexpected size.")
            if item["bytes"] > PART_BYTES:
                raise ReleaseEnvelopeError("R2 part exceeds the deterministic part size.")
        if index_key != f"{artifact_key}.index-{indexes[0]['sha256']}.json":
            raise ReleaseEnvelopeError("R2 index key is not content addressed.")
        public_entry = indexes[0]
    expected_verification = {
        "anonymous_full_get_sha256": "pass",
        "anonymous_range_cors_metadata": "pass",
        "authenticated_post_public_sha256": "pass",
        "authenticated_round_trip_sha256": "pass",
        "multipart_stream_reconstruction": (
            "pass" if mode == "deterministic-parts" else "not-applicable"
        ),
    }
    if receipt["verification"] != expected_verification:
        raise ReleaseEnvelopeError("R2 receipt verification gates are incomplete.")
    expected_publisher = {
        "api": "r2-s3-compatible",
        "atomic_precondition": "If-None-Match: *",
        "signature": "AWS Signature Version 4",
    }
    if receipt["publisher"] != expected_publisher:
        raise ReleaseEnvelopeError("R2 receipt publisher contract differs.")
    return {
        "artifact": {
            "bytes": archive.bytes,
            "content_disposition": artifact["content_disposition"],
            "content_type": artifact["content_type"],
            "logical_key": artifact_key,
            "public_entry_key": public_entry["key"],
            "public_entry_url": public_entry["url"],
            "sha256": archive.sha256,
        },
        "bucket": bucket,
        "objects": objects,
        "public_base_url": base_url,
        "storage": {
            "index_key": storage["index_key"],
            "mode": mode,
            "part_bytes": storage["part_bytes"],
        },
        "verification": expected_verification,
    }


def _validate_r2_public_verification(
    payload: Any,
    *,
    receipt_artifact: ResolvedArtifact,
    receipt: dict[str, Any],
) -> None:
    record = _require_exact_keys(
        payload,
        {"artifact", "checks", "objects", "pass", "receipt", "schema_version"},
        "R2 public-verification record",
    )
    if record["schema_version"] != R2_PUBLIC_VERIFICATION_SCHEMA or record["pass"] is not True:
        raise ReleaseEnvelopeError("R2 public-verification record is not passing v1 evidence.")
    if record["receipt"] != receipt_artifact.identity:
        raise ReleaseEnvelopeError("R2 public-verification record binds a different receipt.")
    if record["artifact"] != receipt["artifact"]:
        raise ReleaseEnvelopeError("R2 public-verification artifact identity differs from the receipt.")
    if record["checks"] != receipt["verification"]:
        raise ReleaseEnvelopeError("R2 public-verification checks differ from the receipt.")
    raw_objects = record["objects"]
    if not isinstance(raw_objects, list) or len(raw_objects) != len(receipt["objects"]):
        raise ReleaseEnvelopeError("R2 public-verification object inventory differs.")
    expected: list[dict[str, Any]] = []
    for item in receipt["objects"]:
        expected.append(
            {
                "bytes": item["bytes"],
                "full_get_status": 200,
                "key": item["key"],
                "range_get_status": 206,
                "sha256": item["sha256"],
                "url": item["url"],
            }
        )
    for raw in raw_objects:
        _require_exact_keys(
            raw,
            {"bytes", "full_get_status", "key", "range_get_status", "sha256", "url"},
            "R2 public-verification object",
        )
    if raw_objects != expected:
        raise ReleaseEnvelopeError("R2 anonymous HTTP object evidence differs from the receipt.")


def _raw_github_url(repository_url: str, commit: str, git_path: str) -> str:
    parsed = urllib.parse.urlsplit(repository_url)
    owner, name = parsed.path.strip("/").split("/", 1)
    path = urllib.parse.quote(git_path, safe="/-._~")
    return f"https://raw.githubusercontent.com/{owner}/{name}/{commit}/{path}"


def _validate_github_public_verification(
    payload: Any,
    *,
    repository: dict[str, Any],
    article: ResolvedArtifact,
    readme: ResolvedArtifact,
) -> None:
    record = _require_exact_keys(
        payload,
        {"content", "pass", "refs", "release", "repository", "schema_version"},
        "GitHub public-verification record",
    )
    if record["schema_version"] != GITHUB_PUBLIC_VERIFICATION_SCHEMA or record["pass"] is not True:
        raise ReleaseEnvelopeError("GitHub public-verification record is not passing v1 evidence.")
    expected_repository = {
        "anonymous_http_status": 200,
        "url": repository["url"],
        "visibility": "public",
    }
    if record["repository"] != expected_repository:
        raise ReleaseEnvelopeError("GitHub repository public evidence differs.")
    expected_release = {
        "anonymous_http_status": 200,
        "tag": repository["release_tag"],
        "target_commit": repository["final_merge_commit"],
        "url": repository["release_url"],
    }
    if record["release"] != expected_release:
        raise ReleaseEnvelopeError("GitHub release public evidence differs.")
    expected_refs = {
        "benchmark_tag": {
            "commit": repository["benchmark_commit"],
            "name": repository["benchmark_tag"],
        },
        "release_tag": {
            "commit": repository["final_merge_commit"],
            "name": repository["release_tag"],
        },
    }
    if record["refs"] != expected_refs:
        raise ReleaseEnvelopeError("GitHub public tag evidence differs.")
    expected_content = {
        "article": {
            "anonymous_http_status": 200,
            "bytes": article.bytes,
            "commit": repository["final_merge_commit"],
            "git_path": "post.md",
            "raw_url": _raw_github_url(
                repository["url"], repository["final_merge_commit"], "post.md"
            ),
            "sha256": article.sha256,
        },
        "readme": {
            "anonymous_http_status": 200,
            "bytes": readme.bytes,
            "commit": repository["final_merge_commit"],
            "git_path": "README.md",
            "raw_url": _raw_github_url(
                repository["url"], repository["final_merge_commit"], "README.md"
            ),
            "sha256": readme.sha256,
        },
    }
    if record["content"] != expected_content:
        raise ReleaseEnvelopeError("GitHub public content evidence differs.")


def _validate_repository_spec(value: Any) -> dict[str, Any]:
    keys = {
        "benchmark_commit",
        "benchmark_tag",
        "final_merge_commit",
        "name",
        "owner",
        "publication_content_commit",
        "release_tag",
        "release_url",
        "url",
        "visibility",
    }
    repository = _require_exact_keys(value, keys, "release repository spec")
    owner = _require_string(repository["owner"], "repository owner")
    name = _require_string(repository["name"], "repository name")
    if _SAFE_REPOSITORY_PART_RE.fullmatch(owner) is None or _SAFE_REPOSITORY_PART_RE.fullmatch(name) is None:
        raise ReleaseEnvelopeError("GitHub owner or repository name is malformed.")
    expected_url = f"https://github.com/{owner}/{name}"
    if repository["url"] != expected_url:
        raise ReleaseEnvelopeError("Repository URL differs from owner/name.")
    _require_https_url(repository["url"], "repository URL")
    if repository["visibility"] != "public":
        raise ReleaseEnvelopeError("External publication requires a publicly visible repository.")
    benchmark_commit = _require_commit(repository["benchmark_commit"], "benchmark commit")
    publication_commit = _require_commit(
        repository["publication_content_commit"], "publication content commit"
    )
    final_commit = _require_commit(repository["final_merge_commit"], "final merge commit")
    if len({benchmark_commit, publication_commit, final_commit}) != 3:
        raise ReleaseEnvelopeError("Benchmark, publication, and final merge commits must differ.")
    benchmark_tag = _require_tag(repository["benchmark_tag"], "benchmark tag")
    release_tag = _require_tag(repository["release_tag"], "release tag")
    expected_release = f"{expected_url}/releases/tag/{urllib.parse.quote(release_tag, safe='-._~')}"
    if repository["release_url"] != expected_release:
        raise ReleaseEnvelopeError("GitHub release URL differs from repository/tag.")
    _require_https_url(repository["release_url"], "GitHub release URL")
    return dict(repository)


def _validate_git(
    *,
    repo: Path,
    repository: dict[str, Any],
    article: ResolvedArtifact,
    readme: ResolvedArtifact,
) -> dict[str, Any]:
    benchmark = repository["benchmark_commit"]
    publication = repository["publication_content_commit"]
    final = repository["final_merge_commit"]
    for commit, label in (
        (benchmark, "benchmark"),
        (publication, "publication content"),
        (final, "final merge"),
    ):
        observed = _git_text(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
        if observed != commit:
            raise ReleaseEnvelopeError(f"{label} commit does not resolve exactly.")
    if _resolve_tag(repo, repository["benchmark_tag"]) != benchmark:
        raise ReleaseEnvelopeError("Benchmark tag does not resolve to the benchmark commit.")
    if _resolve_tag(repo, repository["release_tag"]) != final:
        raise ReleaseEnvelopeError("Release tag does not resolve to the final merge commit.")
    _assert_ancestor(repo, benchmark, publication, "benchmark -> publication content")
    _assert_ancestor(repo, publication, final, "publication content -> final merge")
    parents = _git_text(repo, "rev-list", "--parents", "-n", "1", final).split()
    if len(parents) < 3:
        raise ReleaseEnvelopeError("Final merge commit is not a merge commit.")
    publication_tree = _tree_fingerprint(repo, publication)
    final_tree = _tree_fingerprint(repo, final)
    if final_tree != publication_tree:
        raise ReleaseEnvelopeError(
            "Final merge changed the publication commit's load-bearing tree."
        )
    publication_git_tree = _git_text(repo, "rev-parse", f"{publication}^{{tree}}")
    final_git_tree = _git_text(repo, "rev-parse", f"{final}^{{tree}}")
    if final_git_tree != publication_git_tree:
        raise ReleaseEnvelopeError(
            "Final merge tree differs from the immutable publication content commit."
        )
    origin = _normalize_github_origin(_git_text(repo, "remote", "get-url", "origin"))
    if origin != repository["url"]:
        raise ReleaseEnvelopeError("Git origin differs from the release repository URL.")
    for artifact, git_path in ((article, "post.md"), (readme, "README.md")):
        local = artifact.identity
        if _identity_bytes(_git_blob(repo, publication, git_path)) != local:
            raise ReleaseEnvelopeError(
                f"{git_path} at the publication content commit differs from supplied bytes."
            )
        if _identity_bytes(_git_blob(repo, final, git_path)) != local:
            raise ReleaseEnvelopeError(f"{git_path} changed before the final merge release.")
    return {
        "benchmark_is_ancestor_of_publication": True,
        "final_load_bearing_tree_equal": True,
        "final_load_bearing_tree_sha256": final_tree["sha256"],
        "final_merge_parent_count": len(parents) - 1,
        "final_tree": final_git_tree,
        "final_tree_equal": True,
        "publication_load_bearing_tree_sha256": publication_tree["sha256"],
        "publication_is_ancestor_of_release": True,
        "publication_tree": publication_git_tree,
    }


def _write_once(path: Path, payload: bytes) -> None:
    parent = path.parent.absolute()
    current = parent
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise ReleaseEnvelopeError(f"Envelope output parent is missing: {parent}.") from None
        if stat.S_ISLNK(mode):
            raise ReleaseEnvelopeError(f"Envelope output parent traverses a symlink: {current}.")
        up = current.parent
        if up == current:
            break
        current = up
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ReleaseEnvelopeError("Existing release envelope differs; refusing to overwrite it.")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.tmp-",
            suffix=".json",
            dir=parent,
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        scan_public_artifact(temporary, reject_timestamps=True)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ReleaseEnvelopeError(
                    "Release envelope appeared concurrently with different bytes."
                ) from None
    except (OSError, PublicArtifactPrivacyError, ValueError) as error:
        if isinstance(error, ReleaseEnvelopeError):
            raise
        raise ReleaseEnvelopeError(f"Release envelope write/privacy check failed: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_release_envelope(
    *,
    spec_path: str | Path,
    repo_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Verify every input and write one deterministic external envelope."""

    repo = _assert_repo_root(repo_root)
    spec_artifact_path = _assert_regular_no_symlink_path(
        Path(spec_path), "Release-envelope spec"
    )
    try:
        spec = json.loads(spec_artifact_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseEnvelopeError("Release-envelope spec is not valid UTF-8 JSON.") from error
    spec = _require_exact_keys(spec, {"artifacts", "repository", "schema_version"}, "release spec")
    if spec["schema_version"] != SPEC_SCHEMA:
        raise ReleaseEnvelopeError("Release-envelope spec schema differs.")
    repository = _validate_repository_spec(spec["repository"])
    artifact_specs = _require_exact_keys(
        spec["artifacts"],
        {
            "article",
            "bundle_archive",
            "bundle_manifest",
            "code_freeze",
            "github_public_verification",
            "r2_public_verification",
            "r2_receipt",
            "readme",
        },
        "release spec artifacts",
    )
    artifacts = {
        role: _resolve_artifact(role, raw) for role, raw in artifact_specs.items()
    }
    for role in ("article", "readme"):
        try:
            observed = scan_public_artifact(artifacts[role].path)
        except (FileNotFoundError, OSError, PublicArtifactPrivacyError, ValueError) as error:
            raise ReleaseEnvelopeError(f"{role} privacy scan failed: {error}") from error
        if observed != artifacts[role].identity:
            raise ReleaseEnvelopeError(f"{role} changed during privacy scanning.")

    git_proof = _validate_git(
        repo=repo,
        repository=repository,
        article=artifacts["article"],
        readme=artifacts["readme"],
    )
    manifest, _archive, strict_release_verification = _validate_bundle(
        manifest_artifact=artifacts["bundle_manifest"],
        archive_artifact=artifacts["bundle_archive"],
        code_freeze_artifact=artifacts["code_freeze"],
        article_artifact=artifacts["article"],
    )
    code_freeze_payload = _load_canonical_json(
        artifacts["code_freeze"], "code-freeze record"
    )
    code_freeze = _validate_code_freeze(
        code_freeze_payload,
        repo=repo,
        benchmark_commit=repository["benchmark_commit"],
        benchmark_tag=repository["benchmark_tag"],
        publication_commit=repository["publication_content_commit"],
    )
    r2_receipt_payload = _load_canonical_json(artifacts["r2_receipt"], "R2 receipt")
    r2 = _validate_r2_receipt(r2_receipt_payload, archive=artifacts["bundle_archive"])
    r2_public_payload = _load_canonical_json(
        artifacts["r2_public_verification"], "R2 public-verification record"
    )
    _validate_r2_public_verification(
        r2_public_payload,
        receipt_artifact=artifacts["r2_receipt"],
        receipt=r2,
    )
    github_payload = _load_canonical_json(
        artifacts["github_public_verification"], "GitHub public-verification record"
    )
    _validate_github_public_verification(
        github_payload,
        repository=repository,
        article=artifacts["article"],
        readme=artifacts["readme"],
    )

    envelope: dict[str, Any] = {
        "bundle": {
            "archive": artifacts["bundle_archive"].identity,
            "bundle_name": manifest["bundle_name"],
            "inventory_id": manifest["inventory_id"],
            "manifest": {
                **artifacts["bundle_manifest"].identity,
                "manifest_id": manifest["manifest_id"],
            },
            "strict_release_verification": strict_release_verification,
        },
        "code_freeze": {
            "benchmark_commit": repository["benchmark_commit"],
            **code_freeze,
            "publication_content_commit": repository["publication_content_commit"],
            "record": artifacts["code_freeze"].identity,
        },
        "content": {
            "article": {"git_path": "post.md", **artifacts["article"].identity},
            "readme": {"git_path": "README.md", **artifacts["readme"].identity},
        },
        "github": {
            "public_verification_record": artifacts[
                "github_public_verification"
            ].identity,
            "release_url": repository["release_url"],
            "repository_url": repository["url"],
            "visibility": repository["visibility"],
        },
        "git_proof": git_proof,
        "pass": True,
        "r2": {
            **r2,
            "public_verification_record": artifacts["r2_public_verification"].identity,
            "receipt": artifacts["r2_receipt"].identity,
        },
        "repository": {
            "benchmark": {
                "commit": repository["benchmark_commit"],
                "tag": repository["benchmark_tag"],
            },
            "publication_content_commit": repository["publication_content_commit"],
            "release": {
                "merge_commit": repository["final_merge_commit"],
                "tag": repository["release_tag"],
            },
        },
        "schema_version": ENVELOPE_SCHEMA,
    }
    scan_public_json_value(envelope, reject_timestamps=True)
    payload = canonical_json_bytes(envelope)
    _write_once(Path(output_path), payload)
    if _identity_bytes(Path(output_path).read_bytes()) != _identity_bytes(payload):
        raise ReleaseEnvelopeError("Written release envelope identity differs.")
    return envelope


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic external publication release envelope."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    envelope = build_release_envelope(
        spec_path=args.spec,
        repo_root=args.repo_root,
        output_path=args.output,
    )
    output_identity = _identity_bytes(args.output.read_bytes())
    print(
        "PUBLICATION_EXTERNAL_RELEASE_ENVELOPE_VERIFIED "
        + json.dumps(
            {
                **output_identity,
                "pass": envelope["pass"],
                "schema_version": envelope["schema_version"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
