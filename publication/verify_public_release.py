#!/usr/bin/env python3
"""Generate strict anonymous GitHub and R2 release-verification records.

This tool is intentionally read-only with respect to both services.  It sends
no authentication header, accepts no credential argument, and writes a record
only after the public bytes and immutable references satisfy the external
release-envelope schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PUBLICATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PUBLICATION_ROOT.parent
for _path in (str(PROJECT_ROOT), str(PUBLICATION_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from publish_immutable_r2 import (  # noqa: E402
    RANGE_BYTES,
    REQUIRED_EXPOSED_HEADERS,
)
from release_envelope import (  # noqa: E402
    GITHUB_PUBLIC_VERIFICATION_SCHEMA,
    R2_PUBLIC_VERIFICATION_SCHEMA,
    ReleaseEnvelopeError,
    ResolvedArtifact,
    _validate_r2_receipt,
)
from scan_public_artifact import (  # noqa: E402
    PublicArtifactPrivacyError,
    scan_public_artifact,
)


USER_AGENT = "vectorized-gaussian-renderer-public-release-verifier/1.0"
GITHUB_API_VERSION = "2022-11-28"
MAX_GITHUB_JSON_BYTES = 4 * 1024 * 1024
STREAM_BYTES = 1024 * 1024
MAX_TAG_DEREFERENCES = 8

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_REPOSITORY_PART_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?"
)

Opener = Callable[..., Any]
Sleeper = Callable[[float], None]


class PublicReleaseVerificationError(ValueError):
    """Raised when anonymous publication evidence is incomplete or unequal."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _identity_bytes(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _assert_regular_no_symlink(path: Path, label: str) -> Path:
    absolute = path.absolute()
    current = absolute
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise PublicReleaseVerificationError(f"{label} is missing: {absolute}.") from None
        if stat.S_ISLNK(mode):
            raise PublicReleaseVerificationError(f"{label} traverses a symlink: {current}.")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not stat.S_ISREG(absolute.lstat().st_mode):
        raise PublicReleaseVerificationError(f"{label} must be a regular file.")
    return absolute


def _identity_path(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(STREAM_BYTES), b""):
            digest.update(block)
            byte_count += len(block)
    return {"bytes": byte_count, "sha256": digest.hexdigest()}


def _load_canonical_public_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _assert_regular_no_symlink(path, label)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicReleaseVerificationError(f"{label} is not valid UTF-8 JSON.") from error
    if not isinstance(payload, dict):
        raise PublicReleaseVerificationError(f"{label} must be a JSON object.")
    encoded = canonical_json_bytes(payload)
    if candidate.read_bytes() != encoded:
        raise PublicReleaseVerificationError(f"{label} is not canonically encoded JSON.")
    try:
        observed = scan_public_artifact(candidate)
    except (FileNotFoundError, OSError, PublicArtifactPrivacyError, ValueError) as error:
        raise PublicReleaseVerificationError(f"{label} privacy scan failed: {error}") from error
    expected = _identity_bytes(encoded)
    if observed != expected:
        raise PublicReleaseVerificationError(f"{label} changed during validation.")
    return payload, expected


def _scan_public_input(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    candidate = _assert_regular_no_symlink(path, label)
    try:
        identity = scan_public_artifact(candidate)
    except (FileNotFoundError, OSError, PublicArtifactPrivacyError, ValueError) as error:
        raise PublicReleaseVerificationError(f"{label} privacy scan failed: {error}") from error
    return candidate, identity


def _write_once(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = canonical_json_bytes(payload)
    identity = _identity_bytes(encoded)
    parent = path.parent.absolute()
    current = parent
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise PublicReleaseVerificationError(
                f"Verification output parent is missing: {parent}."
            ) from None
        if stat.S_ISLNK(mode):
            raise PublicReleaseVerificationError(
                f"Verification output parent traverses a symlink: {current}."
            )
        up = current.parent
        if up == current:
            break
        current = up
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise PublicReleaseVerificationError(
                "Existing public-verification record differs; refusing to overwrite it."
            )
        return identity
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.tmp-",
            suffix=".json",
            dir=parent,
            delete=False,
        ) as stream:
            stream.write(encoded)
            temporary = Path(stream.name)
        scan_public_artifact(temporary, reject_timestamps=True)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise PublicReleaseVerificationError(
                    "Public-verification record appeared concurrently with different bytes."
                ) from None
    except (OSError, PublicArtifactPrivacyError, ValueError) as error:
        if isinstance(error, PublicReleaseVerificationError):
            raise
        raise PublicReleaseVerificationError(
            f"Public-verification record write/privacy check failed: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return identity


def _status(response: Any) -> int:
    value = getattr(response, "status", None)
    if value is None:
        value = response.getcode()
    return int(value)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    return value.strip() if isinstance(value, str) else None


def _response_url(response: Any, requested: str) -> str:
    getter = getattr(response, "geturl", None)
    return str(getter()) if callable(getter) else requested


def _open(request: urllib.request.Request, *, opener: Opener, timeout: int) -> Any:
    try:
        return opener(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        raise PublicReleaseVerificationError(
            f"Anonymous HTTP request returned {error.code}: {request.full_url}"
        ) from error
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise PublicReleaseVerificationError(
            f"Anonymous HTTP request failed: {request.full_url}: {error}"
        ) from error


def _assert_response(
    response: Any,
    *,
    request_url: str,
    expected_status: int,
) -> None:
    if _status(response) != expected_status:
        raise PublicReleaseVerificationError(
            f"Anonymous HTTP status differs for {request_url}: "
            f"expected {expected_status}, got {_status(response)}."
        )
    if _response_url(response, request_url) != request_url:
        raise PublicReleaseVerificationError(
            f"Anonymous HTTP redirect is forbidden for immutable evidence: {request_url}."
        )


def _request(
    url: str,
    *,
    accept: str,
    origin: str | None = None,
    method: str = "GET",
    byte_range: tuple[int, int] | None = None,
    github_api: bool = False,
) -> urllib.request.Request:
    headers = {
        "Accept": accept,
        "Accept-Encoding": "identity",
        "Cache-Control": "no-cache",
        "User-Agent": USER_AGENT,
    }
    if origin is not None:
        headers["Origin"] = origin
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    if github_api:
        headers["X-GitHub-Api-Version"] = GITHUB_API_VERSION
    return urllib.request.Request(url, headers=headers, method=method)


def _read_github_json(url: str, *, opener: Opener, timeout: int) -> dict[str, Any]:
    request = _request(
        url,
        accept="application/vnd.github+json",
        github_api=True,
    )
    with _open(request, opener=opener, timeout=timeout) as response:
        _assert_response(response, request_url=url, expected_status=200)
        content_type = (_header(response.headers, "Content-Type") or "").lower()
        if "json" not in content_type:
            raise PublicReleaseVerificationError(
                f"GitHub API response is not JSON: {url}."
            )
        body = response.read(MAX_GITHUB_JSON_BYTES + 1)
    if len(body) > MAX_GITHUB_JSON_BYTES:
        raise PublicReleaseVerificationError(f"GitHub API response is too large: {url}.")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicReleaseVerificationError(
            f"GitHub API response is malformed: {url}."
        ) from error
    if not isinstance(payload, dict):
        raise PublicReleaseVerificationError(f"GitHub API response is not an object: {url}.")
    return payload


def _verify_public_web_page(url: str, *, opener: Opener, timeout: int) -> None:
    request = _request(url, accept="text/html,application/xhtml+xml")
    with _open(request, opener=opener, timeout=timeout) as response:
        _assert_response(response, request_url=url, expected_status=200)
        content_type = (_header(response.headers, "Content-Type") or "").lower()
        if not content_type.startswith("text/html"):
            raise PublicReleaseVerificationError(
                f"Public GitHub page has unexpected content type: {url}."
            )
        if not response.read(1):
            raise PublicReleaseVerificationError(f"Public GitHub page is empty: {url}.")


def _verify_raw_content(
    url: str,
    *,
    expected: dict[str, Any],
    opener: Opener,
    timeout: int,
) -> None:
    request = _request(url, accept="application/octet-stream,text/plain")
    digest = hashlib.sha256()
    byte_count = 0
    with _open(request, opener=opener, timeout=timeout) as response:
        _assert_response(response, request_url=url, expected_status=200)
        content_length = _header(response.headers, "Content-Length")
        if content_length is not None:
            try:
                observed_length = int(content_length)
            except ValueError as error:
                raise PublicReleaseVerificationError(
                    f"Public GitHub content length is malformed: {url}."
                ) from error
            if observed_length != expected["bytes"]:
                raise PublicReleaseVerificationError(
                    f"Public GitHub content length differs: {url}."
                )
        for block in iter(lambda: response.read(STREAM_BYTES), b""):
            digest.update(block)
            byte_count += len(block)
    if {"bytes": byte_count, "sha256": digest.hexdigest()} != expected:
        raise PublicReleaseVerificationError(
            f"Public GitHub content identity differs: {url}."
        )


def _parse_repository_url(repository_url: str) -> tuple[str, str, str]:
    parsed = urllib.parse.urlsplit(repository_url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.endswith("/")
    ):
        raise PublicReleaseVerificationError(
            "Repository URL must be canonical credential-free https://github.com/OWNER/NAME."
        )
    parts = parsed.path.strip("/").split("/")
    if (
        len(parts) != 2
        or any(_REPOSITORY_PART_RE.fullmatch(part) is None for part in parts)
        or parts[1].endswith(".git")
    ):
        raise PublicReleaseVerificationError("Repository URL owner/name is malformed.")
    return parts[0], parts[1], repository_url


def _require_commit(value: str, label: str) -> str:
    if _COMMIT_RE.fullmatch(value) is None:
        raise PublicReleaseVerificationError(f"{label} must be one full lowercase Git commit.")
    return value


def _require_tag(value: str, label: str) -> str:
    if _TAG_RE.fullmatch(value) is None:
        raise PublicReleaseVerificationError(f"{label} must be one simple immutable tag.")
    return value


def _github_api_base(owner: str, name: str) -> str:
    return (
        "https://api.github.com/repos/"
        + urllib.parse.quote(owner, safe="-._~")
        + "/"
        + urllib.parse.quote(name, safe="-._~")
    )


def _resolve_public_github_tag(
    *,
    api_base: str,
    tag: str,
    opener: Opener,
    timeout: int,
) -> str:
    encoded = urllib.parse.quote(tag, safe="-._~")
    ref_url = f"{api_base}/git/ref/tags/{encoded}"
    ref = _read_github_json(ref_url, opener=opener, timeout=timeout)
    if ref.get("ref") != f"refs/tags/{tag}" or not isinstance(ref.get("object"), dict):
        raise PublicReleaseVerificationError(f"GitHub tag ref differs for {tag}.")
    obj = ref["object"]
    seen: set[str] = set()
    for depth in range(MAX_TAG_DEREFERENCES):
        object_type = obj.get("type")
        sha = obj.get("sha")
        if not isinstance(sha, str) or _COMMIT_RE.fullmatch(sha) is None:
            raise PublicReleaseVerificationError(f"GitHub tag object SHA is malformed for {tag}.")
        if object_type == "commit":
            return sha
        if object_type != "tag" or sha in seen:
            raise PublicReleaseVerificationError(f"GitHub tag object chain is malformed for {tag}.")
        seen.add(sha)
        tag_url = f"{api_base}/git/tags/{sha}"
        annotated = _read_github_json(tag_url, opener=opener, timeout=timeout)
        if depth == 0 and annotated.get("tag") != tag:
            raise PublicReleaseVerificationError(
                f"Annotated GitHub tag name differs for {tag}."
            )
        if not isinstance(annotated.get("object"), dict):
            raise PublicReleaseVerificationError(
                f"Annotated GitHub tag target is malformed for {tag}."
            )
        obj = annotated["object"]
    raise PublicReleaseVerificationError(f"GitHub tag chain is too deep for {tag}.")


def _assert_public_repository_api(
    payload: dict[str, Any],
    *,
    owner: str,
    name: str,
    repository_url: str,
) -> None:
    if (
        payload.get("full_name") != f"{owner}/{name}"
        or payload.get("html_url") != repository_url
        or payload.get("private") is not False
        or payload.get("visibility") != "public"
        or payload.get("archived") is not False
        or payload.get("disabled") is not False
    ):
        raise PublicReleaseVerificationError(
            "Anonymous GitHub API does not describe the expected active public repository."
        )


def _assert_public_release_api(
    payload: dict[str, Any],
    *,
    release_tag: str,
    release_url: str,
) -> None:
    if (
        payload.get("tag_name") != release_tag
        or payload.get("html_url") != release_url
        or payload.get("draft") is not False
        or payload.get("prerelease") is not False
        or not isinstance(payload.get("published_at"), str)
        or not payload["published_at"]
    ):
        raise PublicReleaseVerificationError(
            "Anonymous GitHub API does not describe the expected published final release."
        )


def _verify_github_once(
    *,
    repository_url: str,
    benchmark_tag: str,
    benchmark_commit: str,
    release_tag: str,
    final_merge_commit: str,
    article_identity: dict[str, Any],
    readme_identity: dict[str, Any],
    opener: Opener,
    timeout: int,
) -> dict[str, Any]:
    owner, name, repository_url = _parse_repository_url(repository_url)
    api_base = _github_api_base(owner, name)
    repository_api = _read_github_json(api_base, opener=opener, timeout=timeout)
    _assert_public_repository_api(
        repository_api,
        owner=owner,
        name=name,
        repository_url=repository_url,
    )
    _verify_public_web_page(repository_url, opener=opener, timeout=timeout)

    release_url = (
        f"{repository_url}/releases/tag/"
        f"{urllib.parse.quote(release_tag, safe='-._~')}"
    )
    release_api_url = (
        f"{api_base}/releases/tags/{urllib.parse.quote(release_tag, safe='-._~')}"
    )
    release_api = _read_github_json(release_api_url, opener=opener, timeout=timeout)
    _assert_public_release_api(
        release_api,
        release_tag=release_tag,
        release_url=release_url,
    )
    _verify_public_web_page(release_url, opener=opener, timeout=timeout)

    observed_benchmark = _resolve_public_github_tag(
        api_base=api_base,
        tag=benchmark_tag,
        opener=opener,
        timeout=timeout,
    )
    observed_release = _resolve_public_github_tag(
        api_base=api_base,
        tag=release_tag,
        opener=opener,
        timeout=timeout,
    )
    if observed_benchmark != benchmark_commit:
        raise PublicReleaseVerificationError("Public benchmark tag resolves to another commit.")
    if observed_release != final_merge_commit:
        raise PublicReleaseVerificationError("Public release tag resolves to another commit.")

    merge_api = _read_github_json(
        f"{api_base}/git/commits/{final_merge_commit}",
        opener=opener,
        timeout=timeout,
    )
    parents = merge_api.get("parents")
    if (
        merge_api.get("sha") != final_merge_commit
        or not isinstance(parents, list)
        or len(parents) < 2
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("sha"), str)
            or _COMMIT_RE.fullmatch(item["sha"]) is None
            for item in parents
        )
    ):
        raise PublicReleaseVerificationError(
            "Public release commit is not the expected merge commit."
        )

    content: dict[str, dict[str, Any]] = {}
    for role, git_path, identity in (
        ("article", "post.md", article_identity),
        ("readme", "README.md", readme_identity),
    ):
        raw_url = (
            f"https://raw.githubusercontent.com/{owner}/{name}/"
            f"{final_merge_commit}/{urllib.parse.quote(git_path, safe='/-._~')}"
        )
        _verify_raw_content(
            raw_url,
            expected=identity,
            opener=opener,
            timeout=timeout,
        )
        content[role] = {
            "anonymous_http_status": 200,
            "bytes": identity["bytes"],
            "commit": final_merge_commit,
            "git_path": git_path,
            "raw_url": raw_url,
            "sha256": identity["sha256"],
        }

    closing_repository = _read_github_json(api_base, opener=opener, timeout=timeout)
    _assert_public_repository_api(
        closing_repository,
        owner=owner,
        name=name,
        repository_url=repository_url,
    )
    closing_release = _read_github_json(
        release_api_url, opener=opener, timeout=timeout
    )
    _assert_public_release_api(
        closing_release,
        release_tag=release_tag,
        release_url=release_url,
    )
    if _resolve_public_github_tag(
        api_base=api_base,
        tag=benchmark_tag,
        opener=opener,
        timeout=timeout,
    ) != benchmark_commit:
        raise PublicReleaseVerificationError(
            "Public benchmark tag changed during anonymous verification."
        )
    if _resolve_public_github_tag(
        api_base=api_base,
        tag=release_tag,
        opener=opener,
        timeout=timeout,
    ) != final_merge_commit:
        raise PublicReleaseVerificationError(
            "Public release tag changed during anonymous verification."
        )

    return {
        "content": content,
        "pass": True,
        "refs": {
            "benchmark_tag": {"commit": benchmark_commit, "name": benchmark_tag},
            "release_tag": {"commit": final_merge_commit, "name": release_tag},
        },
        "release": {
            "anonymous_http_status": 200,
            "tag": release_tag,
            "target_commit": final_merge_commit,
            "url": release_url,
        },
        "repository": {
            "anonymous_http_status": 200,
            "url": repository_url,
            "visibility": "public",
        },
        "schema_version": GITHUB_PUBLIC_VERIFICATION_SCHEMA,
    }


def verify_github_publication(
    *,
    repository_url: str,
    benchmark_tag: str,
    benchmark_commit: str,
    release_tag: str,
    final_merge_commit: str,
    article_path: str | Path,
    readme_path: str | Path,
    opener: Opener = urllib.request.urlopen,
    timeout: int = 300,
    attempts: int = 3,
    sleeper: Sleeper = time.sleep,
) -> dict[str, Any]:
    """Verify a public GitHub release without credentials and return v1 evidence."""

    _parse_repository_url(repository_url)
    benchmark_tag = _require_tag(benchmark_tag, "Benchmark tag")
    release_tag = _require_tag(release_tag, "Release tag")
    benchmark_commit = _require_commit(benchmark_commit, "Benchmark commit")
    final_merge_commit = _require_commit(final_merge_commit, "Final merge commit")
    if benchmark_commit == final_merge_commit:
        raise PublicReleaseVerificationError(
            "Benchmark and final merge commits must be different."
        )
    article, article_identity = _scan_public_input(Path(article_path), "Article")
    readme, readme_identity = _scan_public_input(Path(readme_path), "README")
    if timeout <= 0 or attempts <= 0:
        raise PublicReleaseVerificationError("Timeout and attempts must be positive.")
    last_error: PublicReleaseVerificationError | None = None
    for attempt in range(1, attempts + 1):
        try:
            record = _verify_github_once(
                repository_url=repository_url,
                benchmark_tag=benchmark_tag,
                benchmark_commit=benchmark_commit,
                release_tag=release_tag,
                final_merge_commit=final_merge_commit,
                article_identity=article_identity,
                readme_identity=readme_identity,
                opener=opener,
                timeout=timeout,
            )
            if _assert_regular_no_symlink(article, "Article") != article:
                raise PublicReleaseVerificationError("Article path changed during verification.")
            if _assert_regular_no_symlink(readme, "README") != readme:
                raise PublicReleaseVerificationError("README path changed during verification.")
            if _identity_path(article) != article_identity:
                raise PublicReleaseVerificationError("Article changed during verification.")
            if _identity_path(readme) != readme_identity:
                raise PublicReleaseVerificationError("README changed during verification.")
            return record
        except PublicReleaseVerificationError as error:
            last_error = error
            if attempt < attempts:
                sleeper(min(2**attempt, 15))
    assert last_error is not None
    raise last_error


def _validate_r2_headers(
    *,
    item: dict[str, Any],
    headers: Mapping[str, str],
    expected_length: int,
    origin: str,
) -> None:
    try:
        content_length = int(_header(headers, "Content-Length") or "-1")
    except ValueError as error:
        raise PublicReleaseVerificationError(
            f"Public R2 Content-Length is malformed for {item['key']}."
        ) from error
    if content_length != expected_length:
        raise PublicReleaseVerificationError(
            f"Public R2 Content-Length differs for {item['key']}."
        )
    content_type = (_header(headers, "Content-Type") or "").split(";", 1)[0].strip().lower()
    if content_type != item["content_type"].lower():
        raise PublicReleaseVerificationError(
            f"Public R2 Content-Type differs for {item['key']}."
        )
    if _header(headers, "Content-Disposition") != item["content_disposition"]:
        raise PublicReleaseVerificationError(
            f"Public R2 Content-Disposition differs for {item['key']}."
        )
    if _header(headers, "Cache-Control") != item["cache_control"]:
        raise PublicReleaseVerificationError(
            f"Public R2 Cache-Control differs for {item['key']}."
        )
    if (_header(headers, "Accept-Ranges") or "").lower() != "bytes":
        raise PublicReleaseVerificationError(
            f"Public R2 range support is absent for {item['key']}."
        )
    if not _header(headers, "ETag"):
        raise PublicReleaseVerificationError(f"Public R2 ETag is absent for {item['key']}.")
    content_encoding = (_header(headers, "Content-Encoding") or "identity").lower()
    if content_encoding != "identity":
        raise PublicReleaseVerificationError(
            f"Public R2 content encoding is not identity for {item['key']}."
        )
    allowed_origin = _header(headers, "Access-Control-Allow-Origin")
    if allowed_origin not in {"*", origin}:
        raise PublicReleaseVerificationError(
            f"Public R2 CORS origin differs for {item['key']}."
        )
    exposed = {
        value.strip().lower()
        for value in (_header(headers, "Access-Control-Expose-Headers") or "").split(",")
        if value.strip()
    }
    if not REQUIRED_EXPOSED_HEADERS.issubset(exposed):
        raise PublicReleaseVerificationError(
            f"Public R2 exposed headers are incomplete for {item['key']}."
        )


def _verify_r2_object(
    *,
    item: dict[str, Any],
    origin: str,
    opener: Opener,
    timeout: int,
    reconstruction: Any | None,
) -> dict[str, Any]:
    url = item["url"]
    head_request = _request(
        url,
        accept="application/octet-stream,application/json",
        origin=origin,
        method="HEAD",
    )
    with _open(head_request, opener=opener, timeout=timeout) as response:
        _assert_response(response, request_url=url, expected_status=200)
        _validate_r2_headers(
            item=item,
            headers=response.headers,
            expected_length=item["bytes"],
            origin=origin,
        )
        if response.read(1):
            raise PublicReleaseVerificationError(
                f"Anonymous R2 HEAD returned a body for {item['key']}."
            )

    full_request = _request(
        url,
        accept="application/octet-stream,application/json",
        origin=origin,
    )
    digest = hashlib.sha256()
    byte_count = 0
    sample_size = min(RANGE_BYTES, item["bytes"])
    head = bytearray()
    tail = bytearray()
    with _open(full_request, opener=opener, timeout=timeout) as response:
        _assert_response(response, request_url=url, expected_status=200)
        _validate_r2_headers(
            item=item,
            headers=response.headers,
            expected_length=item["bytes"],
            origin=origin,
        )
        for block in iter(lambda: response.read(STREAM_BYTES), b""):
            digest.update(block)
            if reconstruction is not None:
                reconstruction.update(block)
            byte_count += len(block)
            if len(head) < sample_size:
                head.extend(block[: sample_size - len(head)])
            tail.extend(block)
            if len(tail) > sample_size:
                del tail[:-sample_size]
    if {"bytes": byte_count, "sha256": digest.hexdigest()} != {
        "bytes": item["bytes"],
        "sha256": item["sha256"],
    }:
        raise PublicReleaseVerificationError(
            f"Anonymous R2 full GET identity differs for {item['key']}."
        )

    ranges = [(0, sample_size - 1, bytes(head))]
    tail_start = item["bytes"] - sample_size
    tail_sample = (tail_start, item["bytes"] - 1, bytes(tail))
    if tail_sample[:2] != ranges[0][:2]:
        ranges.append(tail_sample)
    for start, end, expected in ranges:
        range_request = _request(
            url,
            accept="application/octet-stream,application/json",
            origin=origin,
            byte_range=(start, end),
        )
        with _open(range_request, opener=opener, timeout=timeout) as response:
            _assert_response(response, request_url=url, expected_status=206)
            _validate_r2_headers(
                item=item,
                headers=response.headers,
                expected_length=len(expected),
                origin=origin,
            )
            if _header(response.headers, "Content-Range") != (
                f"bytes {start}-{end}/{item['bytes']}"
            ):
                raise PublicReleaseVerificationError(
                    f"Public R2 Content-Range differs for {item['key']}."
                )
            body = response.read(len(expected) + 1)
        if body != expected:
            raise PublicReleaseVerificationError(
                f"Anonymous R2 range bytes differ for {item['key']}."
            )
    return {
        "bytes": item["bytes"],
        "full_get_status": 200,
        "key": item["key"],
        "range_get_status": 206,
        "sha256": item["sha256"],
        "url": item["url"],
    }


def _verify_r2_once(
    *,
    receipt: dict[str, Any],
    receipt_identity: dict[str, Any],
    origin: str,
    opener: Opener,
    timeout: int,
) -> dict[str, Any]:
    reconstruction = hashlib.sha256()
    reconstructed_bytes = 0
    objects: list[dict[str, Any]] = []
    for item in receipt["objects"]:
        is_part = receipt["storage"]["mode"] == "deterministic-parts" and item["role"] == "part"
        objects.append(
            _verify_r2_object(
                item=item,
                origin=origin,
                opener=opener,
                timeout=timeout,
                reconstruction=reconstruction if is_part else None,
            )
        )
        if is_part:
            reconstructed_bytes += item["bytes"]
    if receipt["storage"]["mode"] == "deterministic-parts" and {
        "bytes": reconstructed_bytes,
        "sha256": reconstruction.hexdigest(),
    } != {
        "bytes": receipt["artifact"]["bytes"],
        "sha256": receipt["artifact"]["sha256"],
    }:
        raise PublicReleaseVerificationError(
            "Anonymous R2 multipart bytes do not reconstruct the archive."
        )
    return {
        "artifact": receipt["artifact"],
        "checks": receipt["verification"],
        "objects": objects,
        "pass": True,
        "receipt": receipt_identity,
        "schema_version": R2_PUBLIC_VERIFICATION_SCHEMA,
    }


def verify_r2_publication(
    *,
    receipt_path: str | Path,
    archive_path: str | Path,
    opener: Opener = urllib.request.urlopen,
    timeout: int = 300,
    attempts: int = 3,
    sleeper: Sleeper = time.sleep,
) -> dict[str, Any]:
    """Verify an R2 publisher receipt anonymously and return v1 evidence."""

    raw_receipt, receipt_identity = _load_canonical_public_json(
        Path(receipt_path), "R2 receipt"
    )
    archive = _assert_regular_no_symlink(Path(archive_path), "Evidence archive")
    archive_identity = _identity_path(archive)
    resolved_archive = ResolvedArtifact(
        role="bundle_archive",
        path=archive,
        bytes=archive_identity["bytes"],
        sha256=archive_identity["sha256"],
    )
    try:
        receipt = _validate_r2_receipt(raw_receipt, archive=resolved_archive)
    except ReleaseEnvelopeError as error:
        raise PublicReleaseVerificationError(f"R2 receipt validation failed: {error}") from error
    if timeout <= 0 or attempts <= 0:
        raise PublicReleaseVerificationError("Timeout and attempts must be positive.")
    last_error: PublicReleaseVerificationError | None = None
    for attempt in range(1, attempts + 1):
        try:
            record = _verify_r2_once(
                receipt=receipt,
                receipt_identity=receipt_identity,
                origin=raw_receipt["cors_origin"],
                opener=opener,
                timeout=timeout,
            )
            receipt_candidate = _assert_regular_no_symlink(
                Path(receipt_path), "R2 receipt"
            )
            archive_candidate = _assert_regular_no_symlink(
                Path(archive_path), "Evidence archive"
            )
            if _identity_path(receipt_candidate) != receipt_identity:
                raise PublicReleaseVerificationError(
                    "R2 receipt changed during anonymous verification."
                )
            if _identity_path(archive_candidate) != archive_identity:
                raise PublicReleaseVerificationError(
                    "Evidence archive changed during anonymous verification."
                )
            return record
        except PublicReleaseVerificationError as error:
            last_error = error
            if attempt < attempts:
                sleeper(min(2**attempt, 15))
    assert last_error is not None
    raise last_error


def _add_network_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate strict anonymous GitHub or R2 release evidence."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    r2 = subparsers.add_parser("r2", help="Verify public R2 bytes from a sealed receipt.")
    r2.add_argument("--receipt", type=Path, required=True)
    r2.add_argument("--archive", type=Path, required=True)
    _add_network_options(r2)

    github = subparsers.add_parser(
        "github", help="Verify a public GitHub repository, refs, release, and content."
    )
    github.add_argument("--repository-url", required=True)
    github.add_argument("--benchmark-tag", required=True)
    github.add_argument("--benchmark-commit", required=True)
    github.add_argument("--release-tag", required=True)
    github.add_argument("--final-merge-commit", required=True)
    github.add_argument("--article", type=Path, required=True)
    github.add_argument("--readme", type=Path, required=True)
    _add_network_options(github)
    return parser.parse_args(arguments)


def run_cli(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        if args.mode == "r2":
            record = verify_r2_publication(
                receipt_path=args.receipt,
                archive_path=args.archive,
                timeout=args.timeout,
                attempts=args.attempts,
            )
        else:
            record = verify_github_publication(
                repository_url=args.repository_url,
                benchmark_tag=args.benchmark_tag,
                benchmark_commit=args.benchmark_commit,
                release_tag=args.release_tag,
                final_merge_commit=args.final_merge_commit,
                article_path=args.article,
                readme_path=args.readme,
                timeout=args.timeout,
                attempts=args.attempts,
            )
        identity = _write_once(args.output, record)
    except (OSError, PublicReleaseVerificationError, ValueError) as error:
        print(f"PUBLIC_RELEASE_VERIFICATION_FAILED {error}", file=sys.stderr)
        return 1
    print(
        "PUBLIC_RELEASE_VERIFICATION_OK "
        + json.dumps(
            {
                "bytes": identity["bytes"],
                "mode": args.mode,
                "sha256": identity["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
