#!/usr/bin/env python3
"""Publish one content-addressed artifact to public R2 without overwrites.

The helper uses R2's S3-compatible API with an atomic ``If-None-Match: *``
PutObject precondition. Credentials are accepted only from the process
environment and are never printed, persisted, or included in receipts. All
remote objects are fully preflighted before the first upload, and an existing
object is accepted only when its complete authenticated byte stream matches
the expected identity.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import http.client
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping


MAX_SINGLE_OBJECT_BYTES = 300 * 1024 * 1024
PART_BYTES = 256 * 1024 * 1024
STREAM_BYTES = 8 * 1024 * 1024
RANGE_BYTES = 64 * 1024
CACHE_CONTROL = "public, max-age=31536000, immutable"
USER_AGENT = "vectorized-gaussian-renderer-publication-verifier/1.0"
RECEIPT_SCHEMA = "vectorized-gaussian-r2-immutable-publication/v2"
INDEX_SCHEMA = "vectorized-gaussian-r2-deterministic-parts/v1"
S3_REGION = "auto"
S3_SERVICE = "s3"
S3_ALGORITHM = "AWS4-HMAC-SHA256"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
MAX_ERROR_RESPONSE_BYTES = 64 * 1024
REQUIRED_EXPOSED_HEADERS = frozenset(
    {
        "accept-ranges",
        "content-length",
        "content-range",
        "content-type",
        "etag",
    }
)
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
class PublicationError(RuntimeError):
    """Base class for fail-closed publication errors."""


class RemoteCollisionError(PublicationError):
    """Raised when an immutable key already contains different bytes."""


@dataclass(frozen=True)
class R2Credentials:
    """Ephemeral R2 S3 credentials; secret fields are excluded from repr."""

    account_id: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    session_token: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Identity:
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ObjectSpec:
    role: str
    key: str
    path: Path
    identity: Identity
    content_type: str
    content_disposition: str
    cache_control: str = CACHE_CONTROL


@dataclass(frozen=True)
class PublicationPlan:
    artifact_key: str
    artifact_identity: Identity
    artifact_content_type: str
    artifact_content_disposition: str
    mode: str
    objects: tuple[ObjectSpec, ...]
    payload_objects: tuple[ObjectSpec, ...]
    index_object: ObjectSpec | None
    part_bytes: int | None


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_stream(stream: BinaryIO) -> Identity:
    digest = hashlib.sha256()
    byte_count = 0
    for block in iter(lambda: stream.read(STREAM_BYTES), b""):
        digest.update(block)
        byte_count += len(block)
    return Identity(bytes=byte_count, sha256=digest.hexdigest())


def sha256_path(path: Path) -> Identity:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def _validate_regular_source(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise PublicationError(f"Artifact does not exist: {path}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PublicationError("Artifact must be a regular, non-symlink file.")
    if info.st_size <= 0:
        raise PublicationError("Empty artifacts cannot satisfy range verification.")
    return info


def _stable_stat_fields(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _validate_key(key: str) -> str:
    if not isinstance(key, str) or not key or len(key) > 1024:
        raise PublicationError("R2 key must be a nonempty string of at most 1024 characters.")
    if key.startswith("/") or "\\" in key or any(ord(character) < 33 for character in key):
        raise PublicationError("R2 key is not a safe relative key.")
    parts = key.split("/")
    if any(
        not part
        or part in {".", ".."}
        or _SAFE_COMPONENT_RE.fullmatch(part) is None
        for part in parts
    ):
        raise PublicationError("R2 key contains an unsafe path component.")
    return key


def _validate_download_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or len(name) > 200
        or _SAFE_COMPONENT_RE.fullmatch(name) is None
    ):
        raise PublicationError(
            "Download name must use only letters, digits, period, underscore, and hyphen."
        )
    return name


def _attachment(name: str) -> str:
    return f'attachment; filename="{_validate_download_name(name)}"'


def _copy_direct(source: Path, destination: Path) -> Identity:
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(STREAM_BYTES), b""):
            writer.write(block)
            digest.update(block)
            byte_count += len(block)
    destination.chmod(0o444)
    return Identity(bytes=byte_count, sha256=digest.hexdigest())


def _split_source(
    source: Path,
    destination_root: Path,
    *,
    part_bytes: int,
) -> tuple[Identity, list[tuple[Path, Identity]]]:
    if part_bytes <= 0:
        raise PublicationError("Part size must be positive.")
    aggregate = hashlib.sha256()
    aggregate_bytes = 0
    parts: list[tuple[Path, Identity]] = []
    with source.open("rb") as reader:
        part_number = 0
        while True:
            path = destination_root / f"part-{part_number:05d}"
            part_digest = hashlib.sha256()
            written = 0
            with path.open("xb") as writer:
                while written < part_bytes:
                    block = reader.read(min(STREAM_BYTES, part_bytes - written))
                    if not block:
                        break
                    writer.write(block)
                    part_digest.update(block)
                    aggregate.update(block)
                    written += len(block)
                    aggregate_bytes += len(block)
            if written == 0:
                path.unlink()
                break
            path.chmod(0o444)
            parts.append(
                (
                    path,
                    Identity(bytes=written, sha256=part_digest.hexdigest()),
                )
            )
            part_number += 1
    return (
        Identity(bytes=aggregate_bytes, sha256=aggregate.hexdigest()),
        parts,
    )


def build_publication_plan(
    *,
    source: Path,
    artifact_key: str,
    staging_root: Path,
    content_type: str = "application/octet-stream",
    download_name: str | None = None,
    max_single_bytes: int = MAX_SINGLE_OBJECT_BYTES,
    part_bytes: int = PART_BYTES,
) -> PublicationPlan:
    """Snapshot an artifact and derive all immutable object identities."""

    source = source.absolute()
    before = _validate_regular_source(source)
    source = source.resolve()
    if max_single_bytes <= 0:
        raise PublicationError("Single-object limit must be positive.")
    if not isinstance(content_type, str) or not re.fullmatch(
        r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+", content_type
    ):
        raise PublicationError("Content type is not a canonical media type.")
    artifact_key = _validate_key(artifact_key)
    download_name = _validate_download_name(download_name or source.name)
    if staging_root.exists() and any(staging_root.iterdir()):
        raise PublicationError("Publication staging directory must be empty.")
    staging_root.mkdir(parents=True, exist_ok=True)

    if before.st_size <= max_single_bytes:
        snapshot = staging_root / "artifact.snapshot"
        artifact_identity = _copy_direct(source, snapshot)
        parts: list[tuple[Path, Identity]] = []
    else:
        snapshot = None
        artifact_identity, parts = _split_source(
            source,
            staging_root,
            part_bytes=part_bytes,
        )
    after = source.lstat()
    if _stable_stat_fields(before) != _stable_stat_fields(after):
        raise PublicationError("Artifact changed while it was being snapshotted.")
    if artifact_identity.bytes != before.st_size:
        raise PublicationError("Artifact byte count changed while it was being snapshotted.")
    if artifact_identity.sha256 not in artifact_key:
        raise PublicationError(
            "R2 key must include the artifact's complete lowercase SHA-256."
        )

    disposition = _attachment(download_name)
    if snapshot is not None:
        direct = ObjectSpec(
            role="artifact",
            key=artifact_key,
            path=snapshot,
            identity=artifact_identity,
            content_type=content_type,
            content_disposition=disposition,
        )
        return PublicationPlan(
            artifact_key=artifact_key,
            artifact_identity=artifact_identity,
            artifact_content_type=content_type,
            artifact_content_disposition=disposition,
            mode="single-object",
            objects=(direct,),
            payload_objects=(direct,),
            index_object=None,
            part_bytes=None,
        )

    part_specs: list[ObjectSpec] = []
    for part_number, (path, identity) in enumerate(parts):
        part_name = f"{download_name}.part-{part_number:05d}"
        part_key = _validate_key(
            f"{artifact_key}.part-{part_number:05d}-{identity.sha256}"
        )
        part_specs.append(
            ObjectSpec(
                role="part",
                key=part_key,
                path=path,
                identity=identity,
                content_type="application/octet-stream",
                content_disposition=_attachment(part_name),
            )
        )
    index_payload = {
        "artifact": {
            "bytes": artifact_identity.bytes,
            "content_disposition": disposition,
            "content_type": content_type,
            "key": artifact_key,
            "sha256": artifact_identity.sha256,
        },
        "cache_control": CACHE_CONTROL,
        "part_bytes": part_bytes,
        "parts": [
            {
                "bytes": part.identity.bytes,
                "key": part.key,
                "sha256": part.identity.sha256,
            }
            for part in part_specs
        ],
        "schema_version": INDEX_SCHEMA,
    }
    index_path = staging_root / "index.json"
    index_path.write_bytes(canonical_json_bytes(index_payload))
    index_path.chmod(0o444)
    index_identity = sha256_path(index_path)
    index_spec = ObjectSpec(
        role="index",
        key=_validate_key(f"{artifact_key}.index-{index_identity.sha256}.json"),
        path=index_path,
        identity=index_identity,
        content_type="application/json",
        content_disposition=_attachment(f"{download_name}.index.json"),
    )
    return PublicationPlan(
        artifact_key=artifact_key,
        artifact_identity=artifact_identity,
        artifact_content_type=content_type,
        artifact_content_disposition=disposition,
        mode="deterministic-parts",
        objects=tuple([*part_specs, index_spec]),
        payload_objects=tuple(part_specs),
        index_object=index_spec,
        part_bytes=part_bytes,
    )


def load_ambient_r2_credentials() -> R2Credentials:
    """Load the standard R2 S3 credential tuple without persisting it."""

    names = (
        "CLOUDFLARE_ACCOUNT_ID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    )
    values = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise PublicationError(
            "Atomic R2 publication requires ambient " + ", ".join(missing) + "."
        )
    account_id = values["CLOUDFLARE_ACCOUNT_ID"]
    access_key_id = values["AWS_ACCESS_KEY_ID"]
    secret_access_key = values["AWS_SECRET_ACCESS_KEY"]
    assert account_id is not None
    assert access_key_id is not None
    assert secret_access_key is not None
    if re.fullmatch(r"[0-9A-Fa-f]{32}", account_id) is None:
        raise PublicationError("CLOUDFLARE_ACCOUNT_ID is not a 32-digit hexadecimal ID.")
    for name, value in (
        ("AWS_ACCESS_KEY_ID", access_key_id),
        ("AWS_SECRET_ACCESS_KEY", secret_access_key),
    ):
        if (
            len(value) > 1024
            or value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise PublicationError(f"{name} is malformed.")
    session_token = os.environ.get("AWS_SESSION_TOKEN")
    if session_token is not None and (
        not session_token
        or len(session_token) > 4096
        or session_token != session_token.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in session_token)
    ):
        raise PublicationError("AWS_SESSION_TOKEN is malformed.")
    return R2Credentials(
        account_id=account_id.lower(),
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
    )


def _canonical_header_value(value: str) -> str:
    return " ".join(value.strip().split())


def _hmac_sha256(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


class ConditionalR2Store:
    """Authenticated R2 S3 access with atomic create-only PutObject."""

    def __init__(
        self,
        bucket: str,
        credentials: R2Credentials,
        *,
        connection_factory: Callable[..., Any] = http.client.HTTPSConnection,
        clock: Callable[[], dt.datetime] | None = None,
        timeout: int = 300,
    ):
        if _SAFE_COMPONENT_RE.fullmatch(bucket) is None:
            raise PublicationError("Bucket name is not safe.")
        if timeout <= 0:
            raise PublicationError("Authenticated R2 timeout must be positive.")
        self.bucket = bucket
        self._credentials = credentials
        self._connection_factory = connection_factory
        self._clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
        self._timeout = timeout
        self._host = f"{credentials.account_id}.r2.cloudflarestorage.com"

    def _object_path(self, key: str) -> str:
        key = _validate_key(key)
        return "/" + urllib.parse.quote(
            f"{self.bucket}/{key}",
            safe="/-._~",
        )

    def _signed_headers(
        self,
        *,
        method: str,
        canonical_uri: str,
        payload_sha256: str,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        now = self._clock()
        if now.tzinfo is None:
            raise PublicationError("R2 signing clock must be timezone-aware.")
        now = now.astimezone(dt.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        signed = {
            "host": self._host,
            "x-amz-content-sha256": payload_sha256,
            "x-amz-date": amz_date,
        }
        if headers:
            for name, value in headers.items():
                lower_name = name.lower()
                if lower_name in signed:
                    raise PublicationError(f"Duplicate signed R2 header: {lower_name}.")
                signed[lower_name] = value
        if self._credentials.session_token is not None:
            signed["x-amz-security-token"] = self._credentials.session_token
        canonical_names = sorted(signed)
        canonical_headers = "".join(
            f"{name}:{_canonical_header_value(signed[name])}\n"
            for name in canonical_names
        )
        signed_header_names = ";".join(canonical_names)
        canonical_request = "\n".join(
            (
                method,
                canonical_uri,
                "",
                canonical_headers,
                signed_header_names,
                payload_sha256,
            )
        )
        scope = f"{date_stamp}/{S3_REGION}/{S3_SERVICE}/aws4_request"
        string_to_sign = "\n".join(
            (
                S3_ALGORITHM,
                amz_date,
                scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            )
        )
        date_key = _hmac_sha256(
            ("AWS4" + self._credentials.secret_access_key).encode("utf-8"),
            date_stamp,
        )
        region_key = _hmac_sha256(date_key, S3_REGION)
        service_key = _hmac_sha256(region_key, S3_SERVICE)
        signing_key = _hmac_sha256(service_key, "aws4_request")
        signature = hmac.new(
            signing_key,
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed["authorization"] = (
            f"{S3_ALGORITHM} Credential={self._credentials.access_key_id}/{scope}, "
            f"SignedHeaders={signed_header_names}, Signature={signature}"
        )
        return signed

    def _start_request(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
    ) -> Any:
        connection = self._connection_factory(self._host, timeout=self._timeout)
        try:
            connection.putrequest(
                method,
                path,
                skip_host=True,
                skip_accept_encoding=True,
            )
            for name in sorted(headers):
                connection.putheader(name, headers[name])
            connection.endheaders()
        except BaseException:
            connection.close()
            raise
        return connection

    @staticmethod
    def _discard_response(response: Any) -> None:
        response.read(MAX_ERROR_RESPONSE_BYTES + 1)

    def get_identity(self, key: str) -> Identity | None:
        """Return the authenticated full-object identity, or None if absent."""

        path = self._object_path(key)
        headers = self._signed_headers(
            method="GET",
            canonical_uri=path,
            payload_sha256=EMPTY_SHA256,
        )
        connection = None
        response = None
        try:
            connection = self._start_request(method="GET", path=path, headers=headers)
            response = connection.getresponse()
            status = int(response.status)
            if status == 200:
                return sha256_stream(response)
            self._discard_response(response)
            if status == 404:
                return None
            raise PublicationError(
                f"Authenticated R2 GET failed for {key} with HTTP {status}; "
                "credentials, authorization, or network access may be unavailable."
            )
        except PublicationError:
            raise
        except (OSError, http.client.HTTPException) as error:
            raise PublicationError(f"Authenticated R2 GET failed for {key}.") from error
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()

    def put_if_absent(self, spec: ObjectSpec) -> bool:
        """Atomically create one object; return False only for identical bytes."""

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(spec.path, flags)
        except OSError as error:
            raise PublicationError(f"Could not open staged object for {spec.key}.") from error
        connection = None
        response = None
        try:
            with os.fdopen(descriptor, "rb") as body:
                before = os.fstat(body.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != spec.identity.bytes:
                    raise PublicationError(f"Staged object identity changed for {spec.key}.")
                local_identity = sha256_stream(body)
                after_hash = os.fstat(body.fileno())
                if (
                    local_identity != spec.identity
                    or _stable_stat_fields(before) != _stable_stat_fields(after_hash)
                ):
                    raise PublicationError(f"Staged object identity changed for {spec.key}.")
                body.seek(0)
                path = self._object_path(spec.key)
                headers = self._signed_headers(
                    method="PUT",
                    canonical_uri=path,
                    payload_sha256=spec.identity.sha256,
                    headers={
                        "cache-control": spec.cache_control,
                        "content-disposition": spec.content_disposition,
                        "content-length": str(spec.identity.bytes),
                        "content-type": spec.content_type,
                        "if-none-match": "*",
                    },
                )
                connection = self._start_request(
                    method="PUT",
                    path=path,
                    headers=headers,
                )
                remaining = spec.identity.bytes
                while remaining:
                    block = body.read(min(STREAM_BYTES, remaining))
                    if not block:
                        raise PublicationError(
                            f"Staged object became truncated for {spec.key}."
                        )
                    connection.send(block)
                    remaining -= len(block)
                if body.read(1):
                    raise PublicationError(f"Staged object grew during upload for {spec.key}.")
                after_send = os.fstat(body.fileno())
                if _stable_stat_fields(before) != _stable_stat_fields(after_send):
                    raise PublicationError(f"Staged object changed during upload for {spec.key}.")
                response = connection.getresponse()
                status = int(response.status)
                self._discard_response(response)
        except PublicationError:
            raise
        except (OSError, http.client.HTTPException) as error:
            raise PublicationError(f"Atomic R2 PUT failed for {spec.key}.") from error
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()

        if status == 200:
            return True
        if status == 412:
            actual = self.get_identity(spec.key)
            if actual == spec.identity:
                return False
            if actual is not None:
                raise RemoteCollisionError(
                    f"Conditional create found unequal bytes at immutable key: {spec.key}"
                )
            raise PublicationError(
                f"Conditional create failed but no object is now readable at {spec.key}."
            )
        raise PublicationError(
            f"Atomic R2 PUT failed for {spec.key} with HTTP {status}; "
            "the object was not accepted as a proven conditional create."
        )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    return value.strip() if isinstance(value, str) else None


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return int(status)


class PublicVerifier:
    """Anonymous public HTTP verifier with CORS and range assertions."""

    def __init__(
        self,
        public_base_url: str,
        *,
        origin: str,
        opener: Callable[..., Any] = urllib.request.urlopen,
        attempts: int = 6,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        parsed = urllib.parse.urlsplit(public_base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise PublicationError("Public base URL must be credential-free HTTPS.")
        origin_parts = urllib.parse.urlsplit(origin)
        if (
            origin_parts.scheme not in {"http", "https"}
            or not origin_parts.netloc
            or origin_parts.username is not None
            or origin_parts.password is not None
            or origin_parts.path not in {"", "/"}
            or origin_parts.query
            or origin_parts.fragment
        ):
            raise PublicationError("CORS origin is not a safe HTTP origin.")
        if attempts < 1:
            raise PublicationError("Public verification attempts must be positive.")
        self.public_base_url = public_base_url.rstrip("/")
        self.origin = f"{origin_parts.scheme}://{origin_parts.netloc}"
        self.opener = opener
        self.attempts = attempts
        self.sleeper = sleeper

    def url(self, key: str) -> str:
        encoded = urllib.parse.quote(key, safe="/-._~")
        return f"{self.public_base_url}/{encoded}"

    def _request(
        self,
        key: str,
        *,
        method: str = "GET",
        byte_range: tuple[int, int] | None = None,
    ) -> Any:
        headers = {
            "Origin": self.origin,
            "User-Agent": USER_AGENT,
        }
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        return urllib.request.Request(
            self.url(key),
            headers=headers,
            method=method,
        )

    def _validate_common_headers(
        self,
        spec: ObjectSpec,
        headers: Mapping[str, str],
        *,
        expected_length: int,
    ) -> None:
        try:
            content_length = int(_header(headers, "Content-Length") or "-1")
        except ValueError as error:
            raise PublicationError(f"Public Content-Length is malformed for {spec.key}.") from error
        if content_length != expected_length:
            raise PublicationError(f"Public Content-Length differs for {spec.key}.")
        content_type = (_header(headers, "Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != spec.content_type.lower():
            raise PublicationError(f"Public Content-Type differs for {spec.key}.")
        if _header(headers, "Content-Disposition") != spec.content_disposition:
            raise PublicationError(f"Public Content-Disposition differs for {spec.key}.")
        if _header(headers, "Cache-Control") != spec.cache_control:
            raise PublicationError(f"Public Cache-Control differs for {spec.key}.")
        if (_header(headers, "Accept-Ranges") or "").lower() != "bytes":
            raise PublicationError(f"Public range support is absent for {spec.key}.")
        if not _header(headers, "ETag"):
            raise PublicationError(f"Public ETag is absent for {spec.key}.")
        allowed_origin = _header(headers, "Access-Control-Allow-Origin")
        if allowed_origin not in {"*", self.origin}:
            raise PublicationError(f"Public CORS origin differs for {spec.key}.")
        exposed = {
            value.strip().lower()
            for value in (_header(headers, "Access-Control-Expose-Headers") or "").split(",")
            if value.strip()
        }
        if not REQUIRED_EXPOSED_HEADERS.issubset(exposed):
            raise PublicationError(f"Public CORS exposed headers are incomplete for {spec.key}.")

    def _verify_once(self, spec: ObjectSpec) -> None:
        head_request = self._request(spec.key, method="HEAD")
        with self.opener(head_request, timeout=120) as response:
            if _response_status(response) != 200:
                raise PublicationError(f"Anonymous HEAD did not return 200 for {spec.key}.")
            self._validate_common_headers(
                spec,
                response.headers,
                expected_length=spec.identity.bytes,
            )
            if response.read(1):
                raise PublicationError(f"Anonymous HEAD returned a body for {spec.key}.")

        request = self._request(spec.key)
        with self.opener(request, timeout=120) as response:
            if _response_status(response) != 200:
                raise PublicationError(f"Anonymous GET did not return 200 for {spec.key}.")
            self._validate_common_headers(
                spec,
                response.headers,
                expected_length=spec.identity.bytes,
            )
            identity = sha256_stream(response)
        if identity != spec.identity:
            raise PublicationError(f"Anonymous GET identity differs for {spec.key}.")

        sample = min(RANGE_BYTES, spec.identity.bytes)
        ranges = [(0, sample - 1)]
        tail = (spec.identity.bytes - sample, spec.identity.bytes - 1)
        if tail != ranges[0]:
            ranges.append(tail)
        with spec.path.open("rb") as local:
            for start, end in ranges:
                local.seek(start)
                expected = local.read(end - start + 1)
                range_request = self._request(spec.key, byte_range=(start, end))
                with self.opener(range_request, timeout=120) as response:
                    if _response_status(response) != 206:
                        raise PublicationError(f"Anonymous range GET did not return 206 for {spec.key}.")
                    self._validate_common_headers(
                        spec,
                        response.headers,
                        expected_length=len(expected),
                    )
                    expected_content_range = f"bytes {start}-{end}/{spec.identity.bytes}"
                    if _header(response.headers, "Content-Range") != expected_content_range:
                        raise PublicationError(f"Public Content-Range differs for {spec.key}.")
                    body = response.read()
                if body != expected:
                    raise PublicationError(f"Anonymous range bytes differ for {spec.key}.")

    def verify(self, spec: ObjectSpec) -> None:
        last_error: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                self._verify_once(spec)
                return
            except (
                OSError,
                TimeoutError,
                urllib.error.URLError,
                PublicationError,
            ) as error:
                last_error = error
                if attempt < self.attempts:
                    self.sleeper(min(2**attempt, 15))
        assert last_error is not None
        raise PublicationError(
            f"Anonymous public verification failed for {spec.key}: {last_error}"
        ) from last_error

    def _reconstruct_once(self, parts: Iterable[ObjectSpec]) -> Identity:
        aggregate = hashlib.sha256()
        aggregate_bytes = 0
        for spec in parts:
            request = self._request(spec.key)
            with self.opener(request, timeout=120) as response:
                if _response_status(response) != 200:
                    raise PublicationError(f"Multipart reconstruction GET failed for {spec.key}.")
                self._validate_common_headers(
                    spec,
                    response.headers,
                    expected_length=spec.identity.bytes,
                )
                part_digest = hashlib.sha256()
                part_bytes = 0
                for block in iter(lambda: response.read(STREAM_BYTES), b""):
                    part_digest.update(block)
                    aggregate.update(block)
                    part_bytes += len(block)
                    aggregate_bytes += len(block)
            if Identity(part_bytes, part_digest.hexdigest()) != spec.identity:
                raise PublicationError(f"Multipart reconstruction part differs for {spec.key}.")
        return Identity(bytes=aggregate_bytes, sha256=aggregate.hexdigest())

    def verify_reconstruction(
        self,
        parts: Iterable[ObjectSpec],
        expected: Identity,
    ) -> None:
        part_list = tuple(parts)
        last_error: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                if self._reconstruct_once(part_list) != expected:
                    raise PublicationError("Multipart stream reconstruction identity differs.")
                return
            except (
                OSError,
                TimeoutError,
                urllib.error.URLError,
                PublicationError,
            ) as error:
                last_error = error
                if attempt < self.attempts:
                    self.sleeper(min(2**attempt, 15))
        assert last_error is not None
        raise PublicationError(
            f"Anonymous multipart reconstruction failed: {last_error}"
        ) from last_error


def _identity_matches(actual: Identity | None, expected: Identity) -> bool:
    return actual is not None and actual == expected


def _object_receipt(spec: ObjectSpec, verifier: PublicVerifier) -> dict[str, Any]:
    return {
        "bytes": spec.identity.bytes,
        "cache_control": spec.cache_control,
        "content_disposition": spec.content_disposition,
        "content_type": spec.content_type,
        "key": spec.key,
        "role": spec.role,
        "sha256": spec.identity.sha256,
        "url": verifier.url(spec.key),
    }


def build_receipt(
    *,
    plan: PublicationPlan,
    bucket: str,
    verifier: PublicVerifier,
) -> dict[str, Any]:
    return {
        "artifact": {
            "bytes": plan.artifact_identity.bytes,
            "content_disposition": plan.artifact_content_disposition,
            "content_type": plan.artifact_content_type,
            "key": plan.artifact_key,
            "sha256": plan.artifact_identity.sha256,
        },
        "bucket": bucket,
        "cors_origin": verifier.origin,
        "objects": [_object_receipt(spec, verifier) for spec in plan.objects],
        "public_base_url": verifier.public_base_url,
        "schema_version": RECEIPT_SCHEMA,
        "storage": {
            "index_key": plan.index_object.key if plan.index_object else None,
            "mode": plan.mode,
            "part_bytes": plan.part_bytes,
        },
        "verification": {
            "anonymous_full_get_sha256": "pass",
            "anonymous_range_cors_metadata": "pass",
            "authenticated_post_public_sha256": "pass",
            "authenticated_round_trip_sha256": "pass",
            "multipart_stream_reconstruction": (
                "pass" if plan.mode == "deterministic-parts" else "not-applicable"
            ),
        },
        "publisher": {
            "api": "r2-s3-compatible",
            "atomic_precondition": "If-None-Match: *",
            "signature": "AWS Signature Version 4",
        },
    }


def write_deterministic_receipt(path: Path, receipt: dict[str, Any]) -> Identity:
    payload = canonical_json_bytes(receipt)
    identity = Identity(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PublicationError("Existing receipt differs; refusing to overwrite it.")
        return identity
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        raise PublicationError("Receipt appeared concurrently; refusing to overwrite it.") from None
    return identity


def publish_plan(
    *,
    plan: PublicationPlan,
    store: Any,
    verifier: PublicVerifier,
    receipt_path: Path,
) -> dict[str, Any]:
    """Preflight, upload missing objects, round-trip verify, and seal receipt."""

    states: dict[str, str] = {}
    for spec in plan.objects:
        actual = store.get_identity(spec.key)
        if actual is None:
            states[spec.key] = "missing"
        elif actual != spec.identity:
            raise RemoteCollisionError(
                f"Immutable key already contains unequal bytes: {spec.key}"
            )
        else:
            states[spec.key] = "identical"

    uploaded = 0
    skipped = 0
    for spec in plan.objects:
        if states[spec.key] == "identical":
            skipped += 1
            continue
        # Re-probe for an early, intelligible collision. The signed conditional
        # PUT remains the authority that closes the race after this read.
        actual = store.get_identity(spec.key)
        if actual is not None:
            if actual != spec.identity:
                raise RemoteCollisionError(
                    f"Immutable key changed after preflight: {spec.key}"
                )
            skipped += 1
            continue
        created = store.put_if_absent(spec)
        if created:
            uploaded += 1
        else:
            skipped += 1

    for spec in plan.objects:
        actual = store.get_identity(spec.key)
        if not _identity_matches(actual, spec.identity):
            raise PublicationError(
                f"Authenticated round-trip identity differs for {spec.key}."
            )
    for spec in plan.objects:
        verifier.verify(spec)
    if plan.mode == "deterministic-parts":
        verifier.verify_reconstruction(
            plan.payload_objects,
            plan.artifact_identity,
        )
    # Close the public-verification window with another origin-authenticated
    # read. This cannot prevent a future authorized overwrite, but it prevents
    # a cached public response from concealing an origin change during this run.
    for spec in plan.objects:
        actual = store.get_identity(spec.key)
        if not _identity_matches(actual, spec.identity):
            raise PublicationError(
                f"Final authenticated identity differs for {spec.key}."
            )

    receipt = build_receipt(
        plan=plan,
        bucket=store.bucket,
        verifier=verifier,
    )
    receipt_identity = write_deterministic_receipt(receipt_path, receipt)
    return {
        "receipt": receipt,
        "receipt_identity": receipt_identity,
        "skipped": skipped,
        "uploaded": uploaded,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish and publicly verify one immutable R2 artifact."
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--public-base-url", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--content-type", default="application/octet-stream")
    parser.add_argument("--download-name")
    parser.add_argument("--origin", default="https://github.com")
    return parser.parse_args()


def run_cli() -> int:
    args = parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="vgr-r2-publication-") as temporary:
            plan = build_publication_plan(
                source=args.file,
                artifact_key=args.key,
                staging_root=Path(temporary),
                content_type=args.content_type,
                download_name=args.download_name,
            )
            store = ConditionalR2Store(
                args.bucket,
                load_ambient_r2_credentials(),
            )
            verifier = PublicVerifier(
                args.public_base_url,
                origin=args.origin,
            )
            result = publish_plan(
                plan=plan,
                store=store,
                verifier=verifier,
                receipt_path=args.receipt,
            )
        receipt_identity: Identity = result["receipt_identity"]
        print(
            json.dumps(
                {
                    "receipt_bytes": receipt_identity.bytes,
                    "receipt_sha256": receipt_identity.sha256,
                    "skipped": result["skipped"],
                    "uploaded": result["uploaded"],
                    "verification": "pass",
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, PublicationError, ValueError) as error:
        print(f"IMMUTABLE_R2_PUBLICATION_FAILED {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
