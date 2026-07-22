from __future__ import annotations

import hashlib
import io
import json
import datetime as dt
import tempfile
import unittest
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from publication.publish_immutable_r2 import (
    CACHE_CONTROL,
    ConditionalR2Store,
    Identity,
    ObjectSpec,
    PublicationError,
    PublicVerifier,
    R2Credentials,
    RemoteCollisionError,
    build_publication_plan,
    load_ambient_r2_credentials,
    publish_plan,
)


@dataclass
class _RemoteObject:
    body: bytes
    content_type: str
    content_disposition: str
    cache_control: str


class _FakeStore:
    def __init__(self, bucket: str = "test-bucket") -> None:
        self.bucket = bucket
        self.objects: dict[str, _RemoteObject] = {}
        self.puts: list[str] = []
        self.probes: list[str] = []
        self.race_body: bytes | None = None

    def seed(self, key: str, body: bytes, spec: ObjectSpec) -> None:
        self.objects[key] = _RemoteObject(
            body=body,
            content_type=spec.content_type,
            content_disposition=spec.content_disposition,
            cache_control=spec.cache_control,
        )

    def get_identity(self, key: str) -> Identity | None:
        self.probes.append(key)
        value = self.objects.get(key)
        if value is None:
            return None
        return Identity(
            bytes=len(value.body),
            sha256=hashlib.sha256(value.body).hexdigest(),
        )

    def put_if_absent(self, spec: ObjectSpec) -> bool:
        if self.race_body is not None:
            body = self.race_body
            self.race_body = None
            self.seed(spec.key, body, spec)
        if spec.key in self.objects:
            value = self.objects[spec.key]
            actual = Identity(len(value.body), hashlib.sha256(value.body).hexdigest())
            if actual != spec.identity:
                raise RemoteCollisionError("atomic fake refused unequal concurrent object")
            return False
        body = spec.path.read_bytes()
        self.objects[spec.key] = _RemoteObject(
            body=body,
            content_type=spec.content_type,
            content_disposition=spec.content_disposition,
            cache_control=spec.cache_control,
        )
        self.puts.append(spec.key)
        return True


class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, *, status: int, headers: dict[str, str]) -> None:
        super().__init__(body)
        self.status = status
        self.headers = headers

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _RecordingS3Connection:
    def __init__(self, host: str, timeout: int, response: _FakeResponse) -> None:
        self.host = host
        self.timeout = timeout
        self.response = response
        self.method: str | None = None
        self.path: str | None = None
        self.headers: dict[str, str] = {}
        self.body = bytearray()
        self.closed = False

    def putrequest(
        self,
        method: str,
        path: str,
        *,
        skip_host: bool,
        skip_accept_encoding: bool,
    ) -> None:
        self.method = method
        self.path = path
        if not skip_host or not skip_accept_encoding:
            raise AssertionError("The signed request must control Host and encoding headers")

    def putheader(self, name: str, value: str) -> None:
        if name in self.headers:
            raise AssertionError(f"duplicate header: {name}")
        self.headers[name] = value

    def endheaders(self) -> None:
        return None

    def send(self, body: bytes) -> None:
        self.body.extend(body)

    def getresponse(self) -> _FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


class _RecordingS3Factory:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = list(responses)
        self.connections: list[_RecordingS3Connection] = []

    def __call__(self, host: str, *, timeout: int) -> _RecordingS3Connection:
        if not self.responses:
            raise AssertionError("unexpected authenticated request")
        connection = _RecordingS3Connection(host, timeout, self.responses.pop(0))
        self.connections.append(connection)
        return connection


class _FakePublicEndpoint:
    def __init__(
        self,
        store: _FakeStore,
        base_url: str,
        *,
        corrupt_cache: bool = False,
        omit_cors: bool = False,
    ) -> None:
        self.store = store
        self.base_url = base_url.rstrip("/") + "/"
        self.corrupt_cache = corrupt_cache
        self.omit_cors = omit_cors
        self.requests: list[dict[str, str | None]] = []

    def __call__(self, request: object, timeout: int = 0) -> _FakeResponse:
        del timeout
        url = request.full_url  # type: ignore[attr-defined]
        if not url.startswith(self.base_url):
            raise AssertionError(f"Unexpected public URL: {url}")
        key = urllib.parse.unquote(url[len(self.base_url) :])
        value = self.store.objects[key]
        method = request.get_method()  # type: ignore[attr-defined]
        byte_range = request.get_header("Range")  # type: ignore[attr-defined]
        origin = request.get_header("Origin")  # type: ignore[attr-defined]
        self.requests.append(
            {
                "key": key,
                "method": method,
                "origin": origin,
                "range": byte_range,
            }
        )
        body = value.body
        status = 200
        content_range = None
        if byte_range is not None:
            if method != "GET":
                raise AssertionError("Only GET range requests are expected")
            unit, bounds = byte_range.split("=", 1)
            if unit != "bytes":
                raise AssertionError(byte_range)
            start_text, end_text = bounds.split("-", 1)
            start, end = int(start_text), int(end_text)
            body = body[start : end + 1]
            status = 206
            content_range = f"bytes {start}-{end}/{len(value.body)}"
        content_length = len(body)
        if method == "HEAD":
            body = b""
            content_length = len(value.body)
        headers = {
            "Accept-Ranges": "bytes",
            "Access-Control-Expose-Headers": (
                "Accept-Ranges, Content-Length, Content-Range, Content-Type, ETag"
            ),
            "Cache-Control": "no-store" if self.corrupt_cache else value.cache_control,
            "Content-Disposition": value.content_disposition,
            "Content-Length": str(content_length),
            "Content-Type": value.content_type,
            "ETag": '"fake-etag"',
        }
        if not self.omit_cors:
            headers["Access-Control-Allow-Origin"] = "*"
        if content_range is not None:
            headers["Content-Range"] = content_range
        return _FakeResponse(body, status=status, headers=headers)


class ImmutableR2PublicationTests(unittest.TestCase):
    public_base = "https://public.example.invalid"
    origin = "https://github.com"

    def _source(self, root: Path, payload: bytes) -> Path:
        path = root / "artifact.bin"
        path.write_bytes(payload)
        return path

    def _key(self, payload: bytes) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        return f"evidence/matched/{digest}.tar"

    def _verifier(
        self,
        store: _FakeStore,
        **endpoint_options: bool,
    ) -> tuple[PublicVerifier, _FakePublicEndpoint]:
        endpoint = _FakePublicEndpoint(
            store,
            self.public_base,
            **endpoint_options,
        )
        verifier = PublicVerifier(
            self.public_base,
            origin=self.origin,
            opener=endpoint,
            attempts=1,
            sleeper=lambda _: None,
        )
        return verifier, endpoint

    def test_single_object_upload_skip_and_deterministic_receipt(self) -> None:
        payload = b"immutable-publication-payload\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, payload)
            receipt_path = root / "receipt.json"
            store = _FakeStore()
            verifier, endpoint = self._verifier(store)

            plan = build_publication_plan(
                source=source,
                artifact_key=self._key(payload),
                staging_root=root / "stage-one",
            )
            first = publish_plan(
                plan=plan,
                store=store,
                verifier=verifier,
                receipt_path=receipt_path,
            )
            first_receipt = receipt_path.read_bytes()
            self.assertEqual(first["uploaded"], 1)
            self.assertEqual(first["skipped"], 0)
            self.assertEqual(store.puts, [plan.objects[0].key])

            second_plan = build_publication_plan(
                source=source,
                artifact_key=self._key(payload),
                staging_root=root / "stage-two",
            )
            second = publish_plan(
                plan=second_plan,
                store=store,
                verifier=verifier,
                receipt_path=receipt_path,
            )
            self.assertEqual(second["uploaded"], 0)
            self.assertEqual(second["skipped"], 1)
            self.assertEqual(receipt_path.read_bytes(), first_receipt)

            decoded = json.loads(first_receipt)
            self.assertNotIn("created_at", decoded)
            self.assertNotIn("uploaded_at", decoded)
            self.assertNotIn(str(root), first_receipt.decode())
            self.assertEqual(decoded["verification"]["authenticated_round_trip_sha256"], "pass")
            self.assertEqual(
                decoded["publisher"]["atomic_precondition"],
                "If-None-Match: *",
            )
            self.assertTrue(any(item["range"] for item in endpoint.requests))
            self.assertTrue(any(item["method"] == "HEAD" for item in endpoint.requests))
            self.assertTrue(all(item["origin"] == self.origin for item in endpoint.requests))

    def test_full_artifact_sha_is_required_in_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, b"content")
            with self.assertRaisesRegex(PublicationError, "complete lowercase SHA-256"):
                build_publication_plan(
                    source=source,
                    artifact_key="evidence/matched/not-the-digest.tar",
                    staging_root=root / "stage",
                )

    def test_derived_multipart_keys_must_fit_r2_key_limit(self) -> None:
        payload = b"multipart"
        digest = hashlib.sha256(payload).hexdigest()
        artifact_key = "a" * (1024 - len(digest)) + digest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, payload)
            with self.assertRaisesRegex(PublicationError, "at most 1024"):
                build_publication_plan(
                    source=source,
                    artifact_key=artifact_key,
                    staging_root=root / "stage",
                    max_single_bytes=1,
                    part_bytes=4,
                )

    def test_symlink_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, b"content")
            link = root / "artifact-link.bin"
            link.symlink_to(source)
            with self.assertRaisesRegex(PublicationError, "non-symlink"):
                build_publication_plan(
                    source=link,
                    artifact_key=self._key(b"content"),
                    staging_root=root / "stage",
                )

    def test_preflight_collision_aborts_before_any_upload(self) -> None:
        payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_publication_plan(
                source=self._source(root, payload),
                artifact_key=self._key(payload),
                staging_root=root / "stage",
                max_single_bytes=10,
                part_bytes=7,
            )
            store = _FakeStore()
            store.seed(plan.objects[-1].key, b"unequal-index", plan.objects[-1])
            verifier, _ = self._verifier(store)
            with self.assertRaises(RemoteCollisionError):
                publish_plan(
                    plan=plan,
                    store=store,
                    verifier=verifier,
                    receipt_path=root / "receipt.json",
                )
            self.assertEqual(store.puts, [])
            self.assertFalse((root / "receipt.json").exists())

    def test_concurrent_unequal_create_is_never_overwritten(self) -> None:
        payload = b"expected-content-addressed-bytes"
        concurrent = b"different-concurrent-writer-bytes"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_publication_plan(
                source=self._source(root, payload),
                artifact_key=self._key(payload),
                staging_root=root / "stage",
            )
            store = _FakeStore()
            store.race_body = concurrent
            verifier, _ = self._verifier(store)
            receipt = root / "receipt.json"

            with self.assertRaisesRegex(RemoteCollisionError, "concurrent"):
                publish_plan(
                    plan=plan,
                    store=store,
                    verifier=verifier,
                    receipt_path=receipt,
                )

            self.assertEqual(store.objects[plan.objects[0].key].body, concurrent)
            self.assertEqual(store.puts, [])
            self.assertFalse(receipt.exists())

    def test_multipart_keys_index_and_public_stream_reconstruction(self) -> None:
        payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, payload)
            plan = build_publication_plan(
                source=source,
                artifact_key=self._key(payload),
                staging_root=root / "stage-one",
                max_single_bytes=10,
                part_bytes=7,
            )
            duplicate_plan = build_publication_plan(
                source=source,
                artifact_key=self._key(payload),
                staging_root=root / "stage-two",
                max_single_bytes=10,
                part_bytes=7,
            )
            self.assertEqual(plan.mode, "deterministic-parts")
            self.assertEqual(
                plan.index_object.path.read_bytes(),  # type: ignore[union-attr]
                duplicate_plan.index_object.path.read_bytes(),  # type: ignore[union-attr]
            )
            self.assertEqual(
                [item.key for item in plan.objects],
                [item.key for item in duplicate_plan.objects],
            )
            for spec in plan.objects:
                self.assertIn(plan.artifact_identity.sha256, spec.key)
            for spec in plan.payload_objects:
                self.assertIn(spec.identity.sha256, spec.key)
            self.assertIn(
                plan.index_object.identity.sha256,  # type: ignore[union-attr]
                plan.index_object.key,  # type: ignore[union-attr]
            )

            store = _FakeStore()
            verifier, endpoint = self._verifier(store)
            result = publish_plan(
                plan=plan,
                store=store,
                verifier=verifier,
                receipt_path=root / "receipt.json",
            )
            receipt = result["receipt"]
            self.assertEqual(result["uploaded"], len(plan.objects))
            self.assertEqual(
                receipt["verification"]["multipart_stream_reconstruction"],
                "pass",
            )
            full_part_gets = [
                item
                for item in endpoint.requests
                if item["key"] in {part.key for part in plan.payload_objects}
                and item["range"] is None
            ]
            self.assertGreaterEqual(len(full_part_gets), 2 * len(plan.payload_objects))

    def test_public_metadata_or_cors_failure_writes_no_receipt(self) -> None:
        payload = b"metadata-must-fail-closed"
        for option in ("corrupt_cache", "omit_cors"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan = build_publication_plan(
                    source=self._source(root, payload),
                    artifact_key=self._key(payload),
                    staging_root=root / "stage",
                )
                store = _FakeStore()
                verifier, _ = self._verifier(store, **{option: True})
                receipt = root / "receipt.json"
                with self.assertRaises(PublicationError):
                    publish_plan(
                        plan=plan,
                        store=store,
                        verifier=verifier,
                        receipt_path=receipt,
                    )
                self.assertFalse(receipt.exists())

    def test_final_authenticated_read_detects_post_public_origin_change(self) -> None:
        class LateMutationStore(_FakeStore):
            def get_identity(self, key: str) -> Identity | None:
                if len(self.probes) == 3 and key in self.objects:
                    current = self.objects[key]
                    self.objects[key] = _RemoteObject(
                        body=b"changed-after-public-verification",
                        content_type=current.content_type,
                        content_disposition=current.content_disposition,
                        cache_control=current.cache_control,
                    )
                return super().get_identity(key)

        payload = b"origin-must-still-match"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_publication_plan(
                source=self._source(root, payload),
                artifact_key=self._key(payload),
                staging_root=root / "stage",
            )
            store = LateMutationStore()
            verifier, _ = self._verifier(store)
            receipt = root / "receipt.json"
            with self.assertRaisesRegex(PublicationError, "Final authenticated"):
                publish_plan(
                    plan=plan,
                    store=store,
                    verifier=verifier,
                    receipt_path=receipt,
                )
            self.assertFalse(receipt.exists())

    def test_s3_put_is_signed_and_conditionally_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "object"
            path.write_bytes(b"x")
            spec = ObjectSpec(
                role="artifact",
                key="evidence/" + hashlib.sha256(b"x").hexdigest(),
                path=path,
                identity=Identity(1, hashlib.sha256(b"x").hexdigest()),
                content_type="application/octet-stream",
                content_disposition='attachment; filename="artifact.bin"',
            )
            response = _FakeResponse(b"", status=200, headers={})
            factory = _RecordingS3Factory([response])
            store = ConditionalR2Store(
                "test-bucket",
                R2Credentials(
                    account_id="a" * 32,
                    access_key_id="test-access-key",
                    secret_access_key="test-secret-key",
                ),
                connection_factory=factory,
                clock=lambda: dt.datetime(2026, 7, 22, tzinfo=dt.timezone.utc),
            )
            self.assertTrue(store.put_if_absent(spec))
            self.assertEqual(len(factory.connections), 1)
            request = factory.connections[0]
            self.assertEqual(request.method, "PUT")
            self.assertEqual(request.path, f"/test-bucket/{spec.key}")
            self.assertEqual(bytes(request.body), b"x")
            self.assertEqual(request.headers["if-none-match"], "*")
            self.assertEqual(request.headers["content-type"], spec.content_type)
            self.assertEqual(
                request.headers["content-disposition"],
                spec.content_disposition,
            )
            self.assertEqual(request.headers["cache-control"], CACHE_CONTROL)
            self.assertEqual(request.headers["x-amz-content-sha256"], spec.identity.sha256)
            self.assertIn("if-none-match", request.headers["authorization"])
            self.assertNotIn("test-secret-key", request.headers["authorization"])
            self.assertTrue(
                request.headers["authorization"].endswith(
                    "Signature="
                    "9e2aa40d554e98967275f04a9e0355edde61a282f7e273bf9de77c6a53af1081"
                )
            )

    def test_s3_precondition_failure_compares_and_preserves_concurrent_bytes(self) -> None:
        expected = b"expected"
        concurrent = b"unequal"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "object"
            path.write_bytes(expected)
            spec = ObjectSpec(
                role="artifact",
                key="evidence/" + hashlib.sha256(expected).hexdigest(),
                path=path,
                identity=Identity(len(expected), hashlib.sha256(expected).hexdigest()),
                content_type="application/octet-stream",
                content_disposition='attachment; filename="artifact.bin"',
            )
            factory = _RecordingS3Factory(
                [
                    _FakeResponse(b"", status=412, headers={}),
                    _FakeResponse(concurrent, status=200, headers={}),
                ]
            )
            store = ConditionalR2Store(
                "test-bucket",
                R2Credentials(
                    account_id="b" * 32,
                    access_key_id="test-access-key",
                    secret_access_key="test-secret-key",
                ),
                connection_factory=factory,
                clock=lambda: dt.datetime(2026, 7, 22, tzinfo=dt.timezone.utc),
            )
            with self.assertRaisesRegex(RemoteCollisionError, "unequal bytes"):
                store.put_if_absent(spec)
            self.assertEqual(factory.connections[0].headers["if-none-match"], "*")
            self.assertEqual(factory.connections[1].method, "GET")

    def test_authenticated_s3_get_distinguishes_missing_from_auth_failure(self) -> None:
        credentials = R2Credentials(
            account_id="d" * 32,
            access_key_id="test-access-key",
            secret_access_key="test-secret-key",
        )
        missing_factory = _RecordingS3Factory(
            [_FakeResponse(b"missing", status=404, headers={})]
        )
        missing_store = ConditionalR2Store(
            "test-bucket",
            credentials,
            connection_factory=missing_factory,
            clock=lambda: dt.datetime(2026, 7, 22, tzinfo=dt.timezone.utc),
        )
        self.assertIsNone(missing_store.get_identity("safe/missing"))

        denied_factory = _RecordingS3Factory(
            [_FakeResponse(b"denied", status=403, headers={})]
        )
        denied_store = ConditionalR2Store(
            "test-bucket",
            credentials,
            connection_factory=denied_factory,
            clock=lambda: dt.datetime(2026, 7, 22, tzinfo=dt.timezone.utc),
        )
        with self.assertRaisesRegex(PublicationError, "credentials"):
            denied_store.get_identity("safe/unknown")

    def test_ambient_credentials_are_required_and_secrets_have_safe_repr(self) -> None:
        required = {
            "CLOUDFLARE_ACCOUNT_ID": "c" * 32,
            "AWS_ACCESS_KEY_ID": "access-id",
            "AWS_SECRET_ACCESS_KEY": "do-not-print-this-secret",
        }
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(PublicationError, "AWS_SECRET_ACCESS_KEY"):
                load_ambient_r2_credentials()
        with mock.patch.dict("os.environ", required, clear=True):
            credentials = load_ambient_r2_credentials()
        rendered = repr(credentials)
        self.assertNotIn("access-id", rendered)
        self.assertNotIn("do-not-print-this-secret", rendered)


if __name__ == "__main__":
    unittest.main()
