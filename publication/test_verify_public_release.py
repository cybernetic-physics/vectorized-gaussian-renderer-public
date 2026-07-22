from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

import publication.release_envelope as release_envelope
import publication.verify_public_release as verifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def identity(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class _Response(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        status: int,
        headers: dict[str, str],
        url: str,
    ) -> None:
        super().__init__(body)
        self.status = status
        self.headers = headers
        self._url = url

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _R2Endpoint:
    def __init__(
        self,
        *,
        url: str,
        body: bytes,
        content_type: str,
        content_disposition: str,
        cache_control: str,
    ) -> None:
        self.url = url
        self.body = body
        self.content_type = content_type
        self.content_disposition = content_disposition
        self.cache_control = cache_control
        self.omit_cors = False
        self.corrupt_full = False
        self.redirect = False
        self.requests: list[dict[str, object]] = []

    def __call__(self, request: object, timeout: int = 0) -> _Response:
        del timeout
        url = request.full_url  # type: ignore[attr-defined]
        method = request.get_method()  # type: ignore[attr-defined]
        byte_range = request.get_header("Range")  # type: ignore[attr-defined]
        origin = request.get_header("Origin")  # type: ignore[attr-defined]
        authorization = request.get_header("Authorization")  # type: ignore[attr-defined]
        self.requests.append(
            {
                "authorization": authorization,
                "method": method,
                "origin": origin,
                "range": byte_range,
                "url": url,
            }
        )
        if url != self.url:
            raise AssertionError(f"unexpected URL: {url}")
        body = self.body
        status = 200
        content_range = None
        if byte_range is not None:
            unit, bounds = byte_range.split("=", 1)
            if unit != "bytes":
                raise AssertionError(byte_range)
            start_text, end_text = bounds.split("-", 1)
            start, end = int(start_text), int(end_text)
            body = body[start : end + 1]
            status = 206
            content_range = f"bytes {start}-{end}/{len(self.body)}"
        elif method == "GET" and self.corrupt_full:
            body = bytes([body[0] ^ 1]) + body[1:]
        content_length = len(body)
        if method == "HEAD":
            body = b""
            content_length = len(self.body)
        headers = {
            "Accept-Ranges": "bytes",
            "Access-Control-Expose-Headers": (
                "Accept-Ranges, Content-Length, Content-Range, Content-Type, ETag"
            ),
            "Cache-Control": self.cache_control,
            "Content-Disposition": self.content_disposition,
            "Content-Length": str(content_length),
            "Content-Type": self.content_type,
            "ETag": '"test-etag"',
        }
        if not self.omit_cors:
            headers["Access-Control-Allow-Origin"] = "*"
        if content_range is not None:
            headers["Content-Range"] = content_range
        response_url = self.url + "/redirected" if self.redirect else self.url
        return _Response(body, status=status, headers=headers, url=response_url)


class _GitHubEndpoint:
    def __init__(self, responses: dict[str, tuple[bytes, str]]) -> None:
        self.responses = responses
        self.redirect_url: str | None = None
        self.second_responses: dict[str, tuple[bytes, str]] = {}
        self.call_counts: dict[str, int] = {}
        self.requests: list[dict[str, object]] = []

    def __call__(self, request: object, timeout: int = 0) -> _Response:
        del timeout
        url = request.full_url  # type: ignore[attr-defined]
        self.requests.append(
            {
                "accept": request.get_header("Accept"),  # type: ignore[attr-defined]
                "authorization": request.get_header("Authorization"),  # type: ignore[attr-defined]
                "method": request.get_method(),  # type: ignore[attr-defined]
                "url": url,
            }
        )
        self.call_counts[url] = self.call_counts.get(url, 0) + 1
        selected = (
            self.second_responses[url]
            if self.call_counts[url] > 1 and url in self.second_responses
            else self.responses[url]
        )
        try:
            body, content_type = selected
        except KeyError as error:
            raise AssertionError(f"unexpected URL: {url}") from error
        return _Response(
            body,
            status=200,
            headers={
                "Content-Length": str(len(body)),
                "Content-Type": content_type,
            },
            url=self.redirect_url if self.redirect_url == url else url,
        )


class PublicReleaseVerificationTest(unittest.TestCase):
    def temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(
            prefix=".public-release-verification-test-",
            dir=PROJECT_ROOT,
        )

    def make_r2_fixture(self, root: Path):
        payload = b"independently verified evidence archive\n"
        archive = root / "evidence.tar"
        archive.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        key = f"evidence/gcp-l4/{digest}.tar"
        base_url = "https://public.example.test"
        url = f"{base_url}/{key}"
        disposition = 'attachment; filename="evidence.tar"'
        receipt = {
            "artifact": {
                "bytes": len(payload),
                "content_disposition": disposition,
                "content_type": "application/x-tar",
                "key": key,
                "sha256": digest,
            },
            "bucket": "renderer-publication",
            "cors_origin": "https://github.com",
            "objects": [
                {
                    "bytes": len(payload),
                    "cache_control": "public, max-age=31536000, immutable",
                    "content_disposition": disposition,
                    "content_type": "application/x-tar",
                    "key": key,
                    "role": "artifact",
                    "sha256": digest,
                    "url": url,
                }
            ],
            "public_base_url": base_url,
            "publisher": {
                "api": "r2-s3-compatible",
                "atomic_precondition": "If-None-Match: *",
                "signature": "AWS Signature Version 4",
            },
            "schema_version": "vectorized-gaussian-r2-immutable-publication/v2",
            "storage": {
                "index_key": None,
                "mode": "single-object",
                "part_bytes": None,
            },
            "verification": {
                "anonymous_full_get_sha256": "pass",
                "anonymous_range_cors_metadata": "pass",
                "authenticated_post_public_sha256": "pass",
                "authenticated_round_trip_sha256": "pass",
                "multipart_stream_reconstruction": "not-applicable",
            },
        }
        receipt_path = root / "receipt.json"
        receipt_path.write_bytes(canonical(receipt))
        endpoint = _R2Endpoint(
            url=url,
            body=payload,
            content_type="application/x-tar",
            content_disposition=disposition,
            cache_control="public, max-age=31536000, immutable",
        )
        return archive, receipt_path, receipt, endpoint

    def test_generates_exact_r2_record_from_anonymous_bytes(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            archive, receipt_path, receipt, endpoint = self.make_r2_fixture(root)
            record = verifier.verify_r2_publication(
                receipt_path=receipt_path,
                archive_path=archive,
                opener=endpoint,
                attempts=1,
            )
            self.assertEqual(
                set(record),
                {"artifact", "checks", "objects", "pass", "receipt", "schema_version"},
            )
            self.assertTrue(record["pass"])
            self.assertEqual(record["schema_version"], "publication-r2-public-verification-v1")
            self.assertEqual(record["checks"], receipt["verification"])
            self.assertEqual(record["objects"][0]["full_get_status"], 200)
            self.assertEqual(record["objects"][0]["range_get_status"], 206)
            self.assertTrue(all(item["authorization"] is None for item in endpoint.requests))
            self.assertEqual(
                [item["method"] for item in endpoint.requests], ["HEAD", "GET", "GET"]
            )

            raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            archive_identity = identity(archive.read_bytes())
            normalized_receipt = release_envelope._validate_r2_receipt(
                raw_receipt,
                archive=release_envelope.ResolvedArtifact(
                    role="bundle_archive",
                    path=archive,
                    bytes=int(archive_identity["bytes"]),
                    sha256=str(archive_identity["sha256"]),
                ),
            )
            receipt_identity = identity(receipt_path.read_bytes())
            release_envelope._validate_r2_public_verification(
                record,
                receipt_artifact=release_envelope.ResolvedArtifact(
                    role="r2_receipt",
                    path=receipt_path,
                    bytes=int(receipt_identity["bytes"]),
                    sha256=str(receipt_identity["sha256"]),
                ),
                receipt=normalized_receipt,
            )

            output = root / "r2-public.json"
            first = verifier._write_once(output, record)
            second = verifier._write_once(output, record)
            self.assertEqual(first, second)
            self.assertEqual(output.read_bytes(), canonical(record))

    def test_r2_rejects_missing_cors_and_writes_no_record(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            archive, receipt_path, _, endpoint = self.make_r2_fixture(root)
            endpoint.omit_cors = True
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "CORS origin"
            ):
                verifier.verify_r2_publication(
                    receipt_path=receipt_path,
                    archive_path=archive,
                    opener=endpoint,
                    attempts=1,
                )

    def test_r2_rejects_corrupt_full_get_even_when_ranges_are_correct(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            archive, receipt_path, _, endpoint = self.make_r2_fixture(root)
            endpoint.corrupt_full = True
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "full GET identity"
            ):
                verifier.verify_r2_publication(
                    receipt_path=receipt_path,
                    archive_path=archive,
                    opener=endpoint,
                    attempts=1,
                )

    def test_r2_rejects_redirect(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            archive, receipt_path, _, endpoint = self.make_r2_fixture(root)
            endpoint.redirect = True
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "redirect is forbidden"
            ):
                verifier.verify_r2_publication(
                    receipt_path=receipt_path,
                    archive_path=archive,
                    opener=endpoint,
                    attempts=1,
                )

    def test_r2_rejects_noncanonical_receipt_before_network(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            archive, receipt_path, receipt, endpoint = self.make_r2_fixture(root)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "not canonically encoded"
            ):
                verifier.verify_r2_publication(
                    receipt_path=receipt_path,
                    archive_path=archive,
                    opener=endpoint,
                    attempts=1,
                )
            self.assertEqual(endpoint.requests, [])

    def make_github_fixture(self, root: Path):
        article_body = b"# Scoped L4 result\n"
        readme_body = b"# Renderer\n\nPublic evidence.\n"
        article = root / "post.md"
        readme = root / "README.md"
        article.write_bytes(article_body)
        readme.write_bytes(readme_body)
        benchmark_commit = "a" * 40
        final_commit = "b" * 40
        benchmark_tag_object = "c" * 40
        benchmark_tag = "benchmark-v1"
        release_tag = "release-v1"
        repository_url = "https://github.com/acme/vgr"
        api_base = "https://api.github.com/repos/acme/vgr"
        release_url = f"{repository_url}/releases/tag/{release_tag}"

        def encoded(value: object) -> bytes:
            return json.dumps(value, sort_keys=True).encode("utf-8")

        responses: dict[str, tuple[bytes, str]] = {
            api_base: (
                encoded(
                    {
                        "archived": False,
                        "disabled": False,
                        "full_name": "acme/vgr",
                        "html_url": repository_url,
                        "private": False,
                        "visibility": "public",
                    }
                ),
                "application/json; charset=utf-8",
            ),
            repository_url: (b"<html>repository</html>", "text/html; charset=utf-8"),
            f"{api_base}/releases/tags/{release_tag}": (
                encoded(
                    {
                        "draft": False,
                        "html_url": release_url,
                        "prerelease": False,
                        "published_at": "2026-07-22T00:00:00Z",
                        "tag_name": release_tag,
                    }
                ),
                "application/json; charset=utf-8",
            ),
            release_url: (b"<html>release</html>", "text/html; charset=utf-8"),
            f"{api_base}/git/ref/tags/{benchmark_tag}": (
                encoded(
                    {
                        "object": {"sha": benchmark_tag_object, "type": "tag"},
                        "ref": f"refs/tags/{benchmark_tag}",
                    }
                ),
                "application/json; charset=utf-8",
            ),
            f"{api_base}/git/tags/{benchmark_tag_object}": (
                encoded(
                    {
                        "object": {"sha": benchmark_commit, "type": "commit"},
                        "tag": benchmark_tag,
                    }
                ),
                "application/json; charset=utf-8",
            ),
            f"{api_base}/git/ref/tags/{release_tag}": (
                encoded(
                    {
                        "object": {"sha": final_commit, "type": "commit"},
                        "ref": f"refs/tags/{release_tag}",
                    }
                ),
                "application/json; charset=utf-8",
            ),
            f"{api_base}/git/commits/{final_commit}": (
                encoded(
                    {
                        "parents": [{"sha": "d" * 40}, {"sha": "e" * 40}],
                        "sha": final_commit,
                    }
                ),
                "application/json; charset=utf-8",
            ),
            (
                f"https://raw.githubusercontent.com/acme/vgr/{final_commit}/post.md"
            ): (article_body, "text/plain; charset=utf-8"),
            (
                f"https://raw.githubusercontent.com/acme/vgr/{final_commit}/README.md"
            ): (readme_body, "text/plain; charset=utf-8"),
        }
        endpoint = _GitHubEndpoint(responses)
        arguments = {
            "repository_url": repository_url,
            "benchmark_tag": benchmark_tag,
            "benchmark_commit": benchmark_commit,
            "release_tag": release_tag,
            "final_merge_commit": final_commit,
            "article_path": article,
            "readme_path": readme,
            "opener": endpoint,
            "attempts": 1,
        }
        return arguments, endpoint, responses

    def test_generates_exact_github_record_from_anonymous_state(self) -> None:
        with self.temporary_directory() as temporary:
            arguments, endpoint, _ = self.make_github_fixture(Path(temporary))
            record = verifier.verify_github_publication(**arguments)
            self.assertEqual(
                set(record),
                {"content", "pass", "refs", "release", "repository", "schema_version"},
            )
            self.assertTrue(record["pass"])
            self.assertEqual(
                record["schema_version"], "publication-github-public-verification-v1"
            )
            self.assertEqual(
                record["refs"]["benchmark_tag"]["commit"], "a" * 40
            )
            self.assertEqual(record["refs"]["release_tag"]["commit"], "b" * 40)
            self.assertTrue(all(item["authorization"] is None for item in endpoint.requests))
            article = Path(arguments["article_path"])
            readme = Path(arguments["readme_path"])
            article_identity = identity(article.read_bytes())
            readme_identity = identity(readme.read_bytes())
            release_envelope._validate_github_public_verification(
                record,
                repository={
                    "benchmark_commit": str(arguments["benchmark_commit"]),
                    "benchmark_tag": str(arguments["benchmark_tag"]),
                    "final_merge_commit": str(arguments["final_merge_commit"]),
                    "release_tag": str(arguments["release_tag"]),
                    "release_url": (
                        str(arguments["repository_url"])
                        + "/releases/tag/"
                        + str(arguments["release_tag"])
                    ),
                    "url": str(arguments["repository_url"]),
                },
                article=release_envelope.ResolvedArtifact(
                    role="article",
                    path=article,
                    bytes=int(article_identity["bytes"]),
                    sha256=str(article_identity["sha256"]),
                ),
                readme=release_envelope.ResolvedArtifact(
                    role="readme",
                    path=readme,
                    bytes=int(readme_identity["bytes"]),
                    sha256=str(readme_identity["sha256"]),
                ),
            )

    def test_github_rejects_private_repository(self) -> None:
        with self.temporary_directory() as temporary:
            arguments, endpoint, responses = self.make_github_fixture(Path(temporary))
            api_base = "https://api.github.com/repos/acme/vgr"
            payload = json.loads(responses[api_base][0])
            payload["private"] = True
            payload["visibility"] = "private"
            responses[api_base] = (json.dumps(payload).encode(), responses[api_base][1])
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "active public repository"
            ):
                verifier.verify_github_publication(**arguments)

    def test_github_rejects_wrong_release_tag_target(self) -> None:
        with self.temporary_directory() as temporary:
            arguments, _, responses = self.make_github_fixture(Path(temporary))
            url = "https://api.github.com/repos/acme/vgr/git/ref/tags/release-v1"
            payload = json.loads(responses[url][0])
            payload["object"]["sha"] = "f" * 40
            responses[url] = (json.dumps(payload).encode(), responses[url][1])
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "release tag resolves"
            ):
                verifier.verify_github_publication(**arguments)

    def test_github_rejects_tag_changed_during_closing_read(self) -> None:
        with self.temporary_directory() as temporary:
            arguments, endpoint, responses = self.make_github_fixture(Path(temporary))
            url = "https://api.github.com/repos/acme/vgr/git/ref/tags/release-v1"
            payload = json.loads(responses[url][0])
            payload["object"]["sha"] = "f" * 40
            endpoint.second_responses[url] = (
                json.dumps(payload).encode(),
                responses[url][1],
            )
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "changed during"
            ):
                verifier.verify_github_publication(**arguments)

    def test_github_rejects_raw_article_mismatch(self) -> None:
        with self.temporary_directory() as temporary:
            arguments, _, responses = self.make_github_fixture(Path(temporary))
            url = (
                "https://raw.githubusercontent.com/acme/vgr/"
                + "b" * 40
                + "/post.md"
            )
            responses[url] = (b"# Altered public article\n", "text/plain")
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "content length differs"
            ):
                verifier.verify_github_publication(**arguments)

    def test_github_rejects_prerelease(self) -> None:
        with self.temporary_directory() as temporary:
            arguments, _, responses = self.make_github_fixture(Path(temporary))
            url = "https://api.github.com/repos/acme/vgr/releases/tags/release-v1"
            payload = json.loads(responses[url][0])
            payload["prerelease"] = True
            responses[url] = (json.dumps(payload).encode(), responses[url][1])
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "published final release"
            ):
                verifier.verify_github_publication(**arguments)

    def test_github_rejects_local_article_changed_during_verification(self) -> None:
        with self.temporary_directory() as temporary:
            arguments, _, _ = self.make_github_fixture(Path(temporary))
            article = Path(arguments["article_path"])

            def mutate_article(**_: object) -> dict[str, object]:
                article.write_text("# Changed while checking\n", encoding="utf-8")
                return {"pass": True}

            with mock.patch.object(
                verifier,
                "_verify_github_once",
                side_effect=mutate_article,
            ):
                with self.assertRaisesRegex(
                    verifier.PublicReleaseVerificationError,
                    "Article changed during verification",
                ):
                    verifier.verify_github_publication(**arguments)

    def test_output_is_write_once(self) -> None:
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            output = root / "record.json"
            first = {"pass": True, "schema_version": "test-v1"}
            verifier._write_once(output, first)
            with self.assertRaisesRegex(
                verifier.PublicReleaseVerificationError, "refusing to overwrite"
            ):
                verifier._write_once(output, {"pass": False, "schema_version": "test-v1"})


if __name__ == "__main__":
    unittest.main()
