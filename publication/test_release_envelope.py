from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import publication.release_envelope as release_envelope


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_FREEZE_SCRIPT = PROJECT_ROOT / "scripts" / "verify_publication_code_freeze.py"


def canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def artifact_record(path: Path) -> dict[str, object]:
    return {"path": str(path), **identity(path)}


class ReleaseFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.repo.mkdir()
        self._git("init", "-b", "main")
        self._git("config", "user.email", "release-test@example.test")
        self._git("config", "user.name", "Release Test")
        self._git("remote", "add", "origin", "https://github.com/acme/vgr.git")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "renderer.py").write_text("VERSION = 1\n", encoding="utf-8")
        (self.repo / "post.md").write_text("# Draft\n", encoding="utf-8")
        (self.repo / "README.md").write_text("# Renderer\n", encoding="utf-8")
        self._git("add", ".")
        self._git("commit", "-m", "benchmark source")
        self.benchmark_commit = self._git("rev-parse", "HEAD")
        self.benchmark_tag = "benchmark-v1"
        self._git("tag", self.benchmark_tag)

        self._git("checkout", "-b", "publication")
        (self.repo / "post.md").write_text(
            "# Matched rendering on one L4\n\nScoped evidence.\n", encoding="utf-8"
        )
        (self.repo / "README.md").write_text(
            "# Renderer\n\nPublication evidence.\n", encoding="utf-8"
        )
        self._git("add", "post.md", "README.md")
        self._git("commit", "-m", "publication content")
        self.publication_commit = self._git("rev-parse", "HEAD")

        self._git("checkout", "main")
        self._git("commit", "--allow-empty", "-m", "release lane")
        self._git("merge", "--no-ff", "publication", "-m", "merge publication")
        self.final_commit = self._git("rev-parse", "HEAD")
        self.release_tag = "release-v1"
        self._git("tag", self.release_tag)

        self.code_freeze_path = root / "code-freeze.json"
        subprocess.run(
            [
                sys.executable,
                str(CODE_FREEZE_SCRIPT),
                "--benchmark-ref",
                self.benchmark_tag,
                "--publication-ref",
                self.publication_commit,
                "--output",
                str(self.code_freeze_path),
            ],
            cwd=self.repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        code_identity = identity(self.code_freeze_path)
        article_identity = identity(self.repo / "post.md")

        self.bundle_root = root / "bundle"
        code_object_path = (
            self.bundle_root
            / "objects"
            / "sha256"
            / str(code_identity["sha256"])[:2]
            / str(code_identity["sha256"])
        )
        code_object_path.parent.mkdir(parents=True)
        code_object_path.write_bytes(self.code_freeze_path.read_bytes())
        article_object_path = (
            self.bundle_root
            / "objects"
            / "sha256"
            / str(article_identity["sha256"])[:2]
            / str(article_identity["sha256"])
        )
        article_object_path.parent.mkdir(parents=True, exist_ok=True)
        article_object_path.write_bytes((self.repo / "post.md").read_bytes())
        self.manifest = {
            "bundle_name": "gcp-l4-matched-v1",
            "inventory_id": "1" * 64,
            "manifest_id": "2" * 64,
            "publication_artifacts": [
                {
                    "bytes": article_identity["bytes"],
                    "logical_path": "publication/article.md",
                    "media_type": "text/markdown",
                    "object_path": str(
                        article_object_path.relative_to(self.bundle_root)
                    ),
                    "roles": ["supporting-artifact"],
                    "sha256": article_identity["sha256"],
                },
                {
                    "bytes": code_identity["bytes"],
                    "logical_path": "publication/code-freeze.json",
                    "media_type": "application/json",
                    "object_path": str(code_object_path.relative_to(self.bundle_root)),
                    "roles": ["code-freeze"],
                    "sha256": code_identity["sha256"],
                }
            ],
            "schema_version": "flashgs-matched-evidence-bundle-v1",
        }
        self.manifest_path = self.bundle_root / "manifest.json"
        self.manifest_path.write_bytes(canonical(self.manifest))

        self.archive_path = root / "gcp-l4-matched-v1.tar"
        self.archive_path.write_bytes(b"deterministic scientific archive\n")
        archive_identity = identity(self.archive_path)

        base_url = "https://public.example.test"
        logical_key = f"evidence/gcp-l4/{archive_identity['sha256']}.tar"
        object_url = f"{base_url}/{logical_key}"
        content_disposition = 'attachment; filename="gcp-l4-matched-v1.tar"'
        self.r2_receipt = {
            "artifact": {
                "bytes": archive_identity["bytes"],
                "content_disposition": content_disposition,
                "content_type": "application/x-tar",
                "key": logical_key,
                "sha256": archive_identity["sha256"],
            },
            "bucket": "renderer-publication",
            "cors_origin": "https://github.com",
            "objects": [
                {
                    "bytes": archive_identity["bytes"],
                    "cache_control": "public, max-age=31536000, immutable",
                    "content_disposition": content_disposition,
                    "content_type": "application/x-tar",
                    "key": logical_key,
                    "role": "artifact",
                    "sha256": archive_identity["sha256"],
                    "url": object_url,
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
        self.r2_receipt_path = root / "r2-receipt.json"
        self.r2_receipt_path.write_bytes(canonical(self.r2_receipt))

        normalized_r2_artifact = {
            "bytes": archive_identity["bytes"],
            "content_disposition": content_disposition,
            "content_type": "application/x-tar",
            "logical_key": logical_key,
            "public_entry_key": logical_key,
            "public_entry_url": object_url,
            "sha256": archive_identity["sha256"],
        }
        self.r2_public = {
            "artifact": normalized_r2_artifact,
            "checks": self.r2_receipt["verification"],
            "objects": [
                {
                    "bytes": archive_identity["bytes"],
                    "full_get_status": 200,
                    "key": logical_key,
                    "range_get_status": 206,
                    "sha256": archive_identity["sha256"],
                    "url": object_url,
                }
            ],
            "pass": True,
            "receipt": identity(self.r2_receipt_path),
            "schema_version": "publication-r2-public-verification-v1",
        }
        self.r2_public_path = root / "r2-public-verification.json"
        self.r2_public_path.write_bytes(canonical(self.r2_public))

        repository_url = "https://github.com/acme/vgr"
        release_url = f"{repository_url}/releases/tag/{self.release_tag}"
        readme_identity = identity(self.repo / "README.md")
        self.github_public = {
            "content": {
                "article": {
                    "anonymous_http_status": 200,
                    "bytes": article_identity["bytes"],
                    "commit": self.final_commit,
                    "git_path": "post.md",
                    "raw_url": (
                        f"https://raw.githubusercontent.com/acme/vgr/"
                        f"{self.final_commit}/post.md"
                    ),
                    "sha256": article_identity["sha256"],
                },
                "readme": {
                    "anonymous_http_status": 200,
                    "bytes": readme_identity["bytes"],
                    "commit": self.final_commit,
                    "git_path": "README.md",
                    "raw_url": (
                        f"https://raw.githubusercontent.com/acme/vgr/"
                        f"{self.final_commit}/README.md"
                    ),
                    "sha256": readme_identity["sha256"],
                },
            },
            "pass": True,
            "refs": {
                "benchmark_tag": {
                    "commit": self.benchmark_commit,
                    "name": self.benchmark_tag,
                },
                "release_tag": {
                    "commit": self.final_commit,
                    "name": self.release_tag,
                },
            },
            "release": {
                "anonymous_http_status": 200,
                "tag": self.release_tag,
                "target_commit": self.final_commit,
                "url": release_url,
            },
            "repository": {
                "anonymous_http_status": 200,
                "url": repository_url,
                "visibility": "public",
            },
            "schema_version": "publication-github-public-verification-v1",
        }
        self.github_public_path = root / "github-public-verification.json"
        self.github_public_path.write_bytes(canonical(self.github_public))

        self.spec = {
            "artifacts": {
                "article": artifact_record(self.repo / "post.md"),
                "bundle_archive": artifact_record(self.archive_path),
                "bundle_manifest": artifact_record(self.manifest_path),
                "code_freeze": artifact_record(self.code_freeze_path),
                "github_public_verification": artifact_record(self.github_public_path),
                "r2_public_verification": artifact_record(self.r2_public_path),
                "r2_receipt": artifact_record(self.r2_receipt_path),
                "readme": artifact_record(self.repo / "README.md"),
            },
            "repository": {
                "benchmark_commit": self.benchmark_commit,
                "benchmark_tag": self.benchmark_tag,
                "final_merge_commit": self.final_commit,
                "name": "vgr",
                "owner": "acme",
                "publication_content_commit": self.publication_commit,
                "release_tag": self.release_tag,
                "release_url": release_url,
                "url": repository_url,
                "visibility": "public",
            },
            "schema_version": "publication-external-release-envelope-spec-v1",
        }
        self.spec_path = root / "release-spec.json"
        self.output_path = root / "release-envelope.json"
        self.write_spec()

    def _git(self, *arguments: str) -> str:
        env = {
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-07-22T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-07-22T00:00:00+00:00",
        }
        completed = subprocess.run(
            ["git", *arguments],
            cwd=self.repo,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    def write_spec(self) -> None:
        self.spec_path.write_bytes(canonical(self.spec))

    def replace_json_artifact(self, role: str, path: Path, payload: object) -> None:
        path.write_bytes(canonical(payload))
        self.spec["artifacts"][role] = artifact_record(path)
        self.write_spec()

    def retarget_release_with_base_change(
        self,
        *,
        relative_path: str,
        payload: str,
    ) -> None:
        branch_slug = relative_path.lower().replace("/", "-").replace(".", "-")
        self._git(
            "checkout",
            "-b",
            f"release-with-{branch_slug}-change",
            self.benchmark_commit,
        )
        changed_path = self.repo / relative_path
        changed_path.parent.mkdir(parents=True, exist_ok=True)
        changed_path.write_text(payload, encoding="utf-8")
        self._git("add", relative_path)
        self._git("commit", "-m", f"unreviewed release change to {relative_path}")
        self._git(
            "merge",
            "--no-ff",
            self.publication_commit,
            "-m",
            "merge publication with changed release base",
        )
        attacked_final = self._git("rev-parse", "HEAD")
        self._git("tag", "-f", self.release_tag, attacked_final)
        self.final_commit = attacked_final
        self.spec["repository"]["final_merge_commit"] = attacked_final
        self.github_public["refs"]["release_tag"]["commit"] = attacked_final
        self.github_public["release"]["target_commit"] = attacked_final
        for git_path, record in (
            ("post.md", self.github_public["content"]["article"]),
            ("README.md", self.github_public["content"]["readme"]),
        ):
            record["commit"] = attacked_final
            record["raw_url"] = (
                f"https://raw.githubusercontent.com/acme/vgr/"
                f"{attacked_final}/{git_path}"
            )
        self.github_public_path.write_bytes(canonical(self.github_public))
        self.spec["artifacts"]["github_public_verification"] = artifact_record(
            self.github_public_path
        )
        self.write_spec()

    def strict_release_verification(self) -> dict[str, object]:
        archive_result = {
            "archive": str(self.archive_path),
            "bytes": self.archive_path.stat().st_size,
            "manifest_id": self.manifest["manifest_id"],
            "sha256": identity(self.archive_path)["sha256"],
        }
        verification = {
            "bytes": 10,
            "logical_path": "publication/verification.json",
            "sha256": "3" * 64,
        }
        summary = {
            "bytes": 11,
            "logical_path": "summary.json",
            "sha256": "4" * 64,
        }
        ablation = {
            "bytes": 12,
            "logical_path": "publication/evidence/custom-vectorization-ablation.json",
            "sha256": "5" * 64,
        }
        aggregate = {
            "schema_version": "publication-aggregate-verification-receipt-v1",
            "pass": True,
            "root_mode": "bundle",
            "verification": verification,
            "summary": summary,
            "ablation": ablation,
        }
        claim_ledger = {
            "schema_version": "publication-claim-ledger-verification-v1",
            "pass": True,
            "root_mode": "bundle",
            "verification": verification,
            "summary": summary,
            "ablation": ablation,
            "claim_count": 1,
        }
        validator = PROJECT_ROOT / "publication/verify_claim_ledger.py"
        return {
            "schema_version": "publication-release-verification-v1",
            "pass": True,
            "manifest_id": self.manifest["manifest_id"],
            "aggregate": aggregate,
            "archive": archive_result,
            "claim_ledger": claim_ledger,
            "claim_validator": {
                "logical_path": "publication/validator/verify_claim_ledger.py",
                **identity(validator),
            },
        }

    def run(self, *, strict_error: Exception | None = None) -> dict[str, object]:
        strict = self.strict_release_verification()
        verifier = mock.patch.object(
            release_envelope,
            "verify_publication_release",
            side_effect=strict_error,
            return_value=(copy.deepcopy(self.manifest), strict),
        )
        with verifier:
            return release_envelope.build_release_envelope(
                spec_path=self.spec_path,
                repo_root=self.repo,
                output_path=self.output_path,
            )


class ReleaseEnvelopeTest(unittest.TestCase):
    def make_fixture(self):
        temporary = tempfile.TemporaryDirectory(
            prefix=".release-envelope-test-", dir=PROJECT_ROOT
        )
        return temporary, ReleaseFixture(Path(temporary.name))

    def test_builds_deterministic_path_free_release_envelope(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            envelope = fixture.run()
            first = fixture.output_path.read_bytes()
            second_envelope = fixture.run()
            self.assertEqual(first, fixture.output_path.read_bytes())
            self.assertEqual(envelope, second_envelope)
            self.assertTrue(envelope["pass"])
            self.assertEqual(
                envelope["repository"]["benchmark"]["commit"],
                fixture.benchmark_commit,
            )
            self.assertEqual(
                envelope["repository"]["release"]["merge_commit"],
                fixture.final_commit,
            )
            self.assertEqual(
                envelope["bundle"]["archive"], identity(fixture.archive_path)
            )
            self.assertTrue(
                envelope["bundle"]["strict_release_verification"]["pass"]
            )
            self.assertTrue(envelope["git_proof"]["final_load_bearing_tree_equal"])
            self.assertTrue(envelope["git_proof"]["final_tree_equal"])
            self.assertEqual(envelope["github"]["visibility"], "public")
            decoded = first.decode("utf-8")
            self.assertNotIn(str(fixture.root), decoded)
            self.assertNotIn("created_at", decoded)
            self.assertNotRegex(decoded, r"/Users/[^/]+/")

    def test_rejects_unknown_spec_field(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            fixture.spec["unexpected"] = True
            fixture.write_spec()
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "unknown fields"):
                fixture.run()

    def test_rejects_artifact_hash_mismatch(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            fixture.spec["artifacts"]["article"]["sha256"] = "0" * 64
            fixture.write_spec()
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "content differs"):
                fixture.run()

    def test_rejects_symlink_artifact(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            link = fixture.root / "article-link.md"
            link.symlink_to(fixture.repo / "post.md")
            fixture.spec["artifacts"]["article"] = {
                "path": str(link),
                **identity(fixture.repo / "post.md"),
            }
            fixture.write_spec()
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "Symlinks"):
                fixture.run()

    def test_rejects_symlink_spec(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            link = fixture.root / "release-spec-link.json"
            link.symlink_to(fixture.spec_path.name)
            with mock.patch.object(
                release_envelope,
                "verify_publication_release",
                return_value=(
                    copy.deepcopy(fixture.manifest),
                    fixture.strict_release_verification(),
                ),
            ):
                with self.assertRaisesRegex(
                    release_envelope.ReleaseEnvelopeError, "traverses a symlink"
                ):
                    release_envelope.build_release_envelope(
                        spec_path=link,
                        repo_root=fixture.repo,
                        output_path=fixture.output_path,
                    )

    def test_rejects_failed_strict_release_verification(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            with self.assertRaisesRegex(
                release_envelope.ReleaseEnvelopeError,
                "Strict publication release verification failed",
            ):
                fixture.run(
                    strict_error=RuntimeError(
                        "Strict claim-ledger verification failed"
                    )
                )

    def test_rejects_final_merge_with_new_load_bearing_code(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            fixture.retarget_release_with_base_change(
                relative_path="src/renderer.py",
                payload="VERSION = 2\n",
            )
            with self.assertRaisesRegex(
                release_envelope.ReleaseEnvelopeError,
                "Final merge changed.*load-bearing tree",
            ):
                fixture.run()

    def test_rejects_final_merge_with_unrelated_tree_change(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            fixture.retarget_release_with_base_change(
                relative_path="UNRELATED.txt",
                payload="not part of the reviewed publication content\n",
            )
            with self.assertRaisesRegex(
                release_envelope.ReleaseEnvelopeError,
                "Final merge tree differs from.*publication content commit",
            ):
                fixture.run()

    def test_rejects_release_tag_that_does_not_resolve_to_final_merge(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            fixture.spec["repository"]["release_tag"] = fixture.benchmark_tag
            fixture.spec["repository"]["release_url"] = (
                "https://github.com/acme/vgr/releases/tag/" + fixture.benchmark_tag
            )
            fixture.write_spec()
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "Release tag"):
                fixture.run()

    def test_rejects_unbundled_code_freeze_record(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            changed = json.loads(fixture.code_freeze_path.read_text(encoding="utf-8"))
            changed["pass"] = False
            changed_path = fixture.root / "changed-code-freeze.json"
            changed_path.write_bytes(canonical(changed))
            fixture.spec["artifacts"]["code_freeze"] = artifact_record(changed_path)
            fixture.write_spec()
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "bundled record"):
                fixture.run()

    def test_rejects_bundle_article_that_differs_from_public_post(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            payload = b"# Different bundled article\n"
            digest = hashlib.sha256(payload).hexdigest()
            object_path = (
                fixture.bundle_root
                / "objects"
                / "sha256"
                / digest[:2]
                / digest
            )
            object_path.parent.mkdir(parents=True, exist_ok=True)
            object_path.write_bytes(payload)
            article = next(
                item
                for item in fixture.manifest["publication_artifacts"]
                if item["logical_path"] == "publication/article.md"
            )
            article.update(
                {
                    "bytes": len(payload),
                    "object_path": str(object_path.relative_to(fixture.bundle_root)),
                    "sha256": digest,
                }
            )
            fixture.manifest_path.write_bytes(canonical(fixture.manifest))
            fixture.spec["artifacts"]["bundle_manifest"] = artifact_record(
                fixture.manifest_path
            )
            fixture.write_spec()
            with self.assertRaisesRegex(
                release_envelope.ReleaseEnvelopeError,
                "Bundled publication article differs from supplied post.md",
            ):
                fixture.run()

    def test_rejects_bundle_without_exactly_one_publication_article(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            fixture.manifest["publication_artifacts"] = [
                item
                for item in fixture.manifest["publication_artifacts"]
                if item["logical_path"] != "publication/article.md"
            ]
            fixture.manifest_path.write_bytes(canonical(fixture.manifest))
            fixture.spec["artifacts"]["bundle_manifest"] = artifact_record(
                fixture.manifest_path
            )
            fixture.write_spec()
            with self.assertRaisesRegex(
                release_envelope.ReleaseEnvelopeError,
                "exactly one publication article",
            ):
                fixture.run()

    def test_rejects_unknown_r2_receipt_field(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            changed = copy.deepcopy(fixture.r2_receipt)
            changed["timestamp"] = "2026-07-22T00:00:00Z"
            fixture.replace_json_artifact(
                "r2_receipt", fixture.r2_receipt_path, changed
            )
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "unknown fields"):
                fixture.run()

    def test_rejects_r2_receipt_for_different_archive(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            changed = copy.deepcopy(fixture.r2_receipt)
            changed["artifact"]["sha256"] = "a" * 64
            changed["artifact"]["key"] = "evidence/gcp-l4/" + "a" * 64 + ".tar"
            fixture.replace_json_artifact(
                "r2_receipt", fixture.r2_receipt_path, changed
            )
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "archive"):
                fixture.run()

    def test_rejects_r2_public_record_with_host_path(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            changed = copy.deepcopy(fixture.r2_public)
            changed["artifact"]["public_entry_url"] = "/home/alice/private/object"
            fixture.replace_json_artifact(
                "r2_public_verification", fixture.r2_public_path, changed
            )
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "Host-user path"):
                fixture.run()

    def test_rejects_unknown_github_public_record_field(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            changed = copy.deepcopy(fixture.github_public)
            changed["repository"]["api_result"] = "public"
            fixture.replace_json_artifact(
                "github_public_verification", fixture.github_public_path, changed
            )
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "repository public evidence"):
                fixture.run()

    def test_rejects_non_public_repository_before_network_record(self) -> None:
        temporary, fixture = self.make_fixture()
        with temporary:
            fixture.spec["repository"]["visibility"] = "private"
            fixture.write_spec()
            with self.assertRaisesRegex(release_envelope.ReleaseEnvelopeError, "publicly visible"):
                fixture.run()


if __name__ == "__main__":
    unittest.main()
