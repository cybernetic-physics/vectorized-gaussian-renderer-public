#!/usr/bin/env python3
"""Mutation tests for the fail-closed release-verifier boundary."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import verify_release as verify_release_module
from evidence_bundle import artifact_object_path
from verify_release import (
    CLAIM_SENTINEL,
    CLAIM_VALIDATOR_LOGICAL_PATH,
    PUBLICATION_ROOT,
    file_sha256,
    parse_args,
    parse_claim_validator_stdout,
    pinned_claim_validator,
    require_matching_claim_artifacts,
    scan_public_bundle_objects,
    verify_publication_release,
)


class ReleaseVerificationTests(unittest.TestCase):
    def receipt(self) -> dict[str, object]:
        identity = lambda logical_path, digest: {  # noqa: E731 - compact immutable fixture
            "bytes": 1,
            "logical_path": logical_path,
            "sha256": digest * 64,
        }
        return {
            "schema_version": "publication-claim-ledger-verification-v1",
            "pass": True,
            "root_mode": "bundle",
            "verification": identity("publication/verification.json", "0"),
            "summary": identity("summary.json", "1"),
            "ablation": identity(
                "publication/evidence/custom-vectorization-ablation.json",
                "2",
            ),
        }

    def manifest(self) -> dict[str, object]:
        validator = PUBLICATION_ROOT / "verify_claim_ledger.py"
        return {
            "publication_artifacts": [
                {
                    "logical_path": CLAIM_VALIDATOR_LOGICAL_PATH,
                    "bytes": validator.stat().st_size,
                    "sha256": file_sha256(validator),
                }
            ]
        }

    def test_exact_structured_sentinel_passes(self) -> None:
        receipt = self.receipt()
        self.assertEqual(
            parse_claim_validator_stdout(CLAIM_SENTINEL + json.dumps(receipt, sort_keys=True) + "\n"),
            receipt,
        )

    def test_substring_spoof_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_claim_validator_stdout("noise PUBLICATION_CLAIM_LEDGER_VERIFIED totally fake\n")

    def test_extra_stdout_line_fails(self) -> None:
        encoded = CLAIM_SENTINEL + json.dumps(self.receipt()) + "\nextra\n"
        with self.assertRaises(RuntimeError):
            parse_claim_validator_stdout(encoded)

    def test_unstructured_or_nonpassing_receipt_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_claim_validator_stdout(CLAIM_SENTINEL + "totally fake\n")
        receipt = self.receipt()
        receipt["pass"] = False
        with self.assertRaises(RuntimeError):
            parse_claim_validator_stdout(CLAIM_SENTINEL + json.dumps(receipt) + "\n")

    def test_validator_is_mandatory_and_hash_pinned(self) -> None:
        with self.assertRaises(RuntimeError):
            pinned_claim_validator({"publication_artifacts": []})
        manifest = self.manifest()
        manifest["publication_artifacts"][0]["sha256"] = "f" * 64  # type: ignore[index]
        with self.assertRaises(RuntimeError):
            pinned_claim_validator(manifest)

    def test_pinned_validator_positive(self) -> None:
        validator, identity = pinned_claim_validator(self.manifest())
        self.assertEqual(validator, PUBLICATION_ROOT / "verify_claim_ledger.py")
        self.assertEqual(identity["logical_path"], CLAIM_VALIDATOR_LOGICAL_PATH)

    def test_all_claim_artifact_identities_must_match(self) -> None:
        claim = self.receipt()
        aggregate = {
            role: claim[role]
            for role in ("verification", "summary", "ablation")
        }
        require_matching_claim_artifacts(claim, aggregate)
        for role in ("verification", "summary", "ablation"):
            mismatch = json.loads(json.dumps(aggregate))
            mismatch[role]["sha256"] = "f" * 64
            with self.assertRaises(RuntimeError):
                require_matching_claim_artifacts(claim, mismatch)

    def test_public_bundle_scan_rejects_binary_secret_and_host_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="public-bundle-scan-") as raw:
            root = Path(raw)
            for index, payload in enumerate(
                (
                    b"binary\x00JUPYTER_TOKEN=" + b"A" * 64 + b"\x00",
                    b"binary\x00/home/alice/private/build.o\x00",
                )
            ):
                with self.subTest(index=index):
                    digest = hashlib.sha256(payload).hexdigest()
                    path = root / artifact_object_path(digest)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                    manifest = {
                        "objects": [
                            {
                                "bytes": len(payload),
                                "sha256": digest,
                            }
                        ]
                    }
                    with self.assertRaisesRegex(
                        RuntimeError, "failed privacy scanning"
                    ):
                        scan_public_bundle_objects(root, manifest)
                    path.unlink()

    def test_public_bundle_scan_accepts_portable_binary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="public-bundle-scan-") as raw:
            root = Path(raw)
            payload = b"binary\x00/workspace/public/build.o\x00"
            digest = hashlib.sha256(payload).hexdigest()
            path = root / artifact_object_path(digest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            manifest = {
                "objects": [{"bytes": len(payload), "sha256": digest}]
            }
            self.assertEqual(scan_public_bundle_objects(root, manifest), 1)

    def test_archive_is_required_and_override_removed(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--bundle-root", "/tmp/bundle"])
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--bundle-root",
                        "/tmp/bundle",
                        "--archive",
                        "/tmp/bundle.tar",
                        "--claim-ledger-validator",
                        "/tmp/fake.py",
                    ]
                )

    def test_complete_release_verifier_rejects_invalid_strict_claim_receipt(self) -> None:
        manifest = self.manifest()
        manifest["manifest_id"] = "3" * 64
        claim = self.receipt()
        aggregate = {
            "schema_version": "publication-aggregate-verification-receipt-v1",
            "pass": True,
            "root_mode": "bundle",
            **{
                role: claim[role]
                for role in ("verification", "summary", "ablation")
            },
        }
        invalid_claim = dict(claim)
        invalid_claim["pass"] = False
        completed = subprocess.CompletedProcess(
            args=["verify_claim_ledger.py"],
            returncode=0,
            stdout=CLAIM_SENTINEL + json.dumps(invalid_claim, sort_keys=True) + "\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory(prefix="strict-release-verifier-") as raw:
            bundle = Path(raw) / "bundle"
            bundle.mkdir()
            archive = Path(raw) / "bundle.tar"
            archive.write_bytes(b"archive")
            archive_receipt = {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "manifest_id": manifest["manifest_id"],
                "sha256": "4" * 64,
            }
            with mock.patch.object(
                verify_release_module,
                "verify_evidence_bundle",
                return_value=manifest,
            ), mock.patch.object(
                verify_release_module,
                "verify_aggregate_verification",
                return_value=aggregate,
            ), mock.patch.object(
                verify_release_module,
                "verify_deterministic_archive",
                return_value=archive_receipt,
            ), mock.patch.object(
                verify_release_module.subprocess,
                "run",
                return_value=completed,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "malformed or non-passing receipt",
                ):
                    verify_publication_release(
                        bundle_root=bundle,
                        archive_path=archive,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
