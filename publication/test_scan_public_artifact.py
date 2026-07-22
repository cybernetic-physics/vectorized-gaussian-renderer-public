from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from publication.scan_public_artifact import (
    PublicArtifactPrivacyError,
    scan_public_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PublicArtifactPrivacyTest(unittest.TestCase):
    def temporary_root(self):
        return tempfile.TemporaryDirectory(
            prefix=".public-privacy-test-", dir=PROJECT_ROOT
        )

    def write_json(self, root: Path, value: object, name: str = "artifact.json") -> Path:
        path = root / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_benign_public_artifact_passes_and_workspace_example_is_not_personal(self) -> None:
        with self.temporary_root() as directory:
            path = self.write_json(
                Path(directory),
                {
                    "command_example": "/workspace/vectorized-gaussian-renderer",
                    "url": "https://example.test/artifact/abc",
                },
            )
            identity = scan_public_artifact(path)
            self.assertEqual(identity["bytes"], path.stat().st_size)
            self.assertEqual(len(identity["sha256"]), 64)

    def test_reuses_bundle_secret_scanner(self) -> None:
        with self.temporary_root() as directory:
            path = Path(directory) / "secret.txt"
            # This is a synthetic token-shaped value, not a credential.
            path.write_text("AKIA" + "A" * 16, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Secret material detected"):
                scan_public_artifact(path)

    def test_rejects_secret_assignments_embedded_in_binary_containers(self) -> None:
        with self.temporary_root() as directory:
            root = Path(directory)
            for name, key in (
                ("capture.nsys-rep", b"OPEN_BUTTON_TOKEN"),
                ("capture.sqlite", b"CONTAINER_API_KEY"),
            ):
                with self.subTest(name=name):
                    path = root / name
                    # Synthetic token-shaped bytes model a captured process
                    # environment; no live credential appears in this test.
                    path.write_bytes(
                        b"SQLite format 3\x00binary-prefix\x00"
                        + key
                        + b"="
                        + b"A" * 64
                        + b"\x00binary-suffix"
                    )
                    with self.assertRaisesRegex(
                        ValueError, "(?i)secret assignment"
                    ):
                        scan_public_artifact(path)

    def test_rejects_host_user_paths_embedded_in_binary_containers(self) -> None:
        with self.temporary_root() as directory:
            path = Path(directory) / "capture.nsys-rep"
            path.write_bytes(
                b"binary-prefix\x00/workspace/public-is-portable\x00"
                b"/home/alice/private/torch-extension.o\x00binary-suffix"
            )
            with self.assertRaisesRegex(
                PublicArtifactPrivacyError, "Host-user path"
            ):
                scan_public_artifact(path)

    def test_rejects_exact_host_user_home_in_binary_containers(self) -> None:
        with self.temporary_root() as directory:
            path = Path(directory) / "capture.nsys-rep"
            path.write_bytes(b"binary-prefix\x00/home/alice\x00binary-suffix")
            with self.assertRaisesRegex(
                PublicArtifactPrivacyError, "Host-user path"
            ):
                scan_public_artifact(path)

    def test_rejects_host_user_paths(self) -> None:
        with self.temporary_root() as directory:
            path = self.write_json(
                Path(directory), {"provenance": "/Users/alice/private/results.json"}
            )
            with self.assertRaisesRegex(PublicArtifactPrivacyError, "Host-user path"):
                scan_public_artifact(path)

    def test_rejects_exact_host_user_home(self) -> None:
        with self.temporary_root() as directory:
            path = self.write_json(Path(directory), {"cwd": "/home/alice"})
            with self.assertRaisesRegex(PublicArtifactPrivacyError, "Host-user path"):
                scan_public_artifact(path)

    def test_rejects_credential_bearing_urls(self) -> None:
        with self.temporary_root() as directory:
            path = self.write_json(
                Path(directory), {"url": "https://user:secret@example.test/object"}
            )
            with self.assertRaisesRegex(PublicArtifactPrivacyError, "Credential-bearing URL"):
                scan_public_artifact(path)

    def test_rejects_sensitive_signed_url_queries(self) -> None:
        with self.temporary_root() as directory:
            path = self.write_json(
                Path(directory), {"url": "https://example.test/x?X-Amz-Signature=deadbeef"}
            )
            with self.assertRaisesRegex(PublicArtifactPrivacyError, "Sensitive URL query"):
                scan_public_artifact(path)

    def test_timestamp_rejection_is_opt_in_for_deterministic_envelopes(self) -> None:
        with self.temporary_root() as directory:
            path = self.write_json(
                Path(directory), {"created_at": "2026-07-22T12:34:56+00:00"}
            )
            scan_public_artifact(path)
            with self.assertRaisesRegex(PublicArtifactPrivacyError, "Timestamp"):
                scan_public_artifact(path, reject_timestamps=True)

    def test_rejects_symlink_input(self) -> None:
        with self.temporary_root() as directory:
            root = Path(directory)
            target = self.write_json(root, {"pass": True}, "target.json")
            link = root / "link.json"
            link.symlink_to(target.name)
            with self.assertRaisesRegex(PublicArtifactPrivacyError, "Symlinks"):
                scan_public_artifact(link)


if __name__ == "__main__":
    unittest.main()
