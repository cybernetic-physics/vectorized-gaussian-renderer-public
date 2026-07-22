from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from publication.write_bundle_inputs import build_roles, validate_manifest_url


class BundleInputTests(unittest.TestCase):
    def test_roles_are_exact_and_extras_are_supporting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            extra = root / "publication/article.md"
            extra.parent.mkdir()
            extra.write_text("article\n", encoding="utf-8")
            roles = build_roles(
                root,
                {"summary.json": frozenset({"canonical-summary"})},
            )
            self.assertEqual(roles["summary.json"], ["canonical-summary"])
            self.assertEqual(
                roles["publication/article.md"],
                ["supporting-artifact"],
            )

    def test_missing_required_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                build_roles(
                    Path(temporary),
                    {"summary.json": frozenset({"canonical-summary"})},
                )

    def test_manifest_url_requires_full_hash_and_no_query(self) -> None:
        digest = "b231602598b1eb039175dcb7edbd475167c2fc92011c5e975a4727c62b9f74b9"
        accepted = f"https://example.test/manifests/{digest}.json"
        self.assertEqual(validate_manifest_url(accepted), accepted)
        with self.assertRaises(ValueError):
            validate_manifest_url("https://example.test/manifest.json")
        with self.assertRaises(ValueError):
            validate_manifest_url(f"{accepted}?token=secret")


if __name__ == "__main__":
    unittest.main()
