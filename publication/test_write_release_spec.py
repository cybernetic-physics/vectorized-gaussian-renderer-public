from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from publication import write_release_spec as writer
from publication.release_envelope import _resolve_artifact, _validate_repository_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WriteReleaseSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".release-spec-test-", dir=PROJECT_ROOT
        )
        self.root = Path(self.temporary.name)
        self.artifacts: dict[str, Path] = {}
        for index, role in enumerate(writer.ARTIFACT_ROLES, start=1):
            path = self.root / f"{index:02d}-{role}.bin"
            path.write_bytes((f"{role}\n" * index).encode("utf-8"))
            self.artifacts[role] = path
        self.output = self.root / "release-spec.json"
        self.arguments = {
            "artifact_paths": self.artifacts,
            "owner": "cybernetic-physics",
            "name": "vectorized-gaussian-renderer-public",
            "benchmark_commit": "1" * 40,
            "benchmark_tag": "benchmark-gcp-l4-matched-v5",
            "publication_content_commit": "2" * 40,
            "final_merge_commit": "3" * 40,
            "release_tag": "vgr-gcp-l4-matched-v5",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_writes_exact_canonical_release_envelope_schema(self) -> None:
        spec = writer.write_release_spec(
            **self.arguments, output_path=self.output
        )
        payload = self.output.read_bytes()
        self.assertEqual(payload, writer.canonical_json_bytes(spec))
        self.assertEqual(json.loads(payload), spec)
        self.assertEqual(
            set(spec), {"artifacts", "repository", "schema_version"}
        )
        self.assertEqual(spec["schema_version"], writer.SPEC_SCHEMA)
        self.assertEqual(set(spec["artifacts"]), set(writer.ARTIFACT_ROLES))
        self.assertEqual(_validate_repository_spec(spec["repository"]), spec["repository"])
        self.assertEqual(
            spec["repository"]["url"],
            "https://github.com/cybernetic-physics/vectorized-gaussian-renderer-public",
        )
        self.assertEqual(
            spec["repository"]["release_url"],
            "https://github.com/cybernetic-physics/vectorized-gaussian-renderer-public/"
            "releases/tag/vgr-gcp-l4-matched-v5",
        )
        self.assertEqual(spec["repository"]["visibility"], "public")
        for role, path in self.artifacts.items():
            expected = {
                "bytes": path.stat().st_size,
                "path": str(path.absolute()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            self.assertEqual(spec["artifacts"][role], expected)
            resolved = _resolve_artifact(role, expected)
            self.assertEqual(resolved.identity, {key: expected[key] for key in ("bytes", "sha256")})

    def test_identical_rerun_is_accepted_without_replacing_output(self) -> None:
        first = writer.write_release_spec(**self.arguments, output_path=self.output)
        before = self.output.stat()
        second = writer.write_release_spec(**self.arguments, output_path=self.output)
        after = self.output.stat()
        self.assertEqual(second, first)
        self.assertEqual((after.st_dev, after.st_ino), (before.st_dev, before.st_ino))

    def test_existing_different_output_is_rejected(self) -> None:
        self.output.write_text("different\n", encoding="utf-8")
        with self.assertRaisesRegex(writer.ReleaseSpecError, "refusing to overwrite"):
            writer.write_release_spec(**self.arguments, output_path=self.output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "different\n")

    def test_symlink_artifact_is_rejected(self) -> None:
        target = self.artifacts["article"]
        link = self.root / "article-link"
        link.symlink_to(target)
        paths = dict(self.artifacts, article=link)
        with self.assertRaisesRegex(writer.ReleaseSpecError, "traverses a symlink"):
            writer.build_release_spec(**dict(self.arguments, artifact_paths=paths))

    def test_symlink_ancestor_is_rejected(self) -> None:
        real = self.root / "real"
        real.mkdir()
        nested = real / "article.bin"
        nested.write_bytes(b"article through ancestor\n")
        link = self.root / "linked-directory"
        link.symlink_to(real, target_is_directory=True)
        paths = dict(self.artifacts, article=link / nested.name)
        with self.assertRaisesRegex(writer.ReleaseSpecError, "traverses a symlink"):
            writer.build_release_spec(**dict(self.arguments, artifact_paths=paths))

    def test_output_symlink_is_rejected_without_touching_target(self) -> None:
        target = self.root / "target.json"
        target.write_bytes(b"do not change\n")
        self.output.symlink_to(target)
        with self.assertRaisesRegex(writer.ReleaseSpecError, "may not be a symlink"):
            writer.write_release_spec(**self.arguments, output_path=self.output)
        self.assertEqual(target.read_bytes(), b"do not change\n")

    def test_output_path_cannot_replace_or_hardlink_an_artifact(self) -> None:
        with self.assertRaisesRegex(writer.ReleaseSpecError, "may not replace"):
            writer.write_release_spec(
                **self.arguments, output_path=self.artifacts["article"]
            )
        hardlink = self.root / "hardlinked-output.json"
        os.link(self.artifacts["article"], hardlink)
        with self.assertRaisesRegex(writer.ReleaseSpecError, "aliases a release artifact"):
            writer.write_release_spec(**self.arguments, output_path=hardlink)

    def test_duplicate_paths_and_hardlinks_across_roles_are_rejected(self) -> None:
        duplicate = dict(self.artifacts, readme=self.artifacts["article"])
        with self.assertRaisesRegex(writer.ReleaseSpecError, "use the same path"):
            writer.build_release_spec(
                **dict(self.arguments, artifact_paths=duplicate)
            )
        hardlink = self.root / "article-hardlink.bin"
        os.link(self.artifacts["article"], hardlink)
        aliases = dict(self.artifacts, readme=hardlink)
        with self.assertRaisesRegex(writer.ReleaseSpecError, "alias one file"):
            writer.build_release_spec(
                **dict(self.arguments, artifact_paths=aliases)
            )

    def test_missing_unknown_and_nonregular_artifacts_are_rejected(self) -> None:
        missing_role = dict(self.artifacts)
        missing_role.pop("article")
        with self.assertRaisesRegex(writer.ReleaseSpecError, "missing or unknown"):
            writer.build_release_spec(
                **dict(self.arguments, artifact_paths=missing_role)
            )
        unknown_role = dict(self.artifacts, unknown=self.artifacts["article"])
        with self.assertRaisesRegex(writer.ReleaseSpecError, "missing or unknown"):
            writer.build_release_spec(
                **dict(self.arguments, artifact_paths=unknown_role)
            )
        missing_file = dict(self.artifacts, article=self.root / "missing")
        with self.assertRaisesRegex(writer.ReleaseSpecError, "is missing"):
            writer.build_release_spec(
                **dict(self.arguments, artifact_paths=missing_file)
            )
        directory = self.root / "directory"
        directory.mkdir()
        nonregular = dict(self.artifacts, article=directory)
        with self.assertRaisesRegex(writer.ReleaseSpecError, "must be a regular file"):
            writer.build_release_spec(
                **dict(self.arguments, artifact_paths=nonregular)
            )
        empty = self.root / "empty.bin"
        empty.touch()
        zero_byte = dict(self.artifacts, article=empty)
        with self.assertRaisesRegex(writer.ReleaseSpecError, "at least one byte"):
            writer.build_release_spec(
                **dict(self.arguments, artifact_paths=zero_byte)
            )

    def test_repository_inputs_use_exact_envelope_validation(self) -> None:
        invalid_cases = (
            {"owner": "bad/owner"},
            {"benchmark_commit": "A" * 40},
            {"benchmark_tag": "bad/tag"},
            {"release_tag": "bad tag"},
            {"final_merge_commit": "2" * 40},
        )
        for changes in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaises(writer.ReleaseSpecError):
                    writer.build_release_spec(**dict(self.arguments, **changes))

    def test_artifact_change_before_atomic_publication_is_rejected(self) -> None:
        original_revalidate = writer._revalidate_snapshots

        def mutate_then_revalidate(snapshots: object) -> None:
            self.artifacts["article"].write_bytes(b"changed after identity\n")
            original_revalidate(snapshots)  # type: ignore[arg-type]

        with mock.patch.object(
            writer, "_revalidate_snapshots", side_effect=mutate_then_revalidate
        ):
            with self.assertRaisesRegex(writer.ReleaseSpecError, "changed after"):
                writer.write_release_spec(**self.arguments, output_path=self.output)
        self.assertFalse(self.output.exists())

    def test_control_character_path_and_output_parent_symlink_are_rejected(self) -> None:
        bad_path = dict(self.artifacts, article=self.root / "bad\npath")
        with self.assertRaisesRegex(writer.ReleaseSpecError, "control-character"):
            writer.build_release_spec(
                **dict(self.arguments, artifact_paths=bad_path)
            )
        real = self.root / "real-output"
        real.mkdir()
        linked = self.root / "linked-output"
        linked.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(writer.ReleaseSpecError, "traverses a symlink"):
            writer.write_release_spec(
                **self.arguments, output_path=linked / "release-spec.json"
            )

    def test_parse_args_requires_all_exact_artifact_inputs(self) -> None:
        arguments: list[str] = []
        for role, path in self.artifacts.items():
            arguments.extend((f"--{role.replace('_', '-')}", str(path)))
        arguments.extend(
            (
                "--owner",
                "cybernetic-physics",
                "--name",
                "vectorized-gaussian-renderer-public",
                "--benchmark-commit",
                "1" * 40,
                "--benchmark-tag",
                "benchmark-gcp-l4-matched-v5",
                "--publication-content-commit",
                "2" * 40,
                "--final-merge-commit",
                "3" * 40,
                "--release-tag",
                "vgr-gcp-l4-matched-v5",
                "--output",
                str(self.output),
            )
        )
        parsed = writer.parse_args(arguments)
        self.assertEqual(parsed.output, self.output)
        self.assertEqual(parsed.bundle_archive, self.artifacts["bundle_archive"])


if __name__ == "__main__":
    unittest.main()
