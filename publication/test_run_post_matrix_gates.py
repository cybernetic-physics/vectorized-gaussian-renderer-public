from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PUBLICATION_ROOT = Path(__file__).resolve().parent
if str(PUBLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLICATION_ROOT))

from aggregate_verification import REQUIRED_GATE_IDS  # noqa: E402
from run_post_matrix_gates import (  # noqa: E402
    REQUIRED_GATES,
    REQUIRED_PYTEST_FRAGMENTS,
    copy_tree_new_or_identical,
    reject_tree_symlinks,
    reject_symlink_components,
    require_clean_git_checkout,
    require_real_directory,
    require_resume_ablation_alias,
    write_b64_redaction_inventory,
    write_new_or_identical,
)


def artifact_record(path: Path) -> dict[str, object]:
    import hashlib

    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class PostMatrixCoordinatorTests(unittest.TestCase):
    def test_release_unit_gate_requires_nonoptional_publication_modules(self) -> None:
        required = set(REQUIRED_PYTEST_FRAGMENTS)
        self.assertIn(
            "test_build_stage_authors_projection_sorting_and_fractional_opacity",
            required,
        )
        self.assertIn(
            "test_checked_out_upstream_transform_preserves_equation_functions",
            required,
        )
        self.assertIn(
            "test_redacts_paths_and_rebinds_transitive_hashes",
            required,
        )
        self.assertIn(
            "test_content_addressed_manifest_matches_canonical_bytes",
            required,
        )

    def test_gate_contract_exactly_matches_aggregate(self) -> None:
        self.assertEqual(REQUIRED_GATES, REQUIRED_GATE_IDS)
        self.assertEqual(len(REQUIRED_GATES), len(set(REQUIRED_GATES)))

    def test_immutable_write_allows_only_identical_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            write_new_or_identical(path, b"one\n")
            write_new_or_identical(path, b"one\n")
            with self.assertRaises(FileExistsError):
                write_new_or_identical(path, b"two\n")

    def test_historical_tree_resume_checks_every_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            (source / "nested").mkdir(parents=True)
            (source / "nested/evidence.json").write_text("one\n", encoding="utf-8")
            copy_tree_new_or_identical(source, destination)
            copy_tree_new_or_identical(source, destination)
            (destination / "nested/evidence.json").write_text("two\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                copy_tree_new_or_identical(source, destination)

    def test_b64_redaction_inventory_binds_every_public_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical = root / "historical"
            historical.mkdir()
            historical_file = historical / "trace.bin"
            historical_file.write_bytes(b"public-trace")
            index = root / "diagnosis-index.json"
            lock = root / "diagnosis-lock.json"
            known = root / "known-failure-manifest.json"
            manifest_path = root / "privacy-redaction-manifest.json"
            for path, payload in (
                (index, "index\n"),
                (lock, "lock\n"),
                (known, "known\n"),
                (manifest_path, "manifest\n"),
            ):
                path.write_text(payload, encoding="utf-8")
            files = []
            for label, path in (
                ("diagnosis-index.json", index),
                ("diagnosis-lock.json", lock),
                ("historical/trace.bin", historical_file),
                ("known-failure-manifest.json", known),
            ):
                record = artifact_record(path)
                files.append(
                    {
                        "bytes": record["bytes"],
                        "logical_path": label,
                        "public_sha256": record["sha256"],
                    }
                )
            output = root / "privacy-redaction-inventory.json"
            inventory = write_b64_redaction_inventory(
                manifest={"files": files},
                historical_root=historical,
                diagnosis_index=index,
                diagnosis_lock=lock,
                known_failure_manifest=known,
                manifest_path=manifest_path,
                output=output,
            )
            self.assertTrue(inventory["pass"])
            self.assertEqual(inventory["file_count"], 4)
            write_b64_redaction_inventory(
                manifest={"files": files},
                historical_root=historical,
                diagnosis_index=index,
                diagnosis_lock=lock,
                known_failure_manifest=known,
                manifest_path=manifest_path,
                output=output,
            )
            historical_file.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "artifact differs"):
                write_b64_redaction_inventory(
                    manifest={"files": files},
                    historical_root=historical,
                    diagnosis_index=index,
                    diagnosis_lock=lock,
                    known_failure_manifest=known,
                    manifest_path=manifest_path,
                    output=root / "changed-inventory.json",
                )

    def test_clean_dependency_checkout_rejects_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "dependency"
            subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
            source = checkout / "source.txt"
            source.write_text("pinned\n", encoding="utf-8")
            subprocess.run(["git", "add", "source.txt"], cwd=checkout, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Publication Test",
                    "-c",
                    "user.email=publication-test@example.invalid",
                    "commit",
                    "-m",
                    "fixture",
                ],
                cwd=checkout,
                check=True,
                capture_output=True,
            )
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                text=True,
            ).strip()
            self.assertEqual(
                require_clean_git_checkout(
                    checkout,
                    expected_commit=commit,
                    label="fixture dependency",
                ),
                checkout.resolve(),
            )
            source.write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be clean"):
                require_clean_git_checkout(
                    checkout,
                    expected_commit=commit,
                    label="fixture dependency",
                )

    def test_resume_directories_reject_symlinked_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            outside = root / "outside"
            output.mkdir()
            outside.mkdir()
            (output / "raw").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                reject_symlink_components(
                    output / "raw/evidence.json",
                    boundary=output,
                    label="raw evidence",
                )
            with self.assertRaises(ValueError):
                require_real_directory(
                    output / "raw",
                    boundary=output,
                    label="raw root",
                )

    def test_resume_tree_rejects_nested_b64_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            outside = root / "outside"
            (output / "raw/b64-artifacts/evidence").mkdir(parents=True)
            outside.mkdir()
            (output / "raw/b64-artifacts/evidence/repair").symlink_to(
                outside,
                target_is_directory=True,
            )
            with self.assertRaises(ValueError):
                reject_tree_symlinks(output)

    def test_resume_ablation_alias_must_match_completed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrix = Path(temporary) / "matrix"
            output = matrix / "post"
            result_path = output / "gates/custom-vectorization-ablation/result.json"
            alias = matrix / "publication/evidence/custom-vectorization-ablation.json"
            result_path.parent.mkdir(parents=True)
            alias.parent.mkdir(parents=True)
            alias.write_text('{"pass": true}\n', encoding="utf-8")
            result = {
                "evidence": {"raw_result": artifact_record(alias)},
                "gate_id": "custom-vectorization-ablation",
                "pass": True,
            }
            result_path.write_text(
                json.dumps(result) + "\n",
                encoding="utf-8",
            )
            require_resume_ablation_alias(output=output, matrix=matrix, alias=alias)
            alias.write_text('{"pass": false}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                require_resume_ablation_alias(output=output, matrix=matrix, alias=alias)

    def test_resume_ablation_result_without_alias_fails_early(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrix = Path(temporary) / "matrix"
            output = matrix / "post"
            result_path = output / "gates/custom-vectorization-ablation/result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text("{}\n", encoding="utf-8")
            alias = matrix / "publication/evidence/custom-vectorization-ablation.json"
            with self.assertRaises(FileNotFoundError):
                require_resume_ablation_alias(output=output, matrix=matrix, alias=alias)


if __name__ == "__main__":
    unittest.main()
