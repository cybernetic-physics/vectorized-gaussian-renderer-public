from __future__ import annotations

import json
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
    copy_tree_new_or_identical,
    reject_tree_symlinks,
    reject_symlink_components,
    require_real_directory,
    require_resume_ablation_alias,
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
