from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from publication.redact_b64_diagnosis import (
    PRIVATE_ROOT_RE,
    RedactionError,
    redact_diagnosis,
    verify_public_derivative,
)


def identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def private_identity(path: Path, recorded_path: str) -> dict[str, object]:
    record = identity(path)
    record["path"] = recorded_path
    return record


def host_path(layout: str, user: str, *parts: str) -> str:
    """Build a synthetic host path without embedding one in public source."""

    return "/" + layout + "/" + "/".join((user, *parts))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class B64DiagnosisRedactionTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, Path]:
        historical = root / "source/historical"
        binary = historical / "binaries/debug.so"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(
            b"\x7fELF/private:" + host_path("home", "alice", "build", "render.cu").encode("ascii") + b"\x00"
        )
        log = historical / "logs/debug.log"
        log.parent.mkdir()
        log.write_text(
            "source=" + host_path("Users", "alice", "work", "render.cu") + "\n",
            encoding="utf-8",
        )
        known = root / "source/B64_KNOWN_FAILURE_CASES.json"
        write_json(known, {"schema_version": "known", "case_count": 1})
        unchanged_reference = historical / "evidence/unchanged-reference.json"
        write_json(
            unchanged_reference,
            {
                "schema_version": "unchanged-reference",
                "known": private_identity(known, "known-failure-manifest.json"),
            },
        )
        attestation = historical / "evidence/attestation.json"
        write_json(
            attestation,
            {
                "schema_version": "attestation",
                "native": private_identity(
                    binary,
                    host_path("home", "alice", "build", "debug.so"),
                ),
                "root": host_path("home", "alice", "worktree"),
            },
        )
        diagnosis = historical / "evidence/diagnosis.json"
        write_json(
            diagnosis,
            {
                "schema_version": "diagnosis",
                "attestation": private_identity(
                    attestation,
                    host_path("home", "alice", "results", "attestation.json"),
                ),
                "native": private_identity(
                    binary,
                    host_path("home", "alice", "build", "debug.so"),
                ),
                "source": host_path("Users", "alice", "work", "render.cu"),
            },
        )
        lock = root / "source/B64_DIAGNOSIS_LOCK.json"
        write_json(
            lock,
            {
                "schema_version": "lock",
                "diagnosis": private_identity(
                    diagnosis,
                    host_path("home", "alice", "results", "diagnosis.json"),
                ),
                "known": private_identity(
                    known,
                    host_path(
                        "Users",
                        "alice",
                        "source",
                        "B64_KNOWN_FAILURE_CASES.json",
                    ),
                ),
            },
        )
        index = root / "source/diagnosis-index.json"
        write_json(
            index,
            {
                "schema_version": "index",
                "diagnosis": private_identity(
                    diagnosis,
                    host_path("home", "alice", "results", "diagnosis.json"),
                ),
                "lock": private_identity(
                    lock,
                    host_path(
                        "Users",
                        "alice",
                        "source",
                        "B64_DIAGNOSIS_LOCK.json",
                    ),
                ),
            },
        )
        return {
            "historical": historical,
            "index": index,
            "lock": lock,
            "known": known,
        }

    def run_once(self, root: Path, destination: str) -> dict[str, Path]:
        inputs = self.fixture(root)
        output = root / destination
        paths = {
            "root": output / "historical",
            "index": output / "diagnosis-index.json",
            "lock": output / "B64_DIAGNOSIS_LOCK.json",
            "manifest": output / "privacy-redaction-manifest.json",
        }
        redact_diagnosis(
            historical_root=inputs["historical"],
            diagnosis_index=inputs["index"],
            diagnosis_lock=inputs["lock"],
            known_failure_manifest=inputs["known"],
            output_root=paths["root"],
            output_index=paths["index"],
            output_lock=paths["lock"],
            output_manifest=paths["manifest"],
        )
        return paths

    def test_redacts_paths_and_rebinds_transitive_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            outputs = self.run_once(base, "public")
            files = [path for path in outputs["root"].rglob("*") if path.is_file()]
            files.extend([outputs["index"], outputs["lock"], outputs["manifest"]])
            for path in files:
                self.assertIsNone(PRIVATE_ROOT_RE.search(path.read_bytes()), path)

            binary = outputs["root"] / "binaries/debug.so"
            attestation = outputs["root"] / "evidence/attestation.json"
            diagnosis = outputs["root"] / "evidence/diagnosis.json"
            lock = json.loads(outputs["lock"].read_text(encoding="utf-8"))
            index = json.loads(outputs["index"].read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(attestation.read_text())["native"]["sha256"],
                identity(binary)["sha256"],
            )
            self.assertEqual(
                json.loads(diagnosis.read_text())["attestation"]["sha256"],
                identity(attestation)["sha256"],
            )
            self.assertEqual(lock["diagnosis"]["sha256"], identity(diagnosis)["sha256"])
            self.assertEqual(index["lock"]["sha256"], identity(outputs["lock"])["sha256"])

            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertTrue(manifest["pass"])
            self.assertGreater(manifest["path_prefix_replacements"], 0)
            self.assertGreater(manifest["artifact_hash_rebindings"], 0)
            self.assertTrue(all(record["reversible"] for record in manifest["files"]))
            unchanged = outputs["root"] / "evidence/unchanged-reference.json"
            source_unchanged = base / "source/historical/evidence/unchanged-reference.json"
            self.assertEqual(unchanged.read_bytes(), source_unchanged.read_bytes())
            unchanged_record = next(
                record
                for record in manifest["files"]
                if record["logical_path"]
                == "historical/evidence/unchanged-reference.json"
            )
            self.assertEqual(unchanged_record["artifact_hash_rebindings"], 0)
            self.assertEqual(
                unchanged_record["source_sha256"],
                unchanged_record["public_sha256"],
            )

    def test_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_root = Path(temporary) / "first-fixture"
            second_root = Path(temporary) / "second-fixture"
            first = self.run_once(first_root, "public")
            second = self.run_once(second_root, "public")
            first_files = {
                path.relative_to(first["root"]): path.read_bytes()
                for path in first["root"].rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second["root"]): path.read_bytes()
                for path in second["root"].rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            self.assertEqual(first["index"].read_bytes(), second["index"].read_bytes())
            self.assertEqual(first["lock"].read_bytes(), second["lock"].read_bytes())
            # Absolute input paths are intentionally excluded from the manifest.
            self.assertEqual(first["manifest"].read_bytes(), second["manifest"].read_bytes())

    def test_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self.fixture(root)
            output = root / "public"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                redact_diagnosis(
                    historical_root=inputs["historical"],
                    diagnosis_index=inputs["index"],
                    diagnosis_lock=inputs["lock"],
                    known_failure_manifest=inputs["known"],
                    output_root=output,
                    output_index=root / "index.json",
                    output_lock=root / "lock.json",
                    output_manifest=root / "manifest.json",
                )

    def test_public_verifier_rejects_changed_derivative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = self.run_once(root, "public")
            changed = outputs["root"] / "logs/debug.log"
            changed.write_bytes(changed.read_bytes() + b"changed\n")
            with self.assertRaisesRegex(RedactionError, "artifact changed"):
                verify_public_derivative(
                    historical_root=outputs["root"],
                    diagnosis_index=outputs["index"],
                    diagnosis_lock=outputs["lock"],
                    known_failure_manifest=root / "source/B64_KNOWN_FAILURE_CASES.json",
                    manifest_path=outputs["manifest"],
                )

    def test_refuses_impossibly_short_root(self) -> None:
        with self.assertRaises(RedactionError):
            from publication.redact_b64_diagnosis import _replacement_for

            _replacement_for(b"/home")


if __name__ == "__main__":
    unittest.main()
