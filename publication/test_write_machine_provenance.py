from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from publication import write_machine_provenance as provenance


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class MachineProvenanceTests(unittest.TestCase):
    def fixture(self, root: Path) -> None:
        source_path = root / "provenance/source-manifest.json"
        source = {
            "schema_version": provenance.SOURCE_MANIFEST_SCHEMA,
            "head": "a" * 40,
            "dirty": False,
            "status_short": [],
        }
        write_json(source_path, source)
        source_artifact = provenance.logical_artifact(source_path, root=root)
        environment = {
            "compiler_versions": {"cxx": "c++ 11.4", "nvcc": "CUDA 12.9"},
            "compute_capability": [8, 9],
            "cuda_runtime": "12.8",
            "driver": "580.159.03",
            "gpu_name": provenance.ACCEPTED_HARDWARE_NAME,
            "gpu_uuid": provenance.ACCEPTED_HARDWARE_UUID,
            "torch": "2.11.0+cu128",
            "torch_cuda_arch_list": "8.9",
            "source_git_commit": "a" * 40,
            "source_provenance": {
                "manifest": {
                    "bytes": source_artifact["bytes"],
                    "path": str(source_path),
                    "sha256": source_artifact["sha256"],
                }
            },
        }
        for batch in provenance.PRIMARY_BATCHES:
            for renderer in provenance.RENDERERS:
                for contract in provenance.CONTRACTS:
                    write_json(
                        root / f"runs/{renderer}/{contract}/b{batch}.json",
                        {
                            "schema_version": provenance.RUN_SCHEMA,
                            "renderer": renderer,
                            "output_contract": contract,
                            "pass": True,
                            "headline_eligible": True,
                            "capacity": {"logical_batch": batch},
                            "environment": environment,
                        },
                    )

    def test_complete_matrix_emits_portable_common_environment(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.fixture(root)
            result = provenance.build_machine_provenance(
                root,
                expected_gpu_uuid=provenance.ACCEPTED_HARDWARE_UUID,
            )
            self.assertTrue(result["pass"])
            self.assertEqual(result["primary_run_count"], 32)
            self.assertEqual(result["gpu_name"], "NVIDIA L4")
            self.assertEqual(result["gpu_uuid"], provenance.ACCEPTED_HARDWARE_UUID)
            encoded = provenance.canonical_json_bytes(result).decode("utf-8")
            self.assertNotIn(str(root), encoded)
            self.assertTrue(
                all(set(item) == {"bytes", "logical_path", "sha256"} for item in result["primary_runs"])
            )

    def test_inconsistent_primary_run_environment_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.fixture(root)
            path = root / "runs/flashgs/rgb/b1024.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["environment"]["driver"] = "different"
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "environments disagree"):
                provenance.build_machine_provenance(
                    root,
                    expected_gpu_uuid=provenance.ACCEPTED_HARDWARE_UUID,
                )

    def test_missing_primary_run_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.fixture(root)
            (root / "runs/custom/full/b64.json").unlink()
            with self.assertRaises(FileNotFoundError):
                provenance.build_machine_provenance(
                    root,
                    expected_gpu_uuid=provenance.ACCEPTED_HARDWARE_UUID,
                )

    def test_symlinked_primary_run_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.fixture(root)
            path = root / "runs/custom/full/b64.json"
            target = root / "real.json"
            path.rename(target)
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                provenance.build_machine_provenance(
                    root,
                    expected_gpu_uuid=provenance.ACCEPTED_HARDWARE_UUID,
                )

    def test_run_source_manifest_binding_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.fixture(root)
            path = root / "runs/custom/rgb/b8.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["environment"]["source_provenance"]["manifest"]["sha256"] = "f" * 64
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "source binding differs"):
                provenance.build_machine_provenance(
                    root,
                    expected_gpu_uuid=provenance.ACCEPTED_HARDWARE_UUID,
                )

    def test_writer_is_idempotent_but_refuses_different_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.fixture(root)
            output = root / "publication/machine-provenance.json"
            arguments = [
                "--matrix-root",
                str(root),
                "--output",
                str(output),
                "--expected-gpu-uuid",
                provenance.ACCEPTED_HARDWARE_UUID,
            ]
            provenance.main(arguments)
            first = output.read_bytes()
            provenance.main(arguments)
            self.assertEqual(output.read_bytes(), first)
            output.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                provenance.main(arguments)


if __name__ == "__main__":
    unittest.main()
