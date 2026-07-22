from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from publication.run_bounded_gate import (
    build_normalized_result,
    load_subcommands,
    parse_args,
    sample_failures,
)


class BoundedGateTests(unittest.TestCase):
    def test_parse_requires_bounded_command(self) -> None:
        parsed = parse_args(
            [
                "--gate-id",
                "native-cuda-smoke",
                "--source-root",
                "/tmp/source",
                "--output-dir",
                "/tmp/output",
                "--expected-gpu-uuid",
                "GPU-test",
                "--timeout-seconds",
                "60",
                "--",
                "python",
                "smoke.py",
            ]
        )
        self.assertEqual(parsed.command, ["python", "smoke.py"])
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--gate-id",
                    "x",
                    "--source-root",
                    "/tmp/source",
                    "--output-dir",
                    "/tmp/output",
                    "--expected-gpu-uuid",
                    "GPU-test",
                    "--timeout-seconds",
                    "0",
                    "--",
                    "true",
                ]
            )

    def test_subcommand_schema_is_strict(self) -> None:
        with self.subTest("valid"), unittest.mock.patch.object(
            Path, "is_symlink", return_value=False
        ), unittest.mock.patch.object(Path, "is_file", return_value=True), unittest.mock.patch.object(
            Path, "read_text", return_value=json.dumps([["python", "run.py"]])
        ):
            self.assertEqual(load_subcommands(Path("plan.json")), [["python", "run.py"]])
        with self.subTest("empty-command"), unittest.mock.patch.object(
            Path, "is_symlink", return_value=False
        ), unittest.mock.patch.object(Path, "is_file", return_value=True), unittest.mock.patch.object(
            Path, "read_text", return_value=json.dumps([[]])
        ):
            with self.assertRaises(ValueError):
                load_subcommands(Path("plan.json"))

    def test_normalized_result_binds_raw_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.json"
            native = root / "native.so"
            evidence = root / "raw.json"
            source.write_text("{}\n", encoding="utf-8")
            native.write_bytes(b"native")
            evidence.write_text('{"pass": true}\n', encoding="utf-8")
            spec = root / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "schema_version": "publication-gate-result-spec-v1",
                        "gate_id": "native-cuda-smoke",
                        "identity": {
                            "gpu_uuid": "GPU-test",
                            "scene_sha256": "a" * 64,
                            "source_manifest": str(source),
                            "native_extension": str(native),
                        },
                        "checks": {"a": True, "b": True, "c": True},
                        "evidence": {"raw_result": str(evidence)},
                    }
                ),
                encoding="utf-8",
            )
            result = build_normalized_result(spec, gate_id="native-cuda-smoke")
            self.assertTrue(result["pass"])
            self.assertEqual(result["evidence"]["raw_result"]["bytes"], evidence.stat().st_size)
            self.assertEqual(result["identity"]["gpu_uuid"], "GPU-test")

    @mock.patch("publication.run_bounded_gate.descendant_of", return_value=True)
    def test_sampler_accepts_only_descendant_on_expected_gpu(self, descendant: mock.Mock) -> None:
        sample = {
            "probe_status": {"success": True},
            "compute_apps": [{"gpu_uuid": "GPU-good", "pid": "42", "process_name": "python"}],
        }
        self.assertEqual(
            sample_failures(sample, expected_gpu_uuid="GPU-good", executor_pid=7),
            [],
        )
        descendant.assert_called_once_with(42, 7)

    @mock.patch("publication.run_bounded_gate.descendant_of", return_value=False)
    def test_sampler_accepts_the_known_direct_child_after_exit(
        self,
        descendant: mock.Mock,
    ) -> None:
        sample = {
            "probe_status": {"success": True},
            "compute_apps": [
                {"gpu_uuid": "GPU-good", "pid": "42", "process_name": "[No data]"}
            ],
        }
        self.assertEqual(
            sample_failures(
                sample,
                expected_gpu_uuid="GPU-good",
                executor_pid=7,
                allowed_pids={42},
            ),
            [],
        )
        descendant.assert_not_called()

    @mock.patch("publication.run_bounded_gate.descendant_of", return_value=True)
    def test_sampler_retains_a_live_verified_descendant(
        self,
        descendant: mock.Mock,
    ) -> None:
        allowed_pids = {41}
        sample = {
            "probe_status": {"success": True},
            "compute_apps": [
                {"gpu_uuid": "GPU-good", "pid": "42", "process_name": "python"}
            ],
        }
        self.assertEqual(
            sample_failures(
                sample,
                expected_gpu_uuid="GPU-good",
                executor_pid=7,
                allowed_pids=allowed_pids,
            ),
            [],
        )
        self.assertEqual(allowed_pids, {41, 42})
        descendant.assert_called_once_with(42, 7)

    @mock.patch("publication.run_bounded_gate.descendant_of", return_value=False)
    def test_sampler_rejects_unrelated_or_wrong_gpu(self, _descendant: mock.Mock) -> None:
        sample = {
            "probe_status": {"success": True},
            "compute_apps": [{"gpu_uuid": "GPU-other", "pid": "42", "process_name": "python"}],
        }
        failures = sample_failures(sample, expected_gpu_uuid="GPU-good", executor_pid=7)
        self.assertEqual(len(failures), 1)
        self.assertIn("unfamiliar", failures[0])


if __name__ == "__main__":
    unittest.main()
