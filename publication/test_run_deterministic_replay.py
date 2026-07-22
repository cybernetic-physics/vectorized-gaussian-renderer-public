from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_deterministic_replay as replay


class DeterministicReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_smoke = self.root / "deterministic_cuda_smoke.py"
        self.source_smoke.write_text("# frozen reference\n", encoding="utf-8")
        self.worker = self.root / "worker.py"
        self.output = self.root / "replay.json"

        self.native_build = self.root / "native-build"
        self.native_build.mkdir()
        self.module_name = "isaacsim_gaussian_renderer_cuda_0123456789ab"
        self.native = self.native_build / f"{self.module_name}.so"
        self.native.write_bytes(b"fake deterministic native extension\0")
        self.native_sources = self.root / "native-sources"
        self.native_sources.mkdir()
        self.renderer_cpp = self.native_sources / "renderer.cpp"
        self.renderer_cuda = self.native_sources / "renderer_cuda.cu"
        self.renderer_cpp.write_text("// renderer\n", encoding="utf-8")
        self.renderer_cuda.write_text("// renderer cuda\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arguments(self, *extra: str) -> list[str]:
        return [
            "--python",
            sys.executable,
            "--source-root",
            str(self.root),
            "--worker-script",
            str(self.worker),
            "--source-smoke",
            str(self.source_smoke),
            "--output",
            str(self.output),
            *extra,
        ]

    def valid_payload(self, digest: str = "a" * 64) -> dict[str, object]:
        arguments = dict(replay.PRODUCTION_ARGUMENTS)
        specs = replay.expected_output_specs(arguments)
        total_pixels = arguments["batch"] * arguments["height"] * arguments["width"]
        foreground_pixels = 1_024
        semantic_foreground_pixels = 512
        output_contract = {
            field: True for field in replay.OUTPUT_CONTRACT_BOOLEAN_FIELDS
        }
        output_contract.update(
            {
                "background_pixel_count": total_pixels - foreground_pixels,
                "foreground_pixel_count": foreground_pixels,
                "output_devices": {name: "cuda:0" for name in replay.OUTPUTS},
                "semantic_background_pixel_count": (
                    total_pixels - semantic_foreground_pixels
                ),
                "semantic_foreground_pixel_count": semantic_foreground_pixels,
            }
        )
        return {
            "arguments": arguments,
            "bitwise_equal": {name: True for name in replay.OUTPUTS},
            "capacity_invariant_bitwise_equal": {
                name: True for name in replay.OUTPUTS
            },
            "center_semantic": 1_000,
            "claim_scope": dict(replay.CLAIM_SCOPE),
            "comparison_method": replay.COMPARISON_METHOD,
            "counters": {
                "active_tiles": 4,
                "intersection_overflow": 0,
                "tile_intersections": 4_096,
                "visible_gaussians": 4_096,
                "visible_overflow": 0,
            },
            "equal_depth_gaussian_id_order": "ascending",
            "equal_depth_records_checked": arguments["gaussians"],
            "fixture_sha256": "f" * 64,
            "native_build_contract": {
                "build_directory": str(self.native_build.resolve()),
                "cuda_flags": list(replay.EXPECTED_CUDA_FLAGS),
                "cxx_flags": list(replay.EXPECTED_CXX_FLAGS),
                "module_name": self.module_name,
                "sources": [
                    str(self.renderer_cpp.resolve()),
                    str(self.renderer_cuda.resolve()),
                ],
                "torch_cuda_arch_list": "8.9",
            },
            "native_extension": replay.artifact_record(self.native),
            "output_contract": output_contract,
            "output_digests": {
                name: {**specs[name], "sha256": digest}
                for name in replay.OUTPUTS
            },
            "pass": True,
            "renderer_configuration": dict(replay.RENDERER_CONFIGURATION),
            "schema_version": replay.WORKER_SCHEMA,
            "source_smoke": replay.artifact_record(self.source_smoke),
        }

    def write_worker(self, body: str) -> None:
        self.worker.write_text(body, encoding="utf-8")

    def write_payload_worker(self, value: dict[str, object]) -> None:
        encoded = json.dumps(value, sort_keys=True)
        self.write_worker(f"print('DETERMINISTIC_DIGEST_SMOKE_OK ' + {encoded!r})\n")

    def validate(self, value: dict[str, object]) -> None:
        replay.validate_worker_payload(
            value,
            expected_arguments=dict(replay.PRODUCTION_ARGUMENTS),
            expected_source_smoke=replay.artifact_record(self.source_smoke),
        )

    def test_two_fresh_equal_processes_are_content_addressed(self) -> None:
        expected = self.valid_payload()
        self.write_payload_worker(expected)
        replay.main(self.arguments())
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertTrue(result["pass"])
        self.assertEqual(result["fresh_processes"], 2)
        self.assertEqual(result["claim_scope"], replay.CLAIM_SCOPE)
        self.assertEqual(result["comparison_method"], replay.COMPARISON_METHOD)
        self.assertTrue(result["cross_process_output_hashes_equal"])
        self.assertTrue(all(record["pid"] > 0 for record in result["processes"]))
        for index in (1, 2):
            child = self.root / f"replay.process-{index}.json"
            self.assertTrue(child.is_file())
            self.assertEqual(json.loads(child.read_text(encoding="utf-8")), expected)

    def test_same_pid_record_does_not_false_fail_fresh_popen_contract(self) -> None:
        expected = self.valid_payload()
        self.write_worker("# mocked worker\n")
        with mock.patch.object(replay, "run_worker", return_value=(777, expected)):
            replay.main(self.arguments())
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertTrue(result["pass"])
        self.assertEqual([process["pid"] for process in result["processes"]], [777, 777])

    def test_invalid_pid_record_fails_closed(self) -> None:
        expected = self.valid_payload()
        self.write_worker("# mocked worker\n")
        with mock.patch.object(replay, "run_worker", return_value=(0, expected)):
            with self.assertRaisesRegex(RuntimeError, "invalid process ID"):
                replay.main(self.arguments())
        self.assertFalse(self.output.exists())

    def test_cross_process_digest_mismatch_fails_without_summary(self) -> None:
        first = json.dumps(self.valid_payload("a" * 64), sort_keys=True)
        second = json.dumps(self.valid_payload("b" * 64), sort_keys=True)
        counter = self.root / "counter"
        self.write_worker(
            "from pathlib import Path\n"
            "import os\n"
            "path = Path(os.environ['FAKE_COUNTER'])\n"
            "value = int(path.read_text()) if path.exists() else 0\n"
            "path.write_text(str(value + 1))\n"
            f"payload = {first!r} if value == 0 else {second!r}\n"
            "print('DETERMINISTIC_DIGEST_SMOKE_OK ' + payload)\n"
        )
        with mock.patch.dict(os.environ, {"FAKE_COUNTER": str(counter)}):
            with self.assertRaises(AssertionError):
                replay.main(self.arguments())
        self.assertFalse(self.output.exists())

    def test_missing_or_null_digest_and_fixture_records_fail_closed(self) -> None:
        for field, replacement in (
            ("output_digests", None),
            ("fixture_sha256", None),
        ):
            with self.subTest(field=field):
                candidate = self.valid_payload()
                candidate[field] = replacement
                with self.assertRaises(ValueError):
                    self.validate(candidate)

    def test_digest_contract_rejects_wrong_keys_hash_dtype_shape_and_bytes(self) -> None:
        mutations = {
            "missing_output": lambda value: value["output_digests"].pop("alpha"),
            "bad_hash": lambda value: value["output_digests"]["rgb"].__setitem__(
                "sha256", "not-a-sha"
            ),
            "semantic_dtype": lambda value: value["output_digests"][
                "semantic_id"
            ].__setitem__("dtype", "torch.float32"),
            "rgb_shape": lambda value: value["output_digests"]["rgb"].__setitem__(
                "shape", [1]
            ),
            "depth_bytes": lambda value: value["output_digests"][
                "depth"
            ].__setitem__("bytes", 4),
            "float_shape": lambda value: value["output_digests"]["depth"].__setitem__(
                "shape", [4.0, 64, 64, 1]
            ),
            "extra_digest_field": lambda value: value["output_digests"][
                "alpha"
            ].__setitem__("extra", True),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(self.valid_payload())
                mutate(candidate)
                with self.assertRaises(ValueError):
                    self.validate(candidate)

    def test_output_contract_rejects_failed_predicates_devices_and_coverage(self) -> None:
        mutations = {
            "failed_predicate": lambda value: value["output_contract"].__setitem__(
                "finite_rgb", False
            ),
            "cpu_device": lambda value: value["output_contract"][
                "output_devices"
            ].__setitem__("rgb", "cpu"),
            "empty_foreground": lambda value: value["output_contract"].__setitem__(
                "foreground_pixel_count", 0
            ),
            "bad_total": lambda value: value["output_contract"].__setitem__(
                "background_pixel_count", 0
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(self.valid_payload())
                mutate(candidate)
                with self.assertRaises(ValueError):
                    self.validate(candidate)

    def test_arguments_counters_tie_order_and_source_are_exact(self) -> None:
        other_source = self.root / "other-smoke.py"
        other_source.write_text("# other\n", encoding="utf-8")
        mutations = {
            "arguments": lambda value: value["arguments"].__setitem__("batch", 8),
            "argument_type": lambda value: value["arguments"].__setitem__(
                "batch", 4.0
            ),
            "counter_keys": lambda value: value["counters"].pop("active_tiles"),
            "overflow": lambda value: value["counters"].__setitem__(
                "intersection_overflow", 1
            ),
            "tie_count": lambda value: value.__setitem__(
                "equal_depth_records_checked", 1
            ),
            "source": lambda value: value.__setitem__(
                "source_smoke", replay.artifact_record(other_source)
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(self.valid_payload())
                mutate(candidate)
                with self.assertRaises(ValueError):
                    self.validate(candidate)

    def test_native_extension_and_build_contract_are_verified(self) -> None:
        mutations = {
            "native_hash": lambda value: value["native_extension"].__setitem__(
                "sha256", "0" * 64
            ),
            "native_flags": lambda value: value["native_build_contract"].__setitem__(
                "cuda_flags", []
            ),
            "native_arch": lambda value: value["native_build_contract"].__setitem__(
                "torch_cuda_arch_list", "9.0"
            ),
            "native_source": lambda value: value["native_build_contract"].__setitem__(
                "sources", [str(self.renderer_cpp), str(self.renderer_cpp)]
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(self.valid_payload())
                mutate(candidate)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.validate(candidate)

    def test_worker_mutation_during_first_process_fails_before_evidence_write(self) -> None:
        encoded = json.dumps(self.valid_payload(), sort_keys=True)
        self.write_worker(
            "from pathlib import Path\n"
            "path = Path(__file__)\n"
            "path.write_text(path.read_text() + '# mutated\\n')\n"
            f"print('DETERMINISTIC_DIGEST_SMOKE_OK ' + {encoded!r})\n"
        )
        with self.assertRaisesRegex(RuntimeError, "changed while process 1 ran"):
            replay.main(self.arguments())
        self.assertFalse((self.root / "replay.process-1.json").exists())
        self.assertFalse(self.output.exists())

    def test_source_mutation_during_first_process_fails_before_evidence_write(self) -> None:
        encoded = json.dumps(self.valid_payload(), sort_keys=True)
        self.write_worker(
            "from pathlib import Path\n"
            "import sys\n"
            "source = Path(sys.argv[sys.argv.index('--source-smoke') + 1])\n"
            "source.write_text('# mutated source\\n')\n"
            f"print('DETERMINISTIC_DIGEST_SMOKE_OK ' + {encoded!r})\n"
        )
        with self.assertRaisesRegex(RuntimeError, "changed while process 1 ran"):
            replay.main(self.arguments())
        self.assertFalse((self.root / "replay.process-1.json").exists())
        self.assertFalse(self.output.exists())

    def test_duplicate_worker_sentinel_fails(self) -> None:
        encoded = json.dumps(self.valid_payload(), sort_keys=True)
        self.write_worker(
            f"print('DETERMINISTIC_DIGEST_SMOKE_OK ' + {encoded!r})\n"
            f"print('DETERMINISTIC_DIGEST_SMOKE_OK ' + {encoded!r})\n"
        )
        with self.assertRaises(RuntimeError):
            replay.main(self.arguments())
        self.assertFalse(self.output.exists())

    def test_nonzero_and_malformed_workers_fail_without_summary(self) -> None:
        cases = {
            "nonzero": ("raise SystemExit(9)\n", subprocess.CalledProcessError),
            "malformed": (
                "print('DETERMINISTIC_DIGEST_SMOKE_OK {bad json')\n",
                ValueError,
            ),
            "missing": ("print('no sentinel')\n", RuntimeError),
        }
        for label, (body, exception) in cases.items():
            with self.subTest(label=label):
                self.write_worker(body)
                with self.assertRaises(exception):
                    replay.main(self.arguments())
                self.assertFalse(self.output.exists())

    def test_nonproduction_scope_is_rejected_before_launch(self) -> None:
        self.write_worker("raise AssertionError('must not launch')\n")
        with self.assertRaisesRegex(ValueError, "one fixed scope"):
            replay.main(self.arguments("--batch", "8"))
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_reused(self) -> None:
        self.worker.write_text("raise SystemExit(99)\n", encoding="utf-8")
        self.output.write_text("old\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            replay.main(self.arguments())


if __name__ == "__main__":
    unittest.main()
