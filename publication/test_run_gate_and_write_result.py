from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PUBLICATION_ROOT = Path(__file__).resolve().parent
RUNNER = PUBLICATION_ROOT / "run_gate_and_write_result.py"


class GateAndResultTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        checks = root / "checks.json"
        source = root / "source.json"
        native = root / "native.so"
        checks.write_text('{"a":true,"b":true,"c":true}\n', encoding="utf-8")
        source.write_text('{"source":true}\n', encoding="utf-8")
        native.write_bytes(b"native")
        return checks, source, native

    def command(
        self,
        *,
        root: Path,
        child: list[str],
    ) -> tuple[list[str], Path, Path]:
        checks, source, native = self.fixture(root)
        evidence = root / "raw.json"
        result = root / "result-spec.json"
        command = [
            sys.executable,
            str(RUNNER),
            "--gate-id",
            "native-cuda-smoke",
            "--checks",
            str(checks),
            "--source-manifest",
            str(source),
            "--native-extension",
            str(native),
            "--scene-sha256",
            "a" * 64,
            "--gpu-uuid",
            "GPU-test",
            "--evidence",
            f"raw_result={evidence}",
            "--result-spec",
            str(result),
            "--",
            *child,
        ]
        return command, evidence, result

    def test_writes_spec_only_after_child_creates_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, evidence, result = self.command(
                root=root,
                child=[
                    sys.executable,
                    "-c",
                    "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('eles\\n'); print('CHILD_OK')",
                    str(root / "raw.json"),
                ],
            )
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(evidence.is_file())
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(payload["gate_id"], "native-cuda-smoke")
            self.assertEqual(payload["identity"]["gpu_uuid"], "GPU-test")
            self.assertEqual(payload["evidence"]["raw_result"], str(evidence.resolve()))
            self.assertIn("CHILD_OK", completed.stdout)
            self.assertEqual(completed.stdout.count("PUBLICATION_GATE_RESULT_SPEC_OK"), 1)

    def test_child_failure_writes_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, evidence, result = self.command(
                root=root,
                child=[sys.executable, "-c", "raise SystemExit(7)"],
            )
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 7)
            self.assertFalse(evidence.exists())
            self.assertFalse(result.exists())

    def test_missing_post_run_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, _evidence, result = self.command(
                root=root,
                child=[sys.executable, "-c", "print('NO_EVIDENCE')"],
            )
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(result.exists())


if __name__ == "__main__":
    unittest.main()
