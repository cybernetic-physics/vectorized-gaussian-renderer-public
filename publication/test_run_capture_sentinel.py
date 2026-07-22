from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from publication.run_capture_sentinel import decode_payload


PUBLICATION_ROOT = Path(__file__).resolve().parent


class CaptureSentinelTests(unittest.TestCase):
    def test_decodes_json_and_python_objects(self) -> None:
        self.assertEqual(decode_payload('{"pass": true}', "json"), {"pass": True})
        self.assertEqual(decode_payload("{'pass': True}", "python"), {"pass": True})
        with self.assertRaises(ValueError):
            decode_payload("[]", "json")

    def test_end_to_end_writes_single_payload_and_streams_log(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "result.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PUBLICATION_ROOT / "run_capture_sentinel.py"),
                    "--sentinel",
                    "DONE",
                    "--format",
                    "json",
                    "--output",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    "print('work'); print('DONE {\\\"pass\\\": true}')",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("work", completed.stdout)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"pass": True})

    def test_duplicate_sentinel_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "result.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PUBLICATION_ROOT / "run_capture_sentinel.py"),
                    "--sentinel",
                    "DONE",
                    "--format",
                    "json",
                    "--output",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    "print('DONE {}'); print('DONE {}')",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
