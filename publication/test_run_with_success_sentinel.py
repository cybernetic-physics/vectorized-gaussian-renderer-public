from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PUBLICATION_ROOT = Path(__file__).resolve().parent


class SuccessSentinelTests(unittest.TestCase):
    def run_case(self, child: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(PUBLICATION_ROOT / "run_with_success_sentinel.py"),
                "--sentinel",
                "PUBLICATION_OK",
                "--",
                sys.executable,
                "-c",
                child,
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_success_emits_one_sentinel_after_child_output(self) -> None:
        completed = self.run_case("print('work')")
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout.splitlines(), ["work", "PUBLICATION_OK"])

    def test_failure_emits_no_sentinel(self) -> None:
        completed = self.run_case("print('work'); raise SystemExit(7)")
        self.assertEqual(completed.returncode, 7)
        self.assertNotIn("PUBLICATION_OK", completed.stdout)

    def test_child_cannot_spoof_reserved_sentinel(self) -> None:
        completed = self.run_case("print('PUBLICATION_OK')")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout.count("PUBLICATION_OK"), 1)


if __name__ == "__main__":
    unittest.main()
