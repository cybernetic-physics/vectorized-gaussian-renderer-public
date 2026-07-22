from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from publication.run_pytest_gate import junit_checks


class PytestGateTests(unittest.TestCase):
    def test_junit_counts_and_required_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "junit.xml"
            path.write_text(
                '<testsuite tests="3" failures="0" errors="0" skipped="1">'
                '<testcase classname="tests.a" name="critical"/>'
                '<testcase classname="tests.a" name="ordinary"/>'
                '<testcase classname="tests.a" name="optional"><skipped/></testcase>'
                '</testsuite>',
                encoding="utf-8",
            )
            checks = junit_checks(path, ["critical"])
            self.assertEqual(checks["collected"], 3)
            self.assertEqual(checks["passed"], 2)
            self.assertEqual(checks["skipped_required"], 0)
            self.assertEqual(checks["required_tests"], ["tests.a::critical"])

    def test_required_fragment_must_be_unique_and_unskipped_is_observable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "junit.xml"
            path.write_text(
                '<testsuite tests="2" failures="0" errors="0" skipped="1">'
                '<testcase classname="tests.a" name="critical"><skipped/></testcase>'
                '<testcase classname="tests.b" name="critical"/>'
                '</testsuite>',
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                junit_checks(path, ["critical"])
            checks = junit_checks(path, ["tests.a::critical"])
            self.assertEqual(checks["skipped_required"], 1)


if __name__ == "__main__":
    unittest.main()
