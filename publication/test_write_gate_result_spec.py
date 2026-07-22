from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from publication.write_gate_result_spec import build_spec, parse_evidence


class GateResultSpecTests(unittest.TestCase):
    def test_builds_exact_spec_and_resolves_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checks = root / "checks.json"
            source = root / "source.json"
            native = root / "native.so"
            evidence = root / "raw.json"
            checks.write_text('{"a": true, "b": true, "c": true}', encoding="utf-8")
            source.write_text("{}", encoding="utf-8")
            native.write_bytes(b"native")
            evidence.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                gate_id="native-cuda-smoke",
                checks=checks,
                source_manifest=source,
                native_extension=native,
                scene_sha256="a" * 64,
                gpu_uuid="GPU-test",
                evidence=[("raw_result", str(evidence))],
            )
            result = build_spec(args)
            self.assertEqual(result["gate_id"], "native-cuda-smoke")
            self.assertEqual(result["evidence"], {"raw_result": str(evidence.resolve())})

    def test_evidence_argument_rejects_unsafe_or_missing_key(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_evidence("missing")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_evidence("bad-key=/tmp/x")


if __name__ == "__main__":
    unittest.main()
