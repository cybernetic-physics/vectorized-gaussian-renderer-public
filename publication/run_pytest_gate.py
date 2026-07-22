#!/usr/bin/env python3
"""Run the exact unit suite and derive its normalized gate spec from JUnit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit-xml", type=Path, required=True)
    parser.add_argument("--result-spec", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--native-extension", type=Path, required=True)
    parser.add_argument("--scene-sha256", required=True)
    parser.add_argument("--required-test-fragment", action="append", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a pytest command is required after --")
    return args


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def junit_checks(path: Path, fragments: list[str]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Pytest did not produce a regular JUnit XML file.")
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        raise RuntimeError("JUnit XML has no test suites.")
    totals = {
        field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    cases: dict[str, ET.Element] = {}
    for case in root.findall(".//testcase"):
        identifier = f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}".strip(":")
        if identifier in cases:
            raise RuntimeError(f"JUnit repeats testcase {identifier!r}.")
        cases[identifier] = case
    required: list[str] = []
    for fragment in fragments:
        matches = [identifier for identifier in cases if fragment in identifier]
        if len(matches) != 1:
            raise RuntimeError(
                f"Required-test fragment {fragment!r} matched {len(matches)} JUnit cases."
            )
        required.append(matches[0])
    skipped_required = sum(
        1 for identifier in required if cases[identifier].find("skipped") is not None
    )
    return {
        "collected": totals["tests"],
        "passed": totals["tests"] - totals["skipped"] - totals["failures"] - totals["errors"],
        "failures": totals["failures"],
        "errors": totals["errors"],
        "skipped_required": skipped_required,
        "required_tests": required,
    }


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    process = subprocess.Popen(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    returncode = process.wait()
    if returncode != 0:
        raise SystemExit(returncode)
    checks = junit_checks(args.junit_xml, args.required_test_fragment)
    if checks["failures"] or checks["errors"] or checks["skipped_required"]:
        raise RuntimeError("Unit contract failed according to JUnit evidence.")
    specification = {
        "schema_version": "publication-gate-result-spec-v1",
        "gate_id": "python-unit-contract",
        "identity": {
            "gpu_uuid": None,
            "native_extension": str(args.native_extension.resolve(strict=True)),
            "scene_sha256": args.scene_sha256,
            "source_manifest": str(args.source_manifest.resolve(strict=True)),
        },
        "checks": checks,
        "evidence": {"junit_xml": str(args.junit_xml.resolve(strict=True))},
    }
    write_new(args.result_spec, canonical_json_bytes(specification))
    print("PYTEST_PUBLICATION_OK", flush=True)


if __name__ == "__main__":
    main()
