#!/usr/bin/env python3
"""Run one real gate command, then write its immutable result specification.

The bounded GPU executor normalizes a result specification after its child
returns.  Most real gates create their evidence during that child process, so
the specification cannot honestly be prepared in advance.  This adapter keeps
the complete child argv visible in the bounded command record, streams its
combined output unchanged, and writes the specification only after exit zero
and after every declared evidence file exists and hashes successfully.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from write_gate_result_spec import build_spec, parse_evidence


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--native-extension", type=Path, required=True)
    parser.add_argument("--scene-sha256", required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--evidence", type=parse_evidence, action="append", required=True)
    parser.add_argument("--result-spec", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command or any(not isinstance(token, str) or not token for token in args.command):
        parser.error("a non-empty command is required after --")
    return args


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.result_spec.exists() or args.result_spec.is_symlink():
        raise FileExistsError(f"Refusing to overwrite result specification: {args.result_spec}")

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

    specification = build_spec(args)
    write_new(args.result_spec, canonical_json_bytes(specification))
    print(
        "PUBLICATION_GATE_RESULT_SPEC_OK "
        + json.dumps(
            {"gate_id": args.gate_id, "output": str(args.result_spec.resolve())},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
