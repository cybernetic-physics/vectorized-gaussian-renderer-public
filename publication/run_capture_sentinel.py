#!/usr/bin/env python3
"""Run a command, stream its combined log, and freeze one sentinel payload."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel", required=True)
    parser.add_argument("--format", choices=("json", "python"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def decode_payload(encoded: str, payload_format: str) -> dict[str, Any]:
    try:
        value = json.loads(encoded) if payload_format == "json" else ast.literal_eval(encoded)
    except (json.JSONDecodeError, SyntaxError, ValueError) as error:
        raise ValueError("Sentinel payload is malformed.") from error
    if not isinstance(value, dict):
        raise ValueError("Sentinel payload is not an object.")
    return value


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


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
    payloads: list[str] = []
    prefix = args.sentinel + " "
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith(prefix):
            payloads.append(line[len(prefix) :].rstrip("\r\n"))
    returncode = process.wait()
    if returncode != 0:
        raise SystemExit(returncode)
    if len(payloads) != 1:
        raise RuntimeError(
            f"Expected exactly one complete {args.sentinel} payload line; observed {len(payloads)}."
        )
    write_new_json(args.output, decode_payload(payloads[0], args.format))


if __name__ == "__main__":
    main()
