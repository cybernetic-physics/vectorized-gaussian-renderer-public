#!/usr/bin/env python3
"""Stream a command and emit one exact success sentinel only on exit zero."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterable


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if any(character in args.sentinel for character in "\r\n") or not args.sentinel:
        parser.error("--sentinel must be one non-empty line")
    return args


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
    existing = 0
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.rstrip("\r\n") == args.sentinel:
            existing += 1
    returncode = process.wait()
    if returncode != 0:
        raise SystemExit(returncode)
    if existing:
        raise RuntimeError(
            f"Child unexpectedly emitted the reserved sentinel {args.sentinel!r}."
        )
    print(args.sentinel, flush=True)


if __name__ == "__main__":
    main()
