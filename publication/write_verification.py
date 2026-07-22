#!/usr/bin/env python3
"""Write publication/verification.json from a declarative post-result spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from aggregate_verification import write_verification_from_spec


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    result = write_verification_from_spec(args.spec, args.output)
    print(
        "PUBLICATION_VERIFICATION_WRITTEN "
        + json.dumps(
            {
                "benchmark_commit": result["benchmark"]["commit"],
                "gate_count": len(result["gates"]),
                "output": str(args.output.resolve()),
                "pass": result["pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
