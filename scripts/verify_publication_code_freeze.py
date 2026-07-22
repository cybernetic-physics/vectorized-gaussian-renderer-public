#!/usr/bin/env python3
"""Prove a publication commit changed prose/artifacts, not benchmark code."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_PUBLICATION_PATHS = {
    "post.md",
    "PUBLICATION_READINESS.md",
    "README.md",
    "RESULTS.md",
}
ALLOWED_PUBLICATION_PREFIXES = (
    "publication/",
    "experiments/flashgs_matched/results/",
)
LOAD_BEARING_PATHS = (
    "src",
    "benchmarks",
    "scripts",
    "exts",
    "tests",
    "patches",
    "pyproject.toml",
    "uv.lock",
    "BENCHMARK_PROTOCOL.md",
    "experiments/flashgs_matched/BENCHMARK_CONTRACT.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-ref", required=True)
    parser.add_argument("--publication-ref", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-untagged-benchmark",
        action="store_true",
        help="Diagnostic only; publication acceptance requires a tag.",
    )
    return parser.parse_args()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def publication_path_allowed(path: str) -> bool:
    return path in ALLOWED_PUBLICATION_PATHS or path.startswith(
        ALLOWED_PUBLICATION_PREFIXES
    )


def tree_fingerprint(ref: str) -> dict[str, object]:
    raw = git("ls-tree", "-r", ref, "--", *LOAD_BEARING_PATHS)
    records = sorted(line for line in raw.splitlines() if line)
    canonical = "\n".join(records).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "record_count": len(records),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    benchmark_commit = git("rev-parse", f"{args.benchmark_ref}^{{commit}}")
    publication_commit = git("rev-parse", f"{args.publication_ref}^{{commit}}")
    tags = [
        value
        for value in git("tag", "--points-at", benchmark_commit).splitlines()
        if value
    ]
    raw_diff = git(
        "diff",
        "--name-status",
        "--find-renames",
        benchmark_commit,
        publication_commit,
    )
    changes: list[dict[str, object]] = []
    rejected: list[str] = []
    for line in raw_diff.splitlines():
        fields = line.split("\t")
        status = fields[0]
        paths = fields[1:]
        allowed = bool(paths) and all(publication_path_allowed(path) for path in paths)
        changes.append({"status": status, "paths": paths, "allowed": allowed})
        if not allowed:
            rejected.extend(paths)
    benchmark_tree = tree_fingerprint(benchmark_commit)
    publication_tree = tree_fingerprint(publication_commit)
    tagged = bool(tags)
    passed = bool(
        not rejected
        and benchmark_tree["sha256"] == publication_tree["sha256"]
        and (tagged or args.allow_untagged_benchmark)
    )
    report = {
        "schema_version": "publication-code-freeze-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_ref": args.benchmark_ref,
        "benchmark_commit": benchmark_commit,
        "benchmark_tags": tags,
        "publication_ref": args.publication_ref,
        "publication_commit": publication_commit,
        "allowed_publication_paths": sorted(ALLOWED_PUBLICATION_PATHS),
        "allowed_publication_prefixes": list(ALLOWED_PUBLICATION_PREFIXES),
        "changes": changes,
        "rejected_paths": sorted(set(rejected)),
        "benchmark_load_bearing_tree": benchmark_tree,
        "publication_load_bearing_tree": publication_tree,
        "load_bearing_tree_equal": (
            benchmark_tree["sha256"] == publication_tree["sha256"]
        ),
        "tag_required": not args.allow_untagged_benchmark,
        "pass": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PUBLICATION_CODE_FREEZE "
        + json.dumps(
            {"output": str(args.output), "pass": passed}, sort_keys=True
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
