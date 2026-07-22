#!/usr/bin/env python3
"""Write a secret-free source/build provenance manifest for an evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def canonical_json_bytes(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def command(root: Path, *args: str) -> str:
    return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional build/cache root to inventory. May be repeated; paths "
            "are recorded under stable native-root-N labels rather than as "
            "absolute machine paths."
        ),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    branch = command(root, "git", "branch", "--show-current") or None
    head = command(root, "git", "rev-parse", "HEAD")
    status = command(root, "git", "status", "--short", "--untracked-files=all")
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        stderr=subprocess.STDOUT,
    )
    tracked = [
        line
        for line in subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=root
        )
        .decode("utf-8")
        .split("\0")
        if line
    ]
    tracked_source = {
        relative: {
            "bytes": (root / relative).stat().st_size,
            "sha256": file_sha256(root / relative),
        }
        for relative in tracked
        if (root / relative).is_file()
    }
    untracked = [
        line
        for line in command(root, "git", "ls-files", "--others", "--exclude-standard").splitlines()
        if line
    ]
    untracked_source = {
        relative: {
            "bytes": (root / relative).stat().st_size,
            "sha256": file_sha256(root / relative),
        }
        for relative in untracked
        if (root / relative).is_file()
    }
    ignored = [
        line
        for line in command(root, "git", "ls-files", "--others", "--ignored", "--exclude-standard").splitlines()
        if line.startswith(("outputs/", "profiles/", "build/"))
    ]
    native = {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and any(part in {"build", "torch_extensions"} for part in path.parts)
        and path.suffix in {".so", ".dylib", ".dll", ".a"}
    }
    for root_index, native_root in enumerate(args.native_root):
        native_root = native_root.resolve()
        if not native_root.exists():
            continue
        for path in sorted(native_root.rglob("*")):
            if not path.is_file() or path.suffix not in {".so", ".dylib", ".dll", ".a"}:
                continue
            key = f"native-root-{root_index}/{path.relative_to(native_root).as_posix()}"
            native[key] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    source_tree_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {"tracked": tracked_source, "untracked": untracked_source}
        )
    ).hexdigest()
    payload = {
        "schema_version": "renderer-source-manifest-v2",
        "branch": branch,
        "detached_head": branch is None,
        "head": head,
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_bytes": len(diff),
        "tracked_source_files": tracked_source,
        "relevant_untracked_source": untracked,
        "relevant_untracked_source_files": untracked_source,
        "source_tree_sha256": source_tree_sha256,
        "ignored_generated_inventory": ignored,
        "native_extension_binaries": native,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
