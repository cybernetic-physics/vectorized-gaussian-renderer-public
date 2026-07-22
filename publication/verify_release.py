#!/usr/bin/env python3
"""Canonical offline verifier for the final relocated publication bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

PUBLICATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PUBLICATION_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
EVALUATION_ROOT = SRC_ROOT / "isaacsim_gaussian_renderer" / "evaluation"
for path in (
    str(PROJECT_ROOT),
    str(EVALUATION_ROOT),
    str(PUBLICATION_ROOT),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from aggregate_verification import verify_aggregate_verification  # noqa: E402
from evidence_bundle import (  # noqa: E402
    resolve_artifact_record,
    verify_deterministic_archive,
    verify_evidence_bundle,
)
from scan_public_artifact import scan_public_artifact  # noqa: E402


CLAIM_VALIDATOR_LOGICAL_PATH = "publication/validator/verify_claim_ledger.py"
CLAIM_SENTINEL = "PUBLICATION_CLAIM_LEDGER_VERIFIED "


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_claim_validator(manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    validator = PUBLICATION_ROOT / "verify_claim_ledger.py"
    if validator.is_symlink() or not validator.is_file():
        raise RuntimeError("The canonical claim validator is missing or is a symlink.")
    matches = [
        item
        for item in manifest.get("publication_artifacts", [])
        if isinstance(item, dict) and item.get("logical_path") == CLAIM_VALIDATOR_LOGICAL_PATH
    ]
    if len(matches) != 1:
        raise RuntimeError("The bundle must inventory exactly one canonical claim validator.")
    bundled = matches[0]
    identity = {
        "bytes": validator.stat().st_size,
        "sha256": file_sha256(validator),
    }
    if bundled.get("bytes") != identity["bytes"] or bundled.get("sha256") != identity["sha256"]:
        raise RuntimeError("The executable claim validator differs from the bundle-pinned validator.")
    return validator, {
        **identity,
        "logical_path": CLAIM_VALIDATOR_LOGICAL_PATH,
    }


def parse_claim_validator_stdout(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    if len(lines) != 1 or not lines[0].startswith(CLAIM_SENTINEL):
        raise RuntimeError("Claim validator output is not the single exact structured sentinel line.")
    encoded = lines[0][len(CLAIM_SENTINEL) :]
    try:
        receipt = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise RuntimeError("Claim validator sentinel payload is not JSON.") from error
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "publication-claim-ledger-verification-v1"
        or receipt.get("pass") is not True
        or receipt.get("root_mode") != "bundle"
    ):
        raise RuntimeError("Claim validator returned a malformed or non-passing receipt.")
    return receipt


def require_matching_claim_artifacts(
    claim_receipt: dict[str, Any],
    aggregate_receipt: dict[str, Any],
) -> None:
    """Require the claim layer to bind the exact aggregate-proven objects."""

    for role in ("verification", "summary", "ablation"):
        if claim_receipt.get(role) != aggregate_receipt.get(role):
            raise RuntimeError(
                f"Claim ledger and aggregate verification bind different {role} artifacts."
            )


def scan_public_bundle_objects(
    bundle_root: Path,
    manifest: dict[str, Any],
) -> int:
    """Privacy-scan every relocated object, including binary profiler data."""

    raw_objects = manifest.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise RuntimeError("Verified bundle has no objects to privacy-scan.")
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, dict):
            raise RuntimeError(f"Verified bundle object {index} is malformed.")
        try:
            resolved = resolve_artifact_record(
                raw,
                bundle_root=bundle_root,
                label=f"Public bundle object {index}",
            )
            observed = scan_public_artifact(Path(resolved["path"]))
        except (FileNotFoundError, OSError, ValueError) as error:
            raise RuntimeError(
                f"Public bundle object {index} failed privacy scanning: {error}"
            ) from error
        expected = {"bytes": resolved["bytes"], "sha256": resolved["sha256"]}
        if observed != expected:
            raise RuntimeError(
                f"Public bundle object {index} changed during privacy scanning."
            )
    return len(raw_objects)


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    return parser.parse_args(arguments)


def verify_publication_release(
    *,
    bundle_root: Path,
    archive_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the complete offline release boundary and return its verified manifest."""

    manifest = verify_evidence_bundle(bundle_root)
    aggregate = verify_aggregate_verification(bundle_root)
    archive = verify_deterministic_archive(bundle_root, archive_path)
    validator, validator_identity = pinned_claim_validator(manifest)
    completed = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--root",
            str(bundle_root.resolve(strict=True)),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Strict claim-ledger verification failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    claim_receipt = parse_claim_validator_stdout(completed.stdout)
    if completed.stderr.strip():
        raise RuntimeError("Claim validator emitted unexpected stderr: " + completed.stderr.strip())
    require_matching_claim_artifacts(claim_receipt, aggregate)
    scan_public_bundle_objects(bundle_root, manifest)
    result: dict[str, Any] = {
        "schema_version": "publication-release-verification-v1",
        "pass": True,
        "manifest_id": manifest["manifest_id"],
        "aggregate": aggregate,
        "archive": archive,
        "claim_ledger": claim_receipt,
        "claim_validator": validator_identity,
    }
    return manifest, result


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    _manifest, result = verify_publication_release(
        bundle_root=args.bundle_root,
        archive_path=args.archive,
    )
    print("PUBLICATION_RELEASE_VERIFIED " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
