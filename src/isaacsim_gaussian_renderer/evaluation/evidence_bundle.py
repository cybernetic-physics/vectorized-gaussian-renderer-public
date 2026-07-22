"""Relocatable, content-addressed matched-benchmark evidence bundles.

The raw benchmark records intentionally preserve the paths used on the
measurement host.  Those paths are provenance, not bundle locators.  This
module resolves every artifact by its recorded byte count and SHA-256 when a
bundle root is supplied, and builds a closed inventory around those objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

EVIDENCE_BUNDLE_SCHEMA = "flashgs-matched-evidence-bundle-v1"
PUBLICATION_INVENTORY_SCHEMA = "flashgs-matched-publication-inventory-v1"
CLAIM_LEDGER_SCHEMA = "flashgs-matched-claim-ledger-v1"
SEMANTIC_VALIDATION_SCHEMA = "flashgs-matched-semantic-validation-v1"
CAMERA_CONTRACT_SCHEMA = "camera-trajectory-v1"
CODE_FREEZE_SCHEMA = "publication-code-freeze-v1"
CAPACITY_AMENDMENT_SCHEMA = "flashgs-matched-l4-capacity-amendment-v1"
FIDELITY_SCHEMA = "flashgs-matched-fidelity-v4"
FLASHGS_ADAPTER_ATTESTATION_SCHEMA = "flashgs-adapter-attestation-v3"
FLASHGS_MATCHED_PORT_CLASSIFICATION = {
    "baseline_name": "FlashGS-derived matched-contract port",
    "integration_only": False,
    "upstream_faithful": False,
    "retained_upstream_topology": [
        "warp-cooperative tile-key emission",
        "CUB tile-and-depth radix ordering",
        "sorted tile-range discovery",
        "16x16 tile compositor with cooperative Gaussian loads",
    ],
    "service_integration_changes": [
        "canonical GPU-resident scene and camera tensors",
        "active PyTorch CUDA stream execution",
        "reusable outputs and workspace",
        "fixed-capacity sorting without a measured-region count readback",
        "one native pipeline invocation per camera",
    ],
    "matched_equation_changes": [
        "pinned-gsplat projection and culling conventions",
        "3.33-sigma opacity-contracted support and matched radius conventions",
        "zero covariance epsilon and half-pixel sample centers",
        "degree-zero canonical RGB instead of upstream spherical-harmonic evaluation",
        "matched alpha cap, per-splat rejection, transmittance termination, and background",
    ],
    "full_sensor_extension_changes": [
        "float32 linear RGB output",
        "accumulated alpha",
        "expected camera-space depth",
        "strongest-contributor semantic ID",
    ],
    "correctness_repairs": [
        "aligned cooperative feature-load guards for one- and two-entry tile tails",
    ],
    "allowed_publication_label": "FlashGS-derived matched-contract port",
    "forbidden_publication_labels": [
        "upstream FlashGS",
        "minimally adapted FlashGS",
        "integration-only FlashGS",
    ],
}
ORACLE_SCHEMA = "flashgs-matched-gsplat-oracle-v4"
PROFILE_SCHEMA = "flashgs-matched-profile-v3"
RENDERER_RUN_SCHEMA = "flashgs-matched-renderer-run-v4"
SOURCE_MANIFEST_SCHEMA = "renderer-source-manifest-v2"
SUMMARY_SCHEMA = "flashgs-matched-summary-v4"

CAPACITY_AMENDMENT_ARTIFACTS = {
    "failed_source_manifest": {
        "bytes": 70_337,
        "sha256": "fc47eb697ed9a7ba985cc1324613b68c3129aed8c0db3a1d9ba7ae78ee0a6650",
    },
    "matrix_invocation": {
        "bytes": 3_097,
        "sha256": "892f0f9b5bed8e22a31caf8d69da564efa8acbed8e2edfe0273aabedc2a1efec",
    },
    "matrix_launch_occupancy": {
        "bytes": 3_335,
        "sha256": "fd8c02e848064f14f1a4c5fb9a2d7d5e724686829b5c59eabd8e51af28f6d46e",
    },
    "capacity_run_occupancy": {
        "bytes": 3_649,
        "sha256": "16281a1d19316297f9477ad579f6442c4696471ed260c8b32d757722ba4b7c2a",
    },
    "capacity_command": {
        "bytes": 1_080,
        "sha256": "36f91369419421a371a4c37148682272d8069bed79ffcc273d23159d3d503e41",
    },
    "capacity_log": {
        "bytes": 2_419,
        "sha256": "d245342ca12f2553d94765f3572b4067cb47064a1299573a6354641f56cfdf4c",
    },
    "matrix_driver_log": {
        "bytes": 2_292,
        "sha256": "b2546ea379b0910215f9fae96a93b920b9f9ef37052ecc29b420008f0120c544",
    },
    "matrix_exit_code": {
        "bytes": 1,
        "sha256": "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
    },
}

REQUIRED_PUBLICATION_ROLES = frozenset(
    {
        "adapter-attestation",
        "build-attestation",
        "camera-contract",
        "canonical-summary",
        "canonical-summary-markdown",
        "capacity-amendment",
        "capacity-calibration",
        "claim-ledger",
        "code-freeze",
        "command-record",
        "command-log",
        "fidelity-csv",
        "fidelity-json",
        "fidelity-summary",
        "flashgs-demand-survey",
        "machine-provenance",
        "matrix-invocation",
        "occupancy-evidence",
        "oracle-capture",
        "oracle-manifest",
        "profile-csv",
        "profile-control",
        "profile-raw",
        "profile-sqlite",
        "profile-summary",
        "profile-wrapper",
        "raw-renderer-run",
        "semantic-validator",
        "source-manifest",
        "summary-command",
        "verification-record",
    }
)
ALLOWED_PUBLICATION_ROLES = REQUIRED_PUBLICATION_ROLES | {
    "supporting-artifact",
}
REQUIRED_EXTERNAL_ROLES = frozenset({"scene-dataset"})
ALLOWED_EXTERNAL_ROLES = REQUIRED_EXTERNAL_ROLES | {
    "external-dependency",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_PRIVATE_KEY_RE = re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
_ASSIGNED_SECRET_RE = re.compile(
    rb"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|aws[_-]?secret[_-]?access[_-]?key|"
    rb"client[_-]?secret|cloudflare[_-]?api[_-]?token|password|r2[_-]?secret[_-]?access[_-]?key|"
    rb"secret[_-]?key|upload[_-]?token|wrangler[_-]?token)\s*[\"']?\s*[:=]\s*"
    rb"[\"']?[A-Za-z0-9_./+=:@-]{12,}"
)
_AUTHORIZATION_RE = re.compile(rb"(?i)authorization\s*:\s*(?:basic|bearer)\s+[A-Za-z0-9_./+=:@-]{12,}")
_TOKEN_PREFIX_RE = re.compile(
    rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|"
    rb"xox[baprs]-[A-Za-z0-9-]{20,}|A(?:KI|SI)A[0-9A-Z]{16})"
)
_SECRET_PATH_COMPONENTS = frozenset(
    {
        ".aws",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ssh",
        ".env",
        "__pycache__",
        "credentials",
        "secrets",
    }
)
_SECRET_BASENAMES = frozenset(
    {
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        ".ds_store",
        "wrangler.toml",
    }
)

_ROLE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "camera-contract": (".json",),
    "capacity-amendment": (".json",),
    "capacity-calibration": (".json",),
    "canonical-summary": (".json",),
    "canonical-summary-markdown": (".md",),
    "claim-ledger": (".json",),
    "code-freeze": (".json",),
    "command-record": (".json",),
    "command-log": (".log", ".txt"),
    "fidelity-csv": (".csv",),
    "fidelity-json": (".json",),
    "fidelity-summary": (".json",),
    "flashgs-demand-survey": (".json",),
    "machine-provenance": (".json",),
    "matrix-invocation": (".json",),
    "occupancy-evidence": (".json",),
    "oracle-capture": (".npz",),
    "oracle-manifest": (".json",),
    "profile-csv": (".csv",),
    "profile-control": (".json",),
    "profile-raw": (".nsys-rep",),
    "profile-sqlite": (".sqlite", ".sqlite3"),
    "profile-summary": (".json",),
    "profile-wrapper": (".json",),
    "raw-renderer-run": (".json",),
    "source-manifest": (".json",),
    "summary-command": (".json",),
    "adapter-attestation": (".json",),
    "build-attestation": (".json",),
    "semantic-validator": (".py",),
    "verification-record": (".json",),
}

MATCHED_BATCHES = (1, 8, 32, 64, 128, 256, 512, 1024)
MATCHED_RENDERERS = ("custom", "flashgs")
MATCHED_CONTRACTS = ("full", "rgb")


def capacity_amendment_relative_path(name: str) -> str:
    if name == "matrix_exit_code":
        suffix = ".txt"
    elif name in {"capacity_log", "matrix_driver_log"}:
        suffix = ".log"
    else:
        suffix = ".json"
    return f"publication/capacity-amendment/{name}{suffix}"


def _matched_required_paths() -> dict[str, frozenset[str]]:
    required: dict[str, set[str]] = {}

    def add(path: str, *roles: str) -> None:
        required.setdefault(path, set()).update(roles)

    add("summary.json", "canonical-summary")
    add("summary.md", "canonical-summary-markdown")
    add("provenance/source-manifest.json", "source-manifest")
    add("provenance/matrix-invocation.json", "matrix-invocation")
    add("provenance/matrix-launch-occupancy.json", "occupancy-evidence")
    add(
        "provenance/flashgs-adapter-attestation.json",
        "adapter-attestation",
    )
    add(
        "provenance/gsplat-build-attestation.json",
        "build-attestation",
    )
    add("publication/claim-ledger.json", "claim-ledger")
    add("publication/capacity-amendment.json", "capacity-amendment")
    for amendment_name in CAPACITY_AMENDMENT_ARTIFACTS:
        add(
            capacity_amendment_relative_path(amendment_name),
            "supporting-artifact",
        )
    add("publication/code-freeze.json", "code-freeze")
    add("publication/machine-provenance.json", "machine-provenance")
    add("publication/verification.json", "verification-record")
    add(
        "publication/validator/summarize_flashgs_matched.py",
        "semantic-validator",
    )

    expected_logs = {
        "flashgs-adapter-attestation",
        "generate-contracts",
        "summarize",
    }
    for batch in MATCHED_BATCHES:
        add(f"contracts/b{batch}.json", "camera-contract")
        add(f"oracle/b{batch}.npz", "oracle-capture")
        add(f"oracle/b{batch}.manifest.json", "oracle-manifest")
        add(f"oracle/b{batch}.camera-bundle.json", "supporting-artifact")
        add(f"oracle/b{batch}.node-occupancy.json", "occupancy-evidence")
        expected_logs.add(f"gsplat-oracle-b{batch}")
        add(
            f"capacity/custom/b{batch}.json",
            "capacity-calibration",
        )
        add(
            f"capacity/custom/b{batch}.node-occupancy.json",
            "occupancy-evidence",
        )
        expected_logs.add(f"capacity-custom-b{batch}")
        for renderer in MATCHED_RENDERERS:
            for contract in MATCHED_CONTRACTS:
                run_stem = f"runs/{renderer}/{contract}/b{batch}"
                add(f"{run_stem}.json", "raw-renderer-run")
                add(f"{run_stem}.fidelity-capture.npz", "supporting-artifact")
                add(f"{run_stem}.node-occupancy.json", "occupancy-evidence")
                fidelity_stem = f"fidelity/{renderer}/{contract}/b{batch}"
                add(
                    f"{fidelity_stem}/matched-fidelity-summary.json",
                    "fidelity-summary",
                )
                add(f"{fidelity_stem}/fidelity_report.json", "fidelity-json")
                add(f"{fidelity_stem}/fidelity_report.csv", "fidelity-csv")
                expected_logs.add(f"{renderer}-{contract}-b{batch}")
                expected_logs.add(f"fidelity-{renderer}-{contract}-b{batch}")
    add(
        "capacity/flashgs/b1024-demand-survey.json",
        "flashgs-demand-survey",
    )
    add(
        "capacity/flashgs/b1024-demand-survey.node-occupancy.json",
        "occupancy-evidence",
    )
    expected_logs.add("flashgs-b1024-demand-survey")
    for trial in (2, 3):
        for renderer in MATCHED_RENDERERS:
            for contract in MATCHED_CONTRACTS:
                repeat_stem = f"repeats/{renderer}/{contract}/b128-trial{trial}"
                add(f"{repeat_stem}.json", "raw-renderer-run")
                add(f"{repeat_stem}.node-occupancy.json", "occupancy-evidence")
                expected_logs.add(f"repeat-{renderer}-{contract}-b128-trial{trial}")
    for batch in (MATCHED_BATCHES[0], MATCHED_BATCHES[-1]):
        for renderer in MATCHED_RENDERERS:
            for contract in MATCHED_CONTRACTS:
                config = f"flashgs-matched-{renderer}-{contract}-b{batch}"
                profile_root = f"profiles/{config}"
                add(f"{profile_root}/profile-summary.json", "profile-summary")
                add(f"{profile_root}/profile-control.json", "profile-control")
                add(
                    f"{profile_root}/profile-control.node-occupancy.json",
                    "occupancy-evidence",
                )
                add(f"{profile_root}/wrapper-evidence.json", "profile-wrapper")
                add(f"{profile_root}/occupancy-preflight.json", "occupancy-evidence")
                add(f"{profile_root}/nsys-summary.json", "supporting-artifact")
                add(f"{profile_root}/nsys-profile-exit-status.txt", "supporting-artifact")
                add(f"{profile_root}/nsys-version.txt", "supporting-artifact")
                add(f"{profile_root}/nsys/{config}.nsys-rep", "profile-raw")
                add(f"{profile_root}/nsys/{config}.sqlite", "profile-sqlite")
                for report in (
                    "cuda_api_sum",
                    "cuda_gpu_kern_sum",
                    "cuda_gpu_mem_size_sum",
                    "cuda_gpu_mem_time_sum",
                    "nvtx_sum",
                ):
                    add(
                        f"{profile_root}/nsys/stats/{config}_{report}.csv",
                        "profile-csv",
                    )
                expected_logs.add(f"profile-{renderer}-{contract}-b{batch}")
    for stem in expected_logs:
        add(f"logs/{stem}.log", "command-log")
        roles = {"command-record"}
        if stem == "summarize":
            roles.add("summary-command")
        add(f"logs/{stem}.command.json", *sorted(roles))
    return {path: frozenset(roles) for path, roles in sorted(required.items())}


def file_sha256(path: str | Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity(record: dict[str, Any], *, label: str) -> tuple[str, int]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} is not an artifact record.")
    sha256 = record.get("sha256")
    byte_count = record.get("bytes")
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise ValueError(f"{label} has no canonical SHA-256.")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError(f"{label} has no valid byte count.")
    return sha256, byte_count


def artifact_object_path(sha256: str) -> str:
    """Return the only legal object path for a SHA-256 identity."""

    if _SHA256_RE.fullmatch(sha256) is None:
        raise ValueError("Artifact object SHA-256 is not canonical lowercase hex.")
    return f"objects/sha256/{sha256[:2]}/{sha256}"


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Unsafe {label}: {value!r}.")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise ValueError(f"Unsafe {label}: {value!r}.")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe {label}: {value!r}.")
    return value


def _assert_no_secret_path(path: str | Path, *, label: str) -> None:
    parts = [part.lower() for part in Path(path).parts]
    basename = parts[-1] if parts else ""
    if (
        basename in _SECRET_BASENAMES
        or basename == ".env"
        or basename.startswith(".env.")
        or any(part in _SECRET_PATH_COMPONENTS for part in parts)
    ):
        raise ValueError(f"{label} uses a secret-bearing path; refusing to package it.")


def _assert_regular_no_symlink(path: Path, *, boundary: Path | None = None) -> None:
    candidate = path.absolute()
    stop = boundary.absolute() if boundary is not None else None
    if stop is not None and not candidate.is_relative_to(stop):
        raise ValueError(f"Evidence path escapes its required boundary: {candidate}.")
    current = candidate
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing artifact: {candidate}.") from None
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symlinks are forbidden in evidence paths: {current}.")
        if stop is not None and current == stop:
            break
        parent = current.parent
        if parent == current or (stop is not None and not current.is_relative_to(stop)):
            break
        current = parent
    mode = candidate.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"Evidence artifact is not a regular file: {candidate}.")


def resolve_artifact_record(
    record: dict[str, Any],
    *,
    bundle_root: str | Path | None = None,
    label: str = "Artifact",
) -> dict[str, Any]:
    """Verify and locate an artifact record.

    Outside bundle mode, the recorded path is used.  In bundle mode, the
    recorded path is deliberately ignored and the object is resolved solely
    from ``sha256`` and ``bytes``.
    """

    sha256, byte_count = _identity(record, label=label)
    if bundle_root is None:
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{label} has no recorded path.")
        path = Path(raw_path)
        _assert_regular_no_symlink(path)
    else:
        raw_root = Path(bundle_root)
        if raw_root.is_symlink():
            raise ValueError("Evidence bundle root may not be a symlink.")
        root = raw_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Evidence bundle root is missing: {root}.")
        path = root.joinpath(*PurePosixPath(artifact_object_path(sha256)).parts)
        _assert_regular_no_symlink(path, boundary=root)
    if path.stat().st_size != byte_count or file_sha256(path) != sha256:
        raise ValueError(f"{label} content differs from its recorded hash/identity.")
    return {"path": str(path.resolve()), "bytes": byte_count, "sha256": sha256}


def _scan_secret_bytes(path: Path, *, textual: bool) -> None:
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            window = overlap + chunk
            if _PRIVATE_KEY_RE.search(window) or _TOKEN_PREFIX_RE.search(window) or _AUTHORIZATION_RE.search(window):
                raise ValueError(f"Secret material detected in {path.name}; refusing to package it.")
            if textual and _ASSIGNED_SECRET_RE.search(window):
                raise ValueError(f"Secret assignment detected in {path.name}; refusing to package it.")
            overlap = window[-512:]


def _looks_like_ply(path: str) -> bool:
    lower = PurePosixPath(path).name.lower()
    return lower.endswith(".ply") or ".ply." in lower


def _has_ply_payload(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(128).replace(b"\r\n", b"\n")
    return header.startswith(b"ply\nformat ")


def _media_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".json"):
        return "application/json"
    if lower.endswith(".md"):
        return "text/markdown"
    if lower.endswith(".csv"):
        return "text/csv"
    if lower.endswith((".log", ".txt")):
        return "text/plain"
    if lower.endswith((".sqlite", ".sqlite3")):
        return "application/vnd.sqlite3"
    if lower.endswith(".npz"):
        return "application/x-npz"
    return "application/octet-stream"


def _is_textual(path: str) -> bool:
    return _media_type(path).startswith("text/") or path.lower().endswith(
        (
            ".cpp",
            ".cu",
            ".cuh",
            ".diff",
            ".h",
            ".hpp",
            ".json",
            ".ninja",
            ".patch",
            ".py",
            ".sh",
            ".toml",
            ".yaml",
            ".yml",
        )
    )


def _read_json_document(path: Path) -> Any | None:
    with path.open("rb") as handle:
        prefix = handle.read(4096).lstrip()
    if not prefix.startswith((b"{", b"[")):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _json_pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _resolve_json_pointer(value: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError(f"Claim JSON field is not an RFC 6901 pointer: {pointer!r}.")
    current = value
    for raw_part in pointer[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", raw_part):
            raise ValueError(f"Claim JSON pointer has invalid escaping: {pointer!r}.")
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValueError(f"Claim JSON pointer does not resolve: {pointer!r}.")
    return current


def _artifact_records(value: Any, pointer: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        keys = set(value)
        if {"path", "bytes", "sha256"}.issubset(keys):
            yield pointer, value
        for key in sorted(value):
            child_pointer = f"{pointer}/{_json_pointer_part(str(key))}"
            yield from _artifact_records(value[key], child_pointer)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _artifact_records(item, f"{pointer}/{index}")


def _walk_regular_tree(root: Path) -> dict[str, Path]:
    if root.is_symlink():
        raise ValueError("Evidence input root may not be a symlink.")
    if not root.is_dir():
        raise FileNotFoundError(f"Evidence input root is missing: {root}.")
    files: dict[str, Path] = {}
    directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        base = Path(directory)
        if base != root:
            directories.add(base.relative_to(root).as_posix())
        for name in directory_names:
            candidate = base / name
            if candidate.is_symlink():
                raise ValueError(f"Symlinked evidence directory is forbidden: {candidate}.")
        for name in file_names:
            candidate = base / name
            _assert_regular_no_symlink(candidate, boundary=root)
            relative = candidate.relative_to(root).as_posix()
            _safe_relative_path(relative, label="evidence input path")
            files[relative] = candidate
    required_directories = {
        parent.as_posix()
        for relative in files
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    if directories != required_directories:
        extras = sorted(directories - required_directories)
        raise ValueError(f"Evidence tree contains empty or unexpected directories: {extras}.")
    return files


def _normalize_roles(value: Any, *, allowed: frozenset[str], label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must declare at least one role.")
    if any(not isinstance(role, str) or role not in allowed for role in value):
        raise ValueError(f"{label} declares an unknown role.")
    roles = sorted(set(value))
    if len(roles) != len(value):
        raise ValueError(f"{label} contains duplicate or unsorted roles.")
    return roles


def _normalize_external_dependency(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Malformed external dependency.")
    allowed_keys = {
        "bytes",
        "content_addressed_key",
        "dependency_id",
        "license",
        "media_type",
        "retrieval_manifest_url",
        "roles",
        "sha256",
    }
    required_keys = allowed_keys - {"retrieval_manifest_url"}
    if not required_keys.issubset(value) or set(value) - allowed_keys:
        raise ValueError("External dependency has missing or unknown fields.")
    dependency_id = value.get("dependency_id")
    if not isinstance(dependency_id, str) or _SLUG_RE.fullmatch(dependency_id) is None:
        raise ValueError("External dependency ID is not a safe slug.")
    sha256, byte_count = _identity(value, label=f"External dependency {dependency_id}")
    roles = _normalize_roles(
        value.get("roles"),
        allowed=ALLOWED_EXTERNAL_ROLES,
        label=f"External dependency {dependency_id}",
    )
    media_type = value.get("media_type")
    license_name = value.get("license")
    object_key = _safe_relative_path(
        value.get("content_addressed_key"),
        label=f"external dependency {dependency_id} key",
    )
    if sha256[:12] not in object_key:
        raise ValueError(f"External dependency {dependency_id} key is not content-addressed.")
    if not isinstance(media_type, str) or re.fullmatch(r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+", media_type) is None:
        raise ValueError(f"External dependency {dependency_id} lacks a media type.")
    if (
        not isinstance(license_name, str)
        or not license_name
        or len(license_name) > 128
        or any(ord(character) < 32 for character in license_name)
    ):
        raise ValueError(f"External dependency {dependency_id} lacks a license.")
    normalized: dict[str, Any] = {
        "bytes": byte_count,
        "content_addressed_key": object_key,
        "dependency_id": dependency_id,
        "license": license_name,
        "media_type": media_type,
        "roles": roles,
        "sha256": sha256,
    }
    retrieval_manifest_url = value.get("retrieval_manifest_url")
    if retrieval_manifest_url is not None:
        if not isinstance(retrieval_manifest_url, str):
            raise ValueError("External retrieval-manifest URL must be a string.")
        parsed = urlsplit(retrieval_manifest_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("External retrieval-manifest URL is not secret-safe HTTPS.")
        normalized["retrieval_manifest_url"] = retrieval_manifest_url
    return normalized


def _normalize_inventory(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Publication inventory must be a JSON object.")
    if payload.get("schema_version") != PUBLICATION_INVENTORY_SCHEMA:
        raise ValueError("Unexpected publication-inventory schema.")
    if set(payload) != {
        "artifacts",
        "bundle_name",
        "external_dependencies",
        "schema_version",
    }:
        raise ValueError("Publication inventory has missing or unknown fields.")
    bundle_name = payload.get("bundle_name")
    if not isinstance(bundle_name, str) or _SLUG_RE.fullmatch(bundle_name) is None:
        raise ValueError("Publication bundle name is not a safe slug.")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("Publication inventory has no artifacts.")
    artifacts: list[dict[str, Any]] = []
    paths: set[str] = set()
    observed_roles: set[str] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise ValueError("Malformed publication artifact inventory entry.")
        if set(raw) != {"bytes", "path", "roles", "sha256"}:
            raise ValueError("Publication artifact has missing or unknown fields.")
        relative = _safe_relative_path(raw.get("path"), label="publication artifact path")
        _assert_no_secret_path(relative, label="Publication artifact")
        if relative in paths:
            raise ValueError(f"Duplicate publication artifact path: {relative}.")
        if _looks_like_ply(relative):
            raise ValueError("Dataset PLY files must be external dependencies, not bundled artifacts.")
        roles = _normalize_roles(
            raw.get("roles"),
            allowed=ALLOWED_PUBLICATION_ROLES,
            label=f"Publication artifact {relative}",
        )
        for role in roles:
            suffixes = _ROLE_SUFFIXES.get(role)
            if suffixes is not None and not relative.lower().endswith(suffixes):
                raise ValueError(f"Publication role {role} has an invalid file type: {relative}.")
        sha256, byte_count = _identity(raw, label=f"Publication artifact {relative}")
        artifacts.append(
            {
                "bytes": byte_count,
                "path": relative,
                "roles": roles,
                "sha256": sha256,
            }
        )
        paths.add(relative)
        observed_roles.update(roles)
    missing_roles = sorted(REQUIRED_PUBLICATION_ROLES - observed_roles)
    if missing_roles:
        raise ValueError(f"Publication inventory is missing required roles: {missing_roles}.")
    role_counts = {role: sum(role in item["roles"] for item in artifacts) for role in ALLOWED_PUBLICATION_ROLES}
    expected_role_counts = {
        role: sum(role in roles for roles in _matched_required_paths().values()) for role in REQUIRED_PUBLICATION_ROLES
    }
    wrong_role_counts = {
        role: {
            "actual": role_counts.get(role, 0),
            "expected": expected,
        }
        for role, expected in expected_role_counts.items()
        if role_counts.get(role, 0) != expected
    }
    if wrong_role_counts:
        raise ValueError(f"Publication inventory role cardinalities differ: {wrong_role_counts}.")
    by_path = {item["path"]: item for item in artifacts}
    missing_paths: list[str] = []
    wrong_roles: list[str] = []
    for required_path, required_roles in _matched_required_paths().items():
        item = by_path.get(required_path)
        if item is None:
            missing_paths.append(required_path)
        elif not required_roles.issubset(item["roles"]):
            wrong_roles.append(f"{required_path}: expected {sorted(required_roles)}, found {item['roles']}")
    if missing_paths or wrong_roles:
        raise ValueError(
            "Publication inventory lacks the frozen matched-result layout; "
            f"missing={missing_paths}, role_mismatches={wrong_roles}."
        )

    raw_external = payload.get("external_dependencies")
    if not isinstance(raw_external, list) or not raw_external:
        raise ValueError("Publication inventory has no external dependencies.")
    external = [_normalize_external_dependency(item) for item in raw_external]
    dependency_ids = [item["dependency_id"] for item in external]
    identities = [(item["sha256"], item["bytes"]) for item in external]
    if len(set(dependency_ids)) != len(dependency_ids):
        raise ValueError("Publication inventory repeats an external dependency ID.")
    if len(set(identities)) != len(identities):
        raise ValueError("Publication inventory repeats an external dependency identity.")
    external_roles = {role for item in external for role in item["roles"]}
    missing_external_roles = sorted(REQUIRED_EXTERNAL_ROLES - external_roles)
    if missing_external_roles:
        raise ValueError(f"Publication inventory is missing external roles: {missing_external_roles}.")
    return {
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "bundle_name": bundle_name,
        "external_dependencies": sorted(external, key=lambda item: item["dependency_id"]),
        "schema_version": PUBLICATION_INVENTORY_SCHEMA,
    }


def load_publication_inventory(path: str | Path) -> dict[str, Any]:
    inventory_path = Path(path)
    _assert_no_secret_path(inventory_path, label="Publication inventory")
    _assert_regular_no_symlink(inventory_path)
    _scan_secret_bytes(inventory_path, textual=True)
    return _normalize_inventory(json.loads(inventory_path.read_text(encoding="utf-8")))


def _validate_role_documents(
    artifacts: list[dict[str, Any]],
    object_paths: dict[str, Path],
) -> None:
    artifacts_by_identity: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for artifact in artifacts:
        artifacts_by_identity.setdefault(
            (artifact["sha256"], artifact["bytes"]),
            [],
        ).append(artifact)
    json_roles = {
        "adapter-attestation",
        "build-attestation",
        "camera-contract",
        "canonical-summary",
        "capacity-amendment",
        "capacity-calibration",
        "claim-ledger",
        "code-freeze",
        "command-record",
        "fidelity-json",
        "fidelity-summary",
        "flashgs-demand-survey",
        "machine-provenance",
        "matrix-invocation",
        "occupancy-evidence",
        "oracle-manifest",
        "profile-control",
        "profile-summary",
        "profile-wrapper",
        "raw-renderer-run",
        "source-manifest",
        "summary-command",
        "verification-record",
    }
    for artifact in artifacts:
        path = object_paths[artifact["sha256"]]
        logical_path = artifact.get("logical_path") or artifact.get("path") or "<unknown>"
        roles = set(artifact["roles"])
        payload: Any | None = None
        if roles.intersection(json_roles):
            payload = _read_json_document(path)
            if not isinstance(payload, dict):
                raise ValueError(f"Publication JSON role is not a JSON object: {logical_path}.")
        if "canonical-summary" in roles and (
            payload.get("schema_version") != SUMMARY_SCHEMA
            or payload.get("pass") is not True
            or payload.get("scientific_pass") is not True
            or payload.get("headline_eligible") is not True
        ):
            raise ValueError("Canonical summary is not a passing headline-eligible v4 result.")
        if "capacity-amendment" in roles:
            failure = payload.get("failed_direct_protocol") or {}
            replacement = payload.get("replacement_protocol") or {}
            evidence = payload.get("evidence") or {}
            if (
                payload.get("schema_version") != CAPACITY_AMENDMENT_SCHEMA
                or payload.get("pass") is not True
                or payload.get("failed_source_head") != "025bdaf2258894f477074dbd955653abcbfb3f62"
                or failure.get("logical_batch") != 512
                or failure.get("physical_batch") != 512
                or failure.get("capacity_headroom") != 1.05
                or failure.get("requested_backend_workspace_bytes") != 21_445_449_872
                or failure.get("timed_rows_completed") != 0
                or failure.get("failure_stage") != "custom-capacity-calibration-workspace-allocation"
                or replacement.get("capacity_headroom") != 1.05
                or replacement.get("alternative_chunk_sizes_swept") is not False
                or replacement.get("timing_observed_before_selection") is not False
                or replacement.get("custom_physical_batches")
                != {
                    "b1-b256": "direct",
                    "b512": 128,
                    "b1024": 128,
                }
                or replacement.get("custom_native_submissions")
                != {
                    "b1-b256": 1,
                    "b512": 4,
                    "b1024": 8,
                }
                or set(evidence) != set(CAPACITY_AMENDMENT_ARTIFACTS)
            ):
                raise ValueError("L4 capacity amendment record differs from the frozen protocol.")
            for evidence_name, expected_identity in CAPACITY_AMENDMENT_ARTIFACTS.items():
                record = evidence.get(evidence_name)
                if not isinstance(record, dict):
                    raise ValueError(f"Capacity amendment evidence is missing: {evidence_name}.")
                identity = _identity(record, label=f"Capacity amendment {evidence_name}")
                if identity != (expected_identity["sha256"], expected_identity["bytes"]):
                    raise ValueError(f"Capacity amendment evidence differs: {evidence_name}.")
                if not artifacts_by_identity.get(identity):
                    raise ValueError(f"Capacity amendment evidence is not inventoried: {evidence_name}.")
        if "claim-ledger" in roles:
            if payload.get("schema_version") != CLAIM_LEDGER_SCHEMA or payload.get("pass") is not True:
                raise ValueError("Claim ledger is not a passing v1 ledger.")
            claims = payload.get("claims")
            if not isinstance(claims, list) or not claims:
                raise ValueError("Claim ledger contains no claims.")
            claim_ids: set[str] = set()
            for claim in claims:
                if not isinstance(claim, dict):
                    raise ValueError("Claim ledger contains a malformed claim.")
                claim_id = claim.get("claim_id")
                required_strings = (
                    "article_location",
                    "equation_contract",
                    "hardware",
                    "output_contract",
                    "scene",
                )
                if (
                    not isinstance(claim_id, str)
                    or not claim_id
                    or claim_id in claim_ids
                    or any(not isinstance(claim.get(key), str) or not claim[key] for key in required_strings)
                    or not isinstance(claim.get("display_value"), str)
                    or not claim["display_value"]
                    or not isinstance(claim.get("json_fields"), list)
                    or not claim["json_fields"]
                    or any(not isinstance(field, str) or not field for field in claim["json_fields"])
                    or not isinstance(claim.get("artifact"), dict)
                    or not isinstance(claim["artifact"].get("path"), str)
                    or not claim["artifact"]["path"]
                ):
                    raise ValueError("Claim ledger claim is incomplete or duplicated.")
                target_identity = _identity(
                    claim["artifact"],
                    label=f"Claim {claim_id} artifact",
                )
                target_artifacts = artifacts_by_identity.get(target_identity, [])
                if not target_artifacts:
                    raise ValueError(f"Claim {claim_id} does not name an inventoried publication artifact.")
                target_path = object_paths[target_identity[0]]
                target_payload = _read_json_document(target_path)
                if target_payload is None:
                    raise ValueError(f"Claim {claim_id} target is not JSON.")
                for field in claim["json_fields"]:
                    _resolve_json_pointer(target_payload, field)
                claim_ids.add(claim_id)
        if "code-freeze" in roles and (
            payload.get("schema_version") != CODE_FREEZE_SCHEMA or payload.get("pass") is not True
        ):
            raise ValueError("Code-freeze record did not pass.")
        if "source-manifest" in roles and (
            payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA or payload.get("dirty") is not False
        ):
            raise ValueError("Source manifest is not a clean v2 manifest.")
        if "adapter-attestation" in roles:
            repair_audit = payload.get("correctness_repair_audit") or {}
            repair_checks = repair_audit.get("checks") or {}
            repair_source = repair_audit.get("render_source") or {}
            render_pairs = [
                pair
                for pair in payload.get("source_pairs") or []
                if isinstance(pair, dict)
                and pair.get("adapter_path") == "src/isaacsim_gaussian_renderer/native/flashgs/render.cu"
            ]
            expected_repair_checks = {
                "repaired_slot_0_predicate_exactly_once",
                "repaired_slot_1_predicate_exactly_once",
                "buggy_slot_0_predicate_absent",
                "buggy_slot_1_predicate_absent",
                "slot_0_load_guard_present",
                "slot_1_load_guard_present",
            }
            if (
                payload.get("schema_version") != FLASHGS_ADAPTER_ATTESTATION_SCHEMA
                or payload.get("upstream_commit") != "cdfc4e4002318423eda356eed02df8e01fa32cb6"
                or payload.get("upstream_clean") is not True
                or payload.get("baseline_classification") != FLASHGS_MATCHED_PORT_CLASSIFICATION
                or repair_audit.get("pass") is not True
                or set(repair_checks) != expected_repair_checks
                or not all(value is True for value in repair_checks.values())
                or len(render_pairs) != 1
                or repair_source.get("bytes") != render_pairs[0].get("adapter_bytes")
                or repair_source.get("sha256") != render_pairs[0].get("adapter_sha256")
                or payload.get("pass") is not True
            ):
                raise ValueError("FlashGS adapter attestation did not pass.")
        if "build-attestation" in roles and payload.get("schema_version") != ("flashgs-matched-gsplat-build-v1"):
            raise ValueError("gsplat build attestation schema differs.")
        if "capacity-calibration" in roles and (
            payload.get("schema_version") != "flashgs-matched-capacity-calibration-v1"
            or payload.get("mode") != "capacity-calibration-only"
            or payload.get("pass") is not True
            or (payload.get("output_validation") or {}).get("pass") is not True
        ):
            raise ValueError(f"Capacity calibration did not pass: {logical_path}.")
        if "flashgs-demand-survey" in roles and (
            payload.get("schema_version") != "flashgs-matched-flashgs-demand-survey-v1"
            or payload.get("mode") != "flashgs-demand-survey-only"
            or payload.get("renderer") != "flashgs"
            or payload.get("pass") is not True
            or payload.get("timing_valid") is not False
            or payload.get("render_outputs_valid") is not False
        ):
            raise ValueError(f"FlashGS demand survey did not pass: {logical_path}.")
        if "profile-control" in roles and (
            payload.get("schema_version") != RENDERER_RUN_SCHEMA
            or payload.get("profile_control") is not True
            or payload.get("pass") is not True
        ):
            raise ValueError(f"Profile control did not pass: {logical_path}.")
        if "profile-wrapper" in roles and (
            payload.get("schema_version") != "flashgs-profile-wrapper-evidence-v1"
            or payload.get("pass") is not True
            or not isinstance(payload.get("command"), list)
            or not payload["command"]
        ):
            raise ValueError(f"Profile wrapper did not pass: {logical_path}.")
        if "matrix-invocation" in roles and (
            payload.get("schema_version") != "flashgs-matched-matrix-invocation-v1"
            or not isinstance(payload.get("argv"), list)
            or not payload["argv"]
        ):
            raise ValueError("Matrix invocation evidence differs.")
        if roles.intersection({"command-record", "summary-command"}) and (
            not isinstance(payload.get("command"), list) or not payload["command"]
        ):
            raise ValueError(f"Command record is incomplete: {logical_path}.")
        if "occupancy-evidence" in roles and (
            not isinstance(payload.get("schema_version"), str) or payload.get("pass") is not True
        ):
            raise ValueError(f"Occupancy evidence did not pass: {logical_path}.")
        if "machine-provenance" in roles and any(
            not payload.get(field)
            for field in (
                "cuda_runtime",
                "driver",
                "gpu_name",
                "gpu_uuid",
                "torch",
            )
        ):
            raise ValueError("Machine provenance is incomplete.")
        expected_schemas = {
            "camera-contract": CAMERA_CONTRACT_SCHEMA,
            "fidelity-summary": FIDELITY_SCHEMA,
            "oracle-manifest": ORACLE_SCHEMA,
            "profile-summary": PROFILE_SCHEMA,
            "raw-renderer-run": RENDERER_RUN_SCHEMA,
        }
        for role, schema in expected_schemas.items():
            if role in roles and payload.get("schema_version") != schema:
                raise ValueError(f"Publication acceptance record has the wrong schema: {logical_path}.")
        if roles.intersection(set(expected_schemas) - {"camera-contract"}) and payload.get("pass") is not True:
            raise ValueError(f"Publication acceptance record did not pass: {logical_path}.")
        if "verification-record" in roles and (
            not isinstance(payload.get("schema_version"), str)
            or not payload["schema_version"]
            or payload.get("pass") is not True
        ):
            raise ValueError(f"Publication verification record did not pass: {logical_path}.")


def _object_entry(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    return {
        "bytes": byte_count,
        "json_document": _read_json_document(path) is not None,
        "object_path": artifact_object_path(sha256),
        "sha256": sha256,
    }


def _reference_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        value["source_sha256"],
        value["json_pointer"],
        value["target_sha256"],
        value["target_bytes"],
        value["storage"],
        value.get("dependency_id", ""),
    )


def _collect_references(
    objects: dict[str, tuple[int, Path]],
    external: list[dict[str, Any]],
    *,
    allow_discovery: bool,
) -> list[dict[str, Any]]:
    external_by_identity = {(item["sha256"], item["bytes"]): item for item in external}
    references: list[dict[str, Any]] = []
    scanned: set[str] = set()
    while True:
        pending = sorted(set(objects) - scanned)
        if not pending:
            break
        source_sha256 = pending[0]
        source_bytes, source_path = objects[source_sha256]
        del source_bytes
        scanned.add(source_sha256)
        payload = _read_json_document(source_path)
        if payload is None:
            continue
        for pointer, record in _artifact_records(payload):
            raw_record_path = record.get("path")
            if (
                not isinstance(raw_record_path, str)
                or not raw_record_path
                or "\\" in raw_record_path
                or "\x00" in raw_record_path
                or any(part in (".", "..") for part in PurePosixPath(raw_record_path).parts)
            ):
                raise ValueError(f"Unsafe recorded artifact path at {source_sha256}{pointer}.")
            if not PurePosixPath(raw_record_path).is_absolute():
                _safe_relative_path(
                    raw_record_path,
                    label=f"recorded artifact path at {source_sha256}{pointer}",
                )
            _assert_no_secret_path(raw_record_path, label="Recorded artifact")
            target_sha256, target_bytes = _identity(
                record,
                label=f"Artifact reference {source_sha256}{pointer}",
            )
            target_identity = (target_sha256, target_bytes)
            external_dependency = external_by_identity.get(target_identity)
            reference: dict[str, Any] = {
                "json_pointer": pointer,
                "source_sha256": source_sha256,
                "storage": "external" if external_dependency else "bundled",
                "target_bytes": target_bytes,
                "target_sha256": target_sha256,
            }
            if external_dependency is not None:
                reference["dependency_id"] = external_dependency["dependency_id"]
            elif target_sha256 in objects:
                existing_bytes, _ = objects[target_sha256]
                if existing_bytes != target_bytes:
                    raise ValueError("One SHA-256 is associated with conflicting byte counts.")
            elif allow_discovery:
                raw_path = record.get("path")
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError("Referenced bundled artifact has no usable source path.")
                source = Path(raw_path)
                _assert_no_secret_path(source, label="Referenced artifact")
                if _looks_like_ply(source.name):
                    raise ValueError("A referenced dataset PLY must be declared as external.")
                resolved = resolve_artifact_record(record, label="Referenced artifact")
                source = Path(resolved["path"])
                if _has_ply_payload(source):
                    raise ValueError("A referenced dataset PLY payload must be declared as external.")
                _scan_secret_bytes(source, textual=_is_textual(source.name))
                objects[target_sha256] = (target_bytes, source)
            else:
                raise ValueError(f"Dangling bundled artifact reference {target_sha256} at {source_sha256}{pointer}.")
            references.append(reference)
    references.sort(key=_reference_key)
    return references


def _canonical_manifest_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _inventory_from_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = [
        {
            "bytes": item.get("bytes"),
            "path": item.get("logical_path"),
            "roles": item.get("roles"),
            "sha256": item.get("sha256"),
        }
        for item in payload.get("publication_artifacts", [])
    ]
    return _normalize_inventory(
        {
            "artifacts": artifacts,
            "bundle_name": payload.get("bundle_name"),
            "external_dependencies": payload.get("external_dependencies"),
            "schema_version": PUBLICATION_INVENTORY_SCHEMA,
        }
    )


SemanticValidator = Callable[[Path, Path], dict[str, Any]]


def _normalize_semantic_validation(
    value: Any,
    *,
    input_files: dict[str, Path],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "batches",
        "pass",
        "recomputed_summary_sha256",
        "schema_version",
        "source_manifest",
        "summary",
        "validator",
    }:
        raise ValueError("Semantic-validation result has missing or unknown fields.")
    if (
        value.get("schema_version") != SEMANTIC_VALIDATION_SCHEMA
        or value.get("pass") is not True
        or value.get("batches") != list(MATCHED_BATCHES)
    ):
        raise ValueError("Canonical matched semantic validation did not pass.")
    logical_records = {
        "source_manifest": "provenance/source-manifest.json",
        "summary": "summary.json",
        "validator": "publication/validator/summarize_flashgs_matched.py",
    }
    normalized: dict[str, Any] = {
        "batches": list(MATCHED_BATCHES),
        "execution_contract": "canonical-validator-executed-before-sealing",
        "pass": True,
        "recomputed_summary_sha256": value.get("recomputed_summary_sha256"),
        "schema_version": SEMANTIC_VALIDATION_SCHEMA,
    }
    for field, logical_path in logical_records.items():
        expected_path = input_files[logical_path]
        expected = {
            "bytes": expected_path.stat().st_size,
            "sha256": file_sha256(expected_path),
        }
        actual = value.get(field)
        if (
            not isinstance(actual, dict)
            or {
                "bytes": actual.get("bytes"),
                "sha256": actual.get("sha256"),
            }
            != expected
        ):
            raise ValueError(f"Semantic-validation {field} identity differs.")
        normalized[field] = {
            "bytes": expected["bytes"],
            "logical_path": logical_path,
            "sha256": expected["sha256"],
        }
    summary_payload = json.loads(input_files["summary.json"].read_text(encoding="utf-8"))
    recomputed = sha256_json(summary_payload)
    if value.get("recomputed_summary_sha256") != recomputed:
        raise ValueError("Semantic-validation rebuilt-summary digest differs.")
    return normalized


def build_evidence_bundle(
    *,
    input_root: str | Path,
    inventory_path: str | Path,
    output_root: str | Path,
    semantic_validator: SemanticValidator,
) -> dict[str, Any]:
    """Build and verify one immutable directory-form evidence bundle."""

    raw_source_root = Path(input_root)
    if raw_source_root.is_symlink():
        raise ValueError("Evidence input root may not be a symlink.")
    source_root = raw_source_root.resolve()
    raw_output = Path(output_root)
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(f"Evidence bundle output already exists: {raw_output}.")
    output = raw_output.absolute()
    if output == source_root or output.is_relative_to(source_root):
        raise ValueError("Evidence bundle output must be outside the input root.")
    inventory = load_publication_inventory(inventory_path)
    input_files = _walk_regular_tree(source_root)
    expected_paths = {item["path"] for item in inventory["artifacts"]}
    actual_paths = set(input_files)
    if expected_paths != actual_paths:
        missing = sorted(expected_paths - actual_paths)
        extras = sorted(actual_paths - expected_paths)
        raise ValueError(f"Publication input tree differs from inventory; missing={missing}, extras={extras}.")

    semantic_validation = _normalize_semantic_validation(
        semantic_validator(source_root, source_root / "summary.json"),
        input_files=input_files,
    )

    external_identities = {(item["sha256"], item["bytes"]) for item in inventory["external_dependencies"]}
    objects: dict[str, tuple[int, Path]] = {}
    for artifact in inventory["artifacts"]:
        relative = artifact["path"]
        source = input_files[relative]
        _assert_no_secret_path(relative, label="Publication artifact")
        _scan_secret_bytes(source, textual=_is_textual(relative))
        if _has_ply_payload(source):
            raise ValueError("Dataset PLY payloads must be external dependencies.")
        actual = {"bytes": source.stat().st_size, "sha256": file_sha256(source)}
        if (actual["sha256"], actual["bytes"]) != (
            artifact["sha256"],
            artifact["bytes"],
        ):
            raise ValueError(f"Publication artifact changed after inventory: {relative}.")
        if (artifact["sha256"], artifact["bytes"]) in external_identities:
            raise ValueError("An external dependency is present inside the publication input tree.")
        previous = objects.get(artifact["sha256"])
        if previous is not None and previous[0] != artifact["bytes"]:
            raise ValueError("One SHA-256 is associated with conflicting byte counts.")
        objects.setdefault(artifact["sha256"], (artifact["bytes"], source))

    references = _collect_references(
        objects,
        inventory["external_dependencies"],
        allow_discovery=True,
    )
    referenced_external = {item["dependency_id"] for item in references if item["storage"] == "external"}
    expected_external = {item["dependency_id"] for item in inventory["external_dependencies"]}
    if referenced_external != expected_external:
        raise ValueError("External dependency inventory is unused or incomplete.")

    object_entries = [_object_entry(path, sha256, byte_count) for sha256, (byte_count, path) in sorted(objects.items())]
    publication_artifacts = [
        {
            "bytes": item["bytes"],
            "logical_path": item["path"],
            "media_type": _media_type(item["path"]),
            "object_path": artifact_object_path(item["sha256"]),
            "roles": item["roles"],
            "sha256": item["sha256"],
        }
        for item in inventory["artifacts"]
    ]
    object_paths = {sha256: path for sha256, (_, path) in objects.items()}
    _validate_role_documents(publication_artifacts, object_paths)
    archive_order = ["manifest.json"] + [item["object_path"] for item in object_entries]
    manifest: dict[str, Any] = {
        "archive_input_order": archive_order,
        "artifact_references": references,
        "bundle_name": inventory["bundle_name"],
        "external_dependencies": inventory["external_dependencies"],
        "inventory_id": sha256_json(inventory),
        "objects": object_entries,
        "publication_artifacts": publication_artifacts,
        "required_external_roles": sorted(REQUIRED_EXTERNAL_ROLES),
        "required_publication_roles": sorted(REQUIRED_PUBLICATION_ROLES),
        "schema_version": EVIDENCE_BUNDLE_SCHEMA,
        "semantic_validation": semantic_validation,
    }
    manifest["manifest_id"] = sha256_json(manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for entry in object_entries:
            destination = staging.joinpath(*PurePosixPath(entry["object_path"]).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(object_paths[entry["sha256"]], destination)
            destination.chmod(0o444)
            if destination.stat().st_size != entry["bytes"] or file_sha256(destination) != entry["sha256"]:
                raise ValueError("Copied evidence object failed integrity verification.")
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_canonical_manifest_bytes(manifest))
        manifest_path.chmod(0o444)
        verify_evidence_bundle(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_evidence_bundle(output)


def _validate_manifest_structure(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError("Unexpected evidence-bundle manifest schema.")
    expected_keys = {
        "archive_input_order",
        "artifact_references",
        "bundle_name",
        "external_dependencies",
        "inventory_id",
        "manifest_id",
        "objects",
        "publication_artifacts",
        "required_external_roles",
        "required_publication_roles",
        "schema_version",
        "semantic_validation",
    }
    if set(payload) != expected_keys:
        raise ValueError("Evidence-bundle manifest has missing or unknown fields.")
    if payload.get("required_publication_roles") != sorted(REQUIRED_PUBLICATION_ROLES):
        raise ValueError("Evidence-bundle required publication roles differ.")
    if payload.get("required_external_roles") != sorted(REQUIRED_EXTERNAL_ROLES):
        raise ValueError("Evidence-bundle required external roles differ.")
    unsigned = dict(payload)
    recorded_manifest_id = unsigned.pop("manifest_id", None)
    if recorded_manifest_id != sha256_json(unsigned):
        raise ValueError("Evidence-bundle manifest ID differs.")
    inventory = _inventory_from_manifest(payload)
    if payload.get("inventory_id") != sha256_json(inventory):
        raise ValueError("Evidence-bundle inventory ID differs.")
    semantic = payload.get("semantic_validation")
    if (
        not isinstance(semantic, dict)
        or semantic.get("schema_version") != SEMANTIC_VALIDATION_SCHEMA
        or semantic.get("pass") is not True
        or semantic.get("batches") != list(MATCHED_BATCHES)
        or semantic.get("execution_contract") != "canonical-validator-executed-before-sealing"
    ):
        raise ValueError("Evidence-bundle semantic-validation attestation differs.")
    return inventory


def _verify_semantic_validation(
    semantic: dict[str, Any],
    artifacts: list[dict[str, Any]],
    object_paths: dict[str, Path],
) -> None:
    expected_keys = {
        "batches",
        "execution_contract",
        "pass",
        "recomputed_summary_sha256",
        "schema_version",
        "source_manifest",
        "summary",
        "validator",
    }
    if set(semantic) != expected_keys:
        raise ValueError("Semantic-validation attestation has missing or unknown fields.")
    by_logical_path = {item["logical_path"]: item for item in artifacts}
    for field, logical_path in {
        "source_manifest": "provenance/source-manifest.json",
        "summary": "summary.json",
        "validator": "publication/validator/summarize_flashgs_matched.py",
    }.items():
        artifact = by_logical_path[logical_path]
        expected = {
            "bytes": artifact["bytes"],
            "logical_path": logical_path,
            "sha256": artifact["sha256"],
        }
        if semantic.get(field) != expected:
            raise ValueError(f"Semantic-validation {field} artifact differs.")
    summary_artifact = by_logical_path["summary.json"]
    summary_payload = json.loads(object_paths[summary_artifact["sha256"]].read_text(encoding="utf-8"))
    if semantic.get("recomputed_summary_sha256") != sha256_json(summary_payload):
        raise ValueError("Semantic-validation summary digest differs.")


def verify_evidence_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Fail closed unless a relocated bundle is complete and self-consistent."""

    raw_root = Path(bundle_root)
    if raw_root.is_symlink():
        raise ValueError("Evidence bundle root may not be a symlink.")
    root = raw_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Evidence bundle root is missing: {root}.")
    files = _walk_regular_tree(root)
    manifest_path = root / "manifest.json"
    if "manifest.json" not in files:
        raise FileNotFoundError("Evidence bundle has no manifest.json.")
    _scan_secret_bytes(manifest_path, textual=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_path.read_bytes() != _canonical_manifest_bytes(payload):
        raise ValueError("Evidence-bundle manifest encoding is not canonical.")
    inventory = _validate_manifest_structure(payload)

    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("Evidence bundle has no objects.")
    object_paths: dict[str, Path] = {}
    normalized_objects: list[dict[str, Any]] = []
    for raw in raw_objects:
        if not isinstance(raw, dict) or set(raw) != {
            "bytes",
            "json_document",
            "object_path",
            "sha256",
        }:
            raise ValueError("Malformed evidence object entry.")
        sha256, byte_count = _identity(raw, label="Evidence object")
        expected_path = artifact_object_path(sha256)
        if raw.get("object_path") != expected_path:
            raise ValueError("Evidence object path is not content-addressed.")
        if sha256 in object_paths:
            raise ValueError("Evidence bundle repeats an object SHA-256.")
        resolved = resolve_artifact_record(
            {"bytes": byte_count, "path": "ignored", "sha256": sha256},
            bundle_root=root,
            label="Evidence object",
        )
        path = Path(resolved["path"])
        _scan_secret_bytes(path, textual=bool(raw.get("json_document")))
        if _has_ply_payload(path):
            raise ValueError("Dataset PLY payloads may not appear in an evidence bundle.")
        detected_json = _read_json_document(path) is not None
        if not isinstance(raw.get("json_document"), bool) or raw["json_document"] is not detected_json:
            raise ValueError("Evidence object JSON classification differs from its bytes.")
        object_paths[sha256] = path
        normalized_objects.append(
            {
                "bytes": byte_count,
                "json_document": detected_json,
                "object_path": expected_path,
                "sha256": sha256,
            }
        )
    if normalized_objects != sorted(normalized_objects, key=lambda item: item["sha256"]):
        raise ValueError("Evidence objects are not deterministically ordered.")

    expected_file_paths = {"manifest.json"} | {item["object_path"] for item in normalized_objects}
    if set(files) != expected_file_paths:
        missing = sorted(expected_file_paths - set(files))
        extras = sorted(set(files) - expected_file_paths)
        raise ValueError(f"Evidence bundle has missing or extra files; missing={missing}, extras={extras}.")
    expected_archive_order = ["manifest.json"] + sorted(item["object_path"] for item in normalized_objects)
    if payload.get("archive_input_order") != expected_archive_order:
        raise ValueError("Evidence-bundle archive input order differs.")

    external = inventory["external_dependencies"]
    external_identities = {(item["sha256"], item["bytes"]) for item in external}
    object_identities = {(item["sha256"], item["bytes"]) for item in normalized_objects}
    if external_identities.intersection(object_identities):
        raise ValueError("An external dependency was included as a bundle object.")

    artifacts = payload.get("publication_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Evidence bundle has no publication-artifact inventory.")
    normalized_artifacts: list[dict[str, Any]] = []
    for raw, expected in zip(artifacts, inventory["artifacts"], strict=True):
        if not isinstance(raw, dict) or set(raw) != {
            "bytes",
            "logical_path",
            "media_type",
            "object_path",
            "roles",
            "sha256",
        }:
            raise ValueError("Malformed publication-artifact manifest entry.")
        if (
            raw.get("logical_path") != expected["path"]
            or raw.get("roles") != expected["roles"]
            or raw.get("sha256") != expected["sha256"]
            or raw.get("bytes") != expected["bytes"]
            or raw.get("object_path") != artifact_object_path(expected["sha256"])
            or raw.get("media_type") != _media_type(expected["path"])
            or expected["sha256"] not in object_paths
        ):
            raise ValueError("Publication-artifact manifest entry differs from inventory.")
        normalized_artifacts.append(raw)
    if len(artifacts) != len(inventory["artifacts"]):
        raise ValueError("Publication-artifact inventory length differs.")
    _validate_role_documents(normalized_artifacts, object_paths)
    _verify_semantic_validation(
        payload["semantic_validation"],
        normalized_artifacts,
        object_paths,
    )

    objects_for_scan = {item["sha256"]: (item["bytes"], object_paths[item["sha256"]]) for item in normalized_objects}
    observed_references = _collect_references(
        objects_for_scan,
        external,
        allow_discovery=False,
    )
    if payload.get("artifact_references") != observed_references:
        raise ValueError("Evidence-bundle artifact-reference graph differs.")
    rooted = {item["sha256"] for item in normalized_artifacts}
    referenced = {item["target_sha256"] for item in observed_references if item["storage"] == "bundled"}
    if set(object_paths) != rooted | referenced:
        raise ValueError("Evidence bundle contains an unreferenced object.")
    referenced_external = {item["dependency_id"] for item in observed_references if item["storage"] == "external"}
    if referenced_external != {item["dependency_id"] for item in external}:
        raise ValueError("Evidence bundle has an unused or dangling external dependency.")
    return payload


def write_deterministic_archive(
    bundle_root: str | Path,
    archive_path: str | Path,
) -> dict[str, Any]:
    """Write an uncompressed deterministic tar from a verified bundle."""

    root = Path(bundle_root).resolve()
    manifest = verify_evidence_bundle(root)
    archive = Path(archive_path)
    if archive.exists() or archive.is_symlink():
        raise FileExistsError(f"Evidence archive already exists: {archive}.")
    if archive.absolute().is_relative_to(root):
        raise ValueError("Evidence archive must be outside the bundle directory.")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="w", format=tarfile.GNU_FORMAT) as handle:
        for relative in manifest["archive_input_order"]:
            source = root.joinpath(*PurePosixPath(relative).parts)
            info = tarfile.TarInfo(relative)
            info.size = source.stat().st_size
            info.mode = 0o444
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with source.open("rb") as stream:
                handle.addfile(info, stream)
    return verify_deterministic_archive(root, archive)


def verify_deterministic_archive(
    bundle_root: str | Path,
    archive_path: str | Path,
) -> dict[str, Any]:
    root = Path(bundle_root).resolve()
    manifest = verify_evidence_bundle(root)
    archive = Path(archive_path)
    _assert_regular_no_symlink(archive)
    with tarfile.open(archive, mode="r:") as handle:
        members = handle.getmembers()
        if [item.name for item in members] != manifest["archive_input_order"]:
            raise ValueError("Evidence archive member order differs.")
        expected_size_before_end_blocks = 0
        with archive.open("rb") as archive_stream:
            for member in members:
                expected_info = tarfile.TarInfo(member.name)
                expected_info.size = member.size
                expected_info.mode = 0o444
                expected_info.mtime = 0
                expected_info.uid = 0
                expected_info.gid = 0
                expected_info.uname = ""
                expected_info.gname = ""
                archive_stream.seek(member.offset)
                if archive_stream.read(tarfile.BLOCKSIZE) != expected_info.tobuf(format=tarfile.GNU_FORMAT):
                    raise ValueError("Evidence archive header is not canonical GNU tar.")
                padded_end = member.offset_data + (
                    (member.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE * tarfile.BLOCKSIZE
                )
                archive_stream.seek(member.offset_data + member.size)
                if any(archive_stream.read(padded_end - member.offset_data - member.size)):
                    raise ValueError("Evidence archive member padding is not canonical.")
                expected_size_before_end_blocks = padded_end
            canonical_size = (
                (expected_size_before_end_blocks + 2 * tarfile.BLOCKSIZE + tarfile.RECORDSIZE - 1)
                // tarfile.RECORDSIZE
                * tarfile.RECORDSIZE
            )
            if archive.stat().st_size != canonical_size:
                raise ValueError("Evidence archive has trailing or missing bytes.")
            archive_stream.seek(expected_size_before_end_blocks)
            if any(archive_stream.read()):
                raise ValueError("Evidence archive end padding is not canonical.")
        for member in members:
            if (
                not member.isfile()
                or member.mode != 0o444
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname
                or member.gname
            ):
                raise ValueError("Evidence archive metadata is not deterministic.")
            relative = _safe_relative_path(member.name, label="archive member")
            source = root.joinpath(*PurePosixPath(relative).parts)
            if member.size != source.stat().st_size:
                raise ValueError("Evidence archive member size differs.")
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError("Evidence archive member cannot be read.")
            digest = hashlib.sha256()
            for chunk in iter(lambda: extracted.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != file_sha256(source):
                raise ValueError("Evidence archive member content differs.")
    return {
        "archive": str(archive.resolve()),
        "bytes": archive.stat().st_size,
        "manifest_id": manifest["manifest_id"],
        "sha256": file_sha256(archive),
    }


def write_publication_inventory(
    *,
    input_root: str | Path,
    bundle_name: str,
    roles_by_path: dict[str, list[str]],
    external_dependencies: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Snapshot a complete result tree into a hash-pinned publication inventory."""

    if _SLUG_RE.fullmatch(bundle_name) is None:
        raise ValueError("Publication bundle name is not a safe slug.")
    raw_root = Path(input_root)
    if raw_root.is_symlink():
        raise ValueError("Evidence input root may not be a symlink.")
    root = raw_root.resolve()
    files = _walk_regular_tree(root)
    if set(files) != set(roles_by_path):
        missing_roles = sorted(set(files) - set(roles_by_path))
        unknown_paths = sorted(set(roles_by_path) - set(files))
        raise ValueError(
            f"Every publication file needs explicit roles; missing={missing_roles}, unknown={unknown_paths}."
        )
    artifacts = []
    for relative, path in sorted(files.items()):
        _assert_no_secret_path(relative, label="Publication artifact")
        _scan_secret_bytes(path, textual=_is_textual(relative))
        if _has_ply_payload(path):
            raise ValueError("Dataset PLY payloads must be external dependencies.")
        artifacts.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "roles": roles_by_path[relative],
                "sha256": file_sha256(path),
            }
        )
    payload = _normalize_inventory(
        {
            "artifacts": artifacts,
            "bundle_name": bundle_name,
            "external_dependencies": external_dependencies,
            "schema_version": PUBLICATION_INVENTORY_SCHEMA,
        }
    )
    output = Path(output_path)
    if output.absolute().is_relative_to(root):
        raise ValueError("Publication inventory output must be outside the input root.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Publication inventory output already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_manifest_bytes(payload))
    return payload
