#!/usr/bin/env python3
"""Strictly bind publication numbers to an immutable article and JSON evidence.

This verifier is intentionally publication-local: it does not change or import
the frozen benchmark implementation.  It accepts either:

* a staging root containing logical paths such as ``summary.json`` and
  ``publication/article.md``; or
* a relocated ``flashgs-matched-evidence-bundle-v1`` root, where those logical
  paths are resolved through ``manifest.json`` into the content-addressed object
  store.

The ledger remains compatible with the repository's required
``flashgs-matched-claim-ledger-v1`` role, but adds a strict extension:

.. code-block:: json

  {
    "schema_version": "flashgs-matched-claim-ledger-v1",
    "strict_schema_version": "publication-claim-ledger-strict-v1",
    "pass": true,
    "baseline_label": "FlashGS-derived matched-contract port",
    "article": {
      "path": "/measurement-host/staging/publication/article.md",
      "logical_path": "publication/article.md",
      "bytes": 1234,
      "sha256": "..."
    },
    "claims": [
      {
        "claim_id": "full-b128-gpu-ratio",
        "article_location": "claim:full-b128-gpu-ratio",
        "display_value": "2.315×",
        "baseline_label": "FlashGS-derived matched-contract port",
        "artifact": {
          "path": "/measurement-host/result/summary.json",
          "logical_path": "summary.json",
          "bytes": 5678,
          "sha256": "..."
        },
        "json_fields": [
          "/primary_full_sensor_dynamic_table/4/flashgs_ms",
          "/primary_full_sensor_dynamic_table/4/custom_ms"
        ],
        "operation": "ratio",
        "format": {
          "notation": "fixed",
          "precision": 3,
          "prefix": "",
          "suffix": "×",
          "grouping": false,
          "trim_trailing_zeros": false
        },
        "hardware": "one identified Google Cloud NVIDIA L4",
        "scene": "Home Scan LOD0",
        "output_contract": "full sensor",
        "equation_contract": "pinned-gsplat matched EWA"
      }
    ]
  }

The article wraps each displayed value in one unique marker pair:

``<!-- claim:full-b128-gpu-ratio -->2.315×<!-- /claim:full-b128-gpu-ratio -->``

The claim set and marker set must be identical.  A marker body must equal both
``display_value`` and the value recomputed from the referenced JSON fields.
Only the fixed operations implemented below are accepted; the ledger cannot
carry executable expressions or arbitrary format strings.  In field order,
``direct`` returns ``v0``; ``ratio`` returns ``v0 / v1``; ``difference``
returns ``v0 - v1``; ``percent`` returns ``100 * v0``; ``min``, ``median``,
and ``max`` select over all fields; and ``range`` renders the inclusive
``[min, max]``.
Fixed/scientific decimal rounding is explicitly round-half-even.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


LEDGER_SCHEMA = "flashgs-matched-claim-ledger-v1"
STRICT_LEDGER_SCHEMA = "publication-claim-ledger-strict-v1"
RECEIPT_SCHEMA = "publication-claim-ledger-verification-v1"
BUNDLE_SCHEMA = "flashgs-matched-evidence-bundle-v1"
LEDGER_LOGICAL_PATH = "publication/claim-ledger.json"
ARTICLE_LOGICAL_PATH = "publication/article.md"
ACCEPTED_BASELINE_LABEL = "FlashGS-derived matched-contract port"
ACCEPTED_HARDWARE = "one identified Google Cloud NVIDIA L4"
ACCEPTED_HARDWARE_NAME = "NVIDIA L4"
ACCEPTED_HARDWARE_UUID = "GPU-ebf6dc95-db46-4e1f-6e95-492c5c787805"
ACCEPTED_SCENE = "Home Scan LOD0"
ACCEPTED_EQUATION_CONTRACT = "pinned-gsplat matched EWA"
SUMMARY_LOGICAL_PATH = "summary.json"
SUMMARY_SCHEMA = "flashgs-matched-summary-v4"
VERIFICATION_LOGICAL_PATH = "publication/verification.json"
VERIFICATION_SCHEMA = "publication-verification-v1"
GATE_RESULT_SCHEMA = "publication-gate-result-v1"
ABLATION_LOGICAL_PATH = "publication/evidence/custom-vectorization-ablation.json"
ABLATION_SCHEMA = "publication-custom-vectorization-ablation-v1"
ABLATION_GATE_ID = "custom-vectorization-ablation"
CUSTOM_ABLATION_BASELINE_LABEL = "Custom B128 P1 physical-chunk schedule"
PRIMARY_BATCHES = (1, 8, 32, 64, 128, 256, 512, 1024)
FORBIDDEN_BASELINE_LABELS = frozenset(
    {
        "upstream FlashGS",
        "minimally adapted FlashGS",
        "integration-only FlashGS",
    }
)
ALLOWED_OPERATIONS = frozenset(
    {"direct", "ratio", "difference", "percent", "min", "median", "max", "range"}
)
CLAIM_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MARKER_TOKEN_RE = re.compile(
    r"<!-- claim:([a-z0-9][a-z0-9._-]*) -->|"
    r"<!-- /claim:([a-z0-9][a-z0-9._-]*) -->"
)
PERFORMANCE_NUMBER_RE = re.compile(
    r"(?i)(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)\s*"
    r"(?:[x\u00d7](?!\s*\d)|ms\b|s\b|fps\b|(?:images?|img)/s\b|"
    r"(?:mega)?pixels?/s\b|(?:k|m|g)?b(?:/s)?\b|gib\b|%)"
)
SCALING_LINE_RE = re.compile(
    r"(?i)\bB\d+\b.*\b(?:faster|slower|speedup|throughput|latency|scal(?:e|es|ed|ing)|flat)\b|"
    r"\b(?:faster|slower|speedup|throughput|latency|scal(?:e|es|ed|ing)|flat)\b.*\bB\d+\b"
)


def _claim_bindings() -> tuple[dict[str, dict[str, Any]], frozenset[str]]:
    bindings: dict[str, dict[str, Any]] = {}
    required: set[str] = set()
    table_by_contract = {
        "full": ("primary_full_sensor_dynamic_table", "full sensor"),
        "rgb": ("rgb_only_dynamic_table", "RGB only"),
    }
    metrics = {
        "custom-gpu-ms": ("custom_ms", "direct"),
        "baseline-gpu-ms": ("flashgs_ms", "direct"),
        "gpu-speedup": (("flashgs_ms", "custom_ms"), "ratio"),
        "custom-wall-ms": ("custom_synchronized_wall_ms", "direct"),
        "baseline-wall-ms": ("flashgs_synchronized_wall_ms", "direct"),
        "wall-speedup": (
            ("flashgs_synchronized_wall_ms", "custom_synchronized_wall_ms"),
            "ratio",
        ),
        "custom-images-per-second": ("custom_images_per_second", "direct"),
        "baseline-images-per-second": ("flashgs_images_per_second", "direct"),
        "custom-megapixels-per-second": ("custom_megapixels_per_second", "direct"),
        "baseline-megapixels-per-second": ("flashgs_megapixels_per_second", "direct"),
    }
    for contract, (table, output_contract) in table_by_contract.items():
        for index, batch in enumerate(PRIMARY_BATCHES):
            for suffix, (field_specification, operation) in metrics.items():
                fields = (
                    [f"/{table}/{index}/{field}" for field in field_specification]
                    if isinstance(field_specification, tuple)
                    else [f"/{table}/{index}/{field_specification}"]
                )
                claim_id = f"{contract}-b{batch}-{suffix}"
                bindings[claim_id] = {
                    "artifact_role": "summary",
                    "baseline_label": ACCEPTED_BASELINE_LABEL,
                    "json_fields": fields,
                    "operation": operation,
                    "output_contract": output_contract,
                }
            required.add(f"{contract}-b{batch}-gpu-speedup")
    for claim_id, pointer, output_contract in (
        (
            "full-geometric-mean-speedup",
            "/aggregate/full_sensor_geometric_mean_speedup_custom_over_flashgs",
            "full sensor",
        ),
        (
            "rgb-geometric-mean-speedup",
            "/aggregate/rgb_geometric_mean_speedup_custom_over_flashgs",
            "RGB only",
        ),
    ):
        bindings[claim_id] = {
            "artifact_role": "summary",
            "baseline_label": ACCEPTED_BASELINE_LABEL,
            "json_fields": [pointer],
            "operation": "direct",
            "output_contract": output_contract,
        }
        required.add(claim_id)

    # The causal vectorization result compares this renderer with itself at
    # B128: one physical view per native submission (P1) versus all 128 views
    # in one native submission (P128).  These are deliberately direct reads of
    # the ratios that aggregate_verification.py has already recomputed from the
    # 100-sample raw runs.  No claim requires the ratio to exceed one.
    for contract, output_contract in (
        ("full", "full sensor, Custom B128 P128/P1 causal ablation"),
        ("rgb", "RGB only, Custom B128 P128/P1 causal ablation"),
    ):
        for trial_index, trial in enumerate((1, 2, 3)):
            for clock, field in (
                ("cuda", "cuda_speedup_p128_over_p1"),
                ("wall", "wall_speedup_p128_over_p1"),
            ):
                claim_id = f"custom-ablation-{contract}-t{trial}-{clock}-p128-over-p1"
                bindings[claim_id] = {
                    "artifact_role": "ablation",
                    "baseline_label": CUSTOM_ABLATION_BASELINE_LABEL,
                    "json_fields": [f"/ratios/{contract}/{trial_index}/{field}"],
                    "operation": "direct",
                    "output_contract": output_contract,
                }
                required.add(claim_id)
        for clock, field in (
            ("cuda", "cuda_speedup_p128_over_p1"),
            ("wall", "wall_speedup_p128_over_p1"),
        ):
            fields = [f"/ratios/{contract}/{index}/{field}" for index in range(3)]
            for statistic in ("min", "median", "max"):
                claim_id = f"custom-ablation-{contract}-{clock}-{statistic}-p128-over-p1"
                bindings[claim_id] = {
                    "artifact_role": "ablation",
                    "baseline_label": CUSTOM_ABLATION_BASELINE_LABEL,
                    "json_fields": fields,
                    "operation": statistic,
                    "output_contract": output_contract,
                }
    return bindings, frozenset(required)


ALLOWED_RELEASE_CLAIMS, REQUIRED_RELEASE_CLAIM_IDS = _claim_bindings()


class ClaimLedgerError(ValueError):
    """Raised when any article, claim, evidence, or formatting gate fails."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def safe_logical_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ClaimLedgerError(f"{label} is not a safe logical path.")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise ClaimLedgerError(f"{label} is not a safe relative logical path: {value!r}.")
    return candidate.as_posix()


def normalized_artifact_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClaimLedgerError(f"{label} is not an artifact record.")
    raw_path = value.get("path")
    logical_path = safe_logical_path(value.get("logical_path"), label=f"{label} logical_path")
    byte_count = value.get("bytes")
    sha256 = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ClaimLedgerError(f"{label} has no provenance path.")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ClaimLedgerError(f"{label} has an invalid byte count.")
    if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
        raise ClaimLedgerError(f"{label} has an invalid SHA-256.")
    return {
        "bytes": byte_count,
        "logical_path": logical_path,
        "path": raw_path,
        "sha256": sha256,
    }


def normalized_identity_record(value: Any, *, label: str) -> dict[str, Any]:
    """Normalize an aggregate record that is addressed by identity, not alias."""

    if not isinstance(value, dict):
        raise ClaimLedgerError(f"{label} is not an artifact identity record.")
    raw_path = value.get("path")
    byte_count = value.get("bytes")
    sha256 = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ClaimLedgerError(f"{label} has no provenance path.")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ClaimLedgerError(f"{label} has an invalid byte count.")
    if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
        raise ClaimLedgerError(f"{label} has an invalid SHA-256.")
    return {"bytes": byte_count, "path": raw_path, "sha256": sha256}


def assert_regular_file(path: Path, *, boundary: Path, label: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ClaimLedgerError(f"{label} crosses a symlink: {current}.")
        if current == boundary or current.parent == current:
            break
        current = current.parent
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ClaimLedgerError(f"{label} is missing: {path}.") from error
    if path.is_symlink() or not resolved.is_file():
        raise ClaimLedgerError(f"{label} is not a regular non-symlink file: {path}.")
    try:
        resolved.relative_to(boundary)
    except ValueError as error:
        raise ClaimLedgerError(f"{label} escapes the verification root: {path}.") from error


@dataclass(frozen=True)
class EvidenceRoot:
    root: Path
    mode: str
    publication_artifacts: dict[str, dict[str, Any]]

    @classmethod
    def open(cls, value: str | Path) -> "EvidenceRoot":
        raw = Path(value)
        if raw.is_symlink():
            raise ClaimLedgerError("Verification root may not be a symlink.")
        try:
            root = raw.resolve(strict=True)
        except FileNotFoundError as error:
            raise ClaimLedgerError(f"Verification root is missing: {raw}.") from error
        if not root.is_dir():
            raise ClaimLedgerError(f"Verification root is not a directory: {root}.")
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink():
            raise ClaimLedgerError("Bundle manifest may not be a symlink.")
        if not manifest_path.is_file():
            return cls(root=root, mode="staging", publication_artifacts={})
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ClaimLedgerError("Bundle manifest is not valid UTF-8 JSON.") from error
        if manifest.get("schema_version") != BUNDLE_SCHEMA:
            raise ClaimLedgerError(
                "A root containing manifest.json must be a relocated "
                f"{BUNDLE_SCHEMA} bundle."
            )
        artifacts: dict[str, dict[str, Any]] = {}
        raw_artifacts = manifest.get("publication_artifacts")
        if not isinstance(raw_artifacts, list):
            raise ClaimLedgerError("Bundle manifest has no publication_artifacts list.")
        for item in raw_artifacts:
            if not isinstance(item, dict):
                raise ClaimLedgerError("Bundle manifest contains a malformed publication artifact.")
            logical_path = safe_logical_path(
                item.get("logical_path"),
                label="Bundle publication artifact logical_path",
            )
            byte_count = item.get("bytes")
            sha256 = item.get("sha256")
            if (
                logical_path in artifacts
                or isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
                or not isinstance(sha256, str)
                or SHA256_RE.fullmatch(sha256) is None
            ):
                raise ClaimLedgerError("Bundle publication artifact identity is malformed or duplicated.")
            artifacts[logical_path] = {
                "bytes": byte_count,
                "logical_path": logical_path,
                "sha256": sha256,
            }
        return cls(root=root, mode="bundle", publication_artifacts=artifacts)

    def _object_path(self, sha256: str) -> Path:
        return self.root / "objects" / "sha256" / sha256[:2] / sha256

    def resolve_logical(self, logical_path: str, *, label: str) -> Path:
        logical = safe_logical_path(logical_path, label=f"{label} logical path")
        if self.mode == "bundle":
            manifest_record = self.publication_artifacts.get(logical)
            if manifest_record is None:
                raise ClaimLedgerError(f"{label} is not inventoried in the relocated bundle: {logical}.")
            path = self._object_path(manifest_record["sha256"])
            expected_bytes = manifest_record["bytes"]
            expected_sha256 = manifest_record["sha256"]
        else:
            path = self.root.joinpath(*PurePosixPath(logical).parts)
            expected_bytes = None
            expected_sha256 = None
        assert_regular_file(path, boundary=self.root, label=label)
        if expected_bytes is not None and (
            path.stat().st_size != expected_bytes or file_sha256(path) != expected_sha256
        ):
            raise ClaimLedgerError(f"{label} differs from its bundle-manifest identity.")
        return path

    def resolve_record(self, value: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
        record = normalized_artifact_record(value, label=label)
        path = self.resolve_logical(record["logical_path"], label=label)
        if self.mode == "bundle":
            manifest_record = self.publication_artifacts[record["logical_path"]]
            if (
                manifest_record["bytes"] != record["bytes"]
                or manifest_record["sha256"] != record["sha256"]
            ):
                raise ClaimLedgerError(f"{label} record differs from the bundle inventory.")
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ClaimLedgerError(f"{label} differs from its ledger byte/SHA binding.")
        return path, record

    def resolve_identity(self, value: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
        """Resolve a nested aggregate-verification artifact by bytes/SHA-256."""

        record = normalized_identity_record(value, label=label)
        if self.mode == "bundle":
            path = self._object_path(record["sha256"])
        else:
            path = Path(record["path"])
            if not path.is_absolute():
                path = self.root / path
        assert_regular_file(path, boundary=self.root, label=label)
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ClaimLedgerError(f"{label} differs from its aggregate bytes/SHA binding.")
        return path, record


def reject_json_constant(value: str) -> None:
    raise ClaimLedgerError(f"Non-finite JSON number is forbidden: {value}.")


def load_decimal_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=Decimal,
            parse_int=Decimal,
            parse_constant=reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidOperation) as error:
        raise ClaimLedgerError(f"{label} is not finite UTF-8 JSON.") from error


def resolve_json_pointer(value: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ClaimLedgerError(f"Claim JSON field is not an RFC 6901 pointer: {pointer!r}.")
    current = value
    for raw_part in pointer[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", raw_part):
            raise ClaimLedgerError(f"Claim JSON pointer has invalid escaping: {pointer!r}.")
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ClaimLedgerError(f"Claim JSON pointer does not resolve: {pointer!r}.")
    return current


def numeric(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
        raise ClaimLedgerError(f"{label} is not a JSON number.")
    result = value if isinstance(value, Decimal) else Decimal(value)
    if not result.is_finite():
        raise ClaimLedgerError(f"{label} is not finite.")
    return result


def require_arity(operation: str, values: list[Decimal], *, exact: int | None = None, minimum: int | None = None) -> None:
    if exact is not None and len(values) != exact:
        raise ClaimLedgerError(f"Operation {operation!r} requires exactly {exact} JSON field(s).")
    if minimum is not None and len(values) < minimum:
        raise ClaimLedgerError(f"Operation {operation!r} requires at least {minimum} JSON field(s).")


def recompute(operation: str, values: list[Decimal]) -> Decimal | tuple[Decimal, Decimal]:
    if operation not in ALLOWED_OPERATIONS:
        raise ClaimLedgerError(f"Unsupported claim operation: {operation!r}.")
    with localcontext() as context:
        context.prec = 50
        if operation == "direct":
            require_arity(operation, values, exact=1)
            return values[0]
        if operation == "ratio":
            require_arity(operation, values, exact=2)
            if values[1] == 0:
                raise ClaimLedgerError("Ratio denominator is zero.")
            return values[0] / values[1]
        if operation == "difference":
            require_arity(operation, values, exact=2)
            return values[0] - values[1]
        if operation == "percent":
            require_arity(operation, values, exact=1)
            return values[0] * Decimal(100)
        if operation == "min":
            require_arity(operation, values, minimum=1)
            return min(values)
        if operation == "median":
            require_arity(operation, values, minimum=1)
            ordered = sorted(values)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[middle]
            return (ordered[middle - 1] + ordered[middle]) / Decimal(2)
        if operation == "max":
            require_arity(operation, values, minimum=1)
            return max(values)
        require_arity(operation, values, minimum=1)
        return min(values), max(values)


def clean_affix(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise ClaimLedgerError(f"{label} must be a string of at most 64 characters.")
    if any(ord(character) < 32 for character in value) or "<!--" in value or "-->" in value:
        raise ClaimLedgerError(f"{label} contains forbidden control or marker text.")
    return value


def normalized_format(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClaimLedgerError("Claim format must be an object.")
    allowed = {
        "notation",
        "precision",
        "prefix",
        "suffix",
        "grouping",
        "trim_trailing_zeros",
        "exponent_case",
    }
    if set(value) - allowed:
        raise ClaimLedgerError(f"Claim format has unknown fields: {sorted(set(value) - allowed)}.")
    notation = value.get("notation")
    precision = value.get("precision")
    grouping = value.get("grouping", False)
    trim = value.get("trim_trailing_zeros", False)
    exponent_case = value.get("exponent_case", "lower")
    if notation not in ("fixed", "scientific"):
        raise ClaimLedgerError("Claim notation must be 'fixed' or 'scientific'.")
    if isinstance(precision, bool) or not isinstance(precision, int) or not 0 <= precision <= 18:
        raise ClaimLedgerError("Claim precision must be an integer in [0, 18].")
    if not isinstance(grouping, bool) or not isinstance(trim, bool):
        raise ClaimLedgerError("Claim grouping and trim_trailing_zeros must be booleans.")
    if exponent_case not in ("lower", "upper"):
        raise ClaimLedgerError("Claim exponent_case must be 'lower' or 'upper'.")
    if notation == "scientific" and (grouping or trim):
        raise ClaimLedgerError("Scientific notation does not allow grouping or zero trimming.")
    return {
        "notation": notation,
        "precision": precision,
        "prefix": clean_affix(value.get("prefix", ""), label="Claim format prefix"),
        "suffix": clean_affix(value.get("suffix", ""), label="Claim format suffix"),
        "grouping": grouping,
        "trim_trailing_zeros": trim,
        "exponent_case": exponent_case,
    }


def format_scalar(value: Decimal, specification: dict[str, Any]) -> str:
    if value == 0:
        value = abs(value)
    precision = specification["precision"]
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        if specification["notation"] == "fixed":
            grouping = "," if specification["grouping"] else ""
            rendered = format(value, f"{grouping}.{precision}f")
            if specification["trim_trailing_zeros"] and "." in rendered:
                rendered = rendered.rstrip("0").rstrip(".")
        else:
            rendered = format(value, f".{precision}E")
            if specification["exponent_case"] == "lower":
                rendered = rendered.lower()
    return specification["prefix"] + rendered + specification["suffix"]


def format_result(value: Decimal | tuple[Decimal, Decimal], specification: dict[str, Any]) -> str:
    if isinstance(value, tuple):
        lower, upper = value
        return f"[{format_scalar(lower, specification)}, {format_scalar(upper, specification)}]"
    return format_scalar(value, specification)


def parse_claim_markers(article: str) -> dict[str, str]:
    tokens = list(MARKER_TOKEN_RE.finditer(article))
    if article.count("<!-- claim") != sum(1 for token in tokens if token.group(1) is not None):
        raise ClaimLedgerError("Article contains a malformed opening claim marker.")
    if article.count("<!-- /claim") != sum(1 for token in tokens if token.group(2) is not None):
        raise ClaimLedgerError("Article contains a malformed closing claim marker.")
    markers: dict[str, str] = {}
    active: tuple[str, int] | None = None
    for token in tokens:
        opening_id = token.group(1)
        closing_id = token.group(2)
        if opening_id is not None:
            if active is not None:
                raise ClaimLedgerError("Claim markers may not be nested.")
            if opening_id in markers:
                raise ClaimLedgerError(f"Article repeats claim marker {opening_id!r}.")
            active = opening_id, token.end()
            continue
        assert closing_id is not None
        if active is None or active[0] != closing_id:
            raise ClaimLedgerError(f"Article has an unmatched claim marker closure {closing_id!r}.")
        markers[closing_id] = article[active[1] : token.start()]
        active = None
    if active is not None:
        raise ClaimLedgerError(f"Article has an unclosed claim marker {active[0]!r}.")
    return markers


def reject_unmarked_performance_claims(article: str) -> None:
    """Reject quantitative performance tokens and B-scaling prose outside markers."""

    characters = list(article)
    marker_lines: set[int] = set()
    tokens = list(MARKER_TOKEN_RE.finditer(article))
    active: tuple[str, int] | None = None
    for token in tokens:
        if token.group(1) is not None:
            active = token.group(1), token.start()
            marker_lines.add(article.count("\n", 0, token.start()))
            continue
        if active is None:  # parse_claim_markers reports the useful error first.
            continue
        start = active[1]
        marker_lines.update(range(article.count("\n", 0, start), article.count("\n", 0, token.end()) + 1))
        for index in range(start, token.end()):
            if characters[index] != "\n":
                characters[index] = " "
        active = None
    unmarked = "".join(characters)
    match = PERFORMANCE_NUMBER_RE.search(unmarked)
    if match is not None:
        line = unmarked.count("\n", 0, match.start()) + 1
        raise ClaimLedgerError(
            f"Article contains an unmarked quantitative performance token on line {line}: {match.group(0)!r}."
        )
    for index, line in enumerate(unmarked.splitlines()):
        if index not in marker_lines and SCALING_LINE_RE.search(line):
            raise ClaimLedgerError(
                f"Article contains an unmarked batch-scaling claim on line {index + 1}."
            )


def require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ClaimLedgerError(f"{label} must be a nonempty string.")
    return value


def require_baseline_label(
    value: Any,
    *,
    label: str,
    expected: str = ACCEPTED_BASELINE_LABEL,
) -> str:
    text = require_nonempty_string(value, label=label)
    if text in FORBIDDEN_BASELINE_LABELS:
        raise ClaimLedgerError(f"{label} uses forbidden baseline label {text!r}.")
    if text != expected:
        raise ClaimLedgerError(
            f"{label} must be exactly {expected!r}; found {text!r}."
        )
    return text


def validate_ablation_payload(value: Any) -> None:
    """Validate the claim-facing surface of the aggregate-proven ablation."""

    expected = {"batch", "pass", "ratios", "run_order", "runs", "schema_version"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ClaimLedgerError("Custom vectorization ablation has missing or unknown fields.")
    if (
        value.get("schema_version") != ABLATION_SCHEMA
        or value.get("pass") is not True
        or value.get("batch") != Decimal(128)
    ):
        raise ClaimLedgerError("Custom vectorization ablation is not a passing B128 v1 result.")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise ClaimLedgerError("Custom vectorization ablation does not contain twelve fresh runs.")
    run_ids = [run.get("run_id") for run in runs if isinstance(run, dict)]
    if (
        len(run_ids) != 12
        or any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
        or len(set(run_ids)) != 12
    ):
        raise ClaimLedgerError("Custom vectorization ablation run identities are malformed.")
    run_order = value.get("run_order")
    if not isinstance(run_order, list) or len(run_order) != 12 or set(run_order) != set(run_ids):
        raise ClaimLedgerError("Custom vectorization ablation run order differs from its run set.")
    ratios = value.get("ratios")
    if not isinstance(ratios, dict) or set(ratios) != {"full", "rgb"}:
        raise ClaimLedgerError("Custom vectorization ablation ratio contracts differ.")
    for contract in ("full", "rgb"):
        rows = ratios[contract]
        if not isinstance(rows, list) or len(rows) != 3:
            raise ClaimLedgerError(f"Custom vectorization ablation {contract} ratios are incomplete.")
        for expected_trial, row in enumerate(rows, 1):
            if not isinstance(row, dict) or set(row) != {
                "cuda_speedup_p128_over_p1",
                "trial",
                "wall_speedup_p128_over_p1",
            }:
                raise ClaimLedgerError("Custom vectorization ablation ratio row is malformed.")
            if row.get("trial") != Decimal(expected_trial):
                raise ClaimLedgerError("Custom vectorization ablation trial order differs.")
            for field in ("cuda_speedup_p128_over_p1", "wall_speedup_p128_over_p1"):
                ratio = numeric(row.get(field), label=f"Ablation {contract} trial {expected_trial} {field}")
                if ratio <= 0:
                    raise ClaimLedgerError("Custom vectorization ablation contains a nonpositive ratio.")


def resolve_validated_ablation(
    evidence_root: EvidenceRoot,
    *,
    verification_value: Any,
    ablation_value: Any,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Bind the claim artifact to the exact raw result used by aggregate verification."""

    verification_path, verification_record = evidence_root.resolve_record(
        verification_value,
        label="Aggregate verification record",
    )
    if verification_record["logical_path"] != VERIFICATION_LOGICAL_PATH:
        raise ClaimLedgerError(
            f"Aggregate verification logical path must be exactly {VERIFICATION_LOGICAL_PATH!r}."
        )
    try:
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ClaimLedgerError("Aggregate verification record is not valid UTF-8 JSON.") from error
    expected_verification_fields = {
        "benchmark",
        "created_at",
        "gates",
        "matrix",
        "pass",
        "required_gate_ids",
        "schema_version",
    }
    if (
        not isinstance(verification, dict)
        or set(verification) != expected_verification_fields
        or verification.get("schema_version") != VERIFICATION_SCHEMA
        or verification.get("pass") is not True
    ):
        raise ClaimLedgerError("Aggregate verification record is malformed or nonpassing.")
    required_gate_ids = verification.get("required_gate_ids")
    if not isinstance(required_gate_ids, list) or required_gate_ids.count(ABLATION_GATE_ID) != 1:
        raise ClaimLedgerError("Aggregate verification does not require exactly one vectorization ablation gate.")
    gates = verification.get("gates")
    if not isinstance(gates, list):
        raise ClaimLedgerError("Aggregate verification gate list is malformed.")
    matches = [gate for gate in gates if isinstance(gate, dict) and gate.get("gate_id") == ABLATION_GATE_ID]
    if len(matches) != 1:
        raise ClaimLedgerError("Aggregate verification does not contain exactly one vectorization ablation gate.")
    result_path, _result_record = evidence_root.resolve_identity(
        matches[0].get("result"),
        label="Vectorization ablation normalized gate result",
    )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ClaimLedgerError("Vectorization ablation gate result is not valid UTF-8 JSON.") from error
    expected_result_fields = {
        "checks",
        "created_at",
        "evidence",
        "gate_id",
        "identity",
        "pass",
        "schema_version",
    }
    if (
        not isinstance(result, dict)
        or set(result) != expected_result_fields
        or result.get("schema_version") != GATE_RESULT_SCHEMA
        or result.get("gate_id") != ABLATION_GATE_ID
        or result.get("pass") is not True
        or not isinstance(result.get("evidence"), dict)
        or set(result["evidence"]) != {"raw_result"}
    ):
        raise ClaimLedgerError("Vectorization ablation normalized gate result is malformed or nonpassing.")
    _raw_path, raw_record = evidence_root.resolve_identity(
        result["evidence"]["raw_result"],
        label="Aggregate-referenced raw vectorization ablation",
    )
    ablation_path, ablation_record = evidence_root.resolve_record(
        ablation_value,
        label="Canonical vectorization ablation",
    )
    if ablation_record["logical_path"] != ABLATION_LOGICAL_PATH:
        raise ClaimLedgerError(f"Ablation logical path must be exactly {ABLATION_LOGICAL_PATH!r}.")
    if (
        ablation_record["bytes"] != raw_record["bytes"]
        or ablation_record["sha256"] != raw_record["sha256"]
    ):
        raise ClaimLedgerError(
            "Claim ablation artifact is not the raw result referenced by aggregate verification."
        )
    ablation = load_decimal_json(ablation_path, label="Canonical vectorization ablation")
    validate_ablation_payload(ablation)
    return verification_record, ablation_record, ablation


def verify_claim_ledger(
    root: str | Path,
    *,
    ledger_logical_path: str = LEDGER_LOGICAL_PATH,
) -> dict[str, Any]:
    evidence_root = EvidenceRoot.open(root)
    ledger_logical = safe_logical_path(ledger_logical_path, label="Ledger logical path")
    ledger_path = evidence_root.resolve_logical(ledger_logical, label="Claim ledger")
    try:
        ledger_bytes = ledger_path.read_bytes()
        ledger = json.loads(ledger_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ClaimLedgerError("Claim ledger is not valid UTF-8 JSON.") from error
    if not isinstance(ledger, dict):
        raise ClaimLedgerError("Claim ledger must be a JSON object.")
    required_root_fields = {
        "ablation",
        "article",
        "baseline_label",
        "claims",
        "pass",
        "schema_version",
        "summary",
        "strict_schema_version",
        "verification",
    }
    if set(ledger) != required_root_fields:
        raise ClaimLedgerError(
            "Strict claim ledger has missing or unknown root fields: "
            f"expected={sorted(required_root_fields)}, actual={sorted(ledger)}."
        )
    if ledger.get("schema_version") != LEDGER_SCHEMA:
        raise ClaimLedgerError(f"Claim ledger schema must be {LEDGER_SCHEMA!r}.")
    if ledger.get("strict_schema_version") != STRICT_LEDGER_SCHEMA:
        raise ClaimLedgerError(f"Strict claim ledger schema must be {STRICT_LEDGER_SCHEMA!r}.")
    if ledger.get("pass") is not True:
        raise ClaimLedgerError("Claim ledger pass must be true before strict recomputation.")
    baseline_label = require_baseline_label(ledger.get("baseline_label"), label="Ledger baseline_label")

    summary_path, summary_record = evidence_root.resolve_record(
        ledger.get("summary"),
        label="Validated matrix summary",
    )
    if summary_record["logical_path"] != SUMMARY_LOGICAL_PATH:
        raise ClaimLedgerError(f"Summary logical path must be exactly {SUMMARY_LOGICAL_PATH!r}.")
    summary = load_decimal_json(summary_path, label="Validated matrix summary")
    if not isinstance(summary, dict):
        raise ClaimLedgerError("Validated matrix summary is not an object.")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("pass") is not True
        or summary.get("scientific_pass") is not True
        or summary.get("headline_eligible") is not True
    ):
        raise ClaimLedgerError("Validated matrix summary is not a passing headline-eligible v4 result.")
    hardware_scope = summary.get("hardware_scope")
    if not isinstance(hardware_scope, dict) or (
        hardware_scope.get("gpu_name") != ACCEPTED_HARDWARE_NAME
        or hardware_scope.get("gpu_uuid") != ACCEPTED_HARDWARE_UUID
    ):
        raise ClaimLedgerError("Validated matrix summary hardware is not the fixed publication L4.")
    for table_name in ("primary_full_sensor_dynamic_table", "rgb_only_dynamic_table"):
        table = summary.get(table_name)
        if (
            not isinstance(table, list)
            or len(table) != len(PRIMARY_BATCHES)
            or tuple(row.get("batch") for row in table if isinstance(row, dict)) != PRIMARY_BATCHES
            or any(not isinstance(row, dict) or row.get("pass") is not True for row in table)
        ):
            raise ClaimLedgerError(f"Validated matrix summary table {table_name!r} is incomplete or failed.")

    verification_record, ablation_record, ablation = resolve_validated_ablation(
        evidence_root,
        verification_value=ledger.get("verification"),
        ablation_value=ledger.get("ablation"),
    )

    article_path, article_record = evidence_root.resolve_record(
        ledger.get("article"),
        label="Publication article",
    )
    if article_record["logical_path"] != ARTICLE_LOGICAL_PATH:
        raise ClaimLedgerError(f"Article logical path must be exactly {ARTICLE_LOGICAL_PATH!r}.")
    try:
        article = article_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ClaimLedgerError("Publication article is not UTF-8 text.") from error
    markers = parse_claim_markers(article)
    if not markers:
        raise ClaimLedgerError("Publication article contains no claim markers.")
    reject_unmarked_performance_claims(article)

    raw_claims = ledger.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ClaimLedgerError("Claim ledger contains no claims.")
    expected_claim_fields = {
        "artifact",
        "article_location",
        "baseline_label",
        "claim_id",
        "display_value",
        "equation_contract",
        "format",
        "hardware",
        "json_fields",
        "operation",
        "output_contract",
        "scene",
    }
    claims_by_id: dict[str, dict[str, Any]] = {}
    for claim in raw_claims:
        if not isinstance(claim, dict) or set(claim) != expected_claim_fields:
            raise ClaimLedgerError("Strict claim ledger contains a malformed claim object.")
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or CLAIM_ID_RE.fullmatch(claim_id) is None:
            raise ClaimLedgerError(f"Claim has an unsafe claim_id: {claim_id!r}.")
        if claim_id in claims_by_id:
            raise ClaimLedgerError(f"Claim ledger repeats claim_id {claim_id!r}.")
        if claim.get("article_location") != f"claim:{claim_id}":
            raise ClaimLedgerError(f"Claim {claim_id!r} article_location does not bind its marker.")
        expected_binding = ALLOWED_RELEASE_CLAIMS.get(claim_id)
        if expected_binding is None:
            raise ClaimLedgerError(f"Claim {claim_id!r} is not an allowlisted release claim.")
        require_baseline_label(
            claim.get("baseline_label"),
            label=f"Claim {claim_id} baseline_label",
            expected=expected_binding["baseline_label"],
        )
        fixed_metadata = {
            "hardware": ACCEPTED_HARDWARE,
            "scene": ACCEPTED_SCENE,
            "equation_contract": ACCEPTED_EQUATION_CONTRACT,
            "output_contract": expected_binding["output_contract"],
        }
        for field, expected in fixed_metadata.items():
            if claim.get(field) != expected:
                raise ClaimLedgerError(
                    f"Claim {claim_id!r} {field} must be exactly {expected!r}."
                )
        display_value = require_nonempty_string(
            claim.get("display_value"),
            label=f"Claim {claim_id} display_value",
        )
        if "<!--" in display_value or "-->" in display_value or "\n" in display_value or "\r" in display_value:
            raise ClaimLedgerError(f"Claim {claim_id!r} display_value contains marker or newline text.")
        operation = claim.get("operation")
        if operation != expected_binding["operation"]:
            raise ClaimLedgerError(f"Claim {claim_id!r} operation differs from its fixed release binding.")
        fields = claim.get("json_fields")
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(pointer, str) or not pointer for pointer in fields)
        ):
            raise ClaimLedgerError(f"Claim {claim_id!r} has invalid json_fields.")
        if fields != expected_binding["json_fields"]:
            raise ClaimLedgerError(f"Claim {claim_id!r} JSON fields differ from its fixed release binding.")
        claims_by_id[claim_id] = claim

    missing_required = REQUIRED_RELEASE_CLAIM_IDS - set(claims_by_id)
    if missing_required:
        raise ClaimLedgerError(f"Claim ledger omits required release claims: {sorted(missing_required)}.")

    marker_ids = set(markers)
    claim_ids = set(claims_by_id)
    if marker_ids != claim_ids:
        raise ClaimLedgerError(
            "Article claim markers and ledger claims differ; "
            f"missing_markers={sorted(claim_ids - marker_ids)}, "
            f"missing_claims={sorted(marker_ids - claim_ids)}."
        )

    target_cache: dict[tuple[str, int], Any] = {}
    target_receipts: dict[tuple[str, int], dict[str, Any]] = {}
    verified_claims: list[dict[str, Any]] = []
    for claim_id in sorted(claims_by_id):
        claim = claims_by_id[claim_id]
        target_path, target_record = evidence_root.resolve_record(
            claim["artifact"],
            label=f"Claim {claim_id} artifact",
        )
        artifact_role = ALLOWED_RELEASE_CLAIMS[claim_id]["artifact_role"]
        expected_record = summary_record if artifact_role == "summary" else ablation_record
        if (
            target_record["logical_path"] != expected_record["logical_path"]
            or target_record["sha256"] != expected_record["sha256"]
            or target_record["bytes"] != expected_record["bytes"]
        ):
            raise ClaimLedgerError(
                f"Claim {claim_id!r} is not bound to the validated {artifact_role} artifact."
            )
        target_key = (target_record["sha256"], target_record["bytes"])
        if target_key not in target_cache:
            target_cache[target_key] = load_decimal_json(
                target_path,
                label=f"Claim {claim_id} artifact",
            )
            target_receipts[target_key] = {
                "bytes": target_record["bytes"],
                "logical_path": target_record["logical_path"],
                "sha256": target_record["sha256"],
            }
        target = target_cache[target_key]
        values = [
            numeric(
                resolve_json_pointer(target, pointer),
                label=f"Claim {claim_id} field {pointer}",
            )
            for pointer in claim["json_fields"]
        ]
        computed = recompute(claim["operation"], values)
        specification = normalized_format(claim["format"])
        rendered = format_result(computed, specification)
        if rendered != claim["display_value"]:
            raise ClaimLedgerError(
                f"Claim {claim_id!r} display_value differs from recomputation: "
                f"{claim['display_value']!r} != {rendered!r}."
            )
        if markers[claim_id] != claim["display_value"]:
            raise ClaimLedgerError(
                f"Claim {claim_id!r} article marker differs from display_value: "
                f"{markers[claim_id]!r} != {claim['display_value']!r}."
            )
        verified_claims.append(
            {
                "artifact_logical_path": target_record["logical_path"],
                "artifact_sha256": target_record["sha256"],
                "claim_id": claim_id,
                "display_value": rendered,
                "json_fields": list(claim["json_fields"]),
                "operation": claim["operation"],
            }
        )

    return {
        "schema_version": RECEIPT_SCHEMA,
        "pass": True,
        "root_mode": evidence_root.mode,
        "baseline_label": baseline_label,
        "ledger": {
            "bytes": len(ledger_bytes),
            "logical_path": ledger_logical,
            "sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        },
        "article": {
            "bytes": article_record["bytes"],
            "logical_path": article_record["logical_path"],
            "sha256": article_record["sha256"],
        },
        "summary": {
            "bytes": summary_record["bytes"],
            "logical_path": summary_record["logical_path"],
            "sha256": summary_record["sha256"],
        },
        "verification": {
            "bytes": verification_record["bytes"],
            "logical_path": verification_record["logical_path"],
            "sha256": verification_record["sha256"],
        },
        "ablation": {
            "bytes": ablation_record["bytes"],
            "logical_path": ablation_record["logical_path"],
            "sha256": ablation_record["sha256"],
        },
        "claim_count": len(verified_claims),
        "marker_count": len(markers),
        "claims": verified_claims,
        "target_artifacts": sorted(
            target_receipts.values(),
            key=lambda item: (item["logical_path"], item["sha256"]),
        ),
    }


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--ledger-logical-path",
        default=LEDGER_LOGICAL_PATH,
        help="Staging/bundle logical path of the strict claim ledger.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Optional path for the deterministic passing verification receipt.",
    )
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    args = parse_args(arguments)
    receipt = verify_claim_ledger(
        args.root,
        ledger_logical_path=args.ledger_logical_path,
    )
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(canonical_json_bytes(receipt))
    print(
        "PUBLICATION_CLAIM_LEDGER_VERIFIED "
        + json.dumps(receipt, sort_keys=True)
    )


if __name__ == "__main__":
    main()
