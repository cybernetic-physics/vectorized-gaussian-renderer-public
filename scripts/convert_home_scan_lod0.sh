#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${HOME_SCAN_ROOT:-/workspace/datasets/home-scan-sog}"
OUTPUT_PLY="${HOME_SCAN_LOD0_PLY:-/workspace/datasets/home-scan-lod0.ply}"
EXPECTED_CHUNKS=32
EXPECTED_COUNT=21497908
EXPECTED_BYTES=1203883212
EXPECTED_SHA256=29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267

command -v splat-transform >/dev/null
test -d "$SOURCE_ROOT"
test -f "$SOURCE_ROOT/lod-meta.json"

mapfile -t inputs < <(
  find "$SOURCE_ROOT" -mindepth 2 -maxdepth 2 -type f -path '*/0_*/meta.json' |
    sort -V
)

if [[ "${#inputs[@]}" -ne "$EXPECTED_CHUNKS" ]]; then
  echo "Expected $EXPECTED_CHUNKS LOD0 chunks, found ${#inputs[@]}." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PLY")"
rm -f "$OUTPUT_PLY" "$OUTPUT_PLY.sha256"
splat-transform --mem "${inputs[@]}" "$OUTPUT_PLY"

actual_sha256="$(sha256sum "$OUTPUT_PLY" | awk '{print $1}')"
actual_bytes="$(stat -c '%s' "$OUTPUT_PLY")"

if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "PLY SHA-256 mismatch: expected $EXPECTED_SHA256, got $actual_sha256." >&2
  exit 1
fi
if [[ "$actual_bytes" != "$EXPECTED_BYTES" ]]; then
  echo "PLY byte-size mismatch: expected $EXPECTED_BYTES, got $actual_bytes." >&2
  exit 1
fi

python3 - "$OUTPUT_PLY" "$EXPECTED_COUNT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = int(sys.argv[2])
with path.open("rb") as stream:
    header = stream.read(4096).split(b"end_header\n", 1)[0].decode("ascii")
line = next(item for item in header.splitlines() if item.startswith("element vertex "))
actual = int(line.rsplit(" ", 1)[1])
if actual != expected:
    raise SystemExit(f"PLY vertex count mismatch: expected {expected}, got {actual}.")
PY

sha256sum "$OUTPUT_PLY" > "$OUTPUT_PLY.sha256"
printf 'HOME_SCAN_LOD0_OK path=%s count=%s bytes=%s sha256=%s\n' \
  "$OUTPUT_PLY" \
  "$EXPECTED_COUNT" \
  "$actual_bytes" \
  "$actual_sha256"
