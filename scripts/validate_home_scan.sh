#!/usr/bin/env bash
set -euo pipefail

HOME_SCAN_ROOT="${HOME_SCAN_ROOT:-/workspace/datasets/home-scan-source}"
EXPECTED_FILES=434
EXPECTED_WEBP=360
EXPECTED_MANIFEST_SHA256=ac0c8226d8f0e5359deb09026d12954bb7096143ebff8061c50303909e8f285d
EXPECTED_LOD0=21497908

test -d "$HOME_SCAN_ROOT"
test -f "$HOME_SCAN_ROOT/lod-meta.json"
jq -e . "$HOME_SCAN_ROOT/lod-meta.json" >/dev/null

file_count="$(find "$HOME_SCAN_ROOT" -type f | wc -l | tr -d ' ')"
webp_count="$(find "$HOME_SCAN_ROOT" -type f -name '*.webp' | wc -l | tr -d ' ')"
lod0_count="$(jq -s 'map(.count) | add' "$HOME_SCAN_ROOT"/0_*/meta.json)"

if [[ "$file_count" != "$EXPECTED_FILES" ]]; then
  echo "Expected $EXPECTED_FILES files, found $file_count." >&2
  exit 1
fi
if [[ "$webp_count" != "$EXPECTED_WEBP" ]]; then
  echo "Expected $EXPECTED_WEBP WebP payloads, found $webp_count." >&2
  exit 1
fi
if [[ "$lod0_count" != "$EXPECTED_LOD0" ]]; then
  echo "Expected $EXPECTED_LOD0 LOD0 records, found $lod0_count." >&2
  exit 1
fi

while IFS= read -r relative_path; do
  test -f "$HOME_SCAN_ROOT/$relative_path"
done < <(jq -r '.filenames[]' "$HOME_SCAN_ROOT/lod-meta.json")

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
(
  cd "$HOME_SCAN_ROOT"
  find . -type f -print0 | sort -z | xargs -0 shasum -a 256
) > "$manifest"

actual_manifest_sha256="$(shasum -a 256 "$manifest" | awk '{print $1}')"
if [[ "$actual_manifest_sha256" != "$EXPECTED_MANIFEST_SHA256" ]]; then
  echo "Home Scan manifest mismatch: expected $EXPECTED_MANIFEST_SHA256, got $actual_manifest_sha256." >&2
  exit 1
fi

printf 'HOME_SCAN_VALID root=%s files=%s webp=%s lod0=%s manifest=%s\n' \
  "$HOME_SCAN_ROOT" \
  "$file_count" \
  "$webp_count" \
  "$lod0_count" \
  "$actual_manifest_sha256"
