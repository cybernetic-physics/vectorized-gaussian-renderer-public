#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_MANIFEST="$PROJECT_ROOT/cloudflare/r2-assets.json"
CORS_FILE="$PROJECT_ROOT/cloudflare/r2-cors.json"
WORKER_CONFIG="$PROJECT_ROOT/cloudflare/r2-multipart-uploader/wrangler.jsonc"
WRANGLER_VERSION="4.111.0"
BUCKET="vectorized-gaussian-renderer-assets"
WORKER_NAME="$(
  printf 'vectorized-gaussian-r2-upload-%s-%s' "$(date +%s)" "$$"
)"
WRANGLER=(npx --yes "wrangler@$WRANGLER_VERSION")

cleanup_worker() {
  if [[ "${worker_deployed:-0}" == "1" ]]; then
    "${WRANGLER[@]}" delete "$WORKER_NAME" --force >/dev/null || true
  fi
}
trap cleanup_worker EXIT

cd "$PROJECT_ROOT"
python3 scripts/publish_r2_artifacts.py --local-only
"${WRANGLER[@]}" r2 bucket cors set "$BUCKET" \
  --file "$CORS_FILE" --force
"${WRANGLER[@]}" r2 bucket cors list "$BUCKET"
if [[ "${SKIP_WRANGLER_UPLOADS:-0}" == "1" ]]; then
  python3 scripts/publish_r2_artifacts.py \
    --method wrangler --verify-only
else
  python3 scripts/publish_r2_artifacts.py --method wrangler
fi

deploy_output="$(
  "${WRANGLER[@]}" deploy \
    --config "$WORKER_CONFIG" \
    --name "$WORKER_NAME" 2>&1
)"
printf '%s\n' "$deploy_output"
worker_deployed=1
worker_url="$(
  printf '%s\n' "$deploy_output" |
    sed -nE 's/.*(https:\/\/[^ ]+\.workers\.dev).*/\1/p' |
    tail -1
)"
if [[ -z "$worker_url" ]]; then
  echo "Could not determine the temporary Worker URL." >&2
  exit 1
fi

upload_token="$(openssl rand -hex 32)"
printf '%s' "$upload_token" |
  "${WRANGLER[@]}" secret put UPLOAD_TOKEN \
    --config "$WORKER_CONFIG" \
    --name "$WORKER_NAME" >/dev/null

R2_UPLOAD_TOKEN="$upload_token" \
  python3 scripts/upload_r2_large_object.py \
    --worker-url "$worker_url" \
    --file .artifacts/datasets/home-scan-lod0.ply \
    --key datasets/home-scan-lod0/29cee1594654/home-scan-lod0.ply \
    --sha256 29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267 \
    --bytes 1203883212 \
    --content-type application/octet-stream \
    --content-disposition 'attachment; filename="home-scan-lod0.ply"'
unset upload_token

cleanup_worker
worker_deployed=0
python3 scripts/publish_r2_artifacts.py --verify-only
