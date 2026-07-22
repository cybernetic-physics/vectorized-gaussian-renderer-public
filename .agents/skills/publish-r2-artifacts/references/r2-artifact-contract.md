# Cloudflare R2 artifact contract

## Current release

| Field | Value |
|---|---|
| Bucket | `vectorized-gaussian-renderer-assets` |
| Public base | `https://pub-243008be935848b6accaf262f04a7b82.r2.dev` |
| Wrangler | `4.111.0` |
| Flyby prefix | `flybys/home-scan/v1-a59ed5ade5b2/` |
| Dataset prefix | `datasets/home-scan-lod0/29cee1594654/` |
| Cache-Control | `public, max-age=31536000, immutable` |

The R2.dev address is suitable for this public preview, not high-volume
production distribution. Add a custom R2 domain before relying on stable
production throughput or cache controls.

## Source of truth

`cloudflare/r2-assets.json` defines each source/key pair and pins:

- source path;
- public object key;
- upload method;
- byte count and SHA-256;
- content type and disposition;
- immutable cache policy.

Do not infer publication state from filenames. Validate the manifest against
local bytes and the public endpoint.

## CORS contract

Apply `cloudflare/r2-cors.json`:

```bash
npx --yes wrangler@4.111.0 r2 bucket cors set \
  vectorized-gaussian-renderer-assets \
  --file cloudflare/r2-cors.json \
  --force

npx --yes wrangler@4.111.0 r2 bucket cors list \
  vectorized-gaussian-renderer-assets
```

The rule permits public `GET`/`HEAD`, accepts `Range`, and exposes
`Accept-Ranges`, `Content-Length`, `Content-Range`, `Content-Type`, and `ETag`.
Test with an `Origin` header because R2 returns CORS headers only for a CORS
request.

## Small-object publication

Wrangler currently supports one object at a time and files up to 315 MB:

```bash
python3 scripts/publish_r2_artifacts.py --method wrangler
```

The script validates local hashes, invokes the pinned Wrangler CLI with
explicit HTTP metadata, downloads every small public object, and compares its
full SHA-256.

## Large-object publication

The canonical Home Scan PLY is 1,203,883,212 bytes, so it uses the R2
multipart API through a short-lived Worker binding instead of Wrangler's
single-object endpoint.

`scripts/publish_home_scan_r2.sh`:

1. deploys `cloudflare/r2-multipart-uploader`;
2. generates an in-memory 256-bit bearer token;
3. stores it as a Worker secret;
4. uploads eighteen parts with retry and abort handling;
5. completes the object;
6. deletes the Worker;
7. verifies public length, type, and matching head/tail byte ranges.

The Worker only accepts authenticated keys under `datasets/`, limits each part
to 96 MiB, and exposes create/part/complete/abort operations. It must not remain
deployed after the upload.

## Public verification

Run:

```bash
python3 scripts/publish_r2_artifacts.py --verify-only
```

Expected evidence:

- `LOCAL_ASSET_OK` for every source;
- `PUBLIC_ASSET_OK` with `full-sha256` for small objects;
- `PUBLIC_ASSET_OK` with `head-and-tail-range` for the PLY.

Python's default `Python-urllib` user agent receives HTTP 403 from the managed
R2.dev endpoint. The verifier therefore sends a named project user agent. Do
not remove it without retesting.

## README media

GitHub README rendering does not provide reliable external MP4 controls. Use a
small animated GIF as the inline preview and make it a link to the H.264 MP4.
Keep full custom and OVRTX MP4 links beside the checksum manifest and camera
contract.

Every dataset link must retain author/source/license attribution. Every visual
comparison must state its renderer equations and whether it is diagnostic or
acceptance evidence.
