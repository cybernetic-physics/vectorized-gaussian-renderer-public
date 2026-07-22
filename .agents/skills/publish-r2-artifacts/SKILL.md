---
name: publish-r2-artifacts
description: Validate and publish renderer videos, images, trajectory evidence archives, camera contracts, manifests, profiler reports, and large Gaussian datasets to Cloudflare R2 with immutable keys, checksums, secret-safe authentication, multipart handling, and public or download round-trip verification. Use when storing benchmark evidence outside Git, publishing Home Scan, adding README media, or auditing an existing R2 release.
---

# Publish R2 Artifacts

Publish only audited bytes and leave a reproducible, secret-free publication
contract in the repository. A failed evaluation bundle may be published only
when its key and manifest clearly retain the failure verdict.

Never print, copy, archive, or persist Wrangler OAuth tokens, S3 credentials,
`.env` files, SSH keys, or secret-bearing configuration.

## Required reference

Read
[references/r2-artifact-contract.md](references/r2-artifact-contract.md)
before changing keys, bucket policy, upload code, public URLs, or README media.

## Public release workflow

1. Inspect `cloudflare/r2-assets.json`.
   - Every object must have a repository-relative source, versioned key, byte
     count, SHA-256, content type, disposition, and upload method.
   - Use a hash-bearing or versioned prefix before applying immutable caching.

2. Validate locally.
   - Run `python3 scripts/publish_r2_artifacts.py --local-only`.
   - Stop on any missing file, byte-count mismatch, or SHA mismatch.
   - Keep large datasets under `.artifacts/`; never add them to Git.

3. Configure public reads.
   - Set `cloudflare/r2-cors.json` with the pinned Wrangler version.
   - Allow only `GET` and `HEAD`; expose range and object metadata needed for
     browser playback and verification.
   - Confirm the bucket's public URL before updating documentation.

4. Upload small objects with the pinned Wrangler version.
   - Wrangler is appropriate only up to its current 315 MB limit.
   - Set `Content-Type`, `Content-Disposition`, and immutable
     `Cache-Control` explicitly.

5. Upload a large public object through the temporary multipart Worker.
   - Deploy `cloudflare/r2-multipart-uploader` using Wrangler.
   - Generate a random token in memory and set it as a Worker secret.
   - Upload equal 64 MiB parts with `scripts/upload_r2_large_object.py`.
   - Complete or abort the multipart upload, then delete the Worker.
   - Never commit or print the token and never create long-lived S3 keys.

6. Verify the public bytes.
   - For small objects, download and compare full SHA-256.
   - For large objects, require exact `Content-Length`, `Content-Type`, and
     matching first and last 64 KiB range responses.
   - Confirm CORS with an `Origin` header and range playback with HTTP 206.

7. Update documentation only after verification.
   - Use an inline GIF for GitHub-compatible playback and link it to the MP4.
   - Link manifests, camera contracts, source attribution, licenses, and
     checksums.
   - State whether media is a visual demo or acceptance evidence.

## External-scene releases

For each prepared external SOG scene, publish the canonical PLY and its
prepared-scene manifest under `datasets/<scene-id>/<ply-sha-prefix>/`, and
publish the custom, OVRTX, side-by-side MP4s plus GIF/poster/storyboard and
camera contracts under `flybys/<scene-id>/<release-id>/`. Add source,
attribution, and license links in the README beside the playable preview.
Keep CC BY-NC scenes marked non-commercial in both the release manifest and
README; do not combine them with claims of commercial redistribution rights.

Generate the per-scene manifest only after local QA is complete:

```bash
python3 scripts/build_r2_scene_release.py \
  --scene-manifest datasets/<scene-id>.manifest.json \
  --canonical-ply .artifacts/datasets/<scene-id>.ply \
  --flyby-root outputs/flyby/<scene-id>-v1 \
  --release-id v1-<short-scene-sha> \
  --output cloudflare/r2-assets-<scene-id>.json
python3 scripts/publish_r2_artifacts.py \
  --manifest cloudflare/r2-assets-<scene-id>.json
```

For the pinned Home Scan release, run:

```bash
scripts/publish_home_scan_r2.sh
```

Set `SKIP_WRANGLER_UPLOADS=1` to resume after small objects already passed
public SHA verification.

## Evaluation evidence archive workflow

Use the skill-local helper for an immutable evaluation handoff that does not
need to appear as one public browser object. It computes local SHA-256, uploads
through Wrangler, downloads every object for full round-trip verification, and
writes a secret-free receipt. Files over the single-object limit are split
into 256 MiB checksummed parts with a verified `.index.json`
reconstruction manifest.

1. Reopen and audit the evidence bundle before publication.
2. Archive the manifest, artifact SHA inventory, logs, profiles, and videos.
   Exclude `.git`, datasets, caches, build trees, and secrets.
3. Choose a content-addressed logical key such as
   `trajectory-evals/<run-id>/<archive-sha256>.tar`—never `latest`.
4. Run:

   ```bash
   python3 .agents/skills/publish-r2-artifacts/scripts/publish_r2_artifact.py \
     --bucket vectorized-gaussian-renderer-assets \
     --key trajectory-evals/<run-id>/<sha256>.tar \
     --file /path/to/archive.tar \
     --receipt /path/to/publication-receipt.json
   ```

5. Require `verification: pass` for the index and every part. Cite the bucket,
   logical key, index key, byte count, archive SHA-256, and receipt together.
6. Keep the source checkout and original remote artifacts until round-trip
   verification completes.

Use the temporary multipart Worker instead when consumers require one public,
range-addressable object rather than an indexed evidence archive.

## Failure rules

- Do not set immutable caching on an unversioned key.
- Do not upload generated media before encoded-frame visual QA.
- Do not use Wrangler for a single object above 315 MB.
- Do not treat an upload response as integrity verification.
- Do not expose a multipart Worker without authentication or leave it deployed.
- Do not upload raw workspaces, `.git`, unreviewed datasets, or secret files.
- Treat missing objects, failed downloads, size mismatches, checksum
  mismatches, and incomplete part indexes as publication failures.
- Treat `r2.dev` as a development/public-preview endpoint; use a custom domain
  before production-scale distribution.
