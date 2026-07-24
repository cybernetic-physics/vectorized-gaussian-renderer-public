# Immutable evidence publication

The publication path is fail-closed and has three layers:

1. the frozen scientific matrix and exact-current validation gates;
2. a relocatable, content-addressed scientific bundle containing the final
   article and strict claim ledger; and
3. an external release envelope binding that bundle to anonymous R2 bytes and
   the final public Git merge, tags, and release.

Run `run_post_matrix_gates.py` only after the frozen matrix has produced a
passing `summary.json`. Before that coordinator starts, convert the private
historical B64 diagnosis into its public derivative with
`redact_b64_diagnosis.py`. The transformer replaces only same-length private
root strings and the SHA-256 strings transitively affected by those replacements;
it proves every output byte round-trips, emits no private root, and writes a
deterministic transformation manifest. The unchanged repair verifier must then
rebuild the five-case, five-culprit, 92-pixel diagnosis graph from the derivative.

```bash
python3 publication/redact_b64_diagnosis.py \
  --historical-root /private/b64-short-tail-pre-fix-5eca4e6 \
  --diagnosis-index /private/diagnosis-index.json \
  --diagnosis-lock experiments/flashgs_matched/B64_DIAGNOSIS_LOCK.json \
  --known-failure-manifest experiments/flashgs_matched/B64_KNOWN_FAILURE_CASES.json \
  --output-root /staging/public-b64/outputs/publication-readiness/b64-short-tail-pre-fix-5eca4e6 \
  --output-index /staging/public-b64/evidence/diagnostics/diagnosis-index.json \
  --output-lock /staging/public-b64/experiments/flashgs_matched/B64_DIAGNOSIS_LOCK.json \
  --output-manifest /staging/public-b64/evidence/diagnostics/privacy-redaction-manifest.json
```

Pass those four public derivative paths to `run_post_matrix_gates.py` through
`--historical-b64-root`, `--diagnosis-index`, `--diagnosis-lock`, and
`--diagnosis-redaction-manifest`. Pass the clean public FlashGS checkout at
commit `cdfc4e4002318423eda356eed02df8e01fa32cb6` through
`--flashgs-source`; it must be the real `FlashGS` sibling of the frozen
benchmark checkout so the upstream-source audit cannot skip. The coordinator's
`--output-root` must be a fresh child of the matrix root. Before GPU work, it
independently verifies the derivative and writes an inventory whose artifact
records bind all 30 public graph inputs. It serializes all GPU work under the
cooperative executor lock, writes the canonical P1-versus-P128 ablation alias,
and creates and immediately revalidates `publication/verification.json`. Any
gate failure stops the ladder; do not reuse a partially failed root as release
evidence.

Then run `finalize_article.py --evidence-root <matrix-root> --template post.md`.
It emits `publication/article.md` and `publication/claim-ledger.json`, refuses
unsupported winner language, and recomputes every displayed value. Copy the
article bytes to `post.md`, update the nonnumeric README release links, commit
the final publication content, and generate the code-freeze record against
that immutable commit. The post-matrix coordinator derives
`publication/machine-provenance.json` from all 32 primary run environments and
binds it through aggregate verification. Add the capacity amendment, frozen
semantic validator, and code-freeze record to the matrix root.

`write_bundle_inputs.py` derives the exact 607-file required-role contract from
the frozen benchmark source and marks only additional reviewed files as
supporting artifacts. Use its outputs with the repository's
`write_matched_evidence_inventory.py`, `build_matched_evidence_bundle.py`, and
`verify_matched_evidence_bundle.py`. Verify the copied bundle and deterministic
archive again from a relocated directory before upload. The canonical
`verify_release.py` boundary additionally privacy-scans every content-addressed
object, treating binary Nsight/SQLite string tables as potential captured
environment data and rejecting secret assignments or host-user paths.

Pass the immutable Home Scan retrieval manifest URL
`https://cdn.jsdelivr.net/gh/cybernetic-physics/vectorized-gaussian-renderer-public@b4a4fd0ec6962d905e9bc33672607e5161d6ee05/publication/manifests/b231602598b1eb039175dcb7edbd475167c2fc92011c5e975a4727c62b9f74b9.json` to `write_bundle_inputs.py`. The URL is pinned to the commit that
introduced the content-addressed file, and anonymous retrieval must reproduce
SHA-256 `b231602598b1eb039175dcb7edbd475167c2fc92011c5e975a4727c62b9f74b9`.

After R2 upload and anonymous verification, merge the publication PR with a
merge commit, create the release tag and GitHub release, verify anonymous raw
article/README and tag URLs, and build the separate external envelope with
`release_envelope.py`. `RELEASE_ENVELOPE.md` documents its strict input schemas.
Deallocate the benchmark GPU only after the bundle is durably mirrored and all
public retrieval checks pass.

`publish_immutable_r2.py` publishes an already-audited evidence archive without
changing the frozen benchmark tree. It uses Cloudflare R2's S3-compatible API
so every mutation can carry the atomic `If-None-Match: *` precondition. Plain
`wrangler r2 object put` is deliberately not used: omitting Wrangler's
`--force` flag does not make its upload create-only.

Export a bucket-scoped R2 Object Read & Write S3 credential tuple only in the
publishing process environment:

```bash
export CLOUDFLARE_ACCOUNT_ID=<32-hex-account-id>
export AWS_ACCESS_KEY_ID=<r2-access-key-id>
export AWS_SECRET_ACCESS_KEY=<r2-secret-access-key>
# Export AWS_SESSION_TOKEN only when the issued credentials require it.
```

The helper has no credential command-line arguments. It reads these values only
to sign HTTPS requests and never prints them, writes them, or includes them in
the receipt. A Wrangler OAuth login or `CLOUDFLARE_API_TOKEN` alone is not this
S3 credential tuple, so the helper fails closed instead of falling back to an
unconditional upload. Keep the credentials out of shell history and unset them
when publication finishes.

The requested logical key must contain the archive's complete lowercase
SHA-256, for example:

```bash
python3 publication/publish_immutable_r2.py \
  --bucket vectorized-gaussian-renderer-assets \
  --public-base-url https://pub-243008be935848b6accaf262f04a7b82.r2.dev \
  --file /path/to/gcp-l4-matched-v9.tar \
  --key evidence/gcp-l4-matched-v9/<full-archive-sha256>.tar \
  --download-name gcp-l4-matched-v9.tar \
  --receipt /path/to/gcp-l4-matched-v9.r2-receipt.json
```

Before the first upload, the helper snapshots the local bytes and downloads
every derived remote key through authenticated, SigV4-signed R2 access. Equal
objects are reused; any unequal object aborts the entire preflight. Missing
objects are re-probed immediately before upload, then created with a signed
`If-None-Match: *` PutObject request. If another writer wins that race, R2
rejects this helper's write; the helper downloads the winner and accepts only
an exact byte-for-byte identity. It never overwrites the concurrent object.

This guarantees create-only behavior for this publication run. R2 does not
make the key permanently immutable: another authorized writer could overwrite
it later. The full-SHA key, authenticated/public verification, and receipt make
such a later replacement detectable, so write credentials must remain tightly
scoped and short-lived.

Artifacts above 300 MiB are split into deterministic 256 MiB parts. Each part
key contains both the full archive SHA-256 and full part SHA-256. The canonical
JSON index contains the archive identity and ordered part identities, and its
key contains its own full SHA-256. Consumers concatenate parts in index order
and require the reconstructed byte count and SHA-256 to equal the `artifact`
record.

After upload or reuse, the helper requires:

- authenticated full-download byte counts and SHA-256 values before and after
  public verification;
- anonymous HTTP HEAD metadata with the configured CORS origin;
- anonymous full public downloads with matching byte counts and SHA-256;
- anonymous head and tail ranges returning HTTP 206;
- an `Origin` request with the required CORS allow/expose headers;
- exact content type, attachment disposition, and immutable cache control; and
- streaming reconstruction of every multipart archive.

The receipt contains no timestamp, local hostname, local path, command
environment, credential, or upload/reuse history. It intentionally records the
public endpoint, bucket, keys, and object metadata. Identical inputs and
endpoint settings produce identical receipt bytes. A receipt is written only
after every verification passes, and an existing unequal receipt is never
overwritten.

The configured R2 bucket is public. Audit raw JSON, profiler databases/reports,
logs, and binary metadata for private hostnames, usernames, paths, and secrets
before invoking this helper. The current `r2.dev` endpoint is a public-preview
endpoint; use an approved custom domain for production-scale distribution.

Generate the separate anonymous R2 record from the sealed receipt and original
archive after the publisher succeeds:

```bash
python3 publication/verify_public_release.py r2 \
  --receipt /path/to/gcp-l4-matched-v9.r2-receipt.json \
  --archive /path/to/gcp-l4-matched-v9.tar \
  --output /path/to/gcp-l4-matched-v9.r2-public.json
```

This second pass uses no R2 credential. It re-downloads every public object,
checks its full SHA-256, HEAD metadata, CORS, and exact head/tail ranges, and
streams multipart objects in receipt order to reconstruct the archive hash.
It rejects redirects, mutable metadata, symlink inputs, noncanonical receipts,
input changes during verification, and an existing unequal output record.

After the merge commit, release tag, final non-draft GitHub release, and public
visibility exist, generate the GitHub record:

```bash
python3 publication/verify_public_release.py github \
  --repository-url https://github.com/cybernetic-physics/vectorized-gaussian-renderer-public \
  --benchmark-tag benchmark-gcp-l4-matched-v9 \
  --benchmark-commit <benchmark-commit> \
  --release-tag vgr-gcp-l4-matched-v9 \
  --final-merge-commit <full-merge-commit> \
  --article post.md \
  --readme README.md \
  --output /path/to/gcp-l4-matched-v9.github-public.json
```

The GitHub pass also uses no credential or `gh` session. It verifies the
anonymous repository and release pages, anonymous API visibility, both tag
targets (including annotated-tag dereferencing), merge-parent shape, and the
exact raw `post.md` and `README.md` bytes at the final merge. It closes the
check by re-reading visibility, release state, and tag targets. Both generated
records exactly match the schemas consumed by `release_envelope.py` and contain
no timestamps or local paths.

Use `write_release_spec.py` to derive the envelope spec from the final article,
README, archive, manifest, code-freeze record, R2 receipt, and both anonymous
verification records. It computes all sizes and SHA-256 values itself, rejects
symlinks and changing or aliased inputs, and writes canonical bytes only once.
The generated spec is local and path-bearing; pass it to `release_envelope.py`
and publish only the resulting path-free envelope. See `RELEASE_ENVELOPE.md`
for the complete command and release-ref inputs.

Run the offline mock suite with:

```bash
python3 -m unittest \
  publication.test_publish_immutable_r2 \
  publication.test_verify_public_release \
  publication.test_write_release_spec -v
```
