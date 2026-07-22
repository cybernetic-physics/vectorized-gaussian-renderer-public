# External release envelope

`release_envelope.py` seals the mutable publication layer around an already
verified scientific evidence bundle. It is intentionally offline: GitHub and
R2 must first be checked anonymously, and those checks must be supplied as
hash-pinned JSON artifacts. The final envelope contains no timestamps,
credentials, or host-local paths.

The tool fails unless all of the following agree:

- the benchmark tag resolves to the benchmark commit;
- the publication content commit descends from the benchmark commit;
- the final release commit is a merge commit containing the publication
  commit, the release tag resolves to that merge, and its complete load-bearing
  tree and full repository tree are byte-identical to the publication commit;
- the local Git origin, GitHub release URL, repository visibility, and
  anonymous GitHub verification record identify the same repository;
- the bundle inventories exactly one `publication/article.md`, and those bytes
  equal `post.md` at both the immutable publication commit and final merge;
- `post.md` and `README.md` have identical bytes at the publication commit,
  final merge, local input, and anonymous raw GitHub URLs;
- the full evidence-bundle, aggregate-verification, strict claim-ledger, and
  deterministic-archive verifiers pass as one canonical release-verification
  boundary, every bundle object passes the binary-aware privacy scan, and the
  envelope embeds the path-free aggregate/claim receipts;
- the supplied code-freeze record is the exact code-freeze object inventoried
  by the bundle and still reconstructs from local Git;
- the R2 receipt binds the archive SHA-256 and byte count, uses a complete
  hash-bearing logical key, and reports every authenticated and anonymous
  verification gate passing;
- the separate R2 public-verification record binds that exact receipt, every
  public object key/URL/hash/size, HTTP 200 full GET, and HTTP 206 range GET.

## Inputs

The spec schema is `publication-external-release-envelope-spec-v1`:

```json
{
  "artifacts": {
    "article": {"bytes": 1, "path": "post.md", "sha256": "<64 lowercase hex>"},
    "bundle_archive": {"bytes": 1, "path": "evidence.tar", "sha256": "<64 lowercase hex>"},
    "bundle_manifest": {"bytes": 1, "path": "bundle/manifest.json", "sha256": "<64 lowercase hex>"},
    "code_freeze": {"bytes": 1, "path": "code-freeze.json", "sha256": "<64 lowercase hex>"},
    "github_public_verification": {"bytes": 1, "path": "github-public.json", "sha256": "<64 lowercase hex>"},
    "r2_public_verification": {"bytes": 1, "path": "r2-public.json", "sha256": "<64 lowercase hex>"},
    "r2_receipt": {"bytes": 1, "path": "r2-receipt.json", "sha256": "<64 lowercase hex>"},
    "readme": {"bytes": 1, "path": "README.md", "sha256": "<64 lowercase hex>"}
  },
  "repository": {
    "benchmark_commit": "<40 lowercase hex>",
    "benchmark_tag": "benchmark-gcp-l4-matched-v2",
    "final_merge_commit": "<40 lowercase hex>",
    "name": "vectorized-gaussian-renderer-public",
    "owner": "cybernetic-physics",
    "publication_content_commit": "<40 lowercase hex>",
    "release_tag": "vgr-gcp-l4-matched-v2",
    "release_url": "https://github.com/cybernetic-physics/vectorized-gaussian-renderer-public/releases/tag/vgr-gcp-l4-matched-v2",
    "url": "https://github.com/cybernetic-physics/vectorized-gaussian-renderer-public",
    "visibility": "public"
  },
  "schema_version": "publication-external-release-envelope-spec-v1"
}
```

All artifact paths may be absolute or relative to the caller, but every one
must be a regular non-symlink file with the declared SHA-256 and byte count.
Paths are never copied to the output.

The R2 public record schema is `publication-r2-public-verification-v1`. Its
root fields are exactly `artifact`, `checks`, `objects`, `pass`, `receipt`, and
`schema_version`. `artifact` is the normalized artifact block emitted into the
envelope; `checks` exactly equals the receipt's `verification` block;
`receipt` is the receipt's `{bytes, sha256}` identity. Each `objects` entry is:

```json
{
  "bytes": 1,
  "full_get_status": 200,
  "key": "<exact public object key>",
  "range_get_status": 206,
  "sha256": "<64 lowercase hex>",
  "url": "https://<public-host>/<exact encoded key>"
}
```

The GitHub public record schema is
`publication-github-public-verification-v1`. Its root fields are exactly
`content`, `pass`, `refs`, `release`, `repository`, and `schema_version`.
`repository` records the canonical URL, `visibility: public`, and anonymous
HTTP 200. `release` records the URL, tag, target merge commit, and anonymous
HTTP 200. `refs` binds the benchmark and release tag targets. `content`
contains `article` and `readme` records with `git_path`, final `commit`,
`bytes`, `sha256`, canonical raw GitHub URL, and anonymous HTTP 200.

All JSON evidence records must use sorted, indented canonical JSON with one
trailing newline. Unknown fields are rejected at every level.

## Build and verify

Generate the code-freeze record with immutable refs; do not use `HEAD` as the
publication ref:

```bash
python3 scripts/verify_publication_code_freeze.py \
  --benchmark-ref benchmark-gcp-l4-matched-v2 \
  --publication-ref <publication-content-commit> \
  --output <code-freeze-record>
```

Do not hand-author the two network evidence records. Generate them with the
anonymous, read-only verifier after R2 publication and after the GitHub release
is publicly visible:

```bash
python3 publication/verify_public_release.py r2 \
  --receipt <r2-receipt.json> \
  --archive <bundle-archive.tar> \
  --output <r2-public.json>

python3 publication/verify_public_release.py github \
  --repository-url https://github.com/cybernetic-physics/vectorized-gaussian-renderer-public \
  --benchmark-tag benchmark-gcp-l4-matched-v2 \
  --benchmark-commit <benchmark-commit> \
  --release-tag vgr-gcp-l4-matched-v2 \
  --final-merge-commit <final-merge-commit> \
  --article post.md \
  --readme README.md \
  --output <github-public.json>
```

The verifier accepts no credential argument and sends no authorization header.
It writes canonical, timestamp-free, path-free, write-once records only after
full public-byte and immutable-ref verification succeeds.

After the GitHub merge/tag/release and R2 publication have been independently
verified, derive the local spec from the exact files. Do not hand-enter byte
counts or hashes:

```bash
python3 publication/write_release_spec.py \
  --article post.md \
  --bundle-archive <bundle-archive.tar> \
  --bundle-manifest <bundle-root/manifest.json> \
  --code-freeze <code-freeze-record.json> \
  --github-public-verification <github-public.json> \
  --r2-public-verification <r2-public.json> \
  --r2-receipt <r2-receipt.json> \
  --readme README.md \
  --owner cybernetic-physics \
  --name vectorized-gaussian-renderer-public \
  --benchmark-commit <benchmark-commit> \
  --benchmark-tag benchmark-gcp-l4-matched-v2 \
  --publication-content-commit <publication-content-commit> \
  --final-merge-commit <final-merge-commit> \
  --release-tag vgr-gcp-l4-matched-v2 \
  --output <release-spec.json>
```

The writer rejects missing, nonregular, symlinked, changing, duplicate, or
aliased inputs. It derives every artifact identity, validates the exact
repository schema used below, rereads each input immediately before an atomic
write, and accepts an existing output only when its canonical bytes are
identical. The local spec contains absolute host paths by design; it is an
offline control-plane input and must not be published. The envelope removes
those paths.

Then build the path-free envelope:

```bash
python3 publication/release_envelope.py \
  --spec <release-spec.json> \
  --repo-root <clean-local-checkout> \
  --output <external-release-envelope.json>
```

The output is write-once: an identical rerun is accepted; different bytes at
the same output path are rejected.

Run the offline adversarial tests with:

```bash
python3 -m unittest \
  publication.test_scan_public_artifact \
  publication.test_verify_public_release \
  publication.test_write_release_spec \
  publication.test_release_envelope -v
```

`scan_public_artifact.py` can also audit any proposed public text/JSON file.
It reuses the evidence bundle's canonical secret scanner and adds checks for
personal host paths, credential-bearing URLs, signed secret queries, and
optional timestamp rejection.
