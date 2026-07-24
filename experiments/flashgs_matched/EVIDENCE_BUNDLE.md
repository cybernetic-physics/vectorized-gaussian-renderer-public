# Matched evidence bundle contract

The publication artifact is a closed, relocatable, content-addressed bundle.
Raw benchmark JSON remains unchanged and may retain absolute measurement-host paths for
provenance. Consumers must resolve its `{path, bytes, sha256}` records through
the bundle object store: in bundle mode, `path` is ignored and the object is
loaded from `objects/sha256/<first-two-hex>/<sha256>` after byte-count and
SHA-256 verification.

This package is local until a separate, explicitly authorized R2 publication
step. None of the commands below upload anything.

## Stage the publication root

Copy only final, reviewed evidence into a new staging directory. The staging
directory must contain every file intended for publication and nothing else.
Do not put the Home Scan PLY, a Git checkout, caches, build scratch, SSH/AWS
configuration, `.env` files, or credentials in it.

The inventory is not a one-file-per-role checklist. It freezes the complete
decisive matrix shape and rejects missing **or additional** acceptance-role
members:

- 8 camera contracts;
- 8 Custom batch-capacity calibrations and one FlashGS B1024 demand survey;
- 32 primary renderer/contract/batch runs and 8 B128 independent repeats;
- 8 oracle captures and manifests, plus 32 raw fidelity reports, 32 matched
  fidelity summaries, and 32 fidelity CSVs;
- 8 bounded profiles, each with its control run, wrapper evidence, raw
  `.nsys-rep`, SQLite export, profile summary, both occupancy records, and all
  five Nsight CSV reports (40 CSVs total);
- the occupancy evidence for matrix launch, every calibration, every timed
  run, every repeat, every oracle, and every profile;
- the log and exact command record for all 100 matrix subprocesses;
- the clean source manifest, matrix invocation, schema-v3 FlashGS adapter
  attestation, gsplat build attestation, canonical summary and Markdown,
  hash-bound direct-B512 capacity-amendment record and its eight original
  artifacts, claim ledger, code freeze, machine record, verification record,
  and frozen canonical validator source.

The accepted baseline label is exactly **FlashGS-derived matched-contract
port**. The adapter-attestation v3 semantic classification explicitly rejects
the labels “upstream FlashGS,” “minimally adapted FlashGS,” and
“integration-only FlashGS.”

Additional files use one or more of the allowlisted supporting roles in
`evidence_bundle.py`. Every file needs an explicit role. The writer rejects an
unclassified file, and the builder rejects any file added, removed, or changed
after the inventory snapshot.

Prepare a JSON object mapping each staging-root-relative path to a sorted role
list. Prepare a separate JSON list for external dependencies. Home Scan is an
external content-addressed dependency and must look like this (using the
published manifest URL only after it has itself been verified):

```json
[
  {
    "bytes": 1203883212,
    "content_addressed_key": "datasets/home-scan-lod0/29cee1594654/home-scan-lod0.ply",
    "dependency_id": "home-scan-lod0",
    "license": "CC-BY-NC-4.0",
    "media_type": "application/octet-stream",
    "retrieval_manifest_url": "https://example.invalid/home-scan-lod0.manifest.json",
    "roles": ["scene-dataset"],
    "sha256": "29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267"
  }
]
```

The content-addressed key must contain the first 12 hexadecimal characters of
the full digest. A retrieval-manifest URL is optional; when present it must be
plain HTTPS without credentials, query parameters, or fragments. The scene
must be referenced by an artifact record in the evidence graph, but its PLY
bytes must not appear in the staging root or object store.

Freeze the inventory:

```bash
python3 scripts/write_matched_evidence_inventory.py \
  --input-root /path/to/final-staging \
  --bundle-name gcp-l4-matched-v9 \
  --roles /path/to/roles.json \
  --external-dependencies /path/to/external-dependencies.json \
  --output /path/to/publication-inventory.json
```

## Build and verify

Build the directory bundle and deterministic, uncompressed tar outside the
staging root:

```bash
python3 scripts/build_matched_evidence_bundle.py \
  --input-root /path/to/final-staging \
  --inventory /path/to/publication-inventory.json \
  --semantic-root /path/to/original-matched-result \
  --project-root /path/to/exact-clean-benchmark-checkout \
  --output-root /path/to/gcp-l4-matched-v9 \
  --archive /path/to/gcp-l4-matched-v9.tar
```

`--semantic-root` is the original matrix result while its recorded host-local
artifact paths are still live. Before sealing any bytes, the builder verifies
that the staged source manifest, summary, and validator match that result and
the clean checkout. It then executes the canonical summary validator in
read-only `--verify-existing` mode over all eight batches. That validator
reloads the 8 Custom calibrations, the FlashGS B1024 demand survey, 40 timed
runs, oracles, raw output arrays,
fidelity reports, occupancies, adapter/build attestations, wrapper evidence,
and command records; it reconstructs the summary and requires exact equality.
A true-looking `pass` field cannot substitute for this reconstruction. A
nonzero validator exit, missing sentinel, changed summary, or any failed
scientific/headline gate leaves no output bundle.

The builder recursively follows every artifact record in every JSON object.
Each target must either be copied into the content-addressed object store or
match a declared external dependency. Missing targets, stale hashes, unused
external dependencies, symlinks, unsafe relative paths, secret-like paths or
content, dataset PLY bytes, and unlisted input files are fatal.

The manifest has no timestamp or host path. Its ID is SHA-256 over canonical
JSON before the `manifest_id` field is added. Object paths and archive inputs
are sorted. Tar headers use UID/GID/mtime zero, empty owner names, and mode
`0444`, so identical bytes and inventory produce an identical manifest ID and
tar SHA-256.

Verify after copying or downloading the package, even when all original
measurement-host paths are dead:

```bash
python3 scripts/verify_matched_evidence_bundle.py \
  --bundle-root /path/to/gcp-l4-matched-v9 \
  --archive /path/to/gcp-l4-matched-v9.tar
```

Verification recomputes every hash and byte count, the inventory ID, manifest
ID, JSON artifact-reference graph, exact matrix cardinalities and paths,
acceptance-role checks, semantic-validation attestation bindings, exact object
set, and deterministic archive metadata and member order. A missing, changed,
dangling, unreferenced, symlinked, or extra object fails verification. The
sealed semantic attestation binds the exact validator, source manifest,
summary, and reconstructed-summary digest, so relocation does not depend on
the dead measurement-host paths.

## Claim ledger

The ledger schema is `flashgs-matched-claim-ledger-v1`. It must have
`pass: true` and at least one unique claim. Every claim records:

- `claim_id` and `article_location`;
- `display_value` exactly as it appears in the article;
- the displayed number's immutable `artifact` record and one or more RFC 6901
  `json_fields` pointers;
- `hardware`, `scene`, `output_contract`, and `equation_contract`.

The target must be an inventoried JSON publication artifact and every pointer
must resolve. Because that artifact is part of the recursively verified graph,
a blog number cannot point to an absent, unclassified, changed, or nonexistent
field. Remove any article number that is not represented in the passing
ledger.
