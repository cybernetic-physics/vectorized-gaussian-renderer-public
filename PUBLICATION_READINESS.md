# Publication readiness: production Gaussian rasterizer

This document is the release gate for the article in [`post.md`](post.md). It
separates the production rasterizer claim from experimental OptiX, Warp, OVRTX,
and historical cached results so that a blog edit cannot silently broaden the
evidence.

## Product definition

The product under test is `CustomCudaBackend` behind `RendererService`:

- inference-only screen-space EWA rasterization;
- one or more packed static Gaussian scenes resident on one CUDA device;
- independently moving, batched pinhole cameras;
- reusable CUDA outputs and workspace;
- degree-zero RGB, alpha, expected depth, and strongest-contributor semantic ID;
- Isaac Sim Kit/Fabric integration; and
- bounded physical sub-batching for logical batches larger than the installed
  workspace.

The headline does not include the experimental OptiX ray-volume tracer, the
Warp mesh tracer, the hybrid mesh compositor, or OVRTX. Those are separate
lanes with different algorithms and evidence.

## Allowed outcome-dependent claims if every P0 evidence gate passes

No renderer winner is predeclared. The final, independently reconstructed
summary selects one of four outcomes:

- If every claim-eligible row and every same-step CUDA/wall comparison favors
  Custom under both output contracts, report Custom as the winner only for one
  identified L4, Home Scan, the frozen camera population and schedule, and this
  low-resolution robotics-shaped renderer microbenchmark.
- If every comparison instead favors the FlashGS-derived matched-contract port,
  report the inverse claim with exactly the same scope.
- If the output contracts differ, report a split-by-contract result and no
  uniform renderer winner.
- If a range touches/crosses parity, report a mixed result and name the affected
  contract without converting its ratio of means into winner language.

Every qualifier is load-bearing. A passing derived-port comparison is not
sufficient to shorten any outcome to “faster than FlashGS.” This is not an
upstream-FlashGS, universal 3DGS, multi-GPU, training, nonlinear-camera, 4K,
end-to-end Isaac/Fabric, policy, or deployed-robot claim.

If a row is missing, it stays missing and no uniform or aggregate claim is
published. If any candidate fails same-equation fidelity, its timing remains
non-equivalent diagnostic evidence and supplies no ratio, geometric mean, or
winner claim. The article finalizer must emit an explicitly incomplete or
fidelity-failure diagnostic article rather than aborting silently or implying a
winner. The current strict ledger cannot bind partial/non-equivalent matrices,
so those diagnostic articles intentionally contain no performance numbers and
no release claim ledger.

## P0: required before publication

### 1. Complete the corrected matched FlashGS matrix

Use [`experiments/flashgs_matched/BENCHMARK_CONTRACT.md`](experiments/flashgs_matched/BENCHMARK_CONTRACT.md)
without changing the frozen trajectory after observing results.

Required dimensions:

| Dimension | Required values |
|---|---|
| Renderer | Custom, FlashGS-derived matched-contract port |
| Output contract | RGB-only, full sensor |
| Logical batch | 1, 8, 32, 64, 128, 256, 512, 1024 |
| Scene | Home Scan LOD0 SHA-256 `29cee1594654…` |
| Resolution | 128×128 |
| Motion | every camera changes every measured frame |
| Caches | projection and rendered-frame caches off |
| Timing | 8 warmups, 100 synchronized measured frames |
| Correctness oracle | pinned gsplat `77ab983…` |

Record for every row:

- GPU and synchronized wall latency distributions;
- images/s and megapixels/s;
- host submission time;
- pre/post steady-state NVML process-resident memory and separately measured
  Torch peak allocated/reserved memory;
- visible Gaussians and generated intersections;
- logical chunks and native submissions;
- allocation and capacity telemetry; and
- per-view fidelity verdicts.

Visible/intersection counters remain backend-native diagnostics with different
culling and record units. Custom's 1×1-bin records and FlashGS's 16×16-tile
records must never be compared as a common unit of work. Report installed sort
capacity and utilization because the fixed path also sorts sentinel entries.

A uniform winner claim requires all eight paired rows, every scheduled row to
pass, and the same renderer to be faster in both CUDA-event and synchronized-
wall latency at every one of the 100 matched trajectory steps in every row.
Raw ordered GPU, wall, and host-submission samples and observed same-step
minimum/median/maximum ratios must be published. Aggregate values are
recomputed from those arrays and inconsistent stored summaries are fatal. No
iid or 95% confidence inference is made from correlated trajectory frames.
B128 plus a separately run B1024 endpoint cannot establish behavior across the
B256/B512 chunking transition.

Run B128 three times in fresh processes for both renderers and output contracts
(the primary row plus two counterbalanced repetitions). The run-level winner
must agree in GPU and synchronized-wall latency across all three. Report
run-level dispersion separately from the 100 correlated camera steps.

Also run the mandatory within-Custom B128 P1-versus-P128 control in three fresh
trials for both output contracts. Its displayed speedup is P1 latency divided by
P128 latency. It establishes only the aggregate effect of that physical-
submission schedule at B128; it does not isolate a kernel mechanism, explain
the cross-renderer result, or establish a batch-scaling law. Report all trials
whether they favor P1, favor P128, or disagree.

Current execution: the corrected matrix is being refrozen as
`benchmark-gcp-l4-matched-v9`; no v9 timing result exists yet. The complete v8
timed rows remain diagnostic because their first profiler control stopped on a
false host-path match, and the fail-closed shutdown caused GCP to assign a new
physical L4 before profiling. The v9 contract still binds every artifact to one
exact launch-time UUID, but treats the NVIDIA L4 model—not an ephemeral source
constant—as the headline hardware class. It will use a fresh output root and
automatic shutdown. Frozen pre-repair traces establish the original sparse
edge-tile omission's exact cause:
the final cooperative load in short tile tails used an unassigned feature
offset and therefore read the first Gaussian's features. An older complete
Freiza run is also invalid because its candidates used 3.0-sigma support while
the oracle used 3.33. Do not restart the long matrix until the separate
post-repair verifier proves every historical case and mismatch.

### 2. Resolve or retain the semantic failure

The interrupted full-sensor affected-row capture from our pre-repair derived
port had mismatches clustered in sparse tile-boundary regions. The port returned
background, Custom matched gsplat at every bad pixel, and missing alpha was
material. A
proposed rectangle-containment cause was rejected: the old predicate accepts
its synthetic test and a randomized CPU comparison found no such false
negative. The faulty guard was inherited from pinned upstream source, but this
does not establish an upstream FlashGS product failure under its intended
equation and workload.

The source-grounded trace instead located the loss inside the optimized
compositor: its final load-enable predicate for short tile-range tails was ahead
of its offset-assignment predicate. The repaired guards add no allocations,
launches, synchronization, or work for ordinary
full warps, but they can restore sparse-tail shader work and therefore
invalidate every old timing.

Run the separate repair gate from a clean source manifest and fresh binary. It
must prove the historical cases and coordinates, both output contracts,
complete affected-row tensors against pinned gsplat, zero overflow, and
bounded sanitizer coverage. If it passes, restart the complete matrix from
that same source and discard every earlier candidate timing from the final
table. If it still fails, publish the failed row with no equivalence or winner
claim. Do not lower the threshold.

The historical RTX 3090 diagnosis retained private build/source roots in JSON,
logs, and debug metadata. Those exact bytes must never enter the public bundle.
Generate the deterministic same-length derivative with
`publication/redact_b64_diagnosis.py`, publish its transformation manifest and
tool, and bind all 30 transformed graph inputs through the generated public
inventory. Rerun the unchanged five-case/92-pixel graph verifier on the
derivative and privacy-scan every resulting file. The article must disclose
this path-only transformation; it must not describe the derivative as
byte-identical to the private capture.

The frozen pre-fix debug traces came from an sm_86 RTX 3090, whereas the
headline rerun uses an sm_89 L4. For the decisive rerun, intermediate
projection alpha and power therefore retain a frozen float32-ULP bound, and the
observed distances are retained per culprit.
This narrowly scoped bound does not apply to renderer outputs: every affected
RGB/alpha/depth/semantic pixel and the complete affected-row captures retain
their existing pinned-gsplat tolerances. Any value beyond the frozen bound must
fail the verifier.

Independent semantic review classified the current adapter as a
FlashGS-derived matched-contract port, not an integration-only patch. It
retains FlashGS's emission, sorting, range, and tile-compositor topology but
changes projection and image formation to match gsplat. The final evidence must
preserve that review and use the derived-port label everywhere.

Before publishing a claim about upstream FlashGS, add a separate RGB control
that preserves upstream projection and shader math and changes only service
integration, current-stream execution, preallocation, float output, and the
smallest no-readback capacity mechanism. Match Custom to that upstream equation
with an independently checked oracle. If the upstream-faithful control does not
agree with the derived-port direction, no “faster than FlashGS” claim is
allowed. Treat any full-sensor alpha/depth/semantic shader as a separately
named extension, because upstream FlashGS does not provide that contract.

The oracle coverage must include stratified measured steps 8, 57, and 107 for
B1, B128, B512, and B1024. B512 and B1024 samples must straddle every physical chunk
boundary, including cameras 127/128. The ordinary final-state `<=8` camera
capture is not enough to validate a 100-step dynamic or chunked workload.

### 3. Preserve profiler controls

After the unprofiled matrix completes, run short Nsight Systems controls for
Custom and FlashGS at B1 and B1024 using
[`scripts/profile_flashgs_matched_nsys.sh`](scripts/profile_flashgs_matched_nsys.sh).
Preserve:

- `.nsys-rep`;
- exported SQLite and CSV;
- parsed JSON summaries;
- the exact command and NVTX capture range;
- matching unprofiled result hashes; and
- zero CUDA runtime, driver, or virtual-memory allocator calls inside the
  measured range.

Profiled timings explain the result; they do not replace the primary timings.

### 4. Run exact-current correctness and Isaac gates

Use a clean, tagged benchmark commit and link it from the article. The later
publication commit may change only allowlisted prose and immutable artifact
references; an automated code-tree hash must prove that renderer, adapter,
harness, tests, and native sources are unchanged. Run in increasing cost after
the matrix releases the sole GPU executor:

1. Python unit and benchmark-contract tests;
2. native CUDA smoke, deterministic smoke, and multi-scene smoke;
3. finite `SimulationApp` headless smoke;
4. Kit-extension lifecycle and Fabric moving-camera tests;
5. all-output fidelity and deterministic replay;
6. Compute Sanitizer memcheck; and
7. a bounded stability soak.

Every production render must verify CUDA residency, finite RGB/alpha, finite
foreground depth, alpha range, foreground coverage, valid semantic IDs, and
zero capacity overflow. Always close `SimulationApp` in `finally`.

Current blocker: the clean matrix and the exact-current CUDA, Isaac/Fabric,
sanitizer, profiler, and stability records have not completed. Local
publication-tool tests do not substitute for those L4 gates.

Before every unprofiled matrix row, preserve a fail-closed occupancy record:
`nvidia-smi`, active CUDA compute processes, relevant Python/Isaac/profiler
processes, and `tmux` sessions. Abort on an unfamiliar workload. Record GPU
UUID, clocks, temperature, power, and process memory immediately before and
after each row. Only one benchmark executor may use the identified L4, and no
other GPU workload may share the node.

### 5. Publish a self-contained evidence package

The article cannot cite a developer's local workspace path. Mirror the final
result root before the remote workspace changes and publish an immutable
manifest containing:

- raw runs, camera contracts, oracle captures, fidelity summaries, and final
  tables;
- source/dependency commits and clean source manifests;
- scene, camera, source, binary, result, and profiler checksums;
- machine/toolchain provenance and commands;
- profiler reports and exports; and
- a canonical summary generated from the raw records.

The Home Scan external-dependency record must use `CC-BY-4.0` and the corrected,
hash-addressed dataset manifest. The older `CC-BY-NC-4.0` example in the frozen
benchmark-bundle documentation is erroneous and must not be copied into a
release artifact.

The package also needs a claim ledger mapping every displayed release-
performance article claim to an immutable artifact path, SHA-256, exact JSON
field(s), hardware, scene,
output contract, and equation. The byte-identical article must be inside the
bundle. A strict publication-only validator must recompute displayed values and
ratios, require one ledger entry per article marker, and bind the article hash;
a nonempty `display_value` and resolvable pointer are not sufficient. Performance
numbers absent from the passing ledger are removed from the article. Frozen
methodology quantities and correctness thresholds are validated through their
own required bundle roles rather than misrepresented as timing claims.

Older Custom-versus-gsplat, OVRTX, OptiX, and archived FlashGS records remain
engineering history. They lack the complete raw/provenance package required by
this release and must not supply its performance or scaling claims. The final
matched result and profiler controls receive new ledger rows only after
immutable packaging. This publication contains no high-resolution control.

Avoid circular provenance: the scientific bundle binds the tagged benchmark
source, exact-current validation, article bytes, and claim ledger. A separate
release envelope, created after the final publication commit, binds that commit
and tag to the bundle manifest ID, archive SHA-256, code-freeze result, R2
receipts, and public retrieval URLs. The release envelope stays outside the
scientific bundle.

Validate local SHA-256 before upload, authenticated remote bytes after upload,
and anonymous HTTP GET, ranges, CORS, cache, and content disposition after
publication. Use full-SHA content-addressed R2 keys, preflight collisions, skip
an identical existing object, and fail rather than overwrite an unequal object.
Prefer a custom R2 domain; otherwise explicitly accept and document the
`r2.dev` URL as the publication endpoint.

Current blocker: the privacy-safe derivative tooling is ready, but the
derivative, repaired gate, and final evidence root must be regenerated and
verified with the exact committed publication tool on the identified L4 before
any new performance result is publishable.

### 6. Package media rather than linking loose files

Each blog clip needs:

- MP4 and GIF preview;
- poster and storyboard;
- immutable camera contract;
- render and artifact manifests;
- `ffprobe` metadata;
- SHA-256 and byte count; and
- verified public HTTP metadata.

The four former `outputs/videos_hq/*.mp4` links were never present in the merged
tree. Do not restore those relative links. The existing Home Scan R2 flyby is a
valid visual demo but is not numerical fidelity evidence.

### 7. Final five-review gate

After the artifacts and final text are frozen, rerun five independent reviews:

1. claim and implementation audit;
2. benchmark/fairness audit;
3. raw-evidence and provenance audit;
4. skeptical reviewer/competitor audit; and
5. editorial/media/reproducibility audit.

Each reviewer must inspect the final commit and canonical evidence package. All
P0 objections must be resolved or disclosed in the article.

## Historical engineering evidence

Earlier L4 summaries motivated the fresh matrix and suggested that Custom
throughput was nearly flat over the tested physical batches. Their complete raw
timing, profiler, occupancy, and content-addressed provenance are not in this
publication package. They therefore supply no release number, winner, or
scaling claim. The fresh matrix must independently establish and package any
retained conclusion.

## P1: valuable follow-up, not required for the scoped L4 article

- If the article attributes the result to a particular design choice beyond the
  mandatory B128 physical-schedule control, add matched ablations:
  conservative foreground cutoff on/off; direct reprojection versus
  materialized projected records; equivalent fused versus staged sensor
  outputs; B1024 physical chunks 64/128/256; and fixed-capacity sentinel sort
  versus emitted-prefix count-readback sort. Report phase/kernel time, host
  submission, installed/used sort capacity, pre/post-cutoff intersections,
  memory, and gsplat fidelity. Without these, describe causes as hypotheses.
- Repeat the decisive matrix on an RTX 4090 and a higher-class datacenter GPU.
- Add a 64/128/256/512/1024 resolution crossover sweep.
- Add another large, adversarial Gaussian scene beyond the held-out Garage
  correctness result.
- Measure an actual policy/training loop with concurrent renderer and model work
  before claiming Tensor-core availability or RL/VLA iteration speed.
- Implementing truly batched FlashGS kernels would be a new renderer baseline,
  not a minimal adaptation.
- Rerun raster versus OptiX in-process with identical full outputs, dynamic
  cameras, scenes, and timing boundaries before promoting the tracer.
- Run clamp-only, k-buffer, alpha-threshold, AABB, and geometry-stress tracer
  ablations before causal optimization claims.
- Update OVRTX system comparisons with current animated transforms and tiled
  cameras; keep them separate from same-equation rasterizer comparisons.

## Release hygiene addressed by the publication branch

- Kit and production backend support defaults both use 3.33 sigma.
- Architecture documentation discloses adaptive-capacity counter reads and the
  scalar emitted-count synchronization path.
- Kit metadata identifies the repository rather than OpenAI.
- A root Apache-2.0 license matches `pyproject.toml`.
- The article separates production raster, experimental OptiX, Warp hybrid, and
  OVRTX claims and removes broken media links and mixed-contract tables.

These fixes do not become release evidence until the GPU gates pass, the final
bundle is anonymously verified, and the publication branch is reviewed and
merged.
