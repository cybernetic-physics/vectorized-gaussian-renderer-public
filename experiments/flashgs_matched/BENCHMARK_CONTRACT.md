# Decisive matched FlashGS benchmark contract

## Question

Can pinned public FlashGS replace `CustomCudaBackend` as the rasterizer inside
the same `RendererService` for a dynamic, massively batched Isaac robotics
sensor without losing correctness or efficiency?

## Frozen primary workload

- GPU: one Google Cloud NVIDIA L4 on instance `vgr-publish-l4w1a-20260722` in
  `us-west1-a`, UUID `GPU-b3c9268d-2b06-d924-90cc-d2171c86ef34`; every paired
  artifact must record that UUID, driver 580.159.03, PyTorch 2.11.0+cu128,
  CUDA runtime 12.8, and the actual CUDA compiler/toolchain identity.
- Scene: full Home Scan LOD0, exactly 21,497,908 Gaussians, PLY SHA-256
  `29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267`.
- Resolution: 128x128.
- Batches: 1, 8, 32, 64, 128, 256, 512, and 1024.
- Motion: an immutable 108-step camera contract per batch; eight warmup steps
  followed by 100 synchronized measured steps. Every camera pose changes at
  every measured step, and no two cameras share a pose within a frame.
- Precision: canonical float32 means, scales, WXYZ rotations, activated
  opacity, covariance, RGB, cameras, and float32 outputs; int64 semantics.
- Scene and all camera contracts are uploaded once and remain GPU-resident.
- Each renderer runs in a fresh process. Output and workspace allocations are
  complete before peak-memory reset and measurement.

### L4 capacity amendment before timing

The first L4 feasibility gate attempted the previously frozen direct B512
schedule before any timed renderer row. It failed while installing a
21,445,449,872-byte (19.973 GiB) backend workspace after the full scene and
outputs were resident. The explicit benchmark-reservation path had already
released its old workspace, so this is a physical-capacity result rather than
old/new workspace double residency. That failed source manifest, command,
occupancy record, and log remain immutable diagnostic evidence.

The replacement protocol is a new source commit and result root. It reuses the
already predeclared 128-camera physical chunk without testing alternative
chunk sizes: Custom B512 is four ordered P128 native submissions and B1024 is
eight. B1 through B256 remain direct. This correction is selected solely for
hardware feasibility; no timed row from that failed matrix invocation had
been observed. It does not establish that P128 is optimal, and the article
must disclose the native-submission count for every row.

The final evidence bundle must include
`publication/capacity-amendment.json` and the eight exact artifacts it binds:
the failed clean source manifest, matrix invocation, matrix-launch occupancy,
capacity-run occupancy, capacity command, allocator log, matrix driver log,
and nonzero exit record. The bundle validator pins their byte counts and
SHA-256 identities; prose alone cannot establish that the schedule was chosen
before headline timing.

## Frozen image-formation equation

Both backends use conventional screen-space 3DGS/EWA with:

- 3.33-sigma maximum support, matching pinned gsplat's
  `GAUSSIAN_EXTEND=3.33f`, further limited where opacity falls below `1/255`;
- inclusive near/far depth planes, rejection only when the projected 2D
  covariance determinant is `<= 0`, and no artificial covariance-diagonal or
  minimum-pixel radius floor; matching `radius_clip=0`, a projection is culled
  only when both integer radii are `<= 0`;
- zero 2D covariance epsilon;
- pixel centers at `(x + 0.5, y + 0.5)`;
- `alpha = min(0.99, opacity * exp(-0.5 * mahalanobis_squared))`;
- per-splat alpha rejection below `1/255`;
- front-to-back `T * alpha` compositing and termination at `T <= 1e-4`;
- black linear-RGB background;
- expected depth `sum(T * alpha * z) / sum(T * alpha)`, with positive infinity
  at zero accumulated alpha; and
- semantic ID from the strongest individual `T * alpha` contribution, with
  `-1` below accumulated alpha 0.01.

Home Scan has no canonical semantic annotation. The representative full-sensor
lane therefore assigns eight deterministic, spatially coherent octant labels
from the scene bounding-box center. The previous Gaussian-index-modulo-1024
assignment is retained as a separately named interleaved/high-cardinality
stress lane. Its fidelity result must not gate or be relabeled as the
representative robotics workload.

Pinned gsplat commit `77ab983ffe43420b2131669cb35776b883ca4c3c` is the correctness oracle.
Its known CUDA 12.8 event-creation compile typo is corrected with the recorded
`patches/gsplat-cuda-event-flags.patch`; the oracle verifies that this is the
only source-tree modification and records the patch SHA-256. The patch changes
two CUDA API names and does not alter projection, rasterization, or compositing.
The first oracle process must compile gsplat into a fresh dedicated
`TORCH_EXTENSIONS_DIR`; a content-hashed build attestation binds every CUDA/JIT
source input, `build.ninja`, build parameters, and the loaded native binary.
Every oracle process also executes a native behavioral probe that distinguishes
3.33 support from 3.0 support, verifies opacity-aware contraction, and verifies
inclusive near/far behavior. A prebuilt, source-local, unattested, stale, or
behaviorally mismatched gsplat binary is fatal.
Every oracle process also preserves and passes the same pre/post whole-node
occupancy gate as the candidate rows; an oracle rendered beside another CUDA
executor is ineligible.
Repository gates are PSNR >= 40 dB, SSIM >= 0.995, LPIPS <= 0.01, alpha MAE
<= 0.005, mean relative depth error <= 0.01, and foreground semantic agreement
>= 0.999 for every sampled view. The semantic gate is evaluated over the union
of pixels where candidate or oracle alpha is at least 0.01; whole-image
semantic agreement remains a diagnostic only.

## Backends

### Custom

- `CustomCudaBackend` compact path, with bounded physical sub-batching only at
  the two memory-limited logical batches.
- One logical `RendererService.render` call. Batches that fit use the existing
  single native submission unchanged; batches 512 and 1024 use a predeclared
  physical camera limit of 128 and respectively four and eight ordered native
  submissions sharing one fixed workspace and full logical output allocation.
- Projection-coherence and rendered-frame caches disabled.
- The existing custom depth/foreground cutoff remains part of its algorithm.
- Synchronization-free fixed-capacity global radix ordering is selected; the
  capacity is calibrated over the complete trajectory outside measurement.

### FlashGS-derived matched-contract port

- Public FlashGS commit `cdfc4e4002318423eda356eed02df8e01fa32cb6`.
- Not upstream-faithful or integration-only: projection, support, sampling,
  alpha, termination, color evaluation, and outputs change to match the pinned
  gsplat contract.
- Retains its warp-cooperative projection/key emission, global CUB radix sort,
  16x16 tile-range construction, and optimized compositor load schedule.
- Accepts GPU view matrices/intrinsics and canonical degree-zero RGB, runs on
  the active PyTorch CUDA stream, writes float outputs, and reuses all storage.
- Uses sentinel-filled fixed-capacity sorting to remove the public example's
  intersection-count CPU readback/synchronization. One hash-bound, untimed
  B1024 survey counts pre-write demand for every camera and every trajectory
  step; each exact batch prefix receives 5% headroom. Survey images are invalid
  by protocol, while every timed row independently proves zero overflow from
  device counters after each render. Any overflow is fatal.
- Executes the native FlashGS pipeline once per camera because upstream has no
  true camera batch. All B executions are included.
- Does not add a camera batching kernel, projection cache, rendered-frame
  cache, custom depth/foreground cutoff, or alternate ordering algorithm.

## Output contracts

1. RGB only: only float32 linear RGB is allocated and written. Internal alpha
   and transmittance remain necessary to conventional compositing.
2. Full sensor: float32 RGB, float32 alpha, float32 expected depth, and int64
   semantic ID. FlashGS accumulates the three added outputs in its existing
   compositor, and their complete cost is measured.

## Timing and metrics

For each measured frame, CUDA events bracket only `RendererService.render` and
the ending event is synchronized before the next frame. No output tensor is
copied to the CPU in the measured loop. Record:

- GPU batch/pipeline latency distribution and synchronized wall latency;
- all 100 ordered GPU, synchronized-wall, and host-submission samples, not just
  aggregate statistics;
- host submission latency measured around the service call before waiting;
- images/s and megapixels/s from summed GPU event time;
- pre/post steady-state NVML process-resident memory plus separately measured
  PyTorch peak allocated/reserved bytes;
- visible Gaussians and generated intersections from hash-bound, out-of-band
  full-trajectory capacity-calibration counters;
- installed fixed-sort capacity, native sort submissions, mean utilization,
  and the fact that unused slots are sentinel entries;
- exact upstream/source/camera/scene/machine provenance; and
- gsplat fidelity for both backends and both output contracts.

Every primary v4 matrix artifact is bound to a clean
`renderer-source-manifest-v2` checkout.
The manifest hashes every tracked source file and records the exact HEAD and
empty status; the runner verifies those fields again in the executing checkout
and records the loaded native-extension binary hash. Run, capture, oracle,
camera-bundle, fidelity, and profile summaries are content-hash linked, so a
stale artifact cannot be resumed under a newer result.

The independently frozen pre-fix trace intermediates were captured on sm_86.
When replayed on the headline sm_89 GPU, their target alpha and power must be
within 128 ordered float32 ULP; the artifact records both distances and rejects
129 ULP. This portability allowance is debug-only. It does not relax any
rendered RGB, alpha, depth, semantic, camera, capacity, or gsplat-oracle gate.

Before any timed Custom row, a separate full-output process calibrates capacity
over all 108 trajectory steps at 3.33 sigma. Before any timed FlashGS row, one
output-invalid B1024 process surveys exact pre-write demand and proves every
smaller camera contract is a byte-exact prefix. The passing renderer-specific
artifact binds source, native binary, scene, camera, GPU, equation, counters,
and installed capacity by SHA-256. Both timed output contracts, both B128
repeat processes, and both endpoint profiler controls consume that exact
artifact. Each timed process constructs its workspace at the bound capacity,
calls `prepare_outputs` once, performs exactly eight warmups, and performs no
trajectory preflight, explicit capacity reservation, or `empty_cache` call.

Strict steady-state memory acceptance uses three identical pre-measurement
Torch/NVML samples and one immediate post-measurement sample taken before
output validation. Current/reserved bytes and cumulative Torch allocation,
free, segment, and reservation counters must remain unchanged. The B1 and
B1024 Nsight controls must additionally report zero CUDA runtime, driver, or
virtual-memory allocator calls inside the exact outer NVTX capture range.

Short Nsight Systems controls at B1 and B1024 must preserve raw `.nsys-rep`,
exported SQLite/CSV, NVTX ranges, and unprofiled timing. Profiler timing is not
substituted into the primary table. Run each bounded control with
`scripts/profile_flashgs_matched_nsys.sh`; it consumes the exact calibration
artifact bound to the paired unprofiled result and calls the same runner in explicitly
non-headline `--profile-control` mode.

A uniform renderer-winner claim requires the complete ordered eight-batch
matrix and the same renderer to be faster in both CUDA-event and synchronized-
wall latency at every one of the 100 matched trajectory steps in every
scheduled row. The summary recomputes all aggregates from raw arrays and
reports ratio-of-mean latency plus the observed same-step
minimum/median/maximum. It makes no iid, run-to-run, or 95% confidence claim:
the trajectory steps are correlated deterministic work and each renderer has
one fresh process per row. Subsets, the interleaved semantic stress topology,
dirty source trees, missing endpoint profiles, or an observed same-step range
crossing 1.0 are diagnostic only. The complete L4 matrix establishes
only the hardware- and workload-scoped claim frozen in this contract; it is not
a claim about RTX 4090s, other GPUs, resolutions, scenes, or ordinary 3DGS
workloads.

B128 is the run-level stability anchor required by `BENCHMARK_PROTOCOL.md`.
After the primary row, run two additional fresh-process trials for each
renderer and output contract with the same calibration artifacts, trajectory,
source, GPU, and 100 measured frames. Counterbalance renderer/contract order.
Publish all three run means, same-step ranges, and descriptive
minimum/median/maximum spread over the three independent speedup ratios. No
significance test or confidence interval is implied. The anchor must reproduce
the primary B128 winner in both CUDA-event and synchronized-wall latency; frame
samples inside a trial are not treated as independent replicates.

Before each unprofiled row, preserve `nvidia-smi`, active compute-process,
Isaac/Python/profiler-process, and `tmux` occupancy evidence. Abort on any
unfamiliar workload. Record UUID, clocks, temperature, power, and process
memory before and after the row. Exactly one GPU executor may be active on the
identified GCP L4 node.

Correctness sampling includes measured steps 8, 57, and 107 at B1, B128, B512,
and B1024. The B512 and B1024 samples include cameras on both sides of each
128-camera physical chunk boundary, including 127/128. Other rows retain the
frozen final-state stratified camera sample. A final-state-only eight-camera
capture cannot validate temporal or chunk-boundary equivalence.

The headline full-sensor table is produced by
`benchmarks/summarize_flashgs_matched.py`. OVRTX remains a separate complete
system comparison because its perspective image-formation equation differs.
The final result root must be staged and sealed under
[`EVIDENCE_BUNDLE.md`](EVIDENCE_BUNDLE.md); the canonical package resolves
artifact records by hash rather than relying on Freiza-local paths, and keeps
the Home Scan PLY as a separately verified external dependency.

## Result records

The earlier Freiza RTX 3090 run is retained in
[FREIZA_RTX3090_RESULTS.md](FREIZA_RTX3090_RESULTS.md) as historical diagnostic
evidence only. Its candidates used 3.0-sigma support while the pinned gsplat
oracle used 3.33-sigma support, so it is equation-mismatched and invalid under
this corrected frozen contract. It must not be cited as the decisive matched
result.

The interrupted source-`3da2ffd` corrected-support run is also diagnostic only.
FlashGS produced 92 semantic mismatches across five tile-boundary regions at
B64 while Custom matched pinned gsplat at those pixels. Frozen, source-grounded
pre-repair traces established the cause: for one- and two-entry tile tails, the
optimized compositor enabled its final cooperative feature load without first
assigning the matching feature offset, so it loaded Gaussian 0's features for
the correct logical Gaussian. The matched-contract port repairs the two guard
offsets; the exact-upstream control remains unchanged.

No timing from either diagnostic run is publishable. A post-repair matrix may
start only after a separate fail-closed repair verifier proves all five frozen
cases and all 92 historical mismatch coordinates against the pinned oracle,
checks both output contracts, records zero overflow, and passes bounded
sanitizer coverage. Every performance row and profile must then be regenerated
from the attested repaired binary.
