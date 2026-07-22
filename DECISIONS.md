# Architectural and benchmark decisions

## D-001: Remote development target

- Date: 2026-07-16
- Decision: Use Vast instance `45069639`, label `gsplat-isaac-dev`, for all
  development and headline measurements until a documented host migration.
- Reason: RTX 4090, 24 GB VRAM, persistent 250 GB workspace, direct SSH.
- Consequence: Host changes invalidate direct performance comparisons.

## D-002: Isaac ML runtime precedence

- Date: 2026-07-16
- Decision: Put Isaac Sim 6.0.1's ML prebundle first on `PYTHONPATH`.
- Reason: A pre-existing conflicting Torch 2.7/NCCL 2.26 installation can
  shadow Isaac's Torch 2.11/NCCL 2.28 and cause `ncclCommShrink` load failures.
- Consequence: All builds must compile against Torch 2.11.0+cu128.

## D-003: Baseline before architecture

- Date: 2026-07-16
- Decision: Do not accept the final custom kernel architecture until fair RTX
  and current-`gsplat` baselines plus profiles are published.
- Reason: The objective requires evidence before optimization decisions.

## D-004: Home Scan validity

- Date: 2026-07-16
- Decision: Reject the invalid `.zip`, but use the valid extracted SOG LOD
  directory found at the same basename.
- Evidence: The extracted directory is 490 MB with 434 files, valid JSON,
  360 WebP payloads, no missing files referenced by `lod-meta.json`, and exact
  per-LOD counts matching metadata.
- Consequence: LOD0, containing 21,497,908 Gaussians, is the full Home Scan
  benchmark input. It was decoded into a 1,203,883,212-byte canonical PLY with
  SHA-256
  `29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267`.
  The source has no semantic IDs, so all renderers must use one documented,
  deterministic per-Gaussian semantic sidecar.

## D-005: Current `gsplat` control

- Date: 2026-07-16
- Decision: Pin the open-source control to commit
  `77ab983ffe43420b2131669cb35776b883ca4c3c`, compile for `sm_89`, and retain
  line information for profiling.
- Reason: Current source is required for a fair, inspectable control.

## D-006: Nsight Compute limitation

- Date: 2026-07-16
- Decision: Keep Nsight Compute installed and fail its preflight explicitly
  when counters are unavailable.
- Evidence: Host sets `RmProfilingAdminOnly: 1`; the container lacks
  `CAP_SYS_ADMIN`.
- Consequence: System tracing remains usable. Occupancy/bandwidth acceptance
  evidence requires a counter-enabled host or host configuration change.

## D-007: Direct Isaac Sim integration boundary

- Date: 2026-07-16
- Decision: Integrate the renderer as an Isaac Sim-native Kit extension and
  headless `SimulationApp` sensor service. Isaac Lab is not a core, build,
  launch, configuration, or runtime dependency.
- Reason: Isaac Sim already owns physics, USD, Fabric, extension lifecycle, and
  standalone application control. Adding a framework layer would obscure the
  integration cost and couple the renderer to camera abstractions it does not
  need.
- Consequence: The core service reads batched camera state directly from
  Fabric/USDRT and exposes GPU tensors. A later Isaac Lab adapter may only be a
  thin optional consumer. A Hydra RenderDelegate remains optional and
  secondary. The concrete process-local boundary is
  `isaacsim_gaussian_renderer.RendererService`, with a Kit extension under
  `exts/isaacsim.gaussian_renderer`; its backend protocol is reserved for the
  project-owned kernels.

## D-008: Custom kernels own the accepted renderer

- Date: 2026-07-16
- Decision: Project-owned GPU kernels must perform the dominant projection,
  visibility, queue construction/binning, required ordering, rasterization,
  and RGB/depth/alpha/semantic compositing work.
- Reason: OVRTX/RTX and stock `gsplat` are the required baselines and
  correctness references, not the deliverable.
- Consequence: A Kit wrapper, per-camera dispatch loop, or unmodified `gsplat`
  path can be retained only as an explicitly labeled control and cannot satisfy
  acceptance.

## D-009: OVRTX Gaussian baseline path

- Date: 2026-07-16
- Decision: Use OVRTX's concrete
  `ParticleField3DGaussianSplat` OpenUSD schema for the RTX correctness and
  performance baseline. Convert 3DGS parameters with NVIDIA's canonical rules:
  exponentiate log-scales, sigmoid raw opacity, normalize WXYZ quaternions, and
  preserve 3DGS SH coefficients in USD vec3-major order. Use the default
  particle-field color pipeline unless a measured fidelity result justifies an
  override, and record the setting in every result.
- Evidence: OVRTX 0.3.0 loaded a canonical four-Gaussian prim and returned
  valid colored LDR plus metric `DistanceToImagePlaneSD` with its default
  tonemapping setting. A four-camera single-RenderProduct tiled probe returned
  valid distance but black LDR, so that path is not yet an admissible color
  baseline. Four single-camera RenderProducts submitted in one
  `Renderer.step()` returned four valid 512x512 color and metric-distance
  outputs without a per-camera host render loop.
- Consequence: The earlier public-API inspection blocker is obsolete and must
  not be used as the RTX verdict. Use the one-step multi-RenderProduct path for
  the current RTX batch baseline and label it distinctly from native tiled
  rendering.

## D-010: Fair OVRTX output and semantic contract

- Date: 2026-07-16
- Decision: Configure the RenderProduct compositing API with output alpha
  enabled and background compositing disabled, then normalize `LdrColor` into
  preallocated sRGB float32 RGB/alpha tensors. Use
  `DistanceToImagePlaneSD` for metric float32 depth. Represent the deterministic
  1,024-class per-Gaussian sidecar as 1,024 `SemanticsAPI`-labeled
  `ParticleField3DGaussianSplat` prims, decode `SemanticIdMap` once, and remap
  `SemanticSegmentation` to int64 class IDs on CUDA with background `-1`.
- Evidence: A four-field proof returns all four semantic labels and visible
  per-pixel IDs. The 1,024-field synthetic-small control returns all expected
  label-map entries, nonzero RGB/alpha/depth/semantic outputs, and persistent
  CUDA contract tensors. The 50-frame B1 and B8 runs are saved under
  `outputs/baselines/ovrtx-fair-b1-128/` and
  `outputs/baselines/ovrtx-fair-b8-128/`.
- Consequence: OVRTX now has an admissible all-output synthetic control. Its
  one-step render submission is batched, but its per-RenderProduct output maps
  still require a host loop and the semantic field split is expensive; these
  limitations stay visible in configuration metadata and timing.

## D-011: Kernel architecture milestone status

- Date: 2026-07-16
- Decision: Publish `KERNEL_ARCHITECTURE.md` and `EXPERIMENT_LEDGER.md` as a
  source-grounded candidate architecture and measurement plan, not as an
  accepted final architecture.
- Reason: Source inspection establishes feasible extension boundaries and
  hypotheses, but D-003 still requires fair RTX/current-`gsplat` baselines and
  profiles before architecture acceptance.
- Consequence: Kernel implementation can be split into measurable subsystems,
  but optimization choices remain behind explicit gates.

## D-012: Fidelity contract and camera bundle

- Date: 2026-07-16
- Decision: Fidelity comparisons use `camera-bundle-v1` plus machine-readable
  render outputs containing RGB, alpha, depth, and semantic tensors. Aggregate
  fidelity PASS requires every view to satisfy the thresholds in `GOALS.md`.
- Reason: Headline speedups are valid only when cameras, scene IDs, background,
  color space, output resolution, and measured tensors are identical.
- Consequence: Missing LPIPS for non-identical images is a fidelity failure,
  not a substitute metric. RTX-only secondary lighting and post-processing are
  triaged separately from Gaussian reconstruction/compositing parity.

## D-013: Independent verification isolation

- Date: 2026-07-16
- Decision: Run independent verification reruns only from
  `/workspace/agent-worktrees/verification` on SSH host `vast-gsplat-isaac`.
- Reason: Verification must not overwrite producer worktrees or producer
  outputs while reproducing setup, benchmark, fidelity, and stability evidence.
- Consequence: Verification scripts fail when pointed at another remote root,
  and source synchronization excludes `outputs/`.

## D-014: First custom renderer architecture

- Date: 2026-07-16
- Decision: Use a project-owned C++/CUDA pipeline with fused anisotropic
  projection/culling, fixed-capacity visible and tile-intersection queues,
  global CUB radix ordering, and fused tile compositing. Deterministic mode adds
  a segmented CUB pass keyed by full depth bits and global Gaussian ID.
- Evidence: The complete custom synthetic matrix reports one native batched
  submission per render, no runtime camera loop, zero measured-loop allocation,
  zero queue overflow, GPU-resident RGB/depth/alpha/semantic outputs, and
  bitwise deterministic replay.
- Consequence: Macro-tile rejection, CUDA graphs, persistent scheduling, and
  mixed-precision storage remain optional measured optimizations rather than
  prerequisites for the current backend.

## D-015: Authored-display RGB contract

- Date: 2026-07-16
- Decision: OVRTX comparison scenes treat the supplied RGB values as authored
  display-sRGB values. The custom comparison path therefore applies no second
  transfer function and declares
  `color_transform = identity_authored_display_rgb`.
- Evidence: On the 16,384-frame OVRTX temporal average, unpremultiplied custom
  identity RGB differs by approximately 0.0101 MAE, while applying a second
  linear-to-sRGB transfer differs by approximately 0.2246 MAE.
- Consequence: The earlier custom matrix that included a linear-to-sRGB
  transfer remains conservative performance evidence but is not the exact
  visual comparison contract. Final comparisons use
  `--authored-display-output`.

## D-016: Large OVRTX batches require a raised descriptor limit

- Date: 2026-07-16
- Decision: Source `scripts/remote_env.sh` before every OVRTX capacity or
  benchmark run so the soft file-descriptor limit is 65,536.
- Evidence: B256 initially emitted descriptor-pool, file-descriptor, and
  misleading cleanup OOM errors under the container's low soft limit. The same
  B256 cases measure successfully with the raised limit; 512x512 uses about
  21.53 GB of driver-reported VRAM and needs a watchdog longer than 300 seconds.
- Consequence: Old B256 failures are retained as invalid-environment evidence,
  not treated as renderer capacity limits. Merged manifests replace them with
  isolated high-limit reruns and preserve provenance.

## D-017: Home Scan dense pixel path

- Date: 2026-07-16
- Decision: For dense large-scene rendering, use precomputed object-space
  covariance, fused projection/depth-bucket counting, warp-aggregated bucket
  scatter, 1x1 pixel tiles, and conservative grouped optical-thickness cutoff.
  The measured Home Scan configuration uses 1,024 buckets in groups of 32.
- Evidence: The full 21,497,908-Gaussian Home Scan dynamic path measures
  7.274 ms with 946,174 retained intersections, zero overflow, zero
  measured-loop allocation, and 3.473 GB driver memory.
- Consequence: The 16x16 path remains the shared/multi-scene matrix default;
  the dense pixel path is selected for scenes where exact early cutoff
  materially reduces intersection work.

## D-018: Safe tight depth range

- Date: 2026-07-16
- Decision: A benchmark may tighten custom depth-bucket bounds to the complete
  camera-space scene bounding sphere plus a numerical margin, provided no
  Gaussian center is culled and the bound is recorded.
- Evidence: Home Scan near/far values 17.968897/51.562119 enclose all scene
  centers and reduce empty depth groups without changing the rendered contract.
- Consequence: This is a scheduling quantization optimization, not scene
  cropping. Broad and tight paths remain separately labeled.

## D-019: Projection-coherence reuse

- Date: 2026-07-16
- Decision: Permit explicit opt-in reuse of geometry/scheduling state only when
  scene revision, projection parameters, tensor identity/version, transforms,
  intrinsics, scene IDs, and active IDs are coherent. Cache contents are
  mode-specific: the direct path retains per-pixel depth cutoffs, sorted
  Gaussian-ID intersections, and dense pixel ranges.
- Evidence: The direct cache smoke produces bitwise-identical coherent
  outputs, changes RGB after a feature-only update while depth/alpha/semantic
  remain unchanged, invalidates on a camera edit, and reports no overflow.
  Nsight records direct projection and sorting only on the miss while the
  compositor executes and rewrites all outputs on every hit.
- Consequence: This is not frame or image caching. Fabric/Warp external writes
  and inference-mode tensor writes must explicitly call
  `invalidate_projection_cache()`. Dynamic-camera performance remains a
  separately reported mode.

## D-020: Direct compact screen-space architecture

- Date: 2026-07-16
- Decision: Use the two-pass direct compact screen-space pipeline for the
  headline dense Home Scan workload. Pass one accumulates per-pixel
  depth-bucket optical thickness; pass two emits qualifying pixel/depth keys
  and Gaussian IDs directly, without materializing projected visible records.
  CUB sorts the global intersection queue, and the fused compositor
  cooperatively reprojects Gaussian IDs on every render.
- Evidence: The full 21,497,908-Gaussian Home Scan at B1024 produces
  146,618,236 visible-Gaussian observations and 167,147,351 qualifying pixel
  intersections with zero overflow. The legacy projected arrays have
  one-element placeholders in this mode. Strict conventional `gsplat`
  compositing parity passes at B1 and B8. A tuned 256-frame Home Scan OVRTX
  control reaches PSNR 47.2704 dB and semantic agreement 0.999084, but that
  finite temporal control is not equation-level OVRTX perspective evidence.
- Consequence: `max_visible_records` is diagnostic/capacity-planning metadata
  in direct mode; `max_intersections` remains the materialized queue capacity.
  Direct mode requires 1x1 pixels, screen-space evaluation, and
  non-deterministic scheduling. The general deterministic projected-record
  path remains available for tests.

## D-021: Headline batch and conservative memory accounting

- Date: 2026-07-16
- Decision: Use B1024 at 128x128 as the largest common successful Home Scan
  batch, and compare the maximum driver-reported memory observed across three
  independent runs.
- Evidence: Custom and OVRTX both pass B1024. OVRTX B2048 fails with Vulkan
  `ERROR_OUT_OF_DEVICE_MEMORY`, renderer allocation failure, CUDA interop
  `cudaErrorMemoryAllocation`, and process termination. At B1024, three custom
  runs average 22,794.59 images/s and three OVRTX runs average 220.456
  images/s. Conservative peak driver memory is 11,966,349,312 bytes custom
  versus 18,896,388,096 bytes OVRTX.
- Consequence: The measured headline is 103.397x throughput and a 0.633261
  memory ratio, or 36.674 percent lower memory. B2048 is retained as boundary
  evidence rather than omitted.

## D-022: Separate EWA and OVRTX perspective fidelity contracts

- Date: 2026-07-16
- Decision: Treat deterministic screen-space 2D EWA/`gsplat` parity and
  stochastic OVRTX perspective 3D-ray parity as separate rendering contracts.
  A passing finite-frame or scene-specific comparison cannot substitute for
  the converged OVRTX perspective acceptance gate.
- Evidence: The production EWA renderer matches pinned `gsplat` at B8 with
  mean PSNR 156.611 dB and 100 percent semantic agreement. Against the
  16,384-frame OVRTX perspective reference it measures PSNR 35.094 dB,
  alpha MAE 0.023973, depth relative error 0.057893, and semantic agreement
  0.932251. The experimental exact 3D-ray diagnostic against 65,536 frames
  reaches PSNR 55.449 dB, alpha MAE 0.001317, depth relative error 0.005638,
  and semantic agreement 0.995544, passing every threshold except semantics.
- Consequence: The production fast path remains the conventional
  EWA/`gsplat` renderer and overall OVRTX perspective acceptance remains
  **FAIL**. The exact-ray path stays diagnostic until it receives equivalent
  performance, Home Scan, matrix, stability, and independent verification.
  The OVRTX `projectionModeHint` is advisory. Bitwise-identical isotropic
  synthetic artifacts were initially an insensitive-fixture result. The
  subsequent repeat-controlled rotated anisotropic off-axis experiment has
  been superseded by a fail-closed seven-process L4 gate with valid runtime
  readback, matched-lane checks, zero repeat noise, and a stable focal control.
  Its two perspective/tangential comparisons are bitwise identical across all
  required outputs. The earlier uncontrolled anisotropic Home delta is not
  proof of mode consumption. OVRTX 0.3.0 RTPT therefore has no demonstrated
  tangential behavior on the tested path, and the tangential lane remains
  blocked.

## D-023: Exact-ray hierarchical 32x32 binning

- Date: 2026-07-16
- Decision: For exact-ray evaluation with 16x16 raster blocks, sort 32x32
  conservative parent bins and let each of four child raster blocks read the
  same ordered range. Keep the per-pixel 3D-ray support test and full float32
  center-Z plus Gaussian-ID ordering unchanged.
- Evidence: Fine-bin and macro-bin artifacts are byte-identical for RGB,
  alpha, depth, semantics, and validity at B1/128x128 and B2/130x129. Across
  eight medium-scene memory-risk rows, the measured maximum memory ratio is
  0.661999, minimum speedup is 18.2079x, and all overflow counters are zero.
- Consequence: The exact-ray workspace/OOM blocker is resolved for those
  targeted rows. The capacity planner scales by the actual ceiling-rounded
  macro-bin grid so odd resolutions cannot silently under-allocate. Full
  matrix, multi-scene, soak, profiler, and independent reruns remain required
  before final architecture acceptance.

## Rejected shortcuts

- Treating the preliminary `gsplat` throughput number as the custom result.
- Comparing only RGB while omitting depth, alpha, or semantic work.
- Calling the OVRTX Vulkan interop sample a portable renderer backend.
- Using the missing/expired Home Scan response as benchmark content.
- Treating the sum of all six redundant LOD levels as one 42.3M-Gaussian scene.
- Selecting an architecture solely from source inspection without profiles.
- Making Isaac Lab the extension owner or a transitive runtime dependency.
- Calling a stock renderer through a Kit extension and labeling it custom.
- Treating composited LDR alpha as Gaussian coverage.
- Treating missing LPIPS as passing or replacing it with an acceptance proxy.

## D-024: Trajectory evidence is a first-class acceptance contract

- Date: 2026-07-16
- Decision: Version and hash camera trajectories independently of render
  outputs; evaluate static cache hits, moving-camera misses, conventional EWA
  fidelity, OVRTX perspective convergence, and Isaac/Fabric latency as
  separate verdicts.
- Reason: A repeated unchanged camera measures projection-cache coherence, not
  moving-camera Isaac performance. OVRTX perspective also uses a different
  stochastic rendering equation from the production EWA path.
- Consequence: `headline-acceptance/v3` fails closed when dynamic trajectory,
  Isaac/Fabric, cache, artifact, stability, dynamic reproducibility, or source
  provenance evidence is missing. Static coherent speed can never satisfy the
  dynamic gate.

## D-025: Projection-mode activation requires behavioral evidence

- Date: 2026-07-20
- Decision: Accept an OVRTX tangential benchmark lane only when composed-token
  readback and a controlled cross-mode effect above same-mode repeat noise both
  pass. A renderer-internal selected-mode state may replace token readback only
  if it directly proves kernel consumption. A complete negative gate blocks
  the lane without being labeled an OVRTX bug.
- Evidence: On L4 with Isaac Sim 6.0.1 and OVRTX 0.3.0, the seven-process gate
  passed stage, readback, output, matched-lane, repeat, lifecycle, and repeated
  focal-control checks. Both perspective/tangential comparisons had zero
  differing RGB, alpha, depth, semantic, or valid-depth elements. The focal
  control changed 10.6567 percent of RGB/alpha elements reproducibly.
- Consequence: The result is `NO_OBSERVABLE_MODE_EFFECT`, not an
  infrastructure failure and not activation proof. The tangential matrix
  remains inadmissible for this configuration. The conclusion does not claim
  global non-support, schema violation, proprietary-renderer defect,
  performance, or custom-renderer fidelity.

## D-026: Active-subset renders define inactive output rows

- Date: 2026-07-20
- Decision: Every active-camera-subset render resets inactive RGB, alpha, and
  semantic rows to zero, zero, and -1 respectively, and inactive depth rows to
  positive infinity. The CUDA and deterministic fake backends share this
  contract for both service-owned and caller-owned output buffers.
- Reason: Leaving inactive rows untouched can expose stale observations when a
  caller reuses or alternates output buffers. A fake backend that preserves
  those rows also makes CPU tests validate behavior different from production.
- Consequence: Consumers can treat every output row as defined after each
  logical render. Preserving a previous inactive observation requires an
  explicit consumer-side cache rather than relying on renderer buffer state.
