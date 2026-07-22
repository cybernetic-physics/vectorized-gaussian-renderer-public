# Task ledger

This file is the living orchestration ledger. Benchmark conditions may change
only through an entry in `DECISIONS.md` and a new configuration ID.

## Team ownership

| Role | Branch/worktree | Measurable ownership | State |
|---|---|---|---|
| Orchestrator | `codex/vectorized-gsplat` | Integrate all work, preserve benchmark contract, maintain ledger, review evidence | active |
| Baseline and benchmarking | `codex/baselines` | RTX/OVRTX and current-gsplat runners, datasets, full matrix, JSON/CSV | active |
| GPU profiling | `codex/profiling` | Nsight/PyTorch traces, memory accounting, bottleneck report | active |
| Kernel architecture | `codex/kernel-architecture` | Measured pipeline design, data layout, experiment plan, rejected alternatives | integrated; first architecture measured |
| Kernel implementation | `codex/kernels` | Custom CUDA/C++ pipeline and bindings | integrated and profiled |
| Isaac integration | `codex/isaac-integration` | Isaac Sim-native Kit extension, `SimulationApp`, Fabric cameras, scene IDs/transforms/subsets/cadence, GPU tensor integration | active |
| Fidelity and correctness | `codex/fidelity` | Deterministic metrics, images, thresholds, discrepancy triage | active |
| Independent verification | `codex/verification` | Clean build/reproduction; no main-renderer implementation | active |

## Phase ledger

### Phase 1: repository and environment

- [x] Persistent SSH alias `vast-gsplat-isaac`.
- [x] Dedicated Vast RTX 4090 instance running.
- [x] Isaac Sim 6.0.1 finite headless smoke passes.
- [x] OVRTX, Kit app template, and current `gsplat` cloned.
- [x] CUDA 12.8 and NVIDIA profiling tools installed.
- [x] Exact environment identifiers recorded in `RESULTS.md`.
- [x] Required project documentation created.
- [x] Isaac Sim-native package and Kit extension scaffolded with no Isaac Lab
  dependency.
- [ ] `scripts/setup.sh` reproduced from a fresh Isaac Sim 6.0.1 container.
- [x] Clean-root setup verification by independent verifier on the retained
  Isaac Sim 6.0.1 instance.

### Benchmark assets

- [x] Specified Home Scan path checked.
- [x] Invalid `.zip` state recorded without treating it as a dataset.
- [x] Valid extracted SOG LOD Home Scan found and structurally validated.
- [x] Home Scan counts and relative manifest checksum recorded.
- [x] Home Scan LOD0 decoded into a canonical 21,497,908-record PLY with a
  locally saved manifest and reproducible conversion script.
- [x] Home Scan copied to the remote persistent dataset workspace.
- [x] Public real-world Gaussian dataset selected, downloaded, and
  checksummed: Hugging Face `Voxel51/gaussian_splatting`, train iteration
  30000, 1,074,761 records, SHA-256
  `a4c1906ce0256f5cb2255aa078d18b468dc01d8604f6eda1aaf22b235147a7a1`.
- [x] Deterministic synthetic scene available.

### Baselines

- [x] Fair RTX/OVRTX Gaussian synthetic-small baseline for B1 and B8 at
  128x128, including GPU-resident RGB/depth/alpha/semantic contract output.
- [ ] Confirm and record Fabric Scene Delegate state for the headline OVRTX
  matrix.
- [ ] Fastest visually equivalent RTX configuration.
- [x] Current pinned-`gsplat` full-output baseline. The canonical logical-batch
  and trajectory paths render RGB, expected depth, alpha, and top-contributor
  semantic IDs, including the shared alpha/background policy and separately
  measured semantic cost. The legacy `run_baselines.py` allocation-only rows
  remain non-citable for semantic parity.
- [x] Shared-scene matrix.
- [x] Multi-scene-ID matrix for the custom backend.
- [x] Baseline numbers published before final architecture selection.

### Profiling

- [x] Preliminary Nsight Systems trace for synthetic `gsplat`.
- [x] Full-output custom trace with immutable Home Scan benchmark config.
- [x] PyTorch profiler and allocation trace.
- [ ] Nsight Compute metrics on a counter-enabled host.
- [x] Bottleneck report with launch, sorting, synchronization, allocator, and
  stage timing evidence.
- [ ] Occupancy, bandwidth, and divergence evidence on a counter-enabled host.

### Custom renderer

- [x] Candidate kernel architecture and experiment ledger documented.
- [x] First architecture accepted from measured throughput, memory, and
  project-owned-kernel evidence; overall fidelity remains FAIL.
- [x] Custom anisotropic projection/covariance and visibility culling.
- [x] Compact fixed-capacity visible queues.
- [x] 16x16 tile binning/scheduling.
- [x] Global CUB radix ordering plus deterministic segmented ordering.
- [x] Fused RGB/depth/alpha/semantic compositing.
- [x] Reusable allocation-free workspace; CUDA graph remains an optional
  measured experiment.
- [x] Multi-scene scene-offset/scene-ID support.
- [x] Deterministic mode with full-depth/global-ID tie-breaking.
- [x] Precomputed object-space covariance.
- [x] Exact conservative grouped depth cutoff for dense Home Scan projection.
- [x] Opt-in projection-coherence reuse with explicit invalidation and
  per-frame raster/output recomputation.
- [x] Direct compact two-pass screen-space path without per-camera projected
  records, using per-pixel cutoffs, Gaussian-ID intersections, global CUB
  ordering, and cooperative reprojection in the compositor.
- [x] Full Home Scan B1024 headline at the largest common successful OVRTX
  batch with three-run throughput and conservative driver-memory accounting.

### Isaac integration

- [x] Loadable Isaac Sim Kit extension and explicit lifecycle service.
- [x] Finite `SimulationApp` smoke test using the extension on Isaac Sim 6.0.1.
- [x] Direct Fabric/USDRT batched camera transform ingestion.
- [x] Cached USD camera intrinsics boundary.
- [x] Shared static scene and batched camera service independent of Isaac Lab.
- [x] Per-environment transform API.
- [x] Scene IDs and cached packed-scene offsets.
- [x] GPU RGB/depth/alpha/semantic output contract.
- [x] Active environment subset.
- [x] Render cadence.
- [x] No hidden simulation stepping or CUDA scene-ID copies in the service
  boundary; production CUDA backend validated in Isaac Sim 6.0.1.
- [x] Integration overhead measured separately and end to end.
- [x] Vectorized headless smoke script with shape/device/value/no-hidden-step checks.

### Fidelity and acceptance

- [x] RTX/OVRTX Gaussian reconstruction/compositing reference outputs and
  temporal convergence harness.
- [x] Deterministic camera-bundle schema shared with benchmarks.
- [x] RGB/depth/alpha/semantic metrics, diff images, and threshold PASS/FAIL
  infrastructure.
- [x] Fidelity CLI consuming machine-readable render outputs.
- [x] 5x throughput criterion for the static-coherent Home Scan mode and all
  matched shared-scene matrix cases. Dynamic Home Scan remains below 5x and is
  reported separately.
- [x] Peak-memory criterion: custom <=80 percent of OVRTX.
- [x] Three-run reproducibility within 5 percent.
- [x] Ten-minute synchronized stability soak with no CUDA errors, output
  failures, measured-loop allocation growth, or queue overflow.
- [x] Single-camera regression report.
- [x] Home Scan final rerun.
- [x] Conventional EWA/`gsplat` parity retained as a separate rendering
  contract.
- [x] Experimental exact 3D-ray OVRTX perspective diagnostic; five of six
  thresholds pass, with semantics at 99.554 percent.
- [x] Historical OVRTX perspective/tangential hint audit: the uncontrolled
  anisotropic Home Scan delta is not behavioral evidence; the later
  repeat-controlled 256-frame activation experiment has valid token readback
  and bitwise-identical outputs across modes and repeats.
- [x] GPU-validate the implemented fail-closed seven-lane activation gate:
  discarded prime plus bracketed `C-P-T-T-P-C`, 256-frame temporal outputs,
  matched stage/camera/scene/provenance checks, composed projection/sorting
  readback, nonblank output contracts, and repeated focal positive control.
  The L4 run is a valid negative (`NO_OBSERVABLE_MODE_EFFECT`): every control
  passed, but both cross-mode pairs were bitwise identical, so tangential
  activation and the downstream tangential matrix remain blocked.
- [ ] Repair temporal resume so the renderer sample sequence advances past
  restored frames; do not use resumed references for acceptance until fixed.
- [ ] Full matched OVRTX tangential rerun after mode activation is proven:
  temporal fidelity, throughput, peak memory, every required batch and
  resolution, shared/multi-scene coverage, public data, and Home Scan. Keep
  the perspective matrix as a separate lane rather than replacing it.
- [ ] Strict OVRTX fidelity thresholds; current result is explicit FAIL.
- [ ] Refreshed direct-compact matrix across all required batches,
  resolutions, shared/multi-scene modes, public real-world data, and Home
  Scan.
- [ ] Overall independent acceptance; clean-root reproduction passes, but the
  fail-closed audit correctly retains the OVRTX fidelity FAIL.

### Independent verification

- [x] Fail-closed evidence verifier for acceptance JSON and CSV artifacts.
- [x] Synthetic fixture tests for pass, missing evidence, invalid statistics,
  and failing CSV rows.
- [x] Independent remote-root guard for verification reruns.
- [x] Benchmark loopholes identified in `VERIFICATION_PLAN.md`.
- [x] Clean-root setup rerun from
  `/workspace/agent-worktrees/verification`.
- [x] Independent 45-test run, production Kit-extension compile/load,
  physics/Fabric vectorized example, deterministic and multi-scene smokes, and
  Compute Sanitizer memcheck.
- [x] Independent raw-evidence audit retained with explicit fidelity failures.
- [ ] Independent rerun of full benchmark and stability evidence after
  fidelity is resolved and an all-pass acceptance bundle can be issued.

## Immediate next actions

### Trajectory evaluation framework

- [x] Add portable `camera-trajectory-v1` JSON/NPZ contracts and time-major
  fidelity flattening.
- [x] Add representative and stress Home Scan route inputs.
- [x] Add explicit spatial, sequential-temporal, and vectorized-temporal
  scenario schedules with expected cache events.
- [x] Add CUDA-event/synchronized-wall trajectory runner with custom and
  pinned-gsplat adapters and separated artifact costs.
- [x] Add expanded spatial and temporal fidelity metrics.
- [x] Add mutation, poisoned-buffer, alternating-buffer, and external-write
  anti-stale cache audit.
- [x] Add reopen/hash/frame/video artifact auditor and black/duplicate
  regression fixture.
- [x] Extend the finite Isaac/Fabric example with static/full/mixed motion,
  deadline accounting, and explicit external-write invalidation.
- [x] Split acceptance into independent fail-closed verdicts.
- [ ] Complete and audit the fresh-host measured suite recorded below; do not
  mark this complete from implementation-only evidence.


1. Complete the refreshed direct-compact workload matrix without conflating
   its EWA equation with OVRTX perspective.
2. Decide whether to productionize and optimize exact-ray rendering or retain
   OVRTX perspective as an explicit unsupported-equation FAIL.
3. Keep the OVRTX tangential lane blocked unless a future version or
   configuration passes composed-token readback and produces a cross-mode
   effect above same-mode repeat noise; do not promote token readback alone.
4. Rerun independent clean-root evidence after the matrix and selected
   fidelity path are final.
5. Add a separately measured robot/mesh compositor for hybrid observations.
6. Obtain Nsight Compute hardware counters on a counter-enabled host.
