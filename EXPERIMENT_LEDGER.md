# Kernel Architecture Experiment Ledger

Status: first architecture implemented and measured. Overall acceptance remains
FAIL because strict OVRTX fidelity is below threshold.

## Integrated outcomes

| Experiment | Outcome |
|---|---|
| E-P1 covariance | Precompute `[G, 6]`; Home Scan projection improved while memory stayed at 25.66% of OVRTX |
| E-O1 ordering | Keep global CUB radix ordering; deterministic segmented tie-break remains selectable |
| E-W1 workspace | PASS: zero measured-loop allocation and zero overflow across current matrices |
| E-D1 determinism | PASS: bitwise repeat and 0.0198% three-run timing spread |
| E-B1 exact macro binning | 32x32 parent bins feeding unchanged 16x16 exact-ray raster blocks are byte-identical to fine bins on square and odd-size fixtures; eight risk rows measure 0.3867-0.6620 OVRTX memory and 18.21-56.45x throughput |
| E-R1 exact semantics | Native strongest-individual semantics is not the OVRTX probability rule; aggregate first-hit label probability improves the 65,536-frame stress result from 0.984741 to 0.995667 but remains below 0.999 |
| Dense Home Scan depth pruning | Accept conservative grouped optical-thickness cutoff, D1024/G32 |
| Projection coherence | Accept as explicit opt-in for unchanged inputs; projected records only, never output images |
| Ten-minute stability | PASS: 600.486 seconds, 7.60M images, zero CUDA errors/allocation growth/overflow |
| CUDA memcheck | PASS: projection-cache mutation/invalidation smoke, zero errors |
| Independent clean-root audit | Build/runtime/integration checks pass; overall audit remains fail-closed on OVRTX fidelity |
| OVRTX fidelity | FAIL: immutable authored-display thresholds remain unmet |
| RTX-compositing 4090 validation (2026-07-20) | OptiX RT-core tracer claims reproduce within ~3% (13.4 ms / 156.7 Mrays/s at 32×256², k-buffer+clamp) with PSNR 36.3 dB raster parity; Warp hybrid faster than claimed (432 img/s at 32×256²); OVRTX unified baseline 1.45× slower than its #51 claim (148.7 vs 221 img/s, repeatable) — same-node hybrid-vs-OVRTX margin is 2.9×. Ray-matched RT-vs-raster (identical cameras, `benchmarks/run_rt_vs_raster.py`): tracer 4.21× on the exterior bundle, 1.54× on interior ego views — the unmatched-camera "10×" is retired. Evidence: `benchmarks/rtx_compositing_validation_4090_results_2026-07-20.json` |

## Required Evidence Before Architecture Acceptance

| ID | Question | Configurations | Metrics | Decision |
|---|---|---|---|---|
| E-A0 | Is there a fair target to beat? | RTX/OVRTX and current `gsplat`, same scene/cameras/outputs | frame time, memory, fidelity, work counts | Blocks final architecture until complete |
| E-P1 | Should covariance be computed or stored? | fused quat-scale vs precomputed `[G, 6]` | projection ms, total ms, persistent MiB | Store covariance only if >=8% total speedup and memory target holds |
| E-C1 | Is two-pass compaction acceptable? | B=1,64,256; 128/256/512; shared and scene IDs | compaction ms, scans, syncs, allocation delta | Keep if <12% total and sync-free |
| E-B1 | Does macro binning help? | fine-only vs 64x64 macro+fine | active tiles, intersections, binning ms, total ms | Use macro if >=15% CTA/intersection reduction and <5% overhead |
| E-O1 | Which ordering strategy wins? | global radix, segmented radix, approximate buckets | sort ms, temp bytes, fidelity | Approximate only if all fidelity thresholds pass |
| E-R1 | Which depth and semantic rule matches RTX? | expected/first-depth x first/max-visibility semantic | depth relative error, semantic agreement, RGB/alpha metrics | Select by fidelity first, performance second |
| E-W1 | Does workspace eliminate allocator churn? | warmup then 50+ measured iterations | peak allocated/reserved, allocation delta, overflow count | Accept only with zero measured-loop allocation delta |
| E-G1 | Does CUDA graph replay help? | graph off/on after fixed workspace | p50, p95, CPU launch overhead, graph recapture count | Enable if >=5% p50 or >=20% CPU-launch improvement |
| E-D1 | What does deterministic mode cost? | deterministic off/on, three runs | output repeatability, frame-time reproducibility, speed loss | Ship selectable if correct; flag if >15% slower |
| E-Z1 | Is binding zero-copy? | Isaac/Warp proxy outputs and direct PyTorch outputs | profiler copies, syncs, binding ms | Must show no CPU output copy |

## Baseline And Profile Dependencies

Architecture acceptance waits for these artifacts from the baseline/profiling
tracks:

- RTX/OVRTX result rows for RGB/depth/alpha/semantic and equivalent Gaussian
  reconstruction/compositing settings.
- Current `gsplat` result rows with full output parity, packed/unpacked flags,
  tile size, render mode, SH degree, and workspace behavior.
- Nsight Systems traces with projection, binning/sort, rasterization, launch
  count, and synchronization visible.
- PyTorch profiler traces for allocation and binding behavior.
- Memory rows for persistent scene, reusable workspace, temporary deltas, and
  driver process memory.
- Fidelity images and metrics for representative and worst-case views.

## Subsystem Experiments

### E-PROJ-001: Projection Parity And Counts

- Inputs: synthetic-small, synthetic-medium, public-real-world once available.
- Modes: shared scene, B=1/8/64; resolution 128 and 256.
- Compare custom `project_compact` intermediates to `gsplat` projection
  references where possible: visible count, depth range, radii, means2d, conics.
- Record: projection ms, visible `V`, rejected by near/far, rejected by radius,
  rejected outside image, workspace bytes.
- Gate: projection output explains downstream fidelity differences before
  raster changes are blamed.

### E-PROJ-002: Multi-Scene Offsets

- Inputs: two synthetic scenes concatenated into one canonical SoA allocation.
- Modes: alternating `scene_ids`, selected active subsets, per-environment
  transforms.
- Record: visible IDs must remain inside `scene_offsets[scene_id]` range.
- Gate: no duplicated static scene tensors and no cross-scene Gaussian leakage.

### E-BIN-001: Fine Tile Baseline

- Inputs: same as E-PROJ-001.
- Modes: tile size 16, global keys.
- Record: `tiles_per_gauss`, `T`, active tile count, empty tile count, binning
  ms, offset ms, temporary bytes.
- Gate: establishes source-comparable baseline before macro alternatives.

### E-BIN-002: Macro Tile Rejection

- Inputs: scenes with sparse and dense screen occupancy.
- Modes: no macro, 64x64 macro, optional 32x32 macro if 64x64 is too coarse.
- Record: macro active count, fine active count, `T`, raster CTA count, total ms.
- Gate: accept only by E-B1.

### E-ORDER-001: Global Radix Sort

- Inputs: all scenes once available.
- Modes: deterministic tie-break off/on.
- Record: sort ms, sort temp bytes, key format, p50/p95 total impact.
- Gate: source-grounded default until an alternative beats it.

### E-ORDER-002: Segmented Sort

- Inputs: high-B configurations where global sort dominates.
- Modes: per-image and per-active-tile segment plans.
- Record: segment metadata bytes, launch count, sort ms, total ms.
- Gate: accept only by E-O1 and deterministic constraints.

### E-ORDER-003: Approximate Buckets

- Inputs: worst-case overlapping depth views.
- Modes: depth buckets with fixed tie-breaks.
- Record: all fidelity metrics, especially alpha/depth/semantic mismatches.
- Gate: reject unless all thresholds pass. Performance alone cannot accept it.

### E-COMP-001: Fused Output Rules

- Inputs: RTX reference views and `gsplat` control views.
- Modes: expected depth vs first depth; first semantic vs max-visibility
  semantic.
- Record: RGB PSNR/SSIM/LPIPS, alpha MAE, valid-pixel depth relative error,
  semantic agreement.
- Gate: select rules by RTX parity.

### E-COMP-002: Fused Versus Split Outputs

- Inputs: same cameras and visibility queues.
- Modes: fused RGB/depth/alpha/semantic; split semantic; split depth.
- Record: kernel ms, memory traffic estimate, output bytes, fidelity.
- Gate: fused path must be >=10% faster for acceptance as default.

### E-WORK-001: Workspace Capacity Sweep

- Inputs: matrix batch sizes and resolutions.
- Modes: capacity multiplier 1.0x, 1.25x, 1.5x measured high-water marks.
- Record: peak memory, overflow, allocation delta, warmup growth iterations.
- Gate: choose smallest capacity plan that avoids overflow in measured runs.

### E-GRAPH-001: Graph Capture

- Inputs: stable workspace configurations from E-WORK-001.
- Modes: graph off/on; active subset fixed and changing.
- Record: CPU launch overhead, frame p50/p95, recapture count, failure modes.
- Gate: enable only by E-G1.

### E-DET-001: Deterministic Replay

- Inputs: synthetic and public scenes.
- Modes: deterministic off/on.
- Record: tensor equality or tolerance, key tie-break collisions, frame-time
  variance across three independent runs.
- Gate: deterministic output must be reproducible and labeled in results.

### E-ZERO-001: PyTorch/Warp Output Alias

- Inputs: local Isaac renderer path and standalone PyTorch harness.
- Modes: direct allocated output tensors; Warp proxy `.torch` views.
- Record: tensor data pointers, profiler copy events, synchronization, binding
  time.
- Gate: no CPU output copy in steady state.

## Result Row Additions

Every custom result row must add:

```text
custom_commit
cuda_extension_loaded
layout_contract_version
scene_layout_id
workspace_capacity_visible
workspace_capacity_intersections
workspace_overflow_count
visible_pairs
tile_intersections
active_tiles
sort_strategy
depth_rule
semantic_rule
deterministic
cuda_graph
projection_ms
compaction_ms
binning_ms
ordering_ms
compositing_ms
binding_ms
allocation_delta_bytes
```

## Open Risks

- Home Scan LOD0 may make SH feature storage or large sort workspaces dominate
  memory; the architecture must be ready to choose RGB-only, SH-on-demand, or
  compressed feature paths only after the fidelity contract is known.
- RTX semantic IDs may not match per-Gaussian semantic labels without an
  explicit scene conversion rule. This is a fidelity issue, not a kernel issue.
- Nsight Compute counters are unavailable on the current host because
  `RmProfilingAdminOnly=1`; occupancy and bandwidth gates may need a host
  configuration change.
- CUDA graph capture may be neutral or harmful if selected-environment subsets
  change active topology frequently.
- Approximate ordering has high fidelity risk because alpha compositing is
  order-dependent.

## Current Milestone Output

- `KERNEL_ARCHITECTURE.md`: implemented first architecture plus retained
  optimization gates.
- `src/isaacsim_gaussian_renderer/native/renderer_cuda.cu`: project-owned
  projection, culling, tile queue, CUB ordering, and fused compositor.
- `src/isaacsim_gaussian_renderer/cuda_backend.py`: packed-scene ownership,
  reusable workspace, counters, scene IDs/transforms, and one-submission
  binding.
- `benchmarks/run_matrix.py`: shared and packed multi-scene custom matrices.
- `benchmarks/compare_custom_gsplat.py`: strict all-output reconstruction and
  compositing parity control.
- `experiments/ovrtx_temporal_fidelity.py`: stochastic OVRTX temporal
  convergence reference.
- `experiments/ovrtx_parameter_sweep.py`: diagnostic-only convention sweep that
  prevents modified scene parameters from being mistaken for acceptance
  evidence.

Throughput and memory are measured separately from fidelity. No optimization
is allowed to convert the current OVRTX fidelity verdict into PASS by changing
scene attributes, cameras, outputs, or color transfer.
