# Results

> **Publication status (2026-07-21):** This file is a cumulative engineering
> log, not a table of currently accepted headline claims. Bold `PASS` labels
> below describe the gate used by the dated experiment in that section. The
> older Custom-versus-OVRTX results compare different image-formation
> equations, and several use projection-cache hits; they are retained as
> system-history evidence, not as a rasterizer speedup claim. A publishable
> performance claim remains pending the complete same-source, same-GPU,
> independently changing-camera Custom-versus-FlashGS matrix defined in
> [`PUBLICATION_READINESS.md`](PUBLICATION_READINESS.md).

## Independent PR #68 audit on an NVIDIA L4

Recorded 2026-07-21 on one Google Cloud NVIDIA L4 after testing the exact PR
head `06bdac2` and then the hardened implementation. The complete compact
record is
`benchmarks/pr68_independent_l4_validation_2026-07-21.json`.

The product contract matters: the tracer reuses one static Gaussian map for
many independently changing cameras and accepts device-resident sparse
secondary rays. It does **not** replace Isaac RTX mesh rendering, and it is not
a drop-in Isaac renderer backend.

| workload | OptiX tracer | custom raster | result |
|---|---:|---:|---:|
| Home, B1 × 128² changing camera | 6.85 ms / 146.1 img/s | 6.13 ms / 163.1 img/s | **0.90×** |
| Home, B8 × 128² changing cameras | 24.35 ms / 328.5 img/s | 58.63 ms / 136.5 img/s | **2.41×** |
| Voxel51, B8 × 128² changing cameras | 4.78 ms / 1,674.0 img/s | 15.15 ms / 528.0 img/s | **3.17×**, fidelity fail |
| Home + real G1, B32 × 256², sparse reflections | 68,618 rays in 3.07 ms | no raster equivalent | wall throughput −0.2% |

The 1,024-camera contract contains 1,024 distinct cameras per step and all
1,024 move between steps. Tracer throughput rises from 138.3 img/s at B1 to
423.5 at B128, then stays at 423.1 at B1024: a useful **3.06× throughput
gain**, but not linear scaling. Nsight Systems attributes essentially all
measured GPU time to the internal OptiX launch; the CUDA camera-unpack kernel
is only about 3.85 microseconds per call.

Home fidelity is informative but does not pass the current publication gates
(B8: 35.24 dB RGB PSNR, 0.0179 alpha MAE, 0.36% mean relative depth error at
alpha > 0.5, 99.56% semantic pixel agreement). Voxel51 exposes a real
limitation: 25.66 dB, 0.0493 alpha MAE,
10.8% depth error, and 81.44% semantic/background agreement. Large, thin,
oblique Gaussians have different ray-volume and projected-splat support; this
is not a sort-key collision and makes the tracer unsuitable as a universal
raster replacement.

The exact PR wrapper also had four production blockers: it round-tripped CUDA
cameras through the CPU, reused `K[0]` for the whole batch, raced non-default
Torch streams, and retained 2.34 GB after `close()`. The hardened path fixes
all four, adds near/far-plane and malformed-dump validation, and passed:

- 296 Python tests and the native CUDA/OptiX build;
- Compute Sanitizer memcheck with zero errors;
- legacy dumps with and without semantics plus the new stable-precision dump;
- bitwise CLI/device-camera/per-camera-intrinsics/current-stream checks inside
  a live Isaac `SimulationApp`;
- an actual Isaac RTX G1 frame while the Home OptiX context stayed alive, and
  a CUDA-resident custom-Home + RTX-G1 depth composite.

**Verdict:** useful after hardening for static-map batched
cameras and especially sparse secondary rays. The original 4090 7.8×/2.5×
numbers below remain valid measurements for those particular B32 camera
bundles, not accepted publication speedups. B1 can be slower, scaling
saturates, and geometry-dependent raster equivalence is not guaranteed.

## Original tracer output contract, zero-copy interop, and reflections (RTX 4090)

Recorded 2026-07-20 on a rented Vast.ai RTX 4090 (instance 45429724, same
known-good stack as the section below). Evidence:
`benchmarks/tracer_output_contract_validation_4090_2026-07-20.json`.

- **Full output contract**: the OptiX tracer now emits depth (weight-averaged
  camera-z / accumulated alpha, +inf when empty) and strongest-contributor
  semantic ids, matching the raster's conventions. Measured parity vs the
  raster on the same camera at the new defaults: **PSNR 36.34 dB, alpha MAE
  0.0024, depth relative error 0.18% on jointly-valid pixels, semantic
  agreement 99.4%** — this passed the then-current v2 experimental gates, but
  does not pass the stricter publication gates
  (`experiments/optix_tracer/compare_raster_reference.py` v2).
- **Zero-copy torch interop**: `libgs_tracer.so` + `gs_tracer_torch.py`
  render dense and sparse launches directly into caller-owned CUDA torch
  tensors. The library path is **bitwise identical to the CLI** (rgba, depth,
  and semantic outputs; `gs_tracer_torch_smoke.py`), with an in-process dense
  launch at 3.8 ms for 8 envs × 256².
- **Reflection pass wired into the hybrid**:
  `run_g1_hybrid_warp.py --reflections` emits world-space reflection rays at
  chrome-robot pixels (Warp kernel), traces them through the gaussian field
  via the zero-copy sparse API, and Fresnel-blends before the depth
  composite. At 32 envs: **68,635 reflection rays/frame for +2.2 ms** —
  414 img/s with live chrome reflections vs 428 without (3.3% of the frame).
- **Per-link articulation** (`run_warp_mesh_tracer.py --per-link`): all 36 G1
  links independently posed per env per frame via per-env mesh + BVH refit —
  0.40 ms/env at 32 envs, still ~9× under the 4 ms/env gate. The refit
  dominates (11.0 of 12.8 ms at 32 envs); per-link ray-space instancing
  (no refit) is the better long-term design and stays a follow-up. Pivots are
  link centroids, not URDF joints — this measures cost, not gait realism.
- **Further optimization, ray-matched and parity-gated**: raising the hit
  threshold to `alpha_min = 2/255` (runtime + export boxes sized for it;
  mean box radius 2.57σ, 9,210 more never-visible gaussians dropped) is the
  new default: **9.39 ms exterior (7.81× the raster) / 38.39 ms interior
  (2.49×)**, vs 12.2/49.3 ms at the previous defaults, at equal-or-better
  full-contract parity. K=24 and K=32 measured strictly worse than K=16.

## RTX 4090 validation of the RTX Gaussian compositing path

Recorded 2026-07-20 on a rented Vast.ai RTX 4090 (instance 45419418, driver
590.48.01, CUDA 12.8, OptiX 9.1.0, torch 2.11.0+cu128, Warp 1.15.0, OVRTX
0.3.0.312915, Isaac Sim 6.0.1, Ubuntu 24.04, AMD EPYC 7742). Scene: Home Scan
LOD0, sha256 `29cee159…`, 21,497,908 gaussians. Raw per-run JSON, parity
report, and environment snapshot:
`benchmarks/rtx_compositing_validation_4090_results_2026-07-20.json`.

Every lane of the compositing PR was re-measured on this one node, against the
numbers claimed in the source PRs (#51/#53/#54):

### OptiX RT-core Gaussian tracer (`experiments/optix_tracer/`)

GAS build 39.7 ms, 0.54 GB compacted (claimed 39 ms / 0.54 GB).

| workload | claimed v1 | measured v1 | claimed k-buffer+clamp | measured |
|---|---:|---:|---:|---:|
| dense 32×256² | 49.6 ms / 42 M | 47.9 ms / 43.7 M | 13.6 ms / 154 M | **13.4 ms / 156.7 M** |
| dense 128×128² | 76.4 / 27 M | 74.8 / 28.0 M | 19.6 / 107 M | **19.1 / 109.6 M** |
| sparse 30k | 33 / 0.9 M | 32.7 / 0.9 M | 8.0 / 3.7 M | **8.0 / 3.8 M** |
| sparse 3M incoherent | 915 / 3.3 M | 906.3 / 3.3 M | 219 / 13.7 M | **216.3 / 13.9 M** |
| unified hybrid (32 chrome robots + reflections) | — | — | 13.9 ms | **13.3 ms** (29,760 refl px/frame) |

Every tracer claim reproduces within ~3%.

**Quality parity vs the raster reference (historical experimental gate,
`experiments/optix_tracer/compare_raster_reference.py`): PASS under that
gate** — PSNR 36.27 dB, MAE 0.0016 on the identical dense-mode env-0 camera,
linear color (then-current gates: ≥25 dB, ≤0.03). This is below the current
publication PSNR threshold.

### Warp mesh tracer and tier-1 hybrid

| lane | claimed | measured |
|---|---:|---:|
| Warp mesh gate, dense body, 32 animated robots | 4.3 ms | **0.79 ms** (same 6,130 hit px/env) |
| Warp mesh gate, real G1 (382,618 tris, new asset) | — | 0.89 ms / 32 robots |
| Warp hybrid 32×256², animated robots | 134.4 ms / 238.2 img/s | **74.0 ms / 432.2 img/s** |
| Warp hybrid, real G1 mesh, 32×256² | — | 74.3 ms / 430.4 img/s |
| Warp hybrid 128², 64–128 envs (ego-view proxy) | ~418 img/s | **518.6 img/s** |

Both Warp lanes are *faster* than claimed: the mesh gate by ~5.4× (Warp
1.15.0 vs the older Warp used in #53) and the hybrid by ~1.8× — the Gaussian
layer now costs 2.29 ms/env at 256² versus 4.17 ms/env when #53 was measured,
because `main`'s kernel improved after the source stack branched. The frame
remains ~99% Gaussian layer (32 envs: 73.2 ms gaussian, 0.27–0.53 ms mesh,
~1 ms compositor).

### OVRTX unified baseline — DOES NOT fully reproduce

| lane | claimed | measured (repeat) |
|---|---:|---:|
| OVRTX unified 32 envs, with robots | 144.8 ms / 221 img/s | **215.1 ms (212.5) / 148.7–150.6 img/s** |
| splats-only control, 32 envs | 128.7 ms | 210.6 ms (189.4) |
| robot increment per env (8–32 envs) | ≤0.5 ms | ≤0.24 ms ✓ |

The "robots are ~free in a unified OVRTX scene" claim reproduces; the
absolute OVRTX frame rate does not (~1.45× slower, repeatable, also in the
splats-only control, so it is not robot- or asset-related). The Isaac-RTX
per-env lane is likewise slower than claimed on this host (rtx 349 ms vs
204 ms at 4 envs). Both NVIDIA-RT-stack lanes underperform their #51 claims
on this node while every custom CUDA/OptiX lane matches or beats its claim —
the cross-node variance is exactly why this validation re-ran all lanes on
one machine.

### Ray-matched RT-vs-raster (corrects the "10×" context claim)

`benchmarks/run_rt_vs_raster.py` — both renderers on **identical cameras**
(via `gs_tracer --cams`), same scene, one node, 3 repeats (spread ≤0.3%),
with an env-0 PSNR gate proving both drew the same image:

| camera set | raster | tracer (k-buffer+clamp) | speedup | hit frac | env-0 PSNR |
|---|---:|---:|---:|---:|---:|
| exterior bundle (the raster lanes' standard cameras) | 73.4 ms | 17.4 ms | **4.21×** | 0.100 | 35.8 dB |
| interior ego-view (RL-observation shape) | 95.6 ms | 62.0 ms | **1.54×** | 0.400 | 40.1 dB |

The "~10×" figure in PR #54 compared the tracer on its own
identity-orientation synthetic cameras (hit fraction 0.085 — mostly empty
rays) against a raster number measured on different views; it does not
survive ray-matching. The honest statement: **4.2× on the standard exterior
views, 1.5× on interior ego views**, in the tracer's favor, while the raster
additionally pays the full output contract (rgb+depth+alpha+semantic) and the
tracer emits RGBA only. The tracer's structural advantages — exact per-ray
depth ordering with no global sort, and sparse secondary-ray batches (30k
rays in 8.0 ms) the raster cannot express at all — are unchanged.

### Tracer optimization sweep (ray-matched, parity-gated)

Six configs through the same harness, 3 repeats each; the parity PSNR column
is the gate that separates real wins from quality trades:

| config | exterior ms (speedup) | interior ms (speedup) | PSNR ext/int |
|---|---:|---:|---:|
| base (3σ AABBs, K=8) | 16.9 (4.32×) | 60.1 (1.57×) | 35.8 / 40.1 |
| + adaptive AABBs | 14.4 (5.08×) | 55.8 (1.69×) | 35.8 / 40.1 |
| + adaptive, K=16 → **new default** | **12.2 (5.97×)** | **49.3 (1.91×)** | **35.8 / 40.1** |
| + adaptive, tmin=0.05 | 12.8 (5.70×) | 50.4 (1.87×) | 35.5 / 37.6 |
| + adaptive, K=4 | 22.1 (3.31×) | 78.8 (1.19×) | 35.6 / 40.0 |
| + adaptive, max-iters=128 | 14.4 (5.06×) | 55.8 (1.69×) | 35.8 / 40.1 |

Two changes are strictly better (faster at *identical* parity PSNR) and are
now the defaults, confirmed end-to-end at **5.97× exterior / 1.91×
interior**:

- **Opacity-adaptive AABBs** (`export_gaussians_bin.py`, default on): a
  gaussian with opacity `o` cannot clear `alpha_min` beyond
  `r = sqrt(2·ln(o/alpha_min))` sigmas, so its 3σ box is oversized for any
  `o < 0.35`; mean box radius drops to 2.80σ and 4,519 never-visible
  gaussians are dropped. −15–19% trace time.
- **K-buffer depth 16** (was 8): avg traversal restarts per ray fall
  2.55 → 1.68 on interior views. −12–14% trace time. K=4 is strictly worse;
  K beyond 16 is untested (register pressure risk).

`--tmin 0.05` buys a further ~4% for −2.5 dB interior PSNR — a knob for
speed-critical demo paths, not the default. `--max-iters 128` does nothing
(the transmittance early-out already dominates).

### Same-hardware verdict

On identical silicon, same scene, same day: Warp hybrid **432 img/s with
animated robots** vs unified OVRTX **149 img/s with static robots** — the
hybrid wins by ~2.9× (the #53 claim was 1.08×). The RT-core tracer beats the
tile raster ray-matched — **6.0× on the standard exterior cameras and 1.9×
on interior ego views with the optimized defaults** (4.2×/1.5× before the
sweep) — with verified visual parity, exact depth ordering, and a sparse
secondary-ray capability the raster has no equivalent for.

## L4 OVRTX projection-mode activation result

Recorded 2026-07-20 on an NVIDIA L4, tested source commit `73ef6ee`.

The fail-closed seven-process activation gate completed as a valid experiment
and returned the scientific classification
`NO_OBSERVABLE_MODE_EFFECT`. Stage authoring, composed-token readback,
candidate-output validation, matched-lane checks, repeat controls, and the
repeated positive control all passed. Activation proof itself failed because
perspective and tangential were bitwise identical in both bracketed
comparisons:

| Perspective vs tangential | RGB | Alpha | Valid depth | Depth | Semantic |
|---|---:|---:|---:|---:|---:|
| Candidate pair, differing elements | 0 / 196,608 | 0 / 65,536 | 0 / 65,536 | 0 / 3,556 jointly valid | 0 / 65,536 |
| Repeat pair, differing elements | 0 / 196,608 | 0 / 65,536 | 0 / 65,536 | 0 / 3,556 jointly valid | 0 / 65,536 |

The camera-authoring/output path was responsive to the repeated focal
control. The 5 percent focal-length change affected 10.6567 percent of RGB
and alpha elements, 2.7237 percent of semantic elements, and 2.4750 percent
of valid-depth elements. Its signed alpha radial second moment changed by
`+767.6951169741515` in both brackets with zero repeat noise.

All seven child processes exited zero without a signal or timeout. Every log
passed the fatal scan and contained exactly one cleanup marker followed by one
capture marker. The outer return code of one is the expected fail-closed
scientific rejection, not an infrastructure failure. The explicit default RTX
Isaac headless lifecycle smoke also exited cleanly. The complete local raw
artifact set has 180 checksum-verified files.

The conclusion is deliberately narrow: OVRTX 0.3.0 composed and read back both
tokens but showed no mode-dependent behavior for this L4/Isaac Sim 6.0.1 RTPT
fixture. Readback proves composition, not render-kernel consumption. This is
not evidence of global OVRTX non-support, a schema violation, or a proprietary
renderer defect, and it makes no throughput or custom-renderer fidelity claim.
The tangential benchmark lane therefore remains blocked. The compact machine,
protocol, metrics, checksums, and limitations record is
`benchmarks/ovrtx_projection_activation_l4_results_2026-07-20.json`.

## L40S CUB double-buffer capacity result

Recorded 2026-07-17 on an NVIDIA L40S, branch
`codex/issue-32-cub-double-buffer-impl`.

The changing-camera emitted-prefix sort now uses CUB's `DoubleBuffer` API and
normalizes whichever key/value buffer owns the sorted output without a device
copy. On the exact Home Scan 384x384 trajectory, the change saves 1.21 GiB of
workspace at B8, 4.05 GiB at B40, and 5.00 GiB at B48. B48 completes with a
20.09 GiB backend workspace and 22.04 GiB peak Torch allocation on the
46,068 MiB L40S.

This is a capacity result, not a speedup. Across two independent ABBA runs per
implementation, mean throughput changes by -0.0367 percent at B8, +0.0449
percent at B40, and +0.0356 percent at B48. Those effects are timing noise.
Nsight Systems explains the result: the candidate still executes eight radix
passes over every 64-bit key, with sorting accounting for 36.42 percent of
traced kernel time.

All trajectory cache-event and final-overflow checks pass. A separate full B48
capture validates RGB, alpha, depth, and semantic contracts across all 240
source views. Every cross-implementation view passes every same-equation
threshold; worst values are 98.53 dB PSNR, 0.9999999986 SSIM, `7.06e-9` LPIPS,
`4.84e-8` alpha MAE, `3.53e-9` mean relative-depth error, and 0.999607 semantic
agreement. An independent full baseline repeat is also not bitwise identical
and provides the nondeterministic noise control. Four Compute Sanitizer
access-safety runs report zero errors. Full leak reporting finds the same
2,097,152-byte, 1-byte, and 8-byte PyTorch caching-allocator teardown
allocations in both baseline and candidate.

The complete machine, protocol, per-run timing, memory, fidelity, profiler,
sanitizer, raw-query, checksum, and limitation record is in
`benchmarks/cub_double_buffer_l40s_results_2026-07-17.json`.

This standalone CUDA A/B does not launch Isaac RTX or prove co-residency with
it, does not prove the B48 result on a 23 GiB L4, and does not replace RTX
rendering. It also exposed a pre-existing stale-cache edge case for mutable
camera tensors created inside `torch.inference_mode()`; that is tracked as
issue 33.

The historical OVRTX acceptance framework remains **FAIL** because its strict
fidelity thresholds are not met. Its throughput, memory, custom-kernel
ownership, GPU-output, integration, deterministic-replay, matrix-coverage,
and three-run-reproducibility sub-gates do not establish the current
Custom-versus-FlashGS publication claim.

## Historical OVRTX system comparison (not a rasterizer headline)

Recorded 2026-07-16 on Vast instance `45069639`, branch
`codex/vectorized-gsplat`.

### Home Scan LOD0

The final Home Scan input is the full 21,497,908-Gaussian LOD0 PLY. The
optimized custom path uses project-owned covariance, projection, visibility,
depth-cutoff, intersection, ordering, and compositing kernels. It uses 1x1
pixels, 32 depth buckets in groups of eight, and retains an opt-in
projection-coherence cache. The current B1024 configuration uses eight
physical P128 submissions for one logical service request; this section does
not support a one-physical-submission claim.

The direct cache stores per-pixel depth cutoffs, sorted Gaussian-ID
intersections, and dense pixel ranges. It does not store a per-camera
projected-record list or rendered images. On every cache hit the compositor
reprojects qualifying Gaussian IDs and rewrites RGB, depth, alpha, and semantic
outputs.

In this dated OVRTX comparison, the largest batch both renderers executed was
B1024 at 128x128. OVRTX B2048 failed with Vulkan out-of-device-memory and CUDA
interop allocation errors.

| Renderer | Mean frame | Images/s | Peak driver memory |
|---|---:|---:|---:|
| Custom direct compact, 3-run mean | 44.923 ms | 22,794.59 | 11.966 GB |
| Fair OVRTX, 3-run mean | 4,645.667 ms | 220.456 | 18.896 GB |

The archived measurement reported 103.397x throughput and 63.326 percent of
OVRTX's memory. Because this comparison uses different image-formation
equations and a cache-heavy workload, those values are not a publishable
rasterizer speedup or memory headline. Independent-process throughput spread
was 0.0212 percent custom and 3.1203 percent OVRTX.

The current direct B1024 run reports 146,618,238 visible-Gaussian observations,
167,147,353 qualifying pixel intersections, 1,786,429 active pixels, zero
overflow, no measured-loop allocation growth, and all four outputs on
`cuda:0`. The legacy visible-record arrays have one-element placeholders in
this mode.

Single-camera evidence is retained separately:

| Renderer | Mean frame | Images/s | Driver memory |
|---|---:|---:|---:|
| Custom direct compact | 0.2643 ms | 3,784.09 | 2.733 GB |
| OVRTX, 3-run mean | 7.7102 ms | 129.705 | 13.764 GB |

Nsight Systems captures at B1 and B1024 confirm that direct projection and
global CUB sorting execute on the first cache miss only, while the fused
reproject/compositor executes on every measured render. Across the profiled
B1024 setup plus 31 compositor calls, projection accounts for 54.81 percent,
raster/compositing 43.14 percent, and sorting 1.03 percent of traced GPU stage
time. Steady cache hits are compositor dominated.

Nsight Compute 2025.1.1 is installed, but this Vast container exposes
`RmProfilingAdminOnly=1` without `CAP_SYS_ADMIN`; the preflight records that
hardware counters are unavailable on this host.

Evidence:

- `outputs/acceptance/b1024/`
- `outputs/compact/ovrtx-home-b2048-128-boundary/run.log`
- `outputs/compact-direct/home-b1-128/result.json`
- `outputs/profiles/direct-home-b1-128/`
- `outputs/profiles/direct-home-b1024-128/`
- `outputs/profiles/ncu-direct-preflight/`

### Current synthetic and multi-scene regression

The previously verified general projected-record path completed 36
shared-scene and 30 packed multi-scene cases across batches 1, 8, 32, 64, 128,
and 256 and resolutions 128, 256, and 512.

- 66/66 custom cases passed output, allocation, and overflow checks.
- Joined against the fair authored-display OVRTX matrix, 36/36 matched cases
  pass the 5x throughput and 80-percent memory gates.
- Minimum matched speedup: 89.537x.
- Maximum matched driver-memory ratio: 23.760 percent.
- Three independent synthetic-medium B256 512x512 runs measured
  19.9887, 19.9903, and 19.9926 ms.
- Relative timing spread: 0.0198 percent, with identical 3.798 GB driver
  memory, zero allocation delta, and zero overflow.

Evidence is under `outputs/current-custom-matrix/`,
`outputs/comparisons/ovrtx-current-custom-full-matrix.json`, and
`outputs/stability/current-medium-b256-512/`.

The refreshed direct-compact matrix across synthetic, public real-world, and
Home Scan scenes is still pending and is not implied by this historical B1024
system comparison.

### Ten-minute soak and CUDA sanitizer

The synchronized full Home Scan B1024 128x128 direct-compact soak completed
after 603.329 seconds:

- 13,400 measured batched render calls.
- 22,794.55 images/s and 44.9230 ms mean GPU frame time.
- Zero measured-loop allocation growth.
- Zero visible or intersection overflow.
- No CUDA errors and no NaN/Inf or invalid output-contract observations.
- GPU-resident RGB, depth, alpha, and semantic output remained valid at every
  synchronized check.

The independent verification checkout also ran Compute Sanitizer memcheck over
the 1x1-tile projection-cache path. Cache miss, coherent hit, feature-only
update, camera mutation, and manual invalidation all passed with
`ERROR SUMMARY: 0 errors`.

Evidence:

- `outputs/soak/direct-home-b1024-128-600s.json`
- `outputs/verification-independent/compute-sanitizer.log`
- `outputs/verification-independent/remote-evidence/`

### Fidelity verdict

Strict parity with pinned conventional `gsplat` passes:

- B1 RGB PSNR 157.71 dB, alpha MAE `1.17e-8`, depth relative error
  `1.13e-7`, semantic agreement 1.0.
- B8 RGB PSNR 156.61 dB, alpha MAE `1.26e-8`, depth relative error
  `1.02e-7`, semantic agreement 1.0.

The immutable authored-display OVRTX temporal reference still fails every
strict acceptance threshold:

| Metric | Required | Measured | Verdict |
|---|---:|---:|---|
| RGB PSNR | >= 40 dB | 35.094 dB | FAIL |
| RGB SSIM | >= 0.995 | 0.985668 | FAIL |
| LPIPS | <= 0.01 | 0.012180 | FAIL |
| Alpha MAE | <= 0.005 | 0.023973 | FAIL |
| Depth relative error | <= 0.01 | 0.057893 | FAIL |
| Semantic agreement | >= 0.999 | 0.932251 | FAIL |

The project therefore does not claim overall acceptance, despite passing
performance and memory gates.

This failure is caused by a rendering-equation difference, not by fewer
Gaussians or a lower output resolution. The production path projects each 3D
Gaussian to a deterministic 2D EWA ellipse and alpha-blends it. OVRTX
perspective behaves like stochastic full-3D ray evaluation with probabilistic
first-hit contributions, changing splat footprints, coverage, depth, and
semantic winners.

An experimental exact-ray diagnostic against the 65,536-frame OVRTX
perspective reference measures:

| Metric | Required | Measured | Verdict |
|---|---:|---:|---|
| RGB PSNR | >= 40 dB | 55.449 dB | PASS |
| RGB SSIM | >= 0.995 | 0.999723 | PASS |
| LPIPS | <= 0.01 | 0.000124 | PASS |
| Alpha MAE | <= 0.005 | 0.001317 | PASS |
| Depth relative error | <= 0.01 | 0.005638 | PASS |
| Semantic agreement | >= 0.999 | 0.995544 | FAIL |

Most remaining semantic disagreements are near ties where finite temporal
sampling can select the runner-up label. The exact-ray path remains
diagnostic: it has not completed the same Home Scan performance, full matrix,
ten-minute soak, and independent acceptance program as the fast EWA path.

A production-native exact-semantics capture against a fresh, resumable
65,536-frame reference separates the semantic rules more clearly. The current
native strongest-individual rule reaches `0.984741` agreement (250 mismatched
pixels), while full aggregate first-hit label probability reaches `0.995667`
(71 mismatched pixels). Aggregate semantics is therefore required, but it
still does not pass the `0.999` threshold at this finite sample count. On this
1,024-label stress fixture, positive exact contributors contain 2.936 distinct
labels per pixel on average, p99 is 8, and the maximum is 14; 133 pixels exceed
an eight-label accumulator and none exceed 16. This is sizing evidence for the
fixture, not a global correctness bound or a Home Scan bound.

A tuned one-camera Home Scan comparison against a 256-frame OVRTX sample
passes the numerical thresholds at PSNR 47.270 dB and semantic agreement
0.999084. It is retained as a finite-frame scene-specific control and is not
used as equation-level OVRTX perspective acceptance evidence.

The OVRTX USD schema defines `projectionModeHint` as advisory. A paired
1,024-frame isotropic synthetic experiment authored `perspective` and
`tangential` into distinct USD stages, but all output tensors were bitwise
identical. A later anisotropic Home Scan pair differed at frame 256, but that
pair had no same-mode repeat control. The fail-closed seven-process L4 gate
described above supersedes that uncontrolled pair: it recorded valid composed
token readback, matched rotated anisotropic off-axis inputs, zero same-mode
repeat noise, a stable focal positive control, and bitwise-identical
perspective/tangential outputs in both brackets. The earlier Home Scan delta
is therefore not evidence that OVRTX consumed the hint.

The supported conclusion is narrower: OVRTX 0.3.0 RealTimePathTracing stores
and composes `projectionModeHint` but has shown no behavioral response to it
in the tested configurations. OpenUSD permits renderers to ignore this
advisory hint. It is therefore observationally inert in those tested
configurations; this evidence does not establish OVRTX's global support
status, a schema violation, or a proprietary-renderer defect. The tangential
benchmark lane remains blocked.

A fresh Home Scan tangential control at B1/128x128 accumulated 1,024 frames
with runtime readback `tangential`. It reached 48.541 dB PSNR, 0.999585 SSIM,
0.000767 LPIPS, 0.000794 alpha MAE, and 0.001313 relative depth error; all of
those thresholds pass. Semantic agreement is 0.998291 (28 mismatched pixels),
so the run correctly remains **FAIL** against the 0.999 semantic gate. This is
a run with an authored and read-back `tangential` token; the current evidence
does not establish that its pixels came from a distinct tangential rendering
path. It is retained as a diagnostic control, not tangential acceptance or
proof that OVRTX uses the conventional EWA equation.

### Exact-ray macro-bin validation

The exact-ray 16x16 compositor now sorts one conservative 32x32 parent bin and
shares that center-Z-ordered range across its four child raster blocks. The
unchanged per-pixel 3D-ray support test rejects the extra candidates. A
committed fine-bin baseline and the macro-bin implementation produced
byte-identical NPZ artifacts for all outputs at B1/128x128 and at B2/130x129,
including odd edge tiles.

Targeted medium-scene validation of the eight memory-risk rows produced zero
visible/intersection overflow and numerically passed both performance gates:

| Batch / resolution | Custom memory | OVRTX memory | Ratio | Speedup |
|---|---:|---:|---:|---:|
| 256 / 128x128 | 1.592 GB | 3.910 GB | 0.4071 | 56.45x |
| 64 / 256x256 | 1.430 GB | 3.698 GB | 0.3867 | 24.01x |
| 128 / 256x256 | 2.343 GB | 4.921 GB | 0.4760 | 24.20x |
| 256 / 256x256 | 4.171 GB | 7.720 GB | 0.5403 | 23.51x |
| 32 / 512x512 | 2.252 GB | 4.914 GB | 0.4584 | 18.40x |
| 64 / 512x512 | 4.010 GB | 7.119 GB | 0.5633 | 18.21x |
| 128 / 512x512 | 7.502 GB | 12.332 GB | 0.6083 | 18.51x |
| 256 / 512x512 | 14.475 GB | 21.865 GB | 0.6620 | 18.73x |

These were five-iteration validation rows, not the final 50-iteration,
multi-scene, reproducibility, soak, and independent acceptance rerun. They
resolve the former exact-ray workspace/OOM failure mechanism but do not change
the overall FAIL verdict.

## Kernel architecture milestone

Recorded 2026-07-16 on branch `codex/kernel-architecture`:

- `KERNEL_ARCHITECTURE.md` defines a source-grounded candidate CUDA renderer
  architecture, canonical SoA scene layout, custom extension boundary,
  quantitative memory/performance model, decision gates, and rejected
  alternatives.
- `EXPERIMENT_LEDGER.md` maps projection/covariance, compaction, macro/tile
  binning, ordering, fused compositing, reusable workspace, CUDA graph,
  deterministic mode, and zero-copy binding decisions to required measurements.
- `src/isaacsim_gaussian_renderer/layout_contract.py` and
  `tests/test_layout_contract.py` add interface/data-layout scaffolding for
  canonical scene and render-batch tensors.

This is not an accepted final architecture. D-003 and D-011 still require fair
RTX/current-`gsplat` baselines and profiles before architecture selection.

## Environment snapshot

Recorded 2026-07-16 on Vast instance `45069639`:

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB |
| Driver | 590.48.01 |
| Compute capability | 8.9 |
| CUDA compiler | 12.8, build 12.8.93 |
| Isaac Sim | 6.0.1-rc.7, release build 42383 |
| Kit | 110.1 |
| PyTorch | 2.11.0+cu128 |
| NCCL | 2.28.9 |
| `gsplat` | `77ab983ffe43420b2131669cb35776b883ca4c3c` |
| OVRTX | `v0.3.0-1-g29d1103` |
| Kit app template | `9becc9cb1cce0448f5914d148127a2aa219609da` |
| GCC/G++ | 13.3.0 |
| CMake | 3.28.3 |
| Ninja | 1.11.1 |
| Nsight Systems | 2024.6.2 |
| Nsight Compute | 2025.1.1 |
| Nsight Graphics | 2026.2 |

## Headless Isaac validation

`scripts/isaac_headless_smoke.py` launches `SimulationApp` directly, creates a
world, adds a ground plane, performs ten rendered steps, and exits. It passed
and emitted
`ISAAC_HEADLESS_SMOKE_OK`.

## Preliminary synthetic `gsplat` control

These numbers are infrastructure smoke results, not a fair baseline and not an
acceptance result. The run predates the complete benchmark protocol, lacks an
RTX comparison and semantic output, and uses a synthetic scene.

| Cameras | Gaussians | Resolution | Outputs | Batch time | Images/s | MP/s | Peak allocated |
|---:|---:|---:|---|---:|---:|---:|---:|
| 8 | 10,000 | 64x64 | RGB, depth, alpha | 0.652 ms | 12,260.8 | 50.2 | 0.007 GiB |
| 64 | 100,000 | 128x128 | RGB, depth, alpha | 3.937 ms | 16,257.9 | 266.4 | 0.500 GiB |

Preliminary Nsight Systems kernel-time distribution for the 64-camera run:

- Forward rasterization: approximately 55.1 percent.
- Radix sort: approximately 17.6 percent.
- Tile intersection: approximately 10.9 percent.
- RGB/depth concatenation: approximately 6.3 percent.
- Projection: approximately 5.0 percent.

Trace:
`/workspace/vectorized-gaussian-renderer/profiles/gsplat-batched.nsys-rep`

## OVRTX Gaussian ingestion smoke

This is a renderer-ingestion and AOV smoke result, not a throughput baseline or
an acceptance result. OVRTX 0.3.0 successfully loaded a concrete
`ParticleField3DGaussianSplat` prim containing four colored Gaussians and
rendered it headlessly through Real-Time Path Tracing.

The canonical probe uses the same DC spherical-harmonic convention as
NVIDIA's `usd-convert-gsplat` converter:
`rgb = clamp(0.5 + 0.28209479177387814 * sh_dc)`. At 512x512 after eight
warmup frames with OVRTX's default particle-field color path:

| Output | Observed result |
|---|---|
| LDR RGB | 126,884 nonzero pixels; 6,157 unique colors; maximum 255 |
| Distance to image plane | 41,520 finite positive pixels near 5.0 m |
| `DepthSD` | 41,520 finite positive pixels; normalized/device-depth values |
| LDR alpha | 255 everywhere; this is composited output alpha, not Gaussian coverage |

The local evidence is under
`outputs/ovrtx-canonical-single-default/`; the reproducible probe is
`experiments/ovrtx_gaussian_smoke.py`. A bundled triangle/mesh scene also
produced finite nonzero LDR and HDR output, isolating the earlier black/NaN
result to the Gaussian tonemapping path rather than camera or AOV setup.

The same canonical scene exposed a current tiled-path limitation. One
RenderProduct targeting four identical cameras produced a 512x512 tiled frame
with 41,277 finite positive distance pixels, but its LDR RGB was entirely
black. Both the default and explicit `skipTonemapping = false` variants failed
the tiled color check. This does not block single-camera RTX correctness, but
the tiled result cannot be used as the fair RGB baseline until repaired.

The viable batched alternative is one host submission containing four
single-camera RenderProducts. With four identical 512x512 cameras, the same
canonical scene produced:

| Output | Observed result |
|---|---|
| LDR RGB | 508,586 nonzero camera-pixels; 8,113 unique colors; maximum 255 |
| Distance to image plane | 166,080 finite positive pixels near 5.0 m |
| Output tensor shape | `[4, 512, 512, channels]` |
| Host submission | One `Renderer.step()` containing all four RenderProducts |

The four outputs were visually inspected and match the single-camera result.
Local evidence is under `outputs/ovrtx-canonical-products-4/`. This path avoids
a per-camera render submission loop and is the current mechanism for the RTX
batch baseline; it is explicitly labeled multi-product rather than tiled.

## Fair OVRTX all-output synthetic control

The fair control keeps one deterministic 10,000-Gaussian scene resident,
submits all camera RenderProducts in one `Renderer.step()`, maps native AOVs to
CUDA, and writes the required contract into preallocated CUDA tensors:

- sRGB float32 RGB normalized from `LdrColor`;
- metric float32 `DistanceToImagePlaneSD`;
- generated coverage alpha as float32;
- int64 semantic class IDs with background `-1`.

OVRTX semantics are prim-level, so the fixed
`semantic_id[i] = i % 1024` sidecar is represented by 1,024 labeled Gaussian
ParticleFields. `SemanticIdMap` is decoded once and native renderer IDs are
remapped on CUDA during every timed frame. This cost is included.

| Cameras | Resolution | Gaussians | Iterations | Mean wall | p95 wall | Images/s | Driver VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128x128 | 10,000 | 50 | 14.349 ms | 14.600 ms | 69.69 | 2.51 GB |
| 8 | 128x128 | 10,000 | 50 | 93.827 ms | 94.570 ms | 85.26 | 2.57 GB |

Both runs have machine-readable `MEASURED`/pass rows and locally saved RGB,
depth, alpha, native semantic, remapped semantic, stage, and ID-map evidence
under `outputs/baselines/ovrtx-fair-b1-128/` and
`outputs/baselines/ovrtx-fair-b8-128/`.

The renderer call is one batched host submission. OVRTX still exposes one
mapped output per RenderProduct, so the baseline has a per-camera host mapping
loop after rendering; the result metadata records this limitation. OVRTX's
Vulkan stream is not available to CUDA events, so synchronized end-to-end wall
time is the primary measure.

Still required for the complete baseline campaign:

- Validate real PLY conversion against the canonical converter.
- Confirm Fabric Scene Delegate settings in the dated OVRTX configuration.
- Run all required batches, resolutions, real-world scenes, and three
  independent repetitions.
- Measure the fastest visually equivalent RTX configuration.

## Public dataset status

Selected public real-world Gaussian source:
Hugging Face `Voxel51/gaussian_splatting`, file
`FO_dataset/train/point_cloud/iteration_30000/point_cloud.ply`.

Remote download attempts on 2026-07-16 used
`scripts/download_public_gaussian_dataset.py` with `curl -L --fail --retry 8`.
The host repeatedly returned HTTP 504 before the PLY completed, so no checksum
is recorded yet.
## Profiling milestone: synthetic shared-scene `gsplat`

Recorded 2026-07-16 on `vast-gsplat-isaac` from
`/workspace/agent-worktrees/profiling` on branch `codex/profiling`.

Artifacts are saved under `outputs/profiles/`, including:

- Nsight Systems raw report:
  `outputs/profiles/gsplat-synth-shared-c64-n100k-128-nsys/nsys/gsplat-synth-shared-c64-n100k-128-nsys.nsys-rep`
- Nsight Systems exported SQLite and CSV summaries in the same `nsys/`
  directory.
- Parsed stage summary:
  `outputs/profiles/gsplat-synth-shared-c64-n100k-128-nsys/gsplat-synth-shared-c64-n100k-128-nsys_nsys_summary.json`
- PyTorch profiler trace:
  `outputs/profiles/torchprof/gsplat-synth-shared-c64-n100k-128-torchprof_torch_trace.json`
- Nsight Compute preflight evidence:
  `outputs/profiles/gsplat-synth-shared-ncu-preflight/gsplat-synth-shared-ncu-preflight_ncu_preflight.json`

Matrix timings used RGB, depth, alpha, and a scalar semantic-ID proxy via
`extra_signals`; the proxy is profiling-cost evidence, not semantic correctness.

| Cameras | Gaussians | Resolution | Mean GPU ms | p95 ms | Images/s | MP/s | Peak allocated | Tile intersections |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100,000 | 128x128 | 0.679 | 0.796 | 1,389.6 | 22.8 | 0.017 GiB | 77,882 |
| 8 | 100,000 | 128x128 | 1.060 | 1.243 | 7,190.2 | 117.8 | 0.103 GiB | 660,439 |
| 64 | 100,000 | 128x128 | 4.996 | 5.032 | 12,708.4 | 208.2 | 0.761 GiB | 5,329,600 |
| 32 | 100,000 | 256x256 | 2.798 | 2.894 | 11,255.7 | 737.7 | 0.496 GiB | 3,255,861 |

Nsight Systems stage distribution for the 64-camera, 100k-Gaussian, 128x128
run:

- Rasterization: 47.5 percent of kernel time.
- Sorting: 17.1 percent.
- Tile intersection: 10.0 percent.
- Concatenate/fill: 8.7 percent.
- Projection: 6.5 percent.
- Offset encoding: 1.0 percent.

CUDA API summary over the captured process recorded 845 kernel launches, 88
synchronization calls, and 29 allocator calls. Nsight Compute hardware counters
remain unavailable on this host because `/proc/driver/nvidia/params` contains
`RmProfilingAdminOnly: 1` and the container lacks `cap_sys_admin`; no occupancy
or bandwidth counters are claimed.

Detailed bottlenecks and optimization hypotheses are in `PROFILE_REPORT.md`.

## Asset status

The requested `.zip` was absent during the dated conversion. The canonical
published PLY and its checksummed dataset manifest supersede the
machine-local extracted SOG directory used at that time.

Validation found 434 files, 360 WebP payloads, 72 chunks, valid JSON, and no
missing files referenced by `lod-meta.json`. Counts are:

| LOD | Chunks | Gaussian records |
|---:|---:|---:|
| 0 | 32 | 21,497,908 |
| 1 | 19 | 10,748,589 |
| 2 | 10 | 5,374,295 |
| 3 | 6 | 2,687,148 |
| 4 | 3 | 1,343,574 |
| 5 | 2 | 671,787 |

The all-file relative manifest SHA-256 is
`ac0c8226d8f0e5359deb09026d12954bb7096143ebff8061c50303909e8f285d`.
LOD0 was decoded by `splat-transform` 2.7.1 from the 32 `0_*` chunks into:

| Artifact | Value |
|---|---|
| Remote path | `/workspace/datasets/home-scan-lod0.ply` |
| Vertex count | 21,497,908 |
| Byte count | 1,203,883,212 |
| SHA-256 | `29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267` |
| SH bands | 0 |
| Conversion elapsed | 43.17 seconds |
| Peak host memory | approximately 2.45 GB |

The validated directory is also present on the persistent remote workspace at
`/workspace/datasets/home-scan-sog` with 434 files and approximately 491 MB.
The reproducible conversion recipe and canonical metadata are saved in
`scripts/convert_home_scan_lod0.sh` and
`datasets/home-scan-lod0.manifest.json`. Full renderer ingestion and the final
Home Scan runtime/performance/memory benchmark now pass.

## Fidelity infrastructure milestone

Added on `codex/fidelity`:

- `camera-bundle-v1` deterministic JSON with canonical bundle SHA-256.
- `render-output-v1` JSON manifest and `.npz` tensor input support.
- Per-view and aggregate RGB PSNR, RGB SSIM, LPIPS, alpha MAE, valid-pixel
  depth relative error, and semantic agreement reporting.
- Worst-view selection, JSON/CSV reports, side-by-side RGB images, and
  RGB/depth/alpha/semantic difference images.
- CLI: `benchmarks/compare_fidelity.py` or installed `compare-fidelity`.
- Synthetic self-comparison and perturbation tests.

This is comparison infrastructure only. It does not claim RTX reference output,
custom renderer fidelity, throughput, memory, or Home Scan acceptance.

## Independent verification

Recorded 2026-07-16 on branch `codex/verification`:

- Added `VERIFICATION_PLAN.md` with the independent evidence contract, rerun
  procedure, and benchmark loopholes.
- Added `isaacsim_gaussian_renderer.verification`, a fail-closed verifier for
  `outputs/verification/acceptance.json` and
  `outputs/verification/benchmark_rows.csv`.
- Added synthetic fixture tests for successful evidence, missing acceptance
  evidence, manipulated statistics, and failing CSV rows.
- Updated verification scripts to require remote root
  `/workspace/agent-worktrees/verification` and exclude `outputs/` during
  optional source synchronization.

The final clean-root audit was then run from
`/workspace/agent-worktrees/verification`:

- Source was synchronized without producer outputs.
- Editable installation under Isaac Sim's Python runtime passed.
- 45/45 tests passed.
- The production CUDA backend compiled and loaded through a real headless Kit
  extension with Fabric Scene Delegate enabled.
- The physics-coupled vectorized example passed with GPU Fabric camera
  ingestion, one batched native renderer submission, GPU-resident outputs,
  bitwise deterministic replay, and no hidden physics stepping.
- Native, deterministic, packed multi-scene, and active-subset CUDA smokes
  passed.
- Compute Sanitizer reported zero errors.
- Local, producer, and verification copies of `renderer_cuda.cu` had the same
  SHA-256:
  `a630cd535a145db7a82d5469c161b4cbd564ca4000d0bbfc36bda62ef5cc7816`.

The independent raw-evidence auditor passed throughput, memory,
reproducibility, single-camera reporting, GPU output residency, project-owned
kernel ownership, and headless Isaac Sim integration. It returned overall
`FAIL` because the six strict OVRTX fidelity metrics fail. The all-pass
evidence verifier separately failed closed because no
`outputs/verification/acceptance.json` was issued. This is the correct result,
not an incomplete verifier run.

Evidence is retained under `outputs/verification-independent/`.

## Publication acceptance table

| Criterion | Required | Current verdict |
|---|---|---|
| Fidelity | All thresholds pass | **PENDING**: the corrected FlashGS adapter and both renderers must pass pinned-`gsplat` fidelity for every scheduled RGB/full row |
| Throughput | Custom beats FlashGS for the claimed scope | **PENDING**: complete matched v4 B1-B1024 dynamic-camera matrix required; interrupted diagnostic timings are ineligible |
| Peak memory | Report both methods under one contract | **PENDING**: complete matched v4 matrix and driver/allocator evidence required |
| Isaac Sim extension | Required | **PASS**: real CUDA backend through headless `SimulationApp` |
| Multi-camera | Required | **PASS** functional contract; comparative dynamic-camera performance remains pending |
| Multi-scene IDs | Required | **PASS**: 30/30 packed multi-scene cases |
| RGB/depth/alpha/semantic | Required | **PASS** runtime contract; all outputs GPU-resident |
| Reproducibility | Complete raw samples and repeated controls | **PENDING** for the matched FlashGS publication matrix; historical OVRTX spread is not transferable |
| Ten-minute stability | >=600 seconds | **PASS** historical custom-only soak; matched publication stability evidence remains separate |
| Independent verification | Required | **FAIL-CLOSED overall**: clean-root build/runtime/sanitizer checks pass; acceptance audit retains the OVRTX fidelity failures |
| Home Scan | Required final rerun | **PENDING**: rerun every matched row from the final corrected source revision |

## Trajectory evaluator implementation

The repository now contains a versioned trajectory contract, ten explicit
scenario schedules, synchronized GPU/wall timing, pinned-gsplat trajectory
control, expanded per-frame and temporal metrics, a cache mutation and
anti-stale-output audit, route validation, and a fail-closed artifact/video
auditor. The historical schema named `headline-acceptance/v3` reports
independent verdicts and forbids the static coherent result from satisfying
the dynamic gate; its name is not a current publication acceptance decision.

### Fresh RTX 4090 initial trajectory suite

Recorded on 2026-07-16 on dedicated Vast instance `45120960`, RTX 4090,
driver 580.119.02, CUDA 12.8 runtime/toolkit, and Isaac Sim 6.0.1. These
results intentionally keep static projection-cache hits separate from moving
cameras.

| Workload | Mean GPU ms | p95 GPU ms | Images/s | Cache result |
|---|---:|---:|---:|---|
| Static Home Scan B1, 128x128 | 0.1315 | 0.1434 | 7,602.17 | 500 measured hits, zero misses |
| Static Home Scan B1024, 128x128, run 1 | 40.9428 | 40.9446 | 25,010.52 | 200 measured hits, zero misses |
| Static Home Scan B1024, 128x128, run 2 | 40.9446 | not retained | 25,009.42 | 200 measured hits, zero misses |
| Static Home Scan B1024, 128x128, run 3 | 41.1056 | not retained | 24,911.46 | 200 measured hits, zero misses |
| Sequential Home Scan B1 T120, 256x256 | 6.3666 | 7.9440 | 157.07 | all moving requests miss |
| Phase-offset Home Scan B256 T120, 128x128, 3-run mean | 530.8507 | 531.3863 | 482.24 | all moving requests miss |

The three independent B256 dynamic runs measured 482.386, 482.213, and
482.135 images/s, a 0.0521 percent relative spread. The valid runs used a
409.6-million-intersection capacity; a deliberately preserved undersized run
failed with 328,263,575 overflowed intersections. Mixed-motion B256 measured
481.839, 479.094, 472.538, and 481.962 images/s at 1, 25, 50, and 100 percent
moving. Because every nonzero motion fraction invalidates the whole batch,
these are cache-miss costs and directly motivate a per-camera projection
cache.

The deterministic EWA trajectory comparison against pinned `gsplat` commit
`77ab983ffe43420b2131669cb35776b883ca4c3c` is **FAIL**. RGB, SSIM, LPIPS,
alpha, and depth pass their per-frame thresholds, but strict semantic
agreement is 0.992709 against the required 0.999, so all 120 frames fail the
all-metrics gate. Tie-aware semantics remain diagnostic only.

OVRTX perspective evidence is separately **FAIL**. A 16-sample full-path
static-all render produced 24 identical black frames (96-119); a dynamic-chunk
retry produced 60 black frames in alternating four-frame chunks. The artifact
guard rejected both, so neither is cited as a fidelity reference or valid
video and higher-sample convergence was not promoted.

The synthetic cache and stale-output canary passed 20 API requests, 20 native
submissions, and 20 renderer executions with 6 expected hits, 14 expected
misses, all required outputs rewritten, zero overflow, and explicit
Fabric/Warp-style external-write invalidation. This run found and fixed two
production bugs: allocator address reuse could produce a false moving-camera
hit, and inactive output slots could retain stale values.

The finite Isaac/Fabric B256 tests returned CUDA-resident RGB, alpha, depth,
and semantic tensors, did not advance simulation during rendering, and passed
external-write invalidation. They fail a 60 Hz latency objective:

| Motion | Physics mean ms | Fabric GPU mean ms | Renderer GPU mean ms | Integrated mean / p95 ms | Deadline misses |
|---|---:|---:|---:|---:|---:|
| Static | 1.596 | 0.256 | 14.540 | 16.552 / 17.293 | 92/300 |
| 50 percent | 5.944 | 0.310 | 14.914 | 21.346 / 31.526 | 300/300 |
| 100 percent | 9.064 | 0.290 | 15.010 | 24.530 / 41.289 | 300/300 |

Nsight Systems confirms that the valid moving-camera capture is projection
dominated: projection is 69.33 percent of kernel time, sorting 17.91 percent,
raster/compositing 11.77 percent, and offsets 0.56 percent. The static capture
contains two cold misses (warmup and measured) plus ten hits and has the
expected two projection kernels per miss and one raster stage per native
execution. Nsight Compute 2025.1.1 remains blocked by
`RmProfilingAdminOnly: 1` without `CAP_SYS_ADMIN`; no hardware-counter claims
are made.

The dynamic B256 soak completed 1,200 native renders over 10 full trajectory
repetitions in 642.376 seconds. It measured 531.1404 ms mean GPU latency,
481.982 images/s, zero allocation or reservation growth across all ten memory
checkpoints, zero visible/intersection overflow, and a passing reopened
120-frame/video artifact audit. Warmup contributed three cache hits; all 1,200
measured moving-camera calls were misses.

The representative route passes storyboard and measured route validation with
0.2283 m sampled minimum clearance, 8.9865 m path length, no duplicate poses,
and 0.8168 mean foreground/depth coverage. The deliberately adversarial stress
route also passes artifact/storyboard integrity while recording its intended
extremes: 0.00079 m sampled minimum clearance, 1.8252 rad maximum per-frame
rotation, 0.00486 near-plane-pressure fraction, and no duplicate poses.

## L4 emitted-memory follow-up

The exact Home Scan B40 path previously completed rendering but failed its
mandatory depth validation with a 108 MiB CUDA allocation request and only
103.38 MiB free. The renderer already caps intersections and tile segments at
`INT_MAX`, so the dense per-tile start and end offsets did not require int64.
Changing both arrays to int32 saves 47,185,920 bytes at B40.

On the same replacement NVIDIA L4, all 240 Home Scan views now pass at B40:

| Case | GPU images/s | Integrated wall images/s | Workspace | Maximum intersections | Verdict |
|---|---:|---:|---:|---:|---|
| Home B8, 384x384 | 39.480 | 38.477 | 5,819,909,015 bytes | 130,434,818 | PASS |
| Garage B8, 256x256 | 75.267 | 71.618 | 2,140,158,295 bytes | 54,077,433 | PASS |
| Home B40, 384x384 | 39.295 | 38.843 | 20,295,721,527 bytes | 538,187,900 | PASS |

The change is a capacity improvement, not an additional throughput claim:
Home B40 remains on the same roughly 39-image/s plateau as Home B8. Strict
72-view Garage fidelity against pinned gsplat passes every frame, the finite
Isaac/CUDA ladder passes, and Compute Sanitizer reports zero errors. The B40
process still used 22,485 MiB of the 23,034 MiB L4, so it does not establish
safe co-residency with a substantial RTX workload or an L40 result.

Nsight Systems measures 28.121 ms of kernel time per useful Home B8 view,
versus 28.529 ms before the memory change. Global radix sorting remains the
largest stage at 38.39 percent. Its scratch plus double-buffered key/value
arrays account for 95.82 percent of the B40 workspace and are the next
meaningful optimization target. Full commands, checksums, metrics, limitations,
and per-tensor memory evidence are in
[`benchmarks/emitted_memory_l4_results_2026-07-17.json`](benchmarks/emitted_memory_l4_results_2026-07-17.json).

## L4 phase-scratch alias follow-up

The compact renderer finishes depth-bucket accumulation before CUB radix
sorting begins on the same CUDA stream. Backing `depth_bucket_tau` with the
otherwise idle leading bytes of `sort_temp` therefore removes a separate dense
allocation without changing native math or launch order.

Against the exact-capacity emitted-memory baseline above:

| Case | GPU images/s | Workspace before | Workspace after | Exact saving | Verdict |
|---|---:|---:|---:|---:|---|
| Home B8, 384x384 | 39.870 | 5,819,909,015 | 5,668,914,071 | 150,994,944 bytes | PASS |
| Garage B8, 256x256 | 75.740 | 2,140,158,295 | 2,073,049,431 | 67,108,864 bytes | PASS |
| Home B40, 384x384 | 39.857 | 20,295,721,527 | 19,540,746,807 | 754,974,720 bytes | PASS |

Home B40 process memory fell from 22,485 to 21,765 MiB. All 240 Home views
pass output, motion, cache, CUDA-residency, and zero-overflow gates. Strict
72-view Garage fidelity against pinned gsplat passes every frame, the focused
CUDA/Kit/Fabric ladder passes, and Compute Sanitizer reports zero errors.

This is a memory result, not a speedup claim. The unprofiled runs moved by
about one percent in the favorable direction, while the fresh Nsight kernel
diagnostic moved by 0.27 percent in the unfavorable direction; those deltas
are classified as noise. Sorting remains the largest kernel stage at 38.33
percent.

The exact Home B48 probe fails at the next physical boundary after observing
575,647,560 intersections. Its 20,908,169,815-byte backend workspace request
is below the configured backend-only ceiling but cannot coexist with the
scene, outputs, and Isaac runtime on the 23,034 MiB L4. B40 therefore remains
the validated maximum from this matrix, and the result still does not prove
safe RTX co-residency or L40 behavior. Full evidence and checksums are in
[`benchmarks/phase_scratch_alias_l4_results_2026-07-17.json`](benchmarks/phase_scratch_alias_l4_results_2026-07-17.json).
