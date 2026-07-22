# Historical Freiza RTX 3090 Custom versus FlashGS diagnostics

> **Status: superseded and invalid as matched evidence.** The Custom and
> FlashGS candidates in this run used a 3.0-sigma support cutoff, while pinned
> gsplat commit `77ab983` used `GAUSSIAN_EXTEND=3.33f`. This violates the
> identical-equation requirement. The timings and memory observations below
> are retained only as historical diagnostics and must not support a renderer
> winner, fidelity, or replacement claim under the corrected 3.33-sigma
> contract.

## Verdict

This was a completed historical run on Freiza GPU 0, an RTX 3090, but it is not
a valid run of the corrected frozen dynamic Home Scan contract and cannot
establish the current L4 headline.

`CustomCudaBackend` completed batches 1 through 512 and was faster than the
FlashGS-derived matched-contract port in every directly comparable row. The geometric-mean
raw speedup over those seven batches was 80.211x for full sensor output and
78.124x for RGB-only output. Custom did not complete batch 1024: both output
contracts independently failed during bounded workspace installation because
42,673,906,888 bytes (39.743 GiB) were required on the 23.56 GiB GPU. FlashGS
completed batch 1024.

The raw timing observations are not a replacement claim. The candidate-to-
oracle image-formation equation is mismatched, the complete requested
matrix is not available for Custom, the full-sensor rows through batch 512 do
not jointly pass the pinned-gsplat semantic fidelity gate, and the strict
steady-state fairness gate detects NVML process-memory growth in multiple
measured runs. A bounded B64 Nsight control found zero CUDA allocator calls and
zero Torch allocation/reservation growth in either renderer's measured range,
but FlashGS's 12 MiB NVML delta remained, so the stricter gate stays closed.

## Historical workload and provenance

| Item | Value |
|---|---|
| Machine | Freiza GPU 0 |
| GPU | NVIDIA GeForce RTX 3090, UUID `GPU-397e1e47-c44e-debe-5e89-fe81ef1b3647`, SM 8.6 |
| Driver / CUDA / Torch | 580.119.02 / 12.8 / 2.7.1+cu128 |
| Scene | Home Scan LOD0, 21,497,908 Gaussians |
| Scene SHA-256 | `29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267` |
| Resolution | 128x128 |
| Batches | 1, 8, 32, 64, 128, 256, 512, 1024 |
| Cameras | Independently changing on every measured frame; zero unchanged camera/frame pairs |
| Timing | 8 warmup frames and 100 synchronized measured frames per successful run |
| Caches | Rendered-frame and projection caches disabled |
| Benchmark source used by renderer runs | B1-B32: `5fb66eba0e7c6b7d8f16c8e3195d27b998ed266c`; B64-B1024: `bf0b6333e66be89b59387bb76e5ef9c8eae93437`; every run records an empty source diff |
| Durable implementation tip | `de2a9cb` on `codex/flashgs-dynamic-20260718`; this report is committed later on the same branch |
| FlashGS | `cdfc4e4002318423eda356eed02df8e01fa32cb6` |
| gsplat oracle | `77ab983ffe43420b2131669cb35776b883ca4c3c` |
| gsplat compatibility patch SHA-256 | `ea30120f728d5c728e082fec686f15056de5813a4e8fcc2c6b4e68aa324ca36d` |

Both candidate backends used the same canonical Gaussian tensors, camera
contracts, active PyTorch CUDA stream, float32 precision, black background,
3.0-sigma support, alpha and transmittance thresholds, front-to-back
compositing, expected-depth definition, and strongest-`T*alpha` semantic rule.
The pinned gsplat oracle, however, used its hard-coded 3.33-sigma support.
Therefore the fidelity results and the overall matched interpretation are
invalid. FlashGS kept its native per-camera operation and invoked it once per
camera; that sequential scheduling cost remains a useful historical
diagnostic.

The source-commit change after B32 does not change executable rendering or
timing code. The complete diff from `5fb66eb` to `bf0b633` is confined to the
matrix driver, result summarizer, and summarizer tests; the renderer backends
and `benchmarks/run_flashgs_matched.py` are identical. It adds fail-closed
fidelity interpretation after the per-renderer artifacts are produced. The
commit split is retained here rather than normalized away in provenance.

## Historical full-sensor diagnostic

The memory columns are the maximum of the before/after NVML process samples,
reported in decimal GB. `Fidelity pass` requires both renderers to pass the
pinned-gsplat full-output thresholds for RGB, alpha, expected depth, and
semantic ID.

| Batch | Custom ms | FlashGS-derived ms | Speedup | Custom GB | FlashGS GB | Fidelity pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 3.226 | 426.196 | 132.113x | 2.567 | 3.454 | FAIL |
| 8 | 22.110 | 1,608.089 | 72.732x | 2.672 | 3.456 | FAIL |
| 32 | 86.633 | 5,911.318 | 68.234x | 3.964 | 3.506 | FAIL |
| 64 | 174.295 | 13,433.734 | 77.075x | 5.295 | 3.534 | FAIL |
| 128 | 343.362 | 26,167.595 | 76.210x | 8.210 | 3.618 | FAIL |
| 256 | 688.242 | 52,361.450 | 76.080x | 13.269 | 3.764 | FAIL |
| 512 | 1,396.817 | 101,833.899 | 72.904x | 24.182 | 4.083 | FAIL |
| 1024 | OOM | 204,110.402 | - | - | 4.664 | N/A; FlashGS PASS |

The raw comparable-row speedup range is 68.234x to 132.113x. The batch-1024
speedup is undefined; assigning it a projected Custom latency would conceal the
actual hardware-capacity result.

At batches 1 through 512, every RGB, alpha, and depth threshold passed. The
combined full-sensor gate failed on semantics: at batch 1 Custom passed with
0.999512 minimum semantic agreement while FlashGS obtained 0.998901; at
batches 8 through 512 both candidates fell below the 0.999 threshold on at
least one sampled view. FlashGS's batch-1024 full output passed all thresholds,
including 0.999512 minimum semantic agreement and 21 mismatched sampled pixels.

## Historical RGB-only diagnostic

`Fidelity pass` here evaluates the requested conventional RGB output only.
Alpha, depth, and semantic values in the RGB fidelity reports are oracle
placeholders and are not candidate measurements.

| Batch | Custom ms | FlashGS-derived ms | Speedup | Custom GB | FlashGS GB | RGB fidelity pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 3.214 | 411.337 | 128.001x | 2.565 | 3.454 | PASS |
| 8 | 21.969 | 1,565.272 | 71.249x | 2.670 | 3.454 | PASS |
| 32 | 86.371 | 5,746.015 | 66.527x | 3.962 | 3.483 | PASS |
| 64 | 173.258 | 13,046.596 | 75.302x | 5.291 | 3.490 | PASS |
| 128 | 341.402 | 25,307.171 | 74.127x | 7.950 | 3.538 | PASS |
| 256 | 684.634 | 50,816.546 | 74.224x | 13.269 | 3.603 | PASS |
| 512 | 1,389.088 | 98,148.883 | 70.657x | 23.994 | 3.741 | PASS |
| 1024 | OOM | 198,157.307 | - | - | 4.060 | N/A; FlashGS PASS |

The raw comparable-row speedup range is 66.527x to 128.001x. Both candidates
pass the RGB oracle gate at every completed paired batch. FlashGS also passes
at batch 1024; Custom produced no image at that batch.

## Batch-1024 hardware outcome

The full and RGB Custom probes independently reached the same bounded capacity
request and failed before the measured loop:

- exception: `RendererCapacityError`, caused by `torch.OutOfMemoryError`;
- phase: full-trajectory capacity calibration / workspace installation;
- required batch-global workspace: 42,673,906,888 bytes (42.674 GB,
  39.743 GiB);
- immediate failed allocation reported by Torch: 10.50 GiB; and
- timing, peak steady-state memory, and fidelity: unavailable because no
  measured frame rendered.

This is an observed hardware limit, not an estimate. FlashGS fits because it
reuses a per-camera workspace instead of installing Custom's batch-global
workspace. Its batch-1024 full run processed a mean 4.013 billion generated
intersections and 3.948 billion visible-Gaussian records per frame with zero
reported capacity overflow. It delivered 5.017 images/s for full output and
5.168 images/s for RGB-only output.

## Strict fairness gate

All successful rows record zero Torch allocation growth, no CPU output copy in
the measured loop, fixed preflighted capacities, and zero capacity overflow.
The source summary nevertheless fails closed when NVML process memory changes
between the before/after samples:

| Contract | Batch | Measured NVML growth |
|---|---:|---|
| Full | 32 | Custom +2 MiB; FlashGS +2 MiB |
| Full | 64 | Custom +4 MiB; FlashGS +26 MiB |
| Full | 128 | FlashGS +64 MiB |
| Full | 256 | FlashGS +128 MiB |
| Full | 512 | Custom +52 MiB; FlashGS +302 MiB |
| RGB | 32 | FlashGS +20 MiB |
| RGB | 64 | FlashGS +12 MiB |
| RGB | 128 | FlashGS +44 MiB |
| RGB | 256 | FlashGS +84 MiB |
| RGB | 512 | Custom +120 MiB; FlashGS +168 MiB |

Batches 1 and 8 pass the strict matched-run fairness checks. The B64 profiler
control shows that the remaining FlashGS delta is not accompanied by an
application-visible allocation call in the captured range, but that does not
prove the driver-memory change is irrelevant. Therefore all-matrix fairness
and replacement claims remain ineligible.

## Representative Nsight Systems control

B64 RGB was used because it was a paired, RGB-fidelity-passing workload that
both backends could run. B1024 could not be paired in this superseded run.
Profiler
timing is diagnostic and is not substituted for the 100-frame unprofiled
numbers above.

| Renderer | Unprofiled ms | Profile-control ms | Kernel ms/frame | Launches / 3 frames | Allocator calls | NVML growth | Dominant stage |
|---|---:|---:|---:|---:|---:|---:|---|
| Custom | 173.258 | 171.342 | 169.518 | 45 | 0 | 0 MiB | projection, 66.416% |
| FlashGS | 13,046.596 | 11,943.334 | 11,934.739 | 1,920 | 0 | 12 MiB | raster, 98.424% |

The trace directly exposes the scheduling difference: 15 Custom launches per
captured frame versus 640 FlashGS launches per captured frame. Custom submitted
the unprofiled B64 batch in 0.184 ms; sequential FlashGS submission took
2,154.828 ms. This supports the narrow explanation for this workload, not a
general claim about FlashGS on its paper's ordinary single-camera scenes.

## Evidence locations and integrity

The compact local mirror is:

`outputs/flashgs-matched/freiza-rtx3090-matrix-v3-canonical-per-camera`

The checksum-verified transferred subset contains 637 files and 37,458,340
bytes. A post-transfer rsync checksum pass found no differing files. Three
local derived files (the regenerated B1-B512 JSON/Markdown summary and the
machine-readable B1024 hardware outcome) bring the current local root to 640
files and 37,493,985 bytes. It includes all run JSON, logs, command records,
camera contracts, pinned-gsplat oracles, fidelity reports/images, imported
`.nsys-rep`, SQLite, CSV, and parsed profiler summaries. Large per-frame output
captures and `.qdstrm` streams remain preserved under the remote root:

`/home/freiza/benchmark-results/vgr-flashgs-dynamic-20260719/matrix-v3-canonical-per-camera`

These generated artifacts are intentionally excluded from Git. Their paths
resolve in the preserved benchmark worktree beneath the local mirror above:

- `summary-through-b512.json`
- `logs/custom-full-b1024.log`, SHA-256
  `0f17e5f1fd496d9e65198c2215fd102ce5858afedc4748980ff24a2f9b061bde`
- `logs/custom-rgb-b1024-hardware-probe-proven-env.log`, SHA-256
  `74570e804fa351cd4748c6282827ffa5a469b390e3155c0ca2b479fe51f5e8fa`
- `runs/flashgs/full/b1024.json`
- `runs/flashgs/rgb/b1024.json`
- `oracle/b1024.manifest.json`
- `profiles/rtx3090-rgb-b64-custom-20260721-v2/profile-summary.json`
- `profiles/rtx3090-rgb-b64-flashgs-20260721-v2/profile-summary.json`

The valid raw qdstrm files remain on Freiza with these SHA-256 values:

- Custom B64: `f2de03c057cd458fbc884c326b72208b84782c9a50826ce710e545bbb235d5a4`
  (2,899,050,615 bytes).
- FlashGS B64: `4c495b49a3dcf4b526997b3206ae38d997f584689757591347e0cf9325984b3f`
  (2,899,047,982 bytes).

The earlier failed Custom profiler capture is retained separately and is not
used as valid evidence.

## Historical interpretation only

Under the superseded 3.0-sigma candidate configuration, Custom was dramatically
faster at every batch that both methods completed, while FlashGS was the only
one that completed batch 1024. Those observations motivated the P128 memory
work, but they do not establish performance or fidelity under the corrected
3.33-sigma matched contract. They support neither a renderer-replacement claim
nor the corrected L4 headline claim.

These approximately 78x to 80x geometric-mean raw results must not be described
as a reproduction or refutation of FlashGS's paper-level 8.7x average. That
paper result concerns different ordinary 3DGS workloads and baselines; this
experiment asks whether unbatched FlashGS can replace a native-batched robotics
renderer under one shared `RendererService` contract.
