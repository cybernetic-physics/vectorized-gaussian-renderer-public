# GPU profiling report

Date: 2026-07-16

Owner branch/worktree: `codex/profiling` in
`/workspace/agent-worktrees/profiling` on `vast-gsplat-isaac`.

## Current custom Home Scan profile

Evidence:
`outputs/profiles/home-scan-projection-cache/`

Captured workload: full Home Scan LOD0, B1, 128x128, tile size 1, 1,024 depth
buckets, group size 32, tight safe depth range, projection cache enabled,
1 warmup plus 20 measured renders.

The benchmark measured 4.707 ms under Nsight instrumentation and 4.650 ms
without profiler overhead. Kernel call counts establish that this is not an
image cache:

| Kernel/stage | Calls | Total GPU time |
|---|---:|---:|
| grouped depth tau | 704 | 68.512 ms |
| pixel binning | 22 | 20.621 ms |
| fused compositor | 22 | 4.532 ms |
| cutoff advancement | 704 | 3.647 ms |
| CUB sorting | per-render sequence | 2.331 ms |
| projection | 1 | 1.986 ms |
| covariance precompute | 1 | 1.163 ms |
| depth scatter | 1 | 0.614 ms |
| tile-range discovery | 22 | 0.134 ms |
| depth prefix | 1 | 0.032 ms |

The remaining steady-state bottleneck is grouped optical-thickness
accumulation, followed by pixel binning. Projection is absent from coherent
measured frames, while all output-producing stages remain active.

## Scope

The historical sections below profile the pinned `gsplat` control on synthetic
shared-scene matrix points. They remain the source-grounded motivation for the
custom architecture. The custom Home Scan profile above is the current
integrated result.

The profiled output contract was RGB, depth, alpha, and one semantic-ID scalar
proxy via `extra_signals`. That proxy exercises an extra rasterized output
channel but is not a semantic correctness implementation.

## Tool snapshot

Evidence:
`outputs/profiles/environment/nvidia_tool_versions.json`

- GPU: NVIDIA GeForce RTX 4090, driver 590.48.01, 24,564 MiB, compute 8.9.
- CUDA compiler: 12.8, V12.8.93.
- Nsight Systems: 2024.6.2.225-246235244400v0.
- Nsight Compute CLI: 2025.1.1.0 build 35528883.
- Nsight Graphics package: `nsight-graphics-for-linux-2026.2.0.0`
  installed; `ngfx --version` exits nonzero in this headless container.
- Nsight Compute counters are host-blocked. Evidence:
  `outputs/profiles/gsplat-synth-shared-ncu-preflight/gsplat-synth-shared-ncu-preflight_ncu_preflight.json`
  records `RmProfilingAdminOnly: 1` and no `cap_sys_admin`.

## Reproducible profiling harness

- Matrix/profiler runner: `benchmarks/profile_gsplat.py`
- Matrix shell wrapper: `scripts/profile_matrix.sh`
- Nsight Systems wrapper: `scripts/profile_nsys.sh`
- Nsight Systems CSV parser: `scripts/parse_nsys_stats.py`
- Nsight Compute preflight/wrapper: `scripts/profile_ncu.sh`

All scripts derive `PROJECT_ROOT` from the active worktree, so they run from
`/workspace/agent-worktrees/profiling` rather than the older remote checkout.

## Matrix timings

Evidence:
`outputs/profiles/matrix/profile_runs.csv`

| Config | Mean GPU ms | p95 ms | Images/s | MP/s | Peak alloc GiB | Visible pairs | Tile intersections |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 N100k 128x128 | 0.679 | 0.796 | 1,389.6 | 22.8 | 0.017 | 50,817 | 77,882 |
| C8 N100k 128x128 | 1.060 | 1.243 | 7,190.2 | 117.8 | 0.103 | 429,680 | 660,439 |
| C64 N100k 128x128 | 4.996 | 5.032 | 12,708.4 | 208.2 | 0.761 | 3,466,382 | 5,329,600 |
| C32 N100k 256x256 | 2.798 | 2.894 | 11,255.7 | 737.7 | 0.496 | 1,704,596 | 3,255,861 |

## Nsight Systems bottlenecks

Primary report:
`outputs/profiles/gsplat-synth-shared-c64-n100k-128-nsys/nsys/gsplat-synth-shared-c64-n100k-128-nsys.nsys-rep`

Parsed summary:
`outputs/profiles/gsplat-synth-shared-c64-n100k-128-nsys/gsplat-synth-shared-c64-n100k-128-nsys_nsys_summary.json`

Captured workload: C64, N100k, 128x128, 8 warmup + 20 measured iterations.
The script process reported 5.261 ms mean GPU event time and 0.761 GiB peak
allocated memory.

Kernel-time distribution across captured render launches:

| Stage | Kernel time ms | Share | Calls |
|---|---:|---:|---:|
| Rasterization | 63.251 | 47.5% | 28 |
| Sorting | 22.792 | 17.1% | 336 |
| Tile intersection | 13.319 | 10.0% | 56 |
| Other kernels | 12.301 | 9.2% | 211 |
| Concatenate/fill | 11.531 | 8.7% | 99 |
| Projection | 8.595 | 6.5% | 56 |
| Offset encoding | 1.346 | 1.0% | 59 |

CUDA API totals over the captured process:

| API bucket | Time ms | Calls |
|---|---:|---:|
| Kernel launch | 215.312 | 845 |
| Synchronization | 106.527 | 88 |
| Allocator | 8.500 | 29 |
| Other CUDA API | 8.029 | 534 |

The launch/sync API time is host-side overhead and overlaps with GPU execution;
it is not additive with kernel time. It is still material evidence that a custom
renderer should reduce launch count and avoid per-render host synchronization.

## PyTorch profiler evidence

Evidence:
`outputs/profiles/torchprof/gsplat-synth-shared-c64-n100k-128-torchprof.json`
and
`outputs/profiles/torchprof/gsplat-synth-shared-c64-n100k-128-torchprof_torch_trace.json`

For four profiler iterations, PyTorch recorded:

- `gsplat::rasterization_3dgs`: 4 calls, 19.341 ms CUDA total, 1.316 GB CUDA
  memory usage attributed across events.
- `gsplat::rasterize_to_pixels_3dgs`: 4 calls, 9.028 ms CUDA total.
- `gsplat::intersect_tile`: 4 calls, 5.261 ms CUDA total.
- CUB radix sort onesweep kernels: 24 calls, 2.925 ms CUDA total.
- `cudaLaunchKernel`: 116 calls.
- `cudaStreamSynchronize`: 8 calls, 11.799 ms CPU time.
- `aten::cat`: 12 calls, 1.642 ms CUDA total and 566 MB CUDA memory usage.
- `aten::empty`: 72 calls and 1.152 GB CUDA memory usage attributed across
  profiler events.

## Source locations tied to bottlenecks

These paths are in `/workspace/src/gsplat` at commit
`77ab983ffe43420b2131669cb35776b883ca4c3c`.

- Python entrypoint and metadata returned to the harness:
  `gsplat/rendering.py:575` calls `torch.ops.gsplat.rasterization_3dgs`;
  `gsplat/rendering.py:657` exposes radii, tile counts, intersections, and
  output metadata.
- Projection: `gsplat/cuda/_wrapper.py:658` wraps `fully_fused_projection`;
  Nsight sees `projection_ewa_3dgs_packed_fwd_kernel`.
- Tile intersection: `gsplat/cuda/_wrapper.py:1035` wraps `isect_tiles`;
  `gsplat/cuda/csrc/IntersectTile.cu:214` is the intersect kernel and
  `gsplat/cuda/csrc/IntersectTile.cu:770` starts the count/scan/emit path.
- Sorting: `gsplat/cuda/csrc/IntersectTile.cu:899` sorts intersection pairs;
  `gsplat/cuda/csrc/IntersectTile.cu:1076` contains the CUB radix-sort helper.
- Raster: `gsplat/cuda/_wrapper.py:1336` wraps `rasterize_to_pixels`;
  `gsplat/cuda/csrc/Rasterization.cpp:340` launches the forward raster kernel
  and `gsplat/cuda/csrc/Rasterization.cpp:589` is the public dispatcher path.

## Optimization hypotheses

1. Fuse or specialize raster compositing for the final required outputs.
   Rasterization is the largest measured GPU stage at 47.5% of kernel time.
   The custom renderer should write RGB, depth, alpha, and semantic outputs in
   one purpose-built pass and avoid generic channel handling where possible.

2. Reduce or replace global intersection sorting.
   Sorting is 17.1% of kernel time and launches 336 captured CUB kernels.
   A custom renderer should evaluate per-tile/local ordering, depth bins, or
   scene/camera coherence before committing to full global pair sorting.

3. Rework tile intersection and scan/emit into a lower-launch pipeline.
   Tile intersection plus offsets is 11.0% of kernel time, and the API summary
   shows high launch/sync counts. A fused count/emit path or persistent
   workspace can reduce host-visible launches and allocator churn.

4. Eliminate intermediate concatenation for depth and semantic channels.
   Concatenate/fill is 8.7% of kernel time; PyTorch attributes 566 MB of CUDA
   memory usage to `aten::cat` in four profiler iterations. The custom renderer
   should pass structured output pointers or channel descriptors instead of
   materializing concatenated feature tensors.

5. Preallocate reusable workspace and make allocator behavior explicit.
   Nsight Systems records 29 allocator API calls and 8.5 ms host API time in
   the captured process; PyTorch records many `aten::empty` events. The final
   renderer should expose deterministic workspace sizing and reuse buffers
   across steady-state frames.

6. Consider CUDA Graph capture only after dynamic allocation and sync points
   are removed. The current captured process has 845 kernel launches and 88
   synchronization API calls. Graph capture can reduce launch overhead, but only
   after the renderer has stable memory addresses, fixed shapes, and no hidden
   host synchronizations in the steady-state path.

## Limitations

- No Nsight Compute occupancy or bandwidth counters were collected because the
  host blocks performance counters. The report intentionally contains no
  invented occupancy, SM utilization, L2, or DRAM metrics.
- The semantic output is a scalar extra-signal proxy for profiling cost only.
  Correct semantic-ID compositing and fidelity remain separate work.
- These are synthetic shared-scene `gsplat` measurements. They are useful for
  profiling direction but are not a fair RTX/OVRTX baseline and not an
  acceptance result.
