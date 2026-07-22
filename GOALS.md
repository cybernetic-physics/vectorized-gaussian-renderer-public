# Project goals

This project builds and independently verifies a custom, massively vectorized
Gaussian-splat renderer integrated directly into headless Isaac Sim and
Omniverse Kit on one NVIDIA RTX 4090.

Isaac Lab is not a build, runtime, configuration, launch, or API dependency.
It may consume the renderer through an optional adapter after the direct Isaac
Sim integration passes, but it does not own the core architecture.

## Acceptance criteria

The final result is successful only when all of the following are measured on
the same RTX 4090, scene, Gaussian attributes, camera batch, resolutions,
outputs, warmup, measured frame range, and fidelity contract:

1. Visual fidelity is equivalent to the supported Omniverse RTX Gaussian
   reconstruction/compositing contract.
2. Aggregate steady-state throughput is at least 5.0 times the RTX baseline.
3. Peak GPU memory is at most 80 percent of the RTX baseline.
4. RGB PSNR is at least 40 dB.
5. RGB SSIM is at least 0.995.
6. LPIPS is at most 0.01.
7. Alpha mean absolute error is at most 0.005.
8. Valid-pixel depth relative error is at most 1 percent.
9. Semantic-ID agreement is at least 99.9 percent.
10. Results reproduce within 5 percent across at least three runs.
11. Single-camera performance is reported alongside vectorized throughput.
12. The implementation returns GPU-resident RGB, depth, alpha, and semantic
    tensors without CPU copies in the steady-state loop.
13. Performance-critical projection, visibility, scheduling/binning, ordering,
    rasterization, and compositing are implemented by project-owned GPU
    kernels.
14. The renderer runs through a finite headless `SimulationApp` workflow and a
    loadable Kit extension without a mandatory viewport or render product.

Passing a synthetic microbenchmark, forwarding work to RTX, or outperforming
unmodified `gsplat` is not the acceptance condition.

## Independent trajectory verdicts

Acceptance is reported as independent, fail-closed verdicts for contract and
provenance, EWA versus pinned `gsplat` fidelity, OVRTX perspective fidelity,
static coherent performance, dynamic trajectory performance, Isaac/Fabric
integration, cache correctness, artifact integrity, stability,
reproducibility, and memory. No arithmetic combination or opaque score may
hide a failed axis. Static coherent projection-cache hits are publishable only
as static performance and cannot satisfy the dynamic renderer gate.

## Required workload support

- One shared Gaussian scene viewed by many cameras.
- Multiple Gaussian scenes selected by batched scene IDs.
- Per-environment transforms without duplicating static scene data.
- Batch sizes 1, 8, 32, 64, 128, 256, and the largest fitting batch.
- Resolutions 128x128, 256x256, and 512x512.
- Small, medium, public-real-world, and Home Scan scenes.
- Deterministic mode and selected-environment rendering.
- Optional render-every-N-simulation-steps scheduling.

## Benchmark integrity

Generated frames, temporal interpolation, cached frames, pre-rendered camera
paths, missing outputs, asynchronous timing without synchronization, unequal
precision, unequal preprocessing, and unequal camera/background/scene data are
forbidden.

## Current state

The repository contains a project-owned vectorized C++/CUDA renderer, a direct
Isaac Sim Kit extension, GPU Fabric camera ingestion, deterministic and
multi-scene support, real PLY ingestion, reproducible matrices, profiling
harnesses, and independent evidence auditing. Stock `gsplat` and OVRTX are
retained only as controls.

Performance, memory, integration, and strict `gsplat` compositing parity have
measured evidence. Overall acceptance remains open until the final OVRTX
fidelity thresholds and independent audit pass; unmet criteria must remain
explicit FAIL results.
