# Profiler playbook

## Contents

- Tool selection
- Baseline-equivalence gate
- Repository entrypoints
- Cache-state profiling
- Graphics diagnostics
- Known-good profiling pattern
- Remote capture discipline
- Reporting pitfalls

## Tool selection

| Question | Primary tool | Supporting tool |
|---|---|---|
| Which stage dominates GPU time? | Nsight Systems | Parsed `nsys stats` |
| Are launches, syncs, or allocations excessive? | Nsight Systems | PyTorch profiler |
| Why is one CUDA kernel slow? | Nsight Compute | CUDA source and SASS |
| What happened in one Vulkan/RTX frame? | Nsight Graphics | Vulkan validation |
| Is framework glue allocating or concatenating? | PyTorch profiler | Nsight Systems |
| Is there illegal memory access or a race? | Compute Sanitizer | CUDA-GDB |
| Which process owns GPU memory? | `nvidia-smi`/NVML | Process sampler |
| Are outputs visually equivalent? | Fidelity harness | Saved images/tensors |

## Baseline-equivalence gate

Do not calculate or publish a speedup until one machine-readable manifest
confirms both renderers used:

- The same canonical scene and checksum.
- The same camera matrices, intrinsics, resolution, and active camera IDs.
- The same perspective/tangential projection selection.
- The same Gaussian activation, quaternion conversion, scale convention,
  support, regularization, and color space.
- RGB, depth, alpha, and semantic outputs.
- The same semantic class count and labeling topology.
- Comparable warmup and measurement scopes.

For OVRTX 0.3.0, explicitly test the authoring conversion from project WXYZ
quaternions to OVRTX `quatf[]` XYZW storage with rotated anisotropic splats.
An identity/isotropic control is insufficient.

Treat these as separate profiles:

- Representative spatially coherent semantics.
- Interleaved/high-cardinality semantic stress.
- Static-camera cache hits.
- Moving-camera or scene-revision cache misses.

A result from one category cannot headline another.

## Repository entrypoints

### Nsight Systems

Custom synthetic renderer:

```bash
source scripts/remote_env.sh
PROFILE_ROOT=outputs/profiles/<unique-id> \
  scripts/profile_custom_nsys.sh <unique-id>
```

Full Home Scan:

```bash
PROFILE_ROOT=outputs/profiles/<unique-id> \
  scripts/profile_home_scan_nsys.sh <unique-id>
```

Stock gsplat:

```bash
PROFILE_ROOT=outputs/profiles/<unique-id> \
  scripts/profile_nsys.sh <unique-id>
```

The wrappers capture CUDA, NVTX, and OS runtime activity, export CSV summaries,
and run `scripts/parse_nsys_stats.py`.

For a standalone Vulkan target, add Vulkan tracing when supported by the
installed Nsight Systems version:

```bash
nsys profile \
  --trace=cuda,nvtx,vulkan,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=outputs/profiles/<id>/<id> \
  <application command>
```

### Nsight Compute

Run the preflight wrapper first:

```bash
scripts/profile_custom_ncu.sh <unique-id>
```

The wrapper records `/proc/driver/nvidia/params` and process capabilities. A
host with `RmProfilingAdminOnly: 1` and no `cap_sys_admin` cannot provide
hardware counters. Move the run to a counter-enabled node instead of weakening
the evidence requirement.

When counters are available, profile one or a small set of kernels and one
iteration. Full-set collection has large overhead.

### Nsight Graphics

Use Nsight Graphics when the question is about:

- Vulkan frame structure.
- RTX dispatches and acceleration-structure work.
- Pipeline, shader, descriptor, synchronization, or resource state.
- A frame-specific rendering artifact.

Requirements:

- A supported NVIDIA GPU and driver.
- A display-capable X11/Wayland or remote-desktop session for interactive use.
- A short, capturable frame and stable reproduction.
- First-use shader compilation completed before capture.

`ngfx` existing in a headless container proves installation only. It does not
prove that frame capture is operational.

### PyTorch profiler

The profile entrypoints accept `--torch-profiler`:

```bash
/isaac-sim/python.sh benchmarks/profile_custom.py \
  --config-id <id> \
  --output-dir outputs/profiles/<id> \
  --scene synthetic-medium \
  --batch 64 \
  --width 128 \
  --height 128 \
  --warmup 8 \
  --iterations 20 \
  --torch-profiler
```

Use the trace for Python/ATen/extension boundaries and allocation behavior. Do
not substitute it for Nsight kernel counters.

### Compute Sanitizer

Use a narrow deterministic smoke:

```bash
compute-sanitizer --tool memcheck \
  /isaac-sim/python.sh scripts/custom_backend_smoke.py
```

Use `racecheck`, `initcheck`, or `synccheck` only when the suspected failure
matches the tool. Sanitizer timing is not performance evidence.

## Cache-state profiling

For a cache-aware renderer, capture at least:

1. Cold first render.
2. Warm unchanged scene/camera render.
3. Camera mutation.
4. Scene revision or explicit invalidation.

Use NVTX ranges and call counts to prove which stages were skipped. Projection,
binning, sorting, range construction, and compositing have different coherence
requirements; never call the cache correct because only output images repeat.
Keep a deterministic cache-on/cache-off output comparison separate from the
non-deterministic performance run.

## Graphics diagnostics

Useful supporting tools:

- `vulkaninfo` for device, extension, and layer availability.
- `VK_LAYER_KHRONOS_validation` for API misuse in a debuggable test.
- `nvidia-smi`, `nvidia-smi pmon`, and NVML for process memory/utilization.
- `cuda-gdb`, `gdb`, `strace`, and core dumps for crashes and hangs.
- `ffmpeg` or ImageMagick for assembling visual artifacts, not for metric
  computation.

Validation layers and capture tools can change timing and behavior. Run them
outside performance measurements.

## Known-good profiling pattern

1. Run the workload unprofiled and save timing.
2. Warm shaders, extensions, kernels, and allocators.
3. Capture one short representative process with NVTX labels.
4. Export raw and summarized evidence.
5. Rank stages by total time and calls.
6. Inspect the source responsible for the top stage.
7. Form one testable optimization hypothesis.
8. Change one variable.
9. Rerun correctness/fidelity tests.
10. Reprofile the same workload and compare.

This pattern previously identified rasterization, sorting, intersection
construction, synchronization, and allocation behavior without conflating
them.

## Remote capture discipline

- Run a short unprofiled control first.
- Warm shader/JIT/native-extension caches before capture.
- Use one run ID and one output directory.
- Redirect console output to a log and retain raw `.nsys-rep`/`.ncu-rep`
  artifacts.
- Export only the tables needed to answer the current question.
- On SSH timeout, inspect the existing process and report before retrying.
- Do not run multiple profilers or benchmark jobs concurrently on one GPU.
- Treat `ERR_NVGPUCTRPERM` as a host-capability result, not a renderer failure.
- Treat installed `ngfx` without a display/capture proof as installation only.

## Reporting pitfalls

- Do not add host API time to GPU time as if they were serialized.
- Do not compare a profiled run directly to an unprofiled competitor.
- Do not cite a synthetic gsplat profile as an OVRTX acceptance result.
- Do not cite an interleaved 1,024-field semantic stress result as the normal
  two-class or spatial-semantic baseline.
- Do not generalize an OVRTX projection model from isotropic centered splats.
- Do not profile cached images or omit required output channels.
- Do not compare static cache-hit timing to moving-camera timing without
  labeling both.
- Do not hide warmup policy, sample count, or synchronization points.
- Do not claim graphics-frame capture from an installation-only check.
- Do not omit blocked counter access; retain the preflight JSON.
