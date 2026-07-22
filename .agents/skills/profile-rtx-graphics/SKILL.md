---
name: profile-rtx-graphics
description: Profile and diagnose CUDA, Vulkan, Omniverse RTX, OVRTX, gsplat, and custom Gaussian-renderer workloads. Use when an agent needs Nsight Systems timelines, Nsight Compute kernel counters, Nsight Graphics frame captures, PyTorch profiler traces, Compute Sanitizer checks, GPU memory evidence, NVTX instrumentation, or a source-grounded graphics bottleneck report.
---

# Profile RTX Graphics

Collect reproducible evidence that distinguishes host submission, GPU kernels,
graphics-frame behavior, memory, correctness, and image fidelity.

## Required reference

Read [references/profiler-playbook.md](references/profiler-playbook.md) before
choosing a profiler or publishing a bottleneck claim.

## Workflow

1. Coordinate the node.
   - Apply `$provision-rtx-workstation`.
   - Confirm the CUDA extensions were compiled for the GPU this node actually
     has before profiling. The custom kernel and `gsplat` (a separate build the
     provisioner does not perform) must target the live compute capability
     (`nvidia-smi --query-gpu=compute_cap`; set `TORCH_CUDA_ARCH_LIST` to
     match). Profiling a binary built for the wrong arch measures nothing — it
     fails to load or falls back — so the arch is part of the frozen contract.
   - Do not profile on a shared GPU while another agent or workload is active.
   - Use a unique `PROFILE_ROOT` and never overwrite another run.

2. Prove baseline equivalence before computing a speedup.
   - Match scene checksum, camera bundle, projection mode, sorting mode,
     support/regularization, output channels, color space, near/far planes,
     semantic class topology, and warmup policy.
   - Verify OVRTX boundary conversions with an anisotropic, non-identity
     quaternion fixture. Identity or isotropic splats cannot expose a WXYZ/XYZW
     layout error.
   - Keep high-cardinality or interleaved semantic stress cases separate from
     the representative headline baseline.
   - If any baseline contract changes, invalidate and rerun both sides. Do not
     carry forward an old speedup.

3. Freeze the workload contract.
   - Record source and dependency commits, scene checksum, camera bundle,
     projection mode, renderer settings, outputs, warmup, iterations, and
     resolution.
   - Keep RGB, depth, alpha, and semantics enabled when the claim concerns the
     production sensor renderer.
   - Profile a short representative workload, not the complete acceptance
     matrix.

4. Add or verify NVTX ranges around meaningful stages and the measured range.
   - Keep shader/JIT warmup outside the measured range.
   - Synchronize only where required by the timing contract.
   - Do not hide normal allocations or output writes to make a profile look
     cleaner.

5. Start with Nsight Systems.
   - Use it for CUDA/Vulkan timelines, launch count, synchronization, allocator
     activity, CPU/GPU overlap, and NVTX ranges.
   - Prefer the repository wrappers:

     ```bash
     scripts/profile_custom_nsys.sh
     scripts/profile_home_scan_nsys.sh
     scripts/profile_nsys.sh
     ```

   - Retain `.nsys-rep`, exported CSV/SQLite, logs, and parsed JSON.

6. Use the next profiler only for a specific unanswered question.
   - Nsight Compute: occupancy, divergence, memory throughput, cache, and
     instruction metrics for selected kernels.
   - Nsight Graphics: Vulkan/DX12/OpenGL frame, draw/dispatch, resource,
     pipeline, and shader debugging.
   - PyTorch profiler: framework operations, allocations, concatenations,
     extension dispatch, and CPU/CUDA call relationships.
   - Compute Sanitizer: memory errors, races, initialization, and synchronization.
   - CUDA-GDB/GDB: crashes, device assertions, and hangs.

7. Preflight counter and display access.
   - Run `scripts/profile_custom_ncu.sh` or `scripts/profile_ncu.sh`; preserve
     their JSON preflight if the host blocks counters.
   - Do not claim Nsight Graphics worked solely because `ngfx` is installed.
     Interactive graphics capture normally requires a display-capable session.

8. Profile cold, warm, and invalidated cache states separately.
   - For an unchanged scene and camera, report what is reused and what still
     executes.
   - Mutate one camera or scene revision and prove the expected cache miss.
   - Do not present a repeated-static-frame cache profile as moving-camera
     performance.

9. Interpret evidence conservatively.
   - Separate GPU kernel time from host CUDA API time; they can overlap.
   - Separate profiler-instrumented timing from unprofiled benchmark timing.
   - For OVRTX/Vulkan, do not present CUDA event timing as end-to-end renderer
     timing when the renderer stream is not visible to those events.
   - Name missing counters, unsupported capture paths, or profiler overhead.

10. Tie every optimization hypothesis to both evidence and source.
   - Report top kernels/stages, call counts, total time, and share.
   - Locate the responsible source path before proposing a rewrite.
   - Reprofile after changes with the same workload contract.

11. Keep trace volume bounded.
   - Redirect verbose profiler output to a run log.
   - Report selected tables and parsed summaries rather than dumping raw
     reports or full benchmark JSON into the session.
   - Poll long captures by PID/status and a short log tail. Do not relaunch on
     an SSH timeout until the existing process is checked.

## Evidence contract

Keep:

- Command and environment.
- Raw profiler report.
- Exported tables.
- Machine-readable summary.
- Unprofiled control timing.
- GPU/driver/tool versions.
- Source and dependency commits.
- Known limitations.

Never publish inferred occupancy, cache behavior, or bandwidth as measured
facts when hardware counters were unavailable.
