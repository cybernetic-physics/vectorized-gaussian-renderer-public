# Contributor and Agent Guidance

This repository owns the vectorized Gaussian renderer, its C++/CUDA kernels,
the direct Isaac Sim service, the Kit extension, benchmarks, experiments, and
verification code. Keep project changes in this repository.

## Repository skills

Use the relevant repository-local skill before performing workstation,
profiling, or graphical-test work:

- [provision-rtx-workstation](.agents/skills/provision-rtx-workstation/SKILL.md)
  for Vast or local RTX-node coordination, software installation, source
  synchronization, environment setup, and provenance.
- [profile-rtx-graphics](.agents/skills/profile-rtx-graphics/SKILL.md) for
  Nsight Systems, Nsight Compute, Nsight Graphics, PyTorch profiler, NVTX,
  Compute Sanitizer, CUDA-GDB, and evidence-backed bottleneck analysis.
- [test-isaac-graphics](.agents/skills/test-isaac-graphics/SKILL.md) for finite
  headless Isaac/Kit tests, GPU output assertions, Fabric/physics integration,
  image regressions, fidelity, determinism, and stability validation.

Read each selected `SKILL.md` completely and open its linked reference before
acting. Reuse the root-level scripts named by the skill rather than creating a
parallel setup, profiling, or test harness.

## Shared RTX node policy

Never assume `vast-gsplat-isaac` or another RTX node is available. Before any
GPU workload, run a read-only occupancy check covering `nvidia-smi`, active
Isaac/OVRTX/Python/profiler processes, and existing `tmux` sessions. If an
unfamiliar workload or agent is present:

- Do not launch another render, benchmark, profiler, sanitizer, or build.
- Do not kill or suspend the existing process.
- Report the conflict and wait for explicit coordination.

Use a dedicated remote worktree and output directory per lane. Keep source
changes locally durable; do not let a remote checkout become the only copy.
Use `rsync --delete` only in a dedicated disposable directory whose ownership
is known.

Root `scripts/setup.sh` is not a general Vast bootstrap. It is intentionally
hard-coded to the independent verification worktree and may delete files there
during synchronization.

## Proven development workflow

**Compile the CUDA extensions for the GPU this node actually has, before
running anything that renders.** The custom renderer kernel and `gsplat` are
built from source against a specific compute capability. Provisioning
(`scripts/provision_lidar_node.sh`) clones `gsplat` and initializes its
submodules but does **not** build it, and `scripts/build_gsplat_main.sh`
defaults to `TORCH_CUDA_ARCH_LIST=8.9` (Ada / RTX 4090). On each fresh node,
detect the capability
(`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`, or
`torch.cuda.get_device_capability()`) and set `TORCH_CUDA_ARCH_LIST` to match
before building — otherwise the first render fails with `no kernel image is
available for execution on the device` and every downstream number is void.

The workflow that has produced the most useful evidence is:

1. Source `scripts/remote_env.sh` so Isaac's bundled Torch/NCCL, OVRTX, CUDA,
   gsplat, and profiler paths are ordered consistently.
2. Complete shader/JIT/extension warmup outside measured ranges.
3. Run bounded tests in increasing cost: unit tests, native CUDA smoke,
   `SimulationApp` smoke, Kit/Fabric tests, fidelity, profiler/sanitizer, then
   stability.
4. Use `try/finally` and always close `SimulationApp`; never leave an
   unbounded update loop in a test.
5. Validate all production outputs: RGB, depth, alpha, semantic ID, CUDA
   residency, finite values, foreground coverage, and zero capacity overflow.
6. Start profiling with Nsight Systems and NVTX. Use Nsight Compute only after
   its counter-access preflight, and use Nsight Graphics only in a
   display-capable capture session.
7. Preserve the raw profiler report, exported CSV/SQLite, parsed JSON, command,
   source/dependency commits, machine versions, and an unprofiled control.
8. Keep performance, image fidelity, Isaac integration, and sanitizer
   correctness as separate verdicts.

## Companion NVIDIA repositories

Use sibling checkouts so reference code remains available without vendoring it:

```text
workspace/
├── vectorized-gaussian-renderer/
├── ovrtx/
└── kit-app-template/
```

Clone them from the renderer repository with:

```bash
git clone --branch main --single-branch \
  https://github.com/NVIDIA-Omniverse/ovrtx.git \
  ../ovrtx

git clone --branch main --single-branch \
  https://github.com/NVIDIA-Omniverse/kit-app-template.git \
  ../kit-app-template
```

### OVRTX

OVRTX is the public NVIDIA SDK, schema, examples, and reference integration
used for RTX Gaussian-rendering and sensor baselines. A local checkout is
expected for work involving:

- OVRTX fidelity and performance comparisons.
- Gaussian USD schema or projection behavior.
- RTX camera, depth, segmentation, LiDAR, or radar conventions.
- OVRTX experiment and benchmark maintenance.

The repository does not contain the proprietary Omniverse RTX renderer source,
and its checkout is separate from the prebuilt OVRTX runtime package. Do not
represent OVRTX as the implementation of this project's custom renderer.

### Kit App Template

The Kit App Template is recommended for work involving:

- `exts/isaacsim.gaussian_renderer`.
- Kit extension lifecycle and packaging.
- Standalone Isaac Sim or Omniverse Kit application scaffolding.

It is not required for CUDA-kernel-only development or ordinary Python unit
tests.

Treat both NVIDIA repositories as read-only references unless a task
explicitly requests changes to them. Record the exact dependency commit in
benchmark or verification evidence when results depend on a particular
revision. Do not copy their source trees into this repository.
