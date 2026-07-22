---
name: provision-rtx-workstation
description: Provision, inspect, and coordinate a Linux NVIDIA RTX workstation for this Isaac Sim Gaussian renderer. Use when an agent needs to rent or connect to a Vast node, prepare an existing local GPU node, install or verify Isaac Sim, CUDA, OVRTX, gsplat, Kit references, NVIDIA profilers, build tools, safely synchronize this repository, or run bounded work on a shared GPU machine.
---

# Provision an RTX Workstation

Prepare a reproducible RTX development node without disrupting another agent
or making the remote copy the only copy of development work.

## Required reference

Read [references/workstation-toolchain.md](references/workstation-toolchain.md)
before installing software or choosing an image. It contains the known-good
versions, package groups, directory layout, and validation commands.

Use the bundled helpers instead of rebuilding remote orchestration in ad hoc
SSH one-liners:

- `scripts/inspect-rtx-node.sh`: read-only host, GPU, process, path, tool, and
  Isaac-runtime preflight.
- `scripts/run-rtx-locked.sh`: run one finite GPU command under the cooperative
  project lock after the node has been confirmed free.

## Workflow

1. Establish authority.
   - Treat renting a machine, changing a running instance, installing system
     packages, and stopping processes as state-changing actions.
   - Require explicit user authorization for those actions in the current task.
   - Treat a later instruction to stop using or sharing the node as immediate
     revocation of workload authority, even if an earlier instruction said to
     leave the machine running.
   - Never expose values from `.env`, API tokens, SSH private keys, or Vast
     credentials in logs or commits.

2. Check whether a shared node is available before launching any workload.

   ```bash
   .agents/skills/provision-rtx-workstation/scripts/inspect-rtx-node.sh \
     "$RTX_HOST" "$REMOTE_ROOT"
   ```

   If an unfamiliar GPU workload, profiler, or agent session is present, do not
   start another run. Report the conflict and wait for coordination. Never kill
   another process merely to free the GPU.

   Assign exactly one GPU executor per node. Other agents may prepare local
   source, tests, and commands, but they must not independently submit work to
   the same GPU.

   If parallel local worktrees are used, launch and verify one worker before
   fan-out. Confirm the local Codex CLI accepts the selected model and the
   worker produces a heartbeat or file change. Do not hand every worker direct
   access to the shared GPU; queue execution through the single GPU executor.

3. Use an isolated worktree and output root.
   - Keep the local repository as the durable source of truth.
   - Use a dedicated remote path such as
     `/workspace/agent-worktrees/<lane>`.
   - Do not develop in a shared `/workspace/vectorized-gaussian-renderer`
     checkout when multiple agents are active.
   - Sync source back locally or commit locally before ending the task.

4. Synchronize conservatively.
   - Exclude `.git`, generated outputs, datasets, environments, and caches when
     copying source.
   - Create the destination directory first and run a dry-run or
     `--itemize-changes` pass before the first real transfer.
   - Use portable macOS/Linux rsync flags. Do not assume the local rsync
     supports `--info=progress2`.
   - Use `rsync --delete` only inside a dedicated disposable worktree whose
     ownership is known.
   - Do not run root `scripts/setup.sh` as a generic bootstrap. It is hard-coded
     for the independent verification worktree and intentionally uses
     destructive synchronization there.

5. Provision in dependency order.
   - NVIDIA driver and GPU access.
   - CUDA toolkit and development tools.
   - Isaac Sim and its bundled Python/Torch runtime.
   - Do not install Isaac Lab into an Isaac-Sim-only workstation. An
     incompatible Isaac Lab installer can replace Isaac Sim's Torch/NCCL pair.
   - OVRTX binary runtime plus sibling OVRTX source checkout.
   - Pinned gsplat source and Kit App Template checkout.
   - Initialize gsplat's git submodules before building its CUDA extension.
   - Nsight Systems, Nsight Compute, Nsight Graphics, Compute Sanitizer,
     CUDA-GDB, CUPTI/NVTX, and graphics diagnostics.
   - Repository Python dependencies and editable install.
   - Compile the CUDA extensions for the GPU this node actually has, before any
     render, benchmark, flyby, or profile. The editable install builds (or JITs)
     the custom renderer kernel; `gsplat` is a **separate** build that
     `scripts/provision_lidar_node.sh` does NOT perform — it only clones and
     inits submodules. Run `scripts/build_gsplat_main.sh` explicitly. Both
     builds target a compute capability: detect the live one with
     `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` (or
     `torch.cuda.get_device_capability()`) and set `TORCH_CUDA_ARCH_LIST` to
     match — `build_gsplat_main.sh` defaults to `8.9` (Ada / RTX 4090), so
     override it on any other card (`8.0` A100, `9.0` H100, `10.0` B200, `12.0`
     RTX 5090). A wrong arch fails at first render with `no kernel image is
     available for execution on the device` and voids every measurement.

6. Source the project environment before Isaac, OVRTX, or profiler commands.

   ```bash
   source scripts/remote_env.sh
   ```

   Preserve the ordering in that file. Isaac Sim's ML prebundle must precede
   unrelated site packages, and OVRTX's `python/` and `bin/` directories must
   be visible.

   Use `/isaac-sim/python.sh` for renderer code, Torch, NumPy, OVRTX, gsplat,
   USD/Kit, NPZ analysis, and project tests. Do not assume either local macOS
   `python3` or remote system `python3` has those modules. When a script imports
   `experiments.*`, include the repository root in `PYTHONPATH` or run it as a
   module.

7. Bound remote work.
   - Run short checks directly with an explicit timeout.
   - Run builds, temporal accumulation, matrices, and soaks in a named `tmux`
     session with a unique log and run ID.
   - Before retrying after an SSH timeout, inspect the existing PID, `tmux`
     session, output directory, and log. Do not launch a duplicate job.
   - Redirect verbose NVCC, Isaac, and profiler output to a log. Poll the status
     and a short tail, not the complete log.
   - Avoid nested `python -c` strings and multi-layer SSH quoting. Add a small
     checked-in script, sync it, and invoke it with normal arguments.
   - Run one finite GPU command under the cooperative lock:

     ```bash
     .agents/skills/provision-rtx-workstation/scripts/run-rtx-locked.sh \
       "$RTX_HOST" "$REMOTE_ROOT" "$RUN_ID" -- \
       /isaac-sim/python.sh scripts/custom_backend_smoke.py
     ```

     Set `RTX_REMOTE_DRY_RUN=1` to inspect the generated remote script without
     connecting. Lock contention exits with status `75`.

8. Validate narrowly before expensive work.

   ```bash
   nvidia-smi
   nvcc --version
   nsys --version
   ncu --version
   compute-sanitizer --version
   /isaac-sim/python.sh scripts/isaac_headless_smoke.py
   /isaac-sim/python.sh -m pytest -q tests
   ```

   Run `$test-isaac-graphics` for the full graphical validation ladder and
   `$profile-rtx-graphics` before collecting profiler evidence.

9. Record provenance.
   - Capture GPU, driver, CUDA, Isaac Sim, OVRTX, gsplat, source commit, tool
     versions, dataset hashes, and output paths.
   - Record any missing display, host capability, or performance-counter
     permission as a limitation instead of silently omitting it.

## Shared-host lock discipline

Prefer a user-coordinated reservation. The bundled runner wraps the complete
finite workload with:

```bash
flock -n /workspace/.vectorized-gaussian-renderer.gpu.lock \
  -c '<complete workload command>'
```

A failed lock acquisition means the node is busy. It is not permission to
bypass the lock.

## Completion criteria

Report:

- Node identity and GPU.
- Isolated remote root.
- Exact runtime and profiler versions.
- Dependency commits.
- Headless smoke and unit-test results.
- Counter/display limitations.
- Where source and generated evidence are stored locally.
