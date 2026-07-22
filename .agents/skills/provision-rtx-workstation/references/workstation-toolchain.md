# RTX workstation toolchain

## Contents

- Known-good project stack
- Recommended directory layout
- Runtime identity and interpreter rules
- Portable source and dataset synchronization
- Base Linux packages
- NVIDIA development and profiling tools
- Project sources and runtimes
- Long-running job discipline
- Headless Isaac startup triage
- Environment validation
- Evidence snapshot
- High-frequency failure map

## Known-good project stack

The repository evidence was produced with:

| Component | Known-good value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9 |
| Driver | 590.48.01 |
| CUDA toolkit | 12.8, `nvcc` 12.8.93 |
| Isaac Sim | 6.0.1 at `/isaac-sim` |
| Isaac Torch | 2.11.0+cu128 |
| NCCL | 2.28.9 |
| OVRTX | 0.3.0 runtime; source checkout at commit `29d1103` |
| gsplat | commit `77ab983ffe43420b2131669cb35776b883ca4c3c` |
| Nsight Systems | 2024.6.2 |
| Nsight Compute | 2025.1.1 |
| Nsight Graphics | 2026.2 |

Newer compatible versions may work, but do not mix versions silently in
benchmark evidence.

## Recommended directory layout

```text
/isaac-sim/
/workspace/
├── agent-worktrees/<lane>/             # isolated renderer checkout
├── datasets/
├── src/
│   ├── gsplat/
│   ├── ovrtx/
│   └── kit-app-template/
└── tools/
    └── ovrtx-0.3.0/
        ├── bin/
        └── python/
```

Keep the canonical source and commits on the local client. Treat remote
worktrees as execution environments.

## Runtime identity and interpreter rules

The target is Isaac Sim directly. Do not install Isaac Lab unless a separate
task explicitly requires it and the compatibility matrix has been checked.
Installing an older Isaac Lab package into an Isaac Sim 6.0 image previously
replaced Torch/NCCL and produced an unresolved `ncclCommShrink` symbol at Isaac
startup.

Use this interpreter matrix:

| Work | Interpreter |
|---|---|
| Isaac, Kit, USD, Torch, CUDA, OVRTX, gsplat, NumPy, NPZ analysis, tests | `/isaac-sim/python.sh` after `source scripts/remote_env.sh` |
| OS-only bootstrap with no project imports | System `python3`, only after checking required modules |
| Local macOS analysis | A declared local environment; never assume system Python has Torch or NumPy |

Do not create a separate venv as an escape hatch until `python3-venv`,
`ensurepip`, module paths, and the required OVRTX native libraries are proven.
The project-supported path is Isaac's Python.

When invoking a script that imports `experiments.*`, use:

```bash
export PYTHONPATH="$VGR_PROJECT_ROOT:$VGR_PROJECT_ROOT/src:${PYTHONPATH:-}"
/isaac-sim/python.sh -m experiments.<module> ...
```

Prefer checked-in scripts over nested SSH heredocs or `python -c` commands with
embedded newlines.

## Portable source and dataset synchronization

The macOS system rsync may not support GNU options such as
`--info=progress2`. Use portable flags and create the destination first:

```bash
ssh "$RTX_HOST" "mkdir -p '$REMOTE_ROOT'"
rsync -azn --itemize-changes \
  --exclude .git/ --exclude outputs/ --exclude build/ \
  "$LOCAL_ROOT/" "$RTX_HOST:$REMOTE_ROOT/"
rsync -az --partial --progress \
  --exclude .git/ --exclude outputs/ --exclude build/ \
  "$LOCAL_ROOT/" "$RTX_HOST:$REMOTE_ROOT/"
```

Do not combine a source transfer with a second remote command until the
transfer succeeds. Do not initialize a remote Git repository merely to inspect
an rsynced execution tree. If Git inspection is intentional and ownership is
different, prefer a command-scoped override:

```bash
git -c safe.directory="$REMOTE_ROOT" -C "$REMOTE_ROOT" status --short
```

For datasets:

1. Create the remote parent.
2. Transfer into a unique dataset directory.
3. Verify file count, byte size, and checksum.
4. Keep datasets outside source worktrees.
5. Never use `--delete` against a shared dataset root.

## Base Linux packages

Install the ordinary build and coordination tools:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake git git-lfs htop jq libcap2-bin \
  ninja-build nodejs npm python3-venv ripgrep rsync tmux
```

For Vulkan and graphical diagnostics, also install packages available for the
distribution:

```bash
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  mesa-utils vulkan-tools vulkan-validationlayers xvfb \
  ffmpeg imagemagick libxt6
```

Isaac Sim 6.0's bundled MaterialX libraries require `libXt.so.6`, including on
a headless node. Verify the package and loader entry instead of accepting
MaterialX load errors as harmless headless warnings:

```bash
dpkg-query -W libxt6
ldconfig -p | rg 'libXt\.so\.6'
```

Package names can vary by Ubuntu release. A physical display is unnecessary
for headless Isaac tests, but Nsight Graphics interactive capture normally
needs a working X11/Wayland or remote-desktop session.

## NVIDIA development and profiling tools

The node needs:

- NVIDIA driver compatible with the selected Isaac Sim and CUDA versions.
- CUDA Toolkit: `nvcc`, runtime libraries, CUDA-GDB, Compute Sanitizer, CUPTI,
  and NVTX.
- Nsight Systems CLI and GUI reader.
- Nsight Compute CLI and GUI reader.
- Nsight Graphics for Vulkan/DX12/OpenGL frame and shader inspection.
- `nvidia-smi`/NVML for process, utilization, and driver-memory evidence.

Prefer an image that already contains the required driver-facing stack. If
installing manually, use NVIDIA's packages matching the pinned versions rather
than copying binaries from another host.

Nsight Compute requires hardware-counter permission. Check:

```bash
grep '^RmProfilingAdminOnly' /proc/driver/nvidia/params || true
capsh --print | sed -n 's/^Current: //p'
```

`RmProfilingAdminOnly: 1` without `cap_sys_admin` means counter collection is
blocked by the host. Preserve that as evidence and use Nsight Systems instead;
do not invent occupancy or bandwidth numbers.

## Project sources and runtimes

Clone reference source beside the renderer:

```bash
git clone --branch main --single-branch \
  https://github.com/NVIDIA-Omniverse/ovrtx.git \
  /workspace/src/ovrtx
git clone --branch main --single-branch \
  https://github.com/NVIDIA-Omniverse/kit-app-template.git \
  /workspace/src/kit-app-template
git clone https://github.com/nerfstudio-project/gsplat.git \
  /workspace/src/gsplat
git -C /workspace/src/gsplat checkout \
  77ab983ffe43420b2131669cb35776b883ca4c3c
git -C /workspace/src/gsplat submodule update --init --recursive
```

Install the OVRTX binary package separately under
`/workspace/tools/ovrtx-0.3.0`. The public OVRTX Git repository is reference
source and examples; it is not the proprietary renderer runtime.

Use Isaac Sim's Python:

```bash
/isaac-sim/python.sh -m pip install --no-deps \
  'lazy_loader>=0.4' \
  'numpy==1.26.4' \
  'pillow>=10' \
  'plyfile==1.1.3'
/isaac-sim/python.sh -m pip install --no-deps -e \
  /workspace/agent-worktrees/<lane>
```

Do not install an unrelated Torch over Isaac Sim's bundled Torch/NCCL pair.

gsplat must be **compiled from source on the node before it is used** —
`scripts/provision_lidar_node.sh` clones it and initializes submodules but does
not build it, so a freshly provisioned node imports gsplat's Python but has no
CUDA backend until you build it. The same applies to the custom renderer
kernel, which the editable install (or first-render JIT) compiles.

Compile for the compute capability of the GPU the node is **actually running**,
not a hardcoded assumption. Detect it and pass it through:

```bash
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"  # e.g. 8.9
test -f /workspace/src/gsplat/gsplat/cuda/csrc/third_party/glm/glm/glm.hpp
tmux new-session -d -s "gsplat-build-$RUN_ID" \
  "cd /workspace/src/gsplat &&
   export TORCH_CUDA_ARCH_LIST=$ARCH MAX_JOBS=16 BUILD_EXPERIMENTAL=0 &&
   /isaac-sim/python.sh -m pip install --no-build-isolation -e . \
   > /workspace/gsplat-build-$RUN_ID.log 2>&1"
```

`scripts/build_gsplat_main.sh` wraps this but hardcodes
`TORCH_CUDA_ARCH_LIST=8.9` (Ada / RTX 4090); override it on any other GPU
(`8.0` A100, `9.0` H100, `10.0` B200, `12.0` RTX 5090). After building, confirm
the backend loads: `/isaac-sim/python.sh -c 'from gsplat import cuda'`. An arch
mismatch surfaces as `no kernel image is available for execution on the device`
at first render — the fix is to rebuild for the live capability, not to touch
CUDA or Isaac. Missing GLM headers mean the submodule checkout is incomplete,
not that CUDA or Isaac is broken.

## Long-running job discipline

Use a unique run ID, output directory, log, and `tmux` session for any job
likely to outlive one tool call:

```bash
tmux new-session -d -s "$RUN_ID" \
  "cd '$REMOTE_ROOT' &&
   source scripts/remote_env.sh &&
   timeout --signal=TERM 2h <command> \
   > 'outputs/logs/$RUN_ID.log' 2>&1"
```

Inspect without replaying the complete log:

```bash
tmux has-session -t "$RUN_ID" 2>/dev/null && echo RUNNING || echo NOT_RUNNING
tail -n 80 "outputs/logs/$RUN_ID.log"
```

On an SSH timeout, reconnect and inspect the existing session and output before
retrying. SSH failure does not prove the remote process stopped. Never launch
the same benchmark twice into the same output directory.

For parallel local worktrees, prove one worker launch before fan-out. A CLI/model
identifier mismatch can terminate every lane before any source change. Give
each worker a local ownership boundary, but route all remote GPU commands
through one executor and one queue.

## Headless Isaac startup triage

Always source `scripts/remote_env.sh` before starting Isaac as root; it sets
`OMNI_KIT_ALLOW_ROOT=1`. Bound compatibility checks and smokes with `timeout`.

Headless containers may emit repeated OmniHub, default-display, or GLFW
warnings. Classify them as non-fatal only when all of the following are true:

- The finite test emitted its unique success marker.
- Required JSON says `"pass": true`.
- No Python traceback, fatal signal, CUDA error, or missing output followed.
- `SimulationApp.close()` completed.

First-run OVRTX shader compilation can take minutes. Warm it once outside
measured ranges and preserve the log rather than polling the full output.

## Environment validation

After `source scripts/remote_env.sh`, verify:

```bash
for tool in nvidia-smi nvcc nsys ncu compute-sanitizer cuda-gdb; do
  command -v "$tool"
done
test -x "$NSIGHT_GRAPHICS_HOME/ngfx"
test -d "$OVRTX_ROOT/python/ovrtx"
test -d "$GSPLAT_SOURCE_PATH"

/isaac-sim/python.sh - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
assert torch.cuda.is_available()
PY
```

Run first-use shader compilation outside measured ranges. It can take minutes
after a fresh OVRTX install, cache reset, or driver update.

## Evidence snapshot

Save, without secrets:

```bash
nvidia-smi
nvcc --version
nsys --version
ncu --version
git rev-parse HEAD
git -C /workspace/src/gsplat rev-parse HEAD
git -C /workspace/src/ovrtx rev-parse HEAD
```

Also record:

- Isaac Sim version and path.
- OVRTX binary version and path.
- `CUDA_VISIBLE_DEVICES`.
- Display/headless mode.
- Performance-counter permission.
- Dataset checksums.
- Worktree and output directories.

## High-frequency failure map

| Symptom | Likely cause | Required response |
|---|---|---|
| `No module named numpy`, `torch`, `ovrtx`, or `gsplat` | Bare local/system Python or unsourced environment | Use Isaac Python after `remote_env.sh`; print `sys.executable` and module origins |
| `No module named experiments` | Repository root absent from `PYTHONPATH` | Add both root and `src`, or use `-m` |
| `glm/gtc/type_ptr.hpp` missing | gsplat submodules absent | Initialize submodules; do not reinstall CUDA |
| `ncclCommShrink` symbol failure | Torch/NCCL stack replaced | Restore Isaac's bundled ML prebundle; remove incompatible Isaac Lab install |
| Root-user rejection | `OMNI_KIT_ALLOW_ROOT` not set | Source `remote_env.sh` before launch |
| `dubious ownership` | Rsynced tree owned by another UID | Avoid remote Git or use command-scoped `safe.directory` |
| Remote path missing | Destination parent not created or wrong lane | Create and verify the exact remote root before transfer |
| SSH timeout | Transient endpoint or long-running process | Inspect the existing job before any retry |
