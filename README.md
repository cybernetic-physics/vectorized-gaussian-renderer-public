# Vectorized Gaussian Renderer for Isaac Sim

The opt-in direct Gaussian LiDAR sibling service is specified in
[`GAUSSIAN_LIDAR.md`](GAUSSIAN_LIDAR.md). Camera-only users do not load or
compile the LiDAR native extension, allocate LiDAR storage, build its LBVH,
launch its kernels, or synchronize for LiDAR.

A custom, batched GPU-resident Gaussian-splat sensor renderer for headless
Isaac Sim and Omniverse Kit. Isaac Sim owns physics, USD, and Fabric scene
state; this project owns persistent Gaussian storage, batched GPU scheduling,
custom rasterization/compositing kernels, and GPU-resident RGB, depth, alpha,
and semantic outputs.

Isaac Lab is not required. A future adapter may consume the public renderer
service only after the direct Isaac Sim path passes.

The current backend is a project-owned C++/CUDA implementation. It performs
anisotropic projection/culling, tile queue construction, emitted-prefix CUB
radix ordering, and fused RGB/depth/alpha/semantic compositing. One logical
service call uses one native submission when the batch fits the configured
physical workspace; explicitly bounded larger batches use an ordered fixed
number of native submissions into the same logical outputs. The default
high-throughput path copies the emitted-intersection count to the host and
synchronizes the current CUDA stream before the global radix sort; a
projection-cache hit reuses the sorted intersections without that
synchronization. Deterministic mode retains the device-only per-tile segmented
ordering path. Pinned `gsplat` is the correctness oracle; OVRTX is a separate
system/visual baseline; and the timed publication baseline is the reviewed
FlashGS-derived matched-contract port. Workspace storage
starts from an explicit estimate and grows through a bounded, fail-closed retry
when device counters prove that estimate insufficient; ordering work remains
emission-bounded after growth.

## Companion NVIDIA checkouts

Keep NVIDIA's OVRTX repository beside this repository when developing the
renderer. OVRTX is the authoritative public SDK, schema, example, and reference
integration used by this project for RTX Gaussian-rendering and sensor
baselines. The OVRTX checkout does not contain the proprietary Omniverse RTX
renderer implementation, and cloning it does not install the OVRTX runtime
package used by the benchmark scripts.

From the root of this repository:

```bash
git clone --branch main --single-branch \
  https://github.com/NVIDIA-Omniverse/ovrtx.git \
  ../ovrtx
```

The expected local layout is:

```text
workspace/
├── vectorized-gaussian-renderer/
├── ovrtx/
└── kit-app-template/              # recommended for Kit integration work
```

The NVIDIA Kit App Template is also recommended for contributors working on
the Isaac Sim extension, Kit lifecycle, packaging, or a future standalone Kit
application:

```bash
git clone --branch main --single-branch \
  https://github.com/NVIDIA-Omniverse/kit-app-template.git \
  ../kit-app-template
```

It is a reference and application scaffold, not a runtime dependency of the
custom CUDA kernels. Contributors working only on native projection,
rasterization, compositing, or unit tests can work without it. Keep both
NVIDIA repositories as separate sibling checkouts rather than copying or
vendoring their source into this repository.

## Agent skills and RTX toolchain

This repository includes reusable agent skills under `.agents/skills/`:

| Skill | Use it for |
|---|---|
| `provision-rtx-workstation` | Vast/local RTX-node setup, shared-host coordination, dependency installation, source synchronization, and provenance |
| `profile-rtx-graphics` | CUDA/Vulkan/RTX profiling, profiler selection, NVTX, memory/counter evidence, and bottleneck reporting |
| `test-isaac-graphics` | Writing and running finite Isaac/Kit graphical tests, GPU-output checks, image regressions, fidelity, and stability |
| `render-gaussian-flyby` | Matched camera contracts, custom/OVRTX flyby rendering, video packaging, and visual QA |
| `publish-r2-artifacts` | Versioned Cloudflare R2 publication, multipart datasets, checksums, CORS, and public verification |

The known-good workstation stack and installation procedure are documented in
[workstation-toolchain.md](.agents/skills/provision-rtx-workstation/references/workstation-toolchain.md).
The profiler decision tree is in
[profiler-playbook.md](.agents/skills/profile-rtx-graphics/references/profiler-playbook.md),
and the test contracts are in
[graphics-test-contracts.md](.agents/skills/test-isaac-graphics/references/graphics-test-contracts.md).
The flyby acceptance contract is in
[flyby-contract.md](.agents/skills/render-gaussian-flyby/references/flyby-contract.md),
and the public-asset contract is in
[r2-artifact-contract.md](.agents/skills/publish-r2-artifacts/references/r2-artifact-contract.md).
The separate visual/physical boundary and collision-proxy workflow are in
[HYBRID_PHYSICS.md](HYBRID_PHYSICS.md).

### Required and recommended tools

| Layer | Tools | Purpose |
|---|---|---|
| RTX runtime | NVIDIA driver, RTX GPU, CUDA Toolkit 12.8, Isaac Sim 6.0.1, OVRTX 0.3.0 runtime | Execute Isaac, RTX/OVRTX, and project-owned CUDA rendering |
| CUDA development | `nvcc`, CMake, Ninja, C++ build tools, CUPTI, NVTX | Build kernels and instrument measured regions |
| System profiling | Nsight Systems (`nsys`) | CUDA/Vulkan timeline, launches, synchronization, allocation, and CPU/GPU overlap |
| Kernel profiling | Nsight Compute (`ncu`) | Occupancy, divergence, cache, instruction, and memory-throughput counters when host permissions allow |
| Graphics profiling | Nsight Graphics (`ngfx`) | Vulkan/DX12/OpenGL frame, resource, pipeline, RTX dispatch, and shader inspection |
| Framework profiling | `torch.profiler` | PyTorch/ATen operations, extension dispatch, allocations, and trace export |
| GPU correctness | Compute Sanitizer | CUDA memory, race, initialization, and synchronization checks |
| Debugging | CUDA-GDB, GDB, `strace`, core dumps | Device assertions, native crashes, and hangs |
| GPU/process evidence | `nvidia-smi`, `nvidia-smi pmon`, NVML | GPU identity, utilization, driver memory, and process ownership |
| Vulkan diagnostics | `vulkaninfo`, Khronos validation layers, X11/Wayland or remote desktop | Device/extension validation and display-capable graphics capture |
| Build and coordination | Git/LFS, `rsync`, `ripgrep`, `tmux`, `flock`, `jq`, `htop` | Reproducible source, remote worktrees, locking, and evidence handling |
| Test and image analysis | Pytest, Ruff, NumPy, Pillow, plyfile, LPIPS/torchvision; optional FFmpeg/ImageMagick | Contract tests, tensor/image artifacts, metrics, and visual packaging |

The validated machine used Nsight Systems 2024.6.2, Nsight Compute 2025.1.1,
and Nsight Graphics 2026.2. Newer compatible versions may work, but benchmark
evidence must record exact versions.

Nsight Compute counters are not guaranteed merely because `ncu` is installed.
The host must grant GPU performance-counter access. Nsight Graphics similarly
requires a capture-capable display session for interactive frame debugging;
installation in a headless container is not proof that capture works.

### Workflow that has worked well

- Keep the local checkout authoritative and use an isolated remote worktree per
  agent or lane.
- Check GPU/process/session occupancy before using any shared RTX node. Do not
  launch or kill work when another agent is active.
- Source `scripts/remote_env.sh` before Isaac, OVRTX, CUDA-extension, or
  profiler commands. Its Python ordering prevents incompatible site packages
  from shadowing Isaac Sim's Torch/NCCL runtime.
- Warm shaders, JIT kernels, extensions, and allocators before measurement.
- Run finite headless smokes before benchmarks or profilers and always close
  `SimulationApp`.
- Validate RGB, depth, alpha, and semantic tensors on GPU, plus determinism,
  capacity counters, simulation cadence, and Fabric camera updates.
- Start performance diagnosis with a short Nsight Systems capture and NVTX,
  then use Nsight Compute, Nsight Graphics, PyTorch profiler, or Compute
  Sanitizer only for the question each tool can answer.
- Preserve raw reports and machine-readable JSON/CSV alongside unprofiled
  timings, commits, settings, camera/scene hashes, and limitations.
- Keep runtime performance, graphical fidelity, integration behavior, and CUDA
  correctness as separate pass/fail decisions.

## Current measured status

The release candidate is being re-evaluated on one identified NVIDIA L4 with a
fresh, source-frozen B1–B1024 changing-camera matrix. The comparison is between
the production Custom CUDA rasterizer and this repository's reviewed
FlashGS-derived matched-contract port, under RGB-only and full-sensor contracts.
Pinned gsplat is the correctness oracle, not the timed baseline.

No current winner claim belongs here until that matrix, its raw fidelity and
profiler evidence, exact-current CUDA/Isaac/sanitizer/stability gates, and the
content-addressed publication bundle all pass. Earlier OVRTX and gsplat tables
remain useful engineering diagnostics but are not the evidence for this release
and are intentionally not repeated on the landing page.

The source-only `benchmark-gcp-l4-matched-v2` tag produced no benchmark result
and was superseded before measurement. Its launcher would have prepended a CUDA
12.9 toolkit library directory to a CUDA 12.8 PyTorch wheel, reproducing an
incompatible cuBLAS load during LPIPS. The v3 launcher leaves PyTorch's bundled
runtime libraries authoritative and performs CUDA matrix, convolution, and
LPIPS-Alex operations before any long render; it also records the loaded CUDA
libraries and exact LPIPS weight hashes.

The source-only v3 tag also produced no benchmark result. Its anonymous-clone
test exposed that four repair-audit tests still read a pre-fix source file from
a legacy Git commit that the history-free public repository intentionally omits.
The v4 source carries that exact pre-fix file as a noncompiled, hash-pinned
reference and proves the audit works without any hidden Git history.

The source-only v4 tag likewise produced no benchmark result. Its first bounded
preflight exposed that this host's normal empty-tmux state is reported as
`error connecting ... (No such file or directory)`, which the fail-closed
occupancy probe did not yet recognize. The v5 source recognizes that specific
empty-server result while continuing to reject unknown telemetry failures.

The source-only v5 tag also produced no benchmark result. Its CUDA runtime
preflight passed, but the wrapper rejected its own CUDA child because it saved
process samples and deferred ancestry checks until after that child had exited.
The v6 source verifies ancestry while each process is live and retains only the
direct child and descendants it actually verified on the expected GPU.

This evaluation is a steady-state, robotics-shaped renderer microbenchmark over
one synthetic camera population, one 21,497,908-Gaussian Home Scan, one 128×128
resolution, and a predeclared direct/P128 physical schedule. Smaller contracts
are deterministic prefixes/subsets of the same frozen B1024 camera population,
not independent workload draws. It does not isolate a speedup caused by
vectorization, prove proportional batch scaling, measure end-to-end
Isaac/Fabric ingestion, physics, a policy or training loop, cover high-resolution
single-camera rendering, describe deployed-robot behavior, or replace Isaac Sim
RTX rendering. The final scoped result and immutable evidence links will be
inserted here only after the publication gate passes.

### Matched Home Scan flyby

The image below is a README-compatible animated preview. Select it to open the
full 24 fps side-by-side MP4.

[![Matched Home Scan flyby: custom 2D EWA beside OVRTX perspective](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/side-by-side-preview.gif)](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/side-by-side.mp4)

Both columns render the same 240-camera, clearance-aware path through the full
21,497,908-Gaussian Home Scan at 384x384. The camera contract SHA-256 is
`a59ed5ade5b276cebc6535b1bbe4de2abe2bc50c9f4cd5e3f6906f90096df45f`.
The custom column is one deterministic EWA render per camera. The OVRTX column
averages 64 temporal perspective frames after four warmup frames. OVRTX stores
the scene in two resident ParticleField prims containing 16,777,215 and
4,720,693 Gaussians because one OVRTX prim cannot address the full scene.

| Artifact | Public R2 object |
|---|---|
| Full side-by-side video | [MP4](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/side-by-side.mp4) |
| Custom EWA video | [MP4](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/custom.mp4) |
| OVRTX perspective video | [MP4](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/ovrtx-perspective.mp4) |
| Poster and 12-frame visual audit | [poster](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/poster.png), [storyboard](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/storyboard.png) |
| Canonical Home Scan LOD0 | [1.20 GB PLY](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/datasets/home-scan-lod0/29cee1594654/home-scan-lod0.ply), [dataset manifest](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/datasets/home-scan-lod0/29cee1594654/manifests/b231602598b1eb039175dcb7edbd475167c2fc92011c5e975a4727c62b9f74b9.json) |

The packaged video was decoded again after H.264 encoding and visually checked
at the beginning, midpoint, end, and twelve evenly spaced frames. The path is
upright, camera-matched, and free of unintended wall traversal. This remains a
visual demonstration, not equation-level fidelity acceptance: the production
EWA and OVRTX perspective models differ as documented below.

Home Scan is by
[Isaiah Sweeney](https://superspl.at/scene/3f89bbd3) and is redistributed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The canonical
PLY SHA-256 is
`29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267`.
The release dependency manifest is published create-only with
`publication/publish_immutable_r2.py`; its full-hash key and anonymous
verification receipt are part of the release evidence. The older
`cloudflare/r2-assets.json` remains the historical flyby-upload inventory and
is not the dependency manifest for this benchmark release.

### Matched Garage High Detail flyby

[![Garage High Detail: custom 2D EWA beside OVRTX perspective](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/garage-high-detail/v3-cf8d65138dbc/side-by-side-preview.gif)](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/garage-high-detail/v3-cf8d65138dbc/side-by-side.mp4)

This decoded and reviewed 72-frame interior walkthrough uses a shared camera
contract (`2061260aecba934745dc235266cc3888c66af077c7bc4126ceb2561c8c9775b1`),
the deterministic custom EWA renderer, and 16 temporal OVRTX perspective
samples per camera at 256x256. It is a matched visual demonstration only, not
equation-level fidelity acceptance.

| Artifact | Public R2 object |
|---|---|
| Full, custom, and OVRTX videos | [side-by-side](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/garage-high-detail/v3-cf8d65138dbc/side-by-side.mp4), [custom](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/garage-high-detail/v3-cf8d65138dbc/custom.mp4), [OVRTX](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/garage-high-detail/v3-cf8d65138dbc/ovrtx-perspective.mp4) |
| Visual audit | [poster](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/garage-high-detail/v3-cf8d65138dbc/poster.png), [storyboard](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/garage-high-detail/v3-cf8d65138dbc/storyboard.png) |
| Canonical scene | [1.28 GB PLY](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/datasets/garage-high-detail/cf8d65138dbc/garage-high-detail.ply) |

Garage - High Detail Test is by
[Ethan](https://superspl.at/user?id=ethan3111), from
[SuperSplat](https://superspl.at/scene/8f4d8e05), and is redistributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its canonical PLY
SHA-256 is `cf8d65138dbc31e3619ce697a875b1a005fe88fa126d5737e66294b9064a171a`.

### Why OVRTX perspective fidelity differs

The original default renderer differs from OVRTX perspective because it uses a
different Gaussian rendering equation, not because it uses fewer Gaussians or
a lower output resolution:

- The fast custom path projects each 3D Gaussian onto the image as a
  deterministic local tangent-plane/EWA ellipse and then alpha-composites the
  ordered 2D footprints. This matches the conventional `gsplat`/3DGS model.
- Based on the measured outputs and the exact-ray diagnostic, OVRTX perspective
  appears to evaluate each camera ray against the full 3D Gaussian and treat
  its contribution as a probabilistic first hit. OVRTX's implementation is
  proprietary, so this is an evidence-backed inference rather than a source
  code claim.
- The different evaluation changes each splat's edge and footprint. Amplified
  RGB differences form circular or elliptical rings around splats, and the
  changed coverage weights propagate into RGB, alpha, weighted depth, and
  semantic output.
- The fast custom compositor selects the semantic ID of the strongest
  individual Gaussian contribution. Temporally accumulated OVRTX semantics
  behave more like the label with the greatest aggregate hit probability.
- The custom EWA render is deterministic in one invocation. OVRTX perspective
  is stochastic and must accumulate many frames before its RGB, alpha, depth,
  and semantic estimates converge.

Fidelity evidence is intentionally split by rendering equation:

| Comparison | PSNR | SSIM | LPIPS | Alpha MAE | Depth rel. | Semantic |
|---|---:|---:|---:|---:|---:|---:|
| Fast EWA vs pinned `gsplat`, B8 | 156.611 dB | 1.000000 | 0.000000004 | 0.000000013 | 0.000000102 | 100.000% |
| Fast EWA vs OVRTX perspective, 16,384 frames | 35.094 dB | 0.985668 | 0.012180 | 0.023973 | 5.789% | 93.225% |
| Experimental exact-ray vs OVRTX perspective, 65,536 frames | 55.449 dB | 0.999723 | 0.000124 | 0.001317 | 0.564% | 99.554% |

The exact-ray diagnostic passes every numerical threshold except the 99.9
percent semantic-agreement requirement. It is not yet the production or
headline-performance path and has not completed the same Home Scan, matrix,
stability, and independent-verification program as the fast EWA renderer.
Most of its remaining semantic disagreements are nearly tied labels: among the
61 pixels where both outputs selected foreground but disagreed, OVRTX selected
the model's second-highest-probability label in 60. Finite temporal sampling
can therefore select the runner-up even when the inferred probability model is
correct.

The strongest current conclusion is:

> The original fast renderer accurately matches conventional `gsplat`/3DGS,
> while OVRTX perspective renders Gaussians more like probabilistic 3D ray
> volumes.

A tuned one-camera Home Scan comparison against a 256-frame temporal OVRTX
sample measures 47.270 dB PSNR and 99.908 percent semantic agreement. It is
retained as a useful finite-sample control, not as equation-level perspective
parity or overall acceptance evidence.

The USD `projectionModeHint` is advisory. A fail-closed seven-process L4 gate
authored and read back `perspective` and `tangential`, verified matched
rotated anisotropic inputs and zero same-mode repeat noise, and produced
bitwise-identical OVRTX RGB, alpha, depth, semantic, and valid-depth tensors in
both comparisons. A repeated 5 percent focal-length control changed 10.6567
percent of RGB/alpha elements, confirming that the camera/output path was
responsive. This valid negative blocks the tangential benchmark lane for the
tested OVRTX 0.3.0 configuration; it does not establish global non-support or
an OVRTX defect.

The historical cache-heavy OVRTX system path stores no per-camera
projected-record list. It retains per-pixel depth cutoffs, sorted Gaussian-ID
intersections, and dense pixel ranges only while scene/camera geometry remains
coherent. The compositor reprojects qualifying Gaussians and rewrites RGB,
depth, alpha, and semantic outputs on every render; rendered images are never
cached. The changing-camera path assessed above instead materializes 44-byte
first-pass records within each render to avoid projecting the same camera and
Gaussian pair twice. It does not reuse records from an earlier camera state.

Projection reuse is explicit and off by default in the backend:

```python
backend = CustomCudaBackend(
    tile_size=1,
    depth_bucket_count=32,
    depth_bucket_group_size=8,
    compact_projection_cache=True,
    enable_projection_cache=True,
)
```

Screen-space rendering defaults to gsplat-compatible `rasterize_mode="classic"`.
Select `rasterize_mode="antialiased"` to apply Mip-Splatting's view-dependent
opacity compensation after the `covariance_epsilon` footprint dilation. Compare
classic only with classic and antialiased only with antialiased; the latter is a
different fidelity contract and is intentionally rejected by the exact-ray
evaluation path.

The bounded mode-matched parity and analytic subpixel check is:

```bash
python benchmarks/compare_custom_gsplat.py \
  --scene synthetic-small --batch 1 --width 128 --height 128 \
  --rasterize-mode antialiased --verify-antialias-fixture \
  --gsplat-compatibility-patch patches/gsplat-cuda-event-flags.patch \
  --warmup 1 --iterations 0 \
  --output outputs/antialias-parity
```

Tensors created inside `torch.inference_mode()` have no PyTorch version counter,
so the backend conservatively treats them as projection and capacity-cache
misses on every submission. Create long-lived camera tensors outside inference
mode if unchanged-input cache hits are required. Continue to call
`backend.invalidate_projection_cache()` after external Fabric/Warp writes that
bypass PyTorch version counters.

Reproduce the historical cache-heavy OVRTX acceptance experiment (not the
current matched release headline):

```bash
scripts/run_custom_headline_repeats.sh
scripts/run_ovrtx_headline_repeats.sh
scripts/summarize_headline_acceptance.sh
```

The summarizer is fail-closed: it reports EWA/`gsplat` fidelity separately,
keeps OVRTX perspective fidelity as the acceptance gate, and records exact-ray,
finite-frame Home, and projection-mode results as diagnostics.

Capture that historical cache-hit path with Nsight Systems:

```bash
scripts/profile_home_scan_nsys.sh
```

Nsight Systems captures at B1 and B1024 show direct projection and global
sorting only on the first cache miss, while the fused reproject/compositor
executes on every render. Nsight Compute 2025.1.1 is installed, but this Vast
host blocks hardware counters with `RmProfilingAdminOnly=1` and no
`CAP_SYS_ADMIN`; the preflight evidence records that limitation explicitly.

The full Home Scan B1024 soak ran for 603.329 seconds and completed 13,400
synchronized measured renders at 22,794.55 images/s with zero allocation
growth, zero queue overflow, and valid GPU RGB, depth, alpha, and semantic
outputs throughout.

## Remote development machine

Only connect after the current task has permission to use the node and a
read-only occupancy check confirms that another agent is not using it.

```bash
ssh vast-gsplat-isaac
cd /workspace/vectorized-gaussian-renderer
source scripts/remote_env.sh
```

The Vast instance uses:

- NVIDIA GeForce RTX 4090
- Isaac Sim 6.0.1 at `/isaac-sim`
- Isaac's bundled Torch 2.11.0+cu128 and NCCL 2.28.9
- Current `gsplat` main checkout at `/workspace/src/gsplat`
- OVRTX and the Kit app template under `/workspace/src`
- CUDA Toolkit 12.8 and NVIDIA profiling tools

The login environment deliberately puts Isaac Sim's ML prebundle first on
`PYTHONPATH`. This prevents an older site-package Torch/NCCL installation from
shadowing Isaac Sim 6.0's compatible runtime.

## Current control benchmark

```bash
/isaac-sim/python.sh benchmarks/batched_render.py \
  --num-cameras 64 \
  --num-gaussians 100000 \
  --width 128 \
  --height 128
```

Preliminary infrastructure smoke numbers on the rented RTX 4090:

- 3.94 ms per 64-camera batch
- 16.3k camera frames/second
- 266 megapixels/second
- 0.50 GiB peak allocated GPU memory

These are not fair baseline or acceptance results: the run is synthetic, lacks
semantic compositing, and uses stock `gsplat`.

The headless OVRTX Gaussian schema/AOV smoke can be reproduced remotely with:

```bash
source /workspace/venvs/ovrtx/bin/activate
export PYTHONPATH=/workspace/tools/ovrtx-0.3.0/python
export LD_LIBRARY_PATH=/workspace/tools/ovrtx-0.3.0/bin:${LD_LIBRARY_PATH:-}
python experiments/ovrtx_gaussian_smoke.py \
  --output-dir outputs/ovrtx-canonical-single-default
```

This probe proves Gaussian ingestion, colored LDR, and metric distance output.
It follows NVIDIA's canonical 3DGS DC-SH conversion. A four-camera
single-RenderProduct tiled probe returns valid distance but black LDR, so that
path is tracked as a renderer limitation rather than benchmark evidence.
OVRTX does return valid full-resolution color and metric distance for four
single-camera RenderProducts submitted together in one `Renderer.step()`:

```bash
python experiments/ovrtx_gaussian_smoke.py \
  --output-dir outputs/ovrtx-canonical-products-4 \
  --cameras 4 \
  --layout products
```

The multi-product path has no per-camera render submission loop and is the
current RTX batch-baseline mechanism. The fair runner enables generated
coverage alpha, groups the deterministic semantic sidecar into 1,024 labeled
Gaussian fields, decodes OVRTX's ID map once, and writes all four required
outputs into preallocated CUDA tensors:

```bash
scripts/run_ovrtx_baseline.sh
```

The 50-frame synthetic-small 128x128 controls measured 14.35 ms for B1 and
93.83 ms for B8. The outputs and machine-readable rows are saved under
`outputs/baselines/ovrtx-fair-b1-128/` and
`outputs/baselines/ovrtx-fair-b8-128/`. These are baseline measurements, not
custom-renderer acceptance results.

### Compile CUDA extensions for the GPU you are running on — first

The custom renderer kernel and `gsplat` are compiled from source against a
specific GPU compute capability. **Build them for the card the node actually
has, before any smoke, benchmark, flyby, or profile run.** Two independent
extensions must be built on each fresh node:

- The custom renderer kernel, built by the editable install
  (`pip install -e .`) or JIT-compiled on first render.
- `gsplat`, the mathematical parity control — **provisioning clones it and
  initializes its submodules but does not build it.** You must run the build
  step yourself.

Detect the running GPU's capability on the node and set the arch to match:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. 8.9
# or: /isaac-sim/python.sh -c 'import torch; print(torch.cuda.get_device_capability())'
```

Common values: `8.9` RTX 4090 (Ada), `8.0` A100, `9.0` H100, `10.0` B200,
`12.0` RTX 5090. A mismatch fails at first render with
`no kernel image is available for execution on the device` (or silently loads
nothing), so the wrong arch invalidates every downstream number.

Build the editable `gsplat` main checkout, specialized for 3DGS with line
information retained for profiling. `scripts/build_gsplat_main.sh` defaults to
`TORCH_CUDA_ARCH_LIST=8.9` (Ada / RTX 4090); **override it for any other GPU:**

```bash
TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" \
  scripts/build_gsplat_main.sh
```

## Profiling

Use the
[profile-rtx-graphics skill](.agents/skills/profile-rtx-graphics/SKILL.md)
for the complete capture and evidence contract.

System timeline:

```bash
scripts/profile_nsys.sh
```

CUDA kernel metrics:

```bash
scripts/profile_ncu.sh
```

Profile the custom renderer under the exact authored-display OVRTX contract:

```bash
scripts/profile_custom_nsys.sh
scripts/profile_custom_ncu.sh
scripts/profile_home_scan_nsys.sh
```

Profiles are written under the active worktree's `profiles/` directory.
The validated Nsight Systems capture is `profiles/gsplat-batched.nsys-rep`.
Its first bottleneck is the forward rasterization kernel at about 55% of
measured GPU kernel time, followed by radix sorting at about 18%.

Nsight Compute is installed, but this particular Vast host sets
`RmProfilingAdminOnly: 1` and does not grant the container `CAP_SYS_ADMIN`.
The `profile_ncu.sh` preflight reports that host limitation instead of failing
mid-capture. Nsight Systems, Nsight Graphics, Compute Sanitizer, CUDA-GDB,
CUPTI/NVTX, nvprof, and the CUDA compiler are installed and on `PATH`.

For a narrow CUDA correctness pass:

```bash
compute-sanitizer --tool memcheck \
  /isaac-sim/python.sh scripts/custom_backend_smoke.py
```

Use Nsight Graphics for a Vulkan/RTX frame only from a display-capable session.
Use `vulkaninfo` and Khronos validation layers for API/device diagnostics, but
keep validation and capture overhead outside performance measurements.

## Graphical test ladder

Use the
[test-isaac-graphics skill](.agents/skills/test-isaac-graphics/SKILL.md)
when adding or running graphical validation. After sourcing
`scripts/remote_env.sh`, the normal increasing-cost ladder is:

```bash
/isaac-sim/python.sh -m pytest -q tests
/isaac-sim/python.sh scripts/custom_backend_smoke.py
/isaac-sim/python.sh scripts/isaac_headless_smoke.py
/isaac-sim/python.sh scripts/isaacsim_extension_smoke.py
/isaac-sim/python.sh scripts/fabric_camera_smoke.py
/isaac-sim/python.sh scripts/deterministic_cuda_smoke.py
/isaac-sim/python.sh scripts/multi_scene_cuda_smoke.py
/isaac-sim/python.sh scripts/projection_cache_cuda_smoke.py
```

Every `SimulationApp` test must be finite and close the application in a
`finally` block. Production-path tests must check CUDA-resident RGB, depth,
alpha, and semantic outputs; finite values; nonempty foreground; valid
semantics; simulation cadence; and zero queue overflow.

Visual comparisons use immutable camera/scene metadata and preserve reference,
candidate, side-by-side, RGB/alpha/depth differences, semantic mismatch, JSON,
and CSV. OVRTX Gaussian tests must identify and retain separate `perspective`
and `tangential` projection results, verify the authored USD hint, and fail
closed if the resulting tensors are bitwise identical.

## Isaac Sim integration

The loadable Kit extension and process-local service are now implemented:

```python
import torch
from isaacsim_gaussian_renderer import RendererService

service = RendererService(custom_backend, height=128, width=128, max_views=64)
service.initialize(stage, torch.device("cuda:0"))
service.load_scene(
    scene_id=0,
    means=means,
    scales=scales,
    rotations=rotations,
    opacities=opacities,
    features=features,
    semantic_ids=semantic_ids,
)
outputs = service.render(camera_transforms, intrinsics, scene_ids)
```

`CustomCudaBackend` enables bounded adaptive capacity by default. The first new
or explicitly invalidated camera/scene input reads device counters. On overflow,
the backend grows visible/intersection storage with headroom, invalidates cached
geometry, and retries into the same caller-owned output tensors. Identical inputs
that have already passed reuse that validation without a host synchronization.
After the counter read has synchronized prior work, growth releases the old
workspace before allocating its replacement, avoiding a transient pair of full
workspaces. If replacement allocation fails, the backend attempts to restore
fresh storage at the old capacity before failing closed.
The default geometric growth factor is 1.25, matching the demand-headroom
factor. This avoids doubling a large workspace when a later camera batch only
slightly exceeds a capacity that was already sized from observed counters. The
Kit extension inherits this backend default unless a caller explicitly
overrides it.
Set `max_workspace_bytes` to a deployment-specific hard ceiling on backend-owned
workspace tensor payloads (scene tensors, outputs, and allocator/driver overhead
are reported separately). If a headroom-sized growth target exceeds that
ceiling, the backend first retries at exact counter-observed demand and records
the adjustment. If observed demand itself exceeds the ceiling, or if the retry
budget is exhausted, the backend raises
`RendererCapacityError`, attaches JSON-serializable telemetry, and marks the
failed output tensors invalid instead of returning a truncated image.
`backend.capacity_stats` reports every attempted capacity, actual counters,
headroom, retry count, final storage, and synchronization role.

Changing camera/scene inputs cannot be made both automatically resizeable and
fully asynchronous: deciding to allocate host-owned Torch tensors requires a
device-counter read. Performance reports therefore retain both GPU-event and
wall timing, perform growth before measured steady state, and state when every
moving-camera frame required validation. `adaptive_capacity=False` is an
explicit expert/manual mode for pre-sized profiling; callers must then invoke
`check_capacity()` and treat any overflow as a failed render.

The current milestone includes:

- `RendererService` with `initialize`, `load_scene`,
  `update_scene_transforms`, `render`, `synchronize`, and `shutdown`.
- Strict GPU tensor contracts for scene data, camera batches, active subsets,
  and preallocated outputs.
- Multi-scene registration with packed offsets.
- Backend protocol injection for project-owned custom kernels.
- Project-owned fused CUDA projection, visibility, tile binning, emitted-prefix
  global CUB ordering (or deterministic device-tile segmented ordering), and
  RGB/depth/alpha/semantic compositing.
- Reusable adaptively sized GPU workspace with bounded grow/retry, an optional
  hard byte ceiling, explicit overflow counters, and no post-warmup measured-loop
  allocation for a validated capacity.
- Packed multi-scene storage, arbitrary scene IDs, per-environment transforms,
  active camera subsets, and deterministic depth/ID tie-breaking.
- Isaac camera-source boundary that caches USD intrinsics and reads
  Fabric/USDRT world transforms with one Warp conversion launch into a
  persistent CUDA Torch tensor and no runtime camera loop. The tensor is not a
  direct zero-copy Torch view of Fabric storage.
- Kit extension lifecycle under `exts/isaacsim.gaussian_renderer`.
- Cached GPU scene-offset metadata and no CUDA scene-ID copy to the host in the
  service render path.
- Finite `SimulationApp` extension and physics-coupled vectorized examples
  using the custom backend inside Isaac Sim 6.0.1:

```bash
/isaac-sim/python.sh scripts/isaacsim_extension_smoke.py
/isaac-sim/python.sh scripts/isaacsim_vectorized_example.py
```

Run the separate OVRTX/custom system matrix with one shared timing policy and
the exact authored-display output convention. It is not the current
same-equation publication baseline:

```bash
scripts/run_acceptance_matrix.sh
```

Remaining work is explicitly separated from the implemented Gaussian sensor
path:

- Close the measured OVRTX temporal fidelity gap or retain an overall FAIL
  verdict while preserving strict `gsplat` compositing parity evidence.
- Rerun the complete independent benchmark/stability evidence bundle if the
  fidelity gate is resolved; the current clean-root audit correctly fails
  closed on fidelity.
- Add a separately measured dynamic robot/mesh compositor when hybrid visual
  observations are required.
- Add a direct USD particle-field import convenience layer; current real-scene
  ingestion is checksummed PLY to canonical GPU tensors.

The deterministic fake backend remains only a lifecycle/contract double.

## Trajectory evaluation

Use a versioned route to run static or dynamic evidence without conflating
cache-hit and moving-camera performance:

```bash
source scripts/remote_env.sh
/isaac-sim/python.sh benchmarks/run_trajectory.py \
  --route benchmarks/camera_paths/home-scan-representative-v1.json \
  --renderer custom \
  --scenario sequential-flyby \
  --scene-path /workspace/datasets/home-scan-lod0.ply \
  --frames 120 --batch 1 --width 256 --height 256
```

The runner writes `outputs/evals/<run-id>/` with trajectory, timing, cache,
tensor, frame, video, and SHA-256 evidence. Compare matched render-output NPZ
files with `benchmarks/compare_trajectory.py`; validate geometry and coverage
with `benchmarks/validate_trajectory_route.py`; then reopen and audit the
bundle with `benchmarks/audit_trajectory_artifacts.py`. The acceptance
summarizer treats static coherent performance, dynamic trajectory performance,
OVRTX perspective fidelity, Isaac/Fabric integration, cache correctness,
artifact integrity, stability, reproducibility, and memory independently.

## License and attribution

Project-owned code is licensed under
[Apache License 2.0](LICENSE). The FlashGS-derived matched-port sources retain their MIT
License; see [the third-party notices](THIRD_PARTY_NOTICES.md) and
`src/isaacsim_gaussian_renderer/native/flashgs/LICENSE.flashgs`. Public scene
assets are not part of the Python package. Their authors, source URLs, and
Creative Commons terms are recorded in `datasets/*.manifest.json` and beside
the corresponding public artifacts above.
