# RTX compositing bottleneck report

**Question.** For robot-visible rendering at scale (N robots inside the Home
Scan gsplat), where does the frame time go — and which robot-layer
implementation should the hybrid compositor use?

**Answer.** The per-env Isaac-RTX robot layer is ~99.7% framework tax, not
ray-tracing cost. Replacing it with a custom tracer (Warp software BVH, and the
OptiX RT-core tracer in `experiments/optix_tracer/`) makes the robot layer
effectively free and returns the bottleneck to the Gaussian layer itself.

Every number below is measured, not estimated, and each section names the
in-repo script that reproduces it. All lanes use the Home Scan LOD0 scene
(sha256 `29cee159…`, 21,497,908 gaussians) at 256×256 unless stated.

**2026-07-21 independent PR #68 L4 audit:** the intended static-map use is
valuable but narrower than a headline raster replacement. On new changing
Home cameras at 128² the OptiX tracer is **0.90× the custom raster at B1** and
**2.41× at B8**. Across 1,024 distinct cameras that all move, throughput grows
3.06× from B1 and saturates at about 423 img/s by B128. A second scene is
3.17× faster at B8 but fails raster fidelity on large, thin, oblique
Gaussians. The sparse-reflection path does reproduce: 68,618 rays cost 3.07 ms
at B32 × 256² on the L4 and change whole-frame throughput by about −0.2%.
See `benchmarks/pr68_independent_l4_validation_2026-07-21.json`. These results
do not contradict the original B32 4090 measurements; they bound where those
measurements generalize.

**2026-07-20 same-node validation:** every lane was re-measured on one rented
RTX 4090 (see `RESULTS.md` and
`benchmarks/rtx_compositing_validation_4090_results_2026-07-20.json`). The
OptiX tracer numbers reproduce within ~3%; the Warp lanes are now faster than
the tables below (kernel and Warp improvements since the original
measurements); the OVRTX lane runs ~1.45× slower than the table below on that
node (repeatable, also in the splats-only control). The per-lane tables keep
the original claimed values; treat the validation file as the same-hardware
record.

## Lane 1 — Isaac-RTX hybrid multi-robot scaling (per-env RTX, no pose cache)

`benchmarks/run_g1_hybrid_multirobot.py` — N G1 robots, one per env, each at
its own scripted-gait phase (`isaacsim_gaussian_renderer.g1_gait`, no RL
policy), each env camera framing its own robot with its own RTX render
product. 256×256, rt_subframes=8:

| envs | fps | images/s | frame ms | rtx ms | gaussian ms | physics ms | compositor ms |
|-----:|----:|---------:|---------:|-------:|------------:|-----------:|--------------:|
| 1 | 8.76 | 8.8 | 114 | 103 | 4.0 | 0.2 | 0.74 |
| 2 | 6.97 | 13.9 | 143 | 138 | 8.5 | 0.3 | 0.98 |
| 4 | 4.98 | 19.9 | 201 | 204 | 16.8 | 0.4 | 0.73 |
| 8 | 2.75 | 22.0 | 364 | 351 | 33.3 | 0.7 | 0.74 |
| 16 | 1.55 | 24.8 | 644 | 600 | 66.4 | 1.6 | 0.83 |
| 32 | 0.72 | 22.9 | 1397 | 1246 | 132.6 | 2.4 | 0.88 |

Least-squares fit: **rtx_ms = 36.6·envs + 56.2 (r² = 0.9976)** — the RTX robot
layer is measured-linear in env count, while the Gaussian layer stays ~4.1 ms
per env and the compositor stays constant (<1 ms). Hybrid throughput therefore
**saturates at ~22–25 images/s** regardless of env count; projecting the fit
gives 9.4 s/frame at 256 envs and 37.5 s/frame at 1024.

Conclusion: per-env RTX robot rendering does not vectorize. Any
massively-vectorized training loop must take robot-self-appearance out of the
per-env render (ego-view observations don't contain the robot, so the training
path needs no RTX at all).

**Compositor decision (carried forward):** the depth compositor is already
GPU-resident (`hybrid_compositor.composite_gaussian_and_mesh`, CUDA tensors)
and costs ~0.8 ms (<0.5% of the frame). A dedicated OVRTX GPU depth compositor
is not worth building — it would optimize a stage that is already negligible.

## Lane 2 — Pure-OVRTX multi-robot scaling (unified splats + meshes scene)

`benchmarks/run_ovrtx_multirobot.py` — same N-robot/N-camera layout, but OVRTX
ray-traces the Home Scan splats AND the G1 meshes in ONE scene (no Isaac RTX,
no compositor), with a splats-only control pass. 256×256, static robots:

| envs | splats-only ms | +robots ms | images/s | robot increment/env |
|-----:|---------------:|-----------:|---------:|--------------------:|
| 1 | 5.8 | 6.0 | 166 | 0.23 |
| 2 | 11.6 | 9.2 | 216 | −1.19 |
| 4 | 25.9 | 17.0 | 235 | −2.20 |
| 8 | 33.1 | 31.2 | 256 | −0.24 |
| 16 | 61.7 | 62.6 | 256 | 0.06 |
| 32 | 128.7 | 144.8 | 221 | 0.50 |

Fits: **with-robots 4.45 ms/env (r² = 0.994)**; splats-only 3.84 ms/env. The
robot increment is **indistinguishable from run noise (≤0.5 ms/env, sometimes
negative)** — the splat BVH dominates ray-tracing cost, so N additional robot
meshes are effectively free. Unified OVRTX sustains **~220–256 images/s with
robots in frame**, ~10× the Isaac-RTX hybrid's 22–25.

Caveats: robots were **static** (OVRTX 0.3 exposes no per-frame USD transform
update in this harness), so acceleration structures could cache — treat these
numbers as an optimistic lower bound versus walking robots. Timing is
`renderer.step` + CUDA sync (render only, no per-frame output-contract copies).

## Lane 3 — The wrapper-tax verdict: Warp mesh tracer gate

Thesis: 36.6 ms/env for a robot layer is framework tax, not ray-tracing cost.
Test: trace the actual robot triangles with a Warp CUDA **software** BVH — one
launch over envs × H × W, per-env rigid pose applied in ray space (zero
per-frame geometry/BVH rebuild, poses animated every frame)
(`benchmarks/run_warp_mesh_tracer.py`).

- Real G1 test asset (740 tris — a sparse rig; its ~112 visible px/env matches
  the Isaac-RTX matte exactly, confirming apples-to-apples): **0.25 ms flat**
  for 1–32 envs (marginal ~0.002 ms/env).
- Dense stress body (399k tris, 6,130 hit px/env — production-mesh class):
  **0.086 ms/env + 1.68 ms** (32 dense animated robots: 4.3 ms total, vs
  Isaac RTX's 1,246 ms for the same env count).

Verdict: **~99.7% of the Isaac-RTX per-env cost was wrapper, not silicon.** A
software tracer beats the 4 ms/env parity target ~40×.

## Lane 4 — Tier-1 custom hybrid: Gaussian + Warp tracer + compositor

`benchmarks/run_g1_hybrid_warp.py` — the Isaac-RTX layer replaced by the Warp
tracer; same batched Gaussian layer and CUDA depth compositor; 200k-tri robot,
per-env animated poses; no Isaac/OVRTX processes at all.

256², vs the other lanes at 32 envs:

| lane | frame ms | images/s | robot layer |
|:-----|--------:|---------:|------------:|
| Isaac-RTX hybrid | 1397 | 22.9 | 1246 ms |
| OVRTX unified (static robots) | 144.8 | 221 | ~free |
| **Warp hybrid (animated robots)** | **134.4** | **238.2** | **1.32 ms** |

Fit 4.169 ms/env (vs OVRTX 4.45). The custom hybrid **beats OVRTX's
static-robot best case with moving robots**, keeps the full output contract
(rgb/depth/alpha/semantic, per-env transforms, deterministic replay), and the
frame is now ~99% Gaussian layer — the bottleneck is back in our own kernel,
which is the right place for it.

128² (RL-relevant resolution), real Home Scan, robots in frame: sustains
**~418 images/s at 64–128 envs** (mesh layer 0.94 ms for all 128 robots).
Historical WS3 figures of ~10,000 img/s were the synthetic 10k-Gaussian scene;
on the real 21.5M Home Scan the Gaussian layer costs ~2.4 ms/env at 128², so
real-scene ego-view observation throughput is ~420 images/s at this
resolution — the Gaussian kernel, not robot rendering, is the optimization
frontier.

## Lane 5 — RT-core Gaussian tracer (OptiX)

`experiments/optix_tracer/` — standalone batched OptiX tracer over the flat
scene dump (`scripts/export_gaussians_bin.py`), following 3DGRT's algorithm
(custom AABB primitives, max-response hits, k-buffer march) with a
clamping-commit optimization: when the k-buffer is full and a candidate lies
beyond the k-th hit, commit it instead of ignoring — commitment clamps
traversal tmax, culling all farther candidates.

Measured on RTX 4090 (see `RESULTS.md` for the validated re-run):

| workload | closest-hit v1 | k-buffer + clamp |
|---|---:|---:|
| dense 32 envs × 256² | 49.6 ms / 42 Mrays/s | **13.6 ms / 154 Mrays/s** |
| dense 128 envs × 128² | 76.4 ms / 27 M | **19.6 ms / 107 M** |
| sparse 30k (secondary-ray shape) | 33 ms / 0.9 M | **8.0 ms / 3.7 M** |
| sparse 3M incoherent | 915 ms / 3.3 M | **219 ms / 13.7 M** |

Context: the table above uses the tracer's built-in synthetic cameras
(identity orientation, hit fraction 0.085), which is NOT the raster lanes'
camera set. The **ray-matched** comparison (`benchmarks/run_rt_vs_raster.py`,
identical cameras via `--cams`, RTX 4090, 3 repeats) measures the tracer at
**5.97× the raster on the standard exterior bundle** (12.2 vs 73.0 ms) and
**1.91× on interior ego views** (49.3 vs 94.1 ms) with the optimized defaults
(opacity-adaptive AABBs + KBUF=16; 4.2×/1.5× before the optimization sweep —
see RESULTS.md), with env-0 PSNR 35.8/40.1 dB proving image agreement — while
the raster additionally pays the full output contract and the tracer emits
RGBA only. The originally cited "~10×" (raster 132.5 ms vs tracer 13.6 ms)
mixed those camera sets and should not be quoted. Exact per-ray depth ordering (no global sort) and sparse
secondary-ray batches remain tracer-only capabilities. GAS: 39 ms build,
0.54 GB compacted. The unified hybrid mode (`--robot 1`) traces gaussians +
N chrome robots + live reflections in one launch.

The independent L4 audit additionally found and fixed integration defects in
the first reusable-library implementation: CUDA camera CPU round-trips,
batch-wide reuse of the first intrinsic matrix, non-default-stream races, and
2.34 GB retained after context close. After hardening, the full contract is
bitwise-clean inside a live Isaac process, the persistent OptiX context is
stable across an actual RTX frame, and Compute Sanitizer reports zero errors.

## Honest caveats (carried forward + current)

- **Lambert-only shading** in the Warp and OptiX robot layers by design — no
  MDL/GI. Visual parity against an RTX-rendered matte is a quality axis, not
  yet a measured gate.
- **Rigid whole-body articulation** in the Warp/OptiX robot layers (per-link
  instancing is the follow-up; the same ray-space transform trick applies per
  link with a small top-level AABB test).
- **OVRTX static-robot bound**: Lane 2 numbers are an optimistic lower bound —
  OVRTX 0.3 exposes no per-frame USD transform update in this harness.
- **OptiX tracer output contract: DONE** (2026-07-20 follow-up): depth,
  alpha, and semantic outputs match the raster's conventions (parity gates in
  `RESULTS.md`), the zero-copy torch path is bitwise-identical to the CLI,
  and the reflection pass is wired into the hybrid (+2.2 ms for 68.6k
  rays/frame at 32 envs). Remaining tracer follow-ups: per-link ray-space
  instancing (refit-free articulation) and OptiX SER for incoherent
  secondary rays.
- **Not a universal projected-raster replacement**: the independent Voxel51
  case fails alpha/depth/background equivalence despite being faster. The
  ray-volume method is accepted for the tested Home route and sparse ray work;
  projected rasterization remains the reference for general primary images.
- Nsight Compute hardware counters were blocked on the measurement host
  (`RmProfilingAdminOnly: 1`); occupancy/bandwidth are not reported.
- **Cross-node OVRTX variance**: the OVRTX and Isaac-RTX lanes measured
  ~1.45–1.7× slower on the 2026-07-20 validation node than on the original
  #51 node (same GPU model and driver); the custom CUDA/OptiX lanes did not
  shift. Absolute OVRTX/Isaac numbers are node-sensitive; only same-node
  comparisons are load-bearing.
- The coupled Isaac physics+render scaling study (synthetic scene; render-bound
  from batch 8, compositing 54.9% of render GPU time) was measured in PR #49;
  its harness scripts are not part of this branch — cite PR #49 for those
  numbers.

## Reproduce

```bash
source scripts/remote_env.sh
# Lane 1 (requires Isaac Sim + imported G1 usda):
/isaac-sim/python.sh benchmarks/run_g1_hybrid_multirobot.py \
  --g1-usda /workspace/assets/g1/g1_29dof/g1_29dof.usda --output outputs/hybrid_multirobot.json
# Lane 2 (requires ovrtx==0.3.0.312915 + G1 usda):
/isaac-sim/python.sh benchmarks/run_ovrtx_multirobot.py \
  --g1-usda /workspace/assets/g1/g1_29dof/g1_29dof.usda --envs 32 \
  --output outputs/ovrtx_multirobot.json
# Lane 3 (Warp; real mesh via scripts/extract_g1_mesh.py, or synthetic):
/isaac-sim/python.sh benchmarks/run_warp_mesh_tracer.py \
  --mesh-npz outputs/g1_mesh.npz --synthetic-tris 400000 \
  --output outputs/warp_mesh_tracer.json
# Lane 4:
/isaac-sim/python.sh benchmarks/run_g1_hybrid_warp.py \
  --mesh-npz outputs/g1_mesh.npz --synthetic-tris 200000 \
  --output outputs/g1_hybrid_warp.json
# Lane 5 (standalone binary, no torch env):
python3 scripts/export_gaussians_bin.py \
  --ply /workspace/datasets/home-scan-lod0.ply \
  --output /workspace/datasets/gaussians.bin
cd experiments/optix_tracer && bash build.sh && \
  ./gs_tracer --dump /workspace/datasets/gaussians.bin --mode dense \
    --envs 32 --width 256 --height 256 --kbuffer 1
# Ray-matched RT-vs-raster (identical cameras, exterior + interior sets):
/isaac-sim/python.sh benchmarks/run_rt_vs_raster.py \
  --ply /workspace/datasets/home-scan-lod0.ply \
  --dump /workspace/datasets/gaussians.bin \
  --tracer-bin experiments/optix_tracer/gs_tracer \
  --output outputs/rt_vs_raster.json
```
