# Kernel Architecture

Status: the first custom CUDA architecture is implemented and benchmarked.
The historical candidate gates below remain useful optimization history.
Overall project acceptance still depends on the OVRTX fidelity thresholds and
independent final audit.

## Implemented architecture

The accepted implementation direction is project-owned C++/CUDA with CUB for
radix sorting:

- Fused anisotropic projection, covariance, visibility culling, and compact
  visible-record emission across the whole active camera batch.
- Bounded, adaptively grown tile-intersection queues and dense 16x16 tile
  scheduling.
- Device tile grouping and segmented CUB radix ordering over emitted records.
- Optional deterministic segmented radix ordering by full depth bits and global
  Gaussian ID.
- One fused tile compositor for RGB, alpha, expected depth, and
  maximum-contribution semantic ID.
- One packed static scene allocation, scene-offset/scene-ID selection,
  per-camera environment transforms, active subsets, and reusable outputs.
- No measured-loop allocation and explicit device overflow counters.
- Precomputed object-space covariance `[G, 6]` while preserving canonical
  scales and quaternions at the public scene boundary.
- Classic screen-space covariance dilation by default, plus an opt-in
  `antialiased` mode that compensates opacity for the dilation. The two modes
  are separate fidelity contracts; antialias compensation is rejected for the
  exact-ray path because that path does not evaluate the dilated 2D Gaussian.
- A dense pixel-tile path with conservative grouped optical-thickness cutoff,
  exact retention of the threshold bucket, and global CUB ordering.
- Optional projection-coherence reuse for unchanged scene/camera inputs.
  This reuses projected records and depth ordering only; all downstream
  scheduling, sorting, compositing, and output writes remain per-frame.

The implementation lives in
`src/isaacsim_gaussian_renderer/native/renderer_cuda.cu` with its binding in
`renderer.cpp` and lifecycle/backend ownership in `cuda_backend.py`.

## Evidence Read So Far

Project controls require RGB, depth, alpha, and semantic output on identical
scene/camera data, with steady-state GPU-resident outputs. The implemented
`isaacsim_gaussian_renderer.RendererService` owns the direct Isaac Sim process
boundary, scene IDs, per-environment transforms, active subsets, cadence,
preallocated GPU outputs, project-owned reusable workspace, and native
C++/CUDA renderer dispatch.

## Measured Home Scan specialization

The full 21.5M-Gaussian Home Scan exposed a different bottleneck from the
synthetic 16x16-tile matrix: enormous pixel-intersection work after projection.
The accepted dense path therefore uses:

1. scene-wide precomputed object-space covariance;
2. one fused projection/visibility/depth-bucket-count kernel;
3. warp-aggregated depth-bucket count and scatter;
4. 1x1 pixel tiles;
5. conservative optical-thickness accumulation in 32-bucket groups over 1,024
   total buckets;
6. a cutoff that retains every Gaussian in the threshold bucket and rejects
   only strictly farther buckets;
7. safe near/far bounds from the complete scene bounding sphere;
8. optional projection/depth-order reuse for a coherent static sensor.

The dynamic path measures 7.274 ms. The coherent path measures 4.650 ms versus
26.016 ms OVRTX and uses 25.66 percent of OVRTX driver memory.

Nsight proves the cache boundary: projection, prefix, and scatter run once in a
22-render capture; tau, cutoff, binning, sorting, tile-range discovery, and
compositing run for every render. External zero-copy or inference-mode writes
must call `invalidate_projection_cache()`.

Source inspection was limited to read-only remote paths:

- `/workspace/src/gsplat` at `77ab983ffe43420b2131669cb35776b883ca4c3c`.
- OVRTX at `29d11037fbcaed0f0f53e7f32d17bd0486fd453b`, with the
  runtime installed under `/workspace/tools/ovrtx-0.3.0`.

Grounding facts from `gsplat`:

- The public `rasterization()` path calls one C++ orchestrator named
  `rasterization_3dgs`, returning render colors, alphas, projection
  intermediates, tile intersections, flattened IDs, and offsets.
- `projection_ewa_3dgs_packed` performs a two-pass visibility compaction:
  per-block counts, prefix sum, then writes `[nnz]` batch IDs, camera IDs,
  Gaussian IDs, radii, means2d, depths, and conics.
- `intersect_tile` performs a first pass for `tiles_per_gauss`, a `cumsum`, a
  second pass for `isect_ids` and `flatten_ids`, then optionally radix-sorts by
  packed `(image, tile, depth)` keys. Packed plus segmented sort is explicitly
  rejected in current source.
- `rasterize_to_pixels_3dgs` launches one CTA per image tile. Its dense path
  supports tile size 16 with 256 threads or tile size 4 with 16 threads, blends
  front-to-back, stops pixels when transmittance reaches the threshold, and
  emits color, alpha, and last contributing intersection ID.
- Current channel dispatch requires compile-time supported channel counts.

Grounding facts from OVRTX:

- Camera outputs are RenderVars such as `LdrColor`, `HdrColor`, `DepthSD`, and
  `SemanticSegmentation`; semantic output is numeric IDs with separate metadata.
- Mapped outputs expose DLPack tensor slots. Image tensors are channel-last
  `[height, width, channels]`.
- CUDA mapping is available via `OVRTX_MAP_DEVICE_TYPE_CUDA`; CUDA array mapping
  is image-only and may avoid an extra image copy on that side of the baseline.

## Non-Negotiable Contract

The custom renderer is an inference-only CUDA extension behind the direct
Isaac Sim `RendererService` backend boundary. It must return these GPU tensors
without CPU copies in the steady-state loop:

```text
rgb         [B, H, W, 3] float16 or float32, linear contract decided by baseline
depth       [B, H, W, 1] float32
alpha       [B, H, W, 1] float32
semantic_id [B, H, W, 1] int32 or uint32
```

Inputs are preloaded and preallocated on one CUDA device:

```text
means          [G_total, 3] float32
quats          [G_total, 4] float32, WXYZ at CUDA boundary
scales         [G_total, 3] float32, activated scale
opacities      [G_total]    float32, activated opacity
features       [G_total, F] or [G_total, K, 3]
semantic_ids   [G_total]    int32 or uint32
scene_offsets  [S + 1]      int64 or int32

viewmats        [B, 4, 4] float32, world-to-camera
intrinsics      [B, 3, 3] float32
env_xforms      [B, 4, 4] float32
scene_ids       [B]      int32
active_ids      [A]      int32, optional subset
background      [B, 3]   float32
```

`src/isaacsim_gaussian_renderer/layout_contract.py` provides CPU-side
scaffolding for these shape, contiguity, device, offset, and dtype constraints.
It is a boundary test, not a renderer.

## Canonical SoA Scene Representation

Use one concatenated structure-of-arrays allocation per attribute. Multi-scene
support is represented by `scene_offsets`, not by Python lists or duplicated
static tensors. For camera `b`, the active Gaussian range is:

```text
scene = scene_ids[b]
start = scene_offsets[scene]
end   = scene_offsets[scene + 1]
```

Shared scene is a special case with all `scene_ids = 0`. Per-environment
transforms are applied at projection time as `viewmat[b] * env_xform[b] *
mean_world`, leaving static Gaussian data shared.

Initial storage decision:

- Keep `means`, `quats`, `scales`, `opacities`, `features`, and `semantic_ids`
  canonical and immutable during rendering.
- Materialize optional derived covariance `[G_total, 6]` only if projection
  profiling proves quaternion-scale conversion is a bottleneck and memory still
  satisfies the 80 percent RTX criterion.
- Keep semantic IDs as a separate array to avoid widening the color feature
  path and to allow nearest/front-most semantic write during compositing.

Rejected for now:

- AoS records for all Gaussian attributes. They simplify single-record loads but
  force unnecessary attribute traffic for culling and semantic/depth-only paths.
- Duplicating static scene data per environment. This violates the required
  per-environment transform support and likely fails Home Scan memory headroom.

## Candidate Pipeline

The candidate is a five-subsystem CUDA pipeline with reusable workspace:

1. `project_compact`: project, compute covariance/conic, frustum/radius/opacity
   cull, and compact visible `(camera, gaussian)` pairs.
2. `bin_tiles`: compute macro-tile and tile coverage counts, prefix sums, and
   write tile intersection keys plus Gaussian references.
3. `order_tiles`: order intersections front-to-back within each image tile.
4. `composite_rgba_depth_semantic`: blend RGB, alpha, and depth, write semantic
   ID from the same contributor rule, and early-exit saturated pixels.
5. `marshal_outputs`: optional dtype conversion only when Isaac buffer dtype
   differs from the custom native output.

Each subsystem must be independently timed with CUDA events and NVTX ranges.
No subsystem is accepted without work counters: visible pairs, tile
intersections, active tiles, sorted keys, contributing samples, saturated
pixels, and temporary bytes.

## Projection And Covariance

Projection should start from the source-proven `gsplat` split:

- Pinhole camera first; other camera models are out of scope unless RTX parity
  requires them for selected benchmark scenes.
- Use quaternion+scale to covariance in-kernel for the first implementation.
- Emit compact packed buffers: `camera_ids`, `gaussian_ids`, `radii`,
  `means2d`, `depths`, `conics`, and projected opacity.
- Include `scene_ids` and `scene_offsets` in the indexing calculation so
  `gaussian_ids` are global canonical IDs.

Measurement gate P1:

- Accept fused quaternion-scale projection only if projection is under 15
  percent of total custom frame time at B=64, 128x128 and B=256, 256x256.
- If projection exceeds 15 percent, test precomputed covariance `[G, 6]`.
  Accept precompute only if it improves total frame time by at least 8 percent
  and persistent memory still leaves peak custom memory at or below 80 percent
  of RTX baseline.

## Visibility Compaction

Start with two-pass compaction because it is source-proven in `gsplat` and gives
exact workspace sizing. Replace `item()`-style CPU scalar reads with device-side
capacity checks and preallocated workspace metadata before steady-state timing.

Workspace buffers:

```text
visible_counts       [A * blocks_per_scene_row] int32
visible_offsets      [A * blocks_per_scene_row] int32
camera_ids           [max_visible] int32
gaussian_ids         [max_visible] int32 or int64 for G > 2^31-1
radii                [max_visible, 2] int16 or int32, measured
means2d              [max_visible, 2] float32
depths               [max_visible] float32
conics               [max_visible, 3] float32
projected_opacity    [max_visible] float32
```

Measurement gate C1:

- Two-pass compaction is acceptable if allocation-free steady state has no CPU
  synchronization and compaction plus prefix scan stays below 12 percent total
  frame time.
- If not, test fixed-capacity per-camera queues with overflow detection and
  scene-dependent headroom. Reject them if overflow handling complicates
  deterministic mode or wastes more than 10 percent peak memory on Home Scan.

## Macro And Tile Binning

Tile binning should preserve `gsplat`'s key idea: encode image/tile/depth and
use offsets for each tile. The custom renderer adds a macro-tile pre-pass for
large scenes and high batch counts:

```text
macro tile: 64x64 pixels, used for coarse rejection and scheduling
fine tile:  16x16 pixels initially, used for compositing CTAs
```

At 512x512, each image has 32x32 fine tiles and 8x8 macro tiles. At B=256 this
is 262,144 fine tile work items before empty-tile rejection, so active-tile
compaction is required for sparse views.

Measurement gate B1:

- Use fine-tile-only binning if active fine tiles exceed 60 percent for the
  representative scene and macro rejection saves less than 5 percent total time.
- Use macro+fine binning if it reduces tile intersections or raster CTAs by at
  least 15 percent and adds less than 5 percent frame time in overhead.

## Ordering Alternatives

The renderer must composite front-to-back consistently with the fidelity
contract. Three ordering alternatives remain open:

1. Global radix sort of all `(image, tile, depth)` keys, matching current
   `gsplat` behavior.
2. Segmented per-image or per-tile radix sort to reduce sort range and improve
   cache locality.
3. Per-tile approximate bucket ordering, only if fidelity metrics prove parity.

Measurement gate O1:

- Keep global radix sort if it is below 20 percent total time and fidelity
  passes.
- Accept segmented sort only if it improves total frame time by at least 8
  percent without increasing launch count or temporary bytes enough to violate
  memory target.
- Reject approximate bucket ordering unless RGB PSNR, SSIM, LPIPS, alpha, depth,
  and semantic thresholds all pass on representative and worst-case views.

Rejected now:

- No-sort blending. Alpha compositing is order-dependent and cannot be proposed
  as an acceptance architecture without fidelity evidence.
- Sorting only by Gaussian ID. It is not a depth order.

## Fused RGB, Depth, Alpha, And Semantic Compositing

The compositing kernel should extend the source-proven per-tile front-to-back
CTA model:

- One CTA per active fine tile initially.
- Tile size 16 with 256 threads as the first candidate.
- Load Gaussian screen-space mean, conic, opacity, depth, RGB/SH result, and
  semantic ID in batches.
- For each pixel maintain transmittance `T`, RGB accumulator, weighted depth
  accumulator, alpha, last/front contributor, and semantic winner.
- Stop a pixel when `T <= TRANSMITTANCE_THRESHOLD`, matching the source pattern.

Depth rule and semantic rule must be aligned with RTX reference before final
selection:

- Candidate depth A: expected depth, weighted by `alpha * T`.
- Candidate depth B: first/median visible hit depth.
- Candidate semantic A: semantic of highest `alpha * T` contributor.
- Candidate semantic B: semantic of first contributor that passes the alpha
  threshold.

Measurement gate R1:

- Choose the depth/semantic rule that matches RTX outputs. A rule cannot be
  selected from performance alone.
- Accept fused output only if it is faster than separate RGB/depth/semantic
  kernels by at least 10 percent and meets all fidelity thresholds.

## Reusable Workspace

No steady-state custom run may allocate temporary tensors in the measured loop.
The extension owns a `RendererWorkspace` sized during warmup:

```text
capacity_visible
capacity_intersections
capacity_active_tiles
buffers for scans, sort temp storage, offsets, keys, values, counters
```

If a frame exceeds capacity, default mode reads the exact device counters,
grows both dependent buffers with bounded headroom, invalidates projection
state, and retries. The retry count and hard workspace-byte ceiling are finite.
If either bound is exhausted, every mode raises a structured error and marks
the attempted outputs invalid rather than silently returning truncated work.
Identical inputs that have already passed validation remain asynchronous;
changed inputs require a counter read because Torch owns the allocations.

Measurement gate W1:

- Workspace is accepted when allocator peaks stop changing during warmup and
  measured iterations report zero allocation delta.
- Workspace capacity policy must be reported in each JSON/CSV result.

## CUDA Graph Capture

CUDA graph capture is optional and gated because dynamic visible/intersection
counts can fight fixed graph shapes.

Candidate graph boundary:

- Capture the fixed sequence after workspace capacities are fixed.
- Keep scalar counters in device memory.
- Avoid host reads inside the captured section.
- Use fixed maximum-capacity buffers; kernels consume actual counts.

Measurement gate G1:

- Enable graph replay only if it improves p50 frame time by at least 5 percent
  or reduces CPU launch overhead by at least 20 percent with unchanged fidelity.
- Disable graph replay for configurations with frequent capacity changes or
  selected-environment subsets that alter launch topology unless a padded
  captured path proves faster.

## Deterministic Mode

Deterministic mode must be a first-class runtime flag.

Required behavior:

- Stable sort keys include deterministic tie-breakers: image, tile, quantized
  depth or raw sortable depth, and global Gaussian ID.
- No unordered floating-point atomics in color/depth/alpha compositing.
- Fixed tile traversal and contributor traversal.
- Same inputs produce bitwise-identical or documented tolerance-identical
  tensors across runs on the same host.

Measurement gate D1:

- Deterministic mode is accepted only after three runs reproduce within the
  project 5 percent throughput criterion and output deltas satisfy fidelity
  thresholds against itself and RTX.
- If deterministic mode costs more than 15 percent throughput, ship it as a
  selectable mode and keep default mode separate in result rows.

## PyTorch And Warp Zero-Copy

The Python boundary should expose custom outputs as ordinary CUDA tensors and
write directly into Isaac/Warp proxy tensor storage when available.

Rules:

- Accept `torch.Tensor` inputs that are contiguous, on the active CUDA device,
  and match `layout_contract.py`.
- Use `data_ptr` and current CUDA stream in C++/CUDA.
- For Warp/Isaac buffers, use the `.torch` proxy view already present in the
  local renderer path.
- Do not require DLPack conversion in the steady-state custom path. DLPack is
  relevant for OVRTX baseline output mapping and independent verification.

Measurement gate Z1:

- Python binding overhead must be less than 3 percent of frame time at B=64,
  128x128 and less than 1 percent at B=256, 256x256.
- PyTorch profiler must show no CPU output copy and no per-frame allocator
  churn in the measured loop.

## Precise Custom Extension Boundaries

Create a separate extension package, tentatively
`isaacsim_gaussian_renderer_cuda`, with these inference-only entry points:

```text
create_workspace(scene_layout, max_batch, max_resolution, options) -> handle
resize_workspace(handle, capacity_plan) -> stats
render_forward(
    handle,
    scene_tensors,
    viewmats,
    intrinsics,
    env_xforms,
    scene_ids,
    active_ids,
    background,
    outputs,
    options,
) -> counters
destroy_workspace(handle)
```

Do not expose projection, sorting, or raster internals as the public Isaac
contract. Expose them only as debug/profiling functions for subsystem tests.

Custom extension owns:

- CUDA kernels and C++ binding.
- Workspace capacity and temp buffers.
- NVTX naming and counters.
- Deterministic-mode switch.

Python/Isaac owns:

- Stage/asset loading and conversion into canonical tensors.
- Camera convention conversion.
- Renderer factory integration.
- Output buffer registration and render cadence.
- Benchmark result serialization.

Out of scope for this milestone:

- Training/backward kernels.
- Multi-GPU.
- Mesh/robot compositing.
- RTX secondary lighting emulation beyond Gaussian reconstruction/compositing
  parity required by `GOALS.md`.

## Quantitative Performance And Memory Model

Symbols:

```text
B  active cameras
G  total Gaussians in selected scenes
V  visible camera-Gaussian pairs after projection
T  tile intersections
P  pixels = B * H * W
C  average contributing Gaussian samples evaluated per pixel
F  feature channels used for RGB/SH
```

Dominant work:

```text
projection/cull      O(B * G_scene_visible_test)
compaction scans     O(B * G / block)
tile enumeration     O(V * average_tiles_per_visible)
ordering             O(T) radix passes or segmented equivalent
compositing          O(P * C) with early exit
```

Persistent scene memory, approximate bytes:

```text
means          12 * G
quats          16 * G
scales         12 * G
opacities       4 * G
semantic_ids    4 * G
features        4 * F * G, or 12 * K * G for SH
scene_offsets   8 * (S + 1)
```

For Home Scan LOD0 at 21,497,908 Gaussians, excluding features:

```text
means                 246 MiB
quats                 328 MiB
scales                246 MiB
opacities              82 MiB
semantic_ids           82 MiB
base subtotal         984 MiB
RGB features          246 MiB
SH degree 3 features 3.84 GiB
optional covar[6]     492 MiB
```

Dynamic workspace, approximate bytes:

```text
visible records:
  camera_id int32           4 * V
  gaussian_id int32/int64   4-8 * V
  radii int32x2             8 * V
  means2d float2            8 * V
  depth float               4 * V
  conic float3             12 * V
  opacity float             4 * V
  total                    44-48 * V

tile intersections:
  key uint64                8 * T
  value int32               4 * T
  sorted key/value         12 * T
  offsets int32             4 * B * tile_h * tile_w
  sort temp                 measured CUB/device-radix temp

outputs at float32/int32:
  rgb                      12 * P
  depth                     4 * P
  alpha                     4 * P
  semantic                  4 * P
  total                    24 * P
```

At B=256, 512x512, output tensors alone are about 1.50 GiB. If `V=20M` and
`T=100M`, visible records are about 0.9 GiB, key/value double buffers are about
2.2 GiB before sort temp, and offsets are about 1 MiB. This makes the Home Scan
memory result highly sensitive to visibility and tile-intersection counts; the
final capacity plan is derived from measured counters and recorded headroom,
not silently assumed from a scene heuristic.

Throughput target model:

```text
custom_frame_ms <= RTX_frame_ms / 5
custom_peak_mem <= 0.8 * RTX_peak_mem
```

Per-subsystem budget will be assigned after baseline/profile publication. Until
then, provisional custom budget fractions are:

```text
projection + compaction   <= 25%
binning + ordering        <= 30%
compositing               <= 35%
binding/output/counters   <= 10%
```

These are planning budgets only and must be replaced with measured gates.

## Decision Gates

| Gate | Decision | Required evidence |
|---|---|---|
| A0 | Accept architecture | Fair RTX and `gsplat` baselines plus profiles exist |
| P1 | Projection storage | Fused vs precomputed covariance timings and memory |
| C1 | Compaction strategy | Sync-free steady-state timing and allocation trace |
| B1 | Macro tiling | Active-tile reduction and total-frame impact |
| O1 | Ordering | Sort time, memory, and fidelity on worst-case views |
| R1 | Depth/semantic rule | RTX parity metrics for RGB/depth/alpha/semantic |
| W1 | Workspace | Zero allocation delta after warmup |
| G1 | CUDA graph | p50 and launch-overhead improvement |
| D1 | Determinism | Three-run output and throughput reproducibility |
| Z1 | Zero-copy binding | PyTorch profiler shows no CPU copy or allocator churn |

## Implementation Subsystems

Split implementation work into measurable PRs:

1. `layout-and-binding`: tensor validators, extension skeleton, stream handling,
   output aliasing, and no-op counters.
2. `projection-compaction`: shared-scene projection, culling, compact visible
   queue, counters, and reference parity tests against `gsplat` projection.
3. `scene-id-transform`: `scene_offsets`, `scene_ids`, per-environment
   transforms, active subset, and selected-environment tests.
4. `tile-binning`: fine-tile count/enumerate, offsets, deterministic key format,
   and counters.
5. `ordering`: global radix sort first, then segmented alternative behind a
   flag.
6. `fused-composite`: RGB/depth/alpha/semantic output in one pass with selectable
   depth and semantic rules for fidelity triage.
7. `workspace`: preallocation, capacity planning, overflow reporting, and
   allocator-peak tests.
8. `cuda-graph`: optional captured replay path with profiler proof.
9. `deterministic-mode`: tie-breaks, reproducibility harness, and result flags.
10. `benchmark-integration`: JSON/CSV fields for all counters and toggles.

Each subsystem must land with at least one measurable counter or benchmark row.
Synthetic microbenchmarks may diagnose behavior but cannot satisfy acceptance.

## Rejected Alternatives

- Finalizing an architecture from source inspection alone.
- Omitting semantic output or computing it in a later CPU pass.
- Returning CPU NumPy/images from custom rendering.
- Relying on generated, interpolated, cached, or pre-rendered frames.
- Comparing only against unmodified `gsplat`.
- Treating packed `gsplat` limitations as permanent custom limitations; packed
  plus segmented sort remains a candidate only after a custom offset design is
  measured.
- Using CUDA array output as the custom primary contract. It is useful for OVRTX
  baseline mapping, but Isaac/Warp/PyTorch integration needs ordinary GPU tensor
  storage for RL observations.
