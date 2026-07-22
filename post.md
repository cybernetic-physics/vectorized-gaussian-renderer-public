# Benchmarking a Batched Robotics Gaussian Renderer Against a FlashGS-Derived Port

*What changes when a 21.5-million-Gaussian map must produce RGB, alpha,
expected depth, and semantics for independently moving cameras in a
steady-state GPU-resident renderer microbenchmark?*

<!-- publication:status -->

The hypothesis is specialization. The
[FlashGS paper](https://openaccess.thecvf.com/content/CVPR2025/html/Feng_FlashGS_Efficient_3D_Gaussian_Splatting_for_Large-scale_and_High-resolution_Rendering_CVPR_2025_paper.html)
targets large, high-resolution conventional 3DGS views. Our production renderer
is built for a different shape of work: one very large,
static Gaussian map; many independently moving, low-resolution cameras in a
robotics-shaped workload;
GPU-resident inputs and outputs; and a four-output sensor contract.

That distinction matters. “Faster Gaussian renderer” is too broad to be true.
The claim we are testing is narrower and more useful:

> **A native batched renderer can outperform a per-camera pipeline that retains
> FlashGS's emission, sorting, range, and tile-compositor topology on dynamic,
> massively batched, low-resolution, robotics-shaped renderer work against the
> same target EWA contract, subject to every sampled view passing the pinned-gsplat gates.**

## What we actually built

The production path is `CustomCudaBackend`: an inference-only, screen-space EWA
Gaussian rasterizer exposed through `RendererService` and an Isaac Sim Kit
extension. It packs a shared scene once, accepts batched CUDA camera tensors,
reuses output and workspace allocations, and returns:

- linear or display-encoded degree-zero RGB;
- accumulated alpha;
- expected camera-space depth, `sum(T * alpha * z) / sum(T * alpha)`; and
- the semantic ID of the strongest individual `T * alpha` contributor.

The renderer uses several CUDA stages, not one giant kernel: projection,
intersection emission, CUB radix ordering, range construction, and fused
compositing. A logical render is one service call. Batches that fit the physical
workspace use one native submission; larger logical batches use a fixed,
predeclared number of ordered native submissions into the same output tensors.

The compact renderer trades an additional projection pass for a per-pixel
foreground representation, suppresses conservatively irrelevant background
intersections before sorting, and fuses robotics sensor outputs in a native
batched CUDA pipeline. It still projects every Gaussian independently for every
camera; there is no cross-camera projection reuse.

The headline configuration deliberately selects the compact 1×1-bin,
conservative depth-frontier path, nondeterministic global radix ordering, and a
capacity calibrated before timing. Those are supported backend options, not the
Kit extension's tile-16/adaptive-capacity defaults. Fixed-capacity sorting
orders the installed key capacity, including sentinel entries, so the result
must report installed capacity and utilization as well as generated records.
The timed headline path is not claimed to be bitwise deterministic; a separate
deterministic ordering mode is exercised by the replay gate.

The high-frequency Isaac path reads batched camera transforms from Fabric,
converts them on the active CUDA stream, and submits the same renderer service.
There is no Python render loop over cameras and no full output readback in the
steady-state path. On changed inputs, the default adaptive-capacity path
synchronously reads device counters. The dynamic emitted-prefix path may also
read one scalar count and synchronize the stream. The benchmark calibrates a
fixed-capacity path outside the measured region so neither read occurs there.

This is **not** a trainer, an RTX replacement, a general replacement for every
3DGS renderer, or the experimental OptiX tracer discussed later. Its present
boundaries are explicit:

- inference only;
- pinhole cameras in the production service;
- float32 geometry, cameras, RGB, alpha, and depth; int64 scene and semantic
  IDs;
- degree-zero color at render time (higher-order SH data may be stored but is
  not evaluated by this path);
- Gaussian layers only—USD meshes require a separately measured compositor;
- no claim of linear throughput scaling with batch size; and
- no claim that its current path is best for high-resolution single-camera
  rendering.

## The workload that changes the answer

The target observation is a batch dimension, not a host loop:

```python
outputs = service.render(
    camera_transforms,  # [B, 4, 4], independently changing every frame
    intrinsics,         # [B, 3, 3], per-camera pinhole K
    scene_ids,          # [B], shared or packed scenes
)

# rgb         [B, H, W, 3] float32
# alpha       [B, H, W, 1] float32
# depth       [B, H, W, 1] float32
# semantic_id [B, H, W, 1] int64
```

Pixels with zero accumulated alpha have positive-infinity expected depth.
Semantic output is `-1` when accumulated alpha is below the configured 0.01
sensor threshold.

The primary case uses the full Home Scan LOD0: 21,497,908 Gaussians at
128×128, with every camera pose changing on every measured frame. Projection
and rendered-frame caches are disabled. Home Scan has no ground-truth semantic
labels, so the benchmark uses eight deterministic spatial-octant labels. Those
labels exercise the semantic compositor; they are not a segmentation-quality
dataset.

Home Scan was created by Isaiah Sweeney ([@luxury_scans](https://superspl.at/user?id=luxury_scans))
and published on [SuperSplat](https://superspl.at/scene/3f89bbd3) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The benchmark uses a
content-addressed conversion of that asset to the canonical LOD0 PLY.

This workload is designed to test shared scene residency, bounded physical
batching, compact per-pixel ordering, and fused sensor outputs. Whether those
choices improve performance is a benchmark question, not an assumption. A 4K
single view exercises a different regime. This publication does not measure
that regime and does not extrapolate its low-resolution result to it.

## Why this is not an upstream FlashGS result

The FlashGS paper targets large, high-resolution conventional 3DGS views. A
local archived run predates this evidence protocol and lacks the complete
source, occupancy, invocation, and content-addressed provenance required here.
We therefore omit it entirely. This article supplies no high-resolution control
and cannot support a direct Custom-versus-upstream-FlashGS headline.

## The decisive matched comparison

The robotics comparison uses a pinned FlashGS-derived matched-contract port.
Independent review found that it is **not** a minimally adapted upstream
FlashGS build: matching the pinned-gsplat equation changes projection, support,
alpha, termination, and output behavior. It retains the recognizable FlashGS
emission, radix-sort, tile-range, and 16×16 compositor topology, but it must be
reported as a derived port. The retained topology derives from
[InternLandMark's MIT-licensed FlashGS](https://github.com/InternLandMark/FlashGS)
at pinned commit `cdfc4e4002318423eda356eed02df8e01fa32cb6`. Its changes are:

- load the same canonical Gaussian tensors once and keep them on the GPU;
- accept the same CUDA cameras and use the active PyTorch stream;
- preallocate and reuse outputs, CUB storage, and workspace;
- use float32 linear RGB instead of the example's quantized output;
- replace FlashGS's internal intersection-count readback with a fixed-capacity
  path derived from one hash-bound B1024 demand survey; every timed row also
  verifies zero overflow outside its timing samples;
- match support, alpha, compositing, background, depth, and semantic rules; and
- add alpha, expected depth, and strongest-contributor semantics in the existing
  compositor for the full-sensor lane.

That fixed-capacity mode differs from public FlashGS's count-readback path. It
trades the synchronization for sorting installed capacity, including unused
sentinel slots. It is part of the tested derived port, not evidence about the
performance of upstream FlashGS.

We do **not** add a new camera-batching kernel, projection cache,
rendered-frame cache, alternate sorter, or custom foreground-cutoff algorithm.
The port exposes the common batched `RendererService` API, but upstream
FlashGS does not have a true camera-batch operation: the adapter invokes its
native pipeline once per camera. That scheduling cost is part of the
replacement question; redesigning it would create a new renderer.

That makes this a renderer-pipeline comparison under a matched target contract,
not a measurement of speedup *caused by vectorization*. A direct within-Custom
test of the aggregate physical-batching treatment requires a matched control
over identical cameras. The required B128 P1-versus-P128 control can establish
only the effect of that physical submission schedule at B128; it does not
isolate a kernel mechanism or establish proportional batch scaling.

Both backends target, and are gated against, the pinned gsplat equation:
3.33-sigma maximum
support with opacity contraction, zero covariance epsilon, half-pixel centers,
`min(0.99, opacity * exp(-0.5 * q))`, `1/255` per-splat rejection,
front-to-back compositing, termination at `T <= 1e-4`, and black linear RGB.

The frozen matrix is:

| Property | Value |
|---|---|
| GPU | one Google Cloud NVIDIA L4, fixed UUID |
| Scene | Home Scan LOD0, 21,497,908 Gaussians |
| Resolution | 128×128 |
| Logical batch | 1, 8, 32, 64, 128, 256, 512, 1024 |
| Custom execution | direct through B256; P128×4 at B512; P128×8 at B1024 |
| FlashGS-derived execution | one P1 native pipeline submission per camera |
| Motion | every camera changes every measured frame |
| Timing | 8 excluded warmups + 100 synchronized measured frames |
| Contracts | RGB-only and RGB + alpha + expected depth + semantic ID |
| Correctness | every sampled view against pinned gsplat |
| Caches | projection and rendered-frame caches disabled |

Why the split schedule? The original direct-B512 capacity-only process tried
to install a backend workspace larger than the CUDA memory available after the
scene and outputs were resident. It failed before any timed row,
with the original 1.05 capacity headroom. We did not lower that headroom or
sweep chunk sizes after seeing the failure: the amended matrix uses the
already-declared P128 path. B512 remains one logical service call composed of
four ordered native submissions; B1024 uses eight. This is a hardware-
feasibility correction, not evidence that P128 is optimal or that throughput
scales linearly with batch size.

B128 is repeated in three fresh processes for both renderers and contracts.
Those run-level ratios, rather than the correlated camera frames inside one
process, provide a descriptive minimum/median/maximum spread—not a confidence
interval—and must agree on the winner in both CUDA-event and synchronized-wall
latency.

The evidence archive records both output contracts, CUDA-event and synchronized-
wall latency, images/s, megapixels/s, pre/post steady-state NVML process memory,
Torch peak allocated/reserved memory, visible Gaussians, generated intersections,
native submissions, host submission time, and short Nsight controls. The
strictly generated article displays the bound absolute GPU/wall latencies,
speedups, submission schedules, same-step range direction, and independent-run
direction; the exact remaining diagnostics stay in the hash-bound archive. A
missing row stays missing and is never interpolated.
Custom and FlashGS visible/intersection counters are backend-native diagnostics
with different culling and record units (including 1×1-bin records versus
16×16-tile records); they are never compared as if they measured the same work.

The primary matrix can establish a scoped empirical winner, but it cannot by
itself prove which custom feature caused that result. The separate B128
P1-versus-P128 control addresses only the aggregate physical schedule at that
logical batch. Causal language about the conservative foreground cutoff,
direct reprojection, fused sensor outputs, chunk size, or fixed-capacity sorting
requires matched ablations of those choices. Until those run, the article
describes them as design hypotheses—not measured speedup attribution.

The timed region is the steady-state `RendererService.render` call. The complete
frozen route is used to calibrate capacity before timing. The benchmark excludes
scene ingestion, online capacity discovery or growth, Isaac physics/Fabric
ingestion, policy or model work, and unseen online motion. Its camera population
is synthetic variation over one authored route, not a trace captured from a
deployed robot. Smaller batch contracts are deterministic prefixes/subsets of
the same frozen B1024 population, not independent workload draws. The
equal-weight geometric mean across the eight predeclared batch points is
descriptive, not a scaling exponent or deployment-weighted average.

<!-- publication:results -->

## Earlier same-equation work is context, not publication evidence yet

A separate NVIDIA L4 investigation compared the production EWA path with pinned
gsplat on independently changing cameras and full sensor outputs. Its tracked
summary records are useful engineering history, but the raw timing samples,
profiler exports, and complete artifact bundle are not currently present in the
publication package. This draft therefore withholds its rates, ratios, memory
figures, fidelity extrema, and kernel shares.

That investigation still motivates two hypotheses for the fresh, fully packaged
matrix: physical batching may reduce submission count without producing linear
throughput scaling, and projection plus ordering may dominate this dynamic
workload. Neither becomes a blog result until its current raw evidence passes
the same source, timing, fidelity, and checksum gates as the L4 matrix.

## Correctness is part of the result

Pinned gsplat is the correctness oracle for the screen-space EWA lanes. Every
sampled view must meet all thresholds, not just an average:

- RGB PSNR at least 40 dB;
- SSIM at least 0.995;
- LPIPS at most 0.01;
- alpha MAE at most 0.005;
- mean relative depth error at most 0.01; and
- foreground semantic agreement at least 0.999, measured over the union of
  pixels where either candidate or oracle alpha is at least 0.01.

The first corrected-support matrix exposed a useful failure in our pre-repair
FlashGS-derived matched-contract port: semantic mismatches clustered in sparse
tile-boundary tails. The port returned background while Custom matched gsplat
at the affected pixels. The missing alpha was material, so this was not only a
semantic tie or threshold artifact.

The exact cause is now established. In short tile tails, the optimized
compositor enabled its final cooperative feature load before assigning that
slot's offset. It therefore composited the first Gaussian's features under the
correct logical Gaussian ID. Source-grounded traces reproduced the mechanism
with the frozen cameras, pixels, and culprit IDs. The bad predicate was inherited
from the pinned upstream compositor source, and the matched-contract port repairs
the affected guard offsets. We did not demonstrate the
same visible failure in an upstream-faithful product workload, so this is not an
upstream FlashGS product-bug claim.

That diagnosis is a correctness result, not a performance result. A separate
post-repair verifier on an earlier revision passed the historical cases and
coordinates; its complete affected-row comparison against pinned gsplat and
Compute Sanitizer also passed. Exact-current reruns must bind the same repaired
native source and binary used by the fresh matrix before these become release
claims. Every interrupted timing remains diagnostic only.

## The OptiX tracer is a separate experiment

The repository also contains an experimental Gaussian ray-volume tracer. It can
express sparse secondary rays and camera models that the current production
service does not. It is not the production rasterizer and should not inherit
the rasterizer's claims.

Historical L4 summaries suggested that this tracer can be competitive on some
camera families while missing the conventional raster-fidelity gate. Their raw
package is likewise absent from this publication bundle, so this article does
not quote the old rates, ratios, or fidelity extrema. Earlier comparisons that
used RGBA-only tracer output against a full-output raster were not product
replacement results in any case.

The tracer's k-buffer, opacity-adaptive bounds, and traversal-clamping idea are
interesting research directions. The historical “clamping” contribution is
not isolated because that experiment changed both k-buffer behavior and
clamping. It needs a clamp-only ablation before it becomes a causal claim.

## Where OVRTX fits

OVRTX remains a valuable NVIDIA system baseline for Gaussian sensing, Kit/RTX
integration, and mesh composition. Its perspective renderer does not implement
the same screen-space EWA equation, so it cannot decide which conventional 3DGS
rasterizer is faster at equal image formation. Historical static/coherent OVRTX
numbers remain system measurements, not evidence that Custom universally beats
Gaussian rasterizers. The existing static-robot result also reflects our old
harness; it is not an OVRTX API limitation.

## Reproduce, inspect, disagree

At release, the content-addressed evidence archive, checksum manifest, retrieval
receipt, and external release envelope will be attached to the
`vgr-gcp-l4-matched-v4` GitHub release. This draft does not assert that the
release or anonymous retrieval checks are already complete. The scientific
archive is immutable; the separate envelope binds its hash to the final Git
merge, release tag, and anonymously verified download objects.

The numerical package must bind every result to:

- renderer, FlashGS, and gsplat commits;
- hashes of tracked source files and loaded native extensions;
- scene and camera-contract SHA-256 values;
- GPU UUID, driver, CUDA, PyTorch, compiler, and command;
- raw per-frame distributions and out-of-band capacity counters;
- unprofiled controls plus raw Nsight reports and exports; and
- immutable artifact keys, byte counts, checksums, and HTTP metadata.

The public [Home Scan visual flyby](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/side-by-side.mp4)
is a visual demo, not a numerical fidelity acceptance artifact. Its
[manifest](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/artifact-manifest.json)
and [camera contract](https://pub-243008be935848b6accaf262f04a7b82.r2.dev/flybys/home-scan/v1-a59ed5ade5b2/camera-path.json)
are immutable. New tracer and robot clips will not be linked until their MP4,
GIF, poster, storyboard, camera contract, ffprobe record, and checksum manifest
are published together.

## Publication gate

This article is ready only when all of the following are true:

- the corrected L4 B1–B1024 matrix is complete for RGB-only and
  full-sensor outputs;
- every scheduled row passes same-equation gsplat fidelity, strict memory,
  source, camera, and fairness audits;
- B1 and B1024 have short Nsight controls with unprofiled timing controls;
- all five B64 tile-boundary discrepancies have an exact reproduced cause and the
  repaired row passes, or the row is explicitly reported as non-equivalent;
- the exact-current Isaac/Fabric test ladder and stability soak pass;
- the public reproduction package contains raw results and immutable checksums;
- every linked media asset returns HTTP 200 with the declared metadata; and
- five independent reviews find no mixed hardware, scene, output, timing, or
  image-equation claim.

<!-- publication:conclusion -->

Primary references: the [FlashGS paper](https://openaccess.thecvf.com/content/CVPR2025/html/Feng_FlashGS_Efficient_3D_Gaussian_Splatting_for_Large-scale_and_High-resolution_Rendering_CVPR_2025_paper.html),
[gsplat rasterization documentation](https://docs.gsplat.studio/main/apis/rasterization.html),
[OVRTX documentation](https://nvidia-omniverse.github.io/ovrtx/), and NVIDIA's
[3DGRT project](https://research.nvidia.com/labs/toronto-ai/3DGRT/).
