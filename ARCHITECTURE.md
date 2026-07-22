# Architecture

## System boundary

The renderer is a headless GPU service and Kit extension inside the Isaac Sim
Python process:

```text
SimulationApp / Isaac Sim
  physics + USD stage + Fabric state
                 |
                 v
Isaac Sim Kit extension
  batched Fabric camera poses + cached USD intrinsics
  scene IDs + environment transforms + active subset
                 |
                 v
project-owned vectorized GPU kernel pipeline
  persistent scene data + reusable queues/workspace
                 |
                 v
GPU-resident RGB + depth + alpha + semantic tensors
                 |
                 v
Python/RL/Replicator-style consumer or optional later adapter
```

Isaac Sim owns simulation stepping and scene state. The extension never
advances physics implicitly. A Kit viewport and full Hydra RenderDelegate are
optional secondary consumers, not requirements for the fast sensor path.

Isaac Lab is outside this boundary and must not be imported by the core package
or extension.

## Canonical data model

Static Gaussian data uses structure-of-arrays storage:

```text
means              [G, 3]
scales             [G, 3]
rotations          [G, 4]       WXYZ
opacities          [G]
features           [G, K]       RGB or spherical harmonics
semantic_ids       [G]
scene_offsets      [S + 1]
```

Per-render data is not replicated per Gaussian:

```text
camera_view         [B, 4, 4]
camera_intrinsics   [B, 3, 3]
environment_xform   [B, 4, 4]
scene_ids           [B]
active_camera_ids   [A]         optional subset, A <= B
```

Outputs:

```text
rgb                 [B, H, W, 3]
depth               [B, H, W, 1]
alpha               [B, H, W, 1]
semantic_id         [B, H, W, 1]
```

All steady-state inputs, work queues, and outputs remain GPU-resident.

## Implemented custom CUDA pipeline

The production backend is
`isaacsim_gaussian_renderer.CustomCudaBackend`. One logical render processes
the selected camera/environment batch on the active PyTorch CUDA stream. It is
one native submission when the batch fits the configured physical workspace;
an explicitly bounded larger batch uses a fixed number of ordered native
submissions into the same logical outputs.

The winning dense-scene path is the direct compact screen-space pipeline:

1. Apply per-environment transforms and project every selected
   camera/Gaussian pair in one GPU grid.
2. Accumulate per-pixel optical thickness into a small fixed depth-bucket
   grid, then compute a conservative foreground cutoff per pixel.
3. Reproject in a second pass and emit only qualifying
   `(global_pixel, float_depth_bits) -> gaussian_id` intersections. No
   per-camera projected-record array is materialized.
4. Globally radix-sort the emitted `(pixel, depth)` keys with CUB on the GPU,
   then build the contiguous range for each pixel.
5. On every render, launch one compositor CTA per selected pixel. The
   compositor cooperatively reprojects the sorted Gaussian IDs and fuses
   front-to-back RGB, alpha, expected depth, and maximum-contribution semantic
   output.
6. Write all four outputs directly into preallocated CUDA tensors owned by the
   service or supplied by the caller.

With projection coherence enabled, only the per-pixel depth cutoffs, sorted
Gaussian-ID intersections, and dense pixel ranges persist across matching
camera/scene inputs. The compositor still reprojects qualifying Gaussians and
rewrites RGB, depth, alpha, and semantic tensors on every render. Rendered
images are never cached.

The backend also retains the general projected-record/tile path for
deterministic tests and less dense workloads:

1. Fused anisotropic projection/culling into bounded visible-record storage.
2. Tile/Gaussian intersection generation.
3. Device tile grouping and segmented CUB radix ordering over emitted records.
4. Optional deterministic segmented ordering by full float depth bits and
   global Gaussian ID.
5. Fused tile compositing into the same output contract.

These screen-space paths implement the conventional 3DGS
equation: project each 3D covariance to a 2D EWA conic, evaluate the conic at
pixel centers, and alpha-composite ordered contributions. They are the
production fast path and match pinned `gsplat`; they are not claimed to use
the same Gaussian equation as OVRTX perspective. “Match” is an empirical
fidelity claim that must be re-established for the exact publication source;
the source defines the intended equation. The compact/global-radix fast
path runs with `deterministic=False`; deterministic tie-breaking is a separate,
slower segmented-ordering mode.

An experimental exact-ray mode evaluates the minimum Mahalanobis distance
between each camera ray and the full 3D Gaussian, then composites the resulting
contributions as probabilistic first hits. That model closely matches
temporally converged OVRTX perspective RGB, alpha, and depth, but remains a
diagnostic path. It has not completed the production Home Scan, matrix,
stability, performance, or independent-verification program, and its semantic
agreement remains below the strict threshold.

Scene attributes use multiple packed structure-of-arrays CUDA tensors. Shared scenes
are not copied per camera or environment. Visible/intersection storage starts
from an estimate and grows through a bounded fail-closed retry when device
counters report overflow. CUB temporary storage grows with it. Growth completes
before measured steady state; unchanged validated inputs avoid another host
counter read, while changed inputs synchronize once because host-owned Torch
storage cannot be resized from a device counter alone.

Macro-tile rejection, CUDA graph replay, mixed-precision scene storage, and a
persistent scheduler remain focused future experiments. They are not required
for the measured acceptance result and are not claimed as implemented.

Stock `gsplat` and OVRTX/RTX remain baseline and correctness backends only.
They cannot perform the dominant projection, scheduling, rasterization, or
compositing work in the accepted custom result.

## Isaac Sim integration

`isaacsim_gaussian_renderer.RendererService` is the current integration
boundary. It is independent of Isaac Lab and stock `gsplat`; project-owned
custom kernels are injected through the `GaussianRendererBackend` protocol.

The first-class integration is a standalone headless sensor renderer loaded as
an Isaac Sim Kit extension. Its current Python-facing service is:

- `initialize(stage, device)`
- `load_scene(scene_id, *, means, scales, rotations, opacities, features, semantic_ids)`
- `update_scene_transforms(scene_ids, transforms)`
- `prepare_outputs(batch)`
- `render(camera_transforms, intrinsics, scene_ids, outputs=None, active_camera_ids=None)`
- `synchronize()`
- `shutdown()`

The extension reads high-frequency batched world transforms from USDRT/Fabric
with one reusable GPU `SelectPrims` query and one Warp conversion kernel on the
current PyTorch stream, caches low-frequency camera intrinsics from USD, and
submits one vectorized GPU work sequence for the selected camera/environment
set. It supports shared and multi-scene registration, scene IDs, active
subsets, render cadence, deterministic mode, reusable output/workspace
ownership, and counters for visible Gaussians and tile intersections.

The steady-state service has no per-camera host render loop, no full output or
scene-tensor copies to the CPU, and no per-frame CUDA storage allocation after
validated capacity and output preparation. The calibrated
fixed-capacity path also avoids device-to-host counter reads. On changed inputs,
the default adaptive-capacity path synchronously reads device counters; on a
dynamic projection-cache miss, the emitted-prefix sort path may also copy one
scalar count to the host and synchronize the current CUDA stream. Benchmark
manifests report which capacity and ordering paths ran. Integration overhead is
reported separately from kernel time and included in end-to-end throughput.

The Fabric transform source keeps a persistent CUDA `torch.Tensor` and fills it
with one Warp kernel reading the selected Fabric arrays; it is not a direct
Torch view of Fabric memory. No CPU copy is involved. Camera topology setup may
iterate over USD paths once, but the high-frequency transform read has no
runtime camera loop.

Mesh/robot compositing is a future extension boundary and must not leak into
the core Gaussian queue representation. RTX may temporarily render dynamic
meshes for a separately measured hybrid compositor, but those costs and
outputs cannot be hidden in the Gaussian headline benchmark.
