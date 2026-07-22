# Gaussian LiDAR contract

`GaussianLidarService` is a separate, explicitly constructed sibling of
`RendererService`. The Kit extension creates it only through
`create_lidar_service(...)`. The existing camera `create_service()` path does
not import the LiDAR CUDA backend, load or compile its native extension, build
an acceleration structure, allocate a LiDAR workspace or output, launch a
LiDAR kernel, or synchronize for LiDAR.

## Supported scope

The production mode is `gaussian_surface`: static Gaussian environments,
batched moving sensor poses, one shared sensor-local ray pattern and one or two
returns. The service consumes and produces CUDA tensors directly and does not
use Isaac Lab. Dynamic USD robot or mesh geometry, scene motion during a scan,
motion-distorted geometry, rolling scene transforms, material-calibrated NIR
response, weather and multipath are outside the initial contract.
No `gaussian_volume` diagnostic is exposed in this release, and the service
never substitutes a volumetric or camera-rendering path for `gaussian_surface`.

Every public call enqueues all and only that invocation's work on PyTorch's
current CUDA stream before returning. There are no private streams, host
workers, pending queues, call-count paths, pointer/shape-keyed result caches,
future-call batching, or cached outputs. Repeated identical inputs still clear
and rewrite every output and trace every selected ray. Legal caches are limited
to explicitly revised static-scene LBVHs and caller-owned immutable scan
patterns.

## Inputs and outputs

Inputs are normalized sensor-local directions `[R, 3]` (float32), time offsets
`[R]` (int32 or int64), sensor-to-world transforms `[B, 4, 4]` (float32 rigid),
scene IDs `[B]` (int64), optional scene-to-world transforms `[B, 4, 4]`,
optional active sensor IDs and an explicit return count K of one or two.
Active sensor IDs are CUDA int64, strictly increasing, unique indices; an
explicit empty tensor selects no sensors, while `None` selects the full batch.

CUDA direction values are checked by the trace kernel without a host round
trip. A direction outside absolute norm tolerance `1e-4` produces no returns,
increments `invalid_directions`, and is reported by `check_errors()` after the
caller's chosen synchronization. CPU contract tests reject it eagerly.

| Name | Shape | Dtype | No-hit value |
|---|---:|---|---|
| `range_m` | `[B,R,K]` | float32 | `+inf` |
| `position_world_m` | `[B,R,K,3]` | float32 | zero |
| `intensity` | `[B,R,K]` | float32 | zero |
| `semantic_id` | `[B,R,K]` | int64 | `-1` |
| `valid` | `[B,R,K]` | bool | false |
| `time_offset_ns` | `[B,R,K]` | int64 | input offset, repeated over K |
| `return_count` | `[B,R]` | int32 | zero |

Inactive sensor rows are rewritten to the same no-hit contract. Range is
Euclidean distance in metres because transforms are rigid and directions are
unit length. Caller-owned outputs must be contiguous. Service-owned outputs
are allocated on first use of a `(B,R,K)` shape and always rewritten; timing
harnesses prewarm or provide ring-buffer outputs before measurement.

## Surface equation

Rotations are normalized WXYZ quaternions. The smallest scale axis is the
surfel normal and the other axes form the tangent ellipse. A ray intersects the
plane through the mean. A candidate is accepted only when all frozen
conditions hold:

Registered semantic IDs are non-negative; `-1` is reserved exclusively for
no-hit outputs.

- `0.05 m <= t <= 200 m` (inclusive);
- `abs(dot(normal, direction)) >= 0.05`;
- `normal_scale / min(tangent_scales) <= 0.35`;
- tangent Mahalanobis distance `q <= 3.0^2`;
- `w >= 0.01`, where
  `w = opacity * exp(-0.5*q) * surface_confidence`.

The normal is two-sided. Default surface confidence is one. Explicit
reflectivity is in `[0,1]`; when absent, constant `0.5` is used and intensity
is nonphysical. A candidate response is
`reflectivity * exp(-0.5*q) * abs(dot(normal, -direction))`.

Candidates share a depth cluster when
`t <= first_t + max(0.02 m, 0.001 * first_t)`. Cluster range is the
candidate-weighted range. Intensity is the candidate-weighted response.
Semantic ID is the label with greatest aggregate candidate weight; exact ties
choose the numerically smallest ID. The next return begins strictly after the
previous interval. K=2 reruns the unchanged K=1 equation. Double accumulators
reduce traversal-order sensitivity. More than 16 distinct labels in a cluster
is explicit `semantic_overflow`, never a silent drop.

## Acceleration and revisions

Each scene owns one CUDA acceleration structure. Gaussian centres receive
30-bit Morton codes, CUB sorts primitive IDs on the current stream, adjacent
primitives form packet leaves of eight by default, and conservative surface
AABBs populate a complete binary LBVH. Rays transform into scene-local space,
so B environments share scene tensors, sorted IDs and node bounds.

Registration and explicit scene revision are the only build triggers. In-place
material, opacity, confidence or semantic edits are read by the next trace.
Geometry, scale or rotation edits require
`revise_scene(scene_id, revision=<strictly increasing integer>)`. Build
temporaries remain alive until explicit synchronization and are then released.
Build time, persistent BVH bytes, caller-owned attribute bytes and workspace
bytes are reported separately.

The fixed tree depth is bounded by scene size. Stack, semantic, direction,
scene-ID and active-ID errors are device counters. Calls started/completed,
rays, node visits, primitive tests, candidates and returns are cumulative
observation-only counters and never influence the algorithm.

## Verification boundaries

`experiments/lidar_cpu_oracle.py` is an independent float64
collect-sort-reduce oracle and is never imported by production code. CUDA
validation covers rotated anisotropic splats, no-hit and range boundaries,
grazing/parallel rays, overlap clustering, two layers, semantic ties,
multi-scene batches, active subsets, rigid equivariance, permutation, explicit
revisions, fixed-state replay and zero overflow. The deferred-work audit uses
distinct padded output buffers, guard canaries, current-stream GPU fingerprints
and later rechecks to detect cross-call writes.

OVRTX LiDAR remains a reference integration. A speed comparison is not claimed
unless geometry, ray pattern, outputs, synchronization and timing are genuinely
equivalent. Intensity is compared only after material calibration.
