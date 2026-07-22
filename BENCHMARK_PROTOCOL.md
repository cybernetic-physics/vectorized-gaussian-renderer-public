# Benchmark protocol

## Immutable comparison contract

Every headline comparison uses a named, immutable configuration contract. The
contract freezes:

- one identified GPU UUID, machine, and driver;
- Identical Gaussian tensors and semantic IDs.
- Identical camera matrices, intrinsics, backgrounds, and color space.
- Identical batch, resolution, and measured frame range.
- RGB, depth, alpha, and semantic output enabled.
- Warmed-up steady state with explicit GPU synchronization.
- No generated or interpolated frames.
- No CPU output shortcut.
- No cached/pre-rendered images.

Any run that changes the contract receives a new benchmark configuration ID and
cannot be compared as a speedup. The OVRTX study retains its historical RTX
4090 contract. The decisive Custom-versus-FlashGS study uses
`experiments/flashgs_matched/BENCHMARK_CONTRACT.md` and one identified Google
Cloud NVIDIA L4; neither can silently inherit another study's hardware or image
equation.

## Implementations

### RTX/OVRTX

- Use the same Gaussian scene as the custom renderer.
- Source `scripts/remote_env.sh`; the OVRTX multi-RenderProduct path requires a
  soft file-descriptor limit of 65,536 for reproducible B256 runs.
- Enable and verify Fabric Scene Delegate for Kit/Isaac RTX runs. The
  standalone OVRTX API does not instantiate a Kit application or Fabric Scene
  Delegate; those rows must record that the setting is not applicable and
  cannot be cited as FSD-specific evidence.
- Record all RTX, tiled-rendering, antialiasing, lighting, and precision
  settings.
- Run both normal RTX and the fastest visually equivalent RTX configuration.
- Run `perspective` and `tangential` as separate complete OVRTX lanes across
  the same fidelity, temporal, throughput, memory, scene, batch, and
  resolution matrix. Neither lane may replace or reuse measurements from the
  other.
- Before accepting a tangential row, require composed-token runtime readback
  and a controlled cross-mode output effect above the same-mode repeat-noise
  envelope. A renderer-internal selected-mode state may replace token
  readback only if it proves kernel consumption directly. Authored or composed
  advisory-token readback by itself is not mode-activation evidence.
- Disable DLSS frame generation and temporal interpolation.
- Separate cold startup, shader warmup, and steady state.

### `gsplat`

- Pin commit `77ab983ffe43420b2131669cb35776b883ca4c3c`.
- Build `gsplat` from source on the benchmark node before measuring — a fresh
  provisioned node has it cloned but unbuilt. Compile for the compute
  capability of the GPU the node is actually running (detect with
  `nvidia-smi --query-gpu=compute_cap`). A binary built
  for a different architecture fails to load or falls back and cannot be cited
  as a parity control.
- Record packed/unpacked mode, rasterization mode, precision, SH degree,
  inference options, and workspace behavior.

### Custom

- Record source commit and build flags.
- Compile the custom CUDA kernel for the compute capability of the GPU the node
  is actually running (the editable install / JIT build must target the live
  architecture). Record the built arch alongside
  the commit; an architecture mismatch voids every timing and fidelity number.
- Prove the custom kernel is loaded and active.
- Include all required outputs and preprocessing.
- Record every optimization toggle.
- Projection-coherence reuse is admissible only as a separately labeled static
  mode. The direct compact path may retain per-pixel depth cutoffs, sorted
  Gaussian-ID intersections, and dense pixel ranges. Every timed render must
  still execute Gaussian evaluation for qualifying intersections, fused
  raster/compositing, and fresh RGB/depth/alpha/semantic writes. Nsight kernel
  counts, cache-coherence mutation tests, and a dynamic cache-off result are
  required beside the static result.
- External zero-copy and inference-mode writes must explicitly invalidate the
  projection cache. Cached/pre-rendered images remain forbidden.
- For OVRTX `LdrColor` comparisons, use
  `--authored-display-output`: scene RGB values are already authored in the
  display-sRGB contract and the custom compositor applies no second transfer.

## Datasets

1. Deterministic synthetic scenes with small and medium Gaussian counts.
2. At least one established public Gaussian-splat scene with checksummed
   download instructions.
3. Home Scan, attributed and checksummed in
   `datasets/home-scan-lod0.manifest.json`.

The named `.zip` is absent and a previous 386-byte copy was an XML
`AccessDenied` response. A valid extracted SOG LOD asset does exist beside it:

- Generator: `splat-transform v2.7.1`
- 434 files, including 360 WebP payloads and valid JSON metadata
- 72 chunks over six LOD levels
- LOD0: 21,497,908 Gaussians
- All LODs combined: 42,323,301 encoded Gaussian records
- Relative file-manifest SHA-256:
  `ac0c8226d8f0e5359deb09026d12954bb7096143ebff8061c50303909e8f285d`

The benchmark uses LOD0 as the full-resolution Home Scan unless a later
decision documents a different fidelity contract. Validate it with
`scripts/validate_home_scan.sh` before conversion or upload.

## Matrix

- Batch: 1, 8, 32, 64, 128, 256, maximum fitting.
- Resolution: 128x128, 256x256, 512x512.
- Scene: synthetic-small, synthetic-medium, public-real-world, Home Scan.
- Scene mode: shared scene and batched scene IDs.
- Output set: RGB + depth + alpha + semantic ID.

### Customer stress gate

Run the bounded fail-closed stress matrix with:

```bash
scripts/run_customer_stress_matrix.sh \
  --public-path /workspace/datasets/public-gaussian/Voxel51_gaussian_splatting/FO_dataset/train/point_cloud/iteration_30000/point_cloud.ply \
  --output outputs/customer-stress-matrix
```

The full gate adds rotated-anisotropic and foreground-overdraw scenes to the
representative synthetic cases. It keeps three capacity roles separate:

- `production-default` is the only role eligible for production acceptance;
- `conservative-non-oracle` measures safety-margin sensitivity without using
  observed demand;
- `hindsight-oracle-diagnostic` is sized from an earlier successful render and
  can never rescue a failed production verdict.

The public-scene lanes record the direct compact production default, an
explicitly conservative cache-disabled forced-recompute result, a cache-enabled
result that is explicitly invalidated before every render, and a
repeated-identical-camera projection-cache result. Cache-hit speedups must be
reported beside both recompute controls and must not be called moving-camera
performance.

Every child command is finite and has a per-case timeout. The matrix exits
nonzero when a production lane, required fidelity artifact, LPIPS dependency,
four-output assertion, capacity counter, or required diagnostic lane fails.
Production results must include `renderer-capacity/v1` telemetry with the
initial and final capacities, all retry attempts, actual device counters,
headroom, workspace bytes/ceiling, and a successful zero-overflow final state.
Growth occurs before steady-state measurement. Moving-camera results separately
report the wall cost of the required new-input counter validation.
Use `--evidence-only` only when collecting a complete FAIL artifact for triage;
it does not change the verdict in `run-manifest.json`. `--quick` is a synthetic
smoke and is not customer acceptance.

## Timing

- Record cold startup separately.
- Warm up until shader/JIT compilation and allocator growth stop.
- Use CUDA events for GPU elapsed time and a synchronized wall-clock timer.
- Use at least 50 measured iterations unless a documented long-run constraint
  requires more time per sample.
- Report mean, standard deviation, p50, and p95. Do not treat changing-camera
  frames from one deterministic trajectory as iid repetitions or derive a
  confidence interval from them.
- Run the contract's designated headline anchor at least three times in fresh
  processes. Report descriptive dispersion over independent run-level means;
  do not infer significance or a confidence interval from three runs or from
  correlated frame samples. The matched FlashGS
  contract designates B128 under both output contracts; the complete matrix
  remains a separate all-batch result.

## Memory

Reset allocator peaks before the measured range and synchronize around
measurements. Record:

- Peak allocated memory.
- Peak reserved memory.
- Persistent scene memory.
- Reusable workspace memory.
- Temporary allocation delta.
- Driver-reported process memory.

RTX memory must be measured with equivalent scene, outputs, and warm state.

## Profiling

Collect:

- Nsight Systems timeline and CUDA/NVTX summary.
- Nsight Compute occupancy/bandwidth metrics when host counter permissions
  allow it.
- PyTorch profiler trace for binding/allocation/launch behavior.
- Kernel launch count and CPU synchronization.
- Projection/culling, binning/sorting, and rasterization durations.
- Visible Gaussian and tile-intersection counts.

This host currently restricts Nsight Compute performance counters with
`RmProfilingAdminOnly: 1`; this is recorded as a host limitation, not silently
omitted.

## Fidelity

For every representative and worst-case view, save:

- Reference and candidate RGB.
- Absolute/directional RGB difference.
- Depth and depth-relative-error image.
- Alpha difference.
- Semantic mismatch mask.

Required thresholds are defined in `GOALS.md`. RTX-only secondary lighting is
reported separately from Gaussian reconstruction/compositing parity.

### Deterministic camera bundle

All benchmark runners that compare fidelity share a `camera-bundle-v1` JSON
file. It contains the immutable comparison inputs that must be identical across
RTX/OVRTX, `gsplat`, and custom runs:

- `width` and `height`.
- `background` as three float values in the declared color space.
- `color_space`, fixed by each immutable configuration. The current fair OVRTX
  control uses `display_srgb` with
  `identity_authored_display_rgb`; linear controls must declare `linear_rgb`.
- `coordinate_system`, fixed to `opencv_world_to_camera`.
- Optional `scene_checksum`.
- Ordered camera records containing `view_id`, `scene_id`, 4x4 world-to-camera
  `viewmat`, and 3x3 `intrinsics`.

The bundle includes a SHA-256 `bundle_id` over canonical JSON. Render outputs
may embed that ID; the fidelity comparator rejects mismatches.

### Machine-readable render outputs

The fidelity comparator consumes either:

- A `.npz` containing `rgb`, `alpha`, `depth`, `semantic`, and optional
  `valid_depth`, `color_space`, `background`, and `camera_bundle_id`.
- A `render-output-v1` JSON manifest pointing at `.npy` arrays with the same
  logical fields.

Canonical tensor shapes are:

- `rgb`: `[views, height, width, 3]`, float in the declared color space.
- `alpha`: `[views, height, width]` or `[views, height, width, 1]`.
- `depth`: `[views, height, width]` or `[views, height, width, 1]`.
- `semantic`: `[views, height, width]` or `[views, height, width, 1]`.
- `valid_depth`: optional boolean mask with the same scalar image shape.

The comparator reports per-view and aggregate RGB PSNR, RGB SSIM, LPIPS, alpha
MAE, valid-pixel depth relative error, and semantic agreement. Aggregate PASS
requires every view to satisfy every threshold; worst-view selection is based
on the smallest threshold margin. If LPIPS is unavailable for non-identical
images, acceptance fails rather than substituting a proxy.

### Fidelity contract boundary

The supported parity target is Gaussian reconstruction/compositing with
identical Gaussian tensors, semantic IDs, cameras, backgrounds, color space,
depth convention, and output resolution. RTX-only secondary effects such as
dynamic mesh lighting, shadows, reflections, denoising, temporal antialiasing,
DLSS frame generation, and other path-traced scene effects are recorded as a
separate RTX-secondary-effects delta. They are not counted as a Gaussian
compositing failure unless the benchmark configuration explicitly enables the
same effect in both implementations through a new decision and configuration
ID.

Rendering equations are part of the immutable contract:

- The production fast path projects each 3D Gaussian to a deterministic 2D EWA
  ellipse and alpha-composites the ordered screen-space contributions. Its
  direct mathematical parity control is pinned conventional `gsplat`.
- OVRTX perspective is evaluated as a separate stochastic temporal contract.
  Current evidence is consistent with full 3D Gaussian ray evaluation and
  probabilistic first-hit compositing. A scene-specific parameter fit or
  finite-frame temporal sample cannot replace a converged perspective
  reference.
- The experimental exact-ray path is diagnostic until it completes the same
  performance, Home Scan, workload-matrix, stability, and independent
  verification program as the production renderer.
- `projectionModeHint` is advisory in the USD particle-field schema. A
  perspective/tangential experiment must prove composed runtime token
  readback and a cross-mode effect above the measured same-mode repeat noise
  envelope on a rotated anisotropic off-axis fixture. Bitwise-identical output
  tensors on an isotropic fixture are recorded as an insensitive fixture, not
  as tangential parity or proof that the hint is globally ignored.
- A resumed stochastic temporal run is admissible only if the fresh renderer
  advances its sample sequence by exactly the restored frame count before new
  accumulation. Until that state transition is verified, each final frame
  budget must run from scratch.

## Result format

Every run emits machine-readable JSON and a flat CSV row containing:

- Git/environment identifiers.
- Immutable configuration ID.
- Dataset checksum.
- Implementation and flags.
- Timing statistics.
- Memory statistics.
- Work counts.
- Fidelity metrics.
- PASS/FAIL fields.

## Trajectory evaluation protocol

Trajectory runs use `camera-trajectory-v1`: time-major `viewmats [T,B,4,4]`,
`intrinsics [T,B,3,3]`, explicit or time-broadcast scene IDs, optional
environment transforms and active-camera IDs, expected cache events, and
portable scene/route provenance. The canonical JSON hash includes per-array
SHA-256 values and excludes absolute workstation paths. The contract can be
flattened in stable time-major order into `camera-bundle-v1` for fidelity.

`benchmarks/run_trajectory.py` keeps scene loading, warmup, synchronized GPU
rendering, CPU readback, frame encoding, video encoding, and artifact hashing
as separate scopes. Spatial batches, B1 sequential paths, and vectorized
temporal batches are different execution semantics. Cache events, native
submissions, renderer executions, and rendered views are recorded separately;
skipped cadence requests are not rendered frames.

The required scenario matrix is static repeat, appearance updates, sequential
flyby, phase-offset vectorized flyby, mixed motion, intrinsics sweep,
environment-transform motion, teleport, active-subset cadence, and multi-scene
trajectory. Representative and stress Home Scan routes are versioned under
`benchmarks/camera_paths/`. Route validation records clearance, path length,
per-frame translation/rotation, angular acceleration, duplicate poses,
foreground/depth coverage, near-plane pressure, and manual storyboard status.

Every durable run uses `outputs/evals/<run-id>/` with manifest, trajectory JSON
and NPZ, timing, cache events, per-frame and summary metrics, tensors, frames,
storyboards, videos, profiles, and a complete artifact SHA-256 manifest.
`benchmarks/audit_trajectory_artifacts.py` reopens files and decoded video
samples and fails on missing, black, clipped, frozen, mismatched, or unhashed
evidence.
