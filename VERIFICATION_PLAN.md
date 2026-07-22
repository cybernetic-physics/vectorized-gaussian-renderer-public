# Independent verification plan

This branch owns verification only. It must not implement renderer kernels,
change acceptance thresholds, relax benchmark criteria, or overwrite producer
outputs.

## Remote isolation

- Independent reruns use SSH host `vast-gsplat-isaac`.
- Independent reruns use remote root `/workspace/agent-worktrees/verification`.
- Verification scripts refuse any other remote root.
- Optional source sync excludes `outputs/` so producer outputs and independent
  evidence are not overwritten by the verifier.

## Harness entrypoint

`scripts/verify_results.sh` runs:

1. Required-artifact checks for benchmark documents, scripts, verifier module,
   and tests.
2. Remote unit tests with `/isaac-sim/python.sh -m pytest -q tests`.
3. Isaac headless extension-smoke validation for
   `ISAACSIM_GAUSSIAN_RENDERER_SMOKE_OK`.
4. Independent evidence validation with:

   ```bash
   /isaac-sim/python.sh -m isaacsim_gaussian_renderer.verification \
     --evidence-dir outputs/verification
   ```

The verifier fails closed when `outputs/verification/acceptance.json`,
`outputs/verification/benchmark_rows.csv`, or any required evidence section is
missing or invalid.

## Evidence contract

`outputs/verification/acceptance.json` uses schema version `1` and must include:

- Setup validation: fresh setup rerun, Isaac headless smoke pass, Vast instance
  `45069639`, host `vast-gsplat-isaac`, remote root
  `/workspace/agent-worktrees/verification`, and retained logs whose paths are
  relative to the evidence directory.
- Dataset manifests: synthetic-small, synthetic-medium, public-real-world, and
  Home Scan LOD0 with SHA-256 manifests and record counts. Home Scan LOD0 must
  match manifest
  `ac0c8226d8f0e5359deb09026d12954bb7096143ebff8061c50303909e8f285d`.
- Benchmark runs: full matrix for OVRTX/RTX, `gsplat`, and custom over required
  batches, resolutions, scenes, `shared-scene` mode, `scene-ids` mode, and
  RGB/depth/alpha/semantic outputs. Normal and fastest-visually-equivalent RTX
  settings remain distinct immutable OVRTX configurations.
- Timing samples: at least 50 measured samples per row. The verifier recomputes
  mean, standard deviation, p50, p95, and 95 percent confidence interval.
- Memory evidence: allocated, reserved, persistent scene, workspace, temporary
  allocation delta, and driver process memory.
- Fidelity evidence: representative and worst-case views for every required
  scene, with saved reference/candidate/diff artifacts.
- Deterministic replay: repeated output hashes and selected-environment replay.
- Shared-scene memory assertions: exactly one static scene copy and no static
  scene duplication for per-environment transforms.
- Active-subset and render-cadence checks.
- CUDA error checks with explicit synchronization around the measured range.
- Ten-minute stability run with no CUDA errors and no NaN/Inf outputs.
- Headline comparison: custom kernel loaded, GPU-resident outputs, no steady
  CPU output copies, single-camera result, three independent custom runs,
  throughput ratio, and peak-memory ratio.

`outputs/verification/benchmark_rows.csv` must contain one flat row per JSON
benchmark run and PASS status for every row.

## Threshold handling

The verifier uses the thresholds already defined in `GOALS.md`:

- Custom throughput at least 5.0x RTX.
- Custom peak memory at most 80 percent of RTX.
- RGB PSNR at least 40 dB.
- RGB SSIM at least 0.995.
- LPIPS at most 0.01.
- Alpha MAE at most 0.005.
- Valid-pixel depth relative error at most 1 percent.
- Semantic-ID agreement at least 99.9 percent.
- Three-run reproducibility within 5 percent.
- Ten-minute stability duration at least 600 seconds.

Changing these values is outside verification ownership.

## Current tests

`tests/test_verification.py` creates synthetic fixture evidence that covers the
complete benchmark matrix and verifies:

- A valid fixture passes.
- Missing `acceptance.json` fails closed.
- Manipulated timing statistics fail because the verifier recomputes them.
- A failing CSV row fails closed.

These tests do not claim acceptance. They only validate the verifier behavior
until real baseline, custom-renderer, fidelity, and stability outputs exist.

## Rerun procedure once implementations are available

1. Run `scripts/setup.sh` against the independent remote root and retain logs.
2. Copy or regenerate immutable dataset manifests under independent evidence.
3. Rerun producer benchmark commands from the independent root without writing
   into producer output directories.
4. Confirm benchmark rows use the same configuration IDs, camera bundles,
   Gaussian tensors, backgrounds, precision, output set, warmup policy, and
   measured frame range as the protocol requires.
5. Run deterministic replay, shared-scene memory, active-subset, cadence, CUDA
   synchronization/error, and ten-minute stability checks.
6. Save all evidence under `outputs/verification/`.
7. Run `scripts/verify_results.sh`.

## Benchmark loopholes to close

- Comparing custom output to a different RTX configuration ID while reporting a
  headline speedup.
- Omitting depth, alpha, or semantic outputs from timing while enabling them in
  fidelity-only paths.
- Reporting asynchronous timings without CUDA synchronization.
- Reusing cached images, generated frames, interpolation, or pre-rendered camera
  paths.
- Measuring CPU tensors or CPU transfer shortcuts as if outputs stayed
  GPU-resident.
- Excluding shader/JIT warmup or allocator growth inconsistently between RTX
  and custom runs.
- Letting custom preprocessing differ from RTX/gsplat preprocessing.
- Comparing different camera matrices, intrinsics, backgrounds, color space,
  precision, or Gaussian tensors.
- Claiming shared-scene scaling while duplicating static scene data per camera
  or per environment.
- Reporting only aggregate vectorized throughput and hiding single-camera
  regressions.
- Averaging away worst-case fidelity failures instead of checking
  representative and worst-case views.
- Publishing mean-only timing without sample count, standard deviation, p50,
  p95, and confidence interval.
- Treating preliminary synthetic `gsplat` numbers as RTX baseline evidence.
- Treating the invalid Home Scan `.zip` response as dataset content or using
  all redundant LODs as one scene.
- Silently omitting Nsight Compute counter evidence when host permissions block
  counters.
