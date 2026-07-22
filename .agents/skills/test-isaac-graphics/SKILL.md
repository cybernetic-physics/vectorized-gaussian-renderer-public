---
name: test-isaac-graphics
description: Author, run, and review finite graphical tests for Isaac Sim, Omniverse Kit, OVRTX, and the custom Gaussian renderer. Use when an agent needs a headless SimulationApp smoke test, Kit-extension lifecycle test, Fabric camera test, GPU render-output assertions, deterministic replay, visual regression artifacts, fidelity metrics, projection-mode coverage, sanitizer validation, or a staged graphics test plan.
---

# Test Isaac Graphics

Build tests that terminate reliably and prove renderer behavior with tensors,
images, metadata, and explicit simulation/graphics contracts.

## Required reference

Read [references/graphics-test-contracts.md](references/graphics-test-contracts.md)
before adding a test or choosing the validation ladder.

Use root `scripts/run_isaac_graphics_gate.py` for lifecycle, extension, and
Fabric acceptance. It adds UUID-matched one-GPU selection, bounded process
groups, fatal-log scanning, exact lifecycle ordering, residual-process checks,
and canonical-runtime/source immutability checks. Use
`scripts/check-isaac-log.sh` for narrower headless runs; neither an exit code
nor a body marker alone is a test result.

## Authoring workflow

1. State the behavior under test.
   - Separate renderer math, Kit lifecycle, Fabric ingestion, physics coupling,
     sensor output, performance, and image fidelity.
   - Use the narrowest existing script as the starting point.

2. Make the application finite.
   - Construct `SimulationApp` before importing Isaac/Kit modules. Repository
     smoke scripts use root `scripts/_isaac_launch.py` so they share the
     fail-closed one-GPU configuration selected by the gate.
   - Wrap the body in `try/finally`.
   - Always call `simulation_app.close()`; repository smoke scripts use
     `close_simulation_app()` so failures retain a nonzero process status.
   - Use bounded steps, iterations, and timeouts.

3. Prove runtime and ingestion boundaries before any long render.
   - Ensure the CUDA extensions are compiled for the GPU this node actually
     has. The custom renderer kernel (editable install / JIT) and `gsplat`
     (a separate build — provisioning clones but does not build it) must target
     the live compute capability
     (`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`; set
     `TORCH_CUDA_ARCH_LIST` to match). A mismatch fails at first render with
     `no kernel image is available for execution on the device`, not a test
     logic error. See `$provision-rtx-workstation`.
   - Run with `/isaac-sim/python.sh` after `source scripts/remote_env.sh`.
   - Print `sys.executable`, Torch/CUDA versions, and module origins when
     diagnosing imports.
   - Test a tiny asymmetric scene containing rotated anisotropic splats.
   - Convert project/Ply-loader WXYZ quaternions to OVRTX `quatf[]` XYZW
     explicitly and test that boundary.
   - Do not launch temporal OVRTX convergence, a matrix, or a soak until this
     format-boundary fixture passes.

4. Use deterministic, explicit inputs.
   - Fix scene seeds, camera matrices, intrinsics, dimensions, projection mode,
     sorting mode, background, color space, near/far planes, and output set.
   - Save scene and camera checksums for visual comparisons.
   - For OVRTX Gaussian coverage, test both `perspective` and `tangential`
     projection modes with off-axis anisotropic splats and identify them in
     every artifact. Isotropic centered splats cannot distinguish the modes.

5. Assert the complete GPU output contract.
   - RGB, depth, alpha, and semantic ID have expected shapes and dtypes.
   - Outputs are CUDA-resident when the production path requires CUDA.
   - RGB/foreground depth are finite.
   - Alpha is in `[0, 1]`.
   - Foreground is nonempty.
   - Foreground semantic IDs are valid.
   - Capacity/overflow counters are zero.

6. Assert integration invariants.
   - Extension lifecycle actually ran.
   - Fabric Scene Delegate is enabled when required.
   - USD/Fabric transform edits reach the GPU.
   - Calling `render()` does not advance physics.
   - Calling `synchronize()` does not hide a render or simulation step.
   - Repeated fixed-state deterministic renders are bitwise equal where
     deterministic mode promises that behavior.

7. Save durable evidence.
   - Print one unique success marker.
   - Write JSON containing configuration, versions, checksums, assertions, and
     schema version, pass status, and failure reason.
   - For visual comparisons, save reference, candidate, side-by-side, RGB
     absolute difference, alpha difference, depth error, and semantic mismatch.

8. Separate representative and stress baselines.
   - Use spatially coherent semantic classes for representative performance and
     fidelity.
   - Keep interleaved/high-cardinality semantics as a separate stress test.
   - For stochastic OVRTX AOVs, record temporal sample count and convergence;
     do not compare one stochastic frame as if it were deterministic coverage.
   - Validate linear HDR independently from LDR/tonemap output in headless
     mode.

9. Run the validation ladder in increasing cost.
   - Unit/contract tests.
   - Native CUDA smoke.
   - Isaac headless lifecycle smoke.
   - Kit extension and Fabric camera smokes.
   - Deterministic, multi-scene, active-subset, and cache-coherence smokes.
   - Fidelity comparison.
   - Profiler or Compute Sanitizer.
   - Stability/soak only after all narrower checks pass.

10. Keep tests isolated.
   - Apply `$provision-rtx-workstation` before using a remote/shared GPU.
   - Use a unique output directory.
   - Do not overwrite producer evidence during independent verification.

## Review rules

- Treat missing images, tensors, metadata, or LPIPS as failures when the test
  contract requires them.
- Do not average away a worst-view failure.
- Do not loosen thresholds to make a test pass.
- Do not call a fake backend a production renderer validation.
- Do not treat a successful process exit as proof of graphical correctness.
- Do not infer quaternion, projection, or semantic behavior from identity,
  isotropic, or single-class fixtures.
- Do not parse evolving JSON with unchecked ad hoc key access; validate the
  schema version and required keys first.

Use `$profile-rtx-graphics` only after the relevant graphical test passes.
