---
name: render-gaussian-flyby
description: Render, package, and visually validate matched Gaussian-scene flybys with the custom CUDA renderer and OVRTX in headless Isaac Sim. Use for scripted camera walkthroughs, immutable camera contracts, Home Scan videos, OVRTX temporal accumulation, side-by-side comparisons, frame/storyboard QA, or renderer-demo artifacts that must not be confused with numerical fidelity acceptance.
---

# Render Gaussian Flyby

Produce a reproducible visual comparison in which both renderers consume the
same scene and camera bytes. Keep visual-demo claims separate from fidelity and
performance acceptance.

## Required reference

Read [references/flyby-contract.md](references/flyby-contract.md) before
rendering or judging a flyby. It contains the camera, OVRTX, packaging, and QA
contracts plus the validated Home Scan commands. For a user-supplied SOG or
LOD tree, also read
[references/external-scene-pipeline.md](references/external-scene-pipeline.md).

## Workflow

1. Establish provenance.
   - Treat the local checkout as authoritative and record its branch/status.
   - Verify the canonical scene byte count and SHA-256 before loading CUDA.
   - Record author, source, and license in every public manifest.

2. Design a safe path.
   - Derive occupancy and clearance from scene geometry for indoor scans.
   - Render a short, low-cost pilot before a final temporal OVRTX run.
   - Visually inspect the pilot; reject upside-down cameras, wall traversal,
     black frames, and large unintentional missing-scan views.

3. Lock one camera contract.
   - Serialize world-to-camera matrices, intrinsics, dimensions, near/far,
     frame rate, scene hash, route hash, and coordinate conventions to NPZ and
     JSON.
   - Hash the exact arrays and metadata.
   - Reuse the same contract for custom and OVRTX; never independently rebuild
     nominally similar paths.

4. Render the custom sequence.
   - Keep the Gaussian scene resident on the GPU.
   - Render bounded camera chunks, assert zero capacity overflow, and save
     contiguous numbered PNGs plus a machine-readable render manifest.

5. Render OVRTX.
   - Use `dynamic-chunks`, row-major `omni:xform` writes, and no renderer reset
     between camera batches.
   - Keep no more than twelve active render products for the validated Home
     Scan path.
   - Accumulate the requested temporal frames per camera and record warmup and
     sample counts.
   - Shard scenes above 16,777,215 Gaussians across ParticleField prims.

6. Package and validate.
   - Require contiguous frame counts and equal camera-contract hashes.
   - Encode H.264/yuv420p MP4s with `+faststart`, an animated GIF preview, a
     poster, a twelve-frame storyboard, and a checksum/probe manifest.
   - Decode frames from the final MP4, not only source PNGs, and inspect at
     least the beginning, midpoint, and end with `view_image`.
   - Do not publish until labels fit and every sampled pair remains aligned.

7. Report honestly.
   - Describe the flyby as a visual demonstration.
   - Do not use a visually close path to override equation-level PSNR, SSIM,
     LPIPS, alpha, depth, or semantic acceptance.
   - Record any temporal accumulation, sharding, tonemapping, or output
     limitations next to the media.

Use `$publish-r2-artifacts` only after this workflow passes encoded-frame QA.

## External SOG scenes

Use the generic, provenance-pinned path for an extracted SOG LOD directory or
a standalone `.sog` archive. Never feed compressed SOG bytes directly to a
renderer benchmark.

```bash
# On the RTX host. Keep source assets and generated PLYs outside Git.
python3 scripts/prepare_splat_benchmark_asset.py \
  --scene-id <stable-kebab-id> \
  --source <input-directory-or.sog> \
  --output-ply /workspace/datasets/<stable-kebab-id>.ply \
  --manifest datasets/<stable-kebab-id>.manifest.json

/isaac-sim/python.sh scripts/render_gaussian_flyby.py \
  --renderer custom \
  --scene-id <stable-kebab-id> \
  --scene-path /workspace/datasets/<stable-kebab-id>.ply \
  --expected-scene-sha256 <manifest-canonical-sha256> \
  --expected-gaussian-count <manifest-gaussian-count> \
  --scene-author '<attributed-author>' \
  --scene-source <attribution-url> \
  --scene-license '<license>' \
  --camera-path orbit --output-root outputs/flyby/<stable-kebab-id>-v1 \
  --overwrite --overwrite-camera-contract

# Repeat with --renderer ovrtx and the exact generated camera contract, then:
python3 scripts/package_gaussian_flyby.py \
  --output-root outputs/flyby/<stable-kebab-id>-v1 \
  --scene-manifest datasets/<stable-kebab-id>.manifest.json
```

The generated artifact manifest, encoded MP4 samples, storyboard, and poster
must be reviewed before the release is added to an R2 asset manifest.

## Failure rules

- A successful encoder exit is insufficient without probes and visual review.
- A matching camera hash is insufficient if runtime transform writes are
  ignored; validate signal and batch-boundary frames.
- Do not call static-all OVRTX output valid when many render products dilute or
  time-slice temporal accumulation.
- Do not reset OVRTX after dynamic camera writes; it can restore authored
  transforms or disrupt convergence.
- Preserve final manifests and camera contracts locally before remote cleanup.
