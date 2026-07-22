# Matched Gaussian flyby contract

## Home Scan source contract

| Field | Value |
|---|---|
| Dataset | `home-scan-lod0` |
| Canonical PLY | `/workspace/datasets/home-scan-lod0.ply` |
| Bytes | `1203883212` |
| Gaussians | `21497908` |
| SHA-256 | `29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267` |
| Author | Isaiah Sweeney |
| Source | `https://superspl.at/scene/3f89bbd3` |
| License | CC BY 4.0 |

Reject the run before CUDA allocation if count or checksum differs.

## Camera contract

The final Home Scan path is
`benchmarks/camera_paths/home-scan-walkthrough-v1.json`. It was derived from an
XZ occupancy/clearance map with Home Scan world-up `[0, -1, 0]`.

The generated `camera-path.npz` must contain:

- OpenCV pinhole world-to-camera matrices `[N, 4, 4]`;
- intrinsics `[N, 3, 3]`;
- frame count, dimensions, focal scale, near/far, and frame rate;
- scene and route hashes;
- explicit world-up and camera-model metadata;
- one SHA-256 covering arrays and canonicalized metadata.

Create it once during the custom run. The OVRTX run must load it without
`--overwrite-camera-contract`.

## Validated final commands

Run on the RTX machine:

```bash
cd /workspace/vectorized-gaussian-renderer
source scripts/remote_env.sh

/isaac-sim/python.sh scripts/render_home_scan_flyby.py \
  --renderer custom \
  --scene-path /workspace/datasets/home-scan-lod0.ply \
  --output-root outputs/flyby/home-scan-v1 \
  --camera-path clearance \
  --frames 240 \
  --width 384 \
  --height 384 \
  --fps 24 \
  --chunk-size 2 \
  --overwrite \
  --overwrite-camera-contract

/isaac-sim/python.sh scripts/render_home_scan_flyby.py \
  --renderer ovrtx \
  --scene-path /workspace/datasets/home-scan-lod0.ply \
  --output-root outputs/flyby/home-scan-v1 \
  --camera-contract outputs/flyby/home-scan-v1/camera-path.npz \
  --frames 240 \
  --width 384 \
  --height 384 \
  --fps 24 \
  --chunk-size 12 \
  --ovrtx-camera-layout dynamic-chunks \
  --ovrtx-camera-write-mode row-major \
  --ovrtx-temporal-samples 64 \
  --ovrtx-warmup-samples 4 \
  --overwrite

python3 scripts/package_home_scan_flyby.py \
  --output-root outputs/flyby/home-scan-v1
```

Do not pass `--ovrtx-reset-between-camera-batches` for the validated path.

## OVRTX invariants

- The full scene exceeds OVRTX's effective 24-bit per-prim index limit. Split
  at 16,777,215 Gaussians; the final counts are `[16777215, 4720693]`.
- Keep scene source tensors alive while asynchronous OVRTX upload is active.
- Project WXYZ quaternions become OVRTX/Fabric XYZW at the upload boundary.
- Dynamic row-major `omni:xform` writes were verified by USD readback.
- `xformOp:transform` did not provide the required runtime recomposition in the
  tested path.
- Resetting after camera writes can restore authored transforms.
- Authoring all 240 products at once produced dark/time-sliced accumulation.
  The validated resident stage uses twelve products and updates them in chunks.

## Signal and playback QA

Before encoding, compare per-sequence signal:

- no completely black frames;
- plausible nonblack and clipped fractions;
- custom and OVRTX mean luma remain close;
- batch-boundary frames show the intended camera positions.

After encoding:

1. Probe codec, pixel format, dimensions, frame count, frame rate, and duration.
2. Extract frames from the final side-by-side MP4 near 5%, 50%, and 90%.
3. Generate a twelve-frame storyboard.
4. Inspect both with `view_image`.
5. Reject clipping labels, swapped columns, frozen cameras, wall traversal,
   inverted orientation, mismatched views, or corrupt frames.

The final package contract is H.264, yuv420p, `+faststart`, 240 frames, 24 fps,
10 seconds, 384x384 per renderer, and 768x384 side-by-side.

## Claim boundary

The custom sequence is deterministic screen-space 2D EWA. The OVRTX sequence
is a 64-frame temporal estimate of a stochastic 3D-ray renderer. The video can
demonstrate path, appearance, stability, and gross alignment. It cannot prove
equation parity or replace the numerical fidelity suite.
