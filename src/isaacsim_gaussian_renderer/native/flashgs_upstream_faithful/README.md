# Upstream-faithful FlashGS RGB control

This is a separate control from the repository's `FlashGSBackend`. The latter
is a FlashGS-derived, gsplat-matched equation port; it is appropriate only for
the matched-rasterizer experiment. `UpstreamFaithfulFlashGSBackend` instead
retains the public FlashGS RGB image-formation equation from commit
`cdfc4e4002318423eda356eed02df8e01fa32cb6` for an explicitly degree-zero
workload.

The native build links public `sort.cu` and `render.cu` byte-for-byte from a
clean pinned sibling checkout. In particular, the following stay upstream:

- EWA projection and `+0.3` low-pass covariance;
- opacity-derived support and ellipse/tile intersection test;
- global depth-key radix sort;
- log2/ex2 alpha, no 0.99 cap, and no matched-gsplat alpha cutoff;
- the public 16x16 load schedule and tile-wide termination behavior;
- black-background uint8 encoding and truncation.

`preprocess.cu` is mechanically generated from the hash-pinned upstream file.
The generator asserts byte identity for every equation helper, and permits only
these integration changes:

1. read a preallocated GPU camera-state record produced from the service's
   OpenCV view/intrinsic tensors;
2. read canonical degree-zero RGB without allocating a 48-float SH record per
   Gaussian;
3. bound key writes and report overflow on-device; and
4. expose a renamed service entrypoint.

The fixed-capacity lane fills unused keys with a synthetic tile ID exactly one
past the real tile grid. Public sort places these entries after all real tiles.
Public range discovery writes the extra preallocated range, while public render
never launches that tile. This removes the upstream example's per-camera CPU
counter readback without modifying the compositor.

The service reuses one native uint8 RGB target and converts it to float32 on the
active PyTorch stream. That conversion is included in render timing. The lane
is RGB-only: adding depth, alpha, or semantic accumulation would modify the
upstream tile shader and belongs in the separate matched-port experiment.
For PLY controls, the three RGB values are produced with public FlashGS's
degree-zero `max(0.5 + SH_C0 * f_dc, 0)` rule and are not upper-clamped before
compositing.

The three-float color input specializes away the public implementation's
48-float SH load and SH evaluation. This preserves the declared degree-zero
equation but is not a byte-identical or performance-identical build of the
unmodified public preprocessing translation unit. Report it as a
degree-zero-specialized upstream-equation control.

Before benchmark admission, run the untimed asymmetric-fixture parity gate:

```bash
python benchmarks/verify_upstream_faithful_flashgs_parity.py \
  --expected-gpu-uuid GPU-... \
  --output outputs/upstream-faithful/parity.json
```

Then run the bounded Home Scan dynamic control under the repository's shared
RTX occupancy/lock workflow. That control is intentionally capped at 64 cameras
and 16 frames; it is a smoke/performance diagnostic, not headline evidence.
Before a comparative publication claim, also match sampled real Home Scan
cameras against the byte-exact reference and implement the same equation and
uint8 output contract in the Custom comparison lane.
