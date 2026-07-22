# TODO

## Benchmark both OVRTX Gaussian projection modes

The current OVRTX benchmark evidence was authored with:

```usda
uniform token projectionModeHint = "perspective"
```

This is not sufficient for judging fidelity against the custom renderer. The
custom CUDA path uses the usual 3DGS-style local tangent-plane/EWA projection,
while OVRTX also exposes a `tangential` projection hint intended to preserve
Gaussian shape for novel-view synthesis.

- [ ] Parameterize the OVRTX benchmark runner with explicit `perspective` and
      `tangential` projection modes instead of hard-coding `perspective`.
- [ ] Record the selected projection mode in every command, result JSON, CSV,
      run manifest, comparison report, and output directory.
- [ ] Rerun the same OVRTX performance and fidelity cases in both modes with
      identical scenes, cameras, resolutions, warmup, iteration count, AOVs,
      semantic mapping, and color/depth conventions.
- [ ] Cover the synthetic matrix, public Gaussian scene, and Home Scan LOD0.
- [ ] Compare both OVRTX modes against the same custom-renderer outputs using
      RGB, alpha, depth, and semantic metrics.
- [ ] Keep the overall fidelity verdict open until the tangential results are
      measured. If tangential materially closes the gap, classify the previous
      mismatch as an OVRTX baseline-configuration problem rather than assuming
      the custom CUDA projection math is wrong.

The perspective results remain useful as a conventional camera-projection
control. They must not be silently replaced by tangential results; both modes
should remain reproducible and reported side by side.
