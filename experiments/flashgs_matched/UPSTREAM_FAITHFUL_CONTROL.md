# Native-equation FlashGS control

The publication needs two different FlashGS comparisons. They answer different
questions and must not be merged into one table.

## Equal-equation rasterizer comparison

`FlashGSBackend` is the FlashGS-derived matched-contract port. It changes
projection/support/compositing details to target the same conventional EWA
equation and full sensor contract as the custom renderer and gsplat oracle.
Use this lane only after its adapter classification, raw fidelity gates, and
tile-boundary bug are resolved. It tests implementation performance under a
shared equation; it is not public FlashGS's native output.

## Degree-zero public-equation FlashGS control

`UpstreamFaithfulFlashGSBackend` retains public FlashGS's native RGB equation
within an explicitly degree-zero workload.
It directly compiles hash-pinned upstream sort and render translation units and
mechanically changes only preprocessing input/capacity plumbing. This lane:

- uses the same Home Scan PLY, activated geometry/opacity, camera poses, GPU,
  logical batch, active stream, preallocation policy, and timing boundary;
- applies public FlashGS's degree-zero `max(0.5 + SH_C0 * f_dc, 0)` color
  transform without the matched lane's upper clamp, so the resulting color
  tensor is intentionally equation-native rather than tensor-identical;
- invokes FlashGS once per camera because upstream has no camera batch kernel;
- preserves native support, log2/ex2 alpha, tile-wide termination, black
  background, and uint8 output encoding;
- measures the uint8-to-float32 service conversion; and
- supports RGB only.

The direct three-float color input specializes away public FlashGS's 48-float
SH load and view-dependent SH evaluation. It is equation-equivalent only for
the declared degree-zero input, and it may be faster than the unmodified public
binary. Therefore call this a **degree-zero-specialized upstream-equation
control**, not an unmodified or byte-exact public-FlashGS performance build.

Because its image equation and quantization differ from the custom/gsplat
contract, report this as a native-behavior control, not as the decisive
same-output speedup. Useful wording is:

> Against our reviewed degree-zero-specialized integration of pinned public
> FlashGS's RGB equation, including per-camera scheduling and native uint8
> output converted to the service's float32 tensor, ...

Do not call it unmodified public FlashGS, full-sensor FlashGS, matched EWA, or a
drop-in sensor replacement.

This lane alone cannot produce a Custom-versus-upstream speedup: Custom still
implements the pinned gsplat equation. Naming upstream FlashGS in a comparative
winner claim additionally requires a Custom compatibility lane that implements
this same degree-zero public equation and output quantization, plus an oracle
showing both sides produce the same image on the sampled Home Scan cameras.

## Admission ladder

1. Source audit passes at upstream commit
   `cdfc4e4002318423eda356eed02df8e01fa32cb6`.
2. `verify_upstream_faithful_flashgs_parity.py` matches byte-exact upstream on
   the rotated-anisotropic, off-axis fixture with zero uint8 mismatches.
3. Sampled real-scene parity matches byte-exact upstream at the publication
   camera steps, including reconstructed uint8 output and intersection count.
4. Compute Sanitizer passes the fixture and B1 128x128 Home control.
5. Bounded dynamic Home control passes at B1, B8, and B64 with zero capacity or
   camera-contract errors.
6. Nsight Systems confirms current-stream execution, one native pipeline per
   camera, and the measured conversion kernel.
7. Only then integrate an independently reviewed native-control matrix. Keep
   its evidence and label separate from the matched-contract matrix.

The bounded hook deliberately stops at B64 and 16 frames. Expanding it to the
publication batches is a later, reviewed matrix change; it must reuse the
repository's cooperative node lock, occupancy telemetry, calibration artifacts,
timing statistics, memory gates, and evidence bundle machinery.
