# FlashGS-derived matched-contract port

These CUDA sources are derived from public FlashGS commit
`cdfc4e4002318423eda356eed02df8e01fa32cb6` (InternLandMark, MIT; see
`LICENSE.flashgs`). The unmodified source remains a separate checkout selected
by `FLASHGS_SOURCE_PATH`; its pinned GLM headers are used at build time.

The retained FlashGS algorithmic structure is:

1. one warp-cooperative projection and tile-key emission pass per camera;
2. one global CUB radix sort of tile/depth keys per camera;
3. tile-range discovery from the sorted keys; and
4. the public optimized 16x16 FlashGS tile compositor and load schedule.

This is not an upstream-faithful or integration-only adapter. Preserving the
pinned-gsplat target equation changes projection, support, sampling, alpha,
termination, color evaluation, and output behavior relative to upstream
FlashGS. The port combines service changes with those equation changes:

- canonical degree-zero RGB, covariance, view-matrix, and intrinsics tensors
  are accepted directly from GPU memory;
- the active PyTorch CUDA stream is used;
- scene, geometry, output, CUB, key/value, and tile-range allocations are
  reused;
- a sentinel-filled fixed-capacity sort removes the public example's
  per-camera intersection-count CPU readback and synchronization;
- float32 linear RGB replaces the public example's quantized `uchar3` output;
- the conventional 3DGS alpha cap, alpha threshold, pixel center, support
  cutoff, black background, and expected-depth definition are matched to the
  custom renderer and pinned gsplat oracle;
- optional alpha, expected depth, and strongest-contribution semantic ID are
  accumulated in the same compositor for the full sensor contract; and
- the cooperative feature-loader guards are aligned with their load offsets
  for one- and two-entry tile tails. The pinned upstream source instead loads
  Gaussian 0's features for the final short-tail slot; this port repairs that
  correctness defect without changing the compositor schedule.

FlashGS still executes one native pipeline per camera. This adaptation does
not add a camera batching kernel, projection cache, rendered-frame cache,
custom depth/foreground cutoff, alternate ordering method, or other custom
rasterizer optimization. Fixed-capacity overflow is counted on the device and
is a fail-closed benchmark condition.

`repair_reference_render.cu` is not compiled. It is the exact pre-repair
production `render.cu` from diagnosis commit
`5eca4e640aabd234158d08b4ed61d061e530ee3f`, with SHA-256
`141ddad0271055b41ef097540a95a9f217cc9e55b52bdec211c4da319486c97a`.
The history-free public repository intentionally does not contain that legacy
commit, so the repair audit reads this tracked, hash-pinned reference instead
of depending on a hidden Git object. The reference and production source differ
only in the two short-tail predicates described above.

## Debug-only pipeline trace

`debug_adapter.cpp`, `debug_trace.cu`, and `debug_ops.h` build as a separate
PyTorch extension through `flashgs_debug_native_loader.py`. They are not linked
into the production renderer. The standalone
`benchmarks/debug_flashgs_pipeline.py` command renders exactly one camera
outside all timing ranges. Its default `all-contributors` discovery scans every
scene Gaussian once, retains every projected target-pixel contributor in a
bounded score buffer, and traces the strongest 32 by individual alpha. The
selection ceiling is configurable up to 64; exact total/truncated counts and
cutoff ties are recorded. The narrower
`old-predicate-false-negatives` mode remains available for testing that one
hypothesis, but is not the default diagnosis.

For every discovered or explicitly supplied ID, the trace records projection
gates, the original predicate's coefficients/discriminants/short-circuit
branches, unsorted and sorted tile-key presence, tile-range membership, and an
exact replay of the target-pixel compositor branch. The JSON is diagnostic
correctness evidence only: it synchronizes and performs CPU readback, so none
of its timing is benchmark evidence. It requires a byte-verified clean source
manifest and a FlashGS adapter attestation bound to that same manifest. Tool
integrity and `cause_identified` are separate verdicts; zero contributors or no
located loss stage never counts as a completed diagnosis. Historical
pre-repair diagnosis and post-repair acceptance use separate, fail-closed
schemas. Run them through the repository's locked RTX-node workflow with the
fixed, zero-overflow per-camera capacity from the matching renderer run.
