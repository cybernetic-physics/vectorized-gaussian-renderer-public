# External Gaussian-scene pipeline

This procedure turns a supplied PlayCanvas SOG archive or extracted SOG LOD
directory into a reproducible benchmark/flyby release without placing the
asset in Git.

1. Inspect the source and its `license.txt`. Record title, author, source URL,
   license, byte count, and SHA-256. Do not publish a source whose license is
   missing or incompatible with the intended distribution.
2. Run `scripts/prepare_splat_benchmark_asset.py` on the RTX host. It selects
   LOD0 chunks in natural numeric order, decodes them with `splat-transform`,
   and writes a canonical PLY plus a manifest containing exact source and PLY
   hashes.
3. Run `scripts/inspect_gaussian_scene.py <canonical.ply>` before rendering.
   Compare raw and canonical scale percentiles with a known-good scene. If a
   small scale tail dominates the range, run an explicitly marked pilot with
   `--max-gaussian-scale <positive-limit>`. That option only clamps the
   in-memory render tensors; it must be recorded in the render manifest and
   must not be described as an unfiltered canonical-scene benchmark.
4. Render custom CUDA first with `scripts/render_gaussian_flyby.py`; it writes
   the immutable camera contract. Run OVRTX with that exact contract. Use an
   orbit pilot before a final walkthrough unless an occupancy/collision route
   has been independently validated for the scene.
5. Package with `scripts/package_gaussian_flyby.py`, inspect decoded beginning,
   middle, and end frames plus the storyboard, and preserve the artifact
   manifest.
6. Add only verified PLY, manifest, video, GIF preview, poster, storyboard,
   and camera-contract bytes to `cloudflare/r2-assets.json` under hash-bearing
   keys. Use `scripts/publish_r2_artifacts.py` and, above 315 MiB, the
   authenticated temporary multipart Worker.

The result is a visual demo and a repeatable benchmark input. It does not by
itself establish equation-level renderer fidelity or physical collision.
