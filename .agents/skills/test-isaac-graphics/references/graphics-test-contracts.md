# Isaac and RTX graphics test contracts

## Contents

- Existing test entrypoints
- Runtime and launch contract
- Finite SimulationApp pattern
- Format-boundary fixture
- Output assertions
- Integration assertions
- Visual and fidelity evidence
- OVRTX stochastic and color-output contracts
- Artifact schema checks
- Recommended execution ladder
- Failure triage

## Existing test entrypoints

| Concern | Entry point |
|---|---|
| Fail-closed combined Isaac graphical acceptance | `scripts/run_isaac_graphics_gate.py` |
| Isaac Sim can start, render, and exit | `scripts/isaac_headless_smoke.py` |
| Kit extension lifecycle and renderer service | `scripts/isaacsim_extension_smoke.py` |
| Physics, Fabric cameras, and vectorized rendering | `scripts/isaacsim_vectorized_example.py` |
| Fabric-to-CUDA camera transforms | `scripts/fabric_camera_smoke.py` |
| Native CUDA outputs and capacity | `scripts/custom_backend_smoke.py` |
| Bitwise repeat and equal-depth order | `scripts/deterministic_cuda_smoke.py` |
| Packed scenes and active subsets | `scripts/multi_scene_cuda_smoke.py` |
| Projection-cache coherence | `scripts/projection_cache_cuda_smoke.py` |
| OVRTX Gaussian ingestion and AOVs | `experiments/ovrtx_gaussian_smoke.py` |
| OVRTX/custom temporal fidelity | `experiments/ovrtx_temporal_fidelity.py` |
| Machine-readable fidelity comparison | `benchmarks/compare_fidelity.py` |
| Ten-minute stability | `benchmarks/run_soak.py` |

Extend the closest script instead of creating a broad end-to-end test for a
narrow bug.

## Runtime and launch contract

Run project and analysis code with:

```bash
source scripts/remote_env.sh
/isaac-sim/python.sh <script-or-module>
```

Do not use bare local or remote `python3` for Torch, NumPy, OVRTX, gsplat, USD,
or NPZ analysis. If a script imports from `experiments`, include both the
repository root and `src` in `PYTHONPATH`, or invoke it with `-m`.

For combined Isaac graphical acceptance, run
`scripts/run_isaac_graphics_gate.py`. Supply the canonical runtime's Python, a
disposable output directory, the exact 40-character source commit, and the
content digest produced by `source_fingerprint()`. The gate refuses inherited
Python packages and moving Kit-registry fallback, and requires the same full
GPU UUID from Torch, Vulkan, and `nvidia-smi`.

```bash
SOURCE_SHA256="$(python3 -c \
  'from pathlib import Path; from scripts.run_isaac_graphics_gate import source_fingerprint; print(source_fingerprint(Path.cwd())["content_sha256"])')"

<canonical-python> scripts/run_isaac_graphics_gate.py \
  --python <canonical-python> \
  --uv <uv-binary> \
  --canonical-runtime <canonical-runtime> \
  --source-root "$PWD" \
  --output-root <new-output-directory> \
  --source-commit "$(git rev-parse HEAD)" \
  --source-sha256 "$SOURCE_SHA256"
```

For a narrower smoke, the Isaac launcher may return zero after a Python
exception. Capture a log, require a unique success marker, reject fatal
patterns, and validate result JSON:

```bash
timeout --signal=TERM 10m \
  /isaac-sim/python.sh scripts/isaacsim_extension_smoke.py \
  > outputs/logs/extension-smoke.log 2>&1

.agents/skills/test-isaac-graphics/scripts/check-isaac-log.sh \
  outputs/logs/extension-smoke.log \
  ISAACSIM_GAUSSIAN_RENDERER_SMOKE_OK
```

Pass the optional third argument when the selected test also writes a JSON
artifact containing `"pass": true`.

Repeated OmniHub, missing-default-display, and GLFW warnings can be benign in
headless mode. They are not permission to ignore a missing marker, traceback,
fatal signal, CUDA error, or absent artifact.

## Finite SimulationApp pattern

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
try:
    # Import Isaac, Kit, USD, Torch, and the extension only after startup.
    # Build deterministic scene state.
    # Run bounded steps/renders.
    # Synchronize before checking asynchronous GPU errors or outputs.
    # Assert behavior and write machine-readable evidence.
    print("UNIQUE_TEST_OK")
finally:
    simulation_app.close()
```

Never leave an infinite update loop in a smoke test.

## Format-boundary fixture

Before a large benchmark or temporal OVRTX run, create a tiny scene that can
expose convention errors:

- At least two anisotropic Gaussians with unequal XYZ scales.
- Non-identity rotations around different axes.
- At least one off-axis camera.
- Distinct colors, opacities, depths, and semantic classes.
- Both `perspective` and `tangential` OVRTX modes.

The project/Ply-loader canonical rotation convention is WXYZ. For the current
OVRTX 0.3.0 tensor authoring path, `quatf[]` storage is XYZW. Convert at the
OVRTX boundary and verify orientation with this fixture. Identity quaternions
and isotropic scales cannot detect a cyclic component error.

Also verify:

- Scale values are activated exactly once.
- Opacity is activated exactly once.
- SH/DC color conversion and color space are named.
- Camera matrix direction and principal point are explicit.
- The OVRTX scene reports the expected Gaussian count.

Do not use temporal convergence or custom-kernel tuning to compensate for a
failed ingestion fixture.

## Output assertions

For a batch `B`, height `H`, and width `W`:

| Output | Expected shape | Typical dtype |
|---|---|---|
| RGB | `[B, H, W, 3]` | `float32` |
| Depth | `[B, H, W, 1]` | `float32` |
| Alpha | `[B, H, W, 1]` | `float32` |
| Semantic ID | `[B, H, W, 1]` | integer |

Check:

- CUDA device residency.
- Contiguity where required by the public contract.
- No NaN/Inf in RGB or valid foreground depth.
- Background depth sentinel and semantic ID `-1` where specified.
- Alpha bounds.
- Nonempty foreground.
- Correct scene IDs and active-camera behavior.
- No visible-record or intersection overflow.

Synchronize before reading asynchronous errors:

```python
service.synchronize()
counters = backend.check_capacity(synchronize=False)
```

## Integration assertions

For Kit extension tests:

- Add the repository `exts/` path.
- Enable `isaacsim.gaussian_renderer`.
- Verify `IExt` startup retained an extension instance.
- Enable and verify `/app/useFabricSceneDelegate`.
- Verify `omni.hydra.usdrt_delegate` is loaded.
- Shut down services before closing `SimulationApp`.

For physics/Fabric tests:

- Advance physics explicitly with `world.step`.
- Read camera transforms before and after a known USD/physics change.
- Assert only intended transforms changed.
- Record simulation time before and after renderer calls to prove the renderer
  did not advance physics.

## Visual and fidelity evidence

Use one immutable camera bundle and scene checksum across reference and
candidate renderers. Record:

- Coordinate system and matrix convention.
- Intrinsics and image dimensions.
- Background and color space.
- Near/far planes.
- Gaussian tensor convention.
- Projection and sorting modes.
- Required AOVs.

Save:

- Canonical reference and candidate tensors.
- RGB side-by-side image.
- RGB absolute-difference heatmap.
- Alpha absolute-difference image.
- Valid-depth relative-error image.
- Semantic mismatch mask.
- JSON and CSV metric reports.

The project acceptance metrics are PSNR, SSIM, LPIPS, alpha MAE, valid-depth
relative error, and semantic agreement. Every view must meet its threshold.

For OVRTX Gaussian tests, retain separate `perspective` and `tangential`
directories and metadata. Never relabel an old perspective result as
tangential.

## OVRTX stochastic and color-output contracts

One OVRTX frame may expose binary alpha/hit behavior while a temporally
accumulated reference represents a probability estimate. Record:

- Frame/sample count.
- RNG or temporal sequence settings when exposed.
- Checkpoint/resume metadata.
- Convergence curves and uncertainty for each metric.
- Whether semantic output is a per-frame hit label, temporal mode, or
  contribution-weighted class decision.

Do not generalize a projection or ray model from isotropic centered splats.
Confirm it on anisotropic off-axis fixtures before changing production math.

In minimal headless graphs, LDR tonemap construction can fail while linear HDR,
depth, or alpha are valid. Test HDR and LDR separately. Use linear HDR as the
canonical fidelity source unless the contract specifically requires LDR, then
test the color transform independently.

Use spatially coherent classes for the representative semantic baseline.
Interleaved or 1,024-field semantics are valuable stress cases, but they change
renderer overhead and stochastic voting behavior and must not define the normal
headline.

## Recommended execution ladder

After sourcing `scripts/remote_env.sh`:

```bash
/isaac-sim/python.sh -m pytest -q tests
/isaac-sim/python.sh scripts/custom_backend_smoke.py
/isaac-sim/python.sh scripts/isaac_headless_smoke.py
/isaac-sim/python.sh scripts/isaacsim_extension_smoke.py
/isaac-sim/python.sh scripts/fabric_camera_smoke.py
/isaac-sim/python.sh scripts/deterministic_cuda_smoke.py
/isaac-sim/python.sh scripts/multi_scene_cuda_smoke.py
/isaac-sim/python.sh scripts/projection_cache_cuda_smoke.py
```

Run fidelity, sanitizer, profiling, and soak tests only after this ladder
passes.

## Artifact schema checks

Every machine-readable artifact must include a schema version and explicit
required fields. Before an ad hoc query, inspect the shape:

```bash
jq 'type, (if type == "object" then keys else length end)' result.json
jq -e '
  .schema_version and
  (.pass | type == "boolean") and
  .configuration and
  .versions
' result.json >/dev/null
```

Do not assume optional keys such as `timing`, `artifacts`, or
`intersections_per_visible` exist across all benchmark schemas.

## Failure triage

- Startup/import failure: verify `SimulationApp` creation and `PYTHONPATH`
  ordering; print the active interpreter and module origins.
- Black or empty output: inspect camera convention, scene orientation,
  projection mode, quaternion layout, color conversion, opacity, HDR versus
  LDR, and RenderProduct layout.
- Wrong depth: verify distance convention and valid-depth mask.
- Wrong semantics: verify renderer ID map, semantic remap, class topology,
  temporal voting, and contribution aggregation.
- Non-determinism: check secondary depth ordering, stale outputs, and hidden
  scene/camera mutation.
- Workspace overflow: stop the run, preserve measured counters, size from the
  observed intersection demand, and rerun into a new output directory.
- Hang on exit: verify every service and `SimulationApp` is closed; inspect
  remaining renderer processes.
- OVRTX first-run delay: allow shader compilation before starting measured
  frames.
- Counter/profile failure: apply `$profile-rtx-graphics`; do not reinterpret a
  profiler permission failure as a renderer failure.
