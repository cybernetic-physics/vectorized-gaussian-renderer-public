# Gaussian LiDAR validation evidence

This file records the finite validation run performed on 2026-07-16. The
implementation contract and frozen thresholds are in `GAUSSIAN_LIDAR.md`.
Artifacts remain under
`/workspace/agent-worktrees/gaussian-lidar-20260716/outputs` on the running
Vast instance and are synchronized to the local ignored `outputs/` directory.

## Machine and dependencies

- Required source base: `6a54ea67a1203683a8d5b708b78d235ea0395092`;
  startup HEAD matched it exactly and the ancestor check passed.
- Vast instance: `45122056`, SSH alias `vast-gsplat-lidar-45122056`,
  `$0.45666666666666667/hour`.
- GPU: one NVIDIA GeForce RTX 4090, 24,564 MiB; driver `590.48.01`.
- Isaac Sim `6.0.1` at `/isaac-sim`; Torch `2.11.0+cu128`; CUDA toolkit
  `12.8`; compute capability `8.9`.
- OVRTX runtime `0.3.0.312915`, source
  `29d11037fbcaed0f0f53e7f32d17bd0486fd453b`.
- gsplat `77ab983ffe43420b2131669cb35776b883ca4c3c`; Kit App Template
  `76542106c2e97d1bbc76bf04259bec19887da554`.
- Nsight Systems `2024.6.2`, Nsight Compute `2025.1.1`, Nsight Graphics
  `2026.2.0.0`, Compute Sanitizer `2025.1.0`, Vulkan `1.3.275`.

## Correctness and integration

Commands were run after `source scripts/remote_env.sh` through the repository
GPU coordination lock.

```bash
/isaac-sim/python.sh -m pytest -q tests
/isaac-sim/python.sh scripts/lidar_cuda_smoke.py
/isaac-sim/python.sh scripts/lidar_randomized_oracle.py
/isaac-sim/python.sh scripts/lidar_deferred_work_audit.py --single-call
/isaac-sim/python.sh scripts/lidar_deferred_work_audit.py
/isaac-sim/python.sh scripts/isaacsim_lidar_smoke.py
```

The final-source unit suite passed 85 tests. The CUDA
smoke matched the independent float64 oracle with maximum errors
`6.85e-8 m` range, `6.85e-8 m` position and `3.57e-8` intensity; all dense
outputs were bitwise stable on fixed-state replay. Its native boundary fixture
returned `0.0500000007 m` and `200.0 m`, K=1 was bitwise equal to the first
K=2 return, and a joint rigid sensor/scene transform was equivariant. Four
held-out randomized seeds had maxima `7.47e-7 m` range, `6.51e-7 m` position and `1.59e-7`
intensity. Batched results were bitwise equal to concatenated individual
results, and input permutation changed float outputs by no more than `2e-6`.
All overflow/error counters were zero.

The deferred-work audit passed a fresh single call and a randomized prime
11-call run. Guard canaries stayed intact, same-stream immediate
fingerprints never changed after later calls, ring and synchronize-each modes
were bitwise equal, in-place input mutations changed results, cloned storage
with equal values did not, and two independent service/current-stream pairs
agreed.

The finite Isaac smoke loaded the extension, explicitly created LiDAR, returned
CUDA tensors, and left physics time unchanged at `0.08333333767950535`.
The captured log contained no hidden Python traceback.

## OVRTX probes

The exact OVRTX source example was used. Mesh-only LiDAR returned 56,348
points. Against those ray directions, the custom Gaussian wall differed from
the analytic mesh plane by at most `8.35e-7 m`; the OVRTX mesh samples differed
by at most `0.0152 m`. Intensity was not compared because OVRTX material and
custom reflectivity normalization were not calibrated.

Gaussian-only and co-located Gaussian-plus-mesh scenes both emitted the OVRTX
error `Gaussians are unsupported for motion BVH` and did not produce a usable
LiDAR frame. No OVRTX speedup is claimed.

## Performance

`scripts/run_lidar_matrix.sh lidar-matrix-exact-20260716` ran three fresh
processes for each of the analytic 4,096-Gaussian and medium
100,000-Gaussian scenes. Every process covered B `1,8,32,64,128`, R
`16,384,65,536,131,072`, K `1,2`, static/changing transforms and synchronized
latency/ring-buffer throughput, with full dense outputs. Steady-state CUDA
allocation delta was zero in all highlighted rows.

| Scene/configuration | Three-run GPU result | Per-environment rate |
|---|---:|---:|
| Analytic, B1/R131072/K1 latency p50 | 0.848-0.852 ms | 1,171.5-1,176.2 Hz |
| Analytic, B64/R131072/K1 throughput | 16.951-16.968 ms | 58.93-58.99 Hz |
| Analytic, B64/R131072/K2 throughput | 19.238-19.311 ms | 51.78-51.98 Hz |
| Medium, B1/R131072/K1 latency p50 | 1.183-1.187 ms | 844.9-850.4 Hz |
| Medium, B1/R131072/K2 latency p50 | 2.389-2.463 ms | 404.1-417.3 Hz |
| Medium, B64/R131072/K1 throughput | 28.736-28.775 ms | 34.75-34.80 Hz |
| Medium, B64/R131072/K2 throughput | 51.676-51.823 ms | 19.30-19.35 Hz |
| Medium, B128/R131072/K1 throughput | 57.013-57.065 ms | 17.52-17.54 Hz |

The medium LBVH occupied 1,186,408 bytes and caller-owned scene attributes
5,600,000 bytes. Initial registration, including cold CUDA value validation,
took 112.51-114.27 ms; explicit same-count rebuilds took 0.988-1.273 ms.
For medium B64/R131072/K1, the three runs sustained 291.5-291.9 million
rays/s, 2,224-2,227 aggregate scans/s and 11.95-11.97 GB/s of dense output
writes, with 25.35 node visits and 19.45 primitive tests per ray. Peak reserved
CUDA memory was 926,941,184 bytes and steady allocation delta was zero.
Medium B1/R131072/K1 latency p95 was 1.202-1.214 ms across the three processes.

The exact native module also completed a finite 600.012-second soak with
319,100 measured calls and 41,825,599,488 traced rays. Allocated memory stayed
at 48,208,896 bytes, reserved memory stayed at 60,817,408 bytes, and every
overflow/error counter was zero. An earlier harness attempt correctly failed
because the audit's own first boolean-index validation reserved 2 MiB; those
audit operations were prewarmed before the final renderer memory baseline.

The available 21,497,908-Gaussian PLY had SHA-256
`ded8e9561875ea3feac5e87a196891600c94c621ce83da1090ca810a1055d935`,
not the required
`29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267`.
The full Home Scan benchmark was therefore not run and no substitute or LOD
result is reported as Home Scan evidence.

## Profilers, sanitizers and camera isolation

`scripts/profile_nsys.sh` captured 14 public LiDAR calls: 14 NVTX submission
ranges, 14 output-clear kernels and 14 trace kernels. Trace consumed 98.55% of
GPU kernel time and output initialization 1.20%. Raw `.nsys-rep`, SQLite,
exported CSV and parsed JSON are retained under
`outputs/profiles/lidar-medium-b64-r131072-k1-exact`. SQLite correlation found
exactly 14 public NVTX ranges and 56 runtime calls inside them, all four kernel
launches per invocation; there were no allocator or free calls inside any
public range.

Nsight Compute counter collection was preflighted and correctly stopped:
`RmProfilingAdminOnly=1` while the container lacks `CAP_SYS_ADMIN`. The JSON
preflight is retained rather than presenting invented kernel counters.
Compute Sanitizer memcheck reported zero errors and racecheck reported zero
hazards against the exact final native module.

The camera-only fresh-process test did not import the LiDAR native loader or
create a LiDAR build artifact. Existing deterministic, projection-cache,
multi-scene CUDA and Isaac extension smokes passed. Three 30-iteration
unprofiled camera processes at the required base and three at the LiDAR branch
all had the same median p50, `0.460800 ms` (0.0% change). A camera-only Nsight
trace contained zero LiDAR kernel rows and its steady-state allocation delta
was zero.

Dynamic geometry, scene motion distortion, robot/mesh intersection and
physically calibrated NIR intensity remain explicitly out of scope.
