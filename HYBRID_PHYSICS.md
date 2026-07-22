# Hybrid Gaussian visuals and PhysX contact

Gaussian splats are an appearance representation. Their densities, opacity,
and learned radiance have no PhysX shape, signed-distance field, material, or
contact-report contract. A robot can therefore be physically correct in the
same Isaac stage while the Gaussian scene is visually correct, but the
Gaussian data itself must never be treated as a collision mesh.

## Vectorizable architecture

```text
shared static PLY ──> Custom CUDA Gaussian renderer ──> RGB/alpha/depth tensors
                                  ^
Fabric batched camera poses ──────┘

coarse static collision proxy ──> PhysX <── Unitree G1 USD/URDF colliders
                                      │
                                      └── batched robot state/contact tensors

dynamic robot mesh RTX/OVRTX layer ──> GPU depth-aware compositor <── Gaussian tensors
```

The canonical PLY is loaded once and shared across camera/environment batches.
The collision proxy is also shared across environments; only each environment's
transform and each robot's articulation state are batched. Do not instantiate
one collider per Gaussian or one Gaussian scene per robot.

## Collision-proxy workflow

Use `scripts/build_splat_collision_proxy.sh` on the RTX host after preparing
the canonical PLY:

```bash
scripts/build_splat_collision_proxy.sh \
  /workspace/datasets/home-scan-lod0.ply \
  /workspace/datasets/home-scan-lod0.voxel.json \
  <known-free-space-seed-x,y,z> 0.08 0.10 interior
```

The generated `.collision.glb` is the PhysX static mesh candidate. For indoor
scans, `interior` uses a seeded navigable-space carve; for exterior scans use
`exterior`. Validate proxy alignment against known floor/wall points and reject
thin, floating, or incomplete visual regions before running contact-sensitive
tasks. A hand-authored floor, walls, boxes, or a reconstructed CAD mesh is
often a better first proxy than a scan-derived mesh.

Then validate that the GLB is actually cooked as a static triangle collider,
rather than merely loading as a USD mesh:

```bash
/isaac-sim/python.sh scripts/isaacsim_collision_proxy_import_smoke.py \
  --collision-glb /workspace/assets/home-scan-collision-proxy-v1/home-scan-lod0.collision.glb \
  --report outputs/home-scan-collision-proxy-v1/contact-report.json
```

The smoke applies `PhysicsCollisionAPI`, `PhysicsMeshCollisionAPI`,
`PhysxCollisionAPI`, and `PhysxTriangleMeshCollisionAPI` to every referenced
proxy mesh, then drops a dynamic sphere above a measured proxy vertex. It
fails if the probe does not move under gravity or falls through. The first
Home Scan run built a 1.41 MB GLB from 21.5 million Gaussians at 8 cm voxel
resolution and passed this contact probe. This validates the static terrain
layer; it is not a claim that the visual Gaussian field itself collides.

## Rendering boundary

The custom kernel returns GPU tensors for the static Gaussian layer only. RTX
or OVRTX may render dynamic USD mesh robots, but that work is a separate layer
with separately measured timing. A depth-aware GPU compositor must select the
nearer valid layer and premultiplied-alpha blend ties; it must not copy either
layer through the CPU. This preserves vectorization while keeping visual and
physical claims honest.

`isaacsim_gaussian_renderer.hybrid_compositor.composite_gaussian_and_mesh`
now supplies the tensor-level, depth-ordered straight-alpha compositor for
matching `[B,H,W,*]` Torch layers. Its CUDA smoke verifies that the compositor
stays on the RTX GPU. `scripts/isaacsim_home_g1_hybrid_composite_smoke.py`
also proves the direct Isaac RTX bridge with an imported G1: Replicator's
`LdrColor` and `distance_to_image_plane` arrive as CUDA tensors, a finite
depth-mask supplies the opaque mesh alpha, and no layer is copied to the host
until the evidence PNGs are written. The remaining engineering is batched
camera/robot orchestration, HDR colour matching, and measured vectorized
throughput—not a missing basic AOV bridge.

## Current validation boundary

`scripts/ovrtx_home_g1_mesh_smoke.py` is the reproducible OVRTX composition
probe: it references the imported G1 USD beside the checksummed Home Scan
field and requires a nontrivial RGB footprint versus the Gaussian-only render.
It is intentionally strict. A successful URDF import or a few alpha-only
pixels does **not** validate a usable robot renderer or the still-missing GPU
depth compositor. Keep this result separate from the completed custom Gaussian
and G1 collision-import smokes.
