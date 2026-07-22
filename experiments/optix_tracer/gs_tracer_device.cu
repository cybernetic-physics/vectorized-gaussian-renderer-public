// Batched OptiX gaussian tracer — device programs.
//
// Gaussian field: 3DGRT-style custom AABB primitives (Apache-2.0 reference:
// nv-tlabs/3dgrut) with a k-buffer march and the clamping-commit optimization.
// Hybrid mode adds a triangle-mesh robot per environment (shared GAS, per-env
// rigid pose applied in ray space) with live chrome reflections: at robot hits
// the reflected ray marches the gaussian field for real scene radiance.
// Gaussians in front of the robot occlude it via tmax clamping — no separate
// compositor pass exists in this pipeline.

#include <optix.h>
#include "gs_tracer.h"

// 16 measured 18-27% faster than 8 at identical parity PSNR on the Home Scan
// (ray-matched sweep, RESULTS.md); 4 is strictly worse.
#ifndef KBUF
#define KBUF 16
#endif

extern "C" __constant__ GsParams params;

struct KBuf {
  int count;
  float t[KBUF];
  float a[KBUF];
  unsigned int prim[KBUF];
};

static __forceinline__ __device__ float3 make3(float x, float y, float z) {
  return make_float3(x, y, z);
}
static __forceinline__ __device__ float3 add3(float3 a, float3 b) {
  return make3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __forceinline__ __device__ float3 sub(float3 a, float3 b) {
  return make3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __forceinline__ __device__ float3 mul3(float3 a, float s) {
  return make3(a.x * s, a.y * s, a.z * s);
}
static __forceinline__ __device__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __forceinline__ __device__ float3 cross3(float3 a, float3 b) {
  return make3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x);
}
static __forceinline__ __device__ float3 norm3(float3 v) {
  float inv = rsqrtf(fmaxf(dot3(v, v), 1e-20f));
  return make3(v.x * inv, v.y * inv, v.z * inv);
}
static __forceinline__ __device__ float3 symv(const float* m, float3 v) {
  return make3(
      m[0] * v.x + m[1] * v.y + m[2] * v.z,
      m[1] * v.x + m[3] * v.y + m[4] * v.z,
      m[2] * v.x + m[4] * v.y + m[5] * v.z);
}
// Row-major 3x3 times vector, and transpose-times-vector.
static __forceinline__ __device__ float3 m3v(const float* R, float3 v) {
  return make3(R[0] * v.x + R[1] * v.y + R[2] * v.z,
               R[3] * v.x + R[4] * v.y + R[5] * v.z,
               R[6] * v.x + R[7] * v.y + R[8] * v.z);
}
static __forceinline__ __device__ float3 m3tv(const float* R, float3 v) {
  return make3(R[0] * v.x + R[3] * v.y + R[6] * v.z,
               R[1] * v.x + R[4] * v.y + R[7] * v.z,
               R[2] * v.x + R[5] * v.y + R[8] * v.z);
}

static __forceinline__ __device__ float gaussian_alpha_at_tstar(
    unsigned int prim, float3 o, float3 d, float* t_star) {
  float3 mu = params.means[prim];
  float3 om = sub(o, mu);
  float t, q;
  const float* P = params.precision6 + 6ull * prim;
  if (params.precision_mode == 1) {
    // B = R diag(1/scale) R^T is the symmetric square root of the precision
    // matrix. q = ||B x||^2 and the corresponding closest-point calculation
    // are equivalent to the official 3DGRT local-coordinate formulation,
    // without explicitly forming the ill-conditioned B^2.
    float3 local_o = symv(P, om);
    float3 local_d = symv(P, d);
    float denominator = fmaxf(dot3(local_d, local_d), 1e-20f);
    t = -dot3(local_o, local_d) / denominator;
    float3 local_x = add3(local_o, mul3(local_d, t));
    q = dot3(local_x, local_x);
  } else {
    float3 Ad = symv(P, d);
    float dAd = fmaxf(dot3(d, Ad), 1e-12f);
    t = -dot3(om, Ad) / dAd;
    float3 x = add3(om, mul3(d, t));
    q = dot3(x, symv(P, x));
  }
  *t_star = t;
  return params.opacity[prim] * __expf(-0.5f * q);
}

static __forceinline__ __device__ KBuf* payload_buf() {
  unsigned long long ptr =
      ((unsigned long long)optixGetPayload_0() << 32) | optixGetPayload_1();
  return (KBuf*)ptr;
}

extern "C" __global__ void __intersection__gs() {
  unsigned int prim = optixGetPrimitiveIndex();
  float3 o = optixGetObjectRayOrigin();
  float3 d = optixGetObjectRayDirection();
  float t_star;
  float alpha = gaussian_alpha_at_tstar(prim, o, d, &t_star);
  if (alpha < params.alpha_min) return;
  if (t_star <= optixGetRayTmin() || t_star >= optixGetRayTmax()) return;
  optixReportIntersection(t_star, 0, __float_as_uint(alpha));
}

extern "C" __global__ void __anyhit__gs() {
  KBuf* buf = payload_buf();
  float t = optixGetRayTmax();
  float alpha = __uint_as_float(optixGetAttribute_0());
  unsigned int prim = optixGetPrimitiveIndex();
  int c = buf->count;
  if (c == KBUF && t >= buf->t[KBUF - 1]) {
    // Buffer full, candidate beyond the k-th: COMMIT so traversal clamps.
    return;
  }
  int i = (c < KBUF) ? c : KBUF - 1;
  while (i > 0 && buf->t[i - 1] > t) {
    buf->t[i] = buf->t[i - 1];
    buf->a[i] = buf->a[i - 1];
    buf->prim[i] = buf->prim[i - 1];
    --i;
  }
  buf->t[i] = t;
  buf->a[i] = alpha;
  buf->prim[i] = prim;
  if (c < KBUF) buf->count = c + 1;
  optixIgnoreIntersection();
}

// single-hit gaussian path (baseline); guarded so k-buffer clamp hits and
// robot-mesh traces never clobber pointer payloads.
extern "C" __global__ void __closesthit__gs() {
  if (params.use_kbuffer) return;
  optixSetPayload_0(optixGetPrimitiveIndex());
  optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_2(optixGetAttribute_0());
}

// robot triangle mesh: report hit t + primitive for normal reconstruction.
extern "C" __global__ void __closesthit__tri() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
}

extern "C" __global__ void __miss__gs() {
  if (params.use_kbuffer == 0) optixSetPayload_0(0xFFFFFFFFu);
}

// Per-ray output-contract accumulator, filled by the march functions.
// depth_sum accumulates weight * (t * zscale) so dense rays produce the same
// camera-z convention as the raster backend; best_w/best_prim track the
// strongest individual contribution for the semantic output.
struct MarchAcc {
  float depth_sum;
  float best_w;
  unsigned int best_prim;
};

// March the gaussian field over [t0, tmax]; returns (rgb, T_remaining).
static __forceinline__ __device__ float4 march_kbuffer(float3 o, float3 d,
                                                       float t0, float tmax,
                                                       float zscale,
                                                       MarchAcc* acc) {
  float3 c = make3(0.f, 0.f, 0.f);
  float T = 1.0f;
  int iters = 0;
  KBuf buf;
  unsigned long long ptr = (unsigned long long)&buf;
  unsigned int p0 = (unsigned int)(ptr >> 32);
  unsigned int p1 = (unsigned int)(ptr & 0xFFFFFFFFull);
  bool done = false;
  while (!done && iters < params.max_iters) {
    buf.count = 0;
    unsigned int q2 = 0u;
    optixTrace(params.handle, o, d, t0, tmax, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, q2);
    ++iters;
    if (buf.count == 0) break;
    for (int i = 0; i < buf.count; ++i) {
      float alpha = fminf(buf.a[i], 0.99f);
      float3 col = params.rgb[buf.prim[i]];
      float w = T * alpha;
      c.x += w * col.x;
      c.y += w * col.y;
      c.z += w * col.z;
      if (acc) {
        acc->depth_sum += w * buf.t[i] * zscale;
        if (w > acc->best_w) { acc->best_w = w; acc->best_prim = buf.prim[i]; }
      }
      T *= (1.0f - alpha);
      if (T < params.transmittance_min) { done = true; break; }
    }
    if (buf.count < KBUF) break;
    float last = buf.t[buf.count - 1];
    t0 = last + fmaxf(1e-4f, last * 1e-5f);
  }
  if (params.iter_count) atomicAdd(params.iter_count, (unsigned long long)iters);
  return make_float4(c.x, c.y, c.z, T);
}

static __forceinline__ __device__ float4 march_single(float3 o, float3 d,
                                                      float t0, float tmax,
                                                      float zscale,
                                                      MarchAcc* acc) {
  float3 c = make3(0.f, 0.f, 0.f);
  float T = 1.0f;
  int iters = 0;
  while (iters < params.max_iters) {
    unsigned int p0 = 0xFFFFFFFFu, p1 = 0u, p2 = 0u;
    optixTrace(params.handle, o, d, t0, tmax, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0, p1, p2);
    ++iters;
    if (p0 == 0xFFFFFFFFu) break;
    float t = __uint_as_float(p1);
    float alpha = fminf(__uint_as_float(p2), 0.99f);
    float3 col = params.rgb[p0];
    float w = T * alpha;
    c.x += w * col.x;
    c.y += w * col.y;
    c.z += w * col.z;
    if (acc) {
      acc->depth_sum += w * t * zscale;
      if (w > acc->best_w) { acc->best_w = w; acc->best_prim = p0; }
    }
    T *= (1.0f - alpha);
    if (T < params.transmittance_min) break;
    t0 = t + fmaxf(1e-4f, t * 1e-5f);
  }
  if (params.iter_count) atomicAdd(params.iter_count, (unsigned long long)iters);
  return make_float4(c.x, c.y, c.z, T);
}

extern "C" __global__ void __raygen__gs() {
  uint3 idx = optixGetLaunchIndex();
  float3 o, d;
  size_t out_idx;
  int env = 0;
  // Converts ray-t to camera z for the depth output (1 in sparse mode, where
  // there is no camera and depth is plain ray distance).
  float zscale = 1.0f;
  float ray_tmin = params.t_min;
  float ray_tmax = params.t_max;
  if (params.mode == 1) {
    int r = idx.x;
    if (r >= params.nrays) return;
    o = params.ray_origins[r];
    d = norm3(params.ray_dirs[r]);
    out_idx = r;
  } else {
    int x = idx.x, y = idx.y;
    env = idx.z;
    o = params.cam_origins[env];
    float fx = params.fx, fy = params.fy, cx = params.cx, cy = params.cy;
    if (params.cam_intrinsics4) {
      float4 k = params.cam_intrinsics4[env];
      fx = k.x; fy = k.y; cx = k.z; cy = k.w;
    }
    d = make3(((float)x + 0.5f - cx) / fx,
              ((float)y + 0.5f - cy) / fy, 1.0f);
    // Camera-space forward component of the normalized direction: for the
    // unnormalized (dx, dy, 1) this is 1/||d||, invariant to the world
    // rotation applied below.
    zscale = rsqrtf(fmaxf(dot3(d, d), 1e-20f));
    // Dense near/far follow the raster and gsplat camera-z convention. OptiX
    // traces a normalized ray, so convert those planes to per-ray distances.
    ray_tmin = params.t_min / zscale;
    ray_tmax = params.t_max / zscale;
    // Optional per-env camera rotation (camera->world, row-major); the
    // built-in synthetic cameras use identity.
    if (params.cam_rots9) d = m3v(params.cam_rots9 + 9 * env, d);
    d = norm3(d);
    out_idx = ((size_t)env * params.height + y) * params.width + x;
  }

  // Optional per-env robot: trace the shared robot GAS with the ray expressed
  // in this env's robot frame (rotation-only transform preserves t).
  float t_rob = 3.0e38f;
  float3 n_world = make3(0.f, 0.f, 0.f);
  if (params.robot_enabled && params.mode == 0) {
    const float* R = params.robot_rot9 + 9 * env;  // world -> robot
    float3 orl = m3v(R, sub(o, params.robot_pos[env]));
    float3 drl = m3v(R, d);
    unsigned int p0 = __float_as_uint(3.0e38f), p1 = 0u, p2 = 0u;
    optixTrace(params.robot_handle, orl, drl, ray_tmin, ray_tmax, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               1, 1, 0, p0, p1, p2);
    t_rob = __uint_as_float(p0);
    if (t_rob < 3.0e37f) {
      uint3 tri = params.robot_tris[p1];
      float3 e1 = sub(params.robot_verts[tri.y], params.robot_verts[tri.x]);
      float3 e2 = sub(params.robot_verts[tri.z], params.robot_verts[tri.x]);
      float3 n_local = norm3(cross3(e1, e2));
      n_world = norm3(m3tv(R, n_local));  // robot -> world
      if (dot3(n_world, d) > 0.f) n_world = mul3(n_world, -1.f);
    }
  }

  bool robot_hit = t_rob < 3.0e37f;
  float gs_tmax = robot_hit ? t_rob : ray_tmax;
  MarchAcc acc;
  acc.depth_sum = 0.f;
  acc.best_w = -1.f;
  acc.best_prim = 0xFFFFFFFFu;
  float4 gs = params.use_kbuffer
                  ? march_kbuffer(o, d, ray_tmin, gs_tmax, zscale, &acc)
                  : march_single(o, d, ray_tmin, gs_tmax, zscale, &acc);
  float3 c = make3(gs.x, gs.y, gs.z);
  float T = gs.w;
  bool robot_is_best = false;

  if (robot_hit && T > 0.01f) {
    // Chrome: live reflection into the gaussian field from the hit point.
    // The reflection march is radiance-only — it must not perturb the primary
    // ray's depth/semantic outputs.
    float cos_i = fminf(fmaxf(-dot3(d, n_world), 0.0f), 1.0f);
    float3 refl = norm3(sub(d, mul3(n_world, 2.0f * dot3(d, n_world))));
    float3 hit_p = add3(o, mul3(d, t_rob));
    float3 hit_off = add3(hit_p, mul3(n_world, 1e-3f));
    float4 rr = march_kbuffer(hit_off, refl, 1e-3f, params.t_max, 1.0f, nullptr);
    float f = params.fresnel_f0 +
              (1.0f - params.fresnel_f0) * powf(1.0f - cos_i, 5.0f);
    float3 base = make3(params.robot_base_r * cos_i, params.robot_base_g * cos_i,
                        params.robot_base_b * cos_i);
    float3 robot_col = make3(
        f * rr.x + (1.0f - f) * base.x,
        f * rr.y + (1.0f - f) * base.y,
        f * rr.z + (1.0f - f) * base.z);
    c = add3(c, mul3(robot_col, T));
    // The opaque robot surface contributes its remaining weight T at t_rob.
    acc.depth_sum += T * t_rob * zscale;
    if (T > acc.best_w) { acc.best_w = T; robot_is_best = true; }
    T = 0.0f;
    if (params.refl_count) atomicAdd(params.refl_count, 1ull);
  }

  float alpha = 1.0f - T;
  params.out[out_idx] = make_float4(c.x, c.y, c.z, alpha);
  if (params.out_depth)
    params.out_depth[out_idx] =
        alpha > 1.0e-8f ? acc.depth_sum / alpha : __int_as_float(0x7f800000);
  if (params.out_semantic) {
    int sem = -1;
    if (acc.best_w >= 0.f && alpha >= params.semantic_min_alpha) {
      if (robot_is_best)
        sem = params.robot_semantic_id;
      else if (params.semantics && acc.best_prim != 0xFFFFFFFFu)
        sem = params.semantics[acc.best_prim];
    }
    params.out_semantic[out_idx] = sem;
  }
}
