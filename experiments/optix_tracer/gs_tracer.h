// Shared launch-parameter layout for the batched OptiX gaussian tracer.
#pragma once

#include <optix.h>

struct GsParams {
  // Scene (SoA device pointers)
  const float3* means;
  // [N*6] xx,xy,xz,yy,yz,zz. precision_mode=0 means the legacy inverse
  // covariance; mode=1 means symmetric square-root precision
  // R diag(1/scale) R^T. Applying the latter twice avoids explicitly forming
  // a matrix whose condition number is squared.
  const float*  precision6;
  int precision_mode;
  const float3* rgb;
  const float*  opacity;
  const int*    semantics;  // [N] compact semantic ids, or nullptr

  OptixTraversableHandle handle;

  // Dense mode: one launch over (width, height, envs)
  const float3* cam_origins;  // [envs]
  const float*  cam_rots9;    // [envs*9] camera->world row-major rotations,
                              // or nullptr for the built-in identity cameras
  const float4* cam_intrinsics4;  // optional per-env {fx, fy, cx, cy}
  float fx, fy, cx, cy;
  int width, height, envs;

  // Sparse mode: one launch over (nrays, 1, 1)
  const float3* ray_origins;
  const float3* ray_dirs;
  int nrays;
  int mode;  // 0 = dense pinhole grid, 1 = sparse ray list

  // Dense mode: OpenCV camera-z clipping planes. Sparse mode: ray distance.
  float t_min;
  float t_max;
  float alpha_min;         // hit threshold (1/255)
  float transmittance_min; // early-out (0.03)
  int   max_iters;
  int   use_kbuffer;       // 1 = 3DGRT k-buffer march, 0 = closest-hit march

  float4* out;                     // dense: [envs*H*W], sparse: [nrays]
  // Full output contract (optional; each may be nullptr). Conventions match
  // the raster backend exactly: depth is the weight-averaged camera-z
  // normalized by accumulated alpha (+inf where alpha ~ 0); semantic is the
  // max-weight contributor's id, -1 unless accumulated alpha >=
  // semantic_min_alpha. Sparse mode has no camera, so depth is ray distance.
  float*  out_depth;               // [rays]
  int*    out_semantic;            // [rays]
  float   semantic_min_alpha;      // raster default 0.01
  int     robot_semantic_id;       // semantic id reported at robot hits
  unsigned long long* iter_count;  // total trace iterations (for stats)

  // Hybrid: per-env robot mesh (shared GAS, rigid pose in ray space) + chrome.
  int robot_enabled;
  OptixTraversableHandle robot_handle;
  const float*  robot_rot9;   // [envs*9] world->robot row-major rotations
  const float3* robot_pos;    // [envs]
  const float3* robot_verts;  // robot-frame vertices (for face normals)
  const uint3*  robot_tris;
  float fresnel_f0;
  float robot_base_r, robot_base_g, robot_base_b;
  unsigned long long* refl_count;
};
