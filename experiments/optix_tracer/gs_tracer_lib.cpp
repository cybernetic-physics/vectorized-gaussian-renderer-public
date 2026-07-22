// Zero-copy C API for the batched OptiX gaussian tracer.
//
// Built as libgs_tracer.so (see build.sh) and consumed from torch via ctypes
// (gs_tracer_torch.py). All output pointers are raw CUDA device pointers —
// torch tensors' data_ptr() — so rendered RGBA/depth/semantic land directly
// in torch memory with no host round-trip and no copies. CUDA camera tensors
// and sparse-ray inputs are consumed on-device; CPU cameras use a staged path.
//
// Calls are synchronous on the caller-provided CUDA stream. Launching on
// torch's current stream preserves producer/consumer ordering without a
// device-wide synchronization or a race with non-default streams.

#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>

#include <optix_function_table_definition.h>

#include "gs_tracer_setup.h"

namespace {

struct GstContext {
  GstScene scene;
  GstPipeline pipe;
  GsParams base = {};
  void* d_params = nullptr;
  // Per-call camera scratch, grown on demand.
  void* d_cam_origins = nullptr;
  void* d_cam_rots = nullptr;
  void* d_cam_intrinsics4 = nullptr;
  int cam_capacity = 0;
};

void ensure_cam_capacity(GstContext* c, int envs) {
  if (envs <= c->cam_capacity) return;
  if (c->d_cam_origins) CUDA_CHECK(cudaFree(c->d_cam_origins));
  if (c->d_cam_rots) CUDA_CHECK(cudaFree(c->d_cam_rots));
  if (c->d_cam_intrinsics4) CUDA_CHECK(cudaFree(c->d_cam_intrinsics4));
  CUDA_CHECK(cudaMalloc(&c->d_cam_origins, (size_t)envs * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->d_cam_rots, (size_t)envs * 9 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->d_cam_intrinsics4,
                        (size_t)envs * 4 * sizeof(float)));
  c->cam_capacity = envs;
}

cudaStream_t stream_from_handle(unsigned long long handle) {
  return reinterpret_cast<cudaStream_t>((uintptr_t)handle);
}

bool configure_trace_range(GsParams* p, float near_plane, float far_plane) {
  if (!std::isfinite(near_plane) || !std::isfinite(far_plane)) return false;
  if (near_plane > 0.f) p->t_min = near_plane;
  if (far_plane > 0.f) p->t_max = far_plane;
  return p->t_min >= 0.f && p->t_max > p->t_min;
}

__global__ void unpack_device_cameras(const float* viewmats,
                                      const float* intrinsics,
                                      float3* origins, float* rotations,
                                      float4* intrinsics4, int envs) {
  int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= envs) return;
  const float* vm = viewmats + (size_t)e * 16;
  const float* k = intrinsics + (size_t)e * 9;
  float tx = vm[3], ty = vm[7], tz = vm[11];
  origins[e] = make_float3(
      -(vm[0] * tx + vm[4] * ty + vm[8] * tz),
      -(vm[1] * tx + vm[5] * ty + vm[9] * tz),
      -(vm[2] * tx + vm[6] * ty + vm[10] * tz));
  float* r = rotations + (size_t)e * 9;
  r[0] = vm[0]; r[1] = vm[4]; r[2] = vm[8];
  r[3] = vm[1]; r[4] = vm[5]; r[5] = vm[9];
  r[6] = vm[2]; r[7] = vm[6]; r[8] = vm[10];
  intrinsics4[e] = make_float4(k[0], k[4], k[2], k[5]);
}

}  // namespace

extern "C" {

// Loads the scene dump, builds the GAS + pipeline. Returns nullptr for a
// recoverable input/setup failure; CUDA/OptiX runtime failures remain fatal in
// this experiments/ implementation.
// info_out receives {count, center xyz, radius, has_semantics}.
void* gst_create(const char* dump_path, const char* ptx_path,
                 double* info_out) {
  if (!dump_path || !ptx_path || !info_out) return nullptr;
  GstContext* c = new GstContext();
  if (!gst_load_dump(dump_path, &c->scene)) {
    delete c;
    return nullptr;
  }
  if (!gst_build_pipeline(c->scene, ptx_path, &c->pipe)) {
    gst_destroy_pipeline(&c->pipe);
    gst_destroy_scene(&c->scene);
    delete c;
    return nullptr;
  }
  gst_release_static_build_inputs(&c->scene);
  gst_base_params(c->scene, &c->base);
  c->base.handle = c->pipe.handle;
  CUDA_CHECK(cudaMalloc(&c->d_params, sizeof(GsParams)));
  info_out[0] = (double)c->scene.n;
  info_out[1] = c->scene.center[0];
  info_out[2] = c->scene.center[1];
  info_out[3] = c->scene.center[2];
  info_out[4] = c->scene.radius;
  info_out[5] = c->scene.has_semantics ? 1.0 : 0.0;
  return c;
}

double gst_gas_build_ms(void* handle) {
  return ((GstContext*)handle)->pipe.gas_ms;
}

// Dense batched pinhole render into caller-owned device buffers.
//   h_cams: HOST float array [envs][12] = origin xyz + camera->world R row-major
//   h_intrinsics4: HOST float array [envs][4] = fx, fy, cx, cy
//   d_out_rgba: device float4 [envs*height*width]  (required)
//   d_out_depth / d_out_semantic: device float / int32 buffers or 0
int gst_render_dense(void* handle, int envs, int width, int height,
                     const float* h_cams, const float* h_intrinsics4,
                     unsigned long long d_out_rgba,
                     unsigned long long d_out_depth,
                     unsigned long long d_out_semantic, int use_kbuffer,
                     int max_iters, float alpha_min, float tmin,
                     float near_plane, float far_plane,
                     unsigned long long stream_handle) {
  if (!handle || !h_cams || !h_intrinsics4 || envs <= 0 || width <= 0 ||
      height <= 0 || !d_out_rgba)
    return 1;
  GstContext* c = (GstContext*)handle;
  cudaStream_t stream = stream_from_handle(stream_handle);
  GsParams p = c->base;
  if (max_iters > 0) p.max_iters = max_iters;
  if (alpha_min > 0.f) p.alpha_min = alpha_min;
  if (tmin > 0.f) p.transmittance_min = tmin;
  if (!configure_trace_range(&p, near_plane, far_plane)) return 2;
  ensure_cam_capacity(c, envs);
  std::vector<float> origins((size_t)envs * 3), rots((size_t)envs * 9);
  for (int e = 0; e < envs; ++e) {
    memcpy(&origins[(size_t)e * 3], h_cams + (size_t)e * 12, 12);
    memcpy(&rots[(size_t)e * 9], h_cams + (size_t)e * 12 + 3, 36);
  }
  CUDA_CHECK(cudaMemcpyAsync(c->d_cam_origins, origins.data(),
                             origins.size() * 4, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(c->d_cam_rots, rots.data(), rots.size() * 4,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(c->d_cam_intrinsics4, h_intrinsics4,
                             (size_t)envs * 4 * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  p.mode = 0;
  p.envs = envs;
  p.width = width;
  p.height = height;
  p.cam_origins = (const float3*)c->d_cam_origins;
  p.cam_rots9 = (const float*)c->d_cam_rots;
  p.cam_intrinsics4 = (const float4*)c->d_cam_intrinsics4;
  p.out = (float4*)d_out_rgba;
  p.out_depth = (float*)d_out_depth;
  p.out_semantic = (int*)d_out_semantic;
  p.use_kbuffer = use_kbuffer;
  CUDA_CHECK(cudaMemcpyAsync(c->d_params, &p, sizeof(GsParams),
                             cudaMemcpyHostToDevice, stream));
  OPTIX_CHECK(optixLaunch(c->pipe.pipeline, stream, (CUdeviceptr)c->d_params,
                          sizeof(GsParams), &c->pipe.sbt, width, height, envs));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

// Dense path for contiguous CUDA float32 viewmats [B,4,4] and intrinsics
// [B,3,3]. A small conversion kernel writes the OptiX camera scratch on the
// same stream, so no camera tensor leaves the GPU.
int gst_render_dense_device(void* handle, int envs, int width, int height,
                            unsigned long long d_viewmats,
                            unsigned long long d_intrinsics,
                            unsigned long long d_out_rgba,
                            unsigned long long d_out_depth,
                            unsigned long long d_out_semantic,
                            int use_kbuffer, int max_iters, float alpha_min,
                            float tmin, float near_plane, float far_plane,
                            unsigned long long stream_handle) {
  if (!handle || envs <= 0 || width <= 0 || height <= 0 || !d_viewmats ||
      !d_intrinsics || !d_out_rgba)
    return 1;
  GstContext* c = (GstContext*)handle;
  cudaStream_t stream = stream_from_handle(stream_handle);
  GsParams p = c->base;
  if (max_iters > 0) p.max_iters = max_iters;
  if (alpha_min > 0.f) p.alpha_min = alpha_min;
  if (tmin > 0.f) p.transmittance_min = tmin;
  if (!configure_trace_range(&p, near_plane, far_plane)) return 2;
  ensure_cam_capacity(c, envs);
  int threads = 128;
  int blocks = (envs + threads - 1) / threads;
  unpack_device_cameras<<<blocks, threads, 0, stream>>>(
      (const float*)d_viewmats, (const float*)d_intrinsics,
      (float3*)c->d_cam_origins, (float*)c->d_cam_rots,
      (float4*)c->d_cam_intrinsics4, envs);
  CUDA_CHECK(cudaGetLastError());

  p.mode = 0;
  p.envs = envs;
  p.width = width;
  p.height = height;
  p.cam_origins = (const float3*)c->d_cam_origins;
  p.cam_rots9 = (const float*)c->d_cam_rots;
  p.cam_intrinsics4 = (const float4*)c->d_cam_intrinsics4;
  p.out = (float4*)d_out_rgba;
  p.out_depth = (float*)d_out_depth;
  p.out_semantic = (int*)d_out_semantic;
  p.use_kbuffer = use_kbuffer;
  CUDA_CHECK(cudaMemcpyAsync(c->d_params, &p, sizeof(GsParams),
                             cudaMemcpyHostToDevice, stream));
  OPTIX_CHECK(optixLaunch(c->pipe.pipeline, stream, (CUdeviceptr)c->d_params,
                          sizeof(GsParams), &c->pipe.sbt,
                          width, height, envs));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

// Sparse ray-list render. d_ray_origins / d_ray_dirs are device float3 [n]
// buffers (e.g. torch tensors); outputs as in gst_render_dense. Depth output
// is plain ray distance (no camera in this mode).
int gst_render_sparse(void* handle, long long nrays,
                      unsigned long long d_ray_origins,
                      unsigned long long d_ray_dirs,
                      unsigned long long d_out_rgba,
                      unsigned long long d_out_depth,
                      unsigned long long d_out_semantic, int use_kbuffer,
                      int max_iters, float alpha_min, float tmin,
                      float near_plane, float far_plane,
                      unsigned long long stream_handle) {
  if (!handle || nrays < 0 || nrays > std::numeric_limits<int>::max()) return 1;
  if (nrays == 0) return 0;
  if (!d_ray_origins || !d_ray_dirs || !d_out_rgba) return 1;
  GstContext* c = (GstContext*)handle;
  cudaStream_t stream = stream_from_handle(stream_handle);
  GsParams p = c->base;
  p.mode = 1;
  p.nrays = (int)nrays;
  p.ray_origins = (const float3*)d_ray_origins;
  p.ray_dirs = (const float3*)d_ray_dirs;
  p.out = (float4*)d_out_rgba;
  p.out_depth = (float*)d_out_depth;
  p.out_semantic = (int*)d_out_semantic;
  p.use_kbuffer = use_kbuffer;
  if (max_iters > 0) p.max_iters = max_iters;
  if (alpha_min > 0.f) p.alpha_min = alpha_min;
  if (tmin > 0.f) p.transmittance_min = tmin;
  if (!configure_trace_range(&p, near_plane, far_plane)) return 2;

  CUDA_CHECK(cudaMemcpyAsync(c->d_params, &p, sizeof(GsParams),
                             cudaMemcpyHostToDevice, stream));
  OPTIX_CHECK(optixLaunch(c->pipe.pipeline, stream, (CUdeviceptr)c->d_params,
                          sizeof(GsParams), &c->pipe.sbt,
                          (unsigned int)nrays, 1, 1));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

void gst_destroy(void* handle) {
  if (!handle) return;
  GstContext* c = (GstContext*)handle;
  cudaDeviceSynchronize();
  if (c->d_params) cudaFree(c->d_params);
  if (c->d_cam_origins) cudaFree(c->d_cam_origins);
  if (c->d_cam_rots) cudaFree(c->d_cam_rots);
  if (c->d_cam_intrinsics4) cudaFree(c->d_cam_intrinsics4);
  gst_destroy_pipeline(&c->pipe);
  gst_destroy_scene(&c->scene);
  delete c;
}

}  // extern "C"
