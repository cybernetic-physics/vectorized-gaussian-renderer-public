// Shared host-side setup for the batched OptiX gaussian tracer: scene-dump
// loading, gaussian GAS build, and module/pipeline/SBT creation. Used by both
// the CLI benchmark (gs_tracer_main.cpp) and the zero-copy library
// (gs_tracer_lib.cpp).
//
// CUDA/OptiX runtime failures remain fatal in this experiments/ harness.
// Recoverable file-contract failures return false so an in-process caller can
// reject malformed input before allocating from an untrusted count field.
#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "gs_tracer.h"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err),   \
              __FILE__, __LINE__);                                           \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

#define OPTIX_CHECK(call)                                                    \
  do {                                                                       \
    OptixResult res = (call);                                                \
    if (res != OPTIX_SUCCESS) {                                              \
      fprintf(stderr, "OptiX error %d at %s:%d\n", (int)res, __FILE__,       \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

struct EmptyRecord {
  __align__(OPTIX_SBT_RECORD_HEADER_SIZE) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

template <typename T>
static CUdeviceptr gst_upload(const std::vector<T>& host) {
  void* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return (CUdeviceptr)d;
}

static void gst_context_log(unsigned int level, const char* tag,
                            const char* msg, void*) {
  if (level <= 3) fprintf(stderr, "[optix][%s] %s\n", tag, msg);
}

struct GstScene {
  int64_t n = 0;
  float center[3] = {0, 0, 0};
  float radius = 0.f;
  bool has_semantics = false;
  bool has_stable_precision = false;
  CUdeviceptr d_means = 0, d_ic = 0, d_rgb = 0, d_opac = 0, d_aabb = 0,
              d_sem = 0;
};

// Loads the export_gaussians_bin.py dump and uploads every array. The
// trailing int32 semantics section is optional (older dumps lack it).
static bool gst_load_dump(const std::string& path, GstScene* s) {
  static_assert(sizeof(float) == 4 && sizeof(int) == 4,
                "The Gaussian dump contract requires 32-bit float/int.");
  std::ifstream in(path, std::ios::binary);
  if (!in) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
  in.seekg(0, std::ios::end);
  std::streampos end = in.tellg();
  if (end < 0) { fprintf(stderr, "cannot size %s\n", path.c_str()); return false; }
  uint64_t file_bytes = (uint64_t)end;
  in.seekg(0, std::ios::beg);
  if (file_bytes < 24) {
    fprintf(stderr, "short header in %s\n", path.c_str());
    return false;
  }
  in.read((char*)&s->n, 8);
  in.read((char*)s->center, 12);
  in.read((char*)&s->radius, 4);
  if (!in) { fprintf(stderr, "short header in %s\n", path.c_str()); return false; }
  int64_t n = s->n;
  if (n <= 0 || (uint64_t)n > std::numeric_limits<unsigned int>::max() ||
      !std::isfinite(s->center[0]) || !std::isfinite(s->center[1]) ||
      !std::isfinite(s->center[2]) || !std::isfinite(s->radius) ||
      s->radius <= 0.f) {
    fprintf(stderr, "invalid dump header in %s (n=%lld radius=%g)\n",
            path.c_str(), (long long)n, s->radius);
    return false;
  }
  const uint64_t header_bytes = 24;
  const uint64_t bytes_per_legacy_primitive = 19 * sizeof(float);
  const uint64_t semantics_bytes = (uint64_t)n * sizeof(int);
  if ((uint64_t)n >
      (std::numeric_limits<uint64_t>::max() - header_bytes) /
          bytes_per_legacy_primitive) {
    fprintf(stderr, "dump size overflow in %s (n=%lld)\n",
            path.c_str(), (long long)n);
    return false;
  }
  const uint64_t no_semantics_size =
      header_bytes + (uint64_t)n * bytes_per_legacy_primitive;
  const uint64_t semantics_size = no_semantics_size + semantics_bytes;
  const uint64_t stable_size = semantics_size + 8;
  if (semantics_size < no_semantics_size || stable_size < semantics_size ||
      (file_bytes != no_semantics_size && file_bytes != semantics_size &&
       file_bytes != stable_size)) {
    fprintf(stderr, "unsupported dump size %llu for %s (n=%lld)\n",
            (unsigned long long)file_bytes, path.c_str(), (long long)n);
    return false;
  }
  s->has_semantics = file_bytes >= semantics_size;
  s->has_stable_precision = file_bytes == stable_size;

  std::vector<float> means((size_t)n * 3), ic((size_t)n * 6),
      rgb((size_t)n * 3), opac((size_t)n), aabb((size_t)n * 6);
  in.read((char*)means.data(), means.size() * 4);
  in.read((char*)ic.data(), ic.size() * 4);
  in.read((char*)rgb.data(), rgb.size() * 4);
  in.read((char*)opac.data(), opac.size() * 4);
  in.read((char*)aabb.data(), aabb.size() * 4);
  if (!in) { fprintf(stderr, "short read on %s\n", path.c_str()); return false; }
  // v3 reuses the original six-float precision slot for the stable square
  // root and appends only an eight-byte representation tag after semantics.
  std::vector<int> sem;
  if (s->has_semantics) {
    sem.resize(n);
    in.read((char*)sem.data(), sem.size() * sizeof(int));
  }
  if (s->has_stable_precision) {
    char tag[8] = {};
    in.read(tag, sizeof(tag));
    if (memcmp(tag, "GST3SQRT", sizeof(tag)) != 0) {
      fprintf(stderr, "invalid stable-precision tag in %s\n", path.c_str());
      return false;
    }
  }
  if (!in) { fprintf(stderr, "short trailing section in %s\n", path.c_str()); return false; }
  s->d_means = gst_upload(means);
  s->d_ic = gst_upload(ic);
  s->d_rgb = gst_upload(rgb);
  s->d_opac = gst_upload(opac);
  s->d_aabb = gst_upload(aabb);
  if (s->has_semantics) s->d_sem = gst_upload(sem);
  return true;
}

struct GstPipeline {
  OptixDeviceContext ctx = nullptr;
  OptixModule module = nullptr;
  OptixPipeline pipeline = nullptr;
  OptixProgramGroup groups[4] = {};
  OptixShaderBindingTable sbt = {};
  OptixTraversableHandle handle = 0;
  OptixAccelBuildOptions accel_opts = {};
  float gas_ms = 0.f;
  uint64_t gas_compacted = 0;
  uint64_t gas_uncompacted = 0;
  CUdeviceptr d_gas = 0;
};

// Builds the gaussian GAS plus module/pipeline/SBT for the dump's AABBs.
static bool gst_build_pipeline(const GstScene& s, const std::string& ptx_path,
                               GstPipeline* pl) {
  CUDA_CHECK(cudaFree(0));
  OPTIX_CHECK(optixInit());
  OptixDeviceContextOptions ctx_opts = {};
  ctx_opts.logCallbackFunction = gst_context_log;
  ctx_opts.logCallbackLevel = 3;
  OPTIX_CHECK(optixDeviceContextCreate(0, &ctx_opts, &pl->ctx));

  OptixBuildInput bi = {};
  bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  unsigned int geom_flags = OPTIX_GEOMETRY_FLAG_NONE;
  CUdeviceptr d_aabb = s.d_aabb;
  bi.customPrimitiveArray.aabbBuffers = &d_aabb;
  bi.customPrimitiveArray.numPrimitives = (unsigned int)s.n;
  bi.customPrimitiveArray.flags = &geom_flags;
  bi.customPrimitiveArray.numSbtRecords = 1;

  pl->accel_opts.buildFlags =
      OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  pl->accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes sizes = {};
  OPTIX_CHECK(
      optixAccelComputeMemoryUsage(pl->ctx, &pl->accel_opts, &bi, 1, &sizes));
  void *d_temp = nullptr, *d_out_gas = nullptr;
  CUDA_CHECK(cudaMalloc(&d_temp, sizes.tempSizeInBytes));
  CUDA_CHECK(cudaMalloc(&d_out_gas, sizes.outputSizeInBytes));
  CUdeviceptr d_compacted_size;
  CUDA_CHECK(cudaMalloc((void**)&d_compacted_size, 8));
  OptixAccelEmitDesc emit = {};
  emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit.result = d_compacted_size;

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0);
  OPTIX_CHECK(optixAccelBuild(pl->ctx, 0, &pl->accel_opts, &bi, 1,
                              (CUdeviceptr)d_temp, sizes.tempSizeInBytes,
                              (CUdeviceptr)d_out_gas, sizes.outputSizeInBytes,
                              &pl->handle, &emit, 1));
  cudaEventRecord(ev1);
  CUDA_CHECK(cudaEventSynchronize(ev1));
  cudaEventElapsedTime(&pl->gas_ms, ev0, ev1);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  CUDA_CHECK(cudaMemcpy(&pl->gas_compacted, (void*)d_compacted_size, 8,
                        cudaMemcpyDeviceToHost));
  pl->gas_uncompacted = sizes.outputSizeInBytes;
  if (pl->gas_compacted < sizes.outputSizeInBytes) {
    void* d_comp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_comp, pl->gas_compacted));
    OPTIX_CHECK(optixAccelCompact(pl->ctx, 0, pl->handle, (CUdeviceptr)d_comp,
                                  pl->gas_compacted, &pl->handle));
    CUDA_CHECK(cudaFree(d_out_gas));
    pl->d_gas = (CUdeviceptr)d_comp;
  } else {
    pl->d_gas = (CUdeviceptr)d_out_gas;
  }
  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree((void*)d_compacted_size));

  std::ifstream pf(ptx_path);
  std::string ptx((std::istreambuf_iterator<char>(pf)),
                  std::istreambuf_iterator<char>());
  if (ptx.empty()) {
    fprintf(stderr, "cannot read %s\n", ptx_path.c_str());
    return false;
  }

  OptixModuleCompileOptions mco = {};
  mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
  OptixPipelineCompileOptions pco = {};
  pco.usesMotionBlur = 0;
  pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pco.numPayloadValues = 3;
  pco.numAttributeValues = 2;  // triangles need 2 (barycentrics); custom uses 1
  pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pco.pipelineLaunchParamsVariableName = "params";
  pco.usesPrimitiveTypeFlags =
      OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  char log[4096];
  size_t log_size = sizeof(log);
  OPTIX_CHECK(optixModuleCreate(pl->ctx, &mco, &pco, ptx.c_str(), ptx.size(),
                                log, &log_size, &pl->module));

  OptixProgramGroupOptions pgo = {};
  OptixProgramGroupDesc pg_desc[4] = {};
  pg_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pg_desc[0].raygen.module = pl->module;
  pg_desc[0].raygen.entryFunctionName = "__raygen__gs";
  pg_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pg_desc[1].miss.module = pl->module;
  pg_desc[1].miss.entryFunctionName = "__miss__gs";
  pg_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pg_desc[2].hitgroup.moduleCH = pl->module;
  pg_desc[2].hitgroup.entryFunctionNameCH = "__closesthit__gs";
  pg_desc[2].hitgroup.moduleIS = pl->module;
  pg_desc[2].hitgroup.entryFunctionNameIS = "__intersection__gs";
  pg_desc[2].hitgroup.moduleAH = pl->module;
  pg_desc[2].hitgroup.entryFunctionNameAH = "__anyhit__gs";
  pg_desc[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pg_desc[3].hitgroup.moduleCH = pl->module;
  pg_desc[3].hitgroup.entryFunctionNameCH = "__closesthit__tri";
  size_t log_size2 = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(pl->ctx, pg_desc, 4, &pgo, log,
                                      &log_size2, pl->groups));

  OptixPipelineLinkOptions plo = {};
  plo.maxTraceDepth = 1;
  size_t log_size3 = sizeof(log);
  OPTIX_CHECK(optixPipelineCreate(pl->ctx, &pco, &plo, pl->groups, 4, log,
                                  &log_size3, &pl->pipeline));
  OptixStackSizes stack = {};
  for (auto& g : pl->groups)
    OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &stack, pl->pipeline));
  unsigned int dctrav, dcstate, cont;
  OPTIX_CHECK(
      optixUtilComputeStackSizes(&stack, 1, 0, 0, &dctrav, &dcstate, &cont));
  OPTIX_CHECK(optixPipelineSetStackSize(pl->pipeline, dctrav, dcstate, cont, 1));

  EmptyRecord rec[4];
  for (int i = 0; i < 4; ++i)
    OPTIX_CHECK(optixSbtRecordPackHeader(pl->groups[i], &rec[i]));
  CUdeviceptr d_rec_rg = gst_upload(std::vector<EmptyRecord>{rec[0]});
  CUdeviceptr d_rec_ms = gst_upload(std::vector<EmptyRecord>{rec[1]});
  // hitgroup 0 = gaussians, hitgroup 1 = robot triangles (trace SBT offset 1)
  CUdeviceptr d_rec_hg = gst_upload(std::vector<EmptyRecord>{rec[2], rec[3]});
  pl->sbt.raygenRecord = d_rec_rg;
  pl->sbt.missRecordBase = d_rec_ms;
  pl->sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
  pl->sbt.missRecordCount = 1;
  pl->sbt.hitgroupRecordBase = d_rec_hg;
  pl->sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
  pl->sbt.hitgroupRecordCount = 2;
  return true;
}

// Release every CUDA and OptiX allocation owned by the reusable library
// context. The CLI is process-scoped, but the torch wrapper can create and
// destroy multiple Home-sized contexts inside one long-lived Isaac process.
static void gst_destroy_scene(GstScene* s) {
  if (s->d_means) cudaFree((void*)s->d_means);
  if (s->d_ic) cudaFree((void*)s->d_ic);
  if (s->d_rgb) cudaFree((void*)s->d_rgb);
  if (s->d_opac) cudaFree((void*)s->d_opac);
  if (s->d_aabb) cudaFree((void*)s->d_aabb);
  if (s->d_sem) cudaFree((void*)s->d_sem);
  *s = GstScene{};
}

static void gst_release_static_build_inputs(GstScene* s) {
  // The compacted GAS owns the acceleration structure after a completed
  // build. Static scenes never refit it, so retaining N float6 AABBs wastes
  // roughly 516 MiB for the 21.5M-Gaussian Home scene.
  if (s->d_aabb) {
    CUDA_CHECK(cudaFree((void*)s->d_aabb));
    s->d_aabb = 0;
  }
}

static void gst_destroy_pipeline(GstPipeline* pl) {
  cudaDeviceSynchronize();
  if (pl->pipeline) optixPipelineDestroy(pl->pipeline);
  for (OptixProgramGroup& group : pl->groups)
    if (group) optixProgramGroupDestroy(group);
  if (pl->module) optixModuleDestroy(pl->module);
  if (pl->sbt.raygenRecord) cudaFree((void*)pl->sbt.raygenRecord);
  if (pl->sbt.missRecordBase) cudaFree((void*)pl->sbt.missRecordBase);
  if (pl->sbt.hitgroupRecordBase) cudaFree((void*)pl->sbt.hitgroupRecordBase);
  if (pl->d_gas) cudaFree((void*)pl->d_gas);
  // The CUDA allocations above back objects owned by this OptiX context.
  // Release them before destroying the context itself.
  if (pl->ctx) optixDeviceContextDestroy(pl->ctx);
  *pl = GstPipeline{};
}

// Populates the scene/threshold part of GsParams from a loaded scene.
static void gst_base_params(const GstScene& s, GsParams* p) {
  p->means = (const float3*)s.d_means;
  p->precision6 = (const float*)s.d_ic;
  p->precision_mode = s.has_stable_precision ? 1 : 0;
  p->rgb = (const float3*)s.d_rgb;
  p->opacity = (const float*)s.d_opac;
  p->semantics = s.has_semantics ? (const int*)s.d_sem : nullptr;
  p->t_min = 1.0e-4f;
  p->t_max = 6.0f * s.radius;
  // 2/255 measured 14-16% faster than 1/255 at equal-or-better full-contract
  // parity (see RESULTS.md); pass --alpha-min 0.003921569 for the raster's
  // exact 1/255 threshold.
  p->alpha_min = 2.0f / 255.0f;
  p->transmittance_min = 0.03f;
  p->semantic_min_alpha = 0.01f;
  p->robot_semantic_id = 1 << 20;
  p->max_iters = 256;
  p->use_kbuffer = 1;
}
