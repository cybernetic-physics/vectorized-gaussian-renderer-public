// Host harness for the batched OptiX gaussian tracer benchmark.
//
// Loads the flat binary scene dump (scripts/export_gaussians_bin.py), builds a
// GAS over per-gaussian AABBs (shared setup in gs_tracer_setup.h), and
// benchmarks two launch shapes:
//   dense : (width, height, envs)  — batched primary-ray grid, one launch
//   sparse: (nrays, 1, 1)          — arbitrary ray list (secondary-ray shape)
// Every launch renders the full output contract: RGBA, camera-z depth, and
// strongest-contributor semantic ids (when the dump carries semantics).

#include <cmath>
#include <cstring>

#include <optix_function_table_definition.h>

#include "gs_tracer_setup.h"

static std::string arg_str(int argc, char** argv, const char* key,
                           const char* dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return argv[i + 1];
  return dflt;
}
static long arg_int(int argc, char** argv, const char* key, long dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return atol(argv[i + 1]);
  return dflt;
}

int main(int argc, char** argv) {
  std::string dump = arg_str(argc, argv, "--dump", "gaussians.bin");
  std::string ptx_path = arg_str(argc, argv, "--ptx", "gs_tracer_device.ptx");
  std::string mode = arg_str(argc, argv, "--mode", "dense");
  // Optional camera file for ray-matched benchmarks: int64 envs,
  // float32 fx,fy,cx,cy, then per env float32 origin[3] + R_c2w[9] row-major.
  std::string cams_path = arg_str(argc, argv, "--cams", "");
  std::string out_json = arg_str(argc, argv, "--out-json", "");
  std::string out_ppm = arg_str(argc, argv, "--out-ppm", "");
  // Prefix for raw output-contract dumps: <prefix>.rgba.f32, <prefix>.depth.f32,
  // <prefix>.semantic.i32 (row-major, env-major).
  std::string out_raw = arg_str(argc, argv, "--out-raw", "");
  int envs = (int)arg_int(argc, argv, "--envs", 32);
  int width = (int)arg_int(argc, argv, "--width", 256);
  int height = (int)arg_int(argc, argv, "--height", 256);
  int frames = (int)arg_int(argc, argv, "--frames", 20);
  int nrays = (int)arg_int(argc, argv, "--nrays", 30000);
  int max_iters = (int)arg_int(argc, argv, "--max-iters", 256);
  int use_kbuffer = (int)arg_int(argc, argv, "--kbuffer", 1);
  // Defaults must be the exact float32 constants the library uses, so the
  // CLI and the zero-copy path stay bit-identical (an atof'd literal differs
  // from the compiled constant by 1 ulp — enough to flip borderline hits).
  float alpha_min = 2.0f / 255.0f;
  float tmin = 0.03f;
  float near_plane = 1.0e-4f;
  float far_plane = 0.0f;  // 0 keeps the scene-derived default.
  {
    std::string s = arg_str(argc, argv, "--alpha-min", "");
    if (!s.empty()) alpha_min = (float)atof(s.c_str());
    s = arg_str(argc, argv, "--tmin", "");
    if (!s.empty()) tmin = (float)atof(s.c_str());
    s = arg_str(argc, argv, "--near-plane", "");
    if (!s.empty()) near_plane = (float)atof(s.c_str());
    s = arg_str(argc, argv, "--far-plane", "");
    if (!s.empty()) far_plane = (float)atof(s.c_str());
  }
  int robot = (int)arg_int(argc, argv, "--robot", 0);
  int robot_tris_n = (int)arg_int(argc, argv, "--robot-tris", 200000);

  GstScene scene;
  if (!gst_load_dump(dump, &scene)) return 1;
  int64_t n = scene.n;
  const float* center = scene.center;
  float radius = scene.radius;
  printf("scene: n=%lld center=(%.2f,%.2f,%.2f) radius=%.2f semantics=%d\n",
         (long long)n, center[0], center[1], center[2], radius,
         scene.has_semantics ? 1 : 0);

  GstPipeline pl;
  if (!gst_build_pipeline(scene, ptx_path, &pl)) return 1;
  gst_release_static_build_inputs(&scene);
  printf("gas: build=%.1f ms uncompacted=%.2f GB compacted=%.2f GB\n",
         pl.gas_ms, pl.gas_uncompacted / 1e9, pl.gas_compacted / 1e9);

  // ---- robot triangle GAS (dense robot-sized ellipsoid, shared by envs) ----
  OptixTraversableHandle robot_handle = 0;
  CUdeviceptr d_rverts = 0, d_rtris = 0;
  unsigned int robot_ntris = 0;
  if (robot) {
    int rows = (int)fmax(8.0, sqrt(robot_tris_n / 2.0));
    int cols = 2 * rows;
    std::vector<float> rv;
    rv.reserve((size_t)(rows + 1) * cols * 3);
    for (int r = 0; r <= rows; ++r) {
      float lat = (float)M_PI * r / rows;
      for (int cix = 0; cix < cols; ++cix) {
        float lon = 2.0f * (float)M_PI * cix / cols;
        rv.push_back(0.10f * sinf(lat) * cosf(lon));
        rv.push_back(0.10f * sinf(lat) * sinf(lon));
        rv.push_back(0.35f * cosf(lat));
      }
    }
    std::vector<unsigned int> rt;
    for (int r = 0; r < rows; ++r)
      for (int cix = 0; cix < cols; ++cix) {
        unsigned int a = r * cols + cix;
        unsigned int b = r * cols + (cix + 1) % cols;
        unsigned int cc = (r + 1) * cols + cix;
        unsigned int dd = (r + 1) * cols + (cix + 1) % cols;
        rt.push_back(a); rt.push_back(b); rt.push_back(dd);
        rt.push_back(a); rt.push_back(dd); rt.push_back(cc);
      }
    robot_ntris = (unsigned int)(rt.size() / 3);
    d_rverts = gst_upload(rv);
    d_rtris = gst_upload(rt);

    OptixBuildInput tbi = {};
    tbi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    unsigned int tflags = OPTIX_GEOMETRY_FLAG_NONE;
    tbi.triangleArray.vertexBuffers = &d_rverts;
    tbi.triangleArray.numVertices = (unsigned int)(rv.size() / 3);
    tbi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    tbi.triangleArray.indexBuffer = d_rtris;
    tbi.triangleArray.numIndexTriplets = robot_ntris;
    tbi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    tbi.triangleArray.flags = &tflags;
    tbi.triangleArray.numSbtRecords = 1;
    OptixAccelBufferSizes tsz = {};
    OPTIX_CHECK(
        optixAccelComputeMemoryUsage(pl.ctx, &pl.accel_opts, &tbi, 1, &tsz));
    void *t_tmp = nullptr, *t_out = nullptr;
    CUDA_CHECK(cudaMalloc(&t_tmp, tsz.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(&t_out, tsz.outputSizeInBytes));
    OPTIX_CHECK(optixAccelBuild(pl.ctx, 0, &pl.accel_opts, &tbi, 1,
                                (CUdeviceptr)t_tmp, tsz.tempSizeInBytes,
                                (CUdeviceptr)t_out, tsz.outputSizeInBytes,
                                &robot_handle, nullptr, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(t_tmp));
    printf("robot: tris=%u gas=%.1f MB\n", robot_ntris,
           tsz.outputSizeInBytes / 1e6);
  }

  // ---- params -------------------------------------------------------------
  GsParams p = {};
  gst_base_params(scene, &p);
  p.handle = pl.handle;
  p.fx = 0.9f * width;
  p.fy = 0.9f * width * ((float)height / (float)width);
  p.cx = width * 0.5f;
  p.cy = height * 0.5f;
  p.width = width; p.height = height; p.envs = envs;
  p.alpha_min = alpha_min;
  p.transmittance_min = tmin;
  p.t_min = near_plane;
  if (far_plane > 0.f) p.t_max = far_plane;
  if (p.t_min < 0.f || p.t_max <= p.t_min) {
    fprintf(stderr, "invalid trace range near=%g far=%g\n", p.t_min, p.t_max);
    return 1;
  }
  p.max_iters = max_iters;
  p.use_kbuffer = use_kbuffer;

  size_t out_count;
  unsigned long long rays_per_launch;
  if (mode == "dense") {
    p.mode = 0;
    std::vector<float> orig(envs * 3);
    if (!cams_path.empty()) {
      std::ifstream cf(cams_path, std::ios::binary);
      if (!cf) { fprintf(stderr, "cannot open %s\n", cams_path.c_str()); return 1; }
      long long cam_envs = 0;
      float k[4];
      cf.read((char*)&cam_envs, 8);
      cf.read((char*)k, 16);
      envs = (int)cam_envs;
      p.envs = envs;
      p.fx = k[0]; p.fy = k[1]; p.cx = k[2]; p.cy = k[3];
      orig.assign((size_t)envs * 3, 0.f);
      std::vector<float> rots((size_t)envs * 9);
      for (int e = 0; e < envs; ++e) {
        cf.read((char*)&orig[(size_t)e * 3], 12);
        cf.read((char*)&rots[(size_t)e * 9], 36);
      }
      if (!cf) { fprintf(stderr, "short read on %s\n", cams_path.c_str()); return 1; }
      p.cam_rots9 = (const float*)gst_upload(rots);
      printf("cams: %d loaded from %s fx=%.2f fy=%.2f\n", envs,
             cams_path.c_str(), p.fx, p.fy);
    } else {
      float dist = 2.2f * radius;
      for (int e = 0; e < envs; ++e) {
        float frac = envs > 1 ? (float)e / (envs - 1) : 0.5f;
        orig[e * 3 + 0] = center[0] + (frac - 0.5f) * 0.2f * radius;
        orig[e * 3 + 1] = center[1];
        orig[e * 3 + 2] = center[2] - dist;
      }
    }
    p.cam_origins = (const float3*)gst_upload(orig);
    out_count = (size_t)envs * height * width;
    rays_per_launch = out_count;

    if (robot) {
      // Robot 2.5 m in front of each env camera, upright (z-up robot in a
      // y-down world), distinct heading per env. Rotation is world->robot.
      std::vector<float> rot9(envs * 9);
      std::vector<float> rpos(envs * 3);
      const float B[9] = {1, 0, 0,  0, 0, 1,  0, -1, 0};  // upright base
      for (int e = 0; e < envs; ++e) {
        float th = 2.0f * (float)M_PI * e / (envs > 0 ? envs : 1);
        float cz = cosf(th), sz = sinf(th);
        const float Rz[9] = {cz, -sz, 0, sz, cz, 0, 0, 0, 1};
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            float acc = 0.f;
            for (int k = 0; k < 3; ++k) acc += Rz[i * 3 + k] * B[k * 3 + j];
            rot9[e * 9 + i * 3 + j] = acc;
          }
        rpos[e * 3 + 0] = orig[e * 3 + 0];
        rpos[e * 3 + 1] = orig[e * 3 + 1] - 0.2f;
        rpos[e * 3 + 2] = orig[e * 3 + 2] + 2.5f;
      }
      p.robot_enabled = 1;
      p.robot_handle = robot_handle;
      p.robot_rot9 = (const float*)gst_upload(rot9);
      p.robot_pos = (const float3*)gst_upload(rpos);
      p.robot_verts = (const float3*)d_rverts;
      p.robot_tris = (const uint3*)d_rtris;
      p.fresnel_f0 = 0.9f;
      p.robot_base_r = 0.58f; p.robot_base_g = 0.60f; p.robot_base_b = 0.63f;
      void* d_refl = nullptr;
      CUDA_CHECK(cudaMalloc(&d_refl, 8));
      CUDA_CHECK(cudaMemset(d_refl, 0, 8));
      p.refl_count = (unsigned long long*)d_refl;
    }
  } else {
    p.mode = 1;
    p.nrays = nrays;
    std::vector<float> ro(nrays * 3), rd(nrays * 3);
    srand(7);
    for (int i = 0; i < nrays; ++i) {
      for (int k = 0; k < 3; ++k) {
        ro[i * 3 + k] = center[k] + ((rand() / (float)RAND_MAX) - 0.5f) * 0.6f * radius;
        rd[i * 3 + k] = (rand() / (float)RAND_MAX) - 0.5f;
      }
    }
    p.ray_origins = (const float3*)gst_upload(ro);
    p.ray_dirs = (const float3*)gst_upload(rd);
    out_count = nrays;
    rays_per_launch = nrays;
  }
  void* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, out_count * sizeof(float4)));
  p.out = (float4*)d_out;
  void* d_depth = nullptr;
  CUDA_CHECK(cudaMalloc(&d_depth, out_count * sizeof(float)));
  p.out_depth = (float*)d_depth;
  void* d_semantic = nullptr;
  CUDA_CHECK(cudaMalloc(&d_semantic, out_count * sizeof(int)));
  p.out_semantic = (int*)d_semantic;
  void* d_iters = nullptr;
  CUDA_CHECK(cudaMalloc(&d_iters, 8));
  CUDA_CHECK(cudaMemset(d_iters, 0, 8));
  p.iter_count = (unsigned long long*)d_iters;
  void* d_params = nullptr;
  CUDA_CHECK(cudaMalloc(&d_params, sizeof(GsParams)));
  CUDA_CHECK(cudaMemcpy(d_params, &p, sizeof(GsParams), cudaMemcpyHostToDevice));

  auto launch = [&]() {
    if (p.mode == 0)
      OPTIX_CHECK(optixLaunch(pl.pipeline, 0, (CUdeviceptr)d_params,
                              sizeof(GsParams), &pl.sbt, width, height, envs));
    else
      OPTIX_CHECK(optixLaunch(pl.pipeline, 0, (CUdeviceptr)d_params,
                              sizeof(GsParams), &pl.sbt, nrays, 1, 1));
  };

  for (int i = 0; i < 3; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemset(d_iters, 0, 8));
  if (p.refl_count) CUDA_CHECK(cudaMemset((void*)p.refl_count, 0, 8));

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0);
  for (int i = 0; i < frames; ++i) launch();
  cudaEventRecord(ev1);
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, ev0, ev1);
  float ms = total_ms / frames;

  unsigned long long iters = 0;
  CUDA_CHECK(cudaMemcpy(&iters, d_iters, 8, cudaMemcpyDeviceToHost));
  double avg_iters = (double)iters / ((double)rays_per_launch * frames);
  unsigned long long refl = 0;
  if (p.refl_count)
    CUDA_CHECK(cudaMemcpy(&refl, (void*)p.refl_count, 8, cudaMemcpyDeviceToHost));
  double mrays = rays_per_launch / (ms * 1e3);  // rays per launch / ms -> Mrays/s

  // Hit stats + optional PPM of env 0 + optional raw output-contract dumps.
  std::vector<float> out_host(out_count * 4);
  CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, out_count * 16,
                        cudaMemcpyDeviceToHost));
  size_t hits = 0;
  for (size_t i = 0; i < out_count; ++i)
    if (out_host[i * 4 + 3] > 0.05f) ++hits;
  if (!out_ppm.empty() && mode == "dense") {
    FILE* f = fopen(out_ppm.c_str(), "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const float* px = &out_host[((size_t)y * width + x) * 4];
        for (int k = 0; k < 3; ++k) {
          float v = px[k] < 0 ? 0 : (px[k] > 1 ? 1 : px[k]);
          fputc((int)(v * 255.f + 0.5f), f);
        }
      }
    fclose(f);
  }
  if (!out_raw.empty()) {
    std::vector<float> depth_host(out_count);
    std::vector<int> sem_host(out_count);
    CUDA_CHECK(cudaMemcpy(depth_host.data(), d_depth, out_count * 4,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sem_host.data(), d_semantic, out_count * 4,
                          cudaMemcpyDeviceToHost));
    auto dump_raw = [&](const std::string& suffix, const void* data,
                        size_t bytes) {
      std::string path = out_raw + suffix;
      FILE* f = fopen(path.c_str(), "wb");
      fwrite(data, 1, bytes, f);
      fclose(f);
    };
    dump_raw(".rgba.f32", out_host.data(), out_count * 16);
    dump_raw(".depth.f32", depth_host.data(), out_count * 4);
    dump_raw(".semantic.i32", sem_host.data(), out_count * 4);
  }

  printf("OPTIX_GS_TRACER_OK mode=%s kbuf=%d robot=%d envs=%d res=%dx%d rays=%llu "
         "ms=%.2f Mrays_s=%.1f avg_traces_per_ray=%.1f hit_frac=%.3f "
         "refl_px_per_frame=%.0f gas_ms=%.1f semantics=%d\n",
         mode.c_str(), use_kbuffer, robot, envs, width, height,
         (unsigned long long)rays_per_launch, ms, mrays, avg_iters,
         out_count ? (double)hits / out_count : 0.0,
         refl / (double)(frames > 0 ? frames : 1), pl.gas_ms,
         scene.has_semantics ? 1 : 0);

  if (!out_json.empty()) {
    FILE* f = fopen(out_json.c_str(), "w");
    fprintf(f,
            "{\"schema_version\":\"optix-gs-tracer/v2\",\"mode\":\"%s\","
            "\"envs\":%d,\"width\":%d,\"height\":%d,\"rays_per_launch\":%llu,"
            "\"mean_launch_ms\":%.4f,\"mrays_per_second\":%.2f,"
            "\"avg_traces_per_ray\":%.2f,\"hit_fraction\":%.4f,"
            "\"gas_build_ms\":%.1f,\"gas_compacted_gb\":%.3f,"
            "\"gaussians\":%lld,\"has_semantics\":%s,"
            "\"outputs\":[\"rgba\",\"depth\",\"semantic\"]}\n",
            mode.c_str(), envs, width, height,
            (unsigned long long)rays_per_launch, ms, mrays, avg_iters,
            out_count ? (double)hits / out_count : 0.0, pl.gas_ms,
            pl.gas_compacted / 1e9, (long long)n,
            scene.has_semantics ? "true" : "false");
    fclose(f);
  }
  return 0;
}
