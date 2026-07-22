#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

namespace {

constexpr int kProjectionThreads = 256;
constexpr int kRasterThreads = 256;
constexpr int kDepthGroupBlocks = 1024;
constexpr int kTileGroupingBlocks = 1024;
constexpr int kMaxTileSize = 16;
constexpr int kExactRayBinTileScale = 2;
constexpr float kTransmittanceThreshold = 1.0e-4f;
constexpr float kAlphaThreshold = 1.0f / 255.0f;
constexpr float kMaxAlpha = 0.99f;
constexpr float kMinAntialiasCompensation = 0.005f;

int emitted_sort_key_end_bit(int64_t global_tile_capacity) {
  TORCH_CHECK(
      global_tile_capacity > 0,
      "global tile capacity must be positive");
  TORCH_CHECK(
      global_tile_capacity <= INT_MAX,
      "global tile capacity must fit int32");
  uint64_t maximum_tile =
      static_cast<uint64_t>(global_tile_capacity - 1);
  int tile_bits = 0;
  while (maximum_tile != 0) {
    ++tile_bits;
    maximum_tile >>= 1;
  }
  // Emitted keys are (uint32 global_tile << 32) | positive_float_depth_bits.
  // The complete low word is significant; bits above the largest possible
  // allocated tile index are provably zero.
  return 32 + tile_bits;
}

enum SceneTensor : int {
  kMeans = 0,
  kCovariances = 1,
  kOpacities = 2,
  kFeatures = 3,
  kSemanticIds = 4,
  kRegisteredSceneIds = 5,
  kSceneOffsets = 6,
};

enum CameraTensor : int {
  kViewmats = 0,
  kIntrinsics = 1,
  kEnvXforms = 2,
  kCameraSceneIds = 3,
  kActiveCameraIds = 4,
};

enum OutputTensor : int {
  kRgb = 0,
  kDepth = 1,
  kAlpha = 2,
  kSemantic = 3,
};

enum WorkspaceTensor : int {
  kVisibleMeans2d = 0,
  kVisibleConics = 1,
  kVisibleDepths = 2,
  kVisibleOpacities = 3,
  kVisibleCameraIds = 4,
  kVisibleGaussianIds = 5,
  kVisibleRadii = 6,
  kKeysIn = 7,
  kKeysOut = 8,
  kValuesIn = 9,
  kValuesOut = 10,
  kTileStarts = 11,
  kTileEnds = 12,
  kCounters = 13,
  kSortTemp = 14,
  kDepthBucketTau = 15,
  kDepthCutoff = 16,
  kDepthBucketCounts = 17,
  kDepthBucketOffsets = 18,
  kDepthBucketWriteOffsets = 19,
  kDepthOrderedVisibleIndices = 20,
  kDepthAccumulatedTau = 21,
  kVisibleRayPrecisions = 22,
  kVisibleRayPrecisionMeans = 23,
};

__device__ __forceinline__ int find_scene_slot(
    int64_t requested_scene_id,
    const int64_t* registered_scene_ids,
    int scene_count) {
  for (int slot = 0; slot < scene_count; ++slot) {
    if (registered_scene_ids[slot] == requested_scene_id) {
      return slot;
    }
  }
  return -1;
}

__device__ __forceinline__ void quaternion_to_rotation(
    float w,
    float x,
    float y,
    float z,
    float* rotation) {
  const float norm = rsqrtf(fmaxf(w * w + x * x + y * y + z * z, 1.0e-20f));
  w *= norm;
  x *= norm;
  y *= norm;
  z *= norm;

  const float xx = x * x;
  const float yy = y * y;
  const float zz = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float wx = w * x;
  const float wy = w * y;
  const float wz = w * z;

  rotation[0] = 1.0f - 2.0f * (yy + zz);
  rotation[1] = 2.0f * (xy - wz);
  rotation[2] = 2.0f * (xz + wy);
  rotation[3] = 2.0f * (xy + wz);
  rotation[4] = 1.0f - 2.0f * (xx + zz);
  rotation[5] = 2.0f * (yz - wx);
  rotation[6] = 2.0f * (xz - wy);
  rotation[7] = 2.0f * (yz + wx);
  rotation[8] = 1.0f - 2.0f * (xx + yy);
}

__device__ __forceinline__ void multiply_rotation_3x3(
    const float* left,
    const float* right,
    float* output) {
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      output[row * 3 + column] =
          left[row * 3 + 0] * right[0 * 3 + column] +
          left[row * 3 + 1] * right[1 * 3 + column] +
          left[row * 3 + 2] * right[2 * 3 + column];
    }
  }
}

__device__ __forceinline__ float linear_to_srgb(float value) {
  value = fminf(fmaxf(value, 0.0f), 1.0f);
  return value <= 0.0031308f
      ? 12.92f * value
      : 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
}

__device__ __forceinline__ int depth_bucket_index(
    float depth,
    float near_plane,
    float far_plane,
    int bucket_count) {
  const float normalized = (depth - near_plane) / (far_plane - near_plane);
  return min(
      bucket_count - 1,
      max(0, static_cast<int>(normalized * bucket_count)));
}

__device__ __forceinline__ float splat_alpha_at_pixel(
    float mean_x,
    float mean_y,
    float conic_0,
    float conic_1,
    float conic_2,
    float opacity,
    int pixel_x,
    int pixel_y) {
  const float delta_x =
      (static_cast<float>(pixel_x) + 0.5f) - mean_x;
  const float delta_y =
      (static_cast<float>(pixel_y) + 0.5f) - mean_y;
  const float power = -0.5f * (
      conic_0 * delta_x * delta_x +
      2.0f * conic_1 * delta_x * delta_y +
      conic_2 * delta_y * delta_y);
  if (power > 0.0f || power < -20.0f) {
    return 0.0f;
  }
  const float alpha = fminf(kMaxAlpha, opacity * __expf(power));
  return alpha >= kAlphaThreshold ? alpha : 0.0f;
}

__device__ __forceinline__ float splat_alpha_at_ray(
    float ray_x,
    float ray_y,
    float precision_00,
    float precision_01,
    float precision_02,
    float precision_11,
    float precision_12,
    float precision_22,
    float precision_mean_0,
    float precision_mean_1,
    float precision_mean_2,
    float mean_precision_mean,
    float opacity,
    float gaussian_support_sigma) {
  const float ray_precision_ray =
      precision_00 * ray_x * ray_x +
      2.0f * precision_01 * ray_x * ray_y +
      2.0f * precision_02 * ray_x +
      precision_11 * ray_y * ray_y +
      2.0f * precision_12 * ray_y +
      precision_22;
  if (!(ray_precision_ray > 0.0f)) {
    return 0.0f;
  }
  const float ray_precision_mean =
      ray_x * precision_mean_0 +
      ray_y * precision_mean_1 +
      precision_mean_2;
  const float mahalanobis_squared = fmaxf(
      0.0f,
      mean_precision_mean -
          ray_precision_mean * ray_precision_mean /
              ray_precision_ray);
  if (
      mahalanobis_squared >
      gaussian_support_sigma * gaussian_support_sigma) {
    return 0.0f;
  }
  return fminf(
      1.0f,
      opacity * __expf(-0.5f * mahalanobis_squared));
}

__device__ __forceinline__ float conservative_optical_thickness(
    float alpha) {
  float term = alpha;
  float result = term;
#pragma unroll
  for (int order = 2; order <= 8; ++order) {
    term *= alpha;
    result += term / static_cast<float>(order);
  }
  return result * 0.999f;
}

__device__ __forceinline__ bool apply_antialias_compensation(
    float raw_covariance_2d_00,
    float covariance_2d_01,
    float raw_covariance_2d_11,
    float blurred_determinant,
    float* opacity) {
  const float original_determinant =
      raw_covariance_2d_00 * raw_covariance_2d_11 -
      covariance_2d_01 * covariance_2d_01;
  const float compensation = sqrtf(fmaxf(
      kMinAntialiasCompensation * kMinAntialiasCompensation,
      original_determinant / blurred_determinant));
  *opacity *= compensation;
  return *opacity >= kAlphaThreshold;
}

struct ScreenProjection {
  float mean_x;
  float mean_y;
  float conic_0;
  float conic_1;
  float conic_2;
  float depth;
  float opacity;
  int radius_x;
  int radius_y;
};

__device__ __forceinline__ bool project_screen_gaussian(
    int32_t gaussian,
    int camera,
    const float* __restrict__ means,
    const float* __restrict__ covariances,
    const float* __restrict__ opacities,
    const float* __restrict__ viewmats,
    const float* __restrict__ intrinsics,
    const float* __restrict__ env_xforms,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float gaussian_support_sigma,
    float covariance_epsilon,
    bool antialiased,
    ScreenProjection* projection) {
  float opacity = opacities[gaussian];
  if (!(opacity >= kAlphaThreshold)) {
    return false;
  }

  const float* mean = means + static_cast<int64_t>(gaussian) * 3;
  const float* env = env_xforms + static_cast<int64_t>(camera) * 16;
  const float* view = viewmats + static_cast<int64_t>(camera) * 16;
  const float env_x =
      env[0] * mean[0] + env[1] * mean[1] + env[2] * mean[2] + env[3];
  const float env_y =
      env[4] * mean[0] + env[5] * mean[1] + env[6] * mean[2] + env[7];
  const float env_z =
      env[8] * mean[0] + env[9] * mean[1] + env[10] * mean[2] + env[11];
  const float camera_x =
      view[0] * env_x + view[1] * env_y + view[2] * env_z + view[3];
  const float camera_y =
      view[4] * env_x + view[5] * env_y + view[6] * env_z + view[7];
  const float camera_z =
      view[8] * env_x + view[9] * env_y + view[10] * env_z + view[11];
  if (camera_z < near_plane || camera_z > far_plane) {
    return false;
  }

  const float* intrinsic =
      intrinsics + static_cast<int64_t>(camera) * 9;
  const float focal_x = intrinsic[0];
  const float focal_y = intrinsic[4];
  const float center_x = intrinsic[2];
  const float center_y = intrinsic[5];
  const float inverse_z = 1.0f / camera_z;
  const float mean_x = focal_x * camera_x * inverse_z + center_x;
  const float mean_y = focal_y * camera_y * inverse_z + center_y;

  float env_rotation[9] = {
      env[0], env[1], env[2],
      env[4], env[5], env[6],
      env[8], env[9], env[10],
  };
  float view_rotation[9] = {
      view[0], view[1], view[2],
      view[4], view[5], view[6],
      view[8], view[9], view[10],
  };
  float view_env_rotation[9];
  multiply_rotation_3x3(view_rotation, env_rotation, view_env_rotation);
  const float* object_covariance =
      covariances + static_cast<int64_t>(gaussian) * 6;
  const float object_covariance_00 = object_covariance[0];
  const float object_covariance_01 = object_covariance[1];
  const float object_covariance_02 = object_covariance[2];
  const float object_covariance_11 = object_covariance[3];
  const float object_covariance_12 = object_covariance[4];
  const float object_covariance_22 = object_covariance[5];
  float rotated_covariance[9];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    const float rotation_0 = view_env_rotation[row * 3 + 0];
    const float rotation_1 = view_env_rotation[row * 3 + 1];
    const float rotation_2 = view_env_rotation[row * 3 + 2];
    rotated_covariance[row * 3 + 0] =
        rotation_0 * object_covariance_00 +
        rotation_1 * object_covariance_01 +
        rotation_2 * object_covariance_02;
    rotated_covariance[row * 3 + 1] =
        rotation_0 * object_covariance_01 +
        rotation_1 * object_covariance_11 +
        rotation_2 * object_covariance_12;
    rotated_covariance[row * 3 + 2] =
        rotation_0 * object_covariance_02 +
        rotation_1 * object_covariance_12 +
        rotation_2 * object_covariance_22;
  }
  const float covariance_00 =
      rotated_covariance[0] * view_env_rotation[0] +
      rotated_covariance[1] * view_env_rotation[1] +
      rotated_covariance[2] * view_env_rotation[2];
  const float covariance_01 =
      rotated_covariance[0] * view_env_rotation[3] +
      rotated_covariance[1] * view_env_rotation[4] +
      rotated_covariance[2] * view_env_rotation[5];
  const float covariance_02 =
      rotated_covariance[0] * view_env_rotation[6] +
      rotated_covariance[1] * view_env_rotation[7] +
      rotated_covariance[2] * view_env_rotation[8];
  const float covariance_11 =
      rotated_covariance[3] * view_env_rotation[3] +
      rotated_covariance[4] * view_env_rotation[4] +
      rotated_covariance[5] * view_env_rotation[5];
  const float covariance_12 =
      rotated_covariance[3] * view_env_rotation[6] +
      rotated_covariance[4] * view_env_rotation[7] +
      rotated_covariance[5] * view_env_rotation[8];
  const float covariance_22 =
      rotated_covariance[6] * view_env_rotation[6] +
      rotated_covariance[7] * view_env_rotation[7] +
      rotated_covariance[8] * view_env_rotation[8];

  const float tan_fov_x = 0.5f * width / focal_x;
  const float tan_fov_y = 0.5f * height / focal_y;
  const float limit_x_positive =
      (width - center_x) / focal_x + 0.3f * tan_fov_x;
  const float limit_x_negative =
      center_x / focal_x + 0.3f * tan_fov_x;
  const float limit_y_positive =
      (height - center_y) / focal_y + 0.3f * tan_fov_y;
  const float limit_y_negative =
      center_y / focal_y + 0.3f * tan_fov_y;
  const float clamped_x = camera_z * fminf(
      limit_x_positive,
      fmaxf(-limit_x_negative, camera_x * inverse_z));
  const float clamped_y = camera_z * fminf(
      limit_y_positive,
      fmaxf(-limit_y_negative, camera_y * inverse_z));
  const float inverse_z_squared = inverse_z * inverse_z;
  const float jacobian_00 = focal_x * inverse_z;
  const float jacobian_02 = -focal_x * clamped_x * inverse_z_squared;
  const float jacobian_11 = focal_y * inverse_z;
  const float jacobian_12 = -focal_y * clamped_y * inverse_z_squared;
  const float raw_covariance_2d_00 =
      jacobian_00 * jacobian_00 * covariance_00 +
      2.0f * jacobian_00 * jacobian_02 * covariance_02 +
      jacobian_02 * jacobian_02 * covariance_22;
  const float covariance_2d_01 =
      jacobian_00 * jacobian_11 * covariance_01 +
      jacobian_00 * jacobian_12 * covariance_02 +
      jacobian_02 * jacobian_11 * covariance_12 +
      jacobian_02 * jacobian_12 * covariance_22;
  const float raw_covariance_2d_11 =
      jacobian_11 * jacobian_11 * covariance_11 +
      2.0f * jacobian_11 * jacobian_12 * covariance_12 +
      jacobian_12 * jacobian_12 * covariance_22;
  const float covariance_2d_00 =
      raw_covariance_2d_00 + covariance_epsilon;
  const float covariance_2d_11 =
      raw_covariance_2d_11 + covariance_epsilon;
  const float determinant =
      covariance_2d_00 * covariance_2d_11 -
      covariance_2d_01 * covariance_2d_01;
  if (!(determinant > 0.0f)) {
    return false;
  }

  if (antialiased) {
    // Mip-Splatting opacity compensation conserves projected energy after
    // covariance_epsilon dilates the screen-space footprint. This is an
    // independent implementation of Eq. 4 from arXiv:2311.16493.
    if (!apply_antialias_compensation(
            raw_covariance_2d_00,
            covariance_2d_01,
            raw_covariance_2d_11,
            determinant,
            &opacity)) {
      return false;
    }
  }

  const float opacity_extend = sqrtf(
      2.0f * __logf(opacity / kAlphaThreshold));
  const float extend = fminf(gaussian_support_sigma, opacity_extend);
  const int radius_x = static_cast<int>(ceilf(
      extend * sqrtf(covariance_2d_00)));
  const int radius_y = static_cast<int>(ceilf(
      extend * sqrtf(covariance_2d_11)));
  if (radius_x <= 0 && radius_y <= 0) {
    return false;
  }
  if (
      mean_x + radius_x <= 0.0f ||
      mean_y + radius_y <= 0.0f ||
      mean_x - radius_x >= width ||
      mean_y - radius_y >= height) {
    return false;
  }

  const float inverse_determinant = 1.0f / determinant;
  projection->mean_x = mean_x;
  projection->mean_y = mean_y;
  projection->conic_0 = covariance_2d_11 * inverse_determinant;
  projection->conic_1 = -covariance_2d_01 * inverse_determinant;
  projection->conic_2 = covariance_2d_00 * inverse_determinant;
  projection->depth = camera_z;
  projection->opacity = fminf(fmaxf(opacity, 0.0f), 1.0f);
  projection->radius_x = radius_x;
  projection->radius_y = radius_y;
  return true;
}

__global__ void precompute_covariances_kernel(
    const float* __restrict__ scales,
    const float* __restrict__ rotations,
    int64_t gaussian_count,
    float* __restrict__ covariances) {
  const int64_t gaussian =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gaussian >= gaussian_count) {
    return;
  }

  float rotation[9];
  const float* quaternion = rotations + gaussian * 4;
  quaternion_to_rotation(
      quaternion[0],
      quaternion[1],
      quaternion[2],
      quaternion[3],
      rotation);
  const float* scale = scales + gaussian * 3;
  const float scale_x_squared = scale[0] * scale[0];
  const float scale_y_squared = scale[1] * scale[1];
  const float scale_z_squared = scale[2] * scale[2];
  float* covariance = covariances + gaussian * 6;
  covariance[0] =
      rotation[0] * rotation[0] * scale_x_squared +
      rotation[1] * rotation[1] * scale_y_squared +
      rotation[2] * rotation[2] * scale_z_squared;
  covariance[1] =
      rotation[0] * rotation[3] * scale_x_squared +
      rotation[1] * rotation[4] * scale_y_squared +
      rotation[2] * rotation[5] * scale_z_squared;
  covariance[2] =
      rotation[0] * rotation[6] * scale_x_squared +
      rotation[1] * rotation[7] * scale_y_squared +
      rotation[2] * rotation[8] * scale_z_squared;
  covariance[3] =
      rotation[3] * rotation[3] * scale_x_squared +
      rotation[4] * rotation[4] * scale_y_squared +
      rotation[5] * rotation[5] * scale_z_squared;
  covariance[4] =
      rotation[3] * rotation[6] * scale_x_squared +
      rotation[4] * rotation[7] * scale_y_squared +
      rotation[5] * rotation[8] * scale_z_squared;
  covariance[5] =
      rotation[6] * rotation[6] * scale_x_squared +
      rotation[7] * rotation[7] * scale_y_squared +
      rotation[8] * rotation[8] * scale_z_squared;
}

__global__ void project_visible_kernel(
    const float* __restrict__ means,
    const float* __restrict__ covariances,
    const float* __restrict__ opacities,
    const int64_t* __restrict__ registered_scene_ids,
    const int64_t* __restrict__ scene_offsets,
    int scene_count,
    const float* __restrict__ viewmats,
    const float* __restrict__ intrinsics,
    const float* __restrict__ env_xforms,
    const int64_t* __restrict__ camera_scene_ids,
    const int64_t* __restrict__ active_camera_ids,
    int active_count,
    int batch_size,
    int64_t max_scene_gaussians,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float gaussian_support_sigma,
    float covariance_epsilon,
    bool antialiased,
    bool ray_gaussian_evaluation,
    float* __restrict__ visible_means2d,
    float* __restrict__ visible_conics,
    float* __restrict__ visible_depths,
    float* __restrict__ visible_opacities,
    int32_t* __restrict__ visible_camera_ids,
    int32_t* __restrict__ visible_gaussian_ids,
    int32_t* __restrict__ visible_radii,
    float* __restrict__ visible_ray_precisions,
    float* __restrict__ visible_ray_precision_means,
    int64_t visible_capacity,
    int depth_bucket_count,
    int64_t* __restrict__ depth_bucket_counts,
    int compact_projection_pass,
    float* __restrict__ compact_depth_bucket_tau,
    const int32_t* __restrict__ compact_depth_cutoff,
    int64_t* __restrict__ counters) {
  const int64_t task_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t task_count =
      static_cast<int64_t>(active_count) * max_scene_gaussians;
  if (task_index >= task_count) {
    return;
  }

  const int active_index = static_cast<int>(task_index / max_scene_gaussians);
  const int64_t local_gaussian = task_index - (
      static_cast<int64_t>(active_index) * max_scene_gaussians);
  const int64_t camera64 = active_camera_ids == nullptr
      ? static_cast<int64_t>(active_index)
      : active_camera_ids[active_index];
  if (camera64 < 0 || camera64 >= batch_size) {
    return;
  }
  const int camera = static_cast<int>(camera64);
  const int scene_slot = find_scene_slot(
      camera_scene_ids[camera],
      registered_scene_ids,
      scene_count);
  if (scene_slot < 0) {
    return;
  }

  const int64_t scene_start = scene_offsets[scene_slot];
  const int64_t scene_end = scene_offsets[scene_slot + 1];
  if (local_gaussian >= scene_end - scene_start) {
    return;
  }
  const int64_t gaussian = scene_start + local_gaussian;
  if (gaussian > INT_MAX) {
    return;
  }

  float opacity = opacities[gaussian];
  if (!(opacity >= kAlphaThreshold)) {
    return;
  }

  const float* mean = means + gaussian * 3;
  const float* env = env_xforms + static_cast<int64_t>(camera) * 16;
  const float* view = viewmats + static_cast<int64_t>(camera) * 16;

  const float env_x =
      env[0] * mean[0] + env[1] * mean[1] + env[2] * mean[2] + env[3];
  const float env_y =
      env[4] * mean[0] + env[5] * mean[1] + env[6] * mean[2] + env[7];
  const float env_z =
      env[8] * mean[0] + env[9] * mean[1] + env[10] * mean[2] + env[11];

  const float camera_x =
      view[0] * env_x + view[1] * env_y + view[2] * env_z + view[3];
  const float camera_y =
      view[4] * env_x + view[5] * env_y + view[6] * env_z + view[7];
  const float camera_z =
      view[8] * env_x + view[9] * env_y + view[10] * env_z + view[11];
  if (camera_z < near_plane || camera_z > far_plane) {
    return;
  }

  const float* intrinsic = intrinsics + static_cast<int64_t>(camera) * 9;
  const float focal_x = intrinsic[0];
  const float focal_y = intrinsic[4];
  const float center_x = intrinsic[2];
  const float center_y = intrinsic[5];
  const float inverse_z = 1.0f / camera_z;
  const float mean_x = focal_x * camera_x * inverse_z + center_x;
  const float mean_y = focal_y * camera_y * inverse_z + center_y;

  float env_rotation[9] = {
      env[0], env[1], env[2],
      env[4], env[5], env[6],
      env[8], env[9], env[10],
  };
  float view_rotation[9] = {
      view[0], view[1], view[2],
      view[4], view[5], view[6],
      view[8], view[9], view[10],
  };
  float view_env_rotation[9];
  multiply_rotation_3x3(view_rotation, env_rotation, view_env_rotation);
  const float* object_covariance = covariances + gaussian * 6;
  const float object_covariance_00 = object_covariance[0];
  const float object_covariance_01 = object_covariance[1];
  const float object_covariance_02 = object_covariance[2];
  const float object_covariance_11 = object_covariance[3];
  const float object_covariance_12 = object_covariance[4];
  const float object_covariance_22 = object_covariance[5];
  float rotated_covariance[9];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    const float rotation_0 = view_env_rotation[row * 3 + 0];
    const float rotation_1 = view_env_rotation[row * 3 + 1];
    const float rotation_2 = view_env_rotation[row * 3 + 2];
    rotated_covariance[row * 3 + 0] =
        rotation_0 * object_covariance_00 +
        rotation_1 * object_covariance_01 +
        rotation_2 * object_covariance_02;
    rotated_covariance[row * 3 + 1] =
        rotation_0 * object_covariance_01 +
        rotation_1 * object_covariance_11 +
        rotation_2 * object_covariance_12;
    rotated_covariance[row * 3 + 2] =
        rotation_0 * object_covariance_02 +
        rotation_1 * object_covariance_12 +
        rotation_2 * object_covariance_22;
  }
  const float covariance_00 =
      rotated_covariance[0] * view_env_rotation[0] +
      rotated_covariance[1] * view_env_rotation[1] +
      rotated_covariance[2] * view_env_rotation[2];
  const float covariance_01 =
      rotated_covariance[0] * view_env_rotation[3] +
      rotated_covariance[1] * view_env_rotation[4] +
      rotated_covariance[2] * view_env_rotation[5];
  const float covariance_02 =
      rotated_covariance[0] * view_env_rotation[6] +
      rotated_covariance[1] * view_env_rotation[7] +
      rotated_covariance[2] * view_env_rotation[8];
  const float covariance_11 =
      rotated_covariance[3] * view_env_rotation[3] +
      rotated_covariance[4] * view_env_rotation[4] +
      rotated_covariance[5] * view_env_rotation[5];
  const float covariance_12 =
      rotated_covariance[3] * view_env_rotation[6] +
      rotated_covariance[4] * view_env_rotation[7] +
      rotated_covariance[5] * view_env_rotation[8];
  const float covariance_22 =
      rotated_covariance[6] * view_env_rotation[6] +
      rotated_covariance[7] * view_env_rotation[7] +
      rotated_covariance[8] * view_env_rotation[8];

  const float tan_fov_x = 0.5f * width / focal_x;
  const float tan_fov_y = 0.5f * height / focal_y;
  const float limit_x_positive =
      (width - center_x) / focal_x + 0.3f * tan_fov_x;
  const float limit_x_negative =
      center_x / focal_x + 0.3f * tan_fov_x;
  const float limit_y_positive =
      (height - center_y) / focal_y + 0.3f * tan_fov_y;
  const float limit_y_negative =
      center_y / focal_y + 0.3f * tan_fov_y;
  const float clamped_x = camera_z * fminf(
      limit_x_positive,
      fmaxf(-limit_x_negative, camera_x * inverse_z));
  const float clamped_y = camera_z * fminf(
      limit_y_positive,
      fmaxf(-limit_y_negative, camera_y * inverse_z));
  const float inverse_z_squared = inverse_z * inverse_z;
  const float jacobian_00 = focal_x * inverse_z;
  const float jacobian_02 = -focal_x * clamped_x * inverse_z_squared;
  const float jacobian_11 = focal_y * inverse_z;
  const float jacobian_12 = -focal_y * clamped_y * inverse_z_squared;

  const float raw_covariance_2d_00 =
      jacobian_00 * jacobian_00 * covariance_00 +
      2.0f * jacobian_00 * jacobian_02 * covariance_02 +
      jacobian_02 * jacobian_02 * covariance_22;
  const float covariance_2d_01 =
      jacobian_00 * jacobian_11 * covariance_01 +
      jacobian_00 * jacobian_12 * covariance_02 +
      jacobian_02 * jacobian_11 * covariance_12 +
      jacobian_02 * jacobian_12 * covariance_22;
  const float raw_covariance_2d_11 =
      jacobian_11 * jacobian_11 * covariance_11 +
      2.0f * jacobian_11 * jacobian_12 * covariance_12 +
      jacobian_12 * jacobian_12 * covariance_22;
  const float covariance_2d_00 =
      raw_covariance_2d_00 + covariance_epsilon;
  const float covariance_2d_11 =
      raw_covariance_2d_11 + covariance_epsilon;
  const float determinant =
      covariance_2d_00 * covariance_2d_11 -
      covariance_2d_01 * covariance_2d_01;
  if (!(determinant > 0.0f)) {
    return;
  }

  if (antialiased) {
    if (!apply_antialias_compensation(
            raw_covariance_2d_00,
            covariance_2d_01,
            raw_covariance_2d_11,
            determinant,
            &opacity)) {
      return;
    }
  }

  const float opacity_extend = sqrtf(
      2.0f * __logf(opacity / kAlphaThreshold));
  const float extend = fminf(gaussian_support_sigma, opacity_extend);
  int radius_x = static_cast<int>(ceilf(
      extend * sqrtf(covariance_2d_00)));
  int radius_y = static_cast<int>(ceilf(
      extend * sqrtf(covariance_2d_11)));
  if (radius_x <= 0 && radius_y <= 0) {
    return;
  }
  float cull_mean_x = mean_x;
  float cull_mean_y = mean_y;

  float precision_00 = 0.0f;
  float precision_01 = 0.0f;
  float precision_02 = 0.0f;
  float precision_11 = 0.0f;
  float precision_12 = 0.0f;
  float precision_22 = 0.0f;
  float precision_mean_0 = 0.0f;
  float precision_mean_1 = 0.0f;
  float precision_mean_2 = 0.0f;
  float precision_mean_quadratic = 0.0f;
  if (ray_gaussian_evaluation) {
    const double covariance_00_double = covariance_00;
    const double covariance_01_double = covariance_01;
    const double covariance_02_double = covariance_02;
    const double covariance_11_double = covariance_11;
    const double covariance_12_double = covariance_12;
    const double covariance_22_double = covariance_22;
    const double determinant_3d =
        covariance_00_double * (
            covariance_11_double * covariance_22_double -
            covariance_12_double * covariance_12_double) -
        covariance_01_double * (
            covariance_01_double * covariance_22_double -
            covariance_02_double * covariance_12_double) +
        covariance_02_double * (
            covariance_01_double * covariance_12_double -
            covariance_02_double * covariance_11_double);
    if (!(determinant_3d > 1.0e-48)) {
      return;
    }
    const double inverse_determinant_3d = 1.0 / determinant_3d;
    const double precision_00_double = (
        covariance_11_double * covariance_22_double -
        covariance_12_double * covariance_12_double) *
        inverse_determinant_3d;
    const double precision_01_double = (
        covariance_02_double * covariance_12_double -
        covariance_01_double * covariance_22_double) *
        inverse_determinant_3d;
    const double precision_02_double = (
        covariance_01_double * covariance_12_double -
        covariance_02_double * covariance_11_double) *
        inverse_determinant_3d;
    const double precision_11_double = (
        covariance_00_double * covariance_22_double -
        covariance_02_double * covariance_02_double) *
        inverse_determinant_3d;
    const double precision_12_double = (
        covariance_01_double * covariance_02_double -
        covariance_00_double * covariance_12_double) *
        inverse_determinant_3d;
    const double precision_22_double = (
        covariance_00_double * covariance_11_double -
        covariance_01_double * covariance_01_double) *
        inverse_determinant_3d;
    const double precision_mean_0_double =
        precision_00_double * static_cast<double>(camera_x) +
        precision_01_double * static_cast<double>(camera_y) +
        precision_02_double * static_cast<double>(camera_z);
    const double precision_mean_1_double =
        precision_01_double * static_cast<double>(camera_x) +
        precision_11_double * static_cast<double>(camera_y) +
        precision_12_double * static_cast<double>(camera_z);
    const double precision_mean_2_double =
        precision_02_double * static_cast<double>(camera_x) +
        precision_12_double * static_cast<double>(camera_y) +
        precision_22_double * static_cast<double>(camera_z);
    const double precision_mean_quadratic_double =
        static_cast<double>(camera_x) * precision_mean_0_double +
        static_cast<double>(camera_y) * precision_mean_1_double +
        static_cast<double>(camera_z) * precision_mean_2_double;
    precision_00 = static_cast<float>(precision_00_double);
    precision_01 = static_cast<float>(precision_01_double);
    precision_02 = static_cast<float>(precision_02_double);
    precision_11 = static_cast<float>(precision_11_double);
    precision_12 = static_cast<float>(precision_12_double);
    precision_22 = static_cast<float>(precision_22_double);
    precision_mean_0 = static_cast<float>(precision_mean_0_double);
    precision_mean_1 = static_cast<float>(precision_mean_1_double);
    precision_mean_2 = static_cast<float>(precision_mean_2_double);
    precision_mean_quadratic = static_cast<float>(
        precision_mean_quadratic_double);

    const double support_squared =
        static_cast<double>(gaussian_support_sigma) *
        static_cast<double>(gaussian_support_sigma);
    const double quadric_scale =
        precision_mean_quadratic_double - support_squared;
    const double conic_00 =
        quadric_scale * precision_00_double -
        precision_mean_0_double * precision_mean_0_double;
    const double conic_01 =
        quadric_scale * precision_01_double -
        precision_mean_0_double * precision_mean_1_double;
    const double conic_02 =
        quadric_scale * precision_02_double -
        precision_mean_0_double * precision_mean_2_double;
    const double conic_11 =
        quadric_scale * precision_11_double -
        precision_mean_1_double * precision_mean_1_double;
    const double conic_12 =
        quadric_scale * precision_12_double -
        precision_mean_1_double * precision_mean_2_double;
    const double conic_22 =
        quadric_scale * precision_22_double -
        precision_mean_2_double * precision_mean_2_double;
    const double conic_determinant =
        conic_00 * conic_11 - conic_01 * conic_01;
    bool finite_ellipse = false;
    if (conic_00 > 0.0 && conic_determinant > 0.0) {
      const double inverse_conic_determinant =
          1.0 / conic_determinant;
      const double center_normalized_x = (
          conic_01 * conic_12 -
          conic_11 * conic_02) * inverse_conic_determinant;
      const double center_normalized_y = (
          conic_01 * conic_02 -
          conic_00 * conic_12) * inverse_conic_determinant;
      const double centered_constant =
          conic_22 +
          conic_02 * center_normalized_x +
          conic_12 * center_normalized_y;
      const double ellipse_scale = -centered_constant;
      if (ellipse_scale > 0.0) {
        const double extent_normalized_x = sqrt(
            ellipse_scale * conic_11 *
            inverse_conic_determinant);
        const double extent_normalized_y = sqrt(
            ellipse_scale * conic_00 *
            inverse_conic_determinant);
        const double cull_mean_x_double =
            static_cast<double>(focal_x) * center_normalized_x +
            static_cast<double>(center_x);
        const double cull_mean_y_double =
            static_cast<double>(focal_y) * center_normalized_y +
            static_cast<double>(center_y);
        cull_mean_x = static_cast<float>(cull_mean_x_double);
        cull_mean_y = static_cast<float>(cull_mean_y_double);
        radius_x = max(
            1,
            static_cast<int>(ceilf(
                static_cast<float>(
                    fabs(static_cast<double>(focal_x)) *
                    extent_normalized_x))));
        radius_y = max(
            1,
            static_cast<int>(ceilf(
                static_cast<float>(
                    fabs(static_cast<double>(focal_y)) *
                    extent_normalized_y))));
        finite_ellipse =
            isfinite(cull_mean_x_double) &&
            isfinite(cull_mean_y_double) &&
            isfinite(extent_normalized_x) &&
            isfinite(extent_normalized_y);
      }
    }
    if (!finite_ellipse) {
      cull_mean_x = 0.5f * width;
      cull_mean_y = 0.5f * height;
      radius_x = max(width, height);
      radius_y = max(width, height);
    }
  }

  if (
      cull_mean_x + radius_x <= 0.0f ||
      cull_mean_y + radius_y <= 0.0f ||
      cull_mean_x - radius_x >= width ||
      cull_mean_y - radius_y >= height) {
    return;
  }

  const float inverse_determinant = 1.0f / determinant;
  if (compact_projection_pass != 0) {
    const bool accumulate_tau =
        compact_projection_pass == 1 || compact_projection_pass == 3;
    const float conic_0 = covariance_2d_11 * inverse_determinant;
    const float conic_1 = -covariance_2d_01 * inverse_determinant;
    const float conic_2 = covariance_2d_00 * inverse_determinant;
    const int min_pixel_x = max(
        0,
        static_cast<int>(floorf(cull_mean_x - radius_x)));
    const int min_pixel_y = max(
        0,
        static_cast<int>(floorf(cull_mean_y - radius_y)));
    const int max_pixel_x_exclusive = min(
        width,
        static_cast<int>(ceilf(cull_mean_x + radius_x)));
    const int max_pixel_y_exclusive = min(
        height,
        static_cast<int>(ceilf(cull_mean_y + radius_y)));
    const int bucket = depth_bucket_index(
        camera_z,
        near_plane,
        far_plane,
        depth_bucket_count);
    const int64_t camera_pixel_base =
        static_cast<int64_t>(camera) * height * width;
    bool retained = false;
    bool contributed = false;
    for (
        int pixel_y = min_pixel_y;
        pixel_y < max_pixel_y_exclusive &&
            (compact_projection_pass != 2 || !retained);
        ++pixel_y) {
      for (
          int pixel_x = min_pixel_x;
          pixel_x < max_pixel_x_exclusive;
          ++pixel_x) {
        const float alpha = splat_alpha_at_pixel(
            cull_mean_x,
            cull_mean_y,
            conic_0,
            conic_1,
            conic_2,
            opacity,
            pixel_x,
            pixel_y);
        if (alpha == 0.0f) {
          continue;
        }
        contributed = true;
        const int64_t global_pixel =
            camera_pixel_base +
            static_cast<int64_t>(pixel_y) * width +
            pixel_x;
        if (accumulate_tau) {
          atomicAdd(
              compact_depth_bucket_tau +
                  global_pixel * depth_bucket_count +
                  bucket,
              conservative_optical_thickness(alpha));
        } else if (bucket <= compact_depth_cutoff[global_pixel]) {
          retained = true;
          break;
        }
      }
    }
    if (
        compact_projection_pass == 1 ||
        (compact_projection_pass == 2 && !retained) ||
        (compact_projection_pass == 3 && !contributed)) {
      return;
    }
  }

  const int64_t visible_index = atomicAdd(
      reinterpret_cast<unsigned long long*>(counters + 0),
      static_cast<unsigned long long>(1));
  if (visible_index >= visible_capacity) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(counters + 2),
        static_cast<unsigned long long>(1));
    return;
  }

  visible_means2d[visible_index * 2 + 0] = cull_mean_x;
  visible_means2d[visible_index * 2 + 1] = cull_mean_y;
  visible_conics[visible_index * 3 + 0] =
      covariance_2d_11 * inverse_determinant;
  visible_conics[visible_index * 3 + 1] =
      -covariance_2d_01 * inverse_determinant;
  visible_conics[visible_index * 3 + 2] =
      covariance_2d_00 * inverse_determinant;
  visible_depths[visible_index] = camera_z;
  visible_opacities[visible_index] = fminf(fmaxf(opacity, 0.0f), 1.0f);
  visible_camera_ids[visible_index] = camera;
  visible_gaussian_ids[visible_index] = static_cast<int32_t>(gaussian);
  visible_radii[visible_index * 2 + 0] = radius_x;
  visible_radii[visible_index * 2 + 1] = radius_y;
  if (ray_gaussian_evaluation) {
    visible_ray_precisions[visible_index * 6 + 0] = precision_00;
    visible_ray_precisions[visible_index * 6 + 1] = precision_01;
    visible_ray_precisions[visible_index * 6 + 2] = precision_02;
    visible_ray_precisions[visible_index * 6 + 3] = precision_11;
    visible_ray_precisions[visible_index * 6 + 4] = precision_12;
    visible_ray_precisions[visible_index * 6 + 5] = precision_22;
    visible_ray_precision_means[visible_index * 4 + 0] =
        precision_mean_0;
    visible_ray_precision_means[visible_index * 4 + 1] =
        precision_mean_1;
    visible_ray_precision_means[visible_index * 4 + 2] =
        precision_mean_2;
    visible_ray_precision_means[visible_index * 4 + 3] =
        precision_mean_quadratic;
  }
  if (depth_bucket_counts == nullptr) {
    return;
  }
  const int bucket = depth_bucket_index(
      camera_z,
      near_plane,
      far_plane,
      depth_bucket_count);
  const unsigned int active_mask = __activemask();
  const unsigned int peer_mask =
      __match_any_sync(active_mask, bucket);
  const int leader_lane = __ffs(peer_mask) - 1;
  const int lane = threadIdx.x & 31;
  if (lane == leader_lane) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(
            depth_bucket_counts + bucket),
        static_cast<unsigned long long>(__popc(peer_mask)));
  }
}

__global__ void compact_project_intersections_kernel(
    const float* __restrict__ means,
    const float* __restrict__ covariances,
    const float* __restrict__ opacities,
    const int64_t* __restrict__ registered_scene_ids,
    const int64_t* __restrict__ scene_offsets,
    int scene_count,
    const float* __restrict__ viewmats,
    const float* __restrict__ intrinsics,
    const float* __restrict__ env_xforms,
    const int64_t* __restrict__ camera_scene_ids,
    const int64_t* __restrict__ active_camera_ids,
    int active_count,
    int batch_size,
    int64_t max_scene_gaussians,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float gaussian_support_sigma,
    float covariance_epsilon,
    bool antialiased,
    int depth_bucket_count,
    int compact_projection_pass,
    float* __restrict__ depth_bucket_tau,
    const int32_t* __restrict__ depth_cutoff,
    bool materialize_projected_records,
    float* __restrict__ visible_means2d,
    float* __restrict__ visible_conics,
    float* __restrict__ visible_depths,
    float* __restrict__ visible_opacities,
    int32_t* __restrict__ visible_camera_ids,
    int32_t* __restrict__ visible_gaussian_ids,
    int32_t* __restrict__ visible_radii,
    int64_t visible_capacity,
    uint64_t* __restrict__ keys,
    int32_t* __restrict__ values,
    int64_t intersection_capacity,
    int64_t* __restrict__ counters) {
  const int64_t task_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t task_count =
      static_cast<int64_t>(active_count) * max_scene_gaussians;
  if (task_index >= task_count) {
    return;
  }

  const int active_index =
      static_cast<int>(task_index / max_scene_gaussians);
  const int64_t local_gaussian =
      task_index -
      static_cast<int64_t>(active_index) * max_scene_gaussians;
  const int64_t camera64 = active_camera_ids == nullptr
      ? static_cast<int64_t>(active_index)
      : active_camera_ids[active_index];
  if (camera64 < 0 || camera64 >= batch_size) {
    return;
  }
  const int camera = static_cast<int>(camera64);
  const int scene_slot = find_scene_slot(
      camera_scene_ids[camera],
      registered_scene_ids,
      scene_count);
  if (scene_slot < 0) {
    return;
  }
  const int64_t scene_start = scene_offsets[scene_slot];
  const int64_t scene_end = scene_offsets[scene_slot + 1];
  if (local_gaussian >= scene_end - scene_start) {
    return;
  }
  const int64_t gaussian64 = scene_start + local_gaussian;
  if (gaussian64 > INT_MAX) {
    return;
  }
  const int32_t gaussian = static_cast<int32_t>(gaussian64);

  ScreenProjection projection;
  if (!project_screen_gaussian(
          gaussian,
          camera,
          means,
          covariances,
          opacities,
          viewmats,
          intrinsics,
          env_xforms,
          width,
          height,
          near_plane,
          far_plane,
          gaussian_support_sigma,
          covariance_epsilon,
          antialiased,
          &projection)) {
    return;
  }

  const int min_pixel_x = max(
      0,
      static_cast<int>(floorf(
          projection.mean_x - projection.radius_x)));
  const int min_pixel_y = max(
      0,
      static_cast<int>(floorf(
          projection.mean_y - projection.radius_y)));
  const int max_pixel_x_exclusive = min(
      width,
      static_cast<int>(ceilf(
          projection.mean_x + projection.radius_x)));
  const int max_pixel_y_exclusive = min(
      height,
      static_cast<int>(ceilf(
          projection.mean_y + projection.radius_y)));
  const int bucket = depth_bucket_index(
      projection.depth,
      near_plane,
      far_plane,
      depth_bucket_count);
  const int64_t camera_pixel_base =
      static_cast<int64_t>(camera) * height * width;
  const uint32_t depth_bits = __float_as_uint(projection.depth);
  bool retained = false;
  bool contributed = false;
  for (
      int pixel_y = min_pixel_y;
      pixel_y < max_pixel_y_exclusive;
      ++pixel_y) {
    for (
        int pixel_x = min_pixel_x;
        pixel_x < max_pixel_x_exclusive;
        ++pixel_x) {
      const float alpha = splat_alpha_at_pixel(
          projection.mean_x,
          projection.mean_y,
          projection.conic_0,
          projection.conic_1,
          projection.conic_2,
          projection.opacity,
          pixel_x,
          pixel_y);
      if (alpha == 0.0f) {
        continue;
      }
      contributed = true;
      const int64_t global_pixel =
          camera_pixel_base +
          static_cast<int64_t>(pixel_y) * width +
          pixel_x;
      if (compact_projection_pass == 1) {
        atomicAdd(
            depth_bucket_tau +
                global_pixel * depth_bucket_count +
                bucket,
            conservative_optical_thickness(alpha));
        continue;
      }
      if (bucket > depth_cutoff[global_pixel]) {
        continue;
      }
      retained = true;
      const int64_t intersection_index = atomicAdd(
          reinterpret_cast<unsigned long long*>(counters + 1),
          static_cast<unsigned long long>(1));
      if (intersection_index >= intersection_capacity) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(counters + 3),
            static_cast<unsigned long long>(1));
        continue;
      }
      keys[intersection_index] =
          (static_cast<uint64_t>(global_pixel) << 32) |
          static_cast<uint64_t>(depth_bits);
      values[intersection_index] = gaussian;
    }
  }
  if (
      compact_projection_pass == 1 &&
      materialize_projected_records && contributed) {
    const int64_t visible_index = atomicAdd(
        reinterpret_cast<unsigned long long*>(counters + 0),
        static_cast<unsigned long long>(1));
    if (visible_index >= visible_capacity || visible_index > INT_MAX) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(counters + 2),
          static_cast<unsigned long long>(1));
      return;
    }
    visible_means2d[visible_index * 2 + 0] = projection.mean_x;
    visible_means2d[visible_index * 2 + 1] = projection.mean_y;
    visible_conics[visible_index * 3 + 0] = projection.conic_0;
    visible_conics[visible_index * 3 + 1] = projection.conic_1;
    visible_conics[visible_index * 3 + 2] = projection.conic_2;
    visible_depths[visible_index] = projection.depth;
    visible_opacities[visible_index] = projection.opacity;
    visible_camera_ids[visible_index] = camera;
    visible_gaussian_ids[visible_index] = gaussian;
    visible_radii[visible_index * 2 + 0] = projection.radius_x;
    visible_radii[visible_index * 2 + 1] = projection.radius_y;
  }
  if (
      compact_projection_pass == 2 && retained &&
      !materialize_projected_records) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(counters + 0),
        static_cast<unsigned long long>(1));
  }
}

__global__ void emit_cached_projected_intersections_kernel(
    const float* __restrict__ visible_means2d,
    const float* __restrict__ visible_conics,
    const float* __restrict__ visible_depths,
    const float* __restrict__ visible_opacities,
    const int32_t* __restrict__ visible_camera_ids,
    const int32_t* __restrict__ visible_radii,
    int64_t visible_capacity,
    int width,
    int height,
    float near_plane,
    float far_plane,
    int depth_bucket_count,
    const int32_t* __restrict__ depth_cutoff,
    uint64_t* __restrict__ keys,
    int32_t* __restrict__ values,
    int64_t intersection_capacity,
    int64_t* __restrict__ counters) {
  const int64_t visible_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t visible_count = min(counters[0], visible_capacity);
  if (visible_index >= visible_count) {
    return;
  }

  const float mean_x = visible_means2d[visible_index * 2 + 0];
  const float mean_y = visible_means2d[visible_index * 2 + 1];
  const float conic_0 = visible_conics[visible_index * 3 + 0];
  const float conic_1 = visible_conics[visible_index * 3 + 1];
  const float conic_2 = visible_conics[visible_index * 3 + 2];
  const float depth = visible_depths[visible_index];
  const float opacity = visible_opacities[visible_index];
  const int camera = visible_camera_ids[visible_index];
  const int radius_x = visible_radii[visible_index * 2 + 0];
  const int radius_y = visible_radii[visible_index * 2 + 1];
  const int min_pixel_x = max(
      0,
      static_cast<int>(floorf(mean_x - radius_x)));
  const int min_pixel_y = max(
      0,
      static_cast<int>(floorf(mean_y - radius_y)));
  const int max_pixel_x_exclusive = min(
      width,
      static_cast<int>(ceilf(mean_x + radius_x)));
  const int max_pixel_y_exclusive = min(
      height,
      static_cast<int>(ceilf(mean_y + radius_y)));
  const int bucket = depth_bucket_index(
      depth,
      near_plane,
      far_plane,
      depth_bucket_count);
  const int64_t camera_pixel_base =
      static_cast<int64_t>(camera) * height * width;
  const uint32_t depth_bits = __float_as_uint(depth);

  for (
      int pixel_y = min_pixel_y;
      pixel_y < max_pixel_y_exclusive;
      ++pixel_y) {
    for (
        int pixel_x = min_pixel_x;
        pixel_x < max_pixel_x_exclusive;
        ++pixel_x) {
      const float alpha = splat_alpha_at_pixel(
          mean_x,
          mean_y,
          conic_0,
          conic_1,
          conic_2,
          opacity,
          pixel_x,
          pixel_y);
      if (alpha == 0.0f) {
        continue;
      }
      const int64_t global_pixel =
          camera_pixel_base +
          static_cast<int64_t>(pixel_y) * width +
          pixel_x;
      if (bucket > depth_cutoff[global_pixel]) {
        continue;
      }
      const int64_t intersection_index = atomicAdd(
          reinterpret_cast<unsigned long long*>(counters + 1),
          static_cast<unsigned long long>(1));
      if (intersection_index >= intersection_capacity) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(counters + 3),
            static_cast<unsigned long long>(1));
        continue;
      }
      keys[intersection_index] =
          (static_cast<uint64_t>(global_pixel) << 32) |
          static_cast<uint64_t>(depth_bits);
      values[intersection_index] = static_cast<int32_t>(visible_index);
    }
  }
}

__global__ void prefix_visible_depth_buckets_kernel(
    const int64_t* __restrict__ depth_bucket_counts,
    int depth_bucket_count,
    int64_t* __restrict__ depth_bucket_offsets,
    int64_t* __restrict__ depth_bucket_write_offsets) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int64_t running = 0;
  for (int bucket = 0; bucket < depth_bucket_count; ++bucket) {
    depth_bucket_offsets[bucket] = running;
    depth_bucket_write_offsets[bucket] = running;
    running += depth_bucket_counts[bucket];
  }
  depth_bucket_offsets[depth_bucket_count] = running;
}

__global__ void scatter_visible_depth_buckets_kernel(
    const float* __restrict__ visible_depths,
    int64_t visible_capacity,
    float near_plane,
    float far_plane,
    int depth_bucket_count,
    int64_t* __restrict__ depth_bucket_write_offsets,
    int32_t* __restrict__ depth_ordered_visible_indices,
    const int64_t* __restrict__ counters) {
  const int64_t visible_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t visible_count = min(counters[0], visible_capacity);
  if (visible_index >= visible_count) {
    return;
  }
  const int bucket = depth_bucket_index(
      visible_depths[visible_index],
      near_plane,
      far_plane,
      depth_bucket_count);
  const unsigned int active_mask = __activemask();
  const unsigned int peer_mask =
      __match_any_sync(active_mask, bucket);
  const int leader_lane = __ffs(peer_mask) - 1;
  const int lane = threadIdx.x & 31;
  unsigned long long subgroup_start = 0;
  if (lane == leader_lane) {
    subgroup_start = atomicAdd(
        reinterpret_cast<unsigned long long*>(
            depth_bucket_write_offsets + bucket),
        static_cast<unsigned long long>(__popc(peer_mask)));
  }
  subgroup_start = __shfl_sync(
      peer_mask,
      subgroup_start,
      leader_lane);
  const unsigned int lower_lane_mask =
      lane == 0 ? 0u : ((1u << lane) - 1u);
  const int subgroup_rank =
      __popc(peer_mask & lower_lane_mask);
  const int64_t ordered_index =
      static_cast<int64_t>(subgroup_start) + subgroup_rank;
  depth_ordered_visible_indices[ordered_index] =
      static_cast<int32_t>(visible_index);
}

template <bool kExactRayEvaluation>
__global__ void accumulate_depth_bucket_group_tau_kernel(
    const float* __restrict__ visible_means2d,
    const float* __restrict__ visible_conics,
    const float* __restrict__ visible_depths,
    const float* __restrict__ visible_opacities,
    const int32_t* __restrict__ visible_camera_ids,
    const int32_t* __restrict__ visible_radii,
    const float* __restrict__ visible_ray_precisions,
    const float* __restrict__ visible_ray_precision_means,
    const float* __restrict__ intrinsics,
    int64_t visible_capacity,
    int width,
    int height,
    float gaussian_support_sigma,
    float near_plane,
    float far_plane,
    int depth_bucket_count,
    int depth_bucket_group_size,
    int group_start_bucket,
    int group_end_bucket,
    const int64_t* __restrict__ depth_bucket_offsets,
    const int32_t* __restrict__ depth_ordered_visible_indices,
    const int32_t* __restrict__ depth_cutoff,
    float* __restrict__ depth_bucket_tau,
    const int64_t* __restrict__ counters) {
  __shared__ int64_t range_start;
  __shared__ int64_t range_end;
  if (threadIdx.x == 0) {
    range_start = depth_bucket_offsets[group_start_bucket];
    range_end = depth_bucket_offsets[group_end_bucket];
  }
  __syncthreads();
  const int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t grid_stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (
      int64_t ordered_index = range_start + global_thread;
      ordered_index < range_end;
      ordered_index += grid_stride) {
    const int32_t visible_index =
        depth_ordered_visible_indices[ordered_index];
    if (
        visible_index < 0 ||
        static_cast<int64_t>(visible_index) >=
            min(counters[0], visible_capacity)) {
      continue;
    }
    const int bucket = depth_bucket_index(
        visible_depths[visible_index],
        near_plane,
        far_plane,
        depth_bucket_count);
    const int local_bucket = bucket - group_start_bucket;
    if (
        local_bucket < 0 ||
        bucket >= group_end_bucket ||
        local_bucket >= depth_bucket_group_size) {
      continue;
    }

    const float mean_x = visible_means2d[visible_index * 2 + 0];
    const float mean_y = visible_means2d[visible_index * 2 + 1];
    const float opacity = visible_opacities[visible_index];
    const int radius_x = visible_radii[visible_index * 2 + 0];
    const int radius_y = visible_radii[visible_index * 2 + 1];
    const int camera = visible_camera_ids[visible_index];
    float conic_0 = 0.0f;
    float conic_1 = 0.0f;
    float conic_2 = 0.0f;
    float precision_00 = 0.0f;
    float precision_01 = 0.0f;
    float precision_02 = 0.0f;
    float precision_11 = 0.0f;
    float precision_12 = 0.0f;
    float precision_22 = 0.0f;
    float precision_mean_0 = 0.0f;
    float precision_mean_1 = 0.0f;
    float precision_mean_2 = 0.0f;
    float mean_precision_mean = 0.0f;
    float focal_x = 1.0f;
    float focal_y = 1.0f;
    float center_x = 0.0f;
    float center_y = 0.0f;
    if constexpr (kExactRayEvaluation) {
      precision_00 = visible_ray_precisions[visible_index * 6 + 0];
      precision_01 = visible_ray_precisions[visible_index * 6 + 1];
      precision_02 = visible_ray_precisions[visible_index * 6 + 2];
      precision_11 = visible_ray_precisions[visible_index * 6 + 3];
      precision_12 = visible_ray_precisions[visible_index * 6 + 4];
      precision_22 = visible_ray_precisions[visible_index * 6 + 5];
      precision_mean_0 =
          visible_ray_precision_means[visible_index * 4 + 0];
      precision_mean_1 =
          visible_ray_precision_means[visible_index * 4 + 1];
      precision_mean_2 =
          visible_ray_precision_means[visible_index * 4 + 2];
      mean_precision_mean =
          visible_ray_precision_means[visible_index * 4 + 3];
      const float* intrinsic =
          intrinsics + static_cast<int64_t>(camera) * 9;
      focal_x = intrinsic[0];
      focal_y = intrinsic[4];
      center_x = intrinsic[2];
      center_y = intrinsic[5];
    } else {
      conic_0 = visible_conics[visible_index * 3 + 0];
      conic_1 = visible_conics[visible_index * 3 + 1];
      conic_2 = visible_conics[visible_index * 3 + 2];
    }
    const int min_pixel_x = max(
        0,
        static_cast<int>(floorf(mean_x - radius_x)));
    const int min_pixel_y = max(
        0,
        static_cast<int>(floorf(mean_y - radius_y)));
    const int max_pixel_x_exclusive = min(
        width,
        static_cast<int>(ceilf(mean_x + radius_x)));
    const int max_pixel_y_exclusive = min(
        height,
        static_cast<int>(ceilf(mean_y + radius_y)));
    if (
        min_pixel_x >= max_pixel_x_exclusive ||
        min_pixel_y >= max_pixel_y_exclusive) {
      continue;
    }

    const int64_t camera_pixel_base =
        static_cast<int64_t>(camera) * height * width;
    for (
        int pixel_y = min_pixel_y;
        pixel_y < max_pixel_y_exclusive;
        ++pixel_y) {
      for (
          int pixel_x = min_pixel_x;
          pixel_x < max_pixel_x_exclusive;
          ++pixel_x) {
        const int64_t global_pixel =
            camera_pixel_base +
            static_cast<int64_t>(pixel_y) * width +
            pixel_x;
        if (depth_cutoff[global_pixel] >= 0) {
          continue;
        }
        float alpha = 0.0f;
        if constexpr (kExactRayEvaluation) {
          const float ray_x = (
              (static_cast<float>(pixel_x) + 0.5f) - center_x) /
              focal_x;
          const float ray_y = (
              (static_cast<float>(pixel_y) + 0.5f) - center_y) /
              focal_y;
          alpha = splat_alpha_at_ray(
              ray_x,
              ray_y,
              precision_00,
              precision_01,
              precision_02,
              precision_11,
              precision_12,
              precision_22,
              precision_mean_0,
              precision_mean_1,
              precision_mean_2,
              mean_precision_mean,
              opacity,
              gaussian_support_sigma);
        } else {
          alpha = splat_alpha_at_pixel(
              mean_x,
              mean_y,
              conic_0,
              conic_1,
              conic_2,
              opacity,
              pixel_x,
              pixel_y);
        }
        if (alpha == 0.0f) {
          continue;
        }
        atomicAdd(
            depth_bucket_tau +
                global_pixel * depth_bucket_group_size +
                local_bucket,
            conservative_optical_thickness(alpha));
      }
    }
  }
}

__global__ void advance_depth_cutoff_kernel(
    const float* __restrict__ depth_bucket_tau,
    float* __restrict__ depth_accumulated_tau,
    int64_t pixel_count,
    int depth_bucket_count,
    int depth_bucket_group_size,
    int group_start_bucket,
    int group_end_bucket,
    int32_t* __restrict__ depth_cutoff) {
  const int64_t pixel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixel_count || depth_cutoff[pixel] >= 0) {
    return;
  }
  constexpr float kTransmittanceTauThreshold = 9.210340371976184f;
  float accumulated_tau = depth_accumulated_tau[pixel];
  int cutoff = -1;
  const int64_t base = pixel * depth_bucket_group_size;
  const int local_bucket_count = group_end_bucket - group_start_bucket;
  for (int local_bucket = 0; local_bucket < local_bucket_count; ++local_bucket) {
    accumulated_tau += depth_bucket_tau[base + local_bucket];
    if (accumulated_tau >= kTransmittanceTauThreshold) {
      cutoff = group_start_bucket + local_bucket;
      break;
    }
  }
  if (cutoff < 0 && group_end_bucket == depth_bucket_count) {
    cutoff = depth_bucket_count - 1;
  }
  depth_accumulated_tau[pixel] = accumulated_tau;
  depth_cutoff[pixel] = cutoff;
}

template <bool kExactRayEvaluation>
__global__ void bin_tiles_kernel(
    const float* __restrict__ visible_means2d,
    const float* __restrict__ visible_conics,
    const float* __restrict__ visible_depths,
    const float* __restrict__ visible_opacities,
    const int32_t* __restrict__ visible_camera_ids,
    const int32_t* __restrict__ visible_radii,
    const float* __restrict__ visible_ray_precisions,
    const float* __restrict__ visible_ray_precision_means,
    const float* __restrict__ intrinsics,
    int64_t visible_capacity,
    int width,
    int height,
    int tiles_x,
    int tiles_y,
    int tile_size,
    float gaussian_support_sigma,
    float near_plane,
    float far_plane,
    int depth_bucket_count,
    const int32_t* __restrict__ depth_cutoff,
    uint64_t* __restrict__ keys,
    int32_t* __restrict__ values,
    int64_t intersection_capacity,
    int64_t* __restrict__ counters) {
  const int64_t visible_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t visible_count = min(counters[0], visible_capacity);
  if (visible_index >= visible_count) {
    return;
  }

  const float mean_x = visible_means2d[visible_index * 2 + 0];
  const float mean_y = visible_means2d[visible_index * 2 + 1];
  const int camera = visible_camera_ids[visible_index];
  const int radius_x = visible_radii[visible_index * 2 + 0];
  const int radius_y = visible_radii[visible_index * 2 + 1];
  const int min_tile_x = max(
      0,
      static_cast<int>(floorf(
          mean_x / tile_size -
          static_cast<float>(radius_x) / tile_size)));
  const int min_tile_y = max(
      0,
      static_cast<int>(floorf(
          mean_y / tile_size -
          static_cast<float>(radius_y) / tile_size)));
  const int max_tile_x_exclusive = min(
      tiles_x,
      static_cast<int>(ceilf(
          mean_x / tile_size +
          static_cast<float>(radius_x) / tile_size)));
  const int max_tile_y_exclusive = min(
      tiles_y,
      static_cast<int>(ceilf(
          mean_y / tile_size +
          static_cast<float>(radius_y) / tile_size)));
  if (
      min_tile_x >= max_tile_x_exclusive ||
      min_tile_y >= max_tile_y_exclusive) {
    return;
  }

  const uint32_t depth_bits = __float_as_uint(visible_depths[visible_index]);
  const uint32_t image_tile_base =
      static_cast<uint32_t>(camera) *
      static_cast<uint32_t>(tiles_x * tiles_y);
  for (int tile_y = min_tile_y; tile_y < max_tile_y_exclusive; ++tile_y) {
    for (int tile_x = min_tile_x; tile_x < max_tile_x_exclusive; ++tile_x) {
      const uint32_t global_tile =
          image_tile_base +
          static_cast<uint32_t>(tile_y * tiles_x + tile_x);
      if (tile_size == 1) {
        float splat_alpha = 0.0f;
        if constexpr (kExactRayEvaluation) {
          const float* intrinsic =
              intrinsics + static_cast<int64_t>(camera) * 9;
          const float ray_x = (
              (static_cast<float>(tile_x) + 0.5f) - intrinsic[2]) /
              intrinsic[0];
          const float ray_y = (
              (static_cast<float>(tile_y) + 0.5f) - intrinsic[5]) /
              intrinsic[4];
          splat_alpha = splat_alpha_at_ray(
              ray_x,
              ray_y,
              visible_ray_precisions[visible_index * 6 + 0],
              visible_ray_precisions[visible_index * 6 + 1],
              visible_ray_precisions[visible_index * 6 + 2],
              visible_ray_precisions[visible_index * 6 + 3],
              visible_ray_precisions[visible_index * 6 + 4],
              visible_ray_precisions[visible_index * 6 + 5],
              visible_ray_precision_means[visible_index * 4 + 0],
              visible_ray_precision_means[visible_index * 4 + 1],
              visible_ray_precision_means[visible_index * 4 + 2],
              visible_ray_precision_means[visible_index * 4 + 3],
              visible_opacities[visible_index],
              gaussian_support_sigma);
        } else {
          splat_alpha = splat_alpha_at_pixel(
              mean_x,
              mean_y,
              visible_conics[visible_index * 3 + 0],
              visible_conics[visible_index * 3 + 1],
              visible_conics[visible_index * 3 + 2],
              visible_opacities[visible_index],
              tile_x,
              tile_y);
        }
        if (splat_alpha == 0.0f) {
          continue;
        }
        if (
            depth_cutoff != nullptr &&
            depth_bucket_index(
                visible_depths[visible_index],
                near_plane,
                far_plane,
                depth_bucket_count) > depth_cutoff[global_tile]) {
          continue;
        }
      }
      const int64_t intersection_index = atomicAdd(
          reinterpret_cast<unsigned long long*>(counters + 1),
          static_cast<unsigned long long>(1));
      if (intersection_index >= intersection_capacity) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(counters + 3),
            static_cast<unsigned long long>(1));
        continue;
      }
      keys[intersection_index] =
          (static_cast<uint64_t>(global_tile) << 32) |
          static_cast<uint64_t>(depth_bits);
      values[intersection_index] = static_cast<int32_t>(visible_index);
    }
  }
}

__global__ void count_intersections_by_tile_kernel(
    const uint64_t* __restrict__ keys,
    int64_t intersection_capacity,
    int64_t global_tile_count,
    const int64_t* __restrict__ counters,
    int32_t* __restrict__ tile_counts) {
  const int64_t first_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t intersection_count = min(
      counters[1],
      intersection_capacity);
  for (
      int64_t index = first_index;
      index < intersection_count;
      index += stride) {
    const int64_t tile = static_cast<int64_t>(keys[index] >> 32);
    if (tile < 0 || tile >= global_tile_count) {
      continue;
    }
    atomicAdd(tile_counts + tile, 1);
  }
}

__global__ void initialize_tile_scatter_kernel(
    const int32_t* __restrict__ tile_starts,
    int64_t global_tile_count,
    int32_t* __restrict__ tile_ends,
    int64_t* __restrict__ counters) {
  const int64_t tile =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tile >= global_tile_count) {
    return;
  }
  const int32_t count = tile_ends[tile];
  tile_ends[tile] = tile_starts[tile];
  if (count > 0) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(counters + 4),
        static_cast<unsigned long long>(1));
  }
}

__global__ void scatter_intersections_by_tile_kernel(
    const uint64_t* __restrict__ input_keys,
    const int32_t* __restrict__ input_values,
    int64_t intersection_capacity,
    int64_t global_tile_count,
    const int64_t* __restrict__ counters,
    int32_t* __restrict__ tile_ends,
    uint64_t* __restrict__ grouped_keys,
    int32_t* __restrict__ grouped_values) {
  const int64_t first_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t intersection_count = min(
      counters[1],
      intersection_capacity);
  for (
      int64_t index = first_index;
      index < intersection_count;
      index += stride) {
    const uint64_t key = input_keys[index];
    const int64_t tile = static_cast<int64_t>(key >> 32);
    if (tile < 0 || tile >= global_tile_count) {
      continue;
    }
    const int64_t grouped_index = static_cast<int64_t>(
        atomicAdd(tile_ends + tile, 1));
    if (grouped_index >= intersection_capacity) {
      continue;
    }
    grouped_keys[grouped_index] = key;
    grouped_values[grouped_index] = input_values[index];
  }
}

__global__ void mark_sorted_tile_ranges_kernel(
    const uint64_t* __restrict__ sorted_keys,
    int64_t intersection_count,
    int64_t global_tile_count,
    int32_t* __restrict__ tile_starts,
    int32_t* __restrict__ tile_ends,
    int64_t* __restrict__ counters) {
  const int64_t first_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (
      int64_t index = first_index;
      index < intersection_count;
      index += stride) {
    const int64_t tile = static_cast<int64_t>(sorted_keys[index] >> 32);
    if (tile < 0 || tile >= global_tile_count) {
      continue;
    }
    const bool begins_tile =
        index == 0 ||
        static_cast<int64_t>(sorted_keys[index - 1] >> 32) != tile;
    const bool ends_tile =
        index + 1 == intersection_count ||
        static_cast<int64_t>(sorted_keys[index + 1] >> 32) != tile;
    if (begins_tile) {
      tile_starts[tile] = static_cast<int32_t>(index);
    }
    if (ends_tile) {
      tile_ends[tile] = static_cast<int32_t>(index + 1);
      atomicAdd(
          reinterpret_cast<unsigned long long*>(counters + 4),
          static_cast<unsigned long long>(1));
    }
  }
}

__global__ void build_deterministic_keys_kernel(
    const float* __restrict__ visible_depths,
    const int32_t* __restrict__ visible_gaussian_ids,
    const int32_t* __restrict__ globally_sorted_values,
    int64_t intersection_capacity,
    const int64_t* __restrict__ counters,
    uint64_t* __restrict__ deterministic_keys,
    int32_t* __restrict__ deterministic_values) {
  const int64_t first_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t intersection_count = min(
      counters[1],
      intersection_capacity);
  for (
      int64_t index = first_index;
      index < intersection_count;
      index += stride) {
    const int32_t visible_index = globally_sorted_values[index];
    const uint32_t depth_bits = __float_as_uint(visible_depths[visible_index]);
    const uint32_t gaussian_id = static_cast<uint32_t>(
        visible_gaussian_ids[visible_index]);
    deterministic_keys[index] =
        (static_cast<uint64_t>(depth_bits) << 32) |
        static_cast<uint64_t>(gaussian_id);
    deterministic_values[index] = visible_index;
  }
}

template <
    bool kExactRayEvaluation,
    bool kDirectProjection,
    bool kFullSensorOutput>
__global__ void composite_tiles_kernel(
    const float* __restrict__ visible_means2d,
    const float* __restrict__ visible_conics,
    const float* __restrict__ visible_depths,
    const float* __restrict__ visible_opacities,
    const float* __restrict__ visible_ray_precisions,
    const float* __restrict__ visible_ray_precision_means,
    const int32_t* __restrict__ visible_gaussian_ids,
    const int32_t* __restrict__ sorted_values,
    const int32_t* __restrict__ tile_starts,
    const int32_t* __restrict__ tile_ends,
    const float* __restrict__ means,
    const float* __restrict__ covariances,
    const float* __restrict__ scene_opacities,
    const float* __restrict__ viewmats,
    const float* __restrict__ env_xforms,
    const float* __restrict__ features,
    int feature_width,
    const int64_t* __restrict__ semantic_ids,
    const float* __restrict__ intrinsics,
    const int64_t* __restrict__ active_camera_ids,
    int active_count,
    int width,
    int height,
    int raster_tiles_x,
    int raster_tiles_y,
    int tile_size,
    int range_tiles_x,
    int range_tiles_y,
    int range_tile_size,
    float gaussian_support_sigma,
    float near_plane,
    float far_plane,
    float covariance_epsilon,
    bool antialiased,
    float semantic_min_alpha,
    bool output_srgb,
    float* __restrict__ rgb,
    float* __restrict__ depth,
    float* __restrict__ alpha,
    int64_t* __restrict__ semantic) {
  __shared__ float shared_mean_x[kRasterThreads];
  __shared__ float shared_mean_y[kRasterThreads];
  __shared__ float shared_conic_0[kRasterThreads];
  __shared__ float shared_conic_1[kRasterThreads];
  __shared__ float shared_conic_2[kRasterThreads];
  __shared__ float shared_depth[
      kFullSensorOutput ? kRasterThreads : 1];
  __shared__ float shared_opacity[kRasterThreads];
  __shared__ float shared_color_r[kRasterThreads];
  __shared__ float shared_color_g[kRasterThreads];
  __shared__ float shared_color_b[kRasterThreads];
  __shared__ int64_t shared_semantic[
      kFullSensorOutput ? kRasterThreads : 1];
  __shared__ float shared_precision_00[kRasterThreads];
  __shared__ float shared_precision_01[kRasterThreads];
  __shared__ float shared_precision_02[kRasterThreads];
  __shared__ float shared_precision_11[kRasterThreads];
  __shared__ float shared_precision_12[kRasterThreads];
  __shared__ float shared_precision_22[kRasterThreads];
  __shared__ float shared_precision_mean_0[kRasterThreads];
  __shared__ float shared_precision_mean_1[kRasterThreads];
  __shared__ float shared_precision_mean_2[kRasterThreads];
  __shared__ float shared_mean_precision_mean[kRasterThreads];

  const int raster_tiles_per_image =
      raster_tiles_x * raster_tiles_y;
  const int active_index = blockIdx.x / raster_tiles_per_image;
  if (active_index >= active_count) {
    return;
  }
  const int image_tile =
      blockIdx.x - active_index * raster_tiles_per_image;
  const int64_t camera64 = active_camera_ids == nullptr
      ? static_cast<int64_t>(active_index)
      : active_camera_ids[active_index];
  if (camera64 < 0 || camera64 > INT_MAX) {
    return;
  }
  const int camera = static_cast<int>(camera64);
  const int tile_x = image_tile % raster_tiles_x;
  const int tile_y = image_tile / raster_tiles_x;
  const int range_tile_x = tile_x * tile_size / range_tile_size;
  const int range_tile_y = tile_y * tile_size / range_tile_size;
  if (
      range_tile_x < 0 || range_tile_x >= range_tiles_x ||
      range_tile_y < 0 || range_tile_y >= range_tiles_y) {
    return;
  }
  const int64_t global_tile =
      static_cast<int64_t>(camera) * range_tiles_x * range_tiles_y +
      static_cast<int64_t>(range_tile_y) * range_tiles_x +
      range_tile_x;
  const int lane = threadIdx.x;
  const int tile_pixels = tile_size * tile_size;
  const int pixel_x = tile_x * tile_size + lane % tile_size;
  const int pixel_y = tile_y * tile_size + lane / tile_size;
  const bool valid_pixel =
      lane < tile_pixels && pixel_x < width && pixel_y < height;
  float ray_x = 0.0f;
  float ray_y = 0.0f;
  if constexpr (kExactRayEvaluation) {
    const float* intrinsic =
        intrinsics + static_cast<int64_t>(camera) * 9;
    ray_x = (
        (static_cast<float>(pixel_x) + 0.5f) - intrinsic[2]) /
        intrinsic[0];
    ray_y = (
        (static_cast<float>(pixel_y) + 0.5f) - intrinsic[5]) /
        intrinsic[4];
  }

  float transmittance = 1.0f;
  float accumulated_r = 0.0f;
  float accumulated_g = 0.0f;
  float accumulated_b = 0.0f;
  float accumulated_depth = 0.0f;
  float best_weight = -1.0f;
  int64_t best_semantic = -1;
  bool pixel_done = false;

  const int64_t range_start = static_cast<int64_t>(
      tile_starts[global_tile]);
  const int64_t range_end = static_cast<int64_t>(
      tile_ends[global_tile]);
  if (range_start >= 0 && range_end > range_start) {
    for (
        int64_t chunk_start = range_start;
        chunk_start < range_end;
        chunk_start += kRasterThreads) {
      const int64_t sorted_index = chunk_start + lane;
      const bool load_valid = sorted_index < range_end;
      if (load_valid) {
        const int32_t sorted_value = sorted_values[sorted_index];
        int32_t gaussian = sorted_value;
        if constexpr (kDirectProjection) {
          ScreenProjection projection{};
          const bool projected = project_screen_gaussian(
              gaussian,
              camera,
              means,
              covariances,
              scene_opacities,
              viewmats,
              intrinsics,
              env_xforms,
              width,
              height,
              near_plane,
              far_plane,
              gaussian_support_sigma,
              covariance_epsilon,
              antialiased,
              &projection);
          shared_mean_x[lane] = projection.mean_x;
          shared_mean_y[lane] = projection.mean_y;
          shared_conic_0[lane] = projection.conic_0;
          shared_conic_1[lane] = projection.conic_1;
          shared_conic_2[lane] = projection.conic_2;
          if constexpr (kFullSensorOutput) {
            shared_depth[lane] = projection.depth;
          }
          shared_opacity[lane] = projected
              ? projection.opacity
              : 0.0f;
        } else {
          const int32_t visible_index = sorted_value;
          gaussian = visible_gaussian_ids[visible_index];
          shared_mean_x[lane] = visible_means2d[
              static_cast<int64_t>(visible_index) * 2 + 0];
          shared_mean_y[lane] = visible_means2d[
              static_cast<int64_t>(visible_index) * 2 + 1];
          shared_conic_0[lane] = visible_conics[
              static_cast<int64_t>(visible_index) * 3 + 0];
          shared_conic_1[lane] = visible_conics[
              static_cast<int64_t>(visible_index) * 3 + 1];
          shared_conic_2[lane] = visible_conics[
              static_cast<int64_t>(visible_index) * 3 + 2];
          if constexpr (kFullSensorOutput) {
            shared_depth[lane] = visible_depths[visible_index];
          }
          shared_opacity[lane] = visible_opacities[visible_index];
        }
        const int64_t feature_offset =
            static_cast<int64_t>(gaussian) * feature_width;
        shared_color_r[lane] = features[feature_offset + 0];
        shared_color_g[lane] = features[feature_offset + 1];
        shared_color_b[lane] = features[feature_offset + 2];
        if constexpr (kFullSensorOutput) {
          shared_semantic[lane] = semantic_ids[gaussian];
        }
        if constexpr (kExactRayEvaluation) {
          const int32_t visible_index = sorted_value;
          shared_precision_00[lane] = visible_ray_precisions[
              static_cast<int64_t>(visible_index) * 6 + 0];
          shared_precision_01[lane] = visible_ray_precisions[
              static_cast<int64_t>(visible_index) * 6 + 1];
          shared_precision_02[lane] = visible_ray_precisions[
              static_cast<int64_t>(visible_index) * 6 + 2];
          shared_precision_11[lane] = visible_ray_precisions[
              static_cast<int64_t>(visible_index) * 6 + 3];
          shared_precision_12[lane] = visible_ray_precisions[
              static_cast<int64_t>(visible_index) * 6 + 4];
          shared_precision_22[lane] = visible_ray_precisions[
              static_cast<int64_t>(visible_index) * 6 + 5];
          shared_precision_mean_0[lane] =
              visible_ray_precision_means[
                  static_cast<int64_t>(visible_index) * 4 + 0];
          shared_precision_mean_1[lane] =
              visible_ray_precision_means[
                  static_cast<int64_t>(visible_index) * 4 + 1];
          shared_precision_mean_2[lane] =
              visible_ray_precision_means[
                  static_cast<int64_t>(visible_index) * 4 + 2];
          shared_mean_precision_mean[lane] =
              visible_ray_precision_means[
                  static_cast<int64_t>(visible_index) * 4 + 3];
        }
      }
      __syncthreads();

      const int chunk_count = static_cast<int>(min(
          static_cast<int64_t>(kRasterThreads),
          range_end - chunk_start));
      if (valid_pixel && !pixel_done) {
#pragma unroll 1
        for (int item = 0; item < chunk_count; ++item) {
          float splat_alpha = 0.0f;
          if constexpr (kExactRayEvaluation) {
            splat_alpha = splat_alpha_at_ray(
                ray_x,
                ray_y,
                shared_precision_00[item],
                shared_precision_01[item],
                shared_precision_02[item],
                shared_precision_11[item],
                shared_precision_12[item],
                shared_precision_22[item],
                shared_precision_mean_0[item],
                shared_precision_mean_1[item],
                shared_precision_mean_2[item],
                shared_mean_precision_mean[item],
                shared_opacity[item],
                gaussian_support_sigma);
            if (!(splat_alpha > 0.0f)) {
              continue;
            }
          } else {
            const float delta_x =
                (static_cast<float>(pixel_x) + 0.5f) -
                shared_mean_x[item];
            const float delta_y =
                (static_cast<float>(pixel_y) + 0.5f) -
                shared_mean_y[item];
            const float power = -0.5f * (
                shared_conic_0[item] * delta_x * delta_x +
                2.0f * shared_conic_1[item] * delta_x * delta_y +
                shared_conic_2[item] * delta_y * delta_y);
            if (power > 0.0f || power < -20.0f) {
              continue;
            }
            splat_alpha = fminf(
                kMaxAlpha,
                shared_opacity[item] * __expf(power));
            if (splat_alpha < kAlphaThreshold) {
              continue;
            }
          }
          const float next_transmittance =
              transmittance * (1.0f - splat_alpha);
          if (
              !kExactRayEvaluation &&
              next_transmittance <= kTransmittanceThreshold) {
            pixel_done = true;
            break;
          }
          const float weight = transmittance * splat_alpha;
          accumulated_r += weight * shared_color_r[item];
          accumulated_g += weight * shared_color_g[item];
          accumulated_b += weight * shared_color_b[item];
          if constexpr (kFullSensorOutput) {
            accumulated_depth += weight * shared_depth[item];
            if (weight > best_weight) {
              best_weight = weight;
              best_semantic = shared_semantic[item];
            }
          }
          transmittance = next_transmittance;
          if (
              kExactRayEvaluation &&
              next_transmittance <= kTransmittanceThreshold) {
            pixel_done = true;
            break;
          }
        }
      }
      const int unfinished_pixels =
          __syncthreads_count(valid_pixel && !pixel_done);
      if (unfinished_pixels == 0) {
        break;
      }
    }
  }

  if (!valid_pixel) {
    return;
  }
  const int64_t pixel_index =
      (static_cast<int64_t>(camera) * height + pixel_y) * width + pixel_x;
  const float accumulated_alpha = 1.0f - transmittance;
  if (output_srgb) {
    accumulated_r = linear_to_srgb(accumulated_r);
    accumulated_g = linear_to_srgb(accumulated_g);
    accumulated_b = linear_to_srgb(accumulated_b);
  }
  rgb[pixel_index * 3 + 0] = accumulated_r;
  rgb[pixel_index * 3 + 1] = accumulated_g;
  rgb[pixel_index * 3 + 2] = accumulated_b;
  if constexpr (kFullSensorOutput) {
    alpha[pixel_index] = accumulated_alpha;
    depth[pixel_index] = accumulated_alpha > 1.0e-8f
        ? accumulated_depth / accumulated_alpha
        : __int_as_float(0x7f800000);
    semantic[pixel_index] =
        best_weight >= 0.0f && accumulated_alpha >= semantic_min_alpha
        ? best_semantic
        : -1;
  }
}

__global__ void mark_logical_active_ids_kernel(
    const int64_t* __restrict__ active_camera_ids,
    int64_t active_count,
    int32_t* __restrict__ active_mask,
    int64_t logical_batch) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= active_count) {
    return;
  }
  const int64_t camera = active_camera_ids[index];
  if (camera >= 0 && camera < logical_batch) {
    atomicExch(active_mask + camera, 1);
  }
}

__global__ void build_chunk_active_ids_kernel(
    const int32_t* __restrict__ active_mask,
    int64_t logical_batch,
    int64_t physical_batch,
    int64_t slot_count,
    int64_t* __restrict__ chunk_active_ids) {
  const int64_t slot =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= slot_count) {
    return;
  }
  chunk_active_ids[slot] =
      slot < logical_batch && active_mask[slot] != 0
      ? slot % physical_batch
      : -1;
}

__global__ void aggregate_chunk_counters_kernel(
    const int64_t* __restrict__ chunk_counters,
    int64_t chunk_stride,
    int64_t chunk_count,
    int64_t* __restrict__ counters,
    int64_t* __restrict__ physical_capacity_counters) {
  const int counter = threadIdx.x;
  if (blockIdx.x != 0 || counter >= 5) {
    return;
  }
  int64_t total = 0;
  int64_t maximum = 0;
  for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
    const int64_t value =
        chunk_counters[chunk * chunk_stride + counter];
    total += value;
    maximum = value > maximum ? value : maximum;
  }
  counters[counter] = total;
  physical_capacity_counters[counter] = maximum;
}

void check_dtype(const torch::Tensor& tensor, torch::ScalarType expected, const char* name) {
  TORCH_CHECK(
      tensor.scalar_type() == expected,
      name,
      " has dtype ",
      tensor.scalar_type(),
      ", expected ",
      expected);
}

}  // namespace

void precompute_covariances_cuda(
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    torch::Tensor covariances) {
  c10::cuda::CUDAGuard device_guard(scales.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      scales.get_device());
  const int64_t gaussian_count = scales.size(0);
  const int64_t blocks =
      (gaussian_count + kProjectionThreads - 1) / kProjectionThreads;
  TORCH_CHECK(blocks <= INT_MAX, "covariance precompute grid exceeds CUDA limit");
  precompute_covariances_kernel<<<
      static_cast<unsigned int>(blocks),
      kProjectionThreads,
      0,
      stream>>>(
      scales.data_ptr<float>(),
      rotations.data_ptr<float>(),
      gaussian_count,
      covariances.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepare_chunked_active_ids_cuda(
    const torch::Tensor& active_camera_ids,
    torch::Tensor active_mask,
    torch::Tensor chunk_active_ids,
    int64_t logical_batch,
    int64_t physical_batch) {
  c10::cuda::CUDAGuard device_guard(active_camera_ids.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      active_camera_ids.get_device());
  const int64_t chunk_count =
      (logical_batch + physical_batch - 1) / physical_batch;
  const int64_t slot_count = chunk_count * physical_batch;
  C10_CUDA_CHECK(cudaMemsetAsync(
      active_mask.data_ptr<int32_t>(),
      0,
      static_cast<std::size_t>(logical_batch) * sizeof(int32_t),
      stream));
  const int64_t active_count = active_camera_ids.numel();
  if (active_count > 0) {
    const int64_t blocks =
        (active_count + kProjectionThreads - 1) / kProjectionThreads;
    TORCH_CHECK(blocks <= INT_MAX, "active-ID grid exceeds CUDA limit");
    mark_logical_active_ids_kernel<<<
        static_cast<unsigned int>(blocks),
        kProjectionThreads,
        0,
        stream>>>(
        active_camera_ids.data_ptr<int64_t>(),
        active_count,
        active_mask.data_ptr<int32_t>(),
        logical_batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int64_t slot_blocks =
      (slot_count + kProjectionThreads - 1) / kProjectionThreads;
  TORCH_CHECK(slot_blocks <= INT_MAX, "chunk active-ID grid exceeds CUDA limit");
  build_chunk_active_ids_kernel<<<
      static_cast<unsigned int>(slot_blocks),
      kProjectionThreads,
      0,
      stream>>>(
      active_mask.data_ptr<int32_t>(),
      logical_batch,
      physical_batch,
      slot_count,
      chunk_active_ids.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void aggregate_chunk_counters_cuda(
    const torch::Tensor& chunk_counters,
    torch::Tensor counters,
    torch::Tensor physical_capacity_counters,
    int64_t chunk_count) {
  c10::cuda::CUDAGuard device_guard(chunk_counters.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      chunk_counters.get_device());
  aggregate_chunk_counters_kernel<<<1, 5, 0, stream>>>(
      chunk_counters.data_ptr<int64_t>(),
      chunk_counters.size(1),
      chunk_count,
      counters.data_ptr<int64_t>(),
      physical_capacity_counters.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t sort_temp_bytes_cuda(int64_t num_items, int64_t num_segments) {
  TORCH_CHECK(
      num_items <= INT_MAX,
      "The first renderer implementation supports at most INT_MAX intersections");
  TORCH_CHECK(
      num_segments <= INT_MAX,
      "The first renderer implementation supports at most INT_MAX tile segments");
  std::size_t radix_storage_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr,
      radix_storage_bytes,
      static_cast<uint64_t*>(nullptr),
      static_cast<uint64_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      static_cast<int>(num_items),
      0,
      64,
      static_cast<cudaStream_t>(0)));
  std::size_t segmented_storage_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr,
      segmented_storage_bytes,
      static_cast<uint64_t*>(nullptr),
      static_cast<uint64_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      static_cast<int>(num_items),
      static_cast<int>(num_segments),
      static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      0,
      64,
      static_cast<cudaStream_t>(0)));
  std::size_t scan_storage_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      nullptr,
      scan_storage_bytes,
      static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      static_cast<int>(num_segments),
      static_cast<cudaStream_t>(0)));
  return static_cast<int64_t>(std::max(
      radix_storage_bytes,
      std::max(segmented_storage_bytes, scan_storage_bytes)));
}

int64_t radix_sort_double_buffer_temp_bytes_cuda(
    int64_t num_items,
    int64_t num_segments) {
  TORCH_CHECK(
      num_items <= INT_MAX,
      "The first renderer implementation supports at most INT_MAX intersections");
  const int sort_key_end_bit = emitted_sort_key_end_bit(num_segments);
  cub::DoubleBuffer<uint64_t> keys(
      static_cast<uint64_t*>(nullptr),
      static_cast<uint64_t*>(nullptr));
  cub::DoubleBuffer<int32_t> values(
      static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr));
  std::size_t storage_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr,
      storage_bytes,
      keys,
      values,
      static_cast<int>(num_items),
      0,
      sort_key_end_bit,
      static_cast<cudaStream_t>(0)));
  return static_cast<int64_t>(storage_bytes);
}

int64_t render_cuda(
    const std::vector<torch::Tensor>& scene,
    const std::vector<torch::Tensor>& cameras,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<torch::Tensor>& workspace,
    int64_t height64,
    int64_t width64,
    int64_t max_scene_gaussians,
    double near_plane64,
    double far_plane64,
    double gaussian_support_sigma64,
    double covariance_epsilon64,
    bool antialiased,
    double semantic_min_alpha64,
    bool ray_gaussian_evaluation,
    int64_t tile_size64,
    int64_t depth_bucket_count64,
    int64_t depth_bucket_group_size64,
    bool compact_projection_cache,
    bool materialize_projected_records,
    bool reuse_projection,
    bool output_srgb,
    bool deterministic,
    bool full_sensor_output,
    bool fixed_capacity_sort) {
  check_dtype(scene[kMeans], torch::kFloat32, "means");
  check_dtype(scene[kCovariances], torch::kFloat32, "covariances");
  check_dtype(scene[kOpacities], torch::kFloat32, "opacities");
  check_dtype(scene[kFeatures], torch::kFloat32, "features");
  check_dtype(scene[kSemanticIds], torch::kInt64, "semantic_ids");
  check_dtype(scene[kRegisteredSceneIds], torch::kInt64, "registered_scene_ids");
  check_dtype(scene[kSceneOffsets], torch::kInt64, "scene_offsets");
  check_dtype(cameras[kViewmats], torch::kFloat32, "viewmats");
  check_dtype(cameras[kIntrinsics], torch::kFloat32, "intrinsics");
  check_dtype(cameras[kEnvXforms], torch::kFloat32, "env_xforms");
  check_dtype(cameras[kCameraSceneIds], torch::kInt64, "camera_scene_ids");
  check_dtype(cameras[kActiveCameraIds], torch::kInt64, "active_camera_ids");
  check_dtype(outputs[kRgb], torch::kFloat32, "rgb");
  check_dtype(outputs[kDepth], torch::kFloat32, "depth");
  check_dtype(outputs[kAlpha], torch::kFloat32, "alpha");
  check_dtype(outputs[kSemantic], torch::kInt64, "semantic");
  check_dtype(workspace[kKeysIn], torch::kUInt64, "keys_in");
  check_dtype(workspace[kKeysOut], torch::kUInt64, "keys_out");
  check_dtype(workspace[kValuesIn], torch::kInt32, "values_in");
  check_dtype(workspace[kValuesOut], torch::kInt32, "values_out");
  check_dtype(workspace[kTileStarts], torch::kInt32, "tile_starts");
  check_dtype(workspace[kTileEnds], torch::kInt32, "tile_ends");
  check_dtype(workspace[kCounters], torch::kInt64, "counters");
  check_dtype(workspace[kSortTemp], torch::kUInt8, "sort_temp");
  check_dtype(
      workspace[kDepthBucketTau],
      torch::kFloat32,
      "depth_bucket_tau");
  check_dtype(workspace[kDepthCutoff], torch::kInt32, "depth_cutoff");
  check_dtype(
      workspace[kDepthBucketCounts],
      torch::kInt64,
      "depth_bucket_counts");
  check_dtype(
      workspace[kDepthBucketOffsets],
      torch::kInt64,
      "depth_bucket_offsets");
  check_dtype(
      workspace[kDepthBucketWriteOffsets],
      torch::kInt64,
      "depth_bucket_write_offsets");
  check_dtype(
      workspace[kDepthOrderedVisibleIndices],
      torch::kInt32,
      "depth_ordered_visible_indices");
  check_dtype(
      workspace[kDepthAccumulatedTau],
      torch::kFloat32,
      "depth_accumulated_tau");
  check_dtype(
      workspace[kVisibleRayPrecisions],
      torch::kFloat32,
      "visible_ray_precisions");
  check_dtype(
      workspace[kVisibleRayPrecisionMeans],
      torch::kFloat32,
      "visible_ray_precision_means");

  TORCH_CHECK(width64 <= INT_MAX && height64 <= INT_MAX, "resolution exceeds int32 range");
  TORCH_CHECK(
      max_scene_gaussians <= INT_MAX,
      "max_scene_gaussians exceeds the first implementation limit");
  const int width = static_cast<int>(width64);
  const int height = static_cast<int>(height64);
  const int batch_size = static_cast<int>(cameras[kViewmats].size(0));
  const int active_count = cameras[kActiveCameraIds].numel() == 0
      ? batch_size
      : static_cast<int>(cameras[kActiveCameraIds].numel());
  const int scene_count = static_cast<int>(scene[kRegisteredSceneIds].numel());
  const int feature_width = static_cast<int>(scene[kFeatures].size(1));
  TORCH_CHECK(
      tile_size64 > 0 && tile_size64 <= kMaxTileSize,
      "tile_size must be in [1, ",
      kMaxTileSize,
      "]");
  TORCH_CHECK(
      (tile_size64 & (tile_size64 - 1)) == 0,
      "tile_size must be a power of two");
  const int tile_size = static_cast<int>(tile_size64);
  TORCH_CHECK(
      depth_bucket_count64 > 0 && depth_bucket_count64 <= INT_MAX,
      "depth_bucket_count must fit int32 and be positive");
  const int depth_bucket_count =
      static_cast<int>(depth_bucket_count64);
  TORCH_CHECK(
      depth_bucket_group_size64 > 0 &&
          depth_bucket_group_size64 <= depth_bucket_count64 &&
          depth_bucket_group_size64 <= INT_MAX,
      "depth_bucket_group_size must be positive and no larger than "
      "depth_bucket_count");
  const int depth_bucket_group_size =
      static_cast<int>(depth_bucket_group_size64);
  TORCH_CHECK(
      !compact_projection_cache || tile_size == 1,
      "compact_projection_cache requires tile_size=1");
  TORCH_CHECK(
      !compact_projection_cache || !ray_gaussian_evaluation,
      "compact_projection_cache supports only screen-space evaluation");
  TORCH_CHECK(
      !compact_projection_cache || !deterministic,
      "compact_projection_cache currently requires deterministic=false");
  TORCH_CHECK(
      !materialize_projected_records || compact_projection_cache,
      "materialize_projected_records requires compact_projection_cache");
  TORCH_CHECK(feature_width >= 3, "features must provide at least RGB channels");
  TORCH_CHECK(
      cameras[kEnvXforms].size(0) >= batch_size,
      "env_xforms must provide one transform per camera");

  const int raster_tiles_x = (width + tile_size - 1) / tile_size;
  const int raster_tiles_y = (height + tile_size - 1) / tile_size;
  const int exact_ray_bin_scale =
      ray_gaussian_evaluation && tile_size == kMaxTileSize
      ? kExactRayBinTileScale
      : 1;
  const int bin_tile_size = tile_size * exact_ray_bin_scale;
  const int bin_tiles_x = (width + bin_tile_size - 1) / bin_tile_size;
  const int bin_tiles_y = (height + bin_tile_size - 1) / bin_tile_size;
  const int64_t bin_tiles_per_image =
      static_cast<int64_t>(bin_tiles_x) * bin_tiles_y;
  const int64_t raster_tiles_per_image =
      static_cast<int64_t>(raster_tiles_x) * raster_tiles_y;
  const int64_t global_tile_count =
      static_cast<int64_t>(batch_size) * bin_tiles_per_image;
  TORCH_CHECK(
      global_tile_count <= UINT32_MAX,
      "global tile count exceeds the uint32 key range");
  TORCH_CHECK(
      global_tile_count <= INT_MAX,
      "device tile grouping supports at most INT_MAX tile segments");
  TORCH_CHECK(
      workspace[kTileStarts].numel() >= global_tile_count &&
          workspace[kTileEnds].numel() >= global_tile_count,
      "tile range workspace is too small for the render batch");
  TORCH_CHECK(
      workspace[kTileStarts].numel() == workspace[kTileEnds].numel(),
      "tile start and end workspaces must have the same capacity");
  const int64_t visible_capacity = workspace[kVisibleDepths].numel();
  const int64_t intersection_capacity = workspace[kKeysIn].numel();
  TORCH_CHECK(visible_capacity > 0, "visible capacity must be positive");
  TORCH_CHECK(
      visible_capacity <= INT_MAX,
      "dense depth ordering requires visible capacity to fit int32");
  if (materialize_projected_records) {
    TORCH_CHECK(
        workspace[kVisibleMeans2d].numel() >= visible_capacity * 2 &&
            workspace[kVisibleConics].numel() >= visible_capacity * 3 &&
            workspace[kVisibleOpacities].numel() >= visible_capacity &&
            workspace[kVisibleCameraIds].numel() >= visible_capacity &&
            workspace[kVisibleGaussianIds].numel() >= visible_capacity &&
            workspace[kVisibleRadii].numel() >= visible_capacity * 2,
        "projected-candidate workspace is too small");
  }
  TORCH_CHECK(intersection_capacity > 0, "intersection capacity must be positive");
  TORCH_CHECK(
      intersection_capacity <= INT_MAX,
      "The first renderer implementation supports at most INT_MAX intersections");
  TORCH_CHECK(
      workspace[kCounters].numel() >= 5,
      "counters workspace must contain at least five int64 values");
  if (ray_gaussian_evaluation) {
    TORCH_CHECK(
        workspace[kVisibleRayPrecisions].numel() >=
            visible_capacity * 6,
        "visible ray-precision workspace is too small");
    TORCH_CHECK(
        workspace[kVisibleRayPrecisionMeans].numel() >=
            visible_capacity * 4,
        "visible ray precision-mean workspace is too small");
  }

  c10::cuda::CUDAGuard device_guard(scene[kMeans].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      scene[kMeans].get_device());

  int64_t sorted_buffer_selector = 1;
  const int32_t* sorted_values =
      workspace[kValuesOut].data_ptr<int32_t>();
  if (!reuse_projection) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        workspace[kCounters].data_ptr<int64_t>(),
        0,
        5 * sizeof(int64_t),
        stream));
    if (fixed_capacity_sort && !deterministic) {
      // UINT64_MAX is outside every valid global-tile key range. Sorting a
      // sentinel-filled fixed capacity removes the emitted-count D2H copy and
      // stream synchronization while retaining FlashGS-comparable global
      // radix ordering and bounded overflow counters.
      C10_CUDA_CHECK(cudaMemsetAsync(
          workspace[kKeysIn].data_ptr<uint64_t>(),
          0xff,
          static_cast<std::size_t>(intersection_capacity) * sizeof(uint64_t),
          stream));
    }
  }
  const int32_t* depth_cutoff = nullptr;
  int64_t* depth_bucket_counts = nullptr;
  if (
      !reuse_projection &&
      tile_size == 1 &&
      !deterministic &&
      !compact_projection_cache) {
    TORCH_CHECK(
        workspace[kDepthBucketCounts].numel() >= depth_bucket_count,
        "depth-bucket count workspace is too small");
    depth_bucket_counts =
        workspace[kDepthBucketCounts].data_ptr<int64_t>();
    C10_CUDA_CHECK(cudaMemsetAsync(
        depth_bucket_counts,
        0,
        static_cast<std::size_t>(depth_bucket_count) *
            sizeof(int64_t),
        stream));
  }
  if (!reuse_projection) {
    const int64_t projection_tasks =
        static_cast<int64_t>(active_count) * max_scene_gaussians;
    const int64_t projection_blocks =
        (projection_tasks + kProjectionThreads - 1) / kProjectionThreads;
    TORCH_CHECK(
        projection_blocks <= INT_MAX,
        "projection grid exceeds CUDA limit");
    auto launch_projection = [&](
        int compact_projection_pass,
        int64_t* launch_depth_bucket_counts,
        float* compact_depth_bucket_tau,
        const int32_t* compact_depth_cutoff) {
      project_visible_kernel<<<
          static_cast<unsigned int>(projection_blocks),
          kProjectionThreads,
          0,
          stream>>>(
          scene[kMeans].data_ptr<float>(),
          scene[kCovariances].data_ptr<float>(),
          scene[kOpacities].data_ptr<float>(),
          scene[kRegisteredSceneIds].data_ptr<int64_t>(),
          scene[kSceneOffsets].data_ptr<int64_t>(),
          scene_count,
          cameras[kViewmats].data_ptr<float>(),
          cameras[kIntrinsics].data_ptr<float>(),
          cameras[kEnvXforms].data_ptr<float>(),
          cameras[kCameraSceneIds].data_ptr<int64_t>(),
          cameras[kActiveCameraIds].numel() == 0
              ? nullptr
              : cameras[kActiveCameraIds].data_ptr<int64_t>(),
          active_count,
          batch_size,
          max_scene_gaussians,
          width,
          height,
          static_cast<float>(near_plane64),
          static_cast<float>(far_plane64),
          static_cast<float>(gaussian_support_sigma64),
          static_cast<float>(covariance_epsilon64),
          antialiased,
          ray_gaussian_evaluation,
          workspace[kVisibleMeans2d].data_ptr<float>(),
          workspace[kVisibleConics].data_ptr<float>(),
          workspace[kVisibleDepths].data_ptr<float>(),
          workspace[kVisibleOpacities].data_ptr<float>(),
          workspace[kVisibleCameraIds].data_ptr<int32_t>(),
          workspace[kVisibleGaussianIds].data_ptr<int32_t>(),
          workspace[kVisibleRadii].data_ptr<int32_t>(),
          workspace[kVisibleRayPrecisions].data_ptr<float>(),
          workspace[kVisibleRayPrecisionMeans].data_ptr<float>(),
          visible_capacity,
          depth_bucket_count,
          launch_depth_bucket_counts,
          compact_projection_pass,
          compact_depth_bucket_tau,
          compact_depth_cutoff,
          workspace[kCounters].data_ptr<int64_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    auto launch_compact_projection = [&](
        int compact_projection_pass,
        float* compact_depth_bucket_tau,
        const int32_t* compact_depth_cutoff) {
      compact_project_intersections_kernel<<<
          static_cast<unsigned int>(projection_blocks),
          kProjectionThreads,
          0,
          stream>>>(
          scene[kMeans].data_ptr<float>(),
          scene[kCovariances].data_ptr<float>(),
          scene[kOpacities].data_ptr<float>(),
          scene[kRegisteredSceneIds].data_ptr<int64_t>(),
          scene[kSceneOffsets].data_ptr<int64_t>(),
          scene_count,
          cameras[kViewmats].data_ptr<float>(),
          cameras[kIntrinsics].data_ptr<float>(),
          cameras[kEnvXforms].data_ptr<float>(),
          cameras[kCameraSceneIds].data_ptr<int64_t>(),
          cameras[kActiveCameraIds].numel() == 0
              ? nullptr
              : cameras[kActiveCameraIds].data_ptr<int64_t>(),
          active_count,
          batch_size,
          max_scene_gaussians,
          width,
          height,
          static_cast<float>(near_plane64),
          static_cast<float>(far_plane64),
          static_cast<float>(gaussian_support_sigma64),
          static_cast<float>(covariance_epsilon64),
          antialiased,
          depth_bucket_count,
          compact_projection_pass,
          compact_depth_bucket_tau,
          compact_depth_cutoff,
          materialize_projected_records,
          workspace[kVisibleMeans2d].data_ptr<float>(),
          workspace[kVisibleConics].data_ptr<float>(),
          workspace[kVisibleDepths].data_ptr<float>(),
          workspace[kVisibleOpacities].data_ptr<float>(),
          workspace[kVisibleCameraIds].data_ptr<int32_t>(),
          workspace[kVisibleGaussianIds].data_ptr<int32_t>(),
          workspace[kVisibleRadii].data_ptr<int32_t>(),
          visible_capacity,
          workspace[kKeysIn].data_ptr<uint64_t>(),
          workspace[kValuesIn].data_ptr<int32_t>(),
          intersection_capacity,
          workspace[kCounters].data_ptr<int64_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    if (compact_projection_cache) {
      TORCH_CHECK(
          workspace[kDepthBucketTau].numel() >=
              global_tile_count * depth_bucket_count,
          "compact depth-bucket workspace is too small for the pixel grid");
      TORCH_CHECK(
          workspace[kDepthCutoff].numel() >= global_tile_count,
          "compact depth-cutoff workspace is too small for the pixel grid");
      TORCH_CHECK(
          workspace[kDepthAccumulatedTau].numel() >= global_tile_count,
          "compact accumulated-tau workspace is too small for the pixel grid");
      C10_CUDA_CHECK(cudaMemsetAsync(
          workspace[kDepthBucketTau].data_ptr<float>(),
          0,
          static_cast<std::size_t>(
              global_tile_count * depth_bucket_count) *
              sizeof(float),
          stream));
      C10_CUDA_CHECK(cudaMemsetAsync(
          workspace[kDepthCutoff].data_ptr<int32_t>(),
          0xff,
          static_cast<std::size_t>(global_tile_count) *
              sizeof(int32_t),
          stream));
      C10_CUDA_CHECK(cudaMemsetAsync(
          workspace[kDepthAccumulatedTau].data_ptr<float>(),
          0,
          static_cast<std::size_t>(global_tile_count) *
              sizeof(float),
          stream));

      launch_compact_projection(
          1,
          workspace[kDepthBucketTau].data_ptr<float>(),
          nullptr);
      const int64_t cutoff_blocks =
          (global_tile_count + kProjectionThreads - 1) /
          kProjectionThreads;
      advance_depth_cutoff_kernel<<<
          static_cast<unsigned int>(cutoff_blocks),
          kProjectionThreads,
          0,
          stream>>>(
          workspace[kDepthBucketTau].data_ptr<float>(),
          workspace[kDepthAccumulatedTau].data_ptr<float>(),
          global_tile_count,
          depth_bucket_count,
          depth_bucket_count,
          0,
          depth_bucket_count,
          workspace[kDepthCutoff].data_ptr<int32_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      if (materialize_projected_records) {
        const int64_t candidate_blocks =
            (visible_capacity + kProjectionThreads - 1) /
            kProjectionThreads;
        emit_cached_projected_intersections_kernel<<<
            static_cast<unsigned int>(candidate_blocks),
            kProjectionThreads,
            0,
            stream>>>(
            workspace[kVisibleMeans2d].data_ptr<float>(),
            workspace[kVisibleConics].data_ptr<float>(),
            workspace[kVisibleDepths].data_ptr<float>(),
            workspace[kVisibleOpacities].data_ptr<float>(),
            workspace[kVisibleCameraIds].data_ptr<int32_t>(),
            workspace[kVisibleRadii].data_ptr<int32_t>(),
            visible_capacity,
            width,
            height,
            static_cast<float>(near_plane64),
            static_cast<float>(far_plane64),
            depth_bucket_count,
            workspace[kDepthCutoff].data_ptr<int32_t>(),
            workspace[kKeysIn].data_ptr<uint64_t>(),
            workspace[kValuesIn].data_ptr<int32_t>(),
            intersection_capacity,
            workspace[kCounters].data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        launch_compact_projection(
            2,
            nullptr,
            workspace[kDepthCutoff].data_ptr<int32_t>());
      }
      depth_cutoff = workspace[kDepthCutoff].data_ptr<int32_t>();
    } else {
      launch_projection(0, depth_bucket_counts, nullptr, nullptr);
    }
  }

  const int64_t visible_blocks =
      (visible_capacity + kProjectionThreads - 1) / kProjectionThreads;
  if (
      !reuse_projection &&
      tile_size == 1 &&
      !deterministic &&
      !compact_projection_cache) {
    TORCH_CHECK(
        workspace[kDepthBucketTau].numel() >=
            global_tile_count * depth_bucket_group_size,
        "depth-bucket group workspace is too small for the pixel grid");
    TORCH_CHECK(
        workspace[kDepthCutoff].numel() >= global_tile_count,
        "depth-cutoff workspace is too small for the pixel grid");
    TORCH_CHECK(
        workspace[kDepthBucketOffsets].numel() >=
            depth_bucket_count + 1,
        "depth-bucket offset workspace is too small");
    TORCH_CHECK(
        workspace[kDepthBucketWriteOffsets].numel() >=
            depth_bucket_count,
        "depth-bucket write-offset workspace is too small");
    TORCH_CHECK(
        workspace[kDepthOrderedVisibleIndices].numel() >=
            visible_capacity,
        "depth-ordered visible-index workspace is too small");
    TORCH_CHECK(
        workspace[kDepthAccumulatedTau].numel() >= global_tile_count,
        "depth accumulated-tau workspace is too small");
    C10_CUDA_CHECK(cudaMemsetAsync(
        workspace[kDepthCutoff].data_ptr<int32_t>(),
        0xff,
        static_cast<std::size_t>(global_tile_count) *
            sizeof(int32_t),
        stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        workspace[kDepthAccumulatedTau].data_ptr<float>(),
        0,
        static_cast<std::size_t>(global_tile_count) *
            sizeof(float),
        stream));

    if (!reuse_projection) {
      prefix_visible_depth_buckets_kernel<<<1, 1, 0, stream>>>(
          workspace[kDepthBucketCounts].data_ptr<int64_t>(),
          depth_bucket_count,
          workspace[kDepthBucketOffsets].data_ptr<int64_t>(),
          workspace[kDepthBucketWriteOffsets].data_ptr<int64_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      scatter_visible_depth_buckets_kernel<<<
          static_cast<unsigned int>(visible_blocks),
          kProjectionThreads,
          0,
          stream>>>(
          workspace[kVisibleDepths].data_ptr<float>(),
          visible_capacity,
          static_cast<float>(near_plane64),
          static_cast<float>(far_plane64),
          depth_bucket_count,
          workspace[kDepthBucketWriteOffsets].data_ptr<int64_t>(),
          workspace[kDepthOrderedVisibleIndices].data_ptr<int32_t>(),
          workspace[kCounters].data_ptr<int64_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int64_t cutoff_blocks =
        (global_tile_count + kProjectionThreads - 1) /
        kProjectionThreads;
    const int64_t grouped_blocks = std::min<int64_t>(
        visible_blocks,
        kDepthGroupBlocks);
    for (
        int group_start_bucket = 0;
        group_start_bucket < depth_bucket_count;
        group_start_bucket += depth_bucket_group_size) {
      const int group_end_bucket = std::min(
          depth_bucket_count,
          group_start_bucket + depth_bucket_group_size);
      C10_CUDA_CHECK(cudaMemsetAsync(
          workspace[kDepthBucketTau].data_ptr<float>(),
          0,
          static_cast<std::size_t>(
              global_tile_count * depth_bucket_group_size) *
              sizeof(float),
          stream));
      if (ray_gaussian_evaluation) {
        accumulate_depth_bucket_group_tau_kernel<true><<<
            static_cast<unsigned int>(grouped_blocks),
            kProjectionThreads,
            0,
            stream>>>(
            workspace[kVisibleMeans2d].data_ptr<float>(),
            workspace[kVisibleConics].data_ptr<float>(),
            workspace[kVisibleDepths].data_ptr<float>(),
            workspace[kVisibleOpacities].data_ptr<float>(),
            workspace[kVisibleCameraIds].data_ptr<int32_t>(),
            workspace[kVisibleRadii].data_ptr<int32_t>(),
            workspace[kVisibleRayPrecisions].data_ptr<float>(),
            workspace[kVisibleRayPrecisionMeans].data_ptr<float>(),
            cameras[kIntrinsics].data_ptr<float>(),
            visible_capacity,
            width,
            height,
            static_cast<float>(gaussian_support_sigma64),
            static_cast<float>(near_plane64),
            static_cast<float>(far_plane64),
            depth_bucket_count,
            depth_bucket_group_size,
            group_start_bucket,
            group_end_bucket,
            workspace[kDepthBucketOffsets].data_ptr<int64_t>(),
            workspace[kDepthOrderedVisibleIndices].data_ptr<int32_t>(),
            workspace[kDepthCutoff].data_ptr<int32_t>(),
            workspace[kDepthBucketTau].data_ptr<float>(),
            workspace[kCounters].data_ptr<int64_t>());
      } else {
        accumulate_depth_bucket_group_tau_kernel<false><<<
            static_cast<unsigned int>(grouped_blocks),
            kProjectionThreads,
            0,
            stream>>>(
            workspace[kVisibleMeans2d].data_ptr<float>(),
            workspace[kVisibleConics].data_ptr<float>(),
            workspace[kVisibleDepths].data_ptr<float>(),
            workspace[kVisibleOpacities].data_ptr<float>(),
            workspace[kVisibleCameraIds].data_ptr<int32_t>(),
            workspace[kVisibleRadii].data_ptr<int32_t>(),
            workspace[kVisibleRayPrecisions].data_ptr<float>(),
            workspace[kVisibleRayPrecisionMeans].data_ptr<float>(),
            cameras[kIntrinsics].data_ptr<float>(),
            visible_capacity,
            width,
            height,
            static_cast<float>(gaussian_support_sigma64),
            static_cast<float>(near_plane64),
            static_cast<float>(far_plane64),
            depth_bucket_count,
            depth_bucket_group_size,
            group_start_bucket,
            group_end_bucket,
            workspace[kDepthBucketOffsets].data_ptr<int64_t>(),
            workspace[kDepthOrderedVisibleIndices].data_ptr<int32_t>(),
            workspace[kDepthCutoff].data_ptr<int32_t>(),
            workspace[kDepthBucketTau].data_ptr<float>(),
            workspace[kCounters].data_ptr<int64_t>());
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      advance_depth_cutoff_kernel<<<
          static_cast<unsigned int>(cutoff_blocks),
          kProjectionThreads,
          0,
          stream>>>(
          workspace[kDepthBucketTau].data_ptr<float>(),
          workspace[kDepthAccumulatedTau].data_ptr<float>(),
          global_tile_count,
          depth_bucket_count,
          depth_bucket_group_size,
          group_start_bucket,
          group_end_bucket,
          workspace[kDepthCutoff].data_ptr<int32_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    depth_cutoff = workspace[kDepthCutoff].data_ptr<int32_t>();
  }

  if (!reuse_projection) {
    if (!compact_projection_cache) {
      const int64_t bin_blocks =
          (visible_capacity + kProjectionThreads - 1) /
          kProjectionThreads;
      if (ray_gaussian_evaluation) {
        bin_tiles_kernel<true><<<
          static_cast<unsigned int>(bin_blocks),
          kProjectionThreads,
          0,
          stream>>>(
        workspace[kVisibleMeans2d].data_ptr<float>(),
        workspace[kVisibleConics].data_ptr<float>(),
        workspace[kVisibleDepths].data_ptr<float>(),
        workspace[kVisibleOpacities].data_ptr<float>(),
        workspace[kVisibleCameraIds].data_ptr<int32_t>(),
        workspace[kVisibleRadii].data_ptr<int32_t>(),
        workspace[kVisibleRayPrecisions].data_ptr<float>(),
        workspace[kVisibleRayPrecisionMeans].data_ptr<float>(),
        cameras[kIntrinsics].data_ptr<float>(),
        visible_capacity,
        width,
        height,
        bin_tiles_x,
        bin_tiles_y,
        bin_tile_size,
        static_cast<float>(gaussian_support_sigma64),
        static_cast<float>(near_plane64),
        static_cast<float>(far_plane64),
        depth_bucket_count,
        depth_cutoff,
        workspace[kKeysIn].data_ptr<uint64_t>(),
        workspace[kValuesIn].data_ptr<int32_t>(),
        intersection_capacity,
        workspace[kCounters].data_ptr<int64_t>());
      } else {
        bin_tiles_kernel<false><<<
          static_cast<unsigned int>(bin_blocks),
          kProjectionThreads,
          0,
          stream>>>(
        workspace[kVisibleMeans2d].data_ptr<float>(),
        workspace[kVisibleConics].data_ptr<float>(),
        workspace[kVisibleDepths].data_ptr<float>(),
        workspace[kVisibleOpacities].data_ptr<float>(),
        workspace[kVisibleCameraIds].data_ptr<int32_t>(),
        workspace[kVisibleRadii].data_ptr<int32_t>(),
        workspace[kVisibleRayPrecisions].data_ptr<float>(),
        workspace[kVisibleRayPrecisionMeans].data_ptr<float>(),
        cameras[kIntrinsics].data_ptr<float>(),
        visible_capacity,
        width,
        height,
        bin_tiles_x,
        bin_tiles_y,
        bin_tile_size,
        static_cast<float>(gaussian_support_sigma64),
        static_cast<float>(near_plane64),
        static_cast<float>(far_plane64),
        depth_bucket_count,
        depth_cutoff,
        workspace[kKeysIn].data_ptr<uint64_t>(),
        workspace[kValuesIn].data_ptr<int32_t>(),
        intersection_capacity,
        workspace[kCounters].data_ptr<int64_t>());
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (!deterministic) {
      // The high-throughput path sorts only the emitted prefix.  Reading the
      // device count introduces an explicit stream synchronization, but it
      // removes the count/scan/atomic-scatter pipeline and the very large
      // number of CUB tile segments from changing-camera batches.
      int64_t emitted_intersection_count = intersection_capacity;
      if (!fixed_capacity_sort) {
        emitted_intersection_count = 0;
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &emitted_intersection_count,
            workspace[kCounters].data_ptr<int64_t>() + 1,
            sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        emitted_intersection_count = std::min<int64_t>(
            std::max<int64_t>(emitted_intersection_count, 0),
            intersection_capacity);
      }
      const int sort_key_end_bit = emitted_sort_key_end_bit(
          workspace[kTileStarts].numel());

      C10_CUDA_CHECK(cudaMemsetAsync(
          workspace[kTileStarts].data_ptr<int32_t>(),
          0,
          static_cast<std::size_t>(global_tile_count) * sizeof(int32_t),
          stream));
      C10_CUDA_CHECK(cudaMemsetAsync(
          workspace[kTileEnds].data_ptr<int32_t>(),
          0,
          static_cast<std::size_t>(global_tile_count) * sizeof(int32_t),
          stream));
      if (emitted_intersection_count > 0) {
        std::size_t sort_temp_bytes =
            static_cast<std::size_t>(workspace[kSortTemp].numel());
        cub::DoubleBuffer<uint64_t> keys(
            workspace[kKeysIn].data_ptr<uint64_t>(),
            workspace[kKeysOut].data_ptr<uint64_t>());
        cub::DoubleBuffer<int32_t> values(
            workspace[kValuesIn].data_ptr<int32_t>(),
            workspace[kValuesOut].data_ptr<int32_t>());
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            workspace[kSortTemp].data_ptr<uint8_t>(),
            sort_temp_bytes,
            keys,
            values,
            static_cast<int>(emitted_intersection_count),
            0,
            sort_key_end_bit,
            stream));
        TORCH_CHECK(
            keys.selector == values.selector,
            "CUB returned different key and value buffer selectors");
        sorted_buffer_selector = values.selector;
        sorted_values = values.Current();

        const int64_t range_blocks = std::min<int64_t>(
            (emitted_intersection_count + kProjectionThreads - 1) /
                kProjectionThreads,
            kTileGroupingBlocks);
        mark_sorted_tile_ranges_kernel<<<
            static_cast<unsigned int>(range_blocks),
            kProjectionThreads,
            0,
            stream>>>(
            keys.Current(),
            emitted_intersection_count,
            global_tile_count,
            workspace[kTileStarts].data_ptr<int32_t>(),
            workspace[kTileEnds].data_ptr<int32_t>(),
            workspace[kCounters].data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      // Deterministic rendering retains the device-only segmented path so its
      // depth/Gaussian-ID tie-break ordering remains unchanged.
      C10_CUDA_CHECK(cudaMemsetAsync(
          workspace[kTileEnds].data_ptr<int32_t>(),
          0,
          static_cast<std::size_t>(global_tile_count) * sizeof(int32_t),
          stream));
      const int64_t grouping_blocks = std::min<int64_t>(
          (intersection_capacity + kProjectionThreads - 1) /
              kProjectionThreads,
          kTileGroupingBlocks);
      count_intersections_by_tile_kernel<<<
          static_cast<unsigned int>(grouping_blocks),
          kProjectionThreads,
          0,
          stream>>>(
          workspace[kKeysIn].data_ptr<uint64_t>(),
          intersection_capacity,
          global_tile_count,
          workspace[kCounters].data_ptr<int64_t>(),
          workspace[kTileEnds].data_ptr<int32_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      std::size_t sort_temp_bytes =
          static_cast<std::size_t>(workspace[kSortTemp].numel());
      C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          workspace[kSortTemp].data_ptr<uint8_t>(),
          sort_temp_bytes,
          workspace[kTileEnds].data_ptr<int32_t>(),
          workspace[kTileStarts].data_ptr<int32_t>(),
          static_cast<int>(global_tile_count),
          stream));

      const int64_t range_blocks =
          (global_tile_count + kProjectionThreads - 1) /
          kProjectionThreads;
      initialize_tile_scatter_kernel<<<
          static_cast<unsigned int>(range_blocks),
          kProjectionThreads,
          0,
          stream>>>(
          workspace[kTileStarts].data_ptr<int32_t>(),
          global_tile_count,
          workspace[kTileEnds].data_ptr<int32_t>(),
          workspace[kCounters].data_ptr<int64_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      scatter_intersections_by_tile_kernel<<<
          static_cast<unsigned int>(grouping_blocks),
          kProjectionThreads,
          0,
          stream>>>(
          workspace[kKeysIn].data_ptr<uint64_t>(),
          workspace[kValuesIn].data_ptr<int32_t>(),
          intersection_capacity,
          global_tile_count,
          workspace[kCounters].data_ptr<int64_t>(),
          workspace[kTileEnds].data_ptr<int32_t>(),
          workspace[kKeysOut].data_ptr<uint64_t>(),
          workspace[kValuesOut].data_ptr<int32_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      build_deterministic_keys_kernel<<<
          static_cast<unsigned int>(grouping_blocks),
          kProjectionThreads,
          0,
          stream>>>(
          workspace[kVisibleDepths].data_ptr<float>(),
          workspace[kVisibleGaussianIds].data_ptr<int32_t>(),
          workspace[kValuesOut].data_ptr<int32_t>(),
          intersection_capacity,
          workspace[kCounters].data_ptr<int64_t>(),
          workspace[kKeysIn].data_ptr<uint64_t>(),
          workspace[kValuesIn].data_ptr<int32_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      sort_temp_bytes =
          static_cast<std::size_t>(workspace[kSortTemp].numel());
      C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
          workspace[kSortTemp].data_ptr<uint8_t>(),
          sort_temp_bytes,
          workspace[kKeysIn].data_ptr<uint64_t>(),
          workspace[kKeysOut].data_ptr<uint64_t>(),
          workspace[kValuesIn].data_ptr<int32_t>(),
          workspace[kValuesOut].data_ptr<int32_t>(),
          static_cast<int>(intersection_capacity),
          static_cast<int>(global_tile_count),
          workspace[kTileStarts].data_ptr<int32_t>(),
          workspace[kTileEnds].data_ptr<int32_t>(),
          0,
          64,
          stream));
    }
  }

  const int64_t raster_blocks =
      static_cast<int64_t>(active_count) * raster_tiles_per_image;
  TORCH_CHECK(raster_blocks <= INT_MAX, "raster grid exceeds CUDA limit");
#define LAUNCH_COMPOSITOR(EXACT_RAY, DIRECT_PROJECTION, FULL_SENSOR)       \
  composite_tiles_kernel<EXACT_RAY, DIRECT_PROJECTION, FULL_SENSOR><<<    \
      static_cast<unsigned int>(raster_blocks),                            \
      kRasterThreads,                                                      \
      0,                                                                   \
      stream>>>(                                                           \
      workspace[kVisibleMeans2d].data_ptr<float>(),                        \
      workspace[kVisibleConics].data_ptr<float>(),                         \
      workspace[kVisibleDepths].data_ptr<float>(),                         \
      workspace[kVisibleOpacities].data_ptr<float>(),                      \
      workspace[kVisibleRayPrecisions].data_ptr<float>(),                  \
      workspace[kVisibleRayPrecisionMeans].data_ptr<float>(),              \
      workspace[kVisibleGaussianIds].data_ptr<int32_t>(),                  \
      sorted_values,                                                       \
      workspace[kTileStarts].data_ptr<int32_t>(),                          \
      workspace[kTileEnds].data_ptr<int32_t>(),                            \
      scene[kMeans].data_ptr<float>(),                                     \
      scene[kCovariances].data_ptr<float>(),                               \
      scene[kOpacities].data_ptr<float>(),                                 \
      cameras[kViewmats].data_ptr<float>(),                                \
      cameras[kEnvXforms].data_ptr<float>(),                               \
      scene[kFeatures].data_ptr<float>(),                                  \
      feature_width,                                                       \
      scene[kSemanticIds].data_ptr<int64_t>(),                             \
      cameras[kIntrinsics].data_ptr<float>(),                              \
      cameras[kActiveCameraIds].numel() == 0                               \
          ? nullptr                                                        \
          : cameras[kActiveCameraIds].data_ptr<int64_t>(),                 \
      active_count,                                                        \
      width,                                                               \
      height,                                                              \
      raster_tiles_x,                                                      \
      raster_tiles_y,                                                      \
      tile_size,                                                           \
      bin_tiles_x,                                                         \
      bin_tiles_y,                                                         \
      bin_tile_size,                                                       \
      static_cast<float>(gaussian_support_sigma64),                        \
      static_cast<float>(near_plane64),                                    \
      static_cast<float>(far_plane64),                                     \
      static_cast<float>(covariance_epsilon64),                            \
      antialiased,                                                         \
      static_cast<float>(semantic_min_alpha64),                            \
      output_srgb,                                                         \
      outputs[kRgb].data_ptr<float>(),                                     \
      outputs[kDepth].data_ptr<float>(),                                   \
      outputs[kAlpha].data_ptr<float>(),                                   \
      outputs[kSemantic].data_ptr<int64_t>())

  if (compact_projection_cache && !materialize_projected_records) {
    if (full_sensor_output) {
      LAUNCH_COMPOSITOR(false, true, true);
    } else {
      LAUNCH_COMPOSITOR(false, true, false);
    }
  } else if (ray_gaussian_evaluation) {
    if (full_sensor_output) {
      LAUNCH_COMPOSITOR(true, false, true);
    } else {
      LAUNCH_COMPOSITOR(true, false, false);
    }
  } else if (full_sensor_output) {
    LAUNCH_COMPOSITOR(false, false, true);
  } else {
    LAUNCH_COMPOSITOR(false, false, false);
  }
#undef LAUNCH_COMPOSITOR
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return sorted_buffer_selector;
}
