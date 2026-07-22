#include "ops.h"

#include <algorithm>
#include <climits>

#define GLM_FORCE_CUDA
#include "glm/glm.hpp"

namespace flashgs_adapter {
namespace {

__device__ __forceinline__ void quaternion_to_rotation(
    float w,
    float x,
    float y,
    float z,
    float* rotation) {
  const float inverse_norm = rsqrtf(fmaxf(w * w + x * x + y * y + z * z, 1.0e-20f));
  w *= inverse_norm;
  x *= inverse_norm;
  y *= inverse_norm;
  z *= inverse_norm;
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

__global__ void precompute_covariances_kernel(
    int64_t count,
    const float* __restrict__ scales,
    const float* __restrict__ rotations,
    float* __restrict__ covariances) {
  const int64_t gaussian =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gaussian >= count) {
    return;
  }
  float rotation[9];
  const float* quaternion = rotations + gaussian * 4;
  quaternion_to_rotation(
      quaternion[0], quaternion[1], quaternion[2], quaternion[3], rotation);
  const float* scale = scales + gaussian * 3;
  const float sx = scale[0] * scale[0];
  const float sy = scale[1] * scale[1];
  const float sz = scale[2] * scale[2];
  float* covariance = covariances + gaussian * 6;
  covariance[0] =
      rotation[0] * rotation[0] * sx +
      rotation[1] * rotation[1] * sy +
      rotation[2] * rotation[2] * sz;
  covariance[1] =
      rotation[0] * rotation[3] * sx +
      rotation[1] * rotation[4] * sy +
      rotation[2] * rotation[5] * sz;
  covariance[2] =
      rotation[0] * rotation[6] * sx +
      rotation[1] * rotation[7] * sy +
      rotation[2] * rotation[8] * sz;
  covariance[3] =
      rotation[3] * rotation[3] * sx +
      rotation[4] * rotation[4] * sy +
      rotation[5] * rotation[5] * sz;
  covariance[4] =
      rotation[3] * rotation[6] * sx +
      rotation[4] * rotation[7] * sy +
      rotation[5] * rotation[8] * sz;
  covariance[5] =
      rotation[6] * rotation[6] * sx +
      rotation[7] * rotation[7] * sy +
      rotation[8] * rotation[8] * sz;
}

__device__ __forceinline__ void get_rect(
    float2 center,
    int radius_x,
    int radius_y,
    int2* rect_min,
    int2* rect_max,
    dim3 grid,
    int block_x,
    int block_y) {
  rect_min->x = min(
      static_cast<int>(grid.x),
      max(0, static_cast<int>((center.x - radius_x) / block_x)));
  rect_min->y = min(
      static_cast<int>(grid.y),
      max(0, static_cast<int>((center.y - radius_y) / block_y)));
  rect_max->x = min(
      static_cast<int>(grid.x),
      max(0, static_cast<int>((center.x + radius_x) / block_x) + 1));
  rect_max->y = min(
      static_cast<int>(grid.y),
      max(0, static_cast<int>((center.y + radius_y) / block_y) + 1));
}

__device__ __forceinline__ bool segment_intersects_ellipse(
    float a,
    float b,
    float c,
    float center,
    float lower,
    float upper) {
  const float delta = b * b - 4.0f * a * c;
  const float t1 = (lower - center) * (2.0f * a) + b;
  const float t2 = (upper - center) * (2.0f * a) + b;
  return delta >= 0.0f &&
      (t1 <= 0.0f || t1 * t1 <= delta) &&
      (t2 >= 0.0f || t2 * t2 <= delta);
}

__device__ __forceinline__ bool block_intersects_ellipse(
    float2 pixel_min,
    float2 pixel_max,
    float2 center,
    float3 conic,
    float half_support_squared) {
  float dx = center.x * 2.0f < pixel_min.x + pixel_max.x
      ? center.x - pixel_min.x
      : center.x - pixel_max.x;
  float a = conic.z;
  float b = -2.0f * conic.y * dx;
  float c = conic.x * dx * dx - 2.0f * half_support_squared;
  if (segment_intersects_ellipse(
          a, b, c, center.y, pixel_min.y, pixel_max.y)) {
    return true;
  }
  float dy = center.y * 2.0f < pixel_min.y + pixel_max.y
      ? center.y - pixel_min.y
      : center.y - pixel_max.y;
  a = conic.x;
  b = -2.0f * conic.y * dy;
  c = conic.z * dy * dy - 2.0f * half_support_squared;
  return segment_intersects_ellipse(
      a, b, c, center.x, pixel_min.x, pixel_max.x);
}

__device__ __forceinline__ bool block_contains_center(
    float2 pixel_min,
    float2 pixel_max,
    float2 center) {
  return center.x >= pixel_min.x && center.x <= pixel_max.x &&
      center.y >= pixel_min.y && center.y <= pixel_max.y;
}

__device__ __forceinline__ int64_t reserve_intersections(
    int count,
    int64_t capacity,
    int64_t* counters) {
  const unsigned long long offset = atomicAdd(
      reinterpret_cast<unsigned long long*>(counters + 1),
      static_cast<unsigned long long>(count));
  const unsigned long long end = offset + static_cast<unsigned long long>(count);
  if (end > static_cast<unsigned long long>(capacity)) {
    const unsigned long long first_dropped = max(
        offset,
        static_cast<unsigned long long>(capacity));
    atomicAdd(
        reinterpret_cast<unsigned long long*>(counters + 2),
        end - first_dropped);
  }
  return static_cast<int64_t>(offset);
}

__device__ __forceinline__ bool project_gaussian(
    int gaussian,
    const glm::vec3* positions,
    const cov3d_t* covariances,
    const float* opacities,
    const float* view,
    const float* intrinsic,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float support_sigma,
    float covariance_epsilon,
    float2* point_xy,
    float3* conic,
    float* depth,
    float* opacity,
    float* half_support_squared,
    int* radius_x,
    int* radius_y) {
  *opacity = opacities[gaussian];
  if (!(*opacity >= FLASHGS_ALPHA_THRESHOLD)) {
    return false;
  }
  const glm::vec3 mean = positions[gaussian];
  const float camera_x =
      view[0] * mean.x + view[1] * mean.y + view[2] * mean.z + view[3];
  const float camera_y =
      view[4] * mean.x + view[5] * mean.y + view[6] * mean.z + view[7];
  const float camera_z =
      view[8] * mean.x + view[9] * mean.y + view[10] * mean.z + view[11];
  if (camera_z < near_plane || camera_z > far_plane) {
    return false;
  }
  const float focal_x = intrinsic[0];
  const float focal_y = intrinsic[4];
  const float center_x = intrinsic[2];
  const float center_y = intrinsic[5];
  const float inverse_z = 1.0f / camera_z;
  point_xy->x = focal_x * camera_x * inverse_z + center_x;
  point_xy->y = focal_y * camera_y * inverse_z + center_y;

  const cov3d_t object = covariances[gaussian];
  const float r00 = view[0];
  const float r01 = view[1];
  const float r02 = view[2];
  const float r10 = view[4];
  const float r11 = view[5];
  const float r12 = view[6];
  const float r20 = view[8];
  const float r21 = view[9];
  const float r22 = view[10];
  const float c00 = object.s[0];
  const float c01 = object.s[1];
  const float c02 = object.s[2];
  const float c11 = object.s[3];
  const float c12 = object.s[4];
  const float c22 = object.s[5];
#define ROTATED_COV(A0, A1, A2, B0, B1, B2) \
  ((A0) * ((B0) * c00 + (B1) * c01 + (B2) * c02) + \
   (A1) * ((B0) * c01 + (B1) * c11 + (B2) * c12) + \
   (A2) * ((B0) * c02 + (B1) * c12 + (B2) * c22))
  const float camera_cov_00 = ROTATED_COV(r00, r01, r02, r00, r01, r02);
  const float camera_cov_01 = ROTATED_COV(r00, r01, r02, r10, r11, r12);
  const float camera_cov_02 = ROTATED_COV(r00, r01, r02, r20, r21, r22);
  const float camera_cov_11 = ROTATED_COV(r10, r11, r12, r10, r11, r12);
  const float camera_cov_12 = ROTATED_COV(r10, r11, r12, r20, r21, r22);
  const float camera_cov_22 = ROTATED_COV(r20, r21, r22, r20, r21, r22);
#undef ROTATED_COV

  const float tan_fov_x = 0.5f * width / focal_x;
  const float tan_fov_y = 0.5f * height / focal_y;
  const float limit_x_positive =
      (width - center_x) / focal_x + 0.3f * tan_fov_x;
  const float limit_x_negative = center_x / focal_x + 0.3f * tan_fov_x;
  const float limit_y_positive =
      (height - center_y) / focal_y + 0.3f * tan_fov_y;
  const float limit_y_negative = center_y / focal_y + 0.3f * tan_fov_y;
  const float clamped_x = camera_z * fminf(
      limit_x_positive, fmaxf(-limit_x_negative, camera_x * inverse_z));
  const float clamped_y = camera_z * fminf(
      limit_y_positive, fmaxf(-limit_y_negative, camera_y * inverse_z));
  const float inverse_z_squared = inverse_z * inverse_z;
  const float j00 = focal_x * inverse_z;
  const float j02 = -focal_x * clamped_x * inverse_z_squared;
  const float j11 = focal_y * inverse_z;
  const float j12 = -focal_y * clamped_y * inverse_z_squared;
  const float cov2d_00 =
      j00 * j00 * camera_cov_00 +
      2.0f * j00 * j02 * camera_cov_02 +
      j02 * j02 * camera_cov_22 + covariance_epsilon;
  const float cov2d_01 =
      j00 * j11 * camera_cov_01 +
      j00 * j12 * camera_cov_02 +
      j02 * j11 * camera_cov_12 +
      j02 * j12 * camera_cov_22;
  const float cov2d_11 =
      j11 * j11 * camera_cov_11 +
      2.0f * j11 * j12 * camera_cov_12 +
      j12 * j12 * camera_cov_22 + covariance_epsilon;
  const float determinant = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01;
  if (!(determinant > 0.0f)) {
    return false;
  }
  const float opacity_extend = sqrtf(
      2.0f * __logf(*opacity / FLASHGS_ALPHA_THRESHOLD));
  const float extend = fminf(support_sigma, opacity_extend);
  *half_support_squared = 0.5f * extend * extend;
  *radius_x = static_cast<int>(ceilf(extend * sqrtf(cov2d_00)));
  *radius_y = static_cast<int>(ceilf(extend * sqrtf(cov2d_11)));
  if (*radius_x <= 0 && *radius_y <= 0) {
    return false;
  }
  if (
      point_xy->x + *radius_x <= 0.0f ||
      point_xy->y + *radius_y <= 0.0f ||
      point_xy->x - *radius_x >= width ||
      point_xy->y - *radius_y >= height) {
    return false;
  }
  const float inverse_determinant = 1.0f / determinant;
  conic->x = cov2d_11 * inverse_determinant;
  conic->y = -cov2d_01 * inverse_determinant;
  conic->z = cov2d_00 * inverse_determinant;
  *depth = camera_z;
  *opacity = fminf(fmaxf(*opacity, 0.0f), 1.0f);
  return true;
}

__global__ void preprocess_kernel(
    int count,
    const glm::vec3* __restrict__ positions,
    const cov3d_t* __restrict__ covariances,
    const float* __restrict__ opacities,
    const glm::vec3* __restrict__ colors,
    const float* __restrict__ viewmat,
    const float* __restrict__ intrinsics,
    int width,
    int height,
    int block_x,
    int block_y,
    float near_plane,
    float far_plane,
    float support_sigma,
    float covariance_epsilon,
    float2* __restrict__ points_xy,
    float4* __restrict__ rgb_depth,
    float4* __restrict__ conic_opacity,
    uint64_t* __restrict__ keys_unsorted,
    uint32_t* __restrict__ values_unsorted,
    int64_t intersection_capacity,
    int64_t* __restrict__ counters,
    dim3 grid) {
  const int lane = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = blockIdx.x * blockDim.z + threadIdx.z;
  const int gaussian = warp_id * FLASHGS_WARP_SIZE + lane;
  bool point_valid = false;
  float2 point_xy{};
  float3 conic{};
  float depth = 0.0f;
  float opacity = 0.0f;
  float half_support_squared = 0.0f;
  int radius_x = 0;
  int radius_y = 0;
  int2 rect_min{};
  int2 rect_max{};
  if (gaussian < count) {
    point_valid = project_gaussian(
        gaussian,
        positions,
        covariances,
        opacities,
        viewmat,
        intrinsics,
        width,
        height,
        near_plane,
        far_plane,
        support_sigma,
        covariance_epsilon,
        &point_xy,
        &conic,
        &depth,
        &opacity,
        &half_support_squared,
        &radius_x,
        &radius_y);
    if (point_valid) {
      get_rect(
          point_xy,
          radius_x,
          radius_y,
          &rect_min,
          &rect_max,
          grid,
          block_x,
          block_y);
      point_valid =
          (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) > 0;
    }
  }

  const bool single_tile = point_valid &&
      (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 1;
  bool gaussian_emitted = false;
  if (single_tile) {
    const float2 pixel_min = {
        rect_min.x * block_x + 0.5f,
        rect_min.y * block_y + 0.5f,
    };
    const float2 pixel_max = {
        fminf(width - 0.5f, pixel_min.x + block_x - 1),
        fminf(height - 0.5f, pixel_min.y + block_y - 1),
    };
    const bool valid = block_contains_center(pixel_min, pixel_max, point_xy) ||
        block_intersects_ellipse(
            pixel_min, pixel_max, point_xy, conic, half_support_squared);
    if (valid) {
      const int64_t offset = reserve_intersections(
          1, intersection_capacity, counters);
      if (offset < intersection_capacity) {
        uint64_t key = static_cast<uint64_t>(
            rect_min.y * static_cast<int>(grid.x) + rect_min.x) << 32;
        key |= static_cast<uint64_t>(__float_as_uint(depth));
        keys_unsorted[offset] = key;
        values_unsorted[offset] = static_cast<uint32_t>(gaussian);
      }
      gaussian_emitted = true;
    }
    point_valid = false;
  }

  unsigned int multi_tiles = __ballot_sync(0xffffffffu, point_valid);
  while (multi_tiles) {
    const int source_lane = __ffs(multi_tiles) - 1;
    multi_tiles &= multi_tiles - 1;
    const float2 source_point = {
        __shfl_sync(0xffffffffu, point_xy.x, source_lane),
        __shfl_sync(0xffffffffu, point_xy.y, source_lane),
    };
    const float3 source_conic = {
        __shfl_sync(0xffffffffu, conic.x, source_lane),
        __shfl_sync(0xffffffffu, conic.y, source_lane),
        __shfl_sync(0xffffffffu, conic.z, source_lane),
    };
    const int2 source_rect_min = {
        __shfl_sync(0xffffffffu, rect_min.x, source_lane),
        __shfl_sync(0xffffffffu, rect_min.y, source_lane),
    };
    const int2 source_rect_max = {
        __shfl_sync(0xffffffffu, rect_max.x, source_lane),
        __shfl_sync(0xffffffffu, rect_max.y, source_lane),
    };
    const float source_depth = __shfl_sync(
        0xffffffffu, depth, source_lane);
    const float source_half_support_squared = __shfl_sync(
        0xffffffffu, half_support_squared, source_lane);
    const int source_gaussian = warp_id * FLASHGS_WARP_SIZE + source_lane;
    for (int y0 = source_rect_min.y; y0 < source_rect_max.y; y0 += blockDim.y) {
      const int y = y0 + threadIdx.y;
      for (int x0 = source_rect_min.x; x0 < source_rect_max.x; x0 += blockDim.x) {
        const int x = x0 + threadIdx.x;
        bool valid = y < source_rect_max.y && x < source_rect_max.x;
        if (valid) {
          const float2 pixel_min = {
              x * block_x + 0.5f,
              y * block_y + 0.5f,
          };
          const float2 pixel_max = {
              fminf(width - 0.5f, pixel_min.x + block_x - 1),
              fminf(height - 0.5f, pixel_min.y + block_y - 1),
          };
          valid = block_contains_center(pixel_min, pixel_max, source_point) ||
              block_intersects_ellipse(
                  pixel_min,
                  pixel_max,
                  source_point,
                  source_conic,
                  source_half_support_squared);
        }
        const unsigned int mask = __ballot_sync(0xffffffffu, valid);
        if (mask == 0) {
          continue;
        }
        int64_t offset = 0;
        if (lane == 0) {
          offset = reserve_intersections(
              __popc(mask), intersection_capacity, counters);
        }
        offset = __shfl_sync(0xffffffffu, offset, 0);
        const int local_offset = __popc(mask & ((1u << lane) - 1u));
        if (valid && offset + local_offset < intersection_capacity) {
          uint64_t key = static_cast<uint64_t>(
              y * static_cast<int>(grid.x) + x) << 32;
          key |= static_cast<uint64_t>(__float_as_uint(source_depth));
          keys_unsorted[offset + local_offset] = key;
          values_unsorted[offset + local_offset] =
              static_cast<uint32_t>(source_gaussian);
        }
        if (lane == source_lane) {
          gaussian_emitted = true;
        }
      }
    }
  }

  if (gaussian_emitted && gaussian < count) {
    points_xy[gaussian] = point_xy;
    conic_opacity[gaussian] = {conic.x, conic.y, conic.z, opacity};
    const glm::vec3 color = colors[gaussian];
    rgb_depth[gaussian] = {color.x, color.y, color.z, depth};
    atomicAdd(
        reinterpret_cast<unsigned long long*>(counters),
        static_cast<unsigned long long>(1));
  }
}

}  // namespace

void precompute_covariances(
    int64_t count,
    const float* scales,
    const float* rotations,
    float* covariances,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int64_t blocks = (count + threads - 1) / threads;
  precompute_covariances_kernel<<<
      static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      count, scales, rotations, covariances);
}

void preprocess(
    int count,
    const glm::vec3* positions,
    const cov3d_t* covariances,
    const float* opacities,
    const glm::vec3* colors,
    const float* viewmat,
    const float* intrinsics,
    int width,
    int height,
    int block_x,
    int block_y,
    float near_plane,
    float far_plane,
    float gaussian_support_sigma,
    float covariance_epsilon,
    float2* points_xy,
    float4* rgb_depth,
    float4* conic_opacity,
    uint64_t* keys_unsorted,
    uint32_t* values_unsorted,
    int64_t intersection_capacity,
    int64_t* counters,
    cudaStream_t stream) {
  const dim3 grid(
      (width + block_x - 1) / block_x,
      (height + block_y - 1) / block_y,
      1);
  preprocess_kernel<<<
      (count + 127) / 128,
      dim3(8, 4, 4),
      0,
      stream>>>(
      count,
      positions,
      covariances,
      opacities,
      colors,
      viewmat,
      intrinsics,
      width,
      height,
      block_x,
      block_y,
      near_plane,
      far_plane,
      gaussian_support_sigma,
      covariance_epsilon,
      points_xy,
      rgb_depth,
      conic_opacity,
      keys_unsorted,
      values_unsorted,
      intersection_capacity,
      counters,
      grid);
}

}  // namespace flashgs_adapter
