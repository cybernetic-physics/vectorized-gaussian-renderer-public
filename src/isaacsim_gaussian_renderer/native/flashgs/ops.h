#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "glm/glm.hpp"

constexpr int FLASHGS_WARP_SIZE = 32;
constexpr float FLASHGS_ALPHA_THRESHOLD = 1.0f / 255.0f;
constexpr float FLASHGS_MAX_ALPHA = 0.99f;
constexpr float FLASHGS_TRANSMITTANCE_THRESHOLD = 1.0e-4f;

namespace flashgs_adapter {

union cov3d_t {
  float2 f2[3];
  float s[6];
};

void precompute_covariances(
    int64_t count,
    const float* scales,
    const float* rotations,
    float* covariances,
    cudaStream_t stream = 0);

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
    cudaStream_t stream = 0);

void sort_gaussian_fixed(
    int capacity,
    char* sorting_space,
    size_t sorting_size,
    uint64_t* keys_unsorted,
    uint32_t* values_unsorted,
    uint64_t* keys_sorted,
    uint32_t* values_sorted,
    int end_bit,
    cudaStream_t stream = 0);

size_t get_sort_buffer_size(int capacity, cudaStream_t stream = 0);

void render_16x16_fixed(
    int capacity,
    int width,
    int height,
    const float2* points_xy,
    const float4* rgb_depth,
    const float4* conic_opacity,
    const int64_t* semantic_ids,
    const uint64_t* keys_sorted,
    const uint32_t* values_sorted,
    int2* ranges,
    float semantic_min_alpha,
    bool full_sensor_output,
    float* out_rgb,
    float* out_depth,
    float* out_alpha,
    int64_t* out_semantic,
    cudaStream_t stream = 0);

}  // namespace flashgs_adapter
