#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "ops.h"

namespace flashgs_debug {

constexpr int kFloatFieldCount = 60;
constexpr int kIntFieldCount = 39;

void score_projected_contributors(
    int count,
    const float* positions,
    const flashgs_adapter::cov3d_t* covariances,
    const float* opacities,
    const int64_t* semantic_ids,
    const float* view,
    const float* intrinsic,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float support_sigma,
    float covariance_epsilon,
    int target_pixel_x,
    int target_pixel_y,
    int64_t expected_semantic_id,
    bool require_old_predicate_rejection,
    float* scores,
    cudaStream_t stream = 0);

void trace_candidates(
    int count,
    const float* positions,
    const flashgs_adapter::cov3d_t* covariances,
    const float* opacities,
    const float* view,
    const float* intrinsic,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float support_sigma,
    float covariance_epsilon,
    int target_pixel_x,
    int target_pixel_y,
    const uint64_t* keys_unsorted,
    const uint32_t* values_unsorted,
    const uint64_t* keys_sorted,
    const uint32_t* values_sorted,
    const int2* ranges,
    const float2* points_xy,
    const float4* conic_opacity,
    int capacity,
    const int64_t* candidate_ids,
    int candidate_count,
    float* float_trace,
    int64_t* int_trace,
    cudaStream_t stream = 0);

}  // namespace flashgs_debug
