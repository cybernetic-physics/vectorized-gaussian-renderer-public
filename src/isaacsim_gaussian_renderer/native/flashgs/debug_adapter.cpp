#include "debug_ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA-resident");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_dtype(
    const torch::Tensor& tensor,
    torch::ScalarType expected,
    const char* name) {
  TORCH_CHECK(
      tensor.scalar_type() == expected,
      name,
      " has dtype ",
      tensor.scalar_type(),
      ", expected ",
      expected);
}

void check_scene_and_camera(
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    const torch::Tensor& opacities,
    const torch::Tensor& viewmat,
    const torch::Tensor& intrinsics) {
  for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
           {&means, "means"},
           {&covariances, "covariances"},
           {&opacities, "opacities"},
           {&viewmat, "viewmat"},
           {&intrinsics, "intrinsics"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
    check_dtype(*item.first, torch::kFloat32, item.second);
    TORCH_CHECK(
        item.first->get_device() == means.get_device(),
        item.second,
        " must share the means device");
  }
  TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be [G, 3]");
  TORCH_CHECK(
      covariances.dim() == 2 && covariances.size(0) == means.size(0) &&
          covariances.size(1) == 6,
      "covariances must be [G, 6]");
  TORCH_CHECK(opacities.numel() == means.size(0), "opacities must be [G]");
  TORCH_CHECK(viewmat.numel() == 16, "viewmat must contain one 4x4 camera");
  TORCH_CHECK(intrinsics.numel() == 9, "intrinsics must contain one 3x3 camera");
  TORCH_CHECK(
      means.size(0) > 0 && means.size(0) <= INT_MAX,
      "Gaussian count must fit positive int32");
}

void check_render_contract(
    int64_t width,
    int64_t height,
    double near_plane,
    double far_plane,
    double support_sigma,
    double covariance_epsilon,
    int64_t target_pixel_x,
    int64_t target_pixel_y) {
  TORCH_CHECK(width > 0 && width <= INT_MAX, "width must fit positive int32");
  TORCH_CHECK(height > 0 && height <= INT_MAX, "height must fit positive int32");
  TORCH_CHECK(near_plane > 0.0 && far_plane > near_plane, "invalid near/far planes");
  TORCH_CHECK(support_sigma > 0.0, "support_sigma must be positive");
  TORCH_CHECK(covariance_epsilon >= 0.0, "covariance_epsilon must be non-negative");
  TORCH_CHECK(
      target_pixel_x >= 0 && target_pixel_x < width &&
          target_pixel_y >= 0 && target_pixel_y < height,
      "target pixel lies outside the image");
}

}  // namespace

torch::Tensor score_projected_contributors(
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    const torch::Tensor& opacities,
    const torch::Tensor& semantic_ids,
    const torch::Tensor& viewmat,
    const torch::Tensor& intrinsics,
    int64_t height,
    int64_t width,
    double near_plane,
    double far_plane,
    double support_sigma,
    double covariance_epsilon,
    int64_t target_pixel_x,
    int64_t target_pixel_y,
    int64_t expected_semantic_id,
    bool require_old_predicate_rejection) {
  check_scene_and_camera(means, covariances, opacities, viewmat, intrinsics);
  check_cuda_contiguous(semantic_ids, "semantic_ids");
  check_dtype(semantic_ids, torch::kInt64, "semantic_ids");
  TORCH_CHECK(
      semantic_ids.get_device() == means.get_device(),
      "semantic_ids must share the means device");
  TORCH_CHECK(semantic_ids.numel() == means.size(0), "semantic_ids must be [G]");
  check_render_contract(
      width,
      height,
      near_plane,
      far_plane,
      support_sigma,
      covariance_epsilon,
      target_pixel_x,
      target_pixel_y);
  c10::cuda::CUDAGuard guard(means.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(means.get_device());
  auto scores = torch::full(
      {means.size(0)},
      -std::numeric_limits<float>::infinity(),
      means.options());
  flashgs_debug::score_projected_contributors(
      static_cast<int>(means.size(0)),
      means.data_ptr<float>(),
      reinterpret_cast<const flashgs_adapter::cov3d_t*>(
          covariances.data_ptr<float>()),
      opacities.data_ptr<float>(),
      semantic_ids.data_ptr<int64_t>(),
      viewmat.data_ptr<float>(),
      intrinsics.data_ptr<float>(),
      static_cast<int>(width),
      static_cast<int>(height),
      static_cast<float>(near_plane),
      static_cast<float>(far_plane),
      static_cast<float>(support_sigma),
      static_cast<float>(covariance_epsilon),
      static_cast<int>(target_pixel_x),
      static_cast<int>(target_pixel_y),
      expected_semantic_id,
      require_old_predicate_rejection,
      scores.data_ptr<float>(),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

std::vector<torch::Tensor> trace_candidates(
    const torch::Tensor& means,
    const torch::Tensor& covariances,
    const torch::Tensor& opacities,
    const torch::Tensor& viewmat,
    const torch::Tensor& intrinsics,
    const torch::Tensor& keys_unsorted,
    const torch::Tensor& values_unsorted,
    const torch::Tensor& keys_sorted,
    const torch::Tensor& values_sorted,
    const torch::Tensor& ranges,
    const torch::Tensor& points_xy,
    const torch::Tensor& conic_opacity,
    const torch::Tensor& candidate_ids,
    int64_t height,
    int64_t width,
    double near_plane,
    double far_plane,
    double support_sigma,
    double covariance_epsilon,
    int64_t target_pixel_x,
    int64_t target_pixel_y) {
  check_scene_and_camera(means, covariances, opacities, viewmat, intrinsics);
  check_render_contract(
      width,
      height,
      near_plane,
      far_plane,
      support_sigma,
      covariance_epsilon,
      target_pixel_x,
      target_pixel_y);
  for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
           {&keys_unsorted, "keys_unsorted"},
           {&values_unsorted, "values_unsorted"},
           {&keys_sorted, "keys_sorted"},
           {&values_sorted, "values_sorted"},
           {&ranges, "ranges"},
           {&points_xy, "points_xy"},
           {&conic_opacity, "conic_opacity"},
           {&candidate_ids, "candidate_ids"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
    TORCH_CHECK(
        item.first->get_device() == means.get_device(),
        item.second,
        " must share the means device");
  }
  check_dtype(keys_unsorted, torch::kUInt64, "keys_unsorted");
  check_dtype(values_unsorted, torch::kInt32, "values_unsorted");
  check_dtype(keys_sorted, torch::kUInt64, "keys_sorted");
  check_dtype(values_sorted, torch::kInt32, "values_sorted");
  check_dtype(ranges, torch::kInt32, "ranges");
  check_dtype(points_xy, torch::kFloat32, "points_xy");
  check_dtype(conic_opacity, torch::kFloat32, "conic_opacity");
  check_dtype(candidate_ids, torch::kInt64, "candidate_ids");
  const int64_t capacity = keys_unsorted.numel();
  TORCH_CHECK(
      capacity >= 0 && capacity <= INT_MAX,
      "capacity must fit non-negative int32");
  TORCH_CHECK(values_unsorted.numel() >= capacity, "values_unsorted is too small");
  TORCH_CHECK(keys_sorted.numel() >= capacity, "keys_sorted is too small");
  TORCH_CHECK(values_sorted.numel() >= capacity, "values_sorted is too small");
  const int64_t tile_count = ((width + 15) / 16) * ((height + 15) / 16);
  TORCH_CHECK(ranges.numel() >= tile_count * 2, "ranges is too small");
  TORCH_CHECK(points_xy.numel() >= means.size(0) * 2, "points_xy is too small");
  TORCH_CHECK(
      conic_opacity.numel() >= means.size(0) * 4,
      "conic_opacity is too small");
  TORCH_CHECK(
      candidate_ids.numel() > 0 && candidate_ids.numel() <= INT_MAX,
      "candidate_ids must fit positive int32");

  c10::cuda::CUDAGuard guard(means.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(means.get_device());
  auto float_trace = torch::empty(
      {candidate_ids.numel(), flashgs_debug::kFloatFieldCount},
      means.options());
  auto int_trace = torch::empty(
      {candidate_ids.numel(), flashgs_debug::kIntFieldCount},
      means.options().dtype(torch::kInt64));
  flashgs_debug::trace_candidates(
      static_cast<int>(means.size(0)),
      means.data_ptr<float>(),
      reinterpret_cast<const flashgs_adapter::cov3d_t*>(
          covariances.data_ptr<float>()),
      opacities.data_ptr<float>(),
      viewmat.data_ptr<float>(),
      intrinsics.data_ptr<float>(),
      static_cast<int>(width),
      static_cast<int>(height),
      static_cast<float>(near_plane),
      static_cast<float>(far_plane),
      static_cast<float>(support_sigma),
      static_cast<float>(covariance_epsilon),
      static_cast<int>(target_pixel_x),
      static_cast<int>(target_pixel_y),
      keys_unsorted.data_ptr<uint64_t>(),
      reinterpret_cast<const uint32_t*>(values_unsorted.data_ptr<int32_t>()),
      keys_sorted.data_ptr<uint64_t>(),
      reinterpret_cast<const uint32_t*>(values_sorted.data_ptr<int32_t>()),
      reinterpret_cast<const int2*>(ranges.data_ptr<int32_t>()),
      reinterpret_cast<const float2*>(points_xy.data_ptr<float>()),
      reinterpret_cast<const float4*>(conic_opacity.data_ptr<float>()),
      static_cast<int>(capacity),
      candidate_ids.data_ptr<int64_t>(),
      static_cast<int>(candidate_ids.numel()),
      float_trace.data_ptr<float>(),
      int_trace.data_ptr<int64_t>(),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {float_trace, int_trace};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "score_projected_contributors",
      &score_projected_contributors,
      "Score every projected target-pixel contributor by individual alpha");
  module.def(
      "trace_candidates",
      &trace_candidates,
      "Trace exact Gaussian IDs through projection, tiles, sort, range, and compositor");
}
