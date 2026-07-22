#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

void check_tensor_group(
    const std::vector<torch::Tensor>& tensors,
    const char* group_name,
    std::size_t expected_size) {
  TORCH_CHECK(
      tensors.size() == expected_size,
      group_name,
      " must contain ",
      expected_size,
      " tensors, got ",
      tensors.size());
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto& tensor = tensors[index];
    TORCH_CHECK(tensor.is_cuda(), group_name, "[", index, "] must be CUDA-resident");
    TORCH_CHECK(tensor.is_contiguous(), group_name, "[", index, "] must be contiguous");
  }
}

}  // namespace

int64_t sort_temp_bytes_cuda(int64_t num_items, int64_t num_segments);
int64_t radix_sort_double_buffer_temp_bytes_cuda(
    int64_t num_items,
    int64_t num_segments);
void precompute_covariances_cuda(
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    torch::Tensor covariances);
void prepare_chunked_active_ids_cuda(
    const torch::Tensor& active_camera_ids,
    torch::Tensor active_mask,
    torch::Tensor chunk_active_ids,
    int64_t logical_batch,
    int64_t physical_batch);
void aggregate_chunk_counters_cuda(
    const torch::Tensor& chunk_counters,
    torch::Tensor counters,
    torch::Tensor physical_capacity_counters,
    int64_t chunk_count);

int64_t render_cuda(
    const std::vector<torch::Tensor>& scene,
    const std::vector<torch::Tensor>& cameras,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<torch::Tensor>& workspace,
    int64_t height,
    int64_t width,
    int64_t max_scene_gaussians,
    double near_plane,
    double far_plane,
    double gaussian_support_sigma,
    double covariance_epsilon,
    bool antialiased,
    double semantic_min_alpha,
    bool ray_gaussian_evaluation,
    int64_t tile_size,
    int64_t depth_bucket_count,
    int64_t depth_bucket_group_size,
    bool compact_projection_cache,
    bool materialize_projected_records,
    bool reuse_projection,
    bool output_srgb,
    bool deterministic,
    bool full_sensor_output,
    bool fixed_capacity_sort);

int64_t sort_temp_bytes(int64_t num_items, int64_t num_segments) {
  TORCH_CHECK(num_items > 0, "num_items must be positive");
  TORCH_CHECK(num_segments > 0, "num_segments must be positive");
  return sort_temp_bytes_cuda(num_items, num_segments);
}

int64_t radix_sort_double_buffer_temp_bytes(
    int64_t num_items,
    int64_t num_segments) {
  TORCH_CHECK(num_items > 0, "num_items must be positive");
  TORCH_CHECK(num_segments > 0, "num_segments must be positive");
  return radix_sort_double_buffer_temp_bytes_cuda(num_items, num_segments);
}

void precompute_covariances(
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    torch::Tensor covariances) {
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA-resident");
  TORCH_CHECK(rotations.is_cuda(), "rotations must be CUDA-resident");
  TORCH_CHECK(covariances.is_cuda(), "covariances must be CUDA-resident");
  TORCH_CHECK(
      scales.device() == rotations.device() &&
          scales.device() == covariances.device(),
      "scales, rotations, and covariances must share a CUDA device");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(rotations.is_contiguous(), "rotations must be contiguous");
  TORCH_CHECK(covariances.is_contiguous(), "covariances must be contiguous");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(
      rotations.scalar_type() == torch::kFloat32,
      "rotations must be float32");
  TORCH_CHECK(
      covariances.scalar_type() == torch::kFloat32,
      "covariances must be float32");
  TORCH_CHECK(
      scales.dim() == 2 && scales.size(1) == 3,
      "scales must have shape [G, 3]");
  TORCH_CHECK(
      rotations.dim() == 2 && rotations.size(1) == 4,
      "rotations must have shape [G, 4]");
  TORCH_CHECK(
      covariances.dim() == 2 && covariances.size(1) == 6,
      "covariances must have shape [G, 6]");
  TORCH_CHECK(
      scales.size(0) == rotations.size(0) &&
          scales.size(0) == covariances.size(0),
      "scales, rotations, and covariances must share G");
  precompute_covariances_cuda(scales, rotations, covariances);
}

void prepare_chunked_active_ids(
    const torch::Tensor& active_camera_ids,
    torch::Tensor active_mask,
    torch::Tensor chunk_active_ids,
    int64_t logical_batch,
    int64_t physical_batch) {
  TORCH_CHECK(
      active_camera_ids.is_cuda() && active_mask.is_cuda() &&
          chunk_active_ids.is_cuda(),
      "chunked active-ID tensors must be CUDA-resident");
  TORCH_CHECK(
      active_camera_ids.is_contiguous() && active_mask.is_contiguous() &&
          chunk_active_ids.is_contiguous(),
      "chunked active-ID tensors must be contiguous");
  TORCH_CHECK(
      active_camera_ids.device() == active_mask.device() &&
          active_camera_ids.device() == chunk_active_ids.device(),
      "chunked active-ID tensors must share a device");
  TORCH_CHECK(
      active_camera_ids.scalar_type() == torch::kInt64,
      "active_camera_ids must be int64");
  TORCH_CHECK(
      active_mask.scalar_type() == torch::kInt32,
      "active_mask must be int32");
  TORCH_CHECK(
      chunk_active_ids.scalar_type() == torch::kInt64,
      "chunk_active_ids must be int64");
  TORCH_CHECK(
      active_camera_ids.dim() == 1 && active_mask.dim() == 1 &&
          chunk_active_ids.dim() == 2,
      "invalid chunked active-ID tensor ranks");
  TORCH_CHECK(
      logical_batch > 0 && physical_batch > 0,
      "logical and physical batches must be positive");
  const int64_t chunk_count =
      (logical_batch + physical_batch - 1) / physical_batch;
  TORCH_CHECK(
      active_mask.numel() >= logical_batch,
      "active_mask is too small for the logical batch");
  TORCH_CHECK(
      chunk_active_ids.size(0) >= chunk_count &&
          chunk_active_ids.size(1) >= physical_batch,
      "chunk_active_ids is too small for the chunk schedule");
  prepare_chunked_active_ids_cuda(
      active_camera_ids,
      active_mask,
      chunk_active_ids,
      logical_batch,
      physical_batch);
}

void aggregate_chunk_counters(
    const torch::Tensor& chunk_counters,
    torch::Tensor counters,
    torch::Tensor physical_capacity_counters,
    int64_t chunk_count) {
  TORCH_CHECK(
      chunk_counters.is_cuda() && counters.is_cuda() &&
          physical_capacity_counters.is_cuda(),
      "chunk counters must be CUDA-resident");
  TORCH_CHECK(
      chunk_counters.is_contiguous() && counters.is_contiguous() &&
          physical_capacity_counters.is_contiguous(),
      "chunk counters must be contiguous");
  TORCH_CHECK(
      chunk_counters.device() == counters.device() &&
          chunk_counters.device() == physical_capacity_counters.device(),
      "chunk counters must share a device");
  TORCH_CHECK(
      chunk_counters.scalar_type() == torch::kInt64 &&
          counters.scalar_type() == torch::kInt64 &&
          physical_capacity_counters.scalar_type() == torch::kInt64,
      "chunk counters must be int64");
  TORCH_CHECK(
      chunk_counters.dim() == 2 && chunk_counters.size(1) >= 5 &&
          counters.numel() >= 5 &&
          physical_capacity_counters.numel() >= 5,
      "chunk counters must have shape [N, >=5] and outputs >=5");
  TORCH_CHECK(
      chunk_count > 0 && chunk_count <= chunk_counters.size(0),
      "chunk_count exceeds the counter workspace");
  aggregate_chunk_counters_cuda(
      chunk_counters,
      counters,
      physical_capacity_counters,
      chunk_count);
}

int64_t render(
    const std::vector<torch::Tensor>& scene,
    const std::vector<torch::Tensor>& cameras,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<torch::Tensor>& workspace,
    int64_t height,
    int64_t width,
    int64_t max_scene_gaussians,
    double near_plane,
    double far_plane,
    double gaussian_support_sigma,
    double covariance_epsilon,
    bool antialiased,
    double semantic_min_alpha,
    bool ray_gaussian_evaluation,
    int64_t tile_size,
    int64_t depth_bucket_count,
    int64_t depth_bucket_group_size,
    bool compact_projection_cache,
    bool materialize_projected_records,
    bool reuse_projection,
    bool output_srgb,
    bool deterministic,
    bool full_sensor_output,
    bool fixed_capacity_sort) {
  check_tensor_group(scene, "scene", 7);
  check_tensor_group(cameras, "cameras", 5);
  check_tensor_group(outputs, "outputs", 4);
  check_tensor_group(workspace, "workspace", 24);

  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  TORCH_CHECK(max_scene_gaussians > 0, "max_scene_gaussians must be positive");
  TORCH_CHECK(near_plane > 0.0, "near_plane must be positive");
  TORCH_CHECK(far_plane > near_plane, "far_plane must be greater than near_plane");
  TORCH_CHECK(
      gaussian_support_sigma > 0.0,
      "gaussian_support_sigma must be positive");
  TORCH_CHECK(
      std::isfinite(covariance_epsilon) && covariance_epsilon >= 0.0,
      "covariance_epsilon must be finite and non-negative");
  TORCH_CHECK(
      std::isfinite(semantic_min_alpha) &&
          semantic_min_alpha >= 0.0 &&
          semantic_min_alpha <= 1.0,
      "semantic_min_alpha must be finite and in [0, 1]");
  TORCH_CHECK(
      tile_size > 0 && tile_size <= 16 && (tile_size & (tile_size - 1)) == 0,
      "tile_size must be a power of two in [1, 16]");
  TORCH_CHECK(depth_bucket_count > 0, "depth_bucket_count must be positive");
  TORCH_CHECK(
      depth_bucket_group_size > 0,
      "depth_bucket_group_size must be positive");
  TORCH_CHECK(
      !compact_projection_cache || tile_size == 1,
      "compact_projection_cache requires tile_size=1");
  TORCH_CHECK(
      !compact_projection_cache || !ray_gaussian_evaluation,
      "compact_projection_cache supports only screen-space evaluation");
  TORCH_CHECK(
      !antialiased || !ray_gaussian_evaluation,
      "antialiased rasterization is incompatible with exact-ray evaluation");
  TORCH_CHECK(
      !compact_projection_cache || !deterministic,
      "compact_projection_cache currently requires deterministic=false");
  TORCH_CHECK(
      !materialize_projected_records || compact_projection_cache,
      "materialize_projected_records requires compact_projection_cache");
  TORCH_CHECK(
      !fixed_capacity_sort || !deterministic,
      "fixed_capacity_sort applies only to the nondeterministic path");

  return render_cuda(
      scene,
      cameras,
      outputs,
      workspace,
      height,
      width,
      max_scene_gaussians,
      near_plane,
      far_plane,
      gaussian_support_sigma,
      covariance_epsilon,
      antialiased,
      semantic_min_alpha,
      ray_gaussian_evaluation,
      tile_size,
      depth_bucket_count,
      depth_bucket_group_size,
      compact_projection_cache,
      materialize_projected_records,
      reuse_projection,
      output_srgb,
      deterministic,
      full_sensor_output,
      fixed_capacity_sort);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "sort_temp_bytes",
      &sort_temp_bytes,
      "Return reusable CUB global-radix, segmented-radix, and scan storage bytes");
  module.def(
      "radix_sort_double_buffer_temp_bytes",
      &radix_sort_double_buffer_temp_bytes,
      "Return bounded-key CUB global-radix storage bytes when caller buffers provide ping-pong storage");
  module.def(
      "precompute_covariances",
      &precompute_covariances,
      "Precompute object-space covariance tensors");
  module.def(
      "prepare_chunked_active_ids",
      &prepare_chunked_active_ids,
      "Prepare padded chunk-local active camera IDs on the CUDA stream");
  module.def(
      "aggregate_chunk_counters",
      &aggregate_chunk_counters,
      "Reduce physical-chunk counters into logical-batch totals");
  module.def(
      "render",
      &render,
      "Run project-owned batched Gaussian projection, ordering, and compositing");
}
