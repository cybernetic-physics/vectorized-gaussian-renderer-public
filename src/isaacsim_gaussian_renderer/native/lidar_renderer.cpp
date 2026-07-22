#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA-resident");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_group(
    const std::vector<torch::Tensor>& tensors,
    const char* name,
    std::size_t size) {
  TORCH_CHECK(tensors.size() == size, name, " must contain ", size, " tensors");
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    check_cuda_contiguous(tensors[index], name);
  }
}

}  // namespace

int64_t lidar_sort_temp_bytes_cuda(int64_t num_items);
void build_scene_lbvh_cuda(
    const std::vector<torch::Tensor>& scene,
    const std::vector<torch::Tensor>& build,
    int64_t packet_size,
    int64_t leaf_count,
    int64_t leaf_capacity,
    double support_sigma,
    double planarity_ratio_max);
void pack_scene_descriptor_cuda(
    torch::Tensor descriptors,
    int64_t slot,
    int64_t scene_id,
    const std::vector<torch::Tensor>& scene,
    int64_t leaf_count,
    int64_t leaf_capacity);
void render_lidar_cuda(
    torch::Tensor descriptors,
    int64_t scene_count,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs,
    torch::Tensor counters,
    int64_t returns,
    int64_t packet_size,
    double near_plane_m,
    double far_plane_m,
    double support_sigma,
    double detection_threshold,
    double planarity_ratio_max,
    double min_incidence_cos,
    double cluster_abs_m,
    double cluster_relative,
    double fallback_reflectivity,
    double direction_norm_tolerance,
    int64_t semantic_slots);

int64_t sort_temp_bytes(int64_t num_items) {
  TORCH_CHECK(num_items > 0, "num_items must be positive");
  return lidar_sort_temp_bytes_cuda(num_items);
}

void build_scene_lbvh(
    const std::vector<torch::Tensor>& scene,
    const std::vector<torch::Tensor>& build,
    int64_t packet_size,
    int64_t leaf_count,
    int64_t leaf_capacity,
    double support_sigma,
    double planarity_ratio_max) {
  check_group(scene, "scene", 3);
  check_group(build, "build", 7);
  TORCH_CHECK(packet_size == 8 || packet_size == 16, "packet_size must be 8 or 16");
  TORCH_CHECK(leaf_count > 0 && leaf_capacity >= leaf_count, "invalid leaf counts");
  TORCH_CHECK(
      (leaf_capacity & (leaf_capacity - 1)) == 0,
      "leaf_capacity must be a power of two");
  TORCH_CHECK(support_sigma > 0.0, "support_sigma must be positive");
  TORCH_CHECK(
      planarity_ratio_max > 0.0 && planarity_ratio_max <= 1.0,
      "planarity_ratio_max must be in (0, 1]");
  const int64_t gaussian_count = scene[0].size(0);
  TORCH_CHECK(scene[0].scalar_type() == torch::kFloat32, "means must be float32");
  TORCH_CHECK(scene[1].scalar_type() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(scene[2].scalar_type() == torch::kFloat32, "rotations must be float32");
  TORCH_CHECK(scene[0].sizes() == torch::IntArrayRef({gaussian_count, 3}), "means shape");
  TORCH_CHECK(scene[1].sizes() == torch::IntArrayRef({gaussian_count, 3}), "scales shape");
  TORCH_CHECK(scene[2].sizes() == torch::IntArrayRef({gaussian_count, 4}), "rotations shape");
  TORCH_CHECK(build[0].numel() == gaussian_count, "keys_in size");
  TORCH_CHECK(build[1].numel() == gaussian_count, "keys_out size");
  TORCH_CHECK(build[2].numel() == gaussian_count, "indices_in size");
  TORCH_CHECK(build[3].numel() == gaussian_count, "sorted_indices size");
  TORCH_CHECK(build[5].sizes() == torch::IntArrayRef({2, 3}), "scene_bounds shape");
  TORCH_CHECK(
      build[6].sizes() == torch::IntArrayRef({2 * leaf_capacity - 1, 6}),
      "node_bounds shape");
  build_scene_lbvh_cuda(
      scene,
      build,
      packet_size,
      leaf_count,
      leaf_capacity,
      support_sigma,
      planarity_ratio_max);
}

void pack_scene_descriptor(
    torch::Tensor descriptors,
    int64_t slot,
    int64_t scene_id,
    const std::vector<torch::Tensor>& scene,
    int64_t leaf_count,
    int64_t leaf_capacity) {
  check_cuda_contiguous(descriptors, "descriptors");
  check_group(scene, "scene", 9);
  TORCH_CHECK(descriptors.scalar_type() == torch::kInt64, "descriptors must be int64");
  TORCH_CHECK(descriptors.dim() == 2 && descriptors.size(1) == 14, "descriptors shape");
  TORCH_CHECK(slot >= 0 && slot < descriptors.size(0), "descriptor slot out of range");
  TORCH_CHECK(leaf_count > 0 && leaf_capacity >= leaf_count, "invalid leaf metadata");
  pack_scene_descriptor_cuda(
      descriptors,
      slot,
      scene_id,
      scene,
      leaf_count,
      leaf_capacity);
}

void render_lidar(
    torch::Tensor descriptors,
    int64_t scene_count,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs,
    torch::Tensor counters,
    int64_t returns,
    int64_t packet_size,
    double near_plane_m,
    double far_plane_m,
    double support_sigma,
    double detection_threshold,
    double planarity_ratio_max,
    double min_incidence_cos,
    double cluster_abs_m,
    double cluster_relative,
    double fallback_reflectivity,
    double direction_norm_tolerance,
    int64_t semantic_slots) {
  check_cuda_contiguous(descriptors, "descriptors");
  check_cuda_contiguous(counters, "counters");
  check_group(inputs, "inputs", 6);
  check_group(outputs, "outputs", 7);
  TORCH_CHECK(scene_count > 0 && scene_count <= descriptors.size(0), "invalid scene_count");
  TORCH_CHECK(returns == 1 || returns == 2, "returns must be 1 or 2");
  TORCH_CHECK(packet_size == 8 || packet_size == 16, "packet_size must be 8 or 16");
  TORCH_CHECK(near_plane_m > 0.0 && far_plane_m > near_plane_m, "invalid range planes");
  TORCH_CHECK(support_sigma > 0.0, "support_sigma must be positive");
  TORCH_CHECK(detection_threshold > 0.0 && detection_threshold <= 1.0, "invalid threshold");
  TORCH_CHECK(planarity_ratio_max > 0.0 && planarity_ratio_max <= 1.0, "invalid planarity");
  TORCH_CHECK(min_incidence_cos > 0.0 && min_incidence_cos <= 1.0, "invalid incidence");
  TORCH_CHECK(cluster_abs_m >= 0.0 && cluster_relative >= 0.0, "invalid cluster");
  TORCH_CHECK(fallback_reflectivity >= 0.0 && fallback_reflectivity <= 1.0, "invalid reflectivity");
  TORCH_CHECK(
      direction_norm_tolerance > 0.0 && direction_norm_tolerance < 1.0,
      "direction tolerance must be in (0, 1)");
  TORCH_CHECK(semantic_slots > 0 && semantic_slots <= 32, "semantic_slots must be in [1, 32]");
  TORCH_CHECK(counters.scalar_type() == torch::kInt64 && counters.numel() == 12, "counters shape");

  const int64_t rays = inputs[0].size(0);
  const int64_t batch = inputs[2].size(0);
  TORCH_CHECK(inputs[0].sizes() == torch::IntArrayRef({rays, 3}), "ray directions shape");
  TORCH_CHECK(inputs[0].scalar_type() == torch::kFloat32, "ray directions dtype");
  TORCH_CHECK(inputs[1].size(0) == rays, "time offsets shape");
  TORCH_CHECK(
      inputs[1].scalar_type() == torch::kInt32 || inputs[1].scalar_type() == torch::kInt64,
      "time offsets dtype");
  TORCH_CHECK(inputs[2].sizes() == torch::IntArrayRef({batch, 4, 4}), "sensor transforms shape");
  TORCH_CHECK(inputs[3].sizes() == torch::IntArrayRef({batch, 4, 4}), "scene transforms shape");
  TORCH_CHECK(inputs[4].sizes() == torch::IntArrayRef({batch}), "scene IDs shape");
  TORCH_CHECK(inputs[5].dim() == 1, "active IDs shape");
  TORCH_CHECK(outputs[0].sizes() == torch::IntArrayRef({batch, rays, returns}), "range shape");
  TORCH_CHECK(outputs[1].sizes() == torch::IntArrayRef({batch, rays, returns, 3}), "position shape");
  TORCH_CHECK(outputs[2].sizes() == torch::IntArrayRef({batch, rays, returns}), "intensity shape");
  TORCH_CHECK(outputs[3].sizes() == torch::IntArrayRef({batch, rays, returns}), "semantic shape");
  TORCH_CHECK(outputs[4].sizes() == torch::IntArrayRef({batch, rays, returns}), "valid shape");
  TORCH_CHECK(outputs[5].sizes() == torch::IntArrayRef({batch, rays, returns}), "time output shape");
  TORCH_CHECK(outputs[6].sizes() == torch::IntArrayRef({batch, rays}), "count shape");
  render_lidar_cuda(
      descriptors,
      scene_count,
      inputs,
      outputs,
      counters,
      returns,
      packet_size,
      near_plane_m,
      far_plane_m,
      support_sigma,
      detection_threshold,
      planarity_ratio_max,
      min_incidence_cos,
      cluster_abs_m,
      cluster_relative,
      fallback_reflectivity,
      direction_norm_tolerance,
      semantic_slots);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("sort_temp_bytes", &sort_temp_bytes, "Return LiDAR Morton-sort temp bytes");
  module.def("build_scene_lbvh", &build_scene_lbvh, "Build a packet LBVH on the current stream");
  module.def("pack_scene_descriptor", &pack_scene_descriptor, "Pack one scene pointer descriptor");
  module.def("render_lidar", &render_lidar, "Trace one complete Gaussian LiDAR invocation");
}
