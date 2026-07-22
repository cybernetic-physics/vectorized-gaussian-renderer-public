#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <vector>

namespace {

void check_tensor_group(
    const std::vector<torch::Tensor>& tensors,
    const char* name,
    std::size_t expected_size) {
  TORCH_CHECK(
      tensors.size() == expected_size,
      name,
      " must contain ",
      expected_size,
      " tensors, got ",
      tensors.size());
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    TORCH_CHECK(tensors[index].is_cuda(), name, "[", index, "] must be CUDA-resident");
    TORCH_CHECK(tensors[index].is_contiguous(), name, "[", index, "] must be contiguous");
  }
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

enum SceneTensor : int {
  kMeans = 0,
  kCovariances = 1,
  kOpacities = 2,
  kColors = 3,
  kSemanticIds = 4,
};

enum CameraTensor : int {
  kViewmats = 0,
  kIntrinsics = 1,
  kSceneIds = 2,
  kActiveCameraIds = 3,
};

enum OutputTensor : int {
  kRgb = 0,
  kDepth = 1,
  kAlpha = 2,
  kSemantic = 3,
};

enum WorkspaceTensor : int {
  kPointsXY = 0,
  kRgbDepth = 1,
  kConicOpacity = 2,
  kKeysUnsorted = 3,
  kValuesUnsorted = 4,
  kKeysSorted = 5,
  kValuesSorted = 6,
  kSortTemp = 7,
  kRanges = 8,
  kCounters = 9,
};

int higher_msb(uint32_t value) {
  int bits = 0;
  do {
    ++bits;
    value >>= 1;
  } while (value != 0);
  return bits;
}

}  // namespace

int64_t get_sort_buffer_size(int64_t capacity) {
  TORCH_CHECK(capacity > 0 && capacity <= INT_MAX, "capacity must fit positive int32");
  return static_cast<int64_t>(
      flashgs_adapter::get_sort_buffer_size(static_cast<int>(capacity)));
}

void precompute_covariances(
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    torch::Tensor covariances) {
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA-resident");
  TORCH_CHECK(rotations.is_cuda(), "rotations must be CUDA-resident");
  TORCH_CHECK(covariances.is_cuda(), "covariances must be CUDA-resident");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(rotations.is_contiguous(), "rotations must be contiguous");
  TORCH_CHECK(covariances.is_contiguous(), "covariances must be contiguous");
  check_dtype(scales, torch::kFloat32, "scales");
  check_dtype(rotations, torch::kFloat32, "rotations");
  check_dtype(covariances, torch::kFloat32, "covariances");
  TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must have shape [G, 3]");
  TORCH_CHECK(rotations.dim() == 2 && rotations.size(1) == 4, "rotations must have shape [G, 4]");
  TORCH_CHECK(covariances.dim() == 2 && covariances.size(1) == 6, "covariances must have shape [G, 6]");
  TORCH_CHECK(
      scales.size(0) == rotations.size(0) && scales.size(0) == covariances.size(0),
      "scales, rotations, and covariances must share G");
  c10::cuda::CUDAGuard guard(scales.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(scales.get_device());
  flashgs_adapter::precompute_covariances(
      scales.size(0),
      scales.data_ptr<float>(),
      rotations.data_ptr<float>(),
      covariances.data_ptr<float>(),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void render_batch(
    const std::vector<torch::Tensor>& scene,
    const std::vector<torch::Tensor>& cameras,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<torch::Tensor>& workspace,
    int64_t height64,
    int64_t width64,
    double near_plane,
    double far_plane,
    double support_sigma,
    double covariance_epsilon,
    double semantic_min_alpha,
    bool full_sensor_output,
    int64_t registered_scene_id) {
  check_tensor_group(scene, "scene", 5);
  check_tensor_group(cameras, "cameras", 4);
  check_tensor_group(outputs, "outputs", 4);
  check_tensor_group(workspace, "workspace", 10);
  check_dtype(scene[kMeans], torch::kFloat32, "means");
  check_dtype(scene[kCovariances], torch::kFloat32, "covariances");
  check_dtype(scene[kOpacities], torch::kFloat32, "opacities");
  check_dtype(scene[kColors], torch::kFloat32, "colors");
  check_dtype(scene[kSemanticIds], torch::kInt64, "semantic_ids");
  check_dtype(cameras[kViewmats], torch::kFloat32, "viewmats");
  check_dtype(cameras[kIntrinsics], torch::kFloat32, "intrinsics");
  check_dtype(cameras[kSceneIds], torch::kInt64, "scene_ids");
  check_dtype(cameras[kActiveCameraIds], torch::kInt64, "active_camera_ids");
  check_dtype(outputs[kRgb], torch::kFloat32, "rgb");
  check_dtype(outputs[kDepth], torch::kFloat32, "depth");
  check_dtype(outputs[kAlpha], torch::kFloat32, "alpha");
  check_dtype(outputs[kSemantic], torch::kInt64, "semantic");
  check_dtype(workspace[kPointsXY], torch::kFloat32, "points_xy");
  check_dtype(workspace[kRgbDepth], torch::kFloat32, "rgb_depth");
  check_dtype(workspace[kConicOpacity], torch::kFloat32, "conic_opacity");
  check_dtype(workspace[kKeysUnsorted], torch::kUInt64, "keys_unsorted");
  check_dtype(workspace[kValuesUnsorted], torch::kInt32, "values_unsorted");
  check_dtype(workspace[kKeysSorted], torch::kUInt64, "keys_sorted");
  check_dtype(workspace[kValuesSorted], torch::kInt32, "values_sorted");
  check_dtype(workspace[kSortTemp], torch::kUInt8, "sort_temp");
  check_dtype(workspace[kRanges], torch::kInt32, "ranges");
  check_dtype(workspace[kCounters], torch::kInt64, "counters");

  TORCH_CHECK(height64 > 0 && height64 <= INT_MAX, "height must fit positive int32");
  TORCH_CHECK(width64 > 0 && width64 <= INT_MAX, "width must fit positive int32");
  TORCH_CHECK(near_plane > 0.0 && far_plane > near_plane, "invalid near/far planes");
  TORCH_CHECK(support_sigma > 0.0, "support_sigma must be positive");
  TORCH_CHECK(covariance_epsilon >= 0.0, "covariance_epsilon must be non-negative");
  TORCH_CHECK(
      semantic_min_alpha >= 0.0 && semantic_min_alpha <= 1.0,
      "semantic_min_alpha must be in [0, 1]");
  TORCH_CHECK(registered_scene_id >= 0, "registered_scene_id must be non-negative");
  const int height = static_cast<int>(height64);
  const int width = static_cast<int>(width64);
  const int64_t gaussian_count64 = scene[kMeans].size(0);
  TORCH_CHECK(
      gaussian_count64 > 0 && gaussian_count64 <= INT_MAX,
      "FlashGS Gaussian count must fit positive int32");
  const int gaussian_count = static_cast<int>(gaussian_count64);
  TORCH_CHECK(
      scene[kMeans].dim() == 2 && scene[kMeans].size(0) == gaussian_count64 &&
          scene[kMeans].size(1) == 3,
      "means must be [G, 3]");
  TORCH_CHECK(
      scene[kCovariances].dim() == 2 &&
          scene[kCovariances].size(0) == gaussian_count64 &&
          scene[kCovariances].size(1) == 6,
      "covariances must be [G, 6]");
  TORCH_CHECK(scene[kOpacities].numel() == gaussian_count64, "opacities must be [G]");
  TORCH_CHECK(
      scene[kColors].dim() == 2 && scene[kColors].size(0) == gaussian_count64 &&
          scene[kColors].size(1) == 3,
      "colors must be [G, 3]");
  TORCH_CHECK(scene[kSemanticIds].numel() == gaussian_count64, "semantic_ids must be [G]");
  const int64_t batch64 = cameras[kViewmats].size(0);
  TORCH_CHECK(batch64 > 0 && batch64 <= INT_MAX, "batch must fit positive int32");
  const int batch = static_cast<int>(batch64);
  TORCH_CHECK(
      cameras[kViewmats].dim() == 3 && cameras[kViewmats].size(0) == batch64 &&
          cameras[kViewmats].size(1) == 4 && cameras[kViewmats].size(2) == 4,
      "viewmats must be [B, 4, 4]");
  TORCH_CHECK(
      cameras[kIntrinsics].dim() == 3 && cameras[kIntrinsics].size(0) == batch64 &&
          cameras[kIntrinsics].size(1) == 3 && cameras[kIntrinsics].size(2) == 3,
      "intrinsics must be [B, 3, 3]");
  TORCH_CHECK(cameras[kSceneIds].numel() == batch64, "scene_ids must be [B]");
  TORCH_CHECK(cameras[kActiveCameraIds].numel() == 0, "the FlashGS-derived matched port does not implement active subsets");
  TORCH_CHECK(outputs[kRgb].numel() >= batch64 * height * width * 3, "RGB output is too small");
  if (full_sensor_output) {
    TORCH_CHECK(outputs[kDepth].numel() >= batch64 * height * width, "depth output is too small");
    TORCH_CHECK(outputs[kAlpha].numel() >= batch64 * height * width, "alpha output is too small");
    TORCH_CHECK(outputs[kSemantic].numel() >= batch64 * height * width, "semantic output is too small");
  }
  TORCH_CHECK(workspace[kPointsXY].numel() >= gaussian_count64 * 2, "points_xy is too small");
  TORCH_CHECK(workspace[kRgbDepth].numel() >= gaussian_count64 * 4, "rgb_depth is too small");
  TORCH_CHECK(workspace[kConicOpacity].numel() >= gaussian_count64 * 4, "conic_opacity is too small");
  const int64_t capacity64 = workspace[kKeysUnsorted].numel();
  TORCH_CHECK(capacity64 > 0 && capacity64 <= INT_MAX, "intersection capacity must fit int32");
  const int capacity = static_cast<int>(capacity64);
  TORCH_CHECK(workspace[kValuesUnsorted].numel() >= capacity64, "values_unsorted is too small");
  TORCH_CHECK(workspace[kKeysSorted].numel() >= capacity64, "keys_sorted is too small");
  TORCH_CHECK(workspace[kValuesSorted].numel() >= capacity64, "values_sorted is too small");
  const int tile_count = ((width + 15) / 16) * ((height + 15) / 16);
  const int sort_end_bit = 32 + higher_msb(static_cast<uint32_t>(tile_count));
  TORCH_CHECK(workspace[kRanges].numel() >= tile_count * 2, "ranges is too small");
  TORCH_CHECK(workspace[kCounters].numel() >= batch64 * 3, "counters must be [max_views, 3]");

  c10::cuda::CUDAGuard guard(scene[kMeans].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(scene[kMeans].get_device());
  const int64_t pixels = static_cast<int64_t>(height) * width;
  for (int camera = 0; camera < batch; ++camera) {
    int64_t* counters = workspace[kCounters].data_ptr<int64_t>() +
        static_cast<int64_t>(camera) * 3;
    C10_CUDA_CHECK(cudaMemsetAsync(counters, 0, 3 * sizeof(int64_t), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        workspace[kKeysUnsorted].data_ptr<uint64_t>(),
        0xff,
        static_cast<std::size_t>(capacity) * sizeof(uint64_t),
        stream));
    flashgs_adapter::preprocess(
        gaussian_count,
        reinterpret_cast<const glm::vec3*>(scene[kMeans].data_ptr<float>()),
        reinterpret_cast<const flashgs_adapter::cov3d_t*>(scene[kCovariances].data_ptr<float>()),
        scene[kOpacities].data_ptr<float>(),
        reinterpret_cast<const glm::vec3*>(scene[kColors].data_ptr<float>()),
        cameras[kViewmats].data_ptr<float>() + static_cast<int64_t>(camera) * 16,
        cameras[kIntrinsics].data_ptr<float>() + static_cast<int64_t>(camera) * 9,
        width,
        height,
        16,
        16,
        static_cast<float>(near_plane),
        static_cast<float>(far_plane),
        static_cast<float>(support_sigma),
        static_cast<float>(covariance_epsilon),
        reinterpret_cast<float2*>(workspace[kPointsXY].data_ptr<float>()),
        reinterpret_cast<float4*>(workspace[kRgbDepth].data_ptr<float>()),
        reinterpret_cast<float4*>(workspace[kConicOpacity].data_ptr<float>()),
        workspace[kKeysUnsorted].data_ptr<uint64_t>(),
        reinterpret_cast<uint32_t*>(workspace[kValuesUnsorted].data_ptr<int32_t>()),
        capacity64,
        counters,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    flashgs_adapter::sort_gaussian_fixed(
        capacity,
        reinterpret_cast<char*>(workspace[kSortTemp].data_ptr<uint8_t>()),
        static_cast<std::size_t>(workspace[kSortTemp].numel()),
        workspace[kKeysUnsorted].data_ptr<uint64_t>(),
        reinterpret_cast<uint32_t*>(workspace[kValuesUnsorted].data_ptr<int32_t>()),
        workspace[kKeysSorted].data_ptr<uint64_t>(),
        reinterpret_cast<uint32_t*>(workspace[kValuesSorted].data_ptr<int32_t>()),
        sort_end_bit,
        stream);
    flashgs_adapter::render_16x16_fixed(
        capacity,
        width,
        height,
        reinterpret_cast<const float2*>(workspace[kPointsXY].data_ptr<float>()),
        reinterpret_cast<const float4*>(workspace[kRgbDepth].data_ptr<float>()),
        reinterpret_cast<const float4*>(workspace[kConicOpacity].data_ptr<float>()),
        scene[kSemanticIds].data_ptr<int64_t>(),
        workspace[kKeysSorted].data_ptr<uint64_t>(),
        reinterpret_cast<const uint32_t*>(workspace[kValuesSorted].data_ptr<int32_t>()),
        reinterpret_cast<int2*>(workspace[kRanges].data_ptr<int32_t>()),
        static_cast<float>(semantic_min_alpha),
        full_sensor_output,
        outputs[kRgb].data_ptr<float>() + static_cast<int64_t>(camera) * pixels * 3,
        full_sensor_output
            ? outputs[kDepth].data_ptr<float>() + static_cast<int64_t>(camera) * pixels
            : outputs[kDepth].data_ptr<float>(),
        full_sensor_output
            ? outputs[kAlpha].data_ptr<float>() + static_cast<int64_t>(camera) * pixels
            : outputs[kAlpha].data_ptr<float>(),
        full_sensor_output
            ? outputs[kSemantic].data_ptr<int64_t>() + static_cast<int64_t>(camera) * pixels
            : outputs[kSemantic].data_ptr<int64_t>(),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "get_sort_buffer_size",
      &get_sort_buffer_size,
      "Return fixed-capacity FlashGS CUB sort storage bytes");
  module.def(
      "precompute_covariances",
      &precompute_covariances,
      "Precompute canonical object-space covariance tensors");
  module.def(
      "render_batch",
      &render_batch,
      "Execute the adapted unbatched FlashGS pipeline once per camera");
}
