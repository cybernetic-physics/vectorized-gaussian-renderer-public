#include "faithful_glue.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <vector>

namespace {

enum SceneTensor : int {
  kMeans = 0,
  kCovariances = 1,
  kOpacities = 2,
  kColors = 3,
};

enum CameraTensor : int {
  kViewmats = 0,
  kIntrinsics = 1,
  kSceneIds = 2,
  kActiveCameraIds = 3,
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
  kCameraState = 9,
  kRgbU8 = 10,
  kCounters = 11,
};

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

}  // namespace

int64_t get_sort_buffer_size(int64_t capacity) {
  TORCH_CHECK(capacity > 0 && capacity <= INT_MAX, "capacity must fit positive int32");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return static_cast<int64_t>(
      flashgs::get_sort_buffer_size(static_cast<int>(capacity), stream));
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
  TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must be [G, 3]");
  TORCH_CHECK(
      rotations.dim() == 2 && rotations.size(1) == 4,
      "rotations must be [G, 4]");
  TORCH_CHECK(
      covariances.dim() == 2 && covariances.size(1) == 6,
      "covariances must be [G, 6]");
  TORCH_CHECK(
      scales.size(0) == rotations.size(0) && scales.size(0) == covariances.size(0),
      "scales, rotations, and covariances must share G");
  TORCH_CHECK(scales.size(0) > 0 && scales.size(0) <= INT_MAX, "G must fit positive int32");
  c10::cuda::CUDAGuard guard(scales.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(scales.get_device());
  flashgs_upstream_faithful::precompute_covariances(
      static_cast<int>(scales.size(0)),
      scales.data_ptr<float>(),
      rotations.data_ptr<float>(),
      reinterpret_cast<flashgs::cov3d_t*>(covariances.data_ptr<float>()),
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
    int64_t registered_scene_id) {
  check_tensor_group(scene, "scene", 4);
  check_tensor_group(cameras, "cameras", 4);
  check_tensor_group(outputs, "outputs", 1);
  check_tensor_group(workspace, "workspace", 12);
  check_dtype(scene[kMeans], torch::kFloat32, "means");
  check_dtype(scene[kCovariances], torch::kFloat32, "covariances");
  check_dtype(scene[kOpacities], torch::kFloat32, "opacities");
  check_dtype(scene[kColors], torch::kFloat32, "colors");
  check_dtype(cameras[kViewmats], torch::kFloat32, "viewmats");
  check_dtype(cameras[kIntrinsics], torch::kFloat32, "intrinsics");
  check_dtype(cameras[kSceneIds], torch::kInt64, "scene_ids");
  check_dtype(cameras[kActiveCameraIds], torch::kInt64, "active_camera_ids");
  check_dtype(outputs[0], torch::kFloat32, "rgb");
  check_dtype(workspace[kPointsXY], torch::kFloat32, "points_xy");
  check_dtype(workspace[kRgbDepth], torch::kFloat32, "rgb_depth");
  check_dtype(workspace[kConicOpacity], torch::kFloat32, "conic_opacity");
  check_dtype(workspace[kKeysUnsorted], torch::kUInt64, "keys_unsorted");
  check_dtype(workspace[kValuesUnsorted], torch::kInt32, "values_unsorted");
  check_dtype(workspace[kKeysSorted], torch::kUInt64, "keys_sorted");
  check_dtype(workspace[kValuesSorted], torch::kInt32, "values_sorted");
  check_dtype(workspace[kSortTemp], torch::kUInt8, "sort_temp");
  check_dtype(workspace[kRanges], torch::kInt32, "ranges");
  check_dtype(workspace[kCameraState], torch::kFloat32, "camera_state");
  check_dtype(workspace[kRgbU8], torch::kUInt8, "rgb_u8");
  check_dtype(workspace[kCounters], torch::kInt32, "counters");

  TORCH_CHECK(height64 > 0 && height64 <= INT_MAX, "height must fit positive int32");
  TORCH_CHECK(width64 > 0 && width64 <= INT_MAX, "width must fit positive int32");
  TORCH_CHECK(
      height64 % 16 == 0 && width64 % 16 == 0,
      "exact upstream 16x16 compositor requires dimensions divisible by 16");
  TORCH_CHECK(near_plane > 0.0 && far_plane > near_plane, "invalid near/far planes");
  TORCH_CHECK(registered_scene_id >= 0, "registered_scene_id must be non-negative");
  const int height = static_cast<int>(height64);
  const int width = static_cast<int>(width64);
  const int64_t gaussian_count64 = scene[kMeans].size(0);
  TORCH_CHECK(
      gaussian_count64 > 0 && gaussian_count64 <= INT_MAX,
      "FlashGS Gaussian count must fit positive int32");
  const int gaussian_count = static_cast<int>(gaussian_count64);
  TORCH_CHECK(
      scene[kMeans].dim() == 2 && scene[kMeans].size(1) == 3,
      "means must be [G, 3]");
  TORCH_CHECK(
      scene[kCovariances].dim() == 2 && scene[kCovariances].size(0) == gaussian_count64 &&
          scene[kCovariances].size(1) == 6,
      "covariances must be [G, 6]");
  TORCH_CHECK(scene[kOpacities].numel() == gaussian_count64, "opacities must be [G]");
  TORCH_CHECK(
      scene[kColors].dim() == 2 && scene[kColors].size(0) == gaussian_count64 &&
          scene[kColors].size(1) == 3,
      "degree-zero colors must be [G, 3]");

  const int64_t batch64 = cameras[kViewmats].size(0);
  TORCH_CHECK(batch64 > 0 && batch64 <= INT_MAX, "batch must fit positive int32");
  const int batch = static_cast<int>(batch64);
  TORCH_CHECK(
      cameras[kViewmats].dim() == 3 && cameras[kViewmats].size(1) == 4 &&
          cameras[kViewmats].size(2) == 4,
      "viewmats must be [B, 4, 4]");
  TORCH_CHECK(
      cameras[kIntrinsics].dim() == 3 && cameras[kIntrinsics].size(0) == batch64 &&
          cameras[kIntrinsics].size(1) == 3 && cameras[kIntrinsics].size(2) == 3,
      "intrinsics must be [B, 3, 3]");
  TORCH_CHECK(cameras[kSceneIds].numel() == batch64, "scene_ids must be [B]");
  TORCH_CHECK(
      cameras[kActiveCameraIds].numel() == 0,
      "upstream-faithful FlashGS does not add active-subset scheduling");

  const int64_t pixels = static_cast<int64_t>(height) * width;
  TORCH_CHECK(outputs[0].numel() >= batch64 * pixels * 3, "RGB output is too small");
  TORCH_CHECK(workspace[kPointsXY].numel() >= gaussian_count64 * 2, "points_xy is too small");
  TORCH_CHECK(workspace[kRgbDepth].numel() >= gaussian_count64 * 4, "rgb_depth is too small");
  TORCH_CHECK(
      workspace[kConicOpacity].numel() >= gaussian_count64 * 4,
      "conic_opacity is too small");
  const int64_t capacity64 = workspace[kKeysUnsorted].numel();
  TORCH_CHECK(capacity64 > 0 && capacity64 <= INT_MAX, "capacity must fit positive int32");
  const int capacity = static_cast<int>(capacity64);
  TORCH_CHECK(workspace[kValuesUnsorted].numel() >= capacity64, "values_unsorted too small");
  TORCH_CHECK(workspace[kKeysSorted].numel() >= capacity64, "keys_sorted too small");
  TORCH_CHECK(workspace[kValuesSorted].numel() >= capacity64, "values_sorted too small");
  const int tile_count = ((width + 15) / 16) * ((height + 15) / 16);
  TORCH_CHECK(
      workspace[kRanges].numel() >= static_cast<int64_t>(tile_count + 1) * 2,
      "ranges must include one sentinel tile");
  TORCH_CHECK(
      workspace[kCameraState].numel() >= 39,
      "camera_state must contain 39 float32 values");
  TORCH_CHECK(workspace[kRgbU8].numel() >= pixels * 3, "rgb_u8 is too small");
  TORCH_CHECK(workspace[kCounters].numel() >= batch64 * 3, "counters must be [B, 3]");

  c10::cuda::CUDAGuard guard(scene[kMeans].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(scene[kMeans].get_device());
  const float3 black = {0.0f, 0.0f, 0.0f};
  for (int camera = 0; camera < batch; ++camera) {
    int* counters = workspace[kCounters].data_ptr<int>() + static_cast<int64_t>(camera) * 3;
    C10_CUDA_CHECK(cudaMemsetAsync(counters, 0, 3 * sizeof(int), stream));
    flashgs_upstream_faithful::fill_sentinel_keys(
        workspace[kKeysUnsorted].data_ptr<uint64_t>(),
        capacity,
        static_cast<uint32_t>(tile_count),
        stream);
    flashgs_upstream_faithful::prepare_camera_state(
        cameras[kViewmats].data_ptr<float>() + static_cast<int64_t>(camera) * 16,
        cameras[kIntrinsics].data_ptr<float>() + static_cast<int64_t>(camera) * 9,
        cameras[kSceneIds].data_ptr<int64_t>() + camera,
        registered_scene_id,
        width,
        height,
        static_cast<float>(far_plane),
        static_cast<float>(near_plane),
        reinterpret_cast<flashgs::upstream_camera_state_t*>(
            workspace[kCameraState].data_ptr<float>()),
        counters + 2,
        stream);
    flashgs::preprocess_degree0_fixed(
        gaussian_count,
        reinterpret_cast<const glm::vec3*>(scene[kMeans].data_ptr<float>()),
        reinterpret_cast<const glm::vec3*>(scene[kColors].data_ptr<float>()),
        scene[kOpacities].data_ptr<float>(),
        reinterpret_cast<flashgs::cov3d_t*>(scene[kCovariances].data_ptr<float>()),
        width,
        height,
        16,
        16,
        reinterpret_cast<const flashgs::upstream_camera_state_t*>(
            workspace[kCameraState].data_ptr<float>()),
        reinterpret_cast<float2*>(workspace[kPointsXY].data_ptr<float>()),
        reinterpret_cast<float4*>(workspace[kRgbDepth].data_ptr<float>()),
        reinterpret_cast<float4*>(workspace[kConicOpacity].data_ptr<float>()),
        workspace[kKeysUnsorted].data_ptr<uint64_t>(),
        reinterpret_cast<uint32_t*>(workspace[kValuesUnsorted].data_ptr<int32_t>()),
        capacity,
        counters,
        counters + 1,
        stream);
    flashgs::sort_gaussian(
        capacity,
        width,
        height,
        16,
        16,
        reinterpret_cast<char*>(workspace[kSortTemp].data_ptr<uint8_t>()),
        static_cast<std::size_t>(workspace[kSortTemp].numel()),
        workspace[kKeysUnsorted].data_ptr<uint64_t>(),
        reinterpret_cast<uint32_t*>(workspace[kValuesUnsorted].data_ptr<int32_t>()),
        workspace[kKeysSorted].data_ptr<uint64_t>(),
        reinterpret_cast<uint32_t*>(workspace[kValuesSorted].data_ptr<int32_t>()),
        stream);
    flashgs::render_16x16(
        capacity,
        width,
        height,
        reinterpret_cast<float2*>(workspace[kPointsXY].data_ptr<float>()),
        reinterpret_cast<float4*>(workspace[kRgbDepth].data_ptr<float>()),
        reinterpret_cast<float4*>(workspace[kConicOpacity].data_ptr<float>()),
        workspace[kKeysSorted].data_ptr<uint64_t>(),
        reinterpret_cast<uint32_t*>(workspace[kValuesSorted].data_ptr<int32_t>()),
        reinterpret_cast<int2*>(workspace[kRanges].data_ptr<int32_t>()),
        black,
        reinterpret_cast<uchar3*>(workspace[kRgbU8].data_ptr<uint8_t>()),
        stream);
    flashgs_upstream_faithful::convert_rgb_u8_to_float(
        workspace[kRgbU8].data_ptr<uint8_t>(),
        outputs[0].data_ptr<float>() + static_cast<int64_t>(camera) * pixels * 3,
        static_cast<int>(pixels * 3),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "get_sort_buffer_size",
      &get_sort_buffer_size,
      "Return exact-upstream CUB storage for the fixed-capacity control");
  module.def(
      "precompute_covariances",
      &precompute_covariances,
      "Precompute upstream-equation object-space covariance on the current stream");
  module.def(
      "render_batch",
      &render_batch,
      "Run exact-upstream FlashGS RGB once per camera, including u8-to-f32 conversion");
}
