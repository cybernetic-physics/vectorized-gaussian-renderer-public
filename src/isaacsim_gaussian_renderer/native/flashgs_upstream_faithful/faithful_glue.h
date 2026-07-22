#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include <ops.h>

namespace flashgs {

// Plain float-compatible state written on the GPU from RendererService's
// row-major OpenCV world-to-camera matrix and centered intrinsic matrix.
struct upstream_camera_state_t
{
	glm::mat4 viewmatrix;
	glm::mat4 projmatrix;
	glm::vec3 cam_position;
	float tan_fovx;
	float tan_fovy;
	float focal_x;
	float focal_y;
};

static_assert(sizeof(upstream_camera_state_t) == 39 * sizeof(float),
	"FlashGS camera state must fit the preallocated 39-float workspace");

void preprocess_degree0_fixed(int P,
	const glm::vec3* positions, const glm::vec3* colors,
	const float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	const upstream_camera_state_t* camera_state,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int capacity, int* curr_offset, int* overflow, cudaStream_t stream = 0);

}  // namespace flashgs

namespace flashgs_upstream_faithful {

void precompute_covariances(
	int count,
	const float* scales,
	const float* rotations,
	flashgs::cov3d_t* covariances,
	cudaStream_t stream);

void prepare_camera_state(
	const float* viewmat,
	const float* intrinsics,
	const int64_t* scene_id,
	int64_t expected_scene_id,
	int width,
	int height,
	float z_far,
	float z_near,
	flashgs::upstream_camera_state_t* state,
	int* camera_contract_errors,
	cudaStream_t stream);

void fill_sentinel_keys(
	uint64_t* keys,
	int capacity,
	uint32_t sentinel_tile,
	cudaStream_t stream);

void convert_rgb_u8_to_float(
	const uint8_t* input,
	float* output,
	int element_count,
	cudaStream_t stream);

}  // namespace flashgs_upstream_faithful
