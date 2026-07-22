#define GLM_FORCE_CUDA
#include "faithful_glue.h"

#include <glm/glm.hpp>

#include <cmath>

namespace flashgs_upstream_faithful {
namespace {

__global__ void precompute_covariances_cuda(
	int count,
	const float* __restrict__ scales,
	const float* __restrict__ rotations,
	flashgs::cov3d_t* __restrict__ covariances)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count)
		return;

	const float* scale_data = scales + idx * 3;
	const float* rotation_data = rotations + idx * 4;
	glm::vec3 scale(scale_data[0], scale_data[1], scale_data[2]);
	// Canonical RendererService quaternions and upstream PLY quaternions are
	// both WXYZ. Canonical registration already normalizes them once.
	glm::vec4 rot(
		rotation_data[0], rotation_data[1], rotation_data[2], rotation_data[3]);

	// This block is the upstream pybind.cpp computeCov3D equation, moved from
	// scene-load CPU code to the active CUDA stream without changing its math.
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	float r = rot.x;
	float x = rot.y;
	float y = rot.z;
	float z = rot.w;
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;
	flashgs::cov3d_t& covariance = covariances[idx];
	covariance.s[0] = Sigma[0][0];
	covariance.s[1] = Sigma[0][1];
	covariance.s[2] = Sigma[0][2];
	covariance.s[3] = Sigma[1][1];
	covariance.s[4] = Sigma[1][2];
	covariance.s[5] = Sigma[2][2];
}

__device__ void record_contract_error(bool condition, int* errors)
{
	if (!condition)
		atomicAdd(errors, 1);
}

__global__ void prepare_camera_state_cuda(
	const float* __restrict__ viewmat,
	const float* __restrict__ intrinsics,
	const int64_t* __restrict__ scene_id,
	int64_t expected_scene_id,
	int width,
	int height,
	float z_far,
	float z_near,
	flashgs::upstream_camera_state_t* __restrict__ state,
	int* __restrict__ errors)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;

	const float tolerance = 1.0e-4f;
	record_contract_error(*scene_id == expected_scene_id, errors);
	record_contract_error(fabsf(intrinsics[1]) <= tolerance, errors);
	record_contract_error(fabsf(intrinsics[3]) <= tolerance, errors);
	record_contract_error(fabsf(intrinsics[6]) <= tolerance, errors);
	record_contract_error(fabsf(intrinsics[7]) <= tolerance, errors);
	record_contract_error(fabsf(intrinsics[8] - 1.0f) <= tolerance, errors);
	record_contract_error(
		fabsf(intrinsics[2] - static_cast<float>(width) * 0.5f) <= tolerance,
		errors);
	record_contract_error(
		fabsf(intrinsics[5] - static_cast<float>(height) * 0.5f) <= tolerance,
		errors);
	record_contract_error(intrinsics[0] > 0.0f && intrinsics[4] > 0.0f, errors);
	record_contract_error(fabsf(viewmat[12]) <= tolerance, errors);
	record_contract_error(fabsf(viewmat[13]) <= tolerance, errors);
	record_contract_error(fabsf(viewmat[14]) <= tolerance, errors);
	record_contract_error(fabsf(viewmat[15] - 1.0f) <= tolerance, errors);

	// Convert row-major [R|t] into the column-major GLM matrix used verbatim by
	// upstream transformPoint4x3/transformPoint4x4.
	float* view = reinterpret_cast<float*>(&state->viewmatrix);
	for (int row = 0; row < 4; ++row)
	{
		for (int column = 0; column < 4; ++column)
			view[column * 4 + row] = viewmat[row * 4 + column];
	}

	// The camera center is -R^T t for the RendererService world-to-camera
	// matrix. Upstream only uses it for view-dependent SH; degree-zero keeps it
	// for exact camera-state parity and future auditability.
	state->cam_position = glm::vec3(
		-(viewmat[0] * viewmat[3] + viewmat[4] * viewmat[7] + viewmat[8] * viewmat[11]),
		-(viewmat[1] * viewmat[3] + viewmat[5] * viewmat[7] + viewmat[9] * viewmat[11]),
		-(viewmat[2] * viewmat[3] + viewmat[6] * viewmat[7] + viewmat[10] * viewmat[11]));

	const float focal_x = intrinsics[0];
	const float focal_y = intrinsics[4];
	float top = height / (2.0f * focal_y) * z_near;
	float bottom = -top;
	float right = width / (2.0f * focal_x) * z_near;
	float left = -right;
	glm::mat4 P;
	float* projection = reinterpret_cast<float*>(&P);
	for (int index = 0; index < 16; ++index)
		projection[index] = 0.0f;
	float z_sign = 1.0f;
	P[0][0] = 2.0f * z_near / (right - left);
	P[1][1] = 2.0f * z_near / (top - bottom);
	P[0][2] = (right + left) / (right - left);
	P[1][2] = (top + bottom) / (top - bottom);
	P[3][2] = z_sign;
	P[2][2] = z_sign * z_far / (z_far - z_near);
	P[2][3] = -(z_far * z_near) / (z_far - z_near);
	state->projmatrix = glm::transpose(P) * state->viewmatrix;
	state->tan_fovx = width / (2.0f * focal_x);
	state->tan_fovy = height / (2.0f * focal_y);
	state->focal_x = focal_x;
	state->focal_y = focal_y;
}

__global__ void fill_sentinel_keys_cuda(
	uint64_t* __restrict__ keys,
	int capacity,
	uint64_t sentinel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < capacity)
		keys[idx] = sentinel;
}

__global__ void convert_rgb_u8_to_float_cuda(
	const uint8_t* __restrict__ input,
	float* __restrict__ output,
	int element_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < element_count)
		output[idx] = static_cast<float>(input[idx]) * (1.0f / 255.0f);
}

}  // namespace

void precompute_covariances(
	int count,
	const float* scales,
	const float* rotations,
	flashgs::cov3d_t* covariances,
	cudaStream_t stream)
{
	precompute_covariances_cuda<<<(count + 255) / 256, 256, 0, stream>>>(
		count, scales, rotations, covariances);
}

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
	cudaStream_t stream)
{
	prepare_camera_state_cuda<<<1, 1, 0, stream>>>(
		viewmat, intrinsics, scene_id, expected_scene_id, width, height,
		z_far, z_near, state, camera_contract_errors);
}

void fill_sentinel_keys(
	uint64_t* keys,
	int capacity,
	uint32_t sentinel_tile,
	cudaStream_t stream)
{
	uint64_t sentinel = (static_cast<uint64_t>(sentinel_tile) << 32) | 0xffffffffull;
	fill_sentinel_keys_cuda<<<(capacity + 255) / 256, 256, 0, stream>>>(
		keys, capacity, sentinel);
}

void convert_rgb_u8_to_float(
	const uint8_t* input,
	float* output,
	int element_count,
	cudaStream_t stream)
{
	convert_rgb_u8_to_float_cuda<<<(element_count + 255) / 256, 256, 0, stream>>>(
		input, output, element_count);
}

}  // namespace flashgs_upstream_faithful
