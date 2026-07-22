#include "ops.h"

namespace flashgs_adapter {
namespace {

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(
	int capacity,
	const uint64_t* point_list_keys,
	int tile_count,
	int2* ranges)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= capacity)
		return;

	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (currtile >= static_cast<uint32_t>(tile_count))
		return;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			if (prevtile < static_cast<uint32_t>(tile_count))
				ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == capacity - 1 ||
		(point_list_keys[idx + 1] >> 32) != currtile)
		ranges[currtile].y = idx + 1;
}

__forceinline__ __device__ float fast_ex2_ftz_f32(float x)
{
	float y;
	asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

template<bool kFullSensorOutput>
struct sensor_accumulator
{
};

template<>
struct sensor_accumulator<true>
{
	float depth;
	float best_weight;
	int32_t best_gaussian;
};

template<bool kFullSensorOutput>
__forceinline__ __device__ void pixel_shader(
	float3& C,
	float& T,
	sensor_accumulator<kFullSensorOutput>& sensor,
	bool& done,
	float2 pixf,
	float2 xy,
	float4 con_o,
	float3 rgb,
	float depth,
	int32_t gaussian)
{
	if (done)
		return;
	float2 d = { xy.x - pixf.x, xy.y - pixf.y };
	float power = -0.5f * (
		con_o.x * d.x * d.x + 2.0f * con_o.y * d.x * d.y +
		con_o.z * d.y * d.y);
	if (power > 0.0f || power < -20.0f)
		return;
	float alpha = fminf(FLASHGS_MAX_ALPHA, con_o.w * __expf(power));
	if (alpha < FLASHGS_ALPHA_THRESHOLD)
		return;
	float next_transmittance = T * (1.0f - alpha);
	if (next_transmittance <= FLASHGS_TRANSMITTANCE_THRESHOLD)
	{
		done = true;
		return;
	}
	float weight = T * alpha;
	C.x += rgb.x * weight;
	C.y += rgb.y * weight;
	C.z += rgb.z * weight;
	if constexpr (kFullSensorOutput)
	{
		sensor.depth += depth * weight;
		if (weight > sensor.best_weight)
		{
			sensor.best_weight = weight;
			sensor.best_gaussian = gaussian;
		}
	}
	T = next_transmittance;
}

template<bool kFullSensorOutput>
__forceinline__ __device__ void write_outputs(
	float* __restrict__ out_rgb,
	float* __restrict__ out_depth,
	float* __restrict__ out_alpha,
	int64_t* __restrict__ out_semantic,
	const int64_t* __restrict__ semantic_ids,
	int2 pix,
	int width,
	int height,
	float3 C,
	float T,
	const sensor_accumulator<kFullSensorOutput>& sensor,
	float semantic_min_alpha)
{
	if (pix.x < width && pix.y < height)
	{
		int pix_id = width * pix.y + pix.x;
		out_rgb[pix_id * 3 + 0] = C.x;
		out_rgb[pix_id * 3 + 1] = C.y;
		out_rgb[pix_id * 3 + 2] = C.z;
		if constexpr (kFullSensorOutput)
		{
			float accumulated_alpha = 1.0f - T;
			out_alpha[pix_id] = accumulated_alpha;
			out_depth[pix_id] = accumulated_alpha > 1.0e-8f
				? sensor.depth / accumulated_alpha
				: __int_as_float(0x7f800000);
			out_semantic[pix_id] =
				sensor.best_weight >= 0.0f &&
					accumulated_alpha >= semantic_min_alpha
					? semantic_ids[sensor.best_gaussian]
					: -1;
		}
	}
}

template<bool kFullSensorOutput>
struct render_load_info
{
	const void* data[FLASHGS_WARP_SIZE] = { nullptr };
	int lg2_scale[FLASHGS_WARP_SIZE] = { 0 };

	render_load_info(const uint32_t* point_list, const float2* points_xy, const float4* rgb_depth, const float4* conic_opacity)
	{
		for (int lane = 0; lane < 32; lane++)
		{
			switch (lane)
			{
			case 0:
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			case 4:
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			case 8:
				data[lane] = &points_xy->x;
				lg2_scale[lane] = 3;
				break;
			case 9:
				data[lane] = &points_xy->y;
				lg2_scale[lane] = 3;
				break;
			case 12:
				data[lane] = &points_xy->x;
				lg2_scale[lane] = 3;
				break;
			case 13:
				data[lane] = &points_xy->y;
				lg2_scale[lane] = 3;
				break;
			case 16:
				data[lane] = &rgb_depth->x;
				lg2_scale[lane] = 4;
				break;
			case 17:
				data[lane] = &rgb_depth->y;
				lg2_scale[lane] = 4;
				break;
			case 18:
				data[lane] = &rgb_depth->z;
				lg2_scale[lane] = 4;
				break;
			case 19:
				if constexpr (kFullSensorOutput)
				{
					data[lane] = &rgb_depth->w;
					lg2_scale[lane] = 4;
				}
				break;
			case 20:
				data[lane] = &rgb_depth->x;
				lg2_scale[lane] = 4;
				break;
			case 21:
				data[lane] = &rgb_depth->y;
				lg2_scale[lane] = 4;
				break;
			case 22:
				data[lane] = &rgb_depth->z;
				lg2_scale[lane] = 4;
				break;
			case 23:
				if constexpr (kFullSensorOutput)
				{
					data[lane] = &rgb_depth->w;
					lg2_scale[lane] = 4;
				}
				break;
			case 24:
				data[lane] = &conic_opacity->x;
				lg2_scale[lane] = 4;
				break;
			case 25:
				data[lane] = &conic_opacity->y;
				lg2_scale[lane] = 4;
				break;
			case 26:
				data[lane] = &conic_opacity->z;
				lg2_scale[lane] = 4;
				break;
			case 27:
				data[lane] = &conic_opacity->w;
				lg2_scale[lane] = 4;
				break;
			case 28:
				data[lane] = &conic_opacity->x;
				lg2_scale[lane] = 4;
				break;
			case 29:
				data[lane] = &conic_opacity->y;
				lg2_scale[lane] = 4;
				break;
			case 30:
				data[lane] = &conic_opacity->z;
				lg2_scale[lane] = 4;
				break;
			case 31:
				data[lane] = &conic_opacity->w;
				lg2_scale[lane] = 4;
				break;
			}
		}
	}
};

template<bool kFullSensorOutput>
__forceinline__ __device__ void get_gaussian_features(
	float2& xy,
	float3& rgb,
	float& depth,
	float4& con_o,
	float buf,
	int offset)
{
	xy = {
		__shfl_sync(~0, buf, 8 + offset),
		__shfl_sync(~0, buf, 9 + offset)
	};
	rgb = {
		__shfl_sync(~0, buf, 16 + offset),
		__shfl_sync(~0, buf, 17 + offset),
		__shfl_sync(~0, buf, 18 + offset)
	};
	if constexpr (kFullSensorOutput)
		depth = __shfl_sync(~0, buf, 19 + offset);
	else
		depth = 0.0f;
	con_o = {
		__shfl_sync(~0, buf, 24 + offset),
		__shfl_sync(~0, buf, 25 + offset),
		__shfl_sync(~0, buf, 26 + offset),
		__shfl_sync(~0, buf, 27 + offset)
	};
}

template<bool kFullSensorOutput>
__forceinline__ __device__ int32_t load_gaussian_id(
	const uint32_t* point_list,
	int point_id,
	int lane)
{
	int32_t value = -1;
	if constexpr (kFullSensorOutput)
	{
		if (lane == 0)
			value = static_cast<int32_t>(point_list[point_id]);
		return __shfl_sync(0xffffffffu, value, 0);
	}
	return -1;
}

template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y, bool kFullSensorOutput>
__global__ void renderCUDA(
	const int2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	const int64_t* __restrict__ semantic_ids,
	render_load_info<kFullSensorOutput> info,
	float semantic_min_alpha,
	float* __restrict__ out_rgb,
	float* __restrict__ out_depth,
	float* __restrict__ out_alpha,
	int64_t* __restrict__ out_semantic)
{
	int2 range = ranges[blockIdx.y * x_blocks + blockIdx.x];
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	// uint2 pix = { blockIdx.x * BLOCK_X + threadIdx.x, blockIdx.y * BLOCK_Y + threadIdx.y };
	int2 pix[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			pix[i][j] = {
				(int)blockIdx.x * BLOCK_X + (int)threadIdx.x * THREAD_X + j,
				(int)blockIdx.y * BLOCK_Y + (int)threadIdx.y * THREAD_Y + i
			};
		}
	}

	// float2 pixf = { (float)pix.x, (float)pix.y };
	float2 pixf[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			pixf[i][j] = {
				(float)pix[i][j].x + 0.5f,
				(float)pix[i][j].y + 0.5f
			};
		}
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
		}
	}

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			C[i][j] = { 0.0f, 0.0f, 0.0f };
		}
	}

	sensor_accumulator<kFullSensorOutput> sensors[THREAD_Y][THREAD_X];
	bool pixel_done[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			if constexpr (kFullSensorOutput)
			{
				sensors[i][j].depth = 0.0f;
				sensors[i][j].best_weight = -1.0f;
				sensors[i][j].best_gaussian = -1;
			}
			pixel_done[i][j] = false;
		}
	}

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset = 0;
		float2 xy;
		float3 rgb;
		float gaussian_depth;
		int32_t gaussian_id;
		float4 con_o;
		if (lane == 0)
		{
			offset = point_id + 2;
		}
		else if (lane == 4)
		{
			offset = point_id + 3;
		}
		else if ((lane & 4) == 0 && point_id + 1 < range.y)
		{
			offset = point_list[point_id + 0];
		}
		else if (point_id + 2 < range.y)
		{
			offset = point_list[point_id + 1];
		}
		const float* ptr = data == nullptr
			? nullptr
			: reinterpret_cast<const float*>(
				reinterpret_cast<const char*>(data) +
				(static_cast<int64_t>(offset) << lg2_scale));
		float buf = 0.0f;
		bool load_enable = data != nullptr;
		if (lane == 0)
		{
			load_enable = load_enable && point_id + 2 < range.y;
		}
		else if (lane == 4)
		{
			load_enable = load_enable && point_id + 3 < range.y;
		}
		else if ((lane & 4) == 0)
		{
			load_enable = load_enable && point_id + 0 < range.y;
		}
		else
		{
			load_enable = load_enable && point_id + 1 < range.y;
		}
		if (load_enable)
		{
			buf = __ldg(ptr); // 0: point_list[point_id + 2], 4: point_list[point_id + 3], 8: features[point_list[point_id + 0]], 12: features[point_list[point_id + 1]]
		}

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

#ifdef _DEBUG
			if (lane == 0)
			{
				printf("point_id = %d\n", point_id);
			}
#endif
			float ldg_buf = 0.0f;
			ptr = data == nullptr
				? nullptr
				: reinterpret_cast<const float*>(
					reinterpret_cast<const char*>(data) +
					(static_cast<int64_t>(offset) << lg2_scale));
			if (load_enable)
			{
				ldg_buf = __ldg(ptr); // 0: point_list[point_id + 4], 4: point_list[point_id + 5], 8: features[point_list[point_id + 2]], 12: features[point_list[point_id + 3]]
#ifdef _DEBUG
				if (lane == 0 && __float_as_int(ldg_buf) != point_list[point_id + 4])
				{
					printf("error1\n");
				}
				else if (lane == 4 && __float_as_int(ldg_buf) != point_list[point_id + 5])
				{
					printf("error2\n");
				}
				else if (lane == 8 && ldg_buf != points_xy[point_list[point_id + 2]].x)
				{
					printf("error3\n");
				}
				else if (lane == 12 && ldg_buf != points_xy[point_list[point_id + 3]].x)
				{
					printf("error4\n");
				}
#endif
			}

			get_gaussian_features<kFullSensorOutput>(
				xy, rgb, gaussian_depth, con_o, buf, 0);
			gaussian_id = load_gaussian_id<kFullSensorOutput>(
				point_list, point_id, lane);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 0]].x)
			{
				printf("error5\n");
			}
#endif

	#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
	#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader<kFullSensorOutput>(
						C[i][j], T[i][j], sensors[i][j], pixel_done[i][j],
						pixf[i][j], xy, con_o, rgb, gaussian_depth,
						gaussian_id);
				}
			}

			get_gaussian_features<kFullSensorOutput>(
				xy, rgb, gaussian_depth, con_o, buf, 4);
			gaussian_id = load_gaussian_id<kFullSensorOutput>(
				point_list, point_id + 1, lane);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 1]].x)
			{
				printf("error6\n");
			}
#endif

	#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
	#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader<kFullSensorOutput>(
						C[i][j], T[i][j], sensors[i][j], pixel_done[i][j],
						pixf[i][j], xy, con_o, rgb, gaussian_depth,
						gaussian_id);
				}
			}

			done = true;
	#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
	#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && pixel_done[i][j];
				}
			}

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

			if (lane == 0)
			{
				load_enable = load_enable && point_id + 4 < range.y;
			}
			else if (lane == 4)
			{
				load_enable = load_enable && point_id + 5 < range.y;
			}
			else if ((lane & 4) == 0)
			{
				load_enable = load_enable && point_id + 2 < range.y;
			}
			else
			{
				load_enable = load_enable && point_id + 3 < range.y;
			}

#ifdef _DEBUG
			if (lane == 0)
			{
				printf("point_id = %d\n", point_id);
			}
#endif
			float ldg_buf = 0.0f;
			ptr = data == nullptr
				? nullptr
				: reinterpret_cast<const float*>(
					reinterpret_cast<const char*>(data) +
					(static_cast<int64_t>(offset) << lg2_scale));
			if (load_enable)
			{
				ldg_buf = __ldg(ptr); // 0: point_list[point_id + 4], 4: point_list[point_id + 5], 8: features[point_list[point_id + 2]], 12: features[point_list[point_id + 3]]
#ifdef _DEBUG
				if (lane == 0 && __float_as_int(ldg_buf) != point_list[point_id + 4])
				{
					printf("error1\n");
				}
				else if (lane == 4 && __float_as_int(ldg_buf) != point_list[point_id + 5])
				{
					printf("error2\n");
				}
				else if (lane == 8 && ldg_buf != points_xy[point_list[point_id + 2]].x)
				{
					printf("error3\n");
				}
				else if (lane == 12 && ldg_buf != points_xy[point_list[point_id + 3]].x)
				{
					printf("error4\n");
				}
#endif
			}

			get_gaussian_features<kFullSensorOutput>(
				xy, rgb, gaussian_depth, con_o, buf, 0);
			gaussian_id = load_gaussian_id<kFullSensorOutput>(
				point_list, point_id, lane);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 0]].x)
			{
				printf("error5\n");
			}
#endif

#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader<kFullSensorOutput>(
						C[i][j], T[i][j], sensors[i][j], pixel_done[i][j],
						pixf[i][j], xy, con_o, rgb, gaussian_depth,
						gaussian_id);
				}
			}

			if (point_id + 1 >= range.y)
				break;

			get_gaussian_features<kFullSensorOutput>(
				xy, rgb, gaussian_depth, con_o, buf, 4);
			gaussian_id = load_gaussian_id<kFullSensorOutput>(
				point_list, point_id + 1, lane);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 1]].x)
			{
				printf("error6\n");
			}
#endif

#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader<kFullSensorOutput>(
						C[i][j], T[i][j], sensors[i][j], pixel_done[i][j],
						pixf[i][j], xy, con_o, rgb, gaussian_depth,
						gaussian_id);
				}
			}

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && pixel_done[i][j];
				}
			}
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				write_outputs<kFullSensorOutput>(
					out_rgb, out_depth, out_alpha, out_semantic,
					semantic_ids,
					pix[i][j], width, height, C[i][j], T[i][j],
					sensors[i][j], semantic_min_alpha);
			}
		}
	}
	else
	{
		sensor_accumulator<kFullSensorOutput> empty_sensor;
		if constexpr (kFullSensorOutput)
		{
			empty_sensor.depth = 0.0f;
			empty_sensor.best_weight = -1.0f;
			empty_sensor.best_gaussian = -1;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				write_outputs<kFullSensorOutput>(
					out_rgb, out_depth, out_alpha, out_semantic,
					semantic_ids,
					pix[i][j], width, height,
					float3{0.0f, 0.0f, 0.0f}, 1.0f,
					empty_sensor, semantic_min_alpha);
			}
		}
	}
}

template<int BLOCK_X, int BLOCK_Y>
void render_fixed(
	int capacity,
	int width,
	int height,
	const float2* points_xy,
	const float4* rgb_depth,
	const float4* conic_opacity,
	const int64_t* semantic_ids,
	const uint64_t* gaussian_keys_sorted,
	const uint32_t* gaussian_values_sorted,
	int2* ranges,
	float semantic_min_alpha,
	bool full_sensor_output,
	float* out_rgb,
	float* out_depth,
	float* out_alpha,
	int64_t* out_semantic,
	cudaStream_t stream)
{
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	cudaMemsetAsync(ranges, 0, sizeof(int2) * grid.x * grid.y, stream);

    identifyTileRanges<<<(capacity + 255) / 256, 256, 0, stream>>>(
        capacity,
        gaussian_keys_sorted,
        static_cast<int>(grid.x * grid.y),
        ranges);

    if (full_sensor_output)
    {
        renderCUDA<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4, true><<<
            grid, dim3(8, 4, 1), 0, stream>>>(
            ranges,
            gaussian_values_sorted,
            width, height, grid.x,
            points_xy,
            rgb_depth,
            conic_opacity,
            semantic_ids,
            render_load_info<true>(
                gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
            semantic_min_alpha,
            out_rgb,
            out_depth,
            out_alpha,
            out_semantic);
    }
    else
    {
        renderCUDA<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4, false><<<
            grid, dim3(8, 4, 1), 0, stream>>>(
            ranges,
            gaussian_values_sorted,
            width, height, grid.x,
            points_xy,
            rgb_depth,
            conic_opacity,
            semantic_ids,
            render_load_info<false>(
                gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
            semantic_min_alpha,
            out_rgb,
            out_depth,
            out_alpha,
            out_semantic);
    }
}

} // namespace

void render_16x16_fixed(
	int capacity,
	int width,
	int height,
	const float2* points_xy,
	const float4* rgb_depth,
	const float4* conic_opacity,
	const int64_t* semantic_ids,
	const uint64_t* gaussian_keys_sorted,
	const uint32_t* gaussian_values_sorted,
	int2* ranges,
	float semantic_min_alpha,
	bool full_sensor_output,
	float* out_rgb,
	float* out_depth,
	float* out_alpha,
	int64_t* out_semantic,
	cudaStream_t stream)
{
    render_fixed<16, 16>(
		capacity,
		width,
		height,
		points_xy,
		rgb_depth,
		conic_opacity,
		semantic_ids,
		gaussian_keys_sorted,
		gaussian_values_sorted,
		ranges,
		semantic_min_alpha,
		full_sensor_output,
		out_rgb,
		out_depth,
		out_alpha,
		out_semantic,
		stream);
}

} // namespace flashgs_adapter
