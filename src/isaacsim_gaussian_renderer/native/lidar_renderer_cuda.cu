#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kDescriptorWidth = 14;
constexpr int kMaxStack = 64;
constexpr int kMaxSemanticSlots = 32;

enum DescriptorField : int {
  kSceneId = 0,
  kGaussianCount = 1,
  kLeafCount = 2,
  kLeafCapacity = 3,
  kNodeCount = 4,
  kMeansPtr = 5,
  kScalesPtr = 6,
  kRotationsPtr = 7,
  kOpacitiesPtr = 8,
  kSemanticsPtr = 9,
  kReflectivityPtr = 10,
  kConfidencePtr = 11,
  kSortedIndicesPtr = 12,
  kNodeBoundsPtr = 13,
};

enum CounterField : int {
  kCallsStarted = 0,
  kCallsCompleted = 1,
  kRaysTraced = 2,
  kNodeVisits = 3,
  kLeafTests = 4,
  kCandidates = 5,
  kReturns = 6,
  kStackOverflow = 7,
  kSemanticOverflow = 8,
  kInvalidDirections = 9,
  kInvalidSceneIds = 10,
  kInvalidActiveSensorIds = 11,
};

struct SceneDescriptor {
  int64_t scene_id;
  int gaussian_count;
  int leaf_count;
  int leaf_capacity;
  int node_count;
  const float* means;
  const float* scales;
  const float* rotations;
  const float* opacities;
  const int64_t* semantics;
  const float* reflectivity;
  const float* confidence;
  const int32_t* sorted_indices;
  const float* node_bounds;
};

struct TraceConfig {
  float near_plane;
  float far_plane;
  float support_sigma;
  float support_q_max;
  float detection_threshold;
  float planarity_ratio_max;
  float min_incidence_cos;
  float cluster_abs;
  float cluster_relative;
  float fallback_reflectivity;
  float direction_norm_tolerance;
  int packet_size;
  int semantic_slots;
};

struct Candidate {
  float t;
  float weight;
  float intensity;
  int64_t semantic;
};

struct Reduction {
  float range;
  float intensity;
  int64_t semantic;
  bool valid;
  bool semantic_overflow;
};

__device__ __forceinline__ void add_counter(int64_t* counters, int index, uint64_t value = 1) {
  atomicAdd(
      reinterpret_cast<unsigned long long*>(counters + index),
      static_cast<unsigned long long>(value));
}

__device__ __forceinline__ float3 add3(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 mul3(float3 a, float value) {
  return make_float3(a.x * value, a.y * value, a.z * value);
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float component(float3 value, int axis) {
  return axis == 0 ? value.x : axis == 1 ? value.y : value.z;
}

__device__ __forceinline__ float3 load3(const float* values) {
  return make_float3(values[0], values[1], values[2]);
}

__device__ __forceinline__ void quaternion_axes(
    const float* quaternion,
    float3 axes[3]) {
  const float w = quaternion[0];
  const float x = quaternion[1];
  const float y = quaternion[2];
  const float z = quaternion[3];
  axes[0] = make_float3(
      1.0f - 2.0f * (y * y + z * z),
      2.0f * (x * y + z * w),
      2.0f * (x * z - y * w));
  axes[1] = make_float3(
      2.0f * (x * y - z * w),
      1.0f - 2.0f * (x * x + z * z),
      2.0f * (y * z + x * w));
  axes[2] = make_float3(
      2.0f * (x * z + y * w),
      2.0f * (y * z - x * w),
      1.0f - 2.0f * (x * x + y * y));
}

__device__ __forceinline__ int smallest_scale_axis(const float* scales) {
  int axis = 0;
  if (scales[1] < scales[axis]) {
    axis = 1;
  }
  if (scales[2] < scales[axis]) {
    axis = 2;
  }
  return axis;
}

__device__ __forceinline__ uint32_t expand_morton_bits(uint32_t value) {
  value &= 0x000003ffu;
  value = (value | (value << 16)) & 0x030000ffu;
  value = (value | (value << 8)) & 0x0300f00fu;
  value = (value | (value << 4)) & 0x030c30c3u;
  value = (value | (value << 2)) & 0x09249249u;
  return value;
}

__global__ void compute_morton_codes_kernel(
    const float* means,
    int gaussian_count,
    const float* scene_bounds,
    uint64_t* keys) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= gaussian_count) {
    return;
  }
  const float* mean = means + 3 * index;
  uint32_t quantized[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    const float low = scene_bounds[axis];
    const float extent = fmaxf(scene_bounds[3 + axis] - low, 1.0e-12f);
    const float normalized = fminf(fmaxf((mean[axis] - low) / extent, 0.0f), 1.0f);
    quantized[axis] = static_cast<uint32_t>(normalized * 1023.0f + 0.5f);
  }
  keys[index] = static_cast<uint64_t>(
      expand_morton_bits(quantized[0]) |
      (expand_morton_bits(quantized[1]) << 1) |
      (expand_morton_bits(quantized[2]) << 2));
}

__global__ void build_leaf_bounds_kernel(
    const float* means,
    const float* scales,
    const float* rotations,
    const int32_t* sorted_indices,
    int gaussian_count,
    int packet_size,
    int leaf_count,
    int leaf_capacity,
    float support_sigma,
    float planarity_ratio_max,
    float* node_bounds) {
  const int leaf = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf >= leaf_capacity) {
    return;
  }
  float low[3] = {CUDART_INF_F, CUDART_INF_F, CUDART_INF_F};
  float high[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
  if (leaf < leaf_count) {
    const int begin = leaf * packet_size;
    const int end = min(begin + packet_size, gaussian_count);
    for (int sorted = begin; sorted < end; ++sorted) {
      const int gaussian = sorted_indices[sorted];
      const float* gaussian_scales = scales + 3 * gaussian;
      const int normal_axis = smallest_scale_axis(gaussian_scales);
      int tangent_axes[2];
      int tangent_count = 0;
      for (int axis = 0; axis < 3; ++axis) {
        if (axis != normal_axis) {
          tangent_axes[tangent_count++] = axis;
        }
      }
      const float tangent_min = fminf(
          gaussian_scales[tangent_axes[0]],
          gaussian_scales[tangent_axes[1]]);
      if (!(gaussian_scales[normal_axis] > 0.0f) ||
          gaussian_scales[normal_axis] / tangent_min > planarity_ratio_max) {
        continue;
      }
      float3 axes[3];
      quaternion_axes(rotations + 4 * gaussian, axes);
      const float3 mean = load3(means + 3 * gaussian);
      const float3 tangent0 = axes[tangent_axes[0]];
      const float3 tangent1 = axes[tangent_axes[1]];
      for (int axis = 0; axis < 3; ++axis) {
        const float extent = support_sigma * (
            fabsf(component(tangent0, axis)) * gaussian_scales[tangent_axes[0]] +
            fabsf(component(tangent1, axis)) * gaussian_scales[tangent_axes[1]]) +
            1.0e-4f;
        const float center = component(mean, axis);
        low[axis] = fminf(low[axis], center - extent);
        high[axis] = fmaxf(high[axis], center + extent);
      }
    }
  }
  float* destination = node_bounds + 6 * (leaf_capacity - 1 + leaf);
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    destination[axis] = low[axis];
    destination[3 + axis] = high[axis];
  }
}

__global__ void build_internal_bounds_kernel(
    float* node_bounds,
    int parent_start,
    int parent_count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= parent_count) {
    return;
  }
  const int node = parent_start + offset;
  const float* left = node_bounds + 6 * (2 * node + 1);
  const float* right = node_bounds + 6 * (2 * node + 2);
  float* destination = node_bounds + 6 * node;
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    destination[axis] = fminf(left[axis], right[axis]);
    destination[3 + axis] = fmaxf(left[3 + axis], right[3 + axis]);
  }
}

__global__ void pack_descriptor_kernel(
    int64_t* descriptors,
    int slot,
    int64_t scene_id,
    int gaussian_count,
    int leaf_count,
    int leaf_capacity,
    const float* means,
    const float* scales,
    const float* rotations,
    const float* opacities,
    const int64_t* semantics,
    const float* reflectivity,
    const float* confidence,
    const int32_t* sorted_indices,
    const float* node_bounds) {
  int64_t* row = descriptors + slot * kDescriptorWidth;
  row[kSceneId] = scene_id;
  row[kGaussianCount] = gaussian_count;
  row[kLeafCount] = leaf_count;
  row[kLeafCapacity] = leaf_capacity;
  row[kNodeCount] = 2 * leaf_capacity - 1;
  row[kMeansPtr] = reinterpret_cast<int64_t>(means);
  row[kScalesPtr] = reinterpret_cast<int64_t>(scales);
  row[kRotationsPtr] = reinterpret_cast<int64_t>(rotations);
  row[kOpacitiesPtr] = reinterpret_cast<int64_t>(opacities);
  row[kSemanticsPtr] = reinterpret_cast<int64_t>(semantics);
  row[kReflectivityPtr] = reinterpret_cast<int64_t>(reflectivity);
  row[kConfidencePtr] = reinterpret_cast<int64_t>(confidence);
  row[kSortedIndicesPtr] = reinterpret_cast<int64_t>(sorted_indices);
  row[kNodeBoundsPtr] = reinterpret_cast<int64_t>(node_bounds);
}

__device__ __forceinline__ SceneDescriptor load_descriptor(const int64_t* row) {
  SceneDescriptor descriptor;
  descriptor.scene_id = row[kSceneId];
  descriptor.gaussian_count = static_cast<int>(row[kGaussianCount]);
  descriptor.leaf_count = static_cast<int>(row[kLeafCount]);
  descriptor.leaf_capacity = static_cast<int>(row[kLeafCapacity]);
  descriptor.node_count = static_cast<int>(row[kNodeCount]);
  descriptor.means = reinterpret_cast<const float*>(row[kMeansPtr]);
  descriptor.scales = reinterpret_cast<const float*>(row[kScalesPtr]);
  descriptor.rotations = reinterpret_cast<const float*>(row[kRotationsPtr]);
  descriptor.opacities = reinterpret_cast<const float*>(row[kOpacitiesPtr]);
  descriptor.semantics = reinterpret_cast<const int64_t*>(row[kSemanticsPtr]);
  descriptor.reflectivity = reinterpret_cast<const float*>(row[kReflectivityPtr]);
  descriptor.confidence = reinterpret_cast<const float*>(row[kConfidencePtr]);
  descriptor.sorted_indices = reinterpret_cast<const int32_t*>(row[kSortedIndicesPtr]);
  descriptor.node_bounds = reinterpret_cast<const float*>(row[kNodeBoundsPtr]);
  return descriptor;
}

__device__ __forceinline__ bool intersect_aabb(
    const float* bounds,
    float3 origin,
    float3 direction,
    float minimum_t,
    float maximum_t,
    float* entry = nullptr) {
  float low_t = minimum_t;
  float high_t = maximum_t;
  for (int axis = 0; axis < 3; ++axis) {
    const float origin_axis = component(origin, axis);
    const float direction_axis = component(direction, axis);
    if (fabsf(direction_axis) < 1.0e-12f) {
      if (origin_axis < bounds[axis] || origin_axis > bounds[3 + axis]) {
        return false;
      }
      continue;
    }
    const float inverse = 1.0f / direction_axis;
    float first = (bounds[axis] - origin_axis) * inverse;
    float second = (bounds[3 + axis] - origin_axis) * inverse;
    if (first > second) {
      const float temporary = first;
      first = second;
      second = temporary;
    }
    low_t = fmaxf(low_t, first);
    high_t = fminf(high_t, second);
    if (low_t > high_t) {
      return false;
    }
  }
  if (entry != nullptr) {
    *entry = low_t;
  }
  return true;
}

__device__ __forceinline__ bool evaluate_candidate(
    const SceneDescriptor& scene,
    int gaussian,
    float3 origin,
    float3 direction,
    const TraceConfig& config,
    Candidate* candidate) {
  const float* scales = scene.scales + 3 * gaussian;
  const int normal_axis = smallest_scale_axis(scales);
  int tangent_axes[2];
  int tangent_count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis != normal_axis) {
      tangent_axes[tangent_count++] = axis;
    }
  }
  const float tangent_min = fminf(scales[tangent_axes[0]], scales[tangent_axes[1]]);
  if (!(scales[normal_axis] > 0.0f) || scales[normal_axis] / tangent_min > config.planarity_ratio_max) {
    return false;
  }
  float3 axes[3];
  quaternion_axes(scene.rotations + 4 * gaussian, axes);
  const float3 normal = axes[normal_axis];
  const float denominator = dot3(normal, direction);
  const float incidence = fabsf(denominator);
  if (incidence < config.min_incidence_cos) {
    return false;
  }
  const float3 mean = load3(scene.means + 3 * gaussian);
  const float t = dot3(normal, sub3(mean, origin)) / denominator;
  if (!(t >= config.near_plane && t <= config.far_plane)) {
    return false;
  }
  const float3 difference = sub3(add3(origin, mul3(direction, t)), mean);
  const float tangent0 = dot3(difference, axes[tangent_axes[0]]) / scales[tangent_axes[0]];
  const float tangent1 = dot3(difference, axes[tangent_axes[1]]) / scales[tangent_axes[1]];
  const float q = tangent0 * tangent0 + tangent1 * tangent1;
  if (!(q <= config.support_q_max)) {
    return false;
  }
  const float support = expf(-0.5f * q);
  const float confidence = scene.confidence == nullptr ? 1.0f : scene.confidence[gaussian];
  const float weight = scene.opacities[gaussian] * support * confidence;
  if (!(weight >= config.detection_threshold) || !isfinite(weight)) {
    return false;
  }
  const float reflectivity = scene.reflectivity == nullptr
      ? config.fallback_reflectivity
      : scene.reflectivity[gaussian];
  candidate->t = t;
  candidate->weight = weight;
  candidate->intensity = fminf(fmaxf(reflectivity * support * incidence, 0.0f), 1.0f);
  candidate->semantic = scene.semantics[gaussian];
  return true;
}

__device__ bool find_nearest_candidate(
    const SceneDescriptor& scene,
    float3 origin,
    float3 direction,
    const TraceConfig& config,
    float minimum_exclusive,
    float* nearest,
    uint64_t* node_visits,
    uint64_t* primitive_tests,
    uint64_t* candidates,
    bool* stack_overflow) {
  int stack[kMaxStack];
  int stack_size = 1;
  stack[0] = 0;
  float best = config.far_plane;
  bool found = false;
  const int leaf_base = scene.leaf_capacity - 1;
  while (stack_size > 0) {
    const int node = stack[--stack_size];
    ++(*node_visits);
    const float* bounds = scene.node_bounds + 6 * node;
    if (!intersect_aabb(bounds, origin, direction, config.near_plane, best) ||
        bounds[0] > bounds[3]) {
      continue;
    }
    if (node >= leaf_base) {
      const int leaf = node - leaf_base;
      if (leaf >= scene.leaf_count) {
        continue;
      }
      const int begin = leaf * config.packet_size;
      const int end = min(begin + config.packet_size, scene.gaussian_count);
      for (int sorted = begin; sorted < end; ++sorted) {
        ++(*primitive_tests);
        Candidate candidate;
        if (evaluate_candidate(
                scene,
                scene.sorted_indices[sorted],
                origin,
                direction,
                config,
                &candidate)) {
          ++(*candidates);
          if (candidate.t > minimum_exclusive && (!found || candidate.t < best)) {
            best = candidate.t;
            found = true;
          }
        }
      }
      continue;
    }
    const int left = 2 * node + 1;
    const int right = left + 1;
    float left_entry = 0.0f;
    float right_entry = 0.0f;
    const bool hit_left = intersect_aabb(
        scene.node_bounds + 6 * left,
        origin,
        direction,
        config.near_plane,
        best,
        &left_entry);
    const bool hit_right = intersect_aabb(
        scene.node_bounds + 6 * right,
        origin,
        direction,
        config.near_plane,
        best,
        &right_entry);
    const int required = static_cast<int>(hit_left) + static_cast<int>(hit_right);
    if (stack_size + required > kMaxStack) {
      *stack_overflow = true;
      return false;
    }
    if (hit_left && hit_right) {
      if (left_entry <= right_entry) {
        stack[stack_size++] = right;
        stack[stack_size++] = left;
      } else {
        stack[stack_size++] = left;
        stack[stack_size++] = right;
      }
    } else if (hit_left) {
      stack[stack_size++] = left;
    } else if (hit_right) {
      stack[stack_size++] = right;
    }
  }
  *nearest = best;
  return found;
}

__device__ Reduction reduce_cluster(
    const SceneDescriptor& scene,
    float3 origin,
    float3 direction,
    const TraceConfig& config,
    float first_t,
    float cluster_end,
    uint64_t* node_visits,
    uint64_t* primitive_tests,
    bool* stack_overflow) {
  int64_t labels[kMaxSemanticSlots];
  double label_weights[kMaxSemanticSlots];
  for (int index = 0; index < config.semantic_slots; ++index) {
    labels[index] = -1;
    label_weights[index] = 0.0;
  }
  int label_count = 0;
  bool semantic_overflow = false;
  double total_weight = 0.0;
  double weighted_range = 0.0;
  double weighted_intensity = 0.0;
  int stack[kMaxStack];
  int stack_size = 1;
  stack[0] = 0;
  const int leaf_base = scene.leaf_capacity - 1;
  while (stack_size > 0) {
    const int node = stack[--stack_size];
    ++(*node_visits);
    const float* bounds = scene.node_bounds + 6 * node;
    if (!intersect_aabb(bounds, origin, direction, first_t, cluster_end) ||
        bounds[0] > bounds[3]) {
      continue;
    }
    if (node >= leaf_base) {
      const int leaf = node - leaf_base;
      if (leaf >= scene.leaf_count) {
        continue;
      }
      const int begin = leaf * config.packet_size;
      const int end = min(begin + config.packet_size, scene.gaussian_count);
      for (int sorted = begin; sorted < end; ++sorted) {
        ++(*primitive_tests);
        Candidate candidate;
        if (!evaluate_candidate(
                scene,
                scene.sorted_indices[sorted],
                origin,
                direction,
                config,
                &candidate) ||
            candidate.t < first_t || candidate.t > cluster_end) {
          continue;
        }
        total_weight += static_cast<double>(candidate.weight);
        weighted_range += static_cast<double>(candidate.weight) * candidate.t;
        weighted_intensity += static_cast<double>(candidate.weight) * candidate.intensity;
        int label_slot = -1;
        for (int index = 0; index < label_count; ++index) {
          if (labels[index] == candidate.semantic) {
            label_slot = index;
            break;
          }
        }
        if (label_slot < 0) {
          if (label_count >= config.semantic_slots) {
            semantic_overflow = true;
          } else {
            label_slot = label_count++;
            labels[label_slot] = candidate.semantic;
          }
        }
        if (label_slot >= 0) {
          label_weights[label_slot] += static_cast<double>(candidate.weight);
        }
      }
      continue;
    }
    const int left = 2 * node + 1;
    const int right = left + 1;
    const bool hit_left = intersect_aabb(
        scene.node_bounds + 6 * left,
        origin,
        direction,
        first_t,
        cluster_end);
    const bool hit_right = intersect_aabb(
        scene.node_bounds + 6 * right,
        origin,
        direction,
        first_t,
        cluster_end);
    const int required = static_cast<int>(hit_left) + static_cast<int>(hit_right);
    if (stack_size + required > kMaxStack) {
      *stack_overflow = true;
      return {0.0f, 0.0f, -1, false, semantic_overflow};
    }
    if (hit_right) {
      stack[stack_size++] = right;
    }
    if (hit_left) {
      stack[stack_size++] = left;
    }
  }
  if (!(total_weight > 0.0) || semantic_overflow) {
    return {0.0f, 0.0f, -1, false, semantic_overflow};
  }
  int winning_slot = 0;
  for (int index = 1; index < label_count; ++index) {
    if (label_weights[index] > label_weights[winning_slot] ||
        (label_weights[index] == label_weights[winning_slot] &&
         labels[index] < labels[winning_slot])) {
      winning_slot = index;
    }
  }
  return {
      static_cast<float>(weighted_range / total_weight),
      static_cast<float>(weighted_intensity / total_weight),
      labels[winning_slot],
      true,
      false};
}

__global__ void begin_call_kernel(int64_t* counters) {
  add_counter(counters, kCallsStarted);
}

__global__ void end_call_kernel(int64_t* counters) {
  add_counter(counters, kCallsCompleted);
}

__global__ void clear_outputs_kernel(
    int64_t total_returns,
    int rays,
    int returns,
    const int32_t* time_offsets_i32,
    const int64_t* time_offsets_i64,
    bool time_input_i64,
    float* ranges,
    float* positions,
    float* intensities,
    int64_t* semantics,
    bool* valid,
    int64_t* output_times,
    int32_t* return_counts) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_returns) {
    return;
  }
  ranges[index] = CUDART_INF_F;
  intensities[index] = 0.0f;
  semantics[index] = -1;
  valid[index] = false;
  positions[3 * index] = 0.0f;
  positions[3 * index + 1] = 0.0f;
  positions[3 * index + 2] = 0.0f;
  const int ray = static_cast<int>((index / returns) % rays);
  output_times[index] = time_input_i64
      ? time_offsets_i64[ray]
      : static_cast<int64_t>(time_offsets_i32[ray]);
  if (index % returns == 0) {
    return_counts[index / returns] = 0;
  }
}

__global__ void trace_lidar_kernel(
    const int64_t* descriptors,
    int scene_count,
    const float* ray_directions,
    int rays,
    const float* sensor_to_world,
    const float* scene_to_world,
    const int64_t* scene_ids,
    const int64_t* active_sensor_ids,
    int active_count,
    int batch,
    int returns,
    TraceConfig config,
    float* ranges,
    float* positions,
    float* intensities,
    int64_t* semantics,
    bool* valid,
    int32_t* return_counts,
    int64_t* counters) {
  const int64_t work_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t work_count = static_cast<int64_t>(active_count) * rays;
  if (work_index >= work_count) {
    return;
  }
  const int active_index = static_cast<int>(work_index / rays);
  const int ray = static_cast<int>(work_index % rays);
  const int sensor = active_sensor_ids == nullptr
      ? active_index
      : static_cast<int>(active_sensor_ids[active_index]);
  if (active_index > 0 && sensor <= static_cast<int>(active_sensor_ids[active_index - 1])) {
    if (ray == 0) {
      add_counter(counters, kInvalidActiveSensorIds);
    }
    return;
  }
  if (sensor < 0 || sensor >= batch) {
    add_counter(counters, kInvalidActiveSensorIds);
    return;
  }
  add_counter(counters, kRaysTraced);
  const float3 local_sensor_direction = load3(ray_directions + 3 * ray);
  const float norm_squared = dot3(local_sensor_direction, local_sensor_direction);
  const float lower_norm = 1.0f - config.direction_norm_tolerance;
  const float upper_norm = 1.0f + config.direction_norm_tolerance;
  if (!isfinite(norm_squared) || norm_squared < lower_norm * lower_norm ||
      norm_squared > upper_norm * upper_norm) {
    add_counter(counters, kInvalidDirections);
    return;
  }
  SceneDescriptor scene{};
  bool found_scene = false;
  for (int slot = 0; slot < scene_count; ++slot) {
    const SceneDescriptor candidate = load_descriptor(descriptors + slot * kDescriptorWidth);
    if (candidate.scene_id == scene_ids[sensor]) {
      scene = candidate;
      found_scene = true;
      break;
    }
  }
  if (!found_scene) {
    add_counter(counters, kInvalidSceneIds);
    return;
  }

  const float* sensor_transform = sensor_to_world + 16 * sensor;
  const float3 world_origin = make_float3(
      sensor_transform[3],
      sensor_transform[7],
      sensor_transform[11]);
  const float3 world_direction = make_float3(
      sensor_transform[0] * local_sensor_direction.x +
          sensor_transform[1] * local_sensor_direction.y +
          sensor_transform[2] * local_sensor_direction.z,
      sensor_transform[4] * local_sensor_direction.x +
          sensor_transform[5] * local_sensor_direction.y +
          sensor_transform[6] * local_sensor_direction.z,
      sensor_transform[8] * local_sensor_direction.x +
          sensor_transform[9] * local_sensor_direction.y +
          sensor_transform[10] * local_sensor_direction.z);
  const float* scene_transform = scene_to_world + 16 * sensor;
  const float3 scene_translation = make_float3(
      scene_transform[3],
      scene_transform[7],
      scene_transform[11]);
  const float3 translated_origin = sub3(world_origin, scene_translation);
  const float3 local_origin = make_float3(
      scene_transform[0] * translated_origin.x +
          scene_transform[4] * translated_origin.y +
          scene_transform[8] * translated_origin.z,
      scene_transform[1] * translated_origin.x +
          scene_transform[5] * translated_origin.y +
          scene_transform[9] * translated_origin.z,
      scene_transform[2] * translated_origin.x +
          scene_transform[6] * translated_origin.y +
          scene_transform[10] * translated_origin.z);
  const float3 local_direction = make_float3(
      scene_transform[0] * world_direction.x +
          scene_transform[4] * world_direction.y +
          scene_transform[8] * world_direction.z,
      scene_transform[1] * world_direction.x +
          scene_transform[5] * world_direction.y +
          scene_transform[9] * world_direction.z,
      scene_transform[2] * world_direction.x +
          scene_transform[6] * world_direction.y +
          scene_transform[10] * world_direction.z);

  uint64_t node_visits = 0;
  uint64_t primitive_tests = 0;
  uint64_t candidate_count = 0;
  uint64_t return_count = 0;
  bool stack_overflow = false;
  float previous_cluster_end = config.near_plane - 1.0e-6f;
  const int64_t output_base = (static_cast<int64_t>(sensor) * rays + ray) * returns;
  for (int return_index = 0; return_index < returns; ++return_index) {
    float first_t = config.far_plane;
    if (!find_nearest_candidate(
            scene,
            local_origin,
            local_direction,
            config,
            previous_cluster_end,
            &first_t,
            &node_visits,
            &primitive_tests,
            &candidate_count,
            &stack_overflow)) {
      break;
    }
    const float cluster_end = fminf(
        config.far_plane,
        first_t + fmaxf(config.cluster_abs, config.cluster_relative * first_t));
    const Reduction reduction = reduce_cluster(
        scene,
        local_origin,
        local_direction,
        config,
        first_t,
        cluster_end,
        &node_visits,
        &primitive_tests,
        &stack_overflow);
    if (stack_overflow) {
      break;
    }
    if (reduction.semantic_overflow) {
      add_counter(counters, kSemanticOverflow);
      break;
    }
    if (!reduction.valid) {
      break;
    }
    const int64_t output_index = output_base + return_index;
    ranges[output_index] = reduction.range;
    intensities[output_index] = reduction.intensity;
    semantics[output_index] = reduction.semantic;
    valid[output_index] = true;
    const float3 world_position = add3(world_origin, mul3(world_direction, reduction.range));
    positions[3 * output_index] = world_position.x;
    positions[3 * output_index + 1] = world_position.y;
    positions[3 * output_index + 2] = world_position.z;
    ++return_count;
    previous_cluster_end = cluster_end;
  }
  return_counts[static_cast<int64_t>(sensor) * rays + ray] = static_cast<int32_t>(return_count);
  add_counter(counters, kNodeVisits, node_visits);
  add_counter(counters, kLeafTests, primitive_tests);
  add_counter(counters, kCandidates, candidate_count);
  add_counter(counters, kReturns, return_count);
  if (stack_overflow) {
    add_counter(counters, kStackOverflow);
  }
}

}  // namespace

int64_t lidar_sort_temp_bytes_cuda(int64_t num_items) {
  std::size_t temp_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_bytes,
      static_cast<const uint64_t*>(nullptr),
      static_cast<uint64_t*>(nullptr),
      static_cast<const int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      num_items,
      0,
      30));
  return static_cast<int64_t>(temp_bytes);
}

void build_scene_lbvh_cuda(
    const std::vector<torch::Tensor>& scene,
    const std::vector<torch::Tensor>& build,
    int64_t packet_size,
    int64_t leaf_count,
    int64_t leaf_capacity,
    double support_sigma,
    double planarity_ratio_max) {
  c10::cuda::CUDAGuard guard(scene[0].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(scene[0].get_device());
  const int gaussian_count = static_cast<int>(scene[0].size(0));
  constexpr int threads = 256;
  compute_morton_codes_kernel<<<(gaussian_count + threads - 1) / threads, threads, 0, stream>>>(
      scene[0].data_ptr<float>(),
      gaussian_count,
      build[5].data_ptr<float>(),
      build[0].data_ptr<uint64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  std::size_t temp_bytes = static_cast<std::size_t>(build[4].numel());
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      build[4].data_ptr<uint8_t>(),
      temp_bytes,
      build[0].data_ptr<uint64_t>(),
      build[1].data_ptr<uint64_t>(),
      build[2].data_ptr<int32_t>(),
      build[3].data_ptr<int32_t>(),
      gaussian_count,
      0,
      30,
      stream));
  build_leaf_bounds_kernel<<<
      (leaf_capacity + threads - 1) / threads,
      threads,
      0,
      stream>>>(
      scene[0].data_ptr<float>(),
      scene[1].data_ptr<float>(),
      scene[2].data_ptr<float>(),
      build[3].data_ptr<int32_t>(),
      gaussian_count,
      static_cast<int>(packet_size),
      static_cast<int>(leaf_count),
      static_cast<int>(leaf_capacity),
      static_cast<float>(support_sigma),
      static_cast<float>(planarity_ratio_max),
      build[6].data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int width = static_cast<int>(leaf_capacity);
  while (width > 1) {
    const int parent_count = width / 2;
    const int parent_start = parent_count - 1;
    build_internal_bounds_kernel<<<
        (parent_count + threads - 1) / threads,
        threads,
        0,
        stream>>>(build[6].data_ptr<float>(), parent_start, parent_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    width = parent_count;
  }
}

void pack_scene_descriptor_cuda(
    torch::Tensor descriptors,
    int64_t slot,
    int64_t scene_id,
    const std::vector<torch::Tensor>& scene,
    int64_t leaf_count,
    int64_t leaf_capacity) {
  c10::cuda::CUDAGuard guard(descriptors.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(descriptors.get_device());
  const float* reflectivity = scene[5].numel() ? scene[5].data_ptr<float>() : nullptr;
  const float* confidence = scene[6].numel() ? scene[6].data_ptr<float>() : nullptr;
  pack_descriptor_kernel<<<1, 1, 0, stream>>>(
      descriptors.data_ptr<int64_t>(),
      static_cast<int>(slot),
      scene_id,
      static_cast<int>(scene[0].size(0)),
      static_cast<int>(leaf_count),
      static_cast<int>(leaf_capacity),
      scene[0].data_ptr<float>(),
      scene[1].data_ptr<float>(),
      scene[2].data_ptr<float>(),
      scene[3].data_ptr<float>(),
      scene[4].data_ptr<int64_t>(),
      reflectivity,
      confidence,
      scene[7].data_ptr<int32_t>(),
      scene[8].data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int64_t semantic_slots) {
  c10::cuda::CUDAGuard guard(descriptors.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(descriptors.get_device());
  const int rays = static_cast<int>(inputs[0].size(0));
  const int batch = static_cast<int>(inputs[2].size(0));
  const int active_count = static_cast<int>(inputs[5].numel());
  const int64_t total_returns = static_cast<int64_t>(batch) * rays * returns;
  constexpr int threads = 256;
  begin_call_kernel<<<1, 1, 0, stream>>>(counters.data_ptr<int64_t>());
  clear_outputs_kernel<<<
      (total_returns + threads - 1) / threads,
      threads,
      0,
      stream>>>(
      total_returns,
      rays,
      static_cast<int>(returns),
      inputs[1].scalar_type() == torch::kInt32 ? inputs[1].data_ptr<int32_t>() : nullptr,
      inputs[1].scalar_type() == torch::kInt64 ? inputs[1].data_ptr<int64_t>() : nullptr,
      inputs[1].scalar_type() == torch::kInt64,
      outputs[0].data_ptr<float>(),
      outputs[1].data_ptr<float>(),
      outputs[2].data_ptr<float>(),
      outputs[3].data_ptr<int64_t>(),
      outputs[4].data_ptr<bool>(),
      outputs[5].data_ptr<int64_t>(),
      outputs[6].data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  TraceConfig config{
      static_cast<float>(near_plane_m),
      static_cast<float>(far_plane_m),
      static_cast<float>(support_sigma),
      static_cast<float>(support_sigma * support_sigma),
      static_cast<float>(detection_threshold),
      static_cast<float>(planarity_ratio_max),
      static_cast<float>(min_incidence_cos),
      static_cast<float>(cluster_abs_m),
      static_cast<float>(cluster_relative),
      static_cast<float>(fallback_reflectivity),
      static_cast<float>(direction_norm_tolerance),
      static_cast<int>(packet_size),
      static_cast<int>(semantic_slots)};
  const int64_t work_count = static_cast<int64_t>(active_count) * rays;
  if (work_count > 0) {
    trace_lidar_kernel<<<
        (work_count + threads - 1) / threads,
        threads,
        0,
        stream>>>(
      descriptors.data_ptr<int64_t>(),
      static_cast<int>(scene_count),
      inputs[0].data_ptr<float>(),
      rays,
      inputs[2].data_ptr<float>(),
      inputs[3].data_ptr<float>(),
      inputs[4].data_ptr<int64_t>(),
      inputs[5].data_ptr<int64_t>(),
      active_count,
      batch,
      static_cast<int>(returns),
      config,
      outputs[0].data_ptr<float>(),
      outputs[1].data_ptr<float>(),
      outputs[2].data_ptr<float>(),
      outputs[3].data_ptr<int64_t>(),
      outputs[4].data_ptr<bool>(),
      outputs[6].data_ptr<int32_t>(),
      counters.data_ptr<int64_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  end_call_kernel<<<1, 1, 0, stream>>>(counters.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
