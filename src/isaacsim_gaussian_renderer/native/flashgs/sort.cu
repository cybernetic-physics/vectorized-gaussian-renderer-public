#include "ops.h"

#include <cub/device/device_radix_sort.cuh>
#include <stdexcept>

namespace flashgs_adapter {

void sort_gaussian_fixed(
    int capacity,
    char* sorting_space,
    size_t sorting_size,
    uint64_t* keys_unsorted,
    uint32_t* values_unsorted,
    uint64_t* keys_sorted,
    uint32_t* values_sorted,
    int end_bit,
    cudaStream_t stream) {
  const cudaError_t status = cub::DeviceRadixSort::SortPairs(
      sorting_space,
      sorting_size,
      keys_unsorted,
      keys_sorted,
      values_unsorted,
      values_sorted,
      capacity,
      0,
      end_bit,
      stream);
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

size_t get_sort_buffer_size(int capacity, cudaStream_t stream) {
  size_t sorting_size = 0;
  const cudaError_t status = cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>(
      nullptr,
      sorting_size,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      capacity,
      0,
      64,
      stream);
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
  return sorting_size;
}

}  // namespace flashgs_adapter
