#include "kernel/RadixSort.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"

namespace idaten
{
	void RadixSort::sort(
		const std::vector<uint32_t>& values,
		TypedCudaMemory<uint32_t>& dst,
		std::vector<uint32_t>* result/*= nullptr*/)
	{
		thrust::host_vector<uint32_t> hostKeys(values.size());
		thrust::host_vector<uint32_t> hostIndices(values.size());

		for (uint32_t i = 0; i < values.size(); i++) {
			hostKeys[i] = values[i];
			hostIndices[i] = i;
		}

		// copy unsorted data from host to device
		thrust::device_vector<uint32_t> deviceKeys = hostKeys;
		thrust::device_vector<uint32_t> deviceIndices = hostIndices;

		thrust::sort_by_key(deviceKeys.begin(), deviceKeys.end(), deviceIndices.begin());

		auto sortedKeys = thrust::raw_pointer_cast(deviceKeys.data());

		dst.init(deviceKeys.size() * sizeof(uint32_t));
		dst.writeByNum(sortedKeys, deviceKeys.size());

		if (result) {
			hostKeys = deviceKeys;
			for (int i = 0; i < hostKeys.size(); i++) {
				result->push_back(hostKeys[i]);
			}
		}
	}

	void RadixSort::sort(
		TypedCudaMemory<uint32_t>& values,
		TypedCudaMemory<uint32_t>& indices,
		TypedCudaMemory<uint32_t>& dst,
		std::vector<uint32_t>* result/*= nullptr*/)
	{
		// copy unsorted data from host to device
		thrust::device_vector<uint32_t> deviceKeys(values.num());
		thrust::device_vector<uint32_t> deviceIndices(indices.num());

		auto keys = thrust::raw_pointer_cast(deviceKeys.data());
		checkCudaErrors(cudaMemcpyAsync(keys, values.ptr(), values.bytes(), cudaMemcpyDeviceToDevice));

		auto ids = thrust::raw_pointer_cast(deviceIndices.data());
		checkCudaErrors(cudaMemcpyAsync(ids, indices.ptr(), indices.bytes(), cudaMemcpyDeviceToDevice));

		thrust::sort_by_key(deviceKeys.begin(), deviceKeys.end(), deviceIndices.begin());

		auto sortedKeys = thrust::raw_pointer_cast(deviceKeys.data());

		dst.init(deviceKeys.size() * sizeof(uint32_t));
		dst.writeByNum(sortedKeys, deviceKeys.size());

		if (result) {
			thrust::host_vector<uint32_t> hostKeys = deviceKeys;
			for (int i = 0; i < hostKeys.size(); i++) {
				result->push_back(hostKeys[i]);
			}
		}
	}

}