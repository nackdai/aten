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
	RadixSort::~RadixSort()
	{
		if (m_deviceKeys) {
			thrust::device_vector<uint32_t>& deviceKeys = *(thrust::device_vector<uint32_t>*)(m_deviceKeys);
			deviceKeys.clear();
			delete m_deviceKeys;
		}
		if (m_deviceIndices) {
			thrust::device_vector<uint32_t>& deviceIndices = *(thrust::device_vector<uint32_t>*)(m_deviceIndices);
			deviceIndices.clear();
			delete m_deviceIndices;
		}
	}

	void RadixSort::init(uint32_t valueNum, uint32_t indexNum)
	{
		if (!m_deviceKeys) {
			m_deviceKeys = new thrust::device_vector<uint32_t>(valueNum);
		}

		if (!m_deviceIndices) {
			m_deviceIndices = new thrust::device_vector<uint32_t>(indexNum);
		}
	}

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
		AT_ASSERT(m_deviceKeys);
		AT_ASSERT(m_deviceIndices);

		// copy unsorted data from host to device
		thrust::device_vector<uint32_t>& deviceKeys = *(thrust::device_vector<uint32_t>*)(m_deviceKeys);
		thrust::device_vector<uint32_t>& deviceIndices = *(thrust::device_vector<uint32_t>*)(m_deviceIndices);

		auto keys = thrust::raw_pointer_cast(deviceKeys.data());
		checkCudaErrors(cudaMemcpyAsync(keys, values.ptr(), values.bytes(), cudaMemcpyDeviceToDevice));

		auto ids = thrust::raw_pointer_cast(deviceIndices.data());
		checkCudaErrors(cudaMemcpyAsync(ids, indices.ptr(), indices.bytes(), cudaMemcpyDeviceToDevice));

		thrust::sort_by_key(deviceKeys.begin(), deviceKeys.begin() + values.num(), deviceIndices.begin());

		auto sortedKeys = thrust::raw_pointer_cast(deviceKeys.data());

		dst.init(deviceKeys.size() * sizeof(uint32_t));
		dst.writeByNum(sortedKeys, values.num());

		if (result) {
			thrust::host_vector<uint32_t> hostKeys = deviceKeys;
			for (int i = 0; i < values.num(); i++) {
				result->push_back(hostKeys[i]);
			}
		}
	}

}