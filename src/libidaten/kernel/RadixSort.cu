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
		if (m_deviceValues) {
			thrust::device_vector<uint32_t>& deviceValues = *(thrust::device_vector<uint32_t>*)(m_deviceValues);
			deviceValues.clear();
			delete m_deviceValues;
		}
	}

	void RadixSort::init(uint32_t num)
	{
		if (!m_deviceKeys) {
			m_deviceKeys = new thrust::device_vector<uint32_t>(num);
		}

		if (!m_deviceValues) {
			m_deviceValues = new thrust::device_vector<uint32_t>(num);
		}
	}

	static void radixSort(
		uint32_t num,
		thrust::device_vector<uint32_t> deviceKeys,
		thrust::device_vector<uint32_t> deviceValues,
		TypedCudaMemory<uint32_t>& refSortedKeys,
		TypedCudaMemory<uint32_t>& refSortedValues,
		std::vector<uint32_t>* resultHostKeys/*= nullptr*/,
		std::vector<uint32_t>* resultHostValues/*= nullptr*/)
	{
		thrust::sort_by_key(deviceKeys.begin(), deviceKeys.begin() + num, deviceValues.begin());

		auto sortedKeys = thrust::raw_pointer_cast(deviceKeys.data());
		auto sortedValues = thrust::raw_pointer_cast(deviceValues.data());

		refSortedKeys.init(deviceKeys.size() * sizeof(uint32_t));
		refSortedKeys.writeByNum(sortedKeys, num);

		refSortedValues.init(deviceValues.size() * sizeof(uint32_t));
		refSortedValues.writeByNum(sortedValues, num);

		if (resultHostKeys) {
			thrust::host_vector<uint32_t> hostKeys = deviceKeys;
			for (int i = 0; i < num; i++) {
				resultHostKeys->push_back(hostKeys[i]);
			}
		}

		if (resultHostValues) {
			thrust::host_vector<uint32_t> hostValues = deviceValues;
			for (int i = 0; i < num; i++) {
				resultHostValues->push_back(hostValues[i]);
			}
		}
	}

	void RadixSort::sort(
		const std::vector<uint32_t>& keys,
		const std::vector<uint32_t>& values,
		TypedCudaMemory<uint32_t>& refSortedKeys,
		TypedCudaMemory<uint32_t>& refSortedValues,
		std::vector<uint32_t>* resultHostKeys/*= nullptr*/,
		std::vector<uint32_t>* resultHostValues/*= nullptr*/)
	{
		AT_ASSERT(keys.size() == values.size());

		uint32_t num = (uint32_t)keys.size();

		thrust::host_vector<uint32_t> hostKeys(num);
		thrust::host_vector<uint32_t> hostValues(num);

		for (uint32_t i = 0; i < num; i++) {
			hostKeys[i] = keys[i];
			hostValues[i] = values[i];
		}

		// copy unsorted data from host to device
		thrust::device_vector<uint32_t> deviceKeys = hostKeys;
		thrust::device_vector<uint32_t> deviceValues = hostValues;

		radixSort(
			num,
			deviceKeys,
			deviceValues,
			refSortedKeys,
			refSortedValues,
			resultHostKeys,
			resultHostValues);
	}

	void RadixSort::sort(
		uint32_t num,
		TypedCudaMemory<uint32_t>& keys,
		TypedCudaMemory<uint32_t>& values,
		TypedCudaMemory<uint32_t>& refSortedKeys,
		TypedCudaMemory<uint32_t>& refSortedValues,
		std::vector<uint32_t>* resultHostKeys/*= nullptr*/,
		std::vector<uint32_t>* resultHostValues/*= nullptr*/)
	{
		AT_ASSERT(m_deviceKeys);
		AT_ASSERT(m_deviceValues);

		AT_ASSERT(keys.num() == values.num());
		AT_ASSERT(keys.num() <= num);

		// copy unsorted data from host to device
		thrust::device_vector<uint32_t>& deviceKeys = *(thrust::device_vector<uint32_t>*)(m_deviceKeys);
		thrust::device_vector<uint32_t>& deviceValues = *(thrust::device_vector<uint32_t>*)(m_deviceValues);

		auto dkeys = thrust::raw_pointer_cast(deviceKeys.data());
		checkCudaErrors(cudaMemcpyAsync(dkeys, keys.ptr(), keys.bytes(), cudaMemcpyDeviceToDevice));

		auto dvalues = thrust::raw_pointer_cast(deviceValues.data());
		checkCudaErrors(cudaMemcpyAsync(dvalues, values.ptr(), values.bytes(), cudaMemcpyDeviceToDevice));

		radixSort(
			num,
			deviceKeys,
			deviceValues,
			refSortedKeys,
			refSortedValues,
			resultHostKeys,
			resultHostValues);
	}
}