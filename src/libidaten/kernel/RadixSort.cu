#include "kernel/RadixSort.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"

namespace idaten
{

	RadixSort::~RadixSort()
	{
	}

	void RadixSort::init(uint32_t num)
	{
		m_32bit = true;
	}

	void RadixSort::initWith64Bit(uint32_t num)
	{
		m_32bit = false;
	}

	template <typename T>
	static void radixSort(
		uint32_t num,
		TypedCudaMemory<T>& keys,
		TypedCudaMemory<uint32_t>& values,
		std::vector<T>* resultHostKeys/*= nullptr*/,
		std::vector<uint32_t>* resultHostValues/*= nullptr*/)
	{
		thrust::sort_by_key(thrust::device, keys.ptr(), keys.ptr() + num, values.ptr());

		if (resultHostKeys) {
			resultHostKeys->resize(num);
			keys.readByNum(resultHostKeys, num);
		}

		if (resultHostValues) {
			resultHostValues->resize(num);
			values.readByNum(resultHostValues, num);
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

		thrust::sort_by_key(thrust::device, deviceKeys.begin(), deviceKeys.begin() + num, deviceValues.begin());

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
		uint32_t num,
		TypedCudaMemory<uint32_t>& keys,
		TypedCudaMemory<uint32_t>& values,
		std::vector<uint32_t>* resultHostKeys/*= nullptr*/,
		std::vector<uint32_t>* resultHostValues/*= nullptr*/)
	{
		AT_ASSERT(keys.num() == values.num());
		AT_ASSERT(keys.num() <= num);

		radixSort(
			num,
			keys,
			values,
			resultHostKeys,
			resultHostValues);
	}

	void RadixSort::sortWith64Bit(
		uint32_t num,
		TypedCudaMemory<uint64_t>& keys,
		TypedCudaMemory<uint32_t>& values,
		std::vector<uint64_t>* resultHostKeys/*= nullptr*/,
		std::vector<uint32_t>* resultHostValues/*= nullptr*/)
	{
		AT_ASSERT(keys.num() == values.num());
		AT_ASSERT(keys.num() <= num);

		radixSort(
			num,
			keys,
			values,
			resultHostKeys,
			resultHostValues);
	}
}