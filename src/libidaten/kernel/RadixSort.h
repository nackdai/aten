#pragma once

#include <vector>

#include "cuda/cudamemory.h"

namespace idaten
{
	class RadixSort {
		friend class LBVHBuilder;

	public:
		RadixSort() {}
		~RadixSort();

	public:
		static void sort(
			const std::vector<uint32_t>& keys,
			const std::vector<uint32_t>& values,
			TypedCudaMemory<uint32_t>& refSortedKeys,
			TypedCudaMemory<uint32_t>& refSortedValues,
			std::vector<uint32_t>* resultHostKeys = nullptr,
			std::vector<uint32_t>* resultHostValues = nullptr);

		void sort(
			uint32_t num,
			TypedCudaMemory<uint32_t>& keys,
			TypedCudaMemory<uint32_t>& values,
			std::vector<uint32_t>* resultHostKeys = nullptr,
			std::vector<uint32_t>* resultHostValues = nullptr);

		void init(uint32_t num);

		void sortWith64Bit(
			uint32_t num,
			TypedCudaMemory<uint64_t>& keys,
			TypedCudaMemory<uint32_t>& values,
			std::vector<uint64_t>* resultHostKeys = nullptr,
			std::vector<uint32_t>* resultHostValues = nullptr);

		void initWith64Bit(uint32_t num);

	private:
		bool m_32bit{ true };
	};
}
