#pragma once

#include <vector>

#include "cuda/cudamemory.h"

namespace idaten
{
	class RadixSort {
		friend class LBVH;
		friend class LBVHBuilder;

	private:
		RadixSort() {}
		~RadixSort();

	public:
		static void sort(
			const std::vector<uint32_t>& values,
			TypedCudaMemory<uint32_t>& dst,
			std::vector<uint32_t>* result = nullptr);

		void sort(
			TypedCudaMemory<uint32_t>& values,
			TypedCudaMemory<uint32_t>& indices,
			TypedCudaMemory<uint32_t>& dst,
			std::vector<uint32_t>* result = nullptr);

		void init(uint32_t valueNum, uint32_t indexNum);

	private:
		// NOTE
		// Avoid "thrust" code in header file, because compile "thrust" is too heavy...
		void* m_deviceKeys{ nullptr };
		void* m_deviceIndices{ nullptr };
	};
}
