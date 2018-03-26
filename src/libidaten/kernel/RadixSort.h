#pragma once

#include <vector>

#include "cuda/cudamemory.h"

namespace idaten
{
	class RadixSort {
	private:
		RadixSort();
		~RadixSort();

	public:
		static void sort(
			const std::vector<uint32_t>& values,
			TypedCudaMemory<uint32_t>& dst,
			std::vector<uint32_t>* result = nullptr);
	};
}