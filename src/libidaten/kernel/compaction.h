#pragma once

#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

namespace idaten {
	class Compaction {
	private:
		Compaction();
		~Compaction();

	public:
		static void init(
			int maxInputNum,
			int blockSize);

		static void clear();

		static void compact(
			idaten::TypedCudaMemory<int>& dst,
			idaten::TypedCudaMemory<int>& bools,
			int* result = nullptr);

		static idaten::TypedCudaMemory<int>& getCount();

#if 0
		// test implementation.
		static void compact();
#endif
	};
}