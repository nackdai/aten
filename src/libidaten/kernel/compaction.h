#pragma once

#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

namespace idaten {
	class StreamCompaction {
	public:
		StreamCompaction() {}
		~StreamCompaction() {}

	public:
		void init(
			int maxInputNum,
			int blockSize);

		void clear();

		void compact(
			idaten::TypedCudaMemory<int>& dst,
			idaten::TypedCudaMemory<int>& bools,
			int* result = nullptr);

		idaten::TypedCudaMemory<int>& getCount();

		void setStream(cudaStream_t stream)
		{
			m_stream = stream;
		}

#if 0
		// test implementation.
		void compact();
#endif

	private:
		void scan(
			const int blocksize,
			idaten::TypedCudaMemory<int>& src,
			idaten::TypedCudaMemory<int>& dst);

	private:
		int m_maxInputNum{ 0 };
		int m_blockSize{ 0 };

		idaten::TypedCudaMemory<int> m_increments;
		idaten::TypedCudaMemory<int> m_tmp;
		idaten::TypedCudaMemory<int> m_work;

		idaten::TypedCudaMemory<int> m_indices;
		idaten::TypedCudaMemory<int> m_iota;
		idaten::TypedCudaMemory<int> m_counts;

		cudaStream_t m_stream{ (cudaStream_t)0 };
	};
}