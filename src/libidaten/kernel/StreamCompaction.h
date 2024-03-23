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
            int32_t maxInputNum,
            int32_t blockSize);

        void clear();

        void compact(
            idaten::TypedCudaMemory<int32_t>& dst,
            idaten::TypedCudaMemory<int32_t>& bools,
            int32_t* result = nullptr);

        idaten::TypedCudaMemory<int32_t>& getCount();

        void SetStream(cudaStream_t stream)
        {
            m_stream = stream;
        }

#if 0
        // test implementation.
        void compact();
#endif

    private:
        void scan(
            const int32_t blocksize,
            idaten::TypedCudaMemory<int32_t>& src,
            idaten::TypedCudaMemory<int32_t>& dst);

    private:
        int32_t m_maxInputNum{ 0 };
        int32_t m_blockSize{ 0 };

        idaten::TypedCudaMemory<int32_t> m_increments;
        idaten::TypedCudaMemory<int32_t> m_tmp;
        idaten::TypedCudaMemory<int32_t> m_work;

        idaten::TypedCudaMemory<int32_t> m_indices;
        idaten::TypedCudaMemory<int32_t> m_iota;
        idaten::TypedCudaMemory<int32_t> m_counts;

        cudaStream_t m_stream{ (cudaStream_t)0 };
    };
}
