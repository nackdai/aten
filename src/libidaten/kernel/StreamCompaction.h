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

        /**
         * @brief Extract indices which specify true in bools.
         * @param[out] dst Destination to store indices.
         * @param[in] bools Array to store true or false (i.e. 1 or 0).
         */
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

        idaten::TypedCudaMemory<int32_t> exclusive_scan_array_;
        idaten::TypedCudaMemory<int32_t> m_counts;

        cudaStream_t m_stream{ (cudaStream_t)0 };
    };
}
