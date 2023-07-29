#include <vector>
#include <numeric>

#include "kernel/StreamCompaction.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

// NOTE
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
// https://github.com/bcrusco/CUDA-Path-Tracer/blob/master/stream_compaction/efficient.cu

// ブロック単位で計算した exclusiveScan の総和値を足したものを計算する.
__global__ void computeBlockCount(
    int32_t* dst,
    int32_t num,    // block count per grid used in exclusiveScan.
    int32_t stride,    // thread count per block used in exclusiveScan.
    const int32_t* src0,
    const int32_t* src1)
{
    int32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= num) {
        return;
    }

    if (index == 0) {
        dst[index] = 0;
    }
    else {
        dst[index] = src0[index * stride - 1] + src1[index * stride - 1];
    }
}

// ブロックごとに前のブロックまでの exclusiveScan の総和値を足したものを加算する.
__global__ void incrementBlocks(
    int32_t* data,
    int32_t num,
    const int32_t* incr)    // value to increment for each blocks.
{
    int32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= num) {
        return;
    }

    data[index] += incr[blockIdx.x];
}

__global__ void exclusiveScan(int32_t* dst, int32_t num, int32_t stride, const int32_t* src)
{
    extern __shared__ int32_t temp[];

    int32_t index = threadIdx.x;
    int32_t offset = 1;

    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n * 2 >= num) {
        return;
    }

    // Copy input data to shared memory
    temp[2 * index] = src[2 * index + (blockIdx.x * blockDim.x * 2)];
    temp[2 * index + 1] = src[2 * index + 1 + (blockIdx.x * blockDim.x * 2)];

    // Up sweep
    for (int32_t d = stride >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (index < d) {
            int32_t ai = offset * (2 * index + 1) - 1;
            int32_t bi = offset * (2 * index + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the root
    if (index == 0) {
        temp[stride - 1] = 0;
    }

    // Down sweep
    for (int32_t d = 1; d < stride; d *= 2) {
        offset >>= 1;
        __syncthreads();

        if (index < d && offset > 0) {
            int32_t ai = offset * (2 * index + 1) - 1;
            int32_t bi = offset * (2 * index + 2) - 1;
            int32_t t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write to output array
    dst[2 * index + (blockIdx.x * blockDim.x * 2)] = temp[2 * index];
    dst[2 * index + 1 + (blockIdx.x * blockDim.x * 2)] = temp[2 * index + 1];
}

__global__ void scatter(
    int32_t* dst,
    int32_t* count,
    int32_t num,
    const int32_t* bools,
    const int32_t* indices,
    const int32_t* src)
{
    int32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= num) {
        return;
    }

    if (bools[idx] > 0) {
        int32_t pos = indices[idx];
        dst[pos] = src[idx];
    }

    if (idx == 0) {
        *count = bools[num - 1] + indices[num - 1];
    }
}

namespace idaten
{
    idaten::TypedCudaMemory<int32_t>& StreamCompaction::getCount()
    {
        return m_counts;
    }

    void StreamCompaction::init(
        int32_t maxInputNum,
        int32_t blockSize)
    {
        if (m_maxInputNum == 0) {
            m_maxInputNum = maxInputNum;
            m_blockSize = blockSize;

            int32_t blockPerGrid = (maxInputNum - 1) / blockSize + 1;

            m_increments.resize(blockPerGrid);
            m_tmp.resize(blockPerGrid);
            m_work.resize(blockPerGrid);

            m_indices.resize(m_maxInputNum);

            std::vector<int32_t> iota(m_maxInputNum);
            std::iota(iota.begin(), iota.end(), 0);

            m_iota.resize(iota.size());
            m_iota.writeFromHostToDeviceByNum(&iota[0], iota.size());

            m_counts.resize(1);
        }
    }

    void StreamCompaction::clear()
    {
        m_maxInputNum = 0;
        m_blockSize = 0;

        m_increments.free();
        m_tmp.free();
        m_work.free();

        m_indices.free();
        m_iota.free();
        m_counts.free();
    }

    void StreamCompaction::scan(
        const int32_t blocksize,
        idaten::TypedCudaMemory<int32_t>& src,
        idaten::TypedCudaMemory<int32_t>& dst)
    {
        AT_ASSERT(dst.num() <= m_maxInputNum);

        int32_t blockPerGrid = (dst.num() - 1) / blocksize + 1;

        exclusiveScan << <blockPerGrid, blocksize / 2, blocksize * sizeof(int32_t), m_stream >> > (
            dst.data(),
            dst.num(),
            blocksize,
            src.data());

        checkCudaKernel(exclusiveScan);

        if (blockPerGrid <= 1) {
            // If number of block is 1, finish.
            return;
        }

        int32_t tmpBlockPerGrid = (blockPerGrid - 1) / blocksize + 1;
        int32_t tmpBlockSize = blockPerGrid;

        computeBlockCount << <tmpBlockPerGrid, tmpBlockSize, 0, m_stream >> > (
            m_increments.data(),
            m_increments.num(),
            blocksize,
            src.data(),
            dst.data());

        checkCudaKernel(computeBlockCount);

        idaten::TypedCudaMemory<int32_t>* input = &m_increments;
        idaten::TypedCudaMemory<int32_t>* output = &m_tmp;

        idaten::TypedCudaMemory<int32_t>* tmpptr = &m_tmp;

        int32_t elementNum = blockPerGrid;

        int32_t count = 1;
        int32_t innerBlockPerGrid = 0;

        std::vector<int32_t> stackBlockPerGrid;

        // Scan blocks.
        for (;;) {
            innerBlockPerGrid = (elementNum - 1) / blocksize + 1;
            stackBlockPerGrid.push_back(elementNum);

            exclusiveScan << <innerBlockPerGrid, blocksize / 2, blocksize * sizeof(int32_t), m_stream >> >(
                m_work.data(),
                m_work.num(),
                blocksize,
                input->data());

            checkCudaKernel(iterate_exclusiveScan);

            if (innerBlockPerGrid <= 1) {
                //cudaMemcpyAsync(tmp.data(), work.data(), work.bytes(), cudaMemcpyAsyncDeviceToDevice);
                tmpptr = &m_work;
                break;
            }

            int32_t innerTmpBlockPerGrid = (innerBlockPerGrid - 1) / blocksize + 1;
            int32_t innerTmpBlockSize = innerBlockPerGrid;

            computeBlockCount << <innerTmpBlockPerGrid, innerTmpBlockSize, 0, m_stream >> > (
                output->data(),
                output->num(),
                blocksize,
                input->data(),
                m_work.data());

            checkCudaKernel(iterate_computeBlockCount);

            // swap.
            auto p = input;
            input = output;
            output = p;

            elementNum = innerBlockPerGrid;
            count++;
        }

#if 1
        input = tmpptr;
        output = &m_increments;

        for (int32_t i = count - 1; i >= 0; i--) {
            // blocks per grid.
            auto bpg = stackBlockPerGrid[i];

            auto threadPerBlock = (output->num() + bpg - 1) / bpg;

            incrementBlocks << <bpg, threadPerBlock, 0, m_stream >> > (
                output->data(),
                output->num(),
                input->data());

            checkCudaKernel(iterate_incrementBlocks);

            // swap.
            auto p = input;
            input = output;
            output = p;
        }

        idaten::TypedCudaMemory<int32_t>* incrResult = (count & 0x1 == 0 ? tmpptr : &m_increments);
#endif

        incrementBlocks << <blockPerGrid, blocksize, 0, m_stream >> > (
            dst.data(),
            dst.num(),
            incrResult->data());

        checkCudaKernel(incrementBlocks);
    }

    void StreamCompaction::compact(
        idaten::TypedCudaMemory<int32_t>& dst,
        idaten::TypedCudaMemory<int32_t>& bools,
        int32_t* result/*= nullptr*/)
    {
        scan(m_blockSize, bools, m_indices);

        int32_t num = dst.num();
        int32_t blockPerGrid = (num - 1) / m_blockSize + 1;

        scatter << <blockPerGrid, m_blockSize, 0, m_stream >> > (
            dst.data(),
            m_counts.data(),
            dst.num(),
            bools.data(),
            m_indices.data(),
            m_iota.data());

        if (result) {
            m_counts.readFromDeviceToHostByNum(result);
        }
    }

#if 0
    // test implementation.
    void StreamCompaction::compact()
    {
#if 1
        const int32_t blocksize = m_blockSize;

        int32_t f[] = { 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3 };
        //int32_t f[] = { 3, 1, 7, 0, 4, 1, 6, 3, 3, 1 };
        //int32_t f[] = { 3, 1, 7, 0, 4, 1, 6, 3 };
        //int32_t f[] = { 0, 25, 25, 25 };

        //int32_t c = aten::nextPow2(AT_COUNTOF(f));
        int32_t c = AT_COUNTOF(f);

        std::vector<int32_t> x(c);
        memcpy(&x[0], f, sizeof(int32_t) * AT_COUNTOF(f));

        idaten::TypedCudaMemory<int32_t> src;
        src.init(x.size());
        src.writeFromHostToDeviceByNum(&x[0], x.size());

        idaten::TypedCudaMemory<int32_t> dst;
        dst.init(x.size());

        scan(blocksize, src, dst);

        std::vector<int32_t> buffer(x.size());
        dst.readFromDeviceToHostByNum(&buffer[0]);

        int32_t xxx = 0;
#else
        const int32_t blocksize = m_blockSize;

        int32_t b[] = { 1, 0, 1, 0, 1, 0, 1, 0 };
        int32_t v[] = { 0, 1, 2, 3, 4, 5, 6, 7 };

        AT_ASSERT(AT_COUNTOF(b) == AT_COUNTOF(v));

        int32_t num = AT_COUNTOF(b);

        std::vector<int32_t> buffer(num);

        idaten::TypedCudaMemory<int32_t> bools;
        bools.init(num);
        bools.writeFromHostToDeviceByNum(b, num);

        idaten::TypedCudaMemory<int32_t> indices;
        indices.init(num);

        scan(blocksize, bools, indices);

        indices.readFromDeviceToHostByNum(&buffer[0]);

        idaten::TypedCudaMemory<int32_t> values;
        values.init(num);
        values.writeFromHostToDeviceByNum(v, num);

        idaten::TypedCudaMemory<int32_t> dst;
        dst.init(num);

        idaten::TypedCudaMemory<int32_t> count;
        count.init(1);

        int32_t blockPerGrid = (num - 1) / blocksize + 1;

        scatter << <blockPerGrid, blocksize >> > (
            dst.data(),
            count.data(),
            dst.maxNum(),
            bools.data(),
            indices.data(),
            values.data());

        dst.readFromDeviceToHostByNum(&buffer[0]);

        int32_t _count = -1;
        count.readFromDeviceToHostByNum(&_count);

        int32_t xxx = 0;
#endif
    }
#endif
}
