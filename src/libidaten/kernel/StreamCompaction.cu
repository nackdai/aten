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
    int* dst,
    int num,    // block count per grid used in exclusiveScan.
    int stride,    // thread count per block used in exclusiveScan.
    const int* src0,
    const int* src1)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

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
    int* data,
    int num,
    const int* incr)    // value to increment for each blocks.
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= num) {
        return;
    }

    data[index] += incr[blockIdx.x];
}

__global__ void exclusiveScan(int* dst, int num, int stride, const int* src)
{
    extern __shared__ int temp[];

    int index = threadIdx.x;
    int offset = 1;

    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n * 2 >= num) {
        return;
    }

    // Copy input data to shared memory
    temp[2 * index] = src[2 * index + (blockIdx.x * blockDim.x * 2)];
    temp[2 * index + 1] = src[2 * index + 1 + (blockIdx.x * blockDim.x * 2)];

    // Up sweep
    for (int d = stride >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (index < d) {
            int ai = offset * (2 * index + 1) - 1;
            int bi = offset * (2 * index + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the root
    if (index == 0) {
        temp[stride - 1] = 0;
    }

    // Down sweep
    for (int d = 1; d < stride; d *= 2) {
        offset >>= 1;
        __syncthreads();

        if (index < d && offset > 0) {
            int ai = offset * (2 * index + 1) - 1;
            int bi = offset * (2 * index + 2) - 1;
            int t = temp[ai];
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
    int* dst,
    int* count,
    int num,
    const int* bools,
    const int* indices,
    const int* src)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= num) {
        return;
    }

    if (bools[idx] > 0) {
        int pos = indices[idx];
        dst[pos] = src[idx];
    }

    if (idx == 0) {
        *count = bools[num - 1] + indices[num - 1];
    }
}

namespace idaten
{
    idaten::TypedCudaMemory<int>& StreamCompaction::getCount()
    {
        return m_counts;
    }

    void StreamCompaction::init(
        int maxInputNum,
        int blockSize)
    {
        if (m_maxInputNum == 0) {
            m_maxInputNum = maxInputNum;
            m_blockSize = blockSize;

            int blockPerGrid = (maxInputNum - 1) / blockSize + 1;

            m_increments.init(blockPerGrid);
            m_tmp.init(blockPerGrid);
            m_work.init(blockPerGrid);

            m_indices.init(m_maxInputNum);

            std::vector<int> iota(m_maxInputNum);
            std::iota(iota.begin(), iota.end(), 0);

            m_iota.init(iota.size());
            m_iota.writeByNum(&iota[0], iota.size());

            m_counts.init(1);
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
        const int blocksize,
        idaten::TypedCudaMemory<int>& src,
        idaten::TypedCudaMemory<int>& dst)
    {
        AT_ASSERT(dst.num() <= m_maxInputNum);

        int blockPerGrid = (dst.num() - 1) / blocksize + 1;

        exclusiveScan << <blockPerGrid, blocksize / 2, blocksize * sizeof(int), m_stream >> > (
            dst.ptr(),
            dst.num(),
            blocksize,
            src.ptr());

        checkCudaKernel(exclusiveScan);

        if (blockPerGrid <= 1) {
            // If number of block is 1, finish.
            return;
        }

        int tmpBlockPerGrid = (blockPerGrid - 1) / blocksize + 1;
        int tmpBlockSize = blockPerGrid;

        computeBlockCount << <tmpBlockPerGrid, tmpBlockSize, 0, m_stream >> > (
            m_increments.ptr(),
            m_increments.num(),
            blocksize,
            src.ptr(),
            dst.ptr());

        checkCudaKernel(computeBlockCount);

        idaten::TypedCudaMemory<int>* input = &m_increments;
        idaten::TypedCudaMemory<int>* output = &m_tmp;

        idaten::TypedCudaMemory<int>* tmpptr = &m_tmp;

        int elementNum = blockPerGrid;

        int count = 1;
        int innerBlockPerGrid = 0;

        std::vector<int> stackBlockPerGrid;

        // Scan blocks.
        for (;;) {
            innerBlockPerGrid = (elementNum - 1) / blocksize + 1;
            stackBlockPerGrid.push_back(elementNum);

            exclusiveScan << <innerBlockPerGrid, blocksize / 2, blocksize * sizeof(int), m_stream >> >(
                m_work.ptr(),
                m_work.num(),
                blocksize,
                input->ptr());

            checkCudaKernel(iterate_exclusiveScan);

            if (innerBlockPerGrid <= 1) {
                //cudaMemcpyAsync(tmp.ptr(), work.ptr(), work.bytes(), cudaMemcpyAsyncDeviceToDevice);
                tmpptr = &m_work;
                break;
            }

            int innerTmpBlockPerGrid = (innerBlockPerGrid - 1) / blocksize + 1;
            int innerTmpBlockSize = innerBlockPerGrid;

            computeBlockCount << <innerTmpBlockPerGrid, innerTmpBlockSize, 0, m_stream >> > (
                output->ptr(),
                output->num(),
                blocksize,
                input->ptr(),
                m_work.ptr());

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

        for (int i = count - 1; i >= 0; i--) {
            // blocks per grid.
            auto bpg = stackBlockPerGrid[i];

            auto threadPerBlock = (output->num() + bpg - 1) / bpg;

            incrementBlocks << <bpg, threadPerBlock, 0, m_stream >> > (
                output->ptr(),
                output->num(),
                input->ptr());

            checkCudaKernel(iterate_incrementBlocks);

            // swap.
            auto p = input;
            input = output;
            output = p;
        }

        idaten::TypedCudaMemory<int>* incrResult = (count & 0x1 == 0 ? tmpptr : &m_increments);
#endif

        incrementBlocks << <blockPerGrid, blocksize, 0, m_stream >> > (
            dst.ptr(),
            dst.num(),
            incrResult->ptr());

        checkCudaKernel(incrementBlocks);
    }

    void StreamCompaction::compact(
        idaten::TypedCudaMemory<int>& dst,
        idaten::TypedCudaMemory<int>& bools,
        int* result/*= nullptr*/)
    {
        scan(m_blockSize, bools, m_indices);

        int num = dst.num();
        int blockPerGrid = (num - 1) / m_blockSize + 1;

        scatter << <blockPerGrid, m_blockSize, 0, m_stream >> > (
            dst.ptr(),
            m_counts.ptr(),
            dst.num(),
            bools.ptr(),
            m_indices.ptr(),
            m_iota.ptr());

        if (result) {
            m_counts.readByNum(result);
        }
    }

#if 0
    // test implementation.
    void StreamCompaction::compact()
    {
#if 1
        const int blocksize = m_blockSize;

        int f[] = { 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3 };
        //int f[] = { 3, 1, 7, 0, 4, 1, 6, 3, 3, 1 };
        //int f[] = { 3, 1, 7, 0, 4, 1, 6, 3 };
        //int f[] = { 0, 25, 25, 25 };

        //int c = aten::nextPow2(AT_COUNTOF(f));
        int c = AT_COUNTOF(f);

        std::vector<int> x(c);
        memcpy(&x[0], f, sizeof(int) * AT_COUNTOF(f));

        idaten::TypedCudaMemory<int> src;
        src.init(x.size());
        src.writeByNum(&x[0], x.size());

        idaten::TypedCudaMemory<int> dst;
        dst.init(x.size());

        scan(blocksize, src, dst);

        std::vector<int> buffer(x.size());
        dst.readByNum(&buffer[0]);

        int xxx = 0;
#else
        const int blocksize = m_blockSize;

        int b[] = { 1, 0, 1, 0, 1, 0, 1, 0 };
        int v[] = { 0, 1, 2, 3, 4, 5, 6, 7 };

        AT_ASSERT(AT_COUNTOF(b) == AT_COUNTOF(v));

        int num = AT_COUNTOF(b);

        std::vector<int> buffer(num);

        idaten::TypedCudaMemory<int> bools;
        bools.init(num);
        bools.writeByNum(b, num);

        idaten::TypedCudaMemory<int> indices;
        indices.init(num);

        scan(blocksize, bools, indices);

        indices.readByNum(&buffer[0]);

        idaten::TypedCudaMemory<int> values;
        values.init(num);
        values.writeByNum(v, num);

        idaten::TypedCudaMemory<int> dst;
        dst.init(num);

        idaten::TypedCudaMemory<int> count;
        count.init(1);

        int blockPerGrid = (num - 1) / blocksize + 1;

        scatter << <blockPerGrid, blocksize >> > (
            dst.ptr(),
            count.ptr(),
            dst.maxNum(),
            bools.ptr(),
            indices.ptr(),
            values.ptr());

        dst.readByNum(&buffer[0]);

        int _count = -1;
        count.readByNum(&_count);

        int xxx = 0;
#endif
    }
#endif
}
