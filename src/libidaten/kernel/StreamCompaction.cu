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
// https://github.com/shineyruan/CUDA-Stream-Compaction

// ブロック単位で計算した exclusiveScan の総和値を足したものを計算する.
/**
 * @brief Add the result of exclusiveScan of 2 blocks.
 * @param[out] dst
 * @param[in] num Number of blocks per grid used in exclusiveScan.
 * @param[in] stride Number of threads per block used in exclusiveScan.
 * @param[in] src0
 * @param[in] src1
 */
__global__ void computeBlockCount(
    int32_t* dst,
    int32_t num,
    int32_t stride,
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

/**
 * @brief Execute exclusive scan.
 * @note https://github.com/shineyruan/CUDA-Stream-Compaction?tab=readme-ov-file#work-efficient-parallel-scan
 * @param[out] dst Output
 * @param[in] num Number of elements in boolean array.
 * @param[in] stride Block size (= number of thread per block).
 * @param[in] bools Boolean array.
 */
__global__ void exclusiveScan(int32_t* dst, int32_t num, int32_t stride, const int32_t* bools)
{
    extern __shared__ int32_t temp[];

    int32_t index = threadIdx.x;
    int32_t offset = 1;

    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n * 2 >= num) {
        return;
    }

    // - Copy input data to shared memory
    // All source data is stored in the one linear list.
    // The sequence is done per block (number of thread per block is blockIdx.x * blockDim.x).
    // So, we need to shift with the block size.
    // On the other hand, the block size (number of threads per block) to compute grid size and
    // the block size which is specified to this kernel are different.
    // The block size which is specified to this kernel is 1/2 of the block size to compute grid size.
    // Therefore, to compute the correct block size (i.e. number of threads per block), we need to multiply with 2.
    temp[2 * index] = bools[2 * index + (blockIdx.x * blockDim.x * 2)];
    temp[2 * index + 1] = bools[2 * index + 1 + (blockIdx.x * blockDim.x * 2)];

    // - Up sweep
    // This for loop means : for d = 0 to log_2(N).
    // N means the number of values which we need to care per block.
    // It means the number of threads per block. So, stride is block size itself.
    // Continuing to divide stride (i.e. the number of thread) by 2 means how many time we can divide stride by 2.
    // It means to compute log_2(N).
    for (int32_t d = stride >> 1; d > 0; d >>= 1) {
        // Sync threads in the block here.
        // So, all threads(= index) executes the following logic per one of loop.
        // e.g. d = 4: thread(i.e. index) = {0, 1, 2, ....}
        //      d = 2: thread(i.e. index) = {0, 1, 2, ....}
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

/**
 * @brief Execute scatter
 * @note https://github.com/shineyruan/CUDA-Stream-Compaction?tab=readme-ov-file#work-efficient-parallel-scan
 * @param[out] dst Result array of stream compaction
 * @param[out] count Number of extracted elements in the sream compaction array.
 * @param[in] num Number of elements in array of destination positions.
 * @param[in] bools Bloolean array.
 * @param[in] dst_positions Array of destination positions. i.e. Result array of exclusive scan.
 */
__global__ void scatter(
    int32_t* dst,
    int32_t* count,
    int32_t num,
    const int32_t* bools,
    const int32_t* dst_positions)
{
    int32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= num) {
        return;
    }

    if (bools[idx] > 0) {
        // If an element is marked as "1" in the boolean array,
        // store it into the corresponding indexed parallel scan position in the output array.
        int32_t pos = dst_positions[idx];
        dst[pos] = idx;
    }

    if (idx == 0) {
        *count = bools[num - 1] + dst_positions[num - 1];
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

            exclusive_scan_array_.resize(m_maxInputNum);

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

        exclusive_scan_array_.free();
        m_counts.free();
    }

    void StreamCompaction::scan(
        const int32_t blocksize,
        idaten::TypedCudaMemory<int32_t>& bools,
        idaten::TypedCudaMemory<int32_t>& dst)
    {
        AT_ASSERT(dst.num() <= m_maxInputNum);

        // (a + b - 1) / b = (a - 1) / b + b / b = (a - 1) / b + 1
        //  => (dst.num() + blocksize - 1) / blocksize = (dst.num() - 1) / blocksize + 1
        int32_t blockPerGrid = static_cast<int32_t>(dst.num() - 1) / blocksize + 1;

        // NOTE:
        // `blocksize * sizeof(int32_t)` means the bytes to allocate for the shared memory in exclusiveScan.
        // Two values are summed into one value. It means the number of values is reduced by 2 (i.e. 1/2).
        // So, we specify the block size as `blocksize / 2`.
        exclusiveScan << <blockPerGrid, blocksize / 2, blocksize * sizeof(int32_t), m_stream >> > (
            dst.data(),
            static_cast<int32_t>(dst.num()),
            blocksize,
            bools.data());

        checkCudaKernel(exclusiveScan);

        if (blockPerGrid <= 1) {
            // If number of block is 1, finish.
            return;
        }

        // ExclusiveScan is done per block. It means the exclusive scan is computed per block.
        // We need to unify the array of ExclusiveScan result and execute ExclusiveScan to the unified array.
        // Continue to unify the array until the number of block per grid is 1.
        // In that case, it means all arrays are unified and it's applied with ExclusiveScan entirely.

        // Subdivide block.
        int32_t tmpBlockPerGrid = (blockPerGrid - 1) / blocksize + 1;
        int32_t tmpBlockSize = blockPerGrid;

        computeBlockCount << <tmpBlockPerGrid, tmpBlockSize, 0, m_stream >> > (
            m_increments.data(),
            static_cast<int32_t>(m_increments.num()),
            blocksize,
            bools.data(),
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
                static_cast<int32_t>(m_work.num()),
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
                static_cast<int32_t>(output->num()),
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

            auto threadPerBlock = static_cast<int32_t>(output->num() + bpg - 1) / bpg;

            incrementBlocks << <bpg, threadPerBlock, 0, m_stream >> > (
                output->data(),
                static_cast<int32_t>(output->num()),
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
            static_cast<int32_t>(dst.num()),
            incrResult->data());

        checkCudaKernel(incrementBlocks);
    }

    void StreamCompaction::compact(
        idaten::TypedCudaMemory<int32_t>& dst,
        idaten::TypedCudaMemory<int32_t>& bools,
        int32_t* result/*= nullptr*/)
    {
        scan(m_blockSize, bools, exclusive_scan_array_);

        const auto num = static_cast<int32_t>(dst.num());
        const auto blockPerGrid = (num - 1) / m_blockSize + 1;

        scatter << <blockPerGrid, m_blockSize, 0, m_stream >> > (
            dst.data(),
            m_counts.data(),
            static_cast<int32_t>(dst.num()),
            bools.data(),
            exclusive_scan_array_.data());

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
