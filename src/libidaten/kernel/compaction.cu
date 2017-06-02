#include "kernel/compaction.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

// NOTE
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

// ブロック単位で計算した exclusiveScan の総和値を足したものを計算する.
__global__ void computeBlockCount(
	int* dst,
	int num,	// block count per grid used in exclusiveScan.
	int stride,	// thread count per block used in exclusiveScan.
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
	const int* incr)	// value to increment for each blocks.
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

namespace idaten {

	void scan(
		idaten::TypedCudaMemory<int>& src,
		idaten::TypedCudaMemory<int>& dst)
	{
		int blocksize = 8;
		int blockPerGrid = (dst.maxNum() - 1) / blocksize + 1;

		exclusiveScan << <blockPerGrid, blocksize / 2, blocksize * sizeof(int) >> > (
			dst.ptr(),
			dst.maxNum(),
			blocksize,
			src.ptr());

		if (blockPerGrid <= 1) {
			return;
		}

		idaten::TypedCudaMemory<int> incr;
		incr.init(blockPerGrid);

		int tmpBlockPerGrid = (blockPerGrid - 1) / blocksize + 1;
		int tmpBlockSize = blockPerGrid;

		computeBlockCount << <tmpBlockPerGrid, tmpBlockSize >> > (
			incr.ptr(),
			incr.maxNum(),
			blocksize,
			src.ptr(),
			dst.ptr());

		idaten::TypedCudaMemory<int> tmp;
		tmp.init(blockPerGrid);

		idaten::TypedCudaMemory<int> work;
		work.init(blockPerGrid);

		idaten::TypedCudaMemory<int>* input = &incr;
		idaten::TypedCudaMemory<int>* output = &tmp;

		int elementNum = blockPerGrid;

		int count = 1;
		int innerBlockPerGrid = 0;

		std::vector<int> stackBlockPerGrid;

		// Scan blocks.
		for (;;) {
			innerBlockPerGrid = (elementNum - 1) / blocksize + 1;
			stackBlockPerGrid.push_back(elementNum);

			exclusiveScan << <innerBlockPerGrid, blocksize / 2, blocksize * sizeof(int) >> >(
				work.ptr(),
				work.maxNum(),
				blocksize,
				input->ptr());

			if (innerBlockPerGrid <= 1) {
				cudaMemcpy(tmp.ptr(), work.ptr(), work.bytes(), cudaMemcpyDeviceToDevice);
				break;
			}

			int innerTmpBlockPerGrid = (innerBlockPerGrid - 1) / blocksize + 1;
			int innerTmpBlockSize = innerBlockPerGrid;

			computeBlockCount << <innerTmpBlockPerGrid, innerTmpBlockSize >> > (
				output->ptr(),
				output->maxNum(),
				blocksize,
				input->ptr(),
				work.ptr());

			// swap.
			auto p = input;
			input = output;
			output = p;

			elementNum = innerBlockPerGrid;
			count++;
		}

#if 1
		input = &tmp;
		output = &incr;

		for (int i = count - 1; i >= 0; i--) {
			// blocks per grid.
			auto bpg = stackBlockPerGrid[i];

			auto threadPerBlock = (output->maxNum() + bpg - 1) / bpg;

			incrementBlocks << <bpg, threadPerBlock >> > (
				output->ptr(),
				output->maxNum(),
				input->ptr());

			// swap.
			auto p = input;
			input = output;
			output = p;
		}

		idaten::TypedCudaMemory<int>* incrResult = (count & 0x1 == 0 ? &tmp : &incr);
#endif

		incrementBlocks << <blockPerGrid, blocksize >> > (
			dst.ptr(),
			dst.maxNum(),
			incrResult->ptr());
	}

	void compact()
	{
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

		scan(src, dst);
	}
}
