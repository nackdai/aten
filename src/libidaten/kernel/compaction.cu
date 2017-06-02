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

__global__ void exclusiveScan(const int* src, int num, int* dst)
{
	extern __shared__ int temp[];

	int index = threadIdx.x;
	int offset = 1;

	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n * 2 >= num) {
		return;
	}

	// Copy input data to shared memory
	temp[2 * index] = src[2 * index + (blockIdx.x * blockDim.x * 2)];
	temp[2 * index + 1] = src[2 * index + 1 + (blockIdx.x * blockDim.x * 2)];

	// Up sweep
	for (int d = num >> 1; d > 0; d >>= 1) {
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
		temp[num - 1] = 0;
	}

	// Down sweep
	for (int d = 1; d < num; d *= 2) {
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
	void compact()
	{
		//int f[] = { 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3 };
		int f[] = { 3, 1, 7, 0, 4, 1, 6, 3, 3, 1 };

		int c = aten::nextPow2(AT_COUNTOF(f));

		std::vector<int> x(c);
		memcpy(&x[0], f, sizeof(int) * AT_COUNTOF(f));

		idaten::TypedCudaMemory<int> src;
		src.init(x.size());
		src.writeByNum(&x[0], x.size());

		idaten::TypedCudaMemory<int> dst;
		dst.init(x.size());

		int blocksize = 8;
		int blockPerGrid = (x.size() - 1) / blocksize + 1;

		exclusiveScan<<<blockPerGrid, blocksize / 2, blocksize  * sizeof(int)>>>(src.ptr(), src.maxNum(), dst.ptr());

		std::vector<int> tmp(src.maxNum());
		dst.readByNum(&tmp[0], tmp.size());

		idaten::TypedCudaMemory<int> incr;
		incr.init(blockPerGrid);

		int tmpBlockPerGrid = (blockPerGrid - 1) / blocksize + 1;
		int tmpBlockSize = blockPerGrid;

		computeBlockCount << <1, tmpBlockSize >> > (
			incr.ptr(),
			incr.maxNum(),
			blocksize,
			src.ptr(),
			dst.ptr());

		std::vector<int> tmp2(incr.maxNum());
		incr.readByNum(&tmp2[0], tmp2.size());

		incrementBlocks << <blockPerGrid, blocksize >> > (
			dst.ptr(),
			dst.maxNum(),
			incr.ptr());

		std::vector<int> tmp3(dst.maxNum());
		dst.readByNum(&tmp3[0], tmp3.size());

		int xxx = 0;
	}
}
