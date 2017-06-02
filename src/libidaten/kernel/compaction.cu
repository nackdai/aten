#include "kernel/compaction.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

__global__ void exclusiveScan(const int* src, int num, int* dst)
{
	extern __shared__ int temp[];

	int index = threadIdx.x;
	int offset = 1;

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

		if (index < d) {
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
		int x[] = { 3, 1, 7, 0, 4, 1, 6, 3, 3, 1, 7, 0, 4, 1, 6, 3 };

		idaten::TypedCudaMemory<int> src;
		src.init(AT_COUNTOF(x));
		src.writeByNum(x, AT_COUNTOF(x));

		idaten::TypedCudaMemory<int> dst;
		dst.init(AT_COUNTOF(x));

		int blocksize = 8;

		exclusiveScan<<<2, blocksize / 2, blocksize  * sizeof(int)>>>(src.ptr(), src.maxNum(), dst.ptr());

		std::vector<int> tmp(16);
		dst.readByNum(&tmp[0], 16);

		int xxx = 0;
	}
}
