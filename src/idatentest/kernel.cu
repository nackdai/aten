#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void onSumArrayOnGPU(float* A, float* B, float* C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

void sumArrayOnGPU(float* A, float* B, float* C, const int N)
{
	dim3 block(1);
	dim3 grid(N);

	onSumArrayOnGPU << <grid, block >> > (A, B, C, N);
}
