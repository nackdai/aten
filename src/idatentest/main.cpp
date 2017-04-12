#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "kernel.h"
#include "aten.h"
#include "idaten.h"

void sumArrayOnHost(float* A, float* B, float* C, const int N)
{
	for (int i = 0; i < N; i++) {
		C[i] = A[i] + B[i];
	}
}

void initData(float* p, int size)
{
	time_t t;
	srand((unsigned int)time(&t));

	for (int i = 0; i < size; i++) {
		p[i] = (float)(rand() & 0xff) / 10.0f;
	}
}

void chekcResult(float* hostRef, float* gpuRef, const int N)
{
	double epsilon = 1e-8;
	bool match = true;

	for (int i = 0; i < N; i++) {
		if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = false;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}

	if (match) {
		printf("Arrays match\n\n");
	}
}

int main()
{
	static const int Elem = 32;
	static const int Bytes = Elem * sizeof(float);

	float h_A[Elem];
	float h_B[Elem];
	float hostRef[Elem];
	float gpuRef[Elem];

	initData(h_A, Elem);
	initData(h_B, Elem);
	sumArrayOnHost(h_A, h_B, hostRef, Elem);

	aten::CudaMemory d_A(h_A, Bytes);
	aten::CudaMemory d_B(h_B, Bytes);
	aten::CudaMemory d_C(Bytes);

	sumArrayOnGPU((float*)d_A.ptr(), (float*)d_B.ptr(), (float*)d_C.ptr(), Elem);

	checkCudaErrors(cudaDeviceSynchronize());

	d_C.read(gpuRef, Bytes);

	chekcResult(hostRef, gpuRef, Elem);

	return 0;
}