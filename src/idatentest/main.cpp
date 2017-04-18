#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "kernel.h"
#include "aten.h"
#include "idaten.h"

#if 0
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
#else
static int WIDTH = 640;
static int HEIGHT = 480;
static const char* TITLE = "app";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::Film g_buffer(WIDTH, HEIGHT);

void display()
{
	renderRayTracing(
		g_buffer.image(), 
		WIDTH, HEIGHT,
		aten::material::getMaterials());

	aten::visualizer::render(g_buffer.image(), false);
}

int main()
{
	aten::timer::init();
	aten::thread::setThreadNum(g_threadnum);

	aten::window::init(WIDTH, HEIGHT, TITLE);

	aten::visualizer::init(WIDTH, HEIGHT);

	aten::Blitter blitter;
	blitter.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/fs.glsl");
	blitter.setIsRenderRGB(true);

	aten::visualizer::addPostProc(&blitter);

	new aten::lambert(aten::vec3(1, 0, 0));
	new aten::lambert(aten::vec3(0, 1, 0));

	aten::window::run(display);

	aten::window::terminate();
}
#endif
