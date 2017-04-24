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

void makeScene(aten::scene* scene)
{
#if 1
	auto emit = new aten::emissive(aten::vec3(36, 36, 36));
	//auto emit = new aten::emissive(aten::vec3(3, 3, 3));

	auto light = new aten::sphere(
		aten::vec3(50.0, 75.0, 81.6),
		5.0,
		emit);

	double r = 1e3;

	auto left = new aten::sphere(
		aten::vec3(r + 1, 40.8, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75f, 0.25f, 0.25f)));

	auto right = new aten::sphere(
		aten::vec3(-r + 99, 40.8, 81.6),
		r,
		new aten::lambert(aten::vec3(0.25, 0.25, 0.75)));

	auto wall = new aten::sphere(
		aten::vec3(50, 40.8, r),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	auto floor = new aten::sphere(
		aten::vec3(50, r, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	auto ceil = new aten::sphere(
		aten::vec3(50, -r + 81.6, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

#if 0
	// —Î‹….
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));
	//new aten::lambert(aten::vec3(1, 1, 1), tex));

	// ‹¾.
	auto mirror = new aten::sphere(
		aten::vec3(27, 16.5, 47),
		16.5,
		new aten::specular(aten::vec3(0.99, 0.99, 0.99)));

	// ƒKƒ‰ƒX.
	auto glass = new aten::sphere(
		aten::vec3(77, 16.5, 78),
		16.5,
		new aten::refraction(aten::vec3(0.99, 0.99, 0.99), 1.5));
#endif

	aten::Light* l = new aten::AreaLight(light, emit->color());

#if 0
	scene->addLight(l);

	scene->add(light);
	scene->add(left);
	scene->add(right);
	scene->add(wall);
	scene->add(floor);
	scene->add(ceil);
	scene->add(green);
	scene->add(mirror);
	scene->add(glass);
#endif
#else
	auto em = new aten::emissive(aten::vec3(1, 0, 0));
	auto lm = new aten::lambert(aten::vec3(0.5, 0.5, 0.5));

	auto s0 = new aten::sphere(aten::vec3(0, 0, -10), 1.0f, em);
	auto s1 = new aten::sphere(aten::vec3(3, 0, -10), 1.0f, lm);

	auto area = new aten::AreaLight(s0, em->color());
#endif
}

void display()
{
	const auto& shapes = aten::transformable::getShapes();

	std::vector<aten::ShapeParameter> shapeparams;
	for (auto s : shapes) {
		auto param = s->getParam();
		param.mtrl.idx = aten::material::findMaterialIdx((aten::material*)param.mtrl.ptr);
		shapeparams.push_back(param);
	}

	const auto& lights = aten::Light::getLights();

	std::vector<aten::LightParameter> lightparams;
	for (auto l : lights) {
		auto param = l->param();
		param.object.idx = aten::transformable::findShapeIdx((aten::transformable*)param.object.ptr);
		lightparams.push_back(param);
	}

	const auto& mtrls = aten::material::getMaterials();

	std::vector<aten::MaterialParameter> mtrlparms;
	for (auto m : mtrls) {
		mtrlparms.push_back(m->param());
	}

	renderRayTracing(
		g_buffer.image(),
		WIDTH, HEIGHT,
		shapeparams,
		mtrlparms,
		lightparams);

	aten::visualizer::render(g_buffer.image(), false);
}

int main()
{
	aten::timer::init();
	aten::thread::setThreadNum(g_threadnum);

	aten::window::init(WIDTH, HEIGHT, TITLE);

	aten::visualizer::init(WIDTH, HEIGHT);

	aten::GammaCorrection gamma;
	gamma.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/gamma_fs.glsl");

	aten::visualizer::addPostProc(&gamma);

	makeScene(nullptr);

	aten::window::run(display);

	aten::window::terminate();
}
#endif
