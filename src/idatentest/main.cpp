#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "aten.h"
#include "idaten.h"

static int WIDTH = 640;
static int HEIGHT = 480;
static const char* TITLE = "app";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::Film g_buffer(WIDTH, HEIGHT);

static aten::PinholeCamera g_camera;

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

#if 1
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
	std::vector<aten::ShapeParameter> shapeparams;
	std::vector<aten::LightParameter> lightparams;
	std::vector<aten::MaterialParameter> mtrlparms;

	aten::DataCollector::collect(
		shapeparams,
		lightparams,
		mtrlparms);

	aten::timer timer;
	timer.begin();

	renderRayTracing(
		g_buffer.image(),
		WIDTH, HEIGHT,
		g_camera.param(),
		shapeparams,
		mtrlparms,
		lightparams);

	auto elapsed = timer.end();
	AT_PRINTF("Elapsed %f[ms]\n", elapsed);

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

	g_camera.init(
		aten::vec3(50.0, 52.0, 295.6),
		aten::vec3(50.0, 40.8, 119.0),
		aten::vec3(0, 1, 0),
		30,
		WIDTH, HEIGHT);

	makeScene(nullptr);

	prepareRayTracing();

	aten::window::run(display);

	aten::window::terminate();
}
