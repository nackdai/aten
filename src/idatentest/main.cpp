#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

#define CORNELLBOX_SUZANNE

static int WIDTH = 512;
static int HEIGHT = 512;
static const char* TITLE = "app";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::Film g_buffer(WIDTH, HEIGHT);

static aten::PinholeCamera g_camera;

static aten::AcceleratedScene<aten::bvh> g_scene;

//static idaten::RayTracing g_tracer;
static idaten::PathTracing g_tracer;

void makeScene(aten::scene* scene)
{
#ifdef CORNELLBOX_SUZANNE
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

#if 0
	// ƒKƒ‰ƒX.
	auto glass = new aten::sphere(
		aten::vec3(77, 16.5, 78),
		16.5,
		new aten::refraction(aten::vec3(0.99, 0.99, 0.99), 1.5));
#else
	aten::AssetManager::registerMtrl(
		"Material.001",
		new aten::lambert(aten::vec3(0.2, 0.2, 0.7)));

	auto obj = aten::ObjLoader::load("../../asset/suzanne.obj");

	aten::mat4 mtxL2W;
	mtxL2W.asRotateByY(Deg2Rad(-25));

	aten::mat4 mtxT;
	mtxT.asTrans(aten::vec3(77, 16.5, 78));

	aten::mat4 mtxS;
	mtxS.asScale(10);

	mtxL2W = mtxT * mtxL2W * mtxS;

	auto glass = new aten::instance<aten::object>(obj, mtxL2W);
#endif

	aten::Light* l = new aten::AreaLight(light, emit->color());

	scene->addLight(l);

#if 1
	scene->add(light);
	scene->add(left);
	scene->add(right);
	scene->add(wall);
	scene->add(floor);
	scene->add(ceil);
	scene->add(green);
	scene->add(mirror);
	scene->add(glass);
#else
	scene->add(light);
	scene->add(glass);
#endif
#else
	aten::AssetManager::registerMtrl(
		"backWall",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"ceiling",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"floor",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"leftWall",
		new aten::lambert(aten::vec3(0.504000, 0.052000, 0.040000)));

	auto emit = new aten::emissive(aten::vec3(36, 33, 24));
	aten::AssetManager::registerMtrl(
		"light",
		emit);

	aten::AssetManager::registerMtrl(
		"rightWall",
		new aten::lambert(aten::vec3(0.112000, 0.360000, 0.072800)));
	aten::AssetManager::registerMtrl(
		"shortBox",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"tallBox",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));

	std::vector<aten::object*> objs;
	aten::ObjLoader::load(objs, "../../asset/cornellbox/orig.obj");

	auto light = new aten::instance<aten::object>(objs[0], aten::mat4::Identity);
	auto box = new aten::instance<aten::object>(objs[1], aten::mat4::Identity);

	scene->add(light);
	scene->add(box);

	auto areaLight = new aten::AreaLight(light, emit->param().baseColor);
	scene->addLight(areaLight);
#endif
}

void display()
{
	aten::timer timer;
	timer.begin();

	g_tracer.render(
		g_buffer.image(),
		WIDTH, HEIGHT);

	auto elapsed = timer.end();
	AT_PRINTF("Elapsed %f[ms]\n", elapsed);

	//aten::visualizer::render(g_buffer.image(), false);
	aten::visualizer::render(false);
	aten::visualizer::takeScreenshot("sc.png");
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

#ifdef CORNELLBOX_SUZANNE
	g_camera.init(
		aten::vec3(50.0, 52.0, 295.6),
		aten::vec3(50.0, 40.8, 119.0),
		aten::vec3(0, 1, 0),
		30,
		WIDTH, HEIGHT);
#else
	g_camera.init(
		aten::vec3(0.f, 1.f, 3.f),
		aten::vec3(0.f, 1.f, 0.f),
		aten::vec3(0, 1, 0),
		45,
		WIDTH, HEIGHT);
#endif

	makeScene(&g_scene);
	g_scene.build();

	g_tracer.prepare();

	idaten::Compaction::init(
		WIDTH * HEIGHT,
		1024);

	{
		std::vector<aten::ShapeParameter> shapeparams;
		std::vector<aten::PrimitiveParamter> primparams;
		std::vector<aten::LightParameter> lightparams;
		std::vector<aten::MaterialParameter> mtrlparms;
		std::vector<aten::vertex> vtxparams;

		aten::DataCollector::collect(
			shapeparams,
			primparams,
			lightparams,
			mtrlparms,
			vtxparams);

		std::vector<std::vector<aten::BVHNode>> nodes;
		std::vector<aten::mat4> mtxs;

		g_scene.getAccel()->collectNodes(nodes, mtxs);
		//aten::bvh::dumpCollectedNodes(nodes, "nodes.txt");

		g_tracer.update(
			aten::visualizer::getTexHandle(),
			WIDTH, HEIGHT,
			g_camera.param(),
			shapeparams,
			mtrlparms,
			lightparams,
			nodes,
			primparams,
			vtxparams,
			mtxs);
	}

	aten::window::run(display);

	aten::window::terminate();
}
