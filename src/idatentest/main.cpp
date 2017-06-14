#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

#include "scenedefs.h"

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
static bool g_isCameraDirty = false;

static aten::AcceleratedScene<aten::bvh> g_scene;

//static idaten::RayTracing g_tracer;
static idaten::PathTracing g_tracer;

void onRun()
{
	if (g_isCameraDirty) {
		g_camera.update();
		g_tracer.updateCamera(g_camera.param());
		g_isCameraDirty = false;
	}

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

void onClose()
{

}

bool g_isMouseLBtnDown = false;
bool g_isMouseRBtnDown = false;
int g_prevX = 0;
int g_prevY = 0;

inline real normalizeHorizontal(int x, real width)
{
	real ret = (real(2) * x - width) / width;
	return ret;
}

inline real normalizeVertical(int y, real height)
{
	real ret = (height - real(2) * y) / height;
	return ret;
}


void onMouseBtn(bool left, bool press, int x, int y)
{
	g_isMouseLBtnDown = false;
	g_isMouseRBtnDown = false;

	if (press) {
		g_prevX = x;
		g_prevY = y;

		g_isMouseLBtnDown = left;
		g_isMouseRBtnDown = !left;
	}
}

void onMouseMove(int x, int y)
{
	if (g_isMouseLBtnDown) {
		real x1 = normalizeHorizontal(g_prevX, WIDTH);
		real y1 = normalizeVertical(g_prevY, HEIGHT);

		real x2 = normalizeHorizontal(x, WIDTH);
		real y2 = normalizeVertical(y, HEIGHT);

		aten::CameraOperator::rotate(
			g_camera,
			x1, y1,
			x2, y2);
	}
	else if (g_isMouseRBtnDown) {
		real offsetX = (real)(g_prevX - x);
		offsetX *= real(0.001);

		real offsetY = (real)(g_prevY - y);
		offsetY *= real(0.001);

		aten::CameraOperator::move(
			g_camera,
			offsetX, offsetY);
	}

	g_prevX = x;
	g_prevY = y;

	g_isCameraDirty = true;
}

void onMouseWheel(int delta)
{
	aten::CameraOperator::dolly(g_camera, delta * real(0.1));
	g_isCameraDirty = true;
}

int main()
{
	aten::timer::init();
	aten::thread::setThreadNum(g_threadnum);

	aten::window::init(
		WIDTH, HEIGHT, TITLE,
		onClose,
		onMouseBtn,
		onMouseMove,
		onMouseWheel);

	aten::visualizer::init(WIDTH, HEIGHT);

	aten::GammaCorrection gamma;
	gamma.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/gamma_fs.glsl");

	aten::visualizer::addPostProc(&gamma);

	aten::vec3 pos, at;
	real vfov;
	Scene::getCameraPosAndAt(pos, at, vfov);

	g_camera.init(
		pos,
		at,
		aten::vec3(0, 1, 0),
		vfov,
		WIDTH, HEIGHT);

	Scene::makeScene(&g_scene);
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

	aten::window::run(onRun);

	aten::window::terminate();
}
