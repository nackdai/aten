#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <imgui.h>

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

	auto cudaelapsed = timer.end();

	//aten::visualizer::render(g_buffer.image(), false);
	aten::visualizer::render(false);
#if 0
	aten::visualizer::takeScreenshot("sc.png");
#endif

	{
		ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Text("cuda : %.3f ms", cudaelapsed);
		aten::window::drawImGui();
	}
}

void onClose()
{

}

bool g_isMouseLBtnDown = false;
bool g_isMouseRBtnDown = false;
int g_prevX = 0;
int g_prevY = 0;

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
		aten::CameraOperator::rotate(
			g_camera,
			WIDTH, HEIGHT,
			g_prevX, g_prevY,
			x, y);
	}
	else if (g_isMouseRBtnDown) {
		aten::CameraOperator::move(
			g_camera,
			g_prevX, g_prevY,
			x, y,
			real(0.001));
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

void onKey(bool press, aten::Key key)
{
	static const real offset = real(0.1);

	if (press) {
		switch (key) {
		case aten::Key::Key_W:
		case aten::Key::Key_UP:
			aten::CameraOperator::moveForward(g_camera, offset);
			break;
		case aten::Key::Key_S:
		case aten::Key::Key_DOWN:
			aten::CameraOperator::moveForward(g_camera, -offset);
			break;
		case aten::Key::Key_D:
		case aten::Key::Key_RIGHT:
			aten::CameraOperator::moveRight(g_camera, offset);
			break;
		case aten::Key::Key_A:
		case aten::Key::Key_LEFT:
			aten::CameraOperator::moveRight(g_camera, -offset);
			break;
		case aten::Key::Key_Z:
			aten::CameraOperator::moveUp(g_camera, offset);
			break;
		case aten::Key::Key_X:
			aten::CameraOperator::moveUp(g_camera, -offset);
			break;
		default:
			break;
		}

		g_isCameraDirty = true;
	}
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
		onMouseWheel,
		onKey);

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

	auto envmap = aten::ImageLoader::load("../../asset/studio015.hdr");
	aten::envmap bg;
	bg.init(envmap);
	aten::ImageBasedLight ibl(&bg);

	g_scene.addImageBasedLight(&ibl);

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

		std::vector<idaten::TextureResource> tex;
		tex.push_back(idaten::TextureResource(envmap->colors(), envmap->width(), envmap->height()));

		// TODO
		for (auto& l : lightparams) {
			if (l.type == aten::LightType::IBL) {
				l.envmap.idx = 0;
			}
		}

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
			mtxs,
			tex, idaten::EnvmapResource(0, ibl.getAvgIlluminace()));
	}

	aten::window::run(onRun);

	aten::window::terminate();
}
