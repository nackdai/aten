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

#define ENABLE_ENVMAP
//#define ENABLE_GEOMRENDERING
//#define ENABLE_TEMPORAL

static int WIDTH = 512;
static int HEIGHT = 512;
static const char* TITLE = "idaten";

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

#if defined(ENABLE_GEOMRENDERING) && defined(ENABLE_TEMPORAL)
// TODO
static idaten::PathTracingTemporalReprojectionGeomtryRendering g_tracer;
#elif defined(ENABLE_GEOMRENDERING)
static idaten::PathTracingGeometryRendering g_tracer;
#elif defined(ENABLE_TEMPORAL)
static idaten::PathTracingTemporalReprojection g_tracer;
#else
static idaten::PathTracing g_tracer;
#endif

static bool g_willShowGUI = true;
static bool g_willTakeScreenShot = false;
static int g_cntScreenShot = 0;

static int g_maxSamples = 1;
static int g_maxDepth = 5;

void onRun()
{
	if (g_isCameraDirty) {
		g_camera.update();
		g_tracer.updateCamera(g_camera.param());
		g_isCameraDirty = false;

#ifndef ENABLE_TEMPORAL
		aten::visualizer::clear();
#endif
	}

	aten::timer timer;
	timer.begin();

	g_tracer.render(
		g_buffer.image(),
#ifdef ENABLE_GEOMRENDERING
		WIDTH >> 1, HEIGHT >> 1,
#else
		WIDTH, HEIGHT,
#endif
		g_maxSamples,
		g_maxDepth);

	auto cudaelapsed = timer.end();

	//aten::visualizer::render(g_buffer.image(), false);
	aten::visualizer::render(false);

	if (g_willTakeScreenShot) {
		static char buffer[1024];
		::sprintf(buffer, "sc_%d.png\0", g_cntScreenShot);

		aten::visualizer::takeScreenshot(buffer);

		g_willTakeScreenShot = false;
		g_cntScreenShot++;

		AT_PRINTF("Take Screenshot[%s]\n", buffer);
	}

	if (g_willShowGUI)
	{
		ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Text("cuda : %.3f ms", cudaelapsed);
		ImGui::Text("%.3f Mrays/sec", (WIDTH * HEIGHT * g_maxSamples) / real(1000 * 1000) * (real(1000) / cudaelapsed));

		int prevSamples = g_maxSamples;
		int prevDepth = g_maxDepth;

		ImGui::SliderInt("Samples", &g_maxSamples, 1, 100);
		ImGui::SliderInt("Depth", &g_maxDepth, 1, 10);

		if (prevSamples != g_maxSamples || prevDepth != g_maxDepth) {
			g_tracer.reset();
		}

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
		g_isCameraDirty = true;
	}
	else if (g_isMouseRBtnDown) {
		aten::CameraOperator::move(
			g_camera,
			g_prevX, g_prevY,
			x, y,
			real(0.001));
		g_isCameraDirty = true;
	}

	g_prevX = x;
	g_prevY = y;
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
		if (key == aten::Key::Key_F1) {
			g_willShowGUI = !g_willShowGUI;
			return;
		}
		else if (key == aten::Key::Key_F2) {
			g_willTakeScreenShot = true;
			return;
		}
	}

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
		case aten::Key::Key_R:
		{
			aten::vec3 pos, at;
			real vfov;
			Scene::getCameraPosAndAt(pos, at, vfov);

			g_camera.init(
				pos,
				at,
				aten::vec3(0, 1, 0),
				vfov,
#ifdef ENABLE_GEOMRENDERING
				WIDTH >> 1, HEIGHT >> 1);
#else
				WIDTH, HEIGHT);
#endif
		}
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
#ifdef ENABLE_GEOMRENDERING
		WIDTH >> 1, HEIGHT >> 1);
#else
		WIDTH, HEIGHT);
#endif

	Scene::makeScene(&g_scene);
	g_scene.build();

	g_tracer.prepare();

	idaten::Compaction::init(
#ifdef ENABLE_GEOMRENDERING
		(WIDTH >> 1) * (HEIGHT >> 1),
#else
		WIDTH * HEIGHT,
#endif
		1024);

#ifdef ENABLE_ENVMAP
	auto envmap = aten::ImageLoader::load("../../asset/studio015.hdr");
	aten::envmap bg;
	bg.init(envmap);
	aten::ImageBasedLight ibl(&bg);

	g_scene.addImageBasedLight(&ibl);
#endif

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
		{
			auto texs = aten::texture::getTextures();

			for (const auto t : texs) {
				tex.push_back(
					idaten::TextureResource(t->colors(), t->width(), t->height()));
			}
		}

#ifdef ENABLE_ENVMAP
		for (auto& l : lightparams) {
			if (l.type == aten::LightType::IBL) {
				l.envmap.idx = envmap->id();
			}
		}
#endif

		g_tracer.update(
			aten::visualizer::getTexHandle(),
#ifdef ENABLE_GEOMRENDERING
			WIDTH >> 1, HEIGHT >> 1,
#else
			WIDTH, HEIGHT,
#endif
			g_camera.param(),
			shapeparams,
			mtrlparms,
			lightparams,
			nodes,
			primparams,
			vtxparams,
			mtxs,
#ifdef ENABLE_ENVMAP
			tex, idaten::EnvmapResource(envmap->id(), ibl.getAvgIlluminace()));
#else
			tex, idaten::EnvmapResource());
#endif
	}

	aten::window::run(onRun);

	aten::window::terminate();
}
