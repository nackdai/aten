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

static int WIDTH = 1280;
static int HEIGHT = 720;
static const char* TITLE = "multigpu";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static aten::AcceleratedScene<aten::GPUBvh> g_scene;

static idaten::GpuProxy<idaten::PathTracing> g_tracer[2];

static const idaten::TileDomain g_tileDomain[2] = {
	{   0, 0, 640, 720 },
	{ 640, 0, 640, 720 },
};

static aten::visualizer* g_visualizer;

static bool g_willShowGUI = true;
static bool g_willTakeScreenShot = false;
static int g_cntScreenShot = 0;

static int g_maxSamples = 1;
static int g_maxBounce = 5;

void onRun(aten::window* window)
{
	if (g_isCameraDirty) {
		g_camera.update();

		auto camparam = g_camera.param();
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		for (int i = 0; i < AT_COUNTOF(g_tracer); i++) {
			g_tracer[i].getRenderer().updateCamera(camparam);
		}
		g_isCameraDirty = false;

		g_visualizer->clear();
	}

	aten::timer timer;
	timer.begin();

	g_tracer[0].render(
		g_tileDomain[0],
		g_maxSamples,
		g_maxBounce);

	auto cudaelapsed = timer.end();

	g_visualizer->render(false);

	if (g_willTakeScreenShot) {
		static char buffer[1024];
		::sprintf(buffer, "sc_%d.png\0", g_cntScreenShot);

		g_visualizer->takeScreenshot(buffer);

		g_willTakeScreenShot = false;
		g_cntScreenShot++;

		AT_PRINTF("Take Screenshot[%s]\n", buffer);
	}

	if (g_willShowGUI)
	{
		ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Text("cuda : %.3f ms", cudaelapsed);
		ImGui::Text("%.3f Mrays/sec", (WIDTH * HEIGHT * g_maxSamples) / real(1000 * 1000) * (real(1000) / cudaelapsed));

		window->drawImGui();
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
	idaten::initCuda();

	aten::timer::init();
	aten::OMPUtil::setThreadNum(g_threadnum);

	aten::initSampler(WIDTH, HEIGHT);

	aten::window::init(
		WIDTH, HEIGHT, TITLE,
		onRun,
		onClose,
		onMouseBtn,
		onMouseMove,
		onMouseWheel,
		onKey);

	g_visualizer = aten::visualizer::init(WIDTH, HEIGHT);

	aten::GammaCorrection gamma;
	gamma.init(
		WIDTH, HEIGHT,
		"../shader/fullscreen_vs.glsl",
		"../shader/gamma_fs.glsl");


	g_visualizer->addPostProc(&gamma);
	//aten::visualizer::addPostProc(&blitter);

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

	auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr");
	aten::envmap bg;
	bg.init(envmap);
	aten::ImageBasedLight ibl(&bg);

	g_scene.addImageBasedLight(&ibl);

	std::vector<aten::GeomParameter> shapeparams;
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

	const auto& nodes = g_scene.getAccel()->getNodes();
	const auto& mtxs = g_scene.getAccel()->getMatrices();
	//aten::bvh::dumpCollectedNodes(nodes, "nodes.txt");

	std::vector<idaten::TextureResource> tex;
	{
		auto texs = aten::texture::getTextures();

		for (const auto t : texs) {
			tex.push_back(
				idaten::TextureResource(t->colors(), t->width(), t->height()));
		}
	}

	for (auto& l : lightparams) {
		if (l.type == aten::LightType::IBL) {
			l.envmap.idx = envmap->id();
		}
	}

	auto camparam = g_camera.param();
	camparam.znear = real(0.1);
	camparam.zfar = real(10000.0);


	g_tracer[0].init(0);
	g_tracer[1].init(1);

	for (int i = 0; i < AT_COUNTOF(g_tracer); i++)
	{
		const auto& tileDomain = g_tileDomain[i];

		g_tracer[i].setCurrent();

		g_tracer[i].getRenderer().getCompaction().init(
			tileDomain.w * tileDomain.h,
			1024);

		g_tracer[i].getRenderer().update(
			aten::visualizer::getTexHandle(),
			WIDTH, HEIGHT,
			camparam,
			shapeparams,
			mtrlparms,
			lightparams,
			nodes,
			primparams,
			vtxparams,
			mtxs,
			tex, idaten::EnvmapResource(envmap->id(), ibl.getAvgIlluminace(), real(4)));
	}

	aten::window::run();

	aten::window::terminate();
}
