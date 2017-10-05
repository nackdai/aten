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
//#define ENABLE_NLM

static int WIDTH = 512;
static int HEIGHT = 512;
static const char* TITLE = "svgf";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::Film g_buffer(WIDTH, HEIGHT);

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static aten::AcceleratedScene<aten::bvh> g_scene;

static idaten::SVGFPathTracing g_tracer;

static aten::TAA g_taa;

static bool g_willShowGUI = true;
static bool g_willTakeScreenShot = false;
static int g_cntScreenShot = 0;

static int g_maxSamples = 1;
static int g_maxBounce = 1;
static int g_curMode = (int)idaten::SVGFPathTracing::Mode::SVGF;
static int g_curAOVMode = (int)idaten::SVGFPathTracing::AOVMode::WireFrame;

static bool g_enableFrameStep = false;
static bool g_frameStep = false;

static bool g_pickPixel = false;

void onRun()
{
	if (g_enableFrameStep && !g_frameStep) {
		return;
	}

	g_frameStep = false;

	if (g_isCameraDirty) {
		g_camera.update();

		auto camparam = g_camera.param();
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		g_tracer.updateCamera(camparam);
		g_isCameraDirty = false;

		aten::visualizer::clear();
	}

	g_taa.update(
		g_tracer.frame(),
		g_camera);

	aten::timer timer;
	timer.begin();

	g_tracer.render(
		g_buffer.image(),
		WIDTH, HEIGHT,
		g_maxSamples,
		g_maxBounce);

	auto cudaelapsed = timer.end();

	aten::visualizer::render(false);

	if (g_willTakeScreenShot)
	{
		static char buffer[1024];
		::sprintf(buffer, "sc_%d.png\0", g_cntScreenShot);

		aten::visualizer::takeScreenshot(buffer);

		g_willTakeScreenShot = false;
		g_cntScreenShot++;

		AT_PRINTF("Take Screenshot[%s]\n", buffer);
	}

	if (g_willShowGUI)
	{
		ImGui::Text("[%d] %.3f ms/frame (%.1f FPS)", g_tracer.frame(), 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Text("cuda : %.3f ms", cudaelapsed);
		ImGui::Text("%.3f Mrays/sec", (WIDTH * HEIGHT * g_maxSamples) / real(1000 * 1000) * (real(1000) / cudaelapsed));

		int prevSamples = g_maxSamples;
		int prevDepth = g_maxBounce;

		ImGui::SliderInt("Samples", &g_maxSamples, 1, 100);
		ImGui::SliderInt("Bounce", &g_maxBounce, 1, 10);

		if (prevSamples != g_maxSamples || prevDepth != g_maxBounce) {
			g_tracer.reset();
		}

		static const char* items[] = { "SVGF", "TF", "PT", "VAR", "AOV" };
		int item_current = g_curMode;
		ImGui::Combo("mode", &item_current, items, AT_COUNTOF(items));

		if (g_curMode != item_current) {
			g_curMode = item_current;
			g_tracer.setMode((idaten::SVGFPathTracing::Mode)g_curMode);
		}

		if (g_curMode == idaten::SVGFPathTracing::Mode::SVGF) {
			auto prevThreshold = g_tracer.getTemporalWeightThreshold();
			auto prevScale = g_tracer.getAtrousTapRadiusScale();

			auto threshold = prevThreshold;
			auto scale = prevScale;

			ImGui::SliderFloat("Threshold", &threshold, 0.0f, 1.0f);
			ImGui::SliderInt("Scale", &scale, 1, 4);

			if (prevThreshold != threshold) {
				g_tracer.setTemporalWeightThreshold(threshold);
			}
			if (prevScale != scale) {
				g_tracer.setAtrousTapRadiusScale(scale);
			}
		}

		if (g_curMode == idaten::SVGFPathTracing::Mode::AOVar) {
			static const char* aovitems[] = { "Normal", "TexColor", "Depth", "Wire" };
			int aov_current = g_curAOVMode;
			ImGui::Combo("aov", &aov_current, aovitems, AT_COUNTOF(aovitems));

			if (g_curAOVMode != aov_current) {
				g_curAOVMode = aov_current;
				g_tracer.setAOVMode((idaten::SVGFPathTracing::AOVMode)g_curAOVMode);
			}
		}

		bool prevEnableTAA = g_taa.isEnableTAA();
		bool enableTAA = prevEnableTAA;
		ImGui::Checkbox("Enable TAA", &enableTAA);

		bool prevCanShowTAADiff = g_taa.canShowTAADiff();
		bool canShowTAADiff = prevCanShowTAADiff;
		ImGui::Checkbox("Show TAA Diff", &canShowTAADiff);

		if (prevEnableTAA != enableTAA) {
			g_taa.enableTAA(enableTAA);
		}
		if (prevCanShowTAADiff != canShowTAADiff) {
			g_taa.showTAADiff(canShowTAADiff);
		}

		auto cam = g_camera.param();
		ImGui::Text("Pos %f/%f/%f", cam.origin.x, cam.origin.y, cam.origin.z);
		ImGui::Text("At  %f/%f/%f", cam.center.x, cam.center.y, cam.center.z);

		aten::window::drawImGui();
	}

	idaten::SVGFPathTracing::PickedInfo info;
	auto isPicked = g_tracer.getPickedPixelInfo(info);
	if (isPicked) {
		AT_PRINTF("[%d, %d]\n", info.ix, info.iy);
		AT_PRINTF("  nml[%f, %f, %f]\n", info.normal.x, info.normal.y, info.normal.z);
		AT_PRINTF("  mesh[%d] mtrl[%d], tri[%d]\n", info.meshid, info.mtrlid, info.triid);
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

		if (g_pickPixel) {
			g_tracer.willPickPixel(x, y);
			g_pickPixel = false;
		}
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
		else if (key == aten::Key::Key_F3) {
			g_enableFrameStep = !g_enableFrameStep;
			return;
		}
		else if (key == aten::Key::Key_SPACE) {
			if (g_enableFrameStep) {
				g_frameStep = true;
				return;
			}
		}
		else if (key == aten::Key::Key_CONTROL) {
			g_pickPixel = true;
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

	aten::initSampler(WIDTH, HEIGHT);

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

	aten::Blitter blitter;
	blitter.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/fs.glsl");
	blitter.setIsRenderRGB(true);

	aten::NonLocalMeanFilterShader nlmshd;
	nlmshd.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl",
		"../shader/nlm_fs.glsl");
	nlmshd.setParam(0.04f, 0.04f);

	g_taa.init(
		WIDTH, HEIGHT,
		"../shader/vs.glsl", "../shader/taa_fs.glsl",
		"../shader/vs.glsl", "../shader/taa_final_fs.glsl");

	aten::visualizer::addPostProc(&g_taa);
	aten::visualizer::addPostProc(&gamma);
	//aten::visualizer::addPostProc(&blitter);

#ifdef ENABLE_NLM
	aten::visualizer::addPostProc(&nlmshd);
#endif

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

	idaten::Compaction::init(
		WIDTH * HEIGHT,
		1024);

#ifdef ENABLE_ENVMAP
	auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr");
	aten::envmap bg;
	bg.init(envmap);
	aten::ImageBasedLight ibl(&bg);

	g_scene.addImageBasedLight(&ibl);
#endif

	{
		auto aabb = g_scene.getAccel()->getBoundingbox();
		auto d = aabb.getLengthBetweenMinAndMax();
		g_tracer.setHitDistanceLimit(d * 0.25f);

		g_tracer.setAovExportBuffer(g_taa.getAovGLTexHandle());

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

		auto camparam = g_camera.param();
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		g_tracer.update(
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
			tex,
#ifdef ENABLE_ENVMAP
			idaten::EnvmapResource(envmap->id(), ibl.getAvgIlluminace(), real(1)));
#else
			idaten::EnvmapResource());
#endif
	}

	g_tracer.setMode((idaten::SVGFPathTracing::Mode)g_curMode);
	g_tracer.setAOVMode((idaten::SVGFPathTracing::AOVMode)g_curAOVMode);

	aten::window::run(onRun);

	aten::window::terminate();
}
