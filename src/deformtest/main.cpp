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
#define ENALBE_GPU_TRACER

static int WIDTH = 512;
static int HEIGHT = 512;
static const char* TITLE = "deform";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

class Lbvh : aten::accelerator {
public:
	Lbvh() : aten::accelerator(aten::AccelType::UserDefs) {}
	~Lbvh() {}

public:
	static accelerator* create()
	{
		auto ret = new Lbvh();
		return ret;
	}

	virtual void build(
		aten::hitable** list,
		uint32_t num,
		aten::aabb* bbox = nullptr) override final
	{
		m_bvh.build(list, num, bbox);

		setBoundingBox(m_bvh.getBoundingbox());
	}

	virtual bool hit(
		const aten::ray& r,
		real t_min, real t_max,
		aten::Intersection& isect) const override final
	{
		AT_ASSERT(false);
		return false;
	}

	virtual bool hit(
		const aten::ray& r,
		real t_min, real t_max,
		aten::Intersection& isect,
		bool enableLod) const override final
	{
		AT_ASSERT(false);
		return false;
	}

	aten::bvh m_bvh;
};

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static aten::AcceleratedScene<aten::GPUBvh> g_scene(Lbvh::create);

static idaten::SVGFPathTracing g_tracer;

static aten::PathTracing g_cputracer;
static aten::FilmProgressive g_buffer(WIDTH, HEIGHT);

static aten::visualizer* g_visualizer;

static float g_avgcuda = 0.0f;

static aten::TAA g_taa;

static aten::FBO g_fbo;

static aten::RasterizeRenderer g_rasterizer;
static aten::RasterizeRenderer g_rasterizerAABB;

static bool g_willShowGUI = true;
static bool g_willTakeScreenShot = false;
static int g_cntScreenShot = 0;

static int g_maxSamples = 1;
static int g_maxBounce = 5;
static int g_curMode = (int)idaten::SVGFPathTracing::Mode::SVGF;
static int g_curAOVMode = (int)idaten::SVGFPathTracing::AOVMode::WireFrame;
static bool g_showAABB = false;

static bool g_enableFrameStep = false;
static bool g_frameStep = false;

static bool g_pickPixel = false;

#ifdef ENALBE_GPU_TRACER
void onRun(aten::window* window)
{
	if (g_enableFrameStep && !g_frameStep) {
		return;
	}

	auto frame = g_tracer.frame();

	g_frameStep = false;

	if (g_isCameraDirty) {
		g_camera.update();

		auto camparam = g_camera.param();
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		g_tracer.updateCamera(camparam);
		g_isCameraDirty = false;

		g_visualizer->clear();
	}

	aten::GLProfiler::begin();

	g_rasterizer.draw(
		g_tracer.frame(),
		&g_scene,
		&g_camera,
		&g_fbo);

	auto rasterizerTime = aten::GLProfiler::end();

	aten::timer timer;
	timer.begin();

	g_tracer.render(
		idaten::TileDomain(0, 0, WIDTH, HEIGHT),
		g_maxSamples,
		g_maxBounce);

	auto cudaelapsed = timer.end();

	g_avgcuda = g_avgcuda * (frame - 1) + cudaelapsed;
	g_avgcuda /= (float)frame;

	aten::GLProfiler::begin();

	g_visualizer->render(false);

	auto visualizerTime = aten::GLProfiler::end();

	if (g_showAABB) {
		g_rasterizerAABB.drawAABB(
			&g_camera,
			g_scene.getAccel());
	}

	if (g_willTakeScreenShot)
	{
		static char buffer[1024];
		::sprintf(buffer, "sc_%d.png\0", g_cntScreenShot);

		g_visualizer->takeScreenshot(buffer);

		g_willTakeScreenShot = false;
		g_cntScreenShot++;

		AT_PRINTF("Take Screenshot[%s]\n", buffer);
	}

	if (g_willShowGUI)
	{
		ImGui::Text("[%d] %.3f ms/frame (%.1f FPS)", g_tracer.frame(), 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Text("cuda : %.3f ms (avg : %.3f ms)", cudaelapsed, g_avgcuda);
		ImGui::Text("%.3f Mrays/sec", (WIDTH * HEIGHT * g_maxSamples) / real(1000 * 1000) * (real(1000) / cudaelapsed));

		if (aten::GLProfiler::isEnabled()) {
			ImGui::Text("GL : [rasterizer %.3f ms] [visualizer %.3f ms]", rasterizerTime, visualizerTime);
		}

		if (ImGui::SliderInt("Samples", &g_maxSamples, 1, 100)
			|| ImGui::SliderInt("Bounce", &g_maxBounce, 1, 10))
		{
			g_tracer.reset();
		}

		static const char* items[] = { "SVGF", "TF", "PT", "VAR", "AOV" };

		if (ImGui::Combo("mode", &g_curMode, items, AT_COUNTOF(items))) {
			g_tracer.setMode((idaten::SVGFPathTracing::Mode)g_curMode);
		}

		if (g_curMode == idaten::SVGFPathTracing::Mode::AOVar) {
			static const char* aovitems[] = { "Normal", "TexColor", "Depth", "Wire", "Barycentric", "Motion", "ObjId" };

			if (ImGui::Combo("aov", &g_curAOVMode, aovitems, AT_COUNTOF(aovitems))) {
				g_tracer.setAOVMode((idaten::SVGFPathTracing::AOVMode)g_curAOVMode);
			}
		}

		bool enableTAA = g_taa.isEnableTAA();
		bool canShowTAADiff = g_taa.canShowTAADiff();

		if (ImGui::Checkbox("Enable TAA", &enableTAA)) {
			g_taa.enableTAA(enableTAA);
		}
		if (ImGui::Checkbox("Show TAA Diff", &canShowTAADiff)) {
			g_taa.showTAADiff(canShowTAADiff);
		}

		ImGui::Checkbox("Show AABB", &g_showAABB);

		auto cam = g_camera.param();
		ImGui::Text("Pos %f/%f/%f", cam.origin.x, cam.origin.y, cam.origin.z);
		ImGui::Text("At  %f/%f/%f", cam.center.x, cam.center.y, cam.center.z);

		window->drawImGui();
	}

	idaten::SVGFPathTracing::PickedInfo info;
	auto isPicked = g_tracer.getPickedPixelInfo(info);
	if (isPicked) {
		AT_PRINTF("[%d, %d]\n", info.ix, info.iy);
		AT_PRINTF("  nml[%f, %f, %f]\n", info.normal.x, info.normal.y, info.normal.z);
		AT_PRINTF("  mesh[%d] mtrl[%d], tri[%d]\n", info.meshid, info.mtrlid, info.triid);
	}
}
#else
void onRun(aten::window* wnd)
{
	static uint32_t g_frameNo = 0;
	static float g_avgElapsed = 0.0f;

	g_camera.update();

	aten::Destination dst;
	{
		dst.width = WIDTH;
		dst.height = HEIGHT;
		dst.maxDepth = 5;
		dst.russianRouletteDepth = 3;
		dst.startDepth = 0;
		dst.sample = 1;
		dst.mutation = 10;
		dst.mltNum = 10;
		dst.buffer = &g_buffer;
	}

	dst.geominfo.albedo_vis = &g_buffer;
	dst.geominfo.depthMax = 1000;

	aten::timer timer;
	timer.begin();

	// Trace rays.
	g_cputracer.render(dst, &g_scene, &g_camera);

	auto elapsed = timer.end();

	g_avgElapsed = g_avgElapsed * g_frameNo + elapsed;
	g_avgElapsed /= (g_frameNo + 1);

	AT_PRINTF("Elapsed %f[ms] / Avg %f[ms]\n", elapsed, g_avgElapsed);

	g_visualizer->render(g_buffer.image(), g_camera.needRevert());

	g_frameNo++;
}
#endif

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
		else if (key == aten::Key::Key_F5) {
			aten::GLProfiler::trigger();
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
	idaten::Skinning skin;
	skin.runMinMaxTest();

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

	aten::GLProfiler::start();

	g_visualizer = aten::visualizer::init(WIDTH, HEIGHT);

	aten::GammaCorrection gamma;
	gamma.init(
		WIDTH, HEIGHT,
		"../shader/fullscreen_vs.glsl",
		"../shader/gamma_fs.glsl");

	aten::Blitter blitter;
	blitter.init(
		WIDTH, HEIGHT,
		"../shader/fullscreen_vs.glsl",
		"../shader/fullscreen_fs.glsl");
	blitter.setIsRenderRGB(true);

	g_taa.init(
		WIDTH, HEIGHT,
		"../shader/fullscreen_vs.glsl", "../shader/taa_fs.glsl",
		"../shader/fullscreen_vs.glsl", "../shader/taa_final_fs.glsl");

	g_visualizer->addPostProc(&g_taa);
	g_visualizer->addPostProc(&gamma);
	//aten::visualizer::addPostProc(&blitter);

	g_rasterizer.init(
		WIDTH, HEIGHT,
		"../shader/ssrt_vs.glsl",
		"../shader/ssrt_gs.glsl",
		"../shader/ssrt_fs.glsl");
	g_rasterizerAABB.init(
		WIDTH, HEIGHT,
		"../shader/simple3d_vs.glsl",
		"../shader/simple3d_fs.glsl");

	g_fbo.asMulti(2);
	g_fbo.init(
		WIDTH, HEIGHT,
		aten::PixelFormat::rgba32f,
		true);

	g_taa.setMotionDepthBufferHandle(g_fbo.getTexHandle(1));

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

#ifdef ENABLE_ENVMAP
	auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr");
	aten::envmap bg;
	bg.init(envmap);
	aten::ImageBasedLight ibl(&bg);

	g_scene.addImageBasedLight(&ibl);
#endif

	{
		auto aabb = g_scene.getAccel()->getBoundingbox();
		auto d = aabb.getDiagonalLenght();
		g_tracer.setHitDistanceLimit(d * 0.25f);

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

		g_tracer.setGBuffer(
			g_fbo.getTexHandle(0),
			g_fbo.getTexHandle(1));
	}

	// For LBVH.
	{
		std::vector<std::vector<aten::PrimitiveParamter>> triangles;
		std::vector<int> triIdOffsets;
		std::vector<aten::vertex> vtxparams;

		aten::DataCollector::collectTriangles(triangles, triIdOffsets, vtxparams);

		const auto& sceneBbox = g_scene.getAccel()->getBoundingbox();
		auto& nodes = g_tracer.getCudaTextureResourceForBvhNodes();
		auto& vtxPos = g_tracer.getCudaTextureResourceForVtxPos();

		// TODO
		// もし、GPUBvh が SBVH だとした場合.
		// ここで取得するノード配列は SBVH のノードである、ThreadedSbvhNode となる.
		// しかし、LBVHBuilder::build で渡すことができるのは、ThreadBVH のノードである ThreadedBvhNode である.
		// そのため、現状、ThreadedBvhNode に無理やりキャストしている.
		// もっとスマートな方法を考えたい.

		auto& cpunodes = g_scene.getAccel()->getNodes();

		for (int i = 0; i < triangles.size(); i++)
		{
			auto& tris = triangles[i];
			auto triIdOffset = triIdOffsets[i];

			// NOTE
			// 0 is for top layer.
			idaten::LBVHBuilder::build(
				nodes[i + 1],	
				tris,
				triIdOffset,
				sceneBbox,
				vtxPos,
				(std::vector<aten::ThreadedBvhNode>*)&cpunodes[i + 1]);
		}
	}

	g_tracer.setMode((idaten::SVGFPathTracing::Mode)g_curMode);
	g_tracer.setAOVMode((idaten::SVGFPathTracing::AOVMode)g_curAOVMode);

	aten::window::run();

	aten::GLProfiler::terminate();

	g_rasterizer.release();
	g_rasterizerAABB.release();
	aten::VertexManager::release();

	aten::window::terminate();
}
