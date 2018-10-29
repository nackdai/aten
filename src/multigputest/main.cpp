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

#include "../common/scenedefs.h"

#define MULTI_GPU_SVGF
#define GPU_NUM    (4)

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
static aten::context g_ctxt;

#ifdef MULTI_GPU_SVGF
static idaten::GpuProxy<idaten::SVGFPathTracingMultiGPU> g_tracer[GPU_NUM];

static aten::TAA g_taa;

static aten::FBO g_fbo;

static aten::RasterizeRenderer g_rasterizer;

using GpuProxy = idaten::GpuProxy<idaten::SVGFPathTracingMultiGPU>;
#else
static idaten::GpuProxy<idaten::PathTracingMultiGPU> g_tracer[GPU_NUM];
using GpuProxy = idaten::GpuProxy<idaten::PathTracingMultiGPU>;
#endif

static int g_curMode = (int)idaten::SVGFPathTracing::Mode::SVGF;

#if (GPU_NUM == 4)
static const idaten::TileDomain g_tileDomain[4] = {
    { 0, 0 * HEIGHT / 4, WIDTH, HEIGHT / 4 },
    { 0, 1 * HEIGHT / 4, WIDTH, HEIGHT / 4 },
    { 0, 2 * HEIGHT / 4, WIDTH, HEIGHT / 4 },
    { 0, 3 * HEIGHT / 4, WIDTH, HEIGHT / 4 },
};
#else
static const idaten::TileDomain g_tileDomain[2] = {
    { 0, 0 * HEIGHT / 2, WIDTH, HEIGHT / 2 },
    { 0, 1 * HEIGHT / 2, WIDTH, HEIGHT / 2 },
};
#endif

static aten::visualizer* g_visualizer = nullptr;

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

#ifdef MULTI_GPU_SVGF
    g_rasterizer.drawSceneForGBuffer(
        g_tracer[0].getRenderer().frame(),
        g_ctxt,
        &g_scene,
        &g_camera,
        &g_fbo);

    for (int i = 0; i < AT_COUNTOF(g_tracer); i++) {
        g_tracer[i].getRenderer().setMode((idaten::SVGFPathTracing::Mode)g_curMode);
    }
#endif

    for (int i = 0; i < AT_COUNTOF(g_tracer); i++) {
        g_tracer[i].render(
            g_tileDomain[i],
            g_maxSamples,
            g_maxBounce);
    }

#if (GPU_NUM == 4)
    GpuProxy::swapCopy(g_tracer, AT_COUNTOF(g_tracer));
#else
    for (int i = 1; i < AT_COUNTOF(g_tracer); i++) {
        g_tracer[0].copyP2P(g_tracer[1]);
    }
#endif

    g_tracer[0].postRender(WIDTH, HEIGHT);

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

        if (ImGui::SliderInt("Samples", &g_maxSamples, 1, 100)
            || ImGui::SliderInt("Bounce", &g_maxBounce, 1, 10))
        {
            for (int i = 1; i < AT_COUNTOF(g_tracer); i++) {
                g_tracer[i].getRenderer().reset();
            }
        }

#ifdef MULTI_GPU_SVGF
        bool enableTAA = g_taa.isEnableTAA();
        if (ImGui::Checkbox("Enable TAA", &enableTAA)) {
            g_taa.enableTAA(enableTAA);
        }
#endif

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
                WIDTH, HEIGHT);
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

    auto window = aten::window::init(
        WIDTH, HEIGHT, TITLE,
        onRun,
        onClose,
        onMouseBtn,
        onMouseMove,
        onMouseWheel,
        onKey);
    window->enableVSync(false);

    g_visualizer = aten::visualizer::init(WIDTH, HEIGHT);

#ifdef MULTI_GPU_SVGF
    g_taa.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl", "../shader/taa_fs.glsl",
        "../shader/fullscreen_vs.glsl", "../shader/taa_final_fs.glsl");

    g_visualizer->addPostProc(&g_taa);

    g_rasterizer.init(
        WIDTH, HEIGHT,
        "../shader/ssrt_vs.glsl",
        "../shader/ssrt_gs.glsl",
        "../shader/ssrt_fs.glsl");

    g_fbo.asMulti(2);
    g_fbo.init(
        WIDTH, HEIGHT,
        aten::PixelFormat::rgba32f,
        true);

    g_taa.setMotionDepthBufferHandle(g_fbo.getTexHandle(1));
#endif

    aten::GammaCorrection gamma;
    gamma.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/gamma_fs.glsl");

    g_visualizer->addPostProc(&gamma);

    aten::vec3 pos, at;
    real vfov;
    Scene::getCameraPosAndAt(pos, at, vfov);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    Scene::makeScene(g_ctxt, &g_scene);
    g_scene.build(g_ctxt);

    auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", g_ctxt);
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
        g_ctxt,
        g_scene,
        shapeparams,
        primparams,
        lightparams,
        mtrlparms,
        vtxparams);

    const auto& nodes = g_scene.getAccel()->getNodes();
    const auto& mtxs = g_scene.getAccel()->getMatrices();

    std::vector<idaten::TextureResource> tex;
    {
        auto texNum = g_ctxt.getTextureNum();

        for (int i = 0; i < texNum; i++) {
            auto t = g_ctxt.getTexture(i);
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

    // Set P2P access between GPUs.
    g_tracer[0].setPeerAccess(1);
    g_tracer[1].setPeerAccess(0);

#if (GPU_NUM == 4)
    g_tracer[2].init(2);
    g_tracer[3].init(3);
    
    g_tracer[2].setPeerAccess(3);
    g_tracer[3].setPeerAccess(2);

    g_tracer[2].setPeerAccess(0);
    g_tracer[0].setPeerAccess(2);
#endif

    for (int i = 0; i < AT_COUNTOF(g_tracer); i++)
    {
        const auto& tileDomain = g_tileDomain[i];

        g_tracer[i].setCurrent();

        g_tracer[i].getRenderer().getCompaction().init(
            tileDomain.w * tileDomain.h,
            1024);

        int w = i == 0 ? WIDTH : tileDomain.w;
        int h = i == 0 ? HEIGHT : tileDomain.h;

        h = i == 2 ? HEIGHT / 2 : h;

        g_tracer[i].getRenderer().update(
            aten::visualizer::getTexHandle(),
            w, h,
            camparam,
            shapeparams,
            mtrlparms,
            lightparams,
            nodes,
            primparams, 0,
            vtxparams, 0,
            mtxs,
            tex, idaten::EnvmapResource(envmap->id(), ibl.getAvgIlluminace(), real(1)));

#ifdef MULTI_GPU_SVGF
        g_tracer[i].getRenderer().setGBuffer(
            g_fbo.getTexHandle(0),
            g_fbo.getTexHandle(1));
#endif
    }

    aten::window::run();

    aten::window::terminate();
}
