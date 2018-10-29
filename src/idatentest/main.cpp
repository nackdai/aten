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

#define ENABLE_ENVMAP
//#define ENABLE_GEOMRENDERING
//#define ENABLE_TEMPORAL
//#define ENABLE_ATROUS

static int WIDTH = 1280;
static int HEIGHT = 720;
static const char* TITLE = "idaten";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static aten::AcceleratedScene<aten::GPUBvh> g_scene;
static aten::context g_ctxt;

aten::ATrousDenoiser atrous;

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

        g_tracer.updateCamera(camparam);
        g_isCameraDirty = false;

#ifndef ENABLE_TEMPORAL
        g_visualizer->clear();
#endif
    }

    atrous.getPositionMap()->clearAsGLTexture(aten::vec4(real(1)));
    atrous.getNormalMap()->clearAsGLTexture(aten::vec4(real(0), real(0), real(0), real(-1)));
    atrous.getAlbedoMap()->clearAsGLTexture(aten::vec4(real(1)));

    aten::timer timer;
    timer.begin();

    g_tracer.render(
#ifdef ENABLE_GEOMRENDERING
        idaten::TileDomain(0, 0, WIDTH >> 1, HEIGHT >> 1),
#else
        idaten::TileDomain(0, 0, WIDTH, HEIGHT),
#endif
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

        int prevSamples = g_maxSamples;
        int prevDepth = g_maxBounce;

        ImGui::SliderInt("Samples", &g_maxSamples, 1, 100);
        ImGui::SliderInt("Bounce", &g_maxBounce, 1, 10);

        if (prevSamples != g_maxSamples || prevDepth != g_maxBounce) {
            g_tracer.reset();
        }

        bool enableProgressive = g_tracer.isProgressive();

        if (ImGui::Checkbox("Progressive", &enableProgressive)) {
            g_tracer.enableProgressive(enableProgressive);
        }

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

    aten::Blitter blitter;
    blitter.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/fullscreen_fs.glsl");
    blitter.setIsRenderRGB(true);

    atrous.init(
        g_ctxt,
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl", "../shader/atrous_fs.glsl",
        "../shader/fullscreen_vs.glsl", "../shader/atrous_final_fs.glsl");

#ifdef ENABLE_ATROUS
    g_visualizer->addPostProc(&atrous);
#endif
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
#ifdef ENABLE_GEOMRENDERING
        WIDTH >> 1, HEIGHT >> 1);
#else
        WIDTH, HEIGHT);
#endif

    Scene::makeScene(g_ctxt, &g_scene);
    g_scene.build(g_ctxt);

    g_tracer.getCompaction().init(
#ifdef ENABLE_GEOMRENDERING
        (WIDTH >> 1) * (HEIGHT >> 1),
#else
        WIDTH * HEIGHT,
#endif
        1024);

#ifdef ENABLE_ENVMAP
    auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", g_ctxt);
    aten::envmap bg;
    bg.init(envmap);
    aten::ImageBasedLight ibl(&bg);

    g_scene.addImageBasedLight(&ibl);
#endif

    {
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
        //aten::bvh::dumpCollectedNodes(nodes, "nodes.txt");

        std::vector<idaten::TextureResource> tex;
        {
            auto texNum = g_ctxt.getTextureNum();

            for (int i = 0; i < texNum; i++) {
                auto t = g_ctxt.getTexture(i);
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
#ifdef ENABLE_GEOMRENDERING
            WIDTH >> 1, HEIGHT >> 1,
#else
            WIDTH, HEIGHT,
#endif
            camparam,
            shapeparams,
            mtrlparms,
            lightparams,
            nodes,
            primparams, 0,
            vtxparams, 0,
            mtxs,
#ifdef ENABLE_ENVMAP
            tex, idaten::EnvmapResource(envmap->id(), ibl.getAvgIlluminace(), real(4)));
#else
            tex, idaten::EnvmapResource());
#endif

#if 1
        g_tracer.enableRenderAOV(
            atrous.getPositionMap()->getGLTexHandle(),
            atrous.getNormalMap()->getGLTexHandle(),
            atrous.getAlbedoMap()->getGLTexHandle(),
            aten::vec3(real(1)));
#endif
    }

    aten::window::run();

    aten::window::terminate();
}
