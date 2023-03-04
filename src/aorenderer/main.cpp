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

static int32_t WIDTH = 1280;
static int32_t HEIGHT = 720;
static const char *TITLE = "AORenderer";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static aten::AcceleratedScene<aten::GPUBvh> g_scene;
static aten::context g_ctxt;

static idaten::AORenderer g_tracer;

static std::shared_ptr<aten::visualizer> g_visualizer;

static bool g_willShowGUI = true;
static bool g_willTakeScreenShot = false;
static int32_t g_cntScreenShot = 0;

static int32_t g_renderMode = 0; // 0: AO, 1: TexView
static int32_t g_viewTexIdx = 0;

void onRun(aten::window *window)
{
    if (g_isCameraDirty)
    {
        g_camera.update();

        auto camparam = g_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        g_tracer.updateCamera(camparam);
        g_isCameraDirty = false;

        g_visualizer->clear();
    }

    aten::timer timer;
    timer.begin();

    if (g_renderMode == 0)
    {
        // AO
        g_tracer.render(
            idaten::TileDomain(0, 0, WIDTH, HEIGHT),
            1,
            5);
    }
    else
    {
        // Texture Viewer
        g_tracer.viewTextures(g_viewTexIdx, WIDTH, HEIGHT);
    }

    auto cudaelapsed = timer.end();

    aten::vec4 clear_color(0, 0.5f, 1.0f, 1.0f);
    aten::RasterizeRenderer::clearBuffer(
        aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
        clear_color,
        1.0f,
        0);

    g_visualizer->render(false);

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
        ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::Text("cuda : %.3f ms", cudaelapsed);

#if 0
        int32_t prevSamples = g_maxSamples;
        int32_t prevDepth = g_maxBounce;

        ImGui::SliderInt("Samples", &g_maxSamples, 1, 100);
        ImGui::SliderInt("Bounce", &g_maxBounce, 1, 10);

        if (prevSamples != g_maxSamples || prevDepth != g_maxBounce) {
            g_tracer.reset();
        }

        bool enableProgressive = g_tracer.isEnableProgressive();

        if (ImGui::Checkbox("Progressive", &enableProgressive)) {
            g_tracer.setEnableProgressive(enableProgressive);
        }
#endif
    }
}

void onClose()
{
}

bool g_isMouseLBtnDown = false;
bool g_isMouseRBtnDown = false;
int32_t g_prevX = 0;
int32_t g_prevY = 0;

void onMouseBtn(bool left, bool press, int32_t x, int32_t y)
{
    g_isMouseLBtnDown = false;
    g_isMouseRBtnDown = false;

    if (press)
    {
        g_prevX = x;
        g_prevY = y;

        g_isMouseLBtnDown = left;
        g_isMouseRBtnDown = !left;
    }
}

void onMouseMove(int32_t x, int32_t y)
{
    if (g_isMouseLBtnDown)
    {
        aten::CameraOperator::rotate(
            g_camera,
            WIDTH, HEIGHT,
            g_prevX, g_prevY,
            x, y);
        g_isCameraDirty = true;
    }
    else if (g_isMouseRBtnDown)
    {
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

void onMouseWheel(int32_t delta)
{
    aten::CameraOperator::dolly(g_camera, delta * real(0.1));
    g_isCameraDirty = true;
}

void onKey(bool press, aten::Key key)
{
    static const real offset = real(0.1);

    if (press)
    {
        if (key == aten::Key::Key_F1)
        {
            g_willShowGUI = !g_willShowGUI;
            return;
        }
        else if (key == aten::Key::Key_F2)
        {
            g_willTakeScreenShot = true;
            return;
        }
    }

    if (press)
    {
        switch (key)
        {
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

int32_t main()
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

    g_visualizer->addPostProc(&blitter);

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

    g_tracer.getCompaction().init(
        WIDTH * HEIGHT,
        1024);

    {
        std::vector<aten::ObjectParameter> shapeparams;
        std::vector<aten::TriangleParameter> primparams;
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

        const auto &nodes = g_scene.getAccel()->getNodes();
        const auto &mtxs = g_scene.getAccel()->getMatrices();

        std::vector<idaten::TextureResource> tex;
        {
            auto texNum = g_ctxt.getTextureNum();

            for (int32_t i = 0; i < texNum; i++)
            {
                auto t = g_ctxt.getTexture(i);
                tex.push_back(
                    idaten::TextureResource(t->colors(), t->width(), t->height()));
            }
        }

        auto camparam = g_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        g_tracer.update(
            g_visualizer->getTexHandle(),
            WIDTH, HEIGHT,
            camparam,
            shapeparams,
            mtrlparms,
            lightparams,
            nodes,
            primparams, 0,
            vtxparams, 0,
            mtxs,
            tex, idaten::EnvmapResource());
    }

    aten::window::run();

    aten::window::terminate();
}
