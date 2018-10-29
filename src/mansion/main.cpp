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

static int WIDTH = 1280;
static int HEIGHT = 720;
static const char* TITLE = "mansion";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static aten::AcceleratedScene<aten::GPUBvh> g_scene;
static aten::context g_ctxt;

static idaten::SVGFPathTracing g_tracer;
static aten::visualizer* g_visualizer;

static float g_avgcuda = 0.0f;
static float g_avgupdate = 0.0f;

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

void onRun(aten::window* window)
{
    if (g_enableFrameStep && !g_frameStep) {
        return;
    }

    auto frame = g_tracer.frame();

    g_frameStep = false;

    float updateTime = 0.0f;

    {
        aten::timer timer;
        timer.begin();

        updateTime = timer.end();

        g_avgupdate = g_avgupdate * (frame - 1) + updateTime;
        g_avgupdate /= (float)frame;
    }

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

    g_rasterizer.drawSceneForGBuffer(
        g_tracer.frame(),
        g_ctxt,
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
        ImGui::Text("update : %.3f ms (avg : %.3f ms)", updateTime, g_avgupdate);
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

        {
            float nTh, zTh;
            g_tracer.getTemporalFilterThreshold(nTh, zTh);

            bool b0 = ImGui::SliderFloat("TF Normal Threshold", &nTh, 0.0f, 1.0f);
            bool b1 = ImGui::SliderFloat("TF Depth Threshold", &zTh, 0.0f, 1.0f);

            if (b0 || b1){
                g_tracer.setTemporalFilterThreshold(nTh, zTh);
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

        auto mtrl = g_ctxt.getMaterial(info.mtrlid);

        AT_PRINTF("  mesh[%d] mtrl[%d](%s), tri[%d]\n", info.meshid, info.mtrlid, mtrl ? mtrl->name() : "none", info.triid);
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

    g_taa.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl", "../shader/taa_fs.glsl",
        "../shader/fullscreen_vs.glsl", "../shader/taa_final_fs.glsl");

    g_visualizer->addPostProc(&g_taa);
    g_visualizer->addPostProc(&gamma);

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
        WIDTH, HEIGHT);

    Scene::makeScene(g_ctxt, &g_scene);
    g_scene.build(g_ctxt);

    auto envmap = aten::ImageLoader::load("../../asset/mansion/HDR_040_Field.hdr", g_ctxt);
    aten::envmap bg;
    bg.init(envmap);
    aten::ImageBasedLight ibl(&bg);

    g_scene.addImageBasedLight(&ibl);

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

        g_tracer.update(
            aten::visualizer::getTexHandle(),
            WIDTH, HEIGHT,
            camparam,
            shapeparams,
            mtrlparms,
            lightparams,
            nodes,
            primparams, 0,
            vtxparams, 0,
            mtxs,
            tex,
            idaten::EnvmapResource(envmap->id(), real(1), real(4)));

        g_tracer.setGBuffer(
            g_fbo.getTexHandle(0),
            g_fbo.getTexHandle(1));
    }

    g_tracer.setMode((idaten::SVGFPathTracing::Mode)g_curMode);
    g_tracer.setAOVMode((idaten::SVGFPathTracing::AOVMode)g_curAOVMode);

    aten::window::run();

    aten::GLProfiler::terminate();

    g_rasterizer.release();
    g_rasterizerAABB.release();
    g_ctxt.release();

    aten::window::terminate();
}
