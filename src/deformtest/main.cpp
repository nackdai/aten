#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <iterator>

#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

#include "../common/scenedefs.h"

//#pragma optimize( "", off)

#define ENABLE_ENVMAP
#define ENABLE_SVGF

static int WIDTH = 1280;
static int HEIGHT = 720;
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
        const aten::context& ctxt,
        aten::hitable** list,
        uint32_t num,
        aten::aabb* bbox = nullptr) override final
    {
        m_bvh.build(ctxt, list, num, bbox);

        setBoundingBox(m_bvh.getBoundingbox());
    }

    virtual bool hit(
        const aten::context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection& isect) const override final
    {
        AT_ASSERT(false);
        return false;
    }

    virtual bool hit(
        const aten::context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        bool enableLod,
        aten::Intersection& isect) const override final
    {
        AT_ASSERT(false);
        return false;
    }

    aten::bvh m_bvh;
};

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static aten::AcceleratedScene<aten::GPUBvh> g_scene;
static aten::context g_ctxt;

#ifdef ENABLE_SVGF
static idaten::SVGFPathTracing g_tracer;
#else
static idaten::PathTracing g_tracer;
#endif

static aten::PathTracing g_cputracer;
static aten::FilmProgressive g_buffer(WIDTH, HEIGHT);

static aten::visualizer* g_visualizer;

static float g_avgcuda = 0.0f;

static aten::TAA g_taa;

static aten::FBO g_fbo;

static aten::RasterizeRenderer g_rasterizer;
static aten::RasterizeRenderer g_rasterizerAABB;

static aten::shader g_shdRasterizeDeformable;

static idaten::Skinning g_skinning;
static aten::Timeline g_timeline;

static idaten::LBVHBuilder g_lbvh;
static int g_triOffset = 0;
static int g_vtxOffset = 0;

static bool g_willShowGUI = true;
static bool g_willTakeScreenShot = false;
static int g_cntScreenShot = 0;

static int g_maxSamples = 1;
static int g_maxBounce = 5;
static int g_curMode = (int)idaten::SVGFPathTracing::Mode::SVGF;
static int g_curAOVMode = (int)idaten::SVGFPathTracing::AOVMode::WireFrame;
static bool g_showAABB = false;

static bool g_pickPixel = false;

void update(int frame)
{
    auto deform = getDeformable();

    if (deform) {
        auto mdl = deform->getHasObjectAsRealType();
        auto anm = getDeformAnm();

        if (anm) {
            aten::mat4 mtxL2W;
            mtxL2W.asScale(0.01);
            mdl->update(mtxL2W, g_timeline.getTime(), anm);
        }
        else {
            mdl->update(aten::mat4(), 0, nullptr);
        }

        g_timeline.advance(1.0f / 60.0f);

        const auto& mtx = mdl->getMatrices();
        g_skinning.update(&mtx[0], mtx.size());

        aten::vec3 aabbMin, aabbMax;

        bool isRestart = (frame == 1);

        // NOTE
        // Add verted offset, in the first frame.
        // In "g_skinning.compute", vertex offset is added to triangle paremters.
        // Added vertex offset is valid permanently, so specify vertex offset just only one time.
        g_skinning.compute(aabbMin, aabbMax, isRestart);

        mdl->setBoundingBox(aten::aabb(aabbMin, aabbMax));
        deform->update(true);

        const auto sceneBbox = aten::aabb(aabbMin, aabbMax);
        auto& nodes = g_tracer.getCudaTextureResourceForBvhNodes();

        auto& vtxPos = g_skinning.getInteropVBO()[0];
        auto& tris = g_skinning.getTriangles();

        // TODO
        int deformPos = nodes.size() - 1;

        // NOTE
        // Vertex offset was added in "g_skinning.compute".
        // But, in "g_lbvh.build", vertex index have to be handled as zero based index.
        // Vertex offset have to be removed from vertex index.
        // So, specify minus vertex offset.
        // This is work around, too complicated...
        g_lbvh.build(
            nodes[deformPos],
            tris,
            g_triOffset,
            sceneBbox,
            vtxPos,
            -g_vtxOffset,
            nullptr);

        // Copy computed vertices, triangles to the tracer.
        g_tracer.updateGeometry(
            g_skinning.getInteropVBO(),
            g_vtxOffset,
            g_skinning.getTriangles(),
            g_triOffset);

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

            auto accel = g_scene.getAccel();
            accel->update(g_ctxt);

            const auto& nodes = g_scene.getAccel()->getNodes();
            const auto& mtxs = g_scene.getAccel()->getMatrices();

            g_tracer.update(
                shapeparams,
                nodes,
                mtxs);
        }
    }
}

void onRun(aten::window* window)
{
    auto frame = g_tracer.frame();

    update(frame);

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

#ifdef ENABLE_SVGF
    g_rasterizer.drawSceneForGBuffer(
        g_tracer.frame(),
        g_ctxt,
        &g_scene,
        &g_camera,
        &g_fbo,
        &g_shdRasterizeDeformable);
#endif

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

#ifdef ENABLE_SVGF
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
        else if (g_curMode == idaten::SVGFPathTracing::Mode::SVGF) {
            int iterCnt = g_tracer.getAtrousIterCount();
            if (ImGui::SliderInt("Atrous Iter", &iterCnt, 1, 5)) {
                g_tracer.setAtrousIterCount(iterCnt);
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
#endif

        auto cam = g_camera.param();
        ImGui::Text("Pos %f/%f/%f", cam.origin.x, cam.origin.y, cam.origin.z);
        ImGui::Text("At  %f/%f/%f", cam.center.x, cam.center.y, cam.center.z);

        window->drawImGui();
    }

#ifdef ENABLE_SVGF
    idaten::SVGFPathTracing::PickedInfo info;
    auto isPicked = g_tracer.getPickedPixelInfo(info);
    if (isPicked) {
        AT_PRINTF("[%d, %d]\n", info.ix, info.iy);
        AT_PRINTF("  nml[%f, %f, %f]\n", info.normal.x, info.normal.y, info.normal.z);
        AT_PRINTF("  mesh[%d] mtrl[%d], tri[%d]\n", info.meshid, info.mtrlid, info.triid);
    }
#endif
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

#ifdef ENABLE_SVGF
        if (g_pickPixel) {
            g_tracer.willPickPixel(x, y);
            g_pickPixel = false;
        }
#endif
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
    static const real offset = real(0.5);

    if (press) {
        if (key == aten::Key::Key_F1) {
            g_willShowGUI = !g_willShowGUI;
            return;
        }
        else if (key == aten::Key::Key_F2) {
            g_willTakeScreenShot = true;
            return;
        }
        else if (key == aten::Key::Key_F5) {
            aten::GLProfiler::trigger();
            return;
        }
        else if (key == aten::Key::Key_SPACE) {
            if (g_timeline.isPaused()) {
                g_timeline.start();
            }
            else {
                g_timeline.pause();
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

    aten::Blitter blitter;
    blitter.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl",
        "../shader/fullscreen_fs.glsl");
    blitter.setIsRenderRGB(true);

#ifdef ENABLE_SVGF
    g_taa.init(
        WIDTH, HEIGHT,
        "../shader/fullscreen_vs.glsl", "../shader/taa_fs.glsl",
        "../shader/fullscreen_vs.glsl", "../shader/taa_final_fs.glsl");

    g_visualizer->addPostProc(&g_taa);
#endif

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

    g_shdRasterizeDeformable.init(
        WIDTH, HEIGHT,
        "./ssrt_deformable_vs.glsl",
        "../shader/ssrt_gs.glsl",
        "../shader/ssrt_fs.glsl");

#ifdef ENABLE_SVGF
    g_fbo.asMulti(2);
    g_fbo.init(
        WIDTH, HEIGHT,
        aten::PixelFormat::rgba32f,
        true);

    g_taa.setMotionDepthBufferHandle(g_fbo.getTexHandle(1));
#endif

    aten::vec3 pos, at;
    real vfov;
    Scene::getCameraPosAndAt(pos, at, vfov);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    aten::accelerator::setUserDefsInternalAccelCreator(Lbvh::create);

    Scene::makeScene(g_ctxt, &g_scene);
    g_scene.build(g_ctxt);

#ifdef ENABLE_ENVMAP
    auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", g_ctxt);
    aten::envmap bg;
    bg.init(envmap);
    aten::ImageBasedLight ibl(&bg);

    g_scene.addImageBasedLight(&ibl);
#endif

    uint32_t advanceVtxNum = 0;
    uint32_t advanceTriNum = 0;
    std::vector<aten::PrimitiveParamter> deformTris;

    auto deform = getDeformable();

    // Initialize skinning.
    if (deform)
    {
        auto mdl = deform->getHasObjectAsRealType();

        mdl->initGLResources(&g_shdRasterizeDeformable);

        auto& vb = mdl->getVBForGPUSkinning();

        std::vector<aten::SkinningVertex> vtx;
        std::vector<uint32_t> idx;

        int vtxIdOffset = g_ctxt.getVertexNum();

        mdl->getGeometryData(g_ctxt, vtx, idx, deformTris);

        g_skinning.initWithTriangles(
            &vtx[0], vtx.size(),
            &deformTris[0], deformTris.size(),
            &vb);

        advanceVtxNum = vtx.size();
        advanceTriNum = deformTris.size();

        auto anm = getDeformAnm();

        if (anm) {
            g_timeline.init(anm->getDesc().time, real(0));
            g_timeline.enableLoop(true);
            g_timeline.start();
        }
    }

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

        g_triOffset = primparams.size();
        g_vtxOffset = vtxparams.size();

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
            primparams, advanceTriNum,
            vtxparams, advanceVtxNum,
            mtxs,
            tex,
#ifdef ENABLE_ENVMAP
            idaten::EnvmapResource(envmap->id(), ibl.getAvgIlluminace(), real(1)));
#else
            idaten::EnvmapResource());
#endif

#ifdef ENABLE_SVGF
        auto aabb = g_scene.getAccel()->getBoundingbox();
        auto d = aabb.getDiagonalLenght();
        g_tracer.setHitDistanceLimit(d * 0.25f);

        g_tracer.setGBuffer(
            g_fbo.getTexHandle(0),
            g_fbo.getTexHandle(1));
#endif
    }

    // For LBVH.
    if (deform) {
        g_skinning.setVtxOffset(g_vtxOffset);
        g_lbvh.init(advanceTriNum);
    }
    else
    {
        std::vector<std::vector<aten::PrimitiveParamter>> triangles;
        std::vector<int> triIdOffsets;

        aten::DataCollector::collectTriangles(g_ctxt, triangles, triIdOffsets);

        uint32_t maxTriNum = 0;
        for (const auto& tris : triangles) {
            maxTriNum = std::max<uint32_t>(maxTriNum, tris.size());
        }

        g_lbvh.init(maxTriNum);

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
            g_lbvh.build(
                nodes[i + 1],    
                tris,
                triIdOffset,
                sceneBbox,
                vtxPos,
                0,
                (std::vector<aten::ThreadedBvhNode>*)&cpunodes[i + 1]);
        }
    }

#ifdef ENABLE_SVGF
    g_tracer.setMode((idaten::SVGFPathTracing::Mode)g_curMode);
    g_tracer.setAOVMode((idaten::SVGFPathTracing::AOVMode)g_curAOVMode);
    //g_tracer.setCanSSRTHitTest(false);
#endif

    aten::window::run();

    aten::GLProfiler::terminate();

    g_rasterizer.release();
    g_rasterizerAABB.release();
    g_ctxt.release();

    aten::window::terminate();
}
