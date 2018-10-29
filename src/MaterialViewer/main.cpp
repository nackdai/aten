#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

#define GPU_RENDERING

#define ENABLE_ENVMAP

static int WIDTH = 1280;
static int HEIGHT = 720;
static const char* TITLE = "MaterialViewer";

#ifdef ENABLE_OMP
static uint32_t g_threadnum = 8;
#else
static uint32_t g_threadnum = 1;
#endif

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static aten::AcceleratedScene<aten::GPUBvh> g_scene;
static aten::context g_ctxt;

static idaten::PathTracing g_tracer;

static aten::PathTracing g_cpuPT;
static aten::FilmProgressive g_buffer(WIDTH, HEIGHT);

static aten::visualizer* g_visualizer;

static aten::texture* g_albedoMap = nullptr;
static aten::texture* g_normalMap = nullptr;

static bool g_willShowGUI = true;
static bool g_willTakeScreenShot = false;
static int g_cntScreenShot = 0;

static int g_maxSamples = 1;
static int g_maxBounce = 5;

static bool g_enableAlbedoMap = true;
static bool g_enableNormalMap = true;

void getCameraPosAndAt(
    aten::vec3& pos,
    aten::vec3& at,
    real& fov)
{
    pos = aten::vec3(0.f, 1.f, 10.f);
    at = aten::vec3(0.f, 1.f, 0.f);
    fov = 45;
}

void makeScene(aten::scene* scene)
{
    aten::MaterialParameter mtrlParam;
    {
        mtrlParam.baseColor = aten::vec3(0.580000, 0.580000, 0.580000);

        mtrlParam.carpaint.clearcoatRoughness = real(0.5);
        mtrlParam.carpaint.flakeLayerRoughness = real(0.5);

        mtrlParam.carpaint.flake_scale = real(100);
        mtrlParam.carpaint.flake_size = real(0.01);
        mtrlParam.carpaint.flake_size_variance = real(0.25);
        mtrlParam.carpaint.flake_normal_orientation = real(0.5);

        mtrlParam.carpaint.flake_reflection = real(0.5);
        mtrlParam.carpaint.flake_transmittance = real(0.5);

        mtrlParam.carpaint.glitterColor = mtrlParam.baseColor;
        mtrlParam.carpaint.flakeColor = mtrlParam.baseColor;

        mtrlParam.carpaint.flake_intensity = real(1);
    }

    auto mtrl = g_ctxt.createMaterialWithMaterialParameter(
        aten::MaterialType::CarPaint,
        mtrlParam,
        nullptr, nullptr, nullptr);

    aten::AssetManager::registerMtrl("m1", mtrl);

    auto obj = aten::ObjLoader::load("../../asset/teapot/teapot.obj", g_ctxt);
    auto teapot = aten::TransformableFactory::createInstance<aten::object>(g_ctxt, obj, aten::mat4::Identity);
    scene->add(teapot);

    // TODO
    //g_albedoMap = aten::ImageLoader::load("../../asset/sponza/01_STUB.JPG");
    //g_normalMap = aten::ImageLoader::load("../../asset/sponza/01_STUB-nml.png");

    obj->getShape(0)->getMaterial()->setTextures(g_albedoMap, g_normalMap, nullptr);
}

aten::material* createMaterial(aten::MaterialType type)
{
    aten::material* mtrl = g_ctxt.createMaterialWithDefaultValue(type);

    if (mtrl) {
        mtrl->setTextures(
            g_enableAlbedoMap ? g_albedoMap : nullptr,
            g_enableNormalMap ? g_normalMap : nullptr,
            nullptr);
    }

    return mtrl;
}

class MaterialParamEditor : public aten::IMaterialParamEditor {
public:
    MaterialParamEditor() {}
    virtual ~MaterialParamEditor() {}

public:
    virtual bool edit(const char* name, real& param, real _min = real(0), real _max = real(1)) override final
    {
        return ImGui::SliderFloat(name, &param, _min, _max);
    }

    virtual bool edit(const char* name, aten::vec3& param) override final
    {
        float f[3] = { param.x, param.y, param.z };
        bool ret = ImGui::ColorEdit3(name, f);

        param.x = f[0];
        param.y = f[1];
        param.z = f[2];

        return ret;
    }

    virtual void edit(const char* name, const char* str) override final
    {
        std::string s(str);
        ImGui::Text("[%s] : (%s)", name, s.empty() ? "none" : str);
    }
};

static MaterialParamEditor g_mtrlParamEditor;

void onRun(aten::window* window)
{
#ifdef GPU_RENDERING
    float updateTime = 0.0f;

    if (g_isCameraDirty) {
        g_camera.update();

        auto camparam = g_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        g_tracer.updateCamera(camparam);
        g_isCameraDirty = false;

        g_visualizer->clear();
    }

    g_tracer.render(
        idaten::TileDomain(0, 0, WIDTH, HEIGHT),
        g_maxSamples,
        g_maxBounce);

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
        if (ImGui::SliderInt("Samples", &g_maxSamples, 1, 100)
            || ImGui::SliderInt("Bounce", &g_maxBounce, 1, 10))
        {
            g_tracer.reset();
        }

        bool isProgressive = g_tracer.isProgressive();

        if (ImGui::Checkbox("Progressive", &isProgressive)) {
            g_tracer.enableProgressive(isProgressive);
            g_tracer.reset();
        }

        auto mtrl = g_ctxt.getMaterial(0);
        bool needUpdateMtrl = false;

        static const char* items[] = {
            "Emissive",
            "Lambert",
            "OrneNayar",
            "Specular",
            "Refraction",
            "Blinn",
            "GGX",
            "Beckman",
            "Velvet",
            "LambertRefraction",
            "MicrofacetRefraction",
            "Disney",
            "CarPaint",
        };
        int mtrlType = (int)mtrl->param().type;
        if (ImGui::Combo("mode", &mtrlType, items, AT_COUNTOF(items))) {
            g_ctxt.deleteAllMaterialsAndClearList();
            mtrl = createMaterial((aten::MaterialType)mtrlType);
            needUpdateMtrl = true;
        }

        {
            bool b0 = ImGui::Checkbox("AlbedoMap", &g_enableAlbedoMap);
            bool b1 = ImGui::Checkbox("NormalMap", &g_enableNormalMap);

            if (b0 || b1) {
                mtrl->setTextures(
                    g_enableAlbedoMap ? g_albedoMap : nullptr,
                    g_enableNormalMap ? g_normalMap : nullptr,
                    nullptr);

                needUpdateMtrl = true;
            }
        }

        if (mtrl->edit(&g_mtrlParamEditor)) {
            needUpdateMtrl = true;
        }

        {
            auto camPos = g_camera.getPos();
            auto camAt = g_camera.getAt();

            ImGui::Text("Pos (%f, %f, %f)", camPos.x, camPos.y, camPos.z);
            ImGui::Text("At  (%f, %f, %f)", camAt.x, camAt.y, camAt.z);
        }

        if (needUpdateMtrl) {
            std::vector<aten::MaterialParameter> params(1);
            params[0] = mtrl->param();
            g_tracer.updateMaterial(params);
            g_tracer.reset();
        }

        window->drawImGui();
    }
#else
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

    g_cpuPT.render(dst, &g_scene, &g_camera);

    g_visualizer->render(g_buffer.image(), g_camera.needRevert());
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
            getCameraPosAndAt(pos, at, vfov);

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

    auto wnd = aten::window::init(
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

    g_visualizer->addPostProc(&gamma);
    //aten::visualizer::addPostProc(&blitter);

    aten::vec3 pos, at;
    real vfov;
    getCameraPosAndAt(pos, at, vfov);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    makeScene(&g_scene);
    g_scene.build(g_ctxt);

    g_tracer.getCompaction().init(
        WIDTH * HEIGHT,
        1024);

    auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", g_ctxt);
    aten::envmap bg;
    bg.init(envmap);
    aten::ImageBasedLight ibl(&bg);

    g_scene.addImageBasedLight(&ibl);

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
            idaten::EnvmapResource(envmap->id(), ibl.getAvgIlluminace(), real(1)));
    }

    aten::window::run();

    aten::GLProfiler::terminate();

    aten::window::terminate();
}
