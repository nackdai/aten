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

static int32_t WIDTH = 1280;
static int32_t HEIGHT = 720;
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

static std::shared_ptr<aten::visualizer> g_visualizer;

static aten::texture* g_albedoMap = nullptr;
static aten::texture* g_normalMap = nullptr;

static bool g_willShowGUI = true;
static bool g_willTakeScreenShot = false;
static int32_t g_cntScreenShot = 0;

static int32_t g_maxSamples = 1;
static int32_t g_maxBounce = 5;

static bool g_enableAlbedoMap = true;
static bool g_enableNormalMap = true;

static struct SceneLight {
    bool is_envmap{ true };

    std::shared_ptr<aten::texture> envmap_texture;
    std::shared_ptr<aten::envmap> envmap;
    std::shared_ptr<aten::ImageBasedLight> ibl;
    std::shared_ptr<aten::PointLight> point_light;
} g_scene_light;

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
    mtrlParam.type = aten::MaterialType::Lambert;
    mtrlParam.baseColor = aten::vec3(0.580000, 0.580000, 0.580000);

    auto mtrl = g_ctxt.CreateMaterialWithMaterialParameter(
        mtrlParam,
        nullptr, nullptr, nullptr);

    aten::AssetManager::registerMtrl("m1", mtrl);

    auto obj = aten::ObjLoader::load("../../asset/teapot/teapot.obj", g_ctxt);
    auto teapot = aten::TransformableFactory::createInstance<aten::PolygonObject>(g_ctxt, obj, aten::mat4::Identity);
    scene->add(teapot);

    // TODO
    //g_albedoMap = aten::ImageLoader::load("../../asset/sponza/01_STUB.JPG");
    //g_normalMap = aten::ImageLoader::load("../../asset/sponza/01_STUB-nml.png");

    obj->getShapes()[0]->GetMaterial()->setTextures(g_albedoMap, g_normalMap, nullptr);
}

std::shared_ptr<aten::material> CreateMaterial(aten::MaterialType type)
{
    auto mtrl = g_ctxt.CreateMaterialWithDefaultValue(type);

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
    virtual bool edit(std::string_view name, real& param, real _min = real(0), real _max = real(1)) override final
    {
        return ImGui::SliderFloat(name.data(), &param, _min, _max);
    }

    virtual bool edit(std::string_view name, aten::vec3& param) override final
    {
        float f[3] = { param.x, param.y, param.z };
        bool ret = ImGui::ColorEdit3(name.data(), f);

        param.x = f[0];
        param.y = f[1];
        param.z = f[2];

        return ret;
    }

    virtual bool edit(std::string_view name, aten::vec4& param) override final
    {
        float f[4] = { param.x, param.y, param.z, param.w };
        bool ret = ImGui::ColorEdit4(name.data(), f);

        param.x = f[0];
        param.y = f[1];
        param.z = f[2];
        param.w = f[3];

        return ret;
    }

    virtual void edit(std::string_view name, std::string_view str) override final
    {
        std::string s(str);
        ImGui::Text("[%s] : (%s)", name.data(), s.empty() ? "none" : str.data());
    }
};

static MaterialParamEditor g_mtrlParamEditor;

void mershallLightParameter(std::vector<aten::LightParameter>& lightparams)
{
    if (g_scene_light.is_envmap) {
        auto result = std::remove_if(lightparams.begin(), lightparams.end(),
            [](const auto& l) {
                return l.type != aten::LightType::IBL;
            }
        );
        lightparams.erase(result, lightparams.end());
    }
    else {
        auto result = std::remove_if(lightparams.begin(), lightparams.end(),
            [](const auto& l) {
                return l.type == aten::LightType::IBL;
            }
        );
        lightparams.erase(result, lightparams.end());
    }
}

void updateLightParameter()
{
    std::vector<aten::LightParameter> lightparams;

    auto lightNum = g_ctxt.GetLightNum();

    for (uint32_t i = 0; i < lightNum; i++) {
        const auto& param = g_ctxt.GetLight(i);
        lightparams.push_back(param);
    }

    mershallLightParameter(lightparams);

    g_tracer.updateLight(lightparams);
    g_tracer.setEnableEnvmap(g_scene_light.is_envmap);
}

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
        WIDTH, HEIGHT,
        g_maxSamples,
        g_maxBounce);

    aten::RasterizeRenderer::clearBuffer(
        aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
        aten::vec4(0, 0.5f, 1.0f, 1.0f),
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
        bool need_renderer_reset = false;

        constexpr char* light_types[] = { "IBL", "PointLight" };
        int32_t lighttype = g_scene_light.is_envmap ? 0 : 1;
        if (ImGui::Combo("light", &lighttype, light_types, AT_COUNTOF(light_types))) {
            auto next_is_envmap = lighttype == 0;
            if (g_scene_light.is_envmap != next_is_envmap) {
                g_scene_light.is_envmap = next_is_envmap;
                updateLightParameter();
                need_renderer_reset = true;
            }
        }

        if (ImGui::SliderInt("Samples", &g_maxSamples, 1, 100)
            || ImGui::SliderInt("Bounce", &g_maxBounce, 1, 10))
        {
            g_tracer.reset();
        }

        bool isProgressive = g_tracer.isEnableProgressive();

        if (ImGui::Checkbox("Progressive", &isProgressive)) {
            g_tracer.setEnableProgressive(isProgressive);
            g_tracer.reset();
        }

        auto mtrl = g_ctxt.GetMaterialInstance(0);
        bool needUpdateMtrl = false;

        constexpr char* mtrl_types[] = {
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
            "Retroreflective",
            "CarPaint",
            "Disney",
        };
        int32_t mtrlType = (int32_t)mtrl->param().type;
        if (ImGui::Combo("mode", &mtrlType, mtrl_types, AT_COUNTOF(mtrl_types))) {
            g_ctxt.DeleteAllMaterialsAndClearList();
            mtrl = CreateMaterial((aten::MaterialType)mtrlType);
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
            need_renderer_reset = true;
        }

        if (need_renderer_reset) {
            g_tracer.reset();
        }
    }
#else
    aten::Destination dst;
    {
        dst.width = WIDTH;
        dst.height = HEIGHT;
        dst.maxDepth = 5;
        dst.russianRouletteDepth = 3;
        dst.sample = 1;
        dst.buffer = &g_buffer;
    }

    g_cpuPT.render(dst, &g_scene, &g_camera);

    aten::RasterizeRenderer::clearBuffer(
        aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
        aten::vec4(0, 0.5f, 1.0f, 1.0f),
        1.0f,
        0);

    g_visualizer->render(g_buffer.image(), g_camera.needRevert());
#endif
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

    if (press) {
        g_prevX = x;
        g_prevY = y;

        g_isMouseLBtnDown = left;
        g_isMouseRBtnDown = !left;
    }
}

void onMouseMove(int32_t x, int32_t y)
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

void onMouseWheel(int32_t delta)
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

    {
        // IBL
        g_scene_light.envmap_texture = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", g_ctxt);
        g_scene_light.envmap = std::make_shared<aten::envmap>();
        g_scene_light.envmap->init(g_scene_light.envmap_texture);
        g_scene_light.ibl = std::make_shared<aten::ImageBasedLight>(g_scene_light.envmap);
        g_scene.addImageBasedLight(g_ctxt, g_scene_light.ibl);

        // PointLight
        g_scene_light.point_light = std::make_shared<aten::PointLight>(
            aten::vec3(0.0, 0.0, 50.0),
            aten::vec3(1.0, 0.0, 0.0),
            400.0f);
        g_ctxt.AddLight(g_scene_light.point_light);
    }

    std::vector<aten::ObjectParameter> shapeparams;
    std::vector<aten::TriangleParameter> primparams;
    std::vector<aten::LightParameter> lightparams;
    std::vector<aten::MaterialParameter> mtrlparms;
    std::vector<aten::vertex> vtxparams;
    std::vector<aten::mat4> mtxs;

    aten::DataCollector::collect(
        g_ctxt,
        shapeparams,
        primparams,
        lightparams,
        mtrlparms,
        vtxparams,
        mtxs);

    const auto& nodes = g_scene.getAccel()->getNodes();

    std::vector<idaten::TextureResource> tex;
    {
        auto texNum = g_ctxt.GetTextureNum();

        for (int32_t i = 0; i < texNum; i++) {
            auto t = g_ctxt.GtTexture(i);
            tex.push_back(
                idaten::TextureResource(t->colors(), t->width(), t->height()));
        }
    }

    mershallLightParameter(lightparams);

    auto camparam = g_camera.param();
    camparam.znear = real(0.1);
    camparam.zfar = real(10000.0);

    g_tracer.update(
        g_visualizer->GetGLTextureHandle(),
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
        g_scene_light.is_envmap
        ? idaten::EnvmapResource(g_scene_light.envmap_texture->id(), g_scene_light.ibl->getAvgIlluminace(), real(1))
        : idaten::EnvmapResource());

    g_tracer.setEnableEnvmap(g_scene_light.is_envmap);

    aten::window::run();

    aten::GLProfiler::terminate();

    aten::window::terminate();
}
