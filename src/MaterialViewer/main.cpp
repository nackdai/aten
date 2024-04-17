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
//#define WHITE_FURNACE_TEST

#ifdef GPU_RENDERING
constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
#else
constexpr int32_t WIDTH = 512;
constexpr int32_t HEIGHT = 512;
#endif
constexpr char* TITLE = "MaterialViewer";

class MaterialParamEditor : public aten::IMaterialParamEditor {
public:
    MaterialParamEditor() = default;
    ~MaterialParamEditor() = default;

public:
    bool edit(std::string_view name, real& param, real _min = real(0), real _max = real(1)) override final
    {
        return ImGui::SliderFloat(name.data(), &param, _min, _max);
    }

    bool edit(std::string_view name, aten::vec3& param) override final
    {
        std::array f = { param.x, param.y, param.z };
        bool ret = ImGui::ColorEdit3(name.data(), f.data());

        param.x = f[0];
        param.y = f[1];
        param.z = f[2];

        return ret;
    }

    bool edit(std::string_view name, aten::vec4& param) override final
    {
        std::array f = { param.x, param.y, param.z, param.w };
        bool ret = ImGui::ColorEdit4(name.data(), f.data());

        param.x = f[0];
        param.y = f[1];
        param.z = f[2];
        param.w = f[3];

        return ret;
    }

    void edit(std::string_view name, std::string_view str) override final
    {
        std::string s(str);
        ImGui::Text("[%s] : (%s)", name.data(), s.empty() ? "none" : str.data());
    }
};

class MaterialViewerApp {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    MaterialViewerApp() = default;
    ~MaterialViewerApp() = default;

    MaterialViewerApp(const MaterialViewerApp&) = delete;
    MaterialViewerApp(MaterialViewerApp&&) = delete;
    MaterialViewerApp operator=(const MaterialViewerApp&) = delete;
    MaterialViewerApp operator=(MaterialViewerApp&&) = delete;

    bool Init()
    {
        visualizer_ = aten::visualizer::init(WIDTH, HEIGHT);

        gamma_.init(
            WIDTH, HEIGHT,
            "../shader/fullscreen_vs.glsl",
            "../shader/gamma_fs.glsl");

        visualizer_->addPostProc(&gamma_);

        aten::vec3 pos, at;
        real vfov;
        GetCameraPosAndAt(pos, at, vfov);

        camera_.Initalize(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            real(0.1), real(10000.0),
            WIDTH, HEIGHT);

        MakeScene(&scene_);
        scene_.build(ctxt_);

#ifndef WHITE_FURNACE_TEST
        // IBL
        scene_light_.is_envmap = true;
        scene_light_.envmap_texture = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_, asset_manager_);
        scene_light_.envmap = std::make_shared<aten::envmap>();
        scene_light_.envmap->init(scene_light_.envmap_texture);
        scene_light_.ibl = std::make_shared<aten::ImageBasedLight>(scene_light_.envmap);
        scene_.addImageBasedLight(ctxt_, scene_light_.ibl);
#endif

#ifdef GPU_RENDERING
#ifndef WHITE_FURNACE_TEST
        // PointLight
        scene_light_.point_light = std::make_shared<aten::PointLight>(
            aten::vec3(0.0, 0.0, 50.0),
            aten::vec3(1.0, 0.0, 0.0),
            400.0f);
        ctxt_.AddLight(scene_light_.point_light);
#endif

        std::vector<aten::ObjectParameter> shapeparams;
        std::vector<aten::TriangleParameter> primparams;
        std::vector<aten::LightParameter> lightparams;
        std::vector<aten::MaterialParameter> mtrlparms;
        std::vector<aten::vertex> vtxparams;
        std::vector<aten::mat4> mtxs;

        aten::DataCollector::collect(
            ctxt_,
            shapeparams,
            primparams,
            lightparams,
            mtrlparms,
            vtxparams,
            mtxs);

        const auto& nodes = scene_.getAccel()->getNodes();

        std::vector<idaten::TextureResource> tex;
        {
            auto texNum = ctxt_.GetTextureNum();

            for (int32_t i = 0; i < texNum; i++) {
                auto t = ctxt_.GtTexture(i);
                tex.push_back(
                    idaten::TextureResource(t->colors(), t->width(), t->height()));
            }
        }

        MershallLightParameter(lightparams);

        auto camparam = camera_.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        renderer_.getCompaction().init(
            WIDTH * HEIGHT,
            1024);

        renderer_.update(
            visualizer_->GetGLTextureHandle(),
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
            scene_light_.is_envmap
            ? idaten::EnvmapResource(scene_light_.envmap_texture->id(), scene_light_.ibl->getAvgIlluminace(), real(1))
            : idaten::EnvmapResource());

        renderer_.setEnableEnvmap(scene_light_.is_envmap);
#endif
        return true;
    }

    bool Run()
    {
#ifdef GPU_RENDERING
        float updateTime = 0.0f;

        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = real(0.1);
            camparam.zfar = real(10000.0);

            renderer_.updateCamera(camparam);
            is_camera_dirty_ = false;

            visualizer_->clear();
            progressive_accumulate_count_ = 0;
        }

        renderer_.render(
            WIDTH, HEIGHT,
            max_samples_,
            max_bounce_);

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        visualizer_->render(false);

        if (will_take_screen_shot_)
        {
            auto screen_shot_file_name = aten::StringFormat("sc_%d.png", screen_shot_count_);

            visualizer_->takeScreenshot(screen_shot_file_name);

            will_take_screen_shot_ = false;
            screen_shot_count_++;

            AT_PRINTF("Take Screenshot[%s]\n", screen_shot_file_name.c_str());
        }

        if (will_show_gui_)
        {
            bool need_renderer_reset = false;

            bool is_aov_rendering = renderer_.GetRenderingMode() == decltype(renderer_)::Mode::AOV;
            if (ImGui::Checkbox("AOV", &is_aov_rendering)) {
                renderer_.SetRenderingMode(
                    is_aov_rendering ? decltype(renderer_)::Mode::AOV : decltype(renderer_)::Mode::PT);
                need_renderer_reset = true;
            }

            constexpr std::array light_types = { "IBL", "PointLight" };
            int32_t lighttype = scene_light_.is_envmap ? 0 : 1;
            if (ImGui::Combo("light", &lighttype, light_types.data(), light_types.size())) {
                auto next_is_envmap = lighttype == 0;
                if (scene_light_.is_envmap != next_is_envmap) {
                    scene_light_.is_envmap = next_is_envmap;
                    UpdateLightParameter();
                    need_renderer_reset = true;
                }
            }

            if (ImGui::SliderInt("Samples", &max_samples_, 1, 100)
                || ImGui::SliderInt("Bounce", &max_bounce_, 1, 10))
            {
                need_renderer_reset = true;
            }

            bool isProgressive = renderer_.IsEnableProgressive();

            if (ImGui::Checkbox("Progressive", &isProgressive)) {
                renderer_.SetEnableProgressive(isProgressive);
                need_renderer_reset = true;
            }

            if (isProgressive) {
                ImGui::Text("%d", progressive_accumulate_count_);
                progressive_accumulate_count_ += 1;
            }

            auto mtrl = ctxt_.GetMaterialInstance(0);
            bool needUpdateMtrl = false;

            constexpr std::array mtrl_types = {
                "Emissive",
                "Lambert",
                "OrneNayar",
                "Specular",
                "Refraction",
                "GGX",
                "Beckman",
                "Velvet",
                "MicrofacetRefraction",
                "Retroreflective",
                "CarPaint",
                "Disney",
            };
            int32_t mtrlType = static_cast<int32_t>(mtrl->param().type);
            if (ImGui::Combo("mode", &mtrlType, mtrl_types.data(), mtrl_types.size())) {
                ctxt_.DeleteAllMaterialsAndClearList();
                mtrl = CreateMaterial(static_cast<aten::MaterialType>(mtrlType));
                mtrl->param().baseColor.set(1.0F);
                needUpdateMtrl = true;
            }

            {
                bool b0 = ImGui::Checkbox("AlbedoMap", &enable_albedo_map_);
                bool b1 = ImGui::Checkbox("NormalMap", &enable_normal_map_);

                if (b0 || b1) {
                    mtrl->setTextures(
                        enable_albedo_map_ ? albedo_map_ : nullptr,
                        enable_normal_map_ ? normal_map_ : nullptr,
                        nullptr);

                    needUpdateMtrl = true;
                }
            }

            if (mtrl->edit(&mtrl_param_editor_)) {
                needUpdateMtrl = true;
            }

            {
                auto camPos = camera_.getPos();
                auto camAt = camera_.getAt();

                ImGui::Text("Pos (%f, %f, %f)", camPos.x, camPos.y, camPos.z);
                ImGui::Text("At  (%f, %f, %f)", camAt.x, camAt.y, camAt.z);
            }

            if (needUpdateMtrl) {
                std::vector<aten::MaterialParameter> params(1);
                params[0] = mtrl->param();
                renderer_.updateMaterial(params);
                need_renderer_reset = true;
            }

            if (need_renderer_reset) {
                renderer_.reset();
                progressive_accumulate_count_ = 0;
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
            dst.buffer = &host_renderer_dst_;
        }

        host_renderer_.render(ctxt_, dst, &scene_, &camera_);

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        visualizer_->renderPixelData(host_renderer_dst_.image().data(), camera_.needRevert());
#endif

        return true;
    }

    void OnClose()
    {
    }

    void OnMouseBtn(bool left, bool press, int32_t x, int32_t y)
    {
        is_mouse_l_btn_down_ = false;
        is_mouse_r_btn_down_ = false;

        if (press)
        {
            prev_mouse_pos_x_ = x;
            prev_mouse_pos_y_ = y;

            is_mouse_l_btn_down_ = left;
            is_mouse_r_btn_down_ = !left;
        }
    }

    void OnMouseMove(int32_t x, int32_t y)
    {
        if (is_mouse_l_btn_down_)
        {
            aten::CameraOperator::rotate(
                camera_,
                WIDTH, HEIGHT,
                prev_mouse_pos_x_, prev_mouse_pos_y_,
                x, y);
            is_camera_dirty_ = true;
        }
        else if (is_mouse_r_btn_down_)
        {
            aten::CameraOperator::move(
                camera_,
                prev_mouse_pos_x_, prev_mouse_pos_y_,
                x, y,
                real(0.001));
            is_camera_dirty_ = true;
        }

        prev_mouse_pos_x_ = x;
        prev_mouse_pos_y_ = y;
    }

    void OnMouseWheel(int32_t delta)
    {
        aten::CameraOperator::dolly(camera_, delta * real(0.1));
        is_camera_dirty_ = true;
    }

    void OnKey(bool press, aten::Key key)
    {
        static const real offset_base = real(0.1);

        if (press)
        {
            if (key == aten::Key::Key_F1)
            {
                will_show_gui_ = !will_show_gui_;
                return;
            }
            else if (key == aten::Key::Key_F2)
            {
                will_take_screen_shot_ = true;
                return;
            }
        }

        auto offset = offset_base;

        if (press)
        {
            switch (key)
            {
            case aten::Key::Key_W:
            case aten::Key::Key_UP:
                aten::CameraOperator::moveForward(camera_, offset);
                break;
            case aten::Key::Key_S:
            case aten::Key::Key_DOWN:
                aten::CameraOperator::moveForward(camera_, -offset);
                break;
            case aten::Key::Key_D:
            case aten::Key::Key_RIGHT:
                aten::CameraOperator::moveRight(camera_, offset);
                break;
            case aten::Key::Key_A:
            case aten::Key::Key_LEFT:
                aten::CameraOperator::moveRight(camera_, -offset);
                break;
            case aten::Key::Key_Z:
                aten::CameraOperator::moveUp(camera_, offset);
                break;
            case aten::Key::Key_X:
                aten::CameraOperator::moveUp(camera_, -offset);
                break;
            case aten::Key::Key_R:
            {
                aten::vec3 pos, at;
                real vfov;
                GetCameraPosAndAt(pos, at, vfov);

                camera_.init(
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

            is_camera_dirty_ = true;
        }
    }

    aten::context& GetContext()
    {
        return ctxt_;
    }

private:
    void GetCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        real& fov)
    {
        pos = aten::vec3(0.f, 1.f, 10.f);
        at = aten::vec3(0.f, 1.f, 0.f);
        fov = 45;
    }

    void MakeScene(aten::scene* scene)
    {
        aten::MaterialParameter mtrlParam;
        mtrlParam.type = aten::MaterialType::Velvet;
#ifdef WHITE_FURNACE_TEST
        mtrlParam.baseColor = aten::vec3(1.0F);
#else
        mtrlParam.baseColor = aten::vec3(0.580000f, 0.580000f, 0.580000f);
#endif
        mtrlParam.standard.ior = 1.5F;
        mtrlParam.standard.roughness = 0.5F;

        auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(
            mtrlParam,
            nullptr, nullptr, nullptr);

#if 0
        constexpr char* asset_path = "../../asset/suzanne/suzanne.obj";
        constexpr char* mtrl_in_asset = "Material.001";
#elif 1
        constexpr char* asset_path = "../../asset/teapot/teapot.obj";
        constexpr char* mtrl_in_asset = "m1";
#else
        constexpr char* asset_path = "../../asset/sphere/sphere.obj";
        constexpr char* mtrl_in_asset = "m1";
#endif

        asset_manager_.registerMtrl(mtrl_in_asset, mtrl);

        auto obj = aten::ObjLoader::load(asset_path, ctxt_, asset_manager_);
        auto poly_obj = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt_, obj, aten::mat4::Identity);
        scene->add(poly_obj);

        // TODO
        //albedo_map_ = aten::ImageLoader::load("../../asset/sponza/01_STUB.JPG");
        //normal_map_ = aten::ImageLoader::load("../../asset/sponza/01_STUB-nml.png");

        obj->getShapes()[0]->GetMaterial()->setTextures(albedo_map_, normal_map_, nullptr);
    }

    std::shared_ptr<aten::material> CreateMaterial(aten::MaterialType type)
    {
        auto mtrl = ctxt_.CreateMaterialWithDefaultValue(type);

        if (mtrl) {
            mtrl->setTextures(
                enable_albedo_map_ ? albedo_map_ : nullptr,
                enable_normal_map_ ? normal_map_ : nullptr,
                nullptr);
        }

        return mtrl;
    }

    void MershallLightParameter(std::vector<aten::LightParameter>& lightparams)
    {
        if (scene_light_.is_envmap) {
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

    void UpdateLightParameter()
    {
        std::vector<aten::LightParameter> lightparams;

        auto lightNum = ctxt_.GetLightNum();

        for (uint32_t i = 0; i < lightNum; i++) {
            const auto& param = ctxt_.GetLight(i);
            lightparams.push_back(param);
        }

        MershallLightParameter(lightparams);

        renderer_.updateLight(lightparams);
        renderer_.setEnableEnvmap(scene_light_.is_envmap);
    }


    struct SceneLight {
        bool is_envmap{ false };

        std::shared_ptr<aten::texture> envmap_texture;
        std::shared_ptr<aten::envmap> envmap;
        std::shared_ptr<aten::ImageBasedLight> ibl;
        std::shared_ptr<aten::PointLight> point_light;
    } scene_light_;

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    aten::AcceleratedScene<aten::GPUBvh> scene_;
    aten::context ctxt_;

    aten::AssetManager asset_manager_;

    idaten::PathTracing renderer_;

    aten::PathTracing host_renderer_;
    aten::FilmProgressive host_renderer_dst_{ WIDTH, HEIGHT };

    std::shared_ptr<aten::visualizer> visualizer_;

    aten::GammaCorrection gamma_;

    aten::texture* albedo_map_{ nullptr };
    aten::texture* normal_map_{ nullptr };

    bool enable_albedo_map_{ false };
    bool enable_normal_map_{ false };

    MaterialParamEditor mtrl_param_editor_;

    int32_t max_samples_{ 1 };
    int32_t max_bounce_{ 5 };

    bool will_show_gui_{ true };
    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };

    int32_t progressive_accumulate_count_{ 0 };
};

int32_t main()
{
    aten::timer::init();
    aten::OMPUtil::setThreadNum(MaterialViewerApp::ThreadNum);

    aten::initSampler(WIDTH, HEIGHT);

    auto app = std::make_shared<MaterialViewerApp>();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        std::bind(&MaterialViewerApp::Run, app),
        std::bind(&MaterialViewerApp::OnClose, app),
        std::bind(&MaterialViewerApp::OnMouseBtn, app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
        std::bind(&MaterialViewerApp::OnMouseMove, app, std::placeholders::_1, std::placeholders::_2),
        std::bind(&MaterialViewerApp::OnMouseWheel, app, std::placeholders::_1),
        std::bind(&MaterialViewerApp::OnKey, app, std::placeholders::_1, std::placeholders::_2));

    if (id >= 0) {
        app->GetContext().SetIsWindowInitialized(true);
    }
    else {
        AT_ASSERT(false);
        return 1;
    }

    app->Init();

    aten::GLProfiler::start();

    wnd->Run();

    aten::GLProfiler::terminate();

    app.reset();

    wnd->Terminate();
}
