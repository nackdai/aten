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

#pragma optimize( "", off)


#define GPU_RENDERING
//#define WHITE_FURNACE_TEST

#ifdef GPU_RENDERING
constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
#else
constexpr int32_t WIDTH = 512;
constexpr int32_t HEIGHT = 512;
#endif
constexpr const char* TITLE = "MaterialViewer";

class MaterialParamEditor : public aten::IMaterialParamEditor {
public:
    MaterialParamEditor() = default;
    ~MaterialParamEditor() = default;

public:
    bool edit(std::string_view name, float& param, float _min, float _max) override final
    {
        return ImGui::SliderFloat(name.data(), &param, _min, _max);
    }

    bool edit(std::string_view name, bool& param) override final
    {
        return ImGui::Checkbox(name.data(), &param);
    }

    bool edit(std::string_view name, aten::vec3& param) override final
    {
        const auto ret = ImGui::ColorEdit3(name.data(), reinterpret_cast<float*>(&param));
        return ret;
    }

    bool edit(std::string_view name, aten::vec4& param) override final
    {
        const auto ret = ImGui::ColorEdit4(name.data(), reinterpret_cast<float*>(&param));
        return ret;
    }

    void edit(std::string_view name, std::string_view str) override final
    {
        ImGui::Text("[%s] : (%s)", name.data(), str.empty() ? "none" : str.data());
    }

    bool edit(std::string_view name, const std::vector<const char*>& elements, int32_t& param) override final
    {
        const auto ret = ImGui::Combo(name.data(), &param, elements.data(), static_cast<int32_t>(elements.size()));
        return ret;
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
        float vfov;
        GetCameraPosAndAt(pos, at, vfov);

        camera_.Initalize(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            float(0.1), float(10000.0),
            WIDTH, HEIGHT);

        MakeScene(&scene_);
        scene_.build(ctxt_);

#ifdef WHITE_FURNACE_TEST
        auto bg = AT_NAME::Background::CreateBackgroundResource(nullptr);
#else
        // IBL
        scene_light_.is_envmap = true;
        scene_light_.envmap_texture = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_);
        auto bg = AT_NAME::Background::CreateBackgroundResource(scene_light_.envmap_texture, aten::vec4(0));
        scene_light_.ibl = std::make_shared<aten::ImageBasedLight>(bg, ctxt_);
        scene_.addImageBasedLight(ctxt_, scene_light_.ibl);

        // PointLight
        scene_light_.point_light = std::make_shared<aten::PointLight>(
            aten::vec3(0.0, 0.0, 50.0),
            aten::vec3(1.0, 0.0, 0.0),
            4000.0f);
#endif

#ifdef GPU_RENDERING
        const auto& nodes = scene_.getAccel()->getNodes();

        auto camparam = camera_.param();
        camparam.znear = float(0.1);
        camparam.zfar = float(10000.0);

        renderer_.getCompaction().init(
            WIDTH * HEIGHT,
            1024);

        renderer_.UpdateSceneData(
            visualizer_->GetGLTextureHandle(),
            WIDTH, HEIGHT,
            camparam, ctxt_, nodes,
            0, 0, bg);

        renderer_.SetEnableEnvmap(scene_light_.is_envmap);
#else
        host_renderer_.SetBG(bg);
        host_renderer_.SetEnableEnvmap(scene_light_.is_envmap);
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
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

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
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Stencil,
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
            if (ImGui::Combo("light", &lighttype, light_types.data(), static_cast<int32_t>(light_types.size()))) {
                ctxt_.ClearAllLights();

                auto next_is_envmap = lighttype == 0;
                if (next_is_envmap) {
                    ctxt_.AddLight(scene_light_.ibl);
                }
                else {
                    ctxt_.AddLight(scene_light_.point_light);
                }

                need_renderer_reset = true;

                scene_light_.is_envmap = next_is_envmap;
                renderer_.SetEnableEnvmap(scene_light_.is_envmap);
                renderer_.updateLight(ctxt_);
            }

            if (!scene_light_.is_envmap) {
                auto& point_light = ctxt_.GetLightInstance(0);
                auto& light_param = point_light->param();

                bool is_updated = false;
                is_updated |= ImGui::ColorEdit3("LightColor", reinterpret_cast<float*>(&light_param.light_color));
                is_updated |= ImGui::SliderFloat("LightIntensity", &light_param.intensity, 0.0F, 10000.0F);
                if (is_updated) {
                    renderer_.updateLight(ctxt_);
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
                "Diffuse",
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
                "Toon",
            };
            if (mtrl) {
                int32_t mtrlType = static_cast<int32_t>(mtrl->param().type);
                if (ImGui::Combo("mode", &mtrlType, mtrl_types.data(), static_cast<int32_t>(mtrl_types.size()))) {
                    auto mtrl_param = mtrl->param();
                    mtrl_param.type = static_cast<aten::MaterialType>(mtrlType);

                    ctxt_.DeleteAllMaterialsAndClearList();
                    mtrl = CreateMaterial(mtrl_param);

                    // TODO
                    auto& param = mtrl->param();
                    param.toon.target_light_idx = 0;
                    param.toon.remap_texture = 0;

                    needUpdateMtrl = true;
                }

                bool b0 = ImGui::Checkbox("AlbedoMap", &enable_albedo_map_);
                bool b1 = ImGui::Checkbox("NormalMap", &enable_normal_map_);

                if (b0 || b1) {
                    mtrl->setTextures(
                        enable_albedo_map_ ? albedo_map_ : nullptr,
                        enable_normal_map_ ? normal_map_ : nullptr,
                        nullptr);

                    needUpdateMtrl = true;
                }

                if (mtrl->edit(&mtrl_param_editor_)) {
                    needUpdateMtrl = true;
                }

                if (needUpdateMtrl) {
                    std::vector<aten::MaterialParameter> params(1);
                    params[0] = mtrl->param();
                    renderer_.updateMaterial(params);
                    need_renderer_reset = true;
                }
            }

            {
                auto camPos = camera_.GetPos();
                auto camAt = camera_.GetAt();

                ImGui::Text("Pos (%f, %f, %f)", camPos.x, camPos.y, camPos.z);
                ImGui::Text("At  (%f, %f, %f)", camAt.x, camAt.y, camAt.z);
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
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Stencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        visualizer_->renderPixelData(host_renderer_dst_.image().data(), camera_.NeedRevert());

#if 0
        {
            auto screen_shot_file_name = aten::StringFormat("sc_%d.png", screen_shot_count_);

            visualizer_->takeScreenshot(screen_shot_file_name);

            will_take_screen_shot_ = false;
            screen_shot_count_++;

            AT_PRINTF("Take Screenshot[%s]\n", screen_shot_file_name.c_str());
        }
#endif
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
            aten::CameraOperator::Rotate(
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
                float(0.001));
            is_camera_dirty_ = true;
        }

        prev_mouse_pos_x_ = x;
        prev_mouse_pos_y_ = y;
    }

    void OnMouseWheel(int32_t delta)
    {
        aten::CameraOperator::Dolly(camera_, delta * float(0.1));
        is_camera_dirty_ = true;
    }

    void OnKey(bool press, aten::Key key)
    {
        static const float offset_base = float(0.1);

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
                aten::CameraOperator::MoveForward(camera_, offset);
                break;
            case aten::Key::Key_S:
            case aten::Key::Key_DOWN:
                aten::CameraOperator::MoveForward(camera_, -offset);
                break;
            case aten::Key::Key_D:
            case aten::Key::Key_RIGHT:
                aten::CameraOperator::MoveRight(camera_, offset);
                break;
            case aten::Key::Key_A:
            case aten::Key::Key_LEFT:
                aten::CameraOperator::MoveRight(camera_, -offset);
                break;
            case aten::Key::Key_Z:
                aten::CameraOperator::MoveUp(camera_, offset);
                break;
            case aten::Key::Key_X:
                aten::CameraOperator::MoveUp(camera_, -offset);
                break;
            case aten::Key::Key_R:
            {
                aten::vec3 pos, at;
                float vfov;
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
        float& fov)
    {
        pos = aten::vec3(0.f, 0.f, 10.f);
        at = aten::vec3(0.f, 0.f, 0.f);
        fov = 45;
    }

    void MakeScene(aten::scene* scene)
    {
        aten::MaterialParameter mtrl_param;
        mtrl_param.type = aten::MaterialType::Retroreflective;
#ifdef WHITE_FURNACE_TEST
        mtrl_param.baseColor = aten::vec3(1.0F);
#else
        mtrl_param.baseColor = aten::vec3(0.580000f, 0.580000f, 0.580000f);
#endif
        mtrl_param.standard.ior = 1.333F;
        mtrl_param.standard.roughness = 0.011F;

#if 1
#if 0
        constexpr const char* asset_path = "../../asset/suzanne/suzanne.obj";
        constexpr const char* mtrl_in_asset = "Material.001";
#elif 0
        constexpr const char* asset_path = "../../asset/teapot/teapot.obj";
        constexpr const char* mtrl_in_asset = "m1";
#else
        constexpr const char* asset_path = "../../asset/sphere/sphere.obj";
        constexpr const char* mtrl_in_asset = "m1";
#endif

        auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(
            mtrl_in_asset,
            mtrl_param,
            nullptr, nullptr, nullptr);

        auto obj = aten::ObjLoader::LoadFirstObj(asset_path, ctxt_);
        auto poly_obj = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt_, obj, aten::mat4::Identity);
        scene->add(poly_obj);

        // For toon ramp.
        auto toon_ramp_tex = aten::ImageLoader::load("../../asset/toon/toon.png", ctxt_);

        // TODO
        //albedo_map_ = aten::ImageLoader::load("../../asset/sponza/01_STUB.JPG");
        //normal_map_ = aten::ImageLoader::load("../../asset/sponza/01_STUB-nml.png");

        obj->getShapes()[0]->GetMaterial()->setTextures(albedo_map_, normal_map_, nullptr);
#else
        constexpr const char* asset_path = "../../asset/cornellbox/bunny_in_box.obj";

        auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(
            "material_0",
            mtrl_param,
            nullptr, nullptr, nullptr);

        auto objs = aten::ObjLoader::Load(asset_path, ctxt_,
            [&](std::string_view name, aten::context& ctxt,
                aten::MaterialType type, const aten::vec3& mtrl_clr,
                const std::string& albedo, const std::string& nml) -> auto {
            (void)albedo;
            (void)nml;

            aten::MaterialParameter param;

            if (name == "light") {
                param.type = aten::MaterialType::Emissive;
                param.baseColor = aten::vec3(1.0F);
            }
            else {
                param.type = aten::MaterialType::Diffuse;
                param.baseColor = mtrl_clr;
            }

            auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(param, nullptr, nullptr, nullptr);
            mtrl->setName(name.data());
            return mtrl;
        }, true, true);

        auto it = std::find_if(objs.begin(), objs.end(), [](const decltype(objs)::value_type& o) { return o->getName() == "light"; });
        if (it != objs.end()) {
            auto light = aten::TransformableFactory::createInstance<aten::PolygonObject>(
                ctxt_,
                *it,
                aten::vec3(0.0f),
                aten::vec3(0.0f),
                aten::vec3(1.0f));
            scene->add(light);

            auto emit = ctxt_.FindMaterialByName("light");
            auto areaLight = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 200.0f);
            ctxt_.AddLight(areaLight);
        }

        for (const auto& obj : objs) {
            if (obj->getName() != "light") {
                auto tranformable = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt_, obj, aten::mat4::Identity);
                scene->add(tranformable);
            }
        }
#endif
    }

    std::shared_ptr<aten::material> CreateMaterial(aten::MaterialParameter& mtrl_param)
    {
        ctxt_.DeleteAllMaterialsAndClearList();

        auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(
            "",
            mtrl_param,
            enable_albedo_map_ ? albedo_map_ : nullptr,
            enable_normal_map_ ? normal_map_ : nullptr,
            nullptr);

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

    struct SceneLight {
        bool is_envmap{ false };

        std::shared_ptr<aten::texture> envmap_texture;
        std::shared_ptr<aten::ImageBasedLight> ibl;
        std::shared_ptr<aten::PointLight> point_light;
    } scene_light_;

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    aten::AcceleratedScene<aten::GPUBvh> scene_;
    aten::context ctxt_;

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

    aten::window::MesageHandlers handlers;
    handlers.OnRun = [&app]() { return app->Run(); };
    handlers.OnClose = [&app]() { app->OnClose(); };
    handlers.OnMouseBtn = [&app](bool left, bool press, int32_t x, int32_t y) { app->OnMouseBtn(left, press, x, y); };
    handlers.OnMouseMove = [&app](int32_t x, int32_t y) { app->OnMouseMove(x, y);  };
    handlers.OnMouseWheel = [&app](int32_t delta) { app->OnMouseWheel(delta); };
    handlers.OnKey = [&app](bool press, aten::Key key) { app->OnKey(press, key); };

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        handlers);

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
