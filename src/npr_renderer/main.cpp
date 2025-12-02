#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <iterator>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

#include "gradient_textue_editor.h"

#include "../common/app_misc.h"
#include "../common/scenedefs.h"

#include "unity_chan_scene.h"
#include "model_loader.h"

//#define NprScene    UnityChanScene
#define NprScene    DummyScene

//#pragma optimize( "", off)

#define ENABLE_ENVMAP
#define TAKE_SC_EVERY_FRAME false
#define DEVICE_RENDERING

#ifdef DEVICE_RENDERING
constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
#else
constexpr int32_t WIDTH = 512;
constexpr int32_t HEIGHT = 512;
#endif

constexpr const char* TITLE = "NPR renderer";

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

    bool edit(std::string_view name, const char* const* elements, size_t size, int32_t& param) override final
    {
        const auto ret = ImGui::Combo(name.data(), &param, elements, size);
        return ret;
    }

    bool CollapsingHeader(std::string_view name) override final
    {
        return ImGui::CollapsingHeader(name.data());
    }
};

class DeformationRendererApp {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    DeformationRendererApp() = default;
    ~DeformationRendererApp() = default;

    DeformationRendererApp(const DeformationRendererApp&) = delete;
    DeformationRendererApp(DeformationRendererApp&&) = delete;
    DeformationRendererApp operator=(const DeformationRendererApp&) = delete;
    DeformationRendererApp operator=(DeformationRendererApp&&) = delete;

    bool Init()
    {
        ctxt_.scene_rendering_config.enable_alpha_blending = true;
        ctxt_.scene_rendering_config.feature_line.enabled = true;
        ctxt_.scene_rendering_config.bvh_hit_min = 1e-4;

        visualizer_ = aten::visualizer::init(WIDTH, HEIGHT);

        gamma_.init(
            WIDTH, HEIGHT,
            "../shader/fullscreen_vs.glsl",
            "../shader/gamma_fs.glsl");

        visualizer_->addPostProc(&gamma_);

        rasterizer_aabb_.init(
            WIDTH, HEIGHT,
            "../shader/simple3d_vs.glsl",
            "../shader/simple3d_fs.glsl");
        aten::vec3 pos, at;
        float vfov;
        NprScene::getCameraPosAndAt(pos, at, vfov);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        aten::accelerator::setUserDefsInternalAccelCreator([] {
            return std::make_shared<aten::GPUBvh>();
        });
        std::ignore = NprScene::makeScene(ctxt_, &scene_);

        // IBL
        scene_light_.is_ibl = false;
        scene_light_.envmap_texture = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_);
        ctxt_.scene_rendering_config.bg = AT_NAME::Background::CreateBackgroundResource(scene_light_.envmap_texture, aten::vec4(0));
        scene_light_.ibl = std::make_shared<aten::ImageBasedLight>(ctxt_.scene_rendering_config.bg, ctxt_);

        // PointLight
        scene_light_.point_light = std::make_shared<aten::PointLight>(
            aten::vec3(0.0, 0.0, 50.0),
            aten::vec3(1.0, 1.0, 1.0),
            4000.0f);

        ctxt_.scene_rendering_config.bg.enable_env_map = scene_light_.is_ibl;

        if (scene_light_.is_ibl) {
            scene_.addImageBasedLight(ctxt_, scene_light_.ibl);
        }
        else {
            ctxt_.AddLight(scene_light_.point_light);
        }

        if constexpr (std::is_same_v<NprScene, UnityChanScene>) {
            BuildScene();
        }

        return true;
    }

    void BuildScene()
    {
        scene_.build(ctxt_);

        {
            const auto& nodes = scene_.getAccel()->getNodes();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

#ifdef DEVICE_RENDERING
            renderer_.UpdateSceneData(
                visualizer_->GetGLTextureHandle(),
                WIDTH, HEIGHT,
                camparam, ctxt_, nodes,
                0, 0);
#else
            ctxt_.CopyBvhNodes(scene_.getAccel()->getNodes());
#endif
        }

        {
            /*deform_mdl_->setScale(aten::vec3(0.01F));

            deform_mdl_->update(ctxt_, true);*/

            auto accel = scene_.getAccel();

            accel->update(ctxt_);

#ifdef DEVICE_RENDERING
            const auto& nodes = scene_.getAccel()->getNodes();

            // Set the created bvh data.
            renderer_.updateBVH(ctxt_, nodes);
#endif
        }

        have_built_scene_ = true;
    }

    bool Run()
    {
        auto frame = renderer_.GetFrameCount();

        update(frame);

        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

#ifdef DEVICE_RENDERING
            renderer_.updateCamera(camparam);
#endif
            is_camera_dirty_ = false;

            visualizer_->clear();
        }

        aten::GLProfiler::begin();

        auto rasterizerTime = aten::GLProfiler::end();

        aten::timer timer;
        timer.begin();

        if (have_built_scene_) {
#ifdef DEVICE_RENDERING
            if (mode_ == RenderingMode::Texture) {
            renderer_.viewTextures(view_texture_idx_, WIDTH, HEIGHT);
        }
            else if (mode_ == RenderingMode::AOV) {
                renderer_.ViewAOV(aov_, WIDTH, HEIGHT);
            }
        else {
            renderer_.render(
                WIDTH, HEIGHT,
                max_samples_,
                max_bounce_);
        }
#else
        aten::Destination dst;
        {
            dst.width = WIDTH;
            dst.height = HEIGHT;
            dst.maxDepth = 6;
            dst.russianRouletteDepth = 3;
            dst.sample = 1;
            dst.buffer = &buffer_;
        }
        renderer_.render(ctxt_, dst, &scene_, &camera_);
#endif
        }

        auto cudaelapsed = timer.end();

        avg_cuda_time_ = avg_cuda_time_ * (frame - 1) + cudaelapsed;
        avg_cuda_time_ /= (float)frame;

        aten::GLProfiler::begin();

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Stencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

#ifdef DEVICE_RENDERING
        visualizer_->render(false);
#else
        visualizer_->renderPixelData(buffer_.image().data(), camera_.NeedRevert());
#endif

        auto visualizerTime = aten::GLProfiler::end();

        if (is_show_aabb_) {
            rasterizer_aabb_.drawAABB(
                &camera_,
                scene_.getAccel());
        }

        if (will_take_screen_shot_)
        {
            auto screen_shot_file_name = aten::StringFormat("sc_%d.png\0", screen_shot_count_);

            visualizer_->takeScreenshot(screen_shot_file_name);

            will_take_screen_shot_ = TAKE_SC_EVERY_FRAME;
            screen_shot_count_++;

            AT_PRINTF("Take Screenshot[%s]\n", screen_shot_file_name.c_str());
        }

        RenderGUI(frame, cudaelapsed, rasterizerTime, visualizerTime);

        return true;
    }

    void RenderGUI(
        uint32_t frame,
        float cudaelapsed,
        double rasterizerTime,
        double visualizerTime
    )
    {
#ifdef DEVICE_RENDERING
        if (will_show_gui_)
        {
            bool need_renderer_reset = false;

            constexpr std::array RenderingModeStr = {
                "PT", "Texture", "AOV"
            };

            ImGui::Combo(
                "RenderingMode",
                reinterpret_cast<int32_t*>(&mode_),
                RenderingModeStr.data(), RenderingModeStr.size());
            if (mode_ == RenderingMode::Texture) {
                const auto tex_num = ctxt_.GetTextureNum();
                ImGui::SliderInt("View texture", &view_texture_idx_, 0, tex_num - 1);
            }
            else if (mode_ == RenderingMode::AOV) {
                constexpr std::array AovStr = {
                    "Albedo", "Normal", "WireFrame", "BaryCentric"
                };
                ImGui::Combo(
                    "AOV",
                    reinterpret_cast<int32_t*>(&aov_),
                    AovStr.data(), AovStr.size());
            }

            ImGui::Text("[%d] %.3f ms/frame (%.1f FPS)", frame, 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("cuda : %.3f ms (avg : %.3f ms)", cudaelapsed, avg_cuda_time_);
            ImGui::Text("%.3f Mrays/sec", (WIDTH * HEIGHT * max_samples_) / float(1000 * 1000) * (float(1000) / cudaelapsed));

            if (aten::GLProfiler::isEnabled()) {
                ImGui::Text("GL : [rasterizer %.3f ms] [visualizer %.3f ms]", rasterizerTime, visualizerTime);
            }

            if (ImGui::SliderInt("Samples", &max_samples_, 1, 100)
                || ImGui::SliderInt("Bounce", &max_bounce_, 1, 10))
            {
                renderer_.reset();
            }

            ImGui::Spacing();

            constexpr std::array light_types = { "IBL", "PointLight" };
            int32_t lighttype = scene_light_.is_ibl ? 0 : 1;
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

                scene_light_.is_ibl = next_is_envmap;
                ctxt_.scene_rendering_config.bg.enable_env_map = scene_light_.is_ibl;
                renderer_.UpdateSceneRenderingConfig(ctxt_);
                renderer_.updateLight(ctxt_);
            }

            if (!scene_light_.is_ibl) {
                auto& point_light = ctxt_.GetLightInstance(0);
                auto& light_param = point_light->param();

                bool is_updated = false;
                is_updated |= ImGui::ColorEdit3("LightColor", reinterpret_cast<float*>(&light_param.light_color));
                is_updated |= ImGui::SliderFloat("LightIntensity", &light_param.intensity, 0.0F, 10000.0F);
                if (is_updated) {
                    renderer_.updateLight(ctxt_);
                }
            }

            ImGui::Spacing();

            auto enable_progressive = renderer_.IsEnableProgressive();
            if (ImGui::Checkbox("Progressive", &enable_progressive))
            {
                renderer_.SetEnableProgressive(enable_progressive);
            }

            auto cam = camera_.param();
            ImGui::Text("Pos %f/%f/%f", cam.origin.x, cam.origin.y, cam.origin.z);
            ImGui::Text("At  %f/%f/%f", cam.center.x, cam.center.y, cam.center.z);

            ImGui::Begin("Material");

            const auto mtrl_num = ctxt_.GetMaterialNum();
            if (mtrl_num > 0) {
                edit_mtrl_idx_ = edit_mtrl_idx_ < 0 ? 0 : edit_mtrl_idx_;
                ImGui::SliderInt("index", &edit_mtrl_idx_, 0, mtrl_num - 1);
            }

            if (edit_mtrl_idx_ >= 0) {
                auto& mtrl_param = ctxt_.GetMaterialInstance(edit_mtrl_idx_)->param();
                auto mtrl = ctxt_.GetMaterialInstance(mtrl_param.id);

                ImGui::Text("%s", mtrl->name());

            bool new_update_mtrl = false;

                if (mtrl_param.type == aten::MaterialType::Toon
                    || mtrl_param.type == aten::MaterialType::StylizedBrdf
            )
            {
                constexpr std::array mtrl_types = {
                    "Toon",
                    "StylizedBrdf",
                };

                    const auto mtrl_idx = mtrl_param.id;

                    int32_t mtrl_type = mtrl_param.type == aten::MaterialType::Toon ? 0 : 1;
                ImGui::Text("%s", mtrl_types[mtrl_type]);

                if (ImGui::Combo("mode", &mtrl_type, mtrl_types.data(), static_cast<int32_t>(mtrl_types.size()))) {
                    auto new_mtrl_type = mtrl_type == 0 ? aten::MaterialType::Toon : aten::MaterialType::StylizedBrdf;
                        if (new_mtrl_type != mtrl_param.type) {
                            auto albedo_map = ctxt_.GetTexture(mtrl_param.albedoMap);
                            auto normal_map = ctxt_.GetTexture(mtrl_param.normalMap);
                            aten::MaterialParameter new_mtrl_param = mtrl_param;
                        new_mtrl_param.type = new_mtrl_type;

                        auto new_mtrl = aten::material::CreateMaterialWithMaterialParameter(
                            new_mtrl_param,
                            albedo_map ? albedo_map.get() : nullptr,
                            normal_map ? normal_map.get() : nullptr,
                            nullptr);
                        new_mtrl->setName(mtrl->name());

                        ctxt_.ReplaceMaterialInstance(mtrl_idx, new_mtrl);
                        mtrl = new_mtrl;
                        need_renderer_reset = true;
                        new_update_mtrl = true;
                    }
                }

                if (mtrl->edit(&mtrl_param_editor_)) {
                    need_renderer_reset = true;
                    new_update_mtrl = true;
                }
            }
            else {
                if (mtrl->edit(&mtrl_param_editor_)) {
                    need_renderer_reset = true;
                    new_update_mtrl = true;
                }
            }

            if (new_update_mtrl) {
                renderer_.updateMaterial(ctxt_.GetMetarialParemeters());
                }
            }

            ImGui::End();

            ImGui::Begin("Feature Line");
            {
                if (aten::npr::FeatureLine::EditFeatureLineConfig(
                    &mtrl_param_editor_, ctxt_.scene_rendering_config.feature_line))
                {
                    renderer_.UpdateSceneRenderingConfig(ctxt_);
                }
            }
            ImGui::End();


            if (edit_mtrl_idx_ >= 0) {
                auto& mtrl_param = ctxt_.GetMaterialInstance(edit_mtrl_idx_)->param();
                if (mtrl_param.type == aten::MaterialType::Toon
                    || mtrl_param.type == aten::MaterialType::StylizedBrdf)
                {
            ImGui::Begin("Gradient texture");

                gradient_tex_editor_.Display();
                if (ImGui::Button("Update")) {

                        auto tex = ctxt_.GetTexture(mtrl_param.toon.remap_texture);
                    gradient_tex_editor_.Read(
                        tex->colors().data(),
                        tex->width(), tex->height());
                        renderer_.UpdateTexture(mtrl_param.toon.remap_texture, ctxt_);
                    }

                    ImGui::End();
                }
            }

            if (need_renderer_reset) {
                renderer_.reset();
            }
        }
#endif
    }

    void Load(std::string_view path)
    {
        std::vector<std::shared_ptr<aten::PolygonObject>> objs;
        ModelLoader::Load(objs, ctxt_, path);

        for (auto& obj : objs) {
            auto obj_instance = aten::TransformableFactory::createInstance<aten::PolygonObject>(
                ctxt_, obj, aten::mat4::Identity);
            scene_.add(obj_instance);
        }

        BuildScene();

        camera_.FitBoundingBox(scene_.GetBoundingBox());
        is_camera_dirty_ = true;
    }

    void OnClose()
    {

    }

    void OnMouseBtn(bool left, bool press, int32_t x, int32_t y)
    {
        is_mouse_l_btn_down_ = false;
        is_mouse_r_btn_down_ = false;

        if (press) {
            prev_mouse_pos_x_ = x;
            prev_mouse_pos_y_ = y;

            is_mouse_l_btn_down_ = left;
            is_mouse_r_btn_down_ = !left;
        }
    }

    void OnMouseMove(int32_t x, int32_t y)
    {
        if (is_mouse_l_btn_down_) {
            aten::CameraOperator::Rotate(
                camera_,
                WIDTH, HEIGHT,
                prev_mouse_pos_x_, prev_mouse_pos_y_,
                x, y);
            is_camera_dirty_ = true;
        }
        else if (is_mouse_r_btn_down_) {
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
        const float offset = float(0.5);

        if (press) {
            if (key == aten::Key::Key_F1) {
                will_show_gui_ = !will_show_gui_;
                return;
            }
            else if (key == aten::Key::Key_F2) {
                will_take_screen_shot_ = true;
                return;
            }
            else if (key == aten::Key::Key_F5) {
                aten::GLProfiler::trigger();
                return;
            }
        }

        if (press) {
            switch (key) {
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
                NprScene::getCameraPosAndAt(pos, at, vfov);

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
    void update(int32_t frame)
    {
    }

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    aten::AcceleratedScene<aten::GPUBvh> scene_;
    aten::context ctxt_;

    bool have_built_scene_{ false };

#ifdef DEVICE_RENDERING
    idaten::NPRPathTracing renderer_;
#else
    aten::PathTracing renderer_;
    aten::FilmProgressive buffer_{ WIDTH, HEIGHT };
#endif

    std::shared_ptr<aten::visualizer> visualizer_;

    float avg_cuda_time_{ 0.0f };

    aten::GammaCorrection gamma_;
    aten::TAA taa_;

    aten::RasterizeRenderer rasterizer_aabb_;

    struct SceneLight {
        bool is_ibl{ false };

        std::shared_ptr<aten::texture> envmap_texture;
        std::shared_ptr<aten::ImageBasedLight> ibl;
        std::shared_ptr<aten::PointLight> point_light;
    } scene_light_;

    MaterialParamEditor mtrl_param_editor_;
    int32_t edit_mtrl_idx_{ -1 };

    GradientTextureEditor gradient_tex_editor_;

    bool will_show_gui_
#ifdef DEVICE_RENDERING
    { true };
#else
    { false };
#endif

    bool will_take_screen_shot_{ TAKE_SC_EVERY_FRAME };
    int32_t screen_shot_count_{ 0 };

    int32_t max_samples_{ 1 };
    int32_t max_bounce_{ 6 };
    bool is_show_aabb_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };

    enum RenderingMode {
        PT,
        Texture,
        AOV,
    };

    RenderingMode mode_{ RenderingMode::PT };
    int32_t view_texture_idx_{ 0 };
    idaten::Renderer::AOV aov_{ static_cast<idaten::Renderer::AOV>(0) };
};

int32_t main()
{
    aten::timer::init();
    aten::OMPUtil::setThreadNum(DeformationRendererApp::ThreadNum);

    aten::initSampler(WIDTH, HEIGHT);

    auto app = std::make_shared<DeformationRendererApp>();

    auto wnd = std::make_shared<aten::window>();

    aten::window::MesageHandlers handlers;
    handlers.OnRun = [&app]() { return app->Run(); };
    handlers.OnClose = [&app]() { app->OnClose(); };
    handlers.OnMouseBtn = [&app](bool left, bool press, int32_t x, int32_t y) { app->OnMouseBtn(left, press, x, y); };
    handlers.OnMouseMove = [&app](int32_t x, int32_t y) { app->OnMouseMove(x, y);  };
    handlers.OnMouseWheel = [&app](int32_t delta) { app->OnMouseWheel(delta); };
    handlers.OnKey = [&app](bool press, aten::Key key) { app->OnKey(press, key); };
    handlers.OnDropFile = [&app](std::string_view path) { app->Load(path); };

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
