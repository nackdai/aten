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

class DeformScene {
public:
    static std::shared_ptr<aten::instance<aten::deformable>> makeScene(
        aten::context& ctxt, aten::scene* scene)
    {
        // NOTE:
        // The part of the normals in unitychan model seems to be broken.
        // That's the part of the hair.
        // It's not sure it happens from the original model or it happens while converting the model data.
        // If we re-compute it by computing cross the edges of the triangle, it's fixed.
        // But, it causes the non smooth normal.

        auto mdl = aten::TransformableFactory::createDeformable(ctxt);
        mdl->read("../../asset/converted_unitychan/unitychan_gpu.mdl");

        aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
        //aten::MaterialLoader::load("../../asset/converted_unitychan/unitychan_mtrl.xml", ctxt);
        aten::MaterialLoader::load("unitychan_toon_test_mtrl.xml", ctxt);

        for (auto& tex : ctxt.GetTextures()) {
            tex->SetFilterMode(aten::TextureFilterMode::Linear);
            tex->SetAddressMode(aten::TextureAddressMode::Wrap);
        }

        auto deformMdl = aten::TransformableFactory::createInstance<aten::deformable>(
            ctxt, mdl,
            aten::vec3(0), aten::vec3(0), aten::vec3(0.01F));
        scene->add(deformMdl);

        auto* mtrl_hair = ctxt.GetMaterialByName("hair");
        mtrl_hair->stencil_type = aten::StencilType::STENCIL;

        auto* mtrl_eyeline = ctxt.GetMaterialByName("eyeline");
        mtrl_eyeline->stencil_type = aten::StencilType::ALWAYS;
        mtrl_eyeline->feature_line.enable = false;

        auto* mtrl_cheek = ctxt.GetMaterialByName("mat_cheek");
        mtrl_cheek->feature_line.enable = false;

        auto* mtrl_eyebase = ctxt.GetMaterialByName("eyebase");
        mtrl_eyebase->feature_line.enable = false;

        auto* mtrl_eye_left = ctxt.GetMaterialByName("eye_L1");
        mtrl_eye_left->feature_line.enable = false;

        auto* mtrl_eye_right = ctxt.GetMaterialByName("eye_R1");
        mtrl_eye_right->feature_line.enable = false;

        aten::ImageLoader::load("FO_CLOTH1.tga", ctxt);

        auto* mtrl_face = ctxt.GetMaterialByName("face");
        AT_ASSERT(mtrl_face->type == aten::MaterialType::Toon);
        mtrl_face->toon.toon_type = aten::MaterialType::Diffuse;
        mtrl_face->toon.remap_texture = ctxt.GetTextureNum() - 1;
        mtrl_face->toon.target_light_idx = 0;
        mtrl_face->feature_line.metric_flag = aten::FeatureLineMetricFlag::Albedo | aten::FeatureLineMetricFlag::Normal | aten::FeatureLineMetricFlag::Depth;

        aten::ImageLoader::setBasePath("./");

        return deformMdl;
    }

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        float& fov)
    {
        pos = aten::vec3(0.f, 1.3f, 0.5f);
        at = aten::vec3(0.f, 1.3f, 0.f);
        fov = 45.0f;
    }
};

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
        DeformScene::getCameraPosAndAt(pos, at, vfov);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        aten::accelerator::setUserDefsInternalAccelCreator([] {
            return std::make_shared<aten::GPUBvh>();
        });
        deform_mdl_ = DeformScene::makeScene(ctxt_, &scene_);
        scene_.build(ctxt_);

        // IBL
        scene_light_.is_ibl = false;
        scene_light_.envmap_texture = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_);
        auto bg = AT_NAME::Background::CreateBackgroundResource(scene_light_.envmap_texture, aten::vec4(0));
        scene_light_.ibl = std::make_shared<aten::ImageBasedLight>(bg, ctxt_);

        // PointLight
        scene_light_.point_light = std::make_shared<aten::PointLight>(
            aten::vec3(0.0, 0.0, 50.0),
            aten::vec3(1.0, 1.0, 1.0),
            4000.0f);

        if (scene_light_.is_ibl) {
            scene_.addImageBasedLight(ctxt_, scene_light_.ibl);
        }
        else {
            ctxt_.AddLight(scene_light_.point_light);
        }

        auto mdl = deform_mdl_->GetHasObjectAsRealType();

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
                0, 0,
                bg);
#else
            ctxt_.CopyBvhNodes(scene_.getAccel()->getNodes());
            renderer_.SetBG(bg);
#endif

            renderer_.SetEnableEnvmap(scene_light_.is_ibl);
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

        return true;
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

#ifdef DEVICE_RENDERING
        if (is_view_texture_) {
            renderer_.viewTextures(view_texture_idx_, WIDTH, HEIGHT);
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

#ifdef DEVICE_RENDERING
        RenderGUI(frame, cudaelapsed, rasterizerTime, visualizerTime);
#endif

        return true;
    }

    void RenderGUI(
        uint32_t frame,
        float cudaelapsed,
        double rasterizerTime,
        double visualizerTime
    )
    {
        if (will_show_gui_)
        {
            bool need_renderer_reset = false;

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
                renderer_.SetEnableEnvmap(scene_light_.is_ibl);
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

            const auto& mtrl_param = ctxt_.GetMaterialByName("face");
            auto mtrl = ctxt_.GetMaterialInstance(mtrl_param->id);
            if (mtrl->edit(&mtrl_param_editor_)) {
                need_renderer_reset = true;
                renderer_.updateMaterial(ctxt_.GetMetarialParemeters());
            }

            ImGui::Spacing();

            if (aten::npr::FeatureLine::EditFeatureLineConfig(
                &mtrl_param_editor_, ctxt_.scene_rendering_config.feature_line))
            {
                renderer_.UpdateSceneRenderingConfig(ctxt_);
            }

            ImGui::Spacing();

            ImGui::Begin("Gradient texture");
            gradient_tex_editor_.Display();
            ImGui::End();

            ImGui::Spacing();

            auto enable_progressive = renderer_.IsEnableProgressive();
            if (ImGui::Checkbox("Progressive", &enable_progressive))
            {
                renderer_.SetEnableProgressive(enable_progressive);
            }

            auto is_view_texture = is_view_texture_;
            if (ImGui::Checkbox("View texture", &is_view_texture)) {
                is_view_texture_ = is_view_texture;
            }
            if (is_view_texture_) {
                const auto tex_num = ctxt_.GetTextureNum();
                ImGui::SliderInt("View texture", &view_texture_idx_, 0, tex_num - 1);
            }

            auto cam = camera_.param();
            ImGui::Text("Pos %f/%f/%f", cam.origin.x, cam.origin.y, cam.origin.z);
            ImGui::Text("At  %f/%f/%f", cam.center.x, cam.center.y, cam.center.z);

            if (need_renderer_reset) {
                renderer_.reset();
            }
        }
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
                DeformScene::getCameraPosAndAt(pos, at, vfov);

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

    std::shared_ptr<aten::instance<aten::deformable>> deform_mdl_;

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

    bool is_view_texture_{ false };
    int32_t view_texture_idx_{ 0 };
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
