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
//#define ENABLE_NPR

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr const char* TITLE = "idaten";

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

class DeviceRendererApp {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    DeviceRendererApp() = default;
    ~DeviceRendererApp() = default;

    DeviceRendererApp(const DeviceRendererApp&) = delete;
    DeviceRendererApp(DeviceRendererApp&&) = delete;
    DeviceRendererApp operator=(const DeviceRendererApp&) = delete;
    DeviceRendererApp operator=(DeviceRendererApp&&) = delete;

    bool Init()
    {
        visualizer_ = aten::visualizer::init(WIDTH, HEIGHT);

        gamma_.init(
            WIDTH, HEIGHT,
            "../shader/fullscreen_vs.glsl",
            "../shader/gamma_fs.glsl");

        visualizer_->addPostProc(&gamma_);

        rasterizer_.init(
            WIDTH, HEIGHT,
            "../shader/ssrt_vs.glsl",
            "../shader/ssrt_gs.glsl",
            "../shader/ssrt_fs.glsl");
        rasterizer_aabb_.init(
            WIDTH, HEIGHT,
            "../shader/simple3d_vs.glsl",
            "../shader/simple3d_fs.glsl");

        aten::vec3 pos, at;
        float vfov;
        Scene::getCameraPosAndAt(pos, at, vfov);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        MakeScene<Scene>(movable_obj_, ctxt_, &scene_);

        scene_.build(ctxt_);

#ifdef ENABLE_ENVMAP
        auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_);
        ctxt_.scene_rendering_config.bg = AT_NAME::Background::CreateBackgroundResource(envmap);
        auto ibl = std::make_shared<aten::ImageBasedLight>(ctxt_.scene_rendering_config.bg, ctxt_);
        ctxt_.scene_rendering_config.bg.avgIllum = ibl->getAvgIlluminace();

        scene_.addImageBasedLight(ctxt_, ibl);
#else
        ctxt_.scene_rendering_config.bg = AT_NAME::Background::CreateBackgroundResource(nullptr, 0.0F);
#endif

        {
            auto aabb = scene_.getAccel()->getBoundingbox();
            auto d = aabb.getDiagonalLenght();
            renderer_.setHitDistanceLimit(d * distance_limit_ratio_);

            const auto& nodes = scene_.getAccel()->getNodes();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            renderer_.UpdateSceneData(
                visualizer_->GetGLTextureHandle(),
                WIDTH, HEIGHT,
                camparam, ctxt_, nodes,
                0, 0);
        }

        return true;
    }

    bool Run()
    {
        if (enable_frame_step_ && !frame_step_)
        {
            return true;
        }

        auto frame = renderer_.GetFrameCount();

        frame_step_ = false;

        float updateTime = 0.0f;

        {
            aten::timer timer;
            timer.begin();

            if (enable_anm_update_)
            {
                update();
            }

            updateTime = timer.end();

            avg_update_time_ = avg_update_time_ * (frame - 1) + updateTime;
            avg_update_time_ /= (float)frame;
        }

        if (is_camera_dirty_)
        {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            renderer_.updateCamera(camparam);
            is_camera_dirty_ = false;

            visualizer_->clear();
        }

        aten::timer timer;
        timer.begin();

        renderer_.render(
            WIDTH, HEIGHT,
            max_samples_,
            max_bounce_);

        auto cudaelapsed = timer.end();

        avg_cuda_time_ = avg_cuda_time_ * (frame - 1) + cudaelapsed;
        avg_cuda_time_ /= (float)frame;

        aten::GLProfiler::begin();

        aten::vec4 clear_color(0, 0.5f, 1.0f, 1.0f);
        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Stencil,
            clear_color,
            1.0f,
            0);

        visualizer_->render(false);

        auto visualizerTime = aten::GLProfiler::end();

        if (is_show_aabb_)
        {
            rasterizer_aabb_.renderSceneDepth(
                ctxt_,
                &scene_,
                &camera_);
            rasterizer_aabb_.drawAABB(
                &camera_,
                scene_.getAccel());
        }

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
            ImGui::Text("[%d] %.3f ms/frame (%.1f FPS)", renderer_.GetFrameCount(), 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("cuda : %.3f ms (avg : %.3f ms)", cudaelapsed, avg_cuda_time_);
            ImGui::Text("update : %.3f ms (avg : %.3f ms)", updateTime, avg_update_time_);
            ImGui::Text("%.3f Mrays/sec", (WIDTH * HEIGHT * max_samples_) / float(1000 * 1000) * (float(1000) / cudaelapsed));

            auto is_input_samples = ImGui::SliderInt("Samples", &max_samples_, 1, 100);
            auto is_input_bounce = ImGui::SliderInt("Bounce", &max_bounce_, 1, 10);

            if (is_input_samples || is_input_bounce)
            {
                renderer_.reset();
            }

            if (ImGui::Checkbox("Progressive", &enable_progressive_))
            {
                renderer_.SetEnableProgressive(enable_progressive_);
            }

            ImGui::Checkbox("Show AABB", &is_show_aabb_);

            if (ImGui::SliderFloat("Distance Limit Ratio", &distance_limit_ratio_, 0.1f, 1.0f))
            {
                auto aabb = scene_.getAccel()->getBoundingbox();
                auto d = aabb.getDiagonalLenght();
                renderer_.setHitDistanceLimit(d * distance_limit_ratio_);
            }

            ImGui::SliderFloat("MoveMultiply", &move_multiply_scale_, 1.0f, 100.0f);

            auto cam = camera_.param();
            ImGui::Text("Pos %f/%f/%f", cam.origin.x, cam.origin.y, cam.origin.z);
            ImGui::Text("At  %f/%f/%f", cam.center.x, cam.center.y, cam.center.z);

#ifdef ENABLE_NPR
            ImGui::Spacing();
            if (aten::npr::FeatureLine::EditFeatureLineConfig(
                &param_editor_, ctxt_.scene_rendering_config.feature_line))
            {
                renderer_.UpdateSceneRenderingConfig(ctxt_);
            }
#endif
        }

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
            else if (key == aten::Key::Key_F3)
            {
                enable_frame_step_ = !enable_frame_step_;
                return;
            }
            else if (key == aten::Key::Key_F4)
            {
                enable_anm_update_ = !enable_anm_update_;
                return;
            }
            else if (key == aten::Key::Key_F5)
            {
                aten::GLProfiler::trigger();
                return;
            }
            else if (key == aten::Key::Key_SPACE)
            {
                if (enable_frame_step_)
                {
                    frame_step_ = true;
                    return;
                }
            }
        }

        auto offset = offset_base * move_multiply_scale_;

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
                Scene::getCameraPosAndAt(pos, at, vfov);

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
    void update()
    {
        static float y = 0.0f;
        static float d = -0.1f;

        if (movable_obj_)
        {
            auto t = movable_obj_->getTrans();

            if (y >= -0.1f)
            {
                d = -0.01f;
            }
            else if (y <= -1.5f)
            {
                d = 0.01f;
            }

            y += d;
            t.y += d;

            movable_obj_->setTrans(t);
            movable_obj_->update(ctxt_);

            auto accel = scene_.getAccel();
            accel->update(ctxt_);

            const auto& nodes = scene_.getAccel()->getNodes();
            renderer_.updateBVH(ctxt_, nodes);
        }
    }

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    aten::AcceleratedScene<aten::GPUBvh> scene_;
    aten::context ctxt_;

#ifdef ENABLE_NPR
    idaten::NPRPathTracing renderer_;
#else
    idaten::PathTracing renderer_;
    //idaten::VolumeRendering renderer_;
#endif

    std::shared_ptr<aten::visualizer> visualizer_;

    aten::GammaCorrection gamma_;

    std::shared_ptr<aten::instance<aten::PolygonObject>> movable_obj_;

    float avg_cuda_time_{ 0.0f };
    float avg_update_time_{ 0.0f };

    aten::RasterizeRenderer rasterizer_;
    aten::RasterizeRenderer rasterizer_aabb_;

    MaterialParamEditor param_editor_;

    bool will_show_gui_{ true };
    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    bool enable_anm_update_{ false };

    int32_t max_samples_{ 1 };
    int32_t max_bounce_{ 5 };
    bool enable_progressive_{ false };
    bool is_show_aabb_{ false };

    float move_multiply_scale_{ 1.0f };

    float distance_limit_ratio_{ 1.0f };

    bool enable_frame_step_{ false };
    bool frame_step_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};

int32_t main()
{
    aten::OMPUtil::setThreadNum(DeviceRendererApp::ThreadNum);

    aten::initSampler(WIDTH, HEIGHT);

    auto app = std::make_shared<DeviceRendererApp>();

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
