#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <iterator>

#include <imgui.h>

#include "aten.h"
#include "atenscene.h"

#include "atmosphere/sky/sky_model.h"
#include "atmosphere/sky/sky_model_device.h"

#define DEVICE_RENDERING

#ifdef DEVICE_RENDERING
constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
#else
constexpr int32_t WIDTH = 256;
constexpr int32_t HEIGHT = 256;
#endif

#define TAKE_SC_EVERY_FRAME (false)

constexpr const char* TITLE = "Atmosphere renderer";

class AtmosphereRendererApp {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    AtmosphereRendererApp() = default;
    ~AtmosphereRendererApp() = default;

    AtmosphereRendererApp(const AtmosphereRendererApp&) = delete;
    AtmosphereRendererApp(AtmosphereRendererApp&&) = delete;
    AtmosphereRendererApp operator=(const AtmosphereRendererApp&) = delete;
    AtmosphereRendererApp operator=(AtmosphereRendererApp&&) = delete;

    bool Init()
    {
        visualizer_ = aten::visualizer::init(WIDTH, HEIGHT);

        gamma_.init(
            WIDTH, HEIGHT,
            "../shader/fullscreen_vs.glsl",
            "../shader/gamma_fs.glsl");

        visualizer_->addPostProc(&gamma_);

        InitCamera();

        return true;
    }

    void InitCamera()
    {
        constexpr aten::Length view_distance_meters = 9000.0_m;
        constexpr float view_zenith_angle_radians = 1.47F;
        constexpr float view_azimuth_angle_radians = -0.1F;

        const float cos_z = aten::cos(view_zenith_angle_radians);
        const float sin_z = aten::sin(view_zenith_angle_radians);
        const float cos_a = aten::cos(view_azimuth_angle_radians);
        const float sin_a = aten::sin(view_azimuth_angle_radians);

        // Y-upにおける各基底ベクトル
        // ux (Right): 水平方向のベクトル
        const std::array ux = { -sin_a, 0.0F, -cos_a };
        // uy (Up): 垂直方向のベクトル (元のuzの要素をY-up用に並び替え)
        const std::array uy = { -cos_z * cos_a, sin_z, -cos_z * sin_a };
        // uz (Forward/View): 視線方向のベクトル
        const std::array uz = { sin_z * cos_a, cos_z, sin_z * sin_a };

        const float l = view_distance_meters.as(aten::MeterUnit::km);

        // 最終的な位置
        aten::vec3 pos{
            uz[0] * l, // X
            uz[1] * l, // Y (ここが高さになる)
            uz[2] * l, // Z
        };
        aten::vec3 at{
            pos + aten::vec3(1.0F)
        };
        const float vfov = 30.0F;

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);
    }

    bool Run()
    {
        update();

        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            is_camera_dirty_ = false;

            visualizer_->clear();
        }

#ifdef DEVICE_RENDERING
        if (!is_sky_initialized_) {
            sky_model_.Init();
            sky_model_.PreCompute();
            is_sky_initialized_ = true;
        }

        sky_model_.Render(
            visualizer_->GetGLTextureHandle(),
            WIDTH, HEIGHT,
            sun_zenith_angle_radians_,
            sun_azimuth_angle_radians_,
            camera_.param());

        aten::vec4 clear_color(0, 0.5f, 1.0f, 1.0f);
        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Stencil,
            clear_color,
            1.0f,
            0);

        visualizer_->render(false);
#else
        if (!is_sky_rendered_) {
            sky_model_.Init();
            sky_model_.PreCompute();

            sky_model_.Render(
                WIDTH, HEIGHT,
                camera_.param(),
                dst_);

            aten::vec4 clear_color(0, 0.5f, 1.0f, 1.0f);
            aten::RasterizeRenderer::clearBuffer(
                aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Stencil,
                clear_color,
                1.0f,
                0);
            is_sky_rendered_ = true;
        }

        visualizer_->renderPixelData(dst_.image().data(), camera_.NeedRevert());
#endif

        RenderGUI();

        return true;
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
                return;
            }
            else if (key == aten::Key::Key_F2) {
                will_take_screen_shot_ = true;
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
                InitCamera();
                break;
            default:
                break;
            }

            is_camera_dirty_ = true;
        }
    }

private:
    void update()
    {
    }

    void RenderGUI()
    {
#ifdef DEVICE_RENDERING
        ImGui::SliderFloat("Sun Zenith Angle (radians)", &sun_zenith_angle_radians_, 0.0F, AT_MATH_PI);
        ImGui::SliderFloat("Sun Azimuth Angle (radians)", &sun_azimuth_angle_radians_, -AT_MATH_PI, AT_MATH_PI);
#endif
    }

#ifdef DEVICE_RENDERING
    idaten::sky::SkyModel sky_model_;

    bool is_sky_initialized_{ false };
#else
    aten::sky::SkyModel sky_model_;

    // TODO
    bool is_sky_rendered_{ false };
#endif

    float sun_zenith_angle_radians_{ 1.3F };
    float sun_azimuth_angle_radians_{ 2.9F };

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    aten::Film dst_{ WIDTH, HEIGHT };

    std::shared_ptr<aten::visualizer> visualizer_;

    aten::GammaCorrection gamma_;

    bool will_take_screen_shot_{ TAKE_SC_EVERY_FRAME };
    int32_t screen_shot_count_{ 0 };

    int32_t max_samples_{ 1 };
    int32_t max_bounce_{ 6 };
    bool is_show_aabb_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};

int32_t main()
{
    aten::OMPUtil::setThreadNum(AtmosphereRendererApp::ThreadNum);

    aten::initSampler(WIDTH, HEIGHT);

    auto app = std::make_shared<AtmosphereRendererApp>();

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

    if (id < 0) {
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
