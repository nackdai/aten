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


constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr char* TITLE = "NanoVDBViewer";

class NanoVDBViewerApp {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    NanoVDBViewerApp() = default;
    ~NanoVDBViewerApp() = default;

    NanoVDBViewerApp(const NanoVDBViewerApp&) = delete;
    NanoVDBViewerApp(NanoVDBViewerApp&&) = delete;
    NanoVDBViewerApp operator=(const NanoVDBViewerApp&) = delete;
    NanoVDBViewerApp operator=(NanoVDBViewerApp&&) = delete;

    bool Init()
    {
        visualizer_ = aten::visualizer::init(WIDTH, HEIGHT);

        blitter_.init(
            WIDTH, HEIGHT,
            "../shader/fullscreen_vs.glsl",
            "../shader/fullscreen_fs.glsl");

        auto bbox = renderer_.LoadNanoVDB("../../asset/vdb/smoke.nvdb");
        if (!bbox) {
            return false;
        }

        visualizer_->addPostProc(&blitter_);

        aten::vec3 pos(0, 0, 1);
        aten::vec3 at(0, 0, 0);
        float vfov = 60.0F;
        camera_.Initalize(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            float(0.1), float(1000.0),
            WIDTH, HEIGHT);

        camera_.FitBoundingBox(bbox.value());

        return true;
    }

    bool Run()
    {
        float updateTime = 0.0f;

        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            visualizer_->clear();
        }

        renderer_.RenderNanoVDB(
            visualizer_->GetGLTextureHandle(),
            WIDTH, HEIGHT,
            camera_.param());

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
            // TODO
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
                float(0.001));
            is_camera_dirty_ = true;
        }

        prev_mouse_pos_x_ = x;
        prev_mouse_pos_y_ = y;
    }

    void OnMouseWheel(int32_t delta)
    {
        aten::CameraOperator::dolly(camera_, delta * float(0.1));
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
                break;
            default:
                break;
            }

            is_camera_dirty_ = true;
        }
    }

private:
    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    idaten::VolumeRendering renderer_;

    std::shared_ptr<aten::visualizer> visualizer_;

    aten::Blitter blitter_;

    bool will_show_gui_{ true };
    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};

int32_t main()
{
    aten::timer::init();
    aten::OMPUtil::setThreadNum(NanoVDBViewerApp::ThreadNum);

    aten::initSampler(WIDTH, HEIGHT);

    auto app = std::make_shared<NanoVDBViewerApp>();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        std::bind(&NanoVDBViewerApp::Run, app),
        std::bind(&NanoVDBViewerApp::OnClose, app),
        std::bind(&NanoVDBViewerApp::OnMouseBtn, app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
        std::bind(&NanoVDBViewerApp::OnMouseMove, app, std::placeholders::_1, std::placeholders::_2),
        std::bind(&NanoVDBViewerApp::OnMouseWheel, app, std::placeholders::_1),
        std::bind(&NanoVDBViewerApp::OnKey, app, std::placeholders::_1, std::placeholders::_2));

    app->Init();

    aten::GLProfiler::start();

    wnd->Run();

    aten::GLProfiler::terminate();

    app.reset();

    wnd->Terminate();
}
