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

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr const char *TITLE = "AORenderer";

class AORendererApp {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    AORendererApp() = default;
    ~AORendererApp() = default;

    AORendererApp(const AORendererApp&) = delete;
    AORendererApp(AORendererApp&&) = delete;
    AORendererApp operator=(const AORendererApp&) = delete;
    AORendererApp operator=(AORendererApp&&) = delete;

    bool Init()
    {
        visualizer_ = aten::visualizer::init(WIDTH, HEIGHT);

        blitter_.init(
            WIDTH, HEIGHT,
            "../shader/fullscreen_vs.glsl",
            "../shader/fullscreen_fs.glsl");

        visualizer_->addPostProc(&blitter_);

        aten::vec3 pos, at;
        float vfov;
        Scene::getCameraPosAndAt(pos, at, vfov);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        Scene::makeScene(ctxt_, &scene_);
        scene_.build(ctxt_);

        renderer_.getCompaction().init(
            WIDTH * HEIGHT,
            1024);

        {
            const auto& nodes = scene_.getAccel()->getNodes();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            auto bg = AT_NAME::Background::CreateBackgroundResource(nullptr);

            renderer_.UpdateSceneData(
                visualizer_->GetGLTextureHandle(),
                WIDTH, HEIGHT,
                camparam, ctxt_, nodes,
                0, 0, bg);
        }

        return true;
    }

    bool Run()
    {
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

        if (render_mode_ == 0)
        {
            // AO
            renderer_.render(
                WIDTH, HEIGHT,
                1, 5);
        }
        else
        {
            // Texture Viewer
            renderer_.viewTextures(view_tex_idx_, WIDTH, HEIGHT);
        }

        auto cudaelapsed = timer.end();

        aten::vec4 clear_color(0, 0.5f, 1.0f, 1.0f);
        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            clear_color,
            1.0f,
            0);

        visualizer_->render(false);

        if (will_take_screenshot_)
        {
            auto screen_shot_file_name = aten::StringFormat("sc_%d.png", cnt_screenshot_);

            visualizer_->takeScreenshot(screen_shot_file_name);

            will_take_screenshot_ = false;
            cnt_screenshot_++;

            AT_PRINTF("Take Screenshot[%s]\n", screen_shot_file_name.c_str());
        }

        if (will_show_gui_)
        {
            ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("cuda : %.3f ms", cudaelapsed);
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
        static const float offset = float(0.1);

        if (press)
        {
            if (key == aten::Key::Key_F1)
            {
                will_show_gui_ = !will_show_gui_;
                return;
            }
            else if (key == aten::Key::Key_F2)
            {
                will_take_screenshot_ = true;
                return;
            }
        }

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

protected:
    aten::PinholeCamera camera_;

    aten::AcceleratedScene<aten::GPUBvh> scene_;
    aten::context ctxt_;

    idaten::AORenderer renderer_;

    std::shared_ptr<aten::visualizer> visualizer_;

    //aten::FilmProgressive buffer_{ WIDTH, HEIGHT };
    aten::Film buffer_{ WIDTH, HEIGHT };

    aten::FBO fbo_;

    // NOTE
    // No need any color correction for AO rendering.
    aten::Blitter blitter_;

    aten::RasterizeRenderer aabb_rasterizer_;
    aten::RasterizeRenderer rasterizer_;

    bool is_camera_dirty_{ false };

    bool will_show_gui_{ true };
    bool will_take_screenshot_{ false };
    int32_t cnt_screenshot_{ 0 };

    // 0: AO, 1: TexView
    int32_t render_mode_{ 0 };
    int32_t view_tex_idx_{ 0 };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };

};

int32_t main()
{
    aten::timer::init();
    aten::OMPUtil::setThreadNum(AORendererApp::ThreadNum);

    aten::initSampler(WIDTH, HEIGHT);

    auto app = std::make_shared<AORendererApp>();

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

    if (!app->Init()) {
        AT_ASSERT(false);
        return 1;
    }

    wnd->Run();

    app.reset();

    wnd->Terminate();
}
