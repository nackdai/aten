#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <iterator>

#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

#include "../common/app_misc.h"
#include "../common/scenedefs.h"

//#pragma optimize( "", off)

#define ENABLE_ENVMAP
#define ENABLE_SVGF

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr const char* TITLE = "deform_mdl_";

class DeformScene {
public:
    static std::shared_ptr<aten::instance<aten::deformable>> makeScene(
        aten::context& ctxt, aten::scene* scene)
    {
        auto mdl = aten::TransformableFactory::createDeformable(ctxt);
        mdl->read("../../asset/converted_unitychan/unitychan_gpu.mdl");

        aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
        aten::MaterialLoader::load("../../asset/converted_unitychan/unitychan_mtrl.xml", ctxt);

        auto deformMdl = aten::TransformableFactory::createInstance<aten::deformable>(
            ctxt, mdl,
            aten::vec3(0), aten::vec3(0), aten::vec3(0.01F));
        scene->add(deformMdl);

        aten::ImageLoader::setBasePath("./");

        return deformMdl;
    }


    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        float& fov)
    {
        pos = aten::vec3(0.f, 1.f, 1.5f);
        at = aten::vec3(0.f, 1.f, 0.f);
        fov = 45.0f;
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
        visualizer_ = aten::visualizer::init(WIDTH, HEIGHT);

        gamma_.init(
            WIDTH, HEIGHT,
            "../shader/fullscreen_vs.glsl",
            "../shader/gamma_fs.glsl");

        visualizer_->addPostProc(&gamma_);

        shader_raasterize_deformable_.init(
            WIDTH, HEIGHT,
            "./ssrt_deformable_vs.glsl",
            "../shader/ssrt_gs.glsl",
            "../shader/ssrt_fs.glsl");

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

#ifdef ENABLE_ENVMAP
        auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_);
        auto bg = AT_NAME::Background::CreateBackgroundResource(envmap);

        auto ibl = std::make_shared<aten::ImageBasedLight>(bg, ctxt_);

        scene_.addImageBasedLight(ctxt_, ibl);
#else
        auto bg = AT_NAME::Background::CreateBackgroundResource(nullptr, aten::vec4(0));
#endif

        auto mdl = deform_mdl_->getHasObjectAsRealType();

        mdl->initGLResources(&shader_raasterize_deformable_);

        {
            const auto& nodes = scene_.getAccel()->getNodes();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            renderer_.UpdateSceneData(
                visualizer_->GetGLTextureHandle(),
                WIDTH, HEIGHT,
                camparam, ctxt_, nodes,
                0, 0,
                bg);
        }

        {
            /*deform_mdl_->setScale(aten::vec3(0.01F));

            deform_mdl_->update(ctxt_, true);*/

            auto accel = scene_.getAccel();

            accel->update(ctxt_);

            const auto& nodes = scene_.getAccel()->getNodes();

            // Set the created bvh data.
            renderer_.updateBVH(ctxt_, nodes);
        }

        return true;
    }

    bool Run()
    {
        auto frame = renderer_.frame();

        update(frame);

        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            renderer_.updateCamera(camparam);
            is_camera_dirty_ = false;

            visualizer_->clear();
        }

        aten::GLProfiler::begin();

        auto rasterizerTime = aten::GLProfiler::end();

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

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Stencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        visualizer_->render(false);

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

            will_take_screen_shot_ = false;
            screen_shot_count_++;

            AT_PRINTF("Take Screenshot[%s]\n", screen_shot_file_name.c_str());
        }

        if (will_show_gui_)
        {
            ImGui::Text("[%d] %.3f ms/frame (%.1f FPS)", renderer_.frame(), 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
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

            auto cam = camera_.param();
            ImGui::Text("Pos %f/%f/%f", cam.origin.x, cam.origin.y, cam.origin.z);
            ImGui::Text("At  %f/%f/%f", cam.center.x, cam.center.y, cam.center.z);
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

    idaten::PathTracing renderer_;

    aten::FilmProgressive buffer_{ WIDTH, HEIGHT };

    std::shared_ptr<aten::visualizer> visualizer_;

    float avg_cuda_time_{ 0.0f };

    aten::GammaCorrection gamma_;
    aten::TAA taa_;

    aten::RasterizeRenderer rasterizer_aabb_;

    aten::shader shader_raasterize_deformable_;

    bool will_show_gui_{ true };
    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    int32_t max_samples_{ 1 };
    int32_t max_bounce_{ 5 };
    bool is_show_aabb_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
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
