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

// NOTE:
// Not work with environment map.

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr const char *TITLE = "ReSTIR";

class ReSTIRRendererApp {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    ReSTIRRendererApp() = default;
    ~ReSTIRRendererApp() = default;

    ReSTIRRendererApp(const ReSTIRRendererApp&) = delete;
    ReSTIRRendererApp(ReSTIRRendererApp&&) = delete;
    ReSTIRRendererApp operator=(const ReSTIRRendererApp&) = delete;
    ReSTIRRendererApp operator=(ReSTIRRendererApp&&) = delete;

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

        fbo_.asMulti(2);
        fbo_.init(
            WIDTH, HEIGHT,
            aten::PixelFormat::rgba32f,
            true);

        aten::vec3 pos, at;
        real vfov;
        Scene::getCameraPosAndAt(pos, at, vfov);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        aten::AssetManager asset_manager;
        Scene::makeScene(ctxt_, &scene_, asset_manager);
        scene_.build(ctxt_);

        {
            auto aabb = scene_.getAccel()->getBoundingbox();
            auto d = aabb.getDiagonalLenght();
            renderer_.setHitDistanceLimit(d * 0.25f);

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

                for (int32_t i = 0; i < texNum; i++)
                {
                    auto t = ctxt_.GtTexture(i);
                    tex.push_back(
                        idaten::TextureResource(t->colors(), t->width(), t->height()));
                }
            }

            auto camparam = camera_.param();
            camparam.znear = real(0.1);
            camparam.zfar = real(10000.0);

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
                idaten::EnvmapResource());

            renderer_.SetGBuffer(
                fbo_.GetGLTextureHandle(0),
                fbo_.GetGLTextureHandle(1));
        }

        renderer_.SetMode(curr_rendering_mode_);
        renderer_.SetAOVMode(curr_aov_mode_);

        return true;
    }

    bool Run()
    {
        if (enable_frame_step_ && !frame_step_)
        {
            return true;
        }

        auto frame = renderer_.frame();

        frame_step_ = false;

        float updateTime = 0.0f;

        {
            aten::timer timer;
            timer.begin();

            updateTime = timer.end();

            avg_update_time_ = avg_update_time_ * (frame - 1) + updateTime;
            avg_update_time_ /= (float)frame;
        }

        if (is_camera_dirty_)
        {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = real(0.1);
            camparam.zfar = real(10000.0);

            renderer_.updateCamera(camparam);
            is_camera_dirty_ = false;

            visualizer_->clear();
        }

        aten::GLProfiler::begin();

        rasterizer_.drawSceneForGBuffer(
            renderer_.frame(),
            ctxt_,
            &scene_,
            &camera_,
            fbo_);

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

        aten::vec4 clear_color(0, 0.5f, 1.0f, 1.0f);
        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            clear_color,
            1.0f,
            0);

        visualizer_->render(false);

        auto visualizerTime = aten::GLProfiler::end();

        if (is_show_aabb_)
        {
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

            AT_PRINTF("Take Screenshot[%s]\n", screen_shot_file_name);
        }

        if (will_show_gui_)
        {
            ImGui::Text("[%d] %.3f ms/frame (%.1f FPS)", renderer_.frame(), 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("cuda : %.3f ms (avg : %.3f ms)", cudaelapsed, avg_cuda_time_);
            ImGui::Text("update : %.3f ms (avg : %.3f ms)", updateTime, avg_update_time_);
            ImGui::Text("%.3f Mrays/sec", (WIDTH * HEIGHT * max_samples_) / real(1000 * 1000) * (real(1000) / cudaelapsed));

            if (aten::GLProfiler::isEnabled())
            {
                ImGui::Text("GL : [rasterizer %.3f ms] [visualizer %.3f ms]", rasterizerTime, visualizerTime);
            }

            auto is_input_samples = ImGui::SliderInt("Samples", &max_samples_, 1, 100);
            auto is_input_bounce = ImGui::SliderInt("Bounce", &max_bounce_, 1, 10);

            if (is_input_samples || is_input_bounce)
            {
                renderer_.reset();
            }

            static const char* items[] = { "ReSTIR", "PT", "AOV" };

            if (ImGui::Combo("mode", (int32_t*)&curr_rendering_mode_, items, AT_COUNTOF(items)))
            {
                renderer_.SetMode(curr_rendering_mode_);
            }

            if (curr_rendering_mode_ == idaten::ReSTIRPathTracing::Mode::ReSTIR)
            {
                static const char* restir_items[] = { "ReSTIR", "SpatialReuse" };

                if (ImGui::Combo("restir mode", (int32_t*)&curr_restir_mode_, restir_items, AT_COUNTOF(restir_items)))
                {
                    renderer_.SetReSTIRMode(curr_restir_mode_);
                }
            }

            if (curr_rendering_mode_ == idaten::ReSTIRPathTracing::Mode::AOVar)
            {
                static const char* aovitems[] = { "Normal", "TexColor", "Depth", "Wire", "Barycentric", "Motion", "ObjId" };

                if (ImGui::Combo("aov", (int32_t*)&curr_aov_mode_, aovitems, AT_COUNTOF(aovitems)))
                {
                    renderer_.SetAOVMode(curr_aov_mode_);
                }
            }

            if (ImGui::Checkbox("Progressive", &enable_progressive_))
            {
                renderer_.SetEnableProgressive(enable_progressive_);
            }

            ImGui::Checkbox("Show AABB", &is_show_aabb_);

            ImGui::SliderFloat("MoveMultiply", &move_multiply_scale_, 1.0f, 100.0f);

            auto cam = camera_.param();
            ImGui::Text("Pos %f/%f/%f", cam.origin.x, cam.origin.y, cam.origin.z);
            ImGui::Text("At  %f/%f/%f", cam.center.x, cam.center.y, cam.center.z);
        }

        idaten::ReSTIRPathTracing::PickedInfo info;
        auto isPicked = renderer_.GetPickedPixelInfo(info);
        if (isPicked)
        {
            AT_PRINTF("[%d, %d]\n", info.ix, info.iy);
            AT_PRINTF("  nml[%f, %f, %f]\n", info.normal.x, info.normal.y, info.normal.z);
            AT_PRINTF("  mesh[%d] mtrl[%d], tri[%d]\n", info.meshid, info.mtrlid, info.triid);
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
            else if (key == aten::Key::Key_F3)
            {
                enable_frame_step_ = !enable_frame_step_;
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
    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    aten::AcceleratedScene<aten::GPUBvh> scene_;
    aten::context ctxt_;

    std::shared_ptr<aten::instance<aten::deformable>> deform_mdl_;
    std::shared_ptr<aten::DeformAnimation> defrom_anm_;

    idaten::ReSTIRPathTracing renderer_;

    std::shared_ptr<aten::visualizer> visualizer_;

    aten::GammaCorrection gamma_;

    aten::FBO fbo_;

    float avg_cuda_time_{ 0.0f };
    float avg_update_time_{ 0.0f };

    aten::RasterizeRenderer rasterizer_;
    aten::RasterizeRenderer rasterizer_aabb_;

    bool will_show_gui_{ true };
    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    int32_t max_samples_{ 1 };
    int32_t max_bounce_{ 5 };
    bool enable_progressive_{ false };
    bool is_show_aabb_{ false };

    idaten::ReSTIRPathTracing::Mode curr_rendering_mode_{ idaten::ReSTIRPathTracing::Mode::ReSTIR };
    idaten::ReSTIRPathTracing::ReSTIRMode curr_restir_mode_{ idaten::ReSTIRPathTracing::ReSTIRMode::ReSTIR };
    idaten::ReSTIRPathTracing::AOVMode curr_aov_mode_{ idaten::ReSTIRPathTracing::AOVMode::WireFrame };

    float move_multiply_scale_{ 1.0f };

    bool enable_frame_step_{ false };
    bool frame_step_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};

int32_t main()
{
    aten::timer::init();
    aten::OMPUtil::setThreadNum(ReSTIRRendererApp::ThreadNum);

    aten::initSampler(WIDTH, HEIGHT);

    auto app = std::make_shared< ReSTIRRendererApp>();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        std::bind(&ReSTIRRendererApp::Run, app),
        std::bind(&ReSTIRRendererApp::OnClose, app),
        std::bind(&ReSTIRRendererApp::OnMouseBtn, app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
        std::bind(&ReSTIRRendererApp::OnMouseMove, app, std::placeholders::_1, std::placeholders::_2),
        std::bind(&ReSTIRRendererApp::OnMouseWheel, app, std::placeholders::_1),
        std::bind(&ReSTIRRendererApp::OnKey, app, std::placeholders::_1, std::placeholders::_2));

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
