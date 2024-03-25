#include <vector>
#include "aten.h"
#include "atenscene.h"
#include "../common/scenedefs.h"

static int32_t WIDTH = 512;
static int32_t HEIGHT = 512;
static const char* TITLE = "app";

#define ENABLE_IBL
// #define ENABLE_EVERY_FRAME_SC
//#define ENABLE_FEATURE_LINE

class HostRendererApp {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    HostRendererApp() = default;
    ~HostRendererApp() = default;

    HostRendererApp(const HostRendererApp&) = delete;
    HostRendererApp(HostRendererApp&&) = delete;
    HostRendererApp operator=(const HostRendererApp&) = delete;
    HostRendererApp operator=(HostRendererApp&&) = delete;

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
        aabb_rasterizer_.init(
            WIDTH, HEIGHT,
            "../shader/simple3d_vs.glsl",
            "../shader/simple3d_fs.glsl");

        if constexpr (std::is_member_function_pointer_v<decltype(&decltype(renderer_)::SetMotionDepthBuffer)>) {
            fbo_.asMulti(2);
            fbo_.init(
                WIDTH, HEIGHT,
                aten::PixelFormat::rgba32f,
                true);
        }

        aten::vec3 lookfrom;
        aten::vec3 lookat;
        real fov;

        Scene::getCameraPosAndAt(lookfrom, lookat, fov);

        camera_.Initalize(
            lookfrom,
            lookat,
            aten::vec3(0, 1, 0),
            fov,
            real(0.1), real(10000.0),
            WIDTH, HEIGHT);

        aten::AssetManager asset_manager;
        Scene::makeScene(ctxt_, &scene_, asset_manager);

        scene_.build(ctxt_);

#ifdef ENABLE_IBL
        envmap_ = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_, asset_manager);

        bg_ = std::make_shared<aten::envmap>();
        bg_->init(envmap_);

        auto ibl = std::make_shared<aten::ImageBasedLight>(bg_);
        scene_.addImageBasedLight(ctxt_, ibl);
#endif

#ifdef ENABLE_FEATURE_LINE
        renderer_.enableFeatureLine(true);
#endif

        return true;
    }

    void update()
    {
        static float y = 0.0f;
        static float d = -0.1f;

        auto obj = getMovableObj();

        if (obj)
        {
            auto t = obj->getTrans();

            if (y >= 0.0f)
            {
                d = -0.1f;
            }
            else if (y <= -0.5f)
            {
                d = 0.1f;
            }

            y += d;
            t.y += d;

            obj->setTrans(t);
            obj->update();

            auto accel = scene_.getAccel();
            accel->update(ctxt_);
        }
    }

    bool Run()
    {
        // update();

        camera_.update();

        aten::Destination dst;
        {
            dst.width = WIDTH;
            dst.height = HEIGHT;
            dst.maxDepth = 3;
            dst.russianRouletteDepth = 3;
            dst.sample = 1;
            dst.buffer = &buffer_;
        }

        if constexpr (std::is_member_function_pointer_v<decltype(&decltype(renderer_)::SetMotionDepthBuffer)>) {
            rasterizer_.drawSceneForGBuffer(
                renderer_.GetFrameCount(),
                ctxt_,
                &scene_,
                &camera_,
                fbo_);

            renderer_.SetMotionDepthBuffer(fbo_, 1);
        }

        const auto frame_cnt = renderer_.GetFrameCount();

        aten::timer timer;
        timer.begin();

        // Trace rays.
        renderer_.render(ctxt_, dst, &scene_, &camera_);

        const auto elapsed = timer.end();
        avg_elapsed_ = avg_elapsed_ * frame_cnt + elapsed;
        avg_elapsed_ /= (frame_cnt + 1);

        AT_PRINTF("Elapsed %f[ms] / Avg %f[ms]\n", elapsed, avg_elapsed_);

        // TODO
        if (need_exporte_as_hdr_)
        {
            need_exporte_as_hdr_ = false;

            // Export to hdr format.
            aten::HDRExporter::save(
                "result.hdr",
                buffer_.image().data(),
                WIDTH, HEIGHT);
        }

        aten::vec4 clear_color(0, 0.5f, 1.0f, 1.0f);
        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            clear_color,
            1.0f,
            0);

        visualizer_->renderPixelData(buffer_.image().data(), camera_.needRevert());

#if 0
        g_rasterizerAABB.drawAABB(
            &g_camera,
            g_scene.getAccel());
#endif

#ifdef ENABLE_EVERY_FRAME_SC
        {
            static char tmp[1024];
            sprintf(tmp, "sc_%d.png\0", g_frameNo);

            g_visualizer->takeScreenshot(tmp);
        }
#endif

        return true;
    }

    aten::context& GetContext()
    {
        return ctxt_;
    }

protected:
    aten::PinholeCamera camera_;

    aten::AcceleratedScene<aten::sbvh> scene_;
    aten::context ctxt_;

    aten::AssetManager asset_manager;

    std::shared_ptr<aten::envmap> bg_;
    std::shared_ptr<aten::texture> envmap_;

    aten::PathTracing renderer_;
    //aten::SVGFRenderer renderer_;
    //aten::ReSTIRRenderer renderer_;

    std::shared_ptr<aten::visualizer> visualizer_;

    //aten::FilmProgressive buffer_{ WIDTH, HEIGHT };
    aten::Film buffer_{ WIDTH, HEIGHT };

    aten::FBO fbo_;

    aten::GammaCorrection gamma_;

    aten::RasterizeRenderer aabb_rasterizer_;
    aten::RasterizeRenderer rasterizer_;

    bool need_exporte_as_hdr_{ false };

    float avg_elapsed_{ 0.0f };
};



int32_t main(int32_t argc, char* argv[])
{
    aten::initSampler(WIDTH, HEIGHT);

    aten::timer::init();
    aten::OMPUtil::setThreadNum(HostRendererApp::ThreadNum);

    auto app = std::make_shared<HostRendererApp>();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT,
        TITLE,
        std::bind(&HostRendererApp::Run, app));

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
