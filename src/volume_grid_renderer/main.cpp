#include <filesystem>
#include <optional>
#include <vector>

#include "aten.h"
#include "idaten.h"
#include "atenscene.h"
#include "../common/scenedefs.h"
#include "../common/app_base.h"

#include "volume/medium.h"
#include "volume/grid.h"

#pragma optimize( "", off)

#define ENABLE_IBL
#define DEVICE_RENDERING

#ifdef DEVICE_RENDERING
#include "volume/grid_loader_device.h"
using NanoVDBBuffer = nanovdb::CudaDeviceBuffer;
#else
#include "volume/grid_loader.h"
using NanoVDBBuffer = nanovdb::HostBuffer;
#endif

constexpr int32_t WIDTH = 512;
constexpr int32_t HEIGHT = 512;
constexpr const char* TITLE = "volume_grid_renderer";

const aten::vec4 BGColor(1.0F);

class VolumeGridRendererApp : public App {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    VolumeGridRendererApp(int32_t width, int32_t height) : App(width, height) {}
    ~VolumeGridRendererApp() = default;

    VolumeGridRendererApp() = delete;
    VolumeGridRendererApp(const VolumeGridRendererApp&) = delete;
    VolumeGridRendererApp(VolumeGridRendererApp&&) = delete;
    VolumeGridRendererApp operator=(const VolumeGridRendererApp&) = delete;
    VolumeGridRendererApp operator=(VolumeGridRendererApp&&) = delete;

    bool Init()
    {
        visualizer_ = aten::visualizer::init(WIDTH, HEIGHT);

        gamma_.init(
            WIDTH, HEIGHT,
            "../shader/fullscreen_vs.glsl",
            "../shader/gamma_fs.glsl");

        visualizer_->addPostProc(&gamma_);

        aabb_rasterizer_.init(
            WIDTH, HEIGHT,
            "../shader/simple3d_vs.glsl",
            "../shader/simple3d_fs.glsl");

        aten::vec3 lookfrom;
        aten::vec3 lookat;
        float fov;

        Scene::getCameraPosAndAt(lookfrom, lookat, fov);

        camera_.Initalize(
            lookfrom,
            lookat,
            aten::vec3(0, 1, 0),
            fov,
            float(0.1), float(10000.0),
            WIDTH, HEIGHT);

        // NOTE:
        // grid_holder_ is used to add the loaded nanovdb grid in LoadNanoVDB.
        // So, it has to be instantiated before calling LoadNanoVDB.
        grid_holder_ = std::make_shared<aten::Grid>(aten::Grid());
        ctxt_.RegisterGridHolder(grid_holder_);

        LoadNanoVDB("../../asset/vdb/smoke.nvdb");

        // TODO
        aten::MaterialParameter param;
        param.type = aten::MaterialType::MaterialTypeMax;
        param.baseColor = aten::vec3(1, 0, 0);
        auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(
            "medium", param, nullptr, nullptr, nullptr);
        const auto mtrl_id = ctxt_.GetMaterialNum() - 1;

        mtrl->param().is_medium = true;
        mtrl->param().medium.grid_idx = 0;
        mtrl->param().medium.sigma_a = 0.0F;
        mtrl->param().medium.sigma_s = 0.9F;
        mtrl->param().medium.phase_function_g = 0.4F;
        mtrl->param().medium.majorant = AT_NAME::HeterogeneousMedium::EvalMajorant(
            grid_handle_->grid<float>(),
            mtrl->param().medium.sigma_a,
            mtrl->param().medium.sigma_s);

        auto grid_obj = aten::GenerateTrianglesFromGridBoundingBox(ctxt_, mtrl_id, grid_handle_->grid<float>());
        auto obj = aten::TransformableFactory::createInstance(
            ctxt_, grid_obj,
            aten::vec3(0.0F), aten::vec3(0.0F), aten::vec3(1.0F));
        scene_.add(obj);

        camera_.FitBoundingBox(grid_obj->getBoundingbox());

        scene_.build(ctxt_);

#ifdef ENABLE_IBL
        envmap_ = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_);
        auto bg = AT_NAME::Background::CreateBackgroundResource(envmap_);

        auto ibl = std::make_shared<aten::ImageBasedLight>(bg, ctxt_);
        scene_.addImageBasedLight(ctxt_, ibl);
#else
        auto bg = AT_NAME::Background::CreateBackgroundResource(nullptr, BGColor);
#endif

#ifdef DEVICE_RENDERING
        {
            auto aabb = scene_.getAccel()->getBoundingbox();
            auto d = aabb.getDiagonalLenght();
            renderer_.setHitDistanceLimit(d * 1.0F);

            const auto& nodes = scene_.getAccel()->getNodes();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            renderer_.UpdateSceneData(
                visualizer_->GetGLTextureHandle(),
                WIDTH, HEIGHT,
                camparam, ctxt_, nodes,
                0, 0, bg,
                [](const aten::context& ctxt) -> auto { return ctxt.GetGrid(); });
        }
#else
        renderer_.EnableRenderGrid(true);
        renderer_.SetBG(bg);
#endif

        return true;
    }

    void update()
    {
        if (is_camera_dirty_)
        {
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
    }

    bool Run()
    {
        update();

        aten::Destination dst;
        {
            dst.width = WIDTH;
            dst.height = HEIGHT;
            dst.maxDepth = 5;
            dst.russianRouletteDepth = 3;
            dst.sample = 1;
            dst.buffer = &buffer_;
        }

        aten::timer timer;
        timer.begin();

#ifdef DEVICE_RENDERING
        renderer_.render(
            WIDTH, HEIGHT,
            dst.sample,
            dst.maxDepth);
#else
        const auto frame_cnt = renderer_.GetFrameCount();

        // Trace rays.
        renderer_.render(ctxt_, dst, &scene_, &camera_);

        const auto elapsed = timer.end();
        avg_elapsed_ = avg_elapsed_ * frame_cnt + elapsed;
        avg_elapsed_ /= (frame_cnt + 1);

        AT_PRINTF("Elapsed %f[ms] / Avg %f[ms]\n", elapsed, avg_elapsed_);
#endif

        aten::vec4 clear_color(0, 0.5f, 1.0f, 1.0f);
        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            clear_color,
            1.0f,
            0);

#ifdef DEVICE_RENDERING
        visualizer_->render(false);
#else
        visualizer_->renderPixelData(buffer_.image().data(), camera_.NeedRevert());
#endif

#ifdef ENABLE_EVERY_FRAME_SC
        {
            const auto frame = renderer_.GetFrameCount();
            const auto file_name = aten::StringFormat("sc_%d.png", frame);
            visualizer_->takeScreenshot(file_name);
        }
#endif

        return true;
    }

    aten::context& GetContext()
    {
        return ctxt_;
    }

private:
    bool LoadNanoVDB(std::string_view nvdb)
    {
        using GridBufferType = decltype(grid_handle_)::value_type::BufferType;

        std::vector<nanovdb::FloatGrid*> grids;

#ifdef DEVICE_RENDERING
        grid_handle_ = idaten::LoadGrid(nvdb, grids, renderer_.GetCudaStream());
#else
        grid_handle_ = aten::LoadGrid(nvdb, grids);
#endif

        if (grid_handle_) {
            for (auto* grid : grids) {
                grid_holder_->AddGrid(grid);
            }
        }

        return static_cast<bool>(grid_handle_);
    }

private:
    aten::AcceleratedScene<aten::sbvh> scene_;
    aten::context ctxt_;

    std::shared_ptr<aten::texture> envmap_;

#ifdef DEVICE_RENDERING
    idaten::VolumeRendering renderer_;
#else
    aten::VolumePathTracing renderer_;
#endif

    std::optional<nanovdb::GridHandle<NanoVDBBuffer>> grid_handle_;
    std::shared_ptr<aten::Grid> grid_holder_;

    std::shared_ptr<aten::visualizer> visualizer_;

    aten::FilmProgressive buffer_{ WIDTH, HEIGHT };
    //aten::Film buffer_{ WIDTH, HEIGHT };

    aten::GammaCorrection gamma_;

    aten::RasterizeRenderer aabb_rasterizer_;

    float avg_elapsed_{ 0.0f };
};



int32_t main(int32_t argc, char* argv[])
{
    aten::initSampler(WIDTH, HEIGHT);

    aten::timer::init();
    aten::OMPUtil::setThreadNum(VolumeGridRendererApp::ThreadNum);

    auto app = std::make_shared<VolumeGridRendererApp>(WIDTH, HEIGHT);

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT,
        TITLE,
        [&app]() { return app->Run(); },
        [&app]() {},
        [&app](bool left, bool press, int32_t x, int32_t y) { app->OnMouseBtn(left, press, x, y); },
        [&app](int32_t x, int32_t y) { app->OnMouseMove(x, y); },
        [&app](int32_t delta) { app->OnMouseWheel(delta); },
        [&app](bool press, aten::Key key) { app->OnKey(press, key); });

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
