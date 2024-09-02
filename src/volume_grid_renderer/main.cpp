#include <filesystem>
#include <vector>

#ifndef NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED
#pragma warning(push)
#pragma warning(disable:4146)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>
#pragma warning(pop)
#endif

#include "aten.h"
#include "idaten.h"
#include "atenscene.h"
#include "../common/scenedefs.h"

#include "volume/medium.h"
#include "volume/grid.h"

#define ENABLE_IBL
#define DEVICE_RENDERING

#ifdef DEVICE_RENDERING
using NanoVDBBuffer = nanovdb::CudaDeviceBuffer;
#else
using NanoVDBBuffer = nanovdb::HostBuffer;
#endif

constexpr int32_t WIDTH = 512;
constexpr int32_t HEIGHT = 512;
constexpr const char* TITLE = "volume_grid_renderer";

const aten::vec4 BGColor(1.0F);

class VolumeGridRendererApp {
public:
    static constexpr int32_t ThreadNum
#ifdef ENABLE_OMP
    { 8 };
#else
    { 1 };
#endif

    VolumeGridRendererApp() = default;
    ~VolumeGridRendererApp() = default;

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
            grid_handle_.grid<float>(),
            mtrl->param().medium.sigma_a,
            mtrl->param().medium.sigma_s);

        auto grid_obj = aten::GenerateTrianglesFromGridBoundingBox(ctxt_, mtrl_id, grid_handle_.grid<float>());
        auto obj = aten::TransformableFactory::createInstance(
            ctxt_, grid_obj,
            aten::vec3(0.0f), aten::vec3(0.0f), aten::vec3(1.0f));
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
    }

    bool Run()
    {
        update();

        camera_.update();

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
        std::filesystem::path p = nvdb;

        if (!std::filesystem::exists(p)) {
            AT_ASSERT(false);
            AT_PRINTF("%s doesn't exist.", nvdb.data());
            return false;
        }

        try {
            auto list = nanovdb::io::readGridMetaData(nvdb.data());
            if (list.size() != 1) {
                // TODO
                // Support only one grid.
                AT_PRINTF("Support only one grid\n");
                return false;
            }

            if (list[0].gridName != "density") {
                AT_PRINTF("Not denstity grid. Allow only density grid\n");
                return false;
            }

            grid_handle_ = nanovdb::io::readGrid<decltype(grid_handle_)::BufferType>(nvdb.data());
#ifdef DEVICE_RENDERING
            grid_handle_.deviceUpload(renderer_.GetCudaStream(), true);
            auto grid = grid_handle_.deviceGrid<float>();
#else
            auto grid = grid_handle_.grid<float>();
#endif
            grid_holder_->AddGrid(grid);

            return true;
        }
        catch (const std::exception& e) {
            AT_PRINTF("An exception occurred: %s\n", e.what());
            return false;
        }
    }

private:
    aten::PinholeCamera camera_;

    aten::AcceleratedScene<aten::sbvh> scene_;
    aten::context ctxt_;

    std::shared_ptr<aten::texture> envmap_;

#ifdef DEVICE_RENDERING
    idaten::VolumeRendering renderer_;
#else
    aten::VolumePathTracing renderer_;
#endif

    nanovdb::GridHandle<NanoVDBBuffer> grid_handle_;
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

    auto app = std::make_shared<VolumeGridRendererApp>();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT,
        TITLE,
        std::bind(&VolumeGridRendererApp::Run, app));

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
