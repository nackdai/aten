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

#include "../common/scenedefs.h"

//#pragma optimize( "", off)

#define ENABLE_ENVMAP
#define ENABLE_SVGF

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr const char* TITLE = "deform_mdl_";

class Lbvh : aten::accelerator {
public:
    Lbvh() : aten::accelerator(aten::AccelType::UserDefs) {}
    ~Lbvh() {}

public:
    static std::shared_ptr<accelerator> create()
    {
        auto ret = std::make_shared<Lbvh>();
        return std::reinterpret_pointer_cast<accelerator>(ret);
    }

    virtual void build(
        const aten::context& ctxt,
        aten::hitable** list,
        uint32_t num,
        aten::aabb* bbox = nullptr) override final
    {
        m_bvh.build(ctxt, list, num, bbox);

        setBoundingBox(m_bvh.getBoundingbox());
    }

    virtual bool hit(
        const aten::context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        aten::Intersection& isect) const override final
    {
        AT_ASSERT(false);
        return false;
    }

    virtual bool HitWithLod(
        const aten::context& ctxt,
        const aten::ray& r,
        real t_min, real t_max,
        bool enableLod,
        aten::Intersection& isect) const override final
    {
        AT_ASSERT(false);
        return false;
    }

    aten::bvh m_bvh;
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

#ifdef ENABLE_SVGF
        taa_.init(
            WIDTH, HEIGHT,
            "../shader/fullscreen_vs.glsl", "../shader/taa_fs.glsl",
            "../shader/fullscreen_vs.glsl", "../shader/taa_final_fs.glsl");

        visualizer_->addPostProc(&taa_);
#endif

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

        shader_raasterize_deformable_.init(
            WIDTH, HEIGHT,
            "./ssrt_deformable_vs.glsl",
            "../shader/ssrt_gs.glsl",
            "../shader/ssrt_fs.glsl");

#ifdef ENABLE_SVGF
        fbo_.asMulti(2);
        fbo_.init(
            WIDTH, HEIGHT,
            aten::PixelFormat::rgba32f,
            true);

        taa_.setMotionDepthBufferHandle(fbo_.GetGLTextureHandle(1));
#endif

        aten::vec3 pos, at;
        real vfov;
        Scene::getCameraPosAndAt(pos, at, vfov);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        aten::accelerator::setUserDefsInternalAccelCreator(Lbvh::create);

        aten::AssetManager asset_manager;
        aten::tie(deform_mdl_, defrom_anm_) = Scene::makeScene(ctxt_, &scene_, asset_manager);
        scene_.build(ctxt_);

#ifdef ENABLE_ENVMAP
        auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_, asset_manager);
        auto bg = AT_NAME::Background::CreateBackgroundResource(envmap);

        auto ibl = std::make_shared<aten::ImageBasedLight>(bg, ctxt_);

        scene_.addImageBasedLight(ctxt_, ibl);
#else
        auto bg = AT_NAME::Background::CreateBackgroundResource(nullptr, aten::vec4(0));
#endif

        size_t advanceVtxNum = 0;
        size_t advanceTriNum = 0;
        std::vector<aten::TriangleParameter> deformTris;

        // Initialize skinning.
        if (deform_mdl_)
        {
            auto mdl = deform_mdl_->getHasObjectAsRealType();

            mdl->initGLResources(&shader_raasterize_deformable_);

            auto& vb = mdl->getVBForGPUSkinning();

            std::vector<aten::SkinningVertex> vtx;
            std::vector<uint32_t> idx;

            int32_t vtxIdOffset = ctxt_.GetVertexNum();

            mdl->getGeometryData(ctxt_, vtx, idx, deformTris);

            skinning_.initWithTriangles(
                &vtx[0], vtx.size(),
                &deformTris[0], deformTris.size(),
                &vb);

            advanceVtxNum = vtx.size();
            advanceTriNum = deformTris.size();

            if (defrom_anm_) {
                timeline_.init(defrom_anm_->getDesc().time, real(0));
                timeline_.enableLoop(true);
                timeline_.start();
            }
        }

        {
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

            tri_offset_ = primparams.size();
            vtx_offset_ = vtxparams.size();

            const auto& nodes = scene_.getAccel()->getNodes();

            std::vector<idaten::TextureResource> tex;
            {
                auto texNum = ctxt_.GetTextureNum();

                for (int32_t i = 0; i < texNum; i++) {
                    auto t = ctxt_.GetTexture(i);
                    tex.push_back(
                        idaten::TextureResource(t->colors(), t->width(), t->height()));
                }
            }

#ifdef ENABLE_ENVMAP
            for (auto& l : lightparams) {
                if (l.type == aten::LightType::IBL) {
                    l.envmapidx = envmap->id();
                }
            }
#endif

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
                primparams, advanceTriNum,
                vtxparams, advanceVtxNum,
                mtxs,
                tex,
                bg);

#ifdef ENABLE_SVGF
            auto aabb = scene_.getAccel()->getBoundingbox();
            auto d = aabb.getDiagonalLenght();
            renderer_.setHitDistanceLimit(d * 0.25f);

            renderer_.SetGBuffer(
                fbo_.GetGLTextureHandle(0),
                fbo_.GetGLTextureHandle(1));
#endif
        }

        // For LBVH.
        if (deform_mdl_) {
            skinning_.setVtxOffset(vtx_offset_);
            lbvh_.init(advanceTriNum);
        }
        else
        {
            std::vector<std::vector<aten::TriangleParameter>> triangles;
            std::vector<int32_t> triIdOffsets;

            aten::DataCollector::collectTriangles(ctxt_, triangles, triIdOffsets);

            uint32_t maxTriNum = 0;
            for (const auto& tris : triangles) {
                maxTriNum = std::max<uint32_t>(maxTriNum, tris.size());
            }

            lbvh_.init(maxTriNum);

            const auto& sceneBbox = scene_.getAccel()->getBoundingbox();
            auto& nodes = renderer_.getCudaTextureResourceForBvhNodes();
            auto& vtxPos = renderer_.getCudaTextureResourceForVtxPos();

            // TODO
            // もし、GPUBvh が SBVH だとした場合.
            // ここで取得するノード配列は SBVH のノードである、ThreadedSbvhNode となる.
            // しかし、LBVHBuilder::build で渡すことができるのは、ThreadBVH のノードである ThreadedBvhNode である.
            // そのため、現状、ThreadedBvhNode に無理やりキャストしている.
            // もっとスマートな方法を考えたい.
            // If GPUBvh is SBVH, what we can retrieve the array of ThreadedSbvhNode as SBVH node type.
            // But, what we can pass to LBVHBuilder::build is ThreadedBvhNode of ThreadBVH's node.
            // So, we need to reinterpret cast ThreadedSbvhNode to ThreadedBvhNode.
            // It's not safe and we neeed to consider the safer way.

            auto& cpunodes = scene_.getAccel()->getNodes();

            for (int32_t i = 0; i < triangles.size(); i++)
            {
                auto& tris = triangles[i];
                auto triIdOffset = triIdOffsets[i];

                // NOTE
                // 0 is for top layer.
                lbvh_.build(
                    nodes[i + 1],
                    tris,
                    triIdOffset,
                    sceneBbox,
                    vtxPos,
                    0,
                    (std::vector<aten::ThreadedBvhNode>*) & cpunodes[i + 1]);
            }
        }

#ifdef ENABLE_SVGF
        renderer_.SetMode((idaten::SVGFPathTracing::Mode)curr_rendering_mode_);
        renderer_.SetAOVMode((aten::SVGFAovMode)curr_aov_mode_);
        //renderer_.SetCanSSRTHitTest(false);
#endif

        return true;
    }

    bool Run()
    {
        auto frame = renderer_.frame();

        update(frame);

        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = real(0.1);
            camparam.zfar = real(10000.0);

            renderer_.updateCamera(camparam);
            is_camera_dirty_ = false;

            visualizer_->clear();
        }

        aten::GLProfiler::begin();

#ifdef ENABLE_SVGF
        rasterizer_.drawSceneForGBuffer(
            renderer_.frame(),
            ctxt_,
            &scene_,
            &camera_,
            fbo_,
            &shader_raasterize_deformable_);
#endif

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
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
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
            ImGui::Text("%.3f Mrays/sec", (WIDTH * HEIGHT * max_samples_) / real(1000 * 1000) * (real(1000) / cudaelapsed));

            if (aten::GLProfiler::isEnabled()) {
                ImGui::Text("GL : [rasterizer %.3f ms] [visualizer %.3f ms]", rasterizerTime, visualizerTime);
            }

            if (ImGui::SliderInt("Samples", &max_samples_, 1, 100)
                || ImGui::SliderInt("Bounce", &max_bounce_, 1, 10))
            {
                renderer_.reset();
            }

#ifdef ENABLE_SVGF
            const char* items[] = { "SVGF", "TF", "PT", "VAR", "AOV" };

            if (ImGui::Combo("mode", &curr_rendering_mode_, items, AT_COUNTOF(items))) {
                renderer_.SetMode((idaten::SVGFPathTracing::Mode)curr_rendering_mode_);
            }

            if (curr_rendering_mode_ == idaten::SVGFPathTracing::Mode::AOVar) {
                const char* aovitems[] = { "Normal", "TexColor", "Depth", "Wire", "Barycentric", "Motion", "ObjId" };

                if (ImGui::Combo("aov", &curr_aov_mode_, aovitems, AT_COUNTOF(aovitems))) {
                    renderer_.SetAOVMode((aten::SVGFAovMode)curr_aov_mode_);
                }
            }
            else if (curr_rendering_mode_ == idaten::SVGFPathTracing::Mode::SVGF) {
                int32_t iterCnt = renderer_.getAtrousIterCount();
                if (ImGui::SliderInt("Atrous Iter", &iterCnt, 1, 5)) {
                    renderer_.setAtrousIterCount(iterCnt);
                }
            }

            bool enableTAA = taa_.isEnableTAA();
            bool canShowTAADiff = taa_.canShowTAADiff();

            if (ImGui::Checkbox("Enable TAA", &enableTAA)) {
                taa_.enableTAA(enableTAA);
            }
            if (ImGui::Checkbox("Show TAA Diff", &canShowTAADiff)) {
                taa_.showTAADiff(canShowTAADiff);
            }

            ImGui::Checkbox("Show AABB", &is_show_aabb_);
#endif

            auto cam = camera_.param();
            ImGui::Text("Pos %f/%f/%f", cam.origin.x, cam.origin.y, cam.origin.z);
            ImGui::Text("At  %f/%f/%f", cam.center.x, cam.center.y, cam.center.z);
        }

#ifdef ENABLE_SVGF
        idaten::SVGFPathTracing::PickedInfo info;
        auto isPicked = renderer_.GetPickedPixelInfo(info);
        if (isPicked) {
            AT_PRINTF("[%d, %d]\n", info.ix, info.iy);
            AT_PRINTF("  nml[%f, %f, %f]\n", info.normal.x, info.normal.y, info.normal.z);
            AT_PRINTF("  mesh[%d] mtrl[%d], tri[%d]\n", info.meshid, info.mtrlid, info.triid);
        }
#endif

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

#ifdef ENABLE_SVGF
            if (will_pick_pixel_info_) {
                renderer_.WillPickPixel(x, y);
                will_pick_pixel_info_ = false;
            }
#endif
        }
    }

    void OnMouseMove(int32_t x, int32_t y)
    {
        if (is_mouse_l_btn_down_) {
            aten::CameraOperator::rotate(
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
        const real offset = real(0.5);

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
            else if (key == aten::Key::Key_SPACE) {
                if (timeline_.isPaused()) {
                    timeline_.start();
                }
                else {
                    timeline_.pause();
                }
            }
            else if (key == aten::Key::Key_CONTROL) {
                will_pick_pixel_info_ = true;
                return;
            }
        }

        if (press) {
            switch (key) {
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
    void update(int32_t frame)
    {
        if (deform_mdl_) {
            auto mdl = deform_mdl_->getHasObjectAsRealType();

            if (defrom_anm_) {
                aten::mat4 mtx_L2W;
                mtx_L2W.asScale(0.01f);
                mdl->update(mtx_L2W, timeline_.getTime(), defrom_anm_.get());
            }
            else {
                mdl->update(aten::mat4(), 0, nullptr);
            }

            timeline_.advance(1.0f / 60.0f);

            const auto& mtx = mdl->getMatrices();
            skinning_.update(&mtx[0], mtx.size());

            aten::vec3 aabbMin, aabbMax;

            bool isRestart = (frame == 1);

            // NOTE
            // Add verted offset, in the first frame.
            // In "skinning_.compute", vertex offset is added to triangle paremters.
            // Added vertex offset is valid permanently, so specify vertex offset just only one time.
            skinning_.compute(aabbMin, aabbMax, isRestart);

            mdl->setBoundingBox(aten::aabb(aabbMin, aabbMax));
            deform_mdl_->update(true);

            const auto sceneBbox = aten::aabb(aabbMin, aabbMax);
            auto& nodes = renderer_.getCudaTextureResourceForBvhNodes();

            auto& vtxPos = skinning_.getInteropVBO()[0];
            auto& tris = skinning_.getTriangles();

            // TODO
            size_t deformPos = nodes.size() - 1;

            // NOTE
            // Vertex offset was added in "skinning_.compute".
            // But, in "lbvh_.build", vertex index have to be handled as zero based index.
            // Vertex offset have to be removed from vertex index.
            // So, specify minus vertex offset.
            // This is work around, too complicated...
            lbvh_.build(
                nodes[deformPos],
                tris,
                tri_offset_,
                sceneBbox,
                vtxPos,
                -vtx_offset_,
                nullptr);

            // Copy computed vertices, triangles to the tracer.
            renderer_.updateGeometry(
                skinning_.getInteropVBO(),
                vtx_offset_,
                skinning_.getTriangles(),
                tri_offset_);

            {
                auto accel = scene_.getAccel();
                accel->update(ctxt_);

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

                renderer_.updateBVH(
                    shapeparams,
                    nodes,
                    mtxs);
            }
        }
    }

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    aten::AcceleratedScene<aten::GPUBvh> scene_;
    aten::context ctxt_;

    std::shared_ptr<aten::instance<aten::deformable>> deform_mdl_;
    std::shared_ptr<aten::DeformAnimation> defrom_anm_;

#ifdef ENABLE_SVGF
    idaten::SVGFPathTracing renderer_;
#else
    idaten::PathTracing renderer_;
#endif

    aten::FilmProgressive buffer_{ WIDTH, HEIGHT };

    std::shared_ptr<aten::visualizer> visualizer_;

    float avg_cuda_time_{ 0.0f };

    aten::GammaCorrection gamma_;
    aten::TAA taa_;

    aten::FBO fbo_;

    aten::RasterizeRenderer rasterizer_;
    aten::RasterizeRenderer rasterizer_aabb_;

    aten::shader shader_raasterize_deformable_;

    idaten::Skinning skinning_;
    aten::Timeline timeline_;

    idaten::LBVHBuilder lbvh_;
    size_t tri_offset_{ 0 };
    int32_t vtx_offset_{ 0 };

    bool will_show_gui_{ true };
    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    int32_t max_samples_{ 1 };
    int32_t max_bounce_{ 5 };
    int32_t curr_rendering_mode_{ static_cast<int32_t>(idaten::SVGFPathTracing::Mode::SVGF) };
    int32_t curr_aov_mode_{ static_cast<int32_t>(aten::SVGFAovMode::WireFrame) };
    bool is_show_aabb_{ false };

    bool will_pick_pixel_info_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};

int32_t main()
{
    AT_ASSERT_LOG(
        (std::is_same_v<Scene, DeformScene> || std::is_same_v<Scene, DeformInBoxScene>),
        "Allow only deformable scene");

    aten::timer::init();
    aten::OMPUtil::setThreadNum(DeformationRendererApp::ThreadNum);

    aten::initSampler(WIDTH, HEIGHT);

    auto app = std::make_shared<DeformationRendererApp>();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        std::bind(&DeformationRendererApp::Run, app),
        std::bind(&DeformationRendererApp::OnClose, app),
        std::bind(&DeformationRendererApp::OnMouseBtn, app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
        std::bind(&DeformationRendererApp::OnMouseMove, app, std::placeholders::_1, std::placeholders::_2),
        std::bind(&DeformationRendererApp::OnMouseWheel, app, std::placeholders::_1),
        std::bind(&DeformationRendererApp::OnKey, app, std::placeholders::_1, std::placeholders::_2));

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
