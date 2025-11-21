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
    static aten::tuple<std::shared_ptr<aten::instance<aten::deformable>>, std::shared_ptr<aten::DeformAnimation>> makeScene(
        aten::context& ctxt, aten::scene* scene)
    {
        auto mdl = aten::TransformableFactory::createDeformable(ctxt);
        mdl->read("../../asset/converted_unitychan/unitychan_gpu.mdl");

        aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
        aten::MaterialLoader::load("../../asset/converted_unitychan/unitychan_mtrl.xml", ctxt);

        for (auto& tex : ctxt.GetTextures()) {
            tex->SetFilterMode(aten::TextureFilterMode::Linear);
            tex->SetAddressMode(aten::TextureAddressMode::Wrap);
        }

        auto deformMdl = aten::TransformableFactory::createInstance<aten::deformable>(ctxt, mdl, aten::mat4::Identity);
        scene->add(deformMdl);

        aten::ImageLoader::setBasePath("./");

        auto deformAnm = std::make_shared<aten::DeformAnimation>();
        deformAnm->read("../../asset/converted_unitychan/unitychan_WAIT00.anm");

        return aten::make_tuple(deformMdl, deformAnm);
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

class DeformInBoxScene {
public:
    static aten::tuple<std::shared_ptr<aten::instance<aten::deformable>>, std::shared_ptr<aten::DeformAnimation>> makeScene(
        aten::context& ctxt, aten::scene* scene)
    {
#if 1
        {
            auto emit = CreateMaterial("light", ctxt, aten::MaterialType::Emissive, aten::vec3(1.0f, 1.0f, 1.0f));

            CreateMaterial("backWall", ctxt, aten::MaterialType::Diffuse, aten::vec3(0.580000f, 0.568000f, 0.544000f));
            CreateMaterial("ceiling", ctxt, aten::MaterialType::Diffuse, aten::vec3(0.580000f, 0.568000f, 0.544000f));
            CreateMaterial("floor", ctxt, aten::MaterialType::Diffuse, aten::vec3(0.580000f, 0.568000f, 0.544000f));
            CreateMaterial("leftWall", ctxt, aten::MaterialType::Diffuse, aten::vec3(0.504000f, 0.052000f, 0.040000f));
            CreateMaterial("rightWall", ctxt, aten::MaterialType::Diffuse, aten::vec3(0.112000f, 0.360000f, 0.072800f));

            auto objs = aten::ObjLoader::Load("../../asset/cornellbox/box.obj", ctxt, nullptr, nullptr, false);

            auto light = aten::TransformableFactory::createInstance<aten::PolygonObject>(
                ctxt,
                objs[0],
                aten::vec3(0.0f),
                aten::vec3(0.0f),
                aten::vec3(1.0f));
            scene->add(light);

            auto areaLight = std::make_shared<aten::AreaLight>(light, emit->param().baseColor, 400.0f);
            ctxt.AddLight(areaLight);

            auto box = aten::TransformableFactory::createInstance<aten::PolygonObject>(ctxt, objs[1], aten::mat4::Identity);
            scene->add(box);
        }
#endif

        auto mdl = aten::TransformableFactory::createDeformable(ctxt);
        mdl->read("../../asset/converted_unitychan/unitychan_gpu.mdl");

        aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
        aten::MaterialLoader::load("../../asset/converted_unitychan/unitychan_mtrl.xml", ctxt);

        auto deformMdl = aten::TransformableFactory::createInstance<aten::deformable>(ctxt, mdl, aten::mat4::Identity);
        scene->add(deformMdl);

        aten::ImageLoader::setBasePath("./");

        auto deformAnm = std::make_shared<aten::DeformAnimation>();
        deformAnm->read("../../asset/converted_unitychan/unitychan_WAIT00.anm");

        return aten::make_tuple(deformMdl, deformAnm);
    }

    static void getCameraPosAndAt(
        aten::vec3& pos,
        aten::vec3& at,
        float& fov)
    {
        pos = aten::vec3(0.f, 1.f, 3.f);
        at = aten::vec3(0.f, 1.f, 0.f);
        fov = 45;
    }
};

#define Scene DeformScene
//#define Scene DeformInBoxScene

// NOTE:
// Dummy for deformable.
// To compute animation and reflect it to bvh, we need to build lbvh per tick.
// It means we call lbvh's build per update().
// It's apart from the theory for the static bvh.
// Therefore, to create accelerator, we inject this as the dummy and have the acutal bvh separately.
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
        return;
    }

    virtual bool hit(
        const aten::context& ctxt,
        const aten::ray& r,
        float t_min, float t_max,
        aten::Intersection& isect) const override final
    {
        AT_ASSERT(false);
        return false;
    }

    virtual bool HitWithLod(
        const aten::context& ctxt,
        const aten::ray& r,
        float t_min, float t_max,
        bool enableLod,
        aten::Intersection& isect,
        aten::HitStopType hit_stop_type = aten::HitStopType::Closest) const override final
    {
        AT_ASSERT(false);
        return false;
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
        float vfov;
        Scene::getCameraPosAndAt(pos, at, vfov);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        aten::accelerator::setUserDefsInternalAccelCreator(Lbvh::create);
        aten::tie(deform_mdl_, defrom_anm_) = Scene::makeScene(ctxt_, &scene_);
        scene_.build(ctxt_);

#ifdef ENABLE_ENVMAP
        auto envmap = aten::ImageLoader::load("../../asset/envmap/studio015.hdr", ctxt_);
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
            auto mdl = deform_mdl_->GetHasObjectAsRealType();

            // NOTE:
            // To compute skinning all vertices together,
            // the deformable model has one vertex buffer for gpu skinning.
            // The vertex buffer is firstly empty.
            // The computed vertices by gpu skinning are sotred in that vertex buffer.
            // And then, the deformable model can be rendered with the vertex buffer
            // regardless of rasterization or ray tracing.
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
                timeline_.init(defrom_anm_->getDesc().time, float(0));
                timeline_.enableLoop(true);
                timeline_.start();
            }
        }

        {
            tri_offset_ = ctxt_.GetTriangleNum();
            vtx_offset_ = ctxt_.GetVertexNum();

            const auto& nodes = scene_.getAccel()->getNodes();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            renderer_.UpdateSceneData(
                visualizer_->GetGLTextureHandle(),
                WIDTH, HEIGHT,
                camparam, ctxt_, nodes,
                advanceTriNum, advanceVtxNum,
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

        skinning_.setVtxOffset(vtx_offset_);
        lbvh_.init(advanceTriNum);

#ifdef ENABLE_SVGF
        renderer_.SetMode((idaten::SVGFPathTracing::Mode)curr_rendering_mode_);
        renderer_.SetAOVMode((aten::SVGFAovMode)curr_aov_mode_);
#endif

        renderer_.SetCanSSRTHitTest(true);

        return true;
    }

    bool Run()
    {
        auto frame = renderer_.GetFrameCount();

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

#ifdef ENABLE_SVGF
        rasterizer_.drawSceneForGBuffer(
            renderer_.GetFrameCount(),
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
            ImGui::Text("[%d] %.3f ms/frame (%.1f FPS)", renderer_.GetFrameCount(), 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
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
    void update(int32_t frame)
    {
        auto mdl = deform_mdl_->GetHasObjectAsRealType();

        // NOTE:
        // Apply local to world matrix to vertices is done in LBVHBuilder.
        // Therefore, no need to care of local to world matrix in instance class.

        // TODO:
        // Ideally, regardless of any bvh, how to deal with local to world matrix should be standardized.

        aten::mat4 mtx_L2W;
        mtx_L2W.asScale(0.01f);
        mdl->update(mtx_L2W, timeline_.getTime(), defrom_anm_.get());

        timeline_.advance(1.0f / 60.0f);

        const auto& mtx = mdl->getMatrices();
        skinning_.update(&mtx[0], mtx.size());

        aten::vec3 aabbMin, aabbMax;

        bool isRestart = (frame == 1);

        // NOTE:
        // We specified the vertex offset.
        // In "skinning_.compute", the the vertex offset is added to triangle paremters.
        // The specified vertex offset can be assumed as that it's valid permanently.
        // So, specifying the vertex offset just only one time is enough.
        // It means we don't need to specify the vertex offset per update.
        skinning_.compute(aabbMin, aabbMax, isRestart);

        mdl->setBoundingBox(aten::aabb(aabbMin, aabbMax));
        deform_mdl_->update(ctxt_, true);

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

            const auto& nodes = scene_.getAccel()->getNodes();

            renderer_.updateBVH(ctxt_, nodes);
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
