#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr char* TITLE = "DeformationModelViewer";

class DeformationModelViewerApp {
public:
    DeformationModelViewerApp() = default;
    ~DeformationModelViewerApp() = default;

    DeformationModelViewerApp(const DeformationModelViewerApp&) = delete;
    DeformationModelViewerApp(DeformationModelViewerApp&&) = delete;
    DeformationModelViewerApp operator=(const DeformationModelViewerApp&) = delete;
    DeformationModelViewerApp operator=(DeformationModelViewerApp&&) = delete;

    bool Init(int32_t argc, char* argv[])
    {
        // TODO
        // Enable to handle and pass the command argument.
#if 0
        if (!ParseArguments(argc, argv)) {
            AT_ASSERT(false);
            return false;
        }
#else
        args_.input = "../../asset/converted_unitychan/unitychan_gpu.mdl";
        args_.tex_dir = "../../asset/unitychan/Texture";
        args_.mtrl = "../../asset/converted_unitychan/unitychan_mtrl.xml";
        args_.anm = "../../asset/converted_unitychan/unitychan_WAIT00.anm";
#endif

        rasterizer_aabb_.init(
            WIDTH, HEIGHT,
            "../shader/simple3d_vs.glsl",
            "../shader/simple3d_fs.glsl");

        mdl_ = aten::TransformableFactory::createDeformable(ctxt_);

        mdl_->read(args_.input.c_str());

        if (!args_.anm.empty()) {
            anm_.read(args_.anm.c_str());

            timeline_.init(anm_.getDesc().time, real(0));
            timeline_.enableLoop(true);
            timeline_.start();
        }

        const auto is_gpu_skinning = mdl_->isEnabledForGPUSkinning();

        if (is_gpu_skinning) {
            renderer_.init(
                WIDTH, HEIGHT,
                "drawobj_vs.glsl",
                "drawobj_fs.glsl");
        }
        else {
            renderer_.init(
                WIDTH, HEIGHT,
                "../shader/skinning_vs.glsl",
                "../shader/skinning_fs.glsl");
        }

        mdl_->initGLResourcesWithDeformableRenderer(renderer_);

        aten::ImageLoader::setBasePath(args_.tex_dir.c_str());

        if (!aten::MaterialLoader::load(args_.mtrl.c_str(), ctxt_, asset_manager_)) {
            return 0;
        }

        auto texNum = ctxt_.GetTextureNum();

        for (int32_t i = 0; i < texNum; i++) {
            auto tex = ctxt_.GetTexture(i);
            tex->initAsGLTexture();
        }

        // TODO
        aten::vec3 pos(0, 71, 225);
        aten::vec3 at(0, 71, 216);
        real vfov = real(45);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        if (is_gpu_skinning) {
            auto& vb = mdl_->getVBForGPUSkinning();

            std::vector<aten::SkinningVertex> vtx;
            std::vector<uint32_t> idx;
            std::vector<aten::TriangleParameter> tris;

            mdl_->getGeometryData(ctxt_, vtx, idx, tris);

            skinning_.initWithTriangles(
                &vtx[0], vtx.size(),
                &tris[0], tris.size(),
                &vb);
        }

        return true;
    }

    bool Run()
    {
        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = real(0.1);
            camparam.zfar = real(10000.0);

            is_camera_dirty_ = false;
        }

        timeline_.advance(1.0f / 60.0f);


        mdl_->update(aten::mat4(), timeline_.getTime(), &anm_);
        //g_mdl.update(aten::mat4(), nullptr, 0);

        aten::vec3 aabbMin, aabbMax;

        bool is_gpu_skinning = mdl_->isEnabledForGPUSkinning();

        if (is_gpu_skinning) {
            const auto& mtx = mdl_->getMatrices();
            skinning_.update(&mtx[0], mtx.size());
            skinning_.update(&mtx[0], mtx.size());
            skinning_.compute(aabbMin, aabbMax);
        }

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        renderer_.render(ctxt_, &camera_, mdl_.get());

        if (will_take_screen_shot_)
        {
            auto screen_shot_file_name = aten::StringFormat("sc_%d.png", screen_shot_count_);

            aten::visualizer::takeScreenshot(screen_shot_file_name, WIDTH, HEIGHT);

            will_take_screen_shot_ = false;
            screen_shot_count_++;

            AT_PRINTF("Take Screenshot[%s]\n", screen_shot_file_name.c_str());
        }

        if (is_show_aabb_) {
            rasterizer_aabb_.drawAABB(
                &camera_,
                aten::aabb(aabbMin, aabbMax));
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
            if (key == aten::Key::Key_F1) {
                is_show_gui_ = !is_show_gui_;
                return;
            }
            else if (key == aten::Key::Key_F2) {
                will_take_screen_shot_ = true;
                return;
            }
            else if (key == aten::Key::Key_F3) {
                is_show_aabb_ = !is_show_aabb_;
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
    bool ParseArguments(int32_t argc, char* argv[])
    {
        cmdline::parser cmd;

        {
            cmd.add<std::string>("input", 'i', "input model filename", true);
            cmd.add<std::string>("anm", 'a', "animation filename", false);
            cmd.add<std::string>("mtrl", 'm', "material filename", true);
            cmd.add<std::string>("tex_dir", 't', "texture directory path", true);

            cmd.add("help", '?', "print usage");
        }

        bool is_cmd_valid = cmd.parse(argc, argv);

        if (cmd.exist("help")) {
            std::cerr << cmd.usage();
            return false;
        }

        if (!is_cmd_valid) {
            std::cerr << cmd.error_full() << std::endl << cmd.usage();
            return false;
        }

        args_.input = cmd.exist("input");

        if (cmd.exist("anm")) {
            args_.anm = cmd.get<std::string>("anm");
        }

        args_.mtrl = cmd.exist("mtrl");


        args_.tex_dir = cmd.exist("tex_dir");


        return true;
    }

    struct Args {
        std::string input;
        std::string anm;
        std::string mtrl;
        std::string tex_dir;
    } args_;

    aten::context ctxt_;

    aten::AssetManager asset_manager_;

    idaten::Skinning skinning_;

    std::shared_ptr<aten::deformable> mdl_;
    aten::DeformAnimation anm_;

    aten::DeformableRenderer renderer_;

    aten::Timeline timeline_;

    aten::RasterizeRenderer rasterizer_aabb_;

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    bool is_show_gui_{ false };
    bool is_show_aabb_{ true };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};

int32_t main(int32_t argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    auto app = std::make_shared<DeformationModelViewerApp>();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        std::bind(&DeformationModelViewerApp::Run, app),
        std::bind(&DeformationModelViewerApp::OnClose, app),
        std::bind(&DeformationModelViewerApp::OnMouseBtn, app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
        std::bind(&DeformationModelViewerApp::OnMouseMove, app, std::placeholders::_1, std::placeholders::_2),
        std::bind(&DeformationModelViewerApp::OnMouseWheel, app, std::placeholders::_1),
        std::bind(&DeformationModelViewerApp::OnKey, app, std::placeholders::_1, std::placeholders::_2));

    if (id >= 0) {
        app->GetContext().SetIsWindowInitialized(true);
    }
    else {
        AT_ASSERT(false);
        return 1;
    }

    if (!app->Init(argc, argv)) {
        return 1;
    }

    wnd->Run();

    app.reset();

    wnd->Terminate();
}
