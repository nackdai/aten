#include <memory>

#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr char* TITLE = "AssimpLoadViewer";

class AssimpLoadViewerApp {
public:
    AssimpLoadViewerApp() = default;
    ~AssimpLoadViewerApp() = default;

    AssimpLoadViewerApp(const AssimpLoadViewerApp&) = delete;
    AssimpLoadViewerApp(AssimpLoadViewerApp&&) = delete;
    AssimpLoadViewerApp operator=(const AssimpLoadViewerApp&) = delete;
    AssimpLoadViewerApp operator=(AssimpLoadViewerApp&&) = delete;

    bool Init(int32_t argc, char* argv[])
    {
        // TODO
        // Enable to handle and pass the command argument.
#if 0
        if (!ParseArguments(argc, argv)) {
            AT_ASSERT(false);
            return false;
        }
#endif

        obj_ = LoadObj("../../3rdparty/assimp/test/models/FBX/box.fbx", "");
        if (!obj_) {
            AT_ASSERT(false);
            return false;
        }

        ctxt_.InitAllTextureAsGLTexture();

        obj_->buildForRasterizeRendering(ctxt_);

        rasterizer_.init(
            WIDTH, HEIGHT,
            "../shader/drawobj_vs.glsl",
            "../shader/drawobj_fs.glsl");

        auto texNum = ctxt_.GetTextureNum();

        for (int32_t i = 0; i < texNum; i++) {
            auto tex = ctxt_.GtTexture(i);
            tex->initAsGLTexture();
        }

        // TODO
        aten::vec3 pos(0.f, 10.0f, 30.0f);
        aten::vec3 at(0.f, 0.f, 0.f);
        real vfov = real(45);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

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

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        rasterizer_.drawObject(
            ctxt_,
            *obj_,
            &camera_,
            is_wireframe_);

        if (will_take_screen_shot_)
        {
            auto screenshot_file_name = aten::StringFormat("sc_%d.png", screen_shot_count_);

            aten::visualizer::takeScreenshot(screenshot_file_name, WIDTH, HEIGHT);

            will_take_screen_shot_ = false;
            screen_shot_count_++;

            AT_PRINTF("Take Screenshot[%s]\n", screenshot_file_name);
        }

        if (will_show_gui_) {
            ImGui::Checkbox("Wireframe,", &is_wireframe_);
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
    std::shared_ptr<aten::PolygonObject> LoadObj(
        std::string_view objpath,
        std::string_view mtrlpath)
    {
        std::string pathname;
        std::string extname;
        std::string filename;

        aten::getStringsFromPath(
            objpath,
            pathname,
            extname,
            filename);

        if (mtrlpath.empty()) {
            aten::MaterialParameter param;
            param.type = aten::MaterialType::Lambert;
            param.baseColor = aten::vec3(1, 1, 1);;

            auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(
                param,
                nullptr, nullptr, nullptr);
            asset_manager_.registerMtrl("dummy", mtrl);
        }
        else {
            aten::MaterialLoader::load(mtrlpath, ctxt_, asset_manager_);
        }

        std::vector<std::shared_ptr<aten::PolygonObject>> objs;
        aten::AssimpImporter::load(
            objpath,
            objs,
            ctxt_, asset_manager_,
            [&](std::string_view name,
                aten::context& ctxt,
                const aten::MaterialParameter& mtrl_param,
                const std::string& albedo,
                const std::string& nml)
            {
                auto mtrl = asset_manager_.getMtrl(name);
                if (!mtrl) {
                    auto albedo_map = albedo.empty()
                        ? nullptr
                        : aten::ImageLoader::load(pathname + albedo, ctxt, asset_manager_);
                    auto nml_map = nml.empty()
                        ? nullptr
                        : aten::ImageLoader::load(pathname + nml, ctxt, asset_manager_);

                    mtrl = ctxt.CreateMaterialWithMaterialParameter(
                        mtrl_param,
                        albedo_map.get(),
                        nml_map.get(),
                        nullptr);
                    mtrl->setName(name);
                }

                return mtrl;
            });

        // NOTE
        // Number of obj is currently only one.
        AT_ASSERT(objs.size() == 1);

        return objs[0];
    }

    bool ParseArguments(int32_t argc, char* argv[])
    {
        cmdline::parser cmd;

        {
            cmd.add<std::string>("input", 'i', "input model filename", true);

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

        args_.input = cmd.get<std::string>("input");

        return true;
    }

    struct Args {
        std::string input;
    } args_;

    aten::context ctxt_;

    aten::AssetManager asset_manager_;

    aten::RasterizeRenderer rasterizer_;
    std::shared_ptr<aten::PolygonObject> obj_;

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    bool will_show_gui_{ true };
    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    bool is_wireframe_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};

int32_t main(int32_t argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    auto app = std::make_shared<AssimpLoadViewerApp>();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        std::bind(&AssimpLoadViewerApp::Run, app),
        std::bind(&AssimpLoadViewerApp::OnClose, app),
        std::bind(&AssimpLoadViewerApp::OnMouseBtn, app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
        std::bind(&AssimpLoadViewerApp::OnMouseMove, app, std::placeholders::_1, std::placeholders::_2),
        std::bind(&AssimpLoadViewerApp::OnMouseWheel, app, std::placeholders::_1),
        std::bind(&AssimpLoadViewerApp::OnKey, app, std::placeholders::_1, std::placeholders::_2));

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
