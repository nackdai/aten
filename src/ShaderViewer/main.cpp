#include <cmdline.h>
#include <imgui.h>

#include "magnifier.h"

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr char* TITLE = "ShaderViewer";

class ShaderViewerApp {
public:
    ShaderViewerApp() = default;
    ~ShaderViewerApp() = default;

    ShaderViewerApp(const ShaderViewerApp&) = delete;
    ShaderViewerApp(ShaderViewerApp&&) = delete;
    ShaderViewerApp operator=(const ShaderViewerApp&) = delete;
    ShaderViewerApp operator=(ShaderViewerApp&&) = delete;

    bool Init(int32_t argc, char* argv[])
    {
        if (!ParseArguments(argc, argv)) {
            AT_ASSERT(false);
            return false;
        }
        return true;
    }

    bool Run()
    {
        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            is_camera_dirty_ = false;
        }

        if (objs_.empty()) {
            LoadObj(args_.model, "");

            obj_enables_.resize(objs_.size(), true);

            ctxt_.InitAllTextureAsGLTexture();

            for (auto& obj : objs_) {
                obj->buildForRasterizeRendering(ctxt_);
            }

            rasterizer_.init(
                WIDTH, HEIGHT,
                "shader/vs.glsl",
                "shader/retroreflective.glsl");

            visualizer_ = aten::visualizer::init(WIDTH, HEIGHT);

            fbo_ = std::make_shared<decltype(fbo_)::element_type>();
            fbo_->init(WIDTH, HEIGHT, aten::PixelFormat::rgba8);

            magnifier_ = std::make_shared<decltype(magnifier_)::element_type>();
            magnifier_->init(
                WIDTH, HEIGHT,
                "shader/fullscreen_vs.glsl",
                "shader/magnifier.glsl");
            visualizer_->addPostProc(magnifier_.get());

            auto texNum = ctxt_.GetTextureNum();

            for (int32_t i = 0; i < texNum; i++) {
                auto tex = ctxt_.GetTexture(i);
                tex->initAsGLTexture();
            }

            // TODO
            aten::vec3 pos(0.f, 0.0f, 30.0f);
            aten::vec3 at(0.f, 0.f, 0.f);
            float vfov = float(45);

            camera_.init(
                pos,
                at,
                aten::vec3(0, 1, 0),
                vfov,
                WIDTH, HEIGHT);
        }

        aten::RasterizeRenderer::beginRender(fbo_);

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        const size_t obj_min = 0;
        const size_t obj_max = objs_.size();

        rasterizer_.drawWithOutsideRenderFunc(
            ctxt_,
            [&](aten::RasterizeRenderer::FuncObjRenderer func) {
                auto& shader = rasterizer_.getShader();
                shader.setUniformVec3("pointLitPos", aten::vec3(0.0f, 0.0f, 50.0f));
                shader.setUniformVec3("pointLitClr", aten::vec3(0.8f, 0.0f, 0.0f));
                shader.setUniformVec3("pointLitAttr", aten::vec3(0.0f, 0.05f, 0.0f));
                shader.setUniformVec3("cameraPos", camera_.GetPos());

                for (size_t i = obj_min; i < obj_max; i++) {
                    auto& obj = objs_[i];
                    func(*obj);
                }
            },
            &camera_,
            false);

        if (visualizer_) {
            aten::Values values{
                {"center_pos", aten::PolymorphicValue(aten::vec4(WIDTH * 0.5f, HEIGHT * 0.5f, 0, 0))},
                {"magnification", aten::PolymorphicValue(0.5f)},
                {"radius", aten::PolymorphicValue(200.0f)},
                {"circle_line_width", aten::PolymorphicValue(2.0f)},
                {"circle_line_color", aten::PolymorphicValue(aten::vec4(1, 0, 0, 0))},
            };
            magnifier_->setParam(values);

            aten::RasterizeRenderer::beginRender();
            aten::RasterizeRenderer::clearBuffer(
                aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
                aten::vec4(0, 0.5f, 1.0f, 1.0f),
                1.0f,
                0);
            fbo_->BindAsTexture();
            visualizer_->renderGLTexture(fbo_->GetGLTextureHandle(), camera_.NeedRevert());
        }

        if (will_take_screen_shot_)
        {
            auto screen_shot_file_name = aten::StringFormat("sc_%d.png", screen_shot_count_);

            aten::visualizer::takeScreenshot(screen_shot_file_name, WIDTH, HEIGHT);

            will_take_screen_shot_ = false;
            screen_shot_count_++;

            AT_PRINTF("Take Screenshot[%s]\n", screen_shot_file_name.c_str());
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
        static const float offset_base = float(0.1);

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
    void LoadObj(
        std::string_view objpath,
        std::string_view mtrlpath)
    {
        if (mtrlpath.empty()) {
            aten::MaterialParameter param;
            param.type = aten::MaterialType::Diffuse;
            param.baseColor = aten::vec3(0.4, 0.4, 0.4);;

            auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(
                "m1",
                param,
                nullptr, nullptr, nullptr);
        }
        else {
            aten::MaterialLoader::load(mtrlpath, ctxt_);
        }

        objs_ = aten::ObjLoader::Load(objpath, ctxt_, nullptr, true);
    }

    bool ParseArguments(int32_t argc, char* argv[])
    {
        cmdline::parser cmd;
        {
            cmd.add<std::string>("model", 'm', "model filename", false);
        }

        bool is_cmd_valid = cmd.parse(argc, argv);

        if (!is_cmd_valid) {
            std::cerr << cmd.error_full() << std::endl << cmd.usage();
            return false;
        }

        if (cmd.exist("model")) {
            args_.model = cmd.get<std::string>("model");
        }

        return true;
    }

    struct Args {
        std::string model{ "../../asset/teapot/teapot.obj" };
    } args_;


    aten::context ctxt_;

    aten::RasterizeRenderer rasterizer_;

    std::vector<std::shared_ptr<aten::PolygonObject>> objs_;
    std::vector<bool> obj_enables_;

    std::shared_ptr<aten::visualizer> visualizer_;
    std::shared_ptr<aten::FBO> fbo_;

    std::shared_ptr<Magnifier> magnifier_;

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    bool will_show_gui_{ true };
    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};



int32_t main(int32_t argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    auto app = std::make_shared<ShaderViewerApp>();

    if (!app->Init(argc, argv)) {
        AT_ASSERT(false);
        return false;
    }

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        std::bind(&ShaderViewerApp::Run, app),
        std::bind(&ShaderViewerApp::OnClose, app),
        std::bind(&ShaderViewerApp::OnMouseBtn, app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
        std::bind(&ShaderViewerApp::OnMouseMove, app, std::placeholders::_1, std::placeholders::_2),
        std::bind(&ShaderViewerApp::OnMouseWheel, app, std::placeholders::_1),
        std::bind(&ShaderViewerApp::OnKey, app, std::placeholders::_1, std::placeholders::_2));

    if (id >= 0) {
        app->GetContext().SetIsWindowInitialized(true);
    }
    else {
        AT_ASSERT(false);
        return 1;
    }

    wnd->Run();

    app.reset();

    wnd->Terminate();

    return 1;
}
