#include <cmdline.h>
#include <imgui.h>

#include "visualizer/atengl.h"

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

#include "StylizedHighlightTest.h"


constexpr int32_t WIDTH = 1280;
constexpr int32_t HEIGHT = 720;
constexpr char* TITLE = "StylizedHighlightTest";

class StylizedHighlightTestApp {
public:
    StylizedHighlightTestApp() = default;
    ~StylizedHighlightTestApp() = default;

    StylizedHighlightTestApp(const StylizedHighlightTestApp&) = delete;
    StylizedHighlightTestApp(StylizedHighlightTestApp&&) = delete;
    StylizedHighlightTestApp operator=(const StylizedHighlightTestApp&) = delete;
    StylizedHighlightTestApp operator=(StylizedHighlightTestApp&&) = delete;

    bool Init(int32_t argc, char* argv[])
    {
        if (!ParseArguments(argc, argv)) {
            AT_ASSERT(false);
            return false;
        }

        // TODO
        aten::vec3 pos(0.f, 100.0f, 300.0f);
        aten::vec3 at(0.f, 0.f, 0.f);
        float vfov = float(45);

        camera_.init(
            pos,
            at,
            aten::vec3(0, 1, 0),
            vfov,
            WIDTH, HEIGHT);

        Load(args_.input);

        rasterizer_.init(
            WIDTH, HEIGHT,
            "highlight_vs.glsl",
            "highlight_fs.glsl");

        tester_.Init(
            WIDTH, HEIGHT,
            "../shader/drawobj_vs.glsl",
            "../shader/drawobj_fs.glsl");

        return true;
    }

    void Load(std::string_view path)
    {
        ctxt_.CleanAll();

        LoadObj(path, "");

        obj_enables_.resize(objs_.size(), true);

        ctxt_.InitAllTextureAsGLTexture();

        for (auto& obj : objs_) {
            obj->buildForRasterizeRendering(ctxt_);
        }

        auto texNum = ctxt_.GetTextureNum();

        for (int32_t i = 0; i < texNum; i++) {
            auto tex = ctxt_.GetTexture(i);
            tex->initAsGLTexture();
        }

        camera_.FitBoundingBox(obj_aabb_);
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

        aten::RasterizeRenderer::beginRender();

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        obj_min_ = std::min<int32_t>(obj_min_, static_cast<int32_t>(objs_.size()));
        obj_max_ = std::min<int32_t>(obj_max_, static_cast<int32_t>(objs_.size()));

        rasterizer_.drawWithOutsideRenderFunc(
            ctxt_,
            [&](aten::RasterizeRenderer::FuncObjRenderer func) {
                auto& shader = rasterizer_.getShader();
                auto h_translation_dt = shader.getHandle("translation_dt");
                CALL_GL_API(::glUniform1f(h_translation_dt, half_trans_t_));

                auto h_scale_t = shader.getHandle("scale_t");
                CALL_GL_API(::glUniform1f(h_scale_t, half_scale_));

                auto h_split_t = shader.getHandle("split_t");
                CALL_GL_API(::glUniform1f(h_split_t, half_split_t_));

                for (size_t i = 0; i < objs_.size(); i++) {
                    auto& obj = objs_[i];
                    func(*obj);
                }
            },
            &camera_,
                is_wireframe_);

        tester_.UpdateHalfVectors(half_trans_t_, half_scale_, half_split_t_);
        tester_.Draw(ctxt_, camera_);

        const auto& aabb_max = obj_aabb_.maxPos();
        const auto& aabb_min = obj_aabb_.minPos();
        ImGui::Text("max(%.3f, %.3f, %.3f)", aabb_max.x, aabb_max.y, aabb_max.z);
        ImGui::Text("min(%.3f, %.3f, %.3f)", aabb_min.x, aabb_min.y, aabb_min.z);

        ImGui::SliderInt("min", &obj_min_, 0, obj_max_);
        ImGui::SliderInt("max", &obj_max_, obj_min_, static_cast<int32_t>(objs_.size() - 1));

        ImGui::SliderFloat("trans t", &half_trans_t_, -1, 1);
        ImGui::SliderFloat("scale t", &half_scale_, 0.0, 1);
        ImGui::SliderFloat("split t", &half_split_t_, 0.0, 1);

        for (size_t i = obj_min_; i < obj_max_; i++) {
            const auto& obj = objs_[i];
            auto mtrl_name = obj->getShapes()[0]->GetMaterial()->name();

            auto name = obj->getName();
            name += mtrl_name;

            bool is_enable = obj_enables_.at(i);
            if (ImGui::Checkbox(name.c_str(), &is_enable)) {
                obj_enables_[i] = is_enable;
            }
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
            else if (key == aten::Key::Key_F3) {
                is_wireframe_ = !is_wireframe_;
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
            param.type = aten::MaterialType::Diffuse;
            param.baseColor = aten::vec3(1, 1, 1);;

            auto mtrl = ctxt_.CreateMaterialWithMaterialParameter(
                "m1",
                param,
                nullptr, nullptr, nullptr);
        }
        else {
            aten::MaterialLoader::load(mtrlpath, ctxt_);
        }

        objs_ = aten::ObjLoader::Load(objpath, ctxt_, nullptr, nullptr, true);

        obj_aabb_.empty();
        for (const auto& obj : objs_) {
            const auto& aabb = obj->getBoundingbox();
            obj_aabb_.expand(aabb);
        }

        return objs_[0];
    }

    bool ParseArguments(int32_t argc, char* argv[])
    {
        cmdline::parser cmd;

        {
            cmd.add<std::string>("input", 'i', "input model filename", false);

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

        if (cmd.exist("input")) {
            args_.input = cmd.get<std::string>("input");
        }

        return true;
    }

    struct Args {
        std::string input{ "../../asset/sphere/sphere.obj" };
    } args_;

    aten::context ctxt_;

    aten::RasterizeRenderer rasterizer_;

    StylizedHighlightTest tester_;
    float half_trans_t_{ 0.0F };
    float half_scale_{ 0.0F };
    float half_split_t_{ 0.0F };

    std::vector<std::shared_ptr<aten::PolygonObject>> objs_;
    std::vector<bool> obj_enables_;
    aten::aabb obj_aabb_;

    int32_t obj_min_{ 0 };
    int32_t obj_max_{ 20 };

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

    auto app = std::make_shared<StylizedHighlightTestApp>();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        [&app]() { return app->Run(); },
        [&app]() { app->OnClose(); },
        [&app](bool left, bool press, int32_t x, int32_t y) { app->OnMouseBtn(left, press, x, y); },
        [&app](int32_t x, int32_t y) { app->OnMouseMove(x, y); },
        [&app](int32_t delta) {app->OnMouseWheel(delta); },
        [&app](bool press, aten::Key key) { app->OnKey(press, key); },
        [&app](std::string_view path) { app->Load(path); });

    if (id >= 0) {
        app->GetContext().SetIsWindowInitialized(true);
    }
    else {
        AT_ASSERT(false);
        return 1;
    }

    app->Init(argc, argv);

    wnd->Run();

    app.reset();

    wnd->Terminate();
}
