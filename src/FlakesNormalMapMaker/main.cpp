#include "aten.h"

#include "FlakesNormalMapMaker.h"

#include <cmdline.h>
#include <imgui.h>

class FlakesNormalMapMakerApp {
public:
    static constexpr const char* TITLE = "FlakesNormalMapMaker";

    FlakesNormalMapMakerApp() = default;
    ~FlakesNormalMapMakerApp() = default;

    FlakesNormalMapMakerApp(const FlakesNormalMapMakerApp&) = delete;
    FlakesNormalMapMakerApp(FlakesNormalMapMakerApp&&) = delete;
    FlakesNormalMapMakerApp operator=(const FlakesNormalMapMakerApp&) = delete;
    FlakesNormalMapMakerApp operator=(FlakesNormalMapMakerApp&&) = delete;

    bool Init()
    {
        return true;
    }

    bool Run()
    {
        if (!maker_.IsValid()) {
            visualizer_ = aten::visualizer::init(args_.width, args_.height);

            if (!maker_.init(
                args_.width, args_.height,
                "../shader/FlakesTextureMaker_vs.glsl",
                "../shader/FlakesTextureMaker_fs.glsl"))
            {
                AT_ASSERT(false);
                return false;
            }

            visualizer_->addPostProc(&maker_);
        }

        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);

        visualizer_->render(false);

        if (NeedGui()) {
            auto flake_param = maker_.GetParameter();
            ImGui::SliderFloat("scale", &flake_param.flake_scale, 0.1f, 100.0f);
            ImGui::SliderFloat("size", &flake_param.flake_size, 0.1f, 1.0f);
            ImGui::SliderFloat("size variance", &flake_param.flake_size_variance, 0.0f, 1.0f);
            ImGui::SliderFloat("orientation", &flake_param.flake_normal_orientation, 0.0f, 1.0f);
            return true;
        }

        return false;
    }

    void OnClose()
    {
    }

    void OnMouseBtn(bool left, bool press, int32_t x, int32_t y)
    {
    }

    void OnMouseMove(int32_t x, int32_t y)
    {
    }

    void OnMouseWheel(int32_t delta)
    {
    }

    void OnKey(bool press, aten::Key key)
    {
    }

    bool ParseArgs(int32_t argc, char* argv[])
    {
        cmdline::parser cmd;
        {
            cmd.add<std::string>("output", 'o', "output flakes normal map(png)", false, "result");
            cmd.add<int32_t>("width", 'w', "texture width", false);
            cmd.add<int32_t>("height", 'h', "texture height", false);
            cmd.add("gui", 'g', "GUI mode");

            cmd.add<std::string>("help", '?', "print usage", false);
        }

        bool is_cmd_valid = cmd.parse(argc, argv);

        if (cmd.exist("help")) {
            std::cerr << cmd.usage();
            return false;
        }

        if (!is_cmd_valid) {
            std::cerr << cmd.error() << std::endl << cmd.usage();
            return false;
        }

        if (cmd.exist("output")) {
            args_.output = cmd.get<std::string>("output");
        }
        else {
            args_.output = "result.png";
        }

        if (cmd.exist("height")) {
            args_.width = cmd.get<int32_t>("height");
        }

        if (cmd.exist("width")) {
            args_.width = cmd.get<int32_t>("width");
        }

        args_.need_gui = cmd.exist("gui");
        will_take_screenshot_ = !args_.need_gui;

        return true;
    }

    int32_t width() const
    {
        return args_.width;
    }

    int32_t height() const
    {
        return args_.height;
    }

    bool NeedGui() const
    {
        return args_.need_gui;
    }

    aten::context& GetContext()
    {
        return ctxt_;
    }

private:
    struct Args {
        std::string output;
        int32_t width{ 1280 };
        int32_t height{ 720 };
        bool need_gui{ true };
    } args_;

    aten::context ctxt_;
    std::shared_ptr<aten::texture> bump_map_;

    std::shared_ptr<aten::visualizer> visualizer_;

    aten::FlakesNormalMapMaker maker_;

    bool will_take_screenshot_{ false };
};

int32_t main(int32_t argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    auto app = std::make_shared<FlakesNormalMapMakerApp>();

    if (!app->ParseArgs(argc, argv)) {
        AT_ASSERT(false);
        return 1;
    }

    if (!app->Init()) {
        AT_ASSERT(false);
        return 1;
    }

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        app->width(), app->height(),
        FlakesNormalMapMakerApp::TITLE,
        !app->NeedGui(),
        std::bind(&FlakesNormalMapMakerApp::Run, app),
        std::bind(&FlakesNormalMapMakerApp::OnClose, app));

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
}
