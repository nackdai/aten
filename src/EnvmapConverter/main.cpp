#include "aten.h"
#include "atenscene.h"

#include <cmdline.h>
#include <imgui.h>

#include "envmap.h"

class EnvmapConvereterApp {
public:
    static constexpr const char* TITLE = "EnvmapConverter";

    EnvmapConvereterApp() = default;
    ~EnvmapConvereterApp() = default;

    EnvmapConvereterApp(const EnvmapConvereterApp&) = delete;
    EnvmapConvereterApp(EnvmapConvereterApp&&) = delete;
    EnvmapConvereterApp operator=(const EnvmapConvereterApp&) = delete;
    EnvmapConvereterApp operator=(EnvmapConvereterApp&&) = delete;

    bool Init()
    {
        src_ = EnvMap::LoadEnvmap(
            ctxt_,
            EnvMapType::Equirect,
            "studio015.hdr");

        dst_ = EnvMap::CreateEmptyEnvmap(EnvMapType::CubeMap, 512, 512);

        EnvMap::Convert(src_, dst_);

        dst_->SaveAsPng("cubemap.png");

        return true;
    }

    bool Run()
    {
        return true;
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
            cmd.add<std::string>("input", 'i', "input bump map(png)", true);
            cmd.add<std::string>("output", 'o', "output normal map(png)", false, "result");
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

        args_.input = cmd.get<std::string>("input");

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
        //will_take_screenshot_ = !args_.need_gui;

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
        std::string input;
        std::string output;
        int32_t width{ -1 };
        int32_t height{ -1 };
        bool need_gui{ false };
    } args_;

    aten::context ctxt_;
    std::shared_ptr<EnvMap> src_;
    std::shared_ptr<EnvMap> dst_;

    std::shared_ptr<aten::visualizer> visualizer_;
};

int32_t main(int32_t argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    auto app = std::make_shared<EnvmapConvereterApp>();

    /*if (!app->ParseArgs(argc, argv)) {
        AT_ASSERT(false);
        return 1;
    }*/

    if (!app->Init()) {
        AT_ASSERT(false);
        return 1;
    }

    /*auto wnd = std::make_shared<aten::window>();

    aten::window::MesageHandlers handlers;
    handlers.OnRun = [&app]() { return app->Run(); };
    handlers.OnClose = [&app]() { app->OnClose(); };

    auto id = wnd->Create(
        app->width(), app->height(),
        EnvmapConvereterApp::TITLE,
        !app->NeedGui(),
        handlers);

    if (id >= 0) {
        app->GetContext().SetIsWindowInitialized(true);
    }
    else {
        AT_ASSERT(false);
        return 1;
    }

    wnd->Run();

    app.reset();

    wnd->Terminate();*/

    return 1;
}
