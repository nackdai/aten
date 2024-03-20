#include "aten.h"

#include "FlakesNormalMapMaker.h"

#include <cmdline.h>
#include <imgui.h>

int32_t WIDTH = 1280;
int32_t HEIGHT = 720;

static const char* TITLE = "FlakesNormalMapMaker";

struct Options {
    std::string output;
    int32_t width{ 1280 };
    int32_t height{ 720 };
} g_opt;

static std::shared_ptr<aten::visualizer> g_visualizer;

static bool g_willShowGUI = true;

bool onRun()
{
    aten::RasterizeRenderer::clearBuffer(
        aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
        aten::vec4(0, 0.5f, 1.0f, 1.0f),
        1.0f,
        0);

    g_visualizer->render(false);

    if (g_willShowGUI) {
        auto flakeParam = aten::FlakesNormalMapMaker::getParameter();

        auto b0 = ImGui::SliderFloat("scale", &flakeParam.flake_scale, 0.1f, 100.0f);
        auto b1 = ImGui::SliderFloat("size", &flakeParam.flake_size, 0.1f, 1.0f);
        auto b2 = ImGui::SliderFloat("size variance", &flakeParam.flake_size_variance, 0.0f, 1.0f);
        auto b3 = ImGui::SliderFloat("orientation", &flakeParam.flake_normal_orientation, 0.0f, 1.0f);

        if (b0 || b1 || b2 || b3) {
            aten::FlakesNormalMapMaker::setParemter(flakeParam);
        }
    }

    return true;
}

void onClose()
{

}

void onMouseBtn(bool left, bool press, int32_t x, int32_t y)
{
}

void onMouseMove(int32_t x, int32_t y)
{
}

void onMouseWheel(int32_t delta)
{
}

void onKey(bool press, aten::Key key)
{
}

bool parseOption(
    int32_t argc, char* argv[],
    cmdline::parser& cmd,
    Options& opt)
{
    {
        cmd.add<std::string>("output", 'o', "output filename", false, "result");
        cmd.add<std::string>("width", 'w', "texture width", false);
        cmd.add<std::string>("height", 'h', "texture height", false);

        cmd.add<std::string>("help", '?', "print usage", false);
    }

    bool isCmdOk = cmd.parse(argc, argv);

    if (cmd.exist("help")) {
        std::cerr << cmd.usage();
        return false;
    }

    if (!isCmdOk) {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("output")) {
        opt.output = cmd.get<std::string>("output");
    }
    else {
        // TODO
        opt.output = "result.png";
    }

    // TODO

    return true;
}

int32_t main(int32_t argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT, TITLE,
        onRun,
        onClose,
        onMouseBtn,
        onMouseMove,
        onMouseWheel,
        onKey);

    if (id < 0) {
        AT_ASSERT(false);
        return 1;
    }

    g_visualizer = aten::visualizer::init(WIDTH, HEIGHT);

    aten::FlakesNormalMapMaker maker;
    maker.init(
        WIDTH, HEIGHT,
        "../shader/FlakesTextureMaker_vs.glsl",
        "../shader/FlakesTextureMaker_fs.glsl");

    g_visualizer->addPostProc(&maker);

    wnd->Run();

    wnd->Terminate();

    return 0;
}
