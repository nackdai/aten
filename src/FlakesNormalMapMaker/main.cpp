#include "aten.h"

#include "FlakesNormalMapMaker.h"

#include <cmdline.h>
#include <imgui.h>

int WIDTH = 1280;
int HEIGHT = 720;

static const char* TITLE = "FlakesNormalMapMaker";

struct Options {
    std::string output;
    int width{ 1280 };
    int height{ 720 };
} g_opt;

static aten::visualizer* g_visualizer;

static bool g_willShowGUI = true;

void onRun(aten::window* window)
{
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

        window->drawImGui();
    }
}

void onClose()
{

}

void onMouseBtn(bool left, bool press, int x, int y)
{
}

void onMouseMove(int x, int y)
{
}

void onMouseWheel(int delta)
{
}

void onKey(bool press, aten::Key key)
{
}

bool parseOption(
    int argc, char* argv[],
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

int main(int argc, char* argv[])
{
    aten::SetCurrentDirectoryFromExe();

    aten::window::init(
        WIDTH, HEIGHT,
        TITLE,
        onRun,
        onClose,
        onMouseBtn,
        onMouseMove,
        onMouseWheel,
        onKey);

    g_visualizer = aten::visualizer::init(WIDTH, HEIGHT);

    aten::FlakesNormalMapMaker maker;
    maker.init(
        WIDTH, HEIGHT,
        "../shader/FlakesTextureMaker_vs.glsl",
        "../shader/FlakesTextureMaker_fs.glsl");

    g_visualizer->addPostProc(&maker);

    aten::window::run();

    aten::window::terminate();

    return 1;
}
