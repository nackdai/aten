#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "lodmaker.h"

static const int WIDTH = 1280;
static const int HEIGHT = 720;

static const char* TITLE = "LodMaker";

struct Options {
    std::string input;
    std::string output;

    std::string inputBasepath;
    std::string inputFilename;
} g_opt;

static aten::RasterizeRenderer g_rasterizer;

static aten::object* g_obj = nullptr;

static aten::context g_ctxt;

static std::vector<std::vector<aten::face*>> g_triangles;
static std::vector<aten::material*> g_mtrls;

static LodMaker g_lodmaker;
static std::vector<aten::vertex> g_lodVtx;
static std::vector<std::vector<int>> g_lodIdx;

static ObjWriter g_writer;

static int g_GridX = 16;
static int g_GridY = 16;
static int g_GridZ = 16;

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static uint32_t g_lodTriCnt = 0;
static bool g_isWireFrame = true;
static bool g_isUpdateBuffer = false;
static bool g_displayLOD = true;

static bool g_willShowGUI = true;

static bool g_isMouseLBtnDown = false;
static bool g_isMouseRBtnDown = false;
static int g_prevX = 0;
static int g_prevY = 0;

#define TEST_LOD

void onRun(aten::window* window)
{
    if (g_isCameraDirty) {
        g_camera.update();

        auto camparam = g_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        g_isCameraDirty = false;
    }

    bool canDisplayLod = g_displayLOD
        && !g_lodmaker.isRunningThread()
        && !g_lodIdx.empty();

    if (canDisplayLod) {
        g_rasterizer.draw(
            g_ctxt,
            g_lodVtx,
            g_lodIdx,
            g_mtrls,
            &g_camera,
            g_isWireFrame,
            g_isUpdateBuffer);

        g_isUpdateBuffer = false;
    }
    else {
        g_rasterizer.drawObject(g_ctxt, *g_obj, &g_camera, g_isWireFrame);
    }

    {
        if (!g_lodmaker.isRunningThread()
            && !g_lodIdx.empty())
        {
            g_lodTriCnt = 0;

            for (auto ids : g_lodIdx) {
                uint32_t cnt = ids.size();
                g_lodTriCnt += cnt / 3;
            }
        }

        uint32_t orgTriCnt = 0;

        for (auto tris : g_triangles) {
            orgTriCnt += tris.size();
        }

        ImGui::Text("Org Polygon [%d]", orgTriCnt);
        ImGui::Text("LOD Polygon [%d]", g_lodTriCnt);
        ImGui::Text("Collapsed [%.3f]%%", g_lodTriCnt / (float)orgTriCnt * 100.0f);

        ImGui::Checkbox("Display LOD", &g_displayLOD);
        ImGui::Checkbox("Wireframe", &g_isWireFrame);

        ImGui::SliderInt("Grid X", &g_GridX, 4, 1024);
        ImGui::SliderInt("Grid Y", &g_GridY, 4, 1024);
        ImGui::SliderInt("Grid Z", &g_GridZ, 4, 1024);

        bool canRunThread = !g_lodmaker.isRunningThread()
            && !g_writer.isRunningThread();

        if (canRunThread) {
            auto& vtxs = g_ctxt.getVertices();

            if (ImGui::Button("Make LOD")) {
                g_lodmaker.runOnThread(
                    [&]() { g_isUpdateBuffer = true;  },
                    g_lodVtx, g_lodIdx,
                    g_obj->getBoundingbox(),
                    vtxs, g_triangles,
                    g_GridX, g_GridY, g_GridZ);
            }
        }
        else {
            if (g_lodmaker.isRunningThread()) {
                ImGui::Text("Making LOD...");
            }
            else {
                ImGui::Text("Writing to obj...");
            }
        }

        if (canRunThread) {
            if (ImGui::Button("Write Obj")) {
                std::string mtrlname = g_opt.inputFilename + ".mtl";

                g_writer.runOnThread(
                    [&]() {},
                    "lod.obj",
                    mtrlname,
                    g_lodVtx, g_lodIdx,
                    g_mtrls);
            }
        }
        else {
            if (g_writer.isRunningThread()) {
                ImGui::Text("Writing to obj...");
            }
            else {
                ImGui::Text("Making LOD...");
            }
        }

        window->drawImGui();
    }
}

void onClose()
{

}

void onMouseBtn(bool left, bool press, int x, int y)
{
    g_isMouseLBtnDown = false;
    g_isMouseRBtnDown = false;

    if (press) {
        g_prevX = x;
        g_prevY = y;

        g_isMouseLBtnDown = left;
        g_isMouseRBtnDown = !left;
    }
}

void onMouseMove(int x, int y)
{
    if (g_isMouseLBtnDown) {
        aten::CameraOperator::rotate(
            g_camera,
            WIDTH, HEIGHT,
            g_prevX, g_prevY,
            x, y);
        g_isCameraDirty = true;
    }
    else if (g_isMouseRBtnDown) {
        aten::CameraOperator::move(
            g_camera,
            g_prevX, g_prevY,
            x, y,
            real(0.001));
        g_isCameraDirty = true;
    }

    g_prevX = x;
    g_prevY = y;
}

void onMouseWheel(int delta)
{
    aten::CameraOperator::dolly(g_camera, delta * real(0.1));
    g_isCameraDirty = true;
}

void onKey(bool press, aten::Key key)
{
    static const real offset = real(0.1);

    if (press) {
        if (key == aten::Key::Key_F1) {
            g_willShowGUI = !g_willShowGUI;
            return;
        }
    }

    if (press) {
        switch (key) {
        case aten::Key::Key_W:
        case aten::Key::Key_UP:
            aten::CameraOperator::moveForward(g_camera, offset);
            break;
        case aten::Key::Key_S:
        case aten::Key::Key_DOWN:
            aten::CameraOperator::moveForward(g_camera, -offset);
            break;
        case aten::Key::Key_D:
        case aten::Key::Key_RIGHT:
            aten::CameraOperator::moveRight(g_camera, offset);
            break;
        case aten::Key::Key_A:
        case aten::Key::Key_LEFT:
            aten::CameraOperator::moveRight(g_camera, -offset);
            break;
        case aten::Key::Key_Z:
            aten::CameraOperator::moveUp(g_camera, offset);
            break;
        case aten::Key::Key_X:
            aten::CameraOperator::moveUp(g_camera, -offset);
            break;
        default:
            break;
        }

        g_isCameraDirty = true;
    }
}

bool parseOption(
    int argc, char* argv[],
    cmdline::parser& cmd,
    Options& opt)
{
    {
        cmd.add<std::string>("input", 'i', "input filename", true);
        cmd.add<std::string>("output", 'o', "output filename base", false, "result");

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

    if (cmd.exist("input")) {
        opt.input = cmd.get<std::string>("input");
    }
    else {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("output")) {
        opt.output = cmd.get<std::string>("output");
    }
    else {
        // TODO
        opt.output = "result.sbvh";
    }

    return true;
}

// TODO
aten::object* loadObj(const Options& opt)
{
    std::vector<aten::object*> objs;

    aten::ObjLoader::load(objs, opt.input, g_ctxt);

    // NOTE
    // ‚P‚Â‚µ‚©‚ä‚é‚³‚È‚¢.
    AT_ASSERT(objs.size() == 1);

    auto obj = objs[0];

    return obj;
}

int main(int argc, char* argv[])
{
    g_opt.input = "../../asset/sponza/sponza.obj";
    //g_opt.input = "../../asset/sponza/lod.obj";
    //g_opt.input = "../../asset/suzanne/suzanne.obj";

    // TODO
#if 0
    cmdline::parser cmd;

    if (!parseOption(argc, argv, cmd, g_opt)) {
        return 0;
    }
#endif

    std::string extname;
    aten::getStringsFromPath(g_opt.input, g_opt.inputBasepath, extname, g_opt.inputFilename);

    aten::SetCurrentDirectoryFromExe();

    aten::AssetManager::suppressWarnings();

    aten::window::init(
        WIDTH, HEIGHT,
        TITLE,
        onRun,
        onClose,
        onMouseBtn,
        onMouseMove,
        onMouseWheel,
        onKey);

    g_obj = loadObj(g_opt);

    g_ctxt.initAllTexAsGLTexture();

    g_obj->buildForRasterizeRendering(g_ctxt);

    {
        auto& vtxs = g_ctxt.getVertices();

        auto stride = sizeof(aten::vertex);
        auto vtxNum = (uint32_t)vtxs.size();

        g_obj->gatherTrianglesAndMaterials(g_triangles, g_mtrls);

        std::vector<uint32_t> idxNums;

        for (int i = 0; i < g_triangles.size(); i++) {
            auto triNum = (uint32_t)g_triangles[i].size();
            idxNums.push_back(triNum * 3);
        }

        g_rasterizer.initBuffer(stride, vtxNum, idxNums);
    }

    // TODO
    aten::vec3 pos(0, 1, 10);
    aten::vec3 at(0, 1, 1);
    real vfov = real(45);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    g_rasterizer.init(
        WIDTH, HEIGHT,
        "../shader/drawobj_vs.glsl",
        "../shader/drawobj_fs.glsl");

    aten::window::run();

    g_lodmaker.terminate();
    g_writer.terminate();

    aten::window::terminate();

    return 1;
}
