#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "VoxelViewer.h"

static const int WIDTH = 1280;
static const int HEIGHT = 720;

static const char* TITLE = "VoxelViewer";

struct Options {
    std::string input;
    std::string output;

    std::string inputBasepath;
    std::string inputFilename;
} g_opt;

static std::vector<aten::object*> g_objs;
static aten::AcceleratedScene<aten::sbvh> g_scene;
static aten::context g_ctxt;

static VoxelViewer g_viewer;
static aten::RasterizeRenderer g_rasterizer;

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static bool g_willShowGUI = true;

static int g_drawVoxelDepth = 1;
static bool g_drawMesh = false;
static bool g_isWireframe = false;

static bool g_isMouseLBtnDown = false;
static bool g_isMouseRBtnDown = false;
static int g_prevX = 0;
static int g_prevY = 0;

static std::vector<std::vector<aten::ThreadedSbvhNode>> g_voxels;

void onRun(aten::window* window)
{
    if (g_isCameraDirty) {
        g_camera.update();

        auto camparam = g_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        g_isCameraDirty = false;
    }

    // TODO
    auto sbvh = (aten::sbvh*)g_objs[0]->getInternalAccelerator();

    if (g_voxels.empty()) {
        auto maxDepth = sbvh->getMaxDepth();
        g_voxels.resize(maxDepth);

        // NOTE
        // nodes[0] is top layer.
        const auto& nodes = g_scene.getAccel()->getNodes();
        g_viewer.bringVoxels(nodes[1], g_voxels);
    }

    g_viewer.draw(
        g_ctxt,
        &g_camera,
        g_voxels,
        g_isWireframe,
        g_drawVoxelDepth);

    if (g_drawMesh) {
        for (auto obj : g_objs) {
            g_rasterizer.drawObject(g_ctxt, *obj, &g_camera, false);
        }
    }

    {
        ImGui::SliderInt("Depth", &g_drawVoxelDepth, 1, sbvh->getMaxDepth());
        ImGui::Checkbox("Wireframe,", &g_isWireframe);
        ImGui::Checkbox("Draw mesh,", &g_drawMesh);

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
void loadObj(const Options& opt)
{
#if 0
    auto emit = new aten::emissive(aten::vec3(36, 33, 24));
    aten::AssetManager::registerMtrl(
        "light",
        emit);

    aten::ObjLoader::load(g_objs, opt.input);
#elif 1
    aten::MaterialParameter mtrlParam;
    mtrlParam.baseColor = aten::vec3(0.7, 0.7, 0.7);
    mtrlParam.ior = 0.2;
    mtrlParam.roughness = 0.2;

    auto mtrl = g_ctxt.createMaterialWithMaterialParameter(
        aten::MaterialType::GGX,
        mtrlParam,
        nullptr, nullptr, nullptr);

    aten::AssetManager::registerMtrl("m", mtrl);

    aten::ObjLoader::load(g_objs, opt.input, g_ctxt);
#else
    aten::ObjLoader::load(g_objs, "../../asset/sponza/sponza.obj");

    g_objs[0]->importInternalAccelTree("../../asset/sponza/sponza.sbvh");
#endif

    for (auto obj : g_objs) {
        auto instance = aten::TransformableFactory::createInstance<aten::object>(g_ctxt, obj, aten::mat4::Identity);
        g_scene.add(instance);
    }

    g_scene.build(g_ctxt);
}

int main(int argc, char* argv[])
{
    //g_opt.input = "../../asset/cornellbox/orig.obj";
    //g_opt.input = "../../asset/sponza/lod.obj";
    //g_opt.input = "../../asset/suzanne/suzanne.obj";
    g_opt.input = "../../asset/bunny/bunny.obj";

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

    loadObj(g_opt);

    g_ctxt.initAllTexAsGLTexture();

    // TODO
    aten::vec3 pos(0, 1, 3);
    aten::vec3 at(0, 1, 0);
    real vfov = real(45);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    g_viewer.init(
        WIDTH, HEIGHT,
        "voxelviewer_vs.glsl",
        "voxelviewer_fs.glsl");

    g_rasterizer.init(
        WIDTH, HEIGHT,
        "../shader/drawobj_vs.glsl",
        "../shader/drawobj_fs.glsl");

    aten::window::run();

    aten::window::terminate();

    return 1;
}
