#include <memory>

#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

static const int32_t WIDTH = 1280;
static const int32_t HEIGHT = 720;

static const char* TITLE = "AssimpLoadViewer";

struct Options {
    std::string input;
    std::string texDir;
};

static aten::context g_ctxt;

static aten::RasterizeRenderer g_rasterizer;

static std::shared_ptr<aten::PolygonObject> g_obj;

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static bool g_willTakeScreenShot = false;
static int32_t g_cntScreenShot = 0;

static bool g_willShowGUI = true;
static bool g_isWireframe = false;

static bool g_isMouseLBtnDown = false;
static bool g_isMouseRBtnDown = false;
static int32_t g_prevX = 0;
static int32_t g_prevY = 0;

bool onRun()
{
    if (g_isCameraDirty) {
        g_camera.update();

        auto camparam = g_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        g_isCameraDirty = false;
    }

    g_rasterizer.drawObject(
        g_ctxt,
        *g_obj,
        &g_camera,
        g_isWireframe);

    if (g_willTakeScreenShot)
    {
        static char buffer[1024];
        ::sprintf(buffer, "sc_%d.png\0", g_cntScreenShot);

        aten::visualizer::takeScreenshot(buffer, WIDTH, HEIGHT);

        g_willTakeScreenShot = false;
        g_cntScreenShot++;

        AT_PRINTF("Take Screenshot[%s]\n", buffer);
    }

    if (g_willShowGUI) {
        ImGui::Checkbox("Wireframe,", &g_isWireframe);
    }

    return true;
}

void onClose()
{

}

void onMouseBtn(bool left, bool press, int32_t x, int32_t y)
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

void onMouseMove(int32_t x, int32_t y)
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

void onMouseWheel(int32_t delta)
{
    aten::CameraOperator::dolly(g_camera, delta * real(0.1));
    g_isCameraDirty = true;
}

void onKey(bool press, aten::Key key)
{
    static const real offset = real(5);

    if (press) {
        if (key == aten::Key::Key_F1) {
            g_willShowGUI = !g_willShowGUI;
            return;
        }
        else if (key == aten::Key::Key_F2) {
            g_willTakeScreenShot = true;
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

std::shared_ptr<aten::PolygonObject> loadObj(
    std::string_view objpath,
    std::string_view mtrlpath,
    aten::AssetManager& asset_manager)
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

        auto mtrl = g_ctxt.CreateMaterialWithMaterialParameter(
            param,
            nullptr, nullptr, nullptr);
        asset_manager.registerMtrl("dummy", mtrl);
    }
    else {
        aten::MaterialLoader::load(mtrlpath, g_ctxt, asset_manager);
    }

    std::vector<std::shared_ptr<aten::PolygonObject>> objs;
    aten::AssimpImporter::load(
        "../../asset/simpleScene/box.fbx",
        objs,
        g_ctxt, asset_manager,
        [&](const std::string& name,
            aten::context& ctxt,
            const aten::MaterialParameter& mtrl_param,
            const std::string& albedo,
            const std::string& nml)
        {
            auto mtrl = asset_manager.getMtrl(name);
            if (!mtrl) {
                auto albedo_map = albedo.empty()
                    ? nullptr
                    : aten::ImageLoader::load(pathname + albedo, ctxt, asset_manager);
                auto nml_map = nml.empty()
                    ? nullptr
                    : aten::ImageLoader::load(pathname + nml, ctxt, asset_manager);

                mtrl = ctxt.CreateMaterialWithMaterialParameter(
                    mtrl_param,
                    albedo_map.get(),
                    nml_map.get(),
                    nullptr);
                mtrl->setName(name.c_str());
            }

            return mtrl;
        });

    // NOTE
    // １つしかゆるさない.
    AT_ASSERT(objs.size() == 1);

    return objs[0];
}

bool parseOption(
    int32_t argc, char* argv[],
    Options& opt)
{
    cmdline::parser cmd;

    {
        cmd.add<std::string>("input", 'i', "input model filename", true);

        cmd.add("help", '?', "print usage");
    }

    bool isCmdOk = cmd.parse(argc, argv);

    if (cmd.exist("help")) {
        std::cerr << cmd.usage();
        return false;
    }

    if (!isCmdOk) {
        std::cerr << cmd.error_full() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("input")) {
        opt.input = cmd.get<std::string>("input");
    }
    else {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    return true;
}

int32_t main(int32_t argc, char* argv[])
{
    Options opt;

#if 0
    if (!parseOption(argc, argv, opt)) {
        return 0;
    }
#endif

    aten::SetCurrentDirectoryFromExe();

    auto wnd = std::make_shared<aten::window>();

    auto id = wnd->Create(
        WIDTH, HEIGHT,
        TITLE,
        onRun,
        onClose,
        onMouseBtn,
        onMouseMove,
        onMouseWheel,
        onKey);

    if (id >= 0) {
        g_ctxt.SetIsWindowInitialized(true);
    }
    else {
        AT_ASSERT(false);
        return 1;
    }

    aten::AssetManager asset_manager;

    g_obj = loadObj("box.fbx", "", asset_manager);

    g_ctxt.InitAllTextureAsGLTexture();

    g_obj->buildForRasterizeRendering(g_ctxt);

    g_rasterizer.init(
        WIDTH, HEIGHT,
        "../shader/drawobj_vs.glsl",
        "../shader/drawobj_fs.glsl");

    auto texNum = g_ctxt.GetTextureNum();

    for (int32_t i = 0; i < texNum; i++) {
        auto tex = g_ctxt.GtTexture(i);
        tex->initAsGLTexture();
    }

    // TODO
    aten::vec3 pos(0.f, 10.0f, 30.0f);
    aten::vec3 at(0.f, 0.f, 0.f);
    real vfov = real(45);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    wnd->Run();

    wnd->Terminate();

    return 1;
}
