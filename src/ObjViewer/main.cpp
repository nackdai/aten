#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

static const int32_t WIDTH = 1280;
static const int32_t HEIGHT = 720;

static const char* TITLE = "ObjViewer";

struct Options {
    std::string input;
    std::string texDir;
};

static aten::context g_ctxt;

static aten::RasterizeRenderer g_rasterizer;

std::vector<std::shared_ptr<aten::PolygonObject>> g_objs;
std::vector<bool> g_objenable;
aten::aabb g_obj_aabb;

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static bool g_willTakeScreenShot = false;
static int32_t g_cntScreenShot = 0;

static bool g_willShowGUI = true;

static bool g_isWireFrame = false;

static bool g_isMouseLBtnDown = false;
static bool g_isMouseRBtnDown = false;
static int32_t g_prevX = 0;
static int32_t g_prevY = 0;

static int32_t obj_min = 0;
static int32_t obj_max = 20;

void onRun(aten::window* window)
{
    if (g_isCameraDirty) {
        g_camera.update();

        auto camparam = g_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        g_isCameraDirty = false;
    }

    aten::RasterizeRenderer::beginRender();

    aten::RasterizeRenderer::clearBuffer(
        aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
        aten::vec4(0, 0.5f, 1.0f, 1.0f),
        1.0f,
        0);

    obj_min = std::min<int32_t>(obj_min, static_cast<int32_t>(g_objs.size()));
    obj_max = std::min<int32_t>(obj_max, static_cast<int32_t>(g_objs.size()));

    g_rasterizer.drawWithOutsideRenderFunc(
        g_ctxt,
        [&] (aten::RasterizeRenderer::FuncObjRenderer func) {
            for (size_t i = obj_min; i < obj_max; i++) {
                auto is_enable = g_objenable[i];

                if (is_enable) {
                    auto& obj = g_objs[i];
                    func(*obj);
                }
            }
        },
        &g_camera,
        g_isWireFrame);

    if (ImGui::Button("Export")) {
        decltype(g_objs) export_objs;
        for (size_t i = obj_min; i < obj_max; i++) {
            auto is_enable = g_objenable[i];

            if (is_enable) {
                export_objs.push_back(g_objs[i]);
            }
        }
        aten::ObjWriter::writeObjects(
            "sponza.obj",
            "sponza.mtl",
            g_ctxt,
            export_objs);
    }

    const auto& aabb_max = g_obj_aabb.maxPos();
    const auto& aabb_min = g_obj_aabb.minPos();
    ImGui::Text("max(%.3f, %.3f, %.3f)", aabb_max.x, aabb_max.y, aabb_max.z);
    ImGui::Text("min(%.3f, %.3f, %.3f)", aabb_min.x, aabb_min.y, aabb_min.z);

    ImGui::SliderInt("min", &obj_min, 0, obj_max);
    ImGui::SliderInt("max", &obj_max, obj_min, static_cast<int32_t>(g_objs.size() - 1));

    if (!g_objs.empty()) {
        if (ImGui::Button("all_enable")) {
            for (auto& is_enable : g_objenable) {
                is_enable = true;
            }
        }

        if (ImGui::Button("all_disable")) {
            for (auto& is_enable : g_objenable) {
                is_enable = false;
            }
        }
    }

    for (size_t i = obj_min; i < obj_max; i++) {
        const auto& obj = g_objs[i];
        auto mtrl_name = obj->getShapes()[0]->getMaterial()->name();

        const auto name_ptr = obj->getName();
        std::string name;
        if (name_ptr != nullptr) {
            name = name_ptr;
        }
        name += mtrl_name;

        bool is_enable = g_objenable.at(i);
        if (ImGui::Checkbox(name.c_str(), &is_enable)) {
            g_objenable[i] = is_enable;
        }
    }

    if (g_willTakeScreenShot)
    {
        static char buffer[1024];
        ::sprintf(buffer, "sc_%d.png\0", g_cntScreenShot);

        aten::visualizer::takeScreenshot(buffer, WIDTH, HEIGHT);

        g_willTakeScreenShot = false;
        g_cntScreenShot++;

        AT_PRINTF("Take Screenshot[%s]\n", buffer);
    }
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
        else if (key == aten::Key::Key_F3) {
            g_isWireFrame = !g_isWireFrame;
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

void loadObj(
    const char* objpath,
    const char* mtrlpath,
    std::vector<std::shared_ptr<aten::PolygonObject>>& objs)
{
    std::string pathname;
    std::string extname;
    std::string filename;

    aten::getStringsFromPath(
        objpath,
        pathname,
        extname,
        filename);

    if (mtrlpath) {
        aten::MaterialLoader::load(mtrlpath, g_ctxt);
    }
    else {
        aten::MaterialParameter param;
        param.type = aten::MaterialType::Lambert;
        param.baseColor = aten::vec3(1, 1, 1);;

        auto mtrl = g_ctxt.createMaterialWithMaterialParameter(
            param,
            nullptr, nullptr, nullptr);
        aten::AssetManager::registerMtrl("dummy", mtrl);
    }

    aten::ObjLoader::load(objs, objpath, g_ctxt, nullptr, true);

    g_obj_aabb.empty();
    for (const auto& obj : objs) {
        const auto& aabb = obj->getBoundingbox();
        g_obj_aabb.expand(aabb);
    }
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

    aten::window::init(
        WIDTH, HEIGHT,
        TITLE,
        onRun,
        onClose,
        onMouseBtn,
        onMouseMove,
        onMouseWheel,
        onKey);

    //loadObj("../../asset/models/sponza/sponza.obj", nullptr, g_objs);
    loadObj("../../asset/bunny/bunny.obj", nullptr, g_objs);
    //loadObj("../../asset/dragon/dragon.obj", nullptr, g_objs);
    //loadObj("../../asset/models/plane/plane.obj", nullptr, g_objs);
    //loadObj("../../asset/cornellbox/orig.obj", nullptr, g_objs);

    g_objenable.resize(g_objs.size(), true);

    g_ctxt.initAllTexAsGLTexture();

    for (auto& obj : g_objs) {
        obj->buildForRasterizeRendering(g_ctxt);
    }

    g_rasterizer.init(
        WIDTH, HEIGHT,
        "../shader/drawobj_vs.glsl",
        "../shader/drawobj_fs.glsl");

    auto texNum = g_ctxt.getTextureNum();

    for (int32_t i = 0; i < texNum; i++) {
        auto tex = g_ctxt.getTexture(i);
        tex->initAsGLTexture();
    }

    // TODO
    aten::vec3 pos(0.f, 100.0f, 300.0f);
    aten::vec3 at(0.f, 0.f, 0.f);
    real vfov = real(45);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    aten::window::run();

    aten::window::terminate();

    return 1;
}
