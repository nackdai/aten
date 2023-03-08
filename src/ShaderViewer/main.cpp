#include <cmdline.h>
#include <imgui.h>

#include "magnifier.h"

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

static const int32_t WIDTH = 1280;
static const int32_t HEIGHT = 720;

static const char* TITLE = "ShaderViewer";

static aten::context g_ctxt;

static aten::RasterizeRenderer g_rasterizer;

std::vector<std::shared_ptr<aten::PolygonObject>> g_objs;
std::vector<bool> g_objenable;

std::shared_ptr<aten::visualizer> g_visualizer;
std::shared_ptr<aten::FBO> g_fbo;
std::shared_ptr<Magnifier> g_magnifier;

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

void onRun(aten::window* window)
{
    if (g_isCameraDirty) {
        g_camera.update();

        auto camparam = g_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        g_isCameraDirty = false;
    }

    aten::RasterizeRenderer::beginRender(g_fbo);

    aten::RasterizeRenderer::clearBuffer(
        aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
        aten::vec4(0, 0.5f, 1.0f, 1.0f),
        1.0f,
        0);

    int32_t obj_min = 0;
    int32_t obj_max = 20;
    obj_min = std::min<int32_t>(obj_min, static_cast<int32_t>(g_objs.size()));
    obj_max = std::min<int32_t>(obj_max, static_cast<int32_t>(g_objs.size()));

    g_rasterizer.drawWithOutsideRenderFunc(
        g_ctxt,
        [&] (aten::RasterizeRenderer::FuncObjRenderer func) {
            auto& shader = g_rasterizer.getShader();
            shader.setUniformVec3("pointLitPos", aten::vec3(0.0f, 0.0f, 50.0f));
            shader.setUniformVec3("pointLitClr", aten::vec3(0.8f, 0.0f, 0.0f));
            shader.setUniformVec3("pointLitAttr", aten::vec3(0.0f, 0.05f, 0.0f));
            shader.setUniformVec3("cameraPos", g_camera.getPos());

            for (size_t i = obj_min; i < obj_max; i++) {
                auto& obj = g_objs[i];
                func(*obj);
            }
        },
        &g_camera,
        g_isWireFrame);

    if (g_visualizer) {
        aten::Values values{
            {"center_pos", aten::PolymorphicValue(aten::vec4(WIDTH * 0.5f, HEIGHT * 0.5f, 0, 0))},
            {"magnification", aten::PolymorphicValue(0.5f)},
            {"radius", aten::PolymorphicValue(200.0f)},
            {"circle_line_width", aten::PolymorphicValue(2.0f)},
            {"circle_line_color", aten::PolymorphicValue(aten::vec4(1, 0, 0, 0))},
        };
        g_magnifier->setParam(values);

        aten::RasterizeRenderer::beginRender();
        aten::RasterizeRenderer::clearBuffer(
            aten::RasterizeRenderer::Buffer::Color | aten::RasterizeRenderer::Buffer::Depth | aten::RasterizeRenderer::Buffer::Sencil,
            aten::vec4(0, 0.5f, 1.0f, 1.0f),
            1.0f,
            0);
        g_fbo->bindAsTexture();
        g_visualizer->renderGLTexture(g_fbo->getTexHandle(), g_camera.needRevert());
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
        param.baseColor = aten::vec3(0.4, 0.4, 0.4);;

        auto mtrl = g_ctxt.createMaterialWithMaterialParameter(
            param,
            nullptr, nullptr, nullptr);
        aten::AssetManager::registerMtrl("m1", mtrl);
    }

    aten::ObjLoader::load(objs, objpath, g_ctxt, nullptr, true);
}

bool parseOption(
    int32_t argc, char* argv[],
    cmdline::parser& cmd)
{
    {
        cmd.add<std::string>("input", 'i', "input model filename", false);
        cmd.add("full_screen", 'f', "full screen rendering");
    }

    bool isCmdOk = cmd.parse(argc, argv);

    if (!isCmdOk) {
        std::cerr << cmd.error_full() << std::endl << cmd.usage();
        return false;
    }

#if 0
    if (cmd.exist("input")) {
        opt.input = cmd.get<std::string>("input");
    }
    else {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }
#endif

    return true;
}

int32_t main(int32_t argc, char* argv[])
{
    cmdline::parser cmd;

    if (!parseOption(argc, argv, cmd)) {
        return 0;
    }

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

    //loadObj("../../asset/sphere/sphere.obj", nullptr, g_objs);
    //loadObj("../../asset/cube/cube.obj", nullptr, g_objs);
    loadObj("../../asset/teapot/teapot.obj", nullptr, g_objs);

    g_objenable.resize(g_objs.size(), true);

    g_ctxt.initAllTexAsGLTexture();

    for (auto& obj : g_objs) {
        obj->buildForRasterizeRendering(g_ctxt);
    }

    g_rasterizer.init(
        WIDTH, HEIGHT,
        "shader/vs.glsl",
        "shader/retroreflective.glsl");

    if (cmd.exist("full_screen")) {
        g_visualizer = aten::visualizer::init(WIDTH, HEIGHT);

        g_fbo = std::make_shared<decltype(g_fbo)::element_type>();
        g_fbo->init(WIDTH, HEIGHT, aten::PixelFormat::rgba8);

        g_magnifier = std::make_shared<decltype(g_magnifier)::element_type>();
        g_magnifier->init(
            WIDTH, HEIGHT,
            "shader/fullscreen_vs.glsl",
            "shader/magnifier.glsl");
        g_visualizer->addPostProc(g_magnifier.get());
    }

    auto texNum = g_ctxt.getTextureNum();

    for (int32_t i = 0; i < texNum; i++) {
        auto tex = g_ctxt.getTexture(i);
        tex->initAsGLTexture();
    }

    // TODO
    aten::vec3 pos(0.f, 0.0f, 30.0f);
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
