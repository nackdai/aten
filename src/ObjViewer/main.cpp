#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "idaten.h"

static const int WIDTH = 1280;
static const int HEIGHT = 720;

static const char* TITLE = "MdlViewer";

struct Options {
    std::string input;
    std::string texDir;
};

static aten::context g_ctxt;

static aten::RasterizeRenderer g_rasterizer;

aten::object* g_obj = nullptr;

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static bool g_willTakeScreenShot = false;
static int g_cntScreenShot = 0;

static bool g_willShowGUI = true;

static bool g_isMouseLBtnDown = false;
static bool g_isMouseRBtnDown = false;
static int g_prevX = 0;
static int g_prevY = 0;

aten::GeomVertexBuffer m_vb;
aten::GeomIndexBuffer m_ib;

void onRun(aten::window* window)
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
        false);

    g_rasterizer.setColor(aten::vec4(0, 0, 0, 1));
    m_ib.draw(m_vb, aten::Primitive::Lines, 0, 1);

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

// TODO
aten::object* loadObj(
    const char* objpath,
    const char* mtrlpath)
{
    std::string pathname;
    std::string extname;
    std::string filename;

    aten::getStringsFromPath(
        objpath,
        pathname,
        extname,
        filename);

    aten::ImageLoader::setBasePath(pathname);

    if (mtrlpath) {
        aten::MaterialLoader::load(mtrlpath, g_ctxt);
    }

    std::vector<aten::object*> objs;

    aten::ObjLoader::load(objs, objpath, g_ctxt);

    // NOTE
    // ‚P‚Â‚µ‚©‚ä‚é‚³‚È‚¢.
    AT_ASSERT(objs.size() == 1);

    auto obj = objs[0];

    return obj;
}

bool parseOption(
    int argc, char* argv[],
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

int main(int argc, char* argv[])
{
    aten::vec3 nml0;
    {
        aten::vec4 v0(-1.010, 0, 0.990, 0);
        aten::vec4 v1(-0.990, 0, -1.040, 0);
        aten::vec4 v2(-1.020, 1.990, -1.040, 0);

        aten::vec4 e0 = normalize(v1 - v0);
        aten::vec4 e1 = normalize(v2 - v0);

        nml0 = normalize(aten::cross(e0, e1));
    }

    aten::vec3 nml1;
    {
        aten::vec4 v0(-1.010, 0, 0.990, 0);
        aten::vec4 v1(-1.020, 1.990, -1.040, 0);
        aten::vec4 v2(-1.020, 1.990, 0.990, 0);

        aten::vec4 e0 = normalize(v1 - v0);
        aten::vec4 e1 = normalize(v2 - v0);

        nml1 = normalize(aten::cross(e0, e1));
    }

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

    g_obj = loadObj("../../asset/cornellbox/orig.obj", nullptr);

    g_ctxt.initAllTexAsGLTexture();

    g_obj->buildForRasterizeRendering(g_ctxt);

    g_rasterizer.init(
        WIDTH, HEIGHT,
        "drawobj_vs.glsl",
        "drawobj_fs.glsl");

    //aten::ImageLoader::setBasePath(opt.texDir.c_str());

    auto texNum = g_ctxt.getTextureNum();

    for (int i = 0; i < texNum; i++) {
        auto tex = g_ctxt.getTexture(i);
        tex->initAsGLTexture();
    }

    {
        aten::vertex vtxs[] =  {
            {
                {-1.01717770f, 1.43442011f, 0.316223562f, 1.0f},
                {0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f},
            },
            {
                {0.00929778442f * 100, -0.394059151f * 100, -0.919038057f * 100, 1.0f},
                {0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f},
            },
        };
        int idxs[] = { 0, 1 };

        m_vb.init(
            sizeof(aten::vertex),
            AT_COUNTOF(vtxs),
            0,
            &vtxs[0]);
        m_ib.init((uint32_t)AT_COUNTOF(idxs), &idxs[0]);
    }

    // TODO
    aten::vec3 pos(0.f, 1.f, 3.f);
    aten::vec3 at(0.f, 1.f, 0.f);
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
