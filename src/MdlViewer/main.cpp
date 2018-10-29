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
    std::string anm;
    std::string mtrl;
    std::string texDir;
};

static idaten::Skinning skinning;

static aten::deformable* g_mdl{ nullptr };
static aten::DeformAnimation g_anm;

static aten::DeformableRenderer g_renderer;

static aten::Timeline g_timeline;

static aten::context g_ctxt;

static aten::RasterizeRenderer g_rasterizerAABB;

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static bool g_willTakeScreenShot = false;
static int g_cntScreenShot = 0;

static bool g_willShowGUI = true;
static bool g_willShowAABB = true;

static bool g_isMouseLBtnDown = false;
static bool g_isMouseRBtnDown = false;
static int g_prevX = 0;
static int g_prevY = 0;

void onRun(aten::window* window)
{
    if (g_isCameraDirty) {
        g_camera.update();

        auto camparam = g_camera.param();
        camparam.znear = real(0.1);
        camparam.zfar = real(10000.0);

        g_isCameraDirty = false;
    }

    g_timeline.advance(1.0f / 60.0f);

    
    g_mdl->update(aten::mat4(), g_timeline.getTime(), &g_anm);
    //g_mdl.update(aten::mat4(), nullptr, 0);

    aten::vec3 aabbMin, aabbMax;

    bool isGPUSkinning = g_mdl->isEnabledForGPUSkinning();

    if (isGPUSkinning) {
        const auto& mtx = g_mdl->getMatrices();
        skinning.update(&mtx[0], mtx.size());
        skinning.compute(aabbMin, aabbMax);
    }

    g_renderer.render(g_ctxt, &g_camera, g_mdl);

    if (g_willTakeScreenShot)
    {
        static char buffer[1024];
        ::sprintf(buffer, "sc_%d.png\0", g_cntScreenShot);

        aten::visualizer::takeScreenshot(buffer, WIDTH, HEIGHT);

        g_willTakeScreenShot = false;
        g_cntScreenShot++;

        AT_PRINTF("Take Screenshot[%s]\n", buffer);
    }

    if (g_willShowAABB) {
        g_rasterizerAABB.drawAABB(
            &g_camera,
            aten::aabb(aabbMin, aabbMax));
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
        else if (key == aten::Key::Key_F3) {
            g_willShowAABB = !g_willShowAABB;
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
    Options& opt)
{
    cmdline::parser cmd;

    {
        cmd.add<std::string>("input", 'i', "input model filename", true);
        cmd.add<std::string>("anm", 'a', "animation filename", false);
        cmd.add<std::string>("mtrl", 'm', "material filename", true);
        cmd.add<std::string>("path", 'p', "texture directory path", true);

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

    if (cmd.exist("anm")) {
        opt.anm = cmd.get<std::string>("anm");
    }

    if (cmd.exist("mtrl")) {
        opt.mtrl = cmd.get<std::string>("mtrl");
    }
    else {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    if (cmd.exist("path")) {
        opt.texDir = cmd.get<std::string>("path");
    }
    else {
        std::cerr << cmd.error() << std::endl << cmd.usage();
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    Options opt;

    if (!parseOption(argc, argv, opt)) {
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

    g_rasterizerAABB.init(
        WIDTH, HEIGHT,
        "../shader/simple3d_vs.glsl",
        "../shader/simple3d_fs.glsl");

    g_mdl = aten::TransformableFactory::createDeformable(g_ctxt);

    g_mdl->read(opt.input.c_str());
    g_anm.read(opt.anm.c_str());

    bool isGPUSkinning = g_mdl->isEnabledForGPUSkinning();

    if (isGPUSkinning) {
        g_renderer.init(
            WIDTH, HEIGHT,
            "drawobj_vs.glsl",
            "drawobj_fs.glsl");
    }
    else {
        g_renderer.init(
            WIDTH, HEIGHT,
            "../shader/skinning_vs.glsl",
            "../shader/skinning_fs.glsl");
    }

    g_mdl->initGLResourcesWithDeformableRenderer(g_renderer);

    g_timeline.init(g_anm.getDesc().time, real(0));
    g_timeline.enableLoop(true);
    g_timeline.start();

    aten::ImageLoader::setBasePath(opt.texDir.c_str());

    if (!aten::MaterialLoader::load(opt.mtrl.c_str(), g_ctxt)) {
        return 0;
    }

    auto texNum = g_ctxt.getTextureNum();

    for (int i = 0; i < texNum; i++) {
        auto tex = g_ctxt.getTexture(i);
        tex->initAsGLTexture();
    }

    // TODO
    aten::vec3 pos(0, 71, 225);
    aten::vec3 at(0, 71, 216);
    real vfov = real(45);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    if (isGPUSkinning) {
        auto& vb = g_mdl->getVBForGPUSkinning();

        std::vector<aten::SkinningVertex> vtx;
        std::vector<uint32_t> idx;
        std::vector<aten::PrimitiveParamter> tris;

        g_mdl->getGeometryData(g_ctxt, vtx, idx, tris);

        skinning.initWithTriangles(
            &vtx[0], vtx.size(),
            &tris[0], tris.size(),
            &vb);
    }

    aten::window::run();

    g_mdl->release();
    delete g_mdl;

    aten::window::terminate();

    return 1;
}
