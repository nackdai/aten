#include <map>

#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "FeatureLineSampleRay.h"

static const int WIDTH = 512;
static const int HEIGHT = 512;

static const char* TITLE = "MeasureEffectiveRetroreflectiveArea";

static aten::context g_ctxt;

static FeatureLineSampleRay g_viewer;
static aten::RasterizeRenderer g_rasterizer;

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static float g_moveMultiply = 1.0f;

static bool g_willShowGUI = true;

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

    g_viewer.draw(
        g_ctxt,
        &g_camera);

    if (!g_willShowGUI) {
        return;
    }

    const auto& pos = g_camera.getPos();
    ImGui::Text("%.3f, %.3d, %.3f", pos.x, pos.y, pos.z);

    ImGui::SliderFloat("cam scale", &g_moveMultiply, 1, 100);
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
            real(0.001) * g_moveMultiply);
        g_isCameraDirty = true;
    }

    g_prevX = x;
    g_prevY = y;
}

void onMouseWheel(int delta)
{
    aten::CameraOperator::dolly(g_camera, delta * real(0.1) * g_moveMultiply);
    g_isCameraDirty = true;
}

void onKey(bool press, aten::Key key)
{
    static const real offset_base = real(0.1);

    if (press) {
        if (key == aten::Key::Key_F1) {
            g_willShowGUI = !g_willShowGUI;
            return;
        }
    }

    auto offset = offset_base * g_moveMultiply;

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

int main(int argc, char* argv[])
{
    // TODO
#if 0
    cmdline::parser cmd;

    if (!parseOption(argc, argv, cmd, g_opt)) {
        return 0;
    }
#endif

    aten::window::init(
        WIDTH, HEIGHT,
        TITLE,
        onRun,
        onClose,
        onMouseBtn,
        onMouseMove,
        onMouseWheel,
        onKey);

    // TODO
    aten::vec3 pos(0, 0, 30);
    aten::vec3 at(0, 0, 10);
    real vfov = real(90);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    g_viewer.init(
        WIDTH, HEIGHT,
        "FeatureLineSampleRay_vs.glsl",
        "FeatureLineSampleRay_fs.glsl");

    g_rasterizer.init(
        WIDTH, HEIGHT,
        "../shader/drawobj_vs.glsl",
        "../shader/drawobj_fs.glsl");

    aten::window::run();

    aten::window::terminate();

    return 1;
}
