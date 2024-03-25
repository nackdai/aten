#include <map>

#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "MeasureEffectiveRetroreflectiveArea.h"

static const int32_t WIDTH = 512;
static const int32_t HEIGHT = 512;

static const char* TITLE = "MeasureEffectiveRetroreflectiveArea";

static aten::context g_ctxt;

static MeasureEffectiveRetroreflectiveArea g_viewer;
static aten::RasterizeRenderer g_rasterizer;

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static bool g_willShowGUI = true;

static int32_t g_drawVoxelDepth = 1;
static bool g_drawMesh = false;
static bool g_isWireframe = false;

static bool g_isMouseLBtnDown = false;
static bool g_isMouseRBtnDown = false;
static int32_t g_prevX = 0;
static int32_t g_prevY = 0;

static std::vector<std::vector<aten::ThreadedSbvhNode>> g_voxels;

bool onRun()
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

    const auto& pos = g_camera.getPos();
    ImGui::Text("%.3f, %.3d, %.3f", pos.x, pos.y, pos.z);

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

void ComputeERA()
{
    AT_PRINTF("\n");

    constexpr int32_t Step = 40;

    const real ThetaMin = 0;
    const real ThetaMax = AT_MATH_PI_HALF;
    const real ThetaStep = (ThetaMax - ThetaMin) / Step;

    const real PhiMin = -AT_MATH_PI;
    const real PhiMax = AT_MATH_PI;
    const real PhiStep = (PhiMax - PhiMin) / Step;

    std::map<real, aten::vec2> AvgERA;

    for (int32_t phi_cnt = 0; phi_cnt < Step; phi_cnt++)
    {
        const auto phi = PhiMin + PhiStep * phi_cnt;
        for (int32_t theta_cnt = 0; theta_cnt < Step; theta_cnt++)
        {
            const auto theta = ThetaMin + ThetaStep * theta_cnt;

            auto hit_rate = g_viewer.HitTest(theta, phi);
            if (hit_rate > 0) {
                auto phi_deg = Rad2Deg(phi);
                auto theta_deg = Rad2Deg(theta);
                AT_PRINTF("%.3f, %.3f, %.3f\n", phi_deg, theta_deg, hit_rate);

                auto it = AvgERA.find(theta_deg);
                if (it == AvgERA.end()) {
                    AvgERA.insert(std::pair<real, aten::vec2>(theta_deg, aten::vec2(hit_rate, 1)));
                }
                else {
                    auto& v = it->second;
                    v.x *= v.y;
                    v.x += hit_rate;
                    v.y++;
                    v.x /= v.y;
                }
            }
        }
    }

    AT_PRINTF("\n\n");
    for (const auto it : AvgERA) {
        const auto theta = it.first;
        const auto hit_rate = it.second.x;
        AT_PRINTF("{real(%.3f), real(%.3f)},\n", theta, hit_rate);
    }
}

int32_t main(int32_t argc, char* argv[])
{
    // TODO
#if 0
    cmdline::parser cmd;

    if (!parseOption(argc, argv, cmd, g_opt)) {
        return 0;
    }
#endif

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

    // TODO
    aten::vec3 pos(0, 0, 3);
    aten::vec3 at(0, 0, 0);
    real vfov = real(90);

    g_camera.init(
        pos,
        at,
        aten::vec3(0, 1, 0),
        vfov,
        WIDTH, HEIGHT);

    g_viewer.init(
        WIDTH, HEIGHT,
        "MeasureEffectiveRetroreflectiveArea_vs.glsl",
        "MeasureEffectiveRetroreflectiveArea_fs.glsl");

    ComputeERA();

    g_rasterizer.init(
        WIDTH, HEIGHT,
        "../shader/drawobj_vs.glsl",
        "../shader/drawobj_fs.glsl");

    wnd->Run();

    wnd->Terminate();

    return 1;
}
