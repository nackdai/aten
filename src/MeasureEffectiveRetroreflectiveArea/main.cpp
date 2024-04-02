#include <map>

#include <cmdline.h>
#include <imgui.h>

#include "aten.h"
#include "atenscene.h"
#include "MeasureEffectiveRetroreflectiveArea.h"

constexpr int32_t WIDTH = 512;
constexpr int32_t HEIGHT = 512;
constexpr char* TITLE = "MeasureEffectiveRetroreflectiveArea";

class ERAApp {
public:
    ERAApp() = default;
    ~ERAApp() = default;

    ERAApp(const ERAApp&) = delete;
    ERAApp(ERAApp&&) = delete;
    ERAApp operator=(const ERAApp&) = delete;
    ERAApp operator=(ERAApp&&) = delete;

    bool Init(int32_t argc, char* argv[])
    {
        if (!ParseArguments(argc, argv)) {
            AT_ASSERT(false);
            return false;
        }

        era_.Init();

        ComputeERA();

        return true;
    }

    bool Run()
    {
        if (!NeedGui()) {
            return false;
        }

        if (!era_.IsValid()) {
            aten::vec3 pos(0, 0, 3);
            aten::vec3 at(0, 0, 0);
            float vfov = float(90);

            camera_.init(
                pos,
                at,
                aten::vec3(0, 1, 0),
                vfov,
                WIDTH, HEIGHT);

            era_.InitDraw(
                WIDTH, HEIGHT,
                "MeasureEffectiveRetroreflectiveArea_vs.glsl",
                "MeasureEffectiveRetroreflectiveArea_fs.glsl");

            rasterizer_.init(
                WIDTH, HEIGHT,
                "../shader/drawobj_vs.glsl",
                "../shader/drawobj_fs.glsl");
        }

        if (is_camera_dirty_) {
            camera_.update();

            auto camparam = camera_.param();
            camparam.znear = float(0.1);
            camparam.zfar = float(10000.0);

            is_camera_dirty_ = false;
        }

        era_.draw(
            ctxt_,
            &camera_);

        return true;
    }

    void OnClose()
    {
    }

    void OnMouseBtn(bool left, bool press, int32_t x, int32_t y)
    {
        is_mouse_l_btn_down_ = false;
        is_mouse_r_btn_down_ = false;

        if (press)
        {
            prev_mouse_pos_x_ = x;
            prev_mouse_pos_y_ = y;

            is_mouse_l_btn_down_ = left;
            is_mouse_r_btn_down_ = !left;
        }
    }

    void OnMouseMove(int32_t x, int32_t y)
    {
        if (is_mouse_l_btn_down_)
        {
            aten::CameraOperator::rotate(
                camera_,
                WIDTH, HEIGHT,
                prev_mouse_pos_x_, prev_mouse_pos_y_,
                x, y);
            is_camera_dirty_ = true;
        }
        else if (is_mouse_r_btn_down_)
        {
            aten::CameraOperator::move(
                camera_,
                prev_mouse_pos_x_, prev_mouse_pos_y_,
                x, y,
                float(0.001));
            is_camera_dirty_ = true;
        }

        prev_mouse_pos_x_ = x;
        prev_mouse_pos_y_ = y;
    }

    void OnMouseWheel(int32_t delta)
    {
        aten::CameraOperator::dolly(camera_, delta * float(0.1));
        is_camera_dirty_ = true;
    }

    void OnKey(bool press, aten::Key key)
    {
        static const float offset_base = float(0.1);

        if (press)
        {
            if (key == aten::Key::Key_F1)
            {
                will_show_gui_ = !will_show_gui_;
                return;
            }
        }

        auto offset = offset_base;

        if (press)
        {
            switch (key)
            {
            case aten::Key::Key_W:
            case aten::Key::Key_UP:
                aten::CameraOperator::moveForward(camera_, offset);
                break;
            case aten::Key::Key_S:
            case aten::Key::Key_DOWN:
                aten::CameraOperator::moveForward(camera_, -offset);
                break;
            case aten::Key::Key_D:
            case aten::Key::Key_RIGHT:
                aten::CameraOperator::moveRight(camera_, offset);
                break;
            case aten::Key::Key_A:
            case aten::Key::Key_LEFT:
                aten::CameraOperator::moveRight(camera_, -offset);
                break;
            case aten::Key::Key_Z:
                aten::CameraOperator::moveUp(camera_, offset);
                break;
            case aten::Key::Key_X:
                aten::CameraOperator::moveUp(camera_, -offset);
                break;
            default:
                break;
            }

            is_camera_dirty_ = true;
        }
    }

    bool NeedGui() const
    {
        return args_.need_gui;
    }

    aten::context& GetContext()
    {
        return ctxt_;
    }

private:
    void ComputeERA()
    {
        AT_PRINTF("\n");

        const int32_t Step = args_.step;

        const float ThetaMin = 0.0f;
        const float ThetaMax = AT_MATH_PI_HALF;
        const float ThetaStep = (ThetaMax - ThetaMin) / Step;

        const float PhiMin = -AT_MATH_PI;
        const float PhiMax = AT_MATH_PI;
        const float PhiStep = (PhiMax - PhiMin) / Step;

        std::map<float, aten::vec2> AvgERA;

        for (int32_t phi_cnt = 0; phi_cnt < Step; phi_cnt++)
        {
            const auto phi = PhiMin + PhiStep * phi_cnt;
            for (int32_t theta_cnt = 0; theta_cnt < Step; theta_cnt++)
            {
                const auto theta = ThetaMin + ThetaStep * theta_cnt;

                auto hit_rate = era_.HitTest(theta, phi);
                if (hit_rate > 0) {
                    auto phi_deg = Rad2Deg(phi);
                    auto theta_deg = Rad2Deg(theta);
                    AT_PRINTF("%.3f, %.3f, %.3f\n", phi_deg, theta_deg, hit_rate);

                    auto it = AvgERA.find(theta_deg);
                    if (it == AvgERA.end()) {
                        AvgERA.insert(std::pair<float, aten::vec2>(theta_deg, aten::vec2(hit_rate, 1)));
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
            AT_PRINTF("{%.3ff, %.3ff},\n", theta, hit_rate);
        }
    }

    bool ParseArguments(int32_t argc, char* argv[])
    {
        cmdline::parser cmd;

        {
            cmd.add<int32_t>("step", 's', "Number of steps", false);
            cmd.add("gui", 'g', "GUI mode");

            cmd.add("help", '?', "print usage");
        }

        bool is_cmd_valid = cmd.parse(argc, argv);

        if (cmd.exist("help")) {
            std::cerr << cmd.usage();
            return false;
        }

        if (!is_cmd_valid) {
            std::cerr << cmd.error_full() << std::endl << cmd.usage();
            return false;
        }

        if (cmd.exist("step")) {
            args_.step = cmd.get<int32_t>("step");
        }

        args_.need_gui = cmd.exist("gui");

        return true;
    }

    struct Args {
        int32_t step{40};
        bool need_gui{ false };
    } args_;

    aten::context ctxt_;

    MeasureEffectiveRetroreflectiveArea era_;
    aten::RasterizeRenderer rasterizer_;

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    bool will_show_gui_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};

int32_t main(int32_t argc, char* argv[])
{
    auto app = std::make_shared<ERAApp>();

    if (!app->Init(argc, argv)) {
        AT_ASSERT(false);
        return 1;
    }

    if (app->NeedGui()) {
        auto wnd = std::make_shared<aten::window>();

        auto id = wnd->Create(
            WIDTH, HEIGHT, TITLE,
            !app->NeedGui(),
            std::bind(&ERAApp::Run, app),
            std::bind(&ERAApp::OnClose, app),
            std::bind(&ERAApp::OnMouseBtn, app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
            std::bind(&ERAApp::OnMouseMove, app, std::placeholders::_1, std::placeholders::_2),
            std::bind(&ERAApp::OnMouseWheel, app, std::placeholders::_1),
            std::bind(&ERAApp::OnKey, app, std::placeholders::_1, std::placeholders::_2));

        if (id >= 0) {
            app->GetContext().SetIsWindowInitialized(true);
        }
        else {
            AT_ASSERT(false);
            return 1;
        }

        wnd->Run();

        app.reset();

        wnd->Terminate();
    }
}
