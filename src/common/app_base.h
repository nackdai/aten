#include "aten.h"

class App {
public:
    App(int32_t width, int32_t height) : screen_width_(width), screen_height_(height) {}
    ~App() = default;

    App() = delete;
    App(const App&) = delete;
    App(App&&) = delete;
    App& operator=(const App&) = delete;
    App& operator=(App&&) = delete;

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
            aten::CameraOperator::Rotate(
                camera_,
                screen_width_, screen_height_,
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
        aten::CameraOperator::Dolly(camera_, delta * float(0.1));
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
            else if (key == aten::Key::Key_F2)
            {
                will_take_screen_shot_ = true;
                return;
            }
            else if (key == aten::Key::Key_F3)
            {
                enable_frame_step_ = !enable_frame_step_;
                return;
            }
            else if (key == aten::Key::Key_F4)
            {
                enable_anm_update_ = !enable_anm_update_;
                return;
            }
            else if (key == aten::Key::Key_F5)
            {
                aten::GLProfiler::trigger();
                return;
            }
            else if (key == aten::Key::Key_SPACE)
            {
                if (enable_frame_step_)
                {
                    frame_step_ = true;
                    return;
                }
            }
        }

        auto offset = offset_base * move_multiply_scale_;

        if (press)
        {
            switch (key)
            {
            case aten::Key::Key_W:
            case aten::Key::Key_UP:
                aten::CameraOperator::MoveForward(camera_, offset);
                break;
            case aten::Key::Key_S:
            case aten::Key::Key_DOWN:
                aten::CameraOperator::MoveForward(camera_, -offset);
                break;
            case aten::Key::Key_D:
            case aten::Key::Key_RIGHT:
                aten::CameraOperator::MoveRight(camera_, offset);
                break;
            case aten::Key::Key_A:
            case aten::Key::Key_LEFT:
                aten::CameraOperator::MoveRight(camera_, -offset);
                break;
            case aten::Key::Key_Z:
                aten::CameraOperator::MoveUp(camera_, offset);
                break;
            case aten::Key::Key_X:
                aten::CameraOperator::MoveUp(camera_, -offset);
                break;
            default:
                break;
            }

            is_camera_dirty_ = true;
        }
    }

protected:
    int32_t screen_width_{ 0 };
    int32_t screen_height_{ 0 };

    aten::PinholeCamera camera_;
    bool is_camera_dirty_{ false };

    float avg_cuda_time_{ 0.0f };
    float avg_update_time_{ 0.0f };

    bool will_show_gui_{ true };
    bool will_take_screen_shot_{ false };
    int32_t screen_shot_count_{ 0 };

    bool enable_anm_update_{ false };

    int32_t max_samples_{ 1 };
    int32_t max_bounce_{ 5 };
    bool enable_progressive_{ false };
    bool is_show_aabb_{ false };

    float move_multiply_scale_{ 1.0f };

    float distance_limit_ratio_{ 1.0f };

    bool enable_frame_step_{ false };
    bool frame_step_{ false };

    bool is_mouse_l_btn_down_{ false };
    bool is_mouse_r_btn_down_{ false };
    int32_t prev_mouse_pos_x_{ 0 };
    int32_t prev_mouse_pos_y_{ 0 };
};
