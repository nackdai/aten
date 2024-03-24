#pragma once

#include <functional>
#include <vector>

#include "defs.h"
#include "math/vec4.h"

struct GLFWwindow;

namespace aten {
    enum Key {
        Key_ESCAPE,
        Key_0,
        Key_1,
        Key_2,
        Key_3,
        Key_4,
        Key_5,
        Key_6,
        Key_7,
        Key_8,
        Key_9,
        Key_A,
        Key_B,
        Key_C,
        Key_D,
        Key_E,
        Key_F,
        Key_G,
        Key_H,
        Key_I,
        Key_J,
        Key_K,
        Key_L,
        Key_M,
        Key_N,
        Key_O,
        Key_P,
        Key_Q,
        Key_R,
        Key_S,
        Key_T,
        Key_U,
        Key_V,
        Key_W,
        Key_X,
        Key_Y,
        Key_Z,
        Key_UP,
        Key_LEFT,
        Key_RIGHT,
        Key_DOWN,
        Key_CONTROL,
        Key_SHIFT,
        Key_RETURN,
        Key_SPACE,
        Key_BACK,
        Key_DELETE,

        Key_F1,
        Key_F2,
        Key_F3,
        Key_F4,
        Key_F5,
        Key_F6,
        Key_F7,
        Key_F8,
        Key_F9,
        Key_F10,
        Key_F11,
        Key_F12,

        Key_NUM,

        // I have no plans to support belows...

        Key_MINUS = Key_NUM,
        Key_EQUALS,

        Key_TAB,

        Key_LBRACKET,
        Key_RBRACKET,

        Key_SEMICOLON,
        Key_APOSTROPHE,
        Key_GRAVE,

        Key_BACKSLASH,
        Key_COMMA,
        Key_PERIOD,
        Key_SLASH,
        Key_MULTIPLY,
        Key_LMENU,

        Key_CAPITAL,
        Key_NUMLOCK,
        Key_SCROLL,
        Key_NUMPAD7,
        Key_NUMPAD8,
        Key_NUMPAD9,
        Key_SUBTRACT,
        Key_NUMPAD4,
        Key_NUMPAD5,
        Key_NUMPAD6,
        Key_ADD,
        Key_NUMPAD1,
        Key_NUMPAD2,
        Key_NUMPAD3,
        Key_NUMPAD0,
        Key_DECIMAL,
        Key_OEM_102,
        Key_F13,
        Key_F14,
        Key_F15,
        Key_KANA,
        Key_ABNT_C1,
        Key_CONVERT,
        Key_NOCONVERT,
        Key_YEN,
        Key_ABNT_C2,
        Key_NUMPADEQUALS,
        Key_PREVTRACK,
        Key_AT,
        Key_COLON,
        Key_UNDERLINE,
        Key_KANJI,
        Key_STOP,
        Key_AX,
        Key_UNLABELED,
        Key_NEXTTRACK,
        Key_NUMPADENTER,
        Key_MUTE,
        Key_CALCULATOR,
        Key_PLAYPAUSE,
        Key_MEDIASTOP,
        Key_VOLUMEDOWN,
        Key_VOLUMEUP,
        Key_WEBHOME,
        Key_NUMPADCOMMA,
        Key_DIVIDE,
        Key_SYSRQ,
        Key_RMENU,
        Key_PAUSE,
        Key_HOME,

        Key_PRIOR,

        Key_END,

        Key_NEXT,
        Key_INSERT,

        Key_LWIN,
        Key_RWIN,
        Key_APPS,
        Key_POWER,
        Key_SLEEP,
        Key_WAKE,
        Key_WEBSEARCH,
        Key_WEBFAVORITES,
        Key_WEBREFRESH,
        Key_WEBSTOP,
        Key_WEBFORWARD,
        Key_WEBBACK,
        Key_MYCOMPUTER,
        Key_MAIL,
        Key_MEDIASELECT,
        Key_BACKSPACE,
        Key_NUMPADSTAR,
        Key_LALT,
        Key_CAPSLOCK,
        Key_NUMPADMINUS,
        Key_NUMPADPLUS,
        Key_NUMPADPERIOD,
        Key_NUMPADSLASH,
        Key_RALT,
        Key_PGUP,
        Key_PGDN,

        Key_UNDEFINED,
    };

    namespace _detail {
        class WindowImpl;
    }

    class window {
    public:
        window() {}
        virtual ~window() = default;

        window(const window&) = delete;
        window(window&&) = delete;
        window& operator=(const window&) = delete;
        window& operator=(window&&) = delete;

        using OnRunFunc = std::function<bool()>;
        using OnCloseFunc = std::function<void()>;
        using OnMouseBtnFunc = std::function<void(bool left, bool press, int32_t x, int32_t y)>;
        using OnMouseMoveFunc = std::function<void(int32_t x, int32_t y)>;
        using OnMouseWheelFunc = std::function<void(int32_t delta)>;
        using OnKeyFunc = std::function<void(bool press, Key key)>;

        int32_t Create(
            int32_t width, int32_t height, std::string_view title,
            OnRunFunc onRun,
            OnCloseFunc _onClose = nullptr,
            OnMouseBtnFunc _onMouseBtn = nullptr,
            OnMouseMoveFunc _onMouseMove = nullptr,
            OnMouseWheelFunc _onMouseWheel = nullptr,
            OnKeyFunc _onKey = nullptr);

        void Run();

        bool SetCurrent(int32_t id);

        bool EnableVSync(int32_t id, bool enabled);

        void Terminate();

        bool IsInitialized() const;

    private:
        std::shared_ptr<_detail::WindowImpl> FindWindowByNativeHandle(GLFWwindow* w);
        std::shared_ptr<_detail::WindowImpl> FindWindowById(int32_t id);

        void Close(GLFWwindow* window);
        void Key(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods);
        void MouseBtn(GLFWwindow* window, int32_t button, int32_t action, int32_t mods);
        void MouseMotion(GLFWwindow* window, double xpos, double ypos);
        void MouseWheel(GLFWwindow* window, double xoffset, double yoffset);
        void FocusWindow(GLFWwindow* window, int32_t focused);

        static void OnClose(GLFWwindow* window)
        {
            if (OnCloseCallback) {
                OnCloseCallback(window);
            }
        }
        static void OnKey(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods)
        {
            if (OnKeyCallback) {
                OnKeyCallback(window, key, scancode, action, mods);
            }
        }
        static void OnMouseBtn(GLFWwindow* window, int32_t button, int32_t action, int32_t mods)
        {
            if (OnMouseBtnCallback) {
                OnMouseBtnCallback(window, button, action, mods);
            }
        }
        static void OnMouseMotion(GLFWwindow* window, double xpos, double ypos)
        {
            if (OnMouseMotionCallback) {
                OnMouseMotionCallback(window, xpos, ypos);
            }
        }
        static void OnMouseWheel(GLFWwindow* window, double xoffset, double yoffset)
        {
            if (OnMouseWheelCallback) {
                OnMouseWheelCallback(window, xoffset, yoffset);
            }
        }
        static void OnFocusWindow(GLFWwindow* window, int32_t focused)
        {
            if (OnFocusWindowCallback) {
                OnFocusWindowCallback(window, focused);
            }
        }

        static std::function<void(GLFWwindow*)> OnCloseCallback;
        static std::function<void(GLFWwindow*, int32_t, int32_t, int32_t, int32_t)> OnKeyCallback;
        static std::function<void(GLFWwindow*, int32_t, int32_t, int32_t)> OnMouseBtnCallback;
        static std::function<void(GLFWwindow*, double, double)> OnMouseMotionCallback;
        static std::function<void(GLFWwindow*, double, double)> OnMouseWheelCallback;
        static std::function<void(GLFWwindow*, int32_t)> OnFocusWindowCallback;

        void* imgui_ctxt_{ nullptr };

        std::vector<std::shared_ptr<_detail::WindowImpl>> windows_;
    };
}
