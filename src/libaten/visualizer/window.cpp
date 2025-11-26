#include <algorithm>
#include <memory>
#include <tuple>

#include "visualizer/atengl.h"
#include "visualizer/window.h"
#include "os/system.h"

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

namespace aten
{
    namespace _detail {
        class WindowImpl {
        public:
            WindowImpl(GLFWwindow* wnd, int32_t id) : wnd_(wnd), id_(id) {}
            ~WindowImpl() = default;

            GLFWwindow* GetNativeHandle()
            {
                return wnd_;
            }

            int32_t id() const
            {
                return id_;
            }

            void OnClose()
            {
                if (msg_handlers_.OnClose) {
                    msg_handlers_.OnClose();
                }
            }

            void OnMouseBtn(bool left, bool press, int32_t x, int32_t y)
            {
                if (msg_handlers_.OnMouseBtn) {
                    msg_handlers_.OnMouseBtn(left, press, x, y);
                }
            }

            void OnMouseMove(int32_t x, int32_t y)
            {
                if (msg_handlers_.OnMouseMove) {
                    msg_handlers_.OnMouseMove(x, y);
                }
            }

            void OnMouseWheel(int32_t delta)
            {
                if (msg_handlers_.OnMouseWheel) {
                    msg_handlers_.OnMouseWheel(delta);
                }
            }

            void OnKey(bool press, aten::Key key)
            {
                if (msg_handlers_.OnKey) {
                    msg_handlers_.OnKey(press, key);
                }
            }

            void OnDropFile(int32_t count, const char** paths)
            {
                // NOTE:
                // Support only 1 file.
                if (msg_handlers_.OnDropFile) {
                    msg_handlers_.OnDropFile(paths[0]);
                }
            }

            auto GetMousePos() const
            {
                return std::tie(mouse_x_, mouse_y_);
            }

            void SetMousePos(int32_t x, int32_t y)
            {
                mouse_x_ = x;
                mouse_y_ = y;
            }

            void DrawImGui()
            {
                AT_ASSERT(wnd_);
                AT_ASSERT(imgui_ctxt_);

                // Rendering
                int32_t display_w, display_h;
                ::glfwGetFramebufferSize(wnd_, &display_w, &display_h);
                CALL_GL_API(glViewport(0, 0, display_w, display_h));

                ImGui::SetCurrentContext(imgui_ctxt_);
                ImGui::Render();
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            }

            void ClearCallbacks()
            {
                // NOTE:
                // If these callbacks are specified with std::bind, these callbacks might handle the instance.
                // For example, if the instance is shared_ptr, these callbacks might keep the ownership of the instances.
                // In that case, the instance might not be released as expected due to that these callbacks keep its ownership.
                // So, in order to release the ownership, set nullptr to the callbacks explicitly.
                msg_handlers_ = window::MesageHandlers();
            }

            GLFWwindow* wnd_{ nullptr };
            int32_t id_{ -1 };

            int32_t mouse_x_{ -1 };
            int32_t mouse_y_{ -1 };

            ImGuiContext* imgui_ctxt_{ nullptr };

            window::MesageHandlers msg_handlers_;
        };
    }

    std::function<void(GLFWwindow*)> window::OnCloseCallback;
    std::function<void(GLFWwindow*, int32_t, int32_t, int32_t, int32_t)> window::OnKeyCallback;
    std::function<void(GLFWwindow*, int32_t, int32_t, int32_t)> window::OnMouseBtnCallback;
    std::function<void(GLFWwindow*, double, double)> window::OnMouseMotionCallback;
    std::function<void(GLFWwindow*, double, double)> window::OnMouseWheelCallback;
    std::function<void(GLFWwindow*, int32_t)> window::OnFocusWindowCallback;
    std::function<void(GLFWwindow*, int, const char**)> window::OnDropFileCallback;

    std::shared_ptr<_detail::WindowImpl> window::FindWindowByNativeHandle(GLFWwindow* w)
    {
        auto found = std::find_if(
            windows_.begin(), windows_.end(),
            [&](const std::shared_ptr<_detail::WindowImpl>& wnd)
        {
            if (wnd->GetNativeHandle() == w) {
                return true;
            }
            return false;
        });

        if (found != windows_.end()) {
            return *found;
        }

        return nullptr;
    }

    std::shared_ptr<_detail::WindowImpl> window::FindWindowById(int32_t id)
    {
        bool is_in_range = (id >= 0) && (id < windows_.size());
        AT_ASSERT(is_in_range);
        return is_in_range ? windows_[id] : nullptr;
    }

    void window::Close(GLFWwindow* window)
    {
        auto wnd = FindWindowByNativeHandle(window);

        if (wnd) {
            wnd->OnClose();
            ::glfwSetWindowShouldClose(window, GL_TRUE);
        }
    }

    namespace _detail {
        inline Key GetKeyMap(int32_t key)
        {
            if ('0' <= key && key <= '9') {
                key = key - '0';
                return (Key)(Key_0 + key);
            }
            else if ('A' <= key && key <= 'Z') {
                key = key - 'A';
                return (Key)(Key_A + key);
            }
            else {
                switch (key) {
                case GLFW_KEY_ESCAPE:
                    return Key_ESCAPE;
                case GLFW_KEY_ENTER:
                    return Key_RETURN;
                case GLFW_KEY_TAB:
                    return Key_TAB;
                case GLFW_KEY_BACKSPACE:
                    return Key_BACKSPACE;
                case GLFW_KEY_INSERT:
                    return Key_INSERT;
                case GLFW_KEY_DELETE:
                    return Key_DELETE;
                case GLFW_KEY_RIGHT:
                    return Key_RIGHT;
                case GLFW_KEY_LEFT:
                    return Key_LEFT;
                case GLFW_KEY_DOWN:
                    return Key_DOWN;
                case GLFW_KEY_UP:
                    return Key_UP;
                case GLFW_KEY_F1:
                    return Key_F1;
                case GLFW_KEY_F2:
                    return Key_F2;
                case GLFW_KEY_F3:
                    return Key_F3;
                case GLFW_KEY_F4:
                    return Key_F4;
                case GLFW_KEY_F5:
                    return Key_F5;
                case GLFW_KEY_F6:
                    return Key_F6;
                case GLFW_KEY_F7:
                    return Key_F7;
                case GLFW_KEY_F8:
                    return Key_F8;
                case GLFW_KEY_F9:
                    return Key_F9;
                case GLFW_KEY_F10:
                    return Key_F10;
                case GLFW_KEY_F11:
                    return Key_F11;
                case GLFW_KEY_F12:
                    return Key_F12;
                case GLFW_KEY_LEFT_SHIFT:
                    return Key_SHIFT;
                case GLFW_KEY_LEFT_CONTROL:
                    return Key_CONTROL;
                case GLFW_KEY_RIGHT_SHIFT:
                    return Key_SHIFT;
                case GLFW_KEY_RIGHT_CONTROL:
                    return Key_CONTROL;
                case GLFW_KEY_SPACE:
                    return Key_SPACE;
                default:
                    break;
                }
            }

            return Key_UNDEFINED;
        }
    }

    void window::Key(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods)
    {
        auto wnd = FindWindowByNativeHandle(window);

        if (wnd) {
            auto k = _detail::GetKeyMap(key);
            bool press = (action == GLFW_PRESS || action == GLFW_REPEAT);

            wnd->OnKey(press, k);
        }

        // For imgui.
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    }

    void window::MouseBtn(GLFWwindow* window, int32_t button, int32_t action, int32_t mods)
    {
        auto wnd = FindWindowByNativeHandle(window);

        if (wnd) {
            int32_t mouseX, mouseY;
            std::tie(mouseX, mouseY) = wnd->GetMousePos();

            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                if (action == GLFW_PRESS) {
                    wnd->OnMouseBtn(true, true, mouseX, mouseY);
                }
                else if (action == GLFW_RELEASE) {
                    wnd->OnMouseBtn(true, false, mouseX, mouseY);
                }
            }
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                if (action == GLFW_PRESS) {
                    wnd->OnMouseBtn(false, true, mouseX, mouseY);
                }
                else if (action == GLFW_RELEASE) {
                    wnd->OnMouseBtn(false, false, mouseX, mouseY);
                }
            }
        }

        // For imgui.
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    }

    void window::MouseMotion(GLFWwindow* window, double xpos, double ypos)
    {
        auto wnd = FindWindowByNativeHandle(window);

        if (wnd) {
            wnd->SetMousePos((int32_t)xpos, (int32_t)ypos);
            wnd->OnMouseMove((int32_t)xpos, (int32_t)ypos);
        }
    }

    void window::MouseWheel(GLFWwindow* window, double xoffset, double yoffset)
    {
        auto wnd = FindWindowByNativeHandle(window);

        if (wnd) {
            auto offset = (int32_t)(yoffset * 10.0f);
            wnd->OnMouseWheel(offset);
        }

        // For imgui.
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    }

    void window::FocusWindow(GLFWwindow* window, int32_t focused)
    {
        auto wnd = FindWindowByNativeHandle(window);
        if (wnd) {
            // TODO
            // As current context.
        }
    }

    void window::DropFile(GLFWwindow* window, int count, const char** paths)
    {
        auto wnd = FindWindowByNativeHandle(window);
        if (wnd) {
            wnd->OnDropFile(count, paths);
        }
    }

    int32_t window::CreateImpl(
        int32_t width, int32_t height, std::string_view title,
        bool is_offscreen,
        const MesageHandlers& msg_handlers)
    {
        auto result = ::glfwInit();
        if (!result) {
            AT_ASSERT(false);
            return -1;
        }

        // Not specify version.
        // Default value sepcify latest version.
        //::glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        //::glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

        ::glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
        ::glfwWindowHint(GLFW_MAXIMIZED, GL_FALSE);

        if (windows_.size() >= 1) {
            // ２つめ以降.
            //::glfwWindowHint(GLFW_FLOATING, GL_TRUE);
        }

        if (is_offscreen) {
            ::glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        }

        auto glfwWindow = ::glfwCreateWindow(
            width,
            height,
            title.data(),
            NULL, NULL);

        if (!glfwWindow) {
            ::glfwTerminate();
            AT_ASSERT(false);
            return -1;
        }

        if (windows_.size() >= 1) {
            // ２つめ以降.
            auto imguiCtxt = ImGui::GetCurrentContext();
            ImGui::SetCurrentContext(imguiCtxt);
        }

        SetCurrentDirectoryFromExe();

        if (!OnCloseCallback) {
            OnCloseCallback = [this](GLFWwindow* window) { this->Close(window); };
        }
        if (!OnKeyCallback) {
            OnKeyCallback = [this](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) { this->Key(window, key, scancode, action, mods); };
        }
        if (!OnMouseBtnCallback) {
            OnMouseBtnCallback = [this](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) { this->MouseBtn(window, button, action, mods); };
        }
        if (!OnMouseMotionCallback) {
            OnMouseMotionCallback = [this](GLFWwindow* window, double xpos, double ypos) { this->MouseMotion(window, xpos, ypos); };
        }
        if (!OnMouseWheelCallback) {
            OnMouseWheelCallback = [this](GLFWwindow* window, double xoffset, double yoffset) { this->MouseWheel(window, xoffset, yoffset); };
        }
        if (!OnFocusWindowCallback) {
            OnFocusWindowCallback = [this](GLFWwindow* window, int32_t focused) { this->FocusWindow(window, focused); };
        }
        if (!OnDropFileCallback) {
            OnDropFileCallback = [this](GLFWwindow* window, int count, const char** paths) { this->DropFile(window, count, paths); };
        }

        ::glfwSetWindowCloseCallback(glfwWindow, &window::OnClose);
        ::glfwSetKeyCallback(glfwWindow, &window::OnKey);
        ::glfwSetMouseButtonCallback(glfwWindow, &window::OnMouseBtn);
        ::glfwSetCursorPosCallback(glfwWindow, &window::OnMouseMotion);
        ::glfwSetScrollCallback(glfwWindow, &window::OnMouseWheel);
        ::glfwSetWindowFocusCallback(glfwWindow, &window::OnFocusWindow);
        ::glfwSetDropCallback(glfwWindow, &window::OnDropFile);

        // For imgui.
        ::glfwSetCharCallback(glfwWindow, ImGui_ImplGlfw_CharCallback);

        ::glfwMakeContextCurrent(glfwWindow);
        ::glfwSwapInterval(1);

        result = glewInit();
        AT_ASSERT(result == GLEW_OK);

        auto version = ::glGetString(GL_VERSION);
        AT_PRINTF("GL Version(%s)\n", version);

        CALL_GL_API(::glClipControl(
            GL_LOWER_LEFT,
            GL_ZERO_TO_ONE));

        CALL_GL_API(::glFrontFace(GL_CCW));

        CALL_GL_API(::glViewport(0, 0, width, height));
        CALL_GL_API(::glDepthRangef(0.0f, 1.0f));

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

        bool succeeded = ImGui_ImplGlfw_InitForOpenGL(glfwWindow, false);
        AT_ASSERT(succeeded);

        succeeded = ImGui_ImplOpenGL3_Init("#version 400");
        AT_ASSERT(succeeded);

        auto wnd = std::make_shared<_detail::WindowImpl>(glfwWindow, static_cast<int32_t>(windows_.size()));
        {
            wnd->msg_handlers_ = msg_handlers;
            wnd->imgui_ctxt_ = ImGui::GetCurrentContext();
        }

        auto id = wnd->id();

        windows_.push_back(std::move(wnd));

        return id;
    }

    void window::Run()
    {
        bool running = true;

        while (running) {
            for (auto& wnd : windows_) {
                auto glfwWnd = wnd->GetNativeHandle();

                ::glfwMakeContextCurrent(glfwWnd);

                ::glfwPollEvents();

                ImGui::SetCurrentContext(wnd->imgui_ctxt_);

                // For imgui.
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                running = wnd->msg_handlers_.OnRun();

                wnd->DrawImGui();

                ::glfwSwapBuffers(glfwWnd);

                if (wnd->id() == 0
                    && glfwWindowShouldClose(glfwWnd))
                {
                    running = false;
                }
            }
        }

        for (auto& wnd : windows_) {
            wnd->ClearCallbacks();
        }
    }

    bool window::SetCurrent(int32_t id)
    {
        auto wnd = FindWindowById(id);
        if (wnd) {
            ::glfwMakeContextCurrent(wnd->GetNativeHandle());
        }
        return wnd != nullptr;
    }

    bool window::EnableVSync(int id, bool enabled)
    {
        auto wnd = FindWindowById(id);
        if (wnd) {
            ::glfwSwapInterval(enabled ? 1 : 0);
        }
        return wnd != nullptr;
    }

    void window::Terminate()
    {
        // For imgui.
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        ImGuiContext* defaultImguiCtxt = nullptr;

        for (uint32_t i = 0; i < windows_.size(); i++) {
            auto wnd = windows_[i];

            ::glfwDestroyWindow(wnd->GetNativeHandle());

            if (i == 0) {
                defaultImguiCtxt = wnd->imgui_ctxt_;
            }
            else {
                ImGui::DestroyContext(wnd->imgui_ctxt_);
            }
        }

        if (defaultImguiCtxt) {
            ImGui::SetCurrentContext(defaultImguiCtxt);
        }

        ::glfwTerminate();

        windows_.clear();
    }

    bool window::IsInitialized() const
    {
        return windows_.size() > 0;
    }
}
