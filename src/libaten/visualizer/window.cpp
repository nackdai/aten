#include "visualizer/atengl.h"

#include "visualizer/window.h"
#include <GLFW/glfw3.h>
#include <Shlwapi.h>

#include <imgui.h>
#include "ui/imgui_impl_glfw_gl3.h"

#include <algorithm>

namespace aten
{
	bool window::SetCurrentDirectoryFromExe()
	{
		static char buf[_MAX_PATH];

		// 実行プログラムのフルパスを取得
		{
			DWORD result = ::GetModuleFileName(
				NULL,
				buf,
				sizeof(buf));
			AT_ASSERT(result > 0);
		}

		// ファイル名を取り除く
		auto result = ::PathRemoveFileSpec(buf);
		AT_ASSERT(result);

		// カレントディレクトリを設定
		result = ::SetCurrentDirectory(buf);
		AT_ASSERT(result);

		return result ? true : false;
	}

	static std::vector<window*> g_windows;

	static std::vector<int> g_mouseX;
	static std::vector<int> g_mouseY;

	window* findWindow(GLFWwindow* w)
	{
		auto found = std::find_if(
			g_windows.begin(),
			g_windows.end(),
			[&](window* wnd)
		{
			if (wnd->getNativeHandle() == w) {
				return true;
			}
			return false;
		});

		if (found != g_windows.end()) {
			return *found;
		}

		return nullptr;
	}

	static void closeCallback(GLFWwindow* window)
	{
		auto wnd = findWindow(window);

		if (wnd) {
			wnd->onClose();
			::glfwSetWindowShouldClose(window, GL_TRUE);
		}
	}

	static inline Key getKeyMap(int key)
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

	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		auto wnd = findWindow(window);

		if (wnd) {
			Key k = getKeyMap(key);
			bool press = (action == GLFW_PRESS || action == GLFW_REPEAT);

			wnd->onKey(press, k);
		}

		// For imgui.
		ImGui_ImplGlfwGL3_KeyCallback(window, key, scancode, action, mods);
	}

	static void mouseCallback(GLFWwindow* window, int button, int action, int mods)
	{
		auto wnd = findWindow(window);

		if (wnd) {
			auto mouseX = g_mouseX[wnd->id()];
			auto mouseY = g_mouseY[wnd->id()];

			if (button == GLFW_MOUSE_BUTTON_LEFT) {
				if (action == GLFW_PRESS) {
					wnd->onMouseBtn(true, true, mouseX, mouseY);
				}
				else if (action == GLFW_RELEASE) {
					wnd->onMouseBtn(true, false, mouseX, mouseY);
				}
			}
			else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
				if (action == GLFW_PRESS) {
					wnd->onMouseBtn(false, true, mouseX, mouseY);
				}
				else if (action == GLFW_RELEASE) {
					wnd->onMouseBtn(false, false, mouseX, mouseY);
				}
			}
		}

		// For imgui.
		ImGui_ImplGlfwGL3_MouseButtonCallback(window, button, action, mods);
	}

	static void motionCallback(GLFWwindow* window, double xpos, double ypos)
	{
		auto wnd = findWindow(window);

		g_mouseX[wnd->id()] = (int)xpos;
		g_mouseY[wnd->id()] = (int)ypos;

		if (wnd) {
			wnd->onMouseMove((int)xpos, (int)ypos);
		}
	}

	static void wheelCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		auto offset = (int)(yoffset * 10.0f);

		auto wnd = findWindow(window);

		if (wnd) {
			wnd->onMouseWheel(offset);
		}

		// For imgui.
		ImGui_ImplGlfwGL3_ScrollCallback(window, xoffset, yoffset);
	}

	static void onFocusWindow(GLFWwindow* window, int focused)
	{
		auto wnd = findWindow(window);
		if (wnd) {
			// TODO
			// As current context.
		}
	}

	window::window(GLFWwindow* wnd, int32_t id)
		: m_wnd(wnd), m_id(id)
	{}

	window* window::init(
		int width, int height, const char* title,
		OnRun onRun,
		OnClose _onClose/*= nullptr*/,
		OnMouseBtn _onMouseBtn/*= nullptr*/,
		OnMouseMove _onMouseMove/*= nullptr*/,
		OnMouseWheel _onMouseWheel/*= nullptr*/,
		OnKey _onKey/*= nullptr*/)
	{
		auto result = ::glfwInit();
		AT_VRETURN(result, false);

		// Not specify version.
		// Default value sepcify latest version.
		//::glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		//::glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

		::glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		::glfwWindowHint(GLFW_MAXIMIZED, GL_FALSE);

		if (g_windows.size() >= 1) {
			// ２つめ以降.
			::glfwWindowHint(GLFW_FLOATING, GL_TRUE);
		}

		auto glfwWindow = ::glfwCreateWindow(
			width,
			height,
			title,
			NULL, NULL);

		if (!glfwWindow) {
			::glfwTerminate();
			AT_VRETURN(false, false);
		}

		if (g_windows.size() >= 1) {
			// ２つめ以降.
			auto imguiCtxt = ImGui::CreateContext();
			ImGui::SetCurrentContext(imguiCtxt);
		}

		// For imgui.
		bool succeeded = ImGui_ImplGlfwGL3_Init(glfwWindow);
		AT_ASSERT(succeeded);

		SetCurrentDirectoryFromExe();

		::glfwSetWindowCloseCallback(
			glfwWindow,
			closeCallback);

		::glfwSetKeyCallback(
			glfwWindow,
			keyCallback);

		::glfwSetMouseButtonCallback(
			glfwWindow,
			mouseCallback);

		::glfwSetCursorPosCallback(
			glfwWindow,
			motionCallback);

		::glfwSetScrollCallback(
			glfwWindow,
			wheelCallback);

		::glfwSetWindowFocusCallback(
			glfwWindow, 
			onFocusWindow);

		// For imgui.
		::glfwSetCharCallback(glfwWindow, ImGui_ImplGlfwGL3_CharCallback);

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

		window* ret = new window(glfwWindow, g_windows.size());
		{
			ret->m_onRun = onRun;
			ret->m_onClose = _onClose;
			ret->m_onMouseBtn = _onMouseBtn;
			ret->m_onMouseMove = _onMouseMove;
			ret->m_onMouseWheel = _onMouseWheel;
			ret->m_onKey = _onKey;

			ret->m_imguiCtxt = ImGui::GetCurrentContext();
		}
		g_windows.push_back(ret);

		g_mouseX.resize(g_windows.size());
		g_mouseY.resize(g_windows.size());

		return ret;
	}

	void window::run()
	{
		bool running = true;

		while (running) {
			for (auto wnd : g_windows) {
				auto glfwWnd = wnd->m_wnd;

				::glfwMakeContextCurrent(glfwWnd);

				::glfwPollEvents();

				ImGui::SetCurrentContext((ImGuiContext*)wnd->m_imguiCtxt);

				// For imgui.
				ImGui_ImplGlfwGL3_NewFrame(glfwWnd);

				wnd->m_onRun(wnd);

				::glfwSwapBuffers(glfwWnd);

				if (wnd->id() == 0 
					&& glfwWindowShouldClose(glfwWnd))
				{
					running = false;
				}
			}
		}
	}

	void window::asCurrent()
	{
		::glfwMakeContextCurrent(m_wnd);
	}

	void window::terminate()
	{
		// For imgui.
		ImGui_ImplGlfwGL3_Shutdown();

		ImGuiContext* defaultImguiCtxt = nullptr;

		for (uint32_t i = 0; i < g_windows.size(); i++) {
			auto wnd = g_windows[i];

			::glfwDestroyWindow(wnd->m_wnd);

			if (i == 0) {
				defaultImguiCtxt = (ImGuiContext*)wnd->m_imguiCtxt;
			}
			else {
				ImGui::DestroyContext((ImGuiContext*)wnd->m_imguiCtxt);
			}
			
			delete wnd;
		}

		if (defaultImguiCtxt) {
			ImGui::SetCurrentContext(defaultImguiCtxt);
		}

		::glfwTerminate();

		g_windows.clear();
	}

	void window::drawImGui()
	{
		// Rendering
		int display_w, display_h;
		::glfwGetFramebufferSize(m_wnd, &display_w, &display_h);
		CALL_GL_API(glViewport(0, 0, display_w, display_h));

		ImGui::SetCurrentContext((ImGuiContext*)m_imguiCtxt);
		ImGui::Render();
	}

	bool window::isInitialized()
	{
		return !g_windows.empty();
	}

	GLFWwindow* window::getNativeHandle()
	{
		return m_wnd;
	}
}