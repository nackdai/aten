#include "visualizer/atengl.h"

#include "visualizer/window.h"
#include <GLFW/glfw3.h>
#include <Shlwapi.h>

#include <imgui.h>
#include "ui/imgui_impl_glfw_gl3.h"

namespace aten {
	static GLFWwindow* g_window{ nullptr };

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

	static int g_mouseX = 0;
	static int g_mouseY = 0;

	static window::OnClose onClose = nullptr;
	static window::OnMouseBtn onMouseBtn = nullptr;
	static window::OnMouseMove onMouseMove = nullptr;
	static window::OnMouseWheel onMouseWheel = nullptr;
	static window::OnKey onKey = nullptr;

	static void closeCallback(GLFWwindow* window)
	{
		if (onClose) {
			onClose();
		}
		::glfwSetWindowShouldClose(window, GL_TRUE);
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
		if (onKey) {
			Key k = getKeyMap(key);
			bool press = (action == GLFW_PRESS || action == GLFW_REPEAT);

			onKey(press, k);
		}

		// For imgui.
		ImGui_ImplGlfwGL3_KeyCallback(window, key, scancode, action, mods);
	}

	static void mouseCallback(GLFWwindow* window, int button, int action, int mods)
	{
		if (onMouseBtn) {
			if (button == GLFW_MOUSE_BUTTON_LEFT) {
				if (action == GLFW_PRESS) {
					onMouseBtn(true, true, g_mouseX, g_mouseY);
				}
				else if (action == GLFW_RELEASE) {
					onMouseBtn(true, false, g_mouseX, g_mouseY);
				}
			}
			else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
				if (action == GLFW_PRESS) {
					onMouseBtn(false, true, g_mouseX, g_mouseY);
				}
				else if (action == GLFW_RELEASE) {
					onMouseBtn(false, false, g_mouseX, g_mouseY);
				}
			}
		}

		// For imgui.
		ImGui_ImplGlfwGL3_MouseButtonCallback(window, button, action, mods);
	}

	static void motionCallback(GLFWwindow* window, double xpos, double ypos)
	{
		g_mouseX = (int)xpos;
		g_mouseY = (int)ypos;

		if (onMouseMove) {
			onMouseMove(g_mouseX, g_mouseY);
		}
	}

	static void wheelCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		auto offset = (int)(yoffset * 10.0f);

		if (onMouseWheel) {
			onMouseWheel(offset);
		}

		// For imgui.
		ImGui_ImplGlfwGL3_ScrollCallback(window, xoffset, yoffset);
	}

	bool window::init(
		int width, int height, const char* title,
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

		g_window = ::glfwCreateWindow(
			width,
			height,
			title,
			NULL, NULL);

		if (!g_window) {
			::glfwTerminate();
			AT_VRETURN(false, false);
		}

		// For imgui.
		bool succeeded = ImGui_ImplGlfwGL3_Init(g_window);
		AT_ASSERT(succeeded);

		SetCurrentDirectoryFromExe();

		onClose = _onClose;
		onMouseBtn = _onMouseBtn;
		onMouseMove = _onMouseMove;
		onMouseWheel = _onMouseWheel;
		onKey = _onKey;

		::glfwSetWindowCloseCallback(
			g_window,
			closeCallback);

		::glfwSetKeyCallback(
			g_window,
			keyCallback);

		::glfwSetMouseButtonCallback(
			g_window,
			mouseCallback);

		::glfwSetCursorPosCallback(
			g_window,
			motionCallback);

		::glfwSetScrollCallback(
			g_window,
			wheelCallback);

		// For imgui.
		::glfwSetCharCallback(g_window, ImGui_ImplGlfwGL3_CharCallback);

		::glfwMakeContextCurrent(g_window);
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

		return true;
	}

	void window::run(window::OnRun onRun)
	{
		AT_ASSERT(g_window);

		while (!glfwWindowShouldClose(g_window)) {
			::glfwPollEvents();

			// For imgui.
			ImGui_ImplGlfwGL3_NewFrame();

			onRun();

			::glfwSwapBuffers(g_window);
		}
	}

	void window::terminate()
	{
		// For imgui.
		ImGui_ImplGlfwGL3_Shutdown();

		if (g_window) {
			::glfwDestroyWindow(g_window);
		}
		::glfwTerminate();
	}

	void window::drawImGui()
	{
		// Rendering
		int display_w, display_h;
		CALL_GL_API(glfwGetFramebufferSize(g_window, &display_w, &display_h));
		CALL_GL_API(glViewport(0, 0, display_w, display_h));
		//CALL_GL_API(glClearColor(clearClr.x, clearClr.y, clearClr.z, clearClr.w));
		//CALL_GL_API(glClear(GL_COLOR_BUFFER_BIT));
		ImGui::Render();
	}

	bool window::isInitialized()
	{
		return (g_window != nullptr);
	}
}