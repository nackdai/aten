#include "visualizer/window.h"
#include <GLFW/glfw3.h>
#include <Shlwapi.h>

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

	static void closeCallback(GLFWwindow* window)
	{
		if (onClose) {
			onClose();
		}
		::glfwSetWindowShouldClose(window, GL_TRUE);
	}

	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		// TODO
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
	}

	bool window::init(
		int width, int height, const char* title,
		OnClose _onClose/*= nullptr*/,
		OnMouseBtn _onMouseBtn/*= nullptr*/,
		OnMouseMove _onMouseMove/*= nullptr*/,
		OnMouseWheel _onMouseWheel/*= nullptr*/)
	{
		auto result = ::glfwInit();
		AT_VRETURN(result, false);

		::glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
		::glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

		g_window = ::glfwCreateWindow(
			width,
			height,
			title,
			NULL, NULL);

		if (!g_window) {
			::glfwTerminate();
			AT_VRETURN(false, false);
		}

		SetCurrentDirectoryFromExe();

		onClose = _onClose;
		onMouseBtn = _onMouseBtn;
		onMouseMove = _onMouseMove;
		onMouseWheel = _onMouseWheel;

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

		::glfwMakeContextCurrent(g_window);
		::glfwSwapInterval(1);

		return true;
	}

	void window::run(window::OnRun onRun)
	{
		AT_ASSERT(g_window);

		while (!glfwWindowShouldClose(g_window)) {
			onRun();

			::glfwSwapBuffers(g_window);
			::glfwPollEvents();
		}
	}

	void window::terminate()
	{
		if (g_window) {
			::glfwDestroyWindow(g_window);
		}
		::glfwTerminate();
	}
}