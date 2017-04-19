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

	bool window::init(int width, int height, const char* title)
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

		::glfwMakeContextCurrent(g_window);
		::glfwSwapInterval(1);

		return true;
	}

	void window::run(std::function<void()> func)
	{
		AT_ASSERT(g_window);

		while (!glfwWindowShouldClose(g_window)) {
			func();

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