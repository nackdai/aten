#include "visualizer/window.h"

namespace aten {
	window window::s_instance;

	bool window::init(int width, int height, const char* title)
	{
		auto result = ::glfwInit();
		AT_VRETURN(result, false);

		::glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
		::glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

		m_window = ::glfwCreateWindow(
			width,
			height,
			title,
			NULL, NULL);

		if (!m_window) {
			::glfwTerminate();
			AT_VRETURN(false, false);
		}

		return true;
	}

	void window::run(std::function<void()> func)
	{
		AT_ASSERT(m_window);

		while (!glfwWindowShouldClose(m_window)) {
			func();

			::glfwSwapBuffers(m_window);
			::glfwPollEvents();
		}
	}

	void window::terminate()
	{
		if (m_window) {
			::glfwDestroyWindow(m_window);
		}
		::glfwTerminate();
	}
}