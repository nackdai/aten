#pragma once

#include <GLFW/glfw3.h>
#include <functional>
#include "defs.h"

namespace aten {
	class window {
		static window s_instance;

	private:
		window() {}
		~window() {}

	public:
		bool init(int width, int height, const char* title);

		void run(std::function<void()> func);

		void terminate();

	private:
		GLFWwindow* m_window{ nullptr };
	};
}