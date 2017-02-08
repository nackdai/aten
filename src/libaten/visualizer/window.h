#pragma once

#include <functional>
#include "defs.h"

namespace aten {
	class window {
	private:
		window() {}
		~window() {}

	public:
		static bool init(int width, int height, const char* title);

		static void run(std::function<void()> func);

		static void terminate();
	};
}