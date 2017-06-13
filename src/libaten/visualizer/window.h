#pragma once

#include <functional>
#include "defs.h"

namespace aten {
	class window {
	private:
		window() {}
		~window() {}

	public:
		using CallbackRun = std::function<void()>;
		using CallbackClose = std::function<void()>;
		using CallbackMouseBtn = std::function<void(bool left, bool down, int x, int y)>;
		using CallbackMouseMove = std::function<void(int x, int y)>;
		using CallbackMouseWheel = std::function<void(int offset)>;

		static bool init(
			int width, int height, const char* title,
			CallbackClose funcClose = nullptr,
			CallbackMouseBtn funcMouseBtn = nullptr,
			CallbackMouseMove funcMouseMove = nullptr,
			CallbackMouseWheel funcMouseWheel = nullptr);

		static void run(CallbackRun func);

		static void terminate();

		static bool SetCurrentDirectoryFromExe();
	};
}