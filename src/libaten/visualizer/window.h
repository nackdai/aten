#pragma once

#include <functional>
#include "defs.h"

namespace aten {
	class window {
	private:
		window() {}
		~window() {}

	public:
		using OnRun = std::function<void()>;
		using OnClose = std::function<void()>;
		using OnMouseBtn = std::function<void(bool left, bool press, int x, int y)>;
		using OnMouseMove = std::function<void(int x, int y)>;
		using OnMouseWheel = std::function<void(int delta)>;

		static bool init(
			int width, int height, const char* title,
			OnClose _onClose = nullptr,
			OnMouseBtn _onMouseBtn = nullptr,
			OnMouseMove _onMouseMove = nullptr,
			OnMouseWheel _onMouseWheel = nullptr);

		static void run(OnRun onRim);

		static void terminate();

		static bool SetCurrentDirectoryFromExe();
	};
}