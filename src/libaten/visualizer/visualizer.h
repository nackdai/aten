#pragma once

#include "defs.h"
#include "visualizer/shader.h"

namespace aten {
	enum PixelFormat {
		rgba8,
		rgba32f
	};

	class visualizer {
	private:
		visualizer() {}
		~visualizer() {}

	public:
		static bool init(int width, int height, PixelFormat fmt);

		static void setShader(shader* shader);

		static void render(const void* pixels);
	};
}