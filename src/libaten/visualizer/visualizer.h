#pragma once

#include "defs.h"

namespace aten {
	class visualizer {
	private:
		visualizer() {}
		~visualizer() {}

	public:
		static bool init(
			int width, int height,
			const char* pathVS,
			const char* pathPS);

		static void beginRender();

		static void endRender(const void* pixels);
	};
}