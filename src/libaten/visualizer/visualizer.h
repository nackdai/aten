#pragma once

#include "defs.h"
#include "math/vec3.h"
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
		class PreProc {
		protected:
			PreProc() {}
			virtual ~PreProc() {}

		public:
			virtual void operator()(
				const vec3* src,
				uint32_t width, uint32_t height,
				vec3* dst) = 0;
		};

	public:
		static bool init(int width, int height, PixelFormat fmt);

		static void addPreProc(PreProc* preproc);

		static void setShader(shader* shader);

		static void render(
			const vec3* pixels,
			bool revert);
	};
}