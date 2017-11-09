#pragma once

#include "defs.h"
#include "types.h"
#include "visualizer/fbo.h"
#include "visualizer/shader.h"

namespace aten {
	class scene;
	class camera;

	class ResterizeRenderer {
		static shader s_shader;

	private:
		ResterizeRenderer() {}
		~ResterizeRenderer() {}

	public:
		static bool init(
			int width, int height,
			const char* pathVS,
			const char* pathFS);

		static bool init(
			int width, int height,
			const char* pathVS,
			const char* pathGS,
			const char* pathFS);

		static void draw(
			int frame,
			scene* scene,
			const camera* cam,
			FBO* fbo = nullptr);
	};
}