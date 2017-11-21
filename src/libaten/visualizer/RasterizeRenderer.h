#pragma once

#include "defs.h"
#include "types.h"
#include "visualizer/fbo.h"
#include "visualizer/shader.h"
#include "visualizer/GeomDataBuffer.h"

namespace aten {
	class scene;
	class camera;
	class accelerator;

	class ResterizeRenderer {
		static shader s_shader;
		static GeomVertexBuffer s_boxvb;

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

		static void drawAABB(
			shader* shd,
			const camera* cam,
			accelerator* accel);
	};
}