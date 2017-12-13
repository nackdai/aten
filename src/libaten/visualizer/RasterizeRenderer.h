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

	class RasterizeRenderer {
	public:
		RasterizeRenderer() {}
		~RasterizeRenderer() {}

	public:
		bool init(
			int width, int height,
			const char* pathVS,
			const char* pathFS);

		bool init(
			int width, int height,
			const char* pathVS,
			const char* pathGS,
			const char* pathFS);

		void draw(
			int frame,
			scene* scene,
			const camera* cam,
			FBO* fbo = nullptr);

		void drawAABB(
			shader* shd,
			const camera* cam,
			accelerator* accel);

	private:
		shader m_shader;
		GeomVertexBuffer m_boxvb;

		int m_width{ 0 };
		int m_height{ 0 };
	};
}