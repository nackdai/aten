#pragma once

#include "defs.h"
#include "types.h"
#include "visualizer/fbo.h"
#include "visualizer/shader.h"
#include "visualizer/GeomDataBuffer.h"
#include "math/mat4.h"
#include "geometry/object.h"

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
			const camera* cam,
			accelerator* accel);

		void draw(
			object* obj, 
			const camera* cam,
			bool isWireFrame);

		void draw(
			std::vector<vertex>& vtxs,
			std::vector<std::vector<int>>& idxs,
			const camera* cam,
			bool isWireFrame);

	private:
		shader m_shader;
		GeomVertexBuffer m_boxvb;

		mat4 m_mtxPrevW2C;

		int m_width{ 0 };
		int m_height{ 0 };

		GeomVertexBuffer m_vb;
		std::vector<GeomIndexBuffer> m_ib;
	};
}