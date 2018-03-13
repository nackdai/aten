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
		static bool s_isInitGlobalVB;

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

		void release()
		{
			m_boxvb.clear();
			m_vb.clear();
			m_ib.clear();
		}

		void draw(
			int frame,
			scene* scene,
			const camera* cam,
			FBO* fbo = nullptr);

		void drawAABB(
			const camera* cam,
			accelerator* accel);

		using FuncSetUniform = std::function<void(shader& shd, const aten::vec3& color, const aten::texture* albedo, int mtrlid)>;

		void draw(
			object* obj, 
			const camera* cam,
			bool isWireFrame,
			FBO* fbo = nullptr,
			FuncSetUniform funcSetUniform = nullptr);

		void initBuffer(
			uint32_t vtxStride,
			uint32_t vtxNum,
			const std::vector<uint32_t>& idxNums);

		void draw(
			const std::vector<vertex>& vtxs,
			const std::vector<std::vector<int>>& idxs,
			const std::vector<material*>& mtrls,
			const camera* cam,
			bool isWireFrame,
			bool updateBuffer);

		static void reset()
		{
			s_isInitGlobalVB = false;
		}
		
	private:
		shader m_shader;
		GeomVertexBuffer m_boxvb;

		mat4 m_mtxPrevW2C;

		int m_width{ 0 };
		int m_height{ 0 };

		bool m_isInitBuffer{ false };
		GeomVertexBuffer m_vb;
		std::vector<GeomIndexBuffer> m_ib;
	};
}