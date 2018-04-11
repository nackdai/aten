#pragma once

#include <vector>

#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"
#include "aten4idaten.h"

namespace idaten
{
	class Skinning {
	public:
		Skinning() {}
		~Skinning() {}

	public:
		void init(
			aten::SkinningVertex* vertices,
			uint32_t vtxNum, 
			uint32_t* indices,
			uint32_t idxNum,
			const aten::GeomVertexBuffer* vb);

		void initWithTriangles(
			aten::SkinningVertex* vertices,
			uint32_t vtxNum,
			aten::PrimitiveParamter* tris,
			uint32_t triNum,
			const aten::GeomVertexBuffer* vb);

		void update(
			const aten::mat4* matrices,
			uint32_t mtxNum);

		void compute();

		bool getComputedResult(aten::vertex* p, uint32_t num);

		void runMinMaxTest();

	private:
		TypedCudaMemory<aten::SkinningVertex> m_vertices;
		TypedCudaMemory<uint32_t> m_indices;
		TypedCudaMemory<aten::mat4> m_matrices;

		TypedCudaMemory<aten::PrimitiveParamter> m_triangles;

		TypedCudaMemory<aten::vertex> m_dst;

		CudaGLBuffer m_interopVBO;
	};
}