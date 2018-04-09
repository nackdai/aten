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
			const aten::GeomVertexBuffer& vb);

		void update(
			aten::mat4* matrices,
			uint32_t mtxNum);

		void compute();

		void runMinMaxTest();

	private:
		TypedCudaMemory<aten::SkinningVertex> m_vertices;
		TypedCudaMemory<uint32_t> m_indices;
		TypedCudaMemory<aten::mat4> m_matrices;

		CudaGLBuffer m_interopVBO;
	};
}