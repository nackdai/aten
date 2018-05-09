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
			const aten::GeomMultiVertexBuffer* vb);

		void initWithTriangles(
			aten::SkinningVertex* vertices,
			uint32_t vtxNum,
			aten::PrimitiveParamter* tris,
			uint32_t triNum,
			const aten::GeomMultiVertexBuffer* vb);

		void update(
			const aten::mat4* matrices,
			uint32_t mtxNum);

		void compute(
			int32_t indexOffset,
			aten::vec3& aabbMin,
			aten::vec3& aabbMax);

		bool getComputedResult(
			aten::vec4* pos, 
			aten::vec4* nml,
			uint32_t num);

		void runMinMaxTest();

		std::vector<CudaGLBuffer>& getInteropVBO()
		{
			return m_interopVBO;
		}

		TypedCudaMemory<aten::PrimitiveParamter>& getTriangles()
		{
			return m_triangles;
		}

	private:
		TypedCudaMemory<aten::SkinningVertex> m_vertices;
		TypedCudaMemory<uint32_t> m_indices;
		TypedCudaMemory<aten::mat4> m_matrices;

		TypedCudaMemory<aten::PrimitiveParamter> m_triangles;

		TypedCudaMemory<aten::vec4> m_dstPos;
		TypedCudaMemory<aten::vec4> m_dstNml;

		TypedCudaMemory<aten::vec3> m_minBuf;
		TypedCudaMemory<aten::vec3> m_maxBuf;

		std::vector<CudaGLBuffer> m_interopVBO;
	};
}