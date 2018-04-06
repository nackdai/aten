#pragma once

#include <vector>

#include "cuda/cudamemory.h"
#include "aten4idaten.h"

namespace idaten
{
	class Skinning {
	public:
		Skinning();
		~Skinning();

	public:
		// TODO
		// aten‘¤‚Å’è‹`‚·‚é.
		struct Vertex {
			aten::vec4 position;
			aten::vec3 normal;
			float uv[2];
			float blendIndex[4];
			float blendWeight[4];
		};

		void init(
			Vertex* vertices,
			uint32_t vtxNum, 
			uint32_t* indices,
			uint32_t idxNum);

		void update(
			aten::mat4* matrices,
			uint32_t mtxNum);

		void compute();

	private:
		TypedCudaMemory<Vertex> m_vertices;
		TypedCudaMemory<uint32_t> m_indices;
		TypedCudaMemory<aten::mat4> m_matrices;
	};
}