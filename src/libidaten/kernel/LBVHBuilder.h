#pragma once

#include <vector>

#include "cuda/cudamemory.h"
#include "accelerator/threaded_bvh.h"
#include "cuda/cudaGLresource.h"

namespace idaten
{
	class CudaTextureResource;

	class LBVHBuilder {
	private:
		LBVHBuilder() {}
		~LBVHBuilder() {}

	public:
		static void build(
			idaten::CudaTextureResource& dst,
			std::vector<aten::PrimitiveParamter>& tris,
			int triIdOffset,
			const aten::aabb& sceneBbox,
			idaten::CudaTextureResource& texRscVtxPos,
			std::vector<aten::ThreadedBvhNode>* threadedBvhNodes = nullptr);

		static void build(
			idaten::CudaTextureResource& dst,
			TypedCudaMemory<aten::PrimitiveParamter> triangles,
			int triIdOffset,
			const aten::aabb& sceneBbox,
			CudaGLBuffer& vboVtxPos,
			std::vector<aten::ThreadedBvhNode>* threadedBvhNodes = nullptr);

		// test implementation.
		static void build();

		struct LBVHNode {
			int order;

			int left;
			int right;
			uint32_t rangeMin;
			uint32_t rangeMax;
			
			int parent;
			bool isLeaf;
		};
	};
}
