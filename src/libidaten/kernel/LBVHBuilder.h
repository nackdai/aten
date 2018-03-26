#pragma once

#include <vector>

#include "cuda/cudamemory.h"
#include "accelerator/threaded_bvh.h"

namespace idaten
{
	class CudaTextureResource;

	class LBVHBuilder {
	public:
		LBVHBuilder() {}
		~LBVHBuilder() {}

	public:
		static void LBVHBuilder::build(
			TypedCudaMemory<aten::ThreadedBvhNode>& nodes,
			const std::vector<aten::PrimitiveParamter>& tris,
			int shapeId,
			int triIdOffset,
			idaten::CudaTextureResource& texRscVtxPos);

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