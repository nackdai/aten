#pragma once

#include <vector>

#include "cuda/cudamemory.h"
#include "accelerator/threaded_bvh.h"

namespace idaten
{
	class CudaTextureResource;

	class LBVHBuilder {
	private:
		LBVHBuilder() {}
		~LBVHBuilder() {}

	public:
		static void LBVHBuilder::build(
			idaten::CudaTextureResource& dst,
			std::vector<aten::PrimitiveParamter>& tris,
			int triIdOffset,
			const aten::aabb& sceneBbox,
			idaten::CudaTextureResource& texRscVtxPos,
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