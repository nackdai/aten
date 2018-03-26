#pragma once

#include <vector>

#include "cuda/cudamemory.h"
#include "accelerator/threaded_bvh.h"

namespace idaten
{
	class LBVHBuilder {
	public:
		LBVHBuilder() {}
		~LBVHBuilder() {}

	public:
		// test implementation.
		void sort();

		struct LBVHNode {
			int order;

			int left;
			int right;
			uint32_t rangeMin;
			uint32_t rangeMax;
			
			int parent;
			bool isLeaf;
		};

	private:
		TypedCudaMemory<LBVHNode> m_nodesLbvh;
		TypedCudaMemory<aten::ThreadedBvhNode> m_nodes;
	};
}