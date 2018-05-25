#pragma once

#include <vector>

#include "cuda/cudamemory.h"
#include "accelerator/threaded_bvh.h"
#include "cuda/cudaGLresource.h"
#include "kernel/RadixSort.h"

namespace idaten
{
	class CudaTextureResource;
	
	class LBVHBuilder {
	public:
		LBVHBuilder() {}
		~LBVHBuilder() {}

	public:
		void build(
			idaten::CudaTextureResource& dst,
			std::vector<aten::PrimitiveParamter>& tris,
			int triIdOffset,
			const aten::aabb& sceneBbox,
			idaten::CudaTextureResource& texRscVtxPos,
			int vtxOffset,
			std::vector<aten::ThreadedBvhNode>* threadedBvhNodes = nullptr);

		void build(
			idaten::CudaTextureResource& dst,
			TypedCudaMemory<aten::PrimitiveParamter>& triangles,
			int triIdOffset,
			const aten::aabb& sceneBbox,
			CudaGLBuffer& vboVtxPos,
			int vtxOffset,
			std::vector<aten::ThreadedBvhNode>* threadedBvhNodes = nullptr);

		// test implementation.
		static void build();

		struct LBVHNode {
			int order;

			int left;
			int right;
			
			int parent;
			bool isLeaf;
		};

		void init(uint32_t maxNum);

	private:
		template <typename T>
		void onBuild(
			idaten::CudaTextureResource& dst,
			TypedCudaMemory<aten::PrimitiveParamter>& triangles,
			int triIdOffset,
			const aten::aabb& sceneBbox,
			T vtxPos,
			int vtxOffset,
			std::vector<aten::ThreadedBvhNode>* threadedBvhNodes);

	private:
		using MORTON_CODE_TYPE = uint32_t;

		TypedCudaMemory<MORTON_CODE_TYPE> m_mortonCodes;
		TypedCudaMemory<uint32_t> m_indices;
		TypedCudaMemory<MORTON_CODE_TYPE> m_sortedMortonCode;
		TypedCudaMemory<uint32_t> m_sortedIndices;
		RadixSort m_sort;
		TypedCudaMemory<LBVHBuilder::LBVHNode> m_nodesLbvh;
		TypedCudaMemory<aten::ThreadedBvhNode> m_nodes;
		uint32_t* m_executedIdxArray{ nullptr };
	};
}
