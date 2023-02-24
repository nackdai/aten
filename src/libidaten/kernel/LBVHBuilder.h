#pragma once

#include <vector>

#include "cuda/cudamemory.h"
#include "accelerator/threaded_bvh.h"
#include "cuda/cudaGLresource.h"
#include "kernel/RadixSort.h"

//#define AT_ENABLE_64BIT_LBVH_MORTON_CODE

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
            std::vector<aten::TriangleParameter>& tris,
            int32_t triIdOffset,
            const aten::aabb& sceneBbox,
            idaten::CudaTextureResource& texRscVtxPos,
            int32_t vtxOffset,
            std::vector<aten::ThreadedBvhNode>* threadedBvhNodes = nullptr);

        void build(
            idaten::CudaTextureResource& dst,
            TypedCudaMemory<aten::TriangleParameter>& triangles,
            int32_t triIdOffset,
            const aten::aabb& sceneBbox,
            CudaGLBuffer& vboVtxPos,
            int32_t vtxOffset,
            std::vector<aten::ThreadedBvhNode>* threadedBvhNodes = nullptr);

        // test implementation.
        static void build();

        struct LBVHNode {
            int32_t order;

            int32_t left;
            int32_t right;

            int32_t parent;
            bool isLeaf;
        };

        void init(uint32_t maxNum);

    private:
        template <typename T>
        void onBuild(
            idaten::CudaTextureResource& dst,
            TypedCudaMemory<aten::TriangleParameter>& triangles,
            int32_t triIdOffset,
            const aten::aabb& sceneBbox,
            T vtxPos,
            int32_t vtxOffset,
            std::vector<aten::ThreadedBvhNode>* threadedBvhNodes);

    private:
#ifdef AT_ENABLE_64BIT_LBVH_MORTON_CODE
        using MORTON_CODE_TYPE = uint64_t;
#else
        using MORTON_CODE_TYPE = uint32_t;
#endif

        TypedCudaMemory<MORTON_CODE_TYPE> m_mortonCodes;
        TypedCudaMemory<uint32_t> m_indices;
        RadixSort m_sort;
        TypedCudaMemory<LBVHBuilder::LBVHNode> m_nodesLbvh;
        TypedCudaMemory<aten::ThreadedBvhNode> m_nodes;
        uint32_t* m_executedIdxArray{ nullptr };
    };
}
