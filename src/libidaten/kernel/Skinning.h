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
            aten::TriangleParameter* tris,
            uint32_t triNum,
            const aten::GeomMultiVertexBuffer* vb);

        void update(
            const aten::mat4* matrices,
            uint32_t mtxNum);

        void compute(
            aten::vec3& aabbMin,
            aten::vec3& aabbMax,
            bool isRestart = true);

        bool getComputedResult(
            aten::vec4* pos,
            aten::vec4* nml,
            uint32_t num);

        std::vector<CudaGLBuffer>& getInteropVBO()
        {
            return m_interopVBO;
        }

        TypedCudaMemory<aten::TriangleParameter>& getTriangles()
        {
            return triangles_;
        }

        void setVtxOffset(int32_t offset);

    private:
        TypedCudaMemory<aten::SkinningVertex> vertices_;
        TypedCudaMemory<uint32_t> m_indices;
        TypedCudaMemory<aten::mat4> matrices_;

        TypedCudaMemory<aten::TriangleParameter> triangles_;

        TypedCudaMemory<aten::vec4> m_dstPos;
        TypedCudaMemory<aten::vec4> m_dstNml;
        TypedCudaMemory<aten::vec4> m_dstPrev;

        TypedCudaMemory<aten::vec3> m_minBuf;
        TypedCudaMemory<aten::vec3> m_maxBuf;

        std::vector<CudaGLBuffer> m_interopVBO;

        int32_t m_prevVtxOffset{ 0 };
        int32_t m_curVtxOffset{ 0 };
    };
}
