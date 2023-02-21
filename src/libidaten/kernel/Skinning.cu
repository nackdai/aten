#include "kernel/Skinning.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"

//#pragma optimize( "", off)

__global__ void computeSkinning(
    bool isRestart,
    uint32_t vtxNum,
    const aten::SkinningVertex* __restrict__ vertices,
    const aten::mat4* __restrict__ matrices,
    aten::vec4* dstPos,
    aten::vec4* dstNml,
    aten::vec4* dstPrev)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= vtxNum) {
        return;
    }

    const auto* vtx = &vertices[idx];

    aten::vec4 srcPos = vtx->position;
    aten::vec4 srcNml = aten::vec4(vtx->normal, 0);

    aten::vec4 resultPos(0);
    aten::vec4 resultNml(0);

    for (int32_t i = 0; i < 4; i++) {
        int32_t idx = int32_t(vtx->blendIndex[i]);
        float weight = vtx->blendWeight[i];

        aten::mat4 mtx = matrices[idx];

        resultPos += weight * mtx * vtx->position;
        resultNml += weight * mtx * srcNml;
    }

    resultNml = normalize(resultNml);

    if (isRestart) {
        dstPrev[idx] = aten::vec4(resultPos.x, resultPos.y, resultPos.z, 1.0f);
    }
    else {
        // Keep previous position.
        dstPrev[idx] = dstPos[idx];
        dstPrev[idx].w = 1.0f;
    }

    dstPos[idx] = aten::vec4(resultPos.x, resultPos.y, resultPos.z, vtx->uv[0]);
    dstNml[idx] = aten::vec4(resultNml.x, resultNml.y, resultNml.z, vtx->uv[1]);
}

__global__ void setTriangleParam(
    uint32_t triNum,
    aten::PrimitiveParamter* triangles,
    int32_t indexOffset,
    const aten::vec4* __restrict__ pos)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= triNum) {
        return;
    }

    auto* tri = &triangles[idx];

    const auto& v0 = pos[tri->idx[0]];
    const auto& v1 = pos[tri->idx[1]];
    const auto& v2 = pos[tri->idx[2]];

    auto a = v1 - v0;
    auto b = v2 - v0;

    tri->area = length(cross(a, b));

    tri->idx[0] += indexOffset;
    tri->idx[1] += indexOffset;
    tri->idx[2] += indexOffset;
}

// NOTE
// http://www.cuvilib.com/Reduction.pdf

static constexpr uint32_t BLOCK_SIZE = 256;

__global__ void getMinMax(
    bool isFinalIter,
    uint32_t num,
    const aten::vec4* __restrict__ src,
    aten::vec3* dstMin,
    aten::vec3* dstMax)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    const auto tid = threadIdx.x;

    // NOTE
    // http://yusuke-ujitoko.hatenablog.com/entry/2016/02/05/012618
    // カーネル呼び出しのときに指定できるのは1つの数だけ.
    // 複数のshared memoryを使いたいときは、shared memoryのサイズの合計を指定して、カーネル内部で切り分ける必要がある.

    // NOTE
    // We can't specify shared memory size as kernel parameter to templated variable.
    // Shared memory variable which is specified memory size as kernel parameter is declared with "extern"
    // And, the shared memory variable is disclosed.
    // Therefore, it conflicts in each template functions.
    // Of course, it makes same case even in other functions which are not template functions.
    // https://stackoverflow.com/questions/20497209/getting-cuda-error-declaration-is-incompatible-with-previous-variable-name

    __shared__ aten::vec3 minPos[BLOCK_SIZE];
    __shared__ aten::vec3 maxPos[BLOCK_SIZE];

    if (isFinalIter) {
        minPos[tid] = dstMin[idx];
        maxPos[tid] = dstMax[idx];
    }
    else {
        auto pos = src[idx];
        minPos[tid] = pos;
        maxPos[tid] = pos;
    }

    if (idx >= num) {
        minPos[tid] = aten::vec3(FLT_MAX);
        maxPos[tid] = aten::vec3(-FLT_MAX);
    }
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        //if (tid < s && tid + s < num) {
        if (tid < s) {
            minPos[tid] = aten::min(minPos[tid], minPos[tid + s]);
            maxPos[tid] = aten::max(maxPos[tid], maxPos[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        dstMin[blockIdx.x] = minPos[0];
        dstMax[blockIdx.x] = maxPos[0];
    }
}

namespace idaten
{
    void Skinning::init(
        aten::SkinningVertex* vertices,
        uint32_t vtxNum,
        uint32_t* indices,
        uint32_t idxNum,
        const aten::GeomMultiVertexBuffer* vb)
    {
        m_vertices.init(vtxNum);
        m_vertices.writeFromHostToDeviceByNum(vertices, vtxNum);

        m_indices.init(idxNum);
        m_indices.writeFromHostToDeviceByNum(indices, idxNum);

        if (vb) {
            auto handles = vb->getVBOHandles();

            m_interopVBO.resize(handles.size());

            for (int32_t i = 0; i < handles.size(); i++) {
                auto glvbo = handles[i];
                m_interopVBO[i].init(glvbo, CudaGLRscRegisterType::WriteOnly);
            }
        }
        else {
            m_dstPos.init(vtxNum);
            m_dstNml.init(vtxNum);
            m_dstPrev.init(vtxNum);
        }
    }


    void Skinning::initWithTriangles(
        aten::SkinningVertex* vertices,
        uint32_t vtxNum,
        aten::PrimitiveParamter* tris,
        uint32_t triNum,
        const aten::GeomMultiVertexBuffer* vb)
    {
        m_vertices.init(vtxNum);
        m_vertices.writeFromHostToDeviceByNum(vertices, vtxNum);

        m_triangles.init(triNum);
        m_triangles.writeFromHostToDeviceByNum(tris, triNum);

        if (vb) {
            auto handles = vb->getVBOHandles();

            // NOTE
            // Only support position, normal, previous position.
            AT_ASSERT(handles.size() == 3);

            m_interopVBO.resize(handles.size());

            for (int32_t i = 0; i < handles.size(); i++) {
                auto glvbo = handles[i];
                m_interopVBO[i].init(glvbo, CudaGLRscRegisterType::ReadWrite);
            }
        }
        else {
            m_dstPos.init(vtxNum);
            m_dstNml.init(vtxNum);
            m_dstPrev.init(vtxNum);
        }
    }

    void Skinning::update(
        const aten::mat4* matrices,
        uint32_t mtxNum)
    {
        if (m_matrices.bytes() == 0) {
            m_matrices.init(mtxNum);
        }

        AT_ASSERT(m_matrices.num() >= mtxNum);

        m_matrices.writeFromHostToDeviceByNum(matrices, mtxNum);
    }

    void Skinning::compute(
        aten::vec3& aabbMin,
        aten::vec3& aabbMax,
        bool isRestart/*= true*/)
    {
        aten::vec4* dstPos = nullptr;
        aten::vec4* dstNml = nullptr;
        aten::vec4* dstPrev = nullptr;
        size_t vtxbytes = 0;

        if (!m_interopVBO.empty()) {
            // NOTE
            // Only support position, normal.

            m_interopVBO[0].map();
            m_interopVBO[0].bind((void**)&dstPos, vtxbytes);

            m_interopVBO[1].map();
            m_interopVBO[1].bind((void**)&dstNml, vtxbytes);

            m_interopVBO[2].map();
            m_interopVBO[2].bind((void**)&dstPrev, vtxbytes);
        }
        else {
            dstPos = m_dstPos.ptr();
            dstNml = m_dstNml.ptr();
            dstPrev = m_dstPrev.ptr();
        }

        int32_t indexOffset = m_curVtxOffset - m_prevVtxOffset;
        m_prevVtxOffset = m_curVtxOffset;

        // Skinning.
        {
            auto willComputeWithTriangles = m_triangles.num() > 0;

            const auto vtxNum = m_vertices.num();

            dim3 block(512);
            dim3 grid((vtxNum + block.x - 1) / block.x);

            if (willComputeWithTriangles) {
                computeSkinning << <grid, block >> > (
                    isRestart,
                    vtxNum,
                    m_vertices.ptr(),
                    m_matrices.ptr(),
                    dstPos, dstNml, dstPrev);

                checkCudaKernel(computeSkinningWithTriangles);

                const auto triNum = m_triangles.num();

                grid = dim3((triNum + block.x - 1) / block.x);

                setTriangleParam << <grid, block >> > (
                    triNum,
                    m_triangles.ptr(),
                    indexOffset,
                    dstPos);

                checkCudaKernel(setTriangleParam);
            }
            else {
                computeSkinning << <grid, block >> > (
                    isRestart,
                    vtxNum,
                    m_vertices.ptr(),
                    m_matrices.ptr(),
                    dstPos, dstNml, dstPrev);

                checkCudaKernel(computeSkinning);
            }
        }

        // Get min/max.
        {
            auto src = dstPos;
            auto num = m_vertices.num();

            dim3 block(BLOCK_SIZE);
            dim3 grid((num + block.x - 1) / block.x);

            m_minBuf.init(grid.x);
            m_maxBuf.init(grid.x);

            auto sharedMemSize = block.x * sizeof(aten::vertex) * 2;

            getMinMax << <grid, block >> > (
                false,
                num,
                src,
                m_minBuf.ptr(),
                m_maxBuf.ptr());

            checkCudaKernel(getMinMax);

            num = grid.x;

            getMinMax << <1, block >> > (
                true,
                num,
                src,
                m_minBuf.ptr(),
                m_maxBuf.ptr());

            checkCudaKernel(getMinMaxFinal);
        }

        m_minBuf.readFromDeviceToHostByNum(&aabbMin, 1);
        m_maxBuf.readFromDeviceToHostByNum(&aabbMax, 1);

        if (!m_interopVBO.empty()) {
            m_interopVBO[0].unmap();
            m_interopVBO[1].unmap();
            m_interopVBO[2].unmap();
        }
    }

    bool Skinning::getComputedResult(
        aten::vec4* pos,
        aten::vec4* nml,
        uint32_t num)
    {
        AT_ASSERT(m_dstPos.bytes() > 0);
        AT_ASSERT(m_dstNml.bytes() > 0);

        m_dstPos.readFromDeviceToHostByNum(pos, num);
        m_dstNml.readFromDeviceToHostByNum(nml, num);

        return true;
    }

    void Skinning::setVtxOffset(int32_t offset)
    {
        AT_ASSERT(offset >= 0);
        m_prevVtxOffset = m_curVtxOffset;
        m_curVtxOffset = offset;
    }
}
