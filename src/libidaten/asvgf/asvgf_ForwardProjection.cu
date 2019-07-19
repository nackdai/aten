#include "asvgf/asvgf.h"

#include "kernel/pt_common.h"
#include "kernel/context.cuh"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

// TEA = Tiny Encryption Algorithm.
// https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
inline __host__ __device__ void encryptTea(uint2& arg)
{
    const uint32_t key[] = {
        0xa341316c, 
        0xc8013ea4, 
        0xad90777d, 
        0x7e95761e,
    };

    uint32_t v0 = arg.x;
    uint32_t v1 = arg.y;
    uint32_t sum = 0;
    uint32_t delta = 0x9e3779b9;

    for (int i = 0; i < 16; i++) {
        sum += delta;
        v0 += ((v1 << 4) + key[0]) ^ (v1 + sum) ^ ((v1 >> 5) + key[1]);
        v1 += ((v0 << 4) + key[2]) ^ (v0 + sum) ^ ((v0 >> 5) + key[3]);
    }

    arg.x = v0;
    arg.y = v1;
}

void _onForwardProjection(
    int ix, int iy,
    uint32_t frame,
    int width, int height,
    int tiledWidth, int tiledHeight,
    const aten::GeomParameter* shapes,
    const aten::PrimitiveParamter* prims,
    const float4* vtxPos,
    const aten::mat4* matrices,
    const float4* visibilityBuf,
    aten::mat4 mtxW2C,
    unsigned int* rngBuffer_0,
    unsigned int* rngBuffer_1)
{
    if (ix >= tiledWidth || iy >= tiledHeight) {
        return;
    }

    int idx = getIdx(ix, iy, tiledWidth);

    // Compute random position in 3x3 tile.
    uint2 tea = make_uint2(idx, frame);
    encryptTea(tea);
    tea.x %= idaten::AdvancedSVGFPathTracing::GradientTileSize;
    tea.y %= idaten::AdvancedSVGFPathTracing::GradientTileSize;

    // Transform to real screen resolution.
    int2 pos = make_int2(
        ix * idaten::AdvancedSVGFPathTracing::GradientTileSize + tea.x,
        iy * idaten::AdvancedSVGFPathTracing::GradientTileSize + tea.y);

    // Not allowed over than screen resolution.
    if (pos.x >= width || pos.y >= height) {
        return;
    }

    // Index to sample previous frame information.
    int idxPrev = getIdx(pos.x, pos.y, width);

    float4 visBuf = visibilityBuf[idxPrev];

    // XY is bary centroid, and -1 is invalid value (no hit).
    if (visBuf.x < 0.0f || visBuf.y < 0.0f) {
        return;
    }

    uint32_t objid = *(uint32_t*)(&visBuf.z);
    uint32_t primid = *(uint32_t*)(&visBuf.w);

    const auto* s = &shapes[objid];
    const auto* t = &prims[primid];

    float4 p0 = vtxPos[t->idx[0]];
    float4 p1 = vtxPos[t->idx[1]];
    float4 p2 = vtxPos[t->idx[2]];

    float4 _p = p0 * visBuf.x + p1 * visBuf.y + p2 * (1.0f - visBuf.x - visBuf.y);
    aten::vec4 pos_cur(_p.x, _p.y, _p.z, 1.0f);

    // Transform to clip coordinate.
    auto mtxL2W = matrices[s->mtxid * 2];
    pos_cur = mtxW2C.apply(mtxL2W.apply(pos_cur));
    pos_cur /= pos_cur.w;

    if (!AT_MATH_IS_IN_BOUND(pos_cur.x, -pos_cur.w, pos_cur.w)
        || !AT_MATH_IS_IN_BOUND(pos_cur.y, -pos_cur.w, pos_cur.w)
        || !AT_MATH_IS_IN_BOUND(pos_cur.z, -pos_cur.w, pos_cur.w))
    {
        // Not in screen.
        return;
    }

    /* pixel coordinate of forward projected sample */
    int2 ipos_curr = make_int2(
        (pos_cur.x * 0.5 + 0.5) * width,
        (pos_cur.y * 0.5 + 0.5) * height);

    int2 pos_grad = make_int2(
        ipos_curr.x / idaten::AdvancedSVGFPathTracing::GradientTileSize,
        ipos_curr.y / idaten::AdvancedSVGFPathTracing::GradientTileSize);
    int2 pos_stratum = make_int2(
        ipos_curr.x % idaten::AdvancedSVGFPathTracing::GradientTileSize,
        ipos_curr.y % idaten::AdvancedSVGFPathTracing::GradientTileSize);

    uint32_t gradient_idx =
        (1 << 31) // mark sample as busy.
        | (pos_stratum.x << (idaten::AdvancedSVGFPathTracing::StratumOffsetShift * 0)) // encode pos in.
        | (pos_stratum.y << (idaten::AdvancedSVGFPathTracing::StratumOffsetShift * 1)) // current frame.
        | (idxPrev << (idaten::AdvancedSVGFPathTracing::StratumOffsetShift * 2));      // pos in prev frame.


    int idxInGradIdxBuffer = getIdx(pos_grad.x, pos_grad.y, tiledWidth);

    int idxCurr = getIdx(ipos_curr.x, ipos_curr.y, width);

    // Forward project rng seed.
    if ((frame & 0x01) == 0) {
        rngBuffer_0[idxCurr] = rngBuffer_1[idxPrev];
    }
    else {
        rngBuffer_1[idxCurr] = rngBuffer_0[idxPrev];
    }
}

__global__ void forwardProjection(
    uint32_t frame,
    int width, int height,
    int tiledWidth, int tiledHeight,
    const aten::GeomParameter* __restrict__ shapes,
    const aten::PrimitiveParamter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    const aten::mat4* __restrict__ matrices,
    const float4* __restrict__ visibilityBuf,
    aten::mat4 mtxW2C,
    cudaSurfaceObject_t motionDetphBuffer,
    unsigned int* gradientIndicices,
    unsigned int* rngBuffer_0,
    unsigned int* rngBuffer_1)
{
    // NOE
    // This kernel is called with 1/3 screen resolution.

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tiledWidth || iy >= tiledHeight) {
        return;
    }

    int idx = getIdx(ix, iy, tiledWidth);

    // Compute random position in 3x3 tile.
    uint2 tea = make_uint2(idx, frame);
    encryptTea(tea);
    tea.x %= idaten::AdvancedSVGFPathTracing::GradientTileSize;
    tea.y %= idaten::AdvancedSVGFPathTracing::GradientTileSize;

    // Transform to real screen resolution.
    int2 pos = make_int2(
        ix * idaten::AdvancedSVGFPathTracing::GradientTileSize + tea.x,
        iy * idaten::AdvancedSVGFPathTracing::GradientTileSize + tea.y);

    // Not allowed over than screen resolution.
    if (pos.x >= width || pos.y >= height) {
        return;
    }

    // Index to sample previous frame information.
    int idxPrev = getIdx(pos.x, pos.y, width);

    float4 visBuf = visibilityBuf[idxPrev];

    // XY is bary centroid, and -1 is invalid value (no hit).
    if (visBuf.x < 0.0f || visBuf.y < 0.0f) {
        return;
    }

    uint32_t objid = __float_as_uint(visBuf.z);
    uint32_t primid = __float_as_uint(visBuf.w);

    const auto* s = &shapes[objid];
    const auto* t = &prims[primid];

    float4 p0 = tex1Dfetch<float4>(vtxPos, t->idx[0]);
    float4 p1 = tex1Dfetch<float4>(vtxPos, t->idx[1]);
    float4 p2 = tex1Dfetch<float4>(vtxPos, t->idx[2]);

    float4 _p = p0 * visBuf.x + p1 * visBuf.y + p2 * (1.0f - visBuf.x - visBuf.y);
    aten::vec4 pos_cur(_p.x, _p.y, _p.z, 1.0f);

    // Transform to clip coordinate.
    if (s->mtxid >= 0) {
        auto mtxL2W = matrices[s->mtxid * 2];
        pos_cur = mtxL2W.apply(pos_cur);
    }
    pos_cur = mtxW2C.apply(pos_cur);
    pos_cur /= pos_cur.w;

    // Transform previous position to predicted current position.
    float4 motionDepth;
    surf2Dread(&motionDepth, motionDetphBuffer, pos.x * sizeof(float4), pos.y);

    pos_cur.x += motionDepth.x;
    pos_cur.y += motionDepth.y;

    if (!AT_MATH_IS_IN_BOUND(pos_cur.x, -pos_cur.w, pos_cur.w)
        || !AT_MATH_IS_IN_BOUND(pos_cur.y, -pos_cur.w, pos_cur.w)
        || !AT_MATH_IS_IN_BOUND(pos_cur.z, -pos_cur.w, pos_cur.w))
    {
        // Not in screen.
        return;
    }

    /* pixel coordinate of forward projected sample */
    int2 ipos_curr = make_int2(
        (pos_cur.x * 0.5 + 0.5) * width,
        (pos_cur.y * 0.5 + 0.5) * height);

    int2 pos_grad = make_int2(
        ipos_curr.x / idaten::AdvancedSVGFPathTracing::GradientTileSize,
        ipos_curr.y / idaten::AdvancedSVGFPathTracing::GradientTileSize);
    int2 pos_stratum = make_int2(
        ipos_curr.x % idaten::AdvancedSVGFPathTracing::GradientTileSize,
        ipos_curr.y % idaten::AdvancedSVGFPathTracing::GradientTileSize);

    uint32_t gradient_idx =
        (1 << 31) // mark sample as busy.
        | (pos_stratum.x << (idaten::AdvancedSVGFPathTracing::StratumOffsetShift * 0)) // encode pos in.
        | (pos_stratum.y << (idaten::AdvancedSVGFPathTracing::StratumOffsetShift * 1)) // current frame.
        | (idxPrev << (idaten::AdvancedSVGFPathTracing::StratumOffsetShift * 2));      // pos in prev frame.

    // NOTE
    // Atomic functions for CUDA.
    // http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?CUDA%A5%A2%A5%C8%A5%DF%A5%C3%A5%AF%B4%D8%BF%F4
    // atomicCAS(unsigned int* address, unsigned int compare, unsigned int val) {
    //     *address = (old == compare) ? val : old;
    //     return *address;
    // }

    int idxInGradIdxBuffer = getIdx(pos_grad.x, pos_grad.y, tiledWidth);

    // Check if this sample is allowed to become a gradient sample.
    if (atomicCAS(&gradientIndicices[idxInGradIdxBuffer], 0U, gradient_idx) != 0U) {
        return;
    }

    int idxCurr = getIdx(ipos_curr.x, ipos_curr.y, width);

    // Forward project rng seed.
    if ((frame & 0x01) == 0) {
        rngBuffer_0[idxCurr] = rngBuffer_1[idxPrev];
    }
    else {
        rngBuffer_1[idxCurr] = rngBuffer_0[idxPrev];
    }

    // TODO
}

namespace idaten {
    void AdvancedSVGFPathTracing::onForwardProjection(
        int width, int height,
        cudaTextureObject_t texVtxPos)
    {
        int tiledWidth = width / GradientTileSize;
        int tiledHeight = height / GradientTileSize;

        m_mtxW2V.lookat(
            m_camParam.origin,
            m_camParam.center,
            m_camParam.up);

        m_mtxV2C.perspective(
            m_camParam.znear,
            m_camParam.zfar,
            m_camParam.vfov,
            m_camParam.aspect);

        aten::mat4 mtxW2C = m_mtxV2C * m_mtxW2V;

        dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        CudaGLResourceMapper rscmap(&m_motionDepthBuffer);
        auto motionDepthBuffer = m_motionDepthBuffer.bind();

        int curRngSeed = (m_frame & 0x01);
        int prevRngSeed = 1 - curRngSeed;

#if 0
        {
            std::vector<float4> vtx(m_vtxparamsPos.size() / sizeof(float4));
            m_vtxparamsPos.read(vtx.data(), sizeof(float4) * vtx.size());

            std::vector<aten::GeomParameter> shapeparam(m_shapeparam.num());
            m_shapeparam.readByNum(shapeparam.data(), shapeparam.size());

            std::vector<aten::PrimitiveParamter> primparams(m_primparams.num());
            m_primparams.readByNum(primparams.data(), primparams.size());

            std::vector<aten::mat4> mtxparams(m_mtxparams.num());
            m_mtxparams.readByNum(mtxparams.data(), mtxparams.size());

            std::vector<float4> visibilityBuffer(m_visibilityBuffer.num());
            m_visibilityBuffer.readByNum(visibilityBuffer.data(), visibilityBuffer.size());

            std::vector<uint32_t> rngSeed[2];

            rngSeed[0].resize(m_rngSeed[0].num());
            m_rngSeed[0].readByNum(rngSeed[0].data(), rngSeed[0].size());

            rngSeed[1].resize(m_rngSeed[1].num());
            m_rngSeed[1].readByNum(rngSeed[1].data(), rngSeed[1].size());

            for (int iy = 0; iy < height; iy++) {
                for (int ix = 0; ix < width; ix++) {
                    _onForwardProjection(
                        ix, iy,
                        m_frame,
                        width, height,
                        tiledWidth, tiledHeight,
                        shapeparam.data(),
                        primparams.data(),
                        vtx.data(),
                        mtxparams.data(),
                        visibilityBuffer.data(),
                        mtxW2C,
                        rngSeed[curRngSeed].data(),
                        rngSeed[prevRngSeed].data());
                }
            }
        }
#endif

        forwardProjection << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_frame,
            width, height,
            tiledWidth, tiledHeight,
            m_shapeparam.ptr(),
            m_primparams.ptr(),
            texVtxPos,
            m_mtxparams.ptr(),
            m_visibilityBuffer.ptr(),
            mtxW2C,
            motionDepthBuffer,
            m_gradientIndices.ptr(),
            m_rngSeed[curRngSeed].ptr(),
            m_rngSeed[prevRngSeed].ptr());

        checkCudaKernel(forwardProjection);
    }
}
