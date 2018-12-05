#include "svgf/svgf.h"

#include "kernel/context.cuh"
#include "kernel/pt_common.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

inline __device__ float gaussFilter3x3(
    int ix, int iy,
    int w, int h,
    const float4* __restrict__ var)
{
    static const float kernel[] = {
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
        1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
    };

    static const int offsetx[] = {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1,
    };

    static const int offsety[] = {
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1,
    };

    float sum = 0;

    int pos = 0;

#pragma unroll
    for (int i = 0; i < 9; i++) {
        int xx = clamp(ix + offsetx[i], 0, w - 1);
        int yy = clamp(iy + offsety[i], 0, h - 1);

        int idx = getIdx(xx, yy, w);

        float tmp = var[idx].w;

        sum += kernel[pos] * tmp;

        pos++;
    }

    return sum;
}

__global__ void atrousFilter(
    idaten::TileDomain tileDomain,
    bool isFirstIter, bool isFinalIter,
    cudaSurfaceObject_t dst,
    float4* tmpBuffer,
    const float4* __restrict__ aovNormalDepth,
    const float4* __restrict__ aovTexclrMeshid,
    const float4* __restrict__ aovColorVariance,
    const float4* __restrict__ aovMomentTemporalWeight,
    const float4* __restrict__ clrVarBuffer,
    float4* nextClrVarBuffer,
    int stepScale,
    int width, int height,
    float cameraDistance)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    const int idx = getIdx(ix, iy, width);

    auto normalDepth = aovNormalDepth[idx];
    auto texclrMeshid = aovTexclrMeshid[idx];

    float3 centerNormal = make_float3(normalDepth.x, normalDepth.y, normalDepth.z);
    float centerDepth = normalDepth.w;
    int centerMeshId = (int)texclrMeshid.w;

    float4 centerColor;

    if (isFirstIter) {
        centerColor = aovColorVariance[idx];
    }
    else {
        centerColor = clrVarBuffer[idx];
    }

    if (centerMeshId < 0) {
        // 背景なので、そのまま出力して終了.
        nextClrVarBuffer[idx] = make_float4(centerColor.x, centerColor.y, centerColor.z, 0.0f);

        if (isFinalIter) {
            centerColor *= texclrMeshid;

            surf2Dwrite(
                centerColor,
                dst,
                ix * sizeof(float4), iy,
                cudaBoundaryModeTrap);
        }

        return;
    }

    float centerLum = AT_NAME::color::luminance(centerColor.x, centerColor.y, centerColor.z);

    // ガウスフィルタ3x3
    float gaussedVarLum;
    
    if (isFirstIter) {
        gaussedVarLum = gaussFilter3x3(ix, iy, width, height, aovColorVariance);
    }
    else {
        gaussedVarLum = gaussFilter3x3(ix, iy, width, height, clrVarBuffer);
    }

    float sqrGaussedVarLum = sqrt(gaussedVarLum);

    static const float sigmaZ = 1.0f;
    static const float sigmaN = 128.0f;
    static const float sigmaL = 4.0f;

    float2 p = make_float2(ix, iy);

    static const float h[] = {
        2.0 / 3.0,  2.0 / 3.0,  2.0 / 3.0,  2.0 / 3.0,
        1.0 / 6.0,  1.0 / 6.0,  1.0 / 6.0,  1.0 / 6.0,
        4.0 / 9.0,  4.0 / 9.0,  4.0 / 9.0,  4.0 / 9.0,
        1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
        1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    };

    static const int offsetx[] = {
        1,  0, -1, 0,
        2,  0, -2, 0,
        1, -1, -1, 1,
        1, -1, -1, 1,
        2, -2, -2, 2,
        2, -2, -2, 2,
    };
    static const int offsety[] = {
        0, 1,  0, -1,
        0, 2,  0, -2,
        1, 1, -1, -1,
        2, 2, -2, -2,
        1, 1, -1, -1,
        2, 2, -2, -2,
    };

    float4 sumC = centerColor;
    float sumV = centerColor.w;
    float weight = 1.0f;

    int pos = 0;

    float pixelDistanceRatio = (centerDepth / cameraDistance) * height;

#pragma unroll
    for (int i = 0; i < 24; i++)
    {
        int xx = clamp(ix + offsetx[i] * stepScale, 0, width - 1);
        int yy = clamp(iy + offsety[i] * stepScale, 0, height - 1);

        float2 u = make_float2(offsetx[i] * stepScale, offsety[i] * stepScale);
        float2 q = make_float2(xx, yy);

        const int qidx = getIdx(xx, yy, width);

        normalDepth = aovNormalDepth[qidx];
        texclrMeshid = aovTexclrMeshid[qidx];

        float3 normal = make_float3(normalDepth.x, normalDepth.y, normalDepth.z);

        float depth = normalDepth.w;
        int meshid = (int)texclrMeshid.w;

        float4 color;
        float variance;

        if (isFirstIter) {
            color = aovColorVariance[qidx];
            variance = color.w;
        }
        else {
            color = clrVarBuffer[qidx];
            variance = color.w;
        }

        float lum = AT_NAME::color::luminance(color.x, color.y, color.z);

        float Wz = 3.0f * fabs(centerDepth - depth) / (pixelDistanceRatio * length(u) + 0.000001f);

        float Wn = powf(max(0.0f, dot(centerNormal, normal)), sigmaN);

        float Wl = min(expf(-fabs(centerLum - lum) / (sigmaL * sqrGaussedVarLum + 0.000001f)), 1.0f);

        float Wm = meshid == centerMeshId ? 1.0f : 0.0f;

        float W = expf(-Wl * Wl - Wz) * Wn * Wm * h[i];
            
        sumC += W * color;
        sumV += W * W * variance;

        weight += W;

        pos++;
    }

    sumC /= weight;
    sumV /= (weight * weight);

    nextClrVarBuffer[idx] = make_float4(sumC.x, sumC.y, sumC.z, sumV);

    if (isFirstIter) {
        // Store color temporary.
        tmpBuffer[idx] = sumC;
    }
    
    if (isFinalIter) {
        texclrMeshid = aovTexclrMeshid[idx];
        sumC *= texclrMeshid;

        surf2Dwrite(
            sumC,
            dst,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }
}

__global__ void copyFromBufferToAov(
    const float4* __restrict__ src,
    float4* aovColorVariance,
    int width, int height)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const int idx = getIdx(ix, iy, width);

    float4 s = src[idx];

    aovColorVariance[idx].x = s.x;
    aovColorVariance[idx].y = s.y;
    aovColorVariance[idx].z = s.z;
}

namespace idaten
{
    void SVGFPathTracing::onAtrousFilter(
        cudaSurfaceObject_t outputSurf,
        int width, int height)
    {
        m_atrousMaxIterCnt = aten::clamp(m_atrousMaxIterCnt, 0U, 5U);

        for (int i = 0; i < m_atrousMaxIterCnt; i++) {
            onAtrousFilterIter(
                i, m_atrousMaxIterCnt,
                outputSurf,
                width, height);
        }
    }

    void SVGFPathTracing::onAtrousFilterIter(
        uint32_t iterCnt,
        uint32_t maxIterCnt,
        cudaSurfaceObject_t outputSurf,
        int width, int height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        int curaov = getCurAovs();

        int cur = iterCnt & 0x01;
        int next = 1 - cur;

        bool isFirstIter = iterCnt == 0 ? true : false;
        bool isFinalIter = iterCnt == maxIterCnt - 1 ? true : false;

        float cameraDistance = height / (2.0f * aten::tan(0.5f * m_camParam.vfov));

        int stepScale = 1 << iterCnt;

        atrousFilter << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            isFirstIter, isFinalIter,
            outputSurf,
            m_tmpBuf.ptr(),
            m_aovNormalDepth[curaov].ptr(),
            m_aovTexclrMeshid[curaov].ptr(),
            m_aovColorVariance[curaov].ptr(),
            m_aovMomentTemporalWeight[curaov].ptr(),
            m_atrousClrVar[cur].ptr(), m_atrousClrVar[next].ptr(),
            stepScale,
            width, height,
            cameraDistance);
        checkCudaKernel(atrousFilter);
    }

    void SVGFPathTracing::onCopyFromTmpBufferToAov(int width, int height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        int curaov = getCurAovs();

        // Copy color from temporary buffer to AOV buffer for next temporal reprojection.
        copyFromBufferToAov << <grid, block, 0, m_stream >> > (
            m_tmpBuf.ptr(),
            m_aovColorVariance[curaov].ptr(),
            width, height);
        checkCudaKernel(copyFromBufferToAov);
    }
}
