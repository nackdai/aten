#include "svgf/svgf.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/pt_common.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

inline __device__ float gaussFilter3x3(
    int32_t ix, int32_t iy,
    int32_t w, int32_t h,
    const float4* __restrict__ var)
{
    static const float kernel[] = {
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
        1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
    };

    static const int32_t offsetx[] = {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1,
    };

    static const int32_t offsety[] = {
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1,
    };

    float sum = 0;

    int32_t pos = 0;

#pragma unroll
    for (int32_t i = 0; i < 9; i++) {
        int32_t xx = clamp(ix + offsetx[i], 0, w - 1);
        int32_t yy = clamp(iy + offsety[i], 0, h - 1);

        int32_t idx = getIdx(xx, yy, w);

        float tmp = var[idx].w;

        sum += kernel[pos] * tmp;

        pos++;
    }

    return sum;
}

__global__ void atrousFilter(
    bool isFirstIter, bool isFinalIter,
    cudaSurfaceObject_t dst,
    float4* tmpBuffer,
    const float4* __restrict__ aovNormalDepth,
    const float4* __restrict__ aovTexclrMeshid,
    const float4* __restrict__ aovColorVariance,
    const float4* __restrict__ aovMomentTemporalWeight,
    const float4* __restrict__ clrVarBuffer,
    float4* nextClrVarBuffer,
    int32_t stepScale,
    int32_t width, int32_t height,
    float cameraDistance)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const int32_t idx = getIdx(ix, iy, width);

    auto normalDepth = aovNormalDepth[idx];
    auto texclrMeshid = aovTexclrMeshid[idx];

    float3 centerNormal = make_float3(normalDepth.x, normalDepth.y, normalDepth.z);
    float centerDepth = normalDepth.w;
    int32_t centerMeshId = (int32_t)texclrMeshid.w;

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

    static const int32_t offsetx[] = {
        1,  0, -1, 0,
        2,  0, -2, 0,
        1, -1, -1, 1,
        1, -1, -1, 1,
        2, -2, -2, 2,
        2, -2, -2, 2,
    };
    static const int32_t offsety[] = {
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

    int32_t pos = 0;

    float pixelDistanceRatio = (centerDepth / cameraDistance) * height;

#pragma unroll
    for (int32_t i = 0; i < 24; i++)
    {
        int32_t xx = clamp(ix + offsetx[i] * stepScale, 0, width - 1);
        int32_t yy = clamp(iy + offsety[i] * stepScale, 0, height - 1);

        float2 u = make_float2(offsetx[i] * stepScale, offsety[i] * stepScale);
        float2 q = make_float2(xx, yy);

        const int32_t qidx = getIdx(xx, yy, width);

        normalDepth = aovNormalDepth[qidx];
        texclrMeshid = aovTexclrMeshid[qidx];

        float3 normal = make_float3(normalDepth.x, normalDepth.y, normalDepth.z);

        float depth = normalDepth.w;
        int32_t meshid = (int32_t)texclrMeshid.w;

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
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const int32_t idx = getIdx(ix, iy, width);

    float4 s = src[idx];

    aovColorVariance[idx].x = s.x;
    aovColorVariance[idx].y = s.y;
    aovColorVariance[idx].z = s.z;
}

namespace idaten
{
    void SVGFPathTracing::onAtrousFilter(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height)
    {
        m_atrousMaxIterCnt = aten::clamp(m_atrousMaxIterCnt, 0U, 5U);

        for (int32_t i = 0; i < m_atrousMaxIterCnt; i++) {
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
        int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        int32_t curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        int32_t cur = iterCnt & 0x01;
        int32_t next = 1 - cur;

        bool isFirstIter = iterCnt == 0 ? true : false;
        bool isFinalIter = iterCnt == maxIterCnt - 1 ? true : false;

        float cameraDistance = height / (2.0f * aten::tan(0.5f * m_cam.vfov));

        int32_t stepScale = 1 << iterCnt;

        atrousFilter << <grid, block, 0, m_stream >> > (
            isFirstIter, isFinalIter,
            outputSurf,
            m_tmpBuf.ptr(),
            curaov.get<AOVBuffer::NormalDepth>().ptr(),
            curaov.get<AOVBuffer::AlbedoMeshId>().ptr(),
            curaov.get<AOVBuffer::ColorVariance>().ptr(),
            curaov.get<AOVBuffer::MomentTemporalWeight>().ptr(),
            m_atrousClrVar[cur].ptr(), m_atrousClrVar[next].ptr(),
            stepScale,
            width, height,
            cameraDistance);
        checkCudaKernel(atrousFilter);
    }

    void SVGFPathTracing::onCopyFromTmpBufferToAov(int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        int32_t curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        // Copy color from temporary buffer to AOV buffer for next temporal reprojection.
        copyFromBufferToAov << <grid, block, 0, m_stream >> > (
            m_tmpBuf.ptr(),
            curaov.get<AOVBuffer::ColorVariance>().ptr(),
            width, height);
        checkCudaKernel(copyFromBufferToAov);
    }
}
