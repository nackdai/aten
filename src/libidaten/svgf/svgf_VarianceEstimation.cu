#include "svgf/svgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void varianceEstimation(
    idaten::TileDomain tileDomain,
    cudaSurfaceObject_t dst,
    const float4* __restrict__ aovNormalDepth,
    float4* aovMomentTemporalWeight,
    float4* aovColorVariance,
    float4* aovTexclrMeshid,
    aten::mat4 mtxC2V,
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
    auto momentTemporalWeight = aovMomentTemporalWeight[idx];
    auto centerColor = aovColorVariance[idx];

    float centerDepth = aovNormalDepth[idx].w;
    int centerMeshId = (int)texclrMeshid.w;

    if (centerMeshId < 0) {
        // 背景なので、分散はゼロ.
        aovMomentTemporalWeight[idx].x = 0;
        aovMomentTemporalWeight[idx].y = 0;
        aovMomentTemporalWeight[idx].z = 1;

        surf2Dwrite(
            make_float4(0),
            dst,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }

    float pixelDistanceRatio = (centerDepth / cameraDistance) * height;

    float3 centerMoment = make_float3(momentTemporalWeight.x, momentTemporalWeight.y, momentTemporalWeight.z);

    int frame = (int)centerMoment.z;

    centerMoment /= centerMoment.z;

    float var = 0.0f;
    float4 color = centerColor;

    if (frame < 4) {
        // 積算フレーム数が４未満 or Disoccludedされている.
        // 7x7birateral filterで輝度を計算.

        float3 centerNormal = make_float3(normalDepth.x, normalDepth.y, normalDepth.z);

        float3 momentSum = make_float3(centerMoment.x, centerMoment.y, centerMoment.z);
        float weight = 1.0f;

        float radius = frame > 1 ? 2 : 3;

        for (int v = -radius; v <= radius; v++)
        {
            for (int u = -radius; u <= radius; u++)
            {
                if (u != 0 || v != 0) {
                    int xx = clamp(ix + u, 0, width - 1);
                    int yy = clamp(iy + v, 0, height - 1);

                    int pidx = getIdx(xx, yy, width);
                    normalDepth = aovNormalDepth[pidx];
                    texclrMeshid = aovTexclrMeshid[pidx];
                    momentTemporalWeight = aovMomentTemporalWeight[pidx];

                    float3 sampleNml = make_float3(normalDepth.x, normalDepth.y, normalDepth.z);
                    float sampleDepth = normalDepth.w;
                    int sampleMeshId = (int)texclrMeshid.w;
                    auto sampleColor = aovColorVariance[pidx];

                    float3 moment = make_float3(momentTemporalWeight.x, momentTemporalWeight.y, momentTemporalWeight.z);
                    moment /= moment.z;

                    float Wz = aten::abs(sampleDepth - centerDepth) / (pixelDistanceRatio * length(make_float2(u, v)) + 1e-2);
                    float Wn = aten::pow(aten::cmpMax(0.0f, dot(sampleNml, centerNormal)), 128.0f);

                    float Wm = centerMeshId == sampleMeshId ? 1.0f : 0.0f;

                    float W = exp(-Wz) * Wn * Wm;

                    momentSum += moment * W;
                    color += sampleColor * W;
                    weight += W;
                }
            }
        }

        momentSum /= weight;
        color /= weight;

        var = 1.0f + 3.0f * (1.0f - frame / 4.0f) * max(0.0, momentSum.y - momentSum.x * momentSum.x);
    }
    else {
        var = max(0.0f, centerMoment.x - centerMoment.y * centerMoment.y);
    }

    color.w = var;
    aovColorVariance[idx] = color;

    surf2Dwrite(
        make_float4(var, var, var, 1),
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

namespace idaten
{
    void SVGFPathTracing::onVarianceEstimation(
        cudaSurfaceObject_t outputSurf,
        int width, int height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        float cameraDistance = height / (2.0f * aten::tan(0.5f * m_camParam.vfov));

        int curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        varianceEstimation << <grid, block, 0, m_stream >> > (
        //varianceEstimation << <1, 1 >> > (
            m_tileDomain,
            outputSurf,
            curaov.get<AOVBuffer::NormalDepth>().ptr(),
            curaov.get<AOVBuffer::MomentTemporalWeight>().ptr(),
            curaov.get<AOVBuffer::ColorVariance>().ptr(),
            curaov.get<AOVBuffer::AlbedoMeshId>().ptr(),
            m_mtxC2V,
            width, height,
            cameraDistance);

        checkCudaKernel(varianceEstimation);
    }
}
