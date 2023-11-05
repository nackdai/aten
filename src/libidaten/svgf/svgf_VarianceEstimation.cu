#include "svgf/svgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"
#include "renderer/svgf/svgf_impl.h"

__global__ void varianceEstimation(
    cudaSurfaceObject_t dst,
    const float4* __restrict__ aovNormalDepth,
    float4* aovMomentTemporalWeight,
    float4* aovColorVariance,
    float4* aovTexclrMeshid,
    aten::mat4 mtx_C2V,
    int32_t width, int32_t height,
    float cameraDistance)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const size_t size = width * height;

    auto aov_normal_depth{ aten::span<float4>(const_cast<float4*>(aovNormalDepth), size) };
    auto aov_texclr_meshid{ aten::span<float4>(aovTexclrMeshid, size) };
    auto aov_color_variance{ aten::span<float4>(aovColorVariance, size) };
    auto aov_moment_temporalweight{ aten::span<float4>(aovMomentTemporalWeight, size) };

    auto result = AT_NAME::svgf::EstimateVariance(
        ix, iy, width, height,
        mtx_C2V, cameraDistance,
        aov_normal_depth,
        aov_texclr_meshid,
        aov_color_variance,
        aov_moment_temporalweight);

    surf2Dwrite(
        result,
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

namespace idaten
{
    void SVGFPathTracing::onVarianceEstimation(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        float cameraDistance = AT_NAME::camera::ComputeScreenDistance(m_cam, height);

        int32_t curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        varianceEstimation << <grid, block, 0, m_stream >> > (
            outputSurf,
            curaov.get<AOVBuffer::NormalDepth>().data(),
            curaov.get<AOVBuffer::MomentTemporalWeight>().data(),
            curaov.get<AOVBuffer::ColorVariance>().data(),
            curaov.get<AOVBuffer::AlbedoMeshId>().data(),
            m_mtx_C2V,
            width, height,
            cameraDistance);

        checkCudaKernel(varianceEstimation);
    }
}
