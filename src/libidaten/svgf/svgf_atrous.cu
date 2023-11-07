#include "svgf/svgf.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/pt_common.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"
#include "renderer/svgf/svgf_impl.h"

__global__ void atrousFilter(
    bool isFirstIter, bool isFinalIter,
    cudaSurfaceObject_t dst,
    float4* temporaryColorBuffer,
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

    const size_t size = width * height;

    auto temporary_color_buffer{ aten::span<float4>(temporaryColorBuffer, size) };
    auto aov_normal_depth{ aten::const_span<float4>(aovNormalDepth, size) };
    auto aov_texclr_meshid{ aten::const_span<float4>(aovTexclrMeshid, size) };
    auto aov_color_variance{ aten::const_span<float4>(aovColorVariance, size) };
    auto aov_moment_temporalweight{ aten::const_span<float4>(aovMomentTemporalWeight, size) };
    auto color_variance_buffer{ aten::const_span<float4>(clrVarBuffer, size) };
    auto next_color_variance_buffer{ aten::span<float4>(nextClrVarBuffer, size) };

    const int32_t idx = getIdx(ix, iy, width);

    auto extracted_center_pixel = AT_NAME::svgf::ExtractCenterPixel<false>(
        idx,
        isFirstIter ? aov_color_variance : color_variance_buffer,
        aov_normal_depth,
        aov_texclr_meshid);

    const auto center_normal{ aten::get<0>(extracted_center_pixel) };
    const float center_depth{ aten::get<1>(extracted_center_pixel) };
    const auto center_meshid{ aten::get<2>(extracted_center_pixel) };
    auto center_color{ aten::get<3>(extracted_center_pixel) };

    auto back_ground_pixel_clr = AT_NAME::svgf::CheckIfBackgroundPixelForAtrous(
        idx, isFinalIter,
        center_meshid, center_color,
        aov_texclr_meshid, next_color_variance_buffer);
    if (back_ground_pixel_clr) {
        if (back_ground_pixel_clr.value()) {
            // Output background color and end the logic.
            const auto& background = back_ground_pixel_clr.value().value();
            surf2Dwrite(
                background,
                dst,
                ix * sizeof(float4), iy,
                cudaBoundaryModeTrap);
            return;
        }
    }

    // 3x3 Gauss filter.
    auto gauss_filtered_variance = AT_NAME::svgf::ComputeGaussFiltereredVariance(
        isFirstIter,
        ix, iy, width, height,
        aov_color_variance,
        color_variance_buffer);

    auto filtered_color_variance{
        AT_NAME::svgf::ExecAtrousWaveletFilter(
            isFirstIter, isFinalIter,
            ix, iy, width, height,
            gauss_filtered_variance,
            center_normal, center_depth, center_meshid, center_color,
            aov_normal_depth, aov_texclr_meshid, aov_color_variance,
            color_variance_buffer,
            stepScale, cameraDistance)
    };

    auto post_process_result = AT_NAME::svgf::PostProcessForAtrousFilter(
        isFirstIter, isFinalIter,
        idx,
        filtered_color_variance,
        aov_texclr_meshid,
        temporary_color_buffer, next_color_variance_buffer);

    if (post_process_result) {
        surf2Dwrite(
            post_process_result.value(),
            dst,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }
}

__global__ void CopyFromTeporaryColorBufferToAov(
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
            temporary_color_buffer_.data(),
            curaov.get<AOVBuffer::NormalDepth>().data(),
            curaov.get<AOVBuffer::AlbedoMeshId>().data(),
            curaov.get<AOVBuffer::ColorVariance>().data(),
            curaov.get<AOVBuffer::MomentTemporalWeight>().data(),
            m_atrousClrVar[cur].data(), m_atrousClrVar[next].data(),
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
        CopyFromTeporaryColorBufferToAov << <grid, block, 0, m_stream >> > (
            temporary_color_buffer_.data(),
            curaov.get<AOVBuffer::ColorVariance>().data(),
            width, height);
        checkCudaKernel(copyFromBufferToAov);
    }
}
