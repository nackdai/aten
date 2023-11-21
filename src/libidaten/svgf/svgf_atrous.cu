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
    int32_t filter_iter_count,
    int32_t width, int32_t height,
    float camera_distance)
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
        isFinalIter, idx,
        center_color,
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
    auto gauss_filtered_variance = AT_NAME::svgf::Exec3x3GaussFilter<&float4::w>(
        ix, iy, width, height,
        isFirstIter ? aov_color_variance : color_variance_buffer);

    auto filtered_color_variance{
        AT_NAME::svgf::ExecAtrousWaveletFilter(
            isFirstIter,
            ix, iy, width, height,
            gauss_filtered_variance,
            center_normal, center_depth, center_meshid, center_color,
            aov_normal_depth, aov_texclr_meshid, aov_color_variance,
            color_variance_buffer,
            filter_iter_count, camera_distance)
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
    const float4* __restrict__ src_buffer,
    float4* aovColorVariance,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const int32_t idx = getIdx(ix, iy, width);

    const auto size = width * height;

    auto src{ aten::const_span<float4>(src_buffer, size) };
    auto dst{ aten::span<float4>(aovColorVariance, size) };

    AT_NAME::svgf::CopyVectorBuffer<3>(idx, src, dst);
}

namespace idaten
{
    void SVGFPathTracing::onAtrousFilter(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height)
    {
        for (int32_t i = 0; i < params_.atrous_iter_cnt; i++) {
            onAtrousFilterIter(
                i,
                outputSurf,
                width, height);
        }
    }

    void SVGFPathTracing::onAtrousFilterIter(
        uint32_t filter_iter_count,
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        auto& curaov = params_.GetCurrAovBuffer();

        auto atrous_clr_variance_buffers = params_.GetAtrousColorVariance(filter_iter_count);
        auto& curr_atrous_clr_variance = aten::get<0>(atrous_clr_variance_buffers);
        auto& next_atrous_clr_variance = aten::get<1>(atrous_clr_variance_buffers);

        bool isFirstIter = (filter_iter_count == 0);
        bool isFinalIter = (filter_iter_count == params_.atrous_iter_cnt - 1);

        auto camera_distance = AT_NAME::camera::ComputeScreenDistance(m_cam, height);

        atrousFilter << <grid, block, 0, m_stream >> > (
            isFirstIter, isFinalIter,
            outputSurf,
            params_.temporary_color_buffer.data(),
            curaov.get<AT_NAME::SVGFAovBufferType::NormalDepth>().data(),
            curaov.get<AT_NAME::SVGFAovBufferType::AlbedoMeshId>().data(),
            curaov.get<AT_NAME::SVGFAovBufferType::ColorVariance>().data(),
            curaov.get<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>().data(),
            curr_atrous_clr_variance.data(),
            next_atrous_clr_variance.data(),
            filter_iter_count,
            width, height,
            camera_distance);
        checkCudaKernel(atrousFilter);
    }

    void SVGFPathTracing::onCopyFromTmpBufferToAov(int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        auto& curaov = params_.GetCurrAovBuffer();

        // Copy color from temporary buffer to AOV buffer for next temporal reprojection.
        CopyFromTeporaryColorBufferToAov << <grid, block, 0, m_stream >> > (
            params_.temporary_color_buffer.data(),
            curaov.get<AT_NAME::SVGFAovBufferType::ColorVariance>().data(),
            width, height);
        checkCudaKernel(copyFromBufferToAov);
    }
}
