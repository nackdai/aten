#include "svgf/svgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"
#include "renderer/svgf/svgf_impl.h"

inline __device__ void computePrevScreenPos(
    int32_t ix, int32_t iy,
    float centerDepth,
    int32_t width, int32_t height,
    aten::vec4* prevPos,
    const aten::mat4* __restrict__ mtxs)
{
    // NOTE
    // Pview = (Xview, Yview, Zview, 1)
    // mtx_V2C = W 0 0  0
    //          0 H 0  0
    //          0 0 A  B
    //          0 0 -1 0
    // mtx_V2C * Pview = (Xclip, Yclip, Zclip, Wclip) = (Xclip, Yclip, Zclip, Zview)
    //  Wclip = Zview = depth
    // Xscr = Xclip / Wclip = Xclip / Zview = Xclip / depth
    // Yscr = Yclip / Wclip = Yclip / Zview = Yclip / depth
    //
    // Xscr * depth = Xclip
    // Xview = mtx_C2V * Xclip

    const aten::mat4 mtx_C2V = mtxs[0];
    const aten::mat4 mtx_V2W = mtxs[1];
    const aten::mat4 mtx_prev_W2V = mtxs[2];
    const aten::mat4 mtx_V2C = mtxs[3];

    float2 uv = make_float2(ix + 0.5, iy + 0.5);
    uv /= make_float2(width - 1, height - 1);    // [0, 1]
    uv = uv * 2.0f - 1.0f;    // [0, 1] -> [-1, 1]

    aten::vec4 pos(uv.x, uv.y, 0, 0);

    // Screen-space -> Clip-space.
    pos.x *= centerDepth;
    pos.y *= centerDepth;

    // Clip-space -> View-space
    pos = mtx_C2V.apply(pos);
    pos.z = -centerDepth;
    pos.w = 1.0;

    pos = mtx_V2W.apply(pos);

    // Reproject previous screen position.
    pos = mtx_prev_W2V.apply(pos);
    *prevPos = mtx_V2C.apply(pos);
    *prevPos /= prevPos->w;

    *prevPos = *prevPos * 0.5 + 0.5;    // [-1, 1] -> [0, 1]
}

__global__ void temporalReprojection(
    const float threshold_normal,
    const float threshold_depth,
    const float4* __restrict__ contributes,
    const aten::CameraParameter camera,
    float4* curAovNormalDepth,
    float4* curAovTexclrMeshid,
    float4* curAovColorVariance,
    float4* curAovMomentTemporalWeight,
    const float4* __restrict__ prevAovNormalDepth,
    const float4* __restrict__ prevAovTexclrMeshid,
    const float4* __restrict__ prevAovColorVariance,
    const float4* __restrict__ prevAovMomentTemporalWeight,
    cudaSurfaceObject_t motion_detph_buffer,
    cudaSurfaceObject_t dst,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const auto idx = getIdx(ix, iy, width);

    const size_t size = width * height;

    auto contribs{ aten::const_span<float4>(contributes, size) };
    auto curr_aov_normal_depth{ aten::span<float4>(curAovNormalDepth, size) };
    auto curr_aov_texclr_meshid{ aten::span<float4>(curAovTexclrMeshid, size) };
    auto curr_aov_color_variance{ aten::span<float4>(curAovColorVariance, size) };
    auto curr_aov_moment_temporalweight{ aten::span<float4>(curAovMomentTemporalWeight, size) };
    auto prev_aov_normal_depth{ aten::const_span<float4>(prevAovNormalDepth, size) };
    auto prev_aov_texclr_meshid{ aten::const_span<float4>(prevAovTexclrMeshid, size) };
    auto prev_aov_color_variance{ aten::const_span<float4>(prevAovColorVariance, size) };
    auto prev_aov_moment_temporalweight{ aten::const_span<float4>(prevAovMomentTemporalWeight, size) };

    auto extracted_center_pixel = AT_NAME::svgf::ExtractCenterPixel(
        idx,
        contribs,
        curr_aov_normal_depth,
        curr_aov_texclr_meshid);

    const auto center_meshid{ aten::get<2>(extracted_center_pixel) };
    auto curr_color{ aten::get<3>(extracted_center_pixel) };

    auto back_ground_pixel_clr = AT_NAME::svgf::CheckIfBackgroundPixel(
        idx, curr_color, center_meshid,
        curr_aov_color_variance, curr_aov_moment_temporalweight);
    if (back_ground_pixel_clr) {
        surf2Dwrite(
            back_ground_pixel_clr.value(),
            dst,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
        return;
    }

    const auto center_normal{ aten::get<0>(extracted_center_pixel) };
    const float center_depth{ aten::get<1>(extracted_center_pixel) };

    auto weight = AT_NAME::svgf::TemporalReprojection(
        ix, iy, width, height,
        threshold_normal, threshold_depth,
        center_normal, center_depth, center_meshid,
        curr_color,
        curr_aov_color_variance, curr_aov_moment_temporalweight,
        prev_aov_normal_depth, prev_aov_texclr_meshid, prev_aov_color_variance,
        motion_detph_buffer);

    AT_NAME::svgf::AccumulateMoments(
        idx, weight,
        curr_aov_color_variance,
        curr_aov_moment_temporalweight,
        prev_aov_moment_temporalweight);

    surf2Dwrite(
        curr_color,
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

__global__ void PropagateTemporalWeight(
    float4* aovMomentTemporalWeight,
    const float4* __restrict__ aovTexclrMeshid,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    auto idx = getIdx(ix, iy, width);

    const size_t size = width * height;

    auto aov_texclr_meshid{ aten::const_span<float4>(aovTexclrMeshid, size) };
    auto aov_moment_temporalweight{ aten::const_span<float4>(aovMomentTemporalWeight, size) };

    auto weight = AT_NAME::svgf::PropagateTemporalWeight(
        ix, iy, width, height,
        aov_texclr_meshid,
        aov_moment_temporalweight);

    if (weight) {
        aovMomentTemporalWeight[idx].w = weight.value();
    }
}

namespace idaten
{
    void SVGFPathTracing::onTemporalReprojection(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        int32_t curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        int32_t prevaov_idx = getPrevAovs();
        auto& prevaov = aov_[prevaov_idx];

        CudaGLResourceMapper<decltype(m_motionDepthBuffer)> rscmap(m_motionDepthBuffer);
        auto motionDepthBuffer = m_motionDepthBuffer.bind();

        temporalReprojection << <grid, block, 0, m_stream >> > (
        //temporalReprojection << <1, 1 >> > (
            m_nmlThresholdTF,
            m_depthThresholdTF,
            temporary_color_buffer_.data(),
            m_cam,
            curaov.get<AOVBuffer::NormalDepth>().data(),
            curaov.get<AOVBuffer::AlbedoMeshId>().data(),
            curaov.get<AOVBuffer::ColorVariance>().data(),
            curaov.get<AOVBuffer::MomentTemporalWeight>().data(),
            prevaov.get<AOVBuffer::NormalDepth>().data(),
            prevaov.get<AOVBuffer::AlbedoMeshId>().data(),
            prevaov.get<AOVBuffer::ColorVariance>().data(),
            prevaov.get<AOVBuffer::MomentTemporalWeight>().data(),
            motionDepthBuffer,
            outputSurf,
            width, height);

        checkCudaKernel(temporalReprojection);

        PropagateTemporalWeight << <grid, block, 0, m_stream >> > (
            curaov.get<AOVBuffer::MomentTemporalWeight>().data(),
            curaov.get<AOVBuffer::AlbedoMeshId>().data(),
            width, height);
        checkCudaKernel(dilateWeight);
    }
}
