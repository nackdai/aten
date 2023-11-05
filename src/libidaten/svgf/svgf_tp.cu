#include "svgf/svgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/svgf/svgf_impl.h"

//#define ENABLE_MEDIAN_FILTER

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

inline __device__ int32_t getLinearIdx(int32_t x, int32_t y, int32_t w, int32_t h)
{
    int32_t max_buffer_size = w * h;
    return clamp(y * w + x, 0, max_buffer_size - 1);
}

// Bilinear sampler
inline __device__ float4 sampleBilinear(
    const float4* buffer,
    float uvx, float uvy,
    int32_t w, int32_t h)
{
    float2 uv = make_float2(uvx, uvy) * make_float2(w, h) - make_float2(0.5f, 0.5f);

    int32_t x = floor(uv.x);
    int32_t y = floor(uv.y);

    float2 uv_ratio = uv - make_float2(x, y);
    float2 uv_inv = make_float2(1.f, 1.f) - uv_ratio;

    int32_t x1 = clamp(x + 1, 0, w - 1);
    int32_t y1 = clamp(y + 1, 0, h - 1);

    float4 r = (buffer[getLinearIdx(x, y, w, h)] * uv_inv.x + buffer[getLinearIdx(x1, y, w, h)] * uv_ratio.x) * uv_inv.y +
        (buffer[getLinearIdx(x, y1, w, h)] * uv_inv.x + buffer[getLinearIdx(x1, y1, w, h)] * uv_ratio.x) * uv_ratio.y;

    return r;
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

    auto contribs{ aten::span<float4>(const_cast<float4*>(contributes), width * height) };
    auto curr_aov_normal_depth{ aten::span<float4>(curAovNormalDepth, width * height) };
    auto curr_aov_texclr_meshid{ aten::span<float4>(curAovTexclrMeshid, width * height) };
    auto curr_aov_color_variance{ aten::span<float4>(curAovColorVariance, width * height) };
    auto curr_aov_moment_temporalweight{ aten::span<float4>(curAovMomentTemporalWeight, width * height) };
    auto prev_aov_normal_depth{ aten::span<float4>(const_cast<float4*>(prevAovNormalDepth), width * height) };
    auto prev_aov_texclr_meshid{ aten::span<float4>(const_cast<float4*>(prevAovTexclrMeshid), width * height) };
    auto prev_aov_color_variance{ aten::span<float4>(const_cast<float4*>(prevAovColorVariance), width * height) };
    auto prev_aov_moment_temporalweight{ aten::span<float4>(const_cast<float4*>(prevAovMomentTemporalWeight), width * height) };

    auto extracted_center_pixel = AT_NAME::svgf::ExtractCenterPixel(
        idx,
        contribs,
        curr_aov_normal_depth,
        curr_aov_texclr_meshid);

    const auto center_meshid = aten::get<2>(extracted_center_pixel);
    auto curr_color = aten::get<3>(extracted_center_pixel);

    auto back_ground_pixel_clr = AT_NAME::svgf::CheckIfPixelIsBackground(
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

    const auto center_normal = aten::get<0>(extracted_center_pixel);
    const float center_depth = aten::get<1>(extracted_center_pixel);

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

__global__ void dilateWeight(
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

    const int32_t centerMeshId = (int32_t)aovTexclrMeshid[idx].w;

    if (centerMeshId < 0) {
        // This pixel is background, so nothing is done.
        return;
    }

    float temporalWeight = aovMomentTemporalWeight[idx].w;

    for (int32_t y = -1; y <= 1; y++) {
        for (int32_t x = -1; x <= 1; x++) {
            int32_t xx = ix + x;
            int32_t yy = iy + y;

            if ((0 <= xx) && (xx < width)
                && (0 <= yy) && (yy < height))
            {
                int32_t pidx = getIdx(xx, yy, width);
                float w = aovMomentTemporalWeight[pidx].w;
                temporalWeight = min(temporalWeight, w);
            }
        }
    }

    aovMomentTemporalWeight[idx].w = temporalWeight;
}

inline __device__ float3 min(float3 a, float3 b)
{
    return make_float3(
        min(a.x, b.x),
        min(a.y, b.y),
        min(a.z, b.z));
}

inline __device__ float3 max(float3 a, float3 b)
{
    return make_float3(
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z));
}

// Macro for sorting.
#define s2(a, b)                temp = a; a = min(a, b); b = max(temp, b);
#define mn3(a, b, c)            s2(a, b); s2(a, c);
#define mx3(a, b, c)            s2(b, c); s2(a, c);

#define mnmx3(a, b, c)            mx3(a, b, c); s2(a, b);                                   // 3 exchanges
#define mnmx4(a, b, c, d)        s2(a, b); s2(c, d); s2(a, c); s2(b, d);                   // 4 exchanges
#define mnmx5(a, b, c, d, e)    s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges

inline __device__ float3 medianFilter(
    int32_t ix, int32_t iy,
    const float4* src,
    int32_t width, int32_t height)
{
    float3 v[9];

    int32_t pos = 0;

    for (int32_t y = -1; y <= 1; y++) {
        for (int32_t x = -1; x <= 1; x++) {
            int32_t xx = clamp(ix + x, 0, static_cast<int32_t>(width - 1));
            int32_t yy = clamp(iy + y, 0, static_cast<int32_t>(height - 1));

            int32_t pidx = getIdx(xx, yy, width);

            auto s = src[pidx];
            v[pos] = make_float3(s.x, s.y, s.z);

            pos++;
        }
    }

    // Sort
    float3 temp;
    mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
    mnmx5(v[1], v[2], v[3], v[4], v[6]);
    mnmx4(v[2], v[3], v[4], v[7]);
    mnmx3(v[3], v[4], v[8]);

    return v[4];
}

__global__ void medianFilter(
    cudaSurfaceObject_t dst,
    float4* curAovColorVariance,
    float4* curAovMomentTemporalWeight,
    const float4* __restrict__ curAovTexclrMeshid,
    const float4* __restrict__ prevAovMomentTemporalWeight,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    auto idx = getIdx(ix, iy, width);

    const int32_t centerMeshId = curAovTexclrMeshid[idx].w;

    if (centerMeshId < 0) {
        // This pixel is background, so nothing is done.
        return;
    }

    auto curColor = medianFilter(ix, iy, curAovColorVariance, width, height);

    curAovColorVariance[idx].x = curColor.x;
    curAovColorVariance[idx].y = curColor.y;
    curAovColorVariance[idx].z = curColor.z;

    // accumulate moments.
    {
        float lum = AT_NAME::color::luminance(curColor.x, curColor.y, curColor.z);
        float3 centerMoment = make_float3(lum * lum, lum, 0);

        // 積算フレーム数のリセット.
        int32_t frame = 1;

        auto momentTemporalWeight = prevAovMomentTemporalWeight[idx];;
        float3 prevMoment = make_float3(momentTemporalWeight.x, momentTemporalWeight.y, momentTemporalWeight.z);

        // 積算フレーム数を１増やす.
        frame = (int32_t)prevMoment.z + 1;

        centerMoment += prevMoment;

        centerMoment.z = frame;

        curAovMomentTemporalWeight[idx].x = centerMoment.x;
        curAovMomentTemporalWeight[idx].y = centerMoment.y;
        curAovMomentTemporalWeight[idx].z = centerMoment.z;
    }

    surf2Dwrite(
        make_float4(curColor, 0),
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
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

#ifdef ENABLE_MEDIAN_FILTER
        medianFilter << <grid, block, 0, m_stream >> > (
            outputSurf,
            m_aovColorVariance[curaov].data(),
            m_aovMomentTemporalWeight[curaov].data(),
            m_aovTexclrMeshid[curaov].data(),
            m_aovMomentTemporalWeight[prevaov].data(),
            width, height);

        checkCudaKernel(medianFilter);
#endif

        dilateWeight << <grid, block, 0, m_stream >> > (
            curaov.get<AOVBuffer::MomentTemporalWeight>().data(),
            curaov.get<AOVBuffer::AlbedoMeshId>().data(),
            width, height);
        checkCudaKernel(dilateWeight);
    }
}
