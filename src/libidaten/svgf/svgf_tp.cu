#include "svgf/svgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

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
    // mtxV2C = W 0 0  0
    //          0 H 0  0
    //          0 0 A  B
    //          0 0 -1 0
    // mtxV2C * Pview = (Xclip, Yclip, Zclip, Wclip) = (Xclip, Yclip, Zclip, Zview)
    //  Wclip = Zview = depth
    // Xscr = Xclip / Wclip = Xclip / Zview = Xclip / depth
    // Yscr = Yclip / Wclip = Yclip / Zview = Yclip / depth
    //
    // Xscr * depth = Xclip
    // Xview = mtxC2V * Xclip

    const aten::mat4 mtxC2V = mtxs[0];
    const aten::mat4 mtxV2W = mtxs[1];
    const aten::mat4 mtxPrevW2V = mtxs[2];
    const aten::mat4 mtxV2C = mtxs[3];

    float2 uv = make_float2(ix + 0.5, iy + 0.5);
    uv /= make_float2(width - 1, height - 1);    // [0, 1]
    uv = uv * 2.0f - 1.0f;    // [0, 1] -> [-1, 1]

    aten::vec4 pos(uv.x, uv.y, 0, 0);

    // Screen-space -> Clip-space.
    pos.x *= centerDepth;
    pos.y *= centerDepth;

    // Clip-space -> View-space
    pos = mtxC2V.apply(pos);
    pos.z = -centerDepth;
    pos.w = 1.0;

    pos = mtxV2W.apply(pos);

    // Reproject previous screen position.
    pos = mtxPrevW2V.apply(pos);
    *prevPos = mtxV2C.apply(pos);
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
    idaten::TileDomain tileDomain,
    const float nThreshold,
    const float zThreshold,
    const float4* __restrict__ contribs,
    const aten::CameraParameter* __restrict__ camera,
    float4* curAovNormalDepth,
    float4* curAovTexclrMeshid,
    float4* curAovColorVariance,
    float4* curAovMomentTemporalWeight,
    const float4* __restrict__ prevAovNormalDepth,
    const float4* __restrict__ prevAovTexclrMeshid,
    const float4* __restrict__ prevAovColorVariance,
    const float4* __restrict__ prevAovMomentTemporalWeight,
    cudaSurfaceObject_t motionDetphBuffer,
    cudaSurfaceObject_t dst,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    const auto idx = getIdx(ix, iy, width);

    auto nmlDepth = curAovNormalDepth[idx];
    auto texclrMeshId = curAovTexclrMeshid[idx];

    const float centerDepth = nmlDepth.w;
    const int32_t centerMeshId = (int32_t)texclrMeshId.w;

    // 今回のフレームのピクセルカラー.
    auto contrib = contribs[idx];
    float4 curColor = make_float4(contrib.x, contrib.y, contrib.z, 1.0f) / contrib.w;
    //curColor.w = 1;

    if (centerMeshId < 0) {
        // 背景なので、そのまま出力して終わり.
        surf2Dwrite(
            curColor,
            dst,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);

        curAovColorVariance[idx] = curColor;
        curAovMomentTemporalWeight[idx] = make_float4(1, 1, 1, curAovMomentTemporalWeight[idx].w);

        return;
    }

    float3 centerNormal = make_float3(nmlDepth.x, nmlDepth.y, nmlDepth.z);

    float4 sum = make_float4(0);
    float weight = 0.0f;

    aten::vec4 centerPrevPos;

#pragma unroll
    for (int32_t y = -1; y <= 1; y++) {
        for (int32_t x = -1; x <= 1; x++) {
            int32_t xx = clamp(ix + x, 0, width - 1);
            int32_t yy = clamp(iy + y, 0, height - 1);

            float4 motionDepth;
            surf2Dread(&motionDepth, motionDetphBuffer, ix * sizeof(float4), iy);

            // 前のフレームのスクリーン座標.
            int32_t px = (int32_t)(xx + motionDepth.x * width);
            int32_t py = (int32_t)(yy + motionDepth.y * height);

            px = clamp(px, 0, width - 1);
            py = clamp(py, 0, height - 1);

            int32_t pidx = getIdx(px, py, width);

            nmlDepth = prevAovNormalDepth[pidx];
            texclrMeshId = prevAovTexclrMeshid[pidx];

            const float prevDepth = nmlDepth.w;
            const int32_t prevMeshId = (int32_t)texclrMeshId.w;
            float3 prevNormal = make_float3(nmlDepth.x, nmlDepth.y, nmlDepth.z);

            // TODO
            // 同じメッシュ上でもライトのそばの明るくなったピクセルを拾ってしまう場合の対策が必要.

            float Wz = clamp((zThreshold - abs(1 - centerDepth / prevDepth)) / zThreshold, 0.0f, 1.0f);
            float Wn = clamp((dot(centerNormal, prevNormal) - nThreshold) / (1.0f - nThreshold), 0.0f, 1.0f);
            float Wm = centerMeshId == prevMeshId ? 1.0f : 0.0f;

            // 前のフレームのピクセルカラーを取得.
            float4 prev = prevAovColorVariance[pidx];
            //float4 prev = sampleBilinear(prevAovColorVariance, prevPos.x, prevPos.y, width, height);

            float W = Wz * Wn * Wm;
            sum += prev * W;
            weight += W;
        }
    }

    if (weight > 0.0f) {
        sum /= weight;
        weight /= 9;
#if 0
        auto w = min(0.8f, weight);
        curColor = (1.0f - w) * curColor + w * sum;
#elif 1
        curColor = 0.2 * curColor + 0.8 * sum;
#else
        curColor = (1.0f - weight) * curColor + weight * sum;
#endif
    }

    curAovMomentTemporalWeight[idx].w = weight;

#ifdef ENABLE_MEDIAN_FILTER
    curAovColorVariance[idx].x = curColor.x;
    curAovColorVariance[idx].y = curColor.y;
    curAovColorVariance[idx].z = curColor.z;
#else
    curAovColorVariance[idx].x = curColor.x;
    curAovColorVariance[idx].y = curColor.y;
    curAovColorVariance[idx].z = curColor.z;

    // TODO
    // 現フレームと過去フレームが同率で加算されるため、どちらかに強い影響がでると影響が弱まるまでに非常に時間がかかる.
    // ex)
    // f0 = 100, f1 = 0, f2 = 0
    // avg = (f0 + f1 + f2) / 3 = 33.3 <- 非常に大きい値が残り続ける.

    // accumulate moments.
    {
        float lum = AT_NAME::color::luminance(curColor.x, curColor.y, curColor.z);
        float3 centerMoment = make_float3(lum * lum, lum, 0);

        // 積算フレーム数のリセット.
        int32_t frame = 1;

        if (weight > 0.0f) {
            auto momentTemporalWeight = prevAovMomentTemporalWeight[idx];;
            float3 prevMoment = make_float3(momentTemporalWeight.x, momentTemporalWeight.y, momentTemporalWeight.z);

            // 積算フレーム数を１増やす.
            frame = (int32_t)prevMoment.z + 1;

            centerMoment += prevMoment;
        }

        centerMoment.z = frame;

        curAovMomentTemporalWeight[idx].x = centerMoment.x;
        curAovMomentTemporalWeight[idx].y = centerMoment.y;
        curAovMomentTemporalWeight[idx].z = centerMoment.z;
    }
#endif

    surf2Dwrite(
        curColor,
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

__global__ void dilateWeight(
    idaten::TileDomain tileDomain,
    float4* aovMomentTemporalWeight,
    const float4* __restrict__ aovTexclrMeshid,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

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
            int32_t xx = clamp(ix + x, 0, width - 1);
            int32_t yy = clamp(iy + y, 0, height - 1);

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
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        int32_t curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        int32_t prevaov_idx = getPrevAovs();
        auto& prevaov = aov_[prevaov_idx];

        CudaGLResourceMapper<decltype(m_motionDepthBuffer)> rscmap(m_motionDepthBuffer);
        auto motionDepthBuffer = m_motionDepthBuffer.bind();

        temporalReprojection << <grid, block, 0, m_stream >> > (
        //temporalReprojection << <1, 1 >> > (
            m_tileDomain,
            m_nmlThresholdTF,
            m_depthThresholdTF,
            m_tmpBuf.ptr(),
            m_cam.ptr(),
            curaov.get<AOVBuffer::NormalDepth>().ptr(),
            curaov.get<AOVBuffer::AlbedoMeshId>().ptr(),
            curaov.get<AOVBuffer::ColorVariance>().ptr(),
            curaov.get<AOVBuffer::MomentTemporalWeight>().ptr(),
            prevaov.get<AOVBuffer::NormalDepth>().ptr(),
            prevaov.get<AOVBuffer::AlbedoMeshId>().ptr(),
            prevaov.get<AOVBuffer::ColorVariance>().ptr(),
            prevaov.get<AOVBuffer::MomentTemporalWeight>().ptr(),
            motionDepthBuffer,
            outputSurf,
            width, height);

        checkCudaKernel(temporalReprojection);

#ifdef ENABLE_MEDIAN_FILTER
        medianFilter << <grid, block, 0, m_stream >> > (
            outputSurf,
            m_aovColorVariance[curaov].ptr(),
            m_aovMomentTemporalWeight[curaov].ptr(),
            m_aovTexclrMeshid[curaov].ptr(),
            m_aovMomentTemporalWeight[prevaov].ptr(),
            width, height);

        checkCudaKernel(medianFilter);
#endif

        dilateWeight << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            curaov.get<AOVBuffer::MomentTemporalWeight>().ptr(),
            curaov.get<AOVBuffer::AlbedoMeshId>().ptr(),
            width, height);
        checkCudaKernel(dilateWeight);
    }
}
