#include "asvgf/asvgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

#if 0
__device__ inline bool isEqualInt2(const int2& a, const int2& b)
{
    return (a.x == b.x) && (a.y == b.y);
}

__global__ void createGradient(
    idaten::TileDomain tileDomain,
    int tileSize,
    float4* outGradient,
    const idaten::SVGFPathTracing::Path* __restrict__ paths,
    const float4* __restrict__ prevAovColorUnfiltered,
    const float4* __restrict__ curAovTexclrMeshid,
    const int4* __restrict__ gradientSample,
    int width, int height,
    int widthInRealRes, int heightInRealRes)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    const int idx = getIdx(ix, iy, width);

    // Get positon in a tile.
    int2 tilePos = make_int2(gradientSample[idx].x, gradientSample[idx].y);

    // Convert tile positon to real resolution position.
    int2 posInRealRes = make_int2(ix, iy) * tileSize + tilePos;

    posInRealRes.x = aten::clamp(posInRealRes.x, 0, widthInRealRes - 1);
    posInRealRes.y = aten::clamp(posInRealRes.y, 0, heightInRealRes - 1);

    const int curIdxInRealRes = getIdx(posInRealRes.x, posInRealRes.y, widthInRealRes);

    auto curColor = paths->contrib[curIdxInRealRes].v;
    float curLum = AT_NAME::color::luminance(curColor.x, curColor.y, curColor.z);

    outGradient[idx] = make_float4(0.0f);

    // Get previous frame index in real resolution.
    const int prevIdxInRealRes = gradientSample[idx].z;

    // Only previous sample position is in resolution.
    if (prevIdxInRealRes >= 0) {
        auto prevColor = prevAovColorUnfiltered[prevIdxInRealRes];
        float prevLum = AT_NAME::color::luminance(prevColor.x, prevColor.y, prevColor.z);

        outGradient[idx].x = max(curLum, prevLum);
        outGradient[idx].y = curLum - prevLum;
    }

    float2 moments = make_float2(curLum, curLum * curLum);
    float sumW = 1.0f;
    int centerMeshId = (int)curAovTexclrMeshid[curIdxInRealRes].w;

    // Compute moment and variance in a tile.
    for (int yy = 0; yy < tileSize; yy++) {
        for (int xx = 0; xx < tileSize; xx++) {
            int2 p = make_int2(ix, iy) * tileSize + make_int2(xx, yy);

            p.x = aten::clamp(p.x, 0, widthInRealRes - 1);
            p.y = aten::clamp(p.y, 0, heightInRealRes - 1);

            if (!isEqualInt2(posInRealRes, p)) {
                int pidx = getIdx(p.x, p.y, widthInRealRes);

                auto clr = paths->contrib[pidx].v;
                int meshId = (int)curAovTexclrMeshid[pidx].w;

                float l = AT_NAME::color::luminance(clr.x, clr.y, clr.z);
                float w = (centerMeshId == meshId ? 1.0f : 0.0f);

                moments += make_float2(l, l * l) * w;
                sumW += w;
            }
        }
    }

    moments /= sumW;

    float variance = max(0.0f, moments.y - moments.x * moments.x);

    outGradient[idx].z = moments.x;
    outGradient[idx].w = variance;
}

namespace idaten
{
    void AdvancedSVGFPathTracing::onCreateGradient(int width, int height)
    {
        // TODO
        // 分割描画.

        int tiledW = getTiledResolution(width);
        int tiledH = getTiledResolution(height);

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (tiledW + block.x - 1) / block.x,
            (tiledH + block.y - 1) / block.y);

        float cameraDistance = height / (2.0f * aten::tan(0.5f * m_camParam.vfov));

        int curaov = getCurAovs();
        int prevaov = getPrevAovs();

        createGradient << <grid, block >> > (
            m_tileDomain,
            m_gradientTileSize,
            m_gradient.ptr(),
            m_paths.ptr(),
            m_aovColorVariance[prevaov].ptr(),
            m_aovTexclrMeshid[curaov].ptr(),
            m_gradientSample.ptr(),
            tiledW, tiledH,
            width, height);

        checkCudaKernel(createGradient);
    }
}
#endif