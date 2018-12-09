#include "asvgf/asvgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

inline __device__ float _gaussFilter3x3(
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

__global__ void atrousGradient(
    idaten::TileDomain tileDomain,
    cudaSurfaceObject_t dst,
    int tileSize,
    const float4* __restrict__ gradient,
    const float4* __restrict__ curAovNormalDepth,
    float4* nextTo,
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

    float centerLum = gradient[idx].z;
    float centerVariance = gradient[idx].w;

    float gaussedVarLum = _gaussFilter3x3(ix, iy, width, height, gradient);
    float sqrGaussedVarLum = sqrt(gaussedVarLum);

    float centerDepth = curAovNormalDepth[idx].w;

    float2 sumClr = make_float2(gradient[idx].x, gradient[idx].y);
    float sumLum = centerLum;
    float sumVariance = centerVariance;
    float sumW = 1.0f;

    static const int r = 1;
    
    for (int yy = -r; yy <= r; yy ++) {
        for (int xx = -r; xx <= r; xx++) {
            if (xx != 0 || yy != 0) {
                int x = aten::clamp(ix + xx * stepScale, 0, width - 1);
                int y = aten::clamp(iy + yy * stepScale, 0, height - 1);

                const int pidx = getIdx(x, y, width);

                float2 color = make_float2(gradient[pidx].x, gradient[pidx].y);
                auto depth = curAovNormalDepth[pidx].w;
                auto luminance = gradient[pidx].z;
                auto variance = gradient[pidx].w;

                float2 offset = make_float2(xx, yy) * stepScale;

                float Wl = abs(luminance - centerLum) / (sqrGaussedVarLum + 1e-10f);
                float Wz = abs(depth - centerDepth) / (cameraDistance * length(offset) * tileSize + 1e-2f);

                float w = expf(-Wl * Wl - Wz) * h[i];

                sumClr += color * w;
                sumLum += luminance * w;
                sumVariance += variance * w * w;
                sumW += w;
            }
        }
    }

    sumClr /= sumW;
    sumLum /= sumW;
    sumVariance /= (sumW * sumW);

    nextTo[idx] = make_float4(sumClr.x, sumClr.y, sumLum, sumVariance);

    surf2Dwrite(
        nextTo[idx].y >= 0.0f ? make_float4(nextTo[idx].y, 0.0f, 0.0f, 1.0f) : make_float4(0.0f, abs(nextTo[idx].y), 0.0f, 1.0f),
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

namespace idaten
{

}