#include "asvgf/asvgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void displayTiledImageOnRealRes(
    idaten::TileDomain tileDomain,
    int tileSize, 
    cudaSurfaceObject_t dst,
    const float4* __restrict__ src,
    int width, int height)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    const int idx = getIdx(ix, iy, width);

    int tiledX = ix / tileSize;
    int tiledY = iy / tileSize;
    int tiledW = (width + tileSize - 1) / tileSize;

    const int tiledIdx = getIdx(tiledX, tiledY, tiledW);

    auto srcClr = src[tiledIdx];

    float4 dstClr = srcClr.y > 0.0f
        ? make_float4(0.0f, srcClr.y, 0.0f, 1.0f)
        : make_float4(abs(srcClr.y), 0.0f, 0.0f, 1.0f);

    surf2Dwrite(
        dstClr,
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

namespace idaten
{

}