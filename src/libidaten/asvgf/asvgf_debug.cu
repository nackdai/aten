#include "asvgf/asvgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

#if 0
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
    void AdvancedSVGFPathTracing::displayTiledData(
        int width, int height,
        TypedCudaMemory<float4>& src,
        cudaSurfaceObject_t outputSurf)
    {
        // TODO
        // •ªŠ„•`‰æ.

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        displayTiledImageOnRealRes << <grid, block >> > (
            m_tileDomain,
            m_gradientTileSize,
            outputSurf,
            src.ptr(),
            width, height);

        checkCudaKernel(displayTiledImageOnRealRes);
    }
}
#endif