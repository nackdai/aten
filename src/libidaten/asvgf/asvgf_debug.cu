#include "asvgf/asvgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void displayBluenoiseTexture(
    cudaSurfaceObject_t dst,
    int width, int height,
    cudaTextureObject_t tex)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const int idx = getIdx(ix, iy, width);

    float u = (float)ix / width;
    float v = (float)iy / height;

    float4 dstClr = tex2DLayered<float4>(tex, u, v, 0);

    surf2Dwrite(
        dstClr,
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

namespace idaten
{
    void AdvancedSVGFPathTracing::onDebug(
        int width, int height,
        cudaSurfaceObject_t outputSurf)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        auto tex = m_bluenoise.bind();

        displayBluenoiseTexture << <grid, block >> > (
            outputSurf,
            width, height,
            tex);

        checkCudaKernel(displayBluenoiseTexture);

        m_bluenoise.unbind();
    }
}
