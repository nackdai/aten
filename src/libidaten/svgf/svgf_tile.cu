#include "svgf/svgf.h"

#include "kernel/pt_common.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

__global__ void copyBufferForTile(
    idaten::TileDomain tileDomain,
    const idaten::Path paths,
    float4* contribs,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    const auto dstIdx = getIdx(
        ix + tileDomain.x,
        iy + tileDomain.y,
        width);

    const auto srcIdx = getIdx(ix, iy, tileDomain.w);

    contribs[dstIdx] = paths.contrib[srcIdx].v;
}

namespace idaten
{
    void SVGFPathTracing::onCopyBufferForTile(int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        copyBufferForTile << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            m_paths,
            m_tmpBuf.ptr(),
            width, height);

        checkCudaKernel(copyBufferForTile);
    }
}
