#include "svgf/svgf.h"

#include "kernel/pt_common.h"
#include "kernel/pt_params.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

__global__ void copyBufferForTile(
    const idaten::Path paths,
    float4* contribs,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const auto dstIdx = getIdx(ix, iy, width);

    const auto srcIdx = getIdx(ix, iy, width);

    contribs[dstIdx] = paths.contrib[srcIdx].v;
}

namespace idaten
{
    void SVGFPathTracing::onCopyBufferForTile(int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        copyBufferForTile << <grid, block, 0, m_stream >> > (
            path_host_->paths,
            m_tmpBuf.ptr(),
            width, height);

        checkCudaKernel(copyBufferForTile);
    }
}
