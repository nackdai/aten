#include "asvgf/asvgf.h"

#include "kernel/pt_common.h"
#include "kernel/context.cuh"
#include "sampler/bluenoiseSampler.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

__global__ void initRngSeedsForASVGF(
    uint32_t frame,
    int width, int height,
    unsigned int* rngBuffer_0,
    unsigned int* rngBuffer_1)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= width) {
        return;
    }

    int idx = getIdx(ix, iy, width);

    auto seed = idaten::BlueNoiseSamplerGPU::makeSeed(ix, iy, frame, width, height);
    rngBuffer_0[idx] = seed;
    rngBuffer_1[idx] = seed;
}

namespace idaten {
    void AdvancedSVGFPathTracing::onInitRngSeeds(int width, int height)
    {
        dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        initRngSeedsForASVGF << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_frame,
            width, height,
            m_rngSeed[0].ptr(),
            m_rngSeed[1].ptr());

        checkCudaKernel(initRngSeedsForASVGF);
    }
}
