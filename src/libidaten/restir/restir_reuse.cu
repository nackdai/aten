#include "restir/restir.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__host__ __device__ void OnComputeSpatialReuse(
    int idx,
    aten::sampler* sampler,
    const idaten::Reservoir* reservoirs,
    idaten::Reservoir* dst_reservoirs,
    const idaten::ReSTIRIntermedidate* intermediates,
    idaten::ReSTIRIntermedidate* dst_intermediates,
    int width, int height)
{
    int ix = idx % width;
    int iy = idx / width;

    int reuse_idx = -1;
    auto new_reservoir = reservoirs[idx];

    auto r = sampler->nextSample();

#pragma unroll
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            const auto xx = ix + x;
            const auto yy = iy + y;

            if (AT_MATH_IS_IN_BOUND(xx, 0, width - 1)
                && AT_MATH_IS_IN_BOUND(yy, 0, height - 1))
            {
                auto new_idx = getIdx(xx, yy, width);
                const auto& reservoir = reservoirs[new_idx];

                if (reservoir.w > 0.0f) {
                    new_reservoir.w += reservoir.w;
                    new_reservoir.m += reservoir.m;

                    if (r <= reservoir.w / new_reservoir.w) {
                        new_reservoir.light_pdf = reservoir.light_pdf;
                        new_reservoir.light_idx = reservoir.light_idx;
                        reuse_idx = new_idx;
                    }
                }
            }
        }
    }

    if (reuse_idx >= 0) {
        dst_reservoirs[idx] = new_reservoir;

        dst_intermediates[idx].light_sample_nml = intermediates[reuse_idx].light_sample_nml;
        dst_intermediates[idx].light_color = intermediates[reuse_idx].light_color;
    }
    else {
        dst_reservoirs[idx] = reservoirs[idx];
        dst_intermediates[idx] = intermediates[idx];
    }
}

__global__ void computeSpatialReuse(
    idaten::Path* paths,
    const idaten::Reservoir* __restrict__ reservoirs,
    idaten::Reservoir* dst_reservoirs,
    const idaten::ReSTIRIntermedidate* __restrict__ intermediates,
    idaten::ReSTIRIntermedidate* dst_intermediates,
    int width, int height)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    auto idx = getIdx(ix, iy, width);

    OnComputeSpatialReuse(
        idx,
        &paths->sampler[idx],
        reservoirs,
        dst_reservoirs,
        intermediates,
        dst_intermediates,
        width, height
    );
}

namespace idaten {
    int ReSTIRPathTracing::computelReuse(
        int width, int height,
        int bounce)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        int src_idx = 0;
        int dst_idx = 1;

        if (bounce == 0) {
            computeSpatialReuse << <grid, block, 0, m_stream >> > (
                m_paths.ptr(),
                m_reservoirs[src_idx].ptr(),
                m_reservoirs[dst_idx].ptr(),
                m_intermediates[src_idx].ptr(),
                m_intermediates[dst_idx].ptr(),
                width, height);

            checkCudaKernel(computeSpatialReuse);
        }

        return dst_idx;
    }
}
