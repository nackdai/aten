#include "ao/ao.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"

#include "aten4idaten.h"
#include "renderer/ao/aorenderer_impl.h"

namespace ao_kernel {
    __global__ void shadeMissAO(
        bool is_first_bounce,
        int32_t width, int32_t height,
        idaten::Path paths)
    {
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        const auto ao_color = AT_NAME::ao::ShadeByAOIfHitMiss(
            idx,
            is_first_bounce,
            paths);

        if (ao_color >= 0) {
            paths.contrib[idx].contrib = make_float3(ao_color);
        }
    }

    __global__ void shadeAO(
        int32_t ao_num_rays, float ao_radius,
        uint32_t frame,
        idaten::Path paths,
        int32_t* hitindices,
        int32_t* hitnum,
        const aten::Intersection* __restrict__ isects,
        aten::ray* rays,
        idaten::context ctxt,
        const aten::ObjectParameter* __restrict__ shapes,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights,
        const aten::TriangleParameter* __restrict__ prims,
        const aten::mat4* __restrict__ matrices,
        const uint32_t* random)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        const auto rnd = random[idx];
        const auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        paths.sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + 5 * 300, scramble);

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        const auto& ray = rays[idx];
        const auto& isect = isects[idx];

        const auto ao_color = AT_NAME::ao::ShandeByAO(
            ao_num_rays, ao_radius,
            paths.sampler[idx], ctxt, ray, isect);

        paths.contrib[idx].contrib = make_float3(ao_color);
    }
}

namespace idaten {
    void AORenderer::missShade(
        int32_t width, int32_t height,
        int32_t bounce)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        bool isFirstBounce = bounce == 0;

        ao_kernel::shadeMissAO << <grid, block >> > (
            isFirstBounce,
            width, height,
            path_host_->paths);

        checkCudaKernel(shadeMiss);
    }

    void AORenderer::ShadeAO(
        int32_t width, int32_t height,
        int32_t bounce, int32_t rrBounce)
    {
        dim3 blockPerGrid((width * height + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        ao_kernel::shadeAO << <blockPerGrid, threadPerBlock >> > (
            //shade<true> << <1, 1 >> > (
            m_ao_num_rays, m_ao_radius,
            m_frame,
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_isects.data(),
            m_rays.data(),
            ctxt_host_->ctxt,
            ctxt_host_->shapeparam.data(),
            ctxt_host_->mtrlparam.data(),
            ctxt_host_->lightparam.data(),
            ctxt_host_->primparams.data(),
            ctxt_host_->mtxparams.data(),
            m_random.data());

        checkCudaKernel(shade);
    }
}
