#include "ao/ao.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"

#include "aten4idaten.h"
#include "renderer/ao/aorenderer_impl.h"

namespace ao {
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

        AT_NAME::ShadeMissAO(
            idx,
            is_first_bounce,
            paths);
    }

    __global__ void shadeAO(
        int32_t ao_num_rays, float ao_radius,
        uint32_t frame,
        idaten::Path paths,
        int32_t* hitindices,
        int32_t* hitnum,
        const aten::Intersection* __restrict__ isects,
        aten::ray* rays,
        int32_t bounce, int32_t rrBounce,
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

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        const auto& ray = rays[idx];

        auto rnd = random[idx];

        const auto& isect = isects[idx];

        AT_NAME::ShandeAO(
            idx,
            frame, rnd,
            ao_num_rays, ao_radius,
            paths, ctxt, ray, isect);
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

        ao::shadeMissAO << <grid, block >> > (
            isFirstBounce,
            width, height,
            path_host_->paths);

        checkCudaKernel(shadeMiss);
    }

    void AORenderer::onShade(
        int32_t width, int32_t height,
        int32_t bounce, int32_t rrBounce)
    {
        dim3 blockPerGrid((width * height + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        ao::shadeAO << <blockPerGrid, threadPerBlock >> > (
            //shade<true> << <1, 1 >> > (
            m_ao_num_rays, m_ao_radius,
            m_frame,
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_isects.data(),
            m_rays.data(),
            bounce, rrBounce,
            ctxt_host_.ctxt,
            ctxt_host_.shapeparam.data(),
            ctxt_host_.mtrlparam.data(),
            ctxt_host_.lightparam.data(),
            ctxt_host_.primparams.data(),
            ctxt_host_.mtxparams.data(),
            m_random.data());

        checkCudaKernel(shade);
    }
}
