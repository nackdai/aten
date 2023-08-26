#include "ao/ao.h"
#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/StreamCompaction.h"
#include "kernel/pt_common.h"
#include "kernel/persistent_thread.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"
#include "renderer/pt_params.h"

//#define ENABLE_SEPARATE_ALBEDO

namespace ao {
    __global__ void shadeMissAO(
        bool isFirstBounce,
        int32_t width, int32_t height,
        idaten::Path paths)
    {
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        if (!paths.attrib[idx].isTerminate && !paths.attrib[idx].isHit) {
            if (isFirstBounce) {
                paths.attrib[idx].isKill = true;
            }

            paths.contrib[idx].contrib = make_float3(1);

            paths.attrib[idx].isTerminate = true;
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
        int32_t bounce, int32_t rrBounce,
        idaten::context ctxt,
        const uint32_t* random)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        const auto& ray = rays[idx];

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
        auto scramble = random[idx] * 0x1fe3434f;
        path.sampler.init(frame, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
        auto rnd = random[idx];
        auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        paths.sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + 5 * 300, scramble);
#endif

        aten::hitrecord rec;

        const auto& isect = isects[idx];

        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        const auto& mtrl = ctxt.GetMaterial(static_cast<uint32_t>(rec.mtrlid));

        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        aten::vec3 orienting_normal = rec.normal;

        // Apply normal map.
        int32_t normalMap = (int32_t)(mtrl.normalMap >= 0 ? ctxt.textures[mtrl.normalMap] : -1);
        AT_NAME::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);

        float3 ao_color = make_float3(0.0f);

        for (int32_t i = 0; i < ao_num_rays; i++) {
            auto nextDir = AT_NAME::lambert::sampleDirection(orienting_normal, &paths.sampler[idx]);
            auto pdfb = AT_NAME::lambert::pdf(orienting_normal, nextDir);

            real c = dot(orienting_normal, nextDir);

            auto ao_ray = aten::ray(rec.p, nextDir, orienting_normal);

            aten::Intersection isectTmp;

            bool isHit = intersectClosest(&ctxt, ao_ray, &isectTmp, ao_radius);

            if (isHit) {
                if (c > 0.0f) {
                    ao_color += make_float3(isectTmp.t / ao_radius * c / pdfb);
                }
            }
            else {
                ao_color = make_float3(1.0f);
            }
        }

        ao_color /= ao_num_rays;
        paths.contrib[idx].contrib = ao_color;
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
            m_random.data());

        checkCudaKernel(shade);
    }
}
