#include "ao/ao.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
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

//#define ENABLE_SEPARATE_ALBEDO

namespace ao {
    __global__ void shadeMissAO(
        bool isFirstBounce,
        idaten::TileDomain tileDomain,
        idaten::Path paths)
    {
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= tileDomain.w || iy >= tileDomain.h) {
            return;
        }

        const auto idx = getIdx(ix, iy, tileDomain.w);

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
        idaten::TileDomain tileDomain,
        uint32_t frame,
        idaten::Path paths,
        int32_t* hitindices,
        int32_t* hitnum,
        const aten::Intersection* __restrict__ isects,
        aten::ray* rays,
        int32_t bounce, int32_t rrBounce,
        const aten::ObjectParameter* __restrict__ shapes,
        aten::MaterialParameter* mtrls,
        cudaTextureObject_t* nodes,
        const aten::TriangleParameter* __restrict__ prims,
        cudaTextureObject_t vtxPos,
        cudaTextureObject_t vtxNml,
        const aten::mat4* __restrict__ matrices,
        cudaTextureObject_t* textures,
        const uint32_t* random)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idaten::Context ctxt;
        {
            ctxt.shapes = shapes;
            ctxt.mtrls = mtrls;
            ctxt.nodes = nodes;
            ctxt.prims = prims;
            ctxt.vtxPos = vtxPos;
            ctxt.vtxNml = vtxNml;
            ctxt.matrices = matrices;
            ctxt.textures = textures;
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

        auto obj = &ctxt.GetObject(static_cast<uint32_t>(isect.objid));
        evalHitResult(&ctxt, obj, ray, &rec, &isect);

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

    __global__ void gatherAO(
        int32_t width, int32_t height,
        cudaSurfaceObject_t outSurface,
        const idaten::Path paths,
        bool enableProgressive)
    {
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        int32_t sample = paths.contrib[idx].samples;
        const auto& contrib = paths.contrib[idx].contrib;

        float4 data;

        if (enableProgressive) {
            surf2Dread(&data, outSurface, ix * sizeof(float4), iy);

            // First data.w value is 0.
            int32_t n = data.w;
            data = n * data + make_float4(contrib.x, contrib.y, contrib.z, 0) / sample;
            data /= (n + 1);
            data.w = n + 1;
        }
        else {
            data = make_float4(contrib.x, contrib.y, contrib.z, 0) / sample;
            data.w = sample;
        }

        surf2Dwrite(
            data,
            outSurface,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }
}

namespace idaten {
    void AORenderer::onShadeMiss(
        int32_t width, int32_t height,
        int32_t bounce)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        bool isFirstBounce = bounce == 0;

        ao::shadeMissAO << <grid, block >> > (
            isFirstBounce,
            m_tileDomain,
            m_paths);

        checkCudaKernel(shadeMiss);
    }

    void AORenderer::onShade(
        int32_t width, int32_t height,
        int32_t bounce, int32_t rrBounce,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 blockPerGrid((m_tileDomain.w * m_tileDomain.h + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        ao::shadeAO << <blockPerGrid, threadPerBlock >> > (
            //shade<true> << <1, 1 >> > (
            m_ao_num_rays, m_ao_radius,
            m_tileDomain,
            m_frame,
            m_paths,
            m_hitidx.ptr(), hitcount.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
            bounce, rrBounce,
            m_shapeparam.ptr(),
            m_mtrlparam.ptr(),
            m_nodetex.ptr(),
            m_primparams.ptr(),
            texVtxPos, texVtxNml,
            m_mtxparams.ptr(),
            m_tex.ptr(),
            m_random.ptr());

        checkCudaKernel(shade);
    }

    void AORenderer::onGather(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        ao::gatherAO << <grid, block >> > (
            width, height,
            outputSurf,
            m_paths,
            m_enableProgressive);
    }
}
