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
        idaten::Path* paths)
    {
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= tileDomain.w || iy >= tileDomain.h) {
            return;
        }

        const auto idx = getIdx(ix, iy, tileDomain.w);

        if (!paths->attrib[idx].isTerminate && !paths->attrib[idx].isHit) {
            if (isFirstBounce) {
                paths->attrib[idx].isKill = true;
            }

            paths->contrib[idx].contrib = make_float3(1);

            paths->attrib[idx].isTerminate = true;
        }
    }

    __global__ void shadeAO(
        int ao_num_rays, float ao_radius,
        idaten::TileDomain tileDomain,
        unsigned int frame,
        idaten::Path* paths,
        int* hitindices,
        int* hitnum,
        const aten::Intersection* __restrict__ isects,
        aten::ray* rays,
        int bounce, int rrBounce,
        const aten::GeomParameter* __restrict__ shapes, int geomnum,
        aten::MaterialParameter* mtrls,
        cudaTextureObject_t* nodes,
        const aten::PrimitiveParamter* __restrict__ prims,
        cudaTextureObject_t vtxPos,
        cudaTextureObject_t vtxNml,
        const aten::mat4* __restrict__ matrices,
        cudaTextureObject_t* textures,
        const unsigned int* random)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idaten::Context ctxt;
        {
            ctxt.geomnum = geomnum;
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
        paths->sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + 5 * 300, scramble);
#endif

        aten::hitrecord rec;

        const auto& isect = isects[idx];

        auto obj = &ctxt.shapes[isect.objid];
        evalHitResult(&ctxt, obj, ray, &rec, &isect);

        aten::MaterialParameter mtrl = ctxt.mtrls[rec.mtrlid];

        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        aten::vec3 orienting_normal = rec.normal;

        // Apply normal map.
        int normalMap = (int)(mtrl.normalMap >= 0 ? ctxt.textures[mtrl.normalMap] : -1);
        if (mtrl.type == aten::MaterialType::Layer) {
            // 最表層の NormalMap を適用.
            auto* topmtrl = &ctxt.mtrls[mtrl.layer[0]];
            normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
        }
        AT_NAME::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);

        float3 ao_color = make_float3(0.0f);

        for (int i = 0; i < ao_num_rays; i++) {
            auto nextDir = AT_NAME::lambert::sampleDirection(orienting_normal, &paths->sampler[idx]);
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
        paths->contrib[idx].contrib = ao_color;
    }

    __global__ void gatherAO(
        int width, int height,
        cudaSurfaceObject_t outSurface,
        const idaten::Path* __restrict__ paths,
        bool enableProgressive)
    {
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        int sample = paths->contrib[idx].samples;
        const auto& contrib = paths->contrib[idx].contrib;

        float4 data;

        if (enableProgressive) {
            surf2Dread(&data, outSurface, ix * sizeof(float4), iy);

            // First data.w value is 0.
            int n = data.w;
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
        int width, int height,
        int bounce)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        bool isFirstBounce = bounce == 0;

        ao::shadeMissAO << <grid, block >> > (
            isFirstBounce,
            m_tileDomain,
            m_paths.ptr());

        checkCudaKernel(shadeMiss);
    }

    void AORenderer::onShade(
        int width, int height,
        int bounce, int rrBounce,
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
            m_paths.ptr(),
            m_hitidx.ptr(), hitcount.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
            bounce, rrBounce,
            m_shapeparam.ptr(), m_shapeparam.num(),
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
        int width, int height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        ao::gatherAO << <grid, block >> > (
            width, height,
            outputSurf,
            m_paths.ptr(),
            m_enableProgressive);
    }
}
