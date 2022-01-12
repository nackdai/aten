#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "kernel/pt_common.h"
#include "kernel/pt_params.h"
#include "kernel/persistent_thread.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/renderer.h"
#include "kernel/pt_standard_impl.h"

#define ENABLE_PERSISTENT_THREAD

namespace idaten {
namespace kernel {
    __global__ void initPath(
        idaten::Path* path,
        idaten::PathThroughput* throughput,
        idaten::PathContrib* contrib,
        idaten::PathAttribute* attrib,
        aten::sampler* sampler)
    {
        path->throughput = throughput;
        path->contrib = contrib;
        path->attrib = attrib;
        path->sampler = sampler;
    }

    __device__ unsigned int g_headDev = 0;

    __global__ void hitTest(
        idaten::TileDomain tileDomain,
        idaten::Path* paths,
        aten::Intersection* isects,
        aten::ray* rays,
        int* hitbools,
        int width, int height,
        const aten::GeomParameter* __restrict__ shapes, int geomnum,
        const aten::LightParameter* __restrict__ lights, int lightnum,
        cudaTextureObject_t* nodes,
        const aten::PrimitiveParamter* __restrict__ prims,
        cudaTextureObject_t vtxPos,
        aten::mat4* matrices,
        int bounce,
        float hitDistLimit)
    {
#ifdef ENABLE_PERSISTENT_THREAD
        // warp-wise head index of tasks in a block
        __shared__ volatile unsigned int headBlock[NUM_WARP_PER_BLOCK];

        volatile unsigned int& headWarp = headBlock[threadIdx.y];

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            g_headDev = 0;
        }

        idaten::Context ctxt;
        {
            ctxt.geomnum = geomnum;
            ctxt.shapes = shapes;
            ctxt.lightnum = lightnum;
            ctxt.lights = lights;
            ctxt.nodes = nodes;
            ctxt.prims = prims;
            ctxt.vtxPos = vtxPos;
            ctxt.matrices = matrices;
        }

        do
        {
            // let lane 0 fetch [wh, wh + WARP_SIZE - 1] for a warp
            if (threadIdx.x == 0) {
                headWarp = atomicAdd(&g_headDev, WARP_SIZE);
            }
            // task index per thread in a warp
            unsigned int idx = headWarp + threadIdx.x;

            if (idx >= tileDomain.w * tileDomain.h) {
                return;
            }

            paths->attrib[idx].isHit = false;

            hitbools[idx] = 0;

            if (paths->attrib[idx].isTerminate) {
                continue;
            }

            aten::Intersection isect;

            float t_max = AT_MATH_INF;

            if (bounce >= 1
                && !paths->attrib[idx].isSingular)
            {
                t_max = hitDistLimit;
            }

            // TODO
            // 近距離でVoxelにすると品質が落ちる.
            // しかも同じオブジェクト間だとそれが起こりやすい.
            //bool enableLod = (bounce >= 2);
            bool enableLod = false;
            int depth = 9;

            bool isHit = intersectClosest(&ctxt, rays[idx], &isect, t_max, enableLod, depth);

#if 0
            isects[idx].t = isect.t;
            isects[idx].objid = isect.objid;
            isects[idx].mtrlid = isect.mtrlid;
            isects[idx].meshid = isect.meshid;
            isects[idx].primid = isect.primid;
            isects[idx].a = isect.a;
            isects[idx].b = isect.b;
#else
            isects[idx] = isect;
#endif

            if (bounce >= 1
                && !paths->attrib[idx].isSingular
                && isect.t > hitDistLimit)
            {
                isHit = false;
                paths->attrib[idx].isTerminate = true;
            }

            paths->attrib[idx].isHit = isHit;

            hitbools[idx] = isHit ? 1 : 0;
        } while (true);
#else
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= tileDomain.w || iy >= tileDomain.h) {
            return;
        }

        const auto idx = getIdx(ix, iy, tileDomain.w);

        paths->attrib[idx].isHit = false;

        hitbools[idx] = 0;

        if (paths->attrib[idx].isTerminate) {
            return;
        }

        idaten::Context ctxt;
        {
            ctxt.geomnum = geomnum;
            ctxt.shapes = shapes;
            ctxt.lightnum = lightnum;
            ctxt.lights = lights;
            ctxt.nodes = nodes;
            ctxt.prims = prims;
            ctxt.vtxPos = vtxPos;
            ctxt.matrices = matrices;
        }

        aten::Intersection isect;

        float t_max = AT_MATH_INF;

        if (bounce >= 1
            && !paths->attrib[idx].isSingular)
        {
            t_max = hitDistLimit;
        }

        bool isHit = intersectClosest(&ctxt, rays[idx], &isect, t_max);

#if 0
        isects[idx].t = isect.t;
        isects[idx].objid = isect.objid;
        isects[idx].mtrlid = isect.mtrlid;
        isects[idx].meshid = isect.meshid;
        isects[idx].area = isect.area;
        isects[idx].primid = isect.primid;
        isects[idx].a = isect.a;
        isects[idx].b = isect.b;
#else
        isects[idx] = isect;
#endif

        if (bounce >= 1
            && !paths->attrib[idx].isSingular
            && isect.t > hitDistLimit)
        {
            isHit = false;
        }

        paths->attrib[idx].isHit = isHit;

        hitbools[idx] = isHit ? 1 : 0;
#endif
    }

    __global__ void hitTestPrimaryRayInScreenSpace(
        idaten::TileDomain tileDomain,
        cudaSurfaceObject_t gbuffer,
        idaten::Path* paths,
        aten::Intersection* isects,
        int* hitbools,
        int width, int height,
        const aten::vec4 camPos,
        const aten::GeomParameter* __restrict__ geoms,
        const aten::PrimitiveParamter* __restrict__ prims,
        const aten::mat4* __restrict__ matrices,
        cudaTextureObject_t vtxPos)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= tileDomain.w || iy >= tileDomain.h) {
            return;
        }

        const auto idx = getIdx(ix, iy, tileDomain.w);

        paths->attrib[idx].isHit = false;

        hitbools[idx] = 0;

        if (paths->attrib[idx].isTerminate) {
            return;
        }

        ix += tileDomain.x;
        iy += tileDomain.y;

        // Sample data from texture.
        float4 data;
        surf2Dread(&data, gbuffer, ix * sizeof(float4), iy);

        // NOTE
        // x : objid
        // y : primid
        // zw : bary centroid

        int objid = __float_as_int(data.x);
        int primid = __float_as_int(data.y);

        isects[idx].objid = objid;
        isects[idx].primid = primid;

        // bary centroid.
        isects[idx].a = data.z;
        isects[idx].b = data.w;

        if (objid >= 0) {
            aten::PrimitiveParamter prim;
            prim.v0 = ((aten::vec4*)prims)[primid * aten::PrimitiveParamter_float4_size + 0];
            prim.v1 = ((aten::vec4*)prims)[primid * aten::PrimitiveParamter_float4_size + 1];

            isects[idx].mtrlid = prim.mtrlid;
            isects[idx].meshid = prim.gemoid;

            const auto* obj = &geoms[objid];

            float4 p0 = tex1Dfetch<float4>(vtxPos, prim.idx[0]);
            float4 p1 = tex1Dfetch<float4>(vtxPos, prim.idx[1]);
            float4 p2 = tex1Dfetch<float4>(vtxPos, prim.idx[2]);

            real a = data.z;
            real b = data.w;
            real c = 1 - a - b;

            // 重心座標系(barycentric coordinates).
            // v0基準.
            // p = (1 - a - b)*v0 + a*v1 + b*v2
            auto p = c * p0 + a * p1 + b * p2;
            aten::vec4 vp(p.x, p.y, p.z, 1.0f);

            if (obj->mtxid >= 0) {
                auto mtxL2W = matrices[obj->mtxid * 2 + 0];
                vp = mtxL2W.apply(vp);
            }

            isects[idx].t = (camPos - vp).length();

            paths->attrib[idx].isHit = true;
            hitbools[idx] = 1;
        }
        else {
            paths->attrib[idx].isHit = false;
            hitbools[idx] = 0;
        }
    }
}
}

namespace idaten
{
    bool StandardPT::initPath(int width, int height)
    {
        if (!m_isInitPath) {
            m_paths.init(1);

            m_pathThroughput.init(width * height);
            m_pathContrib.init(width * height);
            m_pathAttrib.init(width * height);
            m_pathSampler.init(width * height);

            kernel::initPath << <1, 1, 0, m_stream >> > (
                m_paths.ptr(),
                m_pathThroughput.ptr(),
                m_pathContrib.ptr(),
                m_pathAttrib.ptr(),
                m_pathSampler.ptr());

            m_isInitPath = true;

            return true;
        }

        return false;
    }

    void StandardPT::clearPath()
    {
        cudaMemsetAsync(m_pathThroughput.ptr(), 0, m_pathThroughput.bytes(), m_stream);
        cudaMemsetAsync(m_pathContrib.ptr(), 0, m_pathContrib.bytes(), m_stream);
        cudaMemsetAsync(m_pathAttrib.ptr(), 0, m_pathAttrib.bytes(), m_stream);

        if (m_frame == 0) {
            cudaMemsetAsync(m_pathSampler.ptr(), 0, m_pathSampler.bytes(), m_stream);
        }
    }

    void StandardPT::hitTest(
        int width, int height,
        int bounce,
        cudaTextureObject_t texVtxPos)
    {
#ifdef ENABLE_PERSISTENT_THREAD
        kernel::hitTest << <NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK), 0, m_stream >> > (
#else
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        kernel::hitTest << <grid, block >> > (
#endif
            m_tileDomain,
            m_paths.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
            m_hitbools.ptr(),
            width, height,
            m_shapeparam.ptr(), m_shapeparam.num(),
            m_lightparam.ptr(), m_lightparam.num(),
            m_nodetex.ptr(),
            m_primparams.ptr(),
            texVtxPos,
            m_mtxparams.ptr(),
            bounce,
            m_hitDistLimit);

        checkCudaKernel(hitTest);
    }

    void StandardPT::hitTestOnScreenSpace(
        int width, int height,
        idaten::CudaGLSurface& gbuffer,
        cudaTextureObject_t texVtxPos)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        aten::vec4 campos = aten::vec4(m_camParam.origin, 1.0f);

        CudaGLResourceMapper<std::remove_reference_t<decltype(gbuffer)>> rscmap(gbuffer);
        auto binded_gbuffer = gbuffer.bind();

        kernel::hitTestPrimaryRayInScreenSpace << <grid, block >> > (
            m_tileDomain,
            binded_gbuffer,
            m_paths.ptr(),
            m_isects.ptr(),
            m_hitbools.ptr(),
            width, height,
            campos,
            m_shapeparam.ptr(),
            m_primparams.ptr(),
            m_mtxparams.ptr(),
            texVtxPos);

        checkCudaKernel(hitTestPrimaryRayInScreenSpace);
    }
}
