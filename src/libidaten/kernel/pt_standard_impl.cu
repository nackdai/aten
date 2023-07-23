#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "kernel/pt_common.h"
#include "kernel/pt_params.h"
#include "kernel/persistent_thread.h"
#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/renderer.h"
#include "kernel/pt_standard_impl.h"
#include "renderer/aov.h"
#include "renderer/pathtracing_impl.h"

// TODO
// persistend thread works with CUDA 10.1.
// But, it doesn't work with CUDA 11 or later.
//#define ENABLE_PERSISTENT_THREAD

namespace idaten {
namespace kernel {
    __global__ void genPath(
        bool needFillAOV,
        idaten::Path paths,
        aten::ray* rays,
        int32_t width, int32_t height,
        int32_t sample,
        uint32_t frame,
        const aten::CameraParameter camera,
        const void* samplerValues,
        const uint32_t* __restrict__ random)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        AT_NAME::generate_path(
            rays[idx],
            idx, ix, iy,
            sample, frame,
            paths, camera, *random);
    }

    // NOTE
    // https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf

    __device__ uint32_t g_headDev = 0;

    __global__ void hitTest(
        idaten::context ctxt,
        idaten::Path paths,
        aten::Intersection* isects,
        aten::ray* rays,
        int32_t* hitbools,
        int32_t width, int32_t height,
        cudaTextureObject_t* nodes,
        int32_t bounce,
        float hitDistLimit)
    {
#ifdef ENABLE_PERSISTENT_THREAD
        // warp-wise head index of tasks in a block
        __shared__ volatile uint32_t nextRayArray[NUM_WARP_PER_BLOCK];
        __shared__ volatile uint32_t rayCountArray[NUM_WARP_PER_BLOCK];

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            g_headDev = 0;
        }

        if (threadIdx.x == 0) {
            for (int32_t i = 0; i < NUM_WARP_PER_BLOCK; i++) {
                rayCountArray[i] = 0;
            }
        }

        int32_t size = tileDomain.w * tileDomain.h;

        __syncthreads();

        volatile auto& localPoolNextRay = nextRayArray[threadIdx.y];
        volatile auto& localPoolRayCount = rayCountArray[threadIdx.y];

        ctxt.nodes = nodes;

        do
        {
            // let lane 0 fetch [wh, wh + WARP_SIZE - 1] for a warp
            if (localPoolRayCount == 0 && threadIdx.x == 0) {
                localPoolNextRay = atomicAdd(&g_headDev, WARP_SIZE);
                localPoolRayCount = WARP_SIZE;
            }

            // task index per thread in a warp
            uint32_t idx = localPoolNextRay + threadIdx.x;

            if (idx >= size) {
                return;
            }

            if (threadIdx.x == 0) {
                localPoolNextRay += WARP_SIZE;
                localPoolRayCount -= WARP_SIZE;
            }

            paths.attrib[idx].isHit = false;

            hitbools[idx] = 0;

            if (paths.attrib[idx].isTerminate) {
                continue;
            }

            aten::Intersection isect;

            float t_max = AT_MATH_INF;

            if (bounce >= 1
                && !paths.attrib[idx].isSingular)
            {
                t_max = hitDistLimit;
            }

            // TODO
            // 近距離でVoxelにすると品質が落ちる.
            // しかも同じオブジェクト間だとそれが起こりやすい.
            //bool enableLod = (bounce >= 2);
            bool enableLod = false;
            int32_t depth = 9;

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
                && !paths.attrib[idx].isSingular
                && isect.t > hitDistLimit)
            {
                isHit = false;
                paths.attrib[idx].isTerminate = true;
            }

            paths.attrib[idx].isHit = isHit;

            hitbools[idx] = isHit ? 1 : 0;
        } while (true);
#else
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        paths.attrib[idx].isHit = false;

        hitbools[idx] = 0;

        if (paths.attrib[idx].isTerminate) {
            return;
        }

        aten::Intersection isect;

        float t_max = AT_MATH_INF;

        if (bounce >= 1
            && !paths.attrib[idx].isSingular)
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
            && !paths.attrib[idx].isSingular
            && isect.t > hitDistLimit)
        {
            isHit = false;
        }

        paths.attrib[idx].isHit = isHit;

        hitbools[idx] = isHit ? 1 : 0;
#endif
    }

    __global__ void hitTestPrimaryRayInScreenSpace(
        cudaSurfaceObject_t gbuffer,
        const idaten::context ctxt,
        idaten::Path paths,
        aten::Intersection* isects,
        int32_t* hitbools,
        int32_t width, int32_t height,
        const aten::vec4 camPos)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        paths.attrib[idx].isHit = false;

        hitbools[idx] = 0;

        if (paths.attrib[idx].isTerminate) {
            return;
        }

        // Sample data from texture.
        float4 data;
        surf2Dread(&data, gbuffer, ix * sizeof(float4), iy);

        // NOTE
        // x : objid
        // y : primid
        // zw : bary centroid

        int32_t objid = __float_as_int(data.x);
        int32_t primid = __float_as_int(data.y);

        isects[idx].objid = objid;
        isects[idx].triangle_id = primid;

        // bary centroid.
        isects[idx].a = data.z;
        isects[idx].b = data.w;

        if (objid >= 0) {
            aten::TriangleParameter prim;
            prim.v0 = ((aten::vec4*)ctxt.prims)[primid * aten::TriangleParamter_float4_size + 0];
            prim.v1 = ((aten::vec4*)ctxt.prims)[primid * aten::TriangleParamter_float4_size + 1];

            isects[idx].mtrlid = prim.mtrlid;
            isects[idx].meshid = prim.mesh_id;

            const auto* obj = &ctxt.GetObject(static_cast<uint32_t>(objid));

            float4 p0 = tex1Dfetch<float4>(ctxt.vtxPos, prim.idx[0]);
            float4 p1 = tex1Dfetch<float4>(ctxt.vtxPos, prim.idx[1]);
            float4 p2 = tex1Dfetch<float4>(ctxt.vtxPos, prim.idx[2]);

            real a = data.z;
            real b = data.w;
            real c = 1 - a - b;

            // 重心座標系(barycentric coordinates).
            // v0基準.
            // p = (1 - a - b)*v0 + a*v1 + b*v2
            auto p = c * p0 + a * p1 + b * p2;
            aten::vec4 vp(p.x, p.y, p.z, 1.0f);

            if (obj->mtx_id >= 0) {
                const auto& mtxL2W = ctxt.GetMatrix(obj->mtx_id * 2 + 0);
                vp = mtxL2W.apply(vp);
            }

            isects[idx].t = (camPos - vp).length();

            paths.attrib[idx].isHit = true;
            hitbools[idx] = 1;
        }
        else {
            paths.attrib[idx].isHit = false;
            hitbools[idx] = 0;
        }
    }

    __global__ void shadeMiss(
        int32_t bounce,
        float4* aovNormalDepth,
        float4* aovAlbedoMeshid,
        idaten::Path paths,
        int32_t width, int32_t height)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        AT_NAME::shade_miss(
            idx,
            bounce,
            aten::vec3(1),
            paths,
            aovNormalDepth,
            aovAlbedoMeshid);
    }

    __global__ void shadeMissWithEnvmap(
        int32_t bounce,
        const aten::CameraParameter camera,
        float4* aovNormalDepth,
        float4* aovAlbedoMeshid,
        const idaten::context ctxt,
        int32_t envmap_idx,
        real envmap_avg_illum,
        real envmap_multiplyer,
        idaten::Path paths,
        const aten::ray* __restrict__ rays,
        int32_t width, int32_t height)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        AT_NAME::shade_miss_with_envmap(
            idx,
            ix, iy,
            width, height,
            bounce,
            envmap_idx,
            envmap_avg_illum,
            envmap_multiplyer,
            ctxt, camera,
            paths, rays[idx],
            aovNormalDepth,
            aovAlbedoMeshid);
    }
}
}

namespace idaten
{
    bool StandardPT::initPath(int32_t width, int32_t height)
    {
        if (!is_init_context_) {
            ctxt_.shapes = m_shapeparam.ptr();
            ctxt_.mtrls = m_mtrlparam.ptr();
            ctxt_.lightnum = m_lightparam.num();
            ctxt_.lights = m_lightparam.ptr();
            ctxt_.prims = m_primparams.ptr();
            ctxt_.matrices = m_mtxparams.ptr();

            ctxt_.vtxPos = m_vtxparamsPos.bind();
            ctxt_.vtxNml = m_vtxparamsNml.bind();

            ctxt_.nodes = m_nodetex.ptr();

            ctxt_.textures = m_tex.ptr();
        }

        if (!m_isInitPath) {
            m_pathThroughput.init(width * height);
            m_pathContrib.init(width * height);
            m_pathAttrib.init(width * height);
            m_pathSampler.init(width * height);

            m_paths.throughput = m_pathThroughput.ptr();
            m_paths.contrib = m_pathContrib.ptr();
            m_paths.attrib = m_pathAttrib.ptr();
            m_paths.sampler = m_pathSampler.ptr();

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

    void StandardPT::generatePath(
        int32_t width, int32_t height,
        bool needFillAOV,
        int32_t sample, int32_t maxBounce,
        int32_t seed)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        kernel::genPath << <grid, block, 0, m_stream >> > (
            needFillAOV,
            m_paths,
            m_rays.ptr(),
            width, height,
            sample,
            m_frame,
            m_cam,
            m_sobolMatrices.ptr(),
            m_random.ptr());

        checkCudaKernel(genPath);
    }

    void StandardPT::hitTest(
        int32_t width, int32_t height,
        int32_t bounce)
    {
#ifdef ENABLE_PERSISTENT_THREAD
        kernel::hitTest << <NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK), 0, m_stream >> > (
#else
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        kernel::hitTest << <grid, block >> > (
#endif
            ctxt_,
            m_paths,
            m_isects.ptr(),
            m_rays.ptr(),
            m_hitbools.ptr(),
            width, height,
            m_nodetex.ptr(),
            bounce,
            m_hitDistLimit);

        checkCudaKernel(hitTest);
    }

    void StandardPT::hitTestOnScreenSpace(
        int32_t width, int32_t height,
        idaten::CudaGLSurface& gbuffer)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        aten::vec4 campos = aten::vec4(m_cam.origin, 1.0f);

        CudaGLResourceMapper<std::remove_reference_t<decltype(gbuffer)>> rscmap(gbuffer);
        auto binded_gbuffer = gbuffer.bind();

        kernel::hitTestPrimaryRayInScreenSpace << <grid, block >> > (
            binded_gbuffer,
            ctxt_,
            m_paths,
            m_isects.ptr(),
            m_hitbools.ptr(),
            width, height,
            campos);

        checkCudaKernel(hitTestPrimaryRayInScreenSpace);
    }

    void StandardPT::missShade(
        int32_t width, int32_t height,
        int32_t bounce,
        idaten::TypedCudaMemory<float4>& aovNormalDepth,
        idaten::TypedCudaMemory<float4>& aovTexclrMeshid)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        if (m_enableEnvmap && m_envmapRsc.idx >= 0) {
            kernel::shadeMissWithEnvmap << <grid, block, 0, m_stream >> > (
                bounce,
                m_cam,
                aovNormalDepth.ptr(),
                aovTexclrMeshid.ptr(),
                ctxt_,
                m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
                m_paths,
                m_rays.ptr(),
                width, height);
        }
        else {
            kernel::shadeMiss << <grid, block, 0, m_stream >> > (
                bounce,
                aovNormalDepth.ptr(),
                aovTexclrMeshid.ptr(),
                m_paths,
                width, height);
        }

        checkCudaKernel(shadeMiss);
    }
}
