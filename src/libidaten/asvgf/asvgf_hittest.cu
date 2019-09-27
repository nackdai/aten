#include "asvgf/asvgf.h"

#include "kernel/StreamCompaction.h"

#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

// NOTE
// compute capability 7.5
// https://en.wikipedia.org/wiki/CUDA

#define NUM_SM				30    // no. of streaming multiprocessors
#define NUM_WARP_PER_SM     32    // maximum no. of resident warps per SM
#define NUM_BLOCK_PER_SM    16    // maximum no. of resident blocks per SM
#define NUM_BLOCK           (NUM_SM * NUM_BLOCK_PER_SM)
#define NUM_WARP_PER_BLOCK  (NUM_WARP_PER_SM / NUM_BLOCK_PER_SM)
#define WARP_SIZE           32

__device__ unsigned int asvgf_headDev = 0;

__global__ void hitTestASVGF(
    idaten::TileDomain tileDomain,
    idaten::SVGFPathTracing::Path* paths,
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
    // warp-wise head index of tasks in a block
    __shared__ volatile unsigned int headBlock[NUM_WARP_PER_BLOCK];

    volatile unsigned int& headWarp = headBlock[threadIdx.y];

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asvgf_headDev = 0;
    }

    Context ctxt;
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
            headWarp = atomicAdd(&asvgf_headDev, WARP_SIZE);
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

        isects[idx] = isect;

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
}

namespace idaten
{
    void AdvancedSVGFPathTracing::onHitTest(
        int width, int height,
        int bounce,
        cudaTextureObject_t texVtxPos)
    {
        hitTestASVGF << <NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK), 0, m_stream >> > (
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

        checkCudaKernel(hitTestASVGF);
    }
}
