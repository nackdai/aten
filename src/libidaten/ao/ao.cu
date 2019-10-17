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

#define ENABLE_PERSISTENT_THREAD

__global__ void genPathAO(
    idaten::TileDomain tileDomain,
    idaten::AORenderer::Path* paths,
    aten::ray* rays,
    int sample, int maxSamples,
    unsigned int frame,
    const aten::CameraParameter* __restrict__ camera,
    const unsigned int* sobolmatrices,
    const unsigned int* random)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    const auto idx = getIdx(ix, iy, tileDomain.w);

    auto& path = paths[idx];
    path.isHit = false;

    if (path.isKill) {
        path.isTerminate = true;
        return;
    }

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    auto scramble = random[idx] * 0x1fe3434f;
    path.sampler.init(frame, 0, scramble, sobolmatrices);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f * ((frame + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    path.sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 0, scramble);
#endif

    ix += tileDomain.x;
    iy += tileDomain.y;

    float s = (ix + path.sampler.nextSample()) / (float)(camera->width);
    float t = (iy + path.sampler.nextSample()) / (float)(camera->height);

    AT_NAME::CameraSampleResult camsample;
    AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

    rays[idx] = camsample.r;

    path.throughput = aten::vec3(1);
    path.pdfb = 0.0f;
    path.isTerminate = false;
    path.isSingular = false;

    path.samples += 1;

    // Accumulate value, so do not reset.
    //path.contrib = aten::vec3(0);
}

__device__ unsigned int headDev_AO = 0;

__global__ void hitTestAO(
    idaten::TileDomain tileDomain,
    idaten::AORenderer::Path* paths,
    aten::Intersection* isects,
    aten::ray* rays,
    int* hitbools,
    const aten::GeomParameter* __restrict__ shapes, int geomnum,
    const aten::LightParameter* __restrict__ lights, int lightnum,
    cudaTextureObject_t* nodes,
    const aten::PrimitiveParamter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    aten::mat4* matrices)
{
#ifdef ENABLE_PERSISTENT_THREAD
    // warp-wise head index of tasks in a block
    __shared__ volatile unsigned int headBlock[NUM_WARP_PER_BLOCK];

    volatile unsigned int& headWarp = headBlock[threadIdx.y];

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        headDev_AO = 0;
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
            headWarp = atomicAdd(&headDev_AO, WARP_SIZE);
        }
        // task index per thread in a warp
        unsigned int idx = headWarp + threadIdx.x;

        if (idx >= tileDomain.w * tileDomain.h) {
            return;
        }

        auto& path = paths[idx];
        path.isHit = false;

        hitbools[idx] = 0;

        if (path.isTerminate) {
            continue;
        }

        aten::Intersection isect;

        bool isHit = intersectClosest(&ctxt, rays[idx], &isect);

        isects[idx].t = isect.t;
        isects[idx].objid = isect.objid;
        isects[idx].mtrlid = isect.mtrlid;
        isects[idx].meshid = isect.meshid;
        isects[idx].primid = isect.primid;
        isects[idx].a = isect.a;
        isects[idx].b = isect.b;

        path.isHit = isHit;

        hitbools[idx] = isHit ? 1 : 0;
    } while (true);
#else
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    const auto idx = getIdx(ix, iy, tileDomain.w);

    auto& path = paths[idx];
    path.isHit = false;

    hitbools[idx] = 0;

    if (path.isTerminate) {
        return;
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

    aten::Intersection isect;

    bool isHit = intersectClosest(&ctxt, rays[idx], &isect);

    isects[idx].t = isect.t;
    isects[idx].objid = isect.objid;
    isects[idx].mtrlid = isect.mtrlid;
    isects[idx].meshid = isect.meshid;
    isects[idx].area = isect.area;
    isects[idx].primid = isect.primid;
    isects[idx].a = isect.a;
    isects[idx].b = isect.b;

    path.isHit = isHit;

    hitbools[idx] = isHit ? 1 : 0;
#endif
}

__global__ void shadeMissAO(
    bool isFirstBounce,
    idaten::TileDomain tileDomain,
    idaten::AORenderer::Path* paths)
{
    const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    const auto idx = getIdx(ix, iy, tileDomain.w);

    auto& path = paths[idx];

    if (!path.isTerminate && !path.isHit) {
        if (isFirstBounce) {
            path.isKill = true;
        }

        path.contrib = aten::vec3(1);

        path.isTerminate = true;
    }
}

__global__ void shadeAO(
    int num_ao_rays, float ao_radius,
    idaten::TileDomain tileDomain,
    unsigned int frame,
    idaten::AORenderer::Path* paths,
    int* hitindices,
    int* hitnum,
    const aten::Intersection* __restrict__ isects,
    aten::ray* rays,
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

    Context ctxt;
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

    auto& path = paths[idx];
    const auto& ray = rays[idx];

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    auto scramble = random[idx] * 0x1fe3434f;
    path.sampler.init(frame, 0, scramble, sobolmatrices);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f * ((frame + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    path.sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 0, scramble);
#endif

    const auto& isect = isects[idx];

    // Reconstruction world coordnate.
    aten::vec3 world_pos = ray.org + ray.dir * isects[idx].t;

    aten::hitrecord rec;

    auto obj = &ctxt.shapes[isect.objid];
    evalHitResult(&ctxt, obj, ray, &rec, &isect);

    aten::MaterialParameter mtrl = ctxt.mtrls[rec.mtrlid];

    bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

    // 交差位置の法線.
    // 物体からのレイの入出を考慮.
    aten::vec3 orienting_normal = rec.normal;

#if 0
    // Apply normal map.
    int normalMap = mtrl.normalMap;
    if (mtrl.type == aten::MaterialType::Layer) {
        // 最表層の NormalMap を適用.
        auto* topmtrl = &ctxt.mtrls[mtrl.layer[0]];
        normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
    }
    AT_NAME::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);
#endif

    aten::vec3 ao_color(0.0f);

    for (int i = 0; i < num_ao_rays; i++) {
        // Compute hemisphere ray.
        aten::ray ao_ray;

        // Closest hit.
        aten::Intersection isectTmp;

        bool isHit = intersectCloser(&ctxt, ao_ray, &isectTmp, ao_radius);

        if (isHit) {
            aten::vec3 ao;
            ao_color += ao;
        }
    }

    ao_color /= num_ao_rays;
    path.contrib = ao_color;
}

__global__ void gatherAO(
    int width, int height,
    cudaSurfaceObject_t outSurface,
    const idaten::AORenderer::Path* __restrict__ paths,
    bool enableProgressive)
{
    const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const auto idx = getIdx(ix, iy, width);

    const auto& path = paths[idx];

    int sample = path.samples;

    float4 data;

    if (enableProgressive) {
        surf2Dread(&data, outSurface, ix * sizeof(float4), iy);

        // First data.w value is 0.
        int n = data.w;
        data = n * data + make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
        data /= (n + 1);
        data.w = n + 1;
    }
    else {
        data = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
        data.w = sample;
    }

    surf2Dwrite(
        data,
        outSurface,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

namespace idaten {
    void AORenderer::onGenPath(
        int width, int height,
        int sample, int maxSamples,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        genPathAO << <grid, block >> > (
            m_tileDomain,
            m_paths.ptr(),
            m_rays.ptr(),
            sample, maxSamples,
            m_frame,
            m_cam.ptr(),
            m_sobolMatrices.ptr(),
            m_random.ptr());

        checkCudaKernel(genPath);
    }

    void AORenderer::onHitTest(
        int width, int height,
        cudaTextureObject_t texVtxPos)
    {
        dim3 blockPerGrid_HitTest((m_tileDomain.w * m_tileDomain.h + 128 - 1) / 128);
        dim3 threadPerBlock_HitTest(128);

#ifdef ENABLE_PERSISTENT_THREAD
        hitTestAO << <NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK) >> > (
#else
        hitTestAO << <blockPerGrid_HitTest, threadPerBlock_HitTest >> > (
#endif
        //hitTestAO << <1, 1 >> > (
            m_tileDomain,
            m_paths.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
            m_hitbools.ptr(),
            m_shapeparam.ptr(), m_shapeparam.num(),
            m_lightparam.ptr(), m_lightparam.num(),
            m_nodetex.ptr(),
            m_primparams.ptr(),
            texVtxPos,
            m_mtxparams.ptr());

        checkCudaKernel(hitTest);
    }

    void AORenderer::onShadeMiss(
        int width, int height,
        int bounce)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        bool isFirstBounce = bounce == 0;

        shadeMissAO << <grid, block >> > (
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

        shadeAO << <blockPerGrid, threadPerBlock >> > (
        //shade<true> << <1, 1 >> > (
            m_ao_num_rays, m_ao_radius,
            m_tileDomain,
            m_frame,
            m_paths.ptr(),
            m_hitidx.ptr(), hitcount.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
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
        idaten::TypedCudaMemory<idaten::AORenderer::Path>& paths,
        int width, int height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        gatherAO << <grid, block >> > (
            width, height,
            outputSurf,
            paths.ptr(),
            m_enableProgressive);
    }
}
