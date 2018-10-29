#include "svgf/svgf.h"

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

#define ENABLE_PERSISTENT_THREAD

__global__ void genPath(
    idaten::TileDomain tileDomain,
    bool isFillAOV,
    idaten::SVGFPathTracing::Path* paths,
    aten::ray* rays,
    int width, int height,
    int sample, int maxSamples,
    unsigned int frame,
    const aten::CameraParameter* __restrict__ camera,
    const unsigned int* sobolmatrices,
    unsigned int* random)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const auto idx = getIdx(ix, iy, width);

    paths->attrib[idx].isHit = false;

    if (paths->attrib[idx].isKill) {
        paths->attrib[idx].isTerminate = true;
        return;
    }

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    auto scramble = random[idx] * 0x1fe3434f;
    paths->sampler[idx].init(frame, 0, scramble, sobolmatrices);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f * ((frame + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    paths->sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 0, scramble);
#endif

    float r1 = paths->sampler[idx].nextSample();
    float r2 = paths->sampler[idx].nextSample();

    if (isFillAOV) {
        r1 = r2 = 0.5f;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    float s = (ix + r1) / (float)(camera->width);
    float t = (iy + r2) / (float)(camera->height);

    AT_NAME::CameraSampleResult camsample;
    AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

    rays[idx] = camsample.r;

    paths->throughput[idx].throughput = aten::vec3(1);
    paths->throughput[idx].pdfb = 0.0f;
    paths->attrib[idx].isTerminate = false;
    paths->attrib[idx].isSingular = false;

    paths->contrib[idx].samples += 1;

    // Accumulate value, so do not reset.
    //path.contrib = aten::vec3(0);
}

// NOTE
// persistent thread.
// https://gist.github.com/guozhou/b972bb42bbc5cba1f062#file-persistent-cpp-L15

// NOTE
// compute capability 6.0
// http://homepages.math.uic.edu/~jan/mcs572/performance_considerations.pdf
// p3

#define NUM_SM                64    // no. of streaming multiprocessors
#define NUM_WARP_PER_SM        64    // maximum no. of resident warps per SM
#define NUM_BLOCK_PER_SM    32    // maximum no. of resident blocks per SM
#define NUM_BLOCK            (NUM_SM * NUM_BLOCK_PER_SM)
#define NUM_WARP_PER_BLOCK    (NUM_WARP_PER_SM / NUM_BLOCK_PER_SM)
#define WARP_SIZE            32

__device__ unsigned int g_headDev = 0;

__global__ void hitTest(
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
#ifdef ENABLE_PERSISTENT_THREAD
    // warp-wise head index of tasks in a block
    __shared__ volatile unsigned int headBlock[NUM_WARP_PER_BLOCK];

    volatile unsigned int& headWarp = headBlock[threadIdx.y];

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        g_headDev = 0;
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

__global__ void shadeMiss(
    idaten::TileDomain tileDomain,
    int bounce,
    float4* aovNormalDepth,
    float4* aovTexclrMeshid,
    idaten::SVGFPathTracing::Path* paths,
    int width, int height)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    const auto idx = getIdx(ix, iy, tileDomain.w);

    if (!paths->attrib[idx].isTerminate && !paths->attrib[idx].isHit) {
        // TODO
        auto bg = aten::vec3(0);

        if (bounce == 0) {
            paths->attrib[idx].isKill = true;

            ix += tileDomain.x;
            iy += tileDomain.y;
            const auto _idx = getIdx(ix, iy, width);

            // Export bg color to albedo buffer.
            aovTexclrMeshid[_idx] = make_float4(bg.x, bg.y, bg.z, -1);
            aovNormalDepth[_idx].w = -1;

            // For exporting separated albedo.
            bg = aten::vec3(1, 1, 1);
        }

        auto contrib = paths->throughput[idx].throughput * bg;
        paths->contrib[idx].contrib += make_float3(contrib.x, contrib.y, contrib.z);

        paths->attrib[idx].isTerminate = true;
    }
}

__global__ void shadeMissWithEnvmap(
    idaten::TileDomain tileDomain,
    int offsetX, int offsetY,
    int bounce,
    const aten::CameraParameter* __restrict__ camera,
    float4* aovNormalDepth,
    float4* aovTexclrMeshid,
    cudaTextureObject_t* textures,
    int envmapIdx,
    real envmapAvgIllum,
    real envmapMultiplyer,
    idaten::SVGFPathTracing::Path* paths,
    const aten::ray* __restrict__ rays,
    int width, int height)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    const auto idx = getIdx(ix, iy, tileDomain.w);

    if (!paths->attrib[idx].isTerminate && !paths->attrib[idx].isHit) {
        aten::vec3 dir = rays[idx].dir;

        if (bounce == 0) {
            // Suppress jittering envrinment map.
            // So, re-sample ray without random.

            // TODO
            // More efficient way...

            float s = (ix + offsetX) / (float)(width);
            float t = (iy + offsetY) / (float)(height);

            AT_NAME::CameraSampleResult camsample;
            AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

            dir = camsample.r.dir;
        }

        auto uv = AT_NAME::envmap::convertDirectionToUV(dir);

        auto bg = tex2D<float4>(textures[envmapIdx], uv.x, uv.y);
        auto emit = aten::vec3(bg.x, bg.y, bg.z);

        float misW = 1.0f;
        if (bounce == 0
            || (bounce == 1 && paths->attrib[idx].isSingular))
        {
            paths->attrib[idx].isKill = true;

            ix += tileDomain.x;
            iy += tileDomain.y;
            const auto _idx = getIdx(ix, iy, width);

            // Export envmap to albedo buffer.
            aovTexclrMeshid[_idx] = make_float4(emit.x, emit.y, emit.z, -1);
            aovNormalDepth[_idx].w = -1;

            // For exporting separated albedo.
            emit = aten::vec3(1, 1, 1);
        }
        else {
            auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, envmapAvgIllum);
            misW = paths->throughput[idx].pdfb / (pdfLight + paths->throughput[idx].pdfb);

            emit *= envmapMultiplyer;
        }
        
        auto contrib = paths->throughput[idx].throughput * misW * emit;
        paths->contrib[idx].contrib += make_float3(contrib.x, contrib.y, contrib.z);

        paths->attrib[idx].isTerminate = true;
    }
}

__global__ void shade(
    idaten::TileDomain tileDomain,
    float4* aovNormalDepth,
    float4* aovTexclrMeshid,
    aten::mat4 mtxW2C,
    int width, int height,
    idaten::SVGFPathTracing::Path* paths,
    const int* __restrict__ hitindices,
    int* hitnum,
    const aten::Intersection* __restrict__ isects,
    aten::ray* rays,
    int frame,
    int bounce, int rrBounce,
    const aten::GeomParameter* __restrict__ shapes, int geomnum,
    const aten::MaterialParameter* __restrict__ mtrls,
    const aten::LightParameter* __restrict__ lights, int lightnum,
    const aten::PrimitiveParamter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    cudaTextureObject_t vtxNml,
    const aten::mat4* __restrict__ matrices,
    cudaTextureObject_t* textures,
    unsigned int* random,
    idaten::SVGFPathTracing::ShadowRay* shadowRays)
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
        ctxt.lightnum = lightnum;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.vtxPos = vtxPos;
        ctxt.vtxNml = vtxNml;
        ctxt.matrices = matrices;
        ctxt.textures = textures;
    }

    idx = hitindices[idx];

    __shared__ idaten::SVGFPathTracing::ShadowRay shShadowRays[64 * idaten::SVGFPathTracing::ShadowRayNum];
    __shared__ aten::MaterialParameter shMtrls[64];

    const auto ray = rays[idx];

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    auto scramble = random[idx] * 0x1fe3434f;
    paths->sampler[idx].init(frame, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    paths->sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + bounce * 300, scramble);
#endif

    aten::hitrecord rec;

    const auto& isect = isects[idx];

    auto obj = &ctxt.shapes[isect.objid];
    evalHitResult(&ctxt, obj, ray, &rec, &isect);

    bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

    // 交差位置の法線.
    // 物体からのレイの入出を考慮.
    aten::vec3 orienting_normal = rec.normal;

    if (rec.mtrlid >= 0) {
        shMtrls[threadIdx.x] = ctxt.mtrls[rec.mtrlid];

#if 1
        if (rec.isVoxel)
        {
            // Replace to lambert.
            const auto& albedo = ctxt.mtrls[rec.mtrlid].baseColor;
            shMtrls[threadIdx.x] = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
            shMtrls[threadIdx.x].baseColor = albedo;
        }
#endif

        if (shMtrls[threadIdx.x].type != aten::MaterialType::Layer) {
            shMtrls[threadIdx.x].albedoMap = (int)(shMtrls[threadIdx.x].albedoMap >= 0 ? ctxt.textures[shMtrls[threadIdx.x].albedoMap] : -1);
            shMtrls[threadIdx.x].normalMap = (int)(shMtrls[threadIdx.x].normalMap >= 0 ? ctxt.textures[shMtrls[threadIdx.x].normalMap] : -1);
            shMtrls[threadIdx.x].roughnessMap = (int)(shMtrls[threadIdx.x].roughnessMap >= 0 ? ctxt.textures[shMtrls[threadIdx.x].roughnessMap] : -1);
        }
    }
    else {
        // TODO
        shMtrls[threadIdx.x] = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
        shMtrls[threadIdx.x].baseColor = aten::vec3(1.0f);
    }


    // Render AOVs.
    // NOTE
    // 厳密に法線をAOVに保持するなら、法線マップ適用後するべき.
    // しかし、temporal reprojection、atrousなどのフィルタ適用時に法線を参照する際に、法線マップが細かすぎてはじかれてしまうことがある.
    // それにより、フィルタがおもったようにかからずフィルタの品質が下がってしまう問題が発生する.
    if (bounce == 0) {
        int ix = idx % tileDomain.w;
        int iy = idx / tileDomain.w;

        ix += tileDomain.x;
        iy += tileDomain.y;

        const auto _idx = getIdx(ix, iy, width);

        // World coordinate to Clip coordinate.
        aten::vec4 pos = aten::vec4(rec.p, 1);
        pos = mtxW2C.apply(pos);

        // normal, depth
        aovNormalDepth[_idx] = make_float4(orienting_normal.x, orienting_normal.y, orienting_normal.z, pos.w);

        // texture color, meshid.
        auto texcolor = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec3(1.0f));
#if 0
        aovTexclrMeshid[_idx] = make_float4(texcolor.x, texcolor.y, texcolor.z, isect.meshid);
#else
        aovTexclrMeshid[_idx] = make_float4(texcolor.x, texcolor.y, texcolor.z, isect.mtrlid);
#endif

        // For exporting separated albedo.
        shMtrls[threadIdx.x].albedoMap = -1;
    }
    // TODO
    // How to deal Refraction?
    else if (bounce == 1 && paths->attrib[idx].mtrlType == aten::MaterialType::Specular) {
        int ix = idx % tileDomain.w;
        int iy = idx / tileDomain.w;

        ix += tileDomain.x;
        iy += tileDomain.y;

        const auto _idx = getIdx(ix, iy, width);

        // World coordinate to Clip coordinate.
        aten::vec4 pos = aten::vec4(rec.p, 1);
        pos = mtxW2C.apply(pos);

        // normal, depth
        aovNormalDepth[_idx] = make_float4(orienting_normal.x, orienting_normal.y, orienting_normal.z, pos.w);

        // texture color.
        auto texcolor = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec3(1.0f));
#if 0
        aovTexclrMeshid[_idx] = make_float4(texcolor.x, texcolor.y, texcolor.z, isect.meshid);
#else
        aovTexclrMeshid[_idx] = make_float4(texcolor.x, texcolor.y, texcolor.z, isect.mtrlid);
#endif

        // For exporting separated albedo.
        shMtrls[threadIdx.x].albedoMap = -1;
    }

    // Implicit conection to light.
    if (shMtrls[threadIdx.x].attrib.isEmissive) {
        if (!isBackfacing) {
            float weight = 1.0f;

            if (bounce > 0 && !paths->attrib[idx].isSingular) {
                auto cosLight = dot(orienting_normal, -ray.dir);
                auto dist2 = aten::squared_length(rec.p - ray.org);

                if (cosLight >= 0) {
                    auto pdfLight = 1 / rec.area;

                    // Convert pdf area to sradian.
                    // http://www.slideshare.net/h013/edubpt-v100
                    // p31 - p35
                    pdfLight = pdfLight * dist2 / cosLight;

                    weight = paths->throughput[idx].pdfb / (pdfLight + paths->throughput[idx].pdfb);
                }
            }

            auto contrib = paths->throughput[idx].throughput * weight * shMtrls[threadIdx.x].baseColor;
            paths->contrib[idx].contrib += make_float3(contrib.x, contrib.y, contrib.z);
        }

        // When ray hit the light, tracing will finish.
        paths->attrib[idx].isTerminate = true;
        return;
    }

    if (!shMtrls[threadIdx.x].attrib.isTranslucent && isBackfacing) {
        orienting_normal = -orienting_normal;
    }

    // Apply normal map.
    int normalMap = shMtrls[threadIdx.x].normalMap;
    if (shMtrls[threadIdx.x].type == aten::MaterialType::Layer) {
        // 最表層の NormalMap を適用.
        auto* topmtrl = &ctxt.mtrls[shMtrls[threadIdx.x].layer[0]];
        normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
    }
    AT_NAME::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);

    auto albedo = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec3(1), bounce);

#if 1
#pragma unroll
    for (int i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
        shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].isActive = false;
    }

    // Explicit conection to light.
    if (!(shMtrls[threadIdx.x].attrib.isSingular || shMtrls[threadIdx.x].attrib.isTranslucent))
    {
        auto shadowRayOrg = rec.p + AT_MATH_EPSILON * orienting_normal;

        for (int i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
            real lightSelectPdf = 1;
            aten::LightSampleResult sampleres;

            // TODO
            // Importance sampling.
            int lightidx = aten::cmpMin<int>(paths->sampler[idx].nextSample() * lightnum, lightnum - 1);
            lightSelectPdf = 1.0f / lightnum;

            aten::LightParameter light;
            light.pos = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 0];
            light.dir = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 1];
            light.le = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 2];
            light.v0 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 3];
            light.v1 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 4];
            light.v2 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 5];
            //auto light = ctxt.lights[lightidx];

            sampleLight(&sampleres, &ctxt, &light, rec.p, orienting_normal, &paths->sampler[idx], bounce);

            const auto& posLight = sampleres.pos;
            const auto& nmlLight = sampleres.nml;
            real pdfLight = sampleres.pdf;

            auto dirToLight = normalize(sampleres.dir);
            auto distToLight = length(posLight - rec.p);

            auto tmp = rec.p + dirToLight - shadowRayOrg;
            auto shadowRayDir = normalize(tmp);

            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].isActive = true;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].rayorg = shadowRayOrg;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].raydir = shadowRayDir;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].targetLightId = lightidx;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].distToLight = distToLight;
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].lightcontrib = aten::vec3(0);
            {
                auto cosShadow = dot(orienting_normal, dirToLight);

                real pdfb = samplePDF(&ctxt, &shMtrls[threadIdx.x], orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
                auto bsdf = sampleBSDF(&ctxt, &shMtrls[threadIdx.x], orienting_normal, ray.dir, dirToLight, rec.u, rec.v, albedo);

                bsdf *= paths->throughput[idx].throughput;

                // Get light color.
                auto emit = sampleres.finalColor;

                if (light.attrib.isSingular || light.attrib.isInfinite) {
                    if (pdfLight > real(0) && cosShadow >= 0) {
                        // TODO
                        // ジオメトリタームの扱いについて.
                        // singular light の場合は、finalColor に距離の除算が含まれている.
                        // inifinite light の場合は、無限遠方になり、pdfLightに含まれる距離成分と打ち消しあう？.
                        // （打ち消しあうので、pdfLightには距離成分は含んでいない）.
                        auto misW = pdfLight / (pdfb + pdfLight);

                        shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].lightcontrib =
                            (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf / (float)idaten::SVGFPathTracing::ShadowRayNum;
                    }
                }
                else {
                    auto cosLight = dot(nmlLight, -dirToLight);

                    if (cosShadow >= 0 && cosLight >= 0) {
                        auto dist2 = aten::squared_length(sampleres.dir);
                        auto G = cosShadow * cosLight / dist2;

                        if (pdfb > real(0) && pdfLight > real(0)) {
                            // Convert pdf from steradian to area.
                            // http://www.slideshare.net/h013/edubpt-v100
                            // p31 - p35
                            pdfb = pdfb * cosLight / dist2;

                            auto misW = pdfLight / (pdfb + pdfLight);

                            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].lightcontrib =
                                (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf / (float)idaten::SVGFPathTracing::ShadowRayNum;;
                        }
                    }
                }
            }
        }
    }
#endif

    real russianProb = real(1);

    if (bounce > rrBounce) {
        auto t = normalize(paths->throughput[idx].throughput);
        auto p = aten::cmpMax(t.r, aten::cmpMax(t.g, t.b));

        russianProb = paths->sampler[idx].nextSample();

        if (russianProb >= p) {
            //shPaths[threadIdx.x].contrib = aten::vec3(0);
            paths->attrib[idx].isTerminate = true;
        }
        else {
            russianProb = max(p, 0.01f);
        }
    }
            
    AT_NAME::MaterialSampling sampling;

    sampleMaterial(
        &sampling,
        &ctxt,
        &shMtrls[threadIdx.x],
        orienting_normal,
        ray.dir,
        rec.normal,
        &paths->sampler[idx],
        rec.u, rec.v,
        albedo);

    auto nextDir = normalize(sampling.dir);
    auto pdfb = sampling.pdf;
    auto bsdf = sampling.bsdf;

    real c = 1;
    if (!shMtrls[threadIdx.x].attrib.isSingular) {
        // TODO
        // AMDのはabsしているが....
        c = aten::abs(dot(orienting_normal, nextDir));
        //c = dot(orienting_normal, nextDir);
    }

    if (pdfb > 0 && c > 0) {
        paths->throughput[idx].throughput *= bsdf * c / pdfb;
        paths->throughput[idx].throughput /= russianProb;
    }
    else {
        paths->attrib[idx].isTerminate = true;
    }

    // Make next ray.
    rays[idx] = aten::ray(rec.p, nextDir);

    paths->throughput[idx].pdfb = pdfb;
    paths->attrib[idx].isSingular = shMtrls[threadIdx.x].attrib.isSingular;
    paths->attrib[idx].mtrlType = shMtrls[threadIdx.x].type;

#pragma unroll
    for (int i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
        shadowRays[idx * idaten::SVGFPathTracing::ShadowRayNum + i] = shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i];
    }
}

__global__ void hitShadowRay(
    int bounce,
    idaten::SVGFPathTracing::Path* paths,
    int* hitindices,
    int* hitnum,
    const idaten::SVGFPathTracing::ShadowRay* __restrict__ shadowRays,
    const aten::GeomParameter* __restrict__ shapes, int geomnum,
    aten::MaterialParameter* mtrls,
    const aten::LightParameter* __restrict__ lights, int lightnum,
    cudaTextureObject_t* nodes,
    const aten::PrimitiveParamter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    const aten::mat4* __restrict__ matrices)
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
        ctxt.lightnum = lightnum;
        ctxt.lights = lights;
        ctxt.nodes = nodes;
        ctxt.prims = prims;
        ctxt.vtxPos = vtxPos;
        ctxt.matrices = matrices;
    }

    idx = hitindices[idx];

#pragma unroll
    for (int i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
        const auto& shadowRay = shadowRays[idx * idaten::SVGFPathTracing::ShadowRayNum + i];

        if (!shadowRay.isActive) {
            continue;
        }
        auto targetLightId = shadowRay.targetLightId;
        auto distToLight = shadowRay.distToLight;

        auto light = ctxt.lights[targetLightId];
        auto lightobj = (light.objid >= 0 ? &ctxt.shapes[light.objid] : nullptr);

        real distHitObjToRayOrg = AT_MATH_INF;

        // Ray aim to the area light.
        // So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
        const aten::GeomParameter* hitobj = lightobj;

        aten::Intersection isectTmp;

        bool isHit = false;

        aten::ray r(shadowRay.rayorg, shadowRay.raydir);

        // TODO
        bool enableLod = (bounce >= 2);

        isHit = intersectCloser(&ctxt, r, &isectTmp, distToLight - AT_MATH_EPSILON, enableLod);

        if (isHit) {
            hitobj = &ctxt.shapes[isectTmp.objid];
        }

        isHit = AT_NAME::scene::hitLight(
            isHit,
            light.attrib,
            lightobj,
            distToLight,
            distHitObjToRayOrg,
            isectTmp.t,
            hitobj);

        if (isHit) {
            auto contrib = shadowRay.lightcontrib;
            paths->contrib[idx].contrib += make_float3(contrib.x, contrib.y, contrib.z);
        }
    }
}

__global__ void gather(
    idaten::TileDomain tileDomain,
    cudaSurfaceObject_t dst,
    float4* aovColorVariance,
    float4* aovMomentTemporalWeight,
    const idaten::SVGFPathTracing::Path* __restrict__ paths,
    float4* contribs,
    int width, int height)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    auto idx = getIdx(ix, iy, tileDomain.w);

    float4 c = paths->contrib[idx].v;
    int sample = c.w;

    float3 contrib = make_float3(c.x, c.y, c.z) / sample;
    //contrib.w = sample;

    float lum = AT_NAME::color::luminance(contrib.x, contrib.y, contrib.z);

    ix += tileDomain.x;
    iy += tileDomain.y;
    idx = getIdx(ix, iy, width);

    aovMomentTemporalWeight[idx].x += lum * lum;
    aovMomentTemporalWeight[idx].y += lum;
    aovMomentTemporalWeight[idx].z += 1;

    aovColorVariance[idx] = make_float4(contrib.x, contrib.y, contrib.z, aovColorVariance[idx].w);

    contribs[idx] = c;

#if 0
    auto n = aovs[idx].moments.w;

    auto m = aovs[idx].moments / n;

    auto var = m.x - m.y * m.y;

    surf2Dwrite(
        make_float4(var, var, var, 1),
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
#else
    if (dst) {
        surf2Dwrite(
            make_float4(contrib, 0),
            dst,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }
#endif
}

namespace idaten
{
    void SVGFPathTracing::onGenPath(
        int sample, int maxSamples,
        int seed,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        bool isFillAOV = m_mode == Mode::AOVar;

        genPath << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            isFillAOV,
            m_paths.ptr(),
            m_rays.ptr(),
            m_tileDomain.w, m_tileDomain.h,
            sample, maxSamples,
            m_frame,
            m_cam.ptr(),
            m_sobolMatrices.ptr(),
            m_random.ptr());

        checkCudaKernel(genPath);
    }

    void SVGFPathTracing::onHitTest(
        int width, int height,
        int bounce,
        cudaTextureObject_t texVtxPos)
    {
        if (bounce == 0 && m_canSSRTHitTest) {
            onScreenSpaceHitTest(width, height, bounce, texVtxPos);
        }
        else {
#ifdef ENABLE_PERSISTENT_THREAD
            hitTest << <NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK), 0, m_stream >> > (
#else
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid(
                (m_tileDomain.w + block.x - 1) / block.x,
                (m_tileDomain.h + block.y - 1) / block.y);

            hitTest << <grid, block >> > (
#endif
                //hitTest << <1, 1 >> > (
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
    }

    void SVGFPathTracing::onShadeMiss(
        int width, int height,
        int bounce,
        int offsetX/*= -1*/,
        int offsetY/*= -1*/)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        int curaov = getCurAovs();

        offsetX = offsetX < 0 ? m_tileDomain.x : offsetX;
        offsetY = offsetY < 0 ? m_tileDomain.y : offsetY;

        if (m_envmapRsc.idx >= 0) {
            shadeMissWithEnvmap << <grid, block, 0, m_stream >> > (
                m_tileDomain,
                offsetX, offsetY,
                bounce,
                m_cam.ptr(),
                m_aovNormalDepth[curaov].ptr(),
                m_aovTexclrMeshid[curaov].ptr(),
                m_tex.ptr(),
                m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
                m_paths.ptr(),
                m_rays.ptr(),
                width, height);
        }
        else {
            shadeMiss << <grid, block, 0, m_stream >> > (
                m_tileDomain,
                bounce,
                m_aovNormalDepth[curaov].ptr(),
                m_aovTexclrMeshid[curaov].ptr(),
                m_paths.ptr(),
                width, height);
        }

        checkCudaKernel(shadeMiss);
    }

    void SVGFPathTracing::onShade(
        cudaSurfaceObject_t outputSurf,
        int width, int height,
        int bounce, int rrBounce,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        m_mtxW2V.lookat(
            m_camParam.origin,
            m_camParam.center,
            m_camParam.up);

        m_mtxV2C.perspective(
            m_camParam.znear,
            m_camParam.zfar,
            m_camParam.vfov,
            m_camParam.aspect);

        m_mtxC2V = m_mtxV2C;
        m_mtxC2V.invert();

        m_mtxV2W = m_mtxW2V;
        m_mtxV2W.invert();

        aten::mat4 mtxW2C = m_mtxV2C * m_mtxW2V;

        dim3 blockPerGrid(((m_tileDomain.w * m_tileDomain.h) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        int curaov = getCurAovs();

        shade << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_tileDomain,
            m_aovNormalDepth[curaov].ptr(),
            m_aovTexclrMeshid[curaov].ptr(),
            mtxW2C,
            width, height,
            m_paths.ptr(),
            m_hitidx.ptr(), hitcount.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
            m_frame,
            bounce, rrBounce,
            m_shapeparam.ptr(), m_shapeparam.num(),
            m_mtrlparam.ptr(),
            m_lightparam.ptr(), m_lightparam.num(),
            m_primparams.ptr(),
            texVtxPos, texVtxNml,
            m_mtxparams.ptr(),
            m_tex.ptr(),
            m_random.ptr(),
            m_shadowRays.ptr());

        checkCudaKernel(shade);

        hitShadowRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            bounce,
            m_paths.ptr(),
            m_hitidx.ptr(), hitcount.ptr(),
            m_shadowRays.ptr(),
            m_shapeparam.ptr(), m_shapeparam.num(),
            m_mtrlparam.ptr(),
            m_lightparam.ptr(), m_lightparam.num(),
            m_nodetex.ptr(),
            m_primparams.ptr(),
            texVtxPos,
            m_mtxparams.ptr());

        checkCudaKernel(hitShadowRay);
    }

    void SVGFPathTracing::onGather(
        cudaSurfaceObject_t outputSurf,
        int width, int height,
        int maxSamples)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        int curaov = getCurAovs();

        gather << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            outputSurf,
            m_aovColorVariance[curaov].ptr(),
            m_aovMomentTemporalWeight[curaov].ptr(),
            m_paths.ptr(),
            m_tmpBuf.ptr(),
            width, height);

        checkCudaKernel(gather);
    }
}
