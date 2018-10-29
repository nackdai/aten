#include "kernel/pathtracing.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/StreamCompaction.h"
#include "kernel/pt_common.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"
#include "aten4idaten.h"

//#define ENABLE_SEPARATE_ALBEDO

#define ENABLE_PERSISTENT_THREAD

__global__ void genPath(
    idaten::TileDomain tileDomain,
    idaten::PathTracing::Path* paths,
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

__device__ unsigned int headDev = 0;

__global__ void hitTest(
    idaten::TileDomain tileDomain,
    idaten::PathTracing::Path* paths,
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
        headDev = 0;
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
            headWarp = atomicAdd(&headDev, WARP_SIZE);
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

__global__ void shadeMiss(
    bool isFirstBounce, bool needAOV,
    idaten::TileDomain tileDomain,
    cudaSurfaceObject_t* aovs,
    idaten::PathTracing::Path* paths)
{
    const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    const auto idx = getIdx(ix, iy, tileDomain.w);

    auto& path = paths[idx];

    if (!path.isTerminate && !path.isHit) {
        // TODO
        auto bg = aten::vec3(0);

        if (isFirstBounce) {
            path.isKill = true;

            if (needAOV) {
                surf2Dwrite(
                    make_float4(bg.x, bg.y, bg.z, 1),
                    aovs[2],
                    ix * sizeof(float4), iy,
                    cudaBoundaryModeTrap);
            }

#ifdef ENABLE_SEPARATE_ALBEDO
            // For exporting separated albedo.
            bg = aten::vec3(1, 1, 1);
#endif
        }

        path.contrib += path.throughput * bg;

        path.isTerminate = true;
    }
}

__global__ void shadeMissWithEnvmap(
    bool isFirstBounce, bool needAOV,
    idaten::TileDomain tileDomain,
    cudaSurfaceObject_t* aovs,
    cudaTextureObject_t* textures,
    int envmapIdx,
    real envmapAvgIllum,
    real envmapMultiplyer,
    idaten::PathTracing::Path* paths,
    const aten::ray* __restrict__ rays)
{
    const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    const auto idx = getIdx(ix, iy, tileDomain.w);

    auto& path = paths[idx];

    if (!path.isTerminate && !path.isHit) {
        auto r = rays[idx];

        auto uv = AT_NAME::envmap::convertDirectionToUV(r.dir);

        auto bg = tex2D<float4>(textures[envmapIdx], uv.x, uv.y);
        auto emit = aten::vec3(bg.x, bg.y, bg.z);

        float misW = 1.0f;
        if (isFirstBounce) {
            path.isKill = true;

            if (needAOV) {
                surf2Dwrite(
                    make_float4(emit.x, emit.y, emit.z, 1),
                    aovs[2],
                    ix * sizeof(float4), iy,
                    cudaBoundaryModeTrap);
            }

#ifdef ENABLE_SEPARATE_ALBEDO
            // For exporting separated albedo.
            emit = aten::vec3(1, 1, 1);
#endif
        }
        else {
            auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, envmapAvgIllum);
            misW = path.pdfb / (pdfLight + path.pdfb);

            emit *= envmapMultiplyer;
        }

        path.contrib += path.throughput * misW * emit;

        path.isTerminate = true;
    }
}

__global__ void shade(
    bool needAOV,
    idaten::TileDomain tileDomain,
    unsigned int frame,
    cudaSurfaceObject_t* aovs,
    float3 posRange,
    idaten::PathTracing::Path* paths,
    int* hitindices,
    int* hitnum,
    const aten::Intersection* __restrict__ isects,
    aten::ray* rays,
    int bounce, int rrBounce,
    const aten::GeomParameter* __restrict__ shapes, int geomnum,
    aten::MaterialParameter* mtrls,
    const aten::LightParameter* __restrict__ lights, int lightnum,
    cudaTextureObject_t* nodes,
    const aten::PrimitiveParamter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    cudaTextureObject_t vtxNml,
    const aten::mat4* __restrict__ matrices,
    cudaTextureObject_t* textures,
    const unsigned int* random,
    idaten::PathTracing::ShadowRay* shadowRays)
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
        ctxt.vtxNml = vtxNml;
        ctxt.matrices = matrices;
        ctxt.textures = textures;
    }

    idx = hitindices[idx];

    auto& path = paths[idx];
    const auto& ray = rays[idx];

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    auto scramble = random[idx] * 0x1fe3434f;
    path.sampler.init(frame, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    path.sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + bounce * 300, scramble);
#endif

    aten::hitrecord rec;

    const auto& isect = isects[idx];

    auto obj = &ctxt.shapes[isect.objid];
    evalHitResult(&ctxt, obj, ray, &rec, &isect);

    aten::MaterialParameter mtrl = ctxt.mtrls[rec.mtrlid];

    bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

    // 交差位置の法線.
    // 物体からのレイの入出を考慮.
    aten::vec3 orienting_normal = rec.normal;

    if (mtrl.type != aten::MaterialType::Layer) {
        mtrl.albedoMap = (int)(mtrl.albedoMap >= 0 ? ctxt.textures[mtrl.albedoMap] : -1);
        mtrl.normalMap = (int)(mtrl.normalMap >= 0 ? ctxt.textures[mtrl.normalMap] : -1);
        mtrl.roughnessMap = (int)(mtrl.roughnessMap >= 0 ? ctxt.textures[mtrl.roughnessMap] : -1);
    }

#if 1
    if (needAOV) {
        int ix = idx % tileDomain.w;
        int iy = idx / tileDomain.w;

        ix += tileDomain.x;
        iy += tileDomain.y;

        auto p = make_float3(rec.p.x, rec.p.y, rec.p.z);
        p /= posRange;

        auto n = (orienting_normal + 1.0f) * 0.5f;

        // position
        surf2Dwrite(
            make_float4(p.x, p.y, p.z, rec.mtrlid),
            aovs[0],
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);

        // normal
        surf2Dwrite(
            make_float4(n.x, n.y, n.z, isect.meshid),
            aovs[1],
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);

        auto albedo = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, aten::vec3(1.0f));

        surf2Dwrite(
            make_float4(albedo.x, albedo.y, albedo.z, 1),
            aovs[2],
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }
#endif

#ifdef ENABLE_SEPARATE_ALBEDO
    // For exporting separated albedo.
    mtrl.albedoMap = -1;
#endif

    // Implicit conection to light.
    if (mtrl.attrib.isEmissive) {
        if (!isBackfacing) {
            float weight = 1.0f;

            if (bounce > 0 && !path.isSingular) {
                auto cosLight = dot(orienting_normal, -ray.dir);
                auto dist2 = aten::squared_length(rec.p - ray.org);

                if (cosLight >= 0) {
                    auto pdfLight = 1 / rec.area;

                    // Convert pdf area to sradian.
                    // http://www.slideshare.net/h013/edubpt-v100
                    // p31 - p35
                    pdfLight = pdfLight * dist2 / cosLight;

                    weight = path.pdfb / (pdfLight + path.pdfb);
                }
            }

            path.contrib += path.throughput * weight * mtrl.baseColor;
        }

        // When ray hit the light, tracing will finish.
        path.isTerminate = true;
        return;
    }

    if (!mtrl.attrib.isTranslucent && isBackfacing) {
        orienting_normal = -orienting_normal;
    }

    // Apply normal map.
    int normalMap = mtrl.normalMap;
    if (mtrl.type == aten::MaterialType::Layer) {
        // 最表層の NormalMap を適用.
        auto* topmtrl = &ctxt.mtrls[mtrl.layer[0]];
        normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
    }
    AT_NAME::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);

    shadowRays[idx].isActive = false;

    // Explicit conection to light.
    if (!(mtrl.attrib.isSingular || mtrl.attrib.isTranslucent))
    {
        real lightSelectPdf = 1;
        aten::LightSampleResult sampleres;

        // TODO
        // Importance sampling.
        int lightidx = aten::cmpMin<int>(path.sampler.nextSample() * lightnum, lightnum - 1);
        lightSelectPdf = 1.0f / lightnum;

        aten::LightParameter light;
        light.pos = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 0];
        light.dir = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 1];
        light.le = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 2];
        light.v0 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 3];
        light.v1 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 4];
        light.v2 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 5];
        //auto light = ctxt.lights[lightidx];

        sampleLight(&sampleres, &ctxt, &light, rec.p, orienting_normal, &path.sampler);

        const auto& posLight = sampleres.pos;
        const auto& nmlLight = sampleres.nml;
        real pdfLight = sampleres.pdf;

        auto lightobj = sampleres.obj;

        auto dirToLight = normalize(sampleres.dir);
        auto distToLight = length(posLight - rec.p);

        aten::Intersection isectTmp;

        auto shadowRayOrg = rec.p + AT_MATH_EPSILON * orienting_normal;
        auto tmp = rec.p + dirToLight - shadowRayOrg;
        auto shadowRayDir = normalize(tmp);

        shadowRays[idx].isActive = true;
        shadowRays[idx].r = aten::ray(shadowRayOrg, shadowRayDir);
        shadowRays[idx].targetLightId = lightidx;
        shadowRays[idx].distToLight = distToLight;

        {
            auto cosShadow = dot(orienting_normal, dirToLight);

            real pdfb = samplePDF(&ctxt, &mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
            auto bsdf = sampleBSDF(&ctxt, &mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);

            bsdf *= path.throughput;

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
                    shadowRays[idx].lightcontrib = (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
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
                        shadowRays[idx].lightcontrib = (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
                    }
                }
            }
        }
    }

    real russianProb = real(1);

    if (bounce > rrBounce) {
        auto t = normalize(path.throughput);
        auto p = aten::cmpMax(t.r, aten::cmpMax(t.g, t.b));

        russianProb = path.sampler.nextSample();

        if (russianProb >= p) {
            //path.contrib = aten::vec3(0);
            path.isTerminate = true;
        }
        else {
            russianProb = p;
        }
    }
            
    AT_NAME::MaterialSampling sampling;

    sampleMaterial(
        &sampling,
        &ctxt,
        &mtrl,
        orienting_normal,
        ray.dir,
        rec.normal,
        &path.sampler,
        rec.u, rec.v);

    auto nextDir = normalize(sampling.dir);
    auto pdfb = sampling.pdf;
    auto bsdf = sampling.bsdf;

    real c = 1;
    if (!mtrl.attrib.isSingular) {
        // TODO
        // AMDのはabsしているが....
        c = aten::abs(dot(orienting_normal, nextDir));
        //c = dot(orienting_normal, nextDir);
    }

    if (pdfb > 0 && c > 0) {
        path.throughput *= bsdf * c / pdfb;
        path.throughput /= russianProb;
    }
    else {
        path.isTerminate = true;
    }

    // Make next ray.
    rays[idx] = aten::ray(rec.p, nextDir);

    path.pdfb = pdfb;
    path.isSingular = mtrl.attrib.isSingular;
}

__global__ void hitShadowRay(
    idaten::PathTracing::Path* paths,
    int* hitindices,
    int* hitnum,
    const idaten::PathTracing::ShadowRay* __restrict__ shadowRays,
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

    auto& shadowRay = shadowRays[idx];

    if (shadowRay.isActive) {
        auto light = &ctxt.lights[shadowRay.targetLightId];
        auto lightobj = (light->objid >= 0 ? &ctxt.shapes[light->objid] : nullptr);

        real distHitObjToRayOrg = AT_MATH_INF;

        // Ray aim to the area light.
        // So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
        const aten::GeomParameter* hitobj = lightobj;

        aten::Intersection isectTmp;

        bool isHit = false;
        isHit = intersectCloser(&ctxt, shadowRay.r, &isectTmp, shadowRay.distToLight - AT_MATH_EPSILON);

        if (isHit) {
            hitobj = &ctxt.shapes[isectTmp.objid];
        }
        
        isHit = AT_NAME::scene::hitLight(
            isHit, 
            light->attrib,
            lightobj,
            shadowRay.distToLight,
            distHitObjToRayOrg,
            isectTmp.t,
            hitobj);

        if (isHit) {
            paths[idx].contrib += shadowRay.lightcontrib;
        }
    }
}

__global__ void gather(
    int width, int height,
    cudaSurfaceObject_t outSurface,
    const idaten::PathTracing::Path* __restrict__ paths,
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
    void PathTracing::onGenPath(
        int width, int height,
        int sample, int maxSamples,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        genPath << <grid, block >> > (
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

    void PathTracing::onHitTest(
        int width, int height,
        cudaTextureObject_t texVtxPos)
    {
        dim3 blockPerGrid_HitTest((m_tileDomain.w * m_tileDomain.h + 128 - 1) / 128);
        dim3 threadPerBlock_HitTest(128);

#ifdef ENABLE_PERSISTENT_THREAD
        hitTest << <NUM_BLOCK, dim3(WARP_SIZE, NUM_WARP_PER_BLOCK) >> > (
#else
        hitTest << <blockPerGrid_HitTest, threadPerBlock_HitTest >> > (
#endif
        //hitTest << <1, 1 >> > (
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

    void PathTracing::onShadeMiss(
        int width, int height,
        int bounce)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        bool enableAOV = (bounce == 0 && m_enableAOV);

        bool isFirstBounce = bounce == 0;

        if (m_envmapRsc.idx >= 0) {
            shadeMissWithEnvmap << <grid, block >> > (
                isFirstBounce, enableAOV,
                m_tileDomain,
                m_aovCudaRsc.ptr(),
                m_tex.ptr(),
                m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
                m_paths.ptr(),
                m_rays.ptr());
        }
        else {
            shadeMiss << <grid, block >> > (
                isFirstBounce, enableAOV,
                m_tileDomain,
                m_aovCudaRsc.ptr(),
                m_paths.ptr());
        }

        checkCudaKernel(shadeMiss);
    }

    void PathTracing::onShade(
        int width, int height,
        int bounce, int rrBounce,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 blockPerGrid((m_tileDomain.w * m_tileDomain.h + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        bool enableAOV = (bounce == 0 && m_enableAOV);
        float3 posRange = make_float3(m_posRange.x, m_posRange.y, m_posRange.z);

        auto& hitcount = m_compaction.getCount();

        shade << <blockPerGrid, threadPerBlock >> > (
        //shade<true> << <1, 1 >> > (
            enableAOV,
            m_tileDomain,
            m_frame,
            m_aovCudaRsc.ptr(), posRange,
            m_paths.ptr(),
            m_hitidx.ptr(), hitcount.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
            bounce, rrBounce,
            m_shapeparam.ptr(), m_shapeparam.num(),
            m_mtrlparam.ptr(),
            m_lightparam.ptr(), m_lightparam.num(),
            m_nodetex.ptr(),
            m_primparams.ptr(),
            texVtxPos, texVtxNml,
            m_mtxparams.ptr(),
            m_tex.ptr(),
            m_random.ptr(),
            m_shadowRays.ptr());

        checkCudaKernel(shade);

        hitShadowRay << <blockPerGrid, threadPerBlock >> > (
            //hitShadowRay << <1, 1 >> > (
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

    void PathTracing::onGather(
        cudaSurfaceObject_t outputSurf,
        idaten::TypedCudaMemory<idaten::PathTracing::Path>& paths,
        int width, int height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        gather << <grid, block >> > (
            width, height,
            outputSurf,
            paths.ptr(),
            m_enableProgressive);
    }
}
