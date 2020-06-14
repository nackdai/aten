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

__global__ void shadeASVGF(
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
    int sample,
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
    cudaTextureObject_t blueNoise,
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
    paths->sampler[idx].init(frame + sample, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f
        * (((frame + sample) + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    paths->sampler[idx].init(
        (frame + sample) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
        4 + bounce * 300,
        scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_BLUENOISE
    // Not need to do.
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

namespace idaten
{
    void AdvancedSVGFPathTracing::onShade(
        cudaSurfaceObject_t outputSurf,
        int width, int height,
        int sample,
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

        auto blueNoise = m_bluenoise.bind();

        shadeASVGF << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_tileDomain,
            m_aovNormalDepth[curaov].ptr(),
            m_aovTexclrMeshid[curaov].ptr(),
            mtxW2C,
            width, height,
            m_paths.ptr(),
            m_hitidx.ptr(), hitcount.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
            sample,
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
            blueNoise,
            m_shadowRays.ptr());

        checkCudaKernel(shade);

        onShadeByShadowRay(bounce, texVtxPos);

        m_bluenoise.unbind();
    }
}
