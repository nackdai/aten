#include <utility>

#include "restir/restir.h"

#include "aten4idaten.h"
#include "kernel/accelerator.cuh"
#include "kernel/context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

__global__ void shade(
    idaten::TileDomain tileDomain,
    idaten::Reservoir* reservoirs,
    idaten::ReSTIRIntermedidate* intermediates,
    idaten::ReSTIRPathTracing::NormalMaterialStorage* nml_mtrl_buf,
    float4* aovNormalDepth,
    float4* aovTexclrMeshid,
    aten::mat4 mtxW2C,
    int width, int height,
    idaten::Path* paths,
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
    idaten::ShadowRay* shadowRays)
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
        ctxt.lightnum = lightnum;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.vtxPos = vtxPos;
        ctxt.vtxNml = vtxNml;
        ctxt.matrices = matrices;
        ctxt.textures = textures;
    }

    idx = hitindices[idx];

    __shared__ idaten::ShadowRay shShadowRays[64];
    __shared__ aten::MaterialParameter shMtrls[64];
    __shared__ idaten::ReSTIRIntermedidate shIntermediates[64];

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

        // This kernel doesn't have to be called except first bounce.
        // And, in that case, hit material should not be voxel.
        // Therefore, we can ignore voxel check at all.
#if 0
        if (rec.isVoxel) {
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

    auto albedo = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

    // Apply normal map.
    int normalMap = shMtrls[threadIdx.x].normalMap;
    if (shMtrls[threadIdx.x].type == aten::MaterialType::Layer) {
        // 最表層の NormalMap を適用.
        auto* topmtrl = &ctxt.mtrls[shMtrls[threadIdx.x].layer[0]];
        normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
    }
    AT_NAME::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);

    if (!shMtrls[threadIdx.x].attrib.isTranslucent
        && !shMtrls[threadIdx.x].attrib.isEmissive
        && isBackfacing)
    {
        orienting_normal = -orienting_normal;
    }

    shShadowRays[threadIdx.x].isActive = false;

    shIntermediates[threadIdx.x].clear();

    reservoirs[idx].light_idx = -1;

    if (bounce == 0) {
        // Store AOV.
        int ix = idx % tileDomain.w;
        int iy = idx / tileDomain.w;

        ix += tileDomain.x;
        iy += tileDomain.y;

        const auto _idx = getIdx(ix, iy, width);

        // World coordinate to Clip coordinate.
        aten::vec4 pos = aten::vec4(rec.p, 1);
        pos = mtxW2C.apply(pos);

        aovNormalDepth[_idx] = make_float4(orienting_normal.x, orienting_normal.y, orienting_normal.z, pos.w);
        aovTexclrMeshid[_idx] = make_float4(albedo.x, albedo.y, albedo.z, isect.mtrlid);

        nml_mtrl_buf[idx].normal = orienting_normal;
        nml_mtrl_buf[idx].mtrl_idx = rec.mtrlid;
        nml_mtrl_buf[idx].is_voxel = rec.isVoxel;
        nml_mtrl_buf[idx].is_mtrl_valid = (rec.mtrlid >= 0);
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

            auto contrib = paths->throughput[idx].throughput * weight * static_cast<aten::vec3>(shMtrls[threadIdx.x].baseColor);
            paths->contrib[idx].contrib += make_float3(contrib.x, contrib.y, contrib.z);
        }

        // When ray hit the light, tracing will finish.
        paths->attrib[idx].isTerminate = true;
        return;
    }

    ComputeBrdfFunctor compute_brdf_functor(
        ctxt, shMtrls[threadIdx.x], orienting_normal, ray.dir, rec.u, rec.v, albedo);

    // Explicit conection to light.
    if (!(shMtrls[threadIdx.x].attrib.isSingular || shMtrls[threadIdx.x].attrib.isTranslucent))
    {
        aten::LightSampleResult sampleres;
        aten::LightParameter light;

        auto lightidx = sampleLightWithReservoirRIP(
            &sampleres,
            reservoirs[idx],
            &light,
            compute_brdf_functor,
            &ctxt,
            rec.p, orienting_normal,
            &paths->sampler[idx],
            bounce);

        if (lightidx >= 0) {
            const auto& posLight = sampleres.pos;
            const auto& nmlLight = sampleres.nml;
            real pdfLight = sampleres.pdf;

            auto dirToLight = normalize(sampleres.dir);
            auto distToLight = length(posLight - rec.p);

            shShadowRays[threadIdx.x].rayorg = rec.p;
            shShadowRays[threadIdx.x].raydir = dirToLight;
            shShadowRays[threadIdx.x].targetLightId = lightidx;
            shShadowRays[threadIdx.x].distToLight = distToLight;
            shShadowRays[threadIdx.x].lightcontrib = aten::vec3(0);
            shShadowRays[threadIdx.x].isActive = true;

            shIntermediates[threadIdx.x].light_sample_nml = nmlLight;
            shIntermediates[threadIdx.x].light_color = sampleres.finalColor;
            shIntermediates[threadIdx.x].wi = ray.dir;
            shIntermediates[threadIdx.x].mtrl_idx = rec.mtrlid;
            shIntermediates[threadIdx.x].is_voxel = rec.isVoxel;
            shIntermediates[threadIdx.x].is_mtrl_valid = (rec.mtrlid >= 0);
            shIntermediates[threadIdx.x].throughput = paths->throughput[idx].throughput;
            shIntermediates[threadIdx.x].setNml(orienting_normal);
        }
    }

    shadowRays[idx] = shShadowRays[threadIdx.x];
    intermediates[idx] = shIntermediates[threadIdx.x];

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

    // Get normal to add ray offset.
    // In refraction material case, new ray direction might be computed with inverted normal.
    // For example, when a ray go into the refraction surface, inverted normal is used to compute new ray direction.
    auto rayBasedNormal = (!isBackfacing && shMtrls[threadIdx.x].attrib.isTranslucent)
        ? -orienting_normal
        : orienting_normal;

    real c = 1;
    if (!shMtrls[threadIdx.x].attrib.isSingular) {
        // TODO
        // AMDのはabsしているが....
        //c = aten::abs(dot(orienting_normal, nextDir));
        c = dot(rayBasedNormal, nextDir);
    }

    if (pdfb > 0 && c > 0) {
        paths->throughput[idx].throughput *= bsdf * c / pdfb;
        paths->throughput[idx].throughput /= russianProb;
    }
    else {
        paths->attrib[idx].isTerminate = true;
    }

    // Make next ray.
    rays[idx] = aten::ray(rec.p, nextDir, rayBasedNormal);

    paths->throughput[idx].pdfb = pdfb;
    paths->attrib[idx].isSingular = shMtrls[threadIdx.x].attrib.isSingular;
    paths->attrib[idx].mtrlType = shMtrls[threadIdx.x].type;
}

__global__ void hitShadowRay(
    int bounce,
    idaten::Path* paths,
    int* hitindices,
    int* hitnum,
    idaten::Reservoir* reservoirs,
    const idaten::ShadowRay* __restrict__ shadowRays,
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

    idaten::Context ctxt;
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

    const auto& shadowRay = shadowRays[idx];

    if (!shadowRay.isActive) {
        return;
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
        reservoirs[idx].w = 0.0f;
        reservoirs[idx].m = 0;
        reservoirs[idx].light_idx = -1;
        reservoirs[idx].light_pdf = 0.0f;
    }
    else {
        reservoirs[idx].light_idx = targetLightId;
    }
}

__global__ void computeShadowRayContribution(
    const idaten::Reservoir* __restrict__ reservoirs,
    const idaten::ReSTIRIntermedidate* __restrict__ intermediates,
    idaten::Path* paths,
    int* hitindices,
    int* hitnum,
    const float4* __restrict__ aovNormalDepth,
    const float4* __restrict__ aovTexclrMeshid,
    const aten::LightParameter* __restrict__ lights, int lightnum,
    const aten::MaterialParameter* __restrict__ mtrls,
    cudaTextureObject_t* textures,
    const idaten::ShadowRay* __restrict__ shadowRays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= *hitnum) {
        return;
    }

    idx = hitindices[idx];

    if (lightnum <= 0) {
        return;
    }

    idaten::Context ctxt;
    {
        ctxt.mtrls = mtrls;
        ctxt.textures = textures;
    }

    __shared__ aten::MaterialParameter shMtrls[64];

    const auto& reservoir = reservoirs[idx];
    const auto& intermediate = intermediates[idx];

    if (intermediate.is_mtrl_valid) {
        shMtrls[threadIdx.x] = ctxt.mtrls[intermediate.mtrl_idx];

        if (intermediate.is_voxel) {
            // Replace to lambert.
            const auto& albedo = ctxt.mtrls[intermediate.mtrl_idx].baseColor;
            shMtrls[threadIdx.x] = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
            shMtrls[threadIdx.x].baseColor = albedo;
        }

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

    if (!(shMtrls[threadIdx.x].attrib.isSingular || shMtrls[threadIdx.x].attrib.isTranslucent))
    {
        auto albedo_meshid = aovTexclrMeshid[idx];

        const aten::vec3 orienting_normal(
            intermediates[idx].nml_x,
            intermediates[idx].nml_y,
            intermediates[idx].nml_z);
        const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

        int lightidx = reservoir.light_idx;

        if (lightidx >= 0) {
            const auto& light = lights[lightidx];

            auto pdfLight = reservoir.light_pdf;

            auto nmlLight = intermediate.light_sample_nml;
            auto dirToLight = shadowRays[idx].raydir;
            auto distToLight = shadowRays[idx].distToLight;

            aten::vec3 lightcontrib;
            {
                auto cosShadow = dot(orienting_normal, dirToLight);

                // TODO
                // u,v は samplePDF/sampleBSDF 内部では利用されていない
                float u = 0.0f;
                float v = 0.0f;

                real pdfb = samplePDF(&ctxt, &shMtrls[threadIdx.x], orienting_normal, intermediate.wi, dirToLight, u, v);
                auto bsdf = sampleBSDF(&ctxt, &shMtrls[threadIdx.x], orienting_normal, intermediate.wi, dirToLight, u, v, albedo);

                bsdf *= intermediate.throughput;

                // Get light color.
                auto emit = intermediate.light_color;

                if (light.attrib.isSingular || light.attrib.isInfinite) {
                    if (pdfLight > real(0) && cosShadow >= 0) {
                        // TODO
                        // ジオメトリタームの扱いについて.
                        // singular light の場合は、finalColor に距離の除算が含まれている.
                        // inifinite light の場合は、無限遠方になり、pdfLightに含まれる距離成分と打ち消しあう？.
                        // （打ち消しあうので、pdfLightには距離成分は含んでいない）.
                        auto misW = pdfLight / (pdfb + pdfLight);

                        lightcontrib = misW * bsdf * emit * cosShadow / pdfLight;
                    }
                }
                else {
                    auto cosLight = dot(nmlLight, -dirToLight);

                    if (cosShadow >= 0 && cosLight >= 0) {
                        auto dist2 = distToLight * distToLight;
                        auto G = cosShadow * cosLight / dist2;

                        if (pdfb > real(0) && pdfLight > real(0)) {
                            // Convert pdf from steradian to area.
                            // http://www.slideshare.net/h013/edubpt-v100
                            // p31 - p35
                            pdfb = pdfb * cosLight / dist2;

                            auto misW = pdfLight / (pdfb + pdfLight);

                            lightcontrib = misW * (bsdf * emit * G) / pdfLight;
                        }
                    }
                }
            }

            paths->contrib[idx].contrib += make_float3(lightcontrib.x, lightcontrib.y, lightcontrib.z);
        }
    }
}

namespace idaten
{
    void ReSTIRPathTracing::onShadeReSTIR(
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

        int curBufNmlMtrlPos = getCurBufNmlMtrlPos();

        shade << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_tileDomain,
            m_reservoirs[0].ptr(),
            m_intermediates[0].ptr(),
            m_bufNmlMtrl[curBufNmlMtrlPos].ptr(),
            m_aovNormalDepth.ptr(),
            m_aovTexclrMeshid.ptr(),
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
            m_shadowRays.ptr());

        checkCudaKernel(shade);

        onShadeByShadowRayReSTIR(
            width, height,
            bounce, texVtxPos);
    }

    void ReSTIRPathTracing::onShadeByShadowRayReSTIR(
        int width, int height,
        int bounce,
        cudaTextureObject_t texVtxPos)
    {
        dim3 blockPerGrid(((m_tileDomain.w * m_tileDomain.h) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        hitShadowRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            bounce,
            m_paths.ptr(),
            m_hitidx.ptr(), hitcount.ptr(),
            m_reservoirs[0].ptr(),
            m_shadowRays.ptr(),
            m_shapeparam.ptr(), m_shapeparam.num(),
            m_mtrlparam.ptr(),
            m_lightparam.ptr(), m_lightparam.num(),
            m_nodetex.ptr(),
            m_primparams.ptr(),
            texVtxPos,
            m_mtxparams.ptr());

        checkCudaKernel(hitShadowRay);

        const auto target_idx = computelReuse(width, height, bounce);
        const auto reservior_idx = std::get<0>(target_idx);
        const auto intermediate_idx = std::get<1>(target_idx);

        computeShadowRayContribution << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_reservoirs[reservior_idx].ptr(),
            m_intermediates[intermediate_idx].ptr(),
            m_paths.ptr(),
            m_hitidx.ptr(), hitcount.ptr(),
            m_aovNormalDepth.ptr(),
            m_aovTexclrMeshid.ptr(),
            m_lightparam.ptr(), m_lightparam.num(),
            m_mtrlparam.ptr(),
            m_tex.ptr(),
            m_shadowRays.ptr());

        checkCudaKernel(computeShadowRayContribution);
    }
}
