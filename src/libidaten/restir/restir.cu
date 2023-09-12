#include <utility>

#include "restir/restir.h"
#include "restir/restir_sample_light.cuh"

#include "aten4idaten.h"
#include "kernel/accelerator.cuh"
#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "renderer/pathtracing/pathtracing_impl.h"

__global__ void initReSTIRParameters(
    int32_t width, int32_t height,
    idaten::Reservoir* reservoirs,
    idaten::ReSTIRInfo* restir_infos)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    const auto idx = getIdx(ix, iy, width);

    reservoirs[idx].clear();
    restir_infos[idx].clear();
}

__global__ void shade(
    idaten::Reservoir* reservoirs,
    idaten::ReSTIRInfo* restir_infos,
    float4* aovNormalDepth,
    float4* aovTexclrMeshid,
    aten::mat4 mtxW2C,
    int32_t width, int32_t height,
    idaten::Path paths,
    const int32_t* __restrict__ hitindices,
    int32_t* hitnum,
    const aten::Intersection* __restrict__ isects,
    aten::ray* rays,
    int32_t sample,
    int32_t frame,
    int32_t bounce, int32_t rrBounce,
    idaten::context ctxt,
    uint32_t* random,
    idaten::ShadowRay* shadowRays)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= *hitnum) {
        return;
    }

    idx = hitindices[idx];

    __shared__ aten::MaterialParameter shMtrls[64];

    const auto ray = rays[idx];

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
    auto scramble = random[idx] * 0x1fe3434f;
    paths.sampler[idx].init(frame + sample, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
    auto rnd = random[idx];
    auto scramble = rnd * 0x1fe3434f
        * (((frame + sample) + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
    paths.sampler[idx].init(
        (frame + sample) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
        4 + bounce * 300,
        scramble);
#endif

    aten::hitrecord rec;

    const auto& isect = isects[idx];

    const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));
    AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

    bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

    // 交差位置の法線.
    // 物体からのレイの入出を考慮.
    aten::vec3 orienting_normal = rec.normal;

    AT_NAME::FillMaterial(
        shMtrls[threadIdx.x],
        ctxt,
        rec.mtrlid,
        rec.isVoxel);

    auto albedo = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

    // Apply normal map.
    int32_t normalMap = shMtrls[threadIdx.x].normalMap;
    const auto pre_sampled_r = AT_NAME::material::applyNormal(
        &shMtrls[threadIdx.x],
        normalMap,
        orienting_normal, orienting_normal,
        rec.u, rec.v,
        ray.dir,
        &paths.sampler[idx]);

    if (!shMtrls[threadIdx.x].attrib.isTranslucent
        && !shMtrls[threadIdx.x].attrib.isEmissive
        && isBackfacing)
    {
        orienting_normal = -orienting_normal;
    }

    shadowRays[idx].isActive = false;

    auto& restir_info = restir_infos[idx];
    {
        restir_info.clear();
        restir_info.nml = orienting_normal;
        restir_info.is_voxel = rec.isVoxel;
        restir_info.mtrl_idx = rec.mtrlid;
        restir_info.throughput = paths.throughput[idx].throughput;
        restir_info.wi = ray.dir;
        restir_info.u = rec.u;
        restir_info.v = rec.v;
        restir_info.p = rec.p;
        restir_info.pre_sampled_r = pre_sampled_r;
    }

    if (bounce == 0) {
        // Store AOV.
        int32_t ix = idx % width;
        int32_t iy = idx / width;

        const auto _idx = getIdx(ix, iy, width);

        // World coordinate to Clip coordinate.
        aten::vec4 pos = aten::vec4(rec.p, 1);
        pos = mtxW2C.apply(pos);

        aovNormalDepth[_idx] = make_float4(orienting_normal.x, orienting_normal.y, orienting_normal.z, pos.w);
        aovTexclrMeshid[_idx] = make_float4(albedo.x, albedo.y, albedo.z, isect.mtrlid);
    }

    // Implicit conection to light.
    auto is_hit_implicit_light = AT_NAME::HitImplicitLight(
        isBackfacing,
        bounce,
        paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
        ray,
        rec.p, orienting_normal,
        rec.area,
        shMtrls[threadIdx.x]);
    if (is_hit_implicit_light) {
        return;
    }

    // Explicit conection to light.
    if (!(shMtrls[threadIdx.x].attrib.isSingular || shMtrls[threadIdx.x].attrib.isTranslucent))
    {
        auto& reservoir = reservoirs[idx];

        auto lightidx = sampleLightWithReservoirRIP(
            reservoir,
            shMtrls[threadIdx.x],
            &ctxt,
            rec.p, orienting_normal,
            ray.dir,
            rec.u, rec.v, albedo,
            &paths.sampler[idx],
            bounce);

        if (lightidx >= 0) {
            const auto& light = ctxt.lights[lightidx];

            const auto& posLight = reservoir.light_sample_.pos;
            const auto& nmlLight = reservoir.light_sample_.nml;

            auto lightSelectPdf = reservoir.pdf_;

            auto dirToLight = normalize(reservoir.light_sample_.dir);
            auto distToLight = length(posLight - rec.p);

            aten::Intersection isectTmp;

            auto shadowRayOrg = rec.p + AT_MATH_EPSILON * orienting_normal;
            auto tmp = rec.p + dirToLight - shadowRayOrg;
            auto shadowRayDir = normalize(tmp);

            bool isShadowRayActive = false;

            shadowRays[idx].rayorg = shadowRayOrg;
            shadowRays[idx].raydir = shadowRayDir;
            shadowRays[idx].targetLightId = lightidx;
            shadowRays[idx].distToLight = distToLight;
            shadowRays[idx].lightcontrib = aten::vec3(0);
            {
                auto cosShadow = dot(orienting_normal, dirToLight);
                cosShadow = aten::abs(cosShadow);

                if (light.attrib.isInfinite || light.attrib.isSingular) {
                    if (cosShadow >= 0) {
                        isShadowRayActive = true;
                    }
                }
                else {
                    auto cosLight = dot(nmlLight, -dirToLight);

                    if (cosShadow >= 0 && cosLight >= 0) {
                        isShadowRayActive = true;
                    }
                }

                shadowRays[idx].isActive = isShadowRayActive;
            }
        }
    }

    const auto russianProb = AT_NAME::ComputeRussianProbability(
        bounce, rrBounce,
        paths.attrib[idx],
        paths.throughput[idx],
        paths.sampler[idx]);

    AT_NAME::MaterialSampling sampling;

    AT_NAME::material::sampleMaterialWithExternalAlbedo(
        &sampling,
        &shMtrls[threadIdx.x],
        orienting_normal,
        ray.dir,
        rec.normal,
        &paths.sampler[idx],
        pre_sampled_r,
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

    auto c = dot(orienting_normal, nextDir);

    if (pdfb > 0 && c > 0) {
        paths.throughput[idx].throughput *= bsdf * c / pdfb;
        paths.throughput[idx].throughput /= russianProb;
    }
    else {
        paths.attrib[idx].isTerminate = true;
        return;
    }

    // Make next ray.
    rays[idx] = aten::ray(rec.p, nextDir, rayBasedNormal);

    paths.throughput[idx].pdfb = pdfb;
    paths.attrib[idx].isSingular = shMtrls[threadIdx.x].attrib.isSingular;
    paths.attrib[idx].mtrlType = shMtrls[threadIdx.x].type;
}

__global__ void hitShadowRay(
    int32_t bounce,
    idaten::Path paths,
    int32_t* hitindices,
    int32_t* hitnum,
    idaten::Reservoir* reservoirs,
    const idaten::ShadowRay* __restrict__ shadowRays,
    idaten::context ctxt)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= *hitnum) {
        return;
    }

    idx = hitindices[idx];

    auto isHit = AT_NAME::HitShadowRay(idx, bounce, ctxt, paths, shadowRays);

    if (!isHit) {
        reservoirs[idx].w_sum_ = 0.0f;
        reservoirs[idx].pdf_ = 0.0f;
        reservoirs[idx].target_density_ = 0.0f;
        reservoirs[idx].light_idx_ = -1;
    }
}

__global__ void computeShadowRayContribution(
    const idaten::Reservoir* __restrict__ reservoirs,
    const idaten::ReSTIRInfo* __restrict__ restir_infos,
    idaten::Path paths,
    int32_t* hitindices,
    int32_t* hitnum,
    const float4* __restrict__ aovNormalDepth,
    const float4* __restrict__ aovTexclrMeshid,
    const aten::LightParameter* __restrict__ lights, int32_t lightnum,
    const aten::MaterialParameter* __restrict__ mtrls,
    cudaTextureObject_t* textures,
    const idaten::ShadowRay* __restrict__ shadowRays)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= *hitnum) {
        return;
    }

    idx = hitindices[idx];

    if (paths.attrib[idx].isTerminate) {
        return;
    }

    if (lightnum <= 0) {
        return;
    }

    idaten::context ctxt;
    {
        ctxt.mtrls = mtrls;
        ctxt.textures = textures;
    }

    __shared__ aten::MaterialParameter shMtrls[64];

    const auto& reservoir = reservoirs[idx];
    const auto& restir_info = restir_infos[idx];

    AT_NAME::FillMaterial(
        shMtrls[threadIdx.x],
        ctxt,
        restir_info.mtrl_idx,
        restir_info.is_voxel);

    if (!(shMtrls[threadIdx.x].attrib.isSingular || shMtrls[threadIdx.x].attrib.isTranslucent))
    {
        if (reservoir.isValid()) {
            const auto& orienting_normal = restir_info.nml;

            const auto& albedo_meshid = aovTexclrMeshid[idx];
            const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

            const auto& light = lights[reservoir.light_idx_];

            const auto& nmlLight = reservoir.light_sample_.nml;
            const auto& dirToLight = shadowRays[idx].raydir;
            const auto& distToLight = shadowRays[idx].distToLight;

            aten::vec3 lightcontrib;
            {
                auto cosShadow = dot(orienting_normal, dirToLight);

                // TODO
                // 計算済みのalbedoを与えているため
                // u,v は samplePDF/sampleBSDF 内部では利用されていない
                float u = 0.0f;
                float v = 0.0f;

                auto bsdf = AT_NAME::material::sampleBSDFWithExternalAlbedo(
                    &shMtrls[threadIdx.x],
                    orienting_normal,
                    restir_info.wi,
                    dirToLight,
                    u, v,
                    albedo,
                    restir_info.pre_sampled_r);

                bsdf *= restir_info.throughput;

                // Get light color.
                auto emit = reservoir.light_sample_.finalColor;

                cosShadow = aten::abs(cosShadow);

                if (light.attrib.isInfinite || light.attrib.isSingular) {
                    if (cosShadow >= 0) {
                        lightcontrib = bsdf * emit * cosShadow * reservoir.pdf_;
                    }
                }
                else {
                    auto cosLight = dot(nmlLight, -dirToLight);

                    if (cosShadow >= 0 && cosLight >= 0) {
                        auto dist2 = distToLight * distToLight;
                        auto G = cosShadow * cosLight / dist2;

                        lightcontrib = (bsdf * emit * G) * reservoir.pdf_;
                    }
                }
            }

            paths.contrib[idx].contrib += make_float3(lightcontrib.x, lightcontrib.y, lightcontrib.z);
        }
    }
}

namespace idaten
{
    void ReSTIRPathTracing::initReSTIR(int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        initReSTIRParameters << < grid, block, 0, m_stream >> > (
            width, height,
            m_reservoirs[m_curReservoirPos].data(),
            m_restir_infos.data());

        checkCudaKernel(initReSTIRParameters);
    }

    void ReSTIRPathTracing::onShadeReSTIR(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t sample,
        int32_t bounce, int32_t rrBounce)
    {
        m_mtxW2V.lookat(
            m_cam.origin,
            m_cam.center,
            m_cam.up);

        m_mtxV2C.perspective(
            m_cam.znear,
            m_cam.zfar,
            m_cam.vfov,
            m_cam.aspect);

        m_mtxC2V = m_mtxV2C;
        m_mtxC2V.invert();

        m_mtxV2W = m_mtxW2V;
        m_mtxV2W.invert();

        aten::mat4 mtxW2C = m_mtxV2C * m_mtxW2V;

        dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        shade << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_reservoirs[m_curReservoirPos].data(),
            m_restir_infos.data(),
            aov_.normal_depth().data(),
            aov_.albedo_meshid().data(),
            mtxW2C,
            width, height,
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_isects.data(),
            m_rays.data(),
            sample,
            m_frame,
            bounce, rrBounce,
            ctxt_host_.ctxt,
            m_random.data(),
            m_shadowRays.data());

        checkCudaKernel(shade);

        onShadeByShadowRayReSTIR(
            width, height,
            bounce);
    }

    void ReSTIRPathTracing::onShadeByShadowRayReSTIR(
        int32_t width, int32_t height,
        int32_t bounce)
    {
        dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        hitShadowRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            bounce,
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_reservoirs[m_curReservoirPos].data(),
            m_shadowRays.data(),
            ctxt_host_.ctxt);

        checkCudaKernel(hitShadowRay);

        const auto target_idx = computelReuse(
            width, height,
            bounce);

        computeShadowRayContribution << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_reservoirs[target_idx].data(),
            m_restir_infos.data(),
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            aov_.normal_depth().data(),
            aov_.albedo_meshid().data(),
            ctxt_host_.lightparam.data(), ctxt_host_.lightparam.num(),
            ctxt_host_.mtrlparam.data(),
            ctxt_host_.tex.data(),
            m_shadowRays.data());

        checkCudaKernel(computeShadowRayContribution);
    }
}
