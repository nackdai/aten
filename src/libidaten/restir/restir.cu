#include <utility>

#include "restir/restir.h"

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
#include "renderer/restir/restir_impl.h"

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
    aten::mat4 mtx_W2C,
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
    const aten::ObjectParameter* __restrict__ shapes,
    const aten::MaterialParameter* __restrict__ mtrls,
    const aten::LightParameter* __restrict__ lights,
    const aten::TriangleParameter* __restrict__ prims,
    const aten::mat4* __restrict__ matrices,
    uint32_t* random)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= *hitnum) {
        return;
    }

    idx = hitindices[idx];

    ctxt.shapes = shapes;
    ctxt.mtrls = mtrls;
    ctxt.lights = lights;
    ctxt.prims = prims;
    ctxt.matrices = matrices;

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
        pos = mtx_W2C.apply(pos);

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

    // Generate initial candidates.
    if (!(shMtrls[threadIdx.x].attrib.isSingular || shMtrls[threadIdx.x].attrib.isTranslucent))
    {
        auto& reservoir = reservoirs[idx];

        AT_NAME::restir::SampleLightByStreamingRIS(
            reservoir,
            shMtrls[threadIdx.x],
            ctxt,
            rec.p, orienting_normal,
            ray.dir,
            rec.u, rec.v,
            &paths.sampler[idx],
            bounce);
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

__global__ void EvaluateVisibility(
    int32_t bounce,
    int32_t width, int32_t height,
    idaten::Path paths,
    int32_t* hitindices,
    int32_t* hitnum,
    idaten::Reservoir* reservoirs,
    const idaten::ReSTIRInfo* __restrict__ restir_infos,
    idaten::ShadowRay* shadowRays,
    idaten::context ctxt,
    const aten::ObjectParameter* __restrict__ shapes,
    const aten::MaterialParameter* __restrict__ mtrls,
    const aten::LightParameter* __restrict__ lights,
    const aten::TriangleParameter* __restrict__ prims,
    const aten::mat4* __restrict__ matrices)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= *hitnum) {
        return;
    }

    ctxt.shapes = shapes;
    ctxt.mtrls = mtrls;
    ctxt.lights = lights;
    ctxt.prims = prims;
    ctxt.matrices = matrices;

    idx = hitindices[idx];

    const auto size = width * height;

    aten::span reservoirs_as_span(reservoirs, size);
    aten::const_span resitr_infos_as_span(restir_infos, size);
    aten::span shadow_rays(shadowRays, size);

    AT_NAME::restir::EvaluateVisibility(
        idx,
        bounce,
        paths,
        ctxt,
        reservoirs_as_span,
        resitr_infos_as_span,
        shadow_rays);
}

__global__ void ComputePixelColor(
    const idaten::Reservoir* __restrict__ reservoirs,
    const idaten::ReSTIRInfo* __restrict__ restir_infos,
    idaten::Path paths,
    int32_t* hitindices,
    int32_t* hitnum,
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

    aten::const_span lights_as_span(lights, lightnum);

    auto contrib = AT_NAME::restir::ComputeContribution(
        reservoir, restir_info,
        shMtrls[threadIdx.x],
        aovTexclrMeshid[idx],
        shadowRays[idx],
        lights_as_span);
    if (contrib) {
        const auto pixel_color = contrib.value() * paths.throughput[idx].throughput;
        paths.contrib[idx].contrib += make_float3(pixel_color.x, pixel_color.y, pixel_color.z);
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
        m_mtx_W2V.lookat(
            m_cam.origin,
            m_cam.center,
            m_cam.up);

        m_mtx_V2C.perspective(
            m_cam.znear,
            m_cam.zfar,
            m_cam.vfov,
            m_cam.aspect);

        m_mtx_C2V = m_mtx_V2C;
        m_mtx_C2V.invert();

        m_mtx_V2W = m_mtx_W2V;
        m_mtx_V2W.invert();

        aten::mat4 mtx_W2C = m_mtx_V2C * m_mtx_W2V;

        dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        shade << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_reservoirs[m_curReservoirPos].data(),
            m_restir_infos.data(),
            aov_.normal_depth().data(),
            aov_.albedo_meshid().data(),
            mtx_W2C,
            width, height,
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_isects.data(),
            m_rays.data(),
            sample,
            m_frame,
            bounce, rrBounce,
            ctxt_host_.ctxt,
            ctxt_host_.shapeparam.data(),
            ctxt_host_.mtrlparam.data(),
            ctxt_host_.lightparam.data(),
            ctxt_host_.primparams.data(),
            ctxt_host_.mtxparams.data(),
            m_random.data());

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

        EvaluateVisibility << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            bounce,
            width, height,
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_reservoirs[m_curReservoirPos].data(),
            m_restir_infos.data(),
            m_shadowRays.data(),
            ctxt_host_.ctxt,
            ctxt_host_.shapeparam.data(),
            ctxt_host_.mtrlparam.data(),
            ctxt_host_.lightparam.data(),
            ctxt_host_.primparams.data(),
            ctxt_host_.mtxparams.data());

        checkCudaKernel(EvaluateVisibility);

        const auto target_idx = computelReuse(
            width, height,
            bounce);

        ComputePixelColor << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_reservoirs[target_idx].data(),
            m_restir_infos.data(),
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            aov_.albedo_meshid().data(),
            ctxt_host_.lightparam.data(), ctxt_host_.lightparam.num(),
            ctxt_host_.mtrlparam.data(),
            ctxt_host_.tex.data(),
            m_shadowRays.data());

        checkCudaKernel(ComputePixelColor);
    }
}
