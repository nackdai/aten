#include "kernel/pathtracing.h"

#include "aten4idaten.h"
#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_common.h"
#include "kernel/StreamCompaction.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "renderer/pathtracing/pathtracing_impl.h"

namespace pt {
    __global__ void shade(
        float4* aovNormalDepth,
        float4* aovAlbedoMeshId,
        int32_t width, int32_t height,
        idaten::context ctxt,
        const aten::ObjectParameter* __restrict__ shapes,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights,
        const aten::TriangleParameter* __restrict__ prims,
        const aten::mat4* __restrict__ matrices,
        idaten::Path paths,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        const aten::Intersection* __restrict__ isects,
        aten::ray* rays,
        int32_t sample,
        int32_t frame,
        int32_t bounce, int32_t rrBounce,
        uint32_t* random,
        idaten::ShadowRay* shadowRays)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        if (paths.attrib[idx].is_terminated) {
            return;
        }

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        __shared__ idaten::ShadowRay shShadowRays[64];
        __shared__ aten::MaterialParameter shMtrls[64];

        const auto& isect = isects[idx];

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

        const auto ray = rays[idx];
        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

        aten::hitrecord rec;
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

        auto albedo = AT_NAME::sampleTexture(ctxt, shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, shMtrls[threadIdx.x].baseColor, bounce);

        // Apply normal map.
        int32_t normalMap = shMtrls[threadIdx.x].normalMap;
        auto pre_sampled_r = AT_NAME::material::applyNormal(
            ctxt,
            &shMtrls[threadIdx.x],
            normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir,
            &paths.sampler[idx]);

        if (bounce == 0) {
            // Store AOV.
            AT_NAME::FillBasicAOVs(
                aovNormalDepth[idx], orienting_normal, rec, aten::mat4(),
                aovAlbedoMeshId[idx], albedo, isect);
        }

        // Check stencil.
        auto is_stencil = AT_NAME::CheckStencil(
            rays[idx], paths.attrib[idx],
            bounce,
            ctxt,
            rec.p, orienting_normal,
            shMtrls[threadIdx.x]
        );
        if (is_stencil) {
            return;
        }

        // Check transparency or translucency.
        auto is_translucent_by_alpha = AT_NAME::CheckMaterialTranslucentByAlpha(
            ctxt,
            shMtrls[threadIdx.x],
            rec.u, rec.v, rec.p,
            orienting_normal,
            rays[idx],
            paths.sampler[idx],
            paths.attrib[idx],
            paths.throughput[idx]);
        if (is_translucent_by_alpha) {
            // TODO:
            // Basically, shadow ray is treated via shared memory.
            // Therefore, finally it has to be re-stored to original global memory.
            // But, in this case, it doesn't happen yet, so, set value directly here.
            shadowRays[idx].isActive = false;
            return;
        }

        albedo = paths.throughput[idx].transmission * albedo + paths.throughput[idx].alpha_blend_radiance_on_the_way;

        // Implicit conection to light.
        auto is_hit_implicit_light = AT_NAME::HitTeminatedMaterial(
            ctxt, paths.sampler[idx],
            isect.objid,
            isBackfacing,
            bounce,
            paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
            ray,
            rec,
            albedo, shMtrls[threadIdx.x]);
        if (is_hit_implicit_light) {
            return;
        }

        if (!shMtrls[threadIdx.x].attrib.is_translucent && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        shShadowRays[threadIdx.x].isActive = false;

        // Explicit conection to light.
        AT_NAME::FillShadowRay(
            idx,
            shShadowRays[threadIdx.x],
            ctxt,
            bounce,
            paths,
            shMtrls[threadIdx.x],
            ray,
            rec.p, orienting_normal,
            rec.u, rec.v, albedo,
            pre_sampled_r);

        shadowRays[idx] = shShadowRays[threadIdx.x];

        const auto russianProb = AT_NAME::ComputeRussianProbability(
            bounce, rrBounce,
            paths.attrib[idx],
            paths.throughput[idx],
            paths.sampler[idx]);

        AT_NAME::MaterialSampling sampling;

        AT_NAME::material::sampleMaterial(
            &sampling,
            ctxt,
            paths.throughput[idx].throughput,
            &shMtrls[threadIdx.x],
            orienting_normal,
            ray.dir,
            rec.normal,
            &paths.sampler[idx], pre_sampled_r,
            rec.u, rec.v);

        AT_NAME::PrepareForNextBounce(
            idx,
            rec, isBackfacing, russianProb,
            orienting_normal,
            shMtrls[threadIdx.x], sampling,
            albedo,
            paths,
            rays);
    }

    __global__ void hitShadowRay(
        int32_t bounce,
        idaten::context ctxt,
        const aten::ObjectParameter* __restrict__ shapes,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights,
        const aten::TriangleParameter* __restrict__ prims,
        const aten::mat4* __restrict__ matrices,
        const aten::Intersection* __restrict__ isects,
        idaten::Path paths,
        int32_t* hitindices,
        int32_t* hitnum,
        const idaten::ShadowRay* __restrict__ shadowRays)
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

        const auto& isect = isects[idx];
        const auto& mtrl = ctxt.GetMaterial(isect.mtrlid);

        AT_NAME::HitShadowRay(
            idx, bounce,
            ctxt, mtrl,
            paths, shadowRays[idx]);
    }

    __global__ void gather(
        cudaSurfaceObject_t dst,
        const idaten::Path paths,
        bool enableProgressive,
        int32_t width, int32_t height)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        auto idx = getIdx(ix, iy, width);

        const auto& c = paths.contrib[idx].contrib;
        const auto sample = paths.contrib[idx].samples;

        float4 contrib = make_float4(c.x, c.y, c.z, 0.0F);

        if (enableProgressive) {
            float4 data;
            surf2Dread(&data, dst, ix * sizeof(float4), iy);

            // First data.w value is 0.
            int32_t n = data.w;
            contrib = n * data + contrib / sample;
            contrib /= (n + 1);
            contrib.w = n + 1;
        }
        else {
            contrib /= sample;
            contrib.w = 1;
        }

        if (dst) {
            surf2Dwrite(
                contrib,
                dst,
                ix * sizeof(float4), iy,
                cudaBoundaryModeTrap);
        }
    }

    __global__ void DisplayAOV(
        cudaSurfaceObject_t dst,
        int32_t width, int32_t height,
        const float4* __restrict__ aov_normal_depth,
        const float4* __restrict__ aov_albedo_meshid)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        auto idx = getIdx(ix, iy, width);

        float4 normal_depth = aov_normal_depth[idx];

        // (-1, 1) -> (0, 1)
        normal_depth = normal_depth * 0.5F + 0.5F;
        normal_depth.w = 1.0F;

        surf2Dwrite(
            normal_depth,
            dst,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }
}

namespace idaten
{
    void PathTracingImplBase::onHitTest(
        int32_t width, int32_t height,
        int32_t bounce)
    {
        if (bounce == 0
            && CanSSRTHitTest()
            && g_buffer_.IsValid())
        {
            hitTestOnScreenSpace(
                width, height,
                g_buffer_);
        }
        else {
            hitTest(
                width, height,
                bounce);
        }
    }

    void PathTracingImplBase::onShade(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t sample,
        int32_t bounce, int32_t rrBounce, int32_t max_depth)
    {
        dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        pt::shade << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            aov_.normal_depth().data(),
            aov_.albedo_meshid().data(),
            width, height,
            ctxt_host_->ctxt,
            ctxt_host_->shapeparam.data(),
            ctxt_host_->mtrlparam.data(),
            ctxt_host_->lightparam.data(),
            ctxt_host_->primparams.data(),
            ctxt_host_->mtxparams.data(),
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_isects.data(),
            m_rays.data(),
            sample,
            m_frame,
            bounce, rrBounce,
            m_random.data(),
            m_shadowRays.data());

        checkCudaKernel(shade);

        onShadeByShadowRay(width, height, bounce);
    }

    void PathTracingImplBase::onShadeByShadowRay(
        int32_t width, int32_t height,
        int32_t bounce)
    {
        dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        pt::hitShadowRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            bounce,
            ctxt_host_->ctxt,
            ctxt_host_->shapeparam.data(),
            ctxt_host_->mtrlparam.data(),
            ctxt_host_->lightparam.data(),
            ctxt_host_->primparams.data(),
            ctxt_host_->mtxparams.data(),
            m_isects.data(),
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_shadowRays.data());

        checkCudaKernel(hitShadowRay);
    }

    void PathTracingImplBase::onGather(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t maxSamples)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        pt::gather << <grid, block, 0, m_stream >> > (
            outputSurf,
            path_host_->paths,
            m_enableProgressive,
            width, height);

        checkCudaKernel(gather);
    }

    void PathTracingImplBase::DisplayAOV(
        cudaSurfaceObject_t output_surface,
        int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        pt::DisplayAOV << <grid, block, 0, m_stream >> > (
            output_surface,
            width, height,
            aov_.normal_depth().data(),
            aov_.albedo_meshid().data());

        checkCudaKernel(DisplayAOV);
    }
}
