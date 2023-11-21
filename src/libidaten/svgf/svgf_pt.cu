#include "svgf/svgf.h"

#include "kernel/StreamCompaction.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/svgf/svgf_impl.h"

namespace svgf_kernel {
    __global__ void shade(
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
        uint32_t* random,
        idaten::ShadowRay* shadowRays)
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

        __shared__ idaten::ShadowRay shShadowRays[64];
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

        // Render AOVs.
        // NOTE
        // 厳密に法線をAOVに保持するなら、法線マップ適用後するべき.
        // しかし、temporal reprojection、atrousなどのフィルタ適用時に法線を参照する際に、法線マップが細かすぎてはじかれてしまうことがある.
        // それにより、フィルタがおもったようにかからずフィルタの品質が下がってしまう問題が発生する.
        if (bounce == 0) {
            // texture color
            auto texcolor = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec4(1.0f));

            AT_NAME::FillBasicAOVs(
                aovNormalDepth[idx], orienting_normal, rec, mtx_W2C,
                aovTexclrMeshid[idx], texcolor, isect);
            aovTexclrMeshid[idx].w = isect.mtrlid;

            // For exporting separated albedo.
            shMtrls[threadIdx.x].albedoMap = -1;
        }
        // TODO
        // How to deal Refraction?
        else if (bounce == 1 && paths.attrib[idx].mtrlType == aten::MaterialType::Specular) {
            // texture color.
            auto texcolor = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec4(1.0f));

            // TODO
            // No good idea to compute reflected depth.
            AT_NAME::FillBasicAOVs(
                aovNormalDepth[idx], orienting_normal, rec, mtx_W2C,
                aovTexclrMeshid[idx], texcolor, isect);
            aovTexclrMeshid[idx].w = isect.mtrlid;

            // For exporting separated albedo.
            shMtrls[threadIdx.x].albedoMap = -1;
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

        if (!shMtrls[threadIdx.x].attrib.isTranslucent && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        // Apply normal map.
        int32_t normalMap = shMtrls[threadIdx.x].normalMap;
        auto pre_sample_r = AT_NAME::material::applyNormal(
            &shMtrls[threadIdx.x],
            normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir,
            &paths.sampler[idx]);

        auto albedo = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

#if 1
        // Explicit conection to light.
        AT_NAME::FillShadowRay(
            shShadowRays[threadIdx.x],
            ctxt,
            bounce,
            paths.sampler[idx],
            paths.throughput[idx],
            shMtrls[threadIdx.x],
            ray,
            rec.p, orienting_normal,
            rec.u, rec.v, albedo);
#endif

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
            pre_sample_r,
            rec.u, rec.v,
            albedo);

        AT_NAME::PostProcessPathTrancing(
            idx,
            rec, isBackfacing, russianProb,
            orienting_normal,
            shMtrls[threadIdx.x], sampling,
            paths, rays);

        shadowRays[idx] = shShadowRays[threadIdx.x];
    }

    template <bool IsFirstFrameExecution>
    __global__ void gather(
        cudaSurfaceObject_t dst,
        int32_t width, int32_t height,
        const idaten::Path paths,
        float4* temporary_color_buffer,
        float4* aov_color_variance = nullptr,
        float4* aov_moment_temporalweight = nullptr)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        auto idx = getIdx(ix, iy, width);

        const size_t size = static_cast<size_t>(width * height);
        using BufferType = std::remove_pointer_t<decltype(aov_color_variance)>;

        auto contrib = AT_NAME::svgf::PrepareForDenoise<IsFirstFrameExecution>(
            idx,
            paths,
            aten::span<BufferType>(temporary_color_buffer, size),
            aten::span<BufferType>(aov_color_variance, size),
            aten::span<BufferType>(aov_moment_temporalweight, size));

        if (IsFirstFrameExecution) {
            if (dst) {
                contrib.w = 0;
                surf2Dwrite(
                    contrib,
                    dst,
                    ix * sizeof(float4), iy,
                    cudaBoundaryModeTrap);
            }
        }
    }
}

namespace idaten
{
    void SVGFPathTracing::onHitTest(
        int32_t width, int32_t height,
        int32_t bounce)
    {
        if (bounce == 0 && m_canSSRTHitTest) {
            hitTestOnScreenSpace(
                width, height,
                m_gbuffer);
        }
        else {
            hitTest(
                width, height,
                bounce);
        }
    }

    void SVGFPathTracing::onShade(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t sample,
        int32_t bounce, int32_t rrBounce)
    {
        auto mtx_W2C = params_.mtxs.GetW2C();

        dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        auto& curaov = params_.GetCurrAovBuffer();

        svgf_kernel::shade << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            curaov.get<AT_NAME::SVGFAovBufferType::NormalDepth>().data(),
            curaov.get<AT_NAME::SVGFAovBufferType::AlbedoMeshId>().data(),
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
            m_random.data(),
            m_shadowRays.data());

        checkCudaKernel(shade);

        onShadeByShadowRay(width, height, bounce);
    }

    void SVGFPathTracing::onGather(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t maxSamples)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        auto& curaov = params_.GetCurrAovBuffer();

        if (isFirstFrame() || m_mode == Mode::PT) {
            svgf_kernel::gather<true> << <grid, block, 0, m_stream >> > (
                outputSurf,
                width, height,
                path_host_->paths,
                params_.temporary_color_buffer.data(),
                curaov.get<AT_NAME::SVGFAovBufferType::ColorVariance>().data(),
                curaov.get<AT_NAME::SVGFAovBufferType::MomentTemporalWeight>().data());
        }
        else {
            svgf_kernel::gather<false> << <grid, block, 0, m_stream >> > (
                outputSurf,
                width, height,
                path_host_->paths,
                params_.temporary_color_buffer.data());
        }

        checkCudaKernel(gather);
    }
}
