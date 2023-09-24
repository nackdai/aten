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

namespace svgf {
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
        uint32_t* random,
        idaten::ShadowRay* shadowRays)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

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
                aovNormalDepth[idx], orienting_normal, rec, mtxW2C,
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
                aovNormalDepth[idx], orienting_normal, rec, mtxW2C,
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

    __global__ void gather(
        cudaSurfaceObject_t dst,
        float4* aovColorVariance,
        float4* aovMomentTemporalWeight,
        const idaten::Path paths,
        float4* contribs,
        int32_t width, int32_t height)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        auto idx = getIdx(ix, iy, width);

        float4 c = paths.contrib[idx].v;
        int32_t sample = c.w;

        float3 contrib = make_float3(c.x, c.y, c.z) / sample;
        //contrib.w = sample;

        float lum = AT_NAME::color::luminance(contrib.x, contrib.y, contrib.z);

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

        int32_t curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        svgf::shade << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            curaov.get<AOVBuffer::NormalDepth>().data(),
            curaov.get<AOVBuffer::AlbedoMeshId>().data(),
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

        int32_t curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        svgf::gather << <grid, block, 0, m_stream >> > (
            outputSurf,
            curaov.get<AOVBuffer::ColorVariance>().data(),
            curaov.get<AOVBuffer::MomentTemporalWeight>().data(),
            path_host_->paths,
            m_tmpBuf.data(),
            width, height);

        checkCudaKernel(gather);
    }
}
