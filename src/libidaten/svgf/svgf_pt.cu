#include "svgf/svgf.h"

#include "kernel/StreamCompaction.h"

#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/pt_common.h"
#include "kernel/pt_standard_impl.cuh"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

namespace svgf {
    __global__ void shade(
        idaten::TileDomain tileDomain,
        float4* aovNormalDepth,
        float4* aovTexclrMeshid,
        aten::mat4 mtxW2C,
        int32_t width, int32_t height,
        idaten::Path* paths,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        const aten::Intersection* __restrict__ isects,
        aten::ray* rays,
        int32_t sample,
        int32_t frame,
        int32_t bounce, int32_t rrBounce,
        const aten::GeomParameter* __restrict__ shapes, int32_t geomnum,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights, int32_t lightnum,
        const aten::TriangleParameter* __restrict__ prims,
        cudaTextureObject_t vtxPos,
        cudaTextureObject_t vtxNml,
        const aten::mat4* __restrict__ matrices,
        cudaTextureObject_t* textures,
        uint32_t* random,
        idaten::ShadowRay* shadowRays)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

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

        __shared__ idaten::ShadowRay shShadowRays[64 * idaten::SVGFPathTracing::ShadowRayNum];
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
#endif

        aten::hitrecord rec;

        const auto& isect = isects[idx];

        auto obj = &ctxt.shapes[isect.objid];
        evalHitResult(&ctxt, obj, ray, &rec, &isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        aten::vec3 orienting_normal = rec.normal;

        gatherMaterialInfo(
            shMtrls[threadIdx.x],
            &ctxt,
            rec.mtrlid,
            rec.isVoxel);

        // Render AOVs.
        // NOTE
        // 厳密に法線をAOVに保持するなら、法線マップ適用後するべき.
        // しかし、temporal reprojection、atrousなどのフィルタ適用時に法線を参照する際に、法線マップが細かすぎてはじかれてしまうことがある.
        // それにより、フィルタがおもったようにかからずフィルタの品質が下がってしまう問題が発生する.
        if (bounce == 0) {
            const auto _idx = kernel::adjustIndexWithTiledomain(idx, tileDomain, width);

            // texture color
            auto texcolor = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec4(1.0f));

            AT_NAME::FillBasicAOVs(
                aovNormalDepth[_idx], orienting_normal, rec, aten::mat4(),
                aovTexclrMeshid[_idx], texcolor, isect);
            aovTexclrMeshid[_idx].w = isect.mtrlid;

            // For exporting separated albedo.
            shMtrls[threadIdx.x].albedoMap = -1;
        }
        // TODO
        // How to deal Refraction?
        else if (bounce == 1 && paths->attrib[idx].mtrlType == aten::MaterialType::Specular) {
            const auto _idx = kernel::adjustIndexWithTiledomain(idx, tileDomain, width);

            // texture color.
            auto texcolor = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec4(1.0f));

            AT_NAME::FillBasicAOVs(
                aovNormalDepth[_idx], orienting_normal, rec, aten::mat4(),
                aovTexclrMeshid[_idx], texcolor, isect);
            aovTexclrMeshid[_idx].w = isect.mtrlid;

            // For exporting separated albedo.
            shMtrls[threadIdx.x].albedoMap = -1;
        }

        // Implicit conection to light.
        if (shMtrls[threadIdx.x].attrib.isEmissive) {
            kernel::hitImplicitLight(
                isBackfacing,
                bounce,
                paths->contrib[idx],
                paths->attrib[idx],
                paths->throughput[idx],
                ray,
                rec.p, orienting_normal,
                rec.area,
                shMtrls[threadIdx.x]);

            // When ray hit the light, tracing will finish.
            paths->attrib[idx].isTerminate = true;
            return;
        }

        if (!shMtrls[threadIdx.x].attrib.isTranslucent && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        // Apply normal map.
        int32_t normalMap = shMtrls[threadIdx.x].normalMap;
        auto pre_sample_r = applyNormal(
            &shMtrls[threadIdx.x],
            normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir,
            &paths->sampler[idx]);

        auto albedo = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

#if 1
#pragma unroll
        for (int32_t i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
            shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].isActive = false;
        }

        // Explicit conection to light.
        if (!(shMtrls[threadIdx.x].attrib.isSingular || shMtrls[threadIdx.x].attrib.isTranslucent))
        {
            for (int32_t i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
                // TODO
                // Importance sampling.
                int32_t lightidx = aten::cmpMin<int32_t>(paths->sampler[idx].nextSample() * lightnum, lightnum - 1);

                aten::LightParameter light;
                light.pos = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 0];
                light.dir = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 1];
                light.le = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 2];
                light.v0 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 3];
                light.v1 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 4];
                light.v2 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 5];
                //auto light = ctxt.lights[lightidx];

                real lightSelectPdf = 1.0f / lightnum;

                auto isShadowRayActive = kernel::fillShadowRay(
                    shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i],
                    ctxt,
                    bounce,
                    paths->sampler[idx],
                    paths->throughput[idx],
                    lightidx,
                    light,
                    shMtrls[threadIdx.x],
                    ray,
                    rec.p, orienting_normal,
                    rec.u, rec.v, albedo,
                    lightSelectPdf);

                //shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].lightcontrib /= idaten::SVGFPathTracing::ShadowRayNum;
                shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i].isActive = isShadowRayActive;
            }
        }
#endif

        const auto russianProb = kernel::executeRussianProbability(
            bounce, rrBounce,
            paths->attrib[idx],
            paths->throughput[idx],
            paths->sampler[idx]);

        AT_NAME::MaterialSampling sampling;

        sampleMaterial(
            &sampling,
            &ctxt,
            &shMtrls[threadIdx.x],
            orienting_normal,
            ray.dir,
            rec.normal,
            &paths->sampler[idx],
            pre_sample_r,
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
            paths->throughput[idx].throughput *= bsdf * c / pdfb;
            paths->throughput[idx].throughput /= russianProb;
        }
        else {
            paths->attrib[idx].isTerminate = true;
            return;
        }

        // Make next ray.
        rays[idx] = aten::ray(rec.p, nextDir, rayBasedNormal);

        paths->throughput[idx].pdfb = pdfb;
        paths->attrib[idx].isSingular = shMtrls[threadIdx.x].attrib.isSingular;
        paths->attrib[idx].mtrlType = shMtrls[threadIdx.x].type;

#pragma unroll
        for (int32_t i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
            shadowRays[idx * idaten::SVGFPathTracing::ShadowRayNum + i] = shShadowRays[threadIdx.x * idaten::SVGFPathTracing::ShadowRayNum + i];
        }
    }

    __global__ void hitShadowRay(
        int32_t bounce,
        idaten::Path* paths,
        int32_t* hitindices,
        int32_t* hitnum,
        const idaten::ShadowRay* __restrict__ shadowRays,
        const aten::GeomParameter* __restrict__ shapes, int32_t geomnum,
        aten::MaterialParameter* mtrls,
        const aten::LightParameter* __restrict__ lights, int32_t lightnum,
        cudaTextureObject_t* nodes,
        const aten::TriangleParameter* __restrict__ prims,
        cudaTextureObject_t vtxPos,
        const aten::mat4* __restrict__ matrices)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

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

        // TODO
        bool enableLod = (bounce >= 2);

#pragma unroll
        for (int32_t i = 0; i < idaten::SVGFPathTracing::ShadowRayNum; i++) {
            const auto& shadowRay = shadowRays[idx * idaten::SVGFPathTracing::ShadowRayNum + i];

            if (!shadowRay.isActive) {
                continue;
            }

            auto isHit = kernel::hitShadowRay(
                enableLod, ctxt, shadowRay);

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
        const idaten::Path* __restrict__ paths,
        float4* contribs,
        int32_t width, int32_t height)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= tileDomain.w || iy >= tileDomain.h) {
            return;
        }

        auto idx = getIdx(ix, iy, tileDomain.w);

        float4 c = paths->contrib[idx].v;
        int32_t sample = c.w;

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
}

namespace idaten
{
    void SVGFPathTracing::onHitTest(
        int32_t width, int32_t height,
        int32_t bounce,
        cudaTextureObject_t texVtxPos)
    {
        if (bounce == 0 && m_canSSRTHitTest) {
            hitTestOnScreenSpace(
                width, height,
                m_gbuffer,
                texVtxPos);
        }
        else {
            hitTest(
                width, height,
                bounce,
                texVtxPos);
        }
    }

    void SVGFPathTracing::onShade(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t sample,
        int32_t bounce, int32_t rrBounce,
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

        int32_t curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        svgf::shade << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_tileDomain,
            curaov.get<AOVBuffer::NormalDepth>().ptr(),
            curaov.get<AOVBuffer::AlbedoMeshId>().ptr(),
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

        onShadeByShadowRay(bounce, texVtxPos);
    }

    void SVGFPathTracing::onShadeByShadowRay(
        int32_t bounce,
        cudaTextureObject_t texVtxPos)
    {
        dim3 blockPerGrid(((m_tileDomain.w * m_tileDomain.h) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        svgf::hitShadowRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
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
        int32_t width, int32_t height,
        int32_t maxSamples)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        int32_t curaov_idx = getCurAovs();
        auto& curaov = aov_[curaov_idx];

        svgf::gather << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            outputSurf,
            curaov.get<AOVBuffer::ColorVariance>().ptr(),
            curaov.get<AOVBuffer::MomentTemporalWeight>().ptr(),
            m_paths.ptr(),
            m_tmpBuf.ptr(),
            width, height);

        checkCudaKernel(gather);
    }
}
