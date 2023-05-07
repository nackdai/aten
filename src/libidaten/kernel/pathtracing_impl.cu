#include "kernel/pathtracing.h"

#include "aten4idaten.h"
#include "kernel/accelerator.cuh"
#include "kernel/context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/pt_common.h"
#include "kernel/StreamCompaction.h"
#include "kernel/pt_standard_impl.cuh"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

namespace pt {
    __global__ void shade(
        idaten::TileDomain tileDomain,
        float4* aovNormalDepth,
        float4* aovAlbedoMeshId,
        int32_t width, int32_t height,
        idaten::Path paths,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        const aten::Intersection* __restrict__ isects,
        aten::ray* rays,
        int32_t sample,
        int32_t frame,
        int32_t bounce, int32_t rrBounce,
        const aten::ObjectParameter* __restrict__ shapes, int32_t geomnum,
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

        idx = hitindices[idx];

        if (paths.attrib[idx].isKill || paths.attrib[idx].isTerminate) {
            paths.attrib[idx].isTerminate = true;
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

        auto albedo = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

        // Apply normal map.
        int32_t normalMap = shMtrls[threadIdx.x].normalMap;
        auto pre_sampled_r = applyNormal(
            &shMtrls[threadIdx.x],
            normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir,
            &paths.sampler[idx]);

        if (bounce == 0) {
            // Store AOV.
            const auto _idx = kernel::adjustIndexWithTiledomain(idx, tileDomain, width);

            AT_NAME::FillBasicAOVs(
                aovNormalDepth[_idx], orienting_normal, rec, aten::mat4(),
                aovAlbedoMeshId[_idx], albedo, isect);
        }

        // Implicit conection to light.
        if (shMtrls[threadIdx.x].attrib.isEmissive) {
            kernel::hitImplicitLight(
                isBackfacing,
                bounce,
                paths.contrib[idx],
                paths.attrib[idx],
                paths.throughput[idx],
                ray,
                rec.p, orienting_normal,
                rec.area,
                shMtrls[threadIdx.x]);

            // When ray hit the light, tracing will finish.
            paths.attrib[idx].isTerminate = true;
            return;
        }

        if (!shMtrls[threadIdx.x].attrib.isTranslucent && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        shShadowRays[threadIdx.x].isActive = false;

        // Check transparency or translucency.
        // NOTE:
        // If the material itself is originally translucent, we don't care alpha translucency.
        if (!shMtrls[threadIdx.x].attrib.isTranslucent
            && AT_NAME::material::isTranslucentByAlpha(shMtrls[threadIdx.x], rec.u, rec.v))
        {
            const auto alpha = AT_NAME::material::getTranslucentAlpha(shMtrls[threadIdx.x], rec.u, rec.v);
            auto r = paths.sampler[idx].nextSample();

            if (r >= alpha) {
                // Just through the object.
                // NOTE
                // Ray go through to the opposite direction. So, we need to specify inverted normal.
                rays[idx] = aten::ray(rec.p, ray.dir, -orienting_normal);
                paths.throughput[idx].throughput *= static_cast<aten::vec3>(shMtrls[threadIdx.x].baseColor);
                paths.attrib[idx].isSingular = true;
                shadowRays[idx].isActive = false;
                return;
            }
        }

        // Explicit conection to light.
        if (!(shMtrls[threadIdx.x].attrib.isSingular || shMtrls[threadIdx.x].attrib.isTranslucent))
        {
            auto lightidx = aten::cmpMin<int32_t>(paths.sampler[idx].nextSample() * lightnum, lightnum - 1);

            aten::LightParameter light;
            light.pos = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 0];
            light.dir = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 1];
            light.le = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 2];
            light.v0 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 3];
            light.v1 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 4];
            light.v2 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 5];

            auto lightSelectPdf = 1.0f / lightnum;

            auto isShadowRayActive = kernel::fillShadowRay(
                shShadowRays[threadIdx.x],
                ctxt,
                bounce,
                paths.sampler[idx],
                paths.throughput[idx],
                lightidx,
                light,
                shMtrls[threadIdx.x],
                ray,
                rec.p, orienting_normal,
                rec.u, rec.v, albedo,
                lightSelectPdf,
                pre_sampled_r);

            shShadowRays[threadIdx.x].isActive = isShadowRayActive;
        }

        shadowRays[idx] = shShadowRays[threadIdx.x];

        const auto russianProb = kernel::executeRussianProbability(
            bounce, rrBounce,
            paths.attrib[idx],
            paths.throughput[idx],
            paths.sampler[idx]);

        AT_NAME::MaterialSampling sampling;

        sampleMaterial(
            &sampling,
            &ctxt,
            &shMtrls[threadIdx.x],
            orienting_normal,
            ray.dir,
            rec.normal,
            &paths.sampler[idx], pre_sampled_r,
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

        auto c = dot(rayBasedNormal, nextDir);

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
        const idaten::ShadowRay* __restrict__ shadowRays,
        const aten::ObjectParameter* __restrict__ shapes, int32_t geomnum,
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

        idx = hitindices[idx];

        if (paths.attrib[idx].isKill || paths.attrib[idx].isTerminate) {
            paths.attrib[idx].isTerminate = true;
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

        const auto& shadowRay = shadowRays[idx];

        if (!shadowRay.isActive) {
            return;
        }

        // TODO
        bool enableLod = (bounce >= 2);

        auto isHit = kernel::hitShadowRay(
            enableLod, ctxt, shadowRay);

        if (isHit) {
            auto contrib = shadowRay.lightcontrib;
            paths.contrib[idx].contrib += make_float3(contrib.x, contrib.y, contrib.z);
        }
    }

    __global__ void gather(
        idaten::TileDomain tileDomain,
        cudaSurfaceObject_t dst,
        const idaten::Path paths,
        bool enableProgressive,
        int32_t width, int32_t height)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= tileDomain.w || iy >= tileDomain.h) {
            return;
        }

        auto idx = getIdx(ix, iy, tileDomain.w);

        float4 c = paths.contrib[idx].v;
        int32_t sample = c.w;

        float4 contrib = c;

        ix += tileDomain.x;
        iy += tileDomain.y;
        idx = getIdx(ix, iy, width);

        if (enableProgressive) {
            float4 data;
            surf2Dread(&data, dst, ix * sizeof(float4), iy);

            // First data.w value is 0.
            int32_t n = data.w;
            contrib = n * data + make_float4(c.x, c.y, c.z, 0) / sample;
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
}

namespace idaten
{
    void PathTracing::onHitTest(
        int32_t width, int32_t height,
        int32_t bounce,
        cudaTextureObject_t texVtxPos)
    {
        hitTest(
            width, height,
            bounce,
            texVtxPos);
    }

    void PathTracing::onShade(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t sample,
        int32_t bounce, int32_t rrBounce,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 blockPerGrid(((m_tileDomain.w * m_tileDomain.h) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        pt::shade << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            m_tileDomain,
            aov_.normal_depth().ptr(),
            aov_.albedo_meshid().ptr(),
            width, height,
            m_paths,
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

    void PathTracing::onShadeByShadowRay(
        int32_t bounce,
        cudaTextureObject_t texVtxPos)
    {
        dim3 blockPerGrid(((m_tileDomain.w * m_tileDomain.h) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        pt::hitShadowRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            bounce,
            m_paths,
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
        int32_t width, int32_t height,
        int32_t maxSamples)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        pt::gather << <grid, block, 0, m_stream >> > (
            m_tileDomain,
            outputSurf,
            m_paths,
            m_enableProgressive,
            width, height);

        checkCudaKernel(gather);
    }
}
