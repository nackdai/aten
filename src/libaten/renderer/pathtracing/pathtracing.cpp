#include <array>

#include "renderer/pathtracing/pathtracing.h"

#include "accelerator/threaded_bvh_traverser.h"
#include "material/material_impl.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "sampler/cmj.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (-1)
#define BREAK_Y    (-1)
#pragma optimize( "", off)
#endif

namespace aten
{
    void PathTracing::radiance(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        const context& ctxt,
        scene* scene,
        const aten::CameraParameter& camera,
        aten::hitrecord* first_hrec/*= nullptr*/)
    {
        int32_t depth = 0;

        while (depth < m_maxDepth) {
            bool willContinue = true;
            Intersection isect;

            const auto& ray = rays_[idx];

            path_host_.paths.attrib[idx].attr.isHit = false;

            bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect,
                ctxt,
                ray,
                AT_MATH_EPSILON, AT_MATH_INF);

            if (is_hit) {
                if (depth == 0 && first_hrec) {
                    const auto& obj = ctxt.GetObject(isect.objid);
                    AT_NAME::evaluate_hit_result(*first_hrec, obj, ctxt, ray, isect);
                }

                path_host_.paths.attrib[idx].attr.isHit = true;

                shade(
                    idx,
                    path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                    isect, scene, m_rrDepth, depth);

                AT_NAME::AdvanceAlphaBlendPath(
                    ctxt, rays_[idx],
                    path_host_.paths.attrib[idx], path_host_.paths.throughput[idx]);

                const auto& mtrl = ctxt.GetMaterial(isect.mtrlid);

                std::ignore = AT_NAME::HitShadowRay(
                    depth,
                    ctxt, mtrl,
                    path_host_.paths.attrib[idx],
                    path_host_.paths.contrib[idx],
                    shadow_rays_[idx]);

                willContinue = !path_host_.paths.attrib[idx].attr.is_terminated;
            }
            else {
                ShadeMiss(
                    idx,
                    ix, iy,
                    width, height,
                    depth,
                    ctxt, camera,
                    path_host_.paths, rays_[idx]);

                willContinue = false;
            }

            if (!willContinue) {
                break;
            }

            depth++;
        }
    }

    void PathTracing::shade(
        int32_t idx,
        aten::Path& paths,
        const context& ctxt,
        ray* rays,
        aten::ShadowRay* shadow_rays,
        const aten::Intersection& isect,
        scene* scene,
        int32_t rrDepth,
        int32_t bounce)
    {
        if (paths.attrib[idx].attr.is_terminated) {
            return;
        }

        auto* sampler = &paths.sampler[idx];

        const auto& ray = rays[idx];
        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < float(0);

        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        vec3 orienting_normal = rec.normal;

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        auto albedo = AT_NAME::sampleTexture(ctxt, mtrl.albedoMap, rec.u, rec.v, mtrl.baseColor, bounce);

        auto& shadow_ray = shadow_rays[idx];
        shadow_ray.isActive = false;

        // Check stencil.
        auto is_stencil = AT_NAME::CheckStencil(
            rays[idx], paths.attrib[idx],
            bounce,
            ctxt,
            rec.p, orienting_normal,
            mtrl
        );
        if (is_stencil) {
            return;
        }

        // Check transparency or translucency.
        auto is_translucent_by_alpha = AT_NAME::CheckMaterialTranslucentByAlpha(
            ctxt,
            mtrl,
            rec.u, rec.v, rec.p,
            orienting_normal,
            rays[idx],
            paths.attrib[idx],
            paths.throughput[idx]);
        if (is_translucent_by_alpha) {
            return;
        }

        albedo = paths.throughput[idx].alpha_blend.transmission * albedo + paths.throughput[idx].alpha_blend.throughput;

        // Implicit conection to light.
        auto is_hit_implicit_light = AT_NAME::HitTeminatedMaterial(
            ctxt, paths.sampler[idx],
            isect.objid,
            isBackfacing,
            bounce,
            paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
            ray, rec,
            albedo, mtrl);
        if (is_hit_implicit_light) {
            return;
        }

        if (!mtrl.attrib.is_translucent && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        // Apply normal map.
        auto pre_sampled_r = material::applyNormal(
            ctxt,
            &mtrl,
            mtrl.normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir, sampler);

        // Explicit conection to light.
        AT_NAME::FillShadowRay(
            idx,
            shadow_ray,
            ctxt,
            bounce,
            paths,
            mtrl,
            ray,
            rec.p, orienting_normal,
            rec.u, rec.v, albedo,
            pre_sampled_r);

        const auto russianProb = AT_NAME::ComputeRussianProbability(
            bounce, rrDepth,
            paths.attrib[idx], paths.throughput[idx],
            paths.sampler[idx]);

        aten::MaterialSampling sampling;
        material::sampleMaterial(
            &sampling,
            ctxt,
            paths.throughput[idx].throughput,
            &mtrl,
            orienting_normal,
            ray.dir,
            rec.normal,
            sampler, pre_sampled_r,
            rec.u, rec.v);

        AT_NAME::PrepareForNextBounce(
            idx,
            rec, isBackfacing, russianProb,
            orienting_normal,
            mtrl, sampling,
            albedo,
            paths, rays);
    }

    void PathTracing::shadeMiss(
        const aten::context& ctxt,
        int32_t idx,
        scene* scene,
        int32_t depth,
        Path& paths,
        const ray* rays,
        aten::BackgroundResource& bg)
    {
        const auto& ray = rays[idx];

        auto ibl = scene->getIBL();
        aten::vec3 emit = AT_NAME::Background::SampleFromRay(ray.dir, bg, ctxt);
        float misW = 1.0F;

        if (ibl) {
            // TODO
            // Sample IBL properly.
            if (depth == 0) {
                float misW = 1.0F;
                paths.attrib[idx].attr.is_terminated = true;
            }
            else {
                auto pdfLight = ibl->samplePdf(ray, ctxt);
                misW = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);
            }
        }

        paths.contrib[idx].contrib += paths.throughput[idx].throughput * misW * emit;
    }

    void PathTracing::OnRender(
        context& ctxt,
        Destination& dst,
        scene* scene,
        Camera* camera)
    {
        int32_t width = dst.width;
        int32_t height = dst.height;
        uint32_t samples = dst.sample;

        m_maxDepth = dst.maxDepth;
        m_rrDepth = dst.russianRouletteDepth;

        if (m_rrDepth > m_maxDepth) {
            m_rrDepth = m_maxDepth - 1;
        }

        if (rays_.empty()) {
            rays_.resize(width * height);
        }
        if (shadow_rays_.empty()) {
            shadow_rays_.resize(width * height);
        }

        path_host_.init(width, height);
        path_host_.Clear(GetFrameCount());

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            auto thread_idx = OMPUtil::getThreadIdx();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    vec3 col = vec3(0);
                    vec3 col2 = vec3(0);
                    uint32_t cnt = 0;

#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif
                    int32_t idx = y * width + x;
                    aten::hitrecord hrec;

                    for (uint32_t i = 0; i < samples; i++) {
                        const auto rnd = aten::getRandom(idx);
                        const auto& camsample = camera->param();

                        GeneratePath(
                            rays_[idx],
                            idx,
                            x, y, width, height,
                            i, GetFrameCount(),
                            path_host_.paths,
                            camsample,
                            rnd);

                        path_host_.paths.contrib[idx].contrib = aten::vec3(0);

                            radiance(
                                idx,
                                x, y, width, height,
                                ctxt, scene, camsample, &hrec);

                        if (isInvalidColor(path_host_.paths.contrib[idx].contrib)) {
                            AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, i);
                            continue;
                        }

                        auto c = path_host_.paths.contrib[idx].contrib;

                        col += c;
                        col2 += c * c;
                        cnt++;

                        if (path_host_.paths.attrib[idx].attr.is_terminated) {
                            break;
                        }
                    }

                    col /= (float)cnt;

#if 0
                    if (hrec.mtrlid >= 0) {
                        const auto mtrl = ctxt.GetMaterial(hrec.mtrlid);
                        if (mtrl && mtrl->isNPR()) {
                            col = FeatureLine::RenderFeatureLine(
                                col,
                                x, y, width, height,
                                hrec,
                                ctxt, *scene, *camera);
                        }
                    }
#endif

                    dst.buffer->put(x, y, vec4(col, 1));

                    if (dst.variance) {
                        col2 /= (float)cnt;
                        dst.variance->put(x, y, vec4(col2 - col * col, float(1)));
                    }
                }
            }
        }
    }
}
