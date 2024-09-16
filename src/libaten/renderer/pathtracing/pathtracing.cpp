#include <array>

#include "renderer/pathtracing/pathtracing.h"

#include "material/material_impl.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/npr/npr_impl.h"
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

            path_host_.paths.attrib[idx].isHit = false;

            if (scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, isect)) {
                if (depth == 0 && first_hrec) {
                    const auto& obj = ctxt.GetObject(isect.objid);
                    AT_NAME::evaluate_hit_result(*first_hrec, obj, ctxt, ray, isect);
                }

                path_host_.paths.attrib[idx].isHit = true;

                shade(
                    idx,
                    path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                    isect, scene, m_rrDepth, depth);

                std::ignore = AT_NAME::HitShadowRay(
                    idx, depth, ctxt, path_host_.paths, shadow_rays_[idx], scene);

                willContinue = !path_host_.paths.attrib[idx].is_terminated;
            }
            else {
                auto ibl = scene->getIBL();
                if (ibl && enable_envmap_) {
                    ShadeMissWithEnvmap(
                        idx,
                        ix, iy,
                        width, height,
                        depth,
                        bg_,
                        ctxt, camera,
                        path_host_.paths, rays_[idx]);
                }
                else {
                    ShadeMiss(
                        idx,
                        depth,
                        bg_.bg_color,
                        path_host_.paths);
                }

                willContinue = false;
            }

            if (!willContinue) {
                break;
            }

            depth++;
        }
    }

    void PathTracing::radiance_with_feature_line(
        int32_t idx,
        Path& paths,
        const context& ctxt,
        ray* rays,
        ShadowRay* shadow_rays,
        int32_t rrDepth,
        int32_t maxDepth,
        Camera* cam,
        scene* scene,
        aten::BackgroundResource& bg)
    {
        int32_t depth = 0;

        const auto& ray = rays[idx];
        auto* sampler = &paths.sampler[idx];

        const auto& cam_org = cam->param().origin;

        const auto pixel_width = cam->ComputePixelWidthAtDistance(1);

        constexpr size_t SampleRayNum = 8;

        // TODO: These value should be configurable.
        constexpr float feature_line_width = 1;
        constexpr float albedo_threshold = 0.1f;
        constexpr float normal_threshold = 0.1f;
        static const aten::vec3 LineColor(0, 1, 0);

        std::array<AT_NAME::npr::FeatureLine::SampleRayDesc, SampleRayNum> sample_ray_descs;
        AT_NAME::npr::FeatureLine::Disc disc;

        AT_NAME::npr::GenerateSampleRayAndDiscPerQueryRay<SampleRayNum>(
            sample_ray_descs, disc,
            ray, *sampler,
            feature_line_width, pixel_width);

        while (depth < maxDepth) {
            hitrecord hrec_query;

            bool willContinue = true;
            Intersection isect;

            // Current closest distance to feature line point.
            auto closest_feature_line_point_distance = std::numeric_limits<float>::max();

            // Whether the feature line point has been found.
            bool is_found_feature_line_point = false;

            float hit_point_distance = 0;

            if (scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, isect)) {
                const auto& obj = ctxt.GetObject(isect.objid);
                AT_NAME::evaluate_hit_result(hrec_query, obj, ctxt, ray, isect);

                const auto distance_query_ray_hit = length(hrec_query.p - ray.org);

                // disc.centerはquery_ray.orgに一致する.
                // ただし、最初だけは、query_ray.orgはカメラ視点になっているが、
                // accumulated_distanceでカメラとdiscの距離がすでに含まれている.
                hit_point_distance = length(hrec_query.p - disc.center);

                const auto prev_disc = disc;
                disc = AT_NAME::npr::FeatureLine::ComputeDiscAtQueryRayHitPoint(
                    hrec_query.p,
                    ray.dir,
                    prev_disc.radius,
                    hit_point_distance,
                    disc.accumulated_distance);

                for (size_t i = 0; i < SampleRayNum; i++) {
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    auto sample_ray = AT_NAME::npr::GetSampleRay(
                        depth,
                        sample_ray_descs[i],
                        prev_disc, disc);
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    Intersection isect_sample_ray;

                    if (scene->hit(ctxt, sample_ray, AT_MATH_EPSILON, AT_MATH_INF, isect_sample_ray)) {
                        // Query ray hits and then sample ray hits.
                        aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = AT_NAME::npr::EvaluateQueryAndSampleRayHit(
                            sample_ray_descs[i],
                            ctxt, cam_org,
                            ray, hrec_query, distance_query_ray_hit,
                            isect_sample_ray,
                            disc,
                            is_found_feature_line_point,
                            closest_feature_line_point_distance,
                            feature_line_width, pixel_width,
                            albedo_threshold, normal_threshold);
                    }
                    else {
                        // Query ray hits but sample ray doesn't hit anything.
                        aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = AT_NAME::npr::EvaluateQueryRayHitButSampleRayNotHit(
                            sample_ray_descs[i],
                            ray, hrec_query, distance_query_ray_hit,
                            sample_ray, disc,
                            is_found_feature_line_point,
                            closest_feature_line_point_distance,
                            feature_line_width, pixel_width);
                    }

                    const auto mtrl = ctxt.GetMaterial(hrec_query.mtrlid);
                    if (!mtrl.attrib.is_glossy) {
                        // In non glossy material case, sample ray doesn't bounce anymore.
                        // TODO
                        // Even if material is glossy, how glossy depends on parameter (e.g. roughness etc).
                        // So, I need to consider how to indetify if sample ray bounce is necessary based on material.
                        sample_ray_descs[i].is_terminated = true;
                    }
                }

                if (!is_found_feature_line_point) {
                    shade(
                        idx, paths, ctxt, rays, shadow_rays,
                        isect, scene, rrDepth, depth);
                    AT_NAME::HitShadowRay(
                        idx, depth, ctxt, paths, shadow_rays[idx], scene);
                }
            }
            else {
                // Query ray doesn't hit anything, but evaluate a possibility that sample ray might hit something.

                // NOTE:
                // In order to compute sample ray, previous disc and next disc are necessary.
                // In first bounce, initial point is camera original.
                // So, previous disc is not necessary.

                // Query ray doesn't hit to anything, and we can't create the next disc at query ray hit point.
                // But, the disc is necessary. So, the next disc is created forcibly from the dummy query ray hit point.
                AT_NAME::npr::FeatureLine::Disc prev_disc;
                hit_point_distance = CreateNextDiscByDummyQueryRayHitPoint(depth, hit_point_distance, ray, prev_disc, disc);

                for (int32_t i = 0; i < SampleRayNum; i++) {
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    auto sample_ray = AT_NAME::npr::GetSampleRay(
                        depth,
                        sample_ray_descs[i],
                        prev_disc, disc);
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    Intersection isect_sample_ray;
                    if (scene->hit(ctxt, sample_ray, AT_MATH_EPSILON, AT_MATH_INF, isect_sample_ray)) {
                        // Query ray doesn't hit, but sample ray hits.
                        aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = AT_NAME::npr::EvaluateQueryRayNotHitButSampleRayHit(
                            ctxt, ray,
                            isect_sample_ray,
                            disc,
                            is_found_feature_line_point,
                            closest_feature_line_point_distance,
                            feature_line_width, pixel_width);
                    }
                    else {
                        // Sample ray doesn't hit anything.
                        // So, terminate sample ray traversing immediately.
                        sample_ray_descs[i].is_terminated = true;
                        break;
                    }
                }

                if (!is_found_feature_line_point) {
                    shadeMiss(ctxt, idx, scene, depth, paths, rays, bg);
                    willContinue = false;
                }
            }

            if (is_found_feature_line_point) {
                AT_NAME::npr::ComputeFeatureLineContribution<SampleRayNum>(
                    closest_feature_line_point_distance,
                    paths, idx, LineColor);
                willContinue = false;
            }

            if (!willContinue) {
                break;
            }

            depth++;

            disc.accumulated_distance += hit_point_distance;
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

        auto albedo = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, mtrl.baseColor, bounce);

        auto& shadow_ray = shadow_rays[idx];
        shadow_ray.isActive = false;

        // Implicit conection to light.
        auto is_hit_implicit_light = AT_NAME::HitImplicitLight(
            ctxt, isect.objid,
            isBackfacing,
            bounce,
            paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
            ray,
            rec.p, orienting_normal,
            rec.area,
            mtrl);
        if (is_hit_implicit_light) {
            return;
        }

        if (!mtrl.attrib.is_translucent && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        // Apply normal map.
        auto pre_sampled_r = material::applyNormal(
            &mtrl,
            mtrl.normalMap,
            orienting_normal, orienting_normal,
            rec.u, rec.v,
            ray.dir, sampler);

        // Check transparency or translucency.
        auto is_translucent_by_alpha = AT_NAME::CheckMaterialTranslucentByAlpha(
            mtrl,
            rec.u, rec.v, rec.p,
            orienting_normal,
            rays[idx],
            paths.sampler[idx],
            paths.attrib[idx],
            paths.throughput[idx]);
        if (is_translucent_by_alpha) {
            return;
        }

        // Explicit conection to light.
        AT_NAME::FillShadowRay(
            shadow_ray,
            ctxt,
            bounce,
            paths.sampler[idx],
            paths.throughput[idx],
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
                paths.attrib[idx].is_terminated = true;
            }
            else {
                auto pdfLight = ibl->samplePdf(ray, ctxt);
                misW = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);
            }
        }

        paths.contrib[idx].contrib += paths.throughput[idx].throughput * misW * emit;
    }

    void PathTracing::OnRender(
        const context& ctxt,
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

        auto time = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            auto thread_idx = OMPUtil::getThreadIdx();

            auto t = timer::getSystemTime();

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
                            x, y,
                            i, GetFrameCount(),
                            path_host_.paths,
                            camsample,
                            rnd);

                        path_host_.paths.contrib[idx].contrib = aten::vec3(0);

                        if (enable_feature_line_) {
                            radiance_with_feature_line(
                                idx,
                                path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                                m_rrDepth, m_maxDepth,
                                camera,
                                scene,
                                bg_);
                        }
                        else {
                            radiance(
                                idx,
                                x, y, width, height,
                                ctxt, scene, camsample, &hrec);
                        }

                        if (isInvalidColor(path_host_.paths.contrib[idx].contrib)) {
                            AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, i);
                            continue;
                        }

                        auto c = path_host_.paths.contrib[idx].contrib;

                        col += c;
                        col2 += c * c;
                        cnt++;

                        if (path_host_.paths.attrib[idx].is_terminated) {
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
