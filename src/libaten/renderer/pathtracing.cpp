#include <array>

#include "renderer/pathtracing.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/renderer_utility.h"
#include "renderer/pathtracing_impl.h"
#include "renderer/feature_line.h"
#include "sampler/cmj.h"

#include "material/material_impl.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (10)
#define BREAK_Y    (0)
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
        uint32_t depth = 0;

        while (depth < m_maxDepth) {
            hitrecord rec;

            bool willContinue = true;
            Intersection isect;

            const auto& ray = rays_[idx];

            if (scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, rec, isect)) {
                if (depth == 0 && first_hrec) {
                    *first_hrec = rec;
                }

                shade(
                    idx, path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                    rec, scene, m_rrDepth, depth);

                HitShadowRay(
                    scene, idx, depth, ctxt, path_host_.paths, shadow_rays_.data());

                willContinue = !path_host_.paths.attrib[idx].isTerminate;
            }
            else {
                auto ibl = scene->getIBL();
                if (ibl) {
                    shade_miss_with_envmap(
                        idx,
                        ix, iy,
                        width, height,
                        depth,
                        ibl->param().envmapidx,
                        ibl->getAvgIlluminace(),
                        real(1),
                        ctxt, camera,
                        path_host_.paths, rays_[idx]);
                }
                else {
                    shade_miss(
                        idx,
                        depth,
                        bg()->sample(rays_[idx]),
                        path_host_.paths);
                }

                willContinue = false;
            }

            if (depth < m_startDepth && !path_host_.paths.attrib[idx].isTerminate) {
                path_host_.paths.contrib[idx].contrib = vec3(0);
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
        int32_t startDepth,
        int32_t maxDepth,
        camera* cam,
        scene* scene,
        const background* bg)
    {
        uint32_t depth = 0;

        const auto& ray = rays[idx];
        auto* sampler = &paths.sampler[idx];

        const auto& cam_org = cam->param().origin;

        const auto pixel_width = cam->computePixelWidthAtDistance(1);

        constexpr size_t SampleRayNum = 8;

        // TODO: These value should be configurable.
        constexpr real FeatureLineWidth = 1;
        constexpr real ThresholdAlbedo = 0.1f;
        constexpr real ThresholdNormal = 0.1f;
        static const aten::vec3 LineColor(0, 1, 0);

        std::array<aten::FeatureLine::SampleRayDesc, SampleRayNum> sample_ray_descs;

        auto disc = aten::FeatureLine::generateDisc(ray, FeatureLineWidth, pixel_width);
        for (size_t i = 0; i < SampleRayNum; i++) {
            const auto sample_ray = aten::FeatureLine::generateSampleRay(sample_ray_descs[i], *sampler, ray, disc);
            aten::FeatureLine::storeRayToDesc(sample_ray_descs[i], sample_ray);
        }

        disc.accumulated_distance = 1;

        while (depth < maxDepth) {
            hitrecord hrec_query;

            bool willContinue = true;
            Intersection isect;

            auto closest_sample_ray_distance = std::numeric_limits<real>::max();
            int32_t closest_sample_ray_idx = -1;
            real hit_point_distance = 0;

            if (scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, hrec_query, isect)) {
                const auto distance_query_ray_hit = length(hrec_query.p - ray.org);

                // disc.centerはquery_ray.orgに一致する.
                // ただし、最初だけは、query_ray.orgはカメラ視点になっているが、
                // accumulated_distanceでカメラとdiscの距離がすでに含まれている.
                hit_point_distance = length(hrec_query.p - disc.center);

                const auto prev_disc = disc;
                disc = aten::FeatureLine::computeNextDisc(
                    hrec_query.p,
                    ray.dir,
                    prev_disc.radius,
                    hit_point_distance,
                    disc.accumulated_distance);

                for (size_t i = 0; i < SampleRayNum; i++) {
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    if (depth > 0) {
                        // Generate next sample ray.
                        const auto res_next_sample_ray = aten::FeatureLine::computeNextSampleRay(
                            sample_ray_descs[i],
                            prev_disc, disc);
                        const auto is_sample_ray_valid = std::get<0>(res_next_sample_ray);
                        if (!is_sample_ray_valid) {
                            sample_ray_descs[i].is_terminated = true;
                            continue;
                        }
                        const auto sample_ray = std::get<1>(res_next_sample_ray);
                        aten::FeatureLine::storeRayToDesc(sample_ray_descs[i], sample_ray);
                    }

                    Intersection isect_sample_ray;
                    hitrecord hrec_sample;

                    const auto sample_ray = aten::FeatureLine::getRayFromDesc(sample_ray_descs[i]);

                    if (scene->hit(ctxt, sample_ray, AT_MATH_EPSILON, AT_MATH_INF, hrec_sample, isect_sample_ray)) {
                        // If sample ray hit with the different mesh from query ray one, this sample ray won't bounce in next loop.
                        sample_ray_descs[i].is_terminated = isect_sample_ray.meshid != isect.meshid;
                        sample_ray_descs[i].prev_ray_hit_pos = hrec_sample.p;
                        sample_ray_descs[i].prev_ray_hit_nml = hrec_sample.normal;

                        const auto distance_sample_pos_on_query_ray = aten::FeatureLine::computeDistanceBetweenProjectedPosOnRayAndRayOrg(
                            hrec_sample.p, ray);

                        const auto is_line_width = aten::FeatureLine::isInLineWidth(
                            FeatureLineWidth,
                            ray,
                            hrec_sample.p,
                            disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                            pixel_width);
                        if (is_line_width) {
                            const auto query_albedo = material::sampleAlbedoMap(
                                &ctxt.getMaterial(hrec_query.mtrlid)->param(),
                                hrec_query.u, hrec_query.v);
                            const auto sample_albedo = material::sampleAlbedoMap(
                                &ctxt.getMaterial(hrec_sample.mtrlid)->param(),
                                hrec_query.u, hrec_query.v);
                            const auto query_depth = length(hrec_query.p - cam_org);
                            const auto sample_depth = length(hrec_sample.p - cam_org);

                            const auto is_feature_line = aten::FeatureLine::evaluateMetrics(
                                ray.org,
                                hrec_query, hrec_sample,
                                query_albedo, sample_albedo,
                                query_depth, sample_depth,
                                ThresholdAlbedo, ThresholdNormal,
                                2);

                            if (is_feature_line) {
                                if (distance_sample_pos_on_query_ray < closest_sample_ray_distance
                                    && distance_sample_pos_on_query_ray < distance_query_ray_hit)
                                {
                                    // Deal with sample hit point as FeatureLine.
                                    closest_sample_ray_idx = i;
                                    closest_sample_ray_distance = distance_sample_pos_on_query_ray;
                                }
                                else if (distance_query_ray_hit < closest_sample_ray_distance) {
                                    // Deal with query hit point as FeatureLine.
                                    closest_sample_ray_idx = SampleRayNum;
                                    closest_sample_ray_distance = distance_query_ray_hit;
                                }
                            }
                        }
                    }
                    else {
                        const auto query_hit_plane = aten::FeatureLine::computePlane(hrec_query);
                        const auto res_sample_ray_dummy_hit = aten::FeatureLine::computeRayHitPosOnPlane(
                            query_hit_plane, sample_ray);

                        const auto is_hit_sample_ray_dummy_plane = std::get<0>(res_sample_ray_dummy_hit);
                        if (is_hit_sample_ray_dummy_plane) {
                            const auto sample_ray_dummy_hit_pos = std::get<1>(res_sample_ray_dummy_hit);

                            const auto distance_sample_pos_on_query_ray = aten::FeatureLine::computeDistanceBetweenProjectedPosOnRayAndRayOrg(
                                sample_ray_dummy_hit_pos, ray);

                            const auto is_line_width = aten::FeatureLine::isInLineWidth(
                                FeatureLineWidth,
                                ray,
                                sample_ray_dummy_hit_pos,
                                disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                                pixel_width);
                            if (is_line_width) {
                                // If sample ray doesn't hit anything, it is forcibly feature line.
                                if (distance_sample_pos_on_query_ray < closest_sample_ray_distance
                                    && distance_sample_pos_on_query_ray < distance_query_ray_hit)
                                {
                                    // Deal with sample hit point as FeatureLine.
                                    closest_sample_ray_idx = i;
                                    closest_sample_ray_distance = distance_sample_pos_on_query_ray;
                                }
                                else if (distance_query_ray_hit < closest_sample_ray_distance) {
                                    // Deal with query hit point as FeatureLine.
                                    closest_sample_ray_idx = SampleRayNum;
                                    closest_sample_ray_distance = distance_query_ray_hit;
                                }
                            }
                        }

                        sample_ray_descs[i].is_terminated = true;
                    }

                    const auto mtrl = ctxt.getMaterial(hrec_query.mtrlid);
                    if (!mtrl->param().attrib.isGlossy) {
                        // In non glossy material case, sample ray doesn't bounce anymore.
                        // TODO
                        // Even if material is glossy, how glossy depends on parameter (e.g. roughness etc).
                        // So, I need to consider how to indetify if sample ray bounce is necessary based on material.
                        sample_ray_descs[i].is_terminated = true;
                    }
                }

                if (closest_sample_ray_idx < 0) {
                    shade(
                        idx, paths, ctxt, rays, shadow_rays,
                        hrec_query, scene, rrDepth, depth);
                    HitShadowRay(
                        scene, idx, depth, ctxt, paths, shadow_rays);
                }
            }
            else {
                aten::FeatureLine::Disc prev_disc;
                if (depth > 0) {
                    // Make dummy point to compute next sample ray.
                    const auto dummy_query_hit_pos = ray.org + real(100) * ray.dir;
                    hit_point_distance = length(dummy_query_hit_pos - disc.center);

                    prev_disc = disc;
                    disc = aten::FeatureLine::computeNextDisc(
                        dummy_query_hit_pos,
                        ray.dir,
                        prev_disc.radius,
                        hit_point_distance,
                        disc.accumulated_distance);
                }

                for (size_t i = 0; i < SampleRayNum; i++) {
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    auto sample_ray = aten::FeatureLine::getRayFromDesc(sample_ray_descs[i]);
                    if (depth > 0) {
                        // Generate next sample ray.
                        const auto res_next_sample_ray = aten::FeatureLine::computeNextSampleRay(
                            sample_ray_descs[i],
                            prev_disc, disc);
                        const auto is_sample_ray_valid = std::get<0>(res_next_sample_ray);
                        if (!is_sample_ray_valid) {
                            sample_ray_descs[i].is_terminated = true;
                            continue;
                        }
                        sample_ray = std::get<1>(res_next_sample_ray);
                    }

                    Intersection isect_sample_ray;
                    hitrecord hrec_sample;

                    if (scene->hit(ctxt, sample_ray, AT_MATH_EPSILON, AT_MATH_INF, hrec_sample, isect_sample_ray)) {
                        const auto distance_sample_pos_on_query_ray = aten::FeatureLine::computeDistanceBetweenProjectedPosOnRayAndRayOrg(
                            hrec_sample.p, ray);

                        if (distance_sample_pos_on_query_ray < closest_sample_ray_distance) {
                            const auto is_line_width = aten::FeatureLine::isInLineWidth(
                                FeatureLineWidth,
                                ray,
                                hrec_sample.p,
                                disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                                pixel_width);
                            if (is_line_width) {
                                // If sample ray doesn't hit anything, it is forcibly feature line.
                                closest_sample_ray_idx = i;
                                closest_sample_ray_distance = distance_sample_pos_on_query_ray;
                            }
                        }
                    }
                }


                if (closest_sample_ray_idx < 0) {
                    shadeMiss(idx, scene, depth, paths, rays, bg);
                }
                willContinue = false;
            }

            if (depth < startDepth && !paths.attrib[idx].isTerminate) {
                paths.contrib[idx].contrib = vec3(0);
            }
            else if (closest_sample_ray_idx >= 0) {
                auto pdf_feature_line = real(1) / SampleRayNum;
                pdf_feature_line = pdf_feature_line * (closest_sample_ray_distance * closest_sample_ray_distance);
                const auto weight = paths.throughput[idx].pdfb / (paths.throughput[idx].pdfb + pdf_feature_line);
                paths.contrib[idx].contrib += paths.throughput[idx].throughput * weight * LineColor;
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
        const aten::hitrecord& rec,
        scene* scene,
        int32_t rrDepth,
        int32_t bounce)
    {
        const auto& ray = rays[idx];
        auto* sampler = &paths.sampler[idx];

        bool isBackfacing = dot(rec.normal, -ray.dir) < real(0);

        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        vec3 orienting_normal = rec.normal;

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        auto albedo = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

        auto& shadow_ray = shadow_rays[idx];
        shadow_ray.isActive = false;

        // Implicit conection to light.
        if (mtrl.attrib.isEmissive) {
            if (!isBackfacing) {
                real weight = 1.0f;

                if (bounce > 0 && !paths.attrib[idx].isSingular)
                {
                    auto cosLight = dot(orienting_normal, -ray.dir);
                    auto dist2 = aten::squared_length(rec.p - ray.org);

                    if (cosLight >= 0) {
                        auto pdfLight = 1 / rec.area;

                        // Convert pdf area to sradian.
                        // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
                        // p31 - p35
                        pdfLight = pdfLight * dist2 / cosLight;

                        weight = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);
                    }
                }

                auto emit = static_cast<aten::vec3>(mtrl.baseColor);
                paths.contrib[idx].contrib += paths.throughput[idx].throughput * weight * emit;
            }

            paths.attrib[idx].isTerminate = true;
            return;
        }

        if (!mtrl.attrib.isTranslucent && isBackfacing) {
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
        // NOTE:
        // If the material itself is originally translucent, we don't care alpha translucency.
        if (!mtrl.attrib.isTranslucent
            && AT_NAME::material::isTranslucentByAlpha(mtrl, rec.u, rec.v))
        {
            const auto alpha = AT_NAME::material::getTranslucentAlpha(mtrl, rec.u, rec.v);
            auto r = sampler->nextSample();

            if (r >= alpha) {
                // Just through the object.
                // NOTE
                // Ray go through to the opposite direction. So, we need to specify inverted normal.
                rays[idx] = aten::ray(rec.p, ray.dir, -orienting_normal);
                paths.throughput[idx].throughput *= static_cast<aten::vec3>(mtrl.baseColor);
                paths.attrib[idx].isSingular = true;
                shadow_rays[idx].isActive = false;
                return;
            }
        }

        const auto lightnum = ctxt.get_light_num();

        // Explicit conection to light.
        if (lightnum > 0
            && !(mtrl.attrib.isSingular || mtrl.attrib.isTranslucent))
        {
            auto lightidx = aten::cmpMin<int32_t>(paths.sampler[idx].nextSample() * lightnum, lightnum - 1);
            const auto& light_param = ctxt.GetLight(lightidx);
            const auto lightSelectPdf = real(1) / lightnum;

            auto isShadowRayActive = AT_NAME::FillShadowRay(
                shadow_ray,
                ctxt,
                bounce,
                paths.sampler[idx],
                paths.throughput[idx],
                lightidx,
                light_param,
                mtrl,
                ray,
                rec.p, orienting_normal,
                rec.u, rec.v, albedo,
                lightSelectPdf,
                pre_sampled_r);

            shadow_ray.isActive = isShadowRayActive;
        }

        real russianProb = real(1);

        if (bounce > rrDepth) {
            auto t = normalize(paths.throughput[idx].throughput);
            auto p = std::max(t.r, std::max(t.g, t.b));

            russianProb = sampler->nextSample();

            if (russianProb >= p) {
                paths.attrib[idx].isTerminate = true;
                return;
            }
            else {
                russianProb = p;
            }
        }

        aten::MaterialSampling sampling;
        material::sampleMaterial(
            &sampling,
            &mtrl,
            orienting_normal,
            ray.dir,
            rec.normal,
            sampler, pre_sampled_r,
            rec.u, rec.v);

        auto nextDir = normalize(sampling.dir);
        auto pdfb = sampling.pdf;
        auto bsdf = sampling.bsdf;

        // Get normal to add ray offset.
        // In refraction material case, new ray direction might be computed with inverted normal.
        // For example, when a ray go into the refraction surface, inverted normal is used to compute new ray direction.
        auto rayBasedNormal = (!isBackfacing && mtrl.attrib.isTranslucent)
            ? -orienting_normal
            : orienting_normal;

        auto c = dot(rayBasedNormal, static_cast<vec3>(nextDir));

        if (pdfb > 0 && c > 0) {
            paths.throughput[idx].throughput *= bsdf * c / pdfb;
            paths.throughput[idx].throughput /= russianProb;
        }
        else {
            paths.attrib[idx].isTerminate = true;
            return;
        }

        paths.throughput[idx].pdfb = pdfb;
        paths.attrib[idx].isSingular = mtrl.attrib.isSingular;
        paths.attrib[idx].mtrlType = mtrl.type;

        // Make next ray.
        rays[idx] = aten::ray(rec.p, nextDir, rayBasedNormal);
    }

    void PathTracing::HitShadowRay(
        scene* scene,
        int32_t idx,
        int32_t bounce,
        const aten::context& ctxt,
        aten::Path paths,
        const aten::ShadowRay* shadow_rays)
    {
        if (paths.attrib[idx].isKill || paths.attrib[idx].isTerminate) {
            paths.attrib[idx].isTerminate = true;
            return;
        }

        const auto& shadowRay = shadow_rays[idx];

        if (!shadowRay.isActive) {
            return;
        }

        // TODO
        bool enableLod = (bounce >= 2);

        hitrecord tmpRec;

        auto isHit = AT_NAME::HitShadowRay<std::remove_pointer_t<decltype(scene)>>(
            enableLod, ctxt, shadowRay, scene);

        if (isHit) {
            auto contrib = shadowRay.lightcontrib;
            paths.contrib[idx].contrib += contrib;
        }
    }

    void PathTracing::shadeMiss(
        int32_t idx,
        scene* scene,
        int32_t depth,
        Path& paths,
        const ray* rays,
        const background* bg)
    {
        const auto& ray = rays[idx];

        auto ibl = scene->getIBL();
        aten::vec3 emit(real(0));
        real misW = real(1);

        if (ibl) {
            if (depth == 0) {
                emit = ibl->getEnvMap()->sample(ray);
                misW = real(1);
                paths.attrib[idx].isTerminate = true;
            }
            else {
                emit = ibl->getEnvMap()->sample(ray);
                auto pdfLight = ibl->samplePdf(ray);
                misW = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);
            }
        }
        else {
            emit = sampleBG(ray, bg);
            misW = real(1);
        }

        paths.contrib[idx].contrib += paths.throughput[idx].throughput * misW * emit;
    }

    static uint32_t frame = 0;

    void PathTracing::onRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        frame++;

        int32_t width = dst.width;
        int32_t height = dst.height;
        uint32_t samples = dst.sample;

        m_maxDepth = dst.maxDepth;
        m_rrDepth = dst.russianRouletteDepth;
        m_startDepth = dst.startDepth;

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
                    int32_t pos = y * width + x;

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
                        const auto rnd = aten::getRandom(pos);
                        const auto& camsample = camera->param();

                        generate_path(
                            rays_[idx],
                            idx,
                            x, y,
                            i, get_frame_count(),
                            path_host_.paths,
                            camsample,
                            rnd);

                        path_host_.paths.contrib[idx].contrib = aten::vec3(0);

                        if (enable_feature_line_) {
                            radiance_with_feature_line(
                                idx,
                                path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                                m_rrDepth, m_startDepth, m_maxDepth,
                                camera,
                                scene,
                                bg());
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

                        if (path_host_.paths.attrib[idx].isTerminate) {
                            break;
                        }
                    }

                    col /= (real)cnt;

#if 0
                    if (hrec.mtrlid >= 0) {
                        const auto mtrl = ctxt.getMaterial(hrec.mtrlid);
                        if (mtrl && mtrl->isNPR()) {
                            col = FeatureLine::renderFeatureLine(
                                col,
                                x, y, width, height,
                                hrec,
                                ctxt, *scene, *camera);
                        }
                    }
#endif

                    dst.buffer->put(x, y, vec4(col, 1));

                    if (dst.variance) {
                        col2 /= (real)cnt;
                        dst.variance->put(x, y, vec4(col2 - col * col, real(1)));
                    }
                }
            }
        }
    }
}
