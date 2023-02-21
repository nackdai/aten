#include <array>

#include "renderer/pathtracing.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/renderer_utility.h"
#include "renderer/feature_line.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/wanghash.h"
#include "sampler/cmj.h"
#include "sampler/bluenoiseSampler.h"

#include "material/material_impl.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (-1)
#define BREAK_Y    (-1)
#pragma optimize( "", off)
#endif

namespace aten
{
    PathTracing::Path PathTracing::radiance(
        const context& ctxt,
        sampler* sampler,
        uint32_t maxDepth,
        const ray& inRay,
        camera* cam,
        CameraSampleResult& camsample,
        scene* scene,
        aten::hitrecord* first_hrec/*= nullptr*/)
    {
        uint32_t depth = 0;
        uint32_t rrDepth = m_rrDepth;

        Path path;
        path.ray = inRay;

        while (depth < maxDepth) {
            path.rec = hitrecord();

            bool willContinue = true;
            Intersection isect;

            if (scene->hit(ctxt, path.ray, AT_MATH_EPSILON, AT_MATH_INF, path.rec, isect)) {
                if (depth == 0 && first_hrec) {
                    *first_hrec = path.rec;
                }

                willContinue = shade(ctxt, sampler, scene, cam, camsample, depth, path);
            }
            else {
                shadeMiss(scene, depth, path);
                willContinue = false;
            }

            if (depth < m_startDepth && !path.isTerminate) {
                path.contrib = vec3(0);
            }

            if (!willContinue) {
                break;
            }

            depth++;
        }

        return path;
    }

    PathTracing::Path PathTracing::radiance_with_feature_line(
        const context& ctxt,
        sampler* sampler,
        uint32_t maxDepth,
        const ray& inRay,
        camera* cam,
        CameraSampleResult& camsample,
        scene* scene)
    {
        uint32_t depth = 0;
        uint32_t rrDepth = m_rrDepth;

        Path path;
        path.ray = inRay;

        const auto& cam_org = cam->param().origin;

        const auto pixel_width = cam->computePixelWidthAtDistance(1);

        constexpr size_t SampleRayNum = 8;

        // TODO: These value should be configurable.
        constexpr real FeatureLineWidth = 1;
        constexpr real ThresholdAlbedo = 0.1f;
        constexpr real ThresholdNormal = 0.1f;
        static const aten::vec3 LineColor(0, 1, 0);

        std::array<aten::FeatureLine::SampleRayDesc, SampleRayNum> sample_ray_descs;

        auto disc = aten::FeatureLine::generateDisc(path.ray, FeatureLineWidth, pixel_width);
        for (size_t i = 0; i < SampleRayNum; i++) {
            const auto sample_ray = aten::FeatureLine::generateSampleRay(sample_ray_descs[i], *sampler, path.ray, disc);
            aten::FeatureLine::storeRayToDesc(sample_ray_descs[i], sample_ray);
        }

        disc.accumulated_distance = 1;

        while (depth < maxDepth) {
            path.rec = hitrecord();

            bool willContinue = true;
            Intersection isect;

            auto closest_sample_ray_distance = std::numeric_limits<real>::max();
            int32_t closest_sample_ray_idx = -1;
            real hit_point_distance = 0;

            if (scene->hit(ctxt, path.ray, AT_MATH_EPSILON, AT_MATH_INF, path.rec, isect)) {
                const auto distance_query_ray_hit = length(path.rec.p - path.ray.org);

                // disc.centerはquery_ray.orgに一致する.
                // ただし、最初だけは、query_ray.orgはカメラ視点になっているが、
                // accumulated_distanceでカメラとdiscの距離がすでに含まれている.
                hit_point_distance = length(path.rec.p - disc.center);

                const auto prev_disc = disc;
                disc = aten::FeatureLine::computeNextDisc(
                    path.rec.p,
                    path.ray.dir,
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
                            hrec_sample.p, path.ray);

                        const auto is_line_width = aten::FeatureLine::isInLineWidth(
                            FeatureLineWidth,
                            path.ray,
                            hrec_sample.p,
                            disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                            pixel_width);
                        if (is_line_width) {
                            const auto query_albedo = material::sampleAlbedoMap(
                                &ctxt.getMaterial(path.rec.mtrlid)->param(),
                                path.rec.u, path.rec.v);
                            const auto sample_albedo = material::sampleAlbedoMap(
                                &ctxt.getMaterial(hrec_sample.mtrlid)->param(),
                                path.rec.u, path.rec.v);
                            const auto query_depth = length(path.rec.p - cam_org);
                            const auto sample_depth = length(hrec_sample.p - cam_org);

                            const auto is_feature_line = aten::FeatureLine::evaluateMetrics(
                                path.ray.org,
                                path.rec, hrec_sample,
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
                        const auto query_hit_plane = aten::FeatureLine::computePlane(path.rec);
                        const auto res_sample_ray_dummy_hit = aten::FeatureLine::computeRayHitPosOnPlane(
                            query_hit_plane, sample_ray);

                        const auto is_hit_sample_ray_dummy_plane = std::get<0>(res_sample_ray_dummy_hit);
                        if (is_hit_sample_ray_dummy_plane) {
                            const auto sample_ray_dummy_hit_pos = std::get<1>(res_sample_ray_dummy_hit);

                            const auto distance_sample_pos_on_query_ray = aten::FeatureLine::computeDistanceBetweenProjectedPosOnRayAndRayOrg(
                                sample_ray_dummy_hit_pos, path.ray);

                            const auto is_line_width = aten::FeatureLine::isInLineWidth(
                                FeatureLineWidth,
                                path.ray,
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

                    const auto mtrl = ctxt.getMaterial(path.rec.mtrlid);
                    if (!mtrl->param().attrib.isGlossy) {
                        // In non glossy material case, sample ray doesn't bounce anymore.
                        // TODO
                        // Even if material is glossy, how glossy depends on parameter (e.g. roughness etc).
                        // So, I need to consider how to indetify if sample ray bounce is necessary based on material.
                        sample_ray_descs[i].is_terminated = true;
                    }
                }

                if (closest_sample_ray_idx < 0) {
                    willContinue = shade(ctxt, sampler, scene, cam, camsample, depth, path);
                }
            }
            else {
                aten::FeatureLine::Disc prev_disc;
                if (depth > 0) {
                    // Make dummy point to compute next sample ray.
                    const auto dummy_query_hit_pos = path.ray.org + real(100) * path.ray.dir;
                    hit_point_distance = length(dummy_query_hit_pos - disc.center);

                    prev_disc = disc;
                    disc = aten::FeatureLine::computeNextDisc(
                        dummy_query_hit_pos,
                        path.ray.dir,
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
                            hrec_sample.p, path.ray);

                        if (distance_sample_pos_on_query_ray < closest_sample_ray_distance) {
                            const auto is_line_width = aten::FeatureLine::isInLineWidth(
                                FeatureLineWidth,
                                path.ray,
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
                    shadeMiss(scene, depth, path);
                }
                willContinue = false;
            }

            if (depth < m_startDepth && !path.isTerminate) {
                path.contrib = vec3(0);
            }
            else if (closest_sample_ray_idx >= 0) {
                auto pdf_feature_line = real(1) / SampleRayNum;
                pdf_feature_line = pdf_feature_line * (closest_sample_ray_distance * closest_sample_ray_distance);
                const auto weight = path.pdfb / (path.pdfb + pdf_feature_line);
                path.contrib += path.throughput * weight * LineColor;
            }

            if (!willContinue) {
                break;
            }

            depth++;

            disc.accumulated_distance += hit_point_distance;
        }

        return path;
    }

    bool PathTracing::shade(
        const context& ctxt,
        sampler* sampler,
        scene* scene,
        camera* cam,
        CameraSampleResult& camsample,
        int32_t depth,
        Path& path)
    {
        uint32_t rrDepth = m_rrDepth;

        auto mtrl = ctxt.getMaterial(path.rec.mtrlid);

        bool isBackfacing = dot(path.rec.normal, -path.ray.dir) < real(0);

        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        vec3 orienting_normal = path.rec.normal;

        // Implicit conection to light.
        if (mtrl->isEmissive()) {
            if (!isBackfacing) {
                real weight = 1.0f;

                if (depth > 0
                    && !(path.prevMtrl && path.prevMtrl->isSingularOrTranslucent()))
                {
                    auto cosLight = dot(orienting_normal, -path.ray.dir);
                    auto dist2 = aten::squared_length(path.rec.p - path.ray.org);

                    if (cosLight >= 0) {
                        auto pdfLight = 1 / path.rec.area;

                        // Convert pdf area to sradian.
                        // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
                        // p31 - p35
                        pdfLight = pdfLight * dist2 / cosLight;

                        weight = path.pdfb / (pdfLight + path.pdfb);
                    }
                }

                auto emit = static_cast<aten::vec3>(mtrl->color());
                path.contrib += path.throughput * weight * emit;
            }

            path.isTerminate = true;
            return false;
        }

        if (!mtrl->isTranslucent() && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        // Apply normal map.
        auto pre_sampled_r = material::applyNormal(
            &mtrl->param(),
            mtrl->param().normalMap,
            orienting_normal, orienting_normal,
            path.rec.u, path.rec.v,
            path.ray.dir, sampler);

#if 0
        if (depth == 0) {
            auto Wdash = cam->getWdash(
                path.rec.p,
                camsample.posOnImageSensor,
                camsample.posOnLens,
                camsample.posOnObjectplane);
            auto areaPdf = cam->getPdfImageSensorArea(
                path.rec.p, orienting_normal,
                camsample.posOnImageSensor,
                camsample.posOnLens,
                camsample.posOnObjectplane);

            path.throughput *= Wdash;
            path.throughput /= areaPdf;
        }
#endif

        // Check transparency or translucency.
        // NOTE:
        // If the material itself is originally translucent, we don't care alpha translucency.
        if (!mtrl->isTranslucent()
            && AT_NAME::material::isTranslucentByAlpha(mtrl->param(), path.rec.u, path.rec.v))
        {
            const auto alpha = AT_NAME::material::getTranslucentAlpha(mtrl->param(), path.rec.u, path.rec.v);
            auto r = sampler->nextSample();

            if (r >= alpha) {
                // Just through the object.
                // NOTE
                // Ray go through to the opposite direction. So, we need to specify inverted normal.
                path.ray = aten::ray(path.rec.p, path.ray.dir, -orienting_normal);
                path.throughput *= static_cast<aten::vec3>(mtrl->color());
                return true;
            }
        }

        // Explicit conection to light.
        if (!mtrl->isSingularOrTranslucent())
        {
            real lightSelectPdf = 1;
            LightSampleResult sampleres;

            auto light = scene->sampleLight(
                ctxt,
                path.rec.p,
                orienting_normal,
                sampler,
                lightSelectPdf, sampleres);

            if (light) {
                const vec3& posLight = sampleres.pos;
                const vec3& nmlLight = sampleres.nml;
                real pdfLight = sampleres.pdf;

                auto lightobj = sampleres.obj;

                vec3 dirToLight = normalize(sampleres.dir);

                // TODO
                // Do we need to consider offset for shadow ray?
#if 0
                auto shadowRayOrg = path.rec.p + AT_MATH_EPSILON * orienting_normal;
                auto tmp = path.rec.p + dirToLight - shadowRayOrg;
                auto shadowRayDir = normalize(tmp);
#else
                auto shadowRayOrg = path.rec.p;
                auto shadowRayDir = dirToLight;
#endif

                if (dot(shadowRayDir, orienting_normal) > real(0)) {
                    aten::ray shadowRay(shadowRayOrg, shadowRayDir, orienting_normal);

                    hitrecord tmpRec;

                    if (scene->hitLight(ctxt, light.get(), posLight, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
                        // Shadow ray hits the light.
                        auto cosShadow = dot(orienting_normal, dirToLight);

                        auto bsdf = material::sampleBSDF(
                            &mtrl->param(),
                            orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v, pre_sampled_r);
                        auto pdfb = material::samplePDF(
                            &mtrl->param(),
                            orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);

                        bsdf *= path.throughput;

                        // Get light color.
                        auto emit = sampleres.finalColor;

                        if (light->isInfinite() || light->isSingular()) {
                            if (pdfLight > real(0) && cosShadow >= 0) {
                                auto misW = light->isSingular()
                                    ? real(1)
                                    : aten::computeBalanceHeuristic(pdfLight * lightSelectPdf, pdfb);
                                path.contrib += (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
                            }
                        }
                        else {
                            auto cosLight = dot(nmlLight, -dirToLight);

                            if (cosShadow >= 0 && cosLight >= 0) {
                                auto dist2 = squared_length(sampleres.dir);
                                auto G = cosShadow * cosLight / dist2;

                                if (pdfb > real(0) && pdfLight > real(0)) {
                                    // Convert pdf from steradian to area.
                                    // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
                                    // p31 - p35
                                    pdfb = pdfb * cosLight / dist2;
                                    auto misW = aten::computeBalanceHeuristic(pdfLight * lightSelectPdf, pdfb);
                                    path.contrib += (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
                                }
                            }
                        }
                    }
                }
            }
        }

        real russianProb = real(1);

        if (depth > rrDepth) {
            auto t = normalize(path.throughput);
            auto p = std::max(t.r, std::max(t.g, t.b));

            russianProb = sampler->nextSample();

            if (russianProb >= p) {
                path.contrib = vec3();
                return false;
            }
            else {
                russianProb = p;
            }
        }

        aten::MaterialSampling sampling;
        material::sampleMaterial(
            &sampling,
            &mtrl->param(),
            orienting_normal,
            path.ray.dir,
            path.rec.normal,
            sampler, pre_sampled_r,
            path.rec.u, path.rec.v);

        auto nextDir = normalize(sampling.dir);
        auto pdfb = sampling.pdf;
        auto bsdf = sampling.bsdf;

        // Get normal to add ray offset.
        // In refraction material case, new ray direction might be computed with inverted normal.
        // For example, when a ray go into the refraction surface, inverted normal is used to compute new ray direction.
        auto rayBasedNormal = (!isBackfacing && mtrl->isTranslucent())
            ? -orienting_normal
            : orienting_normal;

        auto c = dot(rayBasedNormal, static_cast<vec3>(nextDir));

        if (pdfb > 0 && c > 0) {
            path.throughput *= bsdf * c / pdfb;
            path.throughput /= russianProb;
        }
        else {
            return false;
        }

        path.prevMtrl = mtrl;

        path.pdfb = pdfb;

        // Make next ray.
        path.ray = aten::ray(path.rec.p, nextDir, rayBasedNormal);

        return true;
    }

    void PathTracing::shadeMiss(
        scene* scene,
        int32_t depth,
        Path& path)
    {
        auto ibl = scene->getIBL();
        aten::vec3 emit(real(0));
        real misW = real(1);

        if (ibl) {
            if (depth == 0) {
                emit = ibl->getEnvMap()->sample(path.ray);
                misW = real(1);
                path.isTerminate = true;
            }
            else {
                emit = ibl->getEnvMap()->sample(path.ray);
                auto pdfLight = ibl->samplePdf(path.ray);
                misW = path.pdfb / (pdfLight + path.pdfb);
            }
        }
        else {
            emit = sampleBG(path.ray);
            misW = real(1);
        }

        path.contrib += path.throughput * misW * emit;
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

        auto time = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            auto idx = OMPUtil::getThreadIdx();

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
                    aten::hitrecord hrec;

                    for (uint32_t i = 0; i < samples; i++) {
                        auto scramble = aten::getRandom(pos) * 0x1fe3434f;

                        //XorShift rnd(scramble + t.milliSeconds);
                        //Halton rnd(scramble + t.milliSeconds);
                        //Sobol rnd;
                        //WangHash rnd(scramble + t.milliSeconds);
#if 1
                        CMJ rnd;
                        rnd.init(frame, i, scramble);
#else
                        // Experimental
                        BlueNoiseSampler rnd;
                        for (auto tex : m_noisetex) {
                            rnd.registerNoiseTexture(tex);
                        }
                        rnd.init(x, y, frame, m_maxDepth, 1);
#endif

                        real u = real(x + rnd.nextSample()) / real(width);
                        real v = real(y + rnd.nextSample()) / real(height);

                        auto camsample = camera->sample(u, v, &rnd);

                        auto ray = camsample.r;

                        Path path;
                        if (enable_feature_line_) {
                            path = radiance_with_feature_line(
                                ctxt,
                                &rnd,
                                m_maxDepth,
                                ray,
                                camera,
                                camsample,
                                scene);
                        }
                        else {
                            path = radiance(
                                ctxt,
                                &rnd,
                                m_maxDepth,
                                ray,
                                camera,
                                camsample,
                                scene,
                                &hrec);
                        }

                        if (isInvalidColor(path.contrib)) {
                            AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, i);
                            continue;
                        }

                        auto pdfOnImageSensor = camsample.pdfOnImageSensor;
                        auto pdfOnLens = camsample.pdfOnLens;

                        auto s = camera->getSensitivity(
                            camsample.posOnImageSensor,
                            camsample.posOnLens);

                        auto c = path.contrib * s / (pdfOnImageSensor * pdfOnLens);

                        col += c;
                        col2 += c * c;
                        cnt++;

                        if (path.isTerminate) {
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
