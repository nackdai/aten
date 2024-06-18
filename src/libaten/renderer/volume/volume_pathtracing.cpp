#include "renderer/volume/volume_pathtracing.h"

#include "material/material_impl.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/volume/volume_pathtracing_impl.h"
#include "sampler/cmj.h"
#include "volume/medium.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (-1)
#define BREAK_Y    (-1)
#pragma optimize( "", off)
#endif

namespace aten
{
    void VolumePathTracing::radiance(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        const context& ctxt,
        scene* scene,
        const aten::CameraParameter& camera)
    {
        int32_t depth = 0;
        int32_t loop_count = 0;

        while (depth < m_maxDepth) {
            if (loop_count >= aten::MedisumStackSize) {
                path_host_.attrib[idx].isTerminate = true;
                break;
            }

            bool willContinue = true;
            bool can_update_depth = false;
            Intersection isect;

            const auto& ray = rays_[idx];

            path_host_.paths.attrib[idx].isHit = false;

            if (scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, isect)) {
                path_host_.paths.attrib[idx].isHit = true;

                can_update_depth = nee(
                    idx,
                    path_host_.paths, ctxt, rays_.data(),
                    isect, scene,
                    m_rrDepth, depth);

                willContinue = !path_host_.paths.attrib[idx].isTerminate;
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

            if (can_update_depth) {
                depth++;
            }

            loop_count++;
        }
    }

    bool VolumePathTracing::shade(
        int32_t idx,
        aten::Path& paths,
        const context& ctxt,
        ray* rays,
        const aten::Intersection& isect,
        scene* scene,
        int32_t rrDepth,
        int32_t bounce)
    {
        const auto russianProb = AT_NAME::ComputeRussianProbability(
            bounce, rrDepth,
            paths.attrib[idx], paths.throughput[idx],
            paths.sampler[idx]);
        if (paths.attrib[idx].isTerminate) {
            return false;
        }
        paths.throughput[idx].throughput /= russianProb;

        auto* sampler = &paths.sampler[idx];

        auto ray = rays[idx];
        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < float(0);

        vec3 orienting_normal = rec.normal;

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        bool is_scattered = false;
        aten::ray next_ray(rec.p, ray.dir);

        if (AT_NAME::HasMedium(paths.throughput[idx].mediums)) {
            const auto& medium = AT_NAME::GetCurrentMedium(ctxt, paths.throughput[idx].mediums);
            aten::tie(is_scattered, next_ray) = AT_NAME::HomogeniousMedium::Sample(
                paths.throughput[idx],
                paths.sampler[idx],
                ray, medium, isect.t);

            ray = next_ray;
        }

        bool is_reflected_or_refracted = false;

        if (is_scattered) {
            rays[idx] = ray;
        }
        else {
            // Implicit conection to light.
            auto is_hit_implicit_light = AT_NAME::HitImplicitLight(
                isBackfacing,
                bounce,
                paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
                ray,
                rec.p, orienting_normal,
                rec.area,
                mtrl);
            if (is_hit_implicit_light) {
                return false;
            }

            const auto curr_ray = ray;

            if (mtrl.is_medium) {
                auto ray_base_nml = dot(ray.dir, orienting_normal) > 0
                    ? orienting_normal
                    : -orienting_normal;
                rays[idx] = aten::ray(rec.p, ray.dir, ray_base_nml);
            }
            else {
                auto albedo = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, mtrl.baseColor, bounce);


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

                is_reflected_or_refracted = true;
            }

            AT_NAME::UpdateMedium(curr_ray, rays[idx].dir, orienting_normal, mtrl, paths.throughput[idx].mediums);
        }

        bool will_update_depth = is_scattered || is_reflected_or_refracted;

        return will_update_depth;
    }

    bool VolumePathTracing::nee(
        int32_t idx,
        aten::Path& paths,
        const context& ctxt,
        ray* rays,
        const aten::Intersection& isect,
        scene* scene,
        int32_t rrDepth,
        int32_t bounce)
    {
        const auto russianProb = AT_NAME::ComputeRussianProbability(
            bounce, rrDepth,
            paths.attrib[idx], paths.throughput[idx],
            paths.sampler[idx]);
        if (paths.attrib[idx].isTerminate) {
            return false;
        }
        paths.throughput[idx].throughput /= russianProb;

        auto* sampler = &paths.sampler[idx];

        auto ray = rays[idx];
        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < float(0);

        vec3 orienting_normal = rec.normal;

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        bool is_scattered = false;
        aten::ray next_ray(rec.p, ray.dir);

        if (AT_NAME::HasMedium(paths.throughput[idx].mediums)) {
            const auto& medium = AT_NAME::GetCurrentMedium(ctxt, paths.throughput[idx].mediums);
            aten::tie(is_scattered, next_ray) = AT_NAME::HomogeniousMedium::Sample(
                paths.throughput[idx],
                paths.sampler[idx],
                ray, medium, isect.t);

            if (is_scattered) {
                auto nml = dot(next_ray.dir, orienting_normal) > 0
                    ? orienting_normal
                    : -orienting_normal;

                aten::LightSampleResult light_sample;
                float light_select_prob = 0.0F;
                int target_light_idx = -1;
                aten::tie(light_sample, light_select_prob, target_light_idx) = AT_NAME::SampleLight(
                    ctxt, mtrl, bounce,
                    paths.sampler[idx],
                    next_ray.org, nml);

                if (target_light_idx >= 0) {
                    float transmittance = 1.0F;
                    float is_visilbe_to_light = false;

                    aten::tie(is_visilbe_to_light, transmittance) = AT_NAME::TraverseShadowRay(
                        ctxt, light_sample,
                        next_ray.org, nml,
                        paths.throughput[idx].mediums,
                        scene);

                    if (is_visilbe_to_light) {
                        const auto& medium = AT_NAME::GetCurrentMedium(ctxt, paths.throughput[idx].mediums);
                        const auto phase_f = AT_NAME::HenyeyGreensteinPhaseFunction::Evaluate(
                            medium.phase_function_g,
                            -next_ray.dir, light_sample.dir);

                        const auto dist2 = aten::sqr(light_sample.dist_to_light);

                        // Geometry term.
                        const auto G = 1.0F / dist2;

                        const auto Ls = transmittance * phase_f * G * light_sample.light_color / light_sample.pdf / light_select_prob;

                        paths.contrib[idx].contrib += paths.throughput[idx].throughput * Ls;
                    }
                }
            }

            ray = next_ray;
        }

        bool is_reflected_or_refracted = false;

        if (is_scattered) {
            rays[idx] = ray;
        }
        else {
            // Implicit conection to light.
            auto is_hit_implicit_light = AT_NAME::HitImplicitLight(
                isBackfacing,
                bounce,
                paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
                ray,
                rec.p, orienting_normal,
                rec.area,
                mtrl);
            if (is_hit_implicit_light) {
                return false;
            }

            const auto curr_ray = ray;

            if (mtrl.is_medium) {
                auto ray_base_nml = dot(ray.dir, orienting_normal) > 0
                    ? orienting_normal
                    : -orienting_normal;
                rays[idx] = aten::ray(rec.p, ray.dir, ray_base_nml);
            }
            else {
                auto albedo = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, mtrl.baseColor, bounce);

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

                // NEE
                aten::LightSampleResult light_sample;
                float light_select_prob = 0.0F;
                int target_light_idx = -1;
                aten::tie(light_sample, light_select_prob, target_light_idx) = AT_NAME::SampleLight(
                    ctxt, mtrl, bounce,
                    paths.sampler[idx],
                    rec.p, orienting_normal);

                if (target_light_idx >= 0) {
                    float transmittance = 1.0F;
                    float is_visilbe_to_light = false;

                    aten::tie(is_visilbe_to_light, transmittance) = AT_NAME::TraverseShadowRay(
                        ctxt, light_sample,
                        rec.p, orienting_normal,
                        paths.throughput[idx].mediums,
                        scene);

                    if (is_visilbe_to_light) {
                        auto radiance = AT_NAME::ComputeRadianceNEE(
                            ray, orienting_normal,
                            mtrl, pre_sampled_r, rec.u, rec.v,
                            light_select_prob, light_sample);
                        if (radiance.has_value()) {
                            const auto& r = radiance.value();
                            paths.contrib[idx].contrib += paths.throughput[idx].throughput * transmittance * r * static_cast<aten::vec3>(albedo);
                        }
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

                AT_NAME::PrepareForNextBounce(
                    idx,
                    rec, isBackfacing, russianProb,
                    orienting_normal,
                    mtrl, sampling,
                    albedo,
                    paths, rays);

                is_reflected_or_refracted = true;
            }

            AT_NAME::UpdateMedium(curr_ray, rays[idx].dir, orienting_normal, mtrl, paths.throughput[idx].mediums);
        }

        bool will_update_depth = is_scattered || is_reflected_or_refracted;

        return will_update_depth;
    }

    void VolumePathTracing::shadeMiss(
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
                paths.attrib[idx].isTerminate = true;
            }
            else {
                auto pdfLight = ibl->samplePdf(ray, ctxt);
                misW = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);
            }
        }

        paths.contrib[idx].contrib += paths.throughput[idx].throughput * misW * emit;
    }

    void VolumePathTracing::OnRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
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
        path_host_.init(width, height);
        path_host_.Clear(GetFrameCount());

        for (auto& attrib : path_host_.attrib) {
            attrib.isKill = false;
        }

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

                    if (path_host_.paths.attrib[idx].isKill) {
                        continue;
                    }

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

                        radiance(
                            idx,
                            x, y, width, height,
                            ctxt, scene, camsample);

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

                    col /= (float)cnt;

                    dst.buffer->put(x, y, vec4(col, 1));
                }
            }
        }
    }
}
