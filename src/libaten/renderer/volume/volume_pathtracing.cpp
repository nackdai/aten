#pragma warning(push)
#pragma warning(disable:4146)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#pragma warning(pop)

#include "renderer/volume/volume_pathtracing.h"

#include "accelerator/threaded_bvh_traverser.h"
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
        int32_t loop_count = 0;

        while (!path_host_.paths.attrib[idx].is_terminated) {
            if (loop_count >= aten::MedisumStackSize) {
                path_host_.attrib[idx].is_terminated = true;
                break;
            }

            bool willContinue = true;
            bool will_update_depth = false;

            const auto& ray = rays_[idx];

            path_host_.paths.attrib[idx].isHit = false;

            Intersection isect;
            bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect,
                ctxt,
                ray,
                AT_MATH_EPSILON, AT_MATH_INF);

            if (is_hit) {
                path_host_.paths.attrib[idx].isHit = true;

                Nee(
                    idx,
                    path_host_.paths, ctxt,
                    rays_.data(), shadow_rays_.data(),
                    isect, scene,
                    m_rrDepth);

                TraverseShadowRay(
                    idx, m_maxDepth,
                    path_host_.paths, ctxt, isect,
                    shadow_rays_.data(),
                    scene);
            }
            else {
                ShadeMiss(
                    idx,
                    ix, iy,
                    width, height,
                    path_host_.throughput[idx].medium.depth_count,
                    ctxt, camera,
                    path_host_.paths, rays_[idx]);

                path_host_.paths.attrib[idx].is_terminated = true;
            }

            loop_count++;
        }
    }

    void VolumePathTracing::Shade(
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
        if (paths.attrib[idx].is_terminated) {
            paths.attrib[idx].will_update_depth = false;
            return;
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

        if (AT_NAME::HasMedium(paths.throughput[idx].medium.stack)) {
            aten::ray next_ray;

            aten::tie(is_scattered, next_ray) = AT_NAME::SampleMedium(
                paths.throughput[idx],
                paths.sampler[idx],
                ctxt,
                ray, isect);

            ray = next_ray;
        }

        bool is_reflected_or_refracted = false;

        if (is_scattered) {
            rays[idx] = ray;
        }
        else {
            // Implicit conection to light.
            auto is_hit_implicit_light = AT_NAME::HitImplicitLight(
                ctxt, isect.objid,
                isBackfacing,
                bounce,
                paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
                ray,
                rec,
                mtrl);
            if (is_hit_implicit_light) {
                paths.attrib[idx].will_update_depth = false;
                return;
            }

            const auto curr_ray = ray;

            if (mtrl.is_medium && !AT_NAME::IsSubsurface(mtrl)) {
                auto ray_base_nml = dot(ray.dir, orienting_normal) > 0
                    ? orienting_normal
                    : -orienting_normal;
                rays[idx] = aten::ray(rec.p, ray.dir, ray_base_nml);
            }
            else {
                auto albedo = AT_NAME::sampleTexture(ctxt, mtrl.albedoMap, rec.u, rec.v, mtrl.baseColor, bounce);

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

                is_reflected_or_refracted = true;
            }

            AT_NAME::UpdateMedium(curr_ray, rays[idx].dir, orienting_normal, mtrl, paths.throughput[idx].medium.stack);
        }

        paths.attrib[idx].will_update_depth = is_scattered || is_reflected_or_refracted;

        return;
    }

    void VolumePathTracing::TraverseShadowRay(
        const int32_t idx,
        const int32_t max_depth,
        aten::Path& paths,
        const aten::context& ctxt,
        const aten::Intersection& isect,
        const AT_NAME::ShadowRay* shadow_rays,
        aten::scene* scene)
    {
        const auto& shadow_ray = shadow_rays[idx];

        aten::TraverseShadowRay(
            idx,
            shadow_ray,
            max_depth,
            paths, ctxt, isect);
    }

    void VolumePathTracing::Nee(
        int32_t idx,
        aten::Path& paths,
        const context& ctxt,
        aten::ray* rays,
        AT_NAME::ShadowRay* shadow_rays,
        const aten::Intersection& isect,
        scene* scene,
        int32_t rrDepth)
    {
        const auto bounce = paths.throughput[idx].medium.depth_count;

        const auto russianProb = AT_NAME::ComputeRussianProbability(
            bounce, rrDepth,
            paths.attrib[idx], paths.throughput[idx],
            paths.sampler[idx]);
        if (paths.attrib[idx].is_terminated) {
            paths.attrib[idx].will_update_depth = false;
            return;
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

        auto& shadow_ray = shadow_rays[idx];
        shadow_ray.isActive = false;

        bool is_scattered = false;

        if (AT_NAME::HasMedium(paths.throughput[idx].medium.stack)) {
            aten::ray next_ray;

            aten::tie(is_scattered, next_ray) = AT_NAME::SampleMedium(
                paths.throughput[idx],
                paths.sampler[idx],
                ctxt,
                ray, isect);

            if (is_scattered) {
                shadow_ray.isActive = true;

                shadow_ray.rayorg = next_ray.org;
                shadow_ray.raydir = next_ray.dir;
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
                ctxt, isect.objid,
                isBackfacing,
                bounce,
                paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
                ray,
                rec,
                mtrl);
            if (is_hit_implicit_light) {
                paths.attrib[idx].will_update_depth = false;
                return;
            }

            const auto curr_ray = ray;

            if (mtrl.is_medium && !AT_NAME::IsSubsurface(mtrl)) {
                auto ray_base_nml = dot(ray.dir, orienting_normal) > 0
                    ? orienting_normal
                    : -orienting_normal;
                rays[idx] = aten::ray(rec.p, ray.dir, ray_base_nml);
            }
            else {
                auto albedo = AT_NAME::sampleTexture(ctxt, mtrl.albedoMap, rec.u, rec.v, mtrl.baseColor, bounce);

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

                    aten::tie(is_visilbe_to_light, transmittance) = AT_NAME::TraverseRayInMedium(
                        ctxt, *sampler,
                        light_sample,
                        rec.p, orienting_normal,
                        paths.throughput[idx].medium.stack);

                    if (is_visilbe_to_light) {
                        auto radiance = AT_NAME::ComputeRadianceNEE(
                            ctxt,
                            paths.throughput[idx].throughput,
                            ray.dir, orienting_normal,
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

                is_reflected_or_refracted = true;
            }

            AT_NAME::UpdateMedium(curr_ray, rays[idx].dir, orienting_normal, mtrl, paths.throughput[idx].medium.stack);
        }

        paths.attrib[idx].will_update_depth = is_scattered || is_reflected_or_refracted;

        return;
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
                paths.attrib[idx].is_terminated = true;
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
                        path_host_.paths.attrib[idx].does_use_throughput_depth = true;

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

                        if (path_host_.paths.attrib[idx].is_terminated) {
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
