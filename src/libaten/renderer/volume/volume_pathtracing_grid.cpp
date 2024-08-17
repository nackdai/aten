#pragma warning(push)
#pragma warning(disable:4146)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#pragma warning(pop)

#include "renderer/volume/volume_pathtracing.h"

#include "material/material_impl.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/volume/volume_pathtracing_impl.h"
#include "sampler/cmj.h"
#include "volume/medium.h"
#include "volume/grid.h"

namespace aten
{
    bool VolumePathTracing::ShadeWithGrid(
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
            auto* grid = mtrl.is_medium
                ? AT_NAME::GetGridFromContext(ctxt, mtrl.medium)
                : nullptr;

            if (grid) {
                aten::tie(is_scattered, next_ray) = AT_NAME::HeterogeneousMedium::Sample(
                    paths.throughput[idx],
                    paths.sampler[idx],
                    ray,
                    mtrl.medium,
                    grid);

                ray = next_ray;
            }
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
                rec.p, orienting_normal,
                rec.area,
                mtrl);
            if (is_hit_implicit_light) {
                return false;
            }

            const auto curr_ray = ray;

            if (mtrl.is_medium && !AT_NAME::IsSubsurface(mtrl)) {
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

    bool VolumePathTracing::NeeWithGrid(
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
            auto* grid = mtrl.is_medium
                ? AT_NAME::GetGridFromContext(ctxt, mtrl.medium)
                : nullptr;

            if (grid) {
                aten::tie(is_scattered, next_ray) = AT_NAME::HeterogeneousMedium::Sample(
                    paths.throughput[idx],
                    paths.sampler[idx],
                    ray,
                    mtrl.medium,
                    grid);

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
                        next_ray.org, nml,
                        false);

                    if (target_light_idx >= 0) {
                        float transmittance = 1.0F;
                        float is_visilbe_to_light = false;

                        aten::tie(is_visilbe_to_light, transmittance) = AT_NAME::TraverseShadowRay(
                            ctxt, *sampler,
                            light_sample,
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
                rec.p, orienting_normal,
                rec.area,
                mtrl);
            if (is_hit_implicit_light) {
                return false;
            }

            const auto curr_ray = ray;

            if (mtrl.is_medium && !AT_NAME::IsSubsurface(mtrl)) {
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
                        ctxt, *sampler,
                        light_sample,
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
}
