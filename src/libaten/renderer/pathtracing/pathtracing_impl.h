#pragma once

#include <optional>

#include "defs.h"
#include "camera/camera.h"
#include "camera/pinhole.h"
#include "light/ibl.h"
#include "light/light_impl.h"
#include "math/ray.h"
#include "material/material.h"
#include "material/material_impl.h"
#include "material/toon.h"
#include "misc/tuple.h"
#include "misc/type_traits.h"
#include "renderer/aov.h"
#include "scene/scene.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#include "kernel/accelerator.cuh"
#else
#include "scene/host_scene_context.h"
#endif

#include "renderer/pathtracing/pt_params.h"
#include "renderer/pathtracing/pathtracing_nee_impl.h"

namespace AT_NAME
{
#ifndef __CUDACC__
    inline aten::vec3 make_float3(float x, float y, float z) { return { x, y, z }; }
    inline aten::vec4 make_float4(float x, float y, float z, float w) { return { x, y, z, w }; }
#endif

    inline AT_DEVICE_API void ClearPathAttribute(PathAttribute& attrib)
    {
        attrib.isHit = false;
        attrib.is_terminated = false;
        attrib.is_singular = false;
        attrib.will_update_depth = true;
        attrib.does_use_throughput_depth = false;
        attrib.last_hit_mtrl_idx = -1;
    }

    inline AT_DEVICE_API void GeneratePath(
        aten::ray& generated_ray,
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t sample, uint32_t frame,
        AT_NAME::Path& paths,
        const aten::CameraParameter& camera,
        const uint32_t rnd)
    {
#if IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
        auto scramble = rnd * 0x1fe3434f
            * (((frame + sample) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        paths.sampler[idx].init(
            (frame + sample) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
            0,
            scramble);
#else
        static_assert(false, "Other samplers are not supported");
#endif

        float r1 = paths.sampler[idx].nextSample();
        float r2 = paths.sampler[idx].nextSample();

        float s = (ix + r1) / (float)(camera.width);
        float t = (iy + r2) / (float)(camera.height);

        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        generated_ray = camsample.r;

        paths.throughput[idx].throughput = aten::vec3(1);
        paths.throughput[idx].pdfb = 1.0f;
        paths.throughput[idx].depth_count = 0;
        paths.throughput[idx].mediums.clear();

        ClearPathAttribute(paths.attrib[idx]);

        paths.contrib[idx].samples += 1;
    }

    namespace _detail {
        template <class A, class B>
        inline AT_DEVICE_API void AddVec3(A& dst, const B& add)
        {
            if constexpr (std::is_same_v<A, B> && std::is_same_v<A, aten::vec3>) {
                dst += add;
            }
            else {
                dst += make_float3(add.x, add.y, add.z);
            }
        }

        // NOTE:
        // If template type A doesn't have the member variable "w", we can deal with it as vector 3 type.
        // Otherwise, we can deal with it as vector 4 type.

        template <class T>
        using HasMemberWOp = decltype(std::declval<T>().w);

        template <class A, class B>
        inline AT_DEVICE_API auto CopyVec(A& dst, const B& src)
            -> std::enable_if_t<!aten::is_detected<HasMemberWOp, A>::value, void>
        {
            if constexpr (std::is_same_v<A, B> && std::is_same_v<A, aten::vec3>) {
                dst = src;
            }
            else {
                dst = make_float3(src.x, src.y, src.z);
            }
        }

        template <class A, class B>
        inline AT_DEVICE_API auto CopyVec(A& dst, const B& src)
            -> std::enable_if_t<aten::is_detected<HasMemberWOp, A>::value, void>
        {
            if constexpr (std::is_same_v<A, B> && std::is_same_v<A, aten::vec4>) {
                dst = src;
            }
            else {
                dst = make_float4(src.x, src.y, src.z, src.w);
            }
        }

        template <class T = _detail::v3>
        inline AT_DEVICE_API T MakeVec3(float x, float y, float z)
        {
            if constexpr (std::is_same_v<T, aten::vec3>) {
                return { x, y, z };
            }
            else {
                return make_float3(x, y, z);
            }
        }

        template <class T = _detail::v4>
        inline AT_DEVICE_API T MakeVec4(float x, float y, float z, float w)
        {
            if constexpr (std::is_same_v<T, aten::vec4>) {
                return { x, y, z, w };
            }
            else {
                return make_float4(x, y, z, w);
            }
        }
    }

    template <class AOV_BUFFER_TYPE = aten::vec4>
    inline AT_DEVICE_API void ShadeMiss(
        int32_t idx,
        int32_t bounce,
        const aten::vec3& bg,
        AT_NAME::Path& paths,
        aten::span<AOV_BUFFER_TYPE> aov_normal_depth = nullptr,
        aten::span<AOV_BUFFER_TYPE> aov_albedo_meshid = nullptr)
    {
        bounce = paths.attrib[idx].does_use_throughput_depth
            ? paths.throughput[idx].depth_count
            : bounce;

        if (!paths.attrib[idx].is_terminated && !paths.attrib[idx].isHit) {
            if (bounce == 0) {
                if (!aov_normal_depth.empty() && !aov_albedo_meshid.empty())
                {
                    // Export bg color to albedo buffer.
                    AT_NAME::FillBasicAOVsIfHitMiss(
                        aov_normal_depth[idx],
                        aov_albedo_meshid[idx], bg);
                }
            }

            auto contrib = paths.throughput[idx].throughput * bg;
            _detail::AddVec3(paths.contrib[idx].contrib, contrib);

            paths.attrib[idx].is_terminated = true;
        }
    }

    template <class AOV_BUFFER_TYPE = aten::vec4>
    inline AT_DEVICE_API void ShadeMissWithEnvmap(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        int32_t bounce,
        const aten::BackgroundResource bg,
        const AT_NAME::context& ctxt,
        const aten::CameraParameter& camera,
        AT_NAME::Path& paths,
        const aten::ray& ray,
        aten::span<AOV_BUFFER_TYPE> aov_normal_depth = nullptr,
        aten::span<AOV_BUFFER_TYPE> aov_albedo_meshid = nullptr)
    {
        bounce = paths.attrib[idx].does_use_throughput_depth
            ? paths.throughput[idx].depth_count
            : bounce;

        if (!paths.attrib[idx].is_terminated && !paths.attrib[idx].isHit) {
            aten::vec3 dir = ray.dir;

            if (bounce == 0) {
                // Suppress jittering envrinment map.
                // So, re-sample ray without random.

                // TODO
                // More efficient way...

                float s = ix / (float)(width);
                float t = iy / (float)(height);

                AT_NAME::CameraSampleResult camsample;
                AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

                dir = camsample.r.dir;
            }

            auto emit = AT_NAME::Background::SampleFromRay(dir, bg, ctxt);

            float misW = 1.0f;
            if (bounce == 0
                || (bounce == 1 && paths.attrib[idx].is_singular))
            {
                if (!aov_normal_depth.empty() && !aov_albedo_meshid.empty())
                {
                    // Export bg color to albedo buffer.
                    AT_NAME::FillBasicAOVsIfHitMiss(
                        aov_normal_depth[idx],
                        aov_albedo_meshid[idx], emit);
                }
            }
            else {
                auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, bg.avgIllum);
                misW = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);
            }

            auto contrib = paths.throughput[idx].throughput * misW * emit;
            _detail::AddVec3(paths.contrib[idx].contrib, contrib);

            paths.attrib[idx].is_terminated = true;
        }
    }

    inline AT_DEVICE_API aten::tuple<aten::LightSampleResult, float, int32_t> SampleLight(
        const AT_NAME::context& ctxt,
        const aten::MaterialParameter& mtrl,
        int32_t bounce,
        aten::sampler& sampler,
        const aten::vec3& org,
        const aten::vec3& nml,
        bool is_on_surface = true)
    {
        aten::LightSampleResult light_sample;
        float light_select_prob = 0.0F;
        int target_light_idx = -1;

        const auto lightnum = static_cast<int32_t>(ctxt.GetLightNum());

        bool is_invalid_mtrl = is_on_surface && (mtrl.attrib.is_singular || mtrl.attrib.is_translucent);

        if (lightnum <= 0 || is_invalid_mtrl) {
            return aten::make_tuple(light_sample, light_select_prob, target_light_idx);
        }

        target_light_idx = aten::cmpMin<decltype(lightnum)>(
            static_cast<decltype(lightnum)>(sampler.nextSample() * lightnum), lightnum - 1);
        light_select_prob = 1.0f / lightnum;

        const auto& light = ctxt.GetLight(target_light_idx);
        AT_NAME::Light::sample(light_sample, light, ctxt, org, nml, &sampler, bounce);

        return aten::make_tuple(light_sample, light_select_prob, target_light_idx);
    }

    inline AT_DEVICE_API void FillShadowRay(
        AT_NAME::ShadowRay& shadow_ray,
        const AT_NAME::context& ctxt,
        int32_t bounce,
        aten::sampler& sampler,
        const AT_NAME::PathThroughput& throughtput,
        const aten::MaterialParameter& mtrl,
        const aten::ray& ray,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        float hit_u, float hit_v,
        const aten::vec4& external_albedo,
        float pre_sampled_r = float(0))
    {
        shadow_ray.isActive = false;

        aten::LightSampleResult sampleres;
        float lightSelectPdf = 0.0F;
        int32_t target_light_idx = -1;
        aten::tie(sampleres, lightSelectPdf, target_light_idx) = SampleLight(ctxt, mtrl, bounce, sampler, hit_pos, hit_nml);

        if (target_light_idx < 0) {
            return;
        }

        bool isShadowRayActive = false;

        const auto& posLight = sampleres.pos;

        auto dirToLight = normalize(sampleres.dir);
        auto distToLight = length(posLight - hit_pos);

        auto shadowRayOrg = hit_pos + AT_MATH_EPSILON * hit_nml;
        auto tmp = hit_pos + dirToLight - shadowRayOrg;
        auto shadowRayDir = normalize(tmp);

        shadow_ray.rayorg = shadowRayOrg;
        shadow_ray.raydir = shadowRayDir;
        shadow_ray.targetLightId = target_light_idx;
        shadow_ray.distToLight = distToLight;
        shadow_ray.lightcontrib = aten::vec3(0);

        auto radiance = ComputeRadianceNEE(
            ray.dir, hit_nml,
            mtrl, pre_sampled_r, hit_u, hit_v,
            lightSelectPdf, sampleres);
        if (radiance.has_value()) {
            const auto& r = radiance.value();
            shadow_ray.lightcontrib = throughtput.throughput * r * static_cast<aten::vec3>(external_albedo);
            isShadowRayActive = true;
        }

        shadow_ray.isActive = isShadowRayActive;
    }

    template <class SCENE = void>
    inline AT_DEVICE_API bool HitShadowRay(
        int32_t idx,
        int32_t bounce,
        const AT_NAME::context& ctxt,
        AT_NAME::Path paths,
        const AT_NAME::ShadowRay& shadow_ray,
        SCENE* scene = nullptr)
    {
        if (paths.attrib[idx].is_terminated) {
            return false;
        }

        if (!shadow_ray.isActive) {
            return false;
        }

        // TODO
        bool enableLod = (bounce >= 2);

        aten::hitrecord tmpRec;

        const auto targetLightId = shadow_ray.targetLightId;
        const auto distToLight = shadow_ray.distToLight;

        const auto& light = ctxt.GetLight(targetLightId);
        const auto lightobj = (light.IsValidLightObjectId()
            ? &ctxt.GetObject(static_cast<uint32_t>(light.arealight_objid))
            : nullptr);

        float distHitObjToRayOrg = AT_MATH_INF;

        // Ray aim to the area light.
        // So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
        const aten::ObjectParameter* hitobj = lightobj;

        aten::Intersection isect;

        bool isHit = false;

        aten::ray r(shadow_ray.rayorg, shadow_ray.raydir);

        if constexpr (!std::is_void_v<std::remove_pointer_t<SCENE>>) {
            // NOTE:
            // operation has to be related with template arg SCENE.
            if (scene) {
                isHit = scene->hit(ctxt, r, AT_MATH_EPSILON, distToLight - AT_MATH_EPSILON, isect);
            }
        }
        else {
#ifndef __CUDACC__
            // Dummy to build with clang.
            auto intersectCloser = [](auto... args) -> bool { return true; };
#endif
            isHit = intersectCloser(&ctxt, r, &isect, distToLight - AT_MATH_EPSILON, enableLod);
        }

        if (isHit) {
            hitobj = &ctxt.GetObject(static_cast<uint32_t>(isect.objid));
        }

        isHit = AT_NAME::scene::hitLight(
            isHit,
            light.attrib,
            lightobj,
            distToLight,
            distHitObjToRayOrg,
            isect.t,
            hitobj);

        if (isHit) {
            _detail::AddVec3(paths.contrib[idx].contrib, shadow_ray.lightcontrib);
        }

        return isHit;
    }

    inline AT_DEVICE_API bool HitImplicitLight(
        const AT_NAME::context& ctxt,
        int32_t hit_obj_id,
        bool is_back_facing,
        int32_t bounce,
        AT_NAME::PathContrib& path_contrib,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput,
        const aten::ray& ray,
        const aten::hitrecord& hrec,
        const aten::MaterialParameter& hit_target_mtrl)
    {
        if (!hit_target_mtrl.attrib.is_emissive) {
            return false;
        }

        if (is_back_facing) {
            return false;
        }

        const aten::vec3& hit_pos = hrec.p;
        const aten::vec3& hit_nml = hrec.normal;
        float hit_area = hrec.area;

        const auto& obj = ctxt.GetObject(hit_obj_id);
        const auto& light = ctxt.GetLight(obj.light_id);
        const auto light_color = AT_NAME::AreaLight::ComputeLightColor(light, hit_area);

        float weight = 1.0f;

        if (bounce > 0) {
            auto cosLight = dot(hit_nml, -ray.dir);
            auto dist2 = aten::squared_length(hit_pos - ray.org);

            if (cosLight >= 0) {
                auto pdfLight = 1 / hit_area;

                // If logic comes here, light has area.
                // But, samling is fully based on solid angle (path) because of brdf sampling in previous bounce.
                // So, convert area PDF to solid angle PDF.
                pdfLight = pdfLight * dist2 / cosLight;

                weight = _detail::ComputeBalanceHeuristic(path_throughput.pdfb, pdfLight);
            }
        }

        // NOTE:
        // In the previous bounce, (bsdf * cos / path_pdf) has been computed and multiplied to path_throughput.throughput.
        // Therefore, no need to compute it again here.

        auto contrib{ path_throughput.throughput * weight * light_color };
        _detail::AddVec3(path_contrib.contrib, contrib);

        // When ray hit the light, tracing will finish.
        path_attrib.is_terminated = true;
        return true;
    }

    inline AT_DEVICE_API bool HitTeminateMaterial(
        const AT_NAME::context& ctxt,
        aten::sampler& sampler,
        int32_t hit_obj_id,
        bool is_back_facing,
        int32_t bounce,
        AT_NAME::PathContrib& path_contrib,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput,
        const aten::ray& ray,
        const aten::hitrecord& hrec,
        const aten::MaterialParameter& hit_target_mtrl)
    {
        const auto is_teminate_mtrl = material::IsTerminateMaterial(hit_target_mtrl);
        if (!is_teminate_mtrl) {
            return false;
        }

        const auto mtrl_type = hit_target_mtrl.type;

        switch (mtrl_type) {
        case aten::MaterialType::Emissive:
            // Implicit conection to light.
            return HitImplicitLight(
                ctxt,
                hit_obj_id, is_back_facing, bounce,
                path_contrib, path_attrib, path_throughput,
                ray, hrec, hit_target_mtrl);
        case aten::MaterialType::Toon:
        {
            // Treat toon as a light.
            float toon_pdf = 0.0F;
            const auto toon_bsdf = Toon::bsdf(
                ctxt, hit_target_mtrl, sampler,
                hrec.p, hrec.normal, ray.dir,
                0.0f, 0.0f,
                &toon_pdf);
            aten::vec3 contrib{ 0.0F };
            if (toon_pdf > 0.0F) {
                contrib = path_throughput.throughput * toon_bsdf / toon_pdf;
            }
            _detail::AddVec3(path_contrib.contrib, contrib);
            path_attrib.is_terminated = true;
            return true;
        }
        default:
            AT_ASSERT(false);
            return false;
        }

        return true;
    }

    inline AT_DEVICE_API bool CheckMaterialTranslucentByAlpha(
        const aten::MaterialParameter& mtrl,
        float u, float v,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        aten::ray& ray,
        aten::sampler& sampler,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput)
    {
        // If the material itself is originally translucent, we don't care alpha translucency.
        if (mtrl.attrib.is_translucent) {
            return false;
        }

        if (AT_NAME::material::isTranslucentByAlpha(mtrl, u, v))
        {
            const auto alpha = AT_NAME::material::getTranslucentAlpha(mtrl, u, v);
            auto r = sampler.nextSample();

            if (r >= alpha) {
                // Just through the object.
                // NOTE
                // Ray go through to the opposite direction. So, we need to specify inverted normal.
                ray = aten::ray(hit_pos, ray.dir, -hit_nml);
                path_throughput.throughput *= static_cast<aten::vec3>(mtrl.baseColor);
                path_attrib.is_singular = true;
                return true;
            }
        }

        return false;
    }

    inline AT_DEVICE_API float ComputeRussianProbability(
        int32_t bounce,
        int32_t rr_bounce,
        AT_NAME::PathAttribute& path_attrib,
        AT_NAME::PathThroughput& path_throughput,
        aten::sampler& sampler)
    {
        float russian_prob = 1.0f;

        if (bounce > rr_bounce) {
            if (aten::squared_length(path_throughput.throughput) > 0) {
                russian_prob = aten::max_from_vec3(path_throughput.throughput);
                auto p = sampler.nextSample();
                path_attrib.is_terminated = (p >= russian_prob);
            }
        }

        return russian_prob;
    }

    inline AT_DEVICE_API void PrepareForNextBounce(
        int32_t idx,
        const aten::hitrecord& rec,
        bool is_backfacing,
        float russian_prob,
        const aten::vec3& normal,
        const aten::MaterialParameter& mtrl,
        const AT_NAME::MaterialSampling& sampling,
        const aten::vec3& albedo,
        AT_NAME::Path& paths,
        aten::ray* rays)
    {
        const auto next_dir = normalize(sampling.dir);
        const auto pdfb = sampling.pdf;
        const auto bsdf = sampling.bsdf;

        auto ray_along_normal = normal;

        // Adjust the surface normal to along with the same direction as the next ray.
        ray_along_normal = dot(ray_along_normal, next_dir) >= 0.0f ? ray_along_normal : -ray_along_normal;

        auto c = dot(ray_along_normal, static_cast<aten::vec3>(next_dir));

        if (pdfb > 0 && c > 0) {
            paths.throughput[idx].throughput *= bsdf * c / pdfb;
            paths.throughput[idx].throughput /= russian_prob;
        }
        else {
            paths.attrib[idx].is_terminated = true;
            return;
        }

        paths.throughput[idx].throughput *= albedo;
        paths.throughput[idx].pdfb = pdfb;
        paths.attrib[idx].is_singular = mtrl.attrib.is_singular;
        paths.attrib[idx].last_hit_mtrl_idx = mtrl.id;

        // Make next ray.
        rays[idx] = aten::ray(rec.p, next_dir, ray_along_normal);
    }

    template <class SCENE = void, bool ENABLE_ALPHA_TRANLUCENT = false>
    inline AT_DEVICE_API void ShadePathTracing(
        int32_t idx,
        const AT_NAME::context& ctxt,
        AT_NAME::Path paths,
        aten::ray* rays,
        const aten::Intersection& isect,
        aten::MaterialParameter& mtrl,
        AT_NAME::ShadowRay& shadow_ray,
        int32_t rrDepth,
        int32_t bounce,
        SCENE* scene = nullptr)
    {
        const auto& ray = rays[idx];
        auto* sampler = &paths.sampler[idx];

        const auto& obj = ctxt.GetObject(isect.objid);

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < float(0);

        aten::vec3 orienting_normal = rec.normal;

        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        auto albedo = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, aten::vec4(1), bounce);

        shadow_ray.isActive = false;

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

        if constexpr (ENABLE_ALPHA_TRANLUCENT) {
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

        AT_NAME::MaterialSampling sampling;
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
}
